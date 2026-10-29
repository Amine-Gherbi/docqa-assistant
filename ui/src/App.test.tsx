import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { DEMO_PAPER, DEMO_QUESTIONS } from './demoPaper';
import stylesCss from './styles.css?raw';

vi.mock('./components/PdfPane', () => ({
  PdfPane: () => null,
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState({}, '', '/');
});

function fakeResponse(body: unknown, status: number): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 502 ? 'Bad Gateway' : 'OK',
    json: async () => {
      if (status >= 400 && typeof body !== 'object') {
        throw new Error('not json');
      }
      return body;
    },
  } as Response;
}

const UPLOAD = {
  id: 'up-1',
  status: 'queued',
  collection: 'u_abc',
  filename: 'five-pages.pdf',
  byte_size: 2048,
  pages_done: 0,
  pages_total: null,
  error_code: null,
  error_detail: null,
  created_at: '2026-10-29T00:00:00Z',
  expires_at: '2026-10-30T00:00:00Z',
};

function pdfFile(name = 'five-pages.pdf'): File {
  return new File(['%PDF-1.7\n%test\n'], name, { type: 'application/pdf' });
}

function cssRule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`${escaped}\\s*\\{([^}]*)\\}`).exec(stylesCss);
  return match?.[1] ?? '';
}

describe('real-paper demo', () => {
  it('opens the real paper and shows its title and page count only', () => {
    render(<App />);

    expect(
      screen.getByRole('heading', { level: 1, name: DEMO_PAPER.title }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(`${DEMO_PAPER.page_count} pages`),
    ).toBeInTheDocument();
    expect(screen.queryByText(/\d+\s+citations?/i)).toBeNull();
    expect(screen.queryByText(/fixture-corpus/i)).toBeNull();
  });

  it('offers three example questions and answers each without the API', async () => {
    const user = userEvent.setup();
    render(<App />);

    const examples = screen.getAllByTestId('example-question');
    expect(examples).toHaveLength(3);
    expect(examples.map((button) => button.textContent)).toEqual(
      DEMO_QUESTIONS.map((example) => example.question),
    );

    for (const [index, example] of DEMO_QUESTIONS.entries()) {
      await user.click(examples[index]!);

      expect(screen.getByLabelText('Question')).toHaveValue(example.question);
      const marks = screen.getAllByTestId('claim-citation');
      expect(marks).toHaveLength(example.answer.claims.length);
      expect(marks.length).toBeGreaterThanOrEqual(1);
      expect(marks.length).toBeLessThanOrEqual(3);
      expect(marks.map((mark) => mark.textContent)).toEqual(
        example.answer.claims.map((_, claimIndex) => String(claimIndex + 1)),
      );
    }
  });

  it('styles every claim the same and leaves no claim selected by default', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getAllByTestId('example-question')[0]!);

    const claims = [...document.querySelectorAll<HTMLElement>('.claim')];
    expect(claims.length).toBeGreaterThan(0);
    expect(claims.some((claim) => claim.classList.contains('selected'))).toBe(
      false,
    );
    expect(
      new Set(claims.map((claim) => claim.className)).size,
    ).toBe(1);
  });

  it('wraps source quotes and keeps implementation notes out of the reader view', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getAllByTestId('example-question')[1]!);

    const sourceQuotes = [
      ...document.querySelectorAll<HTMLElement>('.source-quote'),
    ].map((quote) => quote.textContent);
    expect(sourceQuotes).toContain(
      '“SBERT adds a pooling operation to the output of BERT / RoBERTa”',
    );
    expect(screen.queryByText(/offline demo/i)).toBeNull();
    expect(screen.queryByText(/20 of 20/i)).toBeNull();
    expect(screen.queryByText(/fixture/i)).toBeNull();
  });
});

describe('free-instance cold start', () => {
  it('warms up and retries once instead of showing a bare 502', async () => {
    const user = userEvent.setup();
    let releaseWarmup: (() => void) | null = null;
    const warmupGate = new Promise<Response>((resolve) => {
      releaseWarmup = () => resolve(fakeResponse({ status: 'ok' }, 200));
    });
    const fetchMock = vi
      .fn()
      // the warm-up fired on mount
      .mockResolvedValueOnce(fakeResponse({ status: 'ok' }, 200))
      // the first question meets a sleeping instance at the Render edge
      .mockResolvedValueOnce(fakeResponse('Bad Gateway', 502))
      // the guard's own warm-up, held open until the test has seen the message
      .mockReturnValueOnce(warmupGate)
      // the single retry succeeds
      .mockResolvedValueOnce(
        fakeResponse(DEMO_QUESTIONS[0]!.answer, 200),
      );
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    const field = screen.getByLabelText('Question');
    await user.clear(field);
    await user.type(field, 'Which Adam beta2 value does the paper report?');
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    expect(
      await screen.findByText(/starting the free instance/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();

    await act(async () => {
      releaseWarmup!();
    });

    await waitFor(() =>
      expect(screen.queryByText(/starting the free instance/i)).toBeNull(),
    );
    expect(document.querySelectorAll('.claim').length).toBeGreaterThan(0);
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it('does not retry an answer that is not a cold-start status', async () => {
    const user = userEvent.setup();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(fakeResponse({ status: 'ok' }, 200))
      .mockResolvedValueOnce(fakeResponse({ detail: 'bad question' }, 422));
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    const field = screen.getByLabelText('Question');
    await user.clear(field);
    await user.type(field, 'A question with no recorded answer?');
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/starting the free instance/i)).toBeNull();
  });
});

describe('visitor upload', () => {
  it('states the free-tier disclosure immediately above the drop zone', () => {
    render(<App />);

    const disclosure = screen.getByTestId('upload-disclosure');
    const zone = screen.getByTestId('upload-drop-zone');

    expect(disclosure).toHaveTextContent(
      /uploaded text is sent to a third-party model on a free tier/i,
    );
    expect(disclosure).toHaveTextContent(
      /provider may use it to improve its products/i,
    );
    expect(disclosure).toHaveTextContent(
      /do not upload confidential documents/i,
    );
    expect(disclosure).toHaveTextContent(/deleted after 24 hours/i);
    expect(disclosure.nextElementSibling).toBe(zone);
  });

  it('sets the disclosure at the same type size as the primary button', () => {
    expect(cssRule('.primary')).toMatch(/font-size:\s*11px/);
    expect(cssRule('.upload-disclosure')).toMatch(/font-size:\s*11px/);
  });

  it('uploads, polls through the stages, and asks with the upload collection', async () => {
    const user = userEvent.setup();
    const calls: { url: string; init?: RequestInit }[] = [];
    let polls = 0;
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        calls.push({ url, init });
        if (url.endsWith('/health')) {
          return fakeResponse({ status: 'ok' }, 200);
        }
        if (url.endsWith('/session')) {
          return fakeResponse({ client_id: 'c1', new: true }, 200);
        }
        if (url.endsWith('/documents') && init?.method === 'POST') {
          return fakeResponse(UPLOAD, 202);
        }
        if (url.endsWith('/documents/up-1')) {
          polls += 1;
          return fakeResponse(
            polls === 1
              ? {
                  ...UPLOAD,
                  status: 'parsing',
                  pages_done: 2,
                  pages_total: 5,
                }
              : {
                  ...UPLOAD,
                  status: 'ready',
                  pages_done: 5,
                  pages_total: 5,
                },
            200,
          );
        }
        if (url.endsWith('/ask')) {
          return fakeResponse(DEMO_QUESTIONS[0]!.answer, 200);
        }
        throw new Error(`unexpected request ${url}`);
      },
    );
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await user.upload(screen.getByTestId('upload-input'), pdfFile());

    expect(
      await screen.findByText(/parsing page 2 of 5/i),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole(
        'heading',
        { level: 1, name: 'five-pages.pdf' },
        { timeout: 3_000 },
      ),
    ).toBeInTheDocument();

    const field = screen.getByLabelText('Question');
    await user.clear(field);
    await user.type(field, 'What result does this report?');
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    await waitFor(() => {
      expect(calls.some((call) => call.url.endsWith('/ask'))).toBe(true);
    });
    const ask = calls.find((call) => call.url.endsWith('/ask'))!;
    expect(ask.init?.credentials).toBe('include');
    expect(JSON.parse(String(ask.init?.body))).toEqual({
      question: 'What result does this report?',
      collection: 'u_abc',
    });
    const post = calls.find(
      (call) => call.url.endsWith('/documents') && call.init?.method === 'POST',
    )!;
    expect(post.init?.credentials).toBe('include');
    expect(post.init?.body).toBeInstanceOf(FormData);
    expect((post.init?.body as FormData).get('file')).toBeInstanceOf(File);
  });

  it('returns to the demo paper from an open upload', async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) {
          return fakeResponse({ status: 'ok' }, 200);
        }
        if (url.endsWith('/session')) {
          return fakeResponse({ client_id: 'c1', new: true }, 200);
        }
        if (url.endsWith('/documents') ) {
          return fakeResponse(UPLOAD, 202);
        }
        if (url.endsWith('/documents/up-1')) {
          return fakeResponse(
            { ...UPLOAD, status: 'ready', pages_done: 5, pages_total: 5 },
            200,
          );
        }
        throw new Error(`unexpected request ${url}`);
      }),
    );

    render(<App />);
    await user.upload(screen.getByTestId('upload-input'), pdfFile());
    await screen.findByRole(
      'heading',
      { level: 1, name: 'five-pages.pdf' },
      { timeout: 3_000 },
    );

    await user.click(screen.getByRole('button', { name: 'Demo paper' }));

    expect(
      screen.getByRole('heading', { level: 1, name: DEMO_PAPER.title }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(`${DEMO_PAPER.page_count} pages`),
    ).toBeInTheDocument();
  });

  it('restores an upload named by ?document= on reload', async () => {
    window.history.replaceState({}, '', '/?document=up-1');
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) {
          return fakeResponse({ status: 'ok' }, 200);
        }
        if (url.endsWith('/documents/up-1')) {
          return fakeResponse(
            { ...UPLOAD, status: 'ready', pages_done: 5, pages_total: 5 },
            200,
          );
        }
        throw new Error(`unexpected request ${url}`);
      }),
    );

    render(<App />);

    expect(
      await screen.findByRole(
        'heading',
        { level: 1, name: 'five-pages.pdf' },
        { timeout: 3_000 },
      ),
    ).toBeInTheDocument();
  });

  it('explains a refused upload instead of opening a document', async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith('/health')) {
          return fakeResponse({ status: 'ok' }, 200);
        }
        if (url.endsWith('/session')) {
          return fakeResponse({ client_id: 'c1', new: true }, 200);
        }
        if (url.endsWith('/documents') && init?.method === 'POST') {
          return fakeResponse(
            {
              state: 'rejected',
              code: 'too_large',
              detail: 'The file is 30000000 bytes; the limit is 26214400 bytes.',
              limit: 26214400,
            },
            413,
          );
        }
        throw new Error(`unexpected request ${url}`);
      }),
    );

    render(<App />);
    await user.upload(screen.getByTestId('upload-input'), pdfFile());

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/larger than 25 MB/i);
    expect(
      screen.getByRole('heading', { level: 1, name: DEMO_PAPER.title }),
    ).toBeInTheDocument();
  });
});
