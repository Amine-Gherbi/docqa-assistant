import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { DEMO_PAPER, DEMO_QUESTIONS } from './demoPaper';

vi.mock('./components/PdfPane', () => ({
  PdfPane: () => null,
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

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
