import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { DEMO_PAPER, DEMO_QUESTIONS } from './demoPaper';

vi.mock('./components/PdfPane', () => ({
  PdfPane: () => null,
}));

afterEach(cleanup);

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
