import { expect, test, type Page } from '@playwright/test';

interface HighlightState {
  ranges: number;
  text: string;
}

async function readHighlight(page: Page): Promise<HighlightState | null> {
  return page.evaluate(() => {
    const highlight = CSS.highlights.get('docqa-citation');
    if (!highlight) {
      return null;
    }
    const ranges = [...highlight] as Range[];
    return {
      ranges: ranges.length,
      text: ranges.map((range) => range.toString()).join(' '),
    };
  });
}

test('clicking a claim highlights its quoted span on the cited page', async ({
  page,
}) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Load offline demo' }).click();
  await expect(page.getByTestId('demo-note')).toBeVisible();

  const claim = page.getByRole('button', { name: /quokka/i }).first();
  await expect(claim).toBeVisible();
  await claim.click();

  await expect(page.getByTestId('pdf-page-no')).toHaveText('p.3');
  await expect(page.locator('.textLayer span').first()).toBeVisible();

  await expect
    .poll(async () => (await readHighlight(page))?.ranges ?? 0)
    .toBeGreaterThan(0);

  const state = await readHighlight(page);
  expect(state?.text.toLowerCase()).toContain('quokka');

  await page.getByTestId('trace-toggle').click();
  await expect(page.getByTestId('citation-render-miss')).toHaveText('0');
  await expect(page.getByTestId('fixture-check')).toContainText('20/20');

  for (const heading of [
    'Vector retrieval',
    'Lexical retrieval',
    'Reciprocal rank fusion',
    'Cross-encoder rerank',
    'Final context and refusal',
  ]) {
    await expect(
      page.getByRole('heading', { name: heading, exact: true }),
    ).toBeVisible();
  }
});

test('every demo claim resolves on its own page without a miss', async ({
  page,
}) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Load offline demo' }).click();

  const cases = [
    { claim: /quokka/i, pageNo: 'p.3', quote: 'quokka' },
    { claim: /ts_rank_cd rather than BM25/i, pageNo: 'p.3', quote: 'ts_rank_cd' },
    {
      claim: /Reciprocal rank fusion merges/i,
      pageNo: 'p.1',
      quote: 'reciprocal rank fusion',
    },
  ];

  for (const item of cases) {
    await page.getByRole('button', { name: item.claim }).first().click();
    await expect(page.getByTestId('pdf-page-no')).toHaveText(item.pageNo);
    await expect
      .poll(async () => (await readHighlight(page))?.text.toLowerCase() ?? '')
      .toContain(item.quote);
  }

  await page.getByTestId('trace-toggle').click();
  await expect(page.getByTestId('citation-render-miss')).toHaveText('0');
});

test('the eval page renders the failure gallery and an honest empty report', async ({
  page,
}) => {
  await page.goto('/eval');
  await expect(
    page.getByRole('heading', { name: 'Gates, rates, and failure cases' }),
  ).toBeVisible();
  await expect(page.locator('.gallery-card')).toHaveCount(3);
  await expect(page.locator('.gallery-card').first()).toContainText(
    'Correct refusal with competing passages',
  );
  await expect(page.locator('.eval-report')).toContainText(
    'No eval report is being served',
  );
});
