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

async function waitForTextLayer(page: Page): Promise<void> {
  await expect(page.locator('.textLayer span').first()).toBeVisible();
}

test('all three citations scroll to their page, highlight the quote and show the same number', async ({
  page,
}) => {
  await page.goto('/');

  await expect(
    page.getByRole('heading', {
      level: 1,
      name: 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
    }),
  ).toBeVisible();
  await expect(page.locator('.document-meta')).toHaveText('11 pages');
  await expect(page.locator('#question')).toHaveValue(
    'Why is SBERT more efficient than a BERT cross-encoder?',
  );
  await expect(page.getByTestId('example-question')).toHaveCount(3);
  await expect(page.getByTestId('claim-citation')).toHaveCount(3);

  await waitForTextLayer(page);
  await expect(page.locator('.textLayer')).toContainText(
    'Sentence-BERT: Sentence Embeddings',
  );

  const cases = [
    { number: '1', pageNo: 'Page 1', quote: 'both sentences are fed' },
    {
      number: '2',
      pageNo: 'Page 2',
      quote: 'reduced from 65 hours with bert',
    },
    {
      number: '3',
      pageNo: 'Page 8',
      quote: 'faster than universal sentence encoder',
    },
  ];

  for (const [index, item] of cases.entries()) {
    const mark = page.getByTestId('claim-citation').nth(index);
    await expect(mark).toHaveText(item.number);
    await mark.click();

    await expect(page.getByTestId('pdf-page-no')).toHaveText(item.pageNo);
    await expect(page.getByTestId('passage-marker')).toHaveText(item.number);
    await expect
      .poll(async () => (await readHighlight(page))?.text.toLowerCase() ?? '')
      .toContain(item.quote);
  }

  await page.getByTestId('trace-toggle').click();
  await expect(page.getByTestId('citation-render-miss')).toHaveText('0');
  for (const heading of [
    'Meaning search',
    'Keyword search',
    'Merged ranking',
    'Second-pass ranking',
    'Final context and decision',
  ]) {
    await expect(
      page.getByRole('heading', { name: heading, exact: true }),
    ).toBeVisible();
  }

  expect((await page.locator('body').innerText()).toLowerCase()).not.toContain(
    'fixture',
  );
});

test('every recorded claim resolves on its own page without a miss', async ({
  page,
}) => {
  await page.goto('/');
  await waitForTextLayer(page);

  const examples = [
    [
      { pageNo: 'Page 1', quote: 'both sentences are fed' },
      { pageNo: 'Page 2', quote: 'reduced from 65 hours with bert' },
      { pageNo: 'Page 8', quote: 'faster than universal sentence encoder' },
    ],
    [
      { pageNo: 'Page 3', quote: 'pooling operation to the output' },
      {
        pageNo: 'Page 3',
        quote: 'sentence embeddings are semantically meaningful',
      },
      { pageNo: 'Page 3', quote: 'softmax' },
    ],
    [
      { pageNo: 'Page 4', quote: 'directly using the output' },
      { pageNo: 'Page 4', quote: 'worse than computing average' },
      { pageNo: 'Page 5', quote: 'slight improvement of 1-2 points' },
    ],
  ];

  for (const [exampleIndex, items] of examples.entries()) {
    await page.getByTestId('example-question').nth(exampleIndex).click();
    await waitForTextLayer(page);

    for (const [claimIndex, item] of items.entries()) {
      const mark = page.getByTestId('claim-citation').nth(claimIndex);
      await expect(mark).toHaveText(String(claimIndex + 1));
      await mark.click();
      await expect(page.getByTestId('pdf-page-no')).toHaveText(item.pageNo);
      await expect(page.getByTestId('passage-marker')).toHaveText(
        String(claimIndex + 1),
      );
      await expect
        .poll(async () => (await readHighlight(page))?.text.toLowerCase() ?? '')
        .toContain(item.quote);
    }

    expect(
      (await page.locator('body').innerText()).toLowerCase(),
    ).not.toContain('fixture');
  }

  await page.getByTestId('trace-toggle').click();
  await expect(page.getByTestId('citation-render-miss')).toHaveText('0');
});

test('the whole question is visible at 1280 px and 390 px', async ({ page }) => {
  for (const viewport of [
    { width: 1280, height: 900 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto('/');

    const field = page.locator('#question');
    await expect(field).toHaveValue(
      'Why is SBERT more efficient than a BERT cross-encoder?',
    );
    const clipping = await field.evaluate((element) => {
      const textarea = element as HTMLTextAreaElement;
      return {
        horizontal: textarea.scrollWidth - textarea.clientWidth,
        vertical: textarea.scrollHeight - textarea.clientHeight,
      };
    });
    expect(clipping.horizontal).toBeLessThanOrEqual(1);
    expect(clipping.vertical).toBeLessThanOrEqual(1);
    await expect(page.getByTestId('example-question')).toHaveCount(3);
  }
});

test('the eval page renders the failure gallery and an honest empty report', async ({
  page,
}) => {
  await page.goto('/eval');
  await expect(
    page.getByRole('heading', { name: 'Measure the answer, not the demo.' }),
  ).toBeVisible();
  await expect(page.locator('.gallery-card')).toHaveCount(3);
  await expect(page.locator('.gallery-card').first()).toContainText(
    'Correct refusal with competing passages',
  );
  await expect(page.locator('.eval-report')).toContainText(
    'No evaluation report is being served',
  );
});
