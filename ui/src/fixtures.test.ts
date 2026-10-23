import { describe, expect, it } from 'vitest';

import {
  CITATION_FIXTURE,
  FIXTURE_CITATIONS,
  FIXTURE_DEMO_ANSWER,
} from './fixtures';
import { normaliseWithMap } from './normalise';
import { findQuote } from './resolve';

const pages = new Map(
  CITATION_FIXTURE.pages.map((page) => [
    page.page_no,
    normaliseWithMap(page.text),
  ]),
);

describe('committed citation fixtures', () => {
  it('ships the expected twenty citations', () => {
    expect(FIXTURE_CITATIONS).toHaveLength(20);
    expect(CITATION_FIXTURE.pages).toHaveLength(4);
  });

  it('resolves every quote against the extracted page text with zero misses', () => {
    const misses = FIXTURE_CITATIONS.filter((citation) => {
      const page = pages.get(citation.page_no);
      return page === undefined || findQuote(page, citation.quote) === null;
    });
    expect(misses.map((citation) => citation.id)).toEqual([]);
  });

  it('resolves every claim in the offline demo against the same pages', () => {
    const misses: string[] = [];
    for (const claim of FIXTURE_DEMO_ANSWER.claims) {
      for (const citation of claim.citations) {
        const page = pages.get(citation.page_no);
        if (page === undefined || findQuote(page, citation.quote) === null) {
          misses.push(`${citation.chunk_id}@${citation.page_no}`);
        }
      }
    }
    expect(misses).toEqual([]);
  });

  it('reports a miss for a quote that is not on the page', () => {
    const page = pages.get(1);
    expect(page).toBeDefined();
    if (page) {
      expect(findQuote(page, 'This sentence is not in the fixture.')).toBeNull();
    }
  });
});
