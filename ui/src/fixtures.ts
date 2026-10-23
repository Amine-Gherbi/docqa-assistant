/** Typed access to the committed citation fixture and its self-check. */

import rawFixture from './fixtures/citations.json';
import { extractPage, resolveQuote, type ExtractedPage } from './pdfText';
import type { CitationFixture, FixtureCitation } from './types';

export const CITATION_FIXTURE = rawFixture as unknown as CitationFixture;
export const FIXTURE_DOC_ID = CITATION_FIXTURE.doc_id;
export const FIXTURE_PDF_URL = CITATION_FIXTURE.pdf;
export const FIXTURE_CITATIONS: readonly FixtureCitation[] =
  CITATION_FIXTURE.citations;
export const FIXTURE_PAGE_COUNT = CITATION_FIXTURE.pages.length;

export interface FixtureCheckMiss {
  id: string;
  page_no: number;
  quote: string;
}

export interface FixtureCheck {
  total: number;
  resolved: number;
  misses: FixtureCheckMiss[];
  pages_checked: number;
}

interface PageProxyLike {
  getTextContent(params?: {
    disableNormalization?: boolean;
  }): Promise<{ items: readonly unknown[] }>;
}

export interface DocumentProxyLike {
  numPages: number;
  getPage(pageNumber: number): Promise<PageProxyLike>;
}

/**
 * Resolve every committed quote against the text PDF.js actually extracts from
 * the fixture PDF. This is the runtime twin of `scripts/verify_fixtures.mjs`:
 * it reports 20/20 on a healthy extractor and names any citation that drifts.
 */
export async function checkFixtureCitations(
  document: DocumentProxyLike,
): Promise<FixtureCheck> {
  const cache = new Map<number, ExtractedPage>();
  const misses: FixtureCheckMiss[] = [];

  for (const citation of FIXTURE_CITATIONS) {
    let page = cache.get(citation.page_no);
    if (!page) {
      const proxy = await document.getPage(citation.page_no);
      const content = await proxy.getTextContent({
        disableNormalization: true,
      });
      page = extractPage(content.items);
      cache.set(citation.page_no, page);
    }
    if (!resolveQuote(page, citation.quote)) {
      misses.push({
        id: citation.id,
        page_no: citation.page_no,
        quote: citation.quote,
      });
    }
  }

  return {
    total: FIXTURE_CITATIONS.length,
    resolved: FIXTURE_CITATIONS.length - misses.length,
    misses,
    pages_checked: cache.size,
  };
}
