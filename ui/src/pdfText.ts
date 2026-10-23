/** Turn PDF.js text content into a normalised page plus an offset map. */

import { normaliseWithMap, type NormalisedPage } from './normalise';
import {
  buildLogicalText,
  findQuote,
  itemSlices,
  type ItemSlice,
  type LogicalPage,
  type TextItemLike,
} from './resolve';

export interface ExtractedPage {
  /** Text items in DOM order; the index matches `TextLayer.textDivs`. */
  items: TextItemLike[];
  logical: LogicalPage;
  normalised: NormalisedPage;
}

function isTextItem(value: unknown): value is { str: string; hasEOL?: unknown } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'str' in value &&
    typeof (value as { str: unknown }).str === 'string'
  );
}

/**
 * Build the same strings the citation resolver searches, in the same order the
 * text layer uses. Items without a string (marked content) are skipped by both
 * PDF.js and this filter, so `items[i]` lines up with `textDivs[i]`.
 */
export function extractPage(items: readonly unknown[]): ExtractedPage {
  const textItems: TextItemLike[] = items.filter(isTextItem).map((item) => ({
    str: item.str,
    hasEOL: item.hasEOL === true,
  }));
  const logical = buildLogicalText(textItems);
  return {
    items: textItems,
    logical,
    normalised: normaliseWithMap(logical.raw),
  };
}

export interface ResolvedSpan {
  slices: ItemSlice[];
  rawStart: number;
  rawEnd: number;
}

/** Locate a quote on an extracted page and split it into text-item slices. */
export function resolveQuote(
  page: ExtractedPage,
  quote: string,
): ResolvedSpan | null {
  const span = findQuote(page.normalised, quote);
  if (!span) {
    return null;
  }
  const slices = itemSlices(page.logical.ranges, span.rawStart, span.rawEnd);
  if (slices.length === 0) {
    return null;
  }
  return { slices, rawStart: span.rawStart, rawEnd: span.rawEnd };
}
