/** Pure quote resolution: rebuild the page string, find the quote, map offsets. */

import { normaliseText, type NormalisedPage } from './normalise';

export interface TextItemLike {
  str: string;
  hasEOL: boolean;
}

export interface ItemRange {
  itemIndex: number;
  start: number;
  end: number;
}

export interface LogicalPage {
  raw: string;
  ranges: ItemRange[];
}

export interface LocatedSpan {
  rawStart: number;
  rawEnd: number;
}

export interface ItemSlice {
  itemIndex: number;
  from: number;
  to: number;
}

/**
 * Rebuild the page string from text items, in item order.
 *
 * `hasEOL` becomes a newline so a quote spanning a visual line break matches the
 * same collapsed whitespace the server normaliser produces. The ranges let the
 * caller turn a raw offset span back into text nodes in the PDF.js text layer.
 */
export function buildLogicalText(items: readonly TextItemLike[]): LogicalPage {
  let raw = '';
  const ranges: ItemRange[] = [];
  items.forEach((item, itemIndex) => {
    const start = raw.length;
    raw += item.str;
    const end = raw.length;
    if (end > start) {
      ranges.push({ itemIndex, start, end });
    }
    if (item.hasEOL) {
      raw += '\n';
    }
  });
  return { raw, ranges };
}

/** Locate a normalised quote in a normalised page string. */
export function findQuote(
  page: NormalisedPage,
  quote: string,
): LocatedSpan | null {
  const needle = normaliseText(quote);
  if (needle.length === 0) {
    return null;
  }
  const at = page.text.indexOf(needle);
  if (at < 0) {
    return null;
  }
  const first = page.map[at];
  const last = page.map[at + needle.length - 1];
  if (first === undefined || last === undefined) {
    return null;
  }
  return { rawStart: first, rawEnd: last + 1 };
}

/** Split a raw span into per-text-item slices for Range construction. */
export function itemSlices(
  ranges: readonly ItemRange[],
  rawStart: number,
  rawEnd: number,
): ItemSlice[] {
  const slices: ItemSlice[] = [];
  for (const range of ranges) {
    const from = Math.max(rawStart, range.start);
    const to = Math.min(rawEnd, range.end);
    if (from < to) {
      slices.push({
        itemIndex: range.itemIndex,
        from: from - range.start,
        to: to - range.start,
      });
    }
  }
  return slices;
}
