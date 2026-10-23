/** Typographic quotation marks for the reader view. */

const INNER_OPEN = '‘';
const INNER_CLOSE = '’';

/**
 * Turn quotation marks that sit inside an already-quoted phrase into single
 * typographic quotes. Straight and curly double quotes both alternate open and
 * close; apostrophes are left untouched.
 */
export function nestQuotes(text: string): string {
  let open = true;
  let out = '';
  for (const character of text) {
    if (character === '“' || character === '”' || character === '"') {
      out += open ? INNER_OPEN : INNER_CLOSE;
      open = !open;
      continue;
    }
    out += character;
  }
  return out;
}

/** Wrap a quotation with outer double quotes and nest any inner marks. */
export function formatQuoted(text: string): string {
  return `“${nestQuotes(text)}”`;
}
