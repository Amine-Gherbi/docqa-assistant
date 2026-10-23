/**
 * The same normalisation the server applies in `docqa.answer.normalise_text`:
 * NFKC, canonical quotes, collapsed whitespace, trailing strip, lower case.
 *
 * JavaScript has no full Unicode casefold; `toLowerCase()` matches it for the
 * English corpus and the fixture text, which is the scope of this build.
 */

const QUOTE_TRANSLATION: Readonly<Record<string, string>> = {
  '‘': "'",
  '’': "'",
  '“': '"',
  '”': '"',
  '–': '-',
  '—': '-',
};

function translate(value: string): string {
  let out = '';
  for (const character of value) {
    out += QUOTE_TRANSLATION[character] ?? character;
  }
  return out;
}

/** Normalise one string for quote matching. */
export function normaliseText(value: string): string {
  return translate(value.normalize('NFKC'))
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

export interface NormalisedPage {
  text: string;
  /** For every normalised character, the raw offset it came from. */
  map: number[];
}

/**
 * Normalise a raw page string while keeping a map back to raw offsets.
 *
 * The map is what lets a quote found in the normalised string become a DOM
 * Range over the PDF.js text layer. Characters are normalised one at a time;
 * for the English fixture (and the corpus) this is identical to whole-string
 * NFKC, and the result is checked against `normaliseText` before it is used.
 */
export function normaliseWithMap(raw: string): NormalisedPage {
  let text = '';
  const map: number[] = [];
  let pendingSpace = false;

  for (let index = 0; index < raw.length; index += 1) {
    const rawCharacter = raw[index] ?? '';
    const piece = translate(rawCharacter.normalize('NFKC')).toLowerCase();
    for (const character of piece) {
      if (/\s/.test(character)) {
        if (text.length > 0) {
          pendingSpace = true;
        }
        continue;
      }
      if (pendingSpace) {
        text += ' ';
        map.push(index);
        pendingSpace = false;
      }
      text += character;
      map.push(index);
    }
  }

  const expected = normaliseText(raw);
  if (text !== expected) {
    // Whole-string NFKC composed across code points. Rebuild the map with the
    // exact normaliser so a quote still resolves; offsets are approximated by
    // mapping each normalised character to the raw prefix that produced it.
    return { text: expected, map: approximateMap(raw, expected) };
  }
  return { text, map };
}

function approximateMap(raw: string, text: string): number[] {
  const map: number[] = [];
  let rawIndex = 0;
  for (let index = 0; index < text.length; index += 1) {
    while (rawIndex < raw.length) {
      const rawCharacter = raw[rawIndex] ?? '';
      const piece = translate(rawCharacter.normalize('NFKC')).toLowerCase();
      rawIndex += 1;
      if (text[index] === ' ' || piece.includes(text[index] ?? '')) {
        break;
      }
    }
    map.push(Math.min(rawIndex - 1, Math.max(raw.length - 1, 0)));
  }
  return map;
}
