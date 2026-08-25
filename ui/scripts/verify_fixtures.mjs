/**
 * Verify every fixture quote against PDF.js text extraction, then publish
 * `src/fixtures/citations.json` with the exact page strings the UI rebuilds.
 *
 * This is the offline proof for the "100% of citations resolve" requirement:
 * it uses `getTextContent({disableNormalization:true})`, the same call the
 * browser makes. Run after `make_fixture_pdf.py`:
 *
 *     node ui/scripts/verify_fixtures.mjs
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const uiRoot = resolve(here, '..');
const pdfPath = resolve(uiRoot, 'public', 'fixtures', 'fixture-corpus.pdf');
const sourcePath = resolve(here, 'fixture-source.json');
const outPath = resolve(uiRoot, 'src', 'fixtures', 'citations.json');

const QUOTE_TRANSLATION = new Map([
  ['‘', "'"],
  ['’', "'"],
  ['“', '"'],
  ['”', '"'],
  ['–', '-'],
  ['—', '-'],
]);

function translate(value) {
  let out = '';
  for (const character of value) {
    out += QUOTE_TRANSLATION.get(character) ?? character;
  }
  return out;
}

function normalise(value) {
  return translate(value.normalize('NFKC'))
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

const source = JSON.parse(readFileSync(sourcePath, 'utf8'));
const data = new Uint8Array(readFileSync(pdfPath));
const loadingTask = getDocument({ data, verbosity: 0 });
const document = await loadingTask.promise;

const pages = [];
for (let pageNo = 1; pageNo <= document.numPages; pageNo += 1) {
  const page = await document.getPage(pageNo);
  const content = await page.getTextContent({ disableNormalization: true });
  let text = '';
  for (const item of content.items) {
    if (typeof item.str !== 'string') {
      continue;
    }
    text += item.str;
    if (item.hasEOL) {
      text += '\n';
    }
  }
  pages.push({ page_no: pageNo, text });
}
await loadingTask.destroy();

const normalisedPages = new Map(
  pages.map((page) => [page.page_no, normalise(page.text)]),
);

const failures = [];
for (const citation of source.citations) {
  const page = normalisedPages.get(citation.page_no);
  const needle = normalise(citation.quote);
  if (page === undefined || !page.includes(needle)) {
    failures.push(citation.id);
  }
}

if (failures.length > 0) {
  console.error(
    `FAIL: ${failures.length}/${source.citations.length} fixture quotes did not ` +
      `resolve against PDF.js text: ${failures.join(', ')}`,
  );
  process.exit(1);
}

const output = {
  doc_id: source.doc_id,
  pdf: source.pdf,
  generated_with:
    `${source.pinned_with} + pdfjs-dist getTextContent({disableNormalization:true}); ` +
    'verified by ui/scripts/verify_fixtures.mjs',
  pages,
  citations: source.citations,
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
console.log(
  `OK: ${source.citations.length}/${source.citations.length} fixture quotes resolve ` +
    `against PDF.js text; wrote ${outPath}`,
);
