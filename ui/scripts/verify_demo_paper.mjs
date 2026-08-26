/**
 * Verify every recorded demo quote against the redistributed Sentence-BERT PDF.
 *
 * This reads the PDF that ships under `ui/public/papers/`, extracts each page
 * with the same PDF.js call the browser uses, and checks every citation quote
 * against the page it names. Matching uses the citation contract's
 * normalisation: NFKC, canonical quotes, collapsed whitespace, lower case.
 *
 * Run from the repository root:
 *
 *     node ui/scripts/verify_demo_paper.mjs
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const pdfPath = resolve(repoRoot, 'ui', 'public', 'papers', 'sentence-bert.pdf');
const demoPath = resolve(repoRoot, 'ui', 'src', 'fixtures', 'real-paper.json');

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

function pageText(items) {
  let text = '';
  for (const item of items) {
    if (typeof item.str !== 'string') {
      continue;
    }
    text += item.str;
    if (item.hasEOL) {
      text += '\n';
    }
  }
  return text;
}

const demo = JSON.parse(readFileSync(demoPath, 'utf8'));
const data = new Uint8Array(readFileSync(pdfPath));
const loadingTask = getDocument({ data, verbosity: 0 });
const document = await loadingTask.promise;

const failures = [];
const pages = new Map();
for (let pageNo = 1; pageNo <= document.numPages; pageNo += 1) {
  const page = await document.getPage(pageNo);
  const content = await page.getTextContent({ disableNormalization: true });
  pages.set(pageNo, normalise(pageText(content.items)));
}

if (document.numPages !== 11) {
  failures.push(`expected 11 pages, PDF has ${document.numPages}`);
}
if (demo.page_count !== document.numPages) {
  failures.push(
    `real-paper.json says ${demo.page_count} pages, PDF has ${document.numPages}`,
  );
}
if (demo.doc_id !== 'sentence-bert' || demo.pdf !== '/papers/sentence-bert.pdf') {
  failures.push(`demo points at ${demo.doc_id} / ${demo.pdf}`);
}

let checked = 0;
let questionNumber = 0;
for (const question of demo.questions) {
  questionNumber += 1;
  if (question.answer.claims.length < 1 || question.answer.claims.length > 3) {
    failures.push(`question ${questionNumber} has ${question.answer.claims.length} claims`);
  }
  let claimNumber = 0;
  for (const claim of question.answer.claims) {
    claimNumber += 1;
    if (claim.citations.length < 1) {
      failures.push(`question ${questionNumber} claim ${claimNumber} has no citation`);
      continue;
    }
    for (const citation of claim.citations) {
      const page = pages.get(citation.page_no);
      const needle = normalise(citation.quote);
      const ok =
        page !== undefined && needle.length > 0 && page.includes(needle);
      checked += 1;
      console.log(
        `Q${questionNumber} claim ${claimNumber} | page ${citation.page_no} | ${ok ? 'OK  ' : 'FAIL'} | ${JSON.stringify(citation.quote)}`,
      );
      if (!ok) {
        failures.push(
          `question ${questionNumber} claim ${claimNumber}: quote not found on page ${citation.page_no}`,
        );
      }
    }
  }
}

await loadingTask.destroy();

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`FAIL: ${failure}`);
  }
  console.error(
    `FAILED: ${checked - failures.length}/${checked} quotes resolved; ${failures.length} problem(s)`,
  );
  process.exitCode = 1;
} else {
  console.log(
    `OK: ${demo.questions.length} questions, ${checked}/${checked} claims cited, every quote a normalised substring of its cited page; ${document.numPages} pages.`,
  );
}
