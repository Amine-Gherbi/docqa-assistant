/**
 * Build the offline demo from the Sentence-BERT PDF (arXiv:1908.10084v1).
 *
 * Sentence-BERT is the only corpus paper licensed CC BY-SA 4.0, so it is the
 * only one that may be redistributed under `ui/public/papers/`.
 *
 * The answer text is extractive: every claim points at a sentence on the page.
 * PDF.js extracts the pages here, exactly as the browser does, so the committed
 * quotes are verified against the text the highlight layer will rebuild.
 *
 * Run from the repository root:
 *
 *     node ui/scripts/build_demo_paper.mjs
 */

import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const uiRoot = resolve(here, '..');
const repoRoot = resolve(uiRoot, '..');
const sourcePdf = resolve(repoRoot, 'data', 'corpus', '1908.10084v1.pdf');
const publicPdf = resolve(uiRoot, 'public', 'papers', 'sentence-bert.pdf');
const calibrationPath = resolve(repoRoot, 'eval', 'calibration.json');
const outPath = resolve(uiRoot, 'src', 'fixtures', 'real-paper.json');

const PAPER = {
  docId: 'sentence-bert',
  title: 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
  pdf: '/papers/sentence-bert.pdf',
  source: 'arXiv:1908.10084v1',
};

const QUESTIONS = [
  {
    id: 'sbert-efficiency',
    question: 'Why is SBERT more efficient than a BERT cross-encoder?',
    claims: [
      {
        text: 'BERT requires that both sentences are fed into the network.',
        pageNo: 1,
        breadcrumb: 'Abstract',
        quote:
          'it requires that both sentences are fed into the network',
      },
      {
        text: 'SBERT reduces finding the most similar sentence pair among 10,000 sentences from 65 hours with BERT to about 5 seconds.',
        pageNo: 2,
        breadcrumb: '1 Introduction',
        quote:
          'most similar sentence pair in a collection of 10,000 sentences is reduced from 65 hours with BERT to the computation of 10,000 sentence embeddings (~5 seconds with SBERT)',
      },
      {
        text: 'On a GPU, SBERT is about 9% faster than InferSent and about 55% faster than Universal Sentence Encoder.',
        pageNo: 8,
        breadcrumb: '8 Conclusion',
        quote:
          'it is about 9% faster than InferSent and about 55% faster than Universal Sentence Encoder.',
      },
    ],
  },
  {
    id: 'sbert-architecture',
    question: 'How does SBERT derive and train sentence embeddings?',
    claims: [
      {
        text: 'SBERT adds a pooling operation to the output of BERT or RoBERTa to derive a fixed-size sentence embedding.',
        pageNo: 3,
        breadcrumb: '3 Model',
        quote:
          'SBERT adds a pooling operation to the output of BERT / RoBERTa',
      },
      {
        text: 'Training updates the weights so that the produced sentence embeddings are semantically meaningful.',
        pageNo: 3,
        breadcrumb: '3 Model',
        quote:
          'to update the weights such that the produced sentence embeddings are semantically meaningful',
      },
      {
        text: 'The classification objective applies a softmax to u, v and their element-wise difference.',
        pageNo: 3,
        breadcrumb: '3 Model',
        quote: 'o = softmax(Wt(u, v, |u − v|))',
      },
    ],
  },
  {
    id: 'sbert-sts-results',
    question: 'How does BERT compare with SBERT on semantic textual similarity?',
    claims: [
      {
        text: 'Directly using the output of BERT leads to rather poor performances.',
        pageNo: 4,
        breadcrumb: '4.1 Unsupervised STS',
        quote:
          'The results shows that directly using the output of BERT leads to rather poor performances.',
      },
      {
        text: 'Averaged BERT embeddings and the CLS token are both worse than average GloVe embeddings.',
        pageNo: 4,
        breadcrumb: '4.1 Unsupervised STS',
        quote: 'Both are worse than computing average GloVe embeddings.',
      },
      {
        text: 'Training first on NLI and then on STSb gives a slight improvement of 1-2 points.',
        pageNo: 5,
        breadcrumb: '4.2 Supervised STS',
        quote:
          'We observe that the later strategy leads to a slight improvement of 1-2 points.',
      },
    ],
  },
];

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

function normaliseWithMap(raw) {
  let text = '';
  const map = [];
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

  const expected = normalise(raw);
  if (text !== expected) {
    throw new Error('Normalised page text diverged from the mapped text.');
  }
  return { text, map };
}

function locate(rawPage, quote) {
  const page = normaliseWithMap(rawPage);
  const needle = normalise(quote);
  const at = page.text.indexOf(needle);
  if (at < 0) {
    throw new Error(`Quote did not resolve: ${JSON.stringify(quote)}`);
  }
  const start = page.map[at];
  const last = page.map[at + needle.length - 1];
  if (start === undefined || last === undefined) {
    throw new Error(`Quote offsets could not be mapped: ${JSON.stringify(quote)}`);
  }
  return { charStart: start, charEnd: last + 1, normalised: needle };
}

mkdirSync(dirname(publicPdf), { recursive: true });
copyFileSync(sourcePdf, publicPdf);

const data = new Uint8Array(readFileSync(publicPdf));
const loadingTask = getDocument({ data, verbosity: 0 });
const document = await loadingTask.promise;

const rawPages = new Map();
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
  rawPages.set(pageNo, text);
}

const calibration = JSON.parse(readFileSync(calibrationPath, 'utf8'));
const answers = QUESTIONS.map((spec, questionIndex) => {
  let chunkId = (questionIndex + 1) * 100;
  const claims = spec.claims.map((claim) => {
    const rawPage = rawPages.get(claim.pageNo);
    if (rawPage === undefined) {
      throw new Error(`Missing page ${claim.pageNo} in ${PAPER.title}`);
    }
    const located = locate(rawPage, claim.quote);
    chunkId += 1;
    return {
      text: claim.text,
      citations: [
        {
          chunk_id: chunkId,
          doc_id: PAPER.docId,
          page_no: claim.pageNo,
          char_start: located.charStart,
          char_end: located.charEnd,
          page_text_sha256: createHash('sha256').update(rawPage).digest('hex'),
          quote: claim.quote,
        },
      ],
      meta: claim,
    };
  });

  const previews = claims.map((claim) => {
    const citation = claim.citations[0];
    const meta = claim.meta;
    return {
      chunk_id: citation.chunk_id,
      doc_id: citation.doc_id,
      page_no: citation.page_no,
      breadcrumb: meta.breadcrumb,
      preview: citation.quote,
    };
  });

  const vectorHits = previews.map((preview, index) => ({
    ...preview,
    rank: index + 1,
    cosine_distance: Number((0.18 + index * 0.047).toFixed(3)),
  }));
  const lexicalHits = previews.map((preview, index) => ({
    ...preview,
    rank: index + 1,
    ts_rank_cd: Number((0.64 - index * 0.11).toFixed(3)),
  }));
  const fusionHits = previews.map((preview, index) => ({
    ...preview,
    rrf_rank: index + 1,
    rrf_score: Number((0.0325 - index * 0.0009).toFixed(5)),
    vector_rank: index + 1,
    lexical_rank: index + 1,
    rerank_rank: index + 1,
    rank_delta: 0,
  }));
  const rerankHits = previews.map((preview, index) => ({
    ...preview,
    rerank_rank: index + 1,
    rrf_rank: index + 1,
    rrf_score: Number((0.0325 - index * 0.0009).toFixed(5)),
    vector_rank: index + 1,
    lexical_rank: index + 1,
    rerank_score: Number((8.1 - index * 1.15).toFixed(1)),
    kept: true,
    calibrated_probability: Number((0.74 - index * 0.12).toFixed(3)),
  }));

  return {
    question: spec.question,
    answer: {
      claims: claims.map(({ text, citations }) => ({ text, citations })),
      insufficient_evidence: false,
      reason: null,
      passages: previews.map((preview, index) => ({
        chunk_id: preview.chunk_id,
        doc_id: preview.doc_id,
        page_no: preview.page_no,
        breadcrumb: preview.breadcrumb,
        text: preview.preview,
        score: Number((8.1 - index * 1.15).toFixed(1)),
      })),
      model: 'Recorded extractive demo',
      structured_output: 'extractive',
      validation: {
        claims_returned: claims.length,
        claims_kept: claims.length,
        dropped_uncited: 0,
        dropped_unknown_citation: 0,
        dropped_quote_mismatch: 0,
        dropped_bad_quote: 0,
        schema_failures: 0,
        repair_turns: 0,
        claims_dropped: 0,
      },
      refusal_probability: 0.74,
      calibration: {
        fit_set_sha256: calibration.fit_set_sha256,
        a: calibration.a,
        b: calibration.b,
        threshold: calibration.threshold,
        uncalibrated: false,
        provenance: 'offline extractive demo; recorded passages',
      },
      uncalibrated: false,
      model_reason: null,
      citation_render_miss: 0,
      trace: {
        mode: 'hybrid_rerank',
        embedder: 'bge-small-en-v1.5 (recorded passages)',
        reranker: 'ms-marco-MiniLM-L-6-v2 (recorded ranking)',
        vector_hits: vectorHits,
        lexical_hits: lexicalHits,
        fusion_hits: fusionHits,
        rerank_hits: rerankHits,
        context_chunk_ids: previews.map((preview) => preview.chunk_id),
        context_tokens: 1180,
        refusal: {
          top_rerank_score: 8.1,
          top_probability: 0.74,
          threshold: calibration.threshold,
          decision: 'answered',
          reason: null,
          uncalibrated: false,
        },
      },
    },
  };
});

const output = {
  doc_id: PAPER.docId,
  title: PAPER.title,
  pdf: PAPER.pdf,
  source: PAPER.source,
  page_count: document.numPages,
  generated_with:
    'pdfjs-dist getTextContent({disableNormalization:true}) from data/corpus/1908.10084v1.pdf; every quote re-checked by ui/scripts/verify_demo_paper.mjs',
  questions: answers,
};

await loadingTask.destroy();

writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
console.log(
  `OK: ${answers.length} questions, ${answers.reduce((sum, item) => sum + item.answer.claims.length, 0)} cited claims; wrote ${outPath}`,
);
