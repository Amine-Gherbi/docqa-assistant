// k6 concurrency scenario: four simultaneous askers against the local API.
// Run by scripts/verify.sh when k6 is installed; UNVERIFIED on machines that
// do not have it. The answer path is non-streaming by design, so the metric is
// full-response p95, not time-to-first-token.
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 4,
  duration: '30s',
  thresholds: {
    http_req_failed: ['rate<0.01'],
    // The CPU-only local budget is stated in docs/ENV.md; CI keeps 5000.
    http_req_duration: [`p(95)<${__ENV.K6_P95_MS || 5000}`],
  },
};

const QUESTIONS = [
  'How many attention heads does the Transformer use?',
  'What is masked language modelling used for in BERT?',
  'How does dense passage retrieval differ from BM25?',
  'What does the RAG paper retrieve before generation?',
];

export default function () {
  const question = QUESTIONS[__VU % QUESTIONS.length];
  const response = http.post(
    'http://127.0.0.1:8788/ask',
    JSON.stringify({ question }),
    { headers: { 'Content-Type': 'application/json' } },
  );
  check(response, {
    'HTTP 200': (r) => r.status === 200,
    'response has claims or a refusal': (r) => {
      const body = r.json();
      return body !== null && (Array.isArray(body.claims) || body.insufficient_evidence === true);
    },
  });
  sleep(1);
}
