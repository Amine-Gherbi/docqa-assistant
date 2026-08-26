import { useEffect, useState } from 'react';

import { EVAL_REPORT_URL } from '../api';
import galleryJson from '../fixtures/failure-gallery.json';

interface GalleryNumber {
  label: string;
  value: string;
}

interface GalleryPassage {
  chunk_id: number;
  doc_id: string;
  page_no: number;
  breadcrumb: string;
  text: string;
  score: number | null;
}

interface GalleryCase {
  id: string;
  kind: string;
  title: string;
  question: string;
  verdict: string;
  explanation: string;
  numbers: GalleryNumber[];
  passages: GalleryPassage[];
}

interface GalleryFile {
  cases: GalleryCase[];
}

const GALLERY = galleryJson as unknown as GalleryFile;

type ReportState =
  | { status: 'loading' }
  | { status: 'ready'; report: unknown }
  | { status: 'empty' }
  | { status: 'error'; message: string };

const REPORT_URLS = [EVAL_REPORT_URL, '/eval-report.json', '/api/eval/report'];

interface GateResult {
  name: string;
  label: string;
  passed: boolean;
  rate: number;
  successes: number | null;
  denominator: number | null;
  wilson: [number, number] | null;
}

interface MetricRow {
  label: string;
  value: string;
  denominator: string | null;
  ci: string | null;
}

interface StageCount {
  label: string;
  value: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function numberAt(
  record: Record<string, unknown>,
  keys: readonly string[],
): number | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function textAt(
  record: Record<string, unknown>,
  keys: readonly string[],
): string | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }
  return null;
}

function confidenceInterval(
  entry: Record<string, unknown>,
): [number, number] | null {
  for (const key of [
    'wilson_95',
    'wilson_ci',
    'ci',
    'confidence_interval',
  ]) {
    const value = entry[key];
    if (
      Array.isArray(value) &&
      typeof value[0] === 'number' &&
      typeof value[1] === 'number'
    ) {
      return [value[0], value[1]];
    }
  }
  const low = numberAt(entry, ['ci_low', 'wilson_low', 'lower']);
  const high = numberAt(entry, ['ci_high', 'wilson_high', 'upper']);
  return low !== null && high !== null ? [low, high] : null;
}

function metricRow(
  label: string,
  entry: Record<string, unknown>,
): MetricRow | null {
  const value = numberAt(entry, ['value', 'rate', 'score', 'estimate']);
  if (value === null) {
    return null;
  }
  const denominator = numberAt(entry, [
    'n',
    'denominator',
    'count',
    'total',
    'questions',
  ]);
  const ci = confidenceInterval(entry);
  return {
    label: textAt(entry, ['label', 'name']) ?? label,
    value: value.toFixed(3),
    denominator: denominator === null ? null : String(denominator),
    ci: ci === null ? null : `[${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]`,
  };
}

/** Best-effort read of a report whose exact schema is owned by the eval card. */
function metricsFrom(report: unknown): MetricRow[] {
  if (!isRecord(report)) {
    return [];
  }
  const groups = [
    'metrics',
    'rates',
    'aggregates',
    'summary',
    'overall',
    'by_stage',
    'stages',
    'groups',
  ];
  const rows: MetricRow[] = [];
  for (const groupName of groups) {
    const group = report[groupName];
    if (Array.isArray(group)) {
      group.forEach((entry, index) => {
        if (isRecord(entry)) {
          const row = metricRow(`Metric ${index + 1}`, entry);
          if (row) {
            rows.push(row);
          }
        }
      });
    } else if (isRecord(group)) {
      for (const [name, entry] of Object.entries(group)) {
        if (isRecord(entry)) {
          const row = metricRow(name, entry);
          if (row) {
            rows.push(row);
          }
        } else if (typeof entry === 'number') {
          rows.push({
            label: name,
            value: entry.toFixed(3),
            denominator: null,
            ci: null,
          });
        }
      }
    }
  }
  return rows;
}

function readable(value: string): string {
  const words = value.replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

const GATE_LABELS: Record<string, string> = {
  retrieval_recall_at_8: 'Finds the supporting passage in the top eight',
  answer_rate_answerable: 'Answers questions the document can support',
  unanswerable_refusal_rate: 'Withholds an answer when the document cannot support one',
  citation_precision: 'Citations name the supporting passage',
  faithfulness: 'Sentences stay within their cited passage',
};

function readGates(report: unknown): GateResult[] {
  if (!isRecord(report) || !Array.isArray(report.gates)) {
    return [];
  }
  const gates: GateResult[] = [];
  for (const entry of report.gates) {
    if (!isRecord(entry)) {
      continue;
    }
    const name = textAt(entry, ['name']);
    const rate = numberAt(entry, ['rate', 'value']);
    if (name === null || rate === null) {
      continue;
    }
    gates.push({
      name,
      label: GATE_LABELS[name] ?? readable(name),
      passed: entry.passed === true,
      rate,
      successes: numberAt(entry, ['successes', 'success', 'count']),
      denominator: numberAt(entry, ['n', 'denominator', 'total']),
      wilson: confidenceInterval(entry),
    });
  }
  return gates;
}

function readStageCounts(report: unknown): StageCount[] {
  if (!isRecord(report) || !isRecord(report.metrics)) {
    return [];
  }
  const counts = report.metrics.stage_failed_counts;
  if (!isRecord(counts)) {
    return [];
  }
  const labels: Record<string, string> = {
    retrieval: 'Search',
    fusion: 'Merged ranking',
    rerank: 'Rescoring',
    generation: 'Answer writing',
    schema: 'Output format',
  };
  return Object.entries(counts).flatMap(([key, value]) =>
    typeof value === 'number'
      ? [{ label: labels[key] ?? readable(key), value }]
      : [],
  );
}

function readProvenance(report: unknown): Record<string, unknown> | null {
  return isRecord(report) && isRecord(report.provenance)
    ? report.provenance
    : null;
}

function generatorInfo(
  provenance: Record<string, unknown> | null,
): { name: string | null; structuredOutput: string | null } {
  if (!provenance || !isRecord(provenance.generator)) {
    return { name: null, structuredOutput: null };
  }
  return {
    name: textAt(provenance.generator, ['name']),
    structuredOutput: textAt(provenance.generator, ['structured_output']),
  };
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatInterval(interval: [number, number] | null): string {
  return interval === null
    ? '—'
    : `[${formatPercent(interval[0])}, ${formatPercent(interval[1])}]`;
}

function formatTimestamp(value: string | null): string {
  if (value === null) {
    return '—';
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return value;
  }
  return new Intl.DateTimeFormat('en-GB', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

function formatReport(report: unknown): string {
  try {
    return JSON.stringify(report, null, 2);
  } catch {
    return String(report);
  }
}

function ReportView({ report }: { report: unknown }) {
  const gates = readGates(report);
  const metrics = metricsFrom(report);
  const stageCounts = readStageCounts(report);
  const provenance = readProvenance(report);
  const generator = generatorInfo(provenance);
  const noKey = generator.structuredOutput === 'extractive';
  const passedCount = gates.filter((gate) => gate.passed).length;
  const generatedAt = isRecord(report)
    ? textAt(report, ['generated_at', 'generatedAt'])
    : null;
  const questions =
    provenance === null
      ? null
      : numberAt(provenance, ['questions', 'question_count']);
  const answerable =
    provenance === null
      ? null
      : numberAt(provenance, ['answerable', 'answerable_questions']);
  const unanswerable =
    provenance === null
      ? null
      : numberAt(provenance, ['unanswerable', 'unanswerable_questions']);

  return (
    <>
      <div className="section-heading eval-heading">
        <div>
          <p className="eyebrow">Last run</p>
          <h2 id="report-heading">Measured results</h2>
        </div>
        {gates.length > 0 ? (
          <span
            className={
              passedCount === gates.length
                ? 'status-badge all-met'
                : 'status-badge'
            }
          >
            {passedCount} of {gates.length} checks met
          </span>
        ) : null}
      </div>

      {noKey ? (
        <div className="no-key-note">
          <span className="no-key-badge">No-key mode</span>
          <p>
            The extractive answer path produced this run because no generator
            model key was configured. The measured results below are shown
            unchanged.
          </p>
        </div>
      ) : null}

      <dl className="eval-facts">
        <div>
          <dt>Report date</dt>
          <dd>{formatTimestamp(generatedAt)}</dd>
        </div>
        <div>
          <dt>Questions</dt>
          <dd>{questions ?? '—'}</dd>
        </div>
        <div>
          <dt>Answerable</dt>
          <dd>{answerable ?? '—'}</dd>
        </div>
        <div>
          <dt>Unanswerable</dt>
          <dd>{unanswerable ?? '—'}</dd>
        </div>
      </dl>

      {gates.length > 0 ? (
        <div className="table-scroll">
          <table className="data-table eval-table">
            <thead>
              <tr>
                <th>Check</th>
                <th>Result</th>
                <th>Rate</th>
                <th>Count</th>
                <th>95% confidence range</th>
              </tr>
            </thead>
            <tbody>
              {gates.map((gate) => (
                <tr key={gate.name}>
                  <td data-label="Check">{gate.label}</td>
                  <td data-label="Result">
                    <span
                      className={
                        gate.passed ? 'chip kept' : 'chip not-met'
                      }
                    >
                      {gate.passed ? 'Met' : 'Not met'}
                    </span>
                  </td>
                  <td className="num" data-label="Rate">
                    {formatPercent(gate.rate)}
                  </td>
                  <td className="num" data-label="Count">
                    {gate.successes !== null &&
                    gate.denominator !== null
                      ? `${gate.successes}/${gate.denominator}`
                      : '—'}
                  </td>
                  <td className="num" data-label="95% confidence range">
                    {formatInterval(gate.wilson)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : metrics.length > 0 ? (
        <div className="table-scroll">
          <table className="data-table eval-table">
            <thead>
              <tr>
                <th>Rate</th>
                <th>Value</th>
                <th>Denominator</th>
                <th>Wilson 95% interval</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((row, index) => (
                <tr key={`${row.label}-${index}`}>
                  <td data-label="Rate">{readable(row.label)}</td>
                  <td className="num" data-label="Value">
                    {row.value}
                  </td>
                  <td className="num" data-label="Denominator">
                    {row.denominator ?? '—'}
                  </td>
                  <td className="num" data-label="Wilson 95% interval">
                    {row.ci ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="empty">
          The report loaded, but no rate table was recognised in its shape. The
          full report is below.
        </p>
      )}

      {stageCounts.length > 0 ? (
        <section className="stage-counts" aria-labelledby="stage-heading">
          <h3 id="stage-heading">Where questions stopped</h3>
          <dl>
            {stageCounts.map((stage) => (
              <div key={stage.label}>
                <dt>{stage.label}</dt>
                <dd>{stage.value}</dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}

      <details className="raw-report">
        <summary>Full saved report</summary>
        <pre>{formatReport(report)}</pre>
      </details>
    </>
  );
}

function Gallery() {
  return (
    <section className="gallery" aria-labelledby="gallery-heading">
      <div className="section-heading eval-heading">
        <div>
          <p className="eyebrow">Inspectable edge cases</p>
          <h2 id="gallery-heading">Edge cases</h2>
        </div>
        <span className="status-badge">{GALLERY.cases.length} cases</span>
      </div>
      <p className="gallery-intro">
        Three cases where a plausible answer would be wrong, and the pipeline
        withholds, removes, or corrects it.
      </p>
      <div className="gallery-grid">
        {GALLERY.cases.map((item) => (
          <article className="gallery-card" key={item.id}>
            <p className="gallery-kind">{item.kind}</p>
            <h3>{item.title}</h3>
            <p className="gallery-question">
              <span>Question</span>
              {item.question}
            </p>
            <p className="gallery-verdict">{item.verdict}</p>
            <p>{item.explanation}</p>
            <dl className="gallery-numbers">
              {item.numbers.map((number) => (
                <div key={number.label}>
                  <dt>{number.label}</dt>
                  <dd>{number.value}</dd>
                </div>
              ))}
            </dl>
            <ul className="gallery-passages">
              {item.passages.map((passage, index) => (
                <li key={`${item.id}-${passage.chunk_id}-${index}`}>
                  <span className="passage-meta">
                    {passage.doc_id} · page {passage.page_no} ·{' '}
                    {passage.breadcrumb}
                    {passage.score === null
                      ? ''
                      : ` · score ${passage.score.toFixed(2)}`}
                  </span>
                  <span>{passage.text}</span>
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}

export function EvalPage() {
  const [state, setState] = useState<ReportState>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    const load = async (): Promise<void> => {
      for (const url of REPORT_URLS) {
        try {
          const response = await fetch(url, {
            headers: { Accept: 'application/json' },
          });
          if (response.ok) {
            const report: unknown = await response.json();
            if (!cancelled) {
              setState({ status: 'ready', report });
            }
            return;
          }
        } catch {
          // Try the next location.
        }
      }
      if (!cancelled) {
        setState({ status: 'empty' });
      }
    };
    load().catch(() => {
      if (!cancelled) {
        setState({
          status: 'error',
          message: 'The evaluation report could not be read.',
        });
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="page eval-page">
      <header className="eval-head">
        <p className="eyebrow">Evaluation</p>
        <h1>Measure the answer, not the demo.</h1>
        <p className="lede">
          Each rate comes from the saved evaluation run, with its denominator and
          a 95% confidence range. Missing reports are reported as missing.
        </p>
      </header>

      <section className="eval-report" aria-label="Evaluation report">
        {state.status === 'loading' ? (
          <p className="empty">Looking for the saved evaluation report…</p>
        ) : null}
        {state.status === 'ready' ? (
          <ReportView report={state.report} />
        ) : null}
        {state.status === 'error' ? (
          <p className="pane-error" role="alert">
            {state.message}
          </p>
        ) : null}
        {state.status === 'empty' ? (
          <div className="empty-report">
            <h2 id="report-heading">No saved report</h2>
            <p>
              No evaluation report is being served. Run the evaluation and
              refresh this page to load the measured results.
            </p>
          </div>
        ) : null}
      </section>

      <Gallery />
    </main>
  );
}
