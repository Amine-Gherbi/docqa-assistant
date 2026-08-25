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
  | { status: 'ready'; report: unknown; source: string }
  | { status: 'empty' }
  | { status: 'error'; message: string };

const REPORT_URLS = [EVAL_REPORT_URL, '/eval-report.json', '/api/eval/report'];

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

interface MetricRow {
  label: string;
  value: string;
  denominator: string | null;
  ci: string | null;
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
          const row = metricRow(`metric ${index + 1}`, entry);
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

function headlineScalars(report: unknown): [string, string][] {
  if (!isRecord(report)) {
    return [];
  }
  return Object.entries(report)
    .filter(([, value]) => ['string', 'number', 'boolean'].includes(typeof value))
    .slice(0, 12)
    .map(([key, value]) => [key, String(value)]);
}

function formatReport(report: unknown): string {
  try {
    return JSON.stringify(report, null, 2);
  } catch {
    return String(report);
  }
}

function ReportView({ report, source }: { report: unknown; source: string }) {
  const metrics = metricsFrom(report);
  const headline = headlineScalars(report);

  return (
    <>
      <div className="eval-source">
        <span className="chip kept">report loaded</span>
        <code>{source}</code>
      </div>

      {headline.length > 0 ? (
        <dl className="headline-grid">
          {headline.map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      {metrics.length > 0 ? (
        <table className="trace-table eval-table">
          <thead>
            <tr>
              <th>metric</th>
              <th>value</th>
              <th>denominator</th>
              <th>Wilson 95% CI</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((row, index) => (
              <tr key={`${row.label}-${index}`}>
                <td>{row.label}</td>
                <td className="num">{row.value}</td>
                <td className="num">{row.denominator ?? '—'}</td>
                <td className="num">{row.ci ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="empty">
          The report loaded, but no rate table was recognised in its shape.
          The full JSON is below.
        </p>
      )}

      <details className="raw-report">
        <summary>Raw eval-report.json</summary>
        <pre>{formatReport(report)}</pre>
      </details>
    </>
  );
}

function Gallery() {
  return (
    <section className="gallery" aria-labelledby="gallery-heading">
      <header>
        <h2 id="gallery-heading">Failure gallery</h2>
        <p className="muted">
          Three cases where a naive answer looks right and the pipeline refuses,
          drops, or corrects it. Rendered from the committed fixture.
        </p>
      </header>
      <div className="gallery-grid">
        {GALLERY.cases.map((item) => (
          <article className="gallery-card" key={item.id}>
            <p className="gallery-kind">{item.kind}</p>
            <h3>{item.title}</h3>
            <p className="gallery-question">
              <strong>Q:</strong> {item.question}
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
                    {passage.doc_id} · p.{passage.page_no} · chunk{' '}
                    {passage.chunk_id}
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
          const response = await fetch(url, { headers: { Accept: 'application/json' } });
          if (response.ok) {
            const report: unknown = await response.json();
            if (!cancelled) {
              setState({ status: 'ready', report, source: url });
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
        setState({ status: 'error', message: 'The eval report could not be read.' });
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="page eval-page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Evaluation</p>
          <h1>Gates, rates, and failure cases</h1>
          <p className="lede">
            Every rate is printed with a Wilson 95% confidence interval and its
            denominator. If no report exists yet, this page says so instead of
            inventing numbers.
          </p>
        </div>
      </header>

      <section className="eval-report" aria-labelledby="report-heading">
        <h2 id="report-heading">Eval report</h2>
        {state.status === 'loading' ? (
          <p className="empty">Looking for eval-report.json…</p>
        ) : null}
        {state.status === 'ready' ? (
          <ReportView report={state.report} source={state.source} />
        ) : null}
        {state.status === 'error' ? (
          <p className="pane-error" role="alert">
            {state.message}
          </p>
        ) : null}
        {state.status === 'empty' ? (
          <div className="empty-report">
            <p>
              No eval report is being served. Run <code>uv run docqa eval</code>{' '}
              and serve the resulting <code>eval-report.json</code>; this page
              reads it from <code>{EVAL_REPORT_URL}</code> first.
            </p>
          </div>
        ) : null}
      </section>

      <Gallery />
    </main>
  );
}
