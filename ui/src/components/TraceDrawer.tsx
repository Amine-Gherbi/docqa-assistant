import { useEffect, type ReactNode } from 'react';

import type {
  AskResponse,
  FusionHit,
  LexicalHit,
  RerankHit,
  RetrievalTrace,
  VectorHit,
} from '../types';

interface TraceDrawerProps {
  open: boolean;
  onClose: () => void;
  answer: AskResponse | null;
  citationRenderMiss: number;
  missDetails: readonly string[];
}

function Section({
  index,
  title,
  detail,
  children,
}: {
  index: string;
  title: string;
  detail: string;
  children: ReactNode;
}) {
  return (
    <section className="trace-section">
      <header>
        <span className="trace-index">{index}</span>
        <div>
          <h3>{title}</h3>
          <p>{detail}</p>
        </div>
      </header>
      {children}
    </section>
  );
}

function Preview({ text, breadcrumb }: { text: string; breadcrumb: string }) {
  return (
    <span className="preview">
      <span className="preview-breadcrumb">{breadcrumb}</span>
      {text}
    </span>
  );
}

function NullCell() {
  return <span className="null">—</span>;
}

function format(value: number | null, digits = 3): string {
  return value === null ? '—' : value.toFixed(digits);
}

function probability(value: number | null): string {
  return value === null ? '—' : value.toFixed(3);
}

function rankDelta(delta: number | null): string {
  if (delta === null) {
    return '—';
  }
  if (delta === 0) {
    return '0';
  }
  return delta > 0 ? `+${delta}` : `${delta}`;
}

function outputLabel(value: AskResponse['structured_output']): string {
  switch (value) {
    case 'extractive':
      return 'No-key extraction';
    case 'tool':
      return 'Structured tool call';
    case 'json':
      return 'JSON response';
  }
}

function decisionLabel(value: 'answered' | 'refused'): string {
  return value === 'answered' ? 'Answered' : 'Withheld';
}

function reasonLabel(reason: string | null): string {
  if (reason === null) {
    return '—';
  }
  const labels: Record<string, string> = {
    low_score: 'Best passage below the threshold',
    no_context: 'No passages retrieved',
    uncalibrated: 'Calibration unavailable',
  };
  return labels[reason] ?? reason;
}

function VectorTable({ hits }: { hits: readonly VectorHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No meaning-based matches were recorded.</p>;
  }
  return (
    <div className="table-scroll">
      <table className="trace-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Document</th>
            <th>Page</th>
            <th>Distance</th>
            <th>Passage</th>
          </tr>
        </thead>
        <tbody>
          {hits.map((hit) => (
            <tr key={`v-${hit.chunk_id}`}>
              <td>{hit.rank}</td>
              <td>{hit.doc_id}</td>
              <td>{hit.page_no}</td>
              <td className="num">{hit.cosine_distance.toFixed(4)}</td>
              <td>
                <Preview text={hit.preview} breadcrumb={hit.breadcrumb} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LexicalTable({ hits }: { hits: readonly LexicalHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No keyword matches were recorded.</p>;
  }
  return (
    <div className="table-scroll">
      <table className="trace-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Document</th>
            <th>Page</th>
            <th>Keyword score</th>
            <th>Passage</th>
          </tr>
        </thead>
        <tbody>
          {hits.map((hit) => (
            <tr key={`l-${hit.chunk_id}`}>
              <td>{hit.rank}</td>
              <td>{hit.doc_id}</td>
              <td>{hit.page_no}</td>
              <td className="num">{hit.ts_rank_cd.toFixed(4)}</td>
              <td>
                <Preview text={hit.preview} breadcrumb={hit.breadcrumb} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FusionTable({ hits }: { hits: readonly FusionHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No merged ranking was recorded.</p>;
  }
  return (
    <div className="table-scroll">
      <table className="trace-table">
        <thead>
          <tr>
            <th>Merged rank</th>
            <th>Merged score</th>
            <th>Document</th>
            <th>Page</th>
            <th>Meaning rank</th>
            <th>Keyword rank</th>
            <th>Second-pass rank</th>
            <th>Change</th>
            <th>Passage</th>
          </tr>
        </thead>
        <tbody>
          {hits.map((hit) => (
            <tr key={`f-${hit.chunk_id}`}>
              <td>{hit.rrf_rank}</td>
              <td className="num">{hit.rrf_score.toFixed(5)}</td>
              <td>{hit.doc_id}</td>
              <td>{hit.page_no}</td>
              <td>{hit.vector_rank ?? <NullCell />}</td>
              <td>{hit.lexical_rank ?? <NullCell />}</td>
              <td>{hit.rerank_rank ?? <NullCell />}</td>
              <td className="num">{rankDelta(hit.rank_delta)}</td>
              <td>
                <Preview text={hit.preview} breadcrumb={hit.breadcrumb} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RerankTable({ hits }: { hits: readonly RerankHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No rescored candidates were recorded.</p>;
  }
  return (
    <div className="table-scroll">
      <table className="trace-table">
        <thead>
          <tr>
            <th>Second-pass rank</th>
            <th>Document</th>
            <th>Page</th>
            <th>Score</th>
            <th>Evidence probability</th>
            <th>Status</th>
            <th>Merged rank</th>
            <th>Passage</th>
          </tr>
        </thead>
        <tbody>
          {hits.map((hit) => (
            <tr key={`r-${hit.chunk_id}`}>
              <td>{hit.rerank_rank}</td>
              <td>{hit.doc_id}</td>
              <td>{hit.page_no}</td>
              <td className="num">{hit.rerank_score.toFixed(3)}</td>
              <td className="num">{probability(hit.calibrated_probability)}</td>
              <td>
                <span className={hit.kept ? 'chip kept' : 'chip dropped'}>
                  {hit.kept ? 'Included' : 'Excluded'}
                </span>
              </td>
              <td>{hit.rrf_rank}</td>
              <td>
                <Preview text={hit.preview} breadcrumb={hit.breadcrumb} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FinalContext({
  trace,
  citationRenderMiss,
  missDetails,
}: {
  trace: RetrievalTrace;
  citationRenderMiss: number;
  missDetails: readonly string[];
}) {
  return (
    <div className="final-context">
      <dl>
        <div>
          <dt>Passages sent</dt>
          <dd className="num">{trace.context_chunk_ids.length}</dd>
        </div>
        <div>
          <dt>Context tokens</dt>
          <dd className="num">{trace.context_tokens}</dd>
        </div>
        <div>
          <dt>Decision</dt>
          <dd>
            <span className={`chip ${trace.refusal.decision}`}>
              {decisionLabel(trace.refusal.decision)}
            </span>
          </dd>
        </div>
        <div>
          <dt>Best rescored passage</dt>
          <dd className="num">{format(trace.refusal.top_rerank_score)}</dd>
        </div>
        <div>
          <dt>Top evidence probability</dt>
          <dd className="num">{probability(trace.refusal.top_probability)}</dd>
        </div>
        <div>
          <dt>Decision threshold</dt>
          <dd className="num">
            {format(trace.refusal.threshold, 4)}
            {trace.refusal.uncalibrated ? ' (uncalibrated)' : ''}
          </dd>
        </div>
        <div>
          <dt>Reason</dt>
          <dd>{reasonLabel(trace.refusal.reason)}</dd>
        </div>
        <div>
          <dt>Highlight fallbacks</dt>
          <dd className="num" data-testid="citation-render-miss">
            {citationRenderMiss}
          </dd>
        </div>
      </dl>
      {missDetails.length > 0 ? (
        <p className="miss-note">
          {missDetails.length} citation highlight
          {missDetails.length === 1 ? '' : 's'} fell back to the stored page
          rectangle{missDetails.length === 1 ? '' : 's'}.
        </p>
      ) : null}
    </div>
  );
}

/** The five-stage retrieval trail plus the client-side highlight accounting. */
export function TraceDrawer({
  open,
  onClose,
  answer,
  citationRenderMiss,
  missDetails,
}: TraceDrawerProps) {
  const trace = answer?.trace ?? null;

  useEffect(() => {
    if (!open) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <aside
      className="trace-drawer"
      aria-labelledby="retrieval-trail-title"
      role="dialog"
      aria-modal="false"
    >
      <header className="drawer-head">
        <div>
          <p className="eyebrow">Inspectable by default</p>
          <h2 id="retrieval-trail-title">Retrieval trail</h2>
          <p className="drawer-detail">
            {trace
              ? 'Meaning search, keyword search, merged ranking, rescoring, and the evidence check.'
              : 'Run a question to inspect the passages and the decision.'}
          </p>
        </div>
        <button type="button" className="secondary" onClick={onClose}>
          Close
        </button>
      </header>

      {answer ? (
        <div className="validation-strip">
          <span>
            {answer.validation.claims_kept} of{' '}
            {answer.validation.claims_returned} claims passed
          </span>
          <span>
            {answer.validation.dropped_quote_mismatch} removed for a quote
            mismatch
          </span>
          <span>
            {answer.validation.dropped_unknown_citation} removed for an unknown
            source
          </span>
          <span>{answer.validation.schema_failures} format errors</span>
          <span>{answer.validation.repair_turns} repair attempts</span>
          <span>Answer type: {outputLabel(answer.structured_output)}</span>
          {trace ? (
            <span>
              Models: {trace.embedder} · {trace.reranker}
            </span>
          ) : null}
        </div>
      ) : null}

      <div className="drawer-body">
        {!trace ? (
          <p className="empty">
            Ask a question to see the passages found, how they were ranked, and
            why the answer was written or withheld.
          </p>
        ) : (
          <>
            <Section
              index="1"
              title="Meaning search"
              detail="Passages whose meaning is closest to the question."
            >
              <VectorTable hits={trace.vector_hits} />
            </Section>
            <Section
              index="2"
              title="Keyword search"
              detail="Passages that contain the question's exact words."
            >
              <LexicalTable hits={trace.lexical_hits} />
            </Section>
            <Section
              index="3"
              title="Merged ranking"
              detail="The two lists combined by rank, not by score."
            >
              <FusionTable hits={trace.fusion_hits} />
            </Section>
            <Section
              index="4"
              title="Second-pass ranking"
              detail="Each candidate scored again as a question and passage pair."
            >
              <RerankTable hits={trace.rerank_hits} />
            </Section>
            <Section
              index="5"
              title="Final context and decision"
              detail="The passages sent to the answer and the evidence check."
            >
              <FinalContext
                trace={trace}
                citationRenderMiss={citationRenderMiss}
                missDetails={missDetails}
              />
            </Section>
          </>
        )}
      </div>
    </aside>
  );
}
