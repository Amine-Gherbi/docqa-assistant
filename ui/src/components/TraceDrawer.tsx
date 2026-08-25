import type { ReactNode } from 'react';

import type { FixtureCheck } from '../fixtures';
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
  fixtureCheck: FixtureCheck | null;
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

function Preview({ text }: { text: string }) {
  return <span className="preview">{text}</span>;
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

function VectorTable({ hits }: { hits: readonly VectorHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No vector hits recorded.</p>;
  }
  return (
    <table className="trace-table">
      <thead>
        <tr>
          <th>rank</th>
          <th>chunk</th>
          <th>page</th>
          <th>cosine distance</th>
          <th>passage</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((hit) => (
          <tr key={`v-${hit.chunk_id}`}>
            <td>{hit.rank}</td>
            <td>{hit.chunk_id}</td>
            <td>{hit.page_no}</td>
            <td className="num">{hit.cosine_distance.toFixed(4)}</td>
            <td>
              <Preview text={hit.preview} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function LexicalTable({ hits }: { hits: readonly LexicalHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No lexical hits recorded.</p>;
  }
  return (
    <table className="trace-table">
      <thead>
        <tr>
          <th>rank</th>
          <th>chunk</th>
          <th>page</th>
          <th>ts_rank_cd</th>
          <th>passage</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((hit) => (
          <tr key={`l-${hit.chunk_id}`}>
            <td>{hit.rank}</td>
            <td>{hit.chunk_id}</td>
            <td>{hit.page_no}</td>
            <td className="num">{hit.ts_rank_cd.toFixed(4)}</td>
            <td>
              <Preview text={hit.preview} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FusionTable({ hits }: { hits: readonly FusionHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No fused list recorded.</p>;
  }
  return (
    <table className="trace-table">
      <thead>
        <tr>
          <th>RRF rank</th>
          <th>RRF score</th>
          <th>vector rank</th>
          <th>lexical rank</th>
          <th>rerank rank</th>
          <th>Δ rank</th>
          <th>passage</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((hit) => (
          <tr key={`f-${hit.chunk_id}`}>
            <td>{hit.rrf_rank}</td>
            <td className="num">{hit.rrf_score.toFixed(5)}</td>
            <td>{hit.vector_rank ?? <NullCell />}</td>
            <td>{hit.lexical_rank ?? <NullCell />}</td>
            <td>{hit.rerank_rank ?? <NullCell />}</td>
            <td className="num">{rankDelta(hit.rank_delta)}</td>
            <td>
              <Preview text={hit.preview} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RerankTable({ hits }: { hits: readonly RerankHit[] }) {
  if (hits.length === 0) {
    return <p className="empty">No reranked candidates recorded.</p>;
  }
  return (
    <table className="trace-table">
      <thead>
        <tr>
          <th>rerank rank</th>
          <th>rerank score</th>
          <th>calibrated probability</th>
          <th>kept</th>
          <th>RRF rank</th>
          <th>passage</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((hit) => (
          <tr key={`r-${hit.chunk_id}`}>
            <td>{hit.rerank_rank}</td>
            <td className="num">{hit.rerank_score.toFixed(3)}</td>
            <td className="num">{probability(hit.calibrated_probability)}</td>
            <td>
              <span className={hit.kept ? 'chip kept' : 'chip dropped'}>
                {hit.kept ? 'kept' : 'dropped'}
              </span>
            </td>
            <td>{hit.rrf_rank}</td>
            <td>
              <Preview text={hit.preview} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FinalContext({
  trace,
  citationRenderMiss,
  fixtureCheck,
  missDetails,
}: {
  trace: RetrievalTrace;
  citationRenderMiss: number;
  fixtureCheck: FixtureCheck | null;
  missDetails: readonly string[];
}) {
  return (
    <div className="final-context">
      <dl>
        <div>
          <dt>context tokens</dt>
          <dd className="num">{trace.context_tokens}</dd>
        </div>
        <div>
          <dt>context chunks</dt>
          <dd className="num">{trace.context_chunk_ids.join(', ') || '—'}</dd>
        </div>
        <div>
          <dt>decision</dt>
          <dd>
            <span className={`chip ${trace.refusal.decision}`}>
              {trace.refusal.decision}
            </span>
          </dd>
        </div>
        <div>
          <dt>top rerank score</dt>
          <dd className="num">{format(trace.refusal.top_rerank_score)}</dd>
        </div>
        <div>
          <dt>top calibrated probability</dt>
          <dd className="num">{probability(trace.refusal.top_probability)}</dd>
        </div>
        <div>
          <dt>refusal threshold</dt>
          <dd className="num">
            {format(trace.refusal.threshold, 4)}
            {trace.refusal.uncalibrated ? ' (uncalibrated)' : ''}
          </dd>
        </div>
        <div>
          <dt>refusal reason</dt>
          <dd>{trace.refusal.reason ?? '—'}</dd>
        </div>
        <div>
          <dt>citation_render_miss</dt>
          <dd className="num" data-testid="citation-render-miss">
            {citationRenderMiss}
          </dd>
        </div>
        <div>
          <dt>fixture citations</dt>
          <dd data-testid="fixture-check">
            {fixtureCheck
              ? `${fixtureCheck.resolved}/${fixtureCheck.total} resolve`
              : 'checking…'}
          </dd>
        </div>
      </dl>
      {fixtureCheck && fixtureCheck.misses.length > 0 ? (
        <ul className="miss-list">
          {fixtureCheck.misses.map((miss) => (
            <li key={miss.id}>
              {miss.id} (p.{miss.page_no}) did not resolve: “{miss.quote}”
            </li>
          ))}
        </ul>
      ) : null}
      {missDetails.length > 0 ? (
        <p className="miss-note">
          {missDetails.length} citation render attempt
          {missDetails.length === 1 ? '' : 's'} missed and used the rectangle
          fallback.
        </p>
      ) : null}
    </div>
  );
}

/** The five-stage retrieval trace plus the client-side render accounting. */
export function TraceDrawer({
  open,
  onClose,
  answer,
  citationRenderMiss,
  fixtureCheck,
  missDetails,
}: TraceDrawerProps) {
  if (!open) {
    return null;
  }
  const trace = answer?.trace ?? null;

  return (
    <aside className="trace-drawer" aria-label="Retrieval trace">
      <header className="drawer-head">
        <div>
          <h2>Retrieval trace</h2>
          <p className="muted">
            {trace
              ? `${trace.mode} · ${trace.embedder} · ${trace.reranker}`
              : 'No trace on this response.'}
          </p>
        </div>
        <button type="button" className="ghost" onClick={onClose}>
          Close
        </button>
      </header>

      {answer ? (
        <div className="validation-strip">
          <span>
            claims {answer.validation.claims_kept}/{answer.validation.claims_returned} kept
          </span>
          <span>dropped quote mismatch {answer.validation.dropped_quote_mismatch}</span>
          <span>dropped unknown citation {answer.validation.dropped_unknown_citation}</span>
          <span>schema failures {answer.validation.schema_failures}</span>
          <span>repair turns {answer.validation.repair_turns}</span>
          <span>structured output {answer.structured_output}</span>
        </div>
      ) : null}

      <div className="drawer-body">
        {!trace ? (
          <p className="empty">
            Ask a question to inspect vector hits, lexical hits, RRF fusion,
            reranking, and the final context.
          </p>
        ) : (
          <>
            <Section
              index="1"
              title="Vector retrieval"
              detail="Cosine distance over pgvector HNSW; top 20 by distance."
            >
              <VectorTable hits={trace.vector_hits} />
            </Section>
            <Section
              index="2"
              title="Lexical retrieval"
              detail="ts_rank_cd over the GIN index; top 20 by rank."
            >
              <LexicalTable hits={trace.lexical_hits} />
            </Section>
            <Section
              index="3"
              title="Reciprocal rank fusion"
              detail="RRF k=60; rank_delta is rerank rank minus RRF rank."
            >
              <FusionTable hits={trace.fusion_hits} />
            </Section>
            <Section
              index="4"
              title="Cross-encoder rerank"
              detail="ms-marco candidate scores and calibrated probabilities; kept by threshold."
            >
              <RerankTable hits={trace.rerank_hits} />
            </Section>
            <Section
              index="5"
              title="Final context and refusal"
              detail="The passages sent to the generator, the token count, and the refusal number."
            >
              <FinalContext
                trace={trace}
                citationRenderMiss={citationRenderMiss}
                fixtureCheck={fixtureCheck}
                missDetails={missDetails}
              />
            </Section>
          </>
        )}
      </div>
    </aside>
  );
}
