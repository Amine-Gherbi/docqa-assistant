import { useCallback, useRef, useState, type FormEvent } from 'react';

import { askQuestion, DEMO_ANSWER, demoMatches } from './api';
import { EvalPage } from './components/EvalPage';
import { PdfPane } from './components/PdfPane';
import { TraceDrawer } from './components/TraceDrawer';
import { FIXTURE_DOC_ID, type FixtureCheck } from './fixtures';
import type { AskResponse, Citation } from './types';

interface Selection {
  citation: Citation;
  claimIndex: number;
  attemptId: string;
}

function isEvalRoute(): boolean {
  const path = window.location.pathname.replace(/\/+$/, '');
  return path === '/eval';
}

function TopBar({ onEval }: { onEval: boolean }) {
  return (
    <header className="topbar">
      <a className="brand" href="/">
        <span className="brand-mark">P4</span>
        <span>docqa-assistant</span>
      </a>
      <nav className="nav">
        <a
          href="/"
          className={onEval ? '' : 'active'}
          aria-current={onEval ? undefined : 'page'}
        >
          Ask
        </a>
        <a
          href="/eval"
          className={onEval ? 'active' : ''}
          aria-current={onEval ? 'page' : undefined}
        >
          Eval
        </a>
      </nav>
      <span className="ports" title="Windows excludes 8001-8300 on this machine">
        API :8788 · UI :5173
      </span>
    </header>
  );
}

function show(value: number | null, digits = 3): string {
  return value === null ? '—' : value.toFixed(digits);
}

function CitationChips({
  citations,
  selection,
  onSelect,
}: {
  citations: readonly Citation[];
  selection: Selection | null;
  onSelect: (citation: Citation) => void;
}) {
  return (
    <div className="chips">
      {citations.map((citation, index) => {
        const selected =
          selection?.citation.chunk_id === citation.chunk_id &&
          selection.citation.page_no === citation.page_no &&
          selection.citation.quote === citation.quote;
        return (
          <button
            key={`${citation.chunk_id}-${citation.page_no}-${index}`}
            type="button"
            className={selected ? 'citation-chip selected' : 'citation-chip'}
            title={citation.quote}
            onClick={() => onSelect(citation)}
          >
            p.{citation.page_no}
            <span className="chip-chunk">· chunk {citation.chunk_id}</span>
          </button>
        );
      })}
    </div>
  );
}

function RefusalCard({ answer }: { answer: AskResponse }) {
  return (
    <div className="refusal" data-testid="refusal">
      <p className="refusal-kicker">Calibrated refusal</p>
      <h2>Insufficient evidence</h2>
      <p>
        {answer.reason ??
          'The calibrated probability stayed below the fitted threshold, so no answer was generated.'}
      </p>
      <dl className="refusal-numbers">
        <div>
          <dt>refusal probability</dt>
          <dd>{show(answer.refusal_probability)}</dd>
        </div>
        <div>
          <dt>threshold</dt>
          <dd>{show(answer.calibration.threshold, 4)}</dd>
        </div>
        <div>
          <dt>decision</dt>
          <dd>refused</dd>
        </div>
      </dl>
      {answer.model_reason ? (
        <p className="muted">{answer.model_reason}</p>
      ) : null}
      {answer.passages.length > 0 ? (
        <div className="competing">
          <h3>Competing passages</h3>
          <ol>
            {answer.passages.map((passage, index) => (
              <li key={`${passage.chunk_id}-${index}`}>
                <span className="passage-meta">
                  {passage.doc_id} · p.{passage.page_no} · {passage.breadcrumb}
                  {passage.score === null ? '' : ` · ${passage.score.toFixed(2)}`}
                </span>
                <span>{passage.text}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : null}
    </div>
  );
}

function AskPage() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [demoUsed, setDemoUsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [citationRenderMiss, setCitationRenderMiss] = useState(0);
  const [missDetails, setMissDetails] = useState<string[]>([]);
  const [fixtureCheck, setFixtureCheck] = useState<FixtureCheck | null>(null);

  const attemptSeq = useRef(0);
  const missedAttempts = useRef(new Set<string>());

  const handleFixtureCheck = useCallback((check: FixtureCheck) => {
    setFixtureCheck(check);
  }, []);

  const handleMiss = useCallback((attemptId: string) => {
    if (missedAttempts.current.has(attemptId)) {
      return;
    }
    missedAttempts.current.add(attemptId);
    setCitationRenderMiss((count) => count + 1);
    setMissDetails((details) => [...details, attemptId]);
  }, []);

  const selectCitation = (citation: Citation, claimIndex: number): void => {
    attemptSeq.current += 1;
    setSelection({
      citation,
      claimIndex,
      attemptId: `attempt-${attemptSeq.current}`,
    });
  };

  const presentAnswer = (next: AskResponse, fromDemo: boolean): void => {
    attemptSeq.current = 0;
    missedAttempts.current.clear();
    setCitationRenderMiss(0);
    setMissDetails([]);
    setAnswer(next);
    setDemoUsed(fromDemo);
    setError(null);
    const firstCitation = next.claims[0]?.citations[0];
    setSelection(
      firstCitation
        ? { citation: firstCitation, claimIndex: 0, attemptId: 'initial-selection' }
        : null,
    );
  };

  const run = async (text: string): Promise<void> => {
    setLoading(true);
    setError(null);
    setAnswer(null);
    setSelection(null);
    try {
      const response = await askQuestion(text);
      presentAnswer(response, false);
    } catch (cause) {
      if (demoMatches(text)) {
        presentAnswer(DEMO_ANSWER, true);
      } else {
        setError(
          cause instanceof Error
            ? cause.message
            : 'The request to the API failed.',
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAsk = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const text = question.trim();
    if (text.length === 0 || loading) {
      return;
    }
    void run(text);
  };

  const loadDemo = (): void => {
    setQuestion(DEMO_ANSWER.question);
    presentAnswer(DEMO_ANSWER, true);
  };

  const renderableCitation =
    selection && selection.citation.doc_id === FIXTURE_DOC_ID
      ? selection.citation
      : null;
  const trace = answer?.trace ?? null;

  return (
    <main className="workbench">
      <section className="answer-pane" aria-label="Answer">
        <div className="page-head">
          <p className="eyebrow">Ask the corpus</p>
          <h1>An answer where every claim points at a page.</h1>
          <p className="lede">
            Ask a question, click a claim, and the PDF pane resolves the quoted
            span. The trace drawer shows the retrieval path and the refusal
            number behind the answer.
          </p>
        </div>

        <form className="ask-form" onSubmit={handleAsk}>
          <label htmlFor="question">Question</label>
          <div className="ask-row">
            <input
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="e.g. How does hybrid retrieval find a rare term?"
              autoComplete="off"
              disabled={loading}
            />
            <button type="submit" disabled={loading || question.trim().length === 0}>
              {loading ? 'Asking…' : 'Ask'}
            </button>
          </div>
          <div className="ask-actions">
            <button type="button" className="ghost" onClick={loadDemo}>
              Load offline demo
            </button>
            <span className="muted">POST /api/ask → 127.0.0.1:8788</span>
          </div>
        </form>

        {demoUsed ? (
          <p className="demo-note" data-testid="demo-note">
            {DEMO_ANSWER.demo_note}
          </p>
        ) : null}

        {error ? (
          <div className="error-banner" role="alert">
            <strong>The API did not answer.</strong>
            <span>{error}</span>
            <span className="muted">
              Start the API on port 8788, or use the offline demo above.
            </span>
          </div>
        ) : null}

        {loading ? <p className="empty">Retrieving and validating…</p> : null}

        {!answer && !error && !loading ? (
          <div className="intro-card">
            <h2>What this shows</h2>
            <ul>
              <li>
                Hybrid retrieval: vector top-20 ⊕ lexical top-20, fused with
                RRF, reranked, then capped at eight passages.
              </li>
              <li>
                Validation: a claim without a known citation, or whose quote is
                not in the cited chunk, never reaches this pane.
              </li>
              <li>
                Rendering: the quote is found in the PDF.js text layer and
                highlighted; stored rectangles are the fallback.
              </li>
            </ul>
          </div>
        ) : null}

        {answer ? (
          <div className="answer-meta">
            <span className="chip">{answer.model}</span>
            <span className="chip">{answer.structured_output}</span>
            <span className="chip">
              refusal p = {show(answer.refusal_probability)}
            </span>
            {answer.calibration.threshold !== null ? (
              <span className="chip">
                threshold {show(answer.calibration.threshold, 4)}
              </span>
            ) : null}
            {answer.uncalibrated ? (
              <span className="chip warning">uncalibrated</span>
            ) : null}
          </div>
        ) : null}

        {answer?.insufficient_evidence ? <RefusalCard answer={answer} /> : null}

        {answer && !answer.insufficient_evidence ? (
          answer.claims.length > 0 ? (
            <ol className="claims" aria-label="Answer claims">
              {answer.claims.map((claim, claimIndex) => (
                <li
                  key={`${claimIndex}-${claim.text.slice(0, 24)}`}
                  className={
                    selection?.claimIndex === claimIndex
                      ? 'claim selected'
                      : 'claim'
                  }
                >
                  <button
                    type="button"
                    className="claim-text"
                    onClick={() => {
                      const first = claim.citations[0];
                      if (first) {
                        selectCitation(first, claimIndex);
                      }
                    }}
                  >
                    {claim.text}
                  </button>
                  <CitationChips
                    citations={claim.citations}
                    selection={selection}
                    onSelect={(citation) =>
                      selectCitation(citation, claimIndex)
                    }
                  />
                </li>
              ))}
            </ol>
          ) : (
            <div className="error-banner" role="alert">
              <strong>No claim survived validation.</strong>
              <span>
                {answer.validation.claims_returned} claim(s) returned; see the
                validation strip in the trace drawer.
              </span>
            </div>
          )
        ) : null}

        {answer ? (
          <p className="calibration-note">
            Refusal calibrated on fit set{' '}
            <code>{answer.calibration.fit_set_sha256?.slice(0, 12) ?? '—'}</code>
            {answer.calibration.uncalibrated
              ? ' · uncalibrated fallback in use'
              : ''}
          </p>
        ) : null}
      </section>

      <div className="pdf-column">
        {selection && renderableCitation === null ? (
          <p className="doc-note">
            The cited document <code>{selection.citation.doc_id}</code> is not
            the committed fixture, so this build cannot render its PDF. The
            quote still resolved server-side; <code>rects[]</code> are stored
            against the source page.
          </p>
        ) : null}
        <PdfPane
          pdfUrl="/fixtures/fixture-corpus.pdf"
          citation={renderableCitation}
          attemptId={selection?.attemptId ?? ''}
          onMiss={handleMiss}
          onFixtureCheck={handleFixtureCheck}
        />
      </div>

      <button
        type="button"
        className={drawerOpen ? 'trace-toggle open' : 'trace-toggle'}
        aria-expanded={drawerOpen}
        onClick={() => setDrawerOpen((open) => !open)}
        data-testid="trace-toggle"
      >
        {drawerOpen ? 'Hide retrieval trace' : 'Show retrieval trace'}
        {trace ? ` · ${trace.rerank_hits.length} reranked` : ''}
      </button>

      <TraceDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        answer={answer}
        citationRenderMiss={citationRenderMiss}
        fixtureCheck={fixtureCheck}
        missDetails={missDetails}
      />
    </main>
  );
}

export default function App() {
  const onEval = isEvalRoute();
  return (
    <div className="app">
      <TopBar onEval={onEval} />
      {onEval ? <EvalPage /> : <AskPage />}
    </div>
  );
}
