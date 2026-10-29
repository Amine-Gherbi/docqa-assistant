import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react';

import {
  askQuestionWithWarmup,
  uploadStatus,
  uploadedPdfUrl,
  warmApi,
} from './api';
import { EvalPage } from './components/EvalPage';
import { PdfPane } from './components/PdfPane';
import { TraceDrawer } from './components/TraceDrawer';
import { UploadPanel } from './components/UploadPanel';
import {
  DEFAULT_DEMO_QUESTION,
  DEMO_PAPER,
  DEMO_QUESTIONS,
  demoForQuestion,
  type DemoQuestion,
} from './demoPaper';
import { formatQuoted } from './typography';
import type { AskResponse, Citation, UploadDocument } from './types';

interface Selection {
  citation: Citation;
  attemptId: string;
}

interface SourceEntry {
  number: number;
  citation: Citation;
}

function isEvalRoute(): boolean {
  const path = window.location.pathname.replace(/\/+$/, '');
  return path === '/eval';
}

function citationKey(citation: Citation): string {
  return `${citation.chunk_id}|${citation.page_no}|${citation.quote}`;
}

function sourceLabel(docId: string, upload: UploadDocument | null): string {
  if (upload && docId === `upload:${upload.collection}`) {
    return upload.filename;
  }
  return docId === DEMO_PAPER.doc_id ? DEMO_PAPER.title : docId;
}

function uploadDocumentId(upload: UploadDocument): string {
  return `upload:${upload.collection}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function splitTrailingWord(text: string): { lead: string; tail: string } {
  const match = /(^|\s)(\S+)\s*$/.exec(text);
  if (!match) {
    return { lead: '', tail: text };
  }
  const tailStart = match.index + match[1]!.length;
  return { lead: text.slice(0, tailStart), tail: text.slice(tailStart) };
}

function collectSources(answer: AskResponse | null): SourceEntry[] {
  if (!answer) {
    return [];
  }
  const seen = new Map<string, SourceEntry>();
  for (const claim of answer.claims) {
    for (const citation of claim.citations) {
      const key = citationKey(citation);
      if (!seen.has(key)) {
        seen.set(key, { number: seen.size + 1, citation });
      }
    }
  }
  return [...seen.values()];
}

function SiteHeader({ onEval }: { onEval: boolean }) {
  return (
    <header className="site-header">
      <a className="brand" href="/" aria-label="Marginalia home">
        Marginalia
      </a>
      <p className="header-line">
        Answers from your PDFs, with the page and passage behind every sentence.
      </p>
      <nav className="site-nav" aria-label="Primary">
        <a
          href="/"
          className={onEval ? undefined : 'active'}
          aria-current={onEval ? undefined : 'page'}
        >
          Ask
        </a>
        <a
          href="/eval"
          className={onEval ? 'active' : undefined}
          aria-current={onEval ? 'page' : undefined}
        >
          Evaluation
        </a>
      </nav>
    </header>
  );
}

function ClaimText({
  text,
  children,
}: {
  text: string;
  children: ReactNode;
}) {
  const { lead, tail } = splitTrailingWord(text);
  return (
    <p className="claim-text">
      {lead}
      <span className="claim-tail">
        {tail}
        {children}
      </span>
    </p>
  );
}

function CitationMark({
  citation,
  number,
  selected,
  sourceName,
  onSelect,
}: {
  citation: Citation;
  number: number;
  selected: boolean;
  sourceName: string;
  onSelect: () => void;
}) {
  return (
    <sup className="claim-citation">
      <button
        type="button"
        className={selected ? 'citation-sup selected' : 'citation-sup'}
        aria-label={`Source ${number}: ${sourceName}, page ${citation.page_no}`}
        aria-pressed={selected}
        title={citation.quote}
        data-testid="claim-citation"
        onClick={onSelect}
      >
        {number}
      </button>
    </sup>
  );
}

function Sources({
  sources,
  selectedKey,
  labelFor,
  onSelect,
}: {
  sources: readonly SourceEntry[];
  selectedKey: string | null;
  labelFor: (docId: string) => string;
  onSelect: (citation: Citation) => void;
}) {
  if (sources.length === 0) {
    return null;
  }
  return (
    <section className="sources" aria-labelledby="sources-heading">
      <h3 id="sources-heading">Sources</h3>
      <ol className="source-list">
        {sources.map(({ number, citation }) => {
          const selected = selectedKey === citationKey(citation);
          return (
            <li key={citationKey(citation)}>
              <button
                type="button"
                className={selected ? 'source-row selected' : 'source-row'}
                aria-label={`Source ${number}: ${labelFor(citation.doc_id)}, page ${citation.page_no}`}
                aria-pressed={selected}
                data-testid="source-row"
                onClick={() => onSelect(citation)}
              >
                <span className="source-number">{number}</span>
                <span className="source-copy">
                  <span className="source-meta">
                    <span className="source-file">
                      {labelFor(citation.doc_id)}
                    </span>
                    <span aria-hidden="true">·</span>
                    <span className="source-page">
                      Page {citation.page_no}
                    </span>
                  </span>
                  <span className="source-quote" title={citation.quote}>
                    {formatQuoted(citation.quote)}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function ExampleQuestions({
  onChoose,
}: {
  onChoose: (example: DemoQuestion) => void;
}) {
  return (
    <section className="examples" aria-labelledby="examples-heading">
      <h3 id="examples-heading" className="examples-heading">
        Example questions
      </h3>
      <ul className="example-list">
        {DEMO_QUESTIONS.map((example) => (
          <li key={example.question}>
            <button
              type="button"
              className="example-question"
              data-testid="example-question"
              onClick={() => onChoose(example)}
            >
              {example.question}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function refusalCopy(reason: string | null): string {
  switch (reason) {
    case 'no_context':
      return 'No passage in the document matched the question closely enough to support an answer.';
    case 'uncalibrated':
      return 'The evidence threshold was unavailable, so the answer was withheld.';
    case 'low_score':
      return 'The best passage scored below the calibrated threshold, so the answer was withheld.';
    default:
      return 'The retrieved passages did not support a reliable answer, so it was withheld.';
  }
}

function RefusalCard({
  answer,
  labelFor,
}: {
  answer: AskResponse;
  labelFor: (docId: string) => string;
}) {
  return (
    <div className="refusal" data-testid="refusal">
      <p className="eyebrow">No supported answer</p>
      <h3>The document does not settle this question.</h3>
      <p>{refusalCopy(answer.reason)}</p>
      {answer.passages.length > 0 ? (
        <div className="considered">
          <h4>Passages considered</h4>
          <ol>
            {answer.passages.map((passage, index) => (
              <li key={`${passage.chunk_id}-${index}`}>
                <span className="passage-meta">
                  {labelFor(passage.doc_id)} · page {passage.page_no} ·{' '}
                  {passage.breadcrumb}
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
  const [question, setQuestion] = useState(DEFAULT_DEMO_QUESTION.question);
  const [answer, setAnswer] = useState<AskResponse | null>(
    DEFAULT_DEMO_QUESTION.answer,
  );
  const [loading, setLoading] = useState(false);
  const [coldStarting, setColdStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [citationRenderMiss, setCitationRenderMiss] = useState(0);
  const [missDetails, setMissDetails] = useState<string[]>([]);
  const [upload, setUpload] = useState<UploadDocument | null>(null);

  const attemptSeq = useRef(0);
  const missedAttempts = useRef(new Set<string>());
  const documentRef = useRef<HTMLElement | null>(null);
  const questionRef = useRef<HTMLTextAreaElement | null>(null);

  useLayoutEffect(() => {
    const field = questionRef.current;
    if (!field) {
      return;
    }
    field.style.height = 'auto';
    if (field.scrollHeight > 0) {
      const border = field.offsetHeight - field.clientHeight;
      field.style.height = `${field.scrollHeight + border}px`;
    }
  }, [question]);

  // Wake a sleeping free instance as soon as the page loads, so a visitor's
  // first question does not pay the whole cold start in front of an error page.
  useEffect(() => {
    void warmApi();
  }, []);

  // A reload keeps the visitor's document open: `?document=` names an upload
  // this cookie owns. A foreign or swept id silently falls back to the demo.
  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get('document');
    if (!wanted) {
      return;
    }
    let cancelled = false;
    const restore = async (): Promise<void> => {
      try {
        let next = await uploadStatus(wanted);
        for (let attempt = 0; attempt < 120; attempt += 1) {
          if (cancelled || next.status === 'ready' || next.status === 'failed') {
            break;
          }
          await sleep(1_000);
          if (cancelled) {
            return;
          }
          next = await uploadStatus(wanted);
        }
        if (!cancelled && next.status === 'ready') {
          setUpload(next);
        }
      } catch {
        // Unknown, foreign, or swept: leave the demo paper open.
      }
    };
    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleMiss = useCallback((attemptId: string) => {
    if (missedAttempts.current.has(attemptId)) {
      return;
    }
    missedAttempts.current.add(attemptId);
    setCitationRenderMiss((count) => count + 1);
    setMissDetails((details) => [...details, attemptId]);
  }, []);

  const selectCitation = (citation: Citation): void => {
    attemptSeq.current += 1;
    setSelection({
      citation,
      attemptId: `attempt-${attemptSeq.current}`,
    });
    const column = documentRef.current;
    if (column) {
      const bounds = column.getBoundingClientRect();
      const narrow =
        typeof window.matchMedia === 'function' &&
        window.matchMedia('(max-width: 1023px)').matches;
      if (narrow || bounds.top < 0 || bounds.bottom < 120) {
        column.scrollIntoView({ block: 'start', behavior: 'auto' });
      }
    }
  };

  const presentAnswer = (next: AskResponse): void => {
    attemptSeq.current = 0;
    missedAttempts.current.clear();
    setCitationRenderMiss(0);
    setMissDetails([]);
    setAnswer(next);
    setError(null);
    setSelection(null);
  };

  const loadExample = (example: DemoQuestion): void => {
    setQuestion(example.question);
    presentAnswer(example.answer);
  };

  const openUpload = (next: UploadDocument): void => {
    setUpload(next);
    setAnswer(null);
    setSelection(null);
    setError(null);
    setQuestion('');
    window.history.replaceState(
      {},
      '',
      `/?document=${encodeURIComponent(next.id)}`,
    );
  };

  const openDemo = (): void => {
    setUpload(null);
    setQuestion(DEFAULT_DEMO_QUESTION.question);
    presentAnswer(DEFAULT_DEMO_QUESTION.answer);
    window.history.replaceState({}, '', '/');
  };

  const run = async (text: string): Promise<void> => {
    const recorded = upload ? undefined : demoForQuestion(text);
    if (recorded) {
      presentAnswer(recorded.answer);
      return;
    }

    setLoading(true);
    setColdStarting(false);
    setError(null);
    setAnswer(null);
    setSelection(null);
    try {
      const response = await askQuestionWithWarmup(
        text,
        () => setColdStarting(true),
        upload?.collection,
      );
      presentAnswer(response);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'The answer service could not be reached.',
      );
    } finally {
      setLoading(false);
      setColdStarting(false);
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

  const handleQuestionKeyDown = (
    event: KeyboardEvent<HTMLTextAreaElement>,
  ): void => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const sources = useMemo(() => collectSources(answer), [answer]);
  const sourceNumbers = useMemo(
    () =>
      new Map(
        sources.map((entry) => [citationKey(entry.citation), entry.number]),
      ),
    [sources],
  );

  const selectedKey = selection ? citationKey(selection.citation) : null;
  const selectedNumber = selection
    ? (sourceNumbers.get(citationKey(selection.citation)) ?? null)
    : null;
  const openDocumentId = upload ? uploadDocumentId(upload) : DEMO_PAPER.doc_id;
  const documentTitle = upload ? upload.filename : DEMO_PAPER.title;
  const documentPages = upload?.pages_total ?? DEMO_PAPER.page_count;
  const documentUrl = upload ? uploadedPdfUrl(upload.id) : DEMO_PAPER.pdf;
  const renderableCitation =
    selection && selection.citation.doc_id === openDocumentId
      ? selection.citation
      : null;
  const labelFor = useCallback(
    (docId: string) => sourceLabel(docId, upload),
    [upload],
  );
  const trace = answer?.trace ?? null;

  return (
    <main className="workspace">
      <section
        className="document-column"
        aria-label="Document workspace"
        ref={documentRef}
      >
        <div className="document-heading">
          <h1>{documentTitle}</h1>
          <p className="document-meta">{documentPages} pages</p>
        </div>
        {selection && renderableCitation === null ? (
          <p className="doc-note">
            This citation names a document that is not open in the workspace, so
            its page cannot be shown here. The quote still resolved against the
            source page on the server.
          </p>
        ) : null}
        <PdfPane
          pdfUrl={documentUrl}
          citation={renderableCitation}
          citationNumber={selectedNumber}
          attemptId={selection?.attemptId ?? ''}
          onMiss={handleMiss}
        />
      </section>

      <section className="answer-pane" aria-labelledby="ask-heading">
        <UploadPanel
          active={upload}
          onReady={openUpload}
          onDemo={openDemo}
        />

        <header className="ask-head">
          <h2 id="ask-heading">Ask the document.</h2>
        </header>

        <form className="ask-form" onSubmit={handleAsk}>
          <label htmlFor="question" className="visually-hidden">
            Question
          </label>
          <div className="ask-row">
            <textarea
              id="question"
              ref={questionRef}
              value={question}
              rows={2}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleQuestionKeyDown}
              placeholder="What does the paper say about…"
              autoComplete="off"
              disabled={loading}
            />
            <button type="submit" className="primary" disabled={loading}>
              {loading ? 'Asking…' : 'Ask'}
            </button>
          </div>
        </form>

        {upload === null ? (
          <ExampleQuestions onChoose={loadExample} />
        ) : null}

        {error ? (
          <div className="error-banner" role="alert">
            <strong>The answer service did not respond.</strong>
            <span>{error}</span>
            <span className="muted">
              Choose one of the example questions, or start the local service.
            </span>
          </div>
        ) : null}

        {loading ? (
          <div className="loading-state" role="status" aria-live="polite">
            <span className="loading-rule" aria-hidden="true" />
            <p>
              {coldStarting
                ? 'Starting the free instance — this can take up to a minute, then the question is answered automatically.'
                : 'Searching the document for relevant passages, then checking each quote against the page.'}
            </p>
          </div>
        ) : null}

        {answer ? (
          <section
            className="answer-result"
            aria-live="polite"
            aria-label="Answer"
          >
            <header className="answer-heading">
              <p className="eyebrow">Answer</p>
            </header>

            {answer.insufficient_evidence ? (
              <RefusalCard answer={answer} labelFor={labelFor} />
            ) : answer.claims.length > 0 ? (
              <>
                <ol className="claims" aria-label="Answer claims">
                  {answer.claims.map((claim, claimIndex) => (
                    <li
                      key={`${claimIndex}-${claim.text.slice(0, 24)}`}
                      className="claim"
                    >
                      <ClaimText text={claim.text}>
                        {claim.citations.map((citation) => {
                          const number =
                            sourceNumbers.get(citationKey(citation)) ?? 0;
                          return (
                            <CitationMark
                              key={citationKey(citation)}
                              citation={citation}
                              number={number}
                              selected={selectedKey === citationKey(citation)}
                              sourceName={labelFor(citation.doc_id)}
                              onSelect={() => selectCitation(citation)}
                            />
                          );
                        })}
                      </ClaimText>
                    </li>
                  ))}
                </ol>

                <Sources
                  sources={sources}
                  selectedKey={selectedKey}
                  labelFor={labelFor}
                  onSelect={selectCitation}
                />
              </>
            ) : (
              <div className="error-banner" role="alert">
                <strong>No claim passed the citation check.</strong>
                <span>
                  {answer.validation.claims_returned} claim
                  {answer.validation.claims_returned === 1 ? '' : 's'} came back;
                  open the retrieval trail to see why each was removed.
                </span>
              </div>
            )}
          </section>
        ) : null}

        <section className="trail-bar" aria-labelledby="trail-heading">
          <div>
            <p className="eyebrow">Retrieval trail</p>
            <h3 id="trail-heading">How this answer was assembled</h3>
            <p>
              {trace
                ? `${trace.vector_hits.length} meaning matches, ${trace.lexical_hits.length} keyword matches, ${trace.rerank_hits.length} rescored, ${trace.context_chunk_ids.length} passages sent.`
                : 'Run a question to see the passages found, how they were ranked, and the evidence check.'}
            </p>
          </div>
          <button
            type="button"
            className="secondary"
            aria-expanded={drawerOpen}
            onClick={() => setDrawerOpen((open) => !open)}
            data-testid="trace-toggle"
          >
            {drawerOpen ? 'Close trail' : 'Open trail'}
          </button>
        </section>
      </section>

      <TraceDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        answer={answer}
        citationRenderMiss={citationRenderMiss}
        missDetails={missDetails}
      />
    </main>
  );
}

export default function App() {
  const onEval = isEvalRoute();
  return (
    <div className="app">
      <SiteHeader onEval={onEval} />
      {onEval ? <EvalPage /> : <AskPage />}
    </div>
  );
}
