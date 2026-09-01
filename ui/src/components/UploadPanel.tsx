import { useEffect, useRef, useState, type DragEvent } from 'react';

import {
  ensureSession,
  uploadDocument,
  uploadStatus,
} from '../api';
import type { UploadDocument } from '../types';

const MAX_BYTES = 25 * 1024 * 1024;
const POLL_MS = 1_000;

interface UploadPanelProps {
  active: UploadDocument | null;
  onReady: (upload: UploadDocument) => void;
  onDemo: () => void;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/** The stage line a visitor reads while their PDF is processed. */
function uploadStage(upload: UploadDocument): string {
  switch (upload.status) {
    case 'queued':
      return 'Queued';
    case 'parsing':
      return upload.pages_total
        ? `Parsing page ${Math.max(1, upload.pages_done)} of ${upload.pages_total}`
        : 'Parsing the PDF';
    case 'embedding':
      return upload.pages_total
        ? `Embedding page ${Math.max(1, upload.pages_done)} of ${upload.pages_total}`
        : 'Embedding the text';
    case 'ready':
      return 'Ready';
    case 'failed':
      return upload.error_detail ?? 'The PDF could not be processed.';
    default:
      return 'Working';
  }
}

function tooLarge(): string {
  return 'This PDF is larger than 25 MB. Choose a smaller file.';
}

/**
 * Drop zone and stage readout for one visitor upload. The bytes go to the API,
 * which parses them in a worker; this panel only polls the stage.
 */
export function UploadPanel({ active, onReady, onDemo }: UploadPanelProps) {
  const [stage, setStage] = useState<UploadDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  const start = async (file: File): Promise<void> => {
    if (busy) {
      return;
    }
    setError(null);
    if (file.size > MAX_BYTES) {
      setError(tooLarge());
      return;
    }
    let magic: string;
    try {
      magic = new TextDecoder().decode(await file.slice(0, 5).arrayBuffer());
    } catch {
      magic = '';
    }
    if (magic !== '%PDF-') {
      setError('That file is not a PDF. Choose a PDF file.');
      return;
    }

    setBusy(true);
    try {
      await ensureSession();
      let next = await uploadDocument(file);
      if (!aliveRef.current) {
        return;
      }
      setStage(next);
      while (next.status !== 'ready' && next.status !== 'failed') {
        next = await uploadStatus(next.id);
        if (!aliveRef.current) {
          return;
        }
        setStage(next);
        if (next.status === 'ready' || next.status === 'failed') {
          break;
        }
        await sleep(POLL_MS);
        if (!aliveRef.current) {
          return;
        }
      }
      if (next.status === 'failed') {
        setError(next.error_detail ?? 'The PDF could not be processed.');
        return;
      }
      onReady(next);
    } catch (cause) {
      if (aliveRef.current) {
        setError(
          cause instanceof Error
            ? cause.message
            : 'The upload could not be completed.',
        );
      }
    } finally {
      if (aliveRef.current) {
        setBusy(false);
      }
    }
  };

  const handleFiles = (files: FileList | null): void => {
    const file = files?.[0];
    if (file) {
      void start(file);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    setDragging(false);
    handleFiles(event.dataTransfer.files);
  };

  const shown = stage ?? active;

  return (
    <section
      className="upload-panel"
      data-testid="upload-panel"
      aria-labelledby="upload-heading"
    >
      <div className="upload-copy">
        <p className="eyebrow">Your own PDF</p>
        <h2 id="upload-heading">Ask a document you choose.</h2>
      </div>

      <p className="upload-disclosure" data-testid="upload-disclosure">
        Uploaded text is sent to a third-party model on a free tier. The
        provider may use it to improve its products. Do not upload confidential
        documents. Uploaded documents are deleted after 24 hours.
      </p>

      <div
        className={dragging ? 'drop-zone dragging' : 'drop-zone'}
        data-testid="upload-drop-zone"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          id="upload-file"
          data-testid="upload-input"
          className="visually-hidden"
          type="file"
          accept="application/pdf,.pdf"
          disabled={busy}
          onChange={(event) => {
            handleFiles(event.target.files);
            event.target.value = '';
          }}
        />
        <label htmlFor="upload-file" className="primary upload-button">
          Choose a PDF
        </label>
        <span className="drop-hint">or drop it here</span>
      </div>

      <p className="upload-limits">
        PDF only · 25 MB maximum · 80 pages maximum · one at a time · three per
        browser every 24 hours
      </p>

      {shown ? (
        <p className="upload-stage" role="status" data-testid="upload-stage">
          <strong>{shown.filename}</strong> · {uploadStage(shown)}
        </p>
      ) : null}

      {error ? (
        <p className="upload-error" role="alert">
          {error}
        </p>
      ) : null}

      {active ? (
        <button type="button" className="secondary" onClick={onDemo}>
          Demo paper
        </button>
      ) : null}
    </section>
  );
}
