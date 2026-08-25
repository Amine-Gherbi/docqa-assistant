import { useCallback, useEffect, useRef, useState } from 'react';
import {
  getDocument,
  GlobalWorkerOptions,
  TextLayer,
  type PDFDocumentProxy,
  type RenderTask,
} from 'pdfjs-dist';
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

import {
  checkFixtureCitations,
  type DocumentProxyLike,
  type FixtureCheck,
} from '../fixtures';
import { extractPage, resolveQuote, type ExtractedPage } from '../pdfText';
import type { Citation, Rect } from '../types';

GlobalWorkerOptions.workerSrc = workerSrc;

export const HIGHLIGHT_NAME = 'docqa-citation';

const BASE_WIDTH = 612;

interface RenderedPage {
  pageNo: number;
  extracted: ExtractedPage;
  textDivs: HTMLElement[];
  scale: number;
}

interface PdfPaneProps {
  pdfUrl: string;
  citation: Citation | null;
  /** Stable per selection attempt; a miss is counted once per attempt. */
  attemptId: string;
  onMiss: (attemptId: string) => void;
  onFixtureCheck: (check: FixtureCheck) => void;
}

function supportsHighlights(): boolean {
  return (
    typeof CSS !== 'undefined' &&
    'highlights' in CSS &&
    typeof Highlight !== 'undefined'
  );
}

function clearHighlight(): void {
  if (supportsHighlights()) {
    CSS.highlights.delete(HIGHLIGHT_NAME);
  }
}

function textNodesFor(
  textDivs: readonly HTMLElement[],
  slices: readonly { itemIndex: number; from: number; to: number }[],
): Range | null {
  const first = slices[0];
  const last = slices[slices.length - 1];
  if (!first || !last) {
    return null;
  }
  const startNode = textDivs[first.itemIndex]?.firstChild;
  const endNode = textDivs[last.itemIndex]?.firstChild;
  if (!(startNode instanceof Text) || !(endNode instanceof Text)) {
    return null;
  }
  try {
    const range = document.createRange();
    range.setStart(startNode, Math.min(first.from, startNode.length));
    range.setEnd(endNode, Math.min(last.to, endNode.length));
    return range;
  } catch {
    return null;
  }
}

function scrollRangeIntoView(range: Range, host: HTMLElement): void {
  const rect = range.getBoundingClientRect();
  const hostRect = host.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) {
    return;
  }
  const top =
    host.scrollTop + rect.top - hostRect.top - host.clientHeight / 2 + rect.height / 2;
  host.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
}

function drawRects(
  layer: HTMLElement,
  rects: readonly Rect[],
  scale: number,
): void {
  layer.replaceChildren();
  for (const [x0, y0, x1, y1] of rects) {
    const box = document.createElement('div');
    box.className = 'rect-highlight';
    box.style.left = `${x0 * scale}px`;
    box.style.top = `${y0 * scale}px`;
    box.style.width = `${(x1 - x0) * scale}px`;
    box.style.height = `${(y1 - y0) * scale}px`;
    layer.append(box);
  }
}

/**
 * Renders the cited PDF page, then highlights the citation quote with the CSS
 * Custom Highlight API. When the quote cannot be mapped back to a DOM range,
 * the contract's `rects[]` are painted instead and the miss is reported.
 */
export function PdfPane({
  pdfUrl,
  citation,
  attemptId,
  onMiss,
  onFixtureCheck,
}: PdfPaneProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textLayerRef = useRef<HTMLDivElement | null>(null);
  const rectLayerRef = useRef<HTMLDivElement | null>(null);
  const renderedRef = useRef<RenderedPage | null>(null);
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [scale, setScale] = useState(1.2);
  const [renderTick, setRenderTick] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleFixtureCheck = useCallback(
    (check: FixtureCheck) => onFixtureCheck(check),
    [onFixtureCheck],
  );

  useEffect(() => {
    let cancelled = false;
    const task = getDocument({ url: pdfUrl, verbosity: 0 });
    task.promise
      .then(async (loaded) => {
        if (cancelled) {
          return;
        }
        setDocument(loaded);
        try {
          const check = await checkFixtureCitations(
            loaded as unknown as DocumentProxyLike,
          );
          if (!cancelled) {
            handleFixtureCheck(check);
          }
        } catch {
          // The fixture check is advisory; a failure must not blank the page.
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError('The fixture PDF could not be loaded.');
        }
      });
    return () => {
      cancelled = true;
      void task.destroy();
    };
  }, [pdfUrl, handleFixtureCheck]);

  // The selected citation owns the page; with no selection the fixture opens
  // on page 1. This is derived state rather than a syncing effect.
  const pageNo = citation?.page_no ?? 1;

  // Fit the page to the pane while keeping the text layer and canvas in step.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }
    const update = (width: number): void => {
      const next = Math.max(0.55, Math.min(1.6, (width - 48) / BASE_WIDTH));
      setScale((current) => (Math.abs(current - next) < 0.01 ? current : next));
    };
    update(host.clientWidth);
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        update(entry.contentRect.width);
      }
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!document) {
      return;
    }
    let cancelled = false;
    let renderTask: RenderTask | null = null;

    const render = async (): Promise<void> => {
      const page = await document.getPage(pageNo);
      if (cancelled) {
        return;
      }
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      const textLayerDiv = textLayerRef.current;
      if (!canvas || !textLayerDiv) {
        return;
      }

      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * ratio);
      canvas.height = Math.floor(viewport.height * ratio);
      canvas.style.width = `${Math.floor(viewport.width)}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;
      const context = canvas.getContext('2d');
      if (!context) {
        return;
      }
      renderTask = page.render({
        canvas,
        canvasContext: context,
        viewport,
        transform:
          ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
      });
      await renderTask.promise;
      if (cancelled) {
        return;
      }

      const content = await page.getTextContent({
        disableNormalization: true,
      });
      if (cancelled) {
        return;
      }

      textLayerDiv.replaceChildren();
      textLayerDiv.style.setProperty('--total-scale-factor', String(scale));
      textLayerDiv.style.setProperty('--scale-round-x', '1px');
      textLayerDiv.style.setProperty('--scale-round-y', '1px');
      const textLayer = new TextLayer({
        textContentSource: content,
        container: textLayerDiv,
        viewport,
      });
      await textLayer.render();
      if (cancelled) {
        textLayer.cancel();
        return;
      }

      renderedRef.current = {
        pageNo,
        extracted: extractPage(content.items),
        textDivs: textLayer.textDivs,
        scale,
      };
      setRenderTick((tick) => tick + 1);
    };

    render().catch(() => {
      if (!cancelled) {
        setError(`Page ${pageNo} could not be rendered.`);
      }
    });

    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [document, pageNo, scale]);

  useEffect(() => {
    const rendered = renderedRef.current;
    const rectLayer = rectLayerRef.current;
    clearHighlight();
    rectLayer?.replaceChildren();

    if (!citation || !rectLayer) {
      return () => clearHighlight();
    }
    if (!rendered || rendered.pageNo !== citation.page_no) {
      // The cited page is still rendering; the next tick resolves it.
      return () => clearHighlight();
    }

    const resolved = resolveQuote(rendered.extracted, citation.quote);
    const range = resolved
      ? textNodesFor(rendered.textDivs, resolved.slices)
      : null;

    if (range && supportsHighlights()) {
      CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(range));
      const host = hostRef.current;
      if (host) {
        scrollRangeIntoView(range, host);
      }
      return () => clearHighlight();
    }

    // Fallback path: the stored PyMuPDF rectangles, scaled to the viewport.
    if (citation.rects && citation.rects.length > 0) {
      drawRects(rectLayer, citation.rects, rendered.scale);
    }
    if (!range) {
      onMiss(attemptId);
    }
    return () => clearHighlight();
  }, [citation, attemptId, renderTick, onMiss]);

  useEffect(() => () => clearHighlight(), []);

  return (
    <section className="pdf-pane" aria-label="Cited PDF page">
      <header className="pane-head">
        <div>
          <h2>Source page</h2>
          <p className="muted">
            {citation
              ? `${citation.doc_id} · page ${citation.page_no}`
              : 'fixture-corpus · page 1'}
          </p>
        </div>
        <span className="page-badge" data-testid="pdf-page-no">
          p.{pageNo}
        </span>
      </header>

      <div className="pdf-scroll" ref={hostRef}>
        {error ? (
          <p className="pane-error" role="alert">
            {error}
          </p>
        ) : null}
        <div
          className="pdf-page"
          style={{ width: Math.floor(BASE_WIDTH * scale) }}
        >
          <canvas ref={canvasRef} aria-label={`PDF page ${pageNo}`} />
          <div className="textLayer" ref={textLayerRef} />
          <div className="rect-layer" ref={rectLayerRef} />
        </div>
      </div>

      <footer className="pane-foot">
        <span>
          Quote resolution first; <code>rects[]</code> overlay is the fallback.
        </span>
        <span className="muted">
          {supportsHighlights()
            ? 'CSS Custom Highlight API available'
            : 'CSS Custom Highlight API unavailable — rectangles only'}
        </span>
      </footer>
    </section>
  );
}
