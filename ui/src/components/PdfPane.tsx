import { useEffect, useRef, useState } from 'react';
import type {
  PDFDocumentLoadingTask,
  PDFDocumentProxy,
  RenderTask,
} from 'pdfjs-dist';
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

import { extractPage, resolveQuote, type ExtractedPage } from '../pdfText';
import { pdfSourceOptions } from '../pdfSource';
import type { Citation, Rect } from '../types';

type PdfJs = typeof import('pdfjs-dist');

// PDF.js is the largest dependency and the document pane already loads
// asynchronously, so its parse/evaluation is deferred until this component
// mounts instead of being part of the entry chunk. The module is loaded once.
let pdfjsModule: Promise<PdfJs> | null = null;

function loadPdfJs(): Promise<PdfJs> {
  pdfjsModule ??= import('pdfjs-dist').then((module) => {
    module.GlobalWorkerOptions.workerSrc = workerSrc;
    return module;
  });
  return pdfjsModule;
}

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
  /** The same number shown beside the claim and in the source list. */
  citationNumber: number | null;
  /** Stable per selection attempt; a miss is counted once per attempt. */
  attemptId: string;
  onMiss: (attemptId: string) => void;
}

interface MarkerPosition {
  left: number;
  top: number;
}

interface PassageMarker extends MarkerPosition {
  attemptId: string;
  citationKey: string;
  number: number;
}

function citationIdentity(citation: Citation): string {
  return `${citation.chunk_id}|${citation.page_no}|${citation.quote}`;
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
  host.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
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

function markerInMargin(
  page: HTMLElement,
  left: number,
  right: number,
  top: number,
): MarkerPosition | null {
  const pageRect = page.getBoundingClientRect();
  if (pageRect.width === 0) {
    return null;
  }
  const preferred = left - 28;
  const fallback = right + 4;
  const markerLeft = preferred >= 4 ? preferred : fallback;
  return {
    left: Math.max(4, Math.min(pageRect.width - 26, markerLeft)),
    top: Math.max(2, top - 1),
  };
}

function markerForRange(
  range: Range,
  page: HTMLElement,
): MarkerPosition | null {
  const rect = range.getClientRects()[0] ?? range.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) {
    return null;
  }
  const pageRect = page.getBoundingClientRect();
  return markerInMargin(
    page,
    rect.left - pageRect.left,
    rect.right - pageRect.left,
    rect.top - pageRect.top,
  );
}

/**
 * Renders the cited PDF page, then highlights the citation quote with the CSS
 * Custom Highlight API. When the quote cannot be mapped back to a DOM range,
 * the contract's `rects[]` are painted instead and the miss is reported.
 */
export function PdfPane({
  pdfUrl,
  citation,
  citationNumber,
  attemptId,
  onMiss,
}: PdfPaneProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const pageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textLayerRef = useRef<HTMLDivElement | null>(null);
  const rectLayerRef = useRef<HTMLDivElement | null>(null);
  const renderedRef = useRef<RenderedPage | null>(null);
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [scale, setScale] = useState(1.2);
  const [renderTick, setRenderTick] = useState(0);
  const [marker, setMarker] = useState<PassageMarker | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let task: PDFDocumentLoadingTask | null = null;
    loadPdfJs()
      .then((pdfjs) => {
        if (cancelled) {
          return null;
        }
        task = pdfjs.getDocument(pdfSourceOptions(pdfUrl));
        return task.promise;
      })
      .then((loaded) => {
        if (!cancelled && loaded) {
          setDocument(loaded);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError('The paper could not be loaded.');
        }
      });
    return () => {
      cancelled = true;
      void task?.destroy();
    };
  }, [pdfUrl]);

  // The selected citation owns the page; with no selection the paper opens on
  // page 1. This is derived state rather than a syncing effect.
  const pageNo = citation?.page_no ?? 1;

  // Fit the page to the pane while keeping the text layer and canvas in step.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }
    const update = (width: number): void => {
      const next = Math.max(0.45, Math.min(1.6, (width - 48) / BASE_WIDTH));
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
      const pdfjs = await loadPdfJs();
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
      const textLayer = new pdfjs.TextLayer({
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
        setError(`Page ${pageNo} could not be shown.`);
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
    const page = pageRef.current;
    if (range && page && citationNumber !== null) {
      const position = markerForRange(range, page);
      if (position) {
        setMarker({
          ...position,
          attemptId,
          citationKey: citationIdentity(citation),
          number: citationNumber,
        });
      }
    }

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
      const first = citation.rects[0];
      if (!range && first && page && citationNumber !== null) {
        const [x0, y0, x1] = first;
        const position = markerInMargin(
          page,
          x0 * rendered.scale,
          x1 * rendered.scale,
          y0 * rendered.scale,
        );
        if (position) {
          setMarker({
            ...position,
            attemptId,
            citationKey: citationIdentity(citation),
            number: citationNumber,
          });
        }
      }
    }
    if (!range) {
      onMiss(attemptId);
    }
    return () => clearHighlight();
  }, [citation, citationNumber, attemptId, renderTick, onMiss]);

  useEffect(() => () => clearHighlight(), []);

  return (
    <section className="pdf-pane" aria-label="Document page">
      <header className="pane-head">
        <p className="eyebrow">Passage</p>
        <span className="page-badge" data-testid="pdf-page-no">
          Page {pageNo}
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
          ref={pageRef}
          style={{ width: Math.floor(BASE_WIDTH * scale) }}
        >
          <canvas ref={canvasRef} aria-label={`PDF page ${pageNo}`} />
          <div className="textLayer" ref={textLayerRef} />
          <div className="rect-layer" ref={rectLayerRef} />
          {marker &&
          citation &&
          citationNumber !== null &&
          marker.attemptId === attemptId &&
          marker.citationKey === citationIdentity(citation) &&
          marker.number === citationNumber ? (
            <span
              key={attemptId}
              className="passage-marker"
              style={{ left: marker.left, top: marker.top }}
              data-testid="passage-marker"
              aria-hidden="true"
            >
              {marker.number}
            </span>
          ) : null}
        </div>
      </div>

    </section>
  );
}
