import { useEffect, useMemo, useState } from 'react';
import { Expand, ExternalLink, FileText, Minus, PanelRight, PanelRightClose, Plus } from 'lucide-react';
import {
  getSourceInfo,
  sourcePageImageUrl,
  sourcePdfUrl,
} from '../api/chat';
import type { EvidencePassage, EvidenceTab, SourceInfo } from '../types/ui';

interface EvidencePanelProps {
  searchResults?: boolean;
  passage: EvidencePassage | null;
  activeTab: EvidenceTab;
  onTabChange: (tab: EvidenceTab) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  collapsed?: boolean;
  onCollapse?: () => void;
  onExpand?: () => void;
}

function PdfAcrobatIcon() {
  return (
    <svg
      className="pdf-acrobat-icon"
      width="28"
      height="28"
      viewBox="0 0 32 32"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect width="32" height="32" rx="4" fill="#E5252A" />
      <path
        d="M8.8 22.8c2.2-1.35 4.22-4.6 5.48-7.8 1.28-3.22 1.7-6.08.9-7.25-.42-.62-1.02-.3-1.2.28-.56 1.86.42 5.22 2.06 7.88 1.85 3 4.35 5.05 6.45 5.32.7.09 1.02-.45.5-.83-1.6-1.14-5.4-.62-8.7.16-3.05.72-5.75 1.8-6.73 2.73-.45.44-.02.83.34.71.25-.08.54-.2.9-.4z"
        fill="none"
        stroke="#fff"
        strokeWidth="1.25"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function EvidencePanel({
  searchResults = false,
  passage,
  activeTab,
  onTabChange,
  zoom,
  onZoomChange,
  collapsed = false,
  onCollapse,
  onExpand,
}: EvidencePanelProps) {
  const filename = passage?.filename ?? '';
  const page = passage?.page ?? 0;
  const quote = passage?.quote || '';
  const passageKey = `${filename}:${page}:${quote}`;

  const [info, setInfo] = useState<SourceInfo | null>(null);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [quoteExpanded, setQuoteExpanded] = useState(false);
  const [loadedFilename, setLoadedFilename] = useState(filename);
  const [loadingFor, setLoadingFor] = useState('');
  const [quoteKey, setQuoteKey] = useState(passageKey);

  if (loadedFilename !== filename) {
    setLoadedFilename(filename);
    setInfo(null);
    setViewerError(null);
  }
  if (quoteKey !== passageKey) {
    setQuoteKey(passageKey);
    setQuoteExpanded(false);
  }

  useEffect(() => {
    let cancelled = false;
    if (!filename) return undefined;
    getSourceInfo(filename)
      .then((value) => {
        if (!cancelled) setInfo(value);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setViewerError(error instanceof Error ? error.message : 'PDF source indisponible.');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [filename]);

  const pageImage = useMemo(() => {
    if (!passage) return '';
    return sourcePageImageUrl(passage.filename, passage.page, passage.quote, 2.2);
  }, [passage]);

  if (pageImage && loadingFor !== pageImage) {
    setLoadingFor(pageImage);
    setImageLoading(true);
  }

  const pdfUrl = passage ? sourcePdfUrl(passage.filename, passage.page) : '';
  const totalPages = info?.pages ?? null;
  const quoteLong = quote.length > 280;

  function decreaseZoom() {
    onZoomChange(Math.max(70, zoom - 10));
  }

  function increaseZoom() {
    onZoomChange(Math.min(180, zoom + 10));
  }

  if (collapsed) {
    return (
      <aside className="evidence-panel is-collapsed">
        <button
          type="button"
          className="panel-rail-btn"
          onClick={onExpand}
          aria-label="Afficher les sources"
          title="Sources"
        >
          <PanelRight size={20} strokeWidth={1.75} />
        </button>
      </aside>
    );
  }

  return (
    <aside className="evidence-panel">
      <div className="evidence-tabs">
        <button
          type="button"
          className={`evidence-tab${activeTab === 'preuve' ? ' active' : ''}`}
          onClick={() => onTabChange('preuve')}
        >
          {searchResults ? 'Passage' : 'Preuve'}
        </button>
        <button
          type="button"
          className={`evidence-tab${activeTab === 'document' ? ' active' : ''}`}
          onClick={() => onTabChange('document')}
        >
          Document
        </button>
        {onCollapse ? (
          <button
            type="button"
            className="icon-ghost panel-collapse-btn evidence-collapse"
            aria-label="Masquer le panneau des sources"
            title="Masquer les sources"
            onClick={onCollapse}
          >
            <PanelRightClose size={20} strokeWidth={1.75} />
          </button>
        ) : null}
      </div>

      <div className="evidence-content">
        {!passage ? (
          <div className="document-tab-placeholder evidence-empty">
            <FileText size={30} strokeWidth={1.5} />
            <p>Aucune preuve sélectionnée</p>
            <span>Les pages PDF citées apparaîtront ici après une réponse sourcée.</span>
          </div>
        ) : (
          <>
            <div className="evidence-file-row">
              <PdfAcrobatIcon />
              <div className="evidence-file-copy">
                <strong title={passage.filename}>{passage.filename}</strong>
                {info?.title && info.title !== passage.filename ? (
                  <span title={info.title}>{info.title}</span>
                ) : null}
              </div>
            </div>

            <div className="pdf-toolbar-row">
              <span className="pdf-page-label">
                Page {passage.page}{totalPages ? ` / ${totalPages}` : ''}
              </span>
              <div className="pdf-toolbar-actions">
                <div className="zoom-control">
                  <button type="button" aria-label="Zoom arrière" onClick={decreaseZoom}>
                    <Minus size={14} strokeWidth={2} />
                  </button>
                  <span>{zoom}%</span>
                  <button type="button" aria-label="Zoom avant" onClick={increaseZoom}>
                    <Plus size={14} strokeWidth={2} />
                  </button>
                </div>
                <a className="pdf-expand" aria-label="Ouvrir le PDF" href={pdfUrl} target="_blank" rel="noreferrer">
                  <Expand size={15} strokeWidth={1.75} />
                </a>
              </div>
            </div>

            {viewerError ? <div className="viewer-error">{viewerError}</div> : null}

            {activeTab === 'preuve' ? (
              <div className="pdf-real-stage">
                {imageLoading ? <div className="pdf-loading">Chargement de la page…</div> : null}
                <img
                  key={pageImage}
                  src={pageImage}
                  alt={`Page ${passage.page} de ${passage.filename}`}
                  className="pdf-real-page"
                  style={{ width: `${zoom}%` }}
                  onLoad={() => {
                    setImageLoading(false);
                    setViewerError(null);
                  }}
                  onError={() => {
                    setImageLoading(false);
                    setViewerError("Impossible d'afficher cette page PDF. Vérifiez BCT_DOCUMENTS_DIR.");
                  }}
                />
              </div>
            ) : (
              <div className="pdf-document-frame-wrap">
                <iframe
                  className="pdf-document-frame"
                  src={pdfUrl}
                  title={`Document ${passage.filename}`}
                />
              </div>
            )}

            <section className="selected-passage">
              <h3>Passage sélectionné</h3>
              <blockquote className={`selected-quote${quoteLong && !quoteExpanded ? ' is-collapsed' : ''}`}>
                {quote ? `“${quote}”` : 'Aucun extrait textuel fourni par le backend.'}
              </blockquote>
              {quoteLong ? (
                <button
                  type="button"
                  className="quote-expand-btn"
                  onClick={() => setQuoteExpanded((open) => !open)}
                >
                  {quoteExpanded ? 'Réduire' : 'Développer'}
                </button>
              ) : null}
              <p className="selected-source">{passage.sourceLabel}</p>
              <a className="btn-full-source" href={pdfUrl} target="_blank" rel="noreferrer">
                <ExternalLink size={14} strokeWidth={1.75} />
                <span>Voir la source complète</span>
              </a>
            </section>
          </>
        )}
      </div>
    </aside>
  );
}
