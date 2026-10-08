import { useEffect, useMemo, useState } from 'react';
import { Expand, ExternalLink, FileText, Minus, PanelRight, PanelRightClose, Plus } from 'lucide-react';
import {
  getSourceInfo,
  sourcePageImageUrl,
  sourcePdfUrl,
} from '../api/chat';
import type { EvidencePassage, EvidenceTab, SourceInfo } from '../types/ui';
import { t, type UiLocale } from '../uiLocale';

interface EvidencePanelProps {
  locale: UiLocale;
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
  locale,
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
          setViewerError(error instanceof Error ? error.message : t(locale, 'chat.pdfUnavailable'));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [filename, locale]);

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

  return (
    <aside className={`evidence-panel${collapsed ? ' is-collapsed' : ''}`}>
      <button
        type="button"
        className="panel-rail-btn"
        onClick={onExpand}
        aria-label={t(locale, 'chat.showSources')}
        title={t(locale, 'chat.sources')}
        tabIndex={collapsed ? 0 : -1}
        aria-hidden={!collapsed}
      >
        <PanelRight size={20} strokeWidth={1.75} />
      </button>

      <div className="panel-expanded" aria-hidden={collapsed}>
        <div className="evidence-tabs">
          <button
            type="button"
            className={`evidence-tab${activeTab === 'preuve' ? ' active' : ''}`}
            onClick={() => onTabChange('preuve')}
            tabIndex={collapsed ? -1 : 0}
          >
            {t(locale, searchResults ? 'chat.passage' : 'chat.evidence')}
          </button>
          <button
            type="button"
            className={`evidence-tab${activeTab === 'document' ? ' active' : ''}`}
            onClick={() => onTabChange('document')}
            tabIndex={collapsed ? -1 : 0}
          >
            {t(locale, 'chat.document')}
          </button>
          {onCollapse ? (
            <button
              type="button"
              className="icon-ghost panel-collapse-btn evidence-collapse"
              aria-label={t(locale, 'chat.hideSources')}
              title={t(locale, 'chat.hideSources')}
              onClick={onCollapse}
              tabIndex={collapsed ? -1 : 0}
            >
              <PanelRightClose size={20} strokeWidth={1.75} />
            </button>
          ) : null}
        </div>

        <div className="evidence-content">
          {!passage ? (
            <div className="document-tab-placeholder evidence-empty">
              <FileText size={30} strokeWidth={1.5} />
              <p>{t(locale, 'chat.noEvidence')}</p>
              <span>{t(locale, 'chat.noEvidenceHint')}</span>
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
                  {t(locale, 'chat.pageOf', { page: passage.page })}{totalPages ? ` / ${totalPages}` : ''}
                </span>
                <div className="pdf-toolbar-actions">
                  <div className="zoom-control">
                    <button type="button" aria-label={t(locale, 'chat.zoomOut')} onClick={decreaseZoom} tabIndex={collapsed ? -1 : 0}>
                      <Minus size={14} strokeWidth={2} />
                    </button>
                    <span>{zoom}%</span>
                    <button type="button" aria-label={t(locale, 'chat.zoomIn')} onClick={increaseZoom} tabIndex={collapsed ? -1 : 0}>
                      <Plus size={14} strokeWidth={2} />
                    </button>
                  </div>
                  <a className="pdf-expand" aria-label={t(locale, 'chat.openPdf')} href={pdfUrl} target="_blank" rel="noreferrer" tabIndex={collapsed ? -1 : 0}>
                    <Expand size={15} strokeWidth={1.75} />
                  </a>
                </div>
              </div>

              {viewerError ? <div className="viewer-error">{viewerError}</div> : null}

              {activeTab === 'preuve' ? (
                <div className="pdf-real-stage">
                  {imageLoading ? <div className="pdf-loading">{t(locale, 'chat.loadingPage')}</div> : null}
                  <img
                    key={pageImage}
                    src={pageImage}
                    alt={t(locale, 'chat.pageAlt', { page: passage.page, file: passage.filename })}
                    className="pdf-real-page"
                    style={{ width: `${zoom}%` }}
                    onLoad={() => {
                      setImageLoading(false);
                      setViewerError(null);
                    }}
                    onError={() => {
                      setImageLoading(false);
                      setViewerError(t(locale, 'chat.pageError'));
                    }}
                  />
                </div>
              ) : (
                <div className="pdf-document-frame-wrap">
                  <iframe
                    className="pdf-document-frame"
                    src={pdfUrl}
                    title={passage.filename}
                    tabIndex={collapsed ? -1 : 0}
                  />
                </div>
              )}

              <section className="selected-passage">
                <h3>{t(locale, 'chat.selectedPassage')}</h3>
                <blockquote className={`selected-quote${quoteLong && !quoteExpanded ? ' is-collapsed' : ''}`} dir="auto">
                  {quote ? `“${quote}”` : t(locale, 'chat.noExcerpt')}
                </blockquote>
                {quoteLong ? (
                  <button
                    type="button"
                    className="quote-expand-btn"
                    onClick={() => setQuoteExpanded((open) => !open)}
                    tabIndex={collapsed ? -1 : 0}
                  >
                    {t(locale, quoteExpanded ? 'chat.showLess' : 'chat.showMore')}
                  </button>
                ) : null}
                <p className="selected-source">{passage.sourceLabel}</p>
                <a className="btn-full-source" href={pdfUrl} target="_blank" rel="noreferrer" tabIndex={collapsed ? -1 : 0}>
                  <ExternalLink size={14} strokeWidth={1.75} />
                  <span>{t(locale, 'chat.fullSource')}</span>
                </a>
              </section>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
