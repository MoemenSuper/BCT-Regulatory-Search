import { Expand, ExternalLink, FileText, Minus, Plus } from 'lucide-react';
import type { EvidencePassage, EvidenceTab } from '../types/ui';
import { PdfMockViewer } from './PdfMockViewer';

interface EvidencePanelProps {
  passage: EvidencePassage;
  activeTab: EvidenceTab;
  onTabChange: (tab: EvidenceTab) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
}

/** Adobe Acrobat–style PDF mark used in the reference mockup. */
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
      <rect width="32" height="32" rx="6" fill="#E5252A" />
      <path
        fill="#FFFFFF"
        d="M9.8 23.2 15.6 7.8h2.1l5.9 15.4h-2.5l-1.3-3.6h-6.3l-1.3 3.6H9.8zm4.2-5.6h4.3L16.2 12l-2.2 5.6z"
      />
      <path
        fill="#FFFFFF"
        d="M7.2 23.6c2.4-1.2 4.7-3.1 6.7-5.4 1.7-2 3.1-4.2 4.1-6.5.7 1.8 1.8 3.4 3.2 4.7-1.9.9-3.7 2.2-5.2 3.8-1.8 1.8-3.2 3.8-4.2 6-.1 0-.2 0-.3-.1-1.5-1-2.9-1.8-4.3-2.5z"
        opacity="0.35"
      />
    </svg>
  );
}

export function EvidencePanel({
  passage,
  activeTab,
  onTabChange,
  zoom,
  onZoomChange,
}: EvidencePanelProps) {
  const decreaseZoom = () => onZoomChange(Math.max(75, zoom - 10));
  const increaseZoom = () => onZoomChange(Math.min(150, zoom + 10));

  return (
    <aside className="evidence-panel">
      <div className="evidence-tabs">
        <button
          type="button"
          className={`evidence-tab${activeTab === 'preuve' ? ' active' : ''}`}
          onClick={() => onTabChange('preuve')}
        >
          Preuve
        </button>
        <button
          type="button"
          className={`evidence-tab${activeTab === 'document' ? ' active' : ''}`}
          onClick={() => onTabChange('document')}
        >
          Document
        </button>
      </div>

      <div className="evidence-body">
        {activeTab === 'preuve' ? (
          <>
            <div className="pdf-toolbar-row">
              <div className="pdf-file-block">
                <PdfAcrobatIcon />
                <div className="pdf-file-text">
                  <span className="pdf-filename">{passage.filename}</span>
                  <span className="pdf-page-indicator">
                    Page {passage.page} / {passage.totalPages}
                  </span>
                </div>
              </div>

              <div className="pdf-toolbar-actions">
                <div className="pdf-zoom-controls">
                  <button type="button" aria-label="Zoom arrière" onClick={decreaseZoom}>
                    <Minus size={14} strokeWidth={2} />
                  </button>
                  <span>{zoom}%</span>
                  <button type="button" aria-label="Zoom avant" onClick={increaseZoom}>
                    <Plus size={14} strokeWidth={2} />
                  </button>
                </div>
                <button type="button" className="pdf-expand" aria-label="Plein écran">
                  <Expand size={15} strokeWidth={1.75} />
                </button>
              </div>
            </div>

            <PdfMockViewer highlight={passage.quote} zoom={zoom} />

            <section className="selected-passage">
              <h3>Passage sélectionné</h3>
              <blockquote className="selected-quote">&ldquo;{passage.quote}&rdquo;</blockquote>
              <p className="selected-source">{passage.sourceLabel}</p>
              <button type="button" className="btn-full-source">
                <ExternalLink size={14} strokeWidth={1.75} />
                <span>Voir la source complète</span>
              </button>
            </section>
          </>
        ) : (
          <div className="document-tab-placeholder">
            <FileText size={28} strokeWidth={1.5} />
            <p>Aperçu document</p>
            <span>Le rendu PDF réel sera branché ultérieurement.</span>
          </div>
        )}
      </div>
    </aside>
  );
}
