// PDF upload, the indexed document list and background page reading status.
import { useEffect, useState, type FormEvent } from 'react';
import { ArrowUpRight, FileText, FileUp, Loader2, RotateCcw, Trash2 } from 'lucide-react';
import { type EnrichmentProgress, type EnrichmentWorkerState, type IndexedDocument } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { isPdfFile, type UploadEntryStatus, type UploadProgress } from './shared';
import { DocumentSkeleton } from './Skeletons';

function uploadEntryLabel(locale: UiLocale, status: UploadEntryStatus) {
  if (status === 'imported') return t(locale, 'admin.uploadEntryImported');
  if (status === 'duplicate') return t(locale, 'admin.uploadEntryDuplicate');
  if (status === 'failed') return t(locale, 'admin.uploadEntryFailed');
  if (status === 'running') return t(locale, 'admin.uploadEntryRunning');
  return t(locale, 'admin.uploadEntryPending');
}

export function DocumentsPage({
  documents, enrichment, onRetryEnrichment, loading, busy, deletingIds, locale, files, fileKey, docKind, onDocKindChange, uploadProgress, onUpload, onFilesChange, onInvalidFiles, onDeleteDocuments,
}: {
  documents: IndexedDocument[];
  enrichment: EnrichmentWorkerState | null;
  onRetryEnrichment: (documentId: string) => void;
  loading: boolean;
  busy: boolean;
  deletingIds: Set<string>;
  locale: UiLocale;
  files: File[];
  fileKey: number;
  docKind: 'regulatory' | 'statistical' | 'internal';
  onDocKindChange: (value: 'regulatory' | 'statistical' | 'internal') => void;
  uploadProgress: UploadProgress | null;
  onUpload: (event: FormEvent<HTMLFormElement>) => void;
  onFilesChange: (files: File[]) => void;
  onInvalidFiles: () => void;
  onDeleteDocuments: (documentIds: string[], label: string) => void;
}) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  useEffect(() => {
    if (!uploadProgress) setDetailsOpen(false);
  }, [uploadProgress]);
  const dropLabel = dragActive
    ? t(locale, 'admin.dropPdf')
    : files.length
      ? t(locale, 'admin.addMorePdf')
      : t(locale, 'admin.choosePdf');

  function acceptFiles(raw: File[]) {
    const pdfs = raw.filter(isPdfFile);
    if (raw.length > 0 && pdfs.length === 0) {
      onInvalidFiles();
      return;
    }
    if (!pdfs.length) return;
    const seen = new Set(files.map((file) => `${file.name}\0${file.size}\0${file.lastModified}`));
    const merged = [...files];
    for (const file of pdfs) {
      const key = `${file.name}\0${file.size}\0${file.lastModified}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(file);
    }
    onFilesChange(merged);
  }
  const entries = uploadProgress?.entries || [];
  const total = uploadProgress?.total || 0;
  const imported = entries.filter((entry) => entry.status === 'imported').length;
  const duplicates = entries.filter((entry) => entry.status === 'duplicate').length;
  const failed = entries.filter((entry) => entry.status === 'failed').length;
  const success = imported + duplicates;
  const fillPct = total > 0 ? Math.min(100, (success / total) * 100) : 0;
  return (
    <>
      {busy && uploadProgress ? (
        <div className="admin-upload-overlay" role="status" aria-live="assertive" aria-busy="true">
          <div className="admin-upload-overlay-card">
            <p className="admin-upload-overlay-kicker">{t(locale, 'admin.uploading')}</p>
            <h2>{t(locale, 'admin.uploadProgressCount', { done: success, total })}</h2>
            {uploadProgress.currentName ? (
              <p className="admin-upload-overlay-file">{uploadProgress.currentName}</p>
            ) : null}
            <p className="admin-upload-overlay-help">{t(locale, 'admin.uploadProgress')}</p>
            <div
              className="admin-upload-progress-track is-determinate"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={total}
              aria-valuenow={success}
              aria-label={t(locale, 'admin.uploadProgressCount', { done: success, total })}
            >
              <span style={{ width: `${fillPct}%` }} />
            </div>
            <p className="admin-upload-overlay-stats">
              {t(locale, 'admin.uploadStats', { imported, duplicates, failed })}
            </p>
            <button
              type="button"
              className="admin-upload-details-toggle"
              aria-expanded={detailsOpen}
              onClick={() => setDetailsOpen((open) => !open)}
            >
              {detailsOpen ? t(locale, 'admin.uploadHideDetails') : t(locale, 'admin.uploadViewDetails')}
            </button>
            {detailsOpen ? (
              <ul className="admin-upload-details-list">
                {entries.map((entry, index) => (
                  <li key={`${entry.name}-${index}`} className={`is-${entry.status}`}>
                    <strong>{entry.name}</strong>
                    <span>{uploadEntryLabel(locale, entry.status)}</span>
                    {entry.detail ? <em>{entry.detail}</em> : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </div>
      ) : null}
      <section className="admin-document-layout" aria-label={t(locale, 'admin.documents')}>
        <form className="admin-form admin-upload-card" noValidate onSubmit={onUpload}>
          <div className="admin-panel-heading">
            <div>
              <p>{t(locale, 'admin.corpusIntake')}</p>
              <h2>{t(locale, 'admin.uploadPdf')}</h2>
            </div>
            <FileUp aria-hidden="true" size={23} />
          </div>
          <p className="admin-help">{t(locale, 'admin.uploadHelp')}</p>
          <div className="admin-field">
            <label htmlFor="admin-doc-kind">{t(locale, 'admin.docKind')}</label>
            <select
              id="admin-doc-kind"
              name="doc_kind"
              value={docKind}
              disabled={busy}
              onChange={(event) => onDocKindChange(event.target.value as 'regulatory' | 'statistical' | 'internal')}
            >
              <option value="regulatory">{t(locale, 'admin.docKindRegulatory')}</option>
              <option value="statistical">{t(locale, 'admin.docKindStatistical')}</option>
              <option value="internal">{t(locale, 'admin.docKindInternal')}</option>
            </select>
            <p className="admin-help">{t(locale, 'admin.docKindHelp')}</p>
            {docKind !== 'regulatory' ? (
              <p className="admin-help admin-doc-kind-vlm" role="note">
                {t(locale, 'admin.docKindVlmNote')}
              </p>
            ) : null}
          </div>
          <div className="admin-field admin-file-field">
            <label
              className={`admin-file-input${dragActive ? ' is-dragover' : ''}${busy ? ' is-disabled' : ''}`}
              onDragEnter={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (!busy) setDragActive(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (!busy) setDragActive(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setDragActive(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setDragActive(false);
                if (busy) return;
                acceptFiles(Array.from(event.dataTransfer.files || []));
              }}
            >
              <span>{t(locale, 'admin.pdfFile')}</span>
              <input
                key={fileKey}
                name="file"
                type="file"
                accept="application/pdf,.pdf"
                multiple
                disabled={busy}
                onChange={(event) => {
                  acceptFiles(Array.from(event.currentTarget.files || []));
                  event.currentTarget.value = '';
                }}
              />
              <em>{dropLabel}</em>
            </label>
            {files.length ? (
              <div className="admin-selected-files">
                <div className="admin-selected-files-bar">
                  <strong>{t(locale, 'admin.filesSelected', { count: files.length })}</strong>
                  <button type="button" className="admin-action" disabled={busy} onClick={() => onFilesChange([])}>
                    {t(locale, 'admin.clearSelection')}
                  </button>
                </div>
                <ul>
                  {files.map((file) => (
                    <li key={`${file.name}-${file.size}-${file.lastModified}`}>
                      <span title={file.name}>{file.name}</span>
                      <button
                        type="button"
                        className="admin-action"
                        disabled={busy}
                        aria-label={t(locale, 'admin.removeFile', { name: file.name })}
                        onClick={() => onFilesChange(files.filter((entry) => entry !== file))}
                      >
                        {t(locale, 'admin.remove')}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
          <button type="submit" className="admin-primary-button" disabled={busy || !files.length}>
            <FileUp aria-hidden="true" size={18} />
            {busy ? t(locale, 'admin.uploading') : t(locale, 'admin.upload')}
          </button>
        </form>
        <DocumentsList key={docKind} documents={documents} enrichment={enrichment} onRetryEnrichment={onRetryEnrichment} loading={loading} busy={busy} deletingIds={deletingIds} locale={locale} docKind={docKind} onDelete={onDeleteDocuments} />
      </section>
    </>
  );
}

function formatEta(seconds: number): string {
  const minutes = Math.max(1, Math.ceil(seconds / 60));
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function enrichmentLine(doc: IndexedDocument, locale: UiLocale): string {
  const progress: EnrichmentProgress | undefined = doc.enrichment;
  if (!progress?.total) return '';
  if (doc.status === 'ready_degraded') return t(locale, 'admin.enrichDegraded', { count: progress.failed, total: progress.total });
  if (doc.status !== 'enriching') return '';
  const parts = [t(locale, 'admin.enrichProgress', { done: progress.done, total: progress.total })];
  if (progress.eta_seconds) parts.push(t(locale, 'admin.enrichEta', { eta: formatEta(progress.eta_seconds) }));
  if (progress.failed) parts.push(t(locale, 'admin.enrichFailedPages', { count: progress.failed }));
  if (!doc.searchable) parts.push(t(locale, 'admin.enrichNotSearchable'));
  return parts.join(' · ');
}

function enrichmentNotice(state: EnrichmentWorkerState | null, locale: UiLocale): string {
  if (!state) return '';
  if (state.state === 'reading') return t(locale, 'admin.enrichReading', { page: state.page ?? '', name: state.document ?? '' });
  if (state.state === 'waiting_for_chat') return t(locale, 'admin.enrichPausedChat');
  if (state.state === 'activating') return t(locale, 'admin.enrichActivating');
  if (state.state === 'cooldown') return t(locale, 'admin.enrichCooldown', { eta: formatEta(state.cooldown_seconds ?? 60) });
  if (state.state === 'disabled') return t(locale, 'admin.enrichDisabled');
  return '';
}

function DocumentsList({
  documents, enrichment, onRetryEnrichment, loading, busy, deletingIds, locale, docKind, onDelete,
}: {
  documents: IndexedDocument[];
  enrichment: EnrichmentWorkerState | null;
  onRetryEnrichment: (documentId: string) => void;
  loading: boolean;
  busy: boolean;
  deletingIds: Set<string>;
  locale: UiLocale;
  docKind: 'regulatory' | 'statistical' | 'internal';
  onDelete: (documentIds: string[], label: string) => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const rows = (Array.isArray(documents) ? documents : []).filter((doc) => {
    const item = doc as { doc_kind?: string; filename?: string };
    const kind = item.doc_kind === 'statistical' || item.doc_kind === 'internal'
      ? item.doc_kind
      : 'regulatory';
    return kind === docKind;
  });
  const rowIds = rows.map((doc) => (doc as { document_id?: string }).document_id || '').filter(Boolean);
  const selectedIds = rowIds.filter((id) => selected.has(id));
  const allSelected = rowIds.length > 0 && selectedIds.length === rowIds.length;
  function toggle(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }
  return (
    <section className="admin-panel admin-documents-panel">
      <div className="admin-panel-heading">
        <div>
          <p>{t(locale, 'admin.activeCorpus')}</p>
          <h2>{t(locale, 'admin.indexedPdfs')}</h2>
        </div>
        <span className="admin-count">{rows.length} {t(locale, 'admin.readyCount')}</span>
      </div>
      {rows.some((doc) => doc.status === 'enriching') ? (
        <p className="admin-enrich-notice" role="status">
          <Loader2 aria-hidden="true" size={14} className={enrichment?.state === 'reading' ? 'admin-spin' : undefined} />
          <span>{t(locale, 'admin.enrichNotice')}{enrichmentNotice(enrichment, locale) ? ` ${enrichmentNotice(enrichment, locale)}` : ''}</span>
        </p>
      ) : null}
      {loading ? (
        <DocumentSkeleton label={t(locale, 'admin.loading')} />
      ) : rows.length === 0 ? (
        <p className="admin-help">{t(locale, 'admin.indexedEmptyKind')}</p>
      ) : (
        <>
        {rowIds.length ? (
          <div className="admin-doc-toolbar">
            <label className="admin-doc-check">
              <input
                type="checkbox"
                checked={allSelected}
                ref={(node) => { if (node) node.indeterminate = selectedIds.length > 0 && !allSelected; }}
                disabled={busy}
                onChange={() => setSelected(allSelected ? new Set() : new Set(rowIds))}
              />
              {t(locale, 'admin.selectAllPdfs')}
            </label>
            {selectedIds.length ? (
              <button
                type="button"
                className="admin-action delete"
                disabled={busy}
                onClick={() => onDelete(selectedIds, '')}
              >
                {deletingIds.size > 1
                  ? <Loader2 aria-hidden="true" size={14} className="admin-spin" />
                  : <Trash2 aria-hidden="true" size={14} />}
                {t(locale, deletingIds.size > 1 ? 'admin.pdfDeleting' : 'admin.deleteSelected', { count: selectedIds.length })}
              </button>
            ) : null}
          </div>
        ) : null}
        <ul className="admin-doc-list">
          {rows.map((doc, index) => {
            const item = doc;
            const filename = item.filename || '';
            const documentId = item.document_id || '';
            const label = item.title || filename || t(locale, 'admin.pdfDocument');
            const meta = [
              item.pages != null ? t(locale, 'admin.pageCount', { count: item.pages }) : '',
            ].filter(Boolean);
            const openLabel = t(locale, 'admin.openPdf');
            const body = (
              <>
                <span className="admin-document-icon"><FileText aria-hidden="true" size={18} /></span>
                <div>
                  <strong>{label}</strong>
                  {filename ? <span className="admin-doc-filename">{filename}</span> : null}
                  {meta.length ? <span className="admin-doc-meta">{meta.join(' · ')}</span> : null}
                  {item.status && item.status !== 'ready' ? (
                    <span className="admin-doc-enrichment">
                      <DocumentStatusBadge status={item.status} locale={locale} />
                      {enrichmentLine(item, locale)}
                    </span>
                  ) : null}
                </div>
                <ArrowUpRight aria-hidden="true" size={17} />
              </>
            );
            const deleting = Boolean(documentId) && deletingIds.has(documentId);
            return (
              <li key={documentId || filename || String(index)} className={`admin-doc-row${deleting ? ' is-deleting' : ''}`} aria-busy={deleting || undefined}>
                {documentId ? (
                  <input
                    type="checkbox"
                    className="admin-doc-select"
                    checked={selected.has(documentId)}
                    disabled={busy}
                    onChange={() => toggle(documentId)}
                    aria-label={t(locale, 'admin.selectPdf', { name: label })}
                  />
                ) : <span />}
                {filename ? (
                  <a
                    className="admin-doc-link"
                    href={`/api/sources/${encodeURIComponent(filename)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`${openLabel}: ${label}`}
                    title={openLabel}
                  >
                    {body}
                  </a>
                ) : (
                  <div className="admin-doc-link is-disabled">{body}</div>
                )}
                {documentId ? (
                  <span className="admin-doc-actions">
                  {item.enrichment?.failed ? (
                    <button
                      type="button"
                      className="admin-action promote"
                      disabled={busy}
                      onClick={() => onRetryEnrichment(documentId)}
                      title={t(locale, 'admin.enrichRetryHelp')}
                    >
                      <RotateCcw aria-hidden="true" size={14} />
                      {t(locale, 'admin.enrichRetry')}
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="admin-action delete"
                    disabled={busy}
                    onClick={() => onDelete([documentId], label)}
                    aria-label={t(locale, deleting ? 'admin.pdfDeleting' : 'admin.deletePdf')}
                    title={t(locale, 'admin.deletePdf')}
                  >
                    {deleting
                      ? <Loader2 aria-hidden="true" size={14} className="admin-spin" />
                      : <Trash2 aria-hidden="true" size={14} />}
                    {t(locale, deleting ? 'admin.pdfDeleting' : 'admin.delete')}
                  </button>
                  </span>
                ) : null}
              </li>
            );
          })}
        </ul>
        </>
      )}
    </section>
  );
}

function DocumentStatusBadge({ status, locale }: { status: string; locale: UiLocale }) {
  const key = status === 'enriching' ? 'admin.docEnriching' : status === 'ready_degraded' ? 'admin.docDegraded' : 'admin.docReady';
  return <span className={`admin-status ${status === 'enriching' ? 'enriching' : status === 'ready_degraded' ? 'pending' : 'approved'}`}><i aria-hidden="true" />{t(locale, key)}</span>;
}
