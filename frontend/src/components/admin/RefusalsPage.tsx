// Answers the assistant declined, filterable and exportable as CSV.
import { Fragment, useEffect, useRef, useState } from 'react';
import { ChevronRight, Download, ListFilter } from 'lucide-react';
import { type AnswerRefusal, type AnswerRefusalOption, type AnswerRefusalsPage } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { formatWhen } from './shared';
import { BlockSkeleton } from './Skeletons';

export function RefusalsPage({
  page,
  loading,
  busy,
  locale,
  selectedBuckets,
  onBucketsChange,
  onExport,
}: {
  page: AnswerRefusalsPage | null;
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  selectedBuckets: string[];
  onBucketsChange: (buckets: string[]) => void;
  onExport: () => void;
}) {
  const [filterOpen, setFilterOpen] = useState(false);
  // The row opened to read its full question and diagnostics.
  const [openId, setOpenId] = useState<string | null>(null);
  const filterRef = useRef<HTMLDivElement>(null);
  const items = page?.items ?? [];
  const total = page?.total ?? 0;
  const totalAll = page?.total_all ?? total;
  const options: AnswerRefusalOption[] = page?.reason_options?.length
    ? page.reason_options.map((option) => ({
        bucket: option.bucket || option.reason || '',
        title: option.title || option.bucket || option.reason || '',
        count: option.count,
      }))
    : [];

  // Close the filter menu on a click outside it or on Escape.
  useEffect(() => {
    if (!filterOpen) return;
    function onPointer(event: MouseEvent) {
      if (!filterRef.current?.contains(event.target as Node)) setFilterOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setFilterOpen(false);
    }
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [filterOpen]);

  function toggleBucket(bucket: string) {
    if (selectedBuckets.includes(bucket)) {
      onBucketsChange(selectedBuckets.filter((item) => item !== bucket));
      return;
    }
    const next = [...selectedBuckets, bucket];
    // Selecting every reason is the same as no filter.
    if (options.length > 0 && next.length >= options.length) {
      onBucketsChange([]);
      return;
    }
    onBucketsChange(next);
  }

  return (
    <section className="admin-panel">
      <div className="admin-panel-head">
        <h2>
          {t(locale, 'admin.refusalsTitle')}
          <span className="admin-count">{selectedBuckets.length ? `${total} / ${totalAll}` : total}</span>
        </h2>
        <div className="admin-panel-head-actions">
          <div className="admin-popover-anchor" ref={filterRef}>
            <button
              type="button"
              className="admin-btn"
              disabled={busy || loading || options.length === 0}
              aria-expanded={filterOpen}
              onClick={() => setFilterOpen((open) => !open)}
            >
              <ListFilter aria-hidden="true" size={16} />
              {t(locale, 'admin.refusalFilter')}
              {selectedBuckets.length ? <span className="admin-badge-count">{selectedBuckets.length}</span> : null}
            </button>
            {filterOpen ? (
              <div className="admin-popover">
                <p className="admin-help">{t(locale, 'admin.refusalFilterHelp')}</p>
                <ul>
                  {options.map((option) => {
                    const checked = selectedBuckets.includes(option.bucket);
                    return (
                      <li key={option.bucket}>
                        <label>
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={
                              busy ||
                              loading ||
                              // A reason that never happened would only filter to an empty table.
                              (!checked && option.count === 0) ||
                              (!checked && options.length > 1 && selectedBuckets.length >= options.length - 1)
                            }
                            onChange={() => toggleBucket(option.bucket)}
                          />
                          <span>{option.title}</span>
                          <em>{option.count}</em>
                        </label>
                      </li>
                    );
                  })}
                </ul>
                {selectedBuckets.length ? (
                  <button type="button" className="admin-btn small" disabled={busy || loading} onClick={() => onBucketsChange([])}>
                    {t(locale, 'admin.refusalFilterClear')}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
          <button type="button" className="admin-btn" disabled={busy || loading || total === 0} onClick={onExport}>
            <Download aria-hidden="true" size={16} />
            {t(locale, 'admin.exportRefusals')}
          </button>
        </div>
      </div>
      <p className="admin-help">{t(locale, 'admin.refusalsPageHelp')}</p>
      {loading ? (
        <BlockSkeleton label={t(locale, 'admin.loading')} />
      ) : !items.length ? (
        <p className="admin-empty">{t(locale, 'admin.refusalsEmpty')}</p>
      ) : (
        <div className="admin-table-wrap">
          <table className="admin-table">
            <thead>
              <tr>
                <th><span className="sr-only">{t(locale, 'admin.showDetails')}</span></th>
                <th>{t(locale, 'admin.refusalWhen')}</th>
                <th>{t(locale, 'admin.refusalQuestion')}</th>
                <th>{t(locale, 'admin.refusalReason')}</th>
                <th>{t(locale, 'admin.refusalUser')}</th>
                <th>{t(locale, 'admin.refusalStatus')}</th>
                <th>{t(locale, 'admin.refusalProfile')}</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item: AnswerRefusal) => {
                const open = openId === item.refusal_id;
                // The reason is the diagnostics joined with " | "; show them one per line.
                const diagnostics = item.diagnostics?.length ? item.diagnostics : [item.reason];
                return (
                  <Fragment key={item.refusal_id}>
                    <tr className={open ? 'is-open' : undefined}>
                      <td>
                        <button
                          type="button"
                          className="admin-icon-btn admin-expand"
                          aria-expanded={open}
                          aria-label={t(locale, 'admin.showDetails')}
                          title={t(locale, 'admin.showDetails')}
                          onClick={() => setOpenId(open ? null : item.refusal_id)}
                        >
                          <ChevronRight aria-hidden="true" size={16} className="admin-flip" />
                        </button>
                      </td>
                      <td className="num admin-muted">{formatWhen(item.created_at, locale)}</td>
                      <td><span className="admin-question" title={item.question}>{item.question}</span></td>
                      <td><span className="admin-reason" title={item.reason}>{item.reason_title || item.reason}</span></td>
                      <td>{item.user_email || '—'}</td>
                      <td className="admin-code">{item.answer_status}</td>
                      <td className="admin-code">{item.profile || '—'}</td>
                    </tr>
                    {open ? (
                      <tr className="admin-detail-row">
                        <td />
                        <td colSpan={6}>
                          <dl className="admin-detail">
                            <dt>{t(locale, 'admin.refusalQuestion')}</dt>
                            <dd>{item.question}</dd>
                            <dt>{t(locale, 'admin.refusalDiagnostics')}</dt>
                            <dd>
                              <ul className="admin-code">
                                {diagnostics.map((line, index) => <li key={index}>{line}</li>)}
                              </ul>
                            </dd>
                            {item.conversation_id ? (
                              <>
                                <dt>{t(locale, 'admin.refusalConversation')}</dt>
                                <dd className="admin-code">{item.conversation_id}</dd>
                              </>
                            ) : null}
                          </dl>
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
