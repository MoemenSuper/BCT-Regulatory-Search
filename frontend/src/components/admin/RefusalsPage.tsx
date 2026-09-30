// Answers the assistant declined, filterable and exportable as CSV.
import { useState } from 'react';
import { Download, ListFilter } from 'lucide-react';
import { type AnswerRefusal, type AnswerRefusalOption, type AnswerRefusalsPage } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { TableSkeleton } from './Skeletons';

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
    <section className="admin-panel admin-table-panel admin-refusals-page">
      <div className="admin-panel-heading">
        <div>
          <p>{t(locale, 'admin.refusals')}</p>
          <h2>{t(locale, 'admin.refusalsTitle')}</h2>
        </div>
        <div className="admin-refusals-actions">
          <span className="admin-count">{selectedBuckets.length ? `${total} / ${totalAll}` : total}</span>
          <div className="admin-refusal-filter">
            <button
              type="button"
              className={`admin-refresh${selectedBuckets.length ? ' is-active' : ''}`}
              disabled={busy || loading || options.length === 0}
              aria-expanded={filterOpen}
              aria-haspopup="true"
              onClick={() => setFilterOpen((open) => !open)}
            >
              <ListFilter aria-hidden="true" size={17} />
              <span>{t(locale, 'admin.refusalFilter')}</span>
              {selectedBuckets.length ? <em>{selectedBuckets.length}</em> : null}
            </button>
            {filterOpen ? (
              <div className="admin-refusal-filter-menu" role="menu">
                <p>{t(locale, 'admin.refusalFilterHelp')}</p>
                <ul>
                  {options.map((option) => {
                    const checked = selectedBuckets.includes(option.bucket);
                    return (
                      <li key={option.bucket}>
                        <label className={checked ? 'is-selected' : undefined}>
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={
                              busy ||
                              loading ||
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
                  <button type="button" className="admin-action" disabled={busy || loading} onClick={() => onBucketsChange([])}>
                    {t(locale, 'admin.refusalFilterClear')}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
          <button type="button" className="admin-refresh" disabled={busy || loading || total === 0} onClick={onExport}>
            <Download aria-hidden="true" size={17} />
            <span>{t(locale, 'admin.exportRefusals')}</span>
          </button>
        </div>
      </div>
      <p className="admin-help">{t(locale, 'admin.refusalsPageHelp')}</p>
      {loading ? (
        <TableSkeleton label={t(locale, 'admin.loading')} />
      ) : !items.length ? (
        <p className="admin-empty">{t(locale, 'admin.refusalsEmpty')}</p>
      ) : (
        <div className="admin-table-wrap">
          <table className="admin-table admin-refusals-table">
            <thead>
              <tr>
                <th>{t(locale, 'admin.refusalWhen')}</th>
                <th>{t(locale, 'admin.refusalUser')}</th>
                <th>{t(locale, 'admin.refusalStatus')}</th>
                <th>{t(locale, 'admin.refusalProfile')}</th>
                <th>{t(locale, 'admin.refusalReason')}</th>
                <th>{t(locale, 'admin.refusalQuestion')}</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item: AnswerRefusal) => (
                <tr key={item.refusal_id}>
                  <td className="admin-refusal-when">{item.created_at}</td>
                  <td>{item.user_email || '—'}</td>
                  <td><code>{item.answer_status}</code></td>
                  <td>{item.profile || '—'}</td>
                  <td>
                    <span className="admin-refusal-reason" title={item.reason}>
                      {item.reason_title || item.reason}
                    </span>
                  </td>
                  <td className="admin-refusal-question">{item.question}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
