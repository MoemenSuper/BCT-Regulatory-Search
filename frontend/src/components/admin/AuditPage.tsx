// Who did what in the admin dashboard, newest first. Secret values are never logged.
import { type AuditEntry } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { docKindLabel, formatWhen, type DocKind } from './shared';
import { BlockSkeleton } from './Skeletons';

function actionLabel(action: string, locale: UiLocale) {
  const key = `admin.action.${action}`;
  const label = t(locale, key);
  return label === key ? action : label;
}

// Uploads store the document kind as a code; show it in the page language.
function detailLabel(entry: AuditEntry, locale: UiLocale) {
  if (entry.action === 'document.upload' && entry.detail in docKindLabel) {
    return t(locale, docKindLabel[entry.detail as DocKind]);
  }
  return entry.detail || '—';
}

export function AuditPage({ entries, loading, locale }: { entries: AuditEntry[]; loading: boolean; locale: UiLocale }) {
  return (
    <section className="admin-panel">
      <div className="admin-panel-head">
        <h2>{t(locale, 'admin.audit')}<span className="admin-count">{entries.length}</span></h2>
      </div>
      <p className="admin-help">{t(locale, 'admin.auditHelp')}</p>
      {loading ? (
        <BlockSkeleton label={t(locale, 'admin.loading')} />
      ) : !entries.length ? (
        <p className="admin-empty">{t(locale, 'admin.auditEmpty')}</p>
      ) : (
        <div className="admin-table-wrap">
          <table className="admin-table">
            <thead>
              <tr>
                <th>{t(locale, 'admin.refusalWhen')}</th>
                <th>{t(locale, 'admin.auditActor')}</th>
                <th>{t(locale, 'admin.auditAction')}</th>
                <th>{t(locale, 'admin.auditTarget')}</th>
                <th>{t(locale, 'admin.auditDetail')}</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.audit_id}>
                  <td className="num admin-muted">{formatWhen(entry.created_at, locale)}</td>
                  <td>{entry.actor_email}</td>
                  <td><span className="admin-reason">{actionLabel(entry.action, locale)}</span></td>
                  <td className={entry.action.startsWith('config.') ? 'admin-code' : undefined}>{entry.target || '—'}</td>
                  <td className="admin-muted">{detailLabel(entry, locale)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
