// Admin home: the key figures (each opens its page), then the work waiting:
// access requests to decide and the latest refusals to read.
import { ArrowRight } from 'lucide-react';
import { type AdminOverview, type AnswerRefusal } from '../../api/admin';
import { type AuthUser } from '../../api/auth';
import { AvatarMark, displayLabel } from '../ProfileMenu';
import { t, type UiLocale } from '../../uiLocale';
import { formatWhen, type AdminTab } from './shared';
import { MetricsSkeleton } from './Skeletons';

const profileTitleKey: Record<string, string> = {
  cloud: 'admin.profileCloudTitle',
  local_hybrid: 'admin.profileHybridTitle',
  local: 'admin.profileLocalTitle',
};

export function OverviewPage({ overview, users, recentRefusals, busy, loading, locale, onNavigate, onApprove, onReject }: {
  overview: AdminOverview | null;
  users: AuthUser[];
  recentRefusals: AnswerRefusal[];
  busy: boolean;
  loading: boolean;
  locale: UiLocale;
  onNavigate: (tab: AdminTab) => void;
  onApprove: (id: string) => Promise<void>;
  onReject: (id: string) => Promise<void>;
}) {
  if (loading || !overview) return <MetricsSkeleton label={t(locale, 'admin.loading')} />;

  const pending = users.filter((entry) => entry.status === 'pending');
  const profileKey = profileTitleKey[overview.active_profile];
  const metrics = [
    {
      label: t(locale, 'admin.approvedUsers'),
      value: overview.users_approved,
      detail: t(locale, 'admin.pendingReview', { count: overview.users_pending }),
      tone: overview.users_pending > 0 ? 'pending' : undefined,
      tab: 'users' as const,
    },
    {
      label: t(locale, 'admin.indexedPdfs'),
      value: overview.documents_ready,
      detail: overview.documents_ready ? t(locale, 'admin.availableCorpus') : t(locale, 'admin.noActivePdfs'),
      tone: overview.documents_ready > 0 ? undefined : 'warn',
      tab: 'documents' as const,
    },
    {
      label: t(locale, 'admin.runtimeProfile'),
      value: profileKey ? t(locale, profileKey) : overview.active_profile,
      detail: t(locale, 'admin.activeRetrieval'),
      tab: 'configuration' as const,
    },
    {
      label: t(locale, 'admin.supersession'),
      value: overview.supersession.ready ? t(locale, 'admin.ready') : t(locale, 'admin.unavailable'),
      detail: overview.supersession.ready
        ? t(locale, 'admin.supersessionEdges', { count: overview.supersession.edge_count })
        : t(locale, 'admin.supersessionEmpty'),
      tone: overview.supersession.ready ? undefined : 'warn',
      // No admin page manages supersession edges, so this card links nowhere.
      tab: undefined,
    },
  ];

  return (
    <>
      <section className="admin-metrics" aria-label={t(locale, 'admin.overview')}>
        {metrics.map((metric) => {
          const body = (
            <>
              <p>{metric.label}</p>
              <strong className={metric.tone === 'warn' ? 'is-warn' : undefined}>{metric.value}</strong>
              <span className={metric.tone ? `is-${metric.tone}` : undefined}>{metric.detail}</span>
            </>
          );
          const tab = metric.tab;
          return tab
            ? <button type="button" className="admin-metric" key={metric.label} onClick={() => onNavigate(tab)}>{body}</button>
            : <div className="admin-metric" key={metric.label}>{body}</div>;
        })}
      </section>

      <div className="admin-overview-panels">
        <section className="admin-panel" aria-labelledby="pending-heading">
          <div className="admin-panel-head">
            <h2 id="pending-heading">{t(locale, 'admin.accessReview')}<span className="admin-count">{pending.length}</span></h2>
            <button type="button" className="admin-btn small" onClick={() => onNavigate('users')}>
              {t(locale, 'admin.viewAll')}
              <ArrowRight aria-hidden="true" size={14} className="admin-flip" />
            </button>
          </div>
          {pending.length ? (
            <ul className="admin-overview-list">
              {pending.map((entry) => (
                <li key={entry.id}>
                  <div className="admin-user-cell">
                    <AvatarMark user={entry} size={32} />
                    <div>
                      <strong>{displayLabel(entry)}</strong>
                      <span>{entry.email}</span>
                    </div>
                  </div>
                  <div className="admin-actions">
                    <button type="button" className="admin-btn small primary" disabled={busy} onClick={() => void onApprove(entry.id)}>
                      {t(locale, 'admin.approve')}
                    </button>
                    <button type="button" className="admin-btn small" disabled={busy} onClick={() => void onReject(entry.id)}>
                      {t(locale, 'admin.reject')}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="admin-empty">{t(locale, 'admin.noPendingRequests')}</p>
          )}
        </section>

        <section className="admin-panel" aria-labelledby="refusals-heading">
          <div className="admin-panel-head">
            <h2 id="refusals-heading">{t(locale, 'admin.recentRefusals')}<span className="admin-count">{overview.answer_refusals_total}</span></h2>
            <button type="button" className="admin-btn small" onClick={() => onNavigate('refusals')}>
              {t(locale, 'admin.viewAll')}
              <ArrowRight aria-hidden="true" size={14} className="admin-flip" />
            </button>
          </div>
          {recentRefusals.length ? (
            <ul className="admin-overview-list">
              {recentRefusals.map((item) => (
                <li key={item.refusal_id}>
                  <div className="admin-overview-refusal">
                    <span className="admin-question" title={item.question}>{item.question}</span>
                    <span className="admin-muted">{formatWhen(item.created_at, locale)} · {item.user_email || '—'}</span>
                  </div>
                  <span className="admin-reason" title={item.reason}>{item.reason_title || item.reason}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="admin-empty">{t(locale, 'admin.refusalsEmpty')}</p>
          )}
        </section>
      </div>
    </>
  );
}
