// Admin home: usage and corpus counts, with links to the other pages.
import { ArrowUpRight } from 'lucide-react';
import { type AdminOverview } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { type AdminTab } from './shared';
import { OverviewSkeleton } from './Skeletons';

export function OverviewPage({ overview, loading, locale, onNavigate }: { overview: AdminOverview | null; loading: boolean; locale: UiLocale; onNavigate: (tab: AdminTab) => void }) {
  if (loading || !overview) {
    return (
      <div className="admin-overview-page">
        <section className="admin-ops-banner" aria-busy="true">
          <div className="admin-ops-banner-copy">
            <p>{t(locale, 'admin.operations')}</p>
            <h2>{t(locale, 'admin.heroTitle')}</h2>
          </div>
        </section>
        <OverviewSkeleton label={t(locale, 'admin.loading')} />
      </div>
    );
  }

  const metrics = [
    {
      label: t(locale, 'admin.approvedUsers'),
      value: overview.users_approved,
      detail: t(locale, 'admin.pendingReview', { count: overview.users_pending }),
      tone: overview.users_pending > 0 ? 'pending' : undefined,
    },
    {
      label: t(locale, 'admin.indexedPdfs'),
      value: overview.documents_ready,
      detail: t(locale, 'admin.availableCorpus'),
      tone: overview.documents_ready > 0 ? 'ready' : 'warn',
    },
    {
      label: t(locale, 'admin.runtimeProfile'),
      value: overview.active_profile,
      detail: t(locale, 'admin.activeRetrieval'),
    },
    {
      label: t(locale, 'admin.supersession'),
      value: overview.supersession.ready ? t(locale, 'admin.ready') : t(locale, 'admin.unavailable'),
      detail: t(locale, 'admin.supersessionEdges', { count: overview.supersession.edge_count }),
      tone: overview.supersession.ready ? 'ready' : 'warn',
    },
  ];

  const focus = [
    {
      title: t(locale, 'admin.accessReview'),
      detail: overview.users_pending
        ? t(locale, 'admin.pendingRequests', { count: overview.users_pending })
        : t(locale, 'admin.noPendingRequests'),
      action: t(locale, 'admin.reviewUsers'),
      tab: 'users' as const,
      tone: overview.users_pending > 0 ? 'pending' : undefined,
    },
    {
      title: t(locale, 'admin.refusals'),
      detail: overview.answer_refusals_total
        ? t(locale, 'admin.refusalsHelp', { count: overview.answer_refusals_total })
        : t(locale, 'admin.refusalsEmpty'),
      action: t(locale, 'admin.reviewRefusals'),
      tab: 'refusals' as const,
    },
    {
      title: t(locale, 'admin.corpusReadiness'),
      detail: overview.documents_ready
        ? t(locale, 'admin.activePdfCount', { count: overview.documents_ready })
        : t(locale, 'admin.noActivePdfs'),
      action: t(locale, 'admin.inspectDocuments'),
      tab: 'documents' as const,
      tone: overview.documents_ready > 0 ? 'ready' : 'warn',
    },
    {
      title: t(locale, 'admin.supersessionIndex'),
      detail: overview.supersession.ready
        ? t(locale, 'admin.supersessionReady', { count: overview.supersession.edge_count })
        : t(locale, 'admin.supersessionEmpty'),
      action: t(locale, 'admin.openConfiguration'),
      tab: 'configuration' as const,
      tone: overview.supersession.ready ? 'ready' : 'warn',
    },
  ];

  return (
    <div className="admin-overview-page">
      <section className="admin-ops-banner" aria-labelledby="overview-heading">
        <div className="admin-ops-banner-copy">
          <p>{t(locale, 'admin.operations')}</p>
          <h2 id="overview-heading">{t(locale, 'admin.heroTitle')}</h2>
          <span>{t(locale, 'admin.heroText')}</span>
        </div>
        <div className="admin-ops-banner-side" aria-hidden="true">
          <span className="admin-ops-banner-mark" />
        </div>
      </section>

      <section className="admin-metrics-strip" aria-label={t(locale, 'admin.overview')}>
        {metrics.map((metric) => (
          <article className="admin-metric" key={metric.label}>
            <p>{metric.label}</p>
            <strong className={metric.tone ? `is-${metric.tone}` : undefined}>{metric.value}</strong>
            <span className={metric.tone === 'pending' ? 'is-pending' : undefined}>{metric.detail}</span>
          </article>
        ))}
      </section>

      <div className="admin-overview-split">
        <section className="admin-attention-panel" aria-labelledby="attention-heading">
          <header className="admin-attention-heading">
            <h2 id="attention-heading">{t(locale, 'admin.operationalFocus')}</h2>
            <p>{t(locale, 'admin.needsAttention')}</p>
          </header>
          <ul className="admin-attention-list">
            {focus.map((item) => (
              <li key={item.title}>
                <div className="admin-attention-copy">
                  <strong>{item.title}</strong>
                  <p className={item.tone ? `is-${item.tone}` : undefined}>{item.detail}</p>
                </div>
                <button type="button" onClick={() => onNavigate(item.tab)}>
                  {item.action}
                  <ArrowUpRight aria-hidden="true" size={14} strokeWidth={1.8} />
                </button>
              </li>
            ))}
          </ul>
        </section>

        <aside className="admin-safeguard-note">
          <p>{t(locale, 'admin.safeguards')}</p>
          <strong>{t(locale, 'admin.safeguardTitle')}</strong>
          <span>{t(locale, 'admin.safeguardText')}</span>
        </aside>
      </div>
    </div>
  );
}
