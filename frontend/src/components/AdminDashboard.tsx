import { useEffect, useState, type FormEvent } from 'react';
import { flushSync } from 'react-dom';
import { Activity, ArrowUpRight, CheckCircle2, CircleAlert, Database, Download, FileText, FileUp, Gauge, KeyRound, ListFilter, Network, RefreshCw, Settings2, ShieldCheck, ShieldPlus, Trash2, UserCheck, UsersRound, XCircle } from 'lucide-react';
import { approveUser, deleteUser, downloadAnswerRefusalsExport, getConfig, getOverview, listAnswerRefusals, listDocuments, listUsers, promoteUser, rejectUser, resetUserTokens, setCloudRetrievalProvider, setProfile, setSecrets, setUserTokenLimit, uploadDocument, type AdminConfig, type AdminOverview, type AnswerRefusal, type AnswerRefusalOption, type AnswerRefusalsPage } from '../api/admin';
import { logout, type AuthUser } from '../api/auth';
import { LanguageSwitcher } from './LanguageSwitcher';
import { ProfileMenu, displayLabel, AvatarMark } from './ProfileMenu';
import { languageDirection, t, type UiLocale } from '../uiLocale';

type AdminTab = 'overview' | 'users' | 'documents' | 'refusals' | 'configuration';

interface AdminDashboardProps {
  user: AuthUser;
  onUserChange: (user: AuthUser) => void;
  onLogout: () => void;
  locale: UiLocale;
  onLocaleChange: (locale: UiLocale) => void;
}

function isPdfFile(file: File) {
  return file.name.toLowerCase().endsWith('.pdf') && file.size > 0 && file.size <= 50 * 1024 * 1024;
}

export function AdminDashboard({ user, onUserChange, onLogout, locale, onLocaleChange }: AdminDashboardProps) {
  const [tab, setTab] = useState<AdminTab>('overview');
  const [overview, setOverview] = useState<AdminOverview | null>(null);
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [documents, setDocuments] = useState<unknown[]>([]);
  const [refusals, setRefusals] = useState<AnswerRefusalsPage | null>(null);
  const [refusalBuckets, setRefusalBuckets] = useState<string[]>([]);
  const [config, setConfig] = useState<AdminConfig | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [files, setFiles] = useState<File[]>([]);
  const [fileKey, setFileKey] = useState(0);
  const navigation = [
    { id: 'overview' as const, label: t(locale, 'admin.overview'), icon: Gauge },
    { id: 'users' as const, label: t(locale, 'admin.users'), icon: UsersRound },
    { id: 'documents' as const, label: t(locale, 'admin.documents'), icon: FileText },
    { id: 'refusals' as const, label: t(locale, 'admin.refusals'), icon: CircleAlert },
    { id: 'configuration' as const, label: t(locale, 'admin.configuration'), icon: Settings2 },
  ];

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setError(null); setLoading(true);
      try {
        if (tab === 'overview') { const data = await getOverview(); if (!cancelled) setOverview(data); }
        else if (tab === 'users') { const data = await listUsers(); if (!cancelled) setUsers(data); }
        else if (tab === 'documents') { const data = await listDocuments(); if (!cancelled) setDocuments(data); }
        else if (tab === 'refusals') { const data = await listAnswerRefusals(5000, refusalBuckets); if (!cancelled) setRefusals(data); }
        else { const data = await getConfig(); if (!cancelled) setConfig(data); }
      } catch (err) { if (!cancelled) setError(err instanceof Error && err.message ? err.message : t(locale, 'admin.loadFailed')); }
      finally { if (!cancelled) setLoading(false); }
    }
    void load();
    return () => { cancelled = true; };
  }, [tab, locale, refusalBuckets]);

  async function refresh() {
    setError(null); setLoading(true);
    try {
      if (tab === 'overview') setOverview(await getOverview());
      if (tab === 'users') setUsers(await listUsers());
      if (tab === 'documents') setDocuments(await listDocuments());
      if (tab === 'refusals') setRefusals(await listAnswerRefusals(5000, refusalBuckets));
      if (tab === 'configuration') setConfig(await getConfig());
    } catch { setError(t(locale, 'admin.refreshFailed')); }
    finally { setLoading(false); }
  }

  function selectTab(next: AdminTab) { setMessage(null); setError(null); setTab(next); }

  async function handleLogout() { await logout(); onLogout(); }
  async function handleApprove(id: string) { setBusy(true); try { await approveUser(id); setMessage(t(locale, 'admin.approved')); await refresh(); } catch { setError(t(locale, 'admin.userActionFailed')); } finally { setBusy(false); } }
  async function handlePromote(id: string, email: string) {
    if (!window.confirm(t(locale, 'admin.promoteConfirm', { email }))) return;
    setBusy(true);
    try {
      await promoteUser(id);
      setMessage(t(locale, 'admin.promoted'));
      await refresh();
    } catch {
      setError(t(locale, 'admin.userActionFailed'));
    } finally {
      setBusy(false);
    }
  }
  async function handleReject(id: string) { setBusy(true); try { await rejectUser(id); setMessage(t(locale, 'admin.rejected')); await refresh(); } catch { setError(t(locale, 'admin.userActionFailed')); } finally { setBusy(false); } }
  async function handleDelete(id: string, email: string) { if (!window.confirm(t(locale, 'admin.deleteConfirm', { email }))) return; setBusy(true); try { await deleteUser(id); setMessage(t(locale, 'admin.deleted')); await refresh(); } catch { setError(t(locale, 'admin.userActionFailed')); } finally { setBusy(false); } }

  async function handleTokenLimit(id: string, tokenLimit: number) {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await setUserTokenLimit(id, tokenLimit);
      setMessage(t(locale, 'admin.tokenLimitSaved'));
      await refresh();
    } catch {
      setError(t(locale, 'admin.tokenLimitFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleResetTokens(id: string, email: string) {
    if (!window.confirm(t(locale, 'admin.resetTokensConfirm', { email }))) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await resetUserTokens(id);
      setMessage(t(locale, 'admin.tokensReset'));
      await refresh();
    } catch {
      setError(t(locale, 'admin.tokenLimitFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleProfile(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const profile = String(new FormData(event.currentTarget).get('profile') || ''); setBusy(true); setError(null); setMessage(null); try { await setProfile(profile); setMessage(t(locale, 'admin.profileSaved', { profile })); await refresh(); } catch { setError(t(locale, 'admin.configFailed')); } finally { setBusy(false); } }
  async function handleCloudProvider(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const provider = String(new FormData(event.currentTarget).get('cloud_retrieval_provider') || '');
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await setCloudRetrievalProvider(provider);
      setConfig(result.config);
      setMessage(t(locale, 'admin.cloudProviderSaved', { provider: result.cloud_retrieval_provider }));
    } catch {
      setError(t(locale, 'admin.configFailed'));
    } finally {
      setBusy(false);
    }
  }
  async function handleSecrets(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const secrets: Record<string, string | null> = {};
    for (const [key, value] of new FormData(form).entries()) {
      const text = String(value).trim();
      if (text) secrets[key] = text;
    }
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      setConfig(await setSecrets(secrets));
      setMessage(t(locale, 'admin.credentialsSaved'));
      form.reset();
    } catch {
      setError(t(locale, 'admin.configFailed'));
    } finally {
      setBusy(false);
    }
  }
  async function handleExportRefusals() {
    setBusy(true);
    setError(null);
    try {
      await downloadAnswerRefusalsExport(refusalBuckets);
      setMessage(t(locale, 'admin.refusalsExported'));
    } catch {
      setError(t(locale, 'admin.refusalsExportFailed'));
    } finally {
      setBusy(false);
    }
  }
  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    event.stopPropagation();
    const selected = files;
    if (!selected.length) {
      setError(t(locale, 'admin.fileRequired'));
      return;
    }
    if (selected.some((file) => !isPdfFile(file))) {
      setError(t(locale, 'admin.fileInvalid'));
      return;
    }
    flushSync(() => {
      setBusy(true);
      setError(null);
      setMessage(null);
    });
    let ok = 0;
    let duplicates = 0;
    const failures: string[] = [];
    try {
      for (let index = 0; index < selected.length; index += 1) {
        const file = selected[index];
        setMessage(t(locale, 'admin.uploadBatchProgress', { current: index + 1, total: selected.length }));
        const form = new FormData();
        form.append('file', file);
        try {
          const report = await uploadDocument(form) as { duplicate?: boolean };
          if (report.duplicate) duplicates += 1;
          else ok += 1;
        } catch (err) {
          failures.push(`${file.name}: ${err instanceof Error && err.message ? err.message : t(locale, 'admin.uploadFailed')}`);
        }
      }
      setFiles([]);
      setFileKey((value) => value + 1);
      await refresh();
      if (failures.length && !ok && !duplicates) {
        setMessage(null);
        setError(failures[0]);
      } else if (failures.length) {
        setMessage(t(locale, 'admin.uploadBatchPartial', { ok: ok + duplicates, failed: failures.length }));
        setError(failures[0]);
      } else if (selected.length === 1 && duplicates === 1) {
        setMessage(t(locale, 'admin.duplicate'));
      } else if (selected.length === 1) {
        setMessage(t(locale, 'admin.uploadSuccess'));
      } else {
        setMessage(t(locale, 'admin.uploadBatchDone', { count: ok + duplicates }));
      }
    } finally {
      setBusy(false);
    }
  }

  const currentPage = navigation.find((item) => item.id === tab)?.label || t(locale, 'admin.overview');
  return <div className="admin-shell" lang={locale} dir={languageDirection(locale)}>
    <a className="admin-skip-link" href="#admin-content">{t(locale, 'admin.skip')}</a>
    <aside className="admin-sidebar" aria-label={t(locale, 'admin.configuration')}>
      <div className="admin-brand"><img src="/bct-logo-white.png" alt="Banque Centrale de Tunisie" /><span>{t(locale, 'admin.brand')}</span></div>
      <nav className="admin-nav">{navigation.map(({ id, label, icon: Icon }) => <button key={id} type="button" className={tab === id ? 'active' : ''} aria-current={tab === id ? 'page' : undefined} onClick={() => selectTab(id)}><Icon aria-hidden="true" size={19} strokeWidth={1.8} /><span>{label}</span></button>)}</nav>
    </aside>
    <div className="admin-workspace">
      <header className="admin-header">
        <div>
          <p className="admin-breadcrumb">{t(locale, 'admin.configuration')} <span>/</span> {currentPage}</p>
          <h1>{currentPage}</h1>
        </div>
        <div className="admin-header-actions">
          <LanguageSwitcher locale={locale} onChange={onLocaleChange} className="admin-language-switcher" />
          <button type="button" className="admin-refresh" aria-label={t(locale, 'admin.refresh')} onClick={() => void refresh()} disabled={loading || busy}>
            <RefreshCw aria-hidden="true" size={17} className={loading ? 'is-spinning' : ''} />
            <span>{t(locale, 'admin.refresh')}</span>
          </button>
          <ProfileMenu
            user={user}
            locale={locale}
            onUserChange={onUserChange}
            onLogout={() => void handleLogout()}
            variant="header"
          />
        </div>
      </header>
      <main id="admin-content" className="admin-main">
        {error ? <div className="admin-banner error" role="alert"><XCircle aria-hidden="true" size={19} />{error}</div> : null}
        {message ? <div className="admin-banner ok" role="status"><CheckCircle2 aria-hidden="true" size={19} />{message}</div> : null}
        {tab === 'overview' ? <OverviewPage overview={overview} loading={loading} locale={locale} onNavigate={selectTab} /> : null}
        {tab === 'users' ? <UsersPage users={users} currentUser={user} busy={busy} loading={loading} locale={locale} onApprove={handleApprove} onPromote={handlePromote} onReject={handleReject} onDelete={handleDelete} onTokenLimit={handleTokenLimit} onResetTokens={handleResetTokens} /> : null}
        {tab === 'documents' ? (
          <DocumentsPage
            documents={Array.isArray(documents) ? documents : []}
            loading={loading}
            busy={busy}
            locale={locale}
            files={files}
            fileKey={fileKey}
            onUpload={(event) => void handleUpload(event)}
            onFilesChange={(next) => { setFiles(next); setError(null); }}
          />
        ) : null}
        {tab === 'refusals' ? (
          <RefusalsPage
            page={refusals}
            loading={loading}
            busy={busy}
            locale={locale}
            selectedBuckets={refusalBuckets}
            onBucketsChange={setRefusalBuckets}
            onExport={() => void handleExportRefusals()}
          />
        ) : null}
        {tab === 'configuration' ? <ConfigurationPage config={config} loading={loading} busy={busy} locale={locale} onProfile={handleProfile} onCloudProvider={handleCloudProvider} onSecrets={handleSecrets} /> : null}
      </main>
    </div>
  </div>;
}

function OverviewPage({ overview, loading, locale, onNavigate }: { overview: AdminOverview | null; loading: boolean; locale: UiLocale; onNavigate: (tab: AdminTab) => void }) {
  if (loading || !overview) return <><section className="admin-hero"><div className="admin-hero-copy"><p>{t(locale, 'admin.operations')}</p><h2>{t(locale, 'admin.heroTitle')}</h2></div><img src="/bct-building.jpg" alt="" /></section><OverviewSkeleton label={t(locale, 'admin.loading')} /></>;
  const cards = [{ label: t(locale, 'admin.approvedUsers'), value: overview.users_approved, detail: t(locale, 'admin.pendingReview', { count: overview.users_pending }), icon: UsersRound, color: 'blue' }, { label: t(locale, 'admin.indexedPdfs'), value: overview.documents_ready, detail: t(locale, 'admin.availableCorpus'), icon: Database, color: 'red' }, { label: t(locale, 'admin.runtimeProfile'), value: overview.active_profile, detail: t(locale, 'admin.activeRetrieval'), icon: Activity, color: 'navy' }, { label: t(locale, 'admin.graphLite'), value: overview.graph.graph_ready ? t(locale, 'admin.ready') : t(locale, 'admin.unavailable'), detail: overview.graph.graph_enabled ? t(locale, 'admin.graphEnabled') : t(locale, 'admin.graphDisabled'), icon: Network, color: overview.graph.graph_ready ? 'green' : 'gray' }];
  const focus = [{ title: t(locale, 'admin.accessReview'), detail: overview.users_pending ? t(locale, 'admin.pendingRequests', { count: overview.users_pending }) : t(locale, 'admin.noPendingRequests'), action: t(locale, 'admin.reviewUsers'), tab: 'users' as const, icon: UsersRound }, { title: t(locale, 'admin.refusals'), detail: overview.answer_refusals_total ? t(locale, 'admin.refusalsHelp', { count: overview.answer_refusals_total }) : t(locale, 'admin.refusalsEmpty'), action: t(locale, 'admin.reviewRefusals'), tab: 'refusals' as const, icon: CircleAlert }, { title: t(locale, 'admin.corpusReadiness'), detail: overview.documents_ready ? t(locale, 'admin.activePdfCount', { count: overview.documents_ready }) : t(locale, 'admin.noActivePdfs'), action: t(locale, 'admin.inspectDocuments'), tab: 'documents' as const, icon: Database }, { title: t(locale, 'admin.relationshipIndex'), detail: overview.graph.graph_ready ? t(locale, 'admin.graphReady') : overview.graph.graph_enabled ? t(locale, 'admin.graphNotReady') : t(locale, 'admin.graphOff'), action: t(locale, 'admin.openConfiguration'), tab: 'configuration' as const, icon: Network }];
  return <><section className="admin-hero" aria-labelledby="overview-heading"><div className="admin-hero-copy"><p>{t(locale, 'admin.operations')}</p><h2 id="overview-heading">{t(locale, 'admin.heroTitle')}</h2><span>{t(locale, 'admin.heroText')}</span></div><img src="/bct-building.jpg" alt={t(locale, 'auth.eyebrow')} /></section><section className="admin-overview-grid">{cards.map(({ label, value, detail, icon: Icon, color }) => <article className={`admin-stat-card ${color}`} key={label}><div className="admin-stat-icon"><Icon aria-hidden="true" size={21} /></div><p>{label}</p><strong>{value}</strong><span>{detail}</span></article>)}</section><section className="admin-overview-lower"><article className="admin-focus-panel"><div className="admin-panel-heading"><div><p>{t(locale, 'admin.operationalFocus')}</p><h2>{t(locale, 'admin.needsAttention')}</h2></div><CircleAlert aria-hidden="true" size={21} /></div><div className="admin-focus-list">{focus.map(({ title, detail, action, tab, icon: Icon }) => <div className="admin-focus-item" key={title}><span className="admin-focus-icon"><Icon aria-hidden="true" size={18} /></span><div><strong>{title}</strong><p>{detail}</p></div><button type="button" onClick={() => onNavigate(tab)}>{action}<ArrowUpRight aria-hidden="true" size={16} /></button></div>)}</div></article><aside className="admin-grounding-panel"><div className="admin-grounding-mark"><ShieldCheck aria-hidden="true" size={22} /></div><p>{t(locale, 'admin.safeguards')}</p><h2>{t(locale, 'admin.safeguardTitle')}</h2><span>{t(locale, 'admin.safeguardText')}</span></aside></section></>;
}

function RefusalsPage({
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
    if (selectedBuckets.length >= 3) return;
    onBucketsChange([...selectedBuckets, bucket]);
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
                            disabled={busy || loading || (!checked && selectedBuckets.length >= 3)}
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

function formatTokens(value: number | undefined) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value || 0);
}

function formatUsd(value: number | undefined) {
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 4 }).format(value || 0);
}

function UsersPage({ users, currentUser, busy, loading, locale, onApprove, onPromote, onReject, onDelete, onTokenLimit, onResetTokens }: {
  users: AuthUser[];
  currentUser: AuthUser;
  busy: boolean;
  loading: boolean;
  locale: UiLocale;
  onApprove: (id: string) => Promise<void>;
  onPromote: (id: string, email: string) => Promise<void>;
  onReject: (id: string) => Promise<void>;
  onDelete: (id: string, email: string) => Promise<void>;
  onTokenLimit: (id: string, tokenLimit: number) => Promise<void>;
  onResetTokens: (id: string, email: string) => Promise<void>;
}) {
  const [draftLimits, setDraftLimits] = useState<Record<string, string>>({});

  useEffect(() => {
    const next: Record<string, string> = {};
    for (const entry of users) next[entry.id] = String(entry.token_limit ?? 0);
    setDraftLimits(next);
  }, [users]);

  return (
    <>
      <section className="admin-panel admin-table-panel">
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.accessControl')}</p>
            <h2>{t(locale, 'admin.userDirectory')}</h2>
          </div>
          <span className="admin-count">{users.length} {t(locale, 'admin.accounts')}</span>
        </div>
        <p className="admin-help">{t(locale, 'admin.promoteHelp')}</p>
        {loading ? (
          <TableSkeleton label={t(locale, 'admin.loading')} />
        ) : (
          <div className="admin-table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>{t(locale, 'admin.user')}</th>
                  <th>{t(locale, 'admin.role')}</th>
                  <th>{t(locale, 'admin.accountStatus')}</th>
                  <th><span className="sr-only">{t(locale, 'admin.actions')}</span></th>
                </tr>
              </thead>
              <tbody>
                {users.map((entry) => (
                  <tr key={entry.id}>
                    <td>
                      <div className="admin-user-cell">
                        <AvatarMark user={entry} size={32} className="admin-account-mark" />
                        <div>
                          <strong>{displayLabel(entry)}</strong>
                          {entry.display_name?.trim() ? <span className="admin-role">{entry.email}</span> : null}
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="admin-role">
                        {entry.role === 'admin' ? t(locale, 'admin.administrator') : t(locale, 'admin.user')}
                      </span>
                    </td>
                    <td><StatusBadge status={entry.status} locale={locale} /></td>
                    <td className="admin-actions">
                      {entry.status === 'pending' ? (
                        <>
                          <button type="button" className="admin-action accept" disabled={busy} onClick={() => void onApprove(entry.id)}>
                            <UserCheck aria-hidden="true" size={16} />
                            {t(locale, 'admin.approve')}
                          </button>
                          <button type="button" className="admin-action reject" disabled={busy} onClick={() => void onReject(entry.id)}>
                            {t(locale, 'admin.reject')}
                          </button>
                        </>
                      ) : null}
                      {entry.status === 'approved' && entry.role !== 'admin' ? (
                        <button type="button" className="admin-action promote" disabled={busy} onClick={() => void onPromote(entry.id, entry.email)}>
                          <ShieldPlus aria-hidden="true" size={16} />
                          {t(locale, 'admin.promote')}
                        </button>
                      ) : null}
                      {entry.id !== currentUser.id && entry.role !== 'admin' ? (
                        <button type="button" className="admin-action delete" disabled={busy} onClick={() => void onDelete(entry.id, entry.email)}>
                          <Trash2 aria-hidden="true" size={16} />
                          {t(locale, 'admin.delete')}
                        </button>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="admin-panel admin-token-panel">
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.tokenUsage')}</p>
            <h2>{t(locale, 'admin.tokenQuotaTitle')}</h2>
          </div>
        </div>
        <p className="admin-help">{t(locale, 'admin.tokenQuotaHelp')}</p>
        {loading ? (
          <TableSkeleton label={t(locale, 'admin.loading')} />
        ) : (
          <div className="admin-table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>{t(locale, 'admin.user')}</th>
                  <th>{t(locale, 'admin.tokensGroq')}</th>
                  <th>{t(locale, 'admin.tokensEmbed')}</th>
                  <th>{t(locale, 'admin.tokensRerank')}</th>
                  <th>{t(locale, 'admin.tokensUsed')}</th>
                  <th>{t(locale, 'admin.tokenLimit')}</th>
                  <th>{t(locale, 'admin.tokensRemaining')}</th>
                  <th>{t(locale, 'admin.estimatedSpend')}</th>
                  <th><span className="sr-only">{t(locale, 'admin.actions')}</span></th>
                </tr>
              </thead>
              <tbody>
                {users.map((entry) => {
                  const limit = entry.token_limit ?? 0;
                  const used = entry.tokens_used ?? 0;
                  const remaining = limit <= 0 ? null : Math.max(0, limit - used);
                  return (
                    <tr key={`tokens-${entry.id}`}>
                      <td><strong>{entry.email}</strong></td>
                      <td>{formatTokens(entry.tokens_llm ?? 0)}</td>
                      <td>{formatTokens(entry.tokens_embed ?? 0)}</td>
                      <td>{formatTokens(entry.tokens_rerank ?? 0)}</td>
                      <td>{formatTokens(used)}</td>
                      <td>
                        <input
                          className="admin-token-input"
                          type="number"
                          min={0}
                          step={1000}
                          value={draftLimits[entry.id] ?? String(limit)}
                          disabled={busy}
                          aria-label={t(locale, 'admin.tokenLimit')}
                          onChange={(event) => setDraftLimits((current) => ({ ...current, [entry.id]: event.target.value }))}
                        />
                      </td>
                      <td>{remaining === null ? t(locale, 'admin.unlimited') : formatTokens(remaining)}</td>
                      <td>{formatUsd(entry.estimated_spend_usd)}</td>
                      <td className="admin-actions">
                        <button
                          type="button"
                          className="admin-action accept"
                          disabled={busy}
                          onClick={() => {
                            const parsed = Number(draftLimits[entry.id] ?? limit);
                            if (!Number.isFinite(parsed) || parsed < 0) return;
                            void onTokenLimit(entry.id, Math.floor(parsed));
                          }}
                        >
                          {t(locale, 'admin.saveTokenLimit')}
                        </button>
                        <button
                          type="button"
                          className="admin-action reject"
                          disabled={busy || used === 0}
                          onClick={() => void onResetTokens(entry.id, entry.email)}
                        >
                          {t(locale, 'admin.resetTokens')}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}


function DocumentsPage({
  documents, loading, busy, locale, files, fileKey, onUpload, onFilesChange,
}: {
  documents: unknown[];
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  files: File[];
  fileKey: number;
  onUpload: (event: FormEvent<HTMLFormElement>) => void;
  onFilesChange: (files: File[]) => void;
}) {
  const label = files.length
    ? (files.length === 1 ? files[0].name : t(locale, 'admin.filesSelected', { count: files.length }))
    : t(locale, 'admin.choosePdf');
  return (
    <>
      {busy ? (
        <div className="admin-upload-overlay" role="status" aria-live="assertive">
          <div className="admin-upload-overlay-card">
            <p className="admin-upload-overlay-kicker">{t(locale, 'admin.uploading')}</p>
            <h2>{t(locale, 'admin.uploadProgress')}</h2>
            <div className="admin-upload-progress-track" aria-hidden="true"><span /></div>
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
          <div className="admin-field admin-file-field">
            <label className="admin-file-input">
              <span>{t(locale, 'admin.pdfFile')}</span>
              <input
                key={fileKey}
                name="file"
                type="file"
                accept="application/pdf,.pdf"
                multiple
                onChange={(event) => onFilesChange(Array.from(event.currentTarget.files || []))}
              />
              <em>{label}</em>
            </label>
          </div>
          <button type="submit" className="admin-primary-button" disabled={busy || !files.length}>
            <FileUp aria-hidden="true" size={18} />
            {busy ? t(locale, 'admin.uploading') : t(locale, 'admin.upload')}
          </button>
        </form>
        <DocumentsList documents={documents} loading={loading} locale={locale} />
      </section>
    </>
  );
}


function DocumentsList({ documents, loading, locale }: { documents: unknown[]; loading: boolean; locale: UiLocale }) {
  const rows = Array.isArray(documents) ? documents : [];
  return (
    <section className="admin-panel admin-documents-panel">
      <div className="admin-panel-heading">
        <div>
          <p>{t(locale, 'admin.activeCorpus')}</p>
          <h2>{t(locale, 'admin.indexedPdfs')}</h2>
        </div>
        <span className="admin-count">{rows.length} {t(locale, 'admin.readyCount')}</span>
      </div>
      {loading ? (
        <DocumentSkeleton label={t(locale, 'admin.loading')} />
      ) : (
        <ul className="admin-doc-list">
          {rows.map((doc, index) => {
            const item = doc as {
              filename?: string;
              title?: string;
              document_id?: string;
              pages?: number | null;
            };
            const filename = item.filename || '';
            const meta = [
              item.pages != null ? t(locale, 'admin.pageCount', { count: item.pages }) : '',
            ].filter(Boolean);
            const openLabel = t(locale, 'admin.openPdf');
            const body = (
              <>
                <span className="admin-document-icon"><FileText aria-hidden="true" size={18} /></span>
                <div>
                  <strong>{item.title || filename || t(locale, 'admin.pdfDocument')}</strong>
                  {filename ? <span className="admin-doc-filename">{filename}</span> : null}
                  {meta.length ? <span className="admin-doc-meta">{meta.join(' · ')}</span> : null}
                </div>
                <ArrowUpRight aria-hidden="true" size={17} />
              </>
            );
            return (
              <li key={item.document_id || filename || String(index)}>
                {filename ? (
                  <a
                    className="admin-doc-link"
                    href={`/api/sources/${encodeURIComponent(filename)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`${openLabel}: ${item.title || filename}`}
                    title={openLabel}
                  >
                    {body}
                  </a>
                ) : (
                  <div className="admin-doc-link is-disabled">{body}</div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}


function ConfigurationPage({ config, loading, busy, locale, onProfile, onCloudProvider, onSecrets }: {
  config: AdminConfig | null;
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  onProfile: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onCloudProvider: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onSecrets: (event: FormEvent<HTMLFormElement>) => Promise<void>;
}) {
  if (loading || !config) return <ConfigurationSkeleton label={t(locale, 'admin.loading')} />;
  const profileGuides = [
    { value: 'cloud', title: t(locale, 'admin.profileCloudTitle'), body: t(locale, 'admin.profileCloudBody') },
    { value: 'local_hybrid', title: t(locale, 'admin.profileHybridTitle'), body: t(locale, 'admin.profileHybridBody') },
    { value: 'local', title: t(locale, 'admin.profileLocalTitle'), body: t(locale, 'admin.profileLocalBody') },
  ];
  const cloudProviderLabels: Record<string, string> = {
    voyage: t(locale, 'admin.cloudProviderVoyage'),
    google: t(locale, 'admin.cloudProviderGoogle'),
  };
  return (
    <section className="admin-configuration-layout">
      <form className="admin-form admin-panel" onSubmit={(event) => void onProfile(event)}>
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.runtimeControl')}</p>
            <h2>{t(locale, 'admin.executionProfile')}</h2>
          </div>
          <Activity aria-hidden="true" size={22} />
        </div>
        <p className="admin-help">{t(locale, 'admin.profileHelp')}</p>
        <label>
          {t(locale, 'admin.activeProfile')}
          <select name="profile" defaultValue={config.active_profile} key={config.active_profile}>
            {config.profiles.map((profile) => (
              <option key={profile.value} value={profile.value}>{profile.label}</option>
            ))}
          </select>
        </label>
        <button type="submit" className="admin-primary-button" disabled={busy}>
          <CheckCircle2 aria-hidden="true" size={18} />
          {t(locale, 'admin.saveProfile')}
        </button>
        <div className="admin-profile-guide" aria-label={t(locale, 'admin.profileGuide')}>
          <p>{t(locale, 'admin.profileGuide')}</p>
          <ul>
            {profileGuides.map((guide) => (
              <li key={guide.value} className={config.active_profile === guide.value ? 'is-active' : undefined}>
                <strong>{guide.title}</strong>
                <span>{guide.body}</span>
              </li>
            ))}
          </ul>
        </div>
      </form>
      <form className="admin-form admin-panel" onSubmit={(event) => void onCloudProvider(event)}>
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.runtimeControl')}</p>
            <h2>{t(locale, 'admin.cloudProvider')}</h2>
          </div>
          <Network aria-hidden="true" size={22} />
        </div>
        <p className="admin-help">{t(locale, 'admin.cloudProviderHelp')}</p>
        <fieldset className="admin-cloud-provider" disabled={busy}>
          <legend>{t(locale, 'admin.cloudProvider')}</legend>
          {(config.cloud_retrieval_providers || []).map((option) => (
            <label key={option.value} className="admin-choice">
              <input
                type="radio"
                name="cloud_retrieval_provider"
                value={option.value}
                defaultChecked={config.cloud_retrieval_provider === option.value}
                key={`${option.value}-${config.cloud_retrieval_provider}`}
              />
              <span>
                <strong>{cloudProviderLabels[option.value] || option.label}</strong>
                <small>{option.model} · dim {option.dimension}</small>
              </span>
            </label>
          ))}
        </fieldset>
        <button type="submit" className="admin-primary-button" disabled={busy}>
          <CheckCircle2 aria-hidden="true" size={18} />
          {t(locale, 'admin.saveCloudProvider')}
        </button>
      </form>
      <form className="admin-form admin-panel" onSubmit={(event) => void onSecrets(event)}>
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.providerAccess')}</p>
            <h2>{t(locale, 'admin.credentials')}</h2>
          </div>
          <KeyRound aria-hidden="true" size={22} />
        </div>
        <p className="admin-help">{t(locale, 'admin.credentialsHelp')}</p>
        {config.secrets.map((secret) => (
          <label key={secret.key}>
            {secret.key}
            <span className="admin-masked">
              {secret.configured
                ? t(locale, 'admin.configured', { source: secret.source, masked: secret.masked || '' })
                : t(locale, 'admin.notConfigured')}
            </span>
            <input
              name={secret.key}
              type="password"
              autoComplete="off"
              placeholder={secret.configured ? t(locale, 'admin.replaceValue') : t(locale, 'admin.enterValue')}
            />
            <span className="admin-secret-hint">{t(locale, `admin.secretHint.${secret.key}`)}</span>
          </label>
        ))}
        <button type="submit" className="admin-primary-button" disabled={busy}>
          <ShieldCheck aria-hidden="true" size={18} />
          {t(locale, 'admin.saveCredentials')}
        </button>
      </form>
    </section>
  );
}

function StatusBadge({ status, locale }: { status: string; locale: UiLocale }) { const values: Record<UiLocale, Record<string, string>> = { fr: { approved: 'Approuvé', pending: 'En attente', rejected: 'Refusé' }, ar: { approved: 'مقبول', pending: 'قيد الانتظار', rejected: 'مرفوض' }, en: { approved: 'Approved', pending: 'Pending', rejected: 'Rejected' } }; const normalized = status.toLowerCase(); return <span className={`admin-status ${normalized}`}><i aria-hidden="true" />{values[locale][normalized] || status}</span>; }
function OverviewSkeleton({ label }: { label: string }) { return <section className="admin-overview-grid" aria-label={label}><div className="admin-skeleton stat" /><div className="admin-skeleton stat" /><div className="admin-skeleton stat" /><div className="admin-skeleton stat" /></section>; }
function TableSkeleton({ label }: { label: string }) { return <div className="admin-skeleton table" aria-label={label} />; }
function DocumentSkeleton({ label }: { label: string }) { return <div className="admin-skeleton document" aria-label={label} />; }
function ConfigurationSkeleton({ label }: { label: string }) { return <section className="admin-configuration-layout" aria-label={label}><div className="admin-skeleton form" /><div className="admin-skeleton form" /><div className="admin-skeleton form" /></section>; }
