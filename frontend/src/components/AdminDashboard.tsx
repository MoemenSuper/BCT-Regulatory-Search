import { useEffect, useState, type FormEvent } from 'react';
import { flushSync } from 'react-dom';
import { Activity, ArrowUpRight, CheckCircle2, CircleAlert, Download, FileText, FileUp, Gauge, KeyRound, ListFilter, Moon, Network, Settings2, ShieldCheck, ShieldPlus, Sun, Trash2, UserCheck, UsersRound, XCircle } from 'lucide-react';
import { approveUser, deleteUser, downloadAnswerRefusalsExport, getConfig, getOverview, listAnswerRefusals, listDocuments, listUsers, promoteUser, rejectUser, resetUserTokens, setCloudRetrievalProvider, setProfile, setSecrets, setUserTokenLimit, uploadDocument, type AdminConfig, type AdminOverview, type AnswerRefusal, type AnswerRefusalOption, type AnswerRefusalsPage } from '../api/admin';
import { logout, type AuthUser } from '../api/auth';
import { LanguageSwitcher } from './LanguageSwitcher';
import { ProfileMenu, displayLabel, AvatarMark } from './ProfileMenu';
import { languageDirection, t, type UiLocale } from '../uiLocale';

type AdminTab = 'overview' | 'users' | 'documents' | 'refusals' | 'configuration';
type AdminTheme = 'light' | 'dark';

const ADMIN_THEME_KEY = 'bct-admin-theme';

function readAdminTheme(): AdminTheme {
  try {
    const stored = localStorage.getItem(ADMIN_THEME_KEY);
    if (stored === 'dark' || stored === 'light') return stored;
  } catch {
    /* ignore */
  }
  return 'light';
}

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

type UploadEntryStatus = 'pending' | 'running' | 'imported' | 'duplicate' | 'failed';

interface UploadEntry {
  name: string;
  status: UploadEntryStatus;
  detail?: string;
}

interface UploadProgress {
  total: number;
  currentIndex: number;
  currentName: string;
  entries: UploadEntry[];
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
  const [docKind, setDocKind] = useState<'regulatory' | 'statistical' | 'internal'>('regulatory');
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [theme, setTheme] = useState<AdminTheme>(readAdminTheme);
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

  useEffect(() => {
    try {
      localStorage.setItem(ADMIN_THEME_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

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
    const initial: UploadProgress = {
      total: selected.length,
      currentIndex: 0,
      currentName: selected[0]?.name || '',
      entries: selected.map((file) => ({ name: file.name, status: 'pending' })),
    };
    flushSync(() => {
      setBusy(true);
      setError(null);
      setMessage(null);
      setUploadProgress(initial);
    });
    let ok = 0;
    let duplicates = 0;
    const failures: string[] = [];
    try {
      for (let index = 0; index < selected.length; index += 1) {
        const file = selected[index];
        flushSync(() => {
          setUploadProgress((prev) => {
            if (!prev) return prev;
            const entries = prev.entries.map((entry, i) => (
              i === index ? { ...entry, status: 'running' as const, detail: undefined } : entry
            ));
            return {
              ...prev,
              currentIndex: index,
              currentName: file.name,
              entries,
            };
          });
          setMessage(t(locale, 'admin.uploadBatchProgress', { current: index + 1, total: selected.length }));
        });
        const form = new FormData();
        form.append('file', file);
        form.append('doc_kind', docKind);
        try {
          const report = await uploadDocument(form) as { duplicate?: boolean };
          const status: UploadEntryStatus = report.duplicate ? 'duplicate' : 'imported';
          if (report.duplicate) duplicates += 1;
          else ok += 1;
          flushSync(() => {
            setUploadProgress((prev) => {
              if (!prev) return prev;
              const entries = prev.entries.map((entry, i) => (
                i === index
                  ? {
                      ...entry,
                      status,
                      detail: report.duplicate ? t(locale, 'admin.duplicate') : undefined,
                    }
                  : entry
              ));
              return { ...prev, entries };
            });
          });
        } catch (err) {
          const detail = err instanceof Error && err.message ? err.message : t(locale, 'admin.uploadFailed');
          failures.push(`${file.name}: ${detail}`);
          flushSync(() => {
            setUploadProgress((prev) => {
              if (!prev) return prev;
              const entries = prev.entries.map((entry, i) => (
                i === index ? { ...entry, status: 'failed' as const, detail } : entry
              ));
              return { ...prev, entries };
            });
          });
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
      setUploadProgress(null);
    }
  }

  const currentPage = navigation.find((item) => item.id === tab)?.label || t(locale, 'admin.overview');
  return <div className="admin-shell" data-theme={theme} lang={locale} dir={languageDirection(locale)}>
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
          <button
            type="button"
            className="admin-refresh admin-theme-toggle"
            aria-label={theme === 'dark' ? t(locale, 'admin.themeLight') : t(locale, 'admin.themeDark')}
            title={theme === 'dark' ? t(locale, 'admin.themeLight') : t(locale, 'admin.themeDark')}
            aria-pressed={theme === 'dark'}
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
          >
            {theme === 'dark' ? <Moon aria-hidden="true" size={17} /> : <Sun aria-hidden="true" size={17} />}
            <span>{theme === 'dark' ? t(locale, 'admin.themeDark') : t(locale, 'admin.themeLight')}</span>
          </button>
          <LanguageSwitcher locale={locale} onChange={onLocaleChange} className="admin-language-switcher" />
          <ProfileMenu
            user={user}
            locale={locale}
            onUserChange={onUserChange}
            onLogout={() => void handleLogout()}
            variant="header"
            theme={theme}
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
            docKind={docKind}
            onDocKindChange={setDocKind}
            uploadProgress={uploadProgress}
            onUpload={(event) => void handleUpload(event)}
            onFilesChange={(next) => {
              setFiles(next);
              setError(null);
              if (!next.length) setFileKey((key) => key + 1);
            }}
            onInvalidFiles={() => setError(t(locale, 'admin.fileInvalid'))}
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


function uploadEntryLabel(locale: UiLocale, status: UploadEntryStatus) {
  if (status === 'imported') return t(locale, 'admin.uploadEntryImported');
  if (status === 'duplicate') return t(locale, 'admin.uploadEntryDuplicate');
  if (status === 'failed') return t(locale, 'admin.uploadEntryFailed');
  if (status === 'running') return t(locale, 'admin.uploadEntryRunning');
  return t(locale, 'admin.uploadEntryPending');
}

function DocumentsPage({
  documents, loading, busy, locale, files, fileKey, docKind, onDocKindChange, uploadProgress, onUpload, onFilesChange, onInvalidFiles,
}: {
  documents: unknown[];
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  files: File[];
  fileKey: number;
  docKind: 'regulatory' | 'statistical' | 'internal';
  onDocKindChange: (value: 'regulatory' | 'statistical' | 'internal') => void;
  uploadProgress: UploadProgress | null;
  onUpload: (event: FormEvent<HTMLFormElement>) => void;
  onFilesChange: (files: File[]) => void;
  onInvalidFiles: () => void;
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
        <DocumentsList documents={documents} loading={loading} locale={locale} docKind={docKind} />
      </section>
    </>
  );
}


function DocumentsList({
  documents, loading, locale, docKind,
}: {
  documents: unknown[];
  loading: boolean;
  locale: UiLocale;
  docKind: 'regulatory' | 'statistical' | 'internal';
}) {
  const rows = (Array.isArray(documents) ? documents : []).filter((doc) => {
    const item = doc as { doc_kind?: string; filename?: string };
    const kind = item.doc_kind === 'statistical' || item.doc_kind === 'internal'
      ? item.doc_kind
      : 'regulatory';
    return kind === docKind;
  });
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
      ) : rows.length === 0 ? (
        <p className="admin-help">{t(locale, 'admin.indexedEmptyKind')}</p>
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
function OverviewSkeleton({ label }: { label: string }) {
  return (
    <section className="admin-metrics-strip" aria-label={label}>
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
    </section>
  );
}
function TableSkeleton({ label }: { label: string }) { return <div className="admin-skeleton table" aria-label={label} />; }
function DocumentSkeleton({ label }: { label: string }) { return <div className="admin-skeleton document" aria-label={label} />; }
function ConfigurationSkeleton({ label }: { label: string }) { return <section className="admin-configuration-layout" aria-label={label}><div className="admin-skeleton form" /><div className="admin-skeleton form" /><div className="admin-skeleton form" /></section>; }
