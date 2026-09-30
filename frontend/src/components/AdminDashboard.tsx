import { useEffect, useState, type FormEvent } from 'react';
import { flushSync } from 'react-dom';
import { CheckCircle2, CircleAlert, FileText, Gauge, Moon, Settings2, Sun, UsersRound, XCircle } from 'lucide-react';
import { approveUser, deleteDocuments, deleteUser, downloadAnswerRefusalsExport, getConfig, getEnrichmentState, getOverview, listAnswerRefusals, listDocuments, listUsers, promoteUser, rejectUser, resetUserTokens, retryEnrichment, setProfile, setSecrets, setUserTokenLimit, uploadDocument, type AdminConfig, type AdminOverview, type AnswerRefusal, type AnswerRefusalsPage, type EnrichmentWorkerState, type IndexedDocument } from '../api/admin';
import { logout, type AuthUser } from '../api/auth';
import { LanguageSwitcher } from './LanguageSwitcher';
import { ProfileMenu } from './ProfileMenu';
import { languageDirection, t, type UiLocale } from '../uiLocale';
import { isPdfFile, type AdminTab, type DocKind, type UploadEntryStatus, type UploadProgress } from './admin/shared';
import { OverviewPage } from './admin/OverviewPage';
import { RefusalsPage } from './admin/RefusalsPage';
import { UsersPage } from './admin/UsersPage';
import { DocumentsPage } from './admin/DocumentsPage';
import { ConfigurationPage } from './admin/ConfigurationPage';

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

export function AdminDashboard({ user, onUserChange, onLogout, locale, onLocaleChange }: AdminDashboardProps) {
  const [tab, setTab] = useState<AdminTab>('overview');
  const [overview, setOverview] = useState<AdminOverview | null>(null);
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [enrichment, setEnrichment] = useState<EnrichmentWorkerState | null>(null);
  const [refusals, setRefusals] = useState<AnswerRefusalsPage | null>(null);
  const [recentRefusals, setRecentRefusals] = useState<AnswerRefusal[]>([]);
  const [refusalBuckets, setRefusalBuckets] = useState<string[]>([]);
  const [config, setConfig] = useState<AdminConfig | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [deletingIds, setDeletingIds] = useState<Set<string>>(() => new Set());
  const [loading, setLoading] = useState(true);
  const [files, setFiles] = useState<File[]>([]);
  const [fileKey, setFileKey] = useState(0);
  const [docKind, setDocKind] = useState<DocKind>('regulatory');
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [theme, setTheme] = useState<AdminTheme>(readAdminTheme);
  const navigation = [
    { id: 'overview' as const, label: t(locale, 'admin.overview'), icon: Gauge },
    { id: 'users' as const, label: t(locale, 'admin.users'), icon: UsersRound },
    { id: 'documents' as const, label: t(locale, 'admin.documents'), icon: FileText },
    { id: 'refusals' as const, label: t(locale, 'admin.refusals'), icon: CircleAlert },
    { id: 'configuration' as const, label: t(locale, 'admin.configuration'), icon: Settings2 },
  ];

  // The overview also lists pending access requests and the latest refusals.
  async function loadOverview() {
    const [counts, accounts, latest] = await Promise.all([getOverview(), listUsers(), listAnswerRefusals(5)]);
    return { counts, accounts, latest: latest.items };
  }
  function showOverview(data: Awaited<ReturnType<typeof loadOverview>>) {
    setOverview(data.counts);
    setUsers(data.accounts);
    setRecentRefusals(data.latest);
  }

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setError(null); setLoading(true);
      try {
        if (tab === 'overview') { const data = await loadOverview(); if (!cancelled) showOverview(data); }
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

  const enrichingCount = documents.filter((doc) => doc.status === 'enriching').length;
  useEffect(() => {
    if (tab !== 'documents' || !enrichingCount) return;
    let cancelled = false;
    async function poll() {
      try {
        const [docs, state] = await Promise.all([listDocuments(), getEnrichmentState()]);
        if (!cancelled) { setDocuments(docs); setEnrichment(state); }
      } catch { /* next tick retries */ }
    }
    void poll();
    const timer = window.setInterval(() => void poll(), 10000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [tab, enrichingCount]);

  async function handleRetryEnrichment(documentId: string) {
    setBusy(true); setError(null); setMessage(null);
    try {
      await retryEnrichment(documentId);
      setDocuments(await listDocuments());
      setMessage(t(locale, 'admin.enrichRetryQueued'));
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : t(locale, 'admin.enrichRetryFailed'));
    } finally { setBusy(false); }
  }

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
      if (tab === 'overview') showOverview(await loadOverview());
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

  async function handleDeleteDocuments(documentIds: string[], label: string) {
    if (!documentIds.length) return;
    const single = documentIds.length === 1;
    const confirmText = single
      ? t(locale, 'admin.deletePdfConfirm', { name: label })
      : t(locale, 'admin.deletePdfsConfirm', { count: documentIds.length });
    if (!window.confirm(confirmText)) return;
    setBusy(true);
    setDeletingIds(new Set(documentIds));
    setError(null);
    setMessage(null);
    try {
      const { removed, failed } = await deleteDocuments(documentIds);
      const gone = new Set(removed);
      setDocuments((current) => current.filter((doc) => !gone.has(doc.document_id || '')));
      if (failed.length) {
        setError(single
          ? failed[0].error || t(locale, 'admin.pdfDeleteFailed')
          : t(locale, 'admin.pdfsDeletePartial', { ok: removed.length, failed: failed.length, error: failed[0].error }));
      } else {
        setMessage(single ? t(locale, 'admin.pdfDeleted') : t(locale, 'admin.pdfsDeleted', { count: removed.length }));
      }
    } catch (err) {
      const detail = err instanceof Error && err.message ? err.message : t(locale, 'admin.pdfDeleteFailed');
      setError(detail);
    } finally {
      setDeletingIds(new Set());
      setBusy(false);
    }
  }

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
    let enriching = 0;
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
          const report = await uploadDocument(form) as { duplicate?: boolean; status?: string };
          const status: UploadEntryStatus = report.duplicate ? 'duplicate' : 'imported';
          if (report.duplicate) duplicates += 1;
          else ok += 1;
          if (!report.duplicate && report.status === 'enriching') enriching += 1;
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
      } else if (enriching) {
        setMessage(t(locale, 'admin.uploadEnriching', { count: enriching }));
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
  const themeLabel = theme === 'dark' ? t(locale, 'admin.themeLight') : t(locale, 'admin.themeDark');
  return <div className="admin-shell" data-theme={theme} lang={locale} dir={languageDirection(locale)}>
    <a className="admin-skip-link" href="#admin-content">{t(locale, 'admin.skip')}</a>
    <aside className="admin-sidebar">
      <div className="admin-brand"><img src="/bct-logo-white.png" alt="Banque Centrale de Tunisie" /><span>{t(locale, 'admin.brand')}</span></div>
      <nav className="admin-nav">{navigation.map(({ id, label, icon: Icon }) => <button key={id} type="button" className={tab === id ? 'active' : ''} aria-current={tab === id ? 'page' : undefined} onClick={() => selectTab(id)}><Icon aria-hidden="true" size={18} strokeWidth={1.8} /><span>{label}</span></button>)}</nav>
    </aside>
    <div className="admin-workspace">
      <header className="admin-header">
        <h1>{currentPage}</h1>
        <div className="admin-header-actions">
          <button
            type="button"
            className="admin-icon-btn large"
            aria-label={themeLabel}
            title={themeLabel}
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
          >
            {theme === 'dark' ? <Sun aria-hidden="true" size={17} /> : <Moon aria-hidden="true" size={17} />}
          </button>
          <LanguageSwitcher locale={locale} onChange={onLocaleChange} />
          <ProfileMenu
            user={user}
            locale={locale}
            onUserChange={onUserChange}
            onLogout={() => void handleLogout()}
            theme={theme}
          />
        </div>
      </header>
      <main id="admin-content" className="admin-main">
        {error ? <div className="admin-banner error" role="alert"><XCircle aria-hidden="true" size={19} />{error}</div> : null}
        {message ? <div className="admin-banner ok" role="status"><CheckCircle2 aria-hidden="true" size={19} />{message}</div> : null}
        {tab === 'overview' ? <OverviewPage overview={overview} users={users} recentRefusals={recentRefusals} busy={busy} loading={loading} locale={locale} onNavigate={selectTab} onApprove={handleApprove} onReject={handleReject} /> : null}
        {tab === 'users' ? <UsersPage users={users} currentUser={user} busy={busy} loading={loading} locale={locale} onApprove={handleApprove} onPromote={handlePromote} onReject={handleReject} onDelete={handleDelete} onTokenLimit={handleTokenLimit} onResetTokens={handleResetTokens} /> : null}
        {tab === 'documents' ? (
          <DocumentsPage
            documents={Array.isArray(documents) ? documents : []}
            enrichment={enrichment}
            onRetryEnrichment={(id) => void handleRetryEnrichment(id)}
            loading={loading}
            busy={busy}
            deletingIds={deletingIds}
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
            onDeleteDocuments={(documentIds, label) => void handleDeleteDocuments(documentIds, label)}
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
        {tab === 'configuration' ? <ConfigurationPage config={config} loading={loading} busy={busy} locale={locale} onProfile={handleProfile} onSecrets={handleSecrets} /> : null}
      </main>
    </div>
  </div>;
}
