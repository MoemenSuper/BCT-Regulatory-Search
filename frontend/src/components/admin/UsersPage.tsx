// Account approval, roles and token limits — one row per user.
import { useState } from 'react';
import { Check, Pencil, RotateCcw, ShieldPlus, Trash2, X } from 'lucide-react';
import { type AuthUser } from '../../api/auth';
import { displayLabel, AvatarMark } from '../ProfileMenu';
import { t, type UiLocale } from '../../uiLocale';
import { BlockSkeleton } from './Skeletons';

function formatTokens(value: number | undefined) {
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value || 0);
}

function formatUsd(value: number | undefined) {
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 4 }).format(value || 0);
}

export function UsersPage({ users, currentUser, busy, loading, locale, onApprove, onPromote, onReject, onDelete, onTokenLimit, onResetTokens }: {
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
  // Only one row's limit is edited at a time.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftLimit, setDraftLimit] = useState('');
  // Pending requests first: they are what the admin came here for.
  const rows = [...users].sort((a, b) => Number(b.status === 'pending') - Number(a.status === 'pending'));

  async function saveLimit(id: string) {
    const parsed = Number(draftLimit);
    if (!Number.isFinite(parsed) || parsed < 0) return;
    await onTokenLimit(id, Math.floor(parsed));
    setEditingId(null);
  }

  return (
    <section className="admin-panel">
      <div className="admin-panel-head">
        <h2>{t(locale, 'admin.users')}<span className="admin-count">{users.length}</span></h2>
      </div>
      <p className="admin-help">{t(locale, 'admin.promoteHelp')}</p>
      {loading ? (
        <BlockSkeleton label={t(locale, 'admin.loading')} />
      ) : (
        <div className="admin-table-wrap">
          <table className="admin-table">
            <thead>
              <tr>
                <th>{t(locale, 'admin.user')}</th>
                <th>{t(locale, 'admin.role')}</th>
                <th>{t(locale, 'admin.accountStatus')}</th>
                <th>{t(locale, 'admin.tokensUsed')} / {t(locale, 'admin.tokenLimit')}</th>
                <th>{t(locale, 'admin.estimatedSpend')}</th>
                <th><span className="sr-only">{t(locale, 'admin.actions')}</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((entry) => {
                const limit = entry.token_limit ?? 0;
                const used = entry.tokens_used ?? 0;
                const share = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
                const breakdown = [
                  `${t(locale, 'admin.tokensGroq')}: ${formatTokens(entry.tokens_llm)}`,
                  `${t(locale, 'admin.tokensEmbed')}: ${formatTokens(entry.tokens_embed)}`,
                  `${t(locale, 'admin.tokensRerank')}: ${formatTokens(entry.tokens_rerank)}`,
                ].join('\n');
                const isPending = entry.status === 'pending';
                return (
                  <tr key={entry.id}>
                    <td>
                      <div className="admin-user-cell">
                        <AvatarMark user={entry} size={32} />
                        <div>
                          <strong>{displayLabel(entry)}</strong>
                          <span>{entry.email}</span>
                        </div>
                      </div>
                    </td>
                    <td className="admin-role">
                      {entry.role === 'admin' ? t(locale, 'admin.administrator') : t(locale, 'admin.user')}
                    </td>
                    <td><StatusBadge status={entry.status} locale={locale} /></td>
                    <td>
                      {isPending ? <span className="admin-muted">—</span> : editingId === entry.id ? (
                        <form className="admin-token-edit" onSubmit={(event) => { event.preventDefault(); void saveLimit(entry.id); }}>
                          <input
                            type="number"
                            min={0}
                            step={1000}
                            value={draftLimit}
                            disabled={busy}
                            autoFocus
                            aria-label={t(locale, 'admin.tokenLimit')}
                            onChange={(event) => setDraftLimit(event.target.value)}
                            onKeyDown={(event) => { if (event.key === 'Escape') setEditingId(null); }}
                          />
                          <button type="submit" className="admin-icon-btn" disabled={busy} aria-label={t(locale, 'admin.saveTokenLimit')} title={t(locale, 'admin.saveTokenLimit')}>
                            <Check aria-hidden="true" size={16} />
                          </button>
                          <button type="button" className="admin-icon-btn" aria-label={t(locale, 'admin.cancel')} title={t(locale, 'admin.cancel')} onClick={() => setEditingId(null)}>
                            <X aria-hidden="true" size={16} />
                          </button>
                        </form>
                      ) : (
                        <div className="admin-token-cell" title={breakdown}>
                          <div className="admin-token-line num">
                            <span>{formatTokens(used)} / {limit > 0 ? formatTokens(limit) : t(locale, 'admin.unlimited')}</span>
                            <button
                              type="button"
                              className="admin-icon-btn"
                              disabled={busy}
                              aria-label={t(locale, 'admin.editLimit')}
                              title={t(locale, 'admin.editLimit')}
                              onClick={() => { setDraftLimit(String(limit)); setEditingId(entry.id); }}
                            >
                              <Pencil aria-hidden="true" size={14} />
                            </button>
                          </div>
                          {limit > 0 ? (
                            <div className={`admin-meter${share >= 90 ? ' is-high' : ''}`} aria-hidden="true">
                              <span style={{ width: `${share}%` }} />
                            </div>
                          ) : null}
                        </div>
                      )}
                    </td>
                    <td className="num">{isPending ? <span className="admin-muted">—</span> : formatUsd(entry.estimated_spend_usd)}</td>
                    <td>
                      <div className="admin-actions">
                        {isPending ? (
                          <>
                            <button type="button" className="admin-btn small primary" disabled={busy} onClick={() => void onApprove(entry.id)}>
                              {t(locale, 'admin.approve')}
                            </button>
                            <button type="button" className="admin-btn small" disabled={busy} onClick={() => void onReject(entry.id)}>
                              {t(locale, 'admin.reject')}
                            </button>
                          </>
                        ) : null}
                        {entry.status === 'approved' && entry.role !== 'admin' ? (
                          <button type="button" className="admin-btn small" disabled={busy} onClick={() => void onPromote(entry.id, entry.email)}>
                            <ShieldPlus aria-hidden="true" size={14} />
                            {t(locale, 'admin.promote')}
                          </button>
                        ) : null}
                        {!isPending ? (
                          <button type="button" className="admin-btn small" disabled={busy || used === 0} onClick={() => void onResetTokens(entry.id, entry.email)}>
                            <RotateCcw aria-hidden="true" size={14} />
                            {t(locale, 'admin.resetTokens')}
                          </button>
                        ) : null}
                        {entry.id !== currentUser.id && entry.role !== 'admin' ? (
                          <button type="button" className="admin-btn small danger" disabled={busy} onClick={() => void onDelete(entry.id, entry.email)}>
                            <Trash2 aria-hidden="true" size={14} />
                            {t(locale, 'admin.delete')}
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function StatusBadge({ status, locale }: { status: string; locale: UiLocale }) {
  const values: Record<UiLocale, Record<string, string>> = {
    fr: { approved: 'Approuvé', pending: 'En attente', rejected: 'Refusé' },
    ar: { approved: 'مقبول', pending: 'قيد الانتظار', rejected: 'مرفوض' },
    en: { approved: 'Approved', pending: 'Pending', rejected: 'Rejected' },
  };
  const normalized = status.toLowerCase();
  return (
    <span className={`admin-status ${normalized}`}>
      <i aria-hidden="true" />
      {values[locale][normalized] || status}
    </span>
  );
}
