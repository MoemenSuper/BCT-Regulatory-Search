// Account approval, roles and token limits.
import { useEffect, useState } from 'react';
import { ShieldPlus, Trash2, UserCheck } from 'lucide-react';
import { type AuthUser } from '../../api/auth';
import { displayLabel, AvatarMark } from '../ProfileMenu';
import { t, type UiLocale } from '../../uiLocale';
import { TableSkeleton } from './Skeletons';

function formatTokens(value: number | undefined) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value || 0);
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
