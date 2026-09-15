import type { AuthUser } from '../api/auth';
import { AuthShell } from './AuthShell';
import { t, type UiLocale } from '../uiLocale';

interface AccountStatusPageProps {
  user: AuthUser;
  onLogout: () => void;
  locale: UiLocale;
  onLocaleChange: (locale: UiLocale) => void;
}

export function AccountStatusPage({ user, onLogout, locale, onLocaleChange }: AccountStatusPageProps) {
  const pending = user.status === 'pending';
  return (
    <AuthShell
      title={pending ? t(locale, 'auth.pendingTitle') : t(locale, 'auth.deniedTitle')}
      subtitle={
        pending
          ? t(locale, 'auth.pendingSubtitle')
          : t(locale, 'auth.deniedSubtitle')
      }
      locale={locale}
      onLocaleChange={onLocaleChange}
      footer={
        <a
          href="#logout"
          onClick={(event) => {
            event.preventDefault();
            onLogout();
          }}
        >
          {t(locale, 'auth.signOut')}
        </a>
      }
    >
      <div className="status-panel">
        <p>
          {t(locale, 'auth.signedInAs')} <strong>{user.email}</strong>.
          {pending
            ? ` ${t(locale, 'auth.pendingDetail')}`
            : ` ${t(locale, 'auth.deniedDetail')}`}
        </p>
        <button className="btn primary" type="button" onClick={onLogout}>
          {t(locale, 'auth.signOut')}
        </button>
      </div>
    </AuthShell>
  );
}
