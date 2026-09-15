import { useState } from 'react';
import { login, register, type AuthUser } from '../api/auth';
import { AuthShell, CredentialsForm } from './AuthShell';
import { t, type UiLocale } from '../uiLocale';

interface RegisterPageProps {
  onRegistered: (user: AuthUser) => void;
  onGoLogin: () => void;
  locale: UiLocale;
  onLocaleChange: (locale: UiLocale) => void;
}

export function RegisterPage({ onRegistered, onGoLogin, locale, onLocaleChange }: RegisterPageProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(email: string, password: string) {
    setBusy(true);
    setError(null);
    try {
      await register(email, password);
      const session = await login(email, password);
      onRegistered(session.user);
    } catch {
      setError(t(locale, 'auth.registerFailed'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title={t(locale, 'auth.create')}
      subtitle={t(locale, 'auth.registerSubtitle')}
      locale={locale}
      onLocaleChange={onLocaleChange}
      footer={
        <>
          {t(locale, 'auth.hasAccount')}
          <a
            href="#login"
            onClick={(event) => {
              event.preventDefault();
              onGoLogin();
            }}
          >
            {t(locale, 'auth.signIn')}
          </a>
        </>
      }
    >
      <CredentialsForm submitLabel={t(locale, 'auth.signUp')} busy={busy} error={error} onSubmit={handleSubmit} locale={locale} mode="register" />
    </AuthShell>
  );
}
