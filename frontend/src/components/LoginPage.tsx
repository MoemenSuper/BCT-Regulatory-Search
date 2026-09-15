import { useState } from 'react';
import { login, type AuthUser } from '../api/auth';
import { AuthShell, CredentialsForm } from './AuthShell';
import { t, type UiLocale } from '../uiLocale';

interface LoginPageProps {
  onAuthenticated: (user: AuthUser) => void;
  onGoRegister: () => void;
  locale: UiLocale;
  onLocaleChange: (locale: UiLocale) => void;
}

export function LoginPage({ onAuthenticated, onGoRegister, locale, onLocaleChange }: LoginPageProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(email: string, password: string) {
    setBusy(true);
    setError(null);
    try {
      const result = await login(email, password);
      onAuthenticated(result.user);
    } catch {
      setError(t(locale, 'auth.loginFailed'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title={t(locale, 'auth.welcome')}
      subtitle={t(locale, 'auth.signInSubtitle')}
      locale={locale}
      onLocaleChange={onLocaleChange}
      footer={
        <>
          {t(locale, 'auth.noAccount')}
          <a
            href="#register"
            onClick={(event) => {
              event.preventDefault();
              onGoRegister();
            }}
          >
            {t(locale, 'auth.signUp')}
          </a>
        </>
      }
    >
      <CredentialsForm submitLabel={t(locale, 'auth.signIn')} busy={busy} error={error} onSubmit={handleSubmit} locale={locale} mode="login" />
    </AuthShell>
  );
}
