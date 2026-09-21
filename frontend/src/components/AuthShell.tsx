import { useState, type FormEvent, type ReactNode } from 'react';
import { Moon, Sun } from 'lucide-react';
import { LanguageSwitcher } from './LanguageSwitcher';
import { useAuthTheme } from '../hooks/useTheme';
import { languageDirection, t, type UiLocale } from '../uiLocale';

interface AuthShellProps {
  title: string;
  subtitle: string;
  children: ReactNode;
  footer: ReactNode;
  locale: UiLocale;
  onLocaleChange: (locale: UiLocale) => void;
}

export function AuthShell({ title, subtitle, children, footer, locale, onLocaleChange }: AuthShellProps) {
  const { theme, toggleTheme } = useAuthTheme();
  const dark = theme === 'dark';

  return (
    <div className="login-page" data-theme={theme} lang={locale} dir={languageDirection(locale)}>
      <main className="shell" aria-label={t(locale, 'auth.welcome')}>
        <section className="hero" aria-label="Banque Centrale de Tunisie">
          <div className="eyebrow">{t(locale, 'auth.eyebrow')}</div>
          <div className="hero-copy">
            <h1>
              {t(locale, 'auth.heroTitle')
                .split('\n')
                .map((line, index) => (
                  <span key={line}>
                    {index > 0 ? <br /> : null}
                    {line}
                  </span>
                ))}
            </h1>
            <p>{t(locale, 'auth.heroText')}</p>
          </div>
        </section>

        <section className="form-side" aria-label={title}>
          <div className="auth-toolbar">
            <button
              type="button"
              className="auth-theme-toggle"
              onClick={toggleTheme}
              aria-pressed={dark}
              aria-label={dark ? t(locale, 'admin.themeDark') : t(locale, 'admin.themeLight')}
              title={dark ? t(locale, 'admin.themeDark') : t(locale, 'admin.themeLight')}
            >
              {dark ? <Moon size={16} strokeWidth={1.8} aria-hidden="true" /> : <Sun size={16} strokeWidth={1.8} aria-hidden="true" />}
              <span>{dark ? t(locale, 'admin.themeDark') : t(locale, 'admin.themeLight')}</span>
            </button>
            <LanguageSwitcher locale={locale} onChange={onLocaleChange} className="auth-language-switcher" />
          </div>
          <div className="brand" aria-label="Banque Centrale de Tunisie">
            <img
              className="brand-mark"
              src="/bct-logo-white.png"
              alt="Banque Centrale de Tunisie — Central Bank of Tunisia"
            />
          </div>
          <div className="login-box">
            <h2>{title}</h2>
            <p className="subtitle">{subtitle}</p>
            {children}
          </div>
          <p className="signup">{footer}</p>
        </section>
      </main>
    </div>
  );
}

interface CredentialsFormProps {
  submitLabel: string;
  busy: boolean;
  error: string | null;
  success?: string | null;
  onSubmit: (email: string, password: string) => void | Promise<void>;
  locale: UiLocale;
  mode: 'login' | 'register';
}

export function CredentialsForm({
  submitLabel,
  busy,
  error,
  success,
  onSubmit,
  locale,
  mode,
}: CredentialsFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const emailError = email && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) ? null : t(locale, 'auth.emailRequired');
  const passwordError = password.length >= 8 ? null : t(locale, 'auth.passwordRequired');
  const confirmationError =
    mode === 'register' && password !== confirmation ? t(locale, 'auth.passwordMismatch') : null;
  const valid = !emailError && !passwordError && !confirmationError;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched({ email: true, password: true, confirmation: true });
    if (!valid) return;
    await onSubmit(email, password);
  }

  return (
    <form className="credentials-form" onSubmit={(event) => void handleSubmit(event)} noValidate>
      {error ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}
      {success ? (
        <p className="form-success" role="status">
          {success}
        </p>
      ) : null}
      <div className="field">
        <label htmlFor="auth-email">{t(locale, 'auth.email')}</label>
        {touched.email && emailError ? <p className="field-error">{emailError}</p> : null}
        <div className="input-wrap">
          <input
            id="auth-email"
            className="credentials-email"
            type="email"
            autoComplete="username"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            onBlur={() => setTouched((current) => ({ ...current, email: true }))}
            aria-invalid={touched.email && Boolean(emailError)}
            required
          />
        </div>
      </div>
      <div className="field">
        <label htmlFor="auth-password">{t(locale, 'auth.password')}</label>
        {touched.password && passwordError ? <p className="field-error">{passwordError}</p> : null}
        <div className="input-wrap">
          <input
            id="auth-password"
            className="credentials-password"
            type="password"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            onBlur={() => setTouched((current) => ({ ...current, password: true }))}
            aria-invalid={touched.password && Boolean(passwordError)}
            minLength={8}
            required
          />
        </div>
      </div>
      {mode === 'register' ? (
        <div className="field">
          <label htmlFor="auth-password-confirm">{t(locale, 'auth.confirmPassword')}</label>
          {touched.confirmation && confirmationError ? (
            <p className="field-error">{confirmationError}</p>
          ) : null}
          <div className="input-wrap">
            <input
              id="auth-password-confirm"
              className="credentials-password"
              type="password"
              autoComplete="new-password"
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
              onBlur={() => setTouched((current) => ({ ...current, confirmation: true }))}
              aria-invalid={touched.confirmation && Boolean(confirmationError)}
              minLength={8}
              required
            />
          </div>
        </div>
      ) : null}
      <button type="submit" className="btn primary" disabled={busy}>
        {busy ? '…' : submitLabel}
      </button>
    </form>
  );
}
