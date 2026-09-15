import { useState, type FormEvent, type ReactNode } from 'react';
import { LanguageSwitcher } from './LanguageSwitcher';
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
  return (
    <div className="login-page" lang={locale} dir={languageDirection(locale)}>
      <main className="shell" aria-label={t(locale, 'auth.welcome')}>
        <section className="hero" aria-label="Banque Centrale de Tunisie">
          <div className="eyebrow">{t(locale, 'auth.eyebrow')}</div>
          <div className="hero-copy">
            <h1>{t(locale, 'auth.heroTitle').split('\n').map((line, index) => <span key={line}>{index > 0 ? <br /> : null}{line}</span>)}</h1>
            <p>{t(locale, 'auth.heroText')}</p>
          </div>
        </section>

        <section className="form-side" aria-label={title}>
          <LanguageSwitcher locale={locale} onChange={onLocaleChange} className="auth-language-switcher" />
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
  const confirmationError = mode === 'register' && password !== confirmation ? t(locale, 'auth.passwordMismatch') : null;
  const valid = !emailError && !passwordError && !confirmationError;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched({ email: true, password: true, confirmation: true });
    if (!valid) return;
    await onSubmit(email, password);
  }

  return (
    <form onSubmit={handleSubmit}>
      {error ? (
        <div className="form-error" role="alert">
          {error}
        </div>
      ) : null}
      {success ? <div className="form-success">{success}</div> : null}
      <div className="field">
        <label htmlFor="email">{t(locale, 'auth.email')}</label>
        {touched.email && emailError ? <p className="field-error" id="email-error" role="alert">{emailError}</p> : null}
        <input
          id="email"
          name="email"
          type="email"
          autoComplete="email"
          className="credentials-email"
          value={email}
          onChange={(event) => { setEmail(event.target.value); setTouched((current) => ({ ...current, email: true })); }}
          onBlur={() => setTouched((current) => ({ ...current, email: true }))}
          placeholder={t(locale, 'auth.emailPlaceholder')}
          aria-invalid={Boolean(touched.email && emailError)}
          aria-describedby={touched.email && emailError ? 'email-error' : undefined}
        />
      </div>
      <div className="field">
        <label htmlFor="password">{t(locale, 'auth.password')}</label>
        {touched.password && passwordError ? <p className="field-error" id="password-error" role="alert">{passwordError}</p> : null}
        <div className="input-wrap">
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            className="credentials-password"
            value={password}
            onChange={(event) => { setPassword(event.target.value); setTouched((current) => ({ ...current, password: true })); }}
            onBlur={() => setTouched((current) => ({ ...current, password: true }))}
            placeholder={t(locale, 'auth.passwordPlaceholder')}
            aria-invalid={Boolean(touched.password && passwordError)}
            aria-describedby={touched.password && passwordError ? 'password-error' : undefined}
          />
        </div>
      </div>
      {mode === 'register' ? <div className="field">
        <label htmlFor="confirmation">{t(locale, 'auth.confirmPassword')}</label>
        {touched.confirmation && confirmationError ? <p className="field-error" id="confirmation-error" role="alert">{confirmationError}</p> : null}
        <div className="input-wrap"><input id="confirmation" type="password" autoComplete="new-password" className="credentials-password" value={confirmation} onChange={(event) => { setConfirmation(event.target.value); setTouched((current) => ({ ...current, confirmation: true })); }} onBlur={() => setTouched((current) => ({ ...current, confirmation: true }))} placeholder={t(locale, 'auth.passwordPlaceholder')} aria-invalid={Boolean(touched.confirmation && confirmationError)} aria-describedby={touched.confirmation && confirmationError ? 'confirmation-error' : undefined} /></div>
      </div> : null}
      <button className="btn primary" type="submit" disabled={busy || !valid}>
        {busy ? t(locale, 'auth.wait') : submitLabel}
      </button>
    </form>
  );
}
