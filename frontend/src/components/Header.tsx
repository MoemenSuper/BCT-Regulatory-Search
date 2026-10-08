import { LogOut, Moon, Sun } from 'lucide-react';
import type { AuthUser } from '../api/auth';
import type { Theme } from '../hooks/useTheme';
import { LanguageSwitcher } from './LanguageSwitcher';
import { ProfileMenu } from './ProfileMenu';
import { t, type UiLocale } from '../uiLocale';

interface HeaderProps {
  user?: AuthUser;
  locale?: UiLocale;
  onLocaleChange?: (locale: UiLocale) => void;
  onUserChange?: (user: AuthUser) => void;
  onLogout?: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export function Header({
  user,
  locale = 'fr',
  onLocaleChange,
  onUserChange,
  onLogout,
  theme,
  onToggleTheme,
}: HeaderProps) {
  const dark = theme === 'dark';

  return (
    <header className="app-header">
      <div className="header-brand">
        <img
          src="/bct-logo-white.png"
          alt="Banque Centrale de Tunisie — Central Bank of Tunisia"
          className="bct-logo"
        />
        <span className="header-divider" aria-hidden="true" />
        <h1 className="app-title">{t(locale, 'chat.title')}</h1>
      </div>
      <div className="header-actions">
        {onLocaleChange ? <LanguageSwitcher locale={locale} onChange={onLocaleChange} /> : null}
        <button
          type="button"
          className="header-icon-btn"
          onClick={onToggleTheme}
          aria-label={t(locale, dark ? 'chat.themeLight' : 'chat.themeDark')}
          title={t(locale, dark ? 'chat.themeLight' : 'chat.themeDark')}
        >
          {dark ? <Sun size={18} strokeWidth={1.75} aria-hidden="true" /> : <Moon size={18} strokeWidth={1.75} aria-hidden="true" />}
        </button>
        {user && onUserChange && onLogout ? (
          <ProfileMenu
            user={user}
            locale={locale}
            onUserChange={onUserChange}
            onLogout={onLogout}
            theme={theme}
          />
        ) : onLogout ? (
          <button type="button" className="header-icon-btn" onClick={onLogout} aria-label={t(locale, 'auth.signOut')} title={t(locale, 'auth.signOut')}>
            <LogOut size={18} strokeWidth={1.75} aria-hidden="true" />
          </button>
        ) : null}
      </div>
    </header>
  );
}
