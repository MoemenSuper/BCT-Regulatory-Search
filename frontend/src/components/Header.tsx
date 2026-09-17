import { Moon, Sun } from 'lucide-react';
import type { AuthUser } from '../api/auth';
import { useTheme } from '../hooks/useTheme';
import { ProfileMenu } from './ProfileMenu';
import type { UiLocale } from '../uiLocale';

interface HeaderProps {
  user?: AuthUser;
  locale?: UiLocale;
  onUserChange?: (user: AuthUser) => void;
  onLogout?: () => void;
}

export function Header({ user, locale = 'fr', onUserChange, onLogout }: HeaderProps) {
  const { theme, toggleTheme } = useTheme();
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
        <h1 className="app-title">Espace Recherche Réglementaire</h1>
      </div>
      <div className="header-actions">
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-pressed={dark}
          aria-label={dark ? 'Passer en mode clair' : 'Passer en mode sombre'}
          title={dark ? 'Mode clair' : 'Mode sombre'}
        >
          {dark ? <Sun size={18} strokeWidth={1.75} /> : <Moon size={18} strokeWidth={1.75} />}
          <span>{dark ? 'Clair' : 'Sombre'}</span>
        </button>
        {user && onUserChange && onLogout ? (
          <ProfileMenu
            user={user}
            locale={locale}
            onUserChange={onUserChange}
            onLogout={onLogout}
            variant="header"
          />
        ) : onLogout ? (
          <button type="button" className="theme-toggle" onClick={onLogout} aria-label="Sign out">
            <span>Sortir</span>
          </button>
        ) : null}
      </div>
    </header>
  );
}
