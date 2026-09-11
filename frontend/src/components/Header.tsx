import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

export function Header() {
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
    </header>
  );
}
