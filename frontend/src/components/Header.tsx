import {
  Bookmark,
  ChevronDown,
  Clock,
} from 'lucide-react';

export function Header() {
  return (
    <header className="app-header">
      <div className="header-left">
        <img
          src="/bct-logo-white.png"
          alt="Banque Centrale de Tunisie"
          className="bct-logo"
        />
        <span className="header-divider" aria-hidden="true" />
        <h1 className="app-title">Espace Recherche Réglementaire</h1>
      </div>

      <div className="header-right">
        <button type="button" className="header-action">
          <Bookmark size={16} strokeWidth={1.75} />
          <span>Signets</span>
        </button>
        <button type="button" className="header-action">
          <Clock size={16} strokeWidth={1.75} />
          <span>Historique</span>
        </button>

        <div className="lang-toggle" aria-label="Choix de langue">
          <button type="button" className="lang-btn active">
            FR
          </button>
          <button type="button" className="lang-btn">
            عربي
          </button>
        </div>

        <button type="button" className="user-menu">
          <span className="user-avatar" aria-hidden="true">
            FR
          </span>
          <span className="user-name">Rechercheur</span>
          <ChevronDown size={14} strokeWidth={2} />
        </button>
      </div>
    </header>
  );
}
