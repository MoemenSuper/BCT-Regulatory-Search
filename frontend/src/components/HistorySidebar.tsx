import { useEffect, useState } from 'react';
import {
  FileText,
  MoreVertical,
  PanelLeft,
  PanelLeftClose,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import type { HistoryGroup } from '../types/ui';

interface HistorySidebarProps {
  groups: HistoryGroup[];
  selectedId: string | null;
  loading?: boolean;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onNewSearch: () => void;
  onRefresh: () => void;
  collapsed?: boolean;
  onCollapse?: () => void;
  onExpand?: () => void;
}

export function HistorySidebar({
  groups,
  selectedId,
  loading = false,
  onSelect,
  onRename,
  onDelete,
  onNewSearch,
  onRefresh,
  collapsed = false,
  onCollapse,
  onExpand,
}: HistorySidebarProps) {
  const total = groups.reduce((count, group) => count + group.items.length, 0);
  const [menuId, setMenuId] = useState<string | null>(null);

  useEffect(() => {
    if (!menuId) return;
    const close = () => setMenuId(null);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };
    document.addEventListener('click', close);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', close);
      document.removeEventListener('keydown', onKey);
    };
  }, [menuId]);

  if (collapsed) {
    return (
      <aside className="history-sidebar is-collapsed">
        <button
          type="button"
          className="panel-rail-btn"
          onClick={onExpand}
          aria-label="Afficher l’historique"
          title="Historique"
        >
          <PanelLeft size={20} strokeWidth={1.75} />
        </button>
      </aside>
    );
  }

  return (
    <aside id="history-sidebar" className="history-sidebar" tabIndex={-1}>
      <button type="button" className="btn-new-search pressable" onClick={onNewSearch}>
        <Plus size={18} strokeWidth={2.25} />
        <span>Nouvelle recherche</span>
      </button>

      <div className="history-heading-row">
        <h2 className="history-heading">HISTORIQUE DES RECHERCHES</h2>
        {onCollapse ? (
          <button
            type="button"
            className="icon-ghost panel-collapse-btn"
            aria-label="Masquer l’historique"
            title="Masquer l’historique"
            onClick={onCollapse}
          >
            <PanelLeftClose size={20} strokeWidth={1.75} />
          </button>
        ) : null}
      </div>

      <div className="history-scroll">
        {total === 0 && !loading ? (
          <div className="history-empty">
            <FileText size={22} strokeWidth={1.5} />
            <span>Aucune recherche enregistrée.</span>
          </div>
        ) : null}

        {groups.map((group) => (
          <section key={group.label} className="history-group">
            <h3 className="history-date-label">{group.label}</h3>
            <ul className="history-list">
              {group.items.map((item) => {
                const selected = item.id === selectedId;
                return (
                  <li key={item.id} className="history-item">
                    <button
                      type="button"
                      className={`history-card${selected ? ' selected' : ''}`}
                      onClick={() => onSelect(item.id)}
                    >
                      <FileText size={15} strokeWidth={1.75} className="history-doc-icon" />
                      <span className="history-card-text">
                        <span className="history-card-title">{item.title}</span>
                        <span className="history-card-time">
                          {item.time}{item.turnCount > 1 ? ` · ${item.turnCount} échanges` : ''}
                        </span>
                      </span>
                    </button>
                    <button
                      type="button"
                      className="history-more"
                      aria-label="Options de la recherche"
                      aria-haspopup="menu"
                      aria-expanded={menuId === item.id}
                      onClick={(event) => {
                        event.stopPropagation();
                        setMenuId(menuId === item.id ? null : item.id);
                      }}
                    >
                      <MoreVertical size={14} strokeWidth={1.75} />
                    </button>
                    {menuId === item.id ? (
                      <div className="history-menu" role="menu">
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            const title = window.prompt('Nouveau titre', item.title)?.trim();
                            if (title && title !== item.title) onRename(item.id, title);
                          }}
                        >
                          <Pencil size={14} strokeWidth={1.75} />
                          Renommer
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          className="danger"
                          onClick={() => {
                            if (window.confirm('Supprimer définitivement cette recherche ?')) onDelete(item.id);
                          }}
                        >
                          <Trash2 size={14} strokeWidth={1.75} />
                          Supprimer
                        </button>
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>

      <button type="button" className="btn-see-all pressable" onClick={onRefresh}>
        <RefreshCw className={loading ? 'spin' : undefined} size={16} strokeWidth={1.75} />
        <span>{loading ? 'Actualisation…' : 'Actualiser l’historique'}</span>
      </button>
    </aside>
  );
}
