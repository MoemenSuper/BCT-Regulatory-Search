import { FileText, List, MoreVertical, Plus, SlidersHorizontal } from 'lucide-react';
import type { HistoryGroup } from '../types/ui';

interface HistorySidebarProps {
  groups: HistoryGroup[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export function HistorySidebar({ groups, selectedId, onSelect }: HistorySidebarProps) {
  return (
    <aside className="history-sidebar">
      <button type="button" className="btn-new-search">
        <Plus size={18} strokeWidth={2.25} />
        <span>Nouvelle recherche</span>
      </button>

      <div className="history-heading-row">
        <h2 className="history-heading">HISTORIQUE DES RECHERCHES</h2>
        <button type="button" className="icon-ghost" aria-label="Filtres historique">
          <SlidersHorizontal size={14} strokeWidth={1.75} />
        </button>
      </div>

      <div className="history-scroll">
        {groups.map((group) => (
          <section key={group.label} className="history-group">
            <h3 className="history-date-label">{group.label}</h3>
            <ul className="history-list">
              {group.items.map((item) => {
                const selected = item.id === selectedId;
                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`history-card${selected ? ' selected' : ''}`}
                      onClick={() => onSelect(item.id)}
                    >
                      <FileText size={15} strokeWidth={1.75} className="history-doc-icon" />
                      <span className="history-card-text">
                        <span className="history-card-title">{item.title}</span>
                        <span className="history-card-time">{item.time}</span>
                      </span>
                      <span
                        className="history-more"
                        aria-hidden="true"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <MoreVertical size={14} strokeWidth={1.75} />
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>

      <button type="button" className="btn-see-all">
        <List size={16} strokeWidth={1.75} />
        <span>Voir tout l&apos;historique</span>
      </button>
    </aside>
  );
}
