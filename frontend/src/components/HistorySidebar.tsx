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
import { t, type UiLocale } from '../uiLocale';

interface HistorySidebarProps {
  locale: UiLocale;
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
  locale,
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

  useEffect(() => {
    if (collapsed) setMenuId(null);
  }, [collapsed]);

  return (
    <aside
      id="history-sidebar"
      className={`history-sidebar${collapsed ? ' is-collapsed' : ''}`}
      tabIndex={-1}
    >
      <button
        type="button"
        className="panel-rail-btn"
        onClick={onExpand}
        aria-label={t(locale, 'chat.showHistory')}
        title={t(locale, 'chat.history')}
        tabIndex={collapsed ? 0 : -1}
        aria-hidden={!collapsed}
      >
        <PanelLeft size={20} strokeWidth={1.75} />
      </button>

      <div className="panel-expanded" aria-hidden={collapsed}>
        <button type="button" className="btn-new-search pressable" onClick={onNewSearch} tabIndex={collapsed ? -1 : 0}>
          <Plus size={18} strokeWidth={2.25} />
          <span>{t(locale, 'chat.newSearch')}</span>
        </button>

        <div className="history-heading-row">
          <h2 className="history-heading">{t(locale, 'chat.historyTitle')}</h2>
          {onCollapse ? (
            <button
              type="button"
              className="icon-ghost panel-collapse-btn"
              aria-label={t(locale, 'chat.hideHistory')}
              title={t(locale, 'chat.hideHistory')}
              onClick={onCollapse}
              tabIndex={collapsed ? -1 : 0}
            >
              <PanelLeftClose size={20} strokeWidth={1.75} />
            </button>
          ) : null}
        </div>

        <div className="history-scroll">
          {total === 0 && !loading ? (
            <div className="history-empty">
              <FileText size={22} strokeWidth={1.5} />
              <span>{t(locale, 'chat.noHistory')}</span>
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
                        tabIndex={collapsed ? -1 : 0}
                      >
                        <FileText size={15} strokeWidth={1.75} className="history-doc-icon" />
                        <span className="history-card-text">
                          <span className="history-card-title" dir="auto">{item.title}</span>
                          <span className="history-card-time">
                            {item.time}{item.turnCount > 1 ? ` · ${t(locale, 'chat.turns', { count: item.turnCount })}` : ''}
                          </span>
                        </span>
                      </button>
                      <button
                        type="button"
                        className="history-more"
                        aria-label={t(locale, 'chat.searchOptions')}
                        aria-haspopup="menu"
                        aria-expanded={menuId === item.id}
                        tabIndex={collapsed ? -1 : 0}
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
                              const title = window.prompt(t(locale, 'chat.newTitle'), item.title)?.trim();
                              if (title && title !== item.title) onRename(item.id, title);
                            }}
                          >
                            <Pencil size={14} strokeWidth={1.75} />
                            {t(locale, 'chat.rename')}
                          </button>
                          <button
                            type="button"
                            role="menuitem"
                            className="danger"
                            onClick={() => {
                              if (window.confirm(t(locale, 'chat.deleteConfirm'))) onDelete(item.id);
                            }}
                          >
                            <Trash2 size={14} strokeWidth={1.75} />
                            {t(locale, 'chat.delete')}
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

        <button type="button" className="btn-see-all pressable" onClick={onRefresh} tabIndex={collapsed ? -1 : 0}>
          <RefreshCw className={loading ? 'spin' : undefined} size={16} strokeWidth={1.75} />
          <span>{t(locale, loading ? 'chat.refreshing' : 'chat.refresh')}</span>
        </button>
      </div>
    </aside>
  );
}
