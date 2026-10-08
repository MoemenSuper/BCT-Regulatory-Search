import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { FileText } from 'lucide-react';
import {
  deleteConversation,
  getConversation,
  getConversations,
  postChat,
  renameConversation,
} from './api/chat';
import type { AuthUser } from './api/auth';
import { Composer } from './components/Composer';
import { EvidencePanel } from './components/EvidencePanel';
import { Header } from './components/Header';
import { HistorySidebar } from './components/HistorySidebar';
import { LegalFooter } from './components/LegalFooter';
import { PanelResizeHandle } from './components/PanelResizeHandle';
import { ResearchNote } from './components/ResearchNote';
import { SearchStatus } from './components/SearchStatus';
import {
  evidenceFromSource,
  historyGroupsFromConversations,
  noteFromConversationTurn,
} from './data/presentation';
import { useWorkspacePanels } from './hooks/useWorkspacePanels';
import { useChatTheme } from './hooks/useTheme';
import { languageDirection, t, type UiLocale } from './uiLocale';
import type {
  ChatSource,
  ConversationDetail,
  ConversationSummary,
  ConversationTurn,
  EvidenceTab,
} from './types/ui';
import './styles.css';

// Shown on an empty conversation, in the interface language: one click shows what the tool does
// well (compare, summarise, a figure). Each one was checked to get a cited answer from the corpus.
const EXAMPLE_KEYS = ['chat.example1', 'chat.example2', 'chat.example3', 'chat.example4'];

interface AppProps {
  user: AuthUser;
  locale: UiLocale;
  onLocaleChange: (locale: UiLocale) => void;
  onUserChange: (user: AuthUser) => void;
  onLogout?: () => void;
}

export default function App({ user, locale, onLocaleChange, onUserChange, onLogout }: AppProps) {
  const [activeTab, setActiveTab] = useState<EvidenceTab>('preuve');
  const [zoom, setZoom] = useState(100);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [conversationTitle, setConversationTitle] = useState('');
  const [selectedTurnIndex, setSelectedTurnIndex] = useState(0);
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(0);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [history, setHistory] = useState<ConversationSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const threadEndRef = useRef<HTMLDivElement | null>(null);
  const initialHistoryLoaded = useRef(false);

  const refreshHistory = useCallback(async (limit = 100) => {
    setHistoryLoading(true);
    try {
      const items = await getConversations(limit);
      setHistory(items);
      return items;
    } catch {
      return [] as ConversationSummary[];
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  function applyDetail(id: string, detail: ConversationDetail) {
    setConversationId(id);
    setSelectedHistoryId(id);
    setConversationTitle(detail.title || '');
    setTurns(detail.turns || []);
    setSelectedTurnIndex(Math.max(0, (detail.turns?.length || 1) - 1));
    setSelectedSourceIndex(0);
    setActiveTab('preuve');
  }

  function selectTurn(index: number) {
    setSelectedTurnIndex(index);
    if (index !== selectedTurnIndex) setSelectedSourceIndex(0);
    setActiveTab('preuve');
  }

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ block: 'end' });
  }, [turns, pendingQuestion, opening, error]);

  const openConversation = useCallback(async (id: string) => {
    setOpening(true);
    setError(null);
    try {
      applyDetail(id, await getConversation(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : t(locale, 'chat.openFailed'));
    } finally {
      setOpening(false);
    }
  }, [locale]);

  useEffect(() => {
    if (initialHistoryLoaded.current) return;
    initialHistoryLoaded.current = true;
    void refreshHistory().then((items) => {
      if (items[0]) void openConversation(items[0].conversation_id);
    });
  }, [openConversation, refreshHistory]);

  async function runSearch(question: string) {
    const cleaned = question.trim();
    if (!cleaned) return;
    setPendingQuestion(cleaned);
    setError(null);

    const followUp = Boolean(conversationId);
    try {
      const result = await postChat(cleaned, followUp ? conversationId : undefined);
      applyDetail(result.conversation_id, await getConversation(result.conversation_id));
      await refreshHistory();
    } catch (err) {
      const message = err instanceof Error ? err.message : t(locale, 'chat.serverDown');
      setError(t(locale, 'chat.retry', { message }));
    } finally {
      setPendingQuestion(null);
    }
  }

  function startNewSearch() {
    setConversationId(undefined);
    setSelectedHistoryId(null);
    setConversationTitle('');
    setTurns([]);
    setSelectedTurnIndex(0);
    setSelectedSourceIndex(0);
    setActiveTab('preuve');
    setError(null);
  }

  async function renameHistoryItem(id: string, title: string) {
    try {
      await renameConversation(id, title);
      if (id === conversationId) setConversationTitle(title);
      await refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : t(locale, 'chat.renameFailed'));
    }
  }

  async function deleteHistoryItem(id: string) {
    try {
      await deleteConversation(id);
      if (id === conversationId) startNewSearch();
      await refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : t(locale, 'chat.deleteFailed'));
    }
  }

  const busy = pendingQuestion !== null || opening;
  const historyGroups = useMemo(() => historyGroupsFromConversations(history, locale), [history, locale]);
  const activeTurn = turns[selectedTurnIndex];
  const sources: ChatSource[] = activeTurn?.sources || [];
  const passage = evidenceFromSource(sources[selectedSourceIndex], locale);
  const {
    workspaceRef,
    leftCollapsed,
    rightCollapsed,
    resizing,
    leftColumn,
    rightColumn,
    setLeftCollapsed,
    setRightCollapsed,
    startResize,
  } = useWorkspacePanels();
  const { theme, toggleTheme } = useChatTheme();

  return (
    <div className="app-shell" data-theme={theme} lang={locale} dir={languageDirection(locale)}>
      <Header
        user={user}
        locale={locale}
        onLocaleChange={onLocaleChange}
        onUserChange={onUserChange}
        onLogout={onLogout}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <div
        ref={workspaceRef}
        className={`workspace${resizing ? ' is-resizing' : ''}`}
        style={
          {
            '--left-w': `${leftColumn}px`,
            '--right-w': `${rightColumn}px`,
          } as CSSProperties
        }
      >
        <HistorySidebar
          locale={locale}
          groups={historyGroups}
          selectedId={selectedHistoryId}
          loading={historyLoading}
          onSelect={(id) => void openConversation(id)}
          onRename={(id, title) => void renameHistoryItem(id, title)}
          onDelete={(id) => void deleteHistoryItem(id)}
          onNewSearch={startNewSearch}
          onRefresh={() => void refreshHistory(500)}
          collapsed={leftCollapsed}
          onCollapse={() => setLeftCollapsed(true)}
          onExpand={() => setLeftCollapsed(false)}
        />
        <PanelResizeHandle
          side="left"
          locale={locale}
          disabled={leftCollapsed}
          active={resizing === 'left'}
          onResizeStart={(event) => startResize('left', event)}
          onCollapse={() => setLeftCollapsed(true)}
        />

        <main className="center-panel">
          <div className="center-scroll">
            {turns.length === 0 && !pendingQuestion ? (
              <article className="research-note">
                <div className="note-title-row">
                  <span className="note-icon-wrap" aria-hidden="true">
                    <FileText size={18} strokeWidth={1.75} />
                  </span>
                  <h2 className="note-title">{t(locale, 'chat.emptyTitle')}</h2>
                </div>
                <p className="note-body thread-empty-copy">{t(locale, 'chat.emptyBody')}</p>
                <div className="example-questions" aria-label={t(locale, 'chat.examples')}>
                  {EXAMPLE_KEYS.map((key) => t(locale, key)).map((question) => (
                    <button key={question} type="button" disabled={busy} onClick={() => void runSearch(question)}>
                      {/* Non-breaking hyphen on screen only, so "2026-01" never splits across lines. */}
                      {question.replace(/-/g, '‑')}
                    </button>
                  ))}
                </div>
              </article>
            ) : null}

            <div className="conversation-thread" role="log" aria-live="polite" aria-relevant="additions">
              {turns.map((turn, index) => {
                const note = noteFromConversationTurn(
                  turn,
                  conversationId || turn.turn_id,
                  conversationTitle,
                  locale,
                );
                const isActive = index === selectedTurnIndex;
                return (
                  <div
                    key={turn.turn_id}
                    className={`thread-turn${isActive ? ' active' : ''}`}
                    role="button"
                    tabIndex={0}
                    aria-pressed={isActive}
                    onClick={() => selectTurn(index)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        selectTurn(index);
                      }
                    }}
                  >
                    <ResearchNote
                      locale={locale}
                      note={note}
                      conversationId={conversationId || turn.turn_id}
                      turnId={turn.turn_id}
                      compact={index > 0}
                      selectedSourceIndex={isActive ? selectedSourceIndex : -1}
                      onSelectSource={(sourceIndex) => {
                        setSelectedTurnIndex(index);
                        setSelectedSourceIndex(sourceIndex);
                        setActiveTab('preuve');
                      }}
                    />
                  </div>
                );
              })}
            </div>

            {pendingQuestion ? <SearchStatus question={pendingQuestion} /> : null}

            {error ? (
              <div className="note-status note-status-error thread-status" role="alert">
                {error}
              </div>
            ) : null}

            <div ref={threadEndRef} />
          </div>

          <Composer
            locale={locale}
            onSubmit={(question) => void runSearch(question)}
            disabled={busy}
            hasConversation={Boolean(conversationId)}
          />
        </main>

        <PanelResizeHandle
          side="right"
          locale={locale}
          disabled={rightCollapsed}
          active={resizing === 'right'}
          onResizeStart={(event) => startResize('right', event)}
          onCollapse={() => setRightCollapsed(true)}
        />
        <EvidencePanel
          locale={locale}
          searchResults={activeTurn?.answer_status === 'search_results'}
          passage={passage}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          zoom={zoom}
          onZoomChange={setZoom}
          collapsed={rightCollapsed}
          onCollapse={() => setRightCollapsed(true)}
          onExpand={() => setRightCollapsed(false)}
        />
      </div>
      <LegalFooter locale={locale} />
    </div>
  );
}
