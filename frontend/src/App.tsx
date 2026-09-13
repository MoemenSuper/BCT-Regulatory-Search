import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { FileText } from 'lucide-react';
import {
  deleteConversation,
  getConversation,
  getConversations,
  postChat,
  renameConversation,
} from './api/chat';
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
import type {
  ChatSource,
  ConversationDetail,
  ConversationSummary,
  ConversationTurn,
  EvidenceTab,
} from './types/ui';
import './styles.css';

export default function App() {
  const [activeTab, setActiveTab] = useState<EvidenceTab>('preuve');
  const [zoom, setZoom] = useState(100);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [conversationTitle, setConversationTitle] = useState('Note de recherche réglementaire');
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
    setConversationTitle(detail.title || 'Note de recherche réglementaire');
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
      setError(err instanceof Error ? err.message : "Impossible de charger cette recherche.");
    } finally {
      setOpening(false);
    }
  }, []);

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
      const message = err instanceof Error ? err.message : 'Impossible de joindre le backend.';
      setError(`${message} Vérifiez que FastAPI est démarré puis réessayez.`);
    } finally {
      setPendingQuestion(null);
    }
  }

  function startNewSearch() {
    setConversationId(undefined);
    setSelectedHistoryId(null);
    setConversationTitle('Note de recherche réglementaire');
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
      setError(err instanceof Error ? err.message : 'Impossible de renommer cette recherche.');
    }
  }

  async function deleteHistoryItem(id: string) {
    try {
      await deleteConversation(id);
      if (id === conversationId) startNewSearch();
      await refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de supprimer cette recherche.');
    }
  }

  const busy = pendingQuestion !== null || opening;
  const historyGroups = useMemo(() => historyGroupsFromConversations(history), [history]);
  const activeTurn = turns[selectedTurnIndex];
  const sources: ChatSource[] = activeTurn?.sources || [];
  const passage = evidenceFromSource(sources[selectedSourceIndex]);
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

  return (
    <div className="app-shell">
      <Header />
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
                  <h2 className="note-title">Note de recherche réglementaire</h2>
                </div>
                <p className="note-body thread-empty-copy">
                  Posez une question ci-dessous pour démarrer une discussion réglementaire. Les
                  échanges précédents resteront visibles ici et pourront être rouverts depuis
                  l&apos;historique.
                </p>
              </article>
            ) : null}

            <div className="conversation-thread" role="log" aria-live="polite" aria-relevant="additions">
              {turns.map((turn, index) => {
                const note = noteFromConversationTurn(
                  turn,
                  conversationId || turn.turn_id,
                  conversationTitle,
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
                      note={note}
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
            onSubmit={(question) => void runSearch(question)}
            disabled={busy}
            hasConversation={Boolean(conversationId)}
          />
        </main>

        <PanelResizeHandle
          side="right"
          disabled={rightCollapsed}
          active={resizing === 'right'}
          onResizeStart={(event) => startResize('right', event)}
          onCollapse={() => setRightCollapsed(true)}
        />
        <EvidencePanel
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
      <LegalFooter />
    </div>
  );
}
