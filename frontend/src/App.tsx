import { useState } from 'react';
import { postChat } from './api/chat';
import { EvidencePanel } from './components/EvidencePanel';
import { FollowUpBox } from './components/FollowUpBox';
import { Header } from './components/Header';
import { HistorySidebar } from './components/HistorySidebar';
import { ResearchNote } from './components/ResearchNote';
import { SearchBar } from './components/SearchBar';
import {
  INITIAL_SELECTED_HISTORY_ID,
  SAMPLE_QUERY,
  evidenceFromChatSources,
  evidencePassage,
  historyGroups,
  noteFromChatAnswer,
  researchNote,
} from './data/mockData';
import type { EvidencePassage, EvidenceTab, ResearchNoteData } from './types/ui';
import './styles.css';

export default function App() {
  const [selectedHistoryId, setSelectedHistoryId] = useState(INITIAL_SELECTED_HISTORY_ID);
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [activeTab, setActiveTab] = useState<EvidenceTab>('preuve');
  const [zoom, setZoom] = useState(100);
  const [note, setNote] = useState<ResearchNoteData>(researchNote);
  const [passage, setPassage] = useState<EvidencePassage>(evidencePassage);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(question: string) {
    setQuery(question);
    setLoading(true);
    setError(null);

    try {
      const result = await postChat(question);
      setNote(noteFromChatAnswer(question, result.answer, result.sources));
      setPassage(evidenceFromChatSources(result.sources, result.answer));
      setActiveTab('preuve');
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Impossible de joindre le backend.';
      setError(
        `${message} Démarrez FastAPI (uvicorn app:app --reload --port 8000) puis réessayez.`,
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Header />
      <div className="workspace">
        <HistorySidebar
          groups={historyGroups}
          selectedId={selectedHistoryId}
          onSelect={setSelectedHistoryId}
        />

        <main className="center-panel">
          <SearchBar
            value={query}
            onChange={setQuery}
            onSubmit={runSearch}
            disabled={loading}
          />
          <div className="center-scroll">
            <ResearchNote note={note} loading={loading} error={error} />
          </div>
          <FollowUpBox onSubmit={runSearch} disabled={loading} />
        </main>

        <EvidencePanel
          passage={passage}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          zoom={zoom}
          onZoomChange={setZoom}
        />
      </div>
    </div>
  );
}
