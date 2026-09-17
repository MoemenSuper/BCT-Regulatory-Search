import { FileText, MessageSquareText } from 'lucide-react';
import type { ResearchNoteData } from '../types/ui';

interface ResearchNoteProps {
  note: ResearchNoteData;
  compact?: boolean;
  selectedSourceIndex?: number;
  onSelectSource?: (index: number) => void;
}

export function ResearchNote({
  note,
  compact = false,
  selectedSourceIndex = 0,
  onSelectSource,
}: ResearchNoteProps) {
  return (
    <article className={`research-note${compact ? ' research-note-compact' : ''}`}>
      {!compact ? (
        <div className="note-header">
          <div className="note-title-row">
            <span className="note-icon-wrap" aria-hidden="true">
              <FileText size={18} strokeWidth={1.75} />
            </span>
            <h2 className="note-title">{note.title}</h2>
          </div>
        </div>
      ) : (
        <div className="note-turn-badge">
          <MessageSquareText size={14} strokeWidth={1.75} aria-hidden="true" />
          <span>Échange suivant</span>
          <span className="note-turn-date">{note.date}</span>
        </div>
      )}

      {!compact ? (
        <p className="note-meta">
          Date : {note.date}
          <span className="meta-sep">|</span>
          Analyste : {note.analyst}
          <span className="meta-sep">|</span>
          Référence interne : {note.reference}
        </p>
      ) : null}

      <section className="note-section">
        <h3>Question</h3>
        <p className={`note-question${note.question ? '' : ' note-question-empty'}`}>
          {note.question || 'Aucune question active.'}
        </p>
      </section>

      <section className="note-section">
        <h3>{note.searchResults ? 'Recherche' : 'Réponse'}</h3>
        {note.synthesis.map((paragraph, index) => (
          <p key={`${index}-${paragraph.slice(0, 24)}`} className="note-body">
            {paragraph}
          </p>
        ))}
      </section>

      <section className="note-section">
        <h3>{note.searchResults ? 'Résultats à examiner' : 'Sources'}</h3>
        {note.sources.length > 0 ? (
          <ol className="sources-list sources-list-live">
            {note.sources.map((source, index) => (
              <li
                key={`${source.file}-${source.page}-${index}`}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelectSource?.(index);
                }}
              >
                <button
                  type="button"
                  className={`source-link${selectedSourceIndex === index ? ' selected' : ''}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectSource?.(index);
                  }}
                >
                  <span className="source-ref">{note.searchResults ? `${source.id}.` : `[${source.id}]`}</span>
                  <span>{source.citation}</span>
                </button>
                {source.excerpt ? (
                  <p className="note-body" dir="auto">{source.excerpt}</p>
                ) : null}
              </li>
            ))}
          </ol>
        ) : (
          <p className="note-body">Aucune source renvoyée pour cette réponse.</p>
        )}
      </section>
    </article>
  );
}
