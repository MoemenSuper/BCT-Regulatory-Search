import { AlertTriangle, FileText } from 'lucide-react';
import type { ResearchNoteData } from '../types/ui';

interface ResearchNoteProps {
  note: ResearchNoteData;
  loading?: boolean;
  error?: string | null;
}

export function ResearchNote({ note, loading = false, error = null }: ResearchNoteProps) {
  return (
    <article className="research-note">
      <div className="note-header">
        <div className="note-title-row">
          <span className="note-icon-wrap" aria-hidden="true">
            <FileText size={18} strokeWidth={1.75} />
          </span>
          <h2 className="note-title">{note.title}</h2>
        </div>

        <aside className="warning-box">
          <div className="warning-title-row">
            <AlertTriangle size={14} strokeWidth={2} />
            <strong>{note.warningTitle}</strong>
          </div>
          <p>{note.warningText}</p>
        </aside>
      </div>

      <p className="note-meta">
        Date : {note.date}
        <span className="meta-sep">|</span>
        Analyste : {note.analyst}
        <span className="meta-sep">|</span>
        Référence interne : {note.reference}
      </p>

      {loading ? (
        <div className="note-status" role="status">
          Recherche en cours auprès du moteur réglementaire…
        </div>
      ) : null}

      {error ? (
        <div className="note-status note-status-error" role="alert">
          {error}
        </div>
      ) : null}

      <section className="note-section">
        <h3>Question de recherche</h3>
        <p className="note-question">{note.question}</p>
      </section>

      <section className="note-section">
        <h3>Synthèse</h3>
        {note.synthesis.map((paragraph, index) => (
          <p key={index} className="note-body">
            {paragraph}
          </p>
        ))}
      </section>

      {note.keyFindings.length > 0 ? (
        <section className="note-section">
          <h3>Constats clés</h3>
          <ol className="key-findings">
            {note.keyFindings.map((finding, index) => (
              <li key={finding.title}>
                <span className="finding-index">{index + 1}.</span>
                <span>
                  <strong>{finding.title}. </strong>
                  {finding.text}
                </span>
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      <section className="note-section">
        <h3>Sources principales</h3>
        {note.sources.length > 0 ? (
          <ol className="sources-list">
            {note.sources.map((source) => (
              <li key={source.id}>
                <span className="source-ref">[{source.id}]</span> {source.citation}
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
