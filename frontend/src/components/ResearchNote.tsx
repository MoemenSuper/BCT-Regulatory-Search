import { useState, type MouseEvent } from 'react';
import { FileText, MessageSquareText, ThumbsDown, ThumbsUp } from 'lucide-react';
import { postTurnFeedback } from '../api/chat';
import type { ResearchNoteData } from '../types/ui';
import { t, type UiLocale } from '../uiLocale';

interface ResearchNoteProps {
  locale: UiLocale;
  note: ResearchNoteData;
  conversationId: string;
  turnId: string;
  compact?: boolean;
  selectedSourceIndex?: number;
  onSelectSource?: (index: number) => void;
}

type Rating = 'up' | 'down';

function feedbackKey(turnId: string) {
  return `bct-turn-feedback:${turnId}`;
}

function readRating(turnId: string): Rating | null {
  try {
    const value = sessionStorage.getItem(feedbackKey(turnId));
    return value === 'up' || value === 'down' ? value : null;
  } catch {
    return null;
  }
}

function writeRating(turnId: string, rating: Rating | null) {
  try {
    if (rating) sessionStorage.setItem(feedbackKey(turnId), rating);
    else sessionStorage.removeItem(feedbackKey(turnId));
  } catch {
    /* ignore */
  }
}

export function ResearchNote({
  locale,
  note,
  conversationId,
  turnId,
  compact = false,
  selectedSourceIndex = 0,
  onSelectSource,
}: ResearchNoteProps) {
  const [rating, setRating] = useState<Rating | null>(() => readRating(turnId));
  const [busy, setBusy] = useState(false);

  // Clicking the other button switches the rating; clicking the active one takes it back.
  async function sendFeedback(event: MouseEvent, clicked: Rating) {
    event.stopPropagation();
    if (busy) return;
    const next = rating === clicked ? null : clicked;
    setBusy(true);
    try {
      await postTurnFeedback(conversationId, turnId, next ?? 'none');
      writeRating(turnId, next);
      setRating(next);
    } catch {
      /* leave unset so the user can retry */
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className={`research-note${compact ? ' research-note-compact' : ''}`}>
      {!compact ? (
        <div className="note-header">
          <div className="note-title-row">
            <span className="note-icon-wrap" aria-hidden="true">
              <FileText size={18} strokeWidth={1.75} />
            </span>
            <h2 className="note-title" dir="auto">{note.title}</h2>
          </div>
        </div>
      ) : (
        <div className="note-turn-badge">
          <MessageSquareText size={14} strokeWidth={1.75} aria-hidden="true" />
          <span>{t(locale, 'chat.nextTurn')}</span>
          <span className="note-turn-date">{note.date}</span>
        </div>
      )}

      {!compact ? (
        <p className="note-meta">
          {t(locale, 'chat.date', { date: note.date })}
          <span className="meta-sep">|</span>
          {t(locale, 'chat.analyst')}
          <span className="meta-sep">|</span>
          {t(locale, 'chat.reference', { reference: note.reference })}
        </p>
      ) : null}

      <section className="note-section">
        <h3>{t(locale, 'chat.question')}</h3>
        <p className={`note-question${note.question ? '' : ' note-question-empty'}`} dir="auto">
          {note.question || t(locale, 'chat.noQuestion')}
        </p>
      </section>

      <section className="note-section">
        <h3>{t(locale, note.searchResults ? 'chat.search' : 'chat.answer')}</h3>
        {note.synthesis.map((paragraph, index) => (
          <p key={`${index}-${paragraph.slice(0, 24)}`} className="note-body" dir="auto">
            {paragraph}
          </p>
        ))}
      </section>

      <section className="note-section">
        <h3>{t(locale, note.searchResults ? 'chat.resultsToCheck' : 'chat.sources')}</h3>
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
          <p className="note-body">{t(locale, 'chat.noSources')}</p>
        )}
      </section>

      <div className="note-feedback" role="group" aria-label={t(locale, 'chat.rate')}>
        <button
          type="button"
          className={`note-feedback-btn${rating === 'up' ? ' is-active' : ''}`}
          aria-pressed={rating === 'up'}
          aria-label={t(locale, 'chat.helpful')}
          title={t(locale, 'chat.helpful')}
          disabled={busy}
          onClick={(event) => void sendFeedback(event, 'up')}
        >
          <ThumbsUp size={16} strokeWidth={1.8} aria-hidden="true" />
        </button>
        <button
          type="button"
          className={`note-feedback-btn${rating === 'down' ? ' is-active is-down' : ''}`}
          aria-pressed={rating === 'down'}
          aria-label={t(locale, 'chat.wrong')}
          title={t(locale, 'chat.wrong')}
          disabled={busy}
          onClick={(event) => void sendFeedback(event, 'down')}
        >
          <ThumbsDown size={16} strokeWidth={1.8} aria-hidden="true" />
        </button>
      </div>
    </article>
  );
}
