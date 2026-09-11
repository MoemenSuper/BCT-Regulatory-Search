import type {
  ChatSource,
  ConversationSummary,
  EvidencePassage,
  HistoryGroup,
  ResearchNoteData,
} from '../types/ui';

function parseServerDate(value?: string | null): Date | null {
  if (!value) return null;
  const normalized = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value)
    ? `${value.replace(' ', 'T')}Z`
    : value;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
}

function dayKey(date: Date): string {
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

function historyLabel(date: Date): string {
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (dayKey(date) === dayKey(now)) return "AUJOURD'HUI";
  if (dayKey(date) === dayKey(yesterday)) return 'HIER';
  return date
    .toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' })
    .replace('.', '')
    .toUpperCase();
}

function pageLabel(file: string, page: number | string | null | undefined): string {
  return `${file}${page != null && page !== '' ? ` — page ${page}` : ''}`;
}

export function historyGroupsFromConversations(items: ConversationSummary[]): HistoryGroup[] {
  const groups = new Map<string, HistoryGroup>();
  for (const item of items) {
    const date = parseServerDate(item.updated_at) ?? new Date();
    const label = historyLabel(date);
    if (!groups.has(label)) groups.set(label, { label, items: [] });
    groups.get(label)!.items.push({
      id: item.conversation_id,
      title: item.title || item.last_question || 'Recherche réglementaire',
      time: date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }),
      turnCount: item.turn_count,
    });
  }
  return Array.from(groups.values());
}

export function noteFromConversationTurn(
  turn: {
    question: string;
    answer: string;
    sources?: ChatSource[];
    created_at?: string | null;
    answer_status?: string | null;
    turn_id: string;
  },
  conversationId: string,
  conversationTitle?: string,
): ResearchNoteData {
  const sources = turn.sources || [];
  const paragraphs = turn.answer
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);
  const created = parseServerDate(turn.created_at) ?? new Date();

  return {
    searchResults: turn.answer_status === 'search_results',
    title: conversationTitle || 'Note de recherche réglementaire',
    date: created.toLocaleDateString('fr-FR', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    }),
    analyst: 'Rechercheur BCT',
    reference: conversationId
      ? `RR-${conversationId.replace(/-/g, '').slice(0, 10).toUpperCase()}`
      : '—',
    question: turn.question,
    synthesis: paragraphs.length > 0 ? paragraphs : [turn.answer],
    sources: sources.map((source, index) => {
      const page = typeof source.page === 'number' ? source.page : Number(source.page) || null;
      return {
        id: index + 1,
        file: source.file,
        page,
        excerpt: source.excerpt || '',
        citation: pageLabel(source.file, page),
      };
    }),
  };
}

export function evidenceFromSource(source?: ChatSource | null): EvidencePassage | null {
  if (!source?.file) return null;
  const page = typeof source.page === 'number' ? source.page : Number(source.page) || 1;
  return {
    quote: source.excerpt || '',
    sourceLabel: `Source : ${pageLabel(source.file, source.page)}.`,
    filename: source.file,
    page,
  };
}
