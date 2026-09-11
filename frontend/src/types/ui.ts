export interface ChatSource {
  file: string;
  page: number | string | null;
  score?: number;
  excerpt?: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  time: string;
  turnCount: number;
}

export interface HistoryGroup {
  label: string;
  items: HistoryItem[];
}

export interface ResearchSource {
  id: number;
  citation: string;
  file: string;
  page: number | null;
  excerpt: string;
}

export interface ResearchNoteData {
  searchResults?: boolean;
  title: string;
  date: string;
  analyst: string;
  reference: string;
  question: string;
  synthesis: string[];
  sources: ResearchSource[];
}

export interface EvidencePassage {
  quote: string;
  sourceLabel: string;
  filename: string;
  page: number;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  last_question: string;
  turn_count: number;
  updated_at: string;
}

export interface ConversationTurn {
  turn_id: string;
  question: string;
  answer: string;
  sources: ChatSource[];
  answer_status?: string | null;
  created_at?: string | null;
}

export interface ConversationDetail {
  conversation_id: string;
  title?: string;
  turns: ConversationTurn[];
}

export interface SourceInfo {
  filename: string;
  pages: number;
  title: string;
}

export type EvidenceTab = 'preuve' | 'document';
