export interface HistoryItem {
  id: string;
  title: string;
  time: string;
}

export interface HistoryGroup {
  label: string;
  items: HistoryItem[];
}

export interface KeyFinding {
  title: string;
  text: string;
}

export interface ResearchSource {
  id: number;
  citation: string;
}

export interface ResearchNoteData {
  title: string;
  date: string;
  analyst: string;
  reference: string;
  warningTitle: string;
  warningText: string;
  question: string;
  synthesis: string[];
  keyFindings: KeyFinding[];
  sources: ResearchSource[];
}

export interface EvidencePassage {
  quote: string;
  sourceLabel: string;
  filename: string;
  page: number;
  totalPages: number;
}

export type EvidenceTab = 'preuve' | 'document';
