import type {
  ChatSource,
  ConversationDetail,
  ConversationSummary,
  SourceInfo,
} from '../types/ui';

export interface ChatResponse {
  conversation_id: string;
  profile: string;
  status: string;
  answer: string;
  sources: ChatSource[];
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || 'La requête a échoué.';
  } catch {
    return 'La requête a échoué.';
  }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<T>;
}

export async function postChat(
  question: string,
  conversationId?: string,
  profile?: string,
): Promise<ChatResponse> {
  const body: { question: string; conversation_id?: string; profile?: string } = { question };
  if (conversationId) body.conversation_id = conversationId;
  if (profile) body.profile = profile;
  const response = await fetch('/api/chat', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<ChatResponse>;
}

export function getConversations(limit = 100): Promise<ConversationSummary[]> {
  return getJson(`/api/conversations?limit=${Math.max(1, Math.min(limit, 500))}`);
}

export function getConversation(conversationId: string): Promise<ConversationDetail> {
  return getJson(`/api/conversations/${encodeURIComponent(conversationId)}`);
}

export async function renameConversation(conversationId: string, title: string): Promise<void> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'PATCH',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) throw new Error(await readError(response));
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (!response.ok) throw new Error(await readError(response));
}

export async function postTurnFeedback(
  conversationId: string,
  turnId: string,
  rating: 'up' | 'down',
): Promise<void> {
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}/feedback`,
    {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating }),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
}

export function getSourceInfo(filename: string): Promise<SourceInfo> {
  return getJson(`/api/sources/${encodeURIComponent(filename)}/info`);
}

export function sourcePageImageUrl(
  filename: string,
  page: number,
  quote: string,
  scale = 2,
): string {
  const params = new URLSearchParams();
  if (quote.trim()) params.set('quote', quote.trim().slice(0, 2000));
  params.set('scale', String(scale));
  return `/api/sources/${encodeURIComponent(filename)}/page/${page}.png?${params.toString()}`;
}

export function sourcePdfUrl(filename: string, page?: number): string {
  const base = `/api/sources/${encodeURIComponent(filename)}`;
  return page ? `${base}#page=${page}&zoom=page-width` : base;
}
