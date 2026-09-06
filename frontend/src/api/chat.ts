export interface ChatSource {
  file: string;
  page: number | string | null;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}

export async function postChat(question: string): Promise<ChatResponse> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    let detail = 'La requête a échoué.';
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail);
  }

  return response.json() as Promise<ChatResponse>;
}
