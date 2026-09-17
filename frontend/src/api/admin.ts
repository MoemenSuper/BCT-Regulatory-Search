import type { AuthUser } from './auth';

async function readError(response: Response): Promise<string> {
  const text = (await response.text()).trim();
  if (text) {
    try {
      const payload = JSON.parse(text) as { detail?: string | Array<{ msg?: string }> };
      if (typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail;
      if (Array.isArray(payload.detail) && payload.detail[0]?.msg) return String(payload.detail[0].msg);
    } catch {
      return text.slice(0, 300);
    }
  }
  return 'La requête a échoué.';
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: 'include',
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.headers || {}),
    },
  });
  if (!response.ok) throw new Error(await readError(response));
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export interface AdminOverview {
  users_total: number;
  users_pending: number;
  users_approved: number;
  users_rejected: number;
  documents_ready: number;
  active_profile: string;
  answer_refusals_total: number;
  graph: {
    graph_enabled: boolean;
    neo4j_connected: boolean;
    graph_ready: boolean;
  };
}

export interface AnswerRefusal {
  refusal_id: string;
  conversation_id: string | null;
  user_id: string | null;
  user_email: string | null;
  question: string;
  answer_status: string;
  reason: string;
  diagnostics: string[];
  profile: string | null;
  created_at: string;
}

export interface AnswerRefusalsPage {
  total: number;
  items: AnswerRefusal[];
}

export interface SecretInfo {
  key: string;
  configured: boolean;
  masked: string | null;
  source: string;
}

export interface AdminConfig {
  active_profile: string;
  cloud_retrieval_provider: string;
  cloud_retrieval_providers: Array<{
    value: string;
    label: string;
    provider: string;
    model: string;
    dimension: number;
  }>;
  profiles: Array<{
    value: string;
    label: string;
    retrieval: string;
    answer: string;
    qualification: string;
    description: string;
  }>;
  secrets: SecretInfo[];
}

export function getOverview(): Promise<AdminOverview> {
  return request('/api/admin/overview');
}

export function listAnswerRefusals(limit = 500): Promise<AnswerRefusalsPage> {
  return request(`/api/admin/answer-refusals?limit=${Math.max(1, Math.min(limit, 5000))}`);
}

export async function downloadAnswerRefusalsExport(): Promise<void> {
  const response = await fetch('/api/admin/answer-refusals/export', {
    credentials: 'include',
    headers: { Accept: 'text/csv' },
  });
  if (!response.ok) throw new Error(await readError(response));
  const disposition = response.headers.get('Content-Disposition') || '';
  const matched = /filename="([^"]+)"/i.exec(disposition);
  const filename = matched?.[1] || 'answer-refusals.csv';
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function listUsers(): Promise<AuthUser[]> {
  return request('/api/admin/users');
}

export function approveUser(userId: string): Promise<{ user: AuthUser }> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/approve`, { method: 'POST' });
}

export function promoteUser(userId: string): Promise<{ user: AuthUser }> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/promote`, { method: 'POST' });
}

export function rejectUser(userId: string): Promise<{ user: AuthUser }> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/reject`, { method: 'POST' });
}

export function deleteUser(userId: string): Promise<void> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}`, { method: 'DELETE' });
}

export function setUserTokenLimit(userId: string, tokenLimit: number): Promise<{ user: AuthUser }> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/token-limit`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token_limit: tokenLimit }),
  });
}

export function resetUserTokens(userId: string): Promise<{ user: AuthUser }> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/reset-tokens`, { method: 'POST' });
}

export function getConfig(): Promise<AdminConfig> {
  return request('/api/admin/config');
}

export function setProfile(profile: string): Promise<{ active_profile: string }> {
  return request('/api/admin/config/profile', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile }),
  });
}

export function setCloudRetrievalProvider(provider: string): Promise<{
  cloud_retrieval_provider: string;
  config: AdminConfig;
}> {
  return request('/api/admin/config/cloud-retrieval-provider', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });
}

export function setSecrets(secrets: Record<string, string | null>): Promise<AdminConfig> {
  return request('/api/admin/config/secrets', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ secrets }),
  });
}

export function listDocuments(): Promise<unknown[]> {
  return request('/api/documents');
}

export async function uploadDocument(form: FormData): Promise<unknown> {
  const response = await fetch('/api/documents', {
    method: 'POST',
    credentials: 'include',
    body: form,
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}
