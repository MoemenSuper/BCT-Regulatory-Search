export type UserRole = 'user' | 'admin';
export type UserStatus = 'pending' | 'approved' | 'rejected';

export interface AuthUser {
  id: string;
  email: string;
  display_name?: string;
  avatar_icon?: string;
  role: UserRole;
  status: UserStatus;
  created_at: number;
  updated_at: number;
  token_limit?: number;
  tokens_used?: number;
  tokens_llm?: number;
  tokens_embed?: number;
  tokens_rerank?: number;
  tokens_remaining?: number | null;
  estimated_spend_usd?: number;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string | { msg?: string }[] };
    if (typeof payload.detail === 'string') return payload.detail;
    if (Array.isArray(payload.detail) && payload.detail[0]?.msg) return payload.detail[0].msg;
    return 'La requête a échoué.';
  } catch {
    return 'La requête a échoué.';
  }
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

export function getMe(): Promise<{ user: AuthUser }> {
  return request('/api/auth/me');
}

export function login(email: string, password: string): Promise<{ user: AuthUser }> {
  return request('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
}

export function register(email: string, password: string): Promise<{ user: AuthUser; message: string }> {
  return request('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
}

export function logout(): Promise<{ ok: boolean }> {
  return request('/api/auth/logout', { method: 'POST' });
}

export function updateProfile(payload: {
  display_name?: string;
  avatar_icon?: string;
}): Promise<{ user: AuthUser }> {
  return request('/api/auth/profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function changePassword(currentPassword: string, newPassword: string): Promise<{ ok: boolean }> {
  return request('/api/auth/password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
}
