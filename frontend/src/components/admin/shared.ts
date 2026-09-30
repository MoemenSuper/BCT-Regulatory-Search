// Types and helpers shared by the admin pages.
import { parseServerDate } from '../../data/presentation';
import { type UiLocale } from '../../uiLocale';

export type AdminTab = 'overview' | 'users' | 'documents' | 'refusals' | 'configuration';

export type DocKind = 'regulatory' | 'statistical' | 'internal';

export type UploadEntryStatus = 'pending' | 'running' | 'imported' | 'duplicate' | 'failed';

export interface UploadEntry {
  name: string;
  status: UploadEntryStatus;
  detail?: string;
}

export interface UploadProgress {
  total: number;
  currentIndex: number;
  currentName: string;
  entries: UploadEntry[];
}

export function isPdfFile(file: File) {
  return file.name.toLowerCase().endsWith('.pdf') && file.size > 0 && file.size <= 50 * 1024 * 1024;
}

// The server stores UTC timestamps; show them in the admin's local time.
export function formatWhen(value: string, locale: UiLocale) {
  const date = parseServerDate(value);
  if (!date) return value;
  return date.toLocaleString(locale === 'ar' ? 'ar-TN' : locale, { dateStyle: 'medium', timeStyle: 'short' });
}
