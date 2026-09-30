// Types and helpers shared by the admin pages.

export type AdminTab = 'overview' | 'users' | 'documents' | 'refusals' | 'configuration';

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
