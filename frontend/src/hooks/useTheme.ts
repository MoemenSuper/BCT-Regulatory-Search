import { useState } from 'react';

export type Theme = 'light' | 'dark';

const CHAT_THEME_KEY = 'bct-chat-theme';
const AUTH_THEME_KEY = 'bct-auth-theme';
/** @deprecated migrated once from shared html theme */
const LEGACY_THEME_KEY = 'bct-theme';

function readTheme(storageKey: string, fallback: Theme = 'light'): Theme {
  try {
    const stored = localStorage.getItem(storageKey);
    if (stored === 'dark' || stored === 'light') return stored;
    if (storageKey === CHAT_THEME_KEY) {
      const legacy = localStorage.getItem(LEGACY_THEME_KEY);
      if (legacy === 'dark' || legacy === 'light') return legacy;
    }
  } catch {
    /* private mode */
  }
  return fallback;
}

function writeTheme(storageKey: string, theme: Theme) {
  try {
    localStorage.setItem(storageKey, theme);
  } catch {
    /* ignore */
  }
}

/** Per-surface theme. Never writes html[data-theme] — each shell owns its attribute. */
export function useScopedTheme(storageKey: string) {
  const [theme, setThemeState] = useState<Theme>(() => readTheme(storageKey));

  function setTheme(next: Theme) {
    writeTheme(storageKey, next);
    setThemeState(next);
  }

  return {
    theme,
    setTheme,
    toggleTheme: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
  };
}

export function useChatTheme() {
  return useScopedTheme(CHAT_THEME_KEY);
}

export function useAuthTheme() {
  return useScopedTheme(AUTH_THEME_KEY);
}

/** Clear any leftover global theme from older builds; migrate once into chat storage. */
export function clearGlobalThemeAttribute() {
  document.documentElement.removeAttribute('data-theme');
  document.documentElement.style.colorScheme = '';
  try {
    const legacy = localStorage.getItem(LEGACY_THEME_KEY);
    if ((legacy === 'dark' || legacy === 'light') && !localStorage.getItem(CHAT_THEME_KEY)) {
      localStorage.setItem(CHAT_THEME_KEY, legacy);
    }
    localStorage.removeItem(LEGACY_THEME_KEY);
  } catch {
    /* ignore */
  }
}
