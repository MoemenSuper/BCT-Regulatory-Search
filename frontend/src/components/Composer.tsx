import { useState } from 'react';
import { Search } from 'lucide-react';
import { t, type UiLocale } from '../uiLocale';

function PaperPlaneIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M22 2 11 13"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M22 2 15 22l-4-9-9-4 20-7z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

interface ComposerProps {
  locale: UiLocale;
  onSubmit: (value: string) => void;
  disabled?: boolean;
  hasConversation?: boolean;
}

export function Composer({ locale, onSubmit, disabled = false, hasConversation = false }: ComposerProps) {
  const [value, setValue] = useState('');

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        const next = value.trim();
        if (!disabled && next) {
          onSubmit(next);
          setValue('');
        }
      }}
    >
      <label className="composer-label" htmlFor="composer-input">
        {t(locale, hasConversation ? 'chat.continue' : 'chat.ask')}
      </label>
      <div className="composer-row">
        <Search size={18} strokeWidth={1.75} className="composer-icon" aria-hidden="true" />
        <input
          id="composer-input"
          type="text"
          className="composer-input"
          value={value}
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          dir="auto"
          placeholder={t(locale, hasConversation ? 'chat.placeholderFollow' : 'chat.placeholder')}
          aria-label={t(locale, hasConversation ? 'chat.followQuestion' : 'chat.newQuestion')}
        />
        <button type="submit" className="btn-poser" disabled={disabled || !value.trim()}>
          <PaperPlaneIcon />
          <span>{disabled ? '…' : t(locale, 'chat.send')}</span>
        </button>
      </div>
    </form>
  );
}
