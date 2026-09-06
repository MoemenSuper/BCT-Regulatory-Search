import { useState } from 'react';

/** Paper-plane send icon with the folded tail notch from the reference. */
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

interface FollowUpBoxProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
}

export function FollowUpBox({ onSubmit, disabled = false }: FollowUpBoxProps) {
  const [value, setValue] = useState('');

  return (
    <div className="follow-up-box">
      <label className="follow-up-label" htmlFor="follow-up-input">
        Question de suivi
      </label>
      <form
        className="follow-up-row"
        onSubmit={(event) => {
          event.preventDefault();
          const next = value.trim();
          if (!disabled && next) {
            onSubmit(next);
            setValue('');
          }
        }}
      >
        <input
          id="follow-up-input"
          type="text"
          className="follow-up-input"
          placeholder="Posez une question de suivi sur cette recherche..."
          value={value}
          disabled={disabled}
          onChange={(e) => setValue(e.target.value)}
        />
        <button type="submit" className="btn-poser" disabled={disabled}>
          <PaperPlaneIcon />
          <span>{disabled ? '…' : 'Poser'}</span>
        </button>
      </form>
    </div>
  );
}
