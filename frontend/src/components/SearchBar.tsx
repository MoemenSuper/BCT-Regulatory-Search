import { Search, X } from 'lucide-react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  disabled?: boolean;
}

export function SearchBar({ value, onChange, onSubmit, disabled = false }: SearchBarProps) {
  return (
    <form
      className="search-bar"
      onSubmit={(event) => {
        event.preventDefault();
        if (!disabled && value.trim()) onSubmit(value.trim());
      }}
    >
      <Search size={18} strokeWidth={1.75} className="search-bar-icon" />
      <input
        type="text"
        className="search-bar-input"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Recherche réglementaire"
      />
      {value ? (
        <button
          type="button"
          className="search-bar-clear"
          aria-label="Effacer la recherche"
          disabled={disabled}
          onClick={() => onChange('')}
        >
          <X size={16} strokeWidth={1.75} />
        </button>
      ) : null}
    </form>
  );
}
