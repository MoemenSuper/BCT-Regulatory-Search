import type { PointerEvent } from 'react';
import { t, type UiLocale } from '../uiLocale';

interface PanelResizeHandleProps {
  side: 'left' | 'right';
  locale: UiLocale;
  disabled?: boolean;
  active?: boolean;
  onResizeStart: (event: PointerEvent<HTMLButtonElement>) => void;
  onCollapse: () => void;
}

export function PanelResizeHandle({
  side,
  locale,
  disabled = false,
  active = false,
  onResizeStart,
  onCollapse,
}: PanelResizeHandleProps) {
  if (disabled) {
    return <div className="panel-resize is-disabled" aria-hidden="true" />;
  }

  const label = t(locale, side === 'left' ? 'chat.resizeHistory' : 'chat.resizeSources');

  return (
    <button
      type="button"
      className={`panel-resize${active ? ' is-active' : ''}`}
      aria-label={label}
      title={label}
      onPointerDown={onResizeStart}
      onDoubleClick={(event) => {
        event.preventDefault();
        onCollapse();
      }}
    />
  );
}
