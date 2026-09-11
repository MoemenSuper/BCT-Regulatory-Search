import type { PointerEvent } from 'react';

interface PanelResizeHandleProps {
  side: 'left' | 'right';
  disabled?: boolean;
  active?: boolean;
  onResizeStart: (event: PointerEvent<HTMLButtonElement>) => void;
  onCollapse: () => void;
}

export function PanelResizeHandle({
  side,
  disabled = false,
  active = false,
  onResizeStart,
  onCollapse,
}: PanelResizeHandleProps) {
  if (disabled) {
    return <div className="panel-resize is-disabled" aria-hidden="true" />;
  }

  const label =
    side === 'left'
      ? 'Redimensionner l’historique. Double-clic pour masquer.'
      : 'Redimensionner le panneau des sources. Double-clic pour masquer.';

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
