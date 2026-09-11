import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';

const STORAGE_KEY = 'bct-workspace-panels';
const LEFT_DEFAULT = 280;
const RIGHT_DEFAULT = 400;
const LEFT_MIN = 220;
const LEFT_MAX = 460;
const RIGHT_MIN = 300;
const RIGHT_MAX = 640;
const RAIL = 44;
const CENTER_MIN = 360;
const HANDLE = 5;
const GAP = 8;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

interface StoredLayout {
  leftWidth: number;
  rightWidth: number;
  leftCollapsed: boolean;
  rightCollapsed: boolean;
}

function readStored(): StoredLayout {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<StoredLayout>;
      return {
        leftWidth:
          typeof parsed.leftWidth === 'number'
            ? clamp(parsed.leftWidth, LEFT_MIN, LEFT_MAX)
            : LEFT_DEFAULT,
        rightWidth:
          typeof parsed.rightWidth === 'number'
            ? clamp(parsed.rightWidth, RIGHT_MIN, RIGHT_MAX)
            : RIGHT_DEFAULT,
        leftCollapsed: typeof parsed.leftCollapsed === 'boolean' ? parsed.leftCollapsed : false,
        rightCollapsed: typeof parsed.rightCollapsed === 'boolean' ? parsed.rightCollapsed : false,
      };
    }
  } catch {
    /* keep defaults */
  }
  return {
    leftWidth: LEFT_DEFAULT,
    rightWidth: RIGHT_DEFAULT,
    leftCollapsed: false,
    rightCollapsed: false,
  };
}

export function useWorkspacePanels() {
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const [layout, setLayout] = useState(() => readStored());
  const [resizing, setResizing] = useState<'left' | 'right' | null>(null);
  const { leftWidth, rightWidth, leftCollapsed, rightCollapsed } = layout;

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
  }, [layout]);

  useEffect(() => {
    const media = window.matchMedia('(max-width: 900px)');
    const onChange = () => {
      if (media.matches) {
        setLayout((prev) => ({ ...prev, leftCollapsed: false, rightCollapsed: false }));
      }
    };
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  const startResize = useCallback(
    (side: 'left' | 'right', event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      const originX = event.clientX;
      const originLeft = leftWidth;
      const originRight = rightWidth;
      const workspaceWidth = workspaceRef.current?.clientWidth ?? window.innerWidth;
      const chrome = HANDLE * 2 + GAP * 4;
      setResizing(side);

      const onMove = (moveEvent: PointerEvent) => {
        const delta = moveEvent.clientX - originX;
        const maxLeft = Math.min(
          LEFT_MAX,
          workspaceWidth - chrome - (rightCollapsed ? RAIL : originRight) - CENTER_MIN,
        );
        const maxRight = Math.min(
          RIGHT_MAX,
          workspaceWidth - chrome - (leftCollapsed ? RAIL : originLeft) - CENTER_MIN,
        );
        if (side === 'left') {
          const next = clamp(originLeft + delta, LEFT_MIN, Math.max(LEFT_MIN, maxLeft));
          setLayout((prev) => ({ ...prev, leftWidth: next }));
        } else {
          const next = clamp(originRight - delta, RIGHT_MIN, Math.max(RIGHT_MIN, maxRight));
          setLayout((prev) => ({ ...prev, rightWidth: next }));
        }
      };

      const onUp = () => {
        setResizing(null);
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };

      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    },
    [leftCollapsed, leftWidth, rightCollapsed, rightWidth],
  );

  return {
    workspaceRef,
    leftCollapsed,
    rightCollapsed,
    resizing,
    leftColumn: leftCollapsed ? RAIL : leftWidth,
    rightColumn: rightCollapsed ? RAIL : rightWidth,
    setLeftCollapsed: (value: boolean) => setLayout((prev) => ({ ...prev, leftCollapsed: value })),
    setRightCollapsed: (value: boolean) => setLayout((prev) => ({ ...prev, rightCollapsed: value })),
    startResize,
  };
}
