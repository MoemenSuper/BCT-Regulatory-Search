import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type FormEvent } from 'react';
import { createPortal } from 'react-dom';
import { ImagePlus, LogOut, Trash2 } from 'lucide-react';
import {
  changePassword,
  updateProfile,
  type AuthUser,
} from '../api/auth';
import { t, type UiLocale } from '../uiLocale';

const MAX_UPLOAD_BYTES = 2_000_000;
const AVATAR_EDGE = 160;

export function displayLabel(user: AuthUser): string {
  const name = (user.display_name || '').trim();
  if (name) return name;
  return user.email.split('@')[0] || user.email;
}

function isImageAvatar(value: string | undefined): boolean {
  const lower = (value || '').toLowerCase();
  return lower.startsWith('data:image/jpeg;base64,') || lower.startsWith('data:image/png;base64,');
}

export function AvatarMark({
  user,
  size = 36,
  className = '',
}: {
  user: AuthUser;
  size?: number;
  className?: string;
}) {
  if (isImageAvatar(user.avatar_icon)) {
    return (
      <span className={`avatar-mark avatar-mark--image ${className}`.trim()} style={{ width: size, height: size }} aria-hidden="true">
        <img src={user.avatar_icon} alt="" />
      </span>
    );
  }
  const letter = displayLabel(user).charAt(0).toUpperCase() || '?';
  return (
    <span className={`avatar-mark ${className}`.trim()} style={{ width: size, height: size, fontSize: size * 0.4 }} aria-hidden="true">
      {letter}
    </span>
  );
}

async function fileToAvatarDataUrl(file: File): Promise<string> {
  if (file.type !== 'image/jpeg' && file.type !== 'image/png') {
    throw new Error('JPEG_PNG_ONLY');
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error('TOO_LARGE');
  }
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('INVALID'));
      img.src = objectUrl;
    });
    const canvas = document.createElement('canvas');
    canvas.width = AVATAR_EDGE;
    canvas.height = AVATAR_EDGE;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('INVALID');
    const scale = Math.max(AVATAR_EDGE / image.width, AVATAR_EDGE / image.height);
    const width = image.width * scale;
    const height = image.height * scale;
    context.drawImage(image, (AVATAR_EDGE - width) / 2, (AVATAR_EDGE - height) / 2, width, height);
    return canvas.toDataURL('image/jpeg', 0.85);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

interface ProfileMenuProps {
  user: AuthUser;
  locale: UiLocale;
  onUserChange: (user: AuthUser) => void;
  onLogout: () => void;
  variant?: 'header' | 'admin';
  /** Light/dark for the portaled panel — must match the host surface theme. */
  theme: 'light' | 'dark';
}

export function ProfileMenu({
  user,
  locale,
  onUserChange,
  onLogout,
  variant = 'header',
  theme,
}: ProfileMenuProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(user.display_name || '');
  const [avatar, setAvatar] = useState(user.avatar_icon || '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [panelStyle, setPanelStyle] = useState<CSSProperties>({});
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const titleId = useId();

  useEffect(() => {
    setName(user.display_name || '');
    setAvatar(user.avatar_icon || '');
  }, [user.display_name, user.avatar_icon]);

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    const width = Math.min(360, Math.max(300, window.innerWidth - 24));
    if (variant === 'admin') {
      setPanelStyle({
        position: 'fixed',
        left: Math.min(Math.max(12, rect.left), window.innerWidth - width - 12),
        bottom: Math.max(12, window.innerHeight - rect.top + 10),
        width,
        zIndex: 400,
      });
    } else {
      setPanelStyle({
        position: 'fixed',
        top: Math.min(rect.bottom + 10, window.innerHeight - 24),
        right: Math.max(12, window.innerWidth - rect.right),
        width,
        zIndex: 400,
      });
    }
  }, [open, variant]);

  useEffect(() => {
    if (!open) return;
    function onPointer(event: MouseEvent) {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target) || panelRef.current?.contains(target)) return;
      setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    function onReposition() {
      if (!triggerRef.current) return;
      const rect = triggerRef.current.getBoundingClientRect();
      const width = Math.min(360, Math.max(300, window.innerWidth - 24));
      if (variant === 'admin') {
        setPanelStyle({
          position: 'fixed',
          left: Math.min(Math.max(12, rect.left), window.innerWidth - width - 12),
          bottom: Math.max(12, window.innerHeight - rect.top + 10),
          width,
          zIndex: 400,
        });
      } else {
        setPanelStyle({
          position: 'fixed',
          top: Math.min(rect.bottom + 10, window.innerHeight - 24),
          right: Math.max(12, window.innerWidth - rect.right),
          width,
          zIndex: 400,
        });
      }
    }
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', onReposition);
    window.addEventListener('scroll', onReposition, true);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onReposition);
      window.removeEventListener('scroll', onReposition, true);
    };
  }, [open, variant]);

  async function onPickFile(file: File | null) {
    if (!file) return;
    setError(null);
    setMessage(null);
    try {
      setAvatar(await fileToAvatarDataUrl(file));
    } catch (err) {
      const code = err instanceof Error ? err.message : '';
      if (code === 'JPEG_PNG_ONLY') setError(t(locale, 'profile.avatarType'));
      else if (code === 'TOO_LARGE') setError(t(locale, 'profile.avatarTooLarge'));
      else setError(t(locale, 'profile.avatarInvalid'));
    } finally {
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  async function saveProfile(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await updateProfile({
        display_name: name.trim(),
        avatar_icon: avatar,
      });
      onUserChange(result.user);
      setMessage(t(locale, 'profile.saved'));
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : t(locale, 'profile.saveFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function savePassword(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await changePassword(currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      setMessage(t(locale, 'profile.passwordSaved'));
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : t(locale, 'profile.passwordFailed'));
    } finally {
      setBusy(false);
    }
  }

  const previewUser: AuthUser = { ...user, display_name: name, avatar_icon: avatar };

  const panel = open
    ? createPortal(
        <div
          className={`profile-menu-panel profile-menu-panel--${variant}`}
          role="dialog"
          aria-labelledby={titleId}
          ref={panelRef}
          style={panelStyle}
          data-theme={theme}
        >
          <h2 id={titleId}>{t(locale, 'profile.title')}</h2>
          <p className="profile-menu-email">{user.email}</p>
          {message ? <p className="profile-menu-banner ok" role="status">{message}</p> : null}
          {error ? <p className="profile-menu-banner error" role="alert">{error}</p> : null}
          <form className="profile-menu-form" onSubmit={(event) => void saveProfile(event)}>
            <label>
              {t(locale, 'profile.displayName')}
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                maxLength={80}
                autoComplete="nickname"
                placeholder={t(locale, 'profile.displayNamePlaceholder')}
              />
            </label>
            <div className="profile-avatar-upload">
              <span>{t(locale, 'profile.avatar')}</span>
              <div className="profile-avatar-row">
                <AvatarMark user={previewUser} size={56} />
                <div className="profile-avatar-actions">
                  <button type="button" className="profile-secondary" onClick={() => fileRef.current?.click()} disabled={busy}>
                    <ImagePlus size={16} strokeWidth={1.8} aria-hidden="true" />
                    {t(locale, 'profile.avatarUpload')}
                  </button>
                  {avatar ? (
                    <button type="button" className="profile-secondary" onClick={() => setAvatar('')} disabled={busy}>
                      <Trash2 size={16} strokeWidth={1.8} aria-hidden="true" />
                      {t(locale, 'profile.avatarClear')}
                    </button>
                  ) : null}
                </div>
              </div>
              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png,.jpg,.jpeg,.png"
                className="sr-only"
                onChange={(event) => void onPickFile(event.target.files?.[0] || null)}
              />
              <small>{t(locale, 'profile.avatarHelp')}</small>
            </div>
            <button type="submit" className="profile-primary" disabled={busy}>
              {t(locale, 'profile.save')}
            </button>
          </form>
          <form className="profile-menu-form" onSubmit={(event) => void savePassword(event)}>
            <label>
              {t(locale, 'profile.currentPassword')}
              <input
                type="password"
                value={currentPassword}
                onChange={(event) => setCurrentPassword(event.target.value)}
                autoComplete="current-password"
                required
              />
            </label>
            <label>
              {t(locale, 'profile.newPassword')}
              <input
                type="password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                autoComplete="new-password"
                minLength={8}
                required
              />
            </label>
            <button type="submit" className="profile-primary" disabled={busy}>
              {t(locale, 'profile.changePassword')}
            </button>
          </form>
          <button type="button" className="profile-logout" onClick={onLogout}>
            <LogOut size={16} strokeWidth={1.8} aria-hidden="true" />
            {t(locale, 'auth.signOut')}
          </button>
        </div>,
        document.body,
      )
    : null;

  return (
    <div className={`profile-menu profile-menu--${variant}`}>
      <button
        ref={triggerRef}
        type="button"
        className="profile-menu-trigger"
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls={open ? titleId : undefined}
        onClick={() => {
          setOpen((value) => !value);
          setError(null);
          setMessage(null);
        }}
      >
        <AvatarMark user={user} size={variant === 'admin' ? 36 : 34} />
        <span className="profile-menu-trigger-copy">
          <strong>{displayLabel(user)}</strong>
          <small>{user.email}</small>
        </span>
      </button>
      {panel}
    </div>
  );
}
