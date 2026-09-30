// Runtime profile and provider API keys.
import { type FormEvent } from 'react';
import { CheckCircle2 } from 'lucide-react';
import { type AdminConfig } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { BlockSkeleton } from './Skeletons';

export function ConfigurationPage({ config, loading, busy, locale, onProfile, onSecrets }: {
  config: AdminConfig | null;
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  onProfile: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onSecrets: (event: FormEvent<HTMLFormElement>) => Promise<void>;
}) {
  if (loading || !config) {
    return (
      <section className="admin-configuration">
        <div className="admin-panel"><BlockSkeleton label={t(locale, 'admin.loading')} /></div>
        <div className="admin-panel"><BlockSkeleton label={t(locale, 'admin.loading')} /></div>
      </section>
    );
  }
  // Plain-language title and summary for each profile the server offers.
  const profileGuides: Record<string, { title: string; body: string }> = {
    cloud: { title: t(locale, 'admin.profileCloudTitle'), body: t(locale, 'admin.profileCloudBody') },
    local_hybrid: { title: t(locale, 'admin.profileHybridTitle'), body: t(locale, 'admin.profileHybridBody') },
    local: { title: t(locale, 'admin.profileLocalTitle'), body: t(locale, 'admin.profileLocalBody') },
  };
  return (
    <section className="admin-configuration">
      <form className="admin-panel" onSubmit={(event) => void onProfile(event)}>
        <div className="admin-panel-head">
          <h2>{t(locale, 'admin.executionProfile')}</h2>
        </div>
        <div className="admin-panel-body">
          <p className="admin-help">{t(locale, 'admin.profileHelp')}</p>
          <div className="admin-choices" role="radiogroup" aria-label={t(locale, 'admin.activeProfile')} key={config.active_profile}>
            {config.profiles.map((profile) => (
              <label className="admin-choice" key={profile.value}>
                <input type="radio" name="profile" value={profile.value} defaultChecked={profile.value === config.active_profile} />
                <strong>{profileGuides[profile.value]?.title || profile.label}</strong>
                <span>{profileGuides[profile.value]?.body || profile.description}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="admin-form-actions">
          <button type="submit" className="admin-btn primary" disabled={busy}>
            {t(locale, 'admin.saveProfile')}
          </button>
        </div>
      </form>
      <form className="admin-panel" onSubmit={(event) => void onSecrets(event)}>
        <div className="admin-panel-head">
          <h2>{t(locale, 'admin.credentials')}</h2>
        </div>
        <div className="admin-panel-body">
          <p className="admin-help">{t(locale, 'admin.credentialsHelp')}</p>
          {config.secrets.map((secret) => (
            <div className="admin-field" key={secret.key}>
              <div className="admin-secret-head">
                <label className="admin-code" htmlFor={`secret-${secret.key}`}>{secret.key}</label>
                <span className={`admin-secret-state${secret.configured ? ' is-set' : ''}`}>
                  {secret.configured ? <CheckCircle2 aria-hidden="true" size={13} /> : null}
                  {secret.configured
                    ? t(locale, 'admin.configured', { source: secret.source, masked: secret.masked || '' })
                    : t(locale, 'admin.notConfigured')}
                </span>
              </div>
              <input
                id={`secret-${secret.key}`}
                name={secret.key}
                type="password"
                autoComplete="off"
                placeholder={secret.configured ? t(locale, 'admin.replaceValue') : t(locale, 'admin.enterValue')}
                aria-describedby={`secret-${secret.key}-hint`}
              />
              <p className="admin-help" id={`secret-${secret.key}-hint`}>{t(locale, `admin.secretHint.${secret.key}`)}</p>
            </div>
          ))}
        </div>
        <div className="admin-form-actions">
          <button type="submit" className="admin-btn primary" disabled={busy}>
            {t(locale, 'admin.saveCredentials')}
          </button>
        </div>
      </form>
    </section>
  );
}
