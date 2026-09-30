// Runtime profile and provider API keys.
import { type FormEvent } from 'react';
import { Activity, CheckCircle2, KeyRound, ShieldCheck } from 'lucide-react';
import { type AdminConfig } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { ConfigurationSkeleton } from './Skeletons';

export function ConfigurationPage({ config, loading, busy, locale, onProfile, onSecrets }: {
  config: AdminConfig | null;
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  onProfile: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onSecrets: (event: FormEvent<HTMLFormElement>) => Promise<void>;
}) {
  if (loading || !config) return <ConfigurationSkeleton label={t(locale, 'admin.loading')} />;
  const profileGuides = [
    { value: 'cloud', title: t(locale, 'admin.profileCloudTitle'), body: t(locale, 'admin.profileCloudBody') },
    { value: 'local_hybrid', title: t(locale, 'admin.profileHybridTitle'), body: t(locale, 'admin.profileHybridBody') },
    { value: 'local', title: t(locale, 'admin.profileLocalTitle'), body: t(locale, 'admin.profileLocalBody') },
  ];
  return (
    <section className="admin-configuration-layout">
      <form className="admin-form admin-panel" onSubmit={(event) => void onProfile(event)}>
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.runtimeControl')}</p>
            <h2>{t(locale, 'admin.executionProfile')}</h2>
          </div>
          <Activity aria-hidden="true" size={22} />
        </div>
        <p className="admin-help">{t(locale, 'admin.profileHelp')}</p>
        <label>
          {t(locale, 'admin.activeProfile')}
          <select name="profile" defaultValue={config.active_profile} key={config.active_profile}>
            {config.profiles.map((profile) => (
              <option key={profile.value} value={profile.value}>{profile.label}</option>
            ))}
          </select>
        </label>
        <button type="submit" className="admin-primary-button" disabled={busy}>
          <CheckCircle2 aria-hidden="true" size={18} />
          {t(locale, 'admin.saveProfile')}
        </button>
        <div className="admin-profile-guide" aria-label={t(locale, 'admin.profileGuide')}>
          <p>{t(locale, 'admin.profileGuide')}</p>
          <ul>
            {profileGuides.map((guide) => (
              <li key={guide.value} className={config.active_profile === guide.value ? 'is-active' : undefined}>
                <strong>{guide.title}</strong>
                <span>{guide.body}</span>
              </li>
            ))}
          </ul>
        </div>
      </form>
      <form className="admin-form admin-panel" onSubmit={(event) => void onSecrets(event)}>
        <div className="admin-panel-heading">
          <div>
            <p>{t(locale, 'admin.providerAccess')}</p>
            <h2>{t(locale, 'admin.credentials')}</h2>
          </div>
          <KeyRound aria-hidden="true" size={22} />
        </div>
        <p className="admin-help">{t(locale, 'admin.credentialsHelp')}</p>
        {config.secrets.map((secret) => (
          <label key={secret.key}>
            {secret.key}
            <span className="admin-masked">
              {secret.configured
                ? t(locale, 'admin.configured', { source: secret.source, masked: secret.masked || '' })
                : t(locale, 'admin.notConfigured')}
            </span>
            <input
              name={secret.key}
              type="password"
              autoComplete="off"
              placeholder={secret.configured ? t(locale, 'admin.replaceValue') : t(locale, 'admin.enterValue')}
            />
            <span className="admin-secret-hint">{t(locale, `admin.secretHint.${secret.key}`)}</span>
          </label>
        ))}
        <button type="submit" className="admin-primary-button" disabled={busy}>
          <ShieldCheck aria-hidden="true" size={18} />
          {t(locale, 'admin.saveCredentials')}
        </button>
      </form>
    </section>
  );
}
