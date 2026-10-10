// Runtime profile and provider API keys.
import { type FormEvent } from 'react';
import { CheckCircle2 } from 'lucide-react';
import { type AdminConfig } from '../../api/admin';
import { t, type UiLocale } from '../../uiLocale';
import { BlockSkeleton } from './Skeletons';

export function ConfigurationPage({ config, loading, busy, locale, onProfile, onBuildCloudIndex, onSecrets, onClearSecret }: {
  config: AdminConfig | null;
  loading: boolean;
  busy: boolean;
  locale: UiLocale;
  onProfile: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onBuildCloudIndex: () => Promise<void>;
  onSecrets: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onClearSecret: (key: string) => Promise<void>;
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
  const cloud = config.cloud_index;
  // The cloud profile can only be chosen once its index exists (the server refuses it otherwise).
  const cloudLocked = !cloud.ready && config.active_profile !== 'cloud';
  const local = config.local_llm;
  const localProblem: Record<string, string> = {
    unreachable: 'admin.localLlmUnreachable', missing: 'admin.localLlmMissing',
    remote_not_allowed: 'admin.localLlmRemote', cloud_model: 'admin.localLlmCloud',
  };
  const localMessage = t(locale, local.ready ? 'admin.localLlmReady' : localProblem[local.problem || 'unreachable'], {
    model: local.model, installed: local.installed.join(', ') || t(locale, 'admin.localLlmNone'),
  });
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
                <input type="radio" name="profile" value={profile.value} defaultChecked={profile.value === config.active_profile}
                  disabled={(profile.value === 'cloud' && cloudLocked)
                    || (profile.value === 'local' && !local.ready && config.active_profile !== 'local')} />
                <strong>{profileGuides[profile.value]?.title || profile.label}</strong>
                <span>{profileGuides[profile.value]?.body || profile.description}</span>
                {profile.value === 'cloud' ? (
                  <span className="admin-help">
                    {cloud.building ? t(locale, 'admin.cloudIndexBuilding')
                      : cloud.ready ? t(locale, 'admin.cloudIndexReady')
                      : t(locale, 'admin.cloudIndexMissing')}
                    {cloud.error ? <> {t(locale, 'admin.cloudIndexFailed', { error: cloud.error })}</> : null}
                    {!cloud.ready && !cloud.building ? (
                      <button type="button" className="admin-btn" disabled={busy} onClick={() => void onBuildCloudIndex()}>
                        {t(locale, 'admin.cloudIndexBuild')}
                      </button>
                    ) : null}
                  </span>
                ) : null}
                {profile.value === 'local' ? <span className="admin-help">{localMessage}</span> : null}
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
          <p className="admin-help"><strong>{t(locale, 'admin.answeringWith', { provider: t(locale, `admin.choice.${config.answer_provider}`) })}</strong></p>
          {config.secrets.map((secret) => (
            <div className="admin-field" key={secret.key}>
              <div className="admin-secret-head">
                <label className="admin-code" htmlFor={`secret-${secret.key}`}>{secret.key}</label>
                <span className={`admin-secret-state${secret.configured ? ' is-set' : ''}`}>
                  {secret.configured ? <CheckCircle2 aria-hidden="true" size={13} /> : null}
                  {secret.configured
                    ? t(locale, 'admin.configured', {
                        source: t(locale, secret.source === 'store' ? 'admin.sourceStore' : 'admin.sourceEnvironment'),
                        masked: (secret.secret ? secret.masked : secret.value) || '',
                      })
                    : t(locale, 'admin.notConfigured')}
                </span>
              </div>
              <div className="admin-secret-input">
                {secret.choices ? (
                  // An empty choice changes nothing; "Remove" goes back to the .env value or the default.
                  <select id={`secret-${secret.key}`} name={secret.key} defaultValue="" aria-describedby={`secret-${secret.key}-hint`}>
                    <option value="">{t(locale, 'admin.providerKeep')}</option>
                    {secret.choices.map((value) => <option key={value} value={value}>{t(locale, `admin.choice.${value}`)}</option>)}
                  </select>
                ) : (
                  <input
                    id={`secret-${secret.key}`}
                    name={secret.key}
                    type={secret.secret ? 'password' : 'text'}
                    autoComplete="off"
                    placeholder={secret.configured ? t(locale, 'admin.replaceValue') : t(locale, 'admin.enterValue')}
                    aria-describedby={`secret-${secret.key}-hint`}
                  />
                )}
                {/* Only a value saved here can be removed; .env values are edited in the file. */}
                {secret.source === 'store' ? (
                  <button type="button" className="admin-btn" disabled={busy} onClick={() => void onClearSecret(secret.key)}>
                    {t(locale, 'admin.clearSecret')}
                  </button>
                ) : null}
              </div>
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
