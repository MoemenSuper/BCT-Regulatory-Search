import { t, type UiLocale } from '../uiLocale';

export function LanguageSwitcher({ locale, onChange, className = '' }: { locale: UiLocale; onChange: (locale: UiLocale) => void; className?: string }) {
  const options: Array<[UiLocale, string]> = [['fr', 'FR'], ['ar', 'ع'], ['en', 'EN']];
  return <div className={`language-switcher ${className}`.trim()} role="group" aria-label={t(locale, 'language.label')}>
    {options.map(([value, label]) => <button key={value} type="button" aria-pressed={locale === value} onClick={() => onChange(value)} title={t(locale, `language.${value}`)}>{label}</button>)}
  </div>;
}
