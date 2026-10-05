// Relations between texts (supersession edges): found in the PDFs and used at once, checked here.
// The admin sees the page that states each relation, with the sentence highlighted, and approves,
// rejects or adds one. Decisions are kept apart from the index, so a re-upload keeps them.
import { useEffect, useMemo, useState, type FormEvent } from 'react';
import { ExternalLink, FileText, Plus } from 'lucide-react';
import { addRelation, decideRelation, listRelations, type Relation, type RelationStatus } from '../../api/admin';
import { sourcePageImageUrl } from '../../api/chat';
import { t, type UiLocale } from '../../uiLocale';
import { formatWhen } from './shared';
import { BlockSkeleton } from './Skeletons';

const FILTERS: Array<RelationStatus | 'all'> = ['review', 'approved', 'rejected', 'added', 'all'];
const STATUS_TONE: Record<RelationStatus, string> = { review: 'pending', approved: 'approved', rejected: 'rejected', added: 'enriching' };

// "cir:2017:9" -> "Cir 2017-09"
function instrumentLabel(id: string) {
  const [kind, year, number] = id.split(':');
  return `${kind === 'note' ? 'Note' : 'Cir'} ${year}-${String(number).padStart(2, '0')}`;
}

function pdfAtPage(file: string, page = 1) {
  return `/api/sources/${encodeURIComponent(file)}#page=${page}`;
}

export function RelationsPage({ locale }: { locale: UiLocale }) {
  const [items, setItems] = useState<Relation[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<RelationStatus | 'all'>('review');
  const [query, setQuery] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  async function reload() {
    try {
      const data = await listRelations();
      setItems(data.items);
      setSources(data.sources);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { void reload(); }, []);

  const counts = useMemo(() => {
    const result: Record<string, number> = { all: items.length };
    for (const item of items) result[item.status] = (result[item.status] || 0) + 1;
    return result;
  }, [items]);

  const needle = query.trim().toLowerCase();
  const rows = items.filter((item) => (filter === 'all' || item.status === filter) && (!needle
    || `${instrumentLabel(item.source_instrument)} ${instrumentLabel(item.target_instrument)} ${item.source_file}`.toLowerCase().includes(needle)));

  async function decide(item: Relation, status: RelationStatus) {
    setBusyId(item.id);
    try {
      await decideRelation(item.id, status === 'added' ? 'review' : status);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <section className="admin-panel admin-relations">
      <div className="admin-panel-head">
        <h2>{t(locale, 'admin.relations')}<span className="admin-count">{items.length}</span></h2>
      </div>
      <p className="admin-help">{t(locale, 'admin.relationsHelp')}</p>
      <div className="admin-tabs" role="tablist" aria-label={t(locale, 'admin.relations')}>
        {FILTERS.map((value) => (
          <button key={value} type="button" role="tab" aria-selected={filter === value} onClick={() => setFilter(value)}>
            {t(locale, value === 'all' ? 'admin.relAll' : `admin.relStatus.${value}`)}
            <em>{counts[value] || 0}</em>
          </button>
        ))}
      </div>
      <div className="admin-search">
        <input type="search" value={query} placeholder={t(locale, 'admin.relSearch')} aria-label={t(locale, 'admin.relSearch')}
          onChange={(event) => setQuery(event.target.value)} />
      </div>
      {error ? <p className="admin-error" role="alert">{error}</p> : null}
      {loading ? (
        <BlockSkeleton label={t(locale, 'admin.loading')} />
      ) : !rows.length ? (
        <p className="admin-empty">{t(locale, 'admin.relEmpty')}</p>
      ) : (
        <ul className="admin-relation-list">
          {rows.map((item) => (
            <li key={item.id} className="admin-relation">
              <div className="admin-relation-main">
                <div>
                  <strong><FileText aria-hidden="true" size={15} /> {instrumentLabel(item.source_instrument)} · p.{item.source_page}</strong>
                  {' '}{t(locale, `admin.relAction.${item.action}`)}{' '}
                  <strong>{instrumentLabel(item.target_instrument)}</strong>
                  {item.target_article ? ` (${t(locale, 'admin.relArticle', { article: item.target_article })})` : ''}
                  <blockquote dir="auto">“{item.quote}”</blockquote>
                  <span className="admin-muted">
                    {item.source_file}
                    {item.decided_by ? ` · ${t(locale, 'admin.relDecidedBy', { who: item.decided_by })}, ${formatWhen(item.decided_at || '', locale)}` : ''}
                  </span>
                </div>
                <span className={`admin-status ${STATUS_TONE[item.status]}`}><i aria-hidden="true" />{t(locale, `admin.relStatus.${item.status}`)}</span>
              </div>
              <div className="admin-actions">
                <button type="button" className="admin-btn small" aria-expanded={openId === item.id}
                  onClick={() => setOpenId(openId === item.id ? null : item.id)}>
                  {t(locale, openId === item.id ? 'admin.relHidePage' : 'admin.relShowPage')}
                </button>
                {item.status === 'review' ? (
                  <>
                    <button type="button" className="admin-btn small primary" disabled={busyId === item.id} onClick={() => void decide(item, 'approved')}>{t(locale, 'admin.relApprove')}</button>
                    <button type="button" className="admin-btn small danger" disabled={busyId === item.id} onClick={() => void decide(item, 'rejected')}>{t(locale, 'admin.relReject')}</button>
                  </>
                ) : (
                  <button type="button" className="admin-btn small" disabled={busyId === item.id} onClick={() => void decide(item, item.status === 'added' ? 'added' : 'review')}>
                    {t(locale, item.status === 'approved' ? 'admin.relUndo' : item.status === 'rejected' ? 'admin.relRestore' : 'admin.relRemove')}
                  </button>
                )}
              </div>
              {openId === item.id ? (
                <div className="admin-relation-page">
                  <img src={sourcePageImageUrl(item.source_file, item.source_page, item.quote, 1.6)}
                    alt={`${item.source_file}, page ${item.source_page}`} />
                  <p>
                    <a href={pdfAtPage(item.source_file, item.source_page)} target="_blank" rel="noopener noreferrer">
                      <ExternalLink aria-hidden="true" size={14} /> {t(locale, 'admin.relOpenPdf')}
                    </a>
                    {item.target_file ? (
                      <a href={pdfAtPage(item.target_file)} target="_blank" rel="noopener noreferrer">
                        <ExternalLink aria-hidden="true" size={14} /> {t(locale, 'admin.relOpenOld')}
                      </a>
                    ) : null}
                  </p>
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}
      <AddRelationForm locale={locale} sources={sources} onAdded={reload} />
    </section>
  );
}

function AddRelationForm({ locale, sources, onAdded }: { locale: UiLocale; sources: string[]; onAdded: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setBusy(true);
    setError(null);
    try {
      await addRelation({
        source_file: String(data.get('source_file') || ''),
        source_page: Number(data.get('source_page') || 0),
        action: String(data.get('action') || 'REPLACE'),
        target: String(data.get('target') || ''),
        target_article: String(data.get('target_article') || '') || null,
      });
      form.reset();
      await onAdded();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="admin-relation-add" onSubmit={(event) => void submit(event)}>
      <h3><Plus aria-hidden="true" size={16} /> {t(locale, 'admin.relAddTitle')}</h3>
      <p className="admin-help">{t(locale, 'admin.relAddHelp')}</p>
      <div className="admin-relation-fields">
        <label>{t(locale, 'admin.relNewFile')}
          <input name="source_file" list="admin-relation-sources" required placeholder="Cir_2024_01_fr.pdf" />
          <datalist id="admin-relation-sources">{sources.map((name) => <option key={name} value={name} />)}</datalist>
        </label>
        <label>{t(locale, 'admin.relPage')}<input name="source_page" type="number" min={1} required /></label>
        <label>{t(locale, 'admin.relType')}
          <select name="action" defaultValue="REPLACE">
            {['REPLACE', 'ABROGATE', 'MODIFY'].map((value) => <option key={value} value={value}>{t(locale, `admin.relAction.${value}`)}</option>)}
          </select>
        </label>
        <label>{t(locale, 'admin.relOldText')}<input name="target" required placeholder="Cir 2021-01" /></label>
        <label>{t(locale, 'admin.relArticleOptional')}<input name="target_article" /></label>
      </div>
      {error ? <p className="admin-error" role="alert">{error}</p> : null}
      <div className="admin-form-actions">
        <button type="submit" className="admin-btn primary" disabled={busy}>{t(locale, 'admin.relAdd')}</button>
      </div>
    </form>
  );
}
