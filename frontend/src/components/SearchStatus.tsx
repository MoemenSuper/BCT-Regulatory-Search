import { useEffect, useMemo, useState } from 'react';

type QueryLanguage = 'fr' | 'en' | 'ar';

const FR_WORDS = new Set([
  'le', 'la', 'les', 'des', 'un', 'une', 'du', 'de', 'et', 'ou', 'que', 'qui',
  'dans', 'pour', 'avec', 'sur', 'est', 'sont', 'cette', 'ces', 'aux', 'dont',
  'quelle', 'quelles', 'quel', 'quels', 'comment', 'pourquoi', 'selon', 'peut',
  'peuvent', 'faut', 'nous', 'vous', 'pas', 'plus', 'aussi', 'entre', 'contre',
  'parmi', 'apres', 'avant', 'circulaire', 'circulaires', 'banque', 'tunisie',
  'centrale', 'notes', 'article', 'articles', 'alinea', 'reglement', 'devise',
  'changes', 'credit', 'quoi', 'quand', 'donc', 'car', 'mais', 'etre', 'avoir',
  'faire', 'veuillez', 'merci', 'bonjour',
]);

const EN_WORDS = new Set([
  'the', 'is', 'are', 'was', 'were', 'what', 'how', 'does', 'do', 'did', 'can',
  'should', 'which', 'this', 'that', 'these', 'those', 'for', 'with', 'from',
  'about', 'under', 'circular', 'circulars', 'bank', 'tunisia', 'must', 'may',
  'whether', 'please', 'explain', 'describe', 'when', 'where', 'why', 'who',
  'not', 'and', 'of', 'in', 'on', 'to', 'by', 'if', 'any', 'all', 'its', 'their',
  'there', 'here', 'has', 'have', 'will', 'would', 'could',
]);

function detectQueryLanguage(text: string): QueryLanguage {
  const trimmed = text.trim();
  if (!trimmed) return 'fr';

  const arabic = (trimmed.match(/[\u0600-\u06FF]/g) || []).length;
  const latin = (trimmed.match(/[A-Za-z\u00C0-\u024F]/g) || []).length;
  if (arabic >= 4 && arabic >= latin) return 'ar';
  if (arabic >= 8) return 'ar';

  const frenchMarks = (trimmed.match(/[éèêëàâäùûüôöîïçœæÉÈÊËÀÂÙÛÔÎÏÇŒÆ]/g) || []).length;
  const normalized = trimmed
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
  const tokens = normalized.split(/[^a-z]+/).filter((token) => token.length > 1);

  let frScore = frenchMarks * 3;
  let enScore = 0;
  for (const token of tokens) {
    if (FR_WORDS.has(token)) frScore += 2;
    if (EN_WORDS.has(token)) enScore += 2;
  }

  if (enScore > frScore) return 'en';
  return 'fr';
}

const PHRASES: Record<QueryLanguage, string[]> = {
  fr: ['Recherche', 'Lecture des sources', 'Vérification des circulaires', 'Rédaction'],
  en: ['Searching', 'Reading sources', 'Checking circulars', 'Writing'],
  ar: ['جارٍ البحث', 'قراءة المصادر', 'التحقق من المناشير', 'الكتابة'],
};

const ANNOUNCE: Record<QueryLanguage, string> = {
  fr: 'Recherche en cours.',
  en: 'Search in progress.',
  ar: 'البحث جارٍ.',
};

function readPrefersReducedMotion() {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(readPrefersReducedMotion);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduced(media.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  return reduced;
}

interface SearchStatusProps {
  question: string;
}

export function SearchStatus({ question }: SearchStatusProps) {
  const language = useMemo(() => detectQueryLanguage(question), [question]);
  const phrases = PHRASES[language];
  const reducedMotion = usePrefersReducedMotion();
  const [index, setIndex] = useState(0);
  const [rotationKey, setRotationKey] = useState(`${language}:${question}`);
  const nextKey = `${language}:${question}`;
  if (rotationKey !== nextKey) {
    setRotationKey(nextKey);
    setIndex(0);
  }

  useEffect(() => {
    if (reducedMotion) return undefined;
    const timer = window.setInterval(() => {
      setIndex((current) => (current + 1) % phrases.length);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [phrases, reducedMotion, nextKey]);

  const phrase = phrases[reducedMotion ? 0 : index];

  return (
    <div
      className="search-status"
      role="status"
      lang={language}
      dir={language === 'ar' ? 'rtl' : 'ltr'}
    >
      <span className="visually-hidden">{ANNOUNCE[language]}</span>
      <span key={`${language}-${phrase}`} className="search-status-phrase" aria-hidden="true">
        {phrase}
      </span>
    </div>
  );
}
