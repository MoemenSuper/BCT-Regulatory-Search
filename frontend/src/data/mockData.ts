import type { EvidencePassage, HistoryGroup, ResearchNoteData } from '../types/ui';

export const SAMPLE_QUERY =
  "Quelles sont les conditions et procédures de retrait d'agrément d'un établissement de crédit ?";

export const historyGroups: HistoryGroup[] = [
  {
    label: "AUJOURD'HUI",
    items: [
      { id: 'h1', title: "Retrait d'agréments bancaires", time: '10:24' },
      { id: 'h2', title: 'Ratio de solvabilité Bâle III', time: '09:11' },
    ],
  },
  {
    label: 'HIER',
    items: [
      { id: 'h3', title: 'Gouvernance des établissements', time: '16:42' },
      { id: 'h4', title: 'Lutte contre le blanchiment', time: '11:05' },
    ],
  },
  {
    label: '7 MAI 2026',
    items: [
      { id: 'h5', title: 'Publication au JORT des sanctions', time: '14:18' },
      { id: 'h6', title: "Conditions d'octroi d'agrément", time: '10:03' },
    ],
  },
  {
    label: '5 MAI 2026',
    items: [
      { id: 'h7', title: 'Fonds propres prudentiels', time: '15:27' },
      { id: 'h8', title: 'Contrôle sur place — mandats', time: '09:40' },
    ],
  },
  {
    label: '30 AVR. 2026',
    items: [
      { id: 'h9', title: 'Liquidité et réserves obligatoires', time: '17:02' },
      { id: 'h10', title: 'Externalisation des activités', time: '12:55' },
    ],
  },
  {
    label: '28 AVR. 2026',
    items: [
      { id: 'h11', title: 'Protection des déposants', time: '18:21' },
      { id: 'h12', title: 'Dispositif de contrôle interne', time: '08:47' },
    ],
  },
];

export const researchNote: ResearchNoteData = {
  title: 'Note de recherche réglementaire',
  date: '7 mai 2026',
  analyst: 'Rechercheur BCT',
  reference: 'RR-2026-05-007',
  warningTitle: 'Statut non vérifié',
  warningText:
    'La validité actuelle des textes et leur application doivent être confirmées par un expert humain.',
  question: SAMPLE_QUERY,
  synthesis: [
    "Le retrait d'agrément d'un établissement de crédit constitue une mesure exceptionnelle encadrée par la réglementation bancaire tunisienne. Il intervient lorsque l'établissement ne remplit plus les conditions d'agrément, ou lorsqu'il a gravement et/ou de façon répétée manqué à ses obligations légales ou réglementaires. [1] [2]",
    "La procédure comporte en principe une phase contradictoire préalable : information de l'établissement, délai pour présenter des observations, et le cas échéant audition des dirigeants. La décision de retrait est prise par l'autorité compétente et produit des effets immédiats sur l'exercice de l'activité bancaire. [2] [3]",
    "La publicité de la décision (notamment au Journal Officiel de la République Tunisienne) vise à assurer l'opposabilité aux tiers et la protection des déposants et créanciers. Les effets portent sur la cessation des opérations de réception de dépôts et, selon les cas, sur les modalités de liquidation ou de transfert d'activités. [3]",
  ],
  keyFindings: [
    {
      title: 'Motifs de retrait',
      text: "Non-respect durable des conditions d'agrément ; manquements graves ou répétés aux obligations légales et réglementaires ; situations mettant en péril la stabilité ou les intérêts des déposants. [1]",
    },
    {
      title: 'Procédure préalable',
      text: "Notification des griefs, délai d'observation, possibilité d'audition, et constitution d'un dossier administratif avant décision. [2]",
    },
    {
      title: 'Décision',
      text: "Décision motivée de l'autorité compétente, précisant le fondement juridique, la date d'effet et les mesures conservatoires associées. [2] [3]",
    },
    {
      title: 'Publication',
      text: 'Publication au JORT et communication aux établissements concernés afin de garantir la publicité et la sécurité juridique. [3]',
    },
    {
      title: 'Effets',
      text: "Cessation de l'activité agréée, protection des déposants, et organisation éventuelle de la liquidation ou du transfert des engagements. [1] [3]",
    },
  ],
  sources: [
    {
      id: 1,
      citation:
        "Circulaire aux banques n°2026-01 du 05 janvier 2026 — Article 12 (conditions et motifs de retrait d'agrément).",
    },
    {
      id: 2,
      citation:
        "Loi n°2016-48 du 11 juillet 2016, relative aux banques et aux établissements financiers — dispositions sur l'agrément et son retrait.",
    },
    {
      id: 3,
      citation:
        'Circulaire aux banques n°2024-07 relative aux mesures de publicité, aux effets de la décision et à la protection des déposants.',
    },
  ],
};

export const evidencePassage: EvidencePassage = {
  quote:
    "lorsqu'il a gravement et/ou de façon répétée manqué à ses obligations légales ou réglementaires.",
  sourceLabel:
    'Source : Article 12, Circulaire aux banques n°2026-01 du 05 janvier 2026, Page 3.',
  filename: 'Circulaire_aux_banques_n°2026-01.pdf',
  page: 3,
  totalPages: 8,
};

export const INITIAL_SELECTED_HISTORY_ID = 'h1';

/** Build a live note from the FastAPI /chat response while keeping the mock shell. */
export function noteFromChatAnswer(
  question: string,
  answer: string,
  sources: { file: string; page: number | string | null; score: number }[],
): ResearchNoteData {
  const paragraphs = answer
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);

  return {
    ...researchNote,
    question,
    date: new Date().toLocaleDateString('fr-FR', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    }),
    synthesis: paragraphs.length > 0 ? paragraphs : [answer],
    keyFindings: [],
    sources: sources.map((source, index) => ({
      id: index + 1,
      citation: `${source.file}${source.page != null ? ` — page ${source.page}` : ''} (score ${source.score})`,
    })),
  };
}

export function evidenceFromChatSources(
  sources: { file: string; page: number | string | null; score: number }[],
  answer: string,
): EvidencePassage {
  const first = sources[0];
  if (!first) {
    return {
      ...evidencePassage,
      quote: answer.slice(0, 180) + (answer.length > 180 ? '…' : ''),
      sourceLabel: 'Source : réponse générée — aucun extrait documentaire renvoyé.',
      filename: 'Réponse générée',
      page: 1,
      totalPages: 1,
    };
  }

  const pageNum = typeof first.page === 'number' ? first.page : Number(first.page) || 1;

  return {
    ...evidencePassage,
    filename: first.file || evidencePassage.filename,
    page: pageNum,
    sourceLabel: `Source : ${first.file}${first.page != null ? `, page ${first.page}` : ''}.`,
    quote: answer.slice(0, 220) + (answer.length > 220 ? '…' : ''),
  };
}
