import { useEffect, useId, useRef, useState } from 'react';
import { t, type UiLocale } from '../uiLocale';

type NoticeId = 'mentions' | 'avertissement' | 'confidentialite';

const NOTICE_TITLES: Record<NoticeId, string> = {
  mentions: 'chat.legal',
  avertissement: 'chat.warning',
  confidentialite: 'chat.privacy',
};

// The paragraphs of each notice, per interface language.
const NOTICES: Record<UiLocale, Record<NoticeId, string[]>> = {
  fr: {
    mentions: [
      'Espace Recherche Réglementaire est un système d’aide à la consultation du corpus des circulaires et notes de la Banque Centrale de Tunisie. Conception et développement : Moemen Ouerghui.',
      'Les textes réglementaires, dénominations et signes distinctifs de la Banque Centrale de Tunisie restent la propriété de cette institution. Leur usage dans l’interface sert à identifier le corpus consulté.',
      'L’utilisateur demeure responsable de l’interprétation des textes et des suites qu’il leur donne. Les résultats affichés n’engagent pas l’éditeur du système.',
      '© 2026 Moemen Ouerghui. Tous droits réservés sur le logiciel et l’interface, à l’exception des textes réglementaires et des signes distinctifs de la Banque Centrale de Tunisie.',
    ],
    avertissement: [
      'Cet outil assiste la recherche et la localisation de sources. Les réponses sont produites par un modèle d’intelligence artificielle à partir de passages extraits du corpus indexé.',
      'Elles ne valent ni avis juridique, ni instruction interne, ni interprétation officielle de la Banque Centrale de Tunisie. Seul le document source fait foi.',
      'Toute information restituée (citation, pagination, articulation entre textes) doit être vérifiée sur le PDF original et, le cas échéant, soumise aux services compétents avant usage opérationnel.',
    ],
    confidentialite: [
      'Les conversations sont conservées localement afin de reprendre une recherche. Aucune donnée n’est traitée à des fins commerciales.',
      'L’utilisateur s’abstient de saisir des données à caractère personnel, des secrets d’affaires ou tout élément soumis à une obligation de confidentialité, hors environnement dûment autorisé par son institution.',
      'En cas de déploiement interne, la conservation, l’accès et l’effacement des traces relèvent de la politique de sécurité et de protection des données de l’organisme utilisateur.',
    ],
  },
  ar: {
    mentions: [
      '«البحث في النصوص الترتيبية» أداة تساعد على البحث في منشورات البنك المركزي التونسي ومذكّراته. التصميم والتطوير: Moemen Ouerghui.',
      'النصوص الترتيبية واسم البنك المركزي التونسي وشعاره ملك للبنك. تُستعمل في هذه الواجهة فقط للتعريف بالوثائق التي يتم البحث فيها.',
      'المستخدم مسؤول عن فهمه للنصوص وعمّا يقوم به بناءً عليها. النتائج المعروضة لا تُلزم مطوّر الأداة.',
      '© 2026 Moemen Ouerghui. جميع الحقوق محفوظة على البرنامج والواجهة، باستثناء النصوص الترتيبية واسم البنك المركزي التونسي وشعاره.',
    ],
    avertissement: [
      'تساعد هذه الأداة على البحث عن المصادر وإيجادها. يكتب الإجاباتِ نموذجُ ذكاء اصطناعي اعتمادًا على مقاطع مأخوذة من الوثائق.',
      'هذه الإجابات ليست رأيًا قانونيًا ولا تعليمات داخلية ولا تفسيرًا رسميًا من البنك المركزي التونسي. الوثيقة الأصلية وحدها هي المرجع.',
      'يجب التثبّت من كل معلومة (اقتباس، رقم صفحة، علاقة بين نصّين) في ملف PDF الأصلي، وعرضها عند الحاجة على المصالح المختصة قبل استعمالها في العمل.',
    ],
    confidentialite: [
      'تُحفظ المحادثات محليًا حتى تتمكّن من الرجوع إلى بحث سابق. لا تُستعمل أي معطيات لأغراض تجارية.',
      'لا تكتب معطيات شخصية أو أسرارًا مهنية أو أي معلومة سرّية، إلا في بيئة تسمح بها مؤسستك.',
      'عند استعمال الأداة داخل مؤسسة، تخضع مدة حفظ السجلات والنفاذ إليها وحذفها لسياسة الأمن وحماية المعطيات في تلك المؤسسة.',
    ],
  },
  en: {
    mentions: [
      'Regulatory Search is a tool that helps you search the circulars and notes of the Central Bank of Tunisia. Design and development: Moemen Ouerghui.',
      'The regulatory texts, name and logo of the Central Bank of Tunisia belong to the Bank. They are used in this interface only to identify the documents searched.',
      'Users are responsible for how they read the texts and for what they do with them. The results shown do not bind the developer of this tool.',
      '© 2026 Moemen Ouerghui. All rights reserved on the software and interface, except the regulatory texts, name and logo of the Central Bank of Tunisia.',
    ],
    avertissement: [
      'This tool helps you find sources. Answers are written by an artificial intelligence model from passages taken from the indexed documents.',
      'They are not legal advice, internal instructions or an official interpretation by the Central Bank of Tunisia. Only the source document is authoritative.',
      'Check every piece of information (quote, page number, link between two texts) in the original PDF and, when needed, with the relevant department before using it in your work.',
    ],
    confidentialite: [
      'Conversations are stored locally so you can come back to a search. No data is used for commercial purposes.',
      'Do not enter personal data, business secrets or any confidential information, unless your institution allows it in this environment.',
      'When the tool is used inside an institution, how long logs are kept, who can see them and how they are deleted follow that institution’s security and data protection policy.',
    ],
  },
};

export function LegalFooter({ locale }: { locale: UiLocale }) {
  const [openNotice, setOpenNotice] = useState<NoticeId | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (openNotice) {
      if (!dialog.open) dialog.showModal();
    } else if (dialog.open) {
      dialog.close();
    }
  }, [openNotice]);

  return (
    <>
      <footer className="legal-footer">
        <p className="legal-footer-credit">
          {t(locale, 'chat.builtBy')} <strong>Moemen Ouerghui</strong> · 2026
        </p>
        <p className="legal-footer-disclaimer">{t(locale, 'chat.disclaimer')}</p>
        <nav className="legal-footer-nav" aria-label={t(locale, 'chat.legal')}>
          {(Object.keys(NOTICE_TITLES) as NoticeId[]).map((id) => (
            <button key={id} type="button" className="legal-footer-link" onClick={() => setOpenNotice(id)}>
              {t(locale, NOTICE_TITLES[id])}
            </button>
          ))}
        </nav>
      </footer>

      <dialog
        ref={dialogRef}
        className="legal-dialog"
        aria-labelledby={titleId}
        onClose={() => setOpenNotice(null)}
        onClick={(event) => {
          if (event.target === dialogRef.current) setOpenNotice(null);
        }}
      >
        {openNotice ? (
          <div className="legal-dialog-panel">
            <h2 id={titleId} className="legal-dialog-title">
              {t(locale, NOTICE_TITLES[openNotice])}
            </h2>
            {NOTICES[locale][openNotice].map((paragraph) => (
              <p key={paragraph.slice(0, 40)} className="legal-dialog-body">
                {paragraph}
              </p>
            ))}
            <button type="button" className="legal-dialog-close" onClick={() => setOpenNotice(null)}>
              {t(locale, 'chat.close')}
            </button>
          </div>
        ) : null}
      </dialog>
    </>
  );
}
