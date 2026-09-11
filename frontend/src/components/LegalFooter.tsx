import { useEffect, useId, useRef, useState } from 'react';

type NoticeId = 'mentions' | 'avertissement' | 'confidentialite';

const NOTICES: Record<NoticeId, { title: string; body: string[] }> = {
  mentions: {
    title: 'Mentions légales',
    body: [
      'Espace Recherche Réglementaire est un système d’aide à la consultation du corpus des circulaires et notes de la Banque Centrale de Tunisie. Conception et développement : Moemen Ouerghui.',
      'Les textes réglementaires, dénominations et signes distinctifs de la Banque Centrale de Tunisie restent la propriété de cette institution. Leur usage dans l’interface sert à identifier le corpus consulté.',
      'L’utilisateur demeure responsable de l’interprétation des textes et des suites qu’il leur donne. Les résultats affichés n’engagent pas l’éditeur du système.',
      '© 2026 Moemen Ouerghui. Tous droits réservés sur le logiciel et l’interface, à l’exception des textes réglementaires et des signes distinctifs de la Banque Centrale de Tunisie.',
    ],
  },
  avertissement: {
    title: 'Avertissement',
    body: [
      'Cet outil assiste la recherche et la localisation de sources. Les réponses sont produites par un modèle d’intelligence artificielle à partir de passages extraits du corpus indexé.',
      'Elles ne valent ni avis juridique, ni instruction interne, ni interprétation officielle de la Banque Centrale de Tunisie. Seul le document source fait foi.',
      'Toute information restituée (citation, pagination, articulation entre textes) doit être vérifiée sur le PDF original et, le cas échéant, soumise aux services compétents avant usage opérationnel.',
    ],
  },
  confidentialite: {
    title: 'Confidentialité',
    body: [
      'Les conversations sont conservées localement afin de reprendre une recherche. Aucune donnée n’est traitée à des fins commerciales.',
      'L’utilisateur s’abstient de saisir des données à caractère personnel, des secrets d’affaires ou tout élément soumis à une obligation de confidentialité, hors environnement dûment autorisé par son institution.',
      'En cas de déploiement interne, la conservation, l’accès et l’effacement des traces relèvent de la politique de sécurité et de protection des données de l’organisme utilisateur.',
    ],
  },
};

export function LegalFooter() {
  const [openNotice, setOpenNotice] = useState<NoticeId | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const notice = openNotice ? NOTICES[openNotice] : null;

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
          Conçu et développé par <strong>Moemen Ouerghui</strong> · 2026
        </p>
        <p className="legal-footer-disclaimer">
          Aide à la recherche documentaire. Les réponses doivent être vérifiées
          sur la source. Ne constitue pas un avis juridique.
        </p>
        <nav className="legal-footer-nav" aria-label="Mentions légales">
          <button type="button" className="legal-footer-link" onClick={() => setOpenNotice('mentions')}>
            Mentions légales
          </button>
          <button type="button" className="legal-footer-link" onClick={() => setOpenNotice('avertissement')}>
            Avertissement
          </button>
          <button type="button" className="legal-footer-link" onClick={() => setOpenNotice('confidentialite')}>
            Confidentialité
          </button>
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
        {notice ? (
          <div className="legal-dialog-panel">
            <h2 id={titleId} className="legal-dialog-title">
              {notice.title}
            </h2>
            {notice.body.map((paragraph) => (
              <p key={paragraph.slice(0, 40)} className="legal-dialog-body">
                {paragraph}
              </p>
            ))}
            <button type="button" className="legal-dialog-close" onClick={() => setOpenNotice(null)}>
              Fermer
            </button>
          </div>
        ) : null}
      </dialog>
    </>
  );
}
