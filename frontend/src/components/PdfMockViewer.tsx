interface PdfMockViewerProps {
  highlight: string;
  zoom: number;
}

export function PdfMockViewer({ highlight, zoom }: PdfMockViewerProps) {
  return (
    <div className="pdf-mock-stage">
      <div
        className="pdf-mock-page"
        style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}
      >
        <div className="pdf-mock-brand">
          <p className="pdf-ar">البنك المركزي التونسي</p>
          <p className="pdf-brand-fr">Banque Centrale de Tunisie</p>
        </div>

        <p className="pdf-circular-ref">Circulaire aux banques n°2026-01</p>
        <p className="pdf-date">Tunis, le 05 janvier 2026</p>

        <h4 className="pdf-section-title">
          I. CONDITIONS D&apos;AGRÉMENT ET RETRAIT D&apos;AGRÉMENT
        </h4>

        <p className="pdf-article-title">Article 12</p>
        <p className="pdf-body">
          L&apos;agrément d&apos;un établissement de crédit peut être retiré par la Banque Centrale
          de Tunisie lorsque l&apos;établissement ne remplit plus les conditions auxquelles était
          subordonné son agrément, ou
        </p>
        <p className="pdf-body">
          <mark className="pdf-highlight">
            - {highlight}
          </mark>
        </p>
        <p className="pdf-body">
          La décision de retrait est motivée. Elle est notifiée à l&apos;établissement concerné et
          publiée au Journal Officiel de la République Tunisienne. Elle précise, le cas échéant,
          les mesures destinées à préserver les intérêts des déposants et des créanciers.
        </p>

        <p className="pdf-article-title">Article 13</p>
        <p className="pdf-body">
          Préalablement à toute décision de retrait, l&apos;établissement est informé des griefs
          retenus à son encontre et dispose d&apos;un délai pour présenter ses observations écrites
          et, sur demande, être entendu.
        </p>

        <p className="pdf-page-number">3</p>
      </div>
    </div>
  );
}
