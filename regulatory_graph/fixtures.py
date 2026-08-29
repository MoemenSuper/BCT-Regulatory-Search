from datetime import date
from hashlib import sha256

from regulatory_graph.models import (
    ChangeEvent,
    EvidenceSpan,
    GraphPage,
    Instrument,
    InstrumentKind,
    LegalAction,
    Provision,
    ProvisionType,
    ProvisionVersion,
    RegulatoryGraphBundle,
    SourceEdition,
    SourceStatus,
    TargetScope,
    VerificationStatus,
    VersionStatus,
)


SOURCE_SHA256 = "E463736EBB98BE5DBC6E02635E1D635734B9CDF4ACFD598F4F1BA8351DC43078"


def circular_2016_03_fr_bundle() -> RegulatoryGraphBundle:
    source_uid = "BCT:CIRCULAR:2016:03"
    circular_91_24_uid = "BCT:CIRCULAR:1991:24"
    circular_93_08_uid = "BCT:CIRCULAR:1993:08"
    edition_uid = "edition:cir-2016-03-fr"

    article_4_text = (
        "Les banques et les établissements financiers doivent respecter en permanence "
        "un ratio de solvabilité qui ne peut pas être inférieur à 10 %, calculé par le "
        "rapport entre les fonds propres nets et les risques encourus, mesurés par la "
        "somme des agrégats suivants :\n"
        "- Le montant des risques de crédit pondérés, calculé en multipliant les éléments "
        "d’actif et du hors bilan nets par les quotités des risques prévues à l’article 6 "
        "de la présente circulaire ;\n"
        "- Le montant des risques opérationnels, déterminé en multipliant par 12,5 "
        "l’exigence en fonds propres au titre de ces risques calculée conformément aux "
        "dispositions des articles 13 (nouveau) et 14 (nouveau) de la présente circulaire.\n"
        "Les fonds propres nets de base tels que définis par l’article 5 ci-après ne "
        "peuvent être inférieurs en permanence à 7% de la somme des risques encourus "
        "mesurés conformément au premier alinéa du présent article."
    )
    annex_13_declaration = (
        "L’annexe 13 à la circulaire aux banques et établissements financiers "
        "n°93-08 du 30 juillet 1993 relative aux éléments de calcul du ratio de "
        "solvabilité est abrogée et remplacée par l’annexe de la présente circulaire."
    )
    article_16_text = (
        "L’incidence, sur la situation financière et le résultat, des événements "
        "survenant après la date de clôture doit être traitée, par les banques et les "
        "établissements financiers, conformément aux normes comptables en vigueur.\n"
        "Sans préjudice des dispositions de l’alinéa premier, les sommes recouvrées "
        "postérieurement à la date de clôture au titre des concours consentis à la "
        "clientèle ne doivent en aucun cas impacter la classification des actifs et les "
        "provisions constituées conformément aux dispositions de la présente circulaire."
    )
    evidence_quotes = {
        "evidence:cir-2016-03:p2:article-2": (
            "Article 2 : Les dispositions de l’article 4 de la circulaire n°91-24 "
            "du 17 décembre 1991 relative à la division, couverture des risques et suivi "
            "des engagements sont abrogées et remplacées par les dispositions suivantes :"
        ),
        "evidence:cir-2016-03:p3:article-5": annex_13_declaration,
        "evidence:cir-2016-03:p3:article-6": (
            "Article 6 : Les dispositions de l’article 16 de la circulaire n°91-24 "
            "du 17 décembre 1991 relative à la division, couverture des risques et suivi "
            "des engagements sont abrogées et remplacées par les dispositions suivantes :"
        ),
        "evidence:cir-2016-03:p4:effective-date": (
            "Article 7 : Sans préjudice des dates d’entrée en vigueur prévues à l’article "
            "premier, les dispositions de la présente circulaire entrent en vigueur à "
            "compter du 8 aout 2016 à l’exception des dispositions des articles 2 et 3 "
            "qui entrent en vigueur à partir du 30 décembre 2016."
        ),
    }

    instruments = (
        Instrument(
            uid=source_uid,
            authority="BCT",
            kind=InstrumentKind.CIRCULAR,
            year=2016,
            number="03",
            issue_date=date(2016, 7, 29),
            title="Circulaire aux banques et établissements financiers n°2016-03",
            corpus_present=True,
            canonical_citation="Circulaire BCT n°2016-03 du 29 juillet 2016",
            source_status=SourceStatus.LOCAL,
        ),
        Instrument(
            uid=circular_91_24_uid,
            authority="BCT",
            kind=InstrumentKind.CIRCULAR,
            year=1991,
            number="24",
            issue_date=date(1991, 12, 17),
            corpus_present=False,
            canonical_citation="Circulaire BCT n°91-24 du 17 décembre 1991",
            source_status=SourceStatus.EXTERNAL_STUB,
        ),
        Instrument(
            uid=circular_93_08_uid,
            authority="BCT",
            kind=InstrumentKind.CIRCULAR,
            year=1993,
            number="08",
            issue_date=date(1993, 7, 30),
            corpus_present=False,
            canonical_citation="Circulaire BCT n°93-08 du 30 juillet 1993",
            source_status=SourceStatus.EXTERNAL_STUB,
        ),
    )
    source_editions = (
        SourceEdition(
            uid=edition_uid,
            instrument_uid=source_uid,
            language="fr",
            filename="Cir_2016_03_fr.pdf",
            sha256=SOURCE_SHA256,
            extraction_status="native_text_with_visual_page_verification",
            page_count=14,
            is_scan=False,
        ),
    )
    pages = tuple(
        GraphPage(
            uid=f"page:cir-2016-03-fr:{page_number}",
            source_edition_uid=edition_uid,
            page_number=page_number,
            page_label=str(page_number),
        )
        for page_number in (2, 3, 4)
    )
    provisions = (
        Provision(
            uid=f"{circular_91_24_uid}:ARTICLE:4",
            instrument_uid=circular_91_24_uid,
            provision_type=ProvisionType.ARTICLE,
            label="Article 4",
            ordinal=4,
            canonical_path="article/4",
        ),
        Provision(
            uid=f"{circular_93_08_uid}:ANNEX:13",
            instrument_uid=circular_93_08_uid,
            provision_type=ProvisionType.ANNEX,
            label="Annexe 13",
            ordinal=13,
            canonical_path="annex/13",
        ),
        Provision(
            uid=f"{circular_91_24_uid}:ARTICLE:16",
            instrument_uid=circular_91_24_uid,
            provision_type=ProvisionType.ARTICLE,
            label="Article 16",
            ordinal=16,
            canonical_path="article/16",
        ),
    )
    versions = (
        _version(provisions[0].uid, "article-4", article_4_text, date(2016, 12, 30)),
        _version(provisions[1].uid, "annex-13", annex_13_declaration, date(2016, 8, 8), complete=False),
        _version(provisions[2].uid, "article-16", article_16_text, date(2016, 8, 8)),
    )
    evidence_spans = tuple(
        EvidenceSpan(
            uid=uid,
            source_edition_uid=edition_uid,
            quote=quote,
            page_number=int(uid.split(":p", 1)[1].split(":", 1)[0]),
            extraction_method="native_text_visually_verified",
            source_sha256=SOURCE_SHA256,
            extraction_artifact_hash=_hash(quote),
        )
        for uid, quote in evidence_quotes.items()
    )
    effective_evidence_uid = "evidence:cir-2016-03:p4:effective-date"
    events = (
        _replacement_event(
            "article-2",
            provisions[0].uid,
            versions[0].uid,
            date(2016, 12, 30),
            "Article 2 abroge et remplace l’article 4 de la circulaire n°91-24.",
            ("evidence:cir-2016-03:p2:article-2", effective_evidence_uid),
        ),
        _replacement_event(
            "article-5",
            provisions[1].uid,
            versions[1].uid,
            date(2016, 8, 8),
            annex_13_declaration,
            ("evidence:cir-2016-03:p3:article-5", effective_evidence_uid),
        ),
        _replacement_event(
            "article-6",
            provisions[2].uid,
            versions[2].uid,
            date(2016, 8, 8),
            "Article 6 abroge et remplace l’article 16 de la circulaire n°91-24.",
            ("evidence:cir-2016-03:p3:article-6", effective_evidence_uid),
        ),
    )
    return RegulatoryGraphBundle(
        instruments=instruments,
        source_editions=source_editions,
        pages=pages,
        provisions=provisions,
        provision_versions=versions,
        evidence_spans=evidence_spans,
        change_events=events,
    )


def _version(
    provision_uid: str,
    slug: str,
    text: str,
    valid_from: date,
    *,
    complete: bool = True,
) -> ProvisionVersion:
    identity = provision_uid.split(":")
    return ProvisionVersion(
        uid=f"version:bct:{identity[2]}:{identity[3]}:{slug}:{valid_from.isoformat()}",
        provision_uid=provision_uid,
        version_number=1,
        text=text,
        language="fr",
        valid_from=valid_from,
        status=VersionStatus.UNKNOWN,
        content_hash=_hash(text),
        confidence=1.0 if complete else 0.7,
        verification_status=(
            VerificationStatus.VERIFIED if complete else VerificationStatus.NEEDS_REVIEW
        ),
    )


def _replacement_event(
    source_article: str,
    target_provision_uid: str,
    introduced_version_uid: str,
    effective_from: date,
    raw_effect_text: str,
    evidence_uids: tuple[str, ...],
) -> ChangeEvent:
    return ChangeEvent(
        uid=f"event:bct:2016:03:{source_article}",
        source_instrument_uid="BCT:CIRCULAR:2016:03",
        action=LegalAction.REPLACE,
        target_scope=(
            TargetScope.ANNEX if ":ANNEX:" in target_provision_uid else TargetScope.ARTICLE
        ),
        target_provision_uids=(target_provision_uid,),
        effective_from=effective_from,
        raw_effect_text=raw_effect_text,
        evidence_uids=evidence_uids,
        introduces_version_uids=(introduced_version_uid,),
        confidence=1.0,
        extraction_method="manual_visual_verification",
        verification_status=VerificationStatus.VERIFIED,
        validator_reason="Exact operative clause and effective date verified on rendered PDF pages.",
    )


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
