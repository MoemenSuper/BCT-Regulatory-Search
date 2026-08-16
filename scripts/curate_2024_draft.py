from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("queries", []) if isinstance(data, dict) else data


def find(
    items: list[dict[str, Any]], source: str, query_fragment: str
) -> dict[str, Any]:
    matches = [
        item
        for item in items
        if item["expected_source"] == source
        and query_fragment.casefold() in item["query"].casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one match for {source!r} / {query_fragment!r}, got {len(matches)}")
    result = deepcopy(matches[0])
    result["language"] = "ar" if source.lower().endswith("_ar.pdf") else "fr"
    result["relevant"] = True
    result["evidence_method"] = "text_extraction"
    return result


def manual(
    source: str,
    page: int,
    language: str,
    category: str,
    query: str,
    answer: str,
    evidence: str,
    evidence_method: str = "text_extraction",
) -> dict[str, Any]:
    return {
        "query": query,
        "language": language,
        "category": category,
        "relevant": True,
        "expected_source": source,
        "expected_page": page,
        "expected_answer": answer,
        "evidence_quote": evidence,
        "evidence_method": evidence_method,
    }


def stable_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    result: list[dict[str, Any]] = []
    for item in items:
        stem = re.sub(r"[^a-z0-9]+", "_", Path(item["expected_source"]).stem.lower()).strip("_")
        key = (stem, item["category"])
        counts[key] += 1
        clean = deepcopy(item)
        clean["id"] = f"{stem}_{item['category']}_{counts[key]:02d}"
        result.append({"id": clean.pop("id"), **clean})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oss20", type=Path)
    parser.add_argument("oss120", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arabic", type=Path)
    args = parser.parse_args()

    small = load_items(args.oss20)
    large = load_items(args.oss120)
    curated: list[dict[str, Any]] = []

    curated.append(find(small, "Cir_2024_01_fr.pdf", "taux de majoration"))
    curated.append(
        manual(
            "Cir_2024_01_fr.pdf",
            5,
            "fr",
            "amount_or_rate",
            "Quel taux de provisionnement standard s'applique aux engagements agricoles du secteur privé ?",
            "Le taux de provisionnement standard applicable à l'agriculture est de 40 %.",
            "Groupe de contreparties\n\nTPgi\nProfessionnels du secteur privé\n\nAgriculture\n40%",
        )
    )

    item = find(large, "Cir_2024_02_fr.pdf", "avant de commercialiser")
    item["expected_answer"] = (
        "Les banques et établissements financiers doivent mettre en place une politique "
        "de commercialisation et de tarification approuvée par leur organe d'administration "
        "et déclinée en procédures formalisées."
    )
    curated.append(item)

    curated.append(find(small, "Cir_2024_03_fr.pdf", "tenue de compte"))
    curated.append(
        manual(
            "Cir_2024_03_fr.pdf",
            2,
            "fr",
            "amount_or_rate",
            "Quelle commission maximale un commerçant paie-t-il pour un paiement effectué par TPE avec une carte nationale ?",
            "La commission maximale est de 1,3 % du montant payé.",
            "Utilisation des TPE\nsur carte nationale\nCommerçants\néquipés de TPE\nOpération de paiement\n1,3% du\nmontant payé",
        )
    )
    curated.append(
        manual(
            "Cir_2024_03_fr.pdf",
            2,
            "fr",
            "deadline_or_duration",
            "Pendant combien de temps les plafonds et gratuités bancaires applicables à compter du 12 février 2024 restent-ils en vigueur ?",
            "Ils restent applicables pendant douze mois à compter du 12 février 2024.",
            "Article 2 : La présente circulaire entre en vigueur à compter du 12 février 2024.\nLes dispositions de la présente circulaire demeurent applicables durant une\npériode de douze (12) mois à compter de la date de son entrée en vigueur.",
        )
    )

    curated.append(
        manual(
            "Cir_2024_04_fr.pdf",
            2,
            "fr",
            "amount_or_rate",
            "Quelle part maximale du bénéfice de 2023 peut être distribuée lorsque les ratios de solvabilité et Tier 1 dépassent leurs minima réglementaires d'au moins 2,5 % ?",
            "La distribution est limitée à 35 % du bénéfice de l'exercice 2023.",
            "- Dans la limite de 35% du bénéfice de l’exercice 2023 pour les banques et\nles établissements financiers présentant des ratios de solvabilité et Tier 1\narrêtés à fin 2023, après déduction des dividendes à verser, qui dépassent\nles niveaux minimums réglementaires de 2,5% au moins ;",
        )
    )
    curated.append(
        manual(
            "Cir_2024_04_fr.pdf",
            2,
            "fr",
            "exception_or_condition",
            "Dans quelles conditions les dividendes de 2023 peuvent-ils être distribués sans plafond ?",
            "La distribution peut être effectuée sans plafond après accord préalable de la Banque Centrale de Tunisie, si les ratios de solvabilité et Tier 1, calculés après les dividendes, dépassent leurs minima réglementaires respectivement d'au moins 2,5 % et 3,5 %.",
            "- Sans limite et après accord préalable de la Banque Centrale de Tunisie, pour\nles banques et les établissements financiers présentant des ratios de\nsolvabilité et Tier 1 arrêtés à fin 2023, après déduction des dividendes à\nverser, qui dépassent les niveaux minimums réglementaires respectivement\nde 2,5% et 3,5% au moins.",
        )
    )
    curated.append(
        manual(
            "Cir_2024_04_fr.pdf",
            2,
            "fr",
            "required_action",
            "Une banque qui ne respectait pas les normes prudentielles d'adéquation des fonds propres à fin 2023 doit-elle obtenir un accord avant de distribuer des dividendes ?",
            "Oui. Elle doit obtenir l'accord préalable de la Banque Centrale de Tunisie.",
            "Article 2- Sans préjudice des conditions prévues à l’article premier de la présente\ncirculaire, l’accord préalable de la Banque Centrale de Tunisie est requis pour\ntoute distribution de dividendes par les banques et les établissements financiers ne\nrespectant pas, à fin 2023, les normes prudentielles d’adéquation des fonds\npropres prévues par les articles 50, 51 et 52 de la circulaire n°2018-06 susvisée.",
        )
    )

    curated.append(find(large, "Cir_2024_05_fr.pdf", "gestionnaire de système"))
    curated.append(find(large, "Cir_2024_05_fr.pdf", "Règlement Brut"))
    curated.append(find(small, "Cir_2024_06_fr.pdf", "À partir de quelle date"))
    curated.append(find(small, "Cir_2024_06_fr.pdf", "Combien de temps"))
    curated.append(find(small, "Cir_2024_07_fr.pdf", "Quels transferts financiers"))
    frequency = find(small, "Cir_2024_07_fr.pdf", "À quelle fréquence")
    frequency["query"] = (
        "À quelle fréquence les intermédiaires agréés doivent-ils transmettre via le SED "
        "la liste des transferts étrangers reçus par les associations et organismes sans but lucratif ?"
    )
    curated.append(frequency)

    curated.extend(
        [
            manual(
                "Cir_2024_08_fr.pdf",
                1,
                "fr",
                "eligibility_or_scope",
                "Quelles succursales de banques étrangères établies en Tunisie doivent présenter une lettre de garantie pour compléter leur dotation ?",
                "Sont concernées les succursales établies avant l'entrée en vigueur de la loi n°2016-48 dont la dotation atteint au moins la moitié du capital minimum requis sans atteindre ce capital minimum.",
                "Article premier : Les succursales de banques étrangères établies en Tunisie avant\nl’entrée en vigueur de la loi n°2016-48 susvisée et disposant d’une dotation au\nmoins égale à la moitié du capital minimum requis sans toutefois atteindre le\nmontant du capital minimum prévu par l’article 32 de cette même loi, doivent\nprésenter une lettre de garantie conformément au modèle en annexe à la présente\ncirculaire.",
            ),
            manual(
                "Cir_2024_08_fr.pdf",
                1,
                "fr",
                "deadline_or_duration",
                "Dans quel délai une succursale de banque étrangère concernée doit-elle déposer l'original signé de sa lettre de garantie ?",
                "Elle doit déposer l'original signé dans un délai maximal de deux mois à compter de la publication.",
                "Article 3 : La lettre de garantie doit être déposée à la Banque Centrale de Tunisie\n(Pôle Stabilité Financière) en format original et dûment signée par la banque\nétrangère dans un délai ne dépassant pas deux (2) mois à compter de la date de\npublication de la présente circulaire.",
            ),
            find(large, "Cir_2024_08_fr.pdf", "lettre de garantie est‑elle restituée"),
        ]
    )

    visual_rows = [
        (
            2,
            "Quel plafond et quelle échéance s'appliquent à l'achat d'alevins pour un élevage de tilapia d'une capacité de 10 tonnes ?",
            "Le plafond est de 11 milliers de dinars et l'échéance est de 12 mois.",
            "Elevage de Tilapia | 10 tonnes | Achat d’alevins | 11 | 12",
        ),
        (
            2,
            "Quel plafond et quelle échéance s'appliquent aux frais d'élevage et d'assurance d'un élevage de tilapia de 10 tonnes ?",
            "Le plafond est de 44 milliers de dinars et l'échéance est de 12 mois.",
            "Elevage de Tilapia | 10 tonnes | Frais d’élevage et d’assurance | 44 | 12",
        ),
        (
            3,
            "Pour un élevage de poissons en cages de 1000 tonnes, quels plafonds et quelle échéance couvrent l'achat d'alevins et les frais d'élevage et d'assurance ?",
            "Le plafond est de 2140 milliers de dinars pour l'achat d'alevins et de 10500 milliers de dinars pour les frais d'élevage et d'assurance; l'échéance est de 24 mois dans les deux cas.",
            "Elevage de poissons en cages | 1000 tonnes | Achat d’alevins 2140, échéance 24 mois | Frais d’élevage et d’assurance 10500, échéance 24 mois",
        ),
        (
            3,
            "Pour un élevage d'huîtres de 100 tonnes, quels plafonds et quelle échéance couvrent l'achat d'alevins et les frais d'élevage et d'assurance ?",
            "Le plafond est de 70 milliers de dinars pour l'achat d'alevins et de 266 milliers de dinars pour les frais d'élevage et d'assurance; l'échéance est de 24 mois dans les deux cas.",
            "Elevage de coquillages | 100 tonnes | Huître | Achat d’alevins 70, échéance 24 mois | Frais d’élevage et d’assurance 266, échéance 24 mois",
        ),
        (
            3,
            "Pour un élevage de moules de 100 tonnes, quels plafonds et quelle échéance couvrent l'achat d'alevins et les frais d'élevage et d'assurance ?",
            "Le plafond est de 105 milliers de dinars pour l'achat d'alevins et de 266 milliers de dinars pour les frais d'élevage et d'assurance; l'échéance est de 24 mois dans les deux cas.",
            "Elevage de coquillages | 100 tonnes | Moule | Achat d’alevins 105, échéance 24 mois | Frais d’élevage et d’assurance 266, échéance 24 mois",
        ),
    ]
    for page, query, answer, evidence in visual_rows:
        curated.append(
            manual(
                "Cir_2024_09_fr.pdf",
                page,
                "fr",
                "amount_or_rate",
                query,
                answer,
                evidence,
                "visual_review",
            )
        )

    modulation = find(small, "Cir_2024_11_fr.pdf", "ajuster le montant")
    modulation["query"] = (
        "Une banque doit-elle accorder automatiquement la totalité du plafond prévu "
        "pour un crédit de céréaliculture ?"
    )
    modulation["expected_answer"] = (
        "Non. Le montant doit être modulé selon la taille de l'exploitation, les dépenses "
        "à engager et les rendements des dernières campagnes."
    )
    curated.append(modulation)
    curated.append(find(small, "Cir_2024_13_fr.pdf", "À partir de quel mois"))
    curated.append(find(small, "Cir_2024_14_fr.pdf", "montant maximum d’un chèque"))
    curated.append(find(small, "Cir_2024_14_fr.pdf", "Dans quel délai"))

    for source, fragment in [
        ("Note_2024_05_fr.pdf", "Combien de temps"),
        ("Note_2024_05_fr.pdf", "engagement écrit"),
        ("Note_2024_101_fr.pdf", "À partir de quelle date"),
        ("Note_2024_101_fr.pdf", "poids"),
        ("Note_2024_129_fr.pdf", "Combien de jours"),
        ("Note_2024_163_fr.pdf", "date limite"),
        ("Note_2024_77_fr.pdf", "action préalable"),
    ]:
        curated.append(find(small, source, fragment))
    curated.append(
        manual(
            "Note_2024_140_fr.pdf",
            1,
            "fr",
            "effective_date",
            "À partir de quel moment la BFPME peut-elle émettre et recevoir des paiements par ELYSSA-RTGS ?",
            "La BFPME peut émettre et recevoir ces paiements à compter de la publication de son adhésion.",
            "À partir de la date de publication de cette note, la BFPME est autorisée à émettre et à\nrecevoir des paiements via son compte de règlement, mentionné ci-dessous, en utilisant le\nsystème ELYSSA-RTGS.",
        )
    )

    curated.extend(
        [
            manual(
                "Cir_2024_12_ar.pdf",
                1,
                "ar",
                "amount_or_rate",
                "ما هو الحد الأقصى لفارق الفائدة الذي تتكفل به الدولة لفائدة صغار فلاحي الحبوب؟",
                "تتكفل الدولة بفارق الفائدة في حدود ثلاث نقاط.",
                "شروط و  طرق صرف المبالغ المتعلقة باالنتفاع\nب تكفّل الدولة\n بالفارق بين نسبة الفائدة الموظفة على القروض الموسمية لزراعات\n الحبوب ومعدل نسبة الفائدة في السوق النقدية في حدود ثالث نقاط\n بالنسبة للقروض المسندة من قبل البنوك على مواردها الذاتية لفائدة\n.صغار الفالحين في قطاع زراعات الحبوب",
            ),
            manual(
                "Cir_2024_12_ar.pdf",
                3,
                "ar",
                "eligibility_or_scope",
                "ما هي الحدود القصوى للقرض الموسمي الأصلي والقرض التكميلي لصغار فلاحي الحبوب، وهل يلزم تقديم مطلب للانتفاع؟",
                "يجب ألا يتجاوز القرض الأصلي 50 ألف دينار وألا يتجاوز القرض التكميلي 15 ألف دينار، ولا يُطلب من المنتفع تقديم مطلب.",
                "القروض\n الموسمية لزراعات الحبوب المسندة من قبل البنوك\n على مواردها الذاتية\n لفائدة صغار الفالحين\nالتي ال يتجاوز مبلغها األصلي\n خمسين ألف دينار (\n50\n\n ألف\nدينار) و القروض التكميلية المنجرة عنها\n التي ال يتجاوز مبلغها خمسة عشر ألف\n دينار(\n15\n ألف دينار )\n ودون مطالبة\nالمنتفعين بتقديم مطالب في الغرض",
            ),
            manual(
                "Cir_2024_12_ar.pdf",
                4,
                "ar",
                "deadline_or_duration",
                "ما هي آجال تقديم البنوك لمطالب سحب مبالغ تكفل الدولة حسب الموسم الفلاحي؟",
                "آخر أجل هو 31 مارس 2025 لموسم 2022/2023، و31 مارس 2026 لموسم 2023/2024، و31 مارس 2027 لموسم 2024/2025.",
                "وللغرض يتعين على البنك تقديم\nا مط لب سحب\nإلى البنك المركزي التونسي\n في أجل\nأقصاه31\n مارس\n2025\n\n بالنسبة للموسم الفالحي2022\n/\n2023\n ، وفي أجل أقصاه\n31\n\n  مارس2026\n\n  بالنسبة  للموسم  الفالحي2023\n/\n2024\n،\n\n وفي  أجل  أقصاه\n31\n\n مارس2027\n\n بالنسبة للموسم الفالحي2024\n/\n2025",
            ),
        ]
    )

    if args.arabic and args.arabic.exists():
        arabic_items = load_items(args.arabic)
        clean_sources = {
            "Cir_2024_10_ar.pdf",
            "Note_2024_04_ar.pdf",
            "Note_2024_131_ar.pdf",
            "Note_2024_147_ar.pdf",
            "Note_2024_44_ar.pdf",
            "Note_2024_87_ar.pdf",
        }
        for item in arabic_items:
            if item["expected_source"] not in clean_sources:
                continue
            clean = deepcopy(item)
            clean["evidence_method"] = "text_extraction"
            curated.append(clean)

        curated.extend(
            [
                manual(
                    "Note_2024_119_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هو رمز شركة FIRST PAY في مصنفة مؤسسات الدفع، وما هو النشاط المرخص لها؟",
                    "رمز شركة FIRST PAY هو 85، وهي مرخص لها بممارسة نشاط مؤسسة دفع مقيمة.",
                    "قرار لجنة التراخيص عدد 59 المؤرخ\n في\nغرة\n جويلية2024\n\n المتعلق بمنح الترخيص لشركة\n«\nFIRST PAY\n» لممارسة نشاط\n مؤسسة\nدفع مقيمة،\n يعلم البنك المركزي التونسي كافة الوسطاء المقبولين والمؤسسات المالية\nبتحيين\n مصنفة رموز\n هذه المؤسسات كما ي يل:\n\n\n\nالرمز التسمية اإلجتماعية لمؤسسة الدفع\n85\n\n شركة « FIRST PAY »",
                ),
                manual(
                    "Note_2024_136_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هي رموز الاقتطاعات البنكية والبريدية المسندة إلى EUROPEAN NORM TUNISIE وTRY AND BUY وSOMOPRINT؟",
                    "الرموز هي 0142 لشركة EUROPEAN NORM TUNISIE و0143 لشركة TRY AND BUY و0144 لشركة SOMOPRINT.",
                    "الرمز\n المؤسسة المصد ة\nلالقتطاعات\n البنكية والبريدية\n0142\n\n شركة« EUROPEAN NORM TUNISIE »\n\n0143\n\n شركة« TRY AND BUY »\n\n0144\n\n شركة« SOMOPRINT »",
                ),
                manual(
                    "Note_2024_145_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هو رمز الاقتطاعات البنكية والبريدية للجامعة التونسية لشركات التأمين FTUSA؟",
                    "رمز الجامعة التونسية لشركات التأمين FTUSA هو 0145.",
                    "الرمز\n المؤسسة المصد ة\nلالقتطاعات\n البنكية والبريدية\n0145\n ال\nجامعة التونسية لشركات الت\nأمين\n(FTUSA)",
                ),
                manual(
                    "Note_2024_165_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هي رموز الاقتطاعات البنكية والبريدية المسندة إلى شركات SEPAC الثلاث؟",
                    "الرمز 0146 مسند إلى SEPAC SA، والرمز 0147 إلى SEPACS SARL، والرمز 0148 إلى SEPAC INTERNATIONAL SA.",
                    "الرمز\n المؤسسة المصد ة\nلالقتطاعات\n البنكية والبريدية\n0146\n\n\n شركة التعليت الخاص واألنشطة\nالثقافية (شركة\n خمية\nاال\nسم )\n« SEPAC SA »\n\n0147\n\n شركة التعليت الخاص واألنشطة الثقافية والرياضية (ش ركة ذات مسؤ\nولية محدودة )\n« SEPACS SARL »\n\n0148\n شركة سيباع العالمية)\n شركة خمية\nاال سم(\n« SEPAC INTERNATIONAL SA »",
                ),
                manual(
                    "Note_2024_187_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هو رمز الاقتطاعات البنكية والبريدية للشركة التونسية للتأمين وإعادة التأمين STAR؟",
                    "رمز الشركة التونسية للتأمين وإعادة التأمين STAR هو 0150.",
                    "الرمو\n المؤسسة المصدرة\nلالق طاعات\n البنةية والبريدية\n0150\n\n\n\n الشركة ال و سية لل أمين و إ عادة ال أمين\n« STAR »",
                ),
                manual(
                    "Note_2024_178_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هو رمز الاقتطاعات البنكية والبريدية لشركة EL KHAMSA DAY SPA؟",
                    "رمز شركة EL KHAMSA DAY SPA هو 0149.",
                    "الرمو\n المؤسسة المصدرة\nلالق طاعات\n البنةية والبريدية\n0149\n\n« Société EL KHAMSA DAY SPA »",
                ),
                manual(
                    "Note_2024_29_ar.pdf",
                    2,
                    "ar",
                    "amount_or_rate",
                    "ما هي مبالغ القروض التكميلية للهكتار وآجال سدادها للحبوب المروية والبعلية في موسم 2023-2024؟",
                    "المبلغ هو 265 دينارا للهكتار للحبوب المروية، و265 دينارا للحبوب البعلية بالمنطقة 1، و236 دينارا للحبوب البعلية بالمنطقة 2؛ وآخر أجل للسداد هو 31 أوت 2024 في الحالات الثلاث.",
                    "حبوب مروية | هكتار | 265 دينارا | 31 أوت 2024\nحبوب بعلية منطقة 1 | هكتار | 265 دينارا | 31 أوت 2024\nحبوب بعلية منطقة 2 | هكتار | 236 دينارا | 31 أوت 2024",
                    "visual_review",
                ),
                manual(
                    "Note_2024_56_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هما رمزا مؤسستي الدفع Zitouna Pay وPayvago في نظام المقاصة الإلكترونية؟",
                    "رمز Zitouna Pay هو 81، ورمز Payvago هو 84.",
                    "الرمز\n التسمية االجتماعية لمؤسسة الدفع\n81\n\n\"\nZitouna Pay\n \"\n\n84\n\n\"\nPayvago\n \"",
                ),
                manual(
                    "Note_2024_60_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هو رمز الاقتطاعات البنكية والبريدية لشركة L'UNIVERS DU FOOT، ومتى يصبح نافذا؟",
                    "رمز الشركة هو 0137، ويصبح نافذا من تاريخ الإشعار به.",
                    "الرمز\n المؤسسة المصدرة\nلالقتطاعات\n البنكية والبريدية\n0137\n\n\n  شركة« L’UNIVERS DU FOOT »\n\n\n\nالفصل\nالثاني : تدخ  هذه المذكرة حيز التنفيذ بداي ة من تاريخ اإلشعار\nب .ها",
                ),
                manual(
                    "Note_2024_85_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هما رمزا الاقتطاعات البنكية والبريدية لشركتي WINTHROP PHARMA TUNISIE وSANOFI AVENTIS PHARMA TUNISIE؟",
                    "الرمز 0138 مسند إلى WINTHROP PHARMA TUNISIE، والرمز 0139 مسند إلى SANOFI AVENTIS PHARMA TUNISIE.",
                    "الرمز\n المؤسسة المصدرة\nلالقتطاعات\n البنكية ّالبريدية\n0138\n\n\n شركة« WINTHROP PHARMA TUNISIE »\n\n0139\n\n شركة« SANOFI AVENTIS PHARMA TUNISIE »",
                ),
                manual(
                    "Note_2024_86_ar.pdf",
                    1,
                    "ar",
                    "definition",
                    "ما هو رمز الاقتطاعات البنكية والبريدية لشركة DALLAS SPORT SQUARE FIT؟",
                    "رمز شركة DALLAS SPORT SQUARE FIT هو 0140.",
                    "الرمز\n المؤسسة المصدرة\nلالقتطاعات\n البنكية والبريدية\n0140\n\n\n شركة« DALLAS SPORT SQUARE FIT »",
                ),
                manual(
                    "Note_2024_92_ar.pdf",
                    1,
                    "ar",
                    "effective_date",
                    "متى بدأ تداول الأوراق النقدية الإنجليزية الجديدة من فئات 5 و10 و20 و50 جنيها، وما مصير الأوراق الحالية من الفئات نفسها؟",
                    "بدأ تداول الأوراق الجديدة في 05 جوان 2024، وتبقى الأوراق الحالية من الفئات نفسها متداولة إلى جانبها إلى حين إشعار آخر.",
                    "بنك\nأنقلترا (BANK OF ENGLAND)  طرح للتداول\n بداية من\nتاريخ\n05\n جوان\n2024\n أوراق نقدية جديدة من فئ\nـ ة 5\n ،\n10\n ،\n20\n\n و50\n  جني\nها إ\nسترلين\nيا،\n\nهذا ّوقد أعلن بنك أنقلترا أن\nاألوراق\n النقدية\nمن نفس الفئة المتداولة حاليا ستبقى\n\n في التداول جنبا إلى جنب مع هذه\nاألوراق\n النقدية الجديدة إلى حين\nإشعار آخر.",
                    "visual_review",
                ),
            ]
        )

    curated = stable_ids(curated)
    args.output.write_text(
        json.dumps(curated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"curated {len(curated)} questions across "
        f"{len({item['expected_source'] for item in curated})} sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
