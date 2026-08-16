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


def find(items: list[dict[str, Any]], source: str, fragment: str) -> dict[str, Any]:
    matches = [
        item
        for item in items
        if item["expected_source"] == source
        and fragment.casefold() in item["query"].casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one match for {source!r} / {fragment!r}, got {len(matches)}")
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
    parser.add_argument("french", type=Path)
    parser.add_argument("arabic", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    french = load_items(args.french)
    arabic = load_items(args.arabic)
    curated: list[dict[str, Any]] = []

    curated.append(
        manual(
            "Cir_2023_01_fr.pdf",
            2,
            "fr",
            "amount_or_rate",
            "Quel taux annuel de rémunération de l'épargne s'applique à compter du 2 janvier 2023 ?",
            "Le taux annuel de rémunération de l'épargne est fixé à 7 % à compter du 2 janvier 2023.",
            "Article 36 (alinéa premier nouveau) - Le taux de rémunération de l’épargne (TRE) est fixé à un taux annuel de 7%.\nArticle 2 - La présente circulaire entre en vigueur à partir du  2 janvier 2023.",
        )
    )
    curated.append(find(french, "Cir_2023_02_fr.pdf", "réviser les provisions collectives"))
    curated.append(find(french, "Cir_2023_02_fr.pdf", "historique est utilisé"))
    curated.append(find(french, "Cir_2023_03_fr.pdf", "montant maximum d'allocation"))
    curated.append(find(french, "Cir_2023_03_fr.pdf", "Quel document"))
    modulation = find(french, "Cir_2023_04_fr.pdf", "Quel critère")
    modulation["query"] = (
        "Une banque doit-elle accorder automatiquement la totalité du plafond prévu "
        "pour un crédit destiné à une exploitation maraîchère ?"
    )
    modulation["expected_answer"] = (
        "Non. Le montant doit être modulé selon la taille de l'exploitation, les dépenses "
        "à engager et les rendements des dernières campagnes."
    )
    curated.append(modulation)
    curated.extend(
        [
            manual(
                "Cir_2023_05_fr.pdf",
                3,
                "fr",
                "amount_or_rate",
                "À partir de quel poids une entité est-elle considérée comme significative pour le périmètre prudentiel de consolidation ?",
                "Une entité est considérée comme significative lorsque son poids est supérieur ou égal à 1 %.",
                "Condition n°1 : Il est supérieur ou égal à 1% ; ou",
            ),
            find(french, "Cir_2023_05_fr.pdf", "poids est inférieur à 1"),
            find(french, "Cir_2023_06_fr.pdf", "documents complémentaires"),
            find(french, "Cir_2023_06_fr.pdf", "Par quel moyen"),
        ]
    )
    curated.extend(
        [
            manual(
                "Note_2023_06_fr.pdf",
                1,
                "fr",
                "procedure_or_documents",
                "Quel formulaire les sociétés totalement exportatrices résidentes doivent-elles utiliser pour domicilier leurs factures d'importation sous admission temporaire ou en entrepôt ?",
                "Elles doivent utiliser le formulaire électronique du contrat commercial réservé aux importations en admission temporaire ou en entrepôt.",
                "la domiciliation, de factures d’importation de matériels et équipements nécessaires à la production de biens ou de services par les sociétés totalement exportatrices résidentes à travers le système intégré de traitement automatisé des formalités de commerce extérieur, est réalisée par l’utilisation du formulaire électronique du contrat commercial réservé aux importations en admission temporaire ou en entrepôt.",
            ),
            manual(
                "Note_2023_06_fr.pdf",
                1,
                "fr",
                "required_action",
                "Sur quelles données les intermédiaires agréés doivent-ils s'appuyer pour régler les importations effectuées sous admission temporaire ou en entrepôt ?",
                "Ils doivent s'appuyer sur les données transmises par les bureaux de douane via le système intégré de traitement automatisé des formalités de commerce extérieur qui justifient l'entrée des marchandises.",
                "Les Intermédiaires agréés doivent se baser sur les données parvenues par les bureaux de douane via le système intégré de traitement automatisé des formalités de commerce extérieur justifiant l’entrée des marchandises, pour effectuer le règlement financier de ces importations",
            ),
            manual(
                "Note_2023_10_fr.pdf",
                2,
                "fr",
                "amount_or_rate",
                "Quelle part maximale du bénéfice de 2022 peut être distribuée lorsque les ratios de solvabilité et Tier 1 dépassent leurs minima réglementaires d'au moins 2,5 % ?",
                "La distribution est limitée à 35 % du bénéfice de l'exercice 2022.",
                "dans la limite de 35% du bénéfice de l’exercice 2022 pour les banques et les établissements financiers présentant des ratios de solvabilité et Tier 1 arrêtés à fin 2022, après déduction des dividendes à verser, dépassent les niveaux minimums réglementaires de 2,5% au moins",
            ),
            manual(
                "Note_2023_10_fr.pdf",
                2,
                "fr",
                "exception_or_condition",
                "Dans quelles conditions les dividendes de 2022 peuvent-ils être distribués sans plafond ?",
                "La distribution peut être effectuée sans plafond après accord préalable de la Banque Centrale de Tunisie si, après déduction des dividendes, les ratios de solvabilité et Tier 1 dépassent leurs minima réglementaires respectivement d'au moins 2,5 % et 3,5 %.",
                "sans limite et après accord préalable de la Banque Centrale de Tunisie, pour les banques et les établissements financiers présentant des ratios de solvabilité et Tier 1 arrêtés à fin 2022, après déduction des dividendes à verser, dépassent les niveaux minimums réglementaires respectivement de 2,5% et 3,5% au moins.",
            ),
            find(french, "Note_2023_29_fr.pdf", "deuxième composante"),
        ]
    )

    curated.append(find(arabic, "Cir_2023_07_ar.pdf", "فتح الحسابات الخاصة"))
    extension = find(arabic, "Cir_2023_08_ar.pdf", "التاريخ الجديد")
    extension["query"] = "إلى أي تاريخ مُدّد انتفاع المؤسسات الصغرى والمتوسطة بامتياز تكفل الدولة بالفارق في نسبة الفائدة على قروض الاستثمار؟"
    curated.append(extension)
    curated.append(find(arabic, "Note_2023_02_ar.pdf", "التعريفات الديوانية"))
    curated.append(find(arabic, "Note_2023_03_ar.pdf", "SOTACIB Kairouan"))
    curated.extend(
        [
            manual(
                "Note_2023_05_ar.pdf",
                2,
                "ar",
                "amount_or_rate",
                "ما مبلغ القرض التكميلي للهكتار الواحد من الحبوب المروية والحبوب البعلية في المنطقتين 1 و2 لموسم 2022-2023؟",
                "المبلغ هو 257 دينارا للهكتار للحبوب المروية، و257 دينارا للحبوب البعلية في المنطقة 1، و228 دينارا للحبوب البعلية في المنطقة 2.",
                "حبوب مروية | هكتار | 257 دينارا؛ حبوب بعلية منطقة 1 | هكتار | 257 دينارا؛ حبوب بعلية منطقة 2 | هكتار | 228 دينارا؛ موسم 2022-2023",
                "visual_review",
            ),
            manual(
                "Note_2023_05_ar.pdf",
                2,
                "ar",
                "deadline_or_duration",
                "ما أجل تسديد القروض التكميلية المخصصة لتسميد الحبوب ومداواة الأعشاب الطفيلية والأمراض الفطرية لموسم 2022-2023؟",
                "أجل التسديد هو 31 أوت 2023.",
                "أجل التسديد | 31 أوت 2023 | لجميع أصناف الحبوب الواردة بالجدول",
                "visual_review",
            ),
            manual(
                "Note_2023_07_ar.pdf",
                2,
                "ar",
                "deadline_or_duration",
                "ما آخر أجل لتقديم الفلاح المتضرر من الجفاف مطلب جدولة ديونه لموسم 2021-2022؟",
                "آخر أجل هو موفى مارس 2023.",
                "يقدم مطلب الجدولة من طرف الفلاح مباشرة إلى فرع البنك الممول للقرض في أجل لا يتعدى موفى مارس 2023.",
                "visual_review",
            ),
            find(arabic, "Note_2023_08_ar.pdf", "SAUVER TUNISIA"),
            manual(
                "Note_2023_09_ar.pdf",
                1,
                "ar",
                "definition",
                "ما الرمز المسند إلى مؤسسة الدفع المقيمة PAYVAGO في مصنفة الوسطاء المقبولين والمؤسسات المالية؟",
                "الرمز المسند إلى PAYVAGO هو 84.",
                "الرمز\nاالسم االجتماعي\n84\n \n       \n شركة                                             « PAYVAGO »",
            ),
            manual(
                "Note_2023_11_ar.pdf",
                1,
                "ar",
                "definition",
                "ما الرمز المخصص لـ Mutuelle d’Assurance de l’Enseignement (MAE) كمصدر للاقتطاعات البنكية والبريدية؟",
                "الرمز المخصص لها هو 0132.",
                "الرمز المؤسسة المصدرة لإلقتطاعات البنكية والبريدية\n \n0132\n \n \nMutuelle d’Assurance de l’Enseignement (MAE)",
            ),
        ]
    )
    curated.append(find(arabic, "Note_2023_12_ar.pdf", "18 مارس 2023"))
    curated.append(find(arabic, "Note_2023_13_ar.pdf", "1000 درهم"))
    eid = find(arabic, "Note_2023_15_ar.pdf", "21 أفريل 2023")
    eid["query"] = "ما توقيت فتح شبابيك البنوك يوم الجمعة 21 أفريل 2023، وما العمليات المسموح بها خلال ذلك التوقيت؟"
    eid["expected_answer"] = "تفتح من الساعة التاسعة صباحا إلى منتصف النهار، وتقتصر العمليات على التنزيل نقدا والسحب والصرف اليدوي."
    curated.append(eid)
    curated.append(find(arabic, "Note_2023_16_ar.pdf", "29 أفريل 2023"))
    curated.append(find(arabic, "Note_2023_19_ar.pdf", "تقرير المبيعات"))
    curated.append(find(arabic, "Note_2023_20_ar.pdf", "استمرارية عمليات السحب"))
    curated.extend(
        [
            manual(
                "Note_2023_26_ar.pdf",
                1,
                "ar",
                "definition",
                "ما الرمز المخصص لشركة MW ENTERTAINMENT Centre de Sport et Entrainement كمصدر للاقتطاعات البنكية والبريدية؟",
                "الرمز المخصص للشركة هو 0133.",
                "الرمز \n المؤسسة المصدرة\nلالقتطاعات \n البنكية والبريدية \n0133\n \n شركة«MW ENTERTAINMENT \n \nCentre de Sport et Entrainement »",
            ),
            manual(
                "Note_2023_28_ar.pdf",
                1,
                "ar",
                "definition",
                "ما الاسم والاسم المختصر الجديدان للشركة المصنفة بالرمز 43 بدل Société Arab International Lease؟",
                "أصبح اسمها BTK Leasing واسمها المختصر BTKL، مع بقاء الرمز 43.",
                "43\n الشركة العر\nية ب ال\nدولية \n لإليجار\n المالي \n« Société Arab \nInternational Lease »\n \nAIL\n \n \n43\n \n \"\n ب. ت. ك ليزينق\"\n \n« BTK Leasing »\n \n \nBTKL",
            ),
            manual(
                "Note_2023_30_ar.pdf",
                2,
                "ar",
                "deadline_or_duration",
                "ما آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من الجفاف خلال الموسم الفلاحي 2022-2023؟",
                "آخر أجل هو موفى ديسمبر 2023.",
                "يقدم مطلب الجدولة من طرف الفلاح مباشرة إلى فرع البنك الممول للقرض في أجل لا يتعدى موفى ديسمبر 2023.",
                "visual_review",
            ),
            manual(
                "Note_2023_30_ar.pdf",
                2,
                "ar",
                "exception_or_condition",
                "ما المدة القصوى لجدولة ديون الفلاحين المتضررين من الجفاف لموسم 2022-2023؟",
                "تتم الجدولة حالة بحالة على مدة لا تتجاوز خمس سنوات، مع الحفاظ على نسبة الفائدة الأصلية ومراعاة قدرة الفلاح على السداد ونسبة الضرر.",
                "تتم جدولة القروض على مدة لا تتجاوز خمس سنوات وبنفس نسبة الفائدة التي منحت بها، حالة بحالة، مع الأخذ بعين الاعتبار قدرة المنتفع على التسديد ونسبة الضرر من الجفاف.",
                "visual_review",
            ),
            manual(
                "Note_2023_33_ar.pdf",
                1,
                "ar",
                "definition",
                "ما الرمز المخصص لشركة BARAKET & SABAI TRADE كمصدر للاقتطاعات البنكية والبريدية؟",
                "الرمز المخصص للشركة هو 0134.",
                "الرمز \n المؤسسة المصدرة\nلالقتطاعات \n البنكية والبريدية \n0134\n \n شركة«BARAKET & SABAI TRADE »",
            ),
        ]
    )
    curated.append(find(arabic, "Note_2023_35_ar.pdf", "مصاريف مسبقة"))
    curated.append(find(arabic, "Note_2023_40_ar.pdf", "500 درهم"))
    year_end = find(arabic, "Note_2023_44_ar.pdf", "30 ديسمبر 2023")
    year_end["query"] = "ما توقيت فتح شبابيك البنوك يوم السبت 30 ديسمبر 2023، وما العمليات المسموح بها؟"
    year_end["expected_answer"] = "تفتح من الساعة التاسعة صباحا إلى منتصف النهار، وتقتصر العمليات على التنزيل نقدا والسحب والصرف اليدوي."
    curated.append(year_end)
    curated.append(
        manual(
            "Note_2023_45_ar.pdf",
            1,
            "ar",
            "definition",
            "ما الرمز المخصص لـ Mutuelle de la STEG كمصدر للاقتطاعات البنكية والبريدية؟",
            "الرمز المخصص لها هو 0135.",
            "الرمز \n المؤسسة المصدرة\nلالقتطاعات \n البنكية والبريدية \n0135\n \n«Mutuelle de la STEG »",
        )
    )

    output = stable_ids(curated)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
