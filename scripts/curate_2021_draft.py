from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "documents"
PDFS = {path.name: path for path in DOCUMENTS.rglob("*.pdf")}


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("queries", []) if isinstance(data, dict) else data


def full_page(source: str, page: int) -> str:
    document = fitz.open(PDFS[source])
    return document[page - 1].get_text("text").strip()


def manual(
    source: str,
    page: int,
    language: str,
    category: str,
    query: str,
    answer: str,
    evidence: str | None = None,
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
        "evidence_quote": evidence if evidence is not None else full_page(source, page),
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
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    generated = load_items(args.french)
    curated: list[dict[str, Any]] = []
    replaced_sources = {"Cir_2021_06_fr.pdf", "Cir_2021_07_fr.pdf"}
    dropped = {
        ("Cir_2021_05_fr.pdf", 36),
        ("Cir_2021_05_fr.pdf", 37),
    }
    for candidate in generated:
        source = candidate["expected_source"]
        if source in replaced_sources or (source, candidate["expected_page"]) in dropped:
            continue
        item = deepcopy(candidate)
        item["language"] = "fr"
        item["relevant"] = True
        item["evidence_method"] = "text_extraction"
        if source == "Cir_2021_05_fr.pdf" and item["expected_page"] == 31:
            item["query"] = (
                "Dans quel délai un établissement doit-il remettre sa feuille de route pour "
                "se conformer aux exigences de gouvernance et de gestion des risques publiées en 2021 ?"
            )
            item["expected_answer"] = "Il doit la remettre dans les trois mois suivant la publication des exigences."
        curated.append(item)

    visual_rows = [
        (
            "Cir_2021_06_fr.pdf",
            "Quel plafond de crédit par hectare et quelle échéance s'appliquent au blé dur, au blé tendre et aux légumineuses en zones 1 et 2, en sec et en irrigué ?",
            "En sec, le plafond est de 1175 dinars en zone 1 et de 960 dinars en zone 2. En irrigué, il est de 1625 dinars dans les deux zones. L'échéance est le 31 août.",
            "Blé dur, blé tendre et légumineuses | zone 1 en sec 1175 DT/ha | zone 2 en sec 960 DT/ha | irrigué 1625 DT/ha | échéance 31 août",
        ),
        (
            "Cir_2021_06_fr.pdf",
            "Quels plafonds de crédit par hectare s'appliquent à l'orge cultivée en sec dans les zones 1, 2 et 3, et à quelle date le crédit arrive-t-il à échéance ?",
            "Les plafonds sont de 750 dinars en zone 1, 680 dinars en zone 2 et 250 dinars en zone 3. L'échéance est le 31 août.",
            "Orge en sec | zone 1 : 750 DT/ha | zone 2 : 680 DT/ha | zone 3 : 250 DT/ha | échéance 31 août",
        ),
        (
            "Cir_2021_06_fr.pdf",
            "Quels plafonds et quelles échéances s'appliquent aux fourrages d'hiver en sec et aux fourrages d'été en irrigué ?",
            "Le plafond des fourrages d'hiver en sec est de 870 dinars par hectare avec échéance au 31 août. Celui des fourrages d'été en irrigué est de 1140 dinars par hectare avec échéance au 30 septembre.",
            "Fourrages d'hiver en sec | 870 DT/ha | 31 août ; fourrages d'été en irrigué | 1140 DT/ha | 30 septembre",
        ),
        (
            "Cir_2021_07_fr.pdf",
            "Quel plafond de crédit par hectare et quelle échéance s'appliquent à une vigne de table intensive conduite en pergola ?",
            "Le plafond est de 11320 dinars par hectare et l'échéance est le 30 novembre.",
            "Vigne de table intensive (Pergola) | irrigué | 11320 DT/ha | 30 novembre",
        ),
        (
            "Cir_2021_07_fr.pdf",
            "Quels plafonds de crédit par hectare s'appliquent à la vigne de cuve en sec et en irrigué, et quelle est l'échéance ?",
            "Le plafond est de 3085 dinars en sec et de 5210 dinars en irrigué. L'échéance est le 30 septembre.",
            "Vigne de cuve | en sec 3085 DT/ha | en irrigué 5210 DT/ha | 30 septembre",
        ),
        (
            "Cir_2021_07_fr.pdf",
            "Quels plafonds de crédit par hectare s'appliquent aux amandiers du Nord et du Centre-Sud en sec et en irrigué ?",
            "En sec, le plafond est de 2305 dinars au Nord et de 1755 dinars au Centre et au Sud. En irrigué, il est de 4525 dinars dans les deux cas. L'échéance est le 31 août.",
            "Amandiers au Nord | sec 2305 DT/ha | irrigué 4525 DT/ha ; amandiers au Centre et Sud | sec 1755 DT/ha | irrigué 4525 DT/ha | 31 août",
        ),
        (
            "Cir_2021_07_fr.pdf",
            "Quel plafond de crédit par hectare et quelle échéance s'appliquent à la culture irriguée de fraises ?",
            "Le plafond est de 45600 dinars par hectare et l'échéance est le 31 mai.",
            "Fraise | irrigué | 45600 DT/ha | 31 mai",
        ),
        (
            "Cir_2021_07_fr.pdf",
            "Quel plafond de crédit par hectare et quelle échéance s'appliquent à la tomate de saison ?",
            "Le plafond est de 9520 dinars par hectare et l'échéance est le 31 juillet.",
            "Tomate de saison | irrigué | 9520 DT/ha | 31 juillet",
        ),
    ]
    for source, query, answer, evidence in visual_rows:
        curated.append(manual(source, 2, "fr", "amount_or_rate", query, answer, evidence, "visual_review"))

    curated.extend(
        [
            manual(
                "Note_2021_12_fr.pdf",
                3,
                "fr",
                "procedure_or_documents",
                "Quelles pièces doivent accompagner une demande de décaissement au titre des lignes de crédit agricole et d'économie sociale et solidaire ?",
                "La demande doit être accompagnée d'une copie du contrat de prêt signé, du spécimen des signatures habilitées, du RIB destinataire et d'une déclaration d'adhésion aux principes de l'économie sociale et solidaire.",
            ),
            manual(
                "Note_2021_12_fr.pdf",
                4,
                "fr",
                "reporting_or_control",
                "À quelle fréquence et sous quels formats un intermédiaire agréé ou un établissement de leasing doit-il communiquer l'état des déblocages au cabinet Audit et Conseil AWT ?",
                "Il doit communiquer un état des déblocages chaque trimestre, aux formats numérique et papier.",
            ),
            manual(
                "Note_2021_16_fr.pdf",
                2,
                "fr",
                "amount_or_rate",
                "Comment se calcule le plafond de la subvention COVID-19 destinée à soutenir l'emploi d'une entreprise privée ?",
                "Le plafond est le plus petit montant entre trois mois de masse salariale, avec un maximum mensuel de 600 TND par salaire éligible, et 1,5 mois du chiffre d'affaires moyen de 2019, sans dépasser l'équivalent en dinars de 200.000 euros par entreprise.",
            ),
            manual(
                "Note_2021_16_fr.pdf",
                3,
                "fr",
                "amount_or_rate",
                "Quelle part minimale de la subvention de soutien à l'emploi doit servir à payer les salaires du personnel ?",
                "Au moins 50 % de la subvention doit être consacré au paiement partiel ou total des salaires.",
            ),
            manual(
                "Note_2021_16_fr.pdf",
                7,
                "fr",
                "deadline_or_duration",
                "Quelle était la date limite de décaissement des demandes éligibles au programme de soutien de l'emploi privé financé en 2021, sauf prorogation ?",
                "La date limite était le 22 août 2021, sauf prorogation.",
            ),
            manual(
                "Note_2021_16_fr.pdf",
                9,
                "fr",
                "eligibility_or_scope",
                "Quel effectif maximal une entreprise pouvait-elle compter pour être éligible au programme de soutien de l'emploi privé de 2021 ?",
                "L'entreprise devait compter au maximum 250 employés.",
            ),
            manual(
                "Note_2021_28_fr.pdf.pdf",
                1,
                "fr",
                "procedure_or_documents",
                "Que doit faire une banque participant au SGMT avant d'adhérer directement ou indirectement au système interarabe de règlement Buna ?",
                "Elle doit d'abord soumettre une requête à la Banque Centrale de Tunisie. Après vérification des critères d'éligibilité de Buna, la Banque Centrale lui délivre une attestation d'adhésion au SGMT à présenter au gestionnaire avec le dossier requis.",
            ),
        ]
    )

    arabic_rows = [
        ("Note_2021_01_ar.pdf", 2, "required_action", "ما الإجراءات التي كان على البنوك والديوان الوطني للبريد اتخاذها لضمان السحب والدفع الإلكتروني خلال الحجر الصحي من 14 إلى 17 جانفي 2021؟", "كان عليها ضمان التزويد المستمر للموزعات، وجاهزية منصات الدفع، ومعالجة الأعطال والشكاوى، وإعلام البنك المركزي بالتدابير والمستجدات المؤثرة في أنشطة الدفع."),
        ("Note_2021_02_ar.pdf", 1, "effective_date", "متى طُرحت للتداول الورقتان السعوديتان الجديدتان من فئتي 5 و20 ريالا؟", "طُرحت فئة 5 ريالات في 5 أكتوبر 2020، وفئة 20 ريالا في 25 أكتوبر 2020."),
        ("Note_2021_03_ar.pdf", 2, "amount_or_rate", "ما مبالغ القروض التكميلية للهكتار من الحبوب المروية والحبوب البعلية في المنطقتين 1 و2 لموسم 2020-2021، وما أجل سدادها؟", "المبلغ هو 233 دينارا للحبوب المروية، و233 دينارا للحبوب البعلية في المنطقة 1، و213 دينارا للحبوب البعلية في المنطقة 2، وأجل السداد هو 31 أوت 2021."),
        ("Note_2021_04_ar.pdf", 3, "deadline_or_duration", "إلى أي تاريخ واصل البنك المركزي التونسي قبول أوراق الإصدار الرابع للريال القطري المسحوبة من التداول؟", "واصل قبولها إلى غاية 31 مارس 2021 بدخول الغاية."),
        ("Note_2021_07_ar.pdf", 1, "definition", "ما الرمز المخصص لشركة TrustLink كمصدر للاقتطاعات البنكية والبريدية؟", "الرمز المخصص للشركة هو 0117."),
        ("Note_2021_09_ar.pdf", 2, "eligibility_or_scope", "ما نطاقات التعريفات الديوانية للمقاعد وللأثاث وأجزائه التي تنطبق على قائمة الموردين المسجلين؟", "يمتد نطاق المقاعد من 94013000002 إلى 94018000900، ونطاق الأثاث وأجزائه من 94031051007 إلى 94039090008."),
        ("Note_2021_10_ar.pdf", 1, "required_action", "ما التدابير المطلوبة من البنوك والبريد لضمان السحب والدفع الإلكتروني خلال عطلة عيد الشهداء والأيام الموالية في أفريل 2021؟", "يجب ضمان التزويد المستمر للموزعات وجاهزية منصات الدفع، ومعالجة الأعطال والانقطاعات سريعا، وإعلام البنك المركزي بالتدابير والمستجدات المؤثرة في أنشطة الدفع."),
        ("Note_2021_11_ar.pdf", 1, "definition", "ما الرمز المخصص لـ AMICALE DES CADRES DE LA STEG كمصدر للاقتطاعات البنكية والبريدية؟", "الرمز المخصص لها هو 0118."),
        ("Note_2021_13_ar.pdf", 1, "definition", "ما الرمز المخصص للصندوق الوطني للتقاعد والحيطة الاجتماعية CNRPS كمصدر للاقتطاعات البنكية والبريدية؟", "الرمز المخصص له هو 0119."),
        ("Note_2021_14_ar.pdf", 1, "required_action", "ما الخدمات التي وجب على البنوك والبريد ضمان استمراريتها خلال الحجر الصحي من 9 إلى 16 ماي 2021؟", "وجب ضمان استمرارية السحب من الموزعات الآلية والدفع الإلكتروني، مع استمرار التزويد ومعالجة الأعطال وإعلام البنك المركزي بالمستجدات."),
        ("Note_2021_15_ar.pdf", 1, "deadline_or_duration", "ما توقيت فتح شبابيك البنوك يوم السبت 15 ماي 2021، وما العمليات المسموح بها؟", "تفتح من الساعة التاسعة صباحا إلى منتصف النهار، وتقتصر العمليات على التنزيل نقدا والسحب والصرف اليدوي."),
        ("Note_2021_17_ar.pdf", 1, "effective_date", "متى بدأ تداول الورقة النقدية السعودية الجديدة من فئة 200 ريال؟", "بدأ تداولها في 25 فيفري 2021."),
        ("Note_2021_18_ar.pdf", 1, "deadline_or_duration", "إلى متى قبل البنك المركزي التونسي أوراق السلسلة الثامنة للفرنك السويسري بعد سحبها من التداول، وهل بقي استبدالها ممكنا في سويسرا؟", "قبلها البنك المركزي التونسي إلى غاية 30 جوان 2021، وبقي استبدالها لدى البنك الوطني السويسري ممكنا لمدة غير محدودة."),
        ("Note_2021_19_ar.pdf", 1, "definition", "ما الرمز المسند إلى مؤسسة الدفع المقيمة VIAMOBILE في مصنفة الوسطاء المقبولين والمؤسسات المالية؟", "الرمز المسند إليها هو 80."),
        ("Note_2021_20_ar.pdf", 1, "required_action", "ما المطلوب من البنوك والبريد لضمان خدمات السحب والدفع في المناطق ذات الإصابات المرتفعة بكوفيد-19 خلال حجر جوان 2021؟", "يجب ضمان التزويد المستمر للموزعات وجاهزية منصات الدفع الإلكتروني، واتخاذ التدابير اللازمة لمعالجة الأعطال والانقطاعات في أسرع وقت."),
        ("Note_2021_21_ar.pdf", 1, "effective_date", "متى بدأ بنك إنجلترا تداول الورقة الجديدة المصنوعة من البوليمر من فئة 50 جنيها إسترلينيا؟", "بدأ تداولها في 23 جوان 2021."),
        ("Note_2021_22_ar.pdf", 1, "required_action", "ما التدابير المطلوبة لضمان السحب والدفع الإلكتروني خلال عطلة عيد الأضحى وفترات الحجر في جويلية 2021؟", "يجب ضمان التزويد المستمر للموزعات وجاهزية منصات الدفع الإلكتروني، ومعالجة أي عطل أو انقطاع في أسرع وقت."),
        ("Note_2021_23_ar.pdf", 1, "definition", "ما الرمز المخصص لشركة BUTAGAZ Tunisie كمصدر للاقتطاعات البنكية والبريدية؟", "الرمز المخصص لها هو 0120."),
        ("Note_2021_25_ar.pdf", 1, "deadline_or_duration", "ما توقيت فتح شبابيك البنوك يوم السبت 14 أوت 2021، وما العمليات المسموح بها؟", "تفتح من الساعة التاسعة صباحا إلى منتصف النهار، وتقتصر العمليات على التنزيل نقدا والسحب والصرف اليدوي."),
        ("Note_2021_26_ar.pdf", 1, "definition", "ما الرمز المخصص لشركة Société Emna de Sports et Loisir كمصدر للاقتطاعات البنكية والبريدية؟", "الرمز المخصص لها هو 0121."),
        ("Note_2021_27_ar.pdf", 2, "eligibility_or_scope", "ما التعريفات الديوانية لأجهزة الاستقبال التلفزية، وما الشركتان المسجلتان لتوريدها في سبتمبر 2021؟", "التعريفات هي 85287260015 و85287240095 و85287240017 و85287230091 و85287230013 و85287220097 و85287280091 و85287280013 و85287260093، والشركتان هما SODIG وGAN DISTRIBUTION."),
        ("Note_2021_29_ar.pdf", 1, "deadline_or_duration", "ما توقيت فتح شبابيك البنوك يوم السبت 16 أكتوبر 2021، وما العمليات المسموح بها؟", "تفتح من الساعة التاسعة صباحا إلى منتصف النهار، وتقتصر العمليات على التنزيل نقدا والسحب والصرف اليدوي."),
        ("Note_2021_30_ar.pdf", 1, "effective_date", "متى بدأ البنك الملكي الاسكتلندي تداول ورقته الجديدة من فئة 50 جنيها إسترلينيا؟", "بدأ تداولها في 18 أوت 2021."),
        ("Note_2021_31_ar.pdf", 1, "effective_date", "متى بدأ بنك اسكتلندا تداول ورقته الجديدة من فئة 50 جنيها إسترلينيا؟", "بدأ تداولها في 1 جويلية 2021."),
        ("Note_2021_32_ar.pdf", 1, "definition", "ما الرمز المخصص لمجمع المحاسبين بالبلاد التونسية كمصدر للاقتطاعات البنكية والبريدية؟", "الرمز المخصص له هو 0122."),
        ("Note_2021_34_ar.pdf", 1, "required_action", "ما التدابير المطلوبة لضمان استمرارية السحب من الموزعات خلال أيام عيد الثورة من 17 إلى 19 ديسمبر 2021؟", "يجب ضمان التزويد المستمر للموزعات وجاهزية منصات الدفع الإلكتروني، ومعالجة الأعطال والانقطاعات سريعا، وإعلام البنك المركزي بالتدابير والمستجدات المؤثرة في أنشطة الدفع."),
    ]
    for source, page, category, query, answer in arabic_rows:
        curated.append(manual(source, page, "ar", category, query, answer))

    curated.extend(
        [
            manual(
                "Note_2021_33_ar.pdf",
                2,
                "ar",
                "deadline_or_duration",
                "ما آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من الجفاف خلال الموسم الفلاحي 2020-2021؟",
                "آخر أجل هو موفى ديسمبر 2021.",
                "يقدم مطلب الجدولة من طرف الفلاح مباشرة إلى فرع البنك الممول للقرض في أجل لا يتعدى موفى ديسمبر 2021.",
                "visual_review",
            ),
            manual(
                "Note_2021_33_ar.pdf",
                2,
                "ar",
                "deadline_or_duration",
                "ما المدة القصوى لجدولة ديون الفلاحين المتضررين من الجفاف لموسم 2020-2021؟",
                "تتم الجدولة حالة بحالة على مدة لا تتجاوز خمس سنوات، مع الحفاظ على نسبة الفائدة الأصلية ومراعاة قدرة الفلاح على السداد ونسبة الضرر.",
                "تتم جدولة القروض على مدة لا تتجاوز خمس سنوات وبنفس نسبة الفائدة التي منحت بها، حالة بحالة، مع الأخذ بعين الاعتبار قدرة المنتفع على التسديد ونسبة الضرر من الجفاف.",
                "visual_review",
            ),
        ]
    )

    output = stable_ids(curated)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
