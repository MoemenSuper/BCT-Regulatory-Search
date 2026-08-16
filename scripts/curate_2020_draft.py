from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from curate_2021_draft import full_page, manual, stable_ids


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("queries", []) if isinstance(data, dict) else data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("french", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    curated: list[dict] = []
    for candidate in load_items(args.french):
        item = deepcopy(candidate)
        source = item["expected_source"]
        page = item["expected_page"]
        item["language"] = "fr"
        item["relevant"] = True
        item["evidence_method"] = "text_extraction"
        if source == "Cir_2020_02_fr.pdf" and page == 4:
            item["query"] = (
                "Une vente avec un délai de paiement de 200 jours nécessite-t-elle "
                "l'autorisation préalable de la Banque Centrale de Tunisie ?"
            )
            item["expected_answer"] = "Oui. Un délai de paiement de 200 jours nécessite son autorisation préalable."
            item["evidence_quote"] = "Délai de paiement de 200 jours : autorisation préalable de la Banque Centrale de Tunisie requise"
            item["evidence_method"] = "visual_review"
        elif source == "Cir_2020_11_fr.pdf" and page == 16:
            item["query"] = (
                "Dans quel délai les établissements et le switch mobile doivent-ils déclarer "
                "les incidents de fraude, d'intrusion ou d'usage abusif à la Banque Centrale de Tunisie ?"
            )
            item["expected_answer"] = "Au plus tard à la fin du mois suivant la clôture de chaque trimestre."
        elif source == "Note_2020_23_fr.pdf":
            item["query"] = (
                "Une banque qui satisfait aux critères d'éligibilité peut-elle demander le statut "
                "de participant direct à Buna et négocier les modalités avec son gestionnaire ?"
            )
            item["expected_answer"] = "Oui, elle peut postuler à ce statut et négocier directement avec le gestionnaire de Buna."
        curated.append(item)

    curated.extend(
        [
            manual(
                "Cir_2020_03_fr.pdf", 2, "fr", "amount_or_rate",
                "Quel montant un exportateur peut-il transférer chaque année sur son compte professionnel à l'étranger ?",
                "Il peut transférer jusqu'à 25 % des recettes d'exportation rapatriées, dans la limite de 500 000 dinars par année civile.",
            ),
            manual(
                "Cir_2020_15_fr.pdf", 1, "fr", "effective_date",
                "Les mesures temporaires de tarification et de continuité bancaire adoptées le 19 mars 2020 sont-elles toutes restées en vigueur après le 19 juin 2020 ?",
                "Non. Les dispositions correspondant aux articles 2 et 4 ont été abrogées dès la publication du texte du 19 juin 2020.",
            ),
            manual(
                "Cir_2020_18_fr.pdf", 2, "fr", "amount_or_rate",
                "Quels plafonds de crédit par hectare s'appliquaient en 2020 au blé dur, au blé tendre et aux légumineuses en sec et en irrigué, et à quelle échéance ?",
                "En sec, le plafond est de 1 070 dinars en zone 1 et de 875 dinars en zone 2. En irrigué, il est de 1 380 dinars dans les deux zones. L'échéance est le 31 août.",
                "Blé dur, blé tendre et légumineuses | sec zone 1 : 1070 DT/ha | sec zone 2 : 875 DT/ha | irrigué : 1380 DT/ha | échéance : 31 août",
                "visual_review",
            ),
            manual(
                "Cir_2020_18_fr.pdf", 2, "fr", "amount_or_rate",
                "Quels plafonds de crédit par hectare s'appliquent à l'orge en sec dans les zones 1, 2 et 3, et à quelle échéance ?",
                "Les plafonds sont de 645 dinars en zone 1, 605 dinars en zone 2 et 230 dinars en zone 3. L'échéance est le 31 août.",
                "Orge en sec | zone 1 : 645 DT/ha | zone 2 : 605 DT/ha | zone 3 : 230 DT/ha | échéance : 31 août",
                "visual_review",
            ),
            manual(
                "Cir_2020_18_fr.pdf", 2, "fr", "amount_or_rate",
                "Quels plafonds et échéances s'appliquaient en 2020 aux fourrages d'hiver en sec et aux fourrages d'été en irrigué ?",
                "Le plafond est de 750 dinars par hectare pour les fourrages d'hiver en sec, à rembourser le 31 août, et de 1 085 dinars pour les fourrages d'été en irrigué, à rembourser le 30 septembre.",
                "Fourrages d'hiver en sec : 750 DT/ha, 31 août | fourrages d'été en irrigué : 1085 DT/ha, 30 septembre",
                "visual_review",
            ),
            manual(
                "Cir_2020_19_fr.pdf", 2, "fr", "deadline_or_duration",
                "Jusqu'à quelle date les échéances de crédit des entreprises et professionnels du tourisme et de l'artisanat pouvaient-elles être reportées ?",
                "Elles pouvaient être reportées jusqu'à la fin de septembre 2021.",
            ),
        ]
    )

    # One realistic retrieval target for every Arabic source from 2020.  Text-extracted
    # pages are retained verbatim as evidence; visually reviewed rows use a concise
    # transcription because the PDF's embedded OCR corrupts Arabic digits or is absent.
    rows = [
        ("Cir_2020_06_ar.pdf", 2, "deadline_or_duration",
         "إلى أي تاريخ كان على البنوك تأجيل أقساط قروض المؤسسات والمهنيين المتضررين من جائحة كورونا؟",
         "كان التأجيل يشمل الأقساط المستحقة من غرة مارس إلى موفى سبتمبر 2020، مع تعديل جدول السداد تبعا لذلك.",
         "تأجيل أقساط قروض المؤسسات والمهنيين المستحقة من 1 مارس إلى 30 سبتمبر 2020", "visual_review"),
        ("Cir_2020_07_ar.pdf", 2, "eligibility_or_scope",
         "من هم الأفراد الذين شملهم التأجيل الاستثنائي لأقساط القروض خلال أزمة كورونا؟",
         "شمل الأفراد الذين يقل دخلهم الشهري الصافي عن ألف دينار، بالنسبة إلى الأقساط المستحقة من غرة مارس إلى موفى سبتمبر 2020.",
         "الأفراد الذين يقل دخلهم الشهري الصافي عن 1000 دينار | الأقساط من 1 مارس إلى 30 سبتمبر 2020", "visual_review"),
        ("Cir_2020_08_ar.pdf", 2, "procedure_or_documents",
         "كيف كان يمكن للحريف رفض الانتفاع بالتأجيل الاستثنائي لأقساط قرضه؟",
         "كان عليه إعلام البنك بأي وسيلة تترك أثرا كتابيا.", None, "text_extraction"),
        ("Cir_2020_09_ar.pdf", 2, "eligibility_or_scope",
         "هل كان يمكن للطالب أو المتكوّن بالخارج طلب تحويل مسبق لمصاريف الإقامة خلال أزمة كورونا؟",
         "نعم. كان على الوسيط المقبول، بطلب من المنتفع، إنجاز تحويل مسبق لمصاريف الإقامة الخاصة بشهري ماي وجوان 2020.", None, "text_extraction"),
        ("Cir_2020_21_ar.pdf", 2, "deadline_or_duration",
         "إلى أي تاريخ مُدّد إسناد التمويلات الاستثنائية للمؤسسات والمهنيين المتضررين من الجائحة؟",
         "مُدّد أجل إسنادها إلى موفى ديسمبر 2021.", None, "text_extraction"),
        ("Note_2020_02_ar.pdf", 1, "effective_date",
         "متى بدأ تداول الورقة النرويجية الجديدة من فئة 1000 كرونة؟",
         "بدأ تداولها في 14 نوفمبر 2019.", "ورقة نرويجية جديدة من فئة 1000 كرونة | بداية التداول 14 نوفمبر 2019", "visual_review"),
        ("Note_2020_03_ar.pdf", 1, "definition",
         "ما الرمز المسند إلى ASSOCIATION SOCIOS CAB كمصدر للاقتطاعات البنكية والبريدية؟",
         "الرمز هو 0107.", None, "text_extraction"),
        ("Note_2020_04_ar.pdf", 2, "amount_or_rate",
         "ما مبالغ القروض التكميلية للهكتار من الحبوب لموسم 2019-2020، وما أجل سدادها؟",
         "المبلغ 233 دينارا للحبوب المروية، و233 دينارا للحبوب البعلية في المنطقة 1، و213 دينارا للحبوب البعلية في المنطقة 2، وأجل السداد 31 أوت 2020.",
         "حبوب مروية 233 د/هك | حبوب بعلية منطقة 1: 233 د/هك | منطقة 2: 213 د/هك | 31 أوت 2020", "visual_review"),
        ("Note_2020_05_ar.pdf", 1, "definition",
         "ما الرمز المسند إلى Société Slim Fit كمصدر للاقتطاعات البنكية والبريدية؟",
         "الرمز هو 0108.", None, "text_extraction"),
        ("Note_2020_06_ar.pdf", 1, "definition",
         "ما الاسم والرمز الجديدان لشركة Modern Leasing في مصنفة المؤسسات المالية؟",
         "أصبح اسمها BH Leasing ورمزها BHL، مع الإبقاء على الرمز العددي 42.", None, "text_extraction"),
        ("Note_2020_07_ar.pdf", 2, "eligibility_or_scope",
         "ما نطاق التعريفات الديوانية لأجهزة الاستقبال التلفزيوني الخاضعة لقائمة الموردين المسجلين؟",
         "يمتد النطاق من التعريفة 85287220097 إلى التعريفة 85287280091.", None, "text_extraction"),
        ("Note_2020_08_ar.pdf", 1, "effective_date",
         "متى بدأ تداول الأوراق النقدية الصينية الجديدة من فئات 1 و10 و20 و50 يوانا؟",
         "بدأ تداولها في 30 أوت 2019.", None, "text_extraction"),
        ("Note_2020_09_ar.pdf", 1, "effective_date",
         "متى طُرحت الورقة الصينية الجديدة من فئة 100 يوان، وهل عوضت الورقة السابقة فورا؟",
         "طُرحت في ديسمبر 2018، وتداولت بالتوازي مع الورقة السابقة من الفئة نفسها.", None, "text_extraction"),
        ("Note_2020_10_ar.pdf", 1, "effective_date",
         "متى بدأ بنك إنجلترا تداول ورقته المصنوعة من البوليمر من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 20 فيفري 2020.", None, "text_extraction"),
        ("Note_2020_11_ar.pdf", 1, "definition",
         "ما الرمز المسند إلى الشركة التونسية لصناعات التكرير كمصدر للاقتطاعات البنكية والبريدية؟",
         "الرمز هو 0109.", None, "text_extraction"),
        ("Note_2020_12_ar.pdf", 1, "deadline_or_duration",
         "إلى أي تاريخ واصل البنك المركزي التونسي قبول الأوراق القديمة من فئتي 5 و10 جنيهات الصادرة عن Bank of Ireland؟",
         "واصل قبولها إلى غاية 31 مارس 2020 بدخول الغاية.",
         "Bank of Ireland | فئتا 5 و10 جنيهات | قبول الأوراق القديمة إلى 31 مارس 2020", "visual_review"),
        ("Note_2020_13_ar.pdf", 1, "deadline_or_duration",
         "إلى أي تاريخ واصل البنك المركزي التونسي قبول ورقة Danske Bank القديمة من فئة 10 جنيهات؟",
         "واصل قبولها إلى غاية 31 مارس 2020 بدخول الغاية.",
         "Danske Bank | فئة 10 جنيهات | قبول الورقة القديمة إلى 31 مارس 2020", "visual_review"),
        ("Note_2020_14_ar.pdf", 1, "deadline_or_duration",
         "إلى أي تاريخ واصل البنك المركزي التونسي قبول أوراق Ulster Bank القديمة من فئتي 5 و10 جنيهات؟",
         "واصل قبولها إلى غاية 31 مارس 2020 بدخول الغاية.",
         "Ulster Bank | فئتا 5 و10 جنيهات | قبول الأوراق القديمة إلى 31 مارس 2020", "visual_review"),
        ("Note_2020_15_ar.pdf", 1, "required_action",
         "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة عيد الاستقلال؟",
         "كان عليها اتخاذ الإجراءات اللازمة لضمان استمرارية السحب من الموزعات الآلية للأوراق النقدية طوال العطلة.", None, "text_extraction"),
        ("Note_2020_18_ar.pdf", 1, "required_action",
         "ما المطلوب من البنوك والبريد خلال العطلة الاستثنائية ليوم 10 أفريل 2020؟",
         "كان عليها ضمان استمرارية عمليات السحب من الموزعات الآلية للأوراق النقدية خلال العطلة.",
         "العطلة الاستثنائية ليوم 10 أفريل 2020 | تأمين استمرارية السحب من الموزعات الآلية", "visual_review"),
        ("Note_2020_19_ar.pdf", 1, "required_action",
         "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة عيد الشغل؟",
         "كان عليها ضمان استمرارية عمليات السحب من الموزعات الآلية للأوراق النقدية طوال العطلة.", None, "text_extraction"),
        ("Note_2020_20_ar.pdf", 1, "definition",
         "ما الرمزان المسندان إلى BH ASSURANCE وSociété ENERGYM كمصدرين للاقتطاعات؟",
         "الرمز 0110 مسند إلى BH ASSURANCE، والرمز 0111 مسند إلى Société ENERGYM.", None, "text_extraction"),
        ("Note_2020_22_ar.pdf", 1, "required_action",
         "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة عيد الفطر لسنة 2020؟",
         "كان عليها ضمان استمرارية عمليات السحب من الموزعات الآلية للأوراق النقدية طوال العطلة.", None, "text_extraction"),
        ("Note_2020_24_ar.pdf", 1, "required_action",
         "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة عيد الأضحى؟",
         "كان عليها ضمان استمرارية عمليات السحب من الموزعات الآلية للأوراق النقدية طوال العطلة.", None, "text_extraction"),
        ("Note_2020_25_ar.pdf", 2, "eligibility_or_scope",
         "ما الشركتان المسجلتان لتوريد المربعات الخزفية الخاضعة لكراس الشروط؟",
         "هما شركة STAR WORLD TRADING وشركة CHATTI DE COMMERCE.", None, "text_extraction"),
        ("Note_2020_27_ar.pdf", 1, "definition",
         "ما الرمز المسند إلى LLOYD VIE كمصدر للاقتطاعات البنكية والبريدية؟",
         "الرمز هو 0112.", None, "text_extraction"),
        ("Note_2020_29_ar.pdf", 1, "deadline_or_duration",
         "إلى أي تاريخ قبل البنك المركزي التونسي ورقة النرويج القديمة من فئة 1000 كرونة بعد إعلان سحبها؟",
         "واصل قبولها إلى غاية 30 أكتوبر 2020 بدخول الغاية، وبقي استبدالها لدى بنك النرويج ممكنا لمدة لا تقل عن عشر سنوات.",
         "ورقة 1000 كرونة نرويجية | قبول البنك المركزي التونسي إلى 30 أكتوبر 2020 | الاستبدال في النرويج لمدة لا تقل عن 10 سنوات", "visual_review"),
        ("Note_2020_30_ar.pdf", 1, "effective_date",
         "متى بدأ بنك اسكتلندا تداول ورقته الجديدة من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 27 فيفري 2020.", "Bank of Scotland | 20 جنيها إسترلينيا | بداية التداول 27 فيفري 2020", "visual_review"),
        ("Note_2020_31_ar.pdf", 1, "effective_date",
         "متى بدأ Clydesdale Bank تداول ورقته الجديدة من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 27 فيفري 2020.", "Clydesdale Bank | 20 جنيها إسترلينيا | بداية التداول 27 فيفري 2020", "visual_review"),
        ("Note_2020_32_ar.pdf", 1, "effective_date",
         "متى بدأ البنك الملكي الاسكتلندي تداول ورقته الجديدة من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 5 مارس 2020.", "Royal Bank of Scotland | 20 جنيها إسترلينيا | بداية التداول 5 مارس 2020", "visual_review"),
        ("Note_2020_34_ar.pdf", 1, "definition",
         "ما الرمزان المسندان إلى GALAXY GYM وHOPE HEALTHCARE SOLUTIONS كمصدرين للاقتطاعات؟",
         "الرمز 0113 مسند إلى GALAXY GYM، والرمز 0114 مسند إلى HOPE HEALTHCARE SOLUTIONS.",
         "0113 GALAXY GYM | 0114 HOPE HEALTHCARE SOLUTIONS", "visual_review"),
        ("Note_2020_35_ar.pdf", 1, "effective_date",
         "متى بدأ Ulster Bank تداول ورقته الجديدة من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 14 أكتوبر 2020.", "Ulster Bank | 20 جنيها إسترلينيا | بداية التداول 14 أكتوبر 2020", "visual_review"),
        ("Note_2020_36_ar.pdf", 1, "effective_date",
         "متى بدأ Danske Bank تداول ورقته الجديدة من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 20 جويلية 2020.", "Danske Bank | 20 جنيها إسترلينيا | بداية التداول 20 جويلية 2020", "visual_review"),
        ("Note_2020_37_ar.pdf", 1, "effective_date",
         "متى بدأ Bank of Ireland تداول ورقته الجديدة من فئة 20 جنيها إسترلينيا؟",
         "بدأ تداولها في 20 جويلية 2020.", "Bank of Ireland | 20 جنيها إسترلينيا | بداية التداول 20 جويلية 2020", "visual_review"),
        ("Note_2020_39_ar.pdf", 2, "deadline_or_duration",
         "ما آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من الجفاف في موسم 2019-2020، وما المدة القصوى للجدولة؟",
         "آخر أجل هو موفى ديسمبر 2020، وتتم الجدولة على مدة لا تتجاوز خمس سنوات.",
         "مطلب الجدولة إلى موفى ديسمبر 2020 | مدة الجدولة القصوى خمس سنوات", "visual_review"),
        ("Note_2020_40_ar.pdf", 1, "definition",
         "ما الرمز المسند إلى مصرف البركة تونس كمصدر للاقتطاعات البنكية والبريدية؟",
         "الرمز هو 0115.", None, "text_extraction"),
        ("Note_2020_41_ar.pdf", 1, "required_action",
         "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة رأس السنة الإدارية؟",
         "كان عليها ضمان استمرارية عمليات السحب من الموزعات الآلية للأوراق النقدية طوال العطلة.", None, "text_extraction"),
        ("Note_2020_42_ar.pdf", 1, "definition",
         "ما الرمز المسند إلى بنك تمويل المؤسسات الصغرى والمتوسطة كمصدر للاقتطاعات البنكية والبريدية؟",
         "الرمز هو 0116.", None, "text_extraction"),
    ]

    for source, page, category, query, answer, evidence, method in rows:
        if evidence is None:
            evidence = full_page(source, page)
        curated.append(manual(source, page, "ar", category, query, answer, evidence, method))

    output = stable_ids(curated)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
