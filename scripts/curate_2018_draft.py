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
    parser.add_argument("draft", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    replaced = {
        "Cir_2018_03_ar.pdf", "Cir_2018_05_ar.pdf", "Cir_2018_11_fr.pdf",
        "Cir_2018_13_fr.pdf",
    }
    curated: list[dict] = []
    for candidate in load_items(args.draft):
        source = candidate["expected_source"]
        if source.startswith("Note_") or source in replaced:
            continue
        item = deepcopy(candidate)
        item["language"] = "ar" if source.endswith("_ar.pdf") else "fr"
        item["relevant"] = True
        item["evidence_method"] = "text_extraction"
        if source == "Cir_2018_01_fr.pdf" and item["category"] == "exception_or_condition":
            item["query"] = "Quelles importations de produits non prioritaires échappaient aux restrictions de financement appliquées en mars 2018 ?"
        elif source == "Cir_2018_01_fr.pdf":
            item["query"] = "Quel justificatif une entreprise industrielle devait-elle remettre à son intermédiaire agréé pour financer l'importation d'un produit non prioritaire lié à son activité ?"
        elif source == "Cir_2018_02_fr.pdf":
            item["query"] = "À partir de quelle date les pièces tunisiennes de 50, 20 et 10 millimes émises en 2018 ont-elles été mises en circulation ?"
            item["expected_answer"] = "Elles ont été mises en circulation le 19 mars 2018."
        elif source == "Cir_2018_04_fr.pdf" and item["category"] == "deadline_or_duration":
            item["query"] = "Pendant combien de temps l'autorisation d'exporter des riyals saoudiens délivrée à un pèlerin inscrit sur les listes officielles restait-elle valable en 2018 ?"
            item["expected_answer"] = "Elle restait valable pendant quatre mois au maximum à compter de sa délivrance."
        elif source == "Cir_2018_04_fr.pdf":
            item["query"] = "À partir de quelle date les guichets uniques pouvaient-ils délivrer l'allocation touristique pour le pèlerinage de 2018 ?"
            item["expected_answer"] = "Ils pouvaient la délivrer à partir du 14 avril 2018."
        elif source == "Cir_2018_06_fr.pdf" and item["category"] == "amount_or_rate":
            item["expected_answer"] = "Le plafond était de 1,25 % des risques de crédit pondérés."
        elif source == "Cir_2018_06_fr.pdf":
            item["query"] = "Quels éléments entraient dans le calcul des fonds propres nets de base d'une banque selon les normes d'adéquation de 2018 ?"
        elif source == "Cir_2018_07_fr.pdf" and item["category"] == "amount_or_rate":
            item["expected_answer"] = "Une garantie bancaire à première demande de 50 000 dinars était exigée."
        elif source == "Cir_2018_07_fr.pdf":
            item["expected_answer"] = "La décision devait être notifiée dans les deux mois suivant la réception de la demande."
        elif source == "Cir_2018_08_fr.pdf":
            item["expected_answer"] = "Elles devaient vérifier à tout moment que la répartition respectait les quotités publiées par la Banque Centrale sur CAER et sur son site web."
        elif source == "Cir_2018_09_fr.pdf":
            item["query"] = "À partir de quel montant une opération en billets de banque étrangers devait-elle être déclarée à la Banque Centrale dans le dispositif antiblanchiment de 2018 ?"
            item["expected_answer"] = "La déclaration était requise à partir de 5 000 dinars tunisiens."
        elif source == "Cir_2018_10_fr.pdf" and item["category"] == "required_action":
            item["query"] = "Quelle réduction une banque devait-elle appliquer lorsque son ratio crédits/dépôts se situait entre 120 % et 122 % à la fin d'un trimestre ?"
            item["expected_answer"] = "Elle devait appliquer la réduction nécessaire pour ramener le ratio du trimestre suivant à 120 %."
        elif source == "Cir_2018_10_fr.pdf":
            item["expected_answer"] = "Elle devait transmettre son plan d'actions au plus tard dix jours après la déclaration du trimestre concerné."
        elif source == "Cir_2018_12_fr.pdf" and item["category"] == "reporting_or_control":
            item["expected_answer"] = "Les banques devaient afficher en continu leurs conditions indicatives d'offre et de demande de liquidité en dinars, pour des maturités allant du jour le jour à un an."
        elif source == "Cir_2018_12_fr.pdf":
            item["expected_answer"] = "Le TUNIBOR est le taux moyen auquel les banques contributrices sont disposées à se prêter sans garantie, pour les maturités d'une semaine, deux semaines, un, deux, trois, six, neuf et douze mois."
        elif source == "Cir_2018_14_fr.pdf":
            item["expected_answer"] = "Un apport autre qu'en devises nécessitait l'autorisation préalable de la Banque Centrale de Tunisie."
        elif source == "Cir_2018_15_fr.pdf" and item["category"] == "eligibility_or_scope":
            item["expected_answer"] = "La banque correspondante devait avoir une notation à court terme d'au moins A-2 chez Standard & Poor's ou une note équivalente chez Moody's ou Fitch."
        elif source == "Cir_2018_15_fr.pdf":
            item["expected_answer"] = "Oui. Ils pouvaient employer sans autorisation préalable les devises non cessibles de leur clientèle dans les emplois autorisés."
        elif source == "Cir_2018_16_fr.pdf" and item["category"] == "required_action":
            item["expected_answer"] = "Il devait souscrire une assurance de responsabilité civile professionnelle ou une garantie bancaire."
        elif source == "Cir_2018_16_fr.pdf":
            item["expected_answer"] = "Les services devaient être fournis exclusivement en dinars tunisiens et sur le territoire tunisien."
        elif source == "NB-2018_28_1110.pdf":
            item["query"] = "À partir de quand la Société Tunisienne de Banque pouvait-elle exercer en qualité de teneur de marché sur le marché des changes ?"
            item["expected_answer"] = "Ce statut prenait effet à compter de la notification de la décision."
        curated.append(item)

    curated.extend(
        [
            manual("Cir_2018_03_ar.pdf", 1, "ar", "prohibition_or_limit",
                   "كم حسابا بنكيا يمكن فتحه لكل قائمة مترشحة للانتخابات البلدية لسنة 2018؟",
                   "يمكن فتح حساب وحيد لكل قائمة مترشحة، ويُحجّر فتح أكثر من حساب لها.",
                   "الانتخابات البلدية لسنة 2018 | حساب بنكي وحيد لكل قائمة مترشحة | يحجر فتح أكثر من حساب", "visual_review"),
            manual("Cir_2018_03_ar.pdf", 2, "ar", "procedure_or_documents",
                   "ما الوثائق الأساسية المطلوبة لفتح حساب الحملة الانتخابية لقائمة مترشحة للانتخابات البلدية لسنة 2018؟",
                   "يلزم قرار قبول ترشح القائمة، ووثيقة هوية رئيسها، وقرار تكليف الوكيل المالي، ووثيقة هوية الوكيل المالي.",
                   "فتح حساب الحملة | قرار قبول ترشح القائمة | هوية رئيس القائمة | قرار تكليف الوكيل المالي | هوية الوكيل المالي", "visual_review"),
            manual("Cir_2018_03_ar.pdf", 3, "ar", "deadline_or_duration",
                   "في أي أجل كان يجب إغلاق حساب الحملة الانتخابية البلدية لسنة 2018، وكم تدوم مدة حفظ ملفاته؟",
                   "كان يجب إغلاقه في أجل أقصاه 15 يوما من تاريخ غلق الحملة، مع حفظ الملف والوثائق لمدة عشر سنوات من تاريخ إغلاقه.",
                   "غلق حساب الحملة في أجل أقصاه 15 يوما من غلق الحملة | حفظ الملف والوثائق 10 سنوات", "visual_review"),
            manual("Cir_2018_05_ar.pdf", 2, "ar", "amount_or_rate",
                   "ما الحد الأقصى لتمويل دراسة التشخيص المالي والاقتصادي ومرافقة مؤسسة صغرى أو متوسطة ضمن خط إعادة الهيكلة؟",
                   "الحد الأقصى هو 30 ألف دينار للمؤسسة الواحدة.",
                   "دراسة التشخيص والمرافقة | حد أقصى 30 ألف دينار للمؤسسة الواحدة", "visual_review"),
            manual("Cir_2018_05_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "ما نطاق الأصول الثابتة الخام للمؤسسة الصغرى أو المتوسطة المؤهلة لخط دعم إعادة الهيكلة المالية؟",
                   "يجب أن يتراوح حجم الأصول الثابتة الخام بين 100 ألف دينار و30 مليون دينار.",
                   "الأصول الثابتة الخام | من 100 ألف دينار إلى 30 مليون دينار", "visual_review"),
            manual("Cir_2018_05_ar.pdf", 3, "ar", "deadline_or_duration",
                   "في أي أجل كان على البنك المترئس للمجموعة إبداء رأيه في انخراط مؤسسة في خط إعادة الهيكلة، وماذا يحدث عند تجاوزه؟",
                   "كان عليه الرد في أجل أقصاه خمسة أيام عمل من تسلم المطلب، ويُعد تجاوز الأجل موافقة ضمنية.",
                   "رأي البنك المترئس | أجل أقصاه 5 أيام عمل | تجاوز الأجل موافقة ضمنية", "visual_review"),
            manual("Cir_2018_11_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds de crédit par hectare s'appliquaient au blé dur, au blé tendre et aux légumineuses en sec et en irrigué en 2018 ?",
                   "En sec, le plafond était de 910 dinars en zone 1 et de 755 dinars en zone 2. En irrigué, il était de 1 285 dinars dans les deux zones, avec une échéance au 31 août.",
                   "Blé dur, blé tendre et légumineuses | sec zone 1 : 910 DT/ha | sec zone 2 : 755 DT/ha | irrigué : 1285 DT/ha | 31 août", "visual_review"),
            manual("Cir_2018_11_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds de crédit par hectare s'appliquaient à l'orge en sec dans les zones 1, 2 et 3 en 2018 ?",
                   "Les plafonds étaient de 555 dinars en zone 1, 525 dinars en zone 2 et 210 dinars en zone 3, avec une échéance au 31 août.",
                   "Orge en sec | zone 1 : 555 DT/ha | zone 2 : 525 DT/ha | zone 3 : 210 DT/ha | 31 août", "visual_review"),
            manual("Cir_2018_11_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds et échéances s'appliquaient aux fourrages d'hiver en sec et aux fourrages d'été en irrigué en 2018 ?",
                   "Le plafond était de 630 dinars par hectare pour les fourrages d'hiver en sec, à échéance du 31 août, et de 1 005 dinars pour les fourrages d'été en irrigué, à échéance du 30 septembre.",
                   "Fourrages d'hiver en sec : 630 DT/ha, 31 août | fourrages d'été en irrigué : 1005 DT/ha, 30 septembre", "visual_review"),
            manual("Cir_2018_13_fr.pdf", 2, "fr", "effective_date",
                   "Les restrictions de financement des importations de produits non prioritaires adoptées en octobre 2017 restaient-elles applicables après leur abrogation de décembre 2018 ?",
                   "Non. Le dispositif d'octobre 2017 a été abrogé, l'abrogation prenant effet à la notification du texte de décembre 2018."),
            manual("Note_2018_01_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى شركة البحر المتوسط للتأمين وإعادة التأمين COMAR كمصدر للاقتطاعات؟",
                   "الرمز المسند إليها هو 0096.", "COMAR | الرمز 0096", "visual_review"),
            manual("Note_2018_02_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى ودادية أعوان الديوان الوطني للتطهير AMICALE ONAS كمصدر للاقتطاعات؟",
                   "الرمز المسند إليها هو 0097.", "AMICALE ONAS | الرمز 0097", "visual_review"),
            manual("Note_2018_03_ar.pdf", 2, "ar", "amount_or_rate",
                   "كيف حُدّد مبلغ القرض التكميلي لزراعة الحبوب في موسم 2017-2018، وما أجل سداده؟",
                   "حُدّد في حدود 80 % من الكلفة المرجعية المنشورة من وزارة الفلاحة، وكان أجل السداد 31 أوت 2018.",
                   "قرض تكميلي لموسم الحبوب 2017-2018 | 80% من الكلفة المرجعية | أجل السداد 31 أوت 2018", "visual_review"),
            manual("Note_2018_04_ar.pdf", 1, "ar", "effective_date",
                   "متى سحب بنك إنجلترا ورقة تشارلز داروين من فئة 10 جنيهات إسترلينية من التداول؟",
                   "سحبها من التداول ابتداء من غرة مارس 2018.",
                   "Bank of England | ورقة Charles Darwin فئة 10 جنيهات | السحب من التداول 1 مارس 2018", "visual_review"),
            manual("Note_2018_05_ar.pdf", 1, "ar", "effective_date",
                   "متى سحب Bank of Scotland أوراقه القديمة من فئتي 5 و10 جنيهات إسترلينية من التداول؟",
                   "سحبها من التداول ابتداء من غرة مارس 2018.",
                   "Bank of Scotland | فئتا 5 و10 جنيهات | السحب من التداول 1 مارس 2018", "visual_review"),
            manual("Note_2018_06_ar.pdf", 1, "ar", "effective_date",
                   "متى سحب Clydesdale Bank أوراقه القديمة من فئتي 5 و10 جنيهات إسترلينية من التداول؟",
                   "سحبها من التداول ابتداء من غرة مارس 2018.",
                   "Clydesdale Bank | فئتا 5 و10 جنيهات | السحب من التداول 1 مارس 2018", "visual_review"),
            manual("Note_2018_07_ar.pdf", 1, "ar", "effective_date",
                   "متى سحب Royal Bank of Scotland أوراقه القديمة من فئتي 5 و10 جنيهات إسترلينية من التداول؟",
                   "سحبها من التداول ابتداء من غرة مارس 2018.",
                   "Royal Bank of Scotland | فئتا 5 و10 جنيهات | السحب من التداول 1 مارس 2018", "visual_review"),
            manual("Note_2018_11_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة عيد الشهداء؟",
                   "كان عليها اتخاذ الإجراءات اللازمة لضمان استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
            manual("Note_2018_12_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى نقابة موظفي الإدارة العامة لوحدات التدخل كمصدر للاقتطاعات؟",
                   "الرمز المسند إليها هو 0100.", "نقابة موظفي الإدارة العامة لوحدات التدخل | الرمز 0100", "visual_review"),
            manual("Note_2018_13_ar.pdf", 1, "ar", "eligibility_or_scope",
                   "ما شرط منشأ السلع والخدمات التي كانت المصارف الليبية تمول استيرادها من تونس بالدينار التونسي؟",
                   "كان يجب أن تكون السلع أو الخدمات ذات منشأ تونسي."),
            manual("Note_2018_14_ar.pdf", 1, "ar", "deadline_or_duration",
                   "إلى أي تاريخ واصل البنك المركزي التونسي قبول الورقتين النرويجيتين القديمتين من فئتي 100 و200 كرونة؟",
                   "واصل قبولهما إلى غاية 21 ماي 2018 بدخول الغاية، وبقي استبدالهما لدى بنك النرويج ممكنا لمدة لا تقل عن عشر سنوات.",
                   "ورقتا 100 و200 كرونة | قبول البنك المركزي التونسي إلى 21 ماي 2018 | الاستبدال في النرويج لمدة لا تقل عن 10 سنوات", "visual_review"),
            manual("Note_2018_18_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب خلال عطلة عيد الفطر؟",
                   "كان عليها اتخاذ الإجراءات اللازمة لضمان استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
            manual("Note_2018_20_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الطبعة الصينية الجديدة من الورقة النقدية فئة 1 يوان؟",
                   "بدأ تداولها في نوفمبر 2017، بالتوازي مع الطبعة السابقة إلى حين إشعار لاحق.",
                   "طبعة صينية جديدة فئة 1 يوان | بداية التداول نوفمبر 2017 | التداول بالتوازي مع الطبعة السابقة", "visual_review"),
            manual("Note_2018_21_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ مصرف البحرين المركزي تداول الطبعة الجديدة من فئة 5 دنانير؟",
                   "بدأ تداولها في 6 جوان 2018.",
                   "مصرف البحرين المركزي | فئة 5 دنانير | بداية التداول 6 جوان 2018", "visual_review"),
            manual("Note_2018_22_fr.pdf", 1, "fr", "definition",
                   "Quel code ISO a remplacé l'ancien code MRO 478 pour l'ouguiya mauritanienne en 2018 ?",
                   "La nouvelle abréviation est MRU et le nouveau code est 929."),
            manual("Note_2018_23_ar.pdf", 1, "ar", "deadline_or_duration",
                   "في أي توقيت فتحت البنوك شبابيكها يوم السبت 11 أوت 2018 بمناسبة عطلة عيد المرأة؟",
                   "فتحت من الساعة التاسعة صباحا إلى منتصف النهار، واقتصرت العمليات على التنزيل نقدا والسحب والصرف اليدوي.",
                   "السبت 11 أوت 2018 | من الساعة التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي", "visual_review"),
            manual("Note_2018_24_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الورقة السويسرية من فئة 200 فرنك الصادرة ضمن السلسلة التاسعة؟",
                   "بدأ تداولها في 22 أوت 2018.",
                   "ورقة سويسرية فئة 200 فرنك | بداية التداول 22 أوت 2018", "visual_review"),
            manual("Note_2018_29_ar.pdf", 2, "ar", "deadline_or_duration",
                   "ما آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من الجفاف في موسم 2017-2018، وما المدة القصوى للجدولة؟",
                   "آخر أجل هو موفى ديسمبر 2018، وتتم الجدولة على مدة لا تتجاوز خمس سنوات.",
                   "مطلب الجدولة إلى موفى ديسمبر 2018 | مدة الجدولة القصوى 5 سنوات", "visual_review"),
            manual("Note_2018_30_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك النرويج تداول الأوراق الجديدة من فئتي 50 و500 كرونة؟",
                   "بدأ تداولها في 18 أكتوبر 2018.",
                   "Norges Bank | فئتا 50 و500 كرونة | بداية التداول 18 أكتوبر 2018", "visual_review"),
            manual("Note_2018_31_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ مصرف الإمارات العربية المتحدة المركزي تداول الطبعة الجديدة من فئة 100 درهم؟",
                   "بدأ تداولها في 30 أكتوبر 2018.",
                   "مصرف الإمارات العربية المتحدة المركزي | فئة 100 درهم | بداية التداول 30 أكتوبر 2018", "visual_review"),
            manual("Note_2018_33_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك كندا تداول الورقة الجديدة من فئة 10 دولارات كندية؟",
                   "بدأ تداولها في 19 نوفمبر 2018.",
                   "Banque du Canada | فئة 10 دولارات كندية | بداية التداول 19 نوفمبر 2018", "visual_review"),
        ]
    )

    output = stable_ids(curated)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
