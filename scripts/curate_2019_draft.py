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
        "Cir_2019_12_fr.pdf",
        "Note_2019_05_ar.pdf",
        "Note_2019_07_ar.pdf",
        "Note_2019_09_ar.pdf",
        "Note_2019_10_ar.pdf",
        "Note_2019_11_ar.pdf",
        "Note_2019_13_ar.pdf",
        "Note_2019_15_ar.pdf",
        "Note_2019_18_ar.pdf",
        "Note_2019_19_ar.pdf",
        "Note_2019_21_ar.pdf",
        "Note_2019_23_ar.pdf",
    }
    curated: list[dict] = []
    for candidate in load_items(args.draft):
        source = candidate["expected_source"]
        if source in replaced:
            continue
        item = deepcopy(candidate)
        item["language"] = "ar" if source.endswith("_ar.pdf") else "fr"
        item["relevant"] = True
        item["evidence_method"] = "text_extraction"
        if source == "Cir_2019_01_fr.pdf" and item["category"] == "amount_or_rate":
            item["query"] = "Quel montant maximal peut être retiré en espèces d'un compte startup en devises pour les frais de séjour à l'étranger, par voyage et par bénéficiaire ?"
            item["expected_answer"] = "Le plafond est de 30 000 dinars par voyage et par bénéficiaire."
        elif source == "Cir_2019_01_fr.pdf":
            item["expected_answer"] = "Le débit peut être effectué par virement, chèque bancaire, carte de paiement internationale ou, pour certains frais de séjour, en espèces."
        elif source == "Cir_2019_02_fr.pdf" and "Startup" in item["query"]:
            item["expected_answer"] = "L'allocation annuelle maximale est de 100 000 dinars."
        elif source == "Cir_2019_02_fr.pdf":
            item["expected_answer"] = "L'allocation annuelle maximale est de 10 000 dinars."
        elif source == "Cir_2019_03_ar.pdf" and item["category"] == "prohibition_or_limit":
            item["expected_answer"] = "لا يجوز فتح أكثر من حساب واحد للقائمة المترشحة."
        elif source == "Cir_2019_07_fr.pdf" and item["category"] == "deadline_or_duration":
            item["expected_answer"] = "L'activité doit commencer dans les trois mois suivant la notification de l'autorisation."
        elif source == "Cir_2019_07_fr.pdf":
            item["query"] = "Quel justificatif relatif au local faut-il produire pour ouvrir un bureau de change manuel ?"
            item["expected_answer"] = "Il faut notamment produire un contrat de location ou un titre de propriété du local."
        elif source == "Cir_2019_08_fr.pdf" and item["category"] == "definition":
            item["expected_answer"] = "Les financements commerciaux islamiques comprennent la Mourabaha, l'Ijara, l'Istisna'a et le Salam."
        elif source == "Cir_2019_08_fr.pdf":
            item["expected_answer"] = "La Mourabaha peut être accordée aux professionnels et aux particuliers."
        elif source == "Cir_2019_09_fr.pdf" and item["expected_page"] == 2:
            item["query"] = "Que doit faire une personne physique lors de sa première activation du service en ligne permettant de consulter ses données à la Centrale d'Informations ?"
            item["expected_answer"] = "Elle doit s'inscrire en ligne, se présenter pour faire vérifier son identité et déposer le formulaire d'inscription téléchargé."
        elif source == "Cir_2019_09_fr.pdf":
            item["expected_answer"] = "La Banque Centrale peut prendre au maximum deux jours ouvrables pour vérifier les documents et répondre."
        elif source == "Note_2019_02_fr.pdf":
            item["query"] = "À partir de quelle date la pièce tunisienne de 100 millimes émise en 2019 a-t-elle été mise en circulation ?"
            item["expected_answer"] = "Elle a été mise en circulation le 7 février 2019."
        elif source == "Note_2019_08_ar.pdf" and item["category"] == "amount_or_rate":
            item["query"] = "كيف كان يُحدّد سعر بيع العملات للمنحة السياحية الخاصة بموسم الحج لسنة 2019؟"
            item["expected_answer"] = "كان كل وسيط مقبول يحدد سعر بيع العملات وفق أسعاره المعتمدة."
        elif source == "Note_2019_08_ar.pdf":
            item["query"] = "أين كان يمكن للحجيج صرف المنحة السياحية الخاصة بموسم الحج لسنة 2019؟"
            item["expected_answer"] = "كان يمكن صرفها نقدا في الشبابيك الموحدة المفتوحة لهذا الغرض."
        elif source == "Note_2019_16_fr.pdf":
            item["expected_answer"] = "Elles sont publiées dans le système d'échange de données de la Banque Centrale de Tunisie (SED)."
        elif source == "Note_2019_17_fr.pdf" and item["category"] == "amount_or_rate" and item["expected_page"] == 1:
            item["query"] = "Quel était le montant total de la ligne de crédit espagnole destinée aux petits et moyens projets tunisiens et tuniso-espagnols en 2019 ?"
            item["expected_answer"] = "La ligne de crédit s'élevait à 25 000 000 d'euros."
        elif source == "Note_2019_17_fr.pdf":
            item["expected_answer"] = "Le taux maximal était de 2,75 % par an pour une rétrocession en euros."
        elif source == "Note_2019_22_fr.pdf":
            item["query"] = "À quel moment un intermédiaire agréé peut-il communiquer à son client les références de domiciliation électronique d'un titre de commerce extérieur ?"
            item["expected_answer"] = "Il ne peut les communiquer qu'après la finalisation de la prise en charge du titre dans le système intégré."
        curated.append(item)

    curated.extend(
        [
            manual("Cir_2019_06_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds de crédit par hectare s'appliquent au blé dur, au blé tendre et aux légumineuses en sec et en irrigué, et à quelle échéance ?",
                   "En sec, le plafond est de 1 030 dinars en zone 1 et de 855 dinars en zone 2. En irrigué, il est de 1 355 dinars dans les deux zones. L'échéance est le 31 août.",
                   "Blé dur, blé tendre et légumineuses | sec zone 1 : 1030 DT/ha | sec zone 2 : 855 DT/ha | irrigué : 1355 DT/ha | 31 août", "visual_review"),
            manual("Cir_2019_06_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds de crédit par hectare s'appliquent à l'orge en sec dans les zones 1, 2 et 3 ?",
                   "Les plafonds sont de 625 dinars en zone 1, 585 dinars en zone 2 et 245 dinars en zone 3, avec une échéance au 31 août.",
                   "Orge en sec | zone 1 : 625 DT/ha | zone 2 : 585 DT/ha | zone 3 : 245 DT/ha | 31 août", "visual_review"),
            manual("Cir_2019_06_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds et échéances s'appliquent aux fourrages d'hiver en sec et aux fourrages d'été en irrigué ?",
                   "Le plafond est de 745 dinars par hectare pour les fourrages d'hiver en sec, à échéance du 31 août, et de 1 070 dinars pour les fourrages d'été en irrigué, à échéance du 30 septembre.",
                   "Fourrages d'hiver en sec : 745 DT/ha, 31 août | fourrages d'été en irrigué : 1070 DT/ha, 30 septembre", "visual_review"),
            manual("Cir_2019_10_fr.pdf", 2, "fr", "deadline_or_duration",
                   "À quelle date arrive à échéance un crédit de culture saisonnière d'oliviers après la révision de 2019 ?",
                   "L'échéance est fixée au 31 mars au lieu du 31 décembre."),
            manual("Cir_2019_10_fr.pdf", 2, "fr", "deadline_or_duration",
                   "À quelle date arrive à échéance un crédit de campagne destiné au secteur de l'huile d'olive après la révision de 2019 ?",
                   "L'échéance est fixée au 30 juin au lieu du 31 mars."),
            manual("Cir_2019_11_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds de crédit et quelle durée s'appliquent à un élevage de tilapias d'une capacité de 10 tonnes ?",
                   "Le plafond est de 9 000 dinars pour l'achat d'alevins et de 26 000 dinars pour l'élevage et l'assurance, sur 12 mois.",
                   "Tilapia, capacité 10 tonnes | achat d'alevins : 9 milliers de dinars (9000 dinars) | élevage et assurance : 26 milliers (26000 dinars) | 12 mois", "visual_review"),
            manual("Cir_2019_11_fr.pdf", 3, "fr", "amount_or_rate",
                   "Quels plafonds de crédit et quelle durée s'appliquent à un élevage de loups et de daurades en cages d'une capacité de 1 000 tonnes ?",
                   "Le plafond est de 2 100 000 dinars pour l'achat d'alevins et de 6 600 000 dinars pour l'élevage et l'assurance, sur 24 mois.",
                   "Loups et daurades en cages, capacité 1000 tonnes | achat d'alevins : 2100 milliers de dinars (2100000 dinars) | élevage et assurance : 6600 milliers (6600000 dinars) | 24 mois", "visual_review"),
            manual("Cir_2019_12_fr.pdf", 2, "fr", "deadline_or_duration",
                   "Une durée fixe était-elle imposée pour rééchelonner les dettes des oléifacteurs et exportateurs d'huile d'olive en difficulté pendant les campagnes 2017-2018 et 2018-2019 ?",
                   "Non. Le rééchelonnement devait être décidé au cas par cas selon la capacité de remboursement de chaque bénéficiaire."),
            manual("Cir_2019_12_fr.pdf", 3, "fr", "amount_or_rate",
                   "Quels taux minimaux de provisionnement collectif s'appliquaient aux oléifacteurs et aux exportateurs d'huile d'olive concernés par le traitement de l'endettement ?",
                   "Le taux minimal était de 35 % pour les oléifacteurs et de 30 % pour les exportateurs d'huile d'olive."),
            manual("Note_2019_01_ar.pdf", 1, "ar", "deadline_or_duration",
                   "في أي توقيت فتحت البنوك شبابيكها يوم السبت 19 جانفي 2019 بمناسبة عطلة عيد الثورة والشباب، وما العمليات المسموح بها؟",
                   "فتحت من الساعة التاسعة صباحا إلى منتصف النهار، واقتصرت العمليات على التنزيل نقدا والسحب والصرف اليدوي.",
                   "السبت 19 جانفي 2019 | من الساعة التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي", "visual_review"),
            manual("Note_2019_04_ar.pdf", 2, "ar", "amount_or_rate",
                   "ما مبالغ القروض التكميلية للهكتار من الحبوب لموسم 2018-2019، وما أجل سدادها؟",
                   "المبلغ 202 دينار للحبوب المروية، و202 دينار للحبوب البعلية في المنطقة 1، و183 دينارا للحبوب البعلية في المنطقة 2، وأجل السداد 31 أوت 2019.",
                   "حبوب مروية 202 د/هك | حبوب بعلية منطقة 1: 202 د/هك | منطقة 2: 183 د/هك | 31 أوت 2019", "visual_review"),
            manual("Note_2019_05_ar.pdf", 3, "ar", "amount_or_rate",
                   "ما الحد الأقصى للقرض الميسر لتغطية التمويل الذاتي في برنامج المسكن الأول، وكم تدوم فترة الإمهال؟",
                   "الحد الأقصى هو 20 % من الثمن الجملي للمسكن، وفترة الإمهال خمس سنوات دون فائض.",
                   "القرض الميسر للمسكن الأول | حد أقصى 20% من الثمن الجملي | إمهال 5 سنوات دون فائض", "visual_review"),
            manual("Note_2019_06_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الورقة السويسرية من فئة 1000 فرنك الصادرة ضمن السلسلة التاسعة؟",
                   "بدأ تداولها في 13 مارس 2019.",
                   "ورقة سويسرية من فئة 1000 فرنك | بداية التداول 13 مارس 2019", "visual_review"),
            manual("Note_2019_07_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى جمعية Les patriotes étoilistes كمصدر للاقتطاعات البنكية والبريدية؟",
                   "الرمز المسند إليها هو 0102."),
            manual("Note_2019_09_ar.pdf", 1, "ar", "deadline_or_duration",
                   "ما آخر أجل كان على بنك أو مؤسسة مالية استكمال تسجيله لدى إدارة الضرائب الأمريكية والحصول على شهادة المصادقة الإلكترونية لتطبيق FATCA؟",
                   "كان التسجيل مطلوبا قبل موفى ماي 2019، والحصول على شهادة المصادقة الإلكترونية في أقصى أجل يوم 31 ماي 2019.",
                   "FATCA | التسجيل قبل موفى ماي 2019 | شهادة المصادقة الإلكترونية في أقصى أجل 31 ماي 2019", "visual_review"),
            manual("Note_2019_09_ar.pdf", 2, "ar", "deadline_or_duration",
                   "ما آخر أجل لإيداع تصريحات FATCA الخاصة بسنوات 2014 إلى 2018 وإحالتها إلى الجانب الأمريكي؟",
                   "كان آخر أجل يوم 30 سبتمبر 2019.",
                   "تصريحات FATCA لسنوات 2014 و2015 و2016 و2017 و2018 | آخر أجل 30 سبتمبر 2019", "visual_review"),
            manual("Note_2019_10_ar.pdf", 1, "ar", "deadline_or_duration",
                   "في أي توقيت فتحت البنوك شبابيكها يوم الخميس 6 جوان 2019 بمناسبة عطلة عيد الفطر؟",
                   "فتحت من الساعة التاسعة صباحا إلى منتصف النهار، للقيام بالتنزيل نقدا والسحب والصرف اليدوي.",
                   "الخميس 6 جوان 2019 | من الساعة التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي", "visual_review"),
            manual("Note_2019_11_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الأوراق الأوروبية الجديدة من فئتي 100 و200 أورو؟",
                   "بدأ تداولها في 28 ماي 2019.",
                   "أوراق من فئتي 100 و200 أورو | بداية التداول 28 ماي 2019", "visual_review"),
            manual("Note_2019_13_ar.pdf", 3, "ar", "eligibility_or_scope",
                   "من كان المورد المسجل للإطارات المطاطية والعجلات الكاملة في قائمة جوان 2019؟",
                   "كان المورد المسجل هو STE MECHRI DE VENTE PNEUS.",
                   "مورد الإطارات المطاطية والعجلات الكاملة | STE MECHRI DE VENTE PNEUS", "visual_review"),
            manual("Note_2019_15_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى Ola Energy Tunisie كمصدر للاقتطاعات البنكية والبريدية؟",
                   "الرمز المسند إليها هو 0103.",
                   "Ola Energy Tunisie | الرمز 0103", "visual_review"),
            manual("Note_2019_18_ar.pdf", 1, "ar", "deadline_or_duration",
                   "في أي توقيت فتحت البنوك شبابيكها يوم السبت 10 أوت 2019 بمناسبة عيدي الأضحى والمرأة؟",
                   "فتحت من الساعة التاسعة صباحا إلى منتصف النهار، واقتصرت العمليات على التنزيل نقدا والسحب والصرف اليدوي.",
                   "السبت 10 أوت 2019 | من الساعة التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي", "visual_review"),
            manual("Note_2019_19_ar.pdf", 1, "ar", "required_action",
                   "ما التدابير المطلوبة من البنوك والبريد لضمان السحب خلال عطلتي عيد الأضحى وعيد المرأة؟",
                   "كان عليها تزويد الموزعات الآلية باستمرار، ومتابعة مخزون الأوراق النقدية، والتدخل السريع لإصلاح الأعطال وضمان استمرارية السحب."),
            manual("Note_2019_21_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "ما المؤسسة المسجلة لتوريد الحقن ذات الاستعمال الوحيد في قائمة سبتمبر 2019؟",
                   "المؤسسة المسجلة هي STE PHARMA BY CO.",
                   "مورد الحقن ذات الاستعمال الوحيد | STE PHARMA BY CO", "visual_review"),
            manual("Note_2019_23_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى Association Zaytounia pour la Culture et les Sciences كمصدر للاقتطاعات البنكية والبريدية؟",
                   "الرمز المسند إليها هو 0104.",
                   "Association Zaytounia pour la Culture et les Sciences | الرمز 0104", "visual_review"),
            manual("Note_2019_24_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الورقة السويسرية من فئة 100 فرنك الصادرة ضمن السلسلة التاسعة؟",
                   "بدأ تداولها في 12 سبتمبر 2019.",
                   "ورقة سويسرية من فئة 100 فرنك | بداية التداول 12 سبتمبر 2019", "visual_review"),
            manual("Note_2019_25_ar.pdf", 1, "ar", "deadline_or_duration",
                   "إلى أي تاريخ واصل البنك المركزي التونسي قبول الورقتين النرويجيتين القديمتين من فئتي 50 و500 كرونة؟",
                   "واصل قبولهما إلى غاية 2 أكتوبر 2019 بدخول الغاية، وبقي استبدالهما لدى بنك النرويج ممكنا لمدة لا تقل عن عشر سنوات بعد السحب.",
                   "ورقتا 50 و500 كرونة | قبول البنك المركزي التونسي إلى 2 أكتوبر 2019 | الاستبدال في النرويج لمدة لا تقل عن 10 سنوات", "visual_review"),
            manual("Note_2019_26_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى Société d’Administration et Gestion des Stations-Services كمصدر للاقتطاعات؟",
                   "الرمز المسند إليها هو 0105.",
                   "Société d’Administration et Gestion des Stations-Services | الرمز 0105", "visual_review"),
            manual("Note_2019_27_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المسند إلى ENDA Tamweel كمصدر للاقتطاعات البنكية والبريدية؟",
                   "الرمز المسند إليها هو 0106.",
                   "ENDA Tamweel | الرمز 0106", "visual_review"),
        ]
    )

    output = stable_ids(curated)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
