from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from curate_2021_draft import manual, stable_ids


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("queries", []) if isinstance(data, dict) else data


def visual(source: str, page: int, language: str, category: str, query: str, answer: str, evidence: str) -> dict:
    return manual(source, page, language, category, query, answer, evidence, "visual_review")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    curated: list[dict] = []
    for candidate in load_items(args.draft):
        item = deepcopy(candidate)
        item["language"] = "fr"
        item["relevant"] = True
        item["evidence_method"] = "text_extraction"
        curated.append(item)

    curated.extend(
        [
            manual("Cir_2016_03_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quel ratio de solvabilité minimal une banque ou un établissement financier devait-il maintenir en permanence ?",
                   "Le ratio de solvabilité ne pouvait pas être inférieur à 10 % des risques encourus."),
            manual("Cir_2016_03_fr.pdf", 2, "fr", "deadline_or_duration",
                   "À quelles limites l'exposition aux risques devait-elle être ramenée à fin 2017 puis à fin 2018 ?",
                   "La limite devait être ramenée à 75 % des fonds propres nets à fin 2017, puis à 25 % à fin 2018."),
            visual("Cir_2016_04_ar.pdf", 4, "ar", "deadline_or_duration",
                   "كيف يسدد البنك المبالغ المسحوبة من خط تمويل المؤسسات الصغرى والمتوسطة البالغ 50 مليون دولار؟",
                   "يسددها على أقساط سداسية لا تتجاوز 15 قسطا، بعد فترة إمهال مدتها ثلاث سنوات من تاريخ كل عملية سحب.",
                   "خط تمويل 50 مليون دولار | أقساط سداسية لا تتجاوز 15 قسطا | فترة إمهال ثلاث سنوات"),
            visual("Cir_2016_04_ar.pdf", 4, "ar", "amount_or_rate",
                   "ما نسبة الفائدة التي يدفعها البنك عن الأموال غير المسددة من خط تمويل المؤسسات الصغرى والمتوسطة؟",
                   "النسبة السنوية 2 % للمبالغ المسحوبة بالدولار الأمريكي و4 % للمبالغ المسحوبة بالدينار التونسي.",
                   "فائدة سنوية 2 بالمائة للمبالغ بالدولار الأمريكي | 4 بالمائة للمبالغ بالدينار التونسي"),
            manual("Cir_2016_05_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quel était le plafond de crédit par hectare pour le blé dur ou tendre cultivé en sec en zone 1, et pour la même culture en irrigué ?",
                   "Le plafond était de 975 dinars par hectare en sec en zone 1 et de 1 225 dinars par hectare en irrigué, avec échéance au 31 août."),
            manual("Cir_2016_05_fr.pdf", 2, "fr", "amount_or_rate",
                   "Quels plafonds de crédit s'appliquaient à l'orge en sec dans les zones 1, 2 et 3 ?",
                   "Les plafonds étaient de 690 dinars par hectare en zone 1, 510 dinars en zone 2 et 225 dinars en zone 3."),
            manual("Cir_2016_06_fr.pdf", 3, "fr", "reporting_or_control",
                   "Combien de niveaux minimaux un système interne de notation du risque de crédit devait-il comporter ?",
                   "Il devait comporter au moins sept notes pour les contreparties qui ne sont pas en défaut et une note pour celles en défaut."),
            manual("Cir_2016_06_fr.pdf", 6, "fr", "deadline_or_duration",
                   "À quelle fréquence la notation d'une contrepartie devait-elle être mise à jour ?",
                   "Elle devait être mise à jour au moins une fois par an, et plus fréquemment pour les contreparties particulièrement risquées."),
            manual("Cir_2016_07_fr.pdf", 2, "fr", "eligibility_or_scope",
                   "Dans quelles situations un établissement pouvait-il demander une assistance financière exceptionnelle à la Banque Centrale ?",
                   "Il pouvait la demander s'il était solvable mais connaissait une difficulté temporaire de liquidité, ou si sa solvabilité était atteinte et que sa défaillance menaçait la stabilité du système financier. Les banques non résidentes en étaient exclues."),
            manual("Cir_2016_07_fr.pdf", 3, "fr", "deadline_or_duration",
                   "Quelle pouvait être la durée totale maximale d'une assistance financière exceptionnelle, renouvellements compris ?",
                   "L'assistance était accordée pour trois mois au plus et pouvait être renouvelée trois fois, sans dépasser une durée totale de douze mois."),
            manual("Cir_2016_08_fr.pdf", 2, "fr", "amount_or_rate",
                   "Comment se calculait l'allocation de voyages d'affaires d'un exportateur et quel était son plafond annuel ?",
                   "Elle correspondait à 25 % des recettes d'exportation rapatriées, dans la limite de 500 000 dinars par année civile."),
            manual("Cir_2016_08_fr.pdf", 4, "fr", "amount_or_rate",
                   "Comment se calculait l'allocation de voyages d'affaires pour une activité non exportatrice ?",
                   "Elle correspondait à 8 % du chiffre d'affaires hors taxes de l'année précédente, dans la limite de 50 000 dinars par année civile."),
            visual("Cir_2016_09_fr.pdf", 3, "fr", "exception_or_condition",
                   "Une entreprise pouvait-elle verser un acompte à un prestataire étranger sans garantie de restitution ?",
                   "Oui, si la prestation entrait dans son cycle de production et si l'acompte ne dépassait pas 25 % de la valeur de l'opération. Sinon, une garantie de restitution à première demande était requise.",
                   "Acompte sans garantie de restitution | prestation entrant dans le cycle de production | plafond de 25 % de la valeur de l'opération"),
            manual("Cir_2016_09_fr.pdf", 6, "fr", "amount_or_rate",
                   "Quel plafond annuel de Carte Technologique Internationale s'appliquait à une entreprise et à une personne physique ?",
                   "Le plafond était de 10 000 dinars pour une entreprise éligible et de 1 000 dinars pour une personne physique tunisienne résidente titulaire d'au moins le baccalauréat."),
            manual("Cir_2016_10_fr.pdf", 3, "fr", "deadline_or_duration",
                   "Combien de temps une autorisation de sortie de devises restait-elle valable ?",
                   "Elle était personnelle, valable pour un seul voyage et pour deux mois au maximum à compter de sa délivrance."),
            manual("Cir_2016_10_fr.pdf", 3, "fr", "procedure_or_documents",
                   "À qui étaient destinés les trois exemplaires A, B et C d'une autorisation de sortie de devises ?",
                   "La formule A était remise au bénéficiaire, la formule B conservée par l'intermédiaire agréé et la formule C remise aux services des Douanes à la sortie du territoire."),
            manual("Note_2016_01_ar.pdf", 1, "ar", "eligibility_or_scope",
                   "ما الفترة والمحطات التي غطاها جدول استمرارية مكاتب الصرف بمطار تونس قرطاج في بداية 2016؟",
                   "غطى الجدول الفترة من غرة جانفي إلى موفى جوان 2016 بالمحطة الرئيسية والمحطة الفرعية الثانية."),
            manual("Note_2016_02_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "هل كانت شركة BOUZGARROU DESIGN مدرجة ضمن موردي المقاعد والأثاث وأجزائه الخاضعة لكراس الشروط؟",
                   "نعم، أدرجها الملحق ضمن قائمة الموردين المسجلين."),
            manual("Note_2016_03_ar.pdf", 2, "ar", "definition",
                   "كيف تغير تصنيف مؤسسة الوفاق للإيجار المالي مع احتفاظها بالرمز 47؟",
                   "تحولت من مؤسسة مالية باسم EL WIFACK LEASING إلى بنك باسم WIFACK INTERNATIONAL BANK، مع بقاء الرمز 47."),
            manual("Note_2016_04_ar.pdf", 2, "ar", "amount_or_rate",
                   "ما مبلغ القرض التكميلي لكل هكتار من الحبوب المروية في موسم 2015-2016، وما أجل سداده؟",
                   "المبلغ 148 دينارا للهكتار، وأجل السداد 31 أوت 2016."),
            manual("Note_2016_05_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "هل كانت شركة SMART TUNISIE مسجلة ضمن موردي أجهزة الاستقبال التلفزية الخاضعة لكراس الشروط؟",
                   "نعم، أدرجها الملحق ضمن موردي أجهزة الاستقبال التلفزية المسجلين."),
            visual("Note_2016_06_ar.pdf", 1, "ar", "deadline_or_duration",
                   "إلى متى واصل البنك المركزي التونسي قبول الأوراق السويدية القديمة من فئات 20 و50 و1000 كرونة؟",
                   "واصل قبولها إلى غاية 16 ماي 2016 بدخول الغاية، قبل سحبها من التداول في السويد ابتداء من 1 جويلية 2016.",
                   "فئات 20 و50 و1000 كرونة سويدية | قبول البنك المركزي التونسي إلى 16 ماي 2016 | السحب من التداول 1 جويلية 2016"),
            manual("Note_2016_07_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "هل كانت شركة OZALIS DES PIECES DE RECHANGE مدرجة ضمن موردي الإطارات المطاطية والعجلات الكاملة؟",
                   "نعم، أدرجها الملحق ضمن الموردين المسجلين."),
            manual("Note_2016_08_fr.pdf", 1, "fr", "required_action",
                   "Quel document devait être vérifié avant de domicilier l'importation d'un produit soumis à surveillance préalable ?",
                   "Le dossier devait contenir une fiche d'information visée par les services du ministère chargé du Commerce."),
            visual("Note_2016_09_ar.pdf", 1, "ar", "definition",
                   "بماذا تميز أول إصدار من السلسلة التاسعة للأوراق السويسرية من حيث الفئة واللون؟",
                   "كان ورقة من فئة 50 فرنكا، ولونها الرئيسي أخضر.",
                   "أول إصدار من السلسلة التاسعة | فئة 50 فرنكا | اللون الرئيسي الأخضر"),
            manual("Note_2016_10_ar.pdf", 1, "ar", "definition",
                   "ما فئات الأوراق الإماراتية التي أعيد طرحها بعلامات أمان جديدة؟",
                   "شملت فئات 10 و20 و200 و1000 درهم."),
            visual("Note_2016_11_ar.pdf", 1, "ar", "definition",
                   "ما رمز الشركة العربية الدولية للإيجار المالي ARAB INTERNATIONAL LEASE كمصدر للاقتطاعات؟",
                   "رمزها 078.", "ARAB INTERNATIONAL LEASE | الرمز 078"),
            manual("Note_2016_12_fr.pdf", 1, "fr", "procedure_or_documents",
                   "Un étudiant non boursier poursuivant ses études à l'étranger devait-il encore présenter une attestation papier originale ?",
                   "Non. L'intermédiaire agréé pouvait accepter, une seule fois, l'attestation téléchargée depuis le site du ministère pour constituer le dossier de scolarité et exécuter les transferts."),
            visual("Note_2016_13_ar.pdf", 1, "ar", "eligibility_or_scope",
                   "ما الفترة التي غطاها جدول استمرارية مكاتب الصرف بمطار تونس قرطاج في النصف الثاني من 2016؟",
                   "غطى الجدول الفترة من غرة جويلية إلى موفى ديسمبر 2016 بالمحطة الرئيسية والمحطة الفرعية الثانية.",
                   "جدول استمرارية مكاتب الصرف | من غرة جويلية إلى موفى ديسمبر 2016 | المحطة الرئيسية والمحطة الفرعية الثانية"),
            visual("Note_2016_14_ar.pdf", 1, "ar", "deadline_or_duration",
                   "متى فتحت البنوك شبابيكها بمناسبة عطلة عيد الفطر سنة 2016، وما العمليات المسموح بها؟",
                   "فتحت يوم الخميس 7 جويلية 2016 من التاسعة صباحا إلى منتصف النهار، للتنزيل نقدا والسحب والصرف اليدوي فقط.",
                   "الخميس 7 جويلية 2016 | من التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي"),
            manual("Note_2016_15_ar.pdf", 1, "ar", "deadline_or_duration",
                   "متى فتحت البنوك شبابيكها بمناسبة عطلة عيد الجمهورية سنة 2016؟",
                   "فتحت يوم السبت 23 جويلية 2016 من التاسعة صباحا إلى منتصف النهار، للتنزيل نقدا والسحب والصرف اليدوي فقط."),
            manual("Note_2016_16_ar.pdf", 1, "ar", "reporting_or_control",
                   "كيف كان وسيط الصرف يحدد سعر بيع الريالات لمنحة الحج، وما التقرير المطلوب منه؟",
                   "كان يطبق السعر الذي يضبطه ويشهره للعموم بوضوح، ويرسل إلى البنك المركزي تقريرا يوميا عن مبيعات منحة الحج."),
            visual("Note_2016_17_ar.pdf", 1, "ar", "definition",
                   "ما رمزا شركتي BeIN وYOOOPY كمصدرين للاقتطاعات البنكية والبريدية؟",
                   "رمز BeIN هو 079 ورمز YOOOPY هو 080.", "BeIN | الرمز 079 | YOOOPY | الرمز 080"),
            manual("Note_2016_18_ar.pdf", 1, "ar", "eligibility_or_scope",
                   "ما أنواع المعاملات مع إيران التي أصبح استئنافها ممكنا بعد الرفع التدريجي للعقوبات؟",
                   "أصبح ممكنا استئناف المعاملات التجارية والمالية المرتبطة بالنفط والغاز والنقل والخدمات المالية، مع احترام القوانين والتدابير السارية."),
            visual("Note_2016_19_ar.pdf", 2, "ar", "deadline_or_duration",
                   "ما آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من جفاف موسم 2015-2016؟",
                   "كان آخر أجل موفى ديسمبر 2016، ويقدم المطلب إلى فرع البنك الممول مرفقا بشهادة معاينة تثبت الضرر.",
                   "مطلب جدولة ديون جفاف موسم 2015-2016 | آخر أجل موفى ديسمبر 2016 | شهادة معاينة تثبت الضرر"),
            manual("Note_2016_20_ar.pdf", 1, "ar", "required_action",
                   "ما المعلومات التي كان على المؤسسة المالية إرسالها عند تعيين مخاطب وحيد لتبادل معلومات FATCA؟",
                   "كان عليها إرسال هويتها ومعرفها الجبائي ورقم GIIN والاسم المسجل لدى IRS وهوية المخاطب وصفته وعناوينه الإدارية ورقم هاتفه، والإبلاغ فورا عن أي تغيير."),
            visual("Note_2016_21_ar.pdf", 1, "ar", "definition",
                   "ما رموز CTAMA والزيتونة تمكين والعصرية للإيجار المالي كمصادر للاقتطاعات؟",
                   "رمز CTAMA هو 0081، ورمز الزيتونة تمكين 0082، ورمز العصرية للإيجار المالي 0083.",
                   "CTAMA | 0081 | ZITOUNA TAMKEEN | 0082 | MODERN LEASING | 0083"),
            manual("Note_2016_22_fr.pdf", 1, "fr", "exception_or_condition",
                   "Une personne sans matricule fiscal ayant reçu une Carte Technologique Internationale en 2015 pouvait-elle renouveler son allocation en 2016 sans déclaration fiscale ?",
                   "Oui. À titre exceptionnel, l'intermédiaire agréé pouvait renouveler l'allocation sans exiger la déclaration fiscale de 2015."),
            visual("Note_2016_23_ar.pdf", 1, "ar", "deadline_or_duration",
                   "متى فتحت البنوك شبابيكها بمناسبة عطلة عيد الأضحى سنة 2016، وما العمليات المسموح بها؟",
                   "فتحت يوم السبت 10 سبتمبر 2016 من التاسعة صباحا إلى منتصف النهار، للتنزيل نقدا والسحب والصرف اليدوي فقط.",
                   "السبت 10 سبتمبر 2016 | من التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي"),
            visual("Note_2016_24_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك إنجلترا تداول أول ورقة بوليمر من فئة 5 جنيهات إسترلينية؟",
                   "بدأ تداولها في 13 سبتمبر 2016.", "Bank of England | أول ورقة بوليمر فئة 5 جنيهات | بداية التداول 13 سبتمبر 2016"),
            manual("Note_2016_25_ar.pdf", 1, "ar", "other_operational_rule",
                   "ما التغيير الذي أدخله مصرف البحرين المركزي على علامة ضعاف البصر في الطبعتين الجديدتين؟",
                   "نقلت العلامة من أعلى يمين وجه الورقة إلى وسط الجهة اليمنى قرب الحاشية، وأصبحت خطوطا أفقية متتالية بارزة يمكن تحسسها."),
            visual("Note_2016_26_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك السويد تداول الأوراق الجديدة من فئتي 100 و500 كرونة؟",
                   "بدأ تداولها في 3 أكتوبر 2016.", "بنك السويد | فئتا 100 و500 كرونة | بداية التداول 3 أكتوبر 2016"),
            visual("Note_2016_27_fr.pdf", 1, "fr", "effective_date",
                   "Quand la Centrale des Actifs Éligibles au Refinancement devait-elle entrer réellement en production, et quand commençait la dernière phase de tests ?",
                   "La CAER devait entrer en production le 3 janvier 2017, après une dernière phase de tests commençant le 1er novembre 2016.",
                   "CAER | production effective le 3 janvier 2017 | dernière phase de tests à partir du 1er novembre 2016"),
            visual("Note_2016_28_ar.pdf", 1, "ar", "definition",
                   "ما رمز شركة BIGDeal كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0084.", "BIGDeal | الرمز 0084"),
            manual("Note_2016_29_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "ما الشركتان اللتان أضيفتا إلى قائمة موردي أجهزة الاستقبال التلفزية في نوفمبر 2016؟",
                   "هما شركة HDD ELECTRONIQUE وشركة TDISCOUNT."),
            manual("Note_2016_30_fr.pdf", 1, "fr", "procedure_or_documents",
                   "Où un résident devait-il déposer une demande de régularisation d'une allocation touristique restituée hors délai ?",
                   "À compter du 2 janvier 2017, il devait déposer un formulaire F2 auprès du comptoir de la Banque Centrale compétent pour sa zone."),
            manual("Note_2016_31_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "هل كانت شركة SEVAC CERAMIC مدرجة ضمن موردي المربعات الخزفية الخاضعة لكراس الشروط؟",
                   "نعم، أدرجها الملحق ضمن الموردين المسجلين للمربعات الخزفية."),
            visual("Note_2016_32_ar.pdf", 1, "ar", "definition",
                   "ما رمز الرابطة الوطنية للقرآن الكريم كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0085.", "الرابطة الوطنية للقرآن الكريم | الرمز 0085"),
            manual("Note_2016_33_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب النقدي خلال عطلة المولد النبوي الشريف؟",
                   "كان عليها اتخاذ الإجراءات الضرورية لتأمين استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
            visual("Note_2016_34_ar.pdf", 1, "ar", "definition",
                   "ما رمز الجمعية التونسية لمرضى العضلات كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0086.", "الجمعية التونسية لمرضى العضلات | الرمز 0086"),
        ]
    )

    output = stable_ids(curated)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
