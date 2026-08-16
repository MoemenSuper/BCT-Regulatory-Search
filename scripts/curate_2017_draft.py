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
        source = candidate["expected_source"]
        if source.startswith("Note_") or source == "Cir_2017_09_fr.pdf":
            continue
        if source == "CB_2017_08_FR.pdf" and candidate["category"] == "definition":
            item = deepcopy(candidate)
            item["query"] = "À partir de quelle participation au capital le bénéficiaire effectif d'une société était-il considéré comme un actionnaire ou associé important par le dispositif antiblanchiment ?"
        elif source == "Cir_2017_02_fr.pdf" and candidate["category"] == "eligibility_or_scope":
            item = deepcopy(candidate)
            item["query"] = "Quelles conditions une banque devait-elle remplir pour être admise aux opérations de politique monétaire de la Banque Centrale ?"
            item["expected_answer"] = "Elle devait être financièrement solide au regard notamment de ses ratios de fonds propres et de liquidité, et satisfaire aux critères opérationnels fixés par la Banque Centrale."
        elif source == "Cir_2017_03_fr.pdf" and candidate["category"] == "effective_date":
            item = deepcopy(candidate)
            item["query"] = "À partir de quand le taux annuel de rémunération de l'épargne de 4 % devait-il être appliqué ?"
        elif source == "Cir_2017_05_ar.pdf" and candidate["category"] == "eligibility_or_scope":
            item = deepcopy(candidate)
            item["query"] = "هل كان تمديد إجراءات مساندة القطاع السياحي لسنة 2017 متاحا لمؤسسة لم تنتفع سابقا بهذه الإجراءات؟"
            item["expected_answer"] = "لا. شمل التمديد المؤسسات التي سبق لها الانتفاع بالإجراءات الاستثنائية، باستثناء الإجراءات الواردة بالفصل الثاني من المنشور السابق."
        elif source == "Cir_2017_06_fr.pdf" and candidate["category"] == "eligibility_or_scope":
            item = deepcopy(candidate)
            item["query"] = "Les obligations de reporting comptable, prudentiel et statistique de 2017 s'appliquaient-elles uniquement aux banques ?"
            item["expected_answer"] = "Non. Elles s'appliquaient aux banques et aux établissements financiers définis par la loi n° 2016-48."
        elif source == "Cir_2017_12_fr.pdf" and candidate["category"] == "amount_or_rate":
            item = deepcopy(candidate)
            item["query"] = "Quel taux annuel devait rémunérer les comptes spéciaux d'épargne à compter de janvier 2018 ?"
        elif source == "Cir_2017_12_fr.pdf" and candidate["category"] == "effective_date":
            item = deepcopy(candidate)
            item["query"] = "À partir de quand le taux annuel de rémunération de l'épargne de 5 % devait-il être appliqué ?"
        else:
            item = deepcopy(candidate)
        item["language"] = "ar" if source.endswith("_ar.pdf") else "fr"
        item["relevant"] = True
        item["evidence_method"] = "text_extraction"
        curated.append(item)

    curated.extend(
        [
            visual("Cir_2017_09_fr.pdf", 1, "fr", "prohibition_or_limit",
                   "Une banque pouvait-elle financer directement l'importation d'un produit non prioritaire en 2017 ?",
                   "Non. L'importateur devait constituer sur ses fonds propres un dépôt couvrant la totalité de la valeur de l'importation envisagée.",
                   "Produits non prioritaires | dépôt sur fonds propres couvrant la totalité de la valeur des importations envisagées"),
            visual("Cir_2017_09_fr.pdf", 2, "fr", "exception_or_condition",
                   "Quelles importations échappaient à l'obligation de dépôt intégral imposée aux produits non prioritaires ?",
                   "Étaient exclues les importations réalisées dans le cadre de marchés publics au profit de l'État, des entreprises et établissements publics ou des collectivités locales, ainsi que les opérations pour lesquelles un engagement de financement avait déjà commencé à être exécuté avant l'entrée en vigueur du dispositif.",
                   "Exclusions | marchés publics au profit de l'Etat, des entreprises et établissements publics et des collectivités locales | engagements de financement dont l'exécution avait commencé avant l'entrée en vigueur"),
            visual("Note_2017_01_ar.pdf", 2, "ar", "amount_or_rate",
                   "ما مبلغ القرض التكميلي لكل هكتار من الحبوب المروية، وما أجل سداده في موسم 2016-2017؟",
                   "المبلغ 156 دينارا للهكتار، وأجل السداد هو 31 أوت 2017.",
                   "حبوب مروية | 156 دينارا للهكتار | أجل التسديد 31 أوت 2017"),
            visual("Note_2017_04_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الإصدار السادس من الأوراق النقدية السعودية، وما الفئات التي شملها؟",
                   "بدأ التداول في 26 ديسمبر 2016، وشمل فئات 500 و100 و50 و10 و5 ريالات.",
                   "الإصدار السادس | بداية التداول 26 ديسمبر 2016 | فئات 500 و100 و50 و10 و5 ريالات"),
            manual("Note_2017_05_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "هل كانت الشركة الإفريقية لتوزيع السيارات مسجلة ضمن موردي الإطارات المطاطية والعجلات الكاملة الخاضعة لكراس الشروط؟",
                   "نعم، أدرج الملحق الشركة الإفريقية لتوزيع السيارات (STE AFRICAINE DISTRIBUTION AUTOCAR) ضمن الموردين المسجلين."),
            manual("Note_2017_06_ar.pdf", 2, "ar", "other_operational_rule",
                   "ما التغيير الأمني الذي أضيف إلى الطبعة الثانية من الإصدار الخامس للأوراق النقدية الصينية؟",
                   "أضيفت علامة أمان لحماية الورقة من التزييف باستخدام آلات النسخ، إلى جانب تحسينات أمنية أخرى."),
            visual("Note_2017_07_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك اسكتلندا تداول ورقته الجديدة المصنوعة من البوليمر من فئة 5 جنيهات إسترلينية؟",
                   "بدأ تداولها في 4 أكتوبر 2016.", "Bank of Scotland | ورقة بوليمر فئة 5 جنيهات | بداية التداول 4 أكتوبر 2016"),
            visual("Note_2017_08_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك كليدسدال تداول ورقته الجديدة من فئة 5 جنيهات إسترلينية؟",
                   "بدأ تداولها في 27 سبتمبر 2016.", "Clydesdale Bank | ورقة فئة 5 جنيهات | بداية التداول 27 سبتمبر 2016"),
            visual("Note_2017_09_ar.pdf", 1, "ar", "definition",
                   "ما رمز شركة BIGDeal Trading كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0087.", "BIGDeal Trading | الرمز 0087"),
            visual("Note_2017_10_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ البنك الملكي الاسكتلندي تداول ورقته الجديدة من فئة 5 جنيهات إسترلينية؟",
                   "بدأ تداولها في 27 أكتوبر 2016.", "Royal Bank of Scotland | ورقة فئة 5 جنيهات | بداية التداول 27 أكتوبر 2016"),
            visual("Note_2017_11_ar.pdf", 1, "ar", "deadline_or_duration",
                   "متى فتحت البنوك شبابيكها بمناسبة عطلة عيد الاستقلال سنة 2017، وما العمليات التي سمح بها؟",
                   "فتحت يوم السبت 18 مارس 2017 من التاسعة صباحا إلى منتصف النهار، للتنزيل نقدا والسحب والصرف اليدوي فقط.",
                   "السبت 18 مارس 2017 | من التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي"),
            manual("Note_2017_12_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "هل كانت شركة كندور إلكترونيكس إنترناسيونال مسجلة ضمن موردي أجهزة الاستقبال التلفزية الخاضعة لكراس الشروط؟",
                   "نعم، أدرجها الملحق ضمن الموردين المسجلين لأجهزة الاستقبال التلفزية."),
            manual("Note_2017_13_ar.pdf", 1, "ar", "definition",
                   "ما فئات الأوراق البحرينية التي شملتها الطبعة الجديدة من الإصدار الرابع؟",
                   "شملت فئات خمسة دنانير ودينار واحد ونصف دينار."),
            visual("Note_2017_14_ar.pdf", 1, "ar", "deadline_or_duration",
                   "إلى أي تاريخ واصل البنك المركزي التونسي قبول ورقة Bank of England القديمة من فئة 5 جنيهات التي تحمل صورة Elizabeth Fry؟",
                   "واصل قبولها إلى غاية 14 أفريل 2017 بدخول الغاية، قبل سحبها من التداول في إنجلترا يوم 5 ماي 2017.",
                   "ورقة Elizabeth Fry فئة 5 جنيهات | قبول البنك المركزي التونسي إلى 14 أفريل 2017 | السحب من التداول 5 ماي 2017"),
            visual("Note_2017_15_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول ورقة اليورو الجديدة من فئة 50 يورو؟",
                   "بدأ تداولها في 4 أفريل 2017.", "ورقة جديدة فئة 50 يورو | بداية التداول 4 أفريل 2017"),
            manual("Note_2017_16_ar.pdf", 1, "ar", "definition",
                   "ما المؤسسة التي حلّت محل بنك الأعمال التونسي في الرمز 64، وما اسمها المختصر؟",
                   "حلّ البنك الإفريقي للاستثمار والشراكة محلّه، واسمه المختصر CAP Bank."),
            visual("Note_2017_17_ar.pdf", 1, "ar", "definition",
                   "ما رمز شركة SENIOR INTERIM كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0088.", "SENIOR INTERIM | الرمز 0088"),
            visual("Note_2017_18_ar.pdf", 2, "ar", "deadline_or_duration",
                   "قبل كم من الوقت يجب إعلام البنك المركزي بمن سينوب مفوضا أو وكيلا مصرفيا أو يعوضه مؤقتا؟",
                   "يجب إرسال الإفادة قبل بداية مدة النيابة أو التعويض بـ24 ساعة على الأقل.",
                   "إفادة بأسماء وصفات وسلطات ونماذج توقيع النواب | قبل بداية النيابة أو التعويض بـ24 ساعة على الأقل"),
            visual("Note_2017_19_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "ما شروط الدخل والملكية الأساسية لانتفاع عائلة ببرنامج المسكن الأول؟",
                   "يجب ألا تملك العائلة مسكنا، وأن يتراوح دخلها الشهري الخام بين 4.5 و10 مرات الأجر الأدنى المهني المضمون، وأن يكون المنتفع أو قرينه أجيرا.",
                   "لا تمتلك العائلة مسكنا | الدخل الشهري الخام بين 4.5 و10 مرات الأجر الأدنى المهني المضمون | المنتفع أو قرينه أجير"),
            manual("Note_2017_20_fr.pdf", 1, "fr", "required_action",
                   "Quel terme les banques devaient-elles employer en français à la place de « comptoir » pour désigner une implantation de la Banque Centrale ?",
                   "Elles devaient employer le terme « succursale » dans leurs documents physiques et électroniques destinés à la Banque Centrale."),
            manual("Note_2017_21_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب النقدي خلال عطلة عيد الشغل؟",
                   "كان عليها اتخاذ الإجراءات الضرورية لتأمين استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
            visual("Note_2017_22_ar.pdf", 1, "ar", "deadline_or_duration",
                   "إلى متى واصل البنك المركزي التونسي قبول الأوراق السويدية القديمة من فئتي 500 و100 كرونة؟",
                   "واصل قبولها إلى غاية 10 ماي 2017 بدخول الغاية، قبل سحبها من التداول في السويد ابتداء من 1 جويلية 2017.",
                   "فئتا 500 و100 كرونة سويدية | قبول البنك المركزي التونسي إلى 10 ماي 2017 | السحب من التداول 1 جويلية 2017"),
            manual("Note_2017_23_fr.pdf", 1, "fr", "required_action",
                   "Quel document devait figurer dans le dossier avant de domicilier une importation soumise à la surveillance préalable ?",
                   "Une fiche d'information visée par les services du ministère chargé de l'Industrie et du Commerce, Direction générale du commerce extérieur."),
            visual("Note_2017_24_ar.pdf", 1, "ar", "definition",
                   "ما رمز الجمعية التونسية لقرى الأطفال س و س كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0089.", "الجمعية التونسية لقرى الأطفال س و س | الرمز 0089"),
            visual("Note_2017_25_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الورقة السويسرية الجديدة من فئة 20 فرنكا ضمن السلسلة التاسعة؟",
                   "بدأ تداولها في 17 ماي 2017.", "ورقة سويسرية فئة 20 فرنكا | بداية التداول 17 ماي 2017"),
            visual("Note_2017_26_ar.pdf", 1, "ar", "definition",
                   "ما الرمز المعلوماتي المسند إلى مؤسسة MEDFACTOR المختصة في إدارة الديون؟",
                   "رمزها المعلوماتي 26.", "MEDFACTOR | الرمز المعلوماتي 26"),
            visual("Note_2017_27_ar.pdf", 1, "ar", "definition",
                   "ما رمز جمعية صيانة قرية الشفار المصيفية ASVEC كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0090.", "جمعية صيانة قرية الشفار المصيفية ASVEC | الرمز 0090"),
            visual("Note_2017_34_ar.pdf", 1, "ar", "deadline_or_duration",
                   "متى فتحت البنوك شبابيكها بمناسبة عطلة عيد الفطر سنة 2017، وما العمليات المسموح بها؟",
                   "فتحت يوم السبت 24 جوان 2017 من التاسعة صباحا إلى منتصف النهار، للتنزيل نقدا والسحب والصرف اليدوي فقط.",
                   "السبت 24 جوان 2017 | من التاسعة صباحا إلى منتصف النهار | التنزيل نقدا والسحب والصرف اليدوي"),
            visual("Note_2017_36_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الأوراق النرويجية الجديدة من فئتي 100 و200 كرونة؟",
                   "بدأ تداولها في 30 ماي 2017.", "فئتا 100 و200 كرونة نرويجية | بداية التداول 30 ماي 2017"),
            manual("Note_2017_38_ar.pdf", 1, "ar", "reporting_or_control",
                   "كيف كان الوسيط المقبول يحدد سعر بيع العملات لمنحة الحج، وما التزامه تجاه البنك المركزي؟",
                   "كان يطبق السعر الذي يضبطه ويشهره للعموم بوضوح، ويرسل إلى البنك المركزي تقريرا يوميا عن مبيعات المنحة السياحية لموسم الحج."),
            visual("Note_2017_46_ar.pdf", 2, "ar", "definition",
                   "ما الاسم الجديد والاسم المختصر للبنك الذي كان يسمى STUSID BANK؟",
                   "أصبح اسمه البنك التونسي السعودي (TUNISIAN SAUDI BANK) واسمه المختصر TSB، مع بقاء الرمز 21.",
                   "الرمز 21 | STUSID BANK أصبح TUNISIAN SAUDI BANK | الاسم المختصر TSB"),
            manual("Note_2017_55_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "ما الشركات الأربع التي أضيفت إلى قائمة موردي المقاعد والأثاث وأجزائه في أوت 2017؟",
                   "هي شركة GMIHA HOURIA وشركة SANIMEUBLE وشركة TIE & OFFICE (TO) وشركة VOGUE."),
            visual("Note_2017_56_ar.pdf", 1, "ar", "definition",
                   "ما رمز شركة الفولاذ كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0091.", "شركة الفولاذ | الرمز 0091"),
            manual("Note_2017_57_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب النقدي خلال عطلة عيد الأضحى؟",
                   "كان عليها اتخاذ الإجراءات الضرورية لتأمين استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
            visual("Note_2017_60_ar.pdf", 2, "ar", "deadline_or_duration",
                   "ما آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من جفاف موسم 2016-2017، وما المدة القصوى للجدولة؟",
                   "يقدم المطلب إلى فرع البنك الممول في أجل لا يتجاوز موفى ديسمبر 2017، وتتم الجدولة على مدة لا تتجاوز خمس سنوات.",
                   "مطلب الجدولة إلى موفى ديسمبر 2017 | مدة الجدولة لا تتجاوز خمس سنوات"),
            manual("Note_2017_61_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك كليدسدال تداول ورقته الجديدة من فئة 10 جنيهات إسترلينية؟",
                   "بدأ تداولها في 21 سبتمبر 2017."),
            manual("Note_2017_64_ar.pdf", 2, "ar", "eligibility_or_scope",
                   "ما الموردون الثلاثة للمقاعد والأثاث الذين أضيفوا إلى القائمة في أكتوبر 2017؟",
                   "هم المصرف التونسي للبناء (COMPTOIR TUNISIEN DE BATIMENT)، وشركة ARCA TN، وشركة ELGHAZELA TRADING."),
            manual("Note_2017_65_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ البنك الملكي الاسكتلندي تداول ورقته الجديدة من فئة 10 جنيهات إسترلينية؟",
                   "بدأ تداولها في 4 أكتوبر 2017."),
            manual("Note_2017_66_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ بنك اسكتلندا تداول ورقته الجديدة من فئة 10 جنيهات إسترلينية؟",
                   "بدأ تداولها في 11 أكتوبر 2017."),
            visual("Note_2017_67_ar.pdf", 1, "ar", "definition",
                   "ما رمز جمعية Réseau Entreprendre Tunisie كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0092.", "Réseau Entreprendre Tunisie | الرمز 0092"),
            visual("Note_2017_70_ar.pdf", 1, "ar", "effective_date",
                   "متى بدأ تداول الورقة السويسرية الجديدة من فئة 10 فرنكات ضمن السلسلة التاسعة؟",
                   "بدأ تداولها في 11 أكتوبر 2017.", "ورقة سويسرية فئة 10 فرنكات | بداية التداول 11 أكتوبر 2017"),
            visual("Note_2017_73_ar.pdf", 1, "ar", "definition",
                   "ما الرمزان المسندان إلى ERT وUnilever Tunisie كمصدرين للاقتطاعات البنكية والبريدية؟",
                   "رمز ERT هو 0093، ورمز Unilever Tunisie هو 0094.",
                   "ERT | الرمز 0093 | Unilever Tunisie | الرمز 0094"),
            manual("Note_2017_75_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب النقدي خلال عطلة المولد النبوي الشريف لسنة 2017؟",
                   "كان عليها اتخاذ الإجراءات الضرورية لتأمين استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
            manual("Note_2017_78_fr.pdf", 1, "fr", "required_action",
                   "Que devait faire une banque tunisienne lorsqu'un correspondant étranger lui demandait un identifiant LEI avant une transaction ?",
                   "Elle devait demander l'attribution d'un Legal Entity Identifier auprès de l'un des organismes émetteurs répertoriés par la GLEIF."),
            visual("Note_2017_79_ar.pdf", 1, "ar", "definition",
                   "ما رمز شركة ECO MOTOZ NORTH AFRICA كمصدر للاقتطاعات البنكية والبريدية؟",
                   "رمزها 0095.", "ECO MOTOZ NORTH AFRICA | الرمز 0095"),
            manual("Note_2017_81_ar.pdf", 1, "ar", "required_action",
                   "ما المطلوب من البنوك والبريد لضمان السحب النقدي خلال عطلة رأس السنة الميلادية؟",
                   "كان عليها اتخاذ الإجراءات الضرورية لتأمين استمرارية السحب من الموزعات الآلية للأوراق النقدية."),
        ]
    )

    output = stable_ids(curated)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} curated queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
