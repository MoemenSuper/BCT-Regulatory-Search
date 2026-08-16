from __future__ import annotations

import argparse
import json
from pathlib import Path


REMOVE_IDS = {
    "cir_2026_06_ar_eligibility_or_scope_01",
    "cir_2026_06_ar_eligibility_or_scope_02",
    "cir_2026_06_ar_deadline_or_duration_03",
    "cir_2026_06_ar_amount_or_rate_04",
    "cir_2026_06_ar_amount_or_rate_05",
    "note_2026_140_ar_definition_01",
}

UPDATES = {
    "cir_2026_01_fr_amount_or_rate_01": {
        "expected_answer": "Le taux annuel de rémunération de l'épargne est fixé à 6%.",
    },
    "cir_2026_02_fr_definition_01": {
        "expected_answer": "Un bénéficiaire effectif est toute personne physique qui, en dernier lieu, possède ou exerce un contrôle effectif, direct ou indirect, sur le client ou sur la personne physique pour le compte de laquelle une opération de change manuel est réalisée.",
    },
    "cir_2026_02_fr_procedure_or_documents_02": {
        "query": "Que doit faire un bureau de change pour gérer ses risques de blanchiment d'argent, de financement du terrorisme et de financement de la prolifération ?",
        "expected_answer": "Le bureau de change doit mettre en place un dispositif qui identifie et évalue ces risques, puis prévoir des mesures pour les atténuer.",
    },
    "cir_2026_02_fr_procedure_or_documents_06": {
        "query": "Quels renseignements un bureau de change doit-il recueillir au sujet d'une association ?",
        "expected_answer": "Il doit recueillir le nom de l'association, l'adresse de son siège, son identifiant au Registre national des entreprises, l'identité et les numéros de CNI des personnes habilitées à effectuer des opérations financières, ses statuts, la référence de l'extrait du JORT relatif à sa constitution et les éléments permettant d'apprécier sa situation financière.",
    },
    "cir_2026_05_fr_eligibility_or_scope_01": {
        "query": "Quelles institutions figurent sur la liste des adhérents à la Plateforme électronique unique des chèques ?",
    },
    "note_2026_128_ar_definition_01": {
        "query": "ما هو الرمز المسند للمؤسسة المصدرة للاقتطاعات البنكية والبريدية في التحيين الجديد للمصنفة؟",
        "expected_answer": "الرمز المسند للمؤسسة المصدرة للاقتطاعات البنكية والبريدية هو 0197.",
    },
    "note_2026_132_fr_procedure_or_documents_02": {
        "query": "Où les émetteurs de wallets mobiles sont-ils invités à afficher le label TUNPAY ?",
        "expected_answer": "Ils sont invités à déployer le label TUNPAY sur l'ensemble de leur réseau, dans leurs canaux et supports numériques ainsi que dans tous leurs points d'acceptation, notamment chez les commerçants, dans les agences et auprès des agents de paiement.",
    },
    "note_2026_161_ar_deadline_or_duration_01": {
        "query": "ما هو آخر أجل لتقديم مطلب جدولة ديون الفلاحين المتضررين من الجفاف خلال الموسمين 2023-2024 و2024-2025؟",
        "category": "deadline_or_duration",
        "expected_page": 2,
        "expected_answer": "يجب على الفلاح تقديم مطلب الجدولة مباشرة إلى فرع البنك الممول للقرض في أجل لا يتعدى 15 جويلية 2026.",
        "evidence_quote": "يقدم مطلب الجدولة من طرف الفلاح مباشرة إلى فرع البنك الممول للقرض في أجل لا يتعدى 15 جويلية 2026، مصحوبا بشهادة معاينة من المندوبية الجهوية للتنمية الفلاحية.",
    },
    "note_2026_161_ar_prohibition_or_limit_02": {
        "query": "ما هي المدة القصوى لجدولة قروض الفلاحين المتضررين من الجفاف، وهل تتغير نسبة الفائدة؟",
        "category": "deadline_or_duration",
        "expected_page": 2,
        "expected_answer": "تتم جدولة القروض على مدة لا تتجاوز خمس سنوات وبنفس نسبة الفائدة التي مُنحت بها، مع دراسة كل حالة على حدة حسب قدرة الفلاح على السداد ونسبة الضرر من الجفاف.",
        "evidence_quote": "تتم جدولة القروض على مدة لا تتجاوز خمس سنوات وبنفس نسبة الفائدة التي منحت بها، وذلك حالة بحالة مع الأخذ بعين الاعتبار قدرة الفلاح على التسديد ونسبة الضرر من الجفاف.",
    },
    "note_2026_166_ar_definition_01": {
        "expected_answer": "الرمز المخصص لمؤسسة « FACULTE PRIVE SUP SANTE TUNIS » هو 0200.",
        "evidence_quote": "الرمز\nالمؤسسة المصدرة للاقتطاعات البنكية والبريدية\n0200\n« FACULTE PRIVE SUP SANTE TUNIS »",
    },
    "note_2026_22_ar_eligibility_or_scope_01": {
        "query": "هل أصبح الريال العُماني من العملات الأجنبية المسعّرة مقابل الدينار التونسي، وما العمليات التي يشملها ذلك؟",
        "category": "eligibility_or_scope",
        "expected_answer": "نعم، أُدرج الريال العُماني ضمن العملات الأجنبية المسعّرة مقابل الدينار التونسي للعمليات بالحاضر وللأوراق النقدية وصكوك السفر.",
        "evidence_quote": "إدراج الريال العُماني ضمن قائمة العملات الأجنبية المسعّرة مقابل الدينار التونسي للعمليات بالحاضر وللأوراق النقدية وصكوك السفر.",
    },
    "note_2026_59_ar_definition_01": {
        "expected_answer": "تبلغ أبعاد الورقة العُمانية من فئة 50 ريالا في الإصدار السادس 173 مم × 76 مم.",
        "evidence_quote": "الأبعاد\n173 مم × 76 مم\nالفئة\n50 ريالا",
    },
    "note_2026_76_ar_definition_01": {
        "expected_answer": "الرمز المخصص لمؤسسة « SPACE NET » هو 0196.",
        "evidence_quote": "الرمز\nالمؤسسة المصدرة للاقتطاعات البنكية والبريدية\n0196\n« SPACE NET »",
    },
}

MANUAL_ITEMS = [
    {
        "id": "note_2026_140_ar_issuer_codes_01",
        "query": "ما الرمزان المسندان إلى « FREE FITNESS » و« KHAYRI BEN AMOR » في مصنفة مصدري الاقتطاعات البنكية والبريدية؟",
        "language": "ar",
        "category": "definition",
        "relevant": True,
        "expected_source": "Note_2026_140_ar.pdf",
        "expected_page": 1,
        "expected_answer": "الرمز المسند إلى « FREE FITNESS » هو 0198، والرمز المسند إلى « KHAYRI BEN AMOR » هو 0199.",
        "evidence_quote": "الرمز | المؤسسة المصدرة للاقتطاعات البنكية والبريدية\n0198 | « FREE FITNESS »\n0199 | « KHAYRI BEN AMOR »",
    },
    {
        "id": "cir_2026_06_ar_rescheduling_terms_01",
        "query": "ما هي شروط تسوية الديون الفلاحية المتعثرة المصنفة 4 و5 والمصرح بها في 30 سبتمبر 2025؟",
        "language": "ar",
        "category": "procedure_or_documents",
        "relevant": True,
        "expected_source": "Cir_2026_06_ar.pdf",
        "expected_page": 3,
        "expected_answer": "يمكن إعادة جدولة كامل أصل الدين والفوائد التعاقدية الأصلية لمدة أقصاها سبع سنوات، منها سنة إمهال، مع طرح كامل خطايا التأخير. ويشترط دفع 5% من أصل الدين عند تقديم مطلب التسوية.",
        "evidence_quote": "إعادة جدولة كامل أصل الدين وقيمة الفوائد التعاقدية الأصلية الموظفة على مدة أقصاها 7 سنوات، منها سنة إمهال، يتم تحديدها حالة بحالة حسب قدرة كل منتفع على السداد؛ الطرح الكلي لخطايا التأخير الموظفة على هذه الديون. ويشترط للانتفاع بهذه الإجراءات دفع 5% من قيمة أصل الدين عند تقديم مطلب التسوية.",
    },
    {
        "id": "cir_2026_06_ar_full_payment_relief_02",
        "query": "ما الامتيازات الممنوحة عند خلاص كامل الدين الفلاحي المتعثر دون جدولة، وما أجل الخلاص؟",
        "language": "ar",
        "category": "exception_or_condition",
        "relevant": True,
        "expected_source": "Cir_2026_06_ar.pdf",
        "expected_page": 3,
        "expected_answer": "إذا خُلّص كامل الدين دون جدولة خلال أجل أقصاه ستة أشهر من تقديم مطلب التسوية، يطرح البنك كامل خطايا التأخير و50% من قيمة الفوائد التعاقدية الأصلية.",
        "evidence_quote": "بخلاص كامل الدين دون جدولة في أجل أقصاه 6 أشهر من تقديم مطلب التسوية، تتولى البنوك طرح كل خطايا التأخير و50% من قيمة الفوائد التعاقدية الأصلية.",
    },
    {
        "id": "cir_2026_06_ar_application_deadline_03",
        "query": "ما هو آخر أجل لإيداع مطلب تسوية الدين الفلاحي المتعثر، وما التصريح الذي يجب إرفاقه؟",
        "language": "ar",
        "category": "procedure_or_documents",
        "relevant": True,
        "expected_source": "Cir_2026_06_ar.pdf",
        "expected_page": 3,
        "expected_answer": "يجب إيداع المطلب لدى فرع البنك الممول في أجل لا يتعدى 31 ديسمبر 2026، وإرفاقه بتصريح على الشرف يؤكد أن الدين ليس محل تتبعات قضائية في جرائم فساد أو غسل أموال، أو أن حكما باتا بالبراءة صدر بشأنه.",
        "evidence_quote": "أودعوا مطلبا في التسوية لدى فرع البنك الممول في أجل لا يتعدى 31 ديسمبر 2026. ويجب أن يتضمن المطلب تصريحا على الشرف ممضى من طالب التسوية يصرح بموجبه بأن الدين موضوع التسوية ليس محل تتبعات قضائية في جرائم فساد أو غسل أموال أو أنه صادر بشأنه حكم بات بالبراءة.",
    },
    {
        "id": "note_2026_31_ar_irrigated_grain_credit_01",
        "query": "كم يبلغ القرض التكميلي للهكتار الواحد من الحبوب المروية في موسم 2025-2026، وما أجل تسديده؟",
        "language": "ar",
        "category": "amount_or_rate",
        "relevant": True,
        "expected_source": "Note_2026_31_ar.pdf",
        "expected_page": 2,
        "expected_answer": "يبلغ القرض التكميلي 269 دينارا للهكتار الواحد من الحبوب المروية، ويحل أجل تسديده في 31 أوت 2026.",
        "evidence_quote": "حبوب مروية | هكتار | 269 دينارا | 31 أوت 2026",
    },
    {
        "id": "note_2026_31_ar_rainfed_grain_credit_02",
        "query": "ما مبالغ القروض التكميلية للحبوب البعلية في المنطقتين 1 و2 خلال موسم 2025-2026؟",
        "language": "ar",
        "category": "amount_or_rate",
        "relevant": True,
        "expected_source": "Note_2026_31_ar.pdf",
        "expected_page": 2,
        "expected_answer": "يبلغ القرض التكميلي للهكتار 269 دينارا في المنطقة 1 و240 دينارا في المنطقة 2، ويحل أجل التسديد في الحالتين في 31 أوت 2026.",
        "evidence_quote": "حبوب بعلية | منطقة 1 | هكتار | 269 دينارا | 31 أوت 2026\nحبوب بعلية | منطقة 2 | هكتار | 240 دينارا | 31 أوت 2026",
    },
]

VISUALLY_REVIEWED_IDS = {
    "note_2026_161_ar_deadline_or_duration_01",
    "note_2026_161_ar_prohibition_or_limit_02",
    "note_2026_166_ar_definition_01",
    "note_2026_22_ar_eligibility_or_scope_01",
    "note_2026_59_ar_definition_01",
    "note_2026_76_ar_definition_01",
    *(item["id"] for item in MANUAL_ITEMS),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    items = json.loads(args.input.read_text(encoding="utf-8"))
    curated = []
    for item in items:
        if item["id"] in REMOVE_IDS:
            continue
        item = dict(item)
        item.update(UPDATES.get(item["id"], {}))
        curated.append(item)
    curated.extend(MANUAL_ITEMS)
    for item in curated:
        item["evidence_method"] = (
            "visual_review" if item["id"] in VISUALLY_REVIEWED_IDS else "text_extraction"
        )
    curated.sort(key=lambda item: (item["expected_source"].casefold(), item["id"]))
    args.output.write_text(
        json.dumps(curated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"curated {len(items)} generated items into {len(curated)} reviewed items")


if __name__ == "__main__":
    main()
