from __future__ import annotations

import argparse
import json
from pathlib import Path


NEGATIVE_CASES = [
    {
        "id": "negative_fr_out_of_scope_weather",
        "query": "Quel temps fera-t-il demain à Tunis ?",
        "language": "fr",
        "category": "out_of_scope",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "reject_out_of_scope",
    },
    {
        "id": "negative_ar_out_of_scope_recipe",
        "query": "كيف أعد طبق كسكسي تونسي في المنزل؟",
        "language": "ar",
        "category": "out_of_scope",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "reject_out_of_scope",
    },
    {
        "id": "negative_fr_future_exchange_rate",
        "query": "Quel sera demain le taux de change du dinar face à l'euro ?",
        "language": "fr",
        "category": "not_in_corpus",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "abstain",
    },
    {
        "id": "negative_ar_future_regulation",
        "query": "ما هي شروط منشور البنك المركزي الذي سيصدر سنة 2027؟",
        "language": "ar",
        "category": "not_in_corpus",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "abstain",
    },
    {
        "id": "negative_fr_current_status_unknown",
        "query": "Quel est aujourd'hui le plafond en vigueur pour une allocation de voyage d'affaires ?",
        "language": "fr",
        "category": "current_status_unknown",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "abstain",
    },
    {
        "id": "negative_ar_current_status_unknown",
        "query": "هل ما زالت قيود تمويل توريد المنتجات غير ذات الأولوية سارية اليوم؟",
        "language": "ar",
        "category": "current_status_unknown",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "abstain",
    },
    {
        "id": "negative_fr_ambiguous_documents",
        "query": "Quels documents dois-je fournir à ma banque pour effectuer mon transfert ?",
        "language": "fr",
        "category": "ambiguous",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "clarify",
    },
    {
        "id": "negative_ar_ambiguous_allowance",
        "query": "ما هو مبلغ المنحة التي يمكنني الحصول عليها للسفر؟",
        "language": "ar",
        "category": "ambiguous",
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_answer": None,
        "evidence_quote": None,
        "expected_behavior": "clarify",
    },
]


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    items = [item for path in args.inputs for item in load(path)]
    items.extend(NEGATIVE_CASES)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} evaluation queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
