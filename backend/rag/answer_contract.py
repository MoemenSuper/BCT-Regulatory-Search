"""The single entry point for answering: other files import from here, not from the two files
that do the work.

- answer_draft.py: asks the answer model to choose evidence, then to write the answer, with
  fallbacks when a draft is rejected.
- answer_gates.py: the checkers that accept or reject each claim of a draft (exact quotes,
  numbers present in the page, the right document).
"""
from rag.answer_gates import (  # noqa: F401
    ANSWER_SCHEMA_FOR_PROMPT,
    AnswerDraft,
    Claim,
    language_of,
    parse_answer,
    safe_response,
    _load_answer_payload,
    _question_anchors,
)
from rag.answer_draft import (  # noqa: F401
    REFUSAL_BUCKETS,
    evidence_records,
    format_refusal_reason,
    generate_grounded_answer,
    refusal_reason_bucket,
    refusal_reason_title,
    search_response,
    select_evidence,
    try_supersession_partial_answer,
    _annotate_supersession,
)
