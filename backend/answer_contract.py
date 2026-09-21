"""Grounded answers over BCT instruments: public seam for conversation and tests.

Implementation: answer_gates (claim validation) and answer_draft (write ladder).
"""
from answer_gates import (  # noqa: F401
    ANSWER_SCHEMA,
    ANSWER_SCHEMA_FOR_PROMPT,
    AnswerDraft,
    Claim,
    Quote,
    language_of,
    parse_answer,
    safe_response,
    _load_answer_payload,
    _question_anchors,
)
from answer_draft import (  # noqa: F401
    evidence_records,
    format_refusal_reason,
    generate_grounded_answer,
    present_top5_synthesis,
    refusal_reason_bucket,
    refusal_reason_title,
    search_response,
    select_evidence,
    try_literal_evidence_partial,
    try_supersession_partial_answer,
    _annotate_graph_supersession,
    _repair_answer_schema,
)
