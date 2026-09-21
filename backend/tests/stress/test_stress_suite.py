"""Stage-labeled stress suite runner.

Every failure must report: Failure stage = <Stage>
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from answer_contract import parse_answer
from answer_evidence import direct_identity, identity_matches
from jsonl_supersession import extract_edges_from_page_text
from retrieval_selection import (
    expand_adjacent_instrument_pages,
    explicit_instrument_identity,
    page_chunks,
    prefer_named_instrument_hits,
)

from .cases import CASES, Case
from .corpus import page_text
from .stages import Stage, require


# Known product gaps: stress cases that currently expose real validator/generation holes.
_KNOWN_GAPS = {}


def _doc(source: str, page: int, text: str | None = None) -> Document:
    body = text if text is not None else page_text(source, page)
    return Document(
        page_content=body,
        metadata={"source": source, "page": page, "pages": [page]},
    )


def _evidence(source: str, page: int, score: float = 0.9) -> dict:
    return {
        "evidence_id": "E1",
        "source": source,
        "page": page,
        "score": score,
        "text": page_text(source, page),
    }


def _draft(claim: str, quote: str, evidence_id: str = "E1") -> dict:
    return {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": claim,
                "quotes": [{"evidence_id": evidence_id, "quote": quote}],
            }
        ],
    }


def _parse(question: str, evidence: list[dict], claim: str, quote: str):
    return parse_answer(
        json.dumps(_draft(claim, quote), ensure_ascii=False),
        question,
        evidence,
    )


def _top_name(ranked) -> str:
    return Path(str(ranked[0][0].metadata["source"])).name


def _contains(haystack: str, needle: str) -> bool:
    if needle in haystack or needle.casefold() in haystack.casefold():
        return True
    soft = needle.replace("'", "\u2019")
    return soft in haystack or soft.casefold() in haystack.casefold()


def run_case(case: Case) -> None:
    kind = case.kind
    data = case.data
    stage = case.stage
    cid = case.id

    if kind in {"prefer_named", "prefer_named_domain"}:
        named = _doc(*data["named"])
        distractor = _doc(*data["distractor"])
        ranked = prefer_named_instrument_hits(
            [
                (distractor, float(data["distractor_score"])),
                (named, float(data["named_score"])),
            ],
            explicit_instrument_identity(data["query"]),
        )
        require(
            _top_name(ranked) == data["expect_top"],
            stage=stage,
            case_id=cid,
            detail=f"top={_top_name(ranked)} expected={data['expect_top']}",
        )
        return

    if kind == "identity":
        identity = explicit_instrument_identity(data["query"])
        require(
            identity is not None
            and identity["kind"] == data["kind"]
            and identity["year"] == data["year"]
            and identity["number"] == data["number"],
            stage=stage,
            case_id=cid,
            detail=f"identity={identity}",
        )
        return

    if kind == "identity_none":
        require(
            explicit_instrument_identity(data["query"]) is None,
            stage=stage,
            case_id=cid,
            detail="expected no hard instrument anchor",
        )
        return

    if kind == "expand_adjacent":
        hit = _doc(*data["hit"])
        pages = page_chunks([hit, _doc(data["hit"][0], data["must_include_page"])])
        expanded = expand_adjacent_instrument_pages([(hit, 1.0)], pages)
        sources = {
            (Path(str(doc.metadata["source"])).name, int(doc.metadata["page"]))
            for doc, _ in expanded
        }
        require(
            (data["hit"][0], data["must_include_page"]) in sources,
            stage=stage,
            case_id=cid,
            detail=f"expanded={sources}",
        )
        joined = "\n".join(doc.page_content for doc, _ in expanded)
        require(
            _contains(joined, data["must_contain"]),
            stage=stage,
            case_id=cid,
            detail=f"missing {data['must_contain']!r} in expanded pack",
        )
        return

    if kind == "expand_order":
        hit = _doc(*data["hit"])
        neighbor = _doc(data["hit"][0], data["hit"][1] + 1)
        pages = page_chunks([hit, neighbor])
        expanded = expand_adjacent_instrument_pages([(hit, 1.0)], pages)
        page_nums = [int(doc.metadata["page"]) for doc, _ in expanded]
        require(
            page_nums == sorted(page_nums),
            stage=stage,
            case_id=cid,
            detail=f"page order {page_nums}",
        )
        return

    if kind == "page_assembly_operator":
        text = page_text(data["source"], data["page"])
        require(
            _contains(text, data["must_contain"]),
            stage=stage,
            case_id=cid,
            detail=f"operator missing from {data['source']} p.{data['page']}",
        )
        return

    if kind == "pack_needs_both":
        g = page_text(*data["general"])
        e = page_text(*data["exception"])
        require(
            _contains(g, data["general_must"]),
            stage=stage,
            case_id=cid,
            detail="general rule missing from pack page",
        )
        require(
            _contains(e, data["exception_must"]),
            stage=stage,
            case_id=cid,
            detail="exception missing from pack page",
        )
        return

    if kind == "definition_elsewhere":
        text = page_text(data["source"], data["page"])
        require(
            _contains(text, data["term_marker"]),
            stage=stage,
            case_id=cid,
            detail="definition pointer missing",
        )
        for annex_page in data["annex_pages"]:
            annex = page_text(data["source"], annex_page)
            require(
                not annex.strip(),
                stage=stage,
                case_id=cid,
                detail=f"expected empty OCR annex p.{annex_page}, got {len(annex)} chars",
            )
        return

    if kind == "annex_empty":
        for annex_page in data["annex_pages"]:
            annex = page_text(data["source"], annex_page)
            require(
                not annex.strip(),
                stage=stage,
                case_id=cid,
                detail=f"annex p.{annex_page} not empty",
            )
        return

    if kind == "extract_empty":
        text = page_text(data["source"], data["page"])
        edges = extract_edges_from_page_text(
            filename=data["source"], page_number=data["page"], text=text
        )
        require(
            edges == [],
            stage=stage,
            case_id=cid,
            detail=f"unexpected edges={edges}",
        )
        return

    if kind == "extract_action":
        text = page_text(data["source"], data["page"])
        edges = extract_edges_from_page_text(
            filename=data["source"], page_number=data["page"], text=text
        )
        require(bool(edges), stage=stage, case_id=cid, detail="no edges extracted")
        edge = edges[0]
        require(
            edge.action == data["expect_action"],
            stage=stage,
            case_id=cid,
            detail=f"action={edge.action}",
        )
        require(
            edge.target_instrument == data["expect_target"],
            stage=stage,
            case_id=cid,
            detail=f"target={edge.target_instrument}",
        )
        if "expect_source_instrument" in data:
            require(
                edge.source_instrument == data["expect_source_instrument"],
                stage=stage,
                case_id=cid,
                detail=f"source={edge.source_instrument}",
            )
        if "forbid_target" in data:
            require(
                all(e.target_instrument != data["forbid_target"] for e in edges),
                stage=stage,
                case_id=cid,
                detail="forbidden target present",
            )
        return

    if kind in {
        "parse_accept",
        "polarity_reject",
        "parse_reject",
        "parse_reject_topical",
        "regime_reject",
    }:
        evidence = [_evidence(data["source"], data["page"])]
        result = parse_answer(
            json.dumps(_draft(data["claim"], data["quote"]), ensure_ascii=False),
            data["question"],
            evidence,
            temporal_unverified=data.get("temporal_unverified"),
        )
        if kind == "parse_accept":
            require(
                result["status"] in {"answered", "partial_answer"},
                stage=stage,
                case_id=cid,
                detail=f"status={result.get('status')} diagnostics={result.get('diagnostics')}",
            )
            return
        require(
            result["status"] not in {"answered"},
            stage=stage,
            case_id=cid,
            detail=(
                f"incorrect answer accepted: status={result.get('status')} "
                f"answer={result.get('answer')!r}"
            ),
        )
        return

    if kind == "franken_reject":
        evidence = []
        for index, (source, page) in enumerate(data["sources"], start=1):
            record = _evidence(source, page)
            record["evidence_id"] = f"E{index}"
            evidence.append(record)
        quote_source = data["bad_quote_source"]
        eid = next(
            e["evidence_id"]
            for e in evidence
            if Path(str(e["source"])).name == quote_source
        )
        result = parse_answer(
            json.dumps(
                {
                    "status": "answered",
                    "message": "",
                    "claims": [
                        {
                            "text": data["bad_claim"],
                            "quotes": [
                                {"evidence_id": eid, "quote": data["bad_quote"]}
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            data["question"],
            evidence,
        )
        require(
            result["status"] != "answered"
            or "60" not in (result.get("answer") or "")
            or "120" not in (result.get("answer") or ""),
            stage=stage,
            case_id=cid,
            detail=f"franken accepted: {result}",
        )
        return

    if kind == "parse_insufficient":
        result = parse_answer(
            json.dumps(
                {
                    "status": "answered",
                    "message": "",
                    "claims": [
                        {
                            "text": "Le taux est de 42%.",
                            "quotes": [{"evidence_id": "E1", "quote": "42%"}],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            data["question"],
            data["evidence"],
        )
        require(
            result["status"]
            in {"insufficient_evidence", "search_results", "clarification_needed"},
            stage=stage,
            case_id=cid,
            detail=f"status={result.get('status')}",
        )
        return

    if kind == "citation_identity":
        evidence = [_evidence(data["source"], data["page"])]
        result = parse_answer(
            json.dumps(_draft(data["claim"], data["quote"]), ensure_ascii=False),
            data["question"],
            evidence,
            temporal_unverified=data.get("temporal_unverified"),
        )
        require(
            result["status"] in {"answered", "partial_answer"},
            stage=stage,
            case_id=cid,
            detail=f"status={result.get('status')} diagnostics={result.get('diagnostics')}",
        )
        sources = result.get("sources") or []
        require(bool(sources), stage=stage, case_id=cid, detail="no sources")
        require(
            sources[0]["file"] == data["expect_file"],
            stage=stage,
            case_id=cid,
            detail=f"file={sources[0].get('file')}",
        )
        require(
            int(sources[0]["page"]) == int(data["expect_page"]),
            stage=stage,
            case_id=cid,
            detail=f"page={sources[0].get('page')}",
        )
        return

    if kind == "named_only":
        identity = direct_identity(data["question"])
        require(identity is not None, stage=stage, case_id=cid, detail="no identity")
        require(
            identity_matches(data["right_source"], identity),
            stage=stage,
            case_id=cid,
            detail="right source should match",
        )
        require(
            not identity_matches(data["wrong_source"], identity),
            stage=stage,
            case_id=cid,
            detail="wrong source should not match named identity",
        )
        return

    raise AssertionError(
        f"Failure stage = {stage.value}\ncase={cid}\nunknown kind={kind}"
    )


def _parametrize_cases():
    for case in CASES:
        marks = []
        if case.id in _KNOWN_GAPS:
            marks.append(
                pytest.mark.xfail(
                    reason=f"Failure stage = {_KNOWN_GAPS[case.id]}",
                    strict=True,
                )
            )
        yield pytest.param(case, id=case.id, marks=marks)


@pytest.mark.parametrize("case", list(_parametrize_cases()))
def test_stress_case(case: Case):
    run_case(case)


def test_stress_taxonomy_coverage_is_complete():
    """Every major taxonomy bucket from the goal must have executable cases."""
    ids = {c.id for c in CASES}
    required_prefixes = [
        *[f"1.{letter}" for letter in "ABCDEFGHIJKLMNOPQRSTUVW"],
        *[f"2.{letter}" for letter in "ABCDEFGHI"],
        *[f"3.{letter}" for letter in "ABCDEFGHIJKLMN"],
        *[f"4.{letter}" for letter in "ABCDEFGHIJ"],
        *[f"5.{letter}" for letter in "ABCDEFGHIJKLMNOPQRST"],
    ]
    missing = [item for item in required_prefixes if item not in ids]
    require(
        not missing,
        stage=Stage.VALIDATION,
        case_id="coverage_matrix",
        detail=f"missing taxonomy ids: {missing}",
    )
    for prefix in ("6.", "7.", "8.", "9.", "10."):
        require(
            any(i.startswith(prefix) for i in ids),
            stage=Stage.VALIDATION,
            case_id="coverage_matrix",
            detail=f"missing section {prefix}",
        )
