from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from .identity import build_catalog, instrument_from_filename
from .models import RelationshipCandidate
from .store import Neo4jGraphLiteStore
from .validation import validate_extracted_relationship


logger = logging.getLogger(__name__)

# Cheap prefilter: do not spend an LLM call on pages with no obvious BCT relation signal.
_RELATION_SIGNAL = re.compile(
    r"\b(?:modifi\w*|remplac\w*|abrog\w*|cit(?:e|é|ée|és|ées|ent)?|"
    r"référenc\w*|conformément|vu\s+la|vu\s+le|amend\w*|replace\w*|"
    r"repeal\w*|referenc\w*)\b|"
    r"(?:يعدل|تعدل|ينقح|تنقح|يعوض|تعوض|يستبدل|تستبدل|يلغي|تلغي|"
    r"تحيل|يشير|تشير|طبقا|عملا|بالرجوع)",
    re.IGNORECASE,
)
_INSTRUMENT_SIGNAL = re.compile(
    r"\b(?:circulair(?:e|es)|circulars?|notes?)\b.{0,80}"
    r"(?:(?:19|20)\d{2}.{0,20}\d+|\d{2}\s*[-‑–—/]\s*\d+)|"
    r"(?:المنشور|منشور|المذكرة|مذكرة).{0,80}(?:19|20)\d{2}",
    re.IGNORECASE | re.DOTALL,
)


def page_may_contain_relationship(text: str) -> bool:
    return bool(_RELATION_SIGNAL.search(text) and _INSTRUMENT_SIGNAL.search(text))


def _normalize_extraction_json(content: str) -> dict:
    """Coerce common Groq JSON shapes into nodes/relationships dicts."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {"nodes": [], "relationships": []}
    if not isinstance(payload, dict):
        return {"nodes": [], "relationships": []}

    nodes_in = payload.get("nodes")
    if not isinstance(nodes_in, list):
        nodes_in = []
    nodes_out = []
    for index, node in enumerate(nodes_in):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is None:
            node_id = node.get("index", index)
        label = node.get("label") or node.get("type") or "Instrument"
        properties = node.get("properties")
        if not isinstance(properties, dict):
            properties = {
                key: value
                for key, value in node.items()
                if key not in {"id", "index", "label", "type", "properties", "embedding_properties"}
            }
        nodes_out.append(
            {
                "id": str(node_id),
                "label": str(label),
                "properties": properties,
            }
        )

    rels_in = payload.get("relationships")
    if not isinstance(rels_in, list):
        rels_in = payload.get("rels") if isinstance(payload.get("rels"), list) else []
    rels_out = []
    for relationship in rels_in:
        if not isinstance(relationship, dict):
            continue
        start = relationship.get("start_node_id", relationship.get("source"))
        end = relationship.get("end_node_id", relationship.get("target"))
        if start is None or end is None:
            continue
        properties = relationship.get("properties")
        if not isinstance(properties, dict):
            properties = {
                key: value
                for key, value in relationship.items()
                if key
                not in {
                    "type",
                    "start_node_id",
                    "end_node_id",
                    "source",
                    "target",
                    "properties",
                    "embedding_properties",
                }
            }
        rels_out.append(
            {
                "type": str(relationship.get("type") or ""),
                "start_node_id": str(start),
                "end_node_id": str(end),
                "properties": properties,
            }
        )

    return {"nodes": nodes_out, "relationships": rels_out}


_SCHEMA = {
    "node_types": [
        {
            "label": "Instrument",
            "description": "A BCT circular or note explicitly identified in the page.",
            "properties": [
                {"name": "name", "type": "STRING"},
                {"name": "kind", "type": "STRING"},
                {"name": "year", "type": "INTEGER"},
                {"name": "number", "type": "INTEGER"},
            ],
        },
        {
            "label": "Provision",
            "description": "An explicitly named article, section or provision of a BCT instrument.",
            "properties": [
                {"name": "name", "type": "STRING"},
                {"name": "instrument", "type": "STRING"},
                {"name": "label", "type": "STRING"},
            ],
        },
    ],
    "relationship_types": [
        {
            "label": label,
            "properties": [
                {"name": "evidence_quote", "type": "STRING"},
                {"name": "effective_date", "type": "STRING"},
            ],
        }
        for label in ("CITES", "AMENDS", "REPLACES", "ABROGATES")
    ],
    "patterns": [
        ["Instrument", relation, "Instrument"]
        for relation in ("CITES", "AMENDS", "REPLACES", "ABROGATES")
    ]
    + [
        ["Provision", relation, "Provision"]
        for relation in ("AMENDS", "REPLACES", "ABROGATES")
    ],
    "additional_node_types": False,
    "additional_relationship_types": False,
    "additional_patterns": False,
}

_EXTRACTION_PROMPT = r"""
You extract ONLY explicit BCT regulatory relationships from the supplied text.
Use the schema exactly. Prefer extracting nothing over inferring a legal effect.
Return a single JSON object with "nodes" and "relationships" arrays that matches
this exact shape:
{{"nodes":[{{"id":"0","label":"Instrument","properties":{{"name":"Cir_2016_01_fr.pdf","kind":"cir","year":2016,"number":1}}}},{{"id":"1","label":"Instrument","properties":{{"name":"circulaire 2001-11","kind":"cir","year":2001,"number":11}}}}],"relationships":[{{"type":"ABROGATES","start_node_id":"0","end_node_id":"1","properties":{{"evidence_quote":"notamment la circulaire n 2001-11 du 4 mai 2001"}}}}]}}

The text begins with trusted metadata lines SOURCE_DOCUMENT_ID, SOURCE_FILE and
SOURCE_PAGE followed by CONTENT. Those metadata lines identify the instrument
that issued the page; they are not evidence and must never be copied into
`evidence_quote`.

Direction rule: the SOURCE_DOCUMENT is the START node for CITES, AMENDS,
REPLACES or ABROGATES, and the older/referenced BCT instrument is the END node.
For provision-level changes, the source provision belongs to SOURCE_DOCUMENT and
the target provision belongs to the explicitly identified target instrument.

Only emit AMENDS, REPLACES or ABROGATES when the CONTENT itself explicitly says
that the SOURCE_DOCUMENT performs that legal action. A recital describing that
an external law or some other text was previously modified is NOT an action by
the source document. Do not infer replacement from chronology or similar titles.

Every relationship MUST contain `evidence_quote`: a short verbatim quotation
copied from CONTENT, not paraphrased. The quote must include the target BCT
instrument identity and, for AMENDS/REPLACES/ABROGATES, the action wording.
If the quote cannot satisfy this, emit no relationship. `effective_date` is
optional; include it only when the same quoted wording explicitly establishes it.

Instrument properties should identify kind (`cir` or `note`), year and number.
Use four-digit years only (for example expand `91-24` to year 1991, number 24).
Provision nodes are allowed only when the article/section/provision is explicitly
named. Do not create generic people, organizations, topics or concepts.

Text:
{text}

Schema:
{schema}

Examples:
{examples}
"""


@dataclass
class CandidateWriterReport:
    accepted: list[RelationshipCandidate] = field(default_factory=list)
    rejected_count: int = 0


def _graph_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Graph Lite extraction")
    model = os.environ.get(
        "BCT_GRAPH_BUILDER_MODEL",
        os.environ.get("BCT_GROQ_MODEL", "openai/gpt-oss-120b"),
    )
    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0,
        max_tokens=int(os.environ.get("BCT_GRAPH_BUILDER_MAX_TOKENS", "4096")),
    )


class GraphLiteBuilder:
    """Extract relationships via Groq JSON, then gate and store verified edges."""

    def __init__(
        self,
        *,
        driver,
        store: Neo4jGraphLiteStore,
        catalog: dict,
        activate_immediately: bool = True,
    ) -> None:
        self.driver = driver
        self.store = store
        self.catalog = catalog
        self.activate_immediately = bool(activate_immediately)

    def _store_payload(
        self,
        *,
        source_file: str,
        source_page: int,
        source_text: str,
        payload: dict,
    ) -> CandidateWriterReport:
        report = CandidateWriterReport()
        nodes_by_id = {}
        for node in payload.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            label = node.get("label")
            if str(label).casefold() not in {"instrument", "provision"}:
                continue
            nodes_by_id[str(node.get("id", ""))] = {
                "label": label,
                "properties": dict(node.get("properties") or {}),
            }

        for relationship in payload.get("relationships") or []:
            if not isinstance(relationship, dict):
                continue
            candidate = validate_extracted_relationship(
                source_file=source_file,
                source_page=source_page,
                source_text=source_text,
                nodes_by_id=nodes_by_id,
                relationship={
                    "type": relationship.get("type"),
                    "start_node_id": relationship.get("start_node_id"),
                    "end_node_id": relationship.get("end_node_id"),
                    "properties": dict(relationship.get("properties") or {}),
                },
                catalog=self.catalog,
            )
            if candidate is None:
                report.rejected_count += 1
                continue
            self.store.upsert_candidate(candidate, active=self.activate_immediately)
            report.accepted.append(candidate)
        return report

    def ingest_page(self, *, source_file: str, page: int, text: str) -> CandidateWriterReport:
        source = instrument_from_filename(source_file)
        if source is None:
            raise ValueError(f"Cannot parse a BCT instrument identity from {source_file!r}")
        if page < 1 or not text.strip():
            raise ValueError("page must be >= 1 and text must be non-empty")
        if not page_may_contain_relationship(text):
            return CandidateWriterReport()

        payload_text = (
            f"SOURCE_DOCUMENT_ID: {source.id}\n"
            f"SOURCE_FILE: {Path(source_file).name}\n"
            f"SOURCE_PAGE: {page}\n"
            "CONTENT:\n"
            f"{text}"
        )
        prompt = _EXTRACTION_PROMPT.format(
            text=payload_text,
            schema=json.dumps(_SCHEMA, ensure_ascii=False),
            examples="(none)",
        )
        response = _graph_llm().invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return self._store_payload(
            source_file=source_file,
            source_page=page,
            source_text=text,
            payload=_normalize_extraction_json(content),
        )


def create_builder_from_environment(
    *,
    driver,
    store: Neo4jGraphLiteStore,
    documents_dir: str | Path | None = None,
    chunk_files: Iterable[str | Path] = (),
    activate_immediately: bool = True,
) -> GraphLiteBuilder:
    catalog = build_catalog(documents_dir=documents_dir, chunk_files=chunk_files)
    if not catalog:
        raise RuntimeError("No BCT instrument identities were found for Graph Lite catalog validation")
    return GraphLiteBuilder(
        driver=driver,
        store=store,
        catalog=catalog,
        activate_immediately=activate_immediately,
    )
