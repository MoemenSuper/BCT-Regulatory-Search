"""Authority tags, query class, and Google multimodal embed wiring."""
from __future__ import annotations

import json

import numpy as np
import pytest

from document_authority import authority_for_kind, normalize_doc_kind, resolve_doc_kind
from query_authority import (
    allowed_doc_kinds,
    classify_query_authority,
    evidence_kind_allowed,
    select_demonstrations,
)
from answer_gates import parse_answer


def test_doc_kind_maps_to_authority():
    assert normalize_doc_kind("bulletin") == "statistical"
    assert authority_for_kind("statistical") == "secondary"
    assert authority_for_kind("regulatory") == "primary"
    assert resolve_doc_kind(filename="Stats_export_2024_fr.pdf") == "statistical"
    assert resolve_doc_kind(filename="Cir_2026_04_fr.pdf") == "regulatory"


def test_allowed_kinds_never_reject_uncertain():
    assert allowed_doc_kinds("uncertain") == frozenset({"regulatory"})
    assert allowed_doc_kinds("statistical_fact") == frozenset({"statistical", "regulatory"})
    assert evidence_kind_allowed("regulatory_rule", doc_kind="statistical") is False
    assert evidence_kind_allowed("statistical_fact", doc_kind="statistical") is True


def test_dynamic_demos_cover_hard_negative_taux():
    demos = select_demonstrations("Quel est le taux applicable selon la circulaire ?")
    labels = {label for _q, label, _n in demos}
    assert "regulatory_rule" in labels
    assert len(demos) >= 4


def test_classify_query_authority_fails_closed():
    class Boom:
        def invoke(self, _prompt):
            raise RuntimeError("down")

    payload = classify_query_authority(Boom(), "Quel était le volume d'exportations ?")
    assert payload["query_class"] == "uncertain"


def test_classify_query_authority_parses_structured_json():
    class Fake:
        def invoke(self, _prompt):
            return (
                '{"query_class":"statistical_fact","confidence":"high",'
                '"rationale":"volume bulletin"}'
            )

    payload = classify_query_authority(Fake(), "volume d'exportations 2024")
    assert payload["query_class"] == "statistical_fact"


def test_parse_answer_rejects_secondary_on_regulatory_query():
    evidence = [
        {
            "evidence_id": "E1",
            "source": "Bulletin_2024_fr.pdf",
            "page": 1,
            "text": "Le volume d'exportations atteint 12 milliards.",
            "score": 0.9,
            "doc_kind": "statistical",
            "authority": "secondary",
        }
    ]
    draft = {
        "status": "answered",
        "message": "",
        "claims": [
            {
                "text": "Le volume d'exportations atteint 12 milliards.",
                "quotes": [
                    {
                        "evidence_id": "E1",
                        "quote": "Le volume d'exportations atteint 12 milliards.",
                    }
                ],
            }
        ],
    }
    denied = parse_answer(
        json.dumps(draft),
        "Quel est le plafond reglementaire ?",
        evidence,
        query_class="regulatory_rule",
    )
    assert denied["status"] == "insufficient_evidence"

    allowed = parse_answer(
        json.dumps(draft),
        "Quel etait le volume d'exportations ?",
        evidence,
        query_class="statistical_fact",
    )
    assert allowed["status"] == "answered"
    assert allowed["sources"]


def test_secondary_page_images_cover_internal(tmp_path):
    pytest.importorskip("pymupdf")
    import pymupdf
    from pathlib import Path

    from ingestion.models import Page, StructuredDocument
    from ingestion.pipeline import _ensure_secondary_page_images

    pdf_path = tmp_path / "Memo_interne_scan.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Procedure interne")
    doc.save(pdf_path)
    doc.close()

    structured = StructuredDocument(
        filename=pdf_path.name,
        pages=[Page(page_number=1, raw_text="")],
        metadata={"doc_kind": "internal"},
    )
    written = _ensure_secondary_page_images(structured, pdf_path, tmp_path)
    assert written == 1
    assert Path(structured.pages[0].metadata["page_image_path"]).is_file()

    regulatory = StructuredDocument(
        filename="Cir.pdf",
        pages=[Page(page_number=1, raw_text="x")],
        metadata={"doc_kind": "regulatory"},
    )
    assert _ensure_secondary_page_images(regulatory, pdf_path, tmp_path) == 0


def test_google_cloud_spec_is_embedding_2():
    from cloud_embed_clients import CLOUD_EMBED_SPECS

    google = CLOUD_EMBED_SPECS["google"]
    assert google.model == "gemini-embedding-2"
    assert google.dimension == 768


def test_google_embed_document_chunks_passes_image(tmp_path, monkeypatch):
    import sys
    import types

    import cloud_embed_clients as cec

    calls = []
    dim = cec.CLOUD_EMBED_SPECS["google"].dimension
    unit = np.zeros(dim, dtype=np.float32)
    unit[0] = 1.0

    class FakeEmb:
        def __init__(self):
            self.values = unit.tolist()

    class FakeResponse:
        embeddings = [FakeEmb()]
        usage_metadata = None

    class FakeModels:
        def embed_content(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda api_key=None: FakeClient()
    fake_types = types.ModuleType("google.genai.types")

    class Part:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class EmbedContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types.Part = Part
    fake_types.EmbedContentConfig = EmbedContentConfig
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cec, "_gemini_api_keys", lambda: ["test-key"])

    client = cec.GoogleRuntimeClient(tmp_path)
    png = b"\x89PNG\r\n\x1a\nfake"
    vectors = client.embed_document_chunks(
        ["Legende: exportations 2024"],
        images=[png],
        titles=["Bulletin_2024"],
    )
    assert vectors.shape == (1, dim)
    assert abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-5
    assert len(calls) == 1
    contents = calls[0]["contents"]
    assert isinstance(contents, list)
    assert "title: Bulletin_2024" in contents[0]
    assert calls[0]["model"] == "gemini-embedding-2"


def test_chart_suspect_page_triggers_visual_plan(tmp_path, monkeypatch):
    pytest.importorskip("pymupdf")
    import pymupdf

    from ingestion.extract import PdfExtractor, page_chart_signals
    from ingestion.gemini_visual import VisualPage

    path = tmp_path / "Bulletin_chart_fr.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Figure 1. Exportations 2024")
    # Large filled rect approximates a chart footprint for drawing clusters.
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(80, 120, 500, 500))
    shape.finish(color=(0, 0, 0), fill=(0.8, 0.8, 0.9), width=1)
    shape.commit()
    doc.save(path)
    doc.close()

    with pymupdf.open(path) as pdf:
        signals = page_chart_signals(pdf.load_page(0))
    assert signals.suspect or signals.drawing_cluster_count >= 0

    class FakeVisual:
        model = "gemini-test"

        def __init__(self):
            self.calls = 0

        def transcribe(self, *, image_png, source_pdf_sha256, page_number):
            self.calls += 1
            return VisualPage(
                transcription="Figure 1. Exportations 2024",
                items=[],
                uncertain_regions=[],
                complete=True,
                contains_chart=True,
                chart_notes="Serie A: 12",
            )

    fake = FakeVisual()
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_CHART_VISION", "1")
    monkeypatch.delenv("BCT_ALLOW_DEGRADED_INGESTION", raising=False)

    # Force chart suspect for deterministic CI even if drawings don't cluster.
    monkeypatch.setattr(
        "ingestion.extract.page_chart_signals",
        lambda _page: type(signals)(
            image_count=1,
            drawing_cluster_count=1,
            max_image_area_ratio=0.2,
            max_drawing_area_ratio=0.2,
            suspect=True,
        ),
    )
    structured = PdfExtractor(visual_transcriber=fake).extract(path)
    page_out = structured.pages[0]
    assert fake.calls == 1
    assert page_out.metadata.get("has_chart") is True
    assert "Serie A: 12" in page_out.raw_text or page_out.metadata.get("chart_notes") == "Serie A: 12"
