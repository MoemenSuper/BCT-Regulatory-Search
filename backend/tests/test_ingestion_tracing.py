from __future__ import annotations

import pytest
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ingestion.pipeline import IngestionConfig, IngestionPipeline


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    LangfuseResourceManager._instances.clear()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    yield lambda: (client.flush(), exporter.get_finished_spans())[1]
    client.shutdown()
    LangfuseResourceManager._instances.clear()


def test_failed_ingest_emits_error_trace(tmp_path, spans):
    bad = tmp_path / "Stats_fr.pdf"
    bad.write_bytes(b"not a pdf at all")
    pipeline = IngestionPipeline(
        IngestionConfig(
            asset_root=tmp_path / "assets",
            documents_dir=tmp_path / "docs",
            registry_path=tmp_path / "ingestion.sqlite3",
            gemini_cache_dir=tmp_path / "cache",
        )
    )
    try:
        with pytest.raises(ValueError):
            pipeline.ingest(bad, metadata={"doc_kind": "statistical"})
    finally:
        pipeline.close()

    by_name = {span.name: span for span in spans()}
    assert {"ingest-document", "validate-pdf"} <= set(by_name)
    root = by_name["ingest-document"]
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert "PDF signature" in root.attributes["langfuse.observation.status_message"]
    assert by_name["validate-pdf"].parent.span_id == root.context.span_id
