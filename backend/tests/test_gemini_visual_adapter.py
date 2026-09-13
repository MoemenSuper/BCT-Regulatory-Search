import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestion.gemini_visual import GeminiVisualTranscriber


class FakeInteractions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload, ensure_ascii=False), id="interaction-1")


class FakeClient:
    def __init__(self, payload):
        self.interactions = FakeInteractions(payload)


def test_gemini_adapter_uses_structured_output_and_bound_cache(tmp_path: Path):
    payload = {
        "transcription": "التاريخ ١١ أكتوبر ٢٠٢٦",
        "items": [
            {
                "literal": "١١ أكتوبر ٢٠٢٦",
                "kind": "date",
                "context": "التاريخ ١١ أكتوبر ٢٠٢٦",
                "uncertain": False,
            }
        ],
        "uncertain_regions": [],
        "complete": True,
    }
    client = FakeClient(payload)
    adapter = GeminiVisualTranscriber(tmp_path, client=client, model="gemini-3.7-flash")
    image = b"fake-png-bytes"

    first = adapter.transcribe(image_png=image, source_pdf_sha256="abc123", page_number=4)
    second = adapter.transcribe(image_png=image, source_pdf_sha256="abc123", page_number=4)

    assert first == second
    assert first.complete is True
    assert client.interactions.calls == 1
    assert client.interactions.last_kwargs["response_format"]["mime_type"] == "application/json"
    assert client.interactions.last_kwargs["generation_config"]["thinking_level"] == "low"


def test_gemini_adapter_rejects_literal_not_present_in_transcription(tmp_path: Path):
    payload = {
        "transcription": "النص الصحيح",
        "items": [
            {
                "literal": "١١ أكتوبر ٢٠٢٦",
                "kind": "date",
                "context": "١١ أكتوبر ٢٠٢٦",
                "uncertain": False,
            }
        ],
        "uncertain_regions": [],
        "complete": True,
    }
    adapter = GeminiVisualTranscriber(tmp_path, client=FakeClient(payload))

    with pytest.raises(ValueError, match="not bound to its transcription"):
        adapter.transcribe(image_png=b"image", source_pdf_sha256="abc123", page_number=1)
