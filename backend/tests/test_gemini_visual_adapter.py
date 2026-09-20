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


def test_gemini_adapter_marks_literal_not_present_in_transcription_uncertain(tmp_path: Path):
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

    page = adapter.transcribe(image_png=b"image", source_pdf_sha256="abc123", page_number=1)

    # The transcription stays usable and complete; only the index entry is distrusted.
    assert page.complete is True
    assert page.transcription == "النص الصحيح"
    assert page.items[0].uncertain is True
    assert page.uncertain_regions == ["unbound_item:١١ أكتوبر ٢٠٢٦"]


def test_gemini_json_falls_back_to_36_when_primary_quota_exhausted(monkeypatch):
    import ingestion.gemini_visual as gv

    models_used = []

    class FakeInteractions:
        def create(self, **kwargs):
            models_used.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "transcription": "fallback-ok",
                        "items": [],
                        "uncertain_regions": [],
                        "complete": True,
                    }
                ),
                id="fb",
            )

    class FakeClient:
        def __init__(self, api_key):
            self.interactions = FakeInteractions()

    monkeypatch.setenv("GEMINI_API_KEY", "only-key")
    for index in range(2, 16):
        monkeypatch.delenv(f"GEMINI_API_KEY_{index}", raising=False)
    monkeypatch.setenv("BCT_GEMINI_RETRY_SLEEP_SECONDS", "0")
    monkeypatch.setenv("BCT_GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(gv, "_key_cursor", 0)
    monkeypatch.setattr(gv.time, "sleep", lambda _seconds: None)

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", fake_genai)
    monkeypatch.setitem(__import__("sys").modules, "google", SimpleNamespace(genai=fake_genai))
    import google

    monkeypatch.setattr(google, "genai", fake_genai, raising=False)

    text, response_id = gv.gemini_json_from_image(
        b"png",
        prompt="p",
        schema={"type": "object"},
        model="gemini-3.8-flash",
    )
    assert "fallback-ok" in text
    assert response_id == "fb"
    assert "gemini-3.8-flash" in models_used
    assert "gemini-3.6-flash" in models_used


def test_gemini_json_rotates_to_next_key_on_quota(monkeypatch):
    import ingestion.gemini_visual as gv

    calls = []
    created = []

    class FakeInteractions:
        def create(self, **_kwargs):
            key = calls[-1]
            if key == "key-a":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "transcription": "ok",
                        "items": [],
                        "uncertain_regions": [],
                        "complete": True,
                    }
                ),
                id="ok",
            )

    class FakeClient:
        def __init__(self, api_key):
            created.append(api_key)
            calls.append(api_key)
            self.interactions = FakeInteractions()

    monkeypatch.setenv("GEMINI_API_KEY", "key-a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-b")
    for index in range(3, 16):
        monkeypatch.delenv(f"GEMINI_API_KEY_{index}", raising=False)
    monkeypatch.setenv("BCT_GEMINI_RETRY_SLEEP_SECONDS", "0")
    monkeypatch.setattr(gv, "_key_cursor", 0)
    monkeypatch.setattr(gv.time, "sleep", lambda _seconds: None)

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", fake_genai)
    monkeypatch.setitem(__import__("sys").modules, "google", SimpleNamespace(genai=fake_genai))

    # Force import path used inside the function
    import google

    monkeypatch.setattr(google, "genai", fake_genai, raising=False)

    text, response_id = gv.gemini_json_from_image(
        b"png",
        prompt="p",
        schema={"type": "object"},
        model="gemini-3.8-flash",
    )
    assert "ok" in text
    assert response_id == "ok"
    assert created == ["key-a", "key-b"]
