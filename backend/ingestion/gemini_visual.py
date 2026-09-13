from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


PROMPT_VERSION = "bct-faithful-page-transcription-v2"
DEFAULT_MODEL = "gemini-3.7-flash"


class SensitiveLiteral(BaseModel):
    literal: str
    kind: Literal["number", "date", "time", "percentage", "amount", "identifier", "other"]
    context: str
    uncertain: bool = False


class VisualPage(BaseModel):
    transcription: str = Field(description="Faithful verbatim transcription of all visible text in reading order.")
    items: list[SensitiveLiteral] = Field(default_factory=list)
    uncertain_regions: list[str] = Field(default_factory=list)
    complete: bool


_PROMPT = """You are transcribing one page of a public Tunisian regulatory PDF.
Return only data matching the supplied JSON schema.

Rules for `transcription`:
- Transcribe ALL visible textual content faithfully and in reading order.
- Preserve Arabic reading order, visible digit order, punctuation, headings and useful line breaks.
- Do not summarize, translate, normalize, reverse, silently correct, or infer missing text.
- Do not convert Arabic-Indic digits.
- If a character or region is genuinely unclear, preserve what is readable and describe the uncertainty in `uncertain_regions`.

Rules for `items`:
- Include every visible answer-bearing number, date, time, percentage, amount, document reference, account/code, or alphanumeric identifier.
- `literal` must be the exact visible characters and must occur verbatim in `transcription`.
- `context` must be a short verbatim phrase around the literal.
- Set `uncertain=true` when the literal or its association is not visually certain.

Set `complete=true` only when all visible text on the page was inspected and represented. Prefer explicit uncertainty over guessing."""


def _configuration(model: str) -> dict[str, object]:
    return {
        "provider": "Google Gemini API",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "thinking_level": "low",
        "schema": VisualPage.model_json_schema(),
    }


def _configuration_hash(model: str) -> str:
    encoded = json.dumps(_configuration(model), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gemini_json_from_image(
    image_png: bytes,
    *,
    prompt: str,
    schema: dict,
    model: str,
    client=None,
) -> tuple[str, object]:
    """Run one Gemini Interactions JSON call against a page image.

    Returns ``(output_text, response_id)``.
    """
    if client is None:
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Gemini visual ingestion requires google-genai>=2.20.0") from error
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when a page needs Gemini visual extraction")
        client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": base64.b64encode(image_png).decode("ascii"),
                "mime_type": "image/png",
            },
        ],
        generation_config={"thinking_level": "low"},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    )
    return interaction.output_text, getattr(interaction, "id", None)


class GeminiVisualTranscriber:
    """Small current-SDK adapter around Gemini's Interactions API.

    The model is configurable, but defaults to Gemini 3.7 Flash because that is
    the provider/model combination actually tested on the BCT Arabic failure set.
    """

    def __init__(self, cache_dir: str | Path, *, client=None, model: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("BCT_GEMINI_MODEL", DEFAULT_MODEL)
        # Keep provider setup lazy: a clean French PDF should still be ingestible
        # when Gemini is enabled globally but no visual fallback is actually needed.
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Gemini visual ingestion requires google-genai>=2.20.0") from error
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when a page needs Gemini visual extraction")
        self._client = genai.Client(api_key=api_key)
        return self._client

    def transcribe(
        self,
        *,
        image_png: bytes,
        source_pdf_sha256: str,
        page_number: int,
    ) -> VisualPage:
        image_sha = hashlib.sha256(image_png).hexdigest()
        binding = {
            "pdf": source_pdf_sha256.lower(),
            "page": int(page_number),
            "image": image_sha,
            "configuration": _configuration_hash(self.model),
        }
        key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("binding") != binding:
                raise ValueError("Gemini visual cache binding mismatch")
            return VisualPage.model_validate(cached["response"])

        output_text, response_id = gemini_json_from_image(
            image_png,
            prompt=_PROMPT,
            schema=VisualPage.model_json_schema(),
            model=self.model,
            client=self._get_client(),
        )
        parsed = VisualPage.model_validate_json(output_text)
        if not parsed.transcription.strip():
            raise ValueError("Gemini returned an empty page transcription")
        for item in parsed.items:
            if item.literal not in parsed.transcription or item.context not in parsed.transcription:
                raise ValueError("Gemini structured literal/context is not bound to its transcription")

        payload = {
            "binding": binding,
            "response": parsed.model_dump(),
            "response_id": response_id,
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, cache_path)
        return parsed
