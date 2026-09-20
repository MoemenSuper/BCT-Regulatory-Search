from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


PROMPT_VERSION = "bct-faithful-page-transcription-v2"
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.6-flash"
# Tried in order on quota/timeout/transient failure. Extraction quality, not model bake-off.
DEFAULT_MODEL_CHAIN = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
)


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


def _gemini_keys() -> list[str]:
    keys = []
    for name in ("GEMINI_API_KEY", *(f"GEMINI_API_KEY_{n}" for n in range(2, 16))):
        value = (os.environ.get(name) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _quota_exhausted(error: BaseException) -> bool:
    text = str(error)
    return any(token in text for token in ("429", "RESOURCE_EXHAUSTED", "quota", "rate-limit", "too_many_requests"))


def _key_unusable(error: BaseException) -> bool:
    """Dead/revoked project keys should rotate just like quota, not abort the PDF."""
    text = str(error).casefold()
    return any(
        token in text
        for token in (
            "permission_denied",
            "permission denied",
            "403",
            "api key not valid",
            "invalid api key",
            "api_key_invalid",
            "consumer_invalid",
            "billing",
        )
    )


def _transient_network(error: BaseException) -> bool:
    text = str(error).casefold()
    return any(
        token in text
        for token in (
            "apiconnectionerror",
            "connection",
            "timed out",
            "timeout",
            "name or service not known",
            "no address associated with hostname",
            "getaddrinfo",
            "temporary failure in name resolution",
            "errno -5",
            "errno 11001",
            "errno 8",
        )
    )


def _model_chain(primary: str) -> list[str]:
    """Primary first, then further models when keys/quota/timeouts are exhausted."""
    configured = (os.environ.get("BCT_GEMINI_MODEL_CHAIN") or "").strip()
    if configured:
        models = [part.strip() for part in configured.split(",") if part.strip()]
    else:
        fallback = (os.environ.get("BCT_GEMINI_FALLBACK_MODEL") or FALLBACK_MODEL).strip()
        models = [primary]
        for candidate in (fallback, *DEFAULT_MODEL_CHAIN):
            if candidate and candidate.casefold() not in {item.casefold() for item in models}:
                models.append(candidate)
    if primary and primary.casefold() not in {item.casefold() for item in models}:
        models.insert(0, primary)
    elif primary and models and models[0].casefold() != primary.casefold():
        models = [primary, *[item for item in models if item.casefold() != primary.casefold()]]
    return models


# Spread load across GEMINI_API_KEY[_N] instead of always burning slot 1 first.
_key_cursor = 0


def gemini_json_from_image(
    image_png: bytes,
    *,
    prompt: str,
    schema: dict,
    model: str,
    client=None,
    on_rotate=None,
) -> tuple[str, object]:
    """Run one Gemini Interactions JSON call against a page image.

    Returns ``(output_text, response_id)``. Rotates through GEMINI_API_KEY[_N]
    when a key hits quota; after keys are exhausted on the primary model, retries
    the same keys on further models (3.7 / 3.6 / 3.5 by default) so bulk repair can keep
    moving across separate rate pools.
    """
    global _key_cursor
    injected = client is not None
    if injected:
        keys: list[str | None] = [None]
        start = 0
        genai = None
        models = [model]
    else:
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Gemini visual ingestion requires google-genai>=2.20.0") from error
        keys = _gemini_keys()
        if not keys:
            raise RuntimeError("GEMINI_API_KEY is required when a page needs Gemini visual extraction")
        start = _key_cursor % len(keys)
        _key_cursor = start + 1
        models = _model_chain(model)

    clients: dict[int, object] = {}
    if injected:
        clients[0] = client

    last_error: BaseException | None = None
    retry_sleep = float(os.environ.get("BCT_GEMINI_RETRY_SLEEP_SECONDS", "8"))
    timeout = float(os.environ.get("BCT_GEMINI_TIMEOUT_SECONDS", "120"))

    def _try_model(active_model: str) -> tuple[str, object] | None:
        nonlocal last_error
        sweeps = 1 if injected else 2
        for sweep in range(sweeps):
            for offset in range(len(keys)):
                index = (start + offset) % len(keys)
                try:
                    if index not in clients:
                        clients[index] = genai.Client(api_key=keys[index])
                    active = clients[index]

                    def _create(current_model=active_model, current_client=active):
                        return current_client.interactions.create(
                            model=current_model,
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

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        interaction = pool.submit(_create).result(timeout=timeout)
                    if not injected and on_rotate is not None:
                        on_rotate(active, index)
                    return interaction.output_text, getattr(interaction, "id", None)
                except concurrent.futures.TimeoutError as error:
                    last_error = TimeoutError(
                        f"Gemini visual extraction timed out after {int(timeout)}s"
                        f" (model={active_model})"
                    )
                    if injected:
                        raise last_error from error
                    return None  # try next model
                except Exception as error:
                    last_error = error
                    if injected:
                        raise
                    if _quota_exhausted(error) or _key_unusable(error) or _transient_network(error):
                        if offset + 1 < len(keys):
                            continue
                        if sweep + 1 < sweeps:
                            time.sleep(retry_sleep)
                            break
                        return None  # keys exhausted for this model
                    transient = getattr(error, "code", None) in {500, 503} or any(
                        token in str(error) for token in ("500", "503", "high demand")
                    )
                    if transient and sweep + 1 < sweeps:
                        time.sleep(retry_sleep)
                        break
                    raise
            else:
                continue
        return None

    for model_index, active_model in enumerate(models):
        if model_index > 0:
            print(
                f"gemini falling back from {models[model_index - 1]} to {active_model}",
                flush=True,
            )
            clients.clear()
        result = _try_model(active_model)
        if result is not None:
            if model_index > 0:
                print(f"gemini using fallback model {active_model}", flush=True)
            return result

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini visual extraction exhausted all API keys")


class GeminiVisualTranscriber:
    """Small current-SDK adapter around Gemini's Interactions API.

    Defaults to Gemini 3.8 Flash; on quota/timeout walks 3.7 → 3.6 → 3.5 Flash so
    bulk ingest can keep moving across separate rate pools.
    """

    def __init__(self, cache_dir: str | Path, *, client=None, model: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("BCT_GEMINI_MODEL", DEFAULT_MODEL)
        # Keep provider setup lazy: a clean French PDF should still be ingestible
        # when Gemini is enabled globally but no visual fallback is actually needed.
        self._injected_client = client is not None
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Gemini visual ingestion requires google-genai>=2.20.0") from error
        keys = _gemini_keys()
        if not keys:
            raise RuntimeError("GEMINI_API_KEY is required when a page needs Gemini visual extraction")
        self._client = genai.Client(api_key=keys[0])
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

        def _rotate(client, index):
            self._client = client
            print(f"gemini key rotated to slot {index + 1}", flush=True)

        output_text, response_id = gemini_json_from_image(
            image_png,
            prompt=_PROMPT,
            schema=VisualPage.model_json_schema(),
            model=self.model,
            client=self._client if self._injected_client else None,
            on_rotate=_rotate,
        )
        parsed = VisualPage.model_validate_json(output_text)
        if not parsed.transcription.strip():
            raise ValueError("Gemini returned an empty page transcription")
        # The transcription is the evidence; items are an index into it. An item whose
        # literal/context is not found verbatim (Arabic marks, spacing) is kept as
        # uncertain rather than failing the whole page.
        for item in parsed.items:
            if item.literal not in parsed.transcription or item.context not in parsed.transcription:
                item.uncertain = True
                parsed.uncertain_regions.append(f"unbound_item:{item.literal}")

        payload = {
            "binding": binding,
            "response": parsed.model_dump(),
            "response_id": response_id,
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, cache_path)
        return parsed
