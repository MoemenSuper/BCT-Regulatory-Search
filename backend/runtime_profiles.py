"""Coherent runtime profiles for the interactive BCT application."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable


class RuntimeProfile(str, Enum):
    LOCAL = "local"
    LOCAL_HYBRID = "local_hybrid"
    CLOUD = "cloud"


@dataclass(frozen=True)
class ProfileSpec:
    value: RuntimeProfile
    label: str
    retrieval: str
    answer: str
    qualification: str
    description: str


PROFILE_SPECS = {
    RuntimeProfile.LOCAL: ProfileSpec(
        value=RuntimeProfile.LOCAL,
        label="All local (experimental)",
        retrieval="local_e5_bge",
        answer="ollama",
        qualification="experimental_rejected",
        description=(
            "Search: local multilingual-e5-small (no cloud API). "
            "Answer: local Ollama (default qwen3.5:9b-q4_K_M, configurable). "
            "PDF ingestion: EasyOCR (Arabic) + PaddleOCR-VL (charts/tables/hard pages)."
        ),
    ),
    RuntimeProfile.LOCAL_HYBRID: ProfileSpec(
        value=RuntimeProfile.LOCAL_HYBRID,
        label="Local hybrid",
        retrieval="local_e5_bge",
        answer="groq",
        qualification="development",
        description=(
            "Search: local multilingual-e5-small (no cloud API). "
            "Answer: Groq API (model via BCT_GROQ_MODEL). "
            "PDF ingestion: EasyOCR (Arabic) + PaddleOCR-VL (charts/tables/hard pages)."
        ),
    ),
    RuntimeProfile.CLOUD: ProfileSpec(
        value=RuntimeProfile.CLOUD,
        label="Cloud (Voyage + Groq)",
        retrieval="cloud_embed_rerank",
        answer="groq",
        qualification="development_not_legally_qualified",
        description=(
            "Search: Voyage Context-4 embeddings + Voyage rerank. "
            "Answer: Groq. PDF ingestion: Gemini VLM for hard photo/scan/chart pages."
        ),
    ),
}


@dataclass(frozen=True)
class ProfileRuntime:
    spec: ProfileSpec
    retrieval_backend: Any
    answer_provider: str


def parse_profile(value: RuntimeProfile | str) -> RuntimeProfile:
    if isinstance(value, RuntimeProfile):
        return value
    try:
        return RuntimeProfile(value)
    except ValueError as error:
        choices = ", ".join(profile.value for profile in RuntimeProfile)
        raise ValueError(
            f"Unknown runtime profile {value!r}; choose one of: {choices}"
        ) from error


def profile_options() -> list[dict[str, str]]:
    return [
        {
            "value": spec.value.value,
            "label": spec.label,
            "retrieval": spec.retrieval,
            "answer": spec.answer,
            "qualification": spec.qualification,
            "description": spec.description,
        }
        for spec in PROFILE_SPECS.values()
    ]


class RuntimeProfileManager:
    """Resolve profiles while loading the cloud retrieval backend only on demand."""

    def __init__(
        self,
        local_retrieval_backend: Any,
        cloud_retrieval_factory: Callable[[], Any],
        *,
        local_retrieval_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._local_retrieval_backend = local_retrieval_backend
        self._local_retrieval_factory = local_retrieval_factory
        self._cloud_retrieval_factory = cloud_retrieval_factory
        self._cloud_retrieval_backend: Any | None = None
        self._lock = Lock()

    def get(self, value: RuntimeProfile | str) -> ProfileRuntime:
        profile = parse_profile(value)
        spec = PROFILE_SPECS[profile]
        if profile is RuntimeProfile.CLOUD:
            retrieval_backend = self._cloud_backend()
        else:
            retrieval_backend = self._local_backend()
        return ProfileRuntime(
            spec=spec,
            retrieval_backend=retrieval_backend,
            answer_provider=spec.answer,
        )

    # Both backends are built on first use. The check-lock-check pattern keeps the usual
    # case (already built) lock-free, and the lock stops two requests building it twice.

    def _cloud_backend(self) -> Any:
        if self._cloud_retrieval_backend is None:
            with self._lock:
                if self._cloud_retrieval_backend is None:
                    self._cloud_retrieval_backend = self._cloud_retrieval_factory()
        return self._cloud_retrieval_backend

    def _local_backend(self) -> Any:
        if self._local_retrieval_backend is None:
            if self._local_retrieval_factory is None:
                raise RuntimeError("Local retrieval is not configured")
            with self._lock:
                if self._local_retrieval_backend is None:
                    self._local_retrieval_backend = self._local_retrieval_factory()
        return self._local_retrieval_backend

    def reset(self) -> None:
        """Point retrieval at the newly activated corpus after ingest or removal.

        A local backend that is already serving is rebuilt here and swapped in, so
        the admin action pays the index reload instead of the next chat. Model
        weights are process-cached and are not reloaded.
        """
        rebuilt = None
        if self._local_retrieval_factory is not None and self._local_retrieval_backend is not None:
            rebuilt = self._local_retrieval_factory()
        with self._lock:
            self._cloud_retrieval_backend = None
            # Tests may inject a fixed backend and can simply avoid calling reset().
            if self._local_retrieval_factory is not None:
                self._local_retrieval_backend = rebuilt
