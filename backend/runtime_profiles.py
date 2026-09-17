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
            "PDF ingestion may still use Gemini VLM (GEMINI_API_KEY) on hard photo/scan pages."
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
            "PDF ingestion: Gemini VLM (GEMINI_API_KEY) for hard-to-read photo/scan pages."
        ),
    ),
    RuntimeProfile.CLOUD: ProfileSpec(
        value=RuntimeProfile.CLOUD,
        label="Cloud (Voyage + Groq)",
        retrieval="voyage_context_4_rerank_2_5",
        answer="groq",
        qualification="development_not_legally_qualified",
        description=(
            "Search: Voyage Context-4 + rerank (VOYAGE_API_KEY). "
            "Answer: Groq API (GROQ_API_KEY / BCT_GROQ_MODEL). "
            "PDF ingestion: Gemini VLM (GEMINI_API_KEY) for hard-to-read photo/scan pages."
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
            retrieval_backend = self._lazy(
                "_cloud_retrieval_backend", self._cloud_retrieval_factory
            )
        else:
            retrieval_backend = self._local_backend()
        return ProfileRuntime(
            spec=spec,
            retrieval_backend=retrieval_backend,
            answer_provider=spec.answer,
        )

    def _lazy(self, attr: str, factory: Callable[[], Any]) -> Any:
        value = getattr(self, attr)
        if value is None:
            with self._lock:
                value = getattr(self, attr)
                if value is None:
                    value = factory()
                    setattr(self, attr, value)
        return value

    def _local_backend(self) -> Any:
        if self._local_retrieval_backend is not None:
            return self._local_retrieval_backend
        if self._local_retrieval_factory is None:
            raise RuntimeError("Local retrieval is not configured")
        return self._lazy("_local_retrieval_backend", self._local_retrieval_factory)

    def reset(self) -> None:
        """Drop lazy retrieval backends after an ingestion asset promotion."""
        with self._lock:
            self._cloud_retrieval_backend = None
            # The production manager constructs local retrieval lazily. Tests may
            # inject a fixed backend and can simply avoid calling reset().
            if self._local_retrieval_factory is not None:
                self._local_retrieval_backend = None
