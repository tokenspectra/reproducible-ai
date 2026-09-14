"""The engine interface the app routes through.

Three engines back the mode selector (DEMO.md): the deterministic SGLang
instance, the SGLang baseline instance and OpenRouter. They differ in what
they can promise, so the interface makes that explicit rather than leaving
callers to infer it from the engine's name.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EngineId(StrEnum):
    REPRODUCIBLE = "reproducible"
    STANDARD = "standard"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can promise. The UI renders this; nothing infers it from the id."""

    deterministic: bool
    accepts_seed: bool
    returns_evidence: bool
    supports_min_p: bool


@dataclass
class EngineStatus:
    engine: EngineId
    up: bool
    detail: str = ""
    model: str | None = None
    envelope_id: str | None = None
    server_info: dict[str, Any] = field(default_factory=dict)
    checked_at: float = 0.0


@dataclass
class StepEvidence:
    """One generated token and, where the engine provides it, its evidence hash."""

    position: int
    token_id: int | None
    evidence_hash: str | None = None
    evidence_kind: str | None = None


@dataclass
class Chunk:
    """One streamed piece of an assistant turn."""

    text: str = ""
    steps: list[StepEvidence] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 512
    seed: int | None = None

    def rejected_reason(self, caps: EngineCapabilities) -> str | None:
        if self.seed is not None and not caps.accepts_seed:
            return "this engine ignores seeds"
        return None


class Engine(abc.ABC):
    id: EngineId
    capabilities: EngineCapabilities

    @abc.abstractmethod
    async def check(self) -> EngineStatus: ...

    @abc.abstractmethod
    def stream(
        self, messages: list[dict[str, str]], params: SamplingParams
    ) -> AsyncIterator[Chunk]: ...
