"""Engine registry: builds the three engines from settings and holds one HTTP client."""

from __future__ import annotations

import httpx

from overversion_api.config import Settings
from overversion_api.engines.base import (
    Chunk,
    Engine,
    EngineCapabilities,
    EngineId,
    EngineStatus,
    SamplingParams,
    StepEvidence,
)
from overversion_api.engines.openrouter import OpenRouterEngine
from overversion_api.engines.sglang import BaselineEngine, ReproducibleEngine

__all__ = [
    "Chunk",
    "Engine",
    "EngineCapabilities",
    "EngineId",
    "EngineRegistry",
    "EngineStatus",
    "SamplingParams",
    "StepEvidence",
]


class EngineRegistry:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        base = settings.infer_base_url.rstrip("/")
        self.engines: dict[EngineId, Engine] = {
            EngineId.REPRODUCIBLE: ReproducibleEngine(
                base,
                settings.infer_api_key,
                settings.infer_model,
                client,
                timeout_s=settings.infer_timeout_s,
                top_logprobs_k=settings.top_logprobs_k,
            ),
            EngineId.STANDARD: BaselineEngine(
                base + settings.infer_baseline_path,
                settings.infer_api_key,
                settings.infer_model,
                client,
                timeout_s=settings.infer_timeout_s,
            ),
            EngineId.OPENROUTER: OpenRouterEngine(
                settings.openrouter_base_url,
                settings.openrouter_api_key,
                settings.openrouter_model,
                client,
                timeout_s=settings.infer_timeout_s,
                extra_headers={
                    "HTTP-Referer": settings.app_origin,
                    "X-Title": "overversion",
                },
            ),
        }

    def __getitem__(self, engine_id: EngineId) -> Engine:
        return self.engines[engine_id]

    def all(self) -> list[Engine]:
        return list(self.engines.values())
