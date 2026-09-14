"""Shared OpenAI-compatible chat client.

SGLang and OpenRouter both speak the OpenAI chat API, so the wire handling
lives here once. The subclasses differ only in what they send (seed, logprobs)
and in how they report health, which is exactly where the determinism
guarantees do or do not come from.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from reproducible_ai.hashing import hash_top_logprobs

from overversion_api.engines.base import (
    Chunk,
    Engine,
    EngineStatus,
    SamplingParams,
    StepEvidence,
)


class OpenAICompatEngine(Engine):
    """Chat streaming over the OpenAI-compatible protocol."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient,
        *,
        timeout_s: float = 300.0,
        top_logprobs_k: int = 0,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client
        self._timeout_s = timeout_s
        self._top_logprobs_k = top_logprobs_k
        self._extra_body = extra_body or {}
        self._extra_headers = extra_headers or {}

    # -- wire helpers -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self._extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _body(self, messages: list[dict[str, str]], params: SamplingParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            **self._extra_body,
        }
        if params.seed is not None and self.capabilities.accepts_seed:
            body["seed"] = params.seed
        if self._top_logprobs_k:
            body["logprobs"] = True
            body["top_logprobs"] = self._top_logprobs_k
        return body

    async def stream(
        self, messages: list[dict[str, str]], params: SamplingParams
    ) -> AsyncIterator[Chunk]:
        url = f"{self.base_url}/v1/chat/completions"
        position = 0
        async with self._client.stream(
            "POST",
            url,
            json=self._body(messages, params),
            headers=self._headers(),
            timeout=self._timeout_s,
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise EngineHTTPError(f"{self.id}: HTTP {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                steps = self._steps_from(choice, position)
                position += len(steps)
                if text or steps or choice.get("finish_reason"):
                    yield Chunk(text=text, steps=steps, finish_reason=choice.get("finish_reason"))

    def _steps_from(self, choice: dict[str, Any], start_position: int) -> list[StepEvidence]:
        """Turn one streamed choice into step evidence, when the engine sent logprobs."""
        if not self.capabilities.returns_evidence:
            return []
        content = ((choice.get("logprobs") or {}).get("content")) or []
        steps: list[StepEvidence] = []
        for i, item in enumerate(content):
            pairs = [
                (t["token_id"], float(t["logprob"]))
                for t in (item.get("top_logprobs") or [])
                if "token_id" in t and t.get("logprob") is not None
            ]
            steps.append(
                StepEvidence(
                    position=start_position + i,
                    token_id=item.get("token_id"),
                    evidence_hash=hash_top_logprobs(pairs, k=self._top_logprobs_k) if pairs else None,
                    evidence_kind="top_logprobs_scaled" if pairs else None,
                )
            )
        return steps

    async def _get_json(self, path: str) -> dict[str, Any]:
        resp = await self._client.get(
            f"{self.base_url}{path}", headers=self._headers(), timeout=self._timeout_s
        )
        resp.raise_for_status()
        return resp.json()

    def _status(self, *, up: bool, detail: str = "", info: dict[str, Any] | None = None) -> EngineStatus:
        return EngineStatus(
            engine=self.id,
            up=up,
            detail=detail,
            model=self.model,
            server_info=info or {},
            checked_at=time.time(),
        )


class EngineHTTPError(RuntimeError):
    pass
