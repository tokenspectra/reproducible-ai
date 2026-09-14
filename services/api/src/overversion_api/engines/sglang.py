"""The two SGLang engines on the inference box.

Both talk to the same host through Caddy: the deterministic instance at the
root and the baseline instance under /baseline. They are separate engines
rather than a flag because what they promise differs, and because the UI has
to grey them out together when the box is off.

Deterministic-mode facts that shape this client (see
docs/research/sglang-deterministic-audit.md):
  - a seed must always be sent; SGLang defaults it to 42 and does not vary it
    for n > 1, so relying on the default would silently share a seed across
    requests;
  - min_p together with a seed is rejected by SGLang, so the capability says so;
  - returned logprobs are temperature-scaled, so the evidence kind is named
    accordingly and the verifier compares like with like.
"""

from __future__ import annotations

from typing import Any

import httpx

from overversion_api.engines.base import EngineCapabilities, EngineId, EngineStatus
from overversion_api.engines.openai_compat import OpenAICompatEngine


class SGLangEngine(OpenAICompatEngine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def check(self) -> EngineStatus:
        try:
            info = await self._get_json("/server_info")
        except (httpx.HTTPError, ValueError) as e:
            return self._status(up=False, detail=f"{type(e).__name__}: {e}"[:200])
        st = self._status(up=True, info=info)
        st.envelope_id = envelope_id_from(info)
        return st


class ReproducibleEngine(SGLangEngine):
    id = EngineId.REPRODUCIBLE
    capabilities = EngineCapabilities(
        deterministic=True, accepts_seed=True, returns_evidence=True, supports_min_p=False
    )

    async def check(self) -> EngineStatus:
        st = await super().check()
        if st.up and st.server_info.get("enable_deterministic_inference") is False:
            st.up = False
            st.detail = "server is up but deterministic inference is off"
        return st


class BaselineEngine(SGLangEngine):
    id = EngineId.STANDARD
    capabilities = EngineCapabilities(
        deterministic=False, accepts_seed=False, returns_evidence=False, supports_min_p=True
    )


def envelope_id_from(info: dict[str, Any]) -> str:
    """Short, stable id of the serving envelope, from what /server_info exposes.

    SGLang records no weight hash, git sha, kernel or driver version, so this
    is a weak envelope: it pins the fields the server does report. The strong
    envelope is the JSON captured on the box by cloud/nebius (GPU, driver,
    CUDA, image digest, weights manifest hash); the API stores that alongside
    and prefers it when present.
    """
    from reproducible_ai.hashing import blake3_128, canonical_json

    fields = {
        "version": info.get("version"),
        "model_path": info.get("model_path"),
        "served_model_name": info.get("served_model_name"),
        "weight_version": info.get("weight_version"),
        "attention_backend": info.get("attention_backend"),
        "deterministic": info.get("enable_deterministic_inference"),
        "tp_size": info.get("tp_size"),
        "dtype": info.get("dtype"),
        "quantization": info.get("quantization"),
    }
    return "env_" + blake3_128(canonical_json(fields))[:16]
