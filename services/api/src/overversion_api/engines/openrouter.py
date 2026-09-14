"""OpenRouter: always available, promises nothing.

It is a first-class option in the selector rather than only a fallback, so the
app stays usable with the GPU off. It accepts a seed parameter and routes to
whichever provider happens to serve the model, so neither the arithmetic nor
the sampler is under our control: capabilities say no determinism and no
evidence, and turns produced here are labelled accordingly.
"""

from __future__ import annotations

import httpx

from overversion_api.engines.base import EngineCapabilities, EngineId, EngineStatus
from overversion_api.engines.openai_compat import OpenAICompatEngine


class OpenRouterEngine(OpenAICompatEngine):
    id = EngineId.OPENROUTER
    capabilities = EngineCapabilities(
        deterministic=False, accepts_seed=False, returns_evidence=False, supports_min_p=True
    )

    async def check(self) -> EngineStatus:
        if not self.api_key:
            return self._status(up=False, detail="no OPENROUTER_API_KEY configured")
        try:
            await self._get_json("/key")
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (401, 403):
                return self._status(up=False, detail=f"key rejected (HTTP {code})")
            return self._status(up=False, detail=f"HTTP {code}")
        except (httpx.HTTPError, ValueError) as e:
            return self._status(up=False, detail=f"{type(e).__name__}: {e}"[:200])
        return self._status(up=True)
