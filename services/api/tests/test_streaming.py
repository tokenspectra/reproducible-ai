import json

import httpx
import pytest

from overversion_api.engines import EngineId, SamplingParams
from overversion_api.engines.openai_compat import EngineHTTPError
from tests.conftest import make_registry


def sse(*events: dict) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def delta(text: str, logprobs=None, finish=None) -> dict:
    choice = {"delta": {"content": text}, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice]}


def lp(token_id: int, top: list[tuple[int, float]]) -> dict:
    return {
        "token_id": token_id,
        "top_logprobs": [{"token_id": t, "logprob": v} for t, v in top],
    }


async def collect(engine, params=None):
    params = params or SamplingParams(seed=7)
    chunks = []
    async for c in engine.stream([{"role": "user", "content": "hi"}], params):
        chunks.append(c)
    return chunks


async def test_reproducible_sends_seed_and_logprobs_and_hashes(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            content=sse(
                delta("he", [lp(11, [(11, -0.1), (12, -2.0)])]),
                delta("llo", [lp(12, [(12, -0.2), (13, -3.0)])], finish="stop"),
            ),
        )

    registry, client = make_registry(settings, handler)
    async with client:
        chunks = await collect(registry[EngineId.REPRODUCIBLE])
    assert captured["body"]["seed"] == 7
    assert captured["body"]["logprobs"] is True
    assert captured["body"]["top_logprobs"] == 4
    assert captured["auth"] == "Bearer k"
    assert "".join(c.text for c in chunks) == "hello"
    steps = [s for c in chunks for s in c.steps]
    assert [s.position for s in steps] == [0, 1]
    assert all(s.evidence_hash and s.evidence_kind == "top_logprobs_scaled" for s in steps)
    assert steps[0].evidence_hash != steps[1].evidence_hash
    assert chunks[-1].finish_reason == "stop"


async def test_baseline_and_openrouter_send_no_seed_and_no_evidence(settings):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse(delta("x", [lp(1, [(1, -0.5)])], finish="stop")))

    registry, client = make_registry(settings, handler)
    async with client:
        for engine_id in (EngineId.STANDARD, EngineId.OPENROUTER):
            chunks = await collect(registry[engine_id])
            assert [s for c in chunks for s in c.steps] == []
    assert all("seed" not in b and "logprobs" not in b for b in bodies)


async def test_openrouter_sends_attribution_headers(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=sse(delta("x", finish="stop")))

    registry, client = make_registry(settings, handler)
    async with client:
        await collect(registry[EngineId.OPENROUTER])
    assert seen["x-title"] == "overversion"
    assert seen["http-referer"] == settings.app_origin


async def test_http_error_surfaces_body(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model loading"})

    registry, client = make_registry(settings, handler)
    async with client:
        with pytest.raises(EngineHTTPError, match="model loading"):
            await collect(registry[EngineId.REPRODUCIBLE])


async def test_malformed_sse_lines_are_skipped(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        body = b"data: not json\n\n: comment\n\n" + sse(delta("ok", finish="stop"))
        return httpx.Response(200, content=body)

    registry, client = make_registry(settings, handler)
    async with client:
        chunks = await collect(registry[EngineId.REPRODUCIBLE])
    assert "".join(c.text for c in chunks) == "ok"


async def test_evidence_hash_is_order_independent(settings):
    hashes = []

    def make_handler(order):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse(delta("a", [lp(5, order)], finish="stop")))

        return handler

    for order in ([(5, -0.1), (6, -1.0)], [(6, -1.0), (5, -0.1)]):
        registry, client = make_registry(settings, make_handler(order))
        async with client:
            chunks = await collect(registry[EngineId.REPRODUCIBLE])
        hashes.append(next(s.evidence_hash for c in chunks for s in c.steps))
    assert hashes[0] == hashes[1]
