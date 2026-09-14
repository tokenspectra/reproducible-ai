import httpx
import pytest

from overversion_api.engines import EngineId

SERVER_INFO_DET = {
    "version": "0.5.19",
    "model_path": "Qwen/Qwen3-8B",
    "served_model_name": "qwen3-8b",
    "enable_deterministic_inference": True,
    "attention_backend": "fa3",
    "tp_size": 1,
}


def handler_all_up(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/server_info":
        return httpx.Response(200, json=SERVER_INFO_DET)
    if request.url.path == "/baseline/server_info":
        return httpx.Response(200, json={**SERVER_INFO_DET, "enable_deterministic_inference": False})
    if request.url.path.endswith("/key"):
        return httpx.Response(200, json={"data": {"label": "test"}})
    return httpx.Response(404)


def handler_box_down(request: httpx.Request) -> httpx.Response:
    if "infer.test" in request.url.host:
        raise httpx.ConnectError("connection refused", request=request)
    if request.url.path.endswith("/key"):
        return httpx.Response(200, json={"data": {}})
    return httpx.Response(404)


async def test_all_up_prefers_reproducible(tracker_factory):
    tracker, _, client = tracker_factory(handler_all_up)
    async with client:
        statuses = await tracker.refresh()
    assert all(s.up for s in statuses.values())
    assert tracker.default_engine() == EngineId.REPRODUCIBLE
    assert statuses[EngineId.REPRODUCIBLE].envelope_id.startswith("env_")


async def test_box_down_falls_back_to_openrouter(tracker_factory):
    tracker, _, client = tracker_factory(handler_box_down)
    async with client:
        statuses = await tracker.refresh()
    assert not statuses[EngineId.REPRODUCIBLE].up
    assert not statuses[EngineId.STANDARD].up
    assert statuses[EngineId.OPENROUTER].up
    assert tracker.default_engine() == EngineId.OPENROUTER
    assert "ConnectError" in statuses[EngineId.REPRODUCIBLE].detail


async def test_determinism_off_counts_as_down(tracker_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/server_info":
            return httpx.Response(
                200, json={**SERVER_INFO_DET, "enable_deterministic_inference": False}
            )
        return handler_all_up(request)

    tracker, _, client = tracker_factory(handler)
    async with client:
        statuses = await tracker.refresh()
    assert not statuses[EngineId.REPRODUCIBLE].up
    assert "deterministic inference is off" in statuses[EngineId.REPRODUCIBLE].detail
    assert tracker.default_engine() == EngineId.OPENROUTER


async def test_envelope_id_changes_with_server_fields(tracker_factory):
    from overversion_api.engines.sglang import envelope_id_from

    a = envelope_id_from(SERVER_INFO_DET)
    b = envelope_id_from({**SERVER_INFO_DET, "version": "0.5.20"})
    assert a != b and a == envelope_id_from(dict(SERVER_INFO_DET))


async def test_no_openrouter_key_is_down(settings, tracker_factory):
    settings.openrouter_api_key = ""
    tracker, _, client = tracker_factory(handler_box_down)
    async with client:
        statuses = await tracker.refresh()
    assert not statuses[EngineId.OPENROUTER].up
    assert "no OPENROUTER_API_KEY" in statuses[EngineId.OPENROUTER].detail
    assert tracker.default_engine() is None


@pytest.mark.parametrize("code", [401, 500])
async def test_openrouter_http_errors(tracker_factory, code):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/key"):
            return httpx.Response(code, json={"error": "nope"})
        return handler_all_up(request)

    tracker, _, client = tracker_factory(handler)
    async with client:
        statuses = await tracker.refresh()
    assert not statuses[EngineId.OPENROUTER].up
