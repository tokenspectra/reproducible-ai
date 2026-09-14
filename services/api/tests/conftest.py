import httpx
import pytest

from overversion_api.config import Settings
from overversion_api.engines import EngineRegistry
from overversion_api.status import StatusTracker


@pytest.fixture
def settings() -> Settings:
    return Settings(
        infer_base_url="https://infer.test",
        infer_api_key="k",
        openrouter_api_key="or-k",
        status_interval_s=3600.0,
        top_logprobs_k=4,
    )


def make_registry(settings: Settings, handler) -> tuple[EngineRegistry, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EngineRegistry(settings, client), client


@pytest.fixture
def tracker_factory(settings):
    def _make(handler):
        registry, client = make_registry(settings, handler)
        return StatusTracker(registry, settings.status_interval_s), registry, client

    return _make
