"""FastAPI app. Routes for status today; conversations, traces and tokens next."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from overversion_api import db
from overversion_api.config import Settings, get_settings
from overversion_api.engines import EngineRegistry
from overversion_api.routes import router
from overversion_api.status import StatusTracker


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(follow_redirects=True)
        registry = EngineRegistry(settings, client)
        tracker = StatusTracker(registry, settings.status_interval_s)
        app.state.settings = settings
        app.state.client = client
        app.state.registry = registry
        app.state.status = tracker
        db.init_engine(settings.database_url)
        await db.create_all()
        await tracker.start()
        try:
            yield
        finally:
            await tracker.stop()
            await client.aclose()
            await db.dispose()

    app = FastAPI(title="overversion api", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.app_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness of this service only; engine health is /v1/status."""
        return {"status": "ok"}

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        tracker: StatusTracker = app.state.status
        registry: EngineRegistry = app.state.registry
        engines = []
        for engine in registry.all():
            st = tracker.statuses[engine.id]
            caps = engine.capabilities
            engines.append(
                {
                    "id": str(engine.id),
                    "up": st.up,
                    "detail": st.detail,
                    "model": st.model,
                    "envelope_id": st.envelope_id,
                    "checked_at": st.checked_at,
                    "capabilities": {
                        "deterministic": caps.deterministic,
                        "accepts_seed": caps.accepts_seed,
                        "returns_evidence": caps.returns_evidence,
                        "supports_min_p": caps.supports_min_p,
                    },
                }
            )
        default = tracker.default_engine()
        return {"engines": engines, "default_engine": str(default) if default else None}

    return app


app = create_app()
