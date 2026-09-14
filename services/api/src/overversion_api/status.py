"""Background poller behind the status pill.

Checks all three engines on an interval so that a page load never waits on a
network round trip to a box that may be switched off, and so the UI can
preselect an engine that is actually up. The default engine is Reproducible
when it is up, otherwise OpenRouter (DEMO.md).
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from overversion_api.engines import EngineId, EngineRegistry, EngineStatus

DEFAULT_PREFERENCE = (EngineId.REPRODUCIBLE, EngineId.OPENROUTER, EngineId.STANDARD)


class StatusTracker:
    def __init__(self, registry: EngineRegistry, interval_s: float = 10.0) -> None:
        self._registry = registry
        self._interval_s = interval_s
        self._statuses: dict[EngineId, EngineStatus] = {
            e.id: EngineStatus(engine=e.id, up=False, detail="not checked yet")
            for e in registry.all()
        }
        self._task: asyncio.Task[None] | None = None

    @property
    def statuses(self) -> dict[EngineId, EngineStatus]:
        return dict(self._statuses)

    def default_engine(self) -> EngineId | None:
        for engine_id in DEFAULT_PREFERENCE:
            status = self._statuses.get(engine_id)
            if status and status.up:
                return engine_id
        return None

    async def refresh(self) -> dict[EngineId, EngineStatus]:
        results = await asyncio.gather(
            *(e.check() for e in self._registry.all()), return_exceptions=True
        )
        for engine, result in zip(self._registry.all(), results, strict=True):
            if isinstance(result, BaseException):
                self._statuses[engine.id] = EngineStatus(
                    engine=engine.id,
                    up=False,
                    detail=f"{type(result).__name__}: {result}"[:200],
                    checked_at=time.time(),
                )
            else:
                self._statuses[engine.id] = result
        return self.statuses

    async def _loop(self) -> None:
        while True:
            with contextlib.suppress(Exception):
                await self.refresh()
            await asyncio.sleep(self._interval_s)

    async def start(self) -> None:
        await self.refresh()
        self._task = asyncio.create_task(self._loop(), name="status-poller")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
