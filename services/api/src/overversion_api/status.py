"""What the status pill reads.

Two modes. Lazily (the default), a status read refreshes only when the cached
answer is older than the interval, so an idle service sends no traffic at all
and can be slept by the host to save money; the cost is that the first read
after a period of quiet waits for one round of checks, bounded by the short
check timeout. With background polling on, a task refreshes on the interval
instead, which keeps reads instant but means the service never looks idle.

The default engine is Reproducible when it is up, otherwise OpenRouter.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from overversion_api.engines import EngineId, EngineRegistry, EngineStatus

DEFAULT_PREFERENCE = (EngineId.REPRODUCIBLE, EngineId.OPENROUTER, EngineId.STANDARD)


class StatusTracker:
    def __init__(
        self,
        registry: EngineRegistry,
        interval_s: float = 10.0,
        *,
        background: bool = False,
    ) -> None:
        self._registry = registry
        self._interval_s = interval_s
        self._background = background
        self._checked_at = 0.0
        self._lock = asyncio.Lock()
        self._statuses: dict[EngineId, EngineStatus] = {
            e.id: EngineStatus(engine=e.id, up=False, detail="not checked yet")
            for e in registry.all()
        }
        self._task: asyncio.Task[None] | None = None

    @property
    def statuses(self) -> dict[EngineId, EngineStatus]:
        return dict(self._statuses)

    @property
    def stale(self) -> bool:
        return (time.time() - self._checked_at) > self._interval_s

    async def current(self) -> dict[EngineId, EngineStatus]:
        """Statuses, refreshed first if the cached ones have aged out."""
        if self._background or not self.stale:
            return self.statuses
        async with self._lock:
            if not self.stale:  # another request refreshed while we waited
                return self.statuses
            return await self.refresh()

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
        self._checked_at = time.time()
        return self.statuses

    async def _loop(self) -> None:
        while True:
            with contextlib.suppress(Exception):
                await self.refresh()
            await asyncio.sleep(self._interval_s)

    async def start(self) -> None:
        """Start background polling if configured; lazily, do nothing at all.

        Deliberately no refresh here: checking at startup would make every cold
        boot pay for three network round trips before serving its first request.
        """
        if self._background:
            await self.refresh()
            self._task = asyncio.create_task(self._loop(), name="status-poller")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
