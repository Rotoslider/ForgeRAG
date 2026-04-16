"""GPU resource coordinator.

Provides a single place to:
- Check VRAM availability (via torch.cuda.mem_get_info)
- Serialize GPU operations (single-flight semaphore)
- Track loaded models so we can unload idle ones when memory is tight
- Coexist with Flux/Forge on the same GPU by freeing ColPali/VLM when requested

Models register themselves via register_model() with a name and unload
callback. load_scope() is a context manager that acquires the GPU semaphore,
optionally waits for specific idle models to be unloaded first, and records
the activity so the auto-unloader knows when it's safe to reclaim VRAM.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)


class _Loadable(Protocol):
    """Any model that can be asked to unload (sync)."""

    def is_loaded(self) -> bool: ...
    def unload(self) -> None: ...


@dataclass
class ModelEntry:
    name: str
    handle: _Loadable
    last_used: float = field(default_factory=time.time)
    est_vram_bytes: int = 0


class GPUManager:
    """Process-wide coordinator for GPU-heavy operations.

    Usage:
        async with gpu.load_scope("colpali"):
            embeddings = colpali.process_images(paths)   # model loaded inside

    When the scope exits, the model stays loaded (fast re-use) but its
    last_used timestamp is refreshed. A background task unloads models that
    exceed gpu.idle_unload_seconds.
    """

    def __init__(self, *, idle_unload_seconds: int = 300):
        self.idle_unload_seconds = idle_unload_seconds
        self._semaphore = asyncio.Semaphore(1)
        self._models: dict[str, ModelEntry] = {}
        self._watcher_task: asyncio.Task | None = None

    # --------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._watcher_task is None or self._watcher_task.done():
            self._watcher_task = asyncio.create_task(self._idle_watcher())
            logger.info(
                "GPUManager started (idle_unload=%ds)", self.idle_unload_seconds
            )

    async def stop(self) -> None:
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # Unload everything to free VRAM on shutdown
        for entry in list(self._models.values()):
            try:
                if entry.handle.is_loaded():
                    entry.handle.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to unload %s on shutdown: %s", entry.name, exc)
        self._models.clear()

    # ----------------------------------------------------------- registration

    def register(
        self, name: str, handle: _Loadable, est_vram_bytes: int = 0
    ) -> None:
        """Register a model handle with the manager. Does not load the model."""
        self._models[name] = ModelEntry(
            name=name, handle=handle, est_vram_bytes=est_vram_bytes
        )
        logger.info("Registered model %s (est VRAM %.1f GB)", name, est_vram_bytes / 1e9)

    # --------------------------------------------------------- semaphore ops

    @asynccontextmanager
    async def load_scope(self, name: str) -> AsyncIterator[None]:
        """Acquire the GPU semaphore for model `name`. Refreshes last_used."""
        async with self._semaphore:
            entry = self._models.get(name)
            if entry is not None:
                entry.last_used = time.time()
            try:
                yield
            finally:
                if entry is not None:
                    entry.last_used = time.time()

    # -------------------------------------------------------------- watcher

    async def _idle_watcher(self) -> None:
        """Background task: unload models idle longer than threshold."""
        try:
            while True:
                await asyncio.sleep(30)
                now = time.time()
                for entry in list(self._models.values()):
                    try:
                        if (
                            entry.handle.is_loaded()
                            and (now - entry.last_used) > self.idle_unload_seconds
                        ):
                            logger.info(
                                "Auto-unloading idle model %s (idle %.0fs)",
                                entry.name,
                                now - entry.last_used,
                            )
                            entry.handle.unload()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Idle unload of %s failed: %s", entry.name, exc
                        )
        except asyncio.CancelledError:
            pass

    # --------------------------------------------------------------- status

    def gpu_info(self) -> dict[str, Any]:
        """Return VRAM state + loaded model list. Safe if torch is missing."""
        info: dict[str, Any] = {
            "available": False,
            "device_name": None,
            "vram_total_bytes": 0,
            "vram_free_bytes": 0,
            "vram_used_bytes": 0,
            "models": [],
        }
        try:
            import torch

            if not torch.cuda.is_available():
                return info
            info["available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            info["vram_total_bytes"] = int(total_b)
            info["vram_free_bytes"] = int(free_b)
            info["vram_used_bytes"] = int(total_b - free_b)
        except ImportError:
            return info
        except Exception as exc:  # noqa: BLE001
            logger.debug("gpu_info failed: %s", exc)
            return info

        for entry in self._models.values():
            info["models"].append({
                "name": entry.name,
                "loaded": entry.handle.is_loaded(),
                "last_used_s_ago": int(time.time() - entry.last_used),
                "est_vram_bytes": entry.est_vram_bytes,
            })
        return info

    def is_model_loaded(self, name: str) -> bool:
        entry = self._models.get(name)
        return bool(entry and entry.handle.is_loaded())

    async def unload_model(self, name: str) -> bool:
        """Explicitly unload a model by name. Returns True if unloaded."""
        entry = self._models.get(name)
        if entry is None or not entry.handle.is_loaded():
            return False
        async with self._semaphore:
            try:
                entry.handle.unload()
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to unload %s: %s", name, exc)
                return False
