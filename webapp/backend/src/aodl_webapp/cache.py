"""A small thread-safe LRU keyed by strings (plan ids and derived keys).

The routes run in FastAPI's threadpool, so the cache takes a lock around every access; the
values are immutable ``aodl`` objects (``MotionPlan``, ``SimResult``) that are never mutated
after insertion, so handing the same instance to several threads is safe.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    """Least-recently-used mapping ``str -> T`` holding at most ``maxsize`` entries."""

    def __init__(self, maxsize: int) -> None:
        if int(maxsize) < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize!r}")
        self.maxsize = int(maxsize)
        self._data: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        """The cached value, refreshed as most recently used, or ``None``."""
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: str, value: T) -> T:
        """Store ``value`` under ``key`` (evicting the oldest entry when full) and return it."""
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            return value

    def get_or_build(self, key: str, build: Callable[[], T]) -> T:
        """Return the cached value or build, store and return it.

        ``build`` runs outside the lock (it may take seconds), so two concurrent misses on the
        same key both build; the second simply overwrites the first with an equal value.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.put(key, build())

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


__all__ = ["LRUCache"]
