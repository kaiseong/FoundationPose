"""Timing helpers used by capture, tracking, and GUI code."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@contextmanager
def record_elapsed_ms(target: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        target[key] = (time.perf_counter() - start) * 1000.0
