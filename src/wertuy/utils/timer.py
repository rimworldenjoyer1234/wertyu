from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed() -> Iterator[list[float]]:
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
