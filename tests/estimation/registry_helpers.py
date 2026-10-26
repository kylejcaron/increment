from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from increment.declarations import ADJUSTMENTS


@contextmanager
def registered_adjustment(name: str, fn: Callable[..., Any]) -> Iterator[None]:
    """Register and fully remove one test plugin, including its estimator ref."""
    ADJUSTMENTS.register(name, fn)
    try:
        yield
    finally:
        ADJUSTMENTS.unregister(name)
