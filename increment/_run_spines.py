"""Per-unit relations one run has already fitted, shared by the run's views.

A spine is the per-unit relation a warehouse fit reads. Two requests read the
same spine only when every coordinate of its key agrees; the registry lives for
one run and holds fitted evidence, never a warehouse relation, so a fit's own
work relations are dropped when that fit ends.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, cast


@dataclass(frozen=True, slots=True)
class SpineKey:
    """Every coordinate that distinguishes two per-unit relations within one run.

    ``effective_metric`` is the metric definition with its resolved method, control and
    admitted arms; ``horizon`` the observation end, snapshot cutoff and captured event
    edge; ``source_content_digest`` the source's canonical identity (artifact manifests
    and snapshot evidence); ``source_selection`` the relations the scoped source resolved
    and the breakout cell. Each text field is canonical JSON or a SHA-256 digest.
    """

    population: Literal["assigned", "triggered"]
    effective_metric: str
    horizon: str
    source_content_digest: str
    source_selection: str


_RUN_SPINES: ContextVar[dict[SpineKey, object] | None] = ContextVar("_RUN_SPINES", default=None)


@contextmanager
def run_spines() -> Iterator[None]:
    """Share fitted spines among one run's views and discard them when the run ends."""
    registry: dict[SpineKey, object] = {}
    token = _RUN_SPINES.set(registry)
    try:
        yield
    finally:
        _RUN_SPINES.reset(token)
        registry.clear()


def reused_spine[T](key: SpineKey | None, fit: Callable[[], T]) -> T:
    """The run's evidence for *key*, fitted once; outside a run, or without a key, fit."""
    registry = _RUN_SPINES.get()
    if key is None or registry is None:
        return fit()
    if key not in registry:
        registry[key] = fit()
    return cast(T, registry[key])
