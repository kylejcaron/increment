"""Shared runtime validation for power-analysis inputs."""

from __future__ import annotations

import math
import numbers

import numpy as np

from increment.errors import InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "power.planned_looks_positive": "planned_looks must be a positive integer, got {planned_looks!r}",
        "power.planned_looks": "planned_looks must be >= 1, got {planned_looks}",
        "power.finite": "{name} must be finite, got {value!r}",
        "power.require_relative_domain": "{name} must be > -1.0, got {value}",
    },
)
_raise = raiser(_REFUSALS)


def _as_count(value: object) -> int | None:
    """``value`` as an ``int`` when it is an integer of any integer type, such
    as a numpy integer read from a dataframe column; ``None`` otherwise.
    Bools and floats, integral-valued ones included, are not counts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _validate_planned_looks(planned_looks: object) -> int:
    looks = _as_count(planned_looks)
    if looks is None:
        _raise("power.planned_looks_positive", planned_looks=planned_looks)
    if looks < 1:
        _raise("power.planned_looks", planned_looks=planned_looks)
    return looks


def _require_finite(name: str, value: float) -> float:
    """Reject a NaN/inf public numeric input before it reaches a solver."""
    value = float(value)
    if not math.isfinite(value):
        _raise("power.finite", name=name, value=value)
    return value


def _require_relative_domain(name: str, value: float) -> float:
    """A relative lift -- caller input or solved-for result -- cannot
    reduce a positive baseline to zero or below. This ``> -1.0`` floor is
    shared by every relative-lift input and every ``mde_relative`` output.
    """
    _require_finite(name, value)
    if value <= -1.0:
        _raise("power.require_relative_domain", name=name, value=value)
    return value
