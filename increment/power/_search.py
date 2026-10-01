"""Bisection, float64-ordinal, log-scale noncentrality, and null/reason
invariant helpers, plus the minimum-detectable-effect refusal record, shared
by the power solvers in core.py, sequential.py, switchback.py, and
unit_cycle.py.
"""

from __future__ import annotations

import math
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from increment.errors import InvalidRequestError, RefusalSpec

if TYPE_CHECKING:
    from increment.power.core import MdeUnavailableReason

_SIGN_BIT = 1 << 63
_MAGNITUDE_MASK = _SIGN_BIT - 1
_LOG_FLOAT_MAX = math.log(sys.float_info.max)


def float_ordinal(x: float) -> int:
    """Position of `x` in the totally ordered float64 line (+-0 share 0)."""
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    magnitude = bits & _MAGNITUDE_MASK
    return -magnitude if bits & _SIGN_BIT else magnitude


def float_from_ordinal(ordinal: int) -> float:
    bits = (_SIGN_BIT | -ordinal) if ordinal < 0 else ordinal
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def bisect_first_true(lo: int, hi: int, accept: Callable[[int], bool]) -> int:
    """Smallest integer in [lo, hi] where `accept` is True, given `accept`
    switches from False to True exactly once across the range and
    `accept(hi)` is True. Requires lo <= hi. O(log(hi - lo)) evaluations."""
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if accept(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


def require_reason_when_null(
    value: Any, reason: Any, *, name: str, invalid: Callable[[str], None]
) -> None:
    """Enforce that exactly one of `value`/`reason` is set: a value present
    with a reason, or a value absent with no reason, is the defect this
    guards. Calls `invalid(message)` on violation instead of raising
    directly, so each caller keeps its own coded refusal/pydantic-validator
    error type."""
    if (value is None) == (reason is None):
        invalid(f"{name} must have a reason exactly when unavailable")


def _noncentrality(distance: float, log_se_sq: float) -> float:
    """``distance / sqrt(S^2)`` from ``log S^2``, saturating to ``±inf``
    instead of overflowing when the variance underflows."""
    if distance == 0.0:
        return 0.0
    log_abs = math.log(abs(distance)) - 0.5 * log_se_sq
    if log_abs >= _LOG_FLOAT_MAX:
        return math.copysign(math.inf, distance)
    return math.copysign(math.exp(log_abs), distance)


def _se_from_log(log_se_sq: float) -> float:
    """``sqrt(S^2)`` from ``log S^2``, saturating to ``inf`` on overflow."""
    half = 0.5 * log_se_sq
    return math.inf if half >= _LOG_FLOAT_MAX else math.exp(half)


_MDE_NUMERICAL_RESOLUTION = RefusalSpec(
    "power.minimum_detectable_effect.numerical_resolution",
    InvalidRequestError,
    template="the minimum detectable effect at power={target_power} in the {direction} direction could not be resolved: distances {unresolved_interval} enclose power {power_enclosure} ({stopping_reason})",
)


@dataclass(frozen=True, slots=True)
class _MdeRefusal:
    """A minimum detectable effect that does not exist at the requested target.

    ``minimum_detectable_effect`` raises it; supplied-effect solvers keep
    their answer and record ``reason`` beside a null companion effect.
    """

    reason: MdeUnavailableReason
    spec: RefusalSpec
    context: dict[str, object]
