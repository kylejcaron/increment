"""Shared tolerant comparisons for parity payloads and retained state."""

from __future__ import annotations

import math
from typing import Any

TOLERANCE = 1e-9


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def nested_close(expected: Any, actual: Any) -> bool:
    """Compare recursive shape, null availability and numeric values.

    Finite floats use relative ``TOLERANCE`` with the same absolute floor;
    integers and other values compare exactly. Infinities equal only the
    same-signed infinity, and NaNs never compare equal.
    """
    if expected is None or actual is None:
        return expected is None and actual is None
    if _is_real(expected) and _is_real(actual):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        if not (math.isfinite(expected) and math.isfinite(actual)):
            return expected == actual
        return abs(expected - actual) <= TOLERANCE * max(1.0, abs(expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            nested_close(expected[k], actual[k]) for k in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            nested_close(e, a) for e, a in zip(expected, actual, strict=True)
        )
    return expected == actual
