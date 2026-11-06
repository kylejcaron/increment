"""Portable warehouse predicates for floating-point special values."""

from __future__ import annotations

import ibis
import ibis.expr.types as ir


def nonfinite_parts(value: ir.FloatingValue) -> tuple[ir.BooleanValue, ir.BooleanValue]:
    """Return portable ``(is_nan, is_infinite)`` predicates."""
    is_nan = value.isnan()
    is_infinite = value.abs() == ibis.literal(float("inf"), type=value.type())
    return is_nan, is_infinite
