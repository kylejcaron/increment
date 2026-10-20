"""Portable warehouse predicates for floating-point special values."""

from __future__ import annotations

import ibis
import ibis.expr.types as ir


def nonfinite_parts(
    value: ir.FloatingValue, *, dialect: str | None = None
) -> tuple[ir.BooleanValue, ir.BooleanValue]:
    """Return ``(is_nan, is_infinite)`` using backend-portable expressions."""
    backend = dialect or value.get_backend().name
    is_nan = (
        value == ibis.literal(float("nan"), type=value.type())
        if backend == "snowflake"
        else value.isnan()
    )
    is_infinite = value.abs() == ibis.literal(float("inf"), type=value.type())
    return is_nan, is_infinite
