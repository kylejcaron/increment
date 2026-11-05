"""Float literals that warehouse SQL reads as the exact doubles they name."""

from __future__ import annotations

from typing import cast

import ibis
import ibis.expr.types as ir


def exact_double(value: float) -> ir.FloatingScalar:
    """Return ``value`` as ``CAST('<shortest repr>' AS DOUBLE)``.

    DuckDB reads a bare decimal literal as DECIMAL and rounds twice converting
    it to DOUBLE, which can miss the nearest double and differ between ``c``
    and ``-c``. Parsing the shortest round-trip text as a double rounds once.
    """
    return cast("ir.FloatingScalar", ibis.literal(repr(float(value))).cast("float64"))
