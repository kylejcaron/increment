"""Canonical floating moments for an exact Bernoulli arm count."""

from __future__ import annotations

import math
from fractions import Fraction


def canonical_bernoulli_family(n: int, successes: int) -> tuple[float, float, float]:
    """Return the canonical ``(ref_y, cy1, cy2)`` family for integer counts."""
    ref_y = successes / n
    cy1 = float(Fraction(successes) - n * Fraction(ref_y))
    root_n = math.sqrt(n)
    cy2 = float(Fraction(successes * (n - successes), n)) + (cy1 / root_n) * (cy1 / root_n)
    return ref_y, cy1, cy2
