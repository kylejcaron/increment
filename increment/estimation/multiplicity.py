"""Disclosure of the declared role and authoritative family, not a correction."""

from __future__ import annotations

from typing import Literal

from increment._literals import RowRole
from increment.decision import MultiplicityFamily

MultiplicityStatus = Literal[
    "declared_plan",
    "unassigned_in_plan",
    "undeclared_plan",
    "exploratory_unadjusted",
    "exploratory_family",
]


def multiplicity_status(
    role: RowRole | None, family: MultiplicityFamily | str | None
) -> MultiplicityStatus:
    """Project provenance without changing roles, alpha, or family membership.

    ``family`` is the procedure of the source-scoped family containing the
    cell, not a family reconstructed from the currently displayed rows.
    """
    correction = getattr(family, "correction", family)
    if role is None:
        return "undeclared_plan"
    if role == "unassigned":
        return "unassigned_in_plan"
    if role == "exploratory":
        if correction in ("bh", "e_bh"):
            return "exploratory_family"
        return "exploratory_unadjusted"
    return "declared_plan"


def row_multiplicity_status(row) -> MultiplicityStatus:
    """Project a row from its existing effective-family disclosure fields."""
    correction = None
    if getattr(row, "family_q", None) is not None:
        correction = "e_bh" if row.inference == "always_valid" else "bh"
    return multiplicity_status(row.role, correction)


def stamp_multiplicity_status(rows):
    """Return row copies carrying their existing role/family provenance."""
    return [
        row.model_copy(update={"multiplicity_status": row_multiplicity_status(row)}) for row in rows
    ]
