"""Disclosure of the declared role and authoritative family, not a correction."""

from __future__ import annotations

from typing import Literal

from increment._literals import RowRole
from increment._multiplicity import MultiplicityFamily

MultiplicityStatus = Literal[
    "declared_plan",
    "unassigned_in_plan",
    "undeclared_plan",
    "exploratory_unadjusted",
    "exploratory_family",
]

#: Unavailable-cell reasons fixed by arm counts alone (an arm with fewer than two units, or
#: an arm absent from the segment). Counts are ancillary to outcomes under randomization, so
#: such a cell is no hypothesis of a fixed-horizon family; outcome-based reasons stay in the
#: family as non-rejections.
DESIGN_EXCLUSIONS: frozenset[str] = frozenset({"few_units", "no_control_arm", "no_treatment_arm"})


def design_excluded(row: object) -> bool:
    """Whether *row* is unavailable for a reason fixed by arm counts alone."""
    return getattr(row, "excluded", None) in DESIGN_EXCLUSIONS


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
        if correction in ("bh", "e_bh", "bonferroni"):
            return "exploratory_family"
        return "exploratory_unadjusted"
    return "declared_plan"


def row_multiplicity_status(row, *, correction: str | None = None) -> MultiplicityStatus:
    """Project a row from its existing fields and the effective correction, when known.

    Design-excluded exploratory rows sit outside the corrected family.
    """
    if correction is None and getattr(row, "family_q", None) is not None:
        correction = "e_bh" if row.inference == "always_valid" else "bh"
    if row.role == "exploratory" and design_excluded(row):
        return "exploratory_unadjusted"
    return multiplicity_status(row.role, correction)


def stamp_multiplicity_status(rows, *, correction: str | None = None):
    """Return row copies carrying their existing role/family provenance."""
    return [
        row.model_copy(
            update={"multiplicity_status": row_multiplicity_status(row, correction=correction)}
        )
        for row in rows
    ]
