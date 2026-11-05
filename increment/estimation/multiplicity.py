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

#: Failure codes of outcome-based cell degeneracy (zero variance, unusable SE, non-positive
#: mean, a CUPED fit guard, no usable observations): a fixed-horizon family keeps such a cell
#: as a sure non-rejection (p = 1), so its size stays fixed by counts. run(), run_breakout()
#: and stored decisions read the same set.
OUTCOME_NON_REJECTIONS: frozenset[str] = frozenset(
    {
        "estimation.engine.lift_guard",
        "breakout.nonpositive_mean",
        "breakout.zero_variance",
        "breakout.extreme_ratio",
        "estimation.cuped_linear.level_support",
        "estimation.cuped_linear.rank_deficiency",
        "estimation.cuped_linear.leverage",
        "estimation.cuped_linear.residual_df",
        "estimation.cuped_linear.nonpositive_component",
        "estimation.cuped_linear.nonfinite_projection",
        "estimation.cuped_linear.missing_refused",
        "estimation.cuped_linear.support_region",
        "frame.validation.metric_covariate_column",
        "frame.validation.metric_covariate_impute_all_null",
        "readout.cell.missing_metric_observations",
    }
)


def outcome_non_rejection(code: str | None, reason: object = None) -> bool:
    """Whether a failure *code* (with its context *reason*) is an outcome-based non-rejection."""
    return code in OUTCOME_NON_REJECTIONS or (
        code == "evidence.p_value.unavailable" and reason == "zero_relative_variance"
    )


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
