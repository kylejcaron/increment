"""Shared declaration checks for longitudinal source adapters."""

from __future__ import annotations

from collections.abc import Sequence

from increment.estimation.cate import Covariate
from increment.semantics.longitudinal import LongitudinalDesign, invalid


def validate_columns(
    *,
    unit: str,
    group: str,
    cohort: str,
    exposure: str,
    time: str,
    outcome: str,
    cluster: str | None,
    covariates: Sequence[Covariate],
    design: LongitudinalDesign,
) -> None:
    """Validate metadata declarations before either adapter reads data."""
    LongitudinalDesign.model_validate(design.model_dump())
    names = [
        unit,
        group,
        cohort,
        exposure,
        *([cluster] if cluster is not None else []),
        *(c.name for c in covariates),
    ]
    if any(not n or n.startswith("__long_") for n in [*names, time, outcome]):
        invalid("Use nonempty column names outside the reserved __long_ prefix.")
    if len(names) != len(set(names)) or len({unit, time, outcome}) != 3:
        invalid("Roster roles/covariates and outcome roles must use distinct columns.")
    if (cluster is not None) != (design.independence_grain == "cluster"):
        invalid("Supply cluster exactly when independence_grain='cluster'.")
    if any(c.knots is not None for c in covariates):
        invalid(
            "Longitudinal covariates currently support linear continuous and categorical bases; remove knots."
        )


def regression_columns(design: LongitudinalDesign, basis_names: Sequence[str]) -> tuple[str, ...]:
    """Deterministic cohort/arm block labels shared by producers."""
    require_basis_size(design, len(basis_names))
    return tuple(
        f"{j}:{arm}:{name}"
        for j, _ in enumerate(design.cohorts)
        for arm in ("control", "treatment")
        for name in ("intercept", *basis_names)
    )


def require_basis_size(design: LongitudinalDesign, n_basis: int) -> None:
    """Bound regression width before allocating a basis or generating SQL products."""
    if 2 * len(design.cohorts) * (1 + n_basis) > 48:
        invalid(
            "Longitudinal compression supports at most 48 regression columns (1224 moment fields); reduce cohorts or covariates."
        )
