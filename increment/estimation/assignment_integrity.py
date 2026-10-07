"""Assignment-integrity classification over already-consumed source counts.

No source reads or metric sample-size substitution belong in this producer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from increment.estimation.diagnostics import sample_ratio_mismatch
from increment.semantics.design import Encouragement, Observational, Randomized

if TYPE_CHECKING:
    from increment.estimation.readout_types import IntegrityResult


def assignment_integrity(
    design: object,
    counts: Mapping[str, int] | None,
    *,
    population: Literal["assigned", "triggered"] = "assigned",
    randomization_grain: Literal["unit", "cluster"] | None = None,
    cumulative: bool = True,
    constant_allocation: bool = True,
) -> IntegrityResult:
    """Classify the declared law, then check a compatible cumulative prefix.

    ``counts`` must be experiment-level assignment counts at the randomization
    grain, not metric observations or encouragement uptake counts. Call once
    per source snapshot; retain the assigned result when displaying triggered
    rows. Non-rejection is neither certification nor a historical alarm log.
    """
    from increment.estimation.readout_types import IntegrityResult

    allocation = getattr(design, "allocation", None)
    scheme = getattr(design, "allocation_scheme", None)
    common = {
        "analysis_population": population,
        "construction": "none",
        "alpha": None,
        "observed": counts,
        "expected": allocation,
        "randomization_grain": randomization_grain,
        "context": {},
    }
    if isinstance(design, Observational):
        return IntegrityResult(**common, status="not_applicable", code=None)
    if not isinstance(design, (Randomized, Encouragement)):
        return IntegrityResult(
            **common,
            status="unsupported_assignment",
            code="integrity.switchback_assignment_law",
        )
    if population == "triggered":
        common["observed"] = None
        common["context"] = {
            "allocation_scheme": scheme,
            "reason": "post_assignment_selection",
        }
        return IntegrityResult(
            **common,
            status="not_checked_missing_counts",
            code="integrity.triggered_population_not_checked",
        )
    if scheme is None:
        common["context"] = {"allocation_scheme": None}
        return IntegrityResult(
            **common,
            status="not_checked_missing_declaration",
            code="integrity.allocation_scheme_missing",
        )
    if scheme != "independent":
        common["context"] = {"allocation_scheme": scheme}
        return IntegrityResult(
            **common,
            status="unsupported_assignment",
            code="integrity.allocation_scheme_unsupported",
        )
    if allocation is None or not constant_allocation:
        return IntegrityResult(
            **common,
            status="not_checked_missing_declaration",
            code="integrity.allocation_missing_or_nonconstant",
        )
    if counts is None or not cumulative:
        return IntegrityResult(
            **common,
            status="not_checked_missing_counts",
            code="integrity.counts_missing",
        )
    result = sample_ratio_mismatch(
        counts,
        expected=allocation,
        inference="always_valid",
        alpha=0.001,
        grain=randomization_grain or "unit",
    )
    return IntegrityResult(
        status="failed" if result.is_srm else "not_rejected",
        analysis_population=population,
        construction="always_valid",
        alpha=result.alpha,
        observed=result.observed,
        expected=result.expected,
        randomization_grain=result.grain,
        code=None,
        context={
            "allocation_scheme": scheme,
            "log_e_value": result.log_e_value,
            "unassigned_units": result.unassigned_units,
            "mixed_assignment_units": result.mixed_assignment_units,
        },
    )
