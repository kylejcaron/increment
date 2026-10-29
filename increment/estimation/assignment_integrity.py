"""Assignment-integrity classification over already-consumed source counts.

No source reads or metric sample-size substitution belong in this producer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from increment.errors import CapabilityError, RefusalSpec
from increment.estimation.diagnostics import sample_ratio_mismatch
from increment.semantics.design import Encouragement, Observational, Randomized

if TYPE_CHECKING:
    from increment.estimation.readout_types import IntegrityResult

ALLOCATION_SCHEME_MISSING_REFUSAL = RefusalSpec(
    "integrity.allocation_scheme_missing",
    CapabilityError,
    template=(
        "SRM requires a declared target allocation for population "
        "{population!r}. Declare Randomized(control_group=..., "
        "allocation={{...}}, allocation_scheme=...) and retry."
    ),
    keys=frozenset({"population"}),
)
ALLOCATION_MISSING_OR_NONCONSTANT_REFUSAL = RefusalSpec(
    "integrity.allocation_missing_or_nonconstant",
    CapabilityError,
    template=(
        "SRM requires a declared target allocation for population "
        "{population!r}. Declare Randomized(control_group=..., "
        "allocation={{...}}, allocation_scheme=...) and retry."
    ),
    keys=frozenset({"population"}),
)


def assignment_integrity(
    design: object,
    counts: Mapping[str, int] | None,
    *,
    population: Literal["assigned", "triggered"] = "assigned",
    randomization_grain: Literal["unit", "cluster"] | None = None,
    cumulative: bool = True,
    constant_allocation: bool = True,
    order_draws: tuple[tuple[str, str], ...] = (),
) -> IntegrityResult:
    """Classify a declared assignment law using only its matching randomization grain."""
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
    from increment.semantics.assignment import SwitchbackAssignment

    if isinstance(design, SwitchbackAssignment):
        sequence = design.sequence
        probabilities = {"ct": sequence.probability_ct, "tc": 1.0 - sequence.probability_ct}
        expected_grain = sequence.independence_unit
        common.update(
            expected=probabilities,
            randomization_grain=expected_grain,
            context={"randomization_law": sequence.scheme},
        )
        if not constant_allocation:
            common["observed"] = None
            return IntegrityResult(
                **common,
                status="not_checked_missing_declaration",
                code="integrity.allocation_missing_or_nonconstant",
            )
        if population == "triggered":
            common["observed"] = None
            return IntegrityResult(
                **common,
                status="not_checked_missing_counts",
                code="integrity.triggered_population_not_checked",
            )
        observed = {name: sum(order == name for _, order in order_draws) for name in probabilities}
        common["observed"] = observed
        if not cumulative or not order_draws:
            return IntegrityResult(
                **common,
                status="not_checked_missing_counts",
                code="integrity.counts_missing",
            )
        result = sample_ratio_mismatch(
            observed,
            expected=probabilities,
            inference="always_valid",
            alpha=0.001,
            grain="unit",
        )
        return IntegrityResult(
            status="failed" if result.is_srm else "not_rejected",
            analysis_population=population,
            construction="always_valid",
            alpha=result.alpha,
            observed=result.observed,
            expected=result.expected,
            randomization_grain=expected_grain,
            code=None,
            context={
                **common["context"],
                "order_draws": len(order_draws),
                "log_e_value": result.log_e_value,
                "unassigned_units": result.unassigned_units,
                "mixed_assignment_units": result.mixed_assignment_units,
            },
        )
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
