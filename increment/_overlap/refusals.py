"""Coded refusals owned by the overlapping-experiments family."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "overlap.design.duplicate_experiment": "both experiments are named {name!r}; an overlap analysis needs two distinct experiments",
        "overlap.design.arm_labels": "experiment {name!r} uses {label!r} for both control and treatment",
        "overlap.design.treatment_probability": "experiment {name!r} declares treatment_probability={probability!r}; it must be a float strictly between 0 and 1",
        "overlap.design.joint_law_unsupported": RefusalSpec(
            "overlap.design.joint_law_unsupported",
            UnsupportedRequestError,
            template="joint assignment law {joint_law!r} is not supported; only independently randomized experiments are analyzed, because dependent or mutually exclusive layers leave some launch combinations unassigned or confounded with the assignment law. Analyze each experiment separately with Analysis, or declare joint_law='independent' only when the two assignments were drawn independently",
        ),
        "overlap.ingress.missing_columns": "overlap frame is missing required columns {missing!r}",
        "overlap.ingress.columns_unbound": "columns are bound for experiments {bound!r}, but the design declares {declared!r}; bind exactly the declared experiments",
        "overlap.ingress.no_metrics": "at least one metric column is required",
        "overlap.ingress.unit_missing": "{count} row(s) have no unit identifier; every row must belong to an identifiable unit",
        "overlap.ingress.duplicate_units": "{count} unit(s) appear on more than one row (for example {examples!r}); join assignments and outcomes to one row per unit before analysis",
        "overlap.ingress.eligibility_missing": "experiment {experiment!r} has {count} row(s) with missing eligibility; eligibility must be recorded before treatment for every unit",
        "overlap.ingress.unknown_arm": "experiment {experiment!r} has {count} assigned row(s) whose arm is neither {control!r} nor {treatment!r} (for example {examples!r})",
        "overlap.ingress.eligible_unassigned": "experiment {experiment!r} has {count} eligible unit(s) with no assignment; an eligible unit that was never assigned breaks the declared assignment law",
        "overlap.ingress.ineligible_assigned": "experiment {experiment!r} has {count} ineligible unit(s) carrying an assignment; absence from an experiment is recorded as no assignment, never as an arm",
        "overlap.ingress.assignment_time_missing": "experiment {experiment!r} has {count} assigned unit(s) with no assignment time",
        "overlap.timing.window_missing": "{count} jointly eligible unit(s) have no outcome window start or end",
        "overlap.timing.window_order": "{count} jointly eligible unit(s) have an outcome window end at or before its start",
        "overlap.timing.outcome_precedes_assignment": "{count} jointly eligible unit(s) open their outcome window before being assigned to both experiments; outcomes must be measured after the later of the two assignments",
        "overlap.timing.horizon_unaligned": "jointly eligible units have {count} distinct outcome window lengths (for example {examples!r}); fixed-horizon contrasts need one common window length",
        "overlap.timing.dtype": "column {column!r} must hold dates or timestamps, got dtype {dtype}",
        "overlap.timing.awareness_mixed": "timestamps mix naive and timezone-aware values across columns {columns!r}; use one convention",
        "overlap.outcome.missing": "metric {metric!r} has {count} jointly eligible unit(s) with a missing outcome; impute or exclude them before analysis rather than letting absence change the target population",
        "overlap.outcome.nonfinite": "metric {metric!r} has {count} jointly eligible unit(s) with a non-finite outcome",
        "overlap.outcome.non_numeric": "metric {metric!r} column {column!r} is not numeric (dtype {dtype})",
        "overlap.target.empty": "no unit is eligible for both {first!r} and {second!r}; the jointly eligible target population is empty",
        "overlap.request.contrasts_empty": "declare at least one contrast from {allowed!r}",
        "overlap.request.contrast_unknown": "unknown contrast {contrast!r}; declare contrasts from {allowed!r}",
        "overlap.request.contrast_duplicate": "contrast {contrast!r} is declared more than once",
        "overlap.request.metric_unknown": "metric {metric!r} has a role but no outcome; available metrics are {available!r}",
        "overlap.request.role_unknown": "metric {metric!r} has role {role!r}; overlap readouts support 'primary', 'secondary' and 'unassigned'",
        "overlap.request.alpha": "alpha must be a float in (0, 1), got {alpha!r}",
        "overlap.request.q": "q must be a float in (0, 1), got {q!r}",
        "overlap.multiplicity.guardrail_unfinished": RefusalSpec(
            "overlap.multiplicity.guardrail_unfinished",
            UnsupportedRequestError,
            template="metric {metric!r} is declared as a guardrail, but non-inferiority guardrails over joint launch contrasts are not implemented; declare it 'secondary' or 'unassigned', or read the guardrail per experiment through Analysis",
        ),
        "overlap.inference.unrepresentable": RefusalSpec(
            "overlap.inference.unrepresentable",
            CapabilityError,
            template="contrast {contrast!r} for metric {metric!r} has no float64 representation ({nonfinite!r} not finite)",
        ),
        "overlap.planning.cell_probability": "experiment allocation leaves cell {cell!r} with probability {probability!r}; every launch combination needs positive allocation",
        "overlap.planning.sd": "outcome standard deviation for cell {cell!r} must be a finite positive float, got {sd!r}",
        "overlap.planning.n_total": "n_total must be an integer of at least 8 jointly eligible units, got {n_total!r}",
        "overlap.planning.power": "power must be a float in (alpha / 2, 1), got {power!r}",
        "overlap.planning.effect": "effect must be a finite float, got {effect!r}",
        "overlap.planning.family_size": "family_size must be a positive integer, got {family_size!r}",
    },
)
_raise = raiser(_REFUSALS)

OVERLAP_REFUSALS: Mapping[str, RefusalSpec] = MappingProxyType(_REFUSALS)
