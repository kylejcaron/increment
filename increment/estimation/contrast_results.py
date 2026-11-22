"""Typed fixed-horizon switchback contrast results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from operator import index
from typing import TYPE_CHECKING, Literal, SupportsIndex, overload

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from increment._canonical import canonical_json_bytes
from increment._literals import (
    Alternative,
    Backend,
    MetricType,
    PreferredDirection,
    Role,
    default_frame_backend,
)
from increment._unit_cycle import unit_cycle_envelope_cutoff
from increment.declarations import freeze, thaw
from increment.errors import CodedModel, InvalidRequestError, RefusalSpec, refuse
from increment.estimation.readout_types import (
    _restore_collection,
    concat_collection,
    dump_collection,
    partial_metadata,
    refuse_readout,
    validate_collection,
)
from increment.estimation.results import (
    Estimate,
    RejectionSide,
    contrast_p_value,
    contrast_stat_sig,
)
from increment.semantics.unit_cycle import (
    ProspectiveAssumptionProvenance,
    UnitCycleReference,
    UnitCycleTApproximation,
    UnitCycleVarianceEnvelope,
)

if TYPE_CHECKING:
    from increment.estimation.readout_types import ReadoutMetadata


# Independent units and shared blocks have different replication grains.
RandomizationLaw = Literal["independent_bernoulli_order", "shared_schedule", "fixed_count"]
IndependenceGrain = Literal["unit_cycle", "shared_block"]

ContrastAggregation = Literal["sum", "any", "ratio"]
ContrastMethod = Literal[
    "switchback_unit_t_approximation",
    "switchback_block_t",
    "switchback_unit_variance_envelope",
    "switchback_ratio_delta_t",
    "switchback_cuped_jackknife_t",
]
ContrastReference = Literal[
    "unit_t_approximation",
    "block_t",
    "residual_chebyshev",
    "residual_cantelli",
    "switchback_ratio_delta_t",
    "switchback_cuped_jackknife_t",
]
ContrastEstimand = Literal[
    "retained_window_total_difference",
    "mean_unit_retained_window_difference",
    "retained_window_conversion_difference",
    "retained_window_ratio_difference",
]


def _contrast_identity(
    aggregation: ContrastAggregation,
    forecast_column: str | None,
    grain: IndependenceGrain,
    reference: UnitCycleReference | None,
    alternative: Alternative,
) -> tuple[ContrastMethod, ContrastReference, ContrastEstimand]:
    if aggregation == "ratio":
        return (
            "switchback_ratio_delta_t",
            "switchback_ratio_delta_t",
            "retained_window_ratio_difference",
        )
    if forecast_column is not None:
        return (
            "switchback_cuped_jackknife_t",
            "switchback_cuped_jackknife_t",
            "retained_window_total_difference",
        )
    if isinstance(reference, UnitCycleVarianceEnvelope):
        return (
            "switchback_unit_variance_envelope",
            "residual_chebyshev" if alternative == "two-sided" else "residual_cantelli",
            "mean_unit_retained_window_difference",
        )
    estimand: ContrastEstimand = (
        "retained_window_conversion_difference"
        if aggregation == "any"
        else "retained_window_total_difference"
    )
    if grain == "shared_block":
        return "switchback_block_t", "block_t", estimand
    return "switchback_unit_t_approximation", "unit_t_approximation", estimand


_REFUSALS = {
    "estimation.contrast.assignment_metadata": RefusalSpec(
        "estimation.contrast.assignment_metadata",
        InvalidRequestError,
        template="{randomization_law} requires {expected_grain}, got {independence_grain}",
    ),
    "estimation.contrast.exposure": RefusalSpec(
        "estimation.contrast.exposure",
        InvalidRequestError,
        template="The stored exposure does not match the declared switchback law.",
    ),
    "estimation.contrast.replication_counts": RefusalSpec(
        "estimation.contrast.replication_counts",
        InvalidRequestError,
        template="contrast counts must match the declared independent replicate grain",
    ),
    "estimation.contrast.order_counts": RefusalSpec(
        "estimation.contrast.order_counts",
        InvalidRequestError,
        template="CT and TC counts must sum to the declared number of order draws",
    ),
    "estimation.contrast.reference_metadata": RefusalSpec(
        "estimation.contrast.reference_metadata",
        InvalidRequestError,
        template="contrast method, reference and degrees of freedom must match its replicates",
    ),
    "estimation.contrast.legacy_envelope": RefusalSpec(
        "estimation.contrast.legacy_envelope",
        InvalidRequestError,
        template=(
            "this switchback envelope result was saved before it recorded which side of the "
            "null its evidence rejects on; recompute it from the source data"
        ),
        keys=frozenset({"missing_fields", "remedy"}),
    ),
    "estimation.contrast.retained_window": RefusalSpec(
        "estimation.contrast.retained_window",
        InvalidRequestError,
        template="retained_steps={retained_steps} must equal observation_steps={observation_steps} minus carryover_order={carryover_order} and be positive",
    ),
}


def _valid_missing_key(
    context: Mapping[str, object], metric: str, reason: str, key_length: int
) -> bool:
    count = context.get("missing_count")
    key = context.get("first_missing_key")
    return (
        context.get("metric") == metric
        and context.get("reason") == reason
        and type(count) is int
        and count > 0
        and isinstance(key, (tuple, list))
        and len(key) == key_length
        and isinstance(key[0], str)
        and all(type(index) is int for index in key[1:])
    )


def _valid_missing_response(
    context: Mapping[str, object], metric: str, control: str, treatment: str
) -> bool:
    return _valid_missing_key(context, metric, "unobserved_response", 4) and context.get("arm") in (
        control,
        treatment,
    )


def _valid_missing_forecast(
    context: Mapping[str, object], metric: str, forecast_column: str | None
) -> bool:
    return (
        forecast_column is not None
        and context.get("column") == forecast_column
        and _valid_missing_key(context, metric, "missing_forecast", 3)
    )


def _validate_retained_window(
    observation_steps: int, retained_steps: int, carryover_order: int
) -> None:
    if retained_steps != observation_steps - carryover_order or retained_steps < 1:
        refuse(
            _REFUSALS["estimation.contrast.retained_window"],
            observation_steps=observation_steps,
            retained_steps=retained_steps,
            carryover_order=carryover_order,
        )


def _validate_assignment_metadata(law: RandomizationLaw, grain: IndependenceGrain) -> None:
    if law == "fixed_count":
        return
    expected = "unit_cycle" if law == "independent_bernoulli_order" else "shared_block"
    if grain != expected:
        refuse(
            _REFUSALS["estimation.contrast.assignment_metadata"],
            randomization_law=law,
            independence_grain=grain,
            expected_grain=expected,
        )


def _validate_replication_counts(
    law: RandomizationLaw,
    grain: IndependenceGrain,
    n_units: int,
    n_cycles: int,
    n_blocks: int | None,
) -> int:
    if grain == "unit_cycle":
        if n_blocks is None and n_cycles >= n_units:
            return n_units
    elif n_blocks is not None and n_cycles == n_units * n_blocks:
        return n_blocks
    refuse(
        _REFUSALS["estimation.contrast.replication_counts"],
        randomization_law=law,
        n_units=n_units,
        n_cycles=n_cycles,
        n_blocks=n_blocks,
    )


def _validate_order_counts(
    law: RandomizationLaw,
    n_cycles: int,
    n_blocks: int | None,
    ct_cycles: int | None,
    tc_cycles: int | None,
) -> None:
    if law != "shared_schedule" and ct_cycles is None and tc_cycles is None:
        return
    draws = n_cycles if law == "independent_bernoulli_order" else n_blocks
    if (
        law == "fixed_count"
        or ct_cycles is None
        or tc_cycles is None
        or ct_cycles + tc_cycles != draws
    ):
        refuse(
            _REFUSALS["estimation.contrast.order_counts"],
            randomization_law=law,
            n_cycles=n_cycles,
            n_blocks=n_blocks,
            ct_cycles=ct_cycles,
            tc_cycles=tc_cycles,
        )


class _FixedCountMetadata(CodedModel, BaseModel):
    """Law-specific metadata is absent from historical paired payloads."""

    periods_per_cycle: int | None = Field(default=None, strict=True, exclude_if=lambda x: x is None)
    treated_periods: int | None = Field(default=None, strict=True, exclude_if=lambda x: x is None)
    treatment_probability: float | None = Field(default=None, exclude_if=lambda x: x is None)

    def _validate_exposure(self, law: RandomizationLaw, probability_ct: float | None) -> None:
        k, m, p = self.periods_per_cycle, self.treated_periods, self.treatment_probability
        if law == "fixed_count":
            if (
                k is not None
                and m is not None
                and k >= 3
                and 0 < m < k
                and p == m / k
                and probability_ct is None
            ):
                return
        elif k is None and m is None and p is None and probability_ct is not None:
            return
        refuse(
            _REFUSALS["estimation.contrast.exposure"],
            randomization_law=law,
            periods_per_cycle=k,
            treated_periods=m,
            treatment_probability=p,
            probability_ct=probability_ct,
        )


class ContrastResult(_FixedCountMetadata):
    """One additive contrast with realized unit-cycle or shared-block order counts."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    metric: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    method: ContrastMethod = "switchback_unit_t_approximation"
    method_role: Literal["decision"] = "decision"
    role: Role = "unassigned"
    estimand: ContrastEstimand
    aggregation: ContrastAggregation
    probability_ct: float | None = Field(gt=0.0, lt=1.0)
    # Positive carryover orders discard additional post-washout steps.
    randomization_law: RandomizationLaw
    independence_grain: IndependenceGrain
    carryover_order: int = Field(ge=0, strict=True)
    observation_steps: int = Field(ge=1, strict=True)
    retained_steps: int = Field(ge=1, strict=True)
    identifying_assumption: Literal["no_residual_carryover_after_discarded_steps"] = (
        "no_residual_carryover_after_discarded_steps"
    )
    estimate: Estimate | None
    standard_error: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Descriptive sample SE over independent replicates; not the envelope reference.",
    )
    alternative: Alternative
    preferred_direction: PreferredDirection | None = None
    null_abs: float = Field(allow_inf_nan=False)
    alpha: float = Field(gt=0.0, lt=1.0)
    n_units: int = Field(ge=1, strict=True)
    n_cycles: int = Field(ge=1, strict=True)
    n_blocks: int | None = Field(default=None, ge=1, strict=True)
    dof: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    assignment: Literal["switchback"] = "switchback"
    inference: Literal["fixed"] = "fixed"
    reference: ContrastReference = "unit_t_approximation"
    dof_unavailable_reason: Literal["not_applicable"] | None = None
    standard_error_unavailable_reason: Literal["insufficient_replicates"] | None = None
    mean_slope: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    ct_cycles: int | None = Field(default=None, ge=0, strict=True)
    tc_cycles: int | None = Field(default=None, ge=0, strict=True)
    minimum_cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    maximum_cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    washout_steps: int = Field(default=0, ge=0, strict=True)
    effective_alpha: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    refusal_probability_upper: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    residual_cutoff: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    residual_p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    #: Sign of the tested residual before point rounding; "split" when it is zero.
    #: Omitted for historical paired t references.
    residual_side: RejectionSide | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reference_spec: UnitCycleReference | None = None
    provenance: ProspectiveAssumptionProvenance | None = None
    response_meaning: Literal["retained_total", "pre_normalized_retained_mean"] | None = None
    value_scale: Literal["absolute"] = "absolute"
    analysis_population: Literal["assigned"] = "assigned"
    source_snapshot_id: str | None = None
    #: The declared ``Metric.type`` behind ``metric``; None when unknown (see ``LiftEstimate``).
    metric_type: MetricType | None = Field(default=None, exclude_if=lambda value: value is None)
    decision_scope_complete: bool | None = None
    decision_scope_reason_code: str | None = None
    decision_scope_reason_context: Mapping[str, object] | None = None
    forecast_column: str | None = Field(default=None, min_length=1, exclude_if=lambda v: v is None)
    failure_code: str | None = Field(default=None, exclude_if=lambda v: v is None)
    failure_context: Mapping[str, object] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )

    @model_validator(mode="after")
    def _freeze_scope_context(self):
        for name in ("decision_scope_reason_context", "failure_context"):
            context = getattr(self, name)
            if context is not None:
                canonical_json_bytes(thaw(context))
                object.__setattr__(self, name, freeze(context))
        return self

    @field_serializer("failure_context", "decision_scope_reason_context", when_used="json")
    def _dump_reason_context(self, value):
        return thaw(value)

    @property
    def sampling_available(self) -> bool:
        return self.failure_code is None

    @property
    def sampling_reason_code(self) -> str | None:
        return self.failure_code

    @property
    def sampling_reason_context(self) -> Mapping[str, object] | None:
        return self.failure_context

    @property
    def reference_qualification(self) -> str | None:
        if (
            self.sampling_available
            and self.standard_error
            and self.reference
            in (
                "block_t",
                "unit_t_approximation",
                "switchback_ratio_delta_t",
                "switchback_cuped_jackknife_t",
            )
        ):
            return (
                "This switchback test uses a t approximation. Depending on the schedule and "
                "outcomes, false positives can be more frequent than the chosen test level."
            )
        return None

    def _valid_t_residual(self) -> bool:
        p, side = self.residual_p_value, self.residual_side
        if p is None or side is None:
            return p is None and side is None  # Historical t references omitted both.
        if self.standard_error is None or self.standard_error == 0:
            return False
        if side == "split":
            return p == (1.0 if self.alternative == "two-sided" else 0.5)
        return self.alternative == "two-sided" or side == self.alternative or p >= 0.5

    def _check_metric_result(self, replicates: int) -> bool:
        """Validate fitted references and unavailable tests without inventing uncertainty."""
        estimate = self.estimate
        reason = None if self.failure_context is None else self.failure_context.get("reason")
        adjusted = self.forecast_column is not None
        ratio = self.aggregation == "ratio"
        if (
            (self.method, self.reference, self.estimand)
            != _contrast_identity(
                self.aggregation,
                self.forecast_column,
                self.independence_grain,
                self.reference_spec,
                self.alternative,
            )
            or (adjusted and self.aggregation != "sum")
            or (
                (self.reference_spec is not None)
                != (not (ratio or adjusted) and self.independence_grain == "unit_cycle")
            )
            or any(
                value is not None
                for value in (
                    self.effective_alpha,
                    self.refusal_probability_upper,
                    self.residual_cutoff,
                    self.provenance,
                    self.response_meaning,
                    self.dof_unavailable_reason,
                    self.standard_error_unavailable_reason,
                )
            )
            or (
                estimate is not None
                and (estimate.log_mean is not None or estimate.log_se is not None)
            )
        ):
            return False
        if self.failure_code is None:
            return (
                (ratio or adjusted)
                and self._valid_t_residual()
                and self.failure_context is None
                and estimate is not None
                and self.standard_error is not None
                and self.standard_error > 0
                and self.dof == replicates - (2 if adjusted else 1)
                and estimate.alpha == self.alpha
                and estimate.open_side
                == (
                    "upper"
                    if self.alternative == "greater"
                    else "lower"
                    if self.alternative == "less"
                    else None
                )
            )
        if self.residual_p_value is not None or self.residual_side is not None:
            return False
        if reason == "zero_standard_error":
            return (
                not (ratio or adjusted)
                and self.method in ("switchback_unit_t_approximation", "switchback_block_t")
                and self.failure_code == "evidence.p_value.unavailable"
                and self.failure_context == {"metric": self.metric, "reason": reason}
                and estimate is not None
                and estimate.lb is None
                and estimate.ub is None
                and self.standard_error == 0
                and self.dof == replicates - 1
            )
        if (
            self.failure_context is None
            or self.failure_context.get("metric") != self.metric
            or self.standard_error is not None
            or self.dof is not None
            or (estimate is not None and (estimate.lb is not None or estimate.ub is not None))
        ):
            return False
        if reason == "unobserved_response":
            return (
                self.failure_code == "source.frame.switchback.response_unobserved"
                and estimate is None
                and _valid_missing_response(
                    self.failure_context, self.metric, self.control_group, self.treatment_group
                )
            )
        if reason == "missing_forecast":
            return (
                self.failure_code == "source.frame.switchback.metric"
                and estimate is None
                and _valid_missing_forecast(self.failure_context, self.metric, self.forecast_column)
            )
        if self.failure_code != "evidence.p_value.unavailable" or not (adjusted or ratio):
            return False
        if reason in ("zero_denominator", "weak_denominator"):
            return (
                ratio
                and self.failure_context.get("arm") in (self.control_group, self.treatment_group)
                and (estimate is None) == (reason == "zero_denominator")
            )
        if reason == "rank_deficient":
            return adjusted and estimate is None
        if reason == "high_leverage":
            return adjusted and estimate is not None
        if reason == "zero_spread":
            return estimate is not None
        return reason in ("numeric_range", "insufficient_replicates")

    @model_validator(mode="after")
    def _consistent_design(self) -> ContrastResult:
        _validate_retained_window(self.observation_steps, self.retained_steps, self.carryover_order)
        _validate_assignment_metadata(self.randomization_law, self.independence_grain)
        self._validate_exposure(self.randomization_law, self.probability_ct)
        replicates = _validate_replication_counts(
            self.randomization_law,
            self.independence_grain,
            self.n_units,
            self.n_cycles,
            self.n_blocks,
        )
        _validate_order_counts(
            self.randomization_law, self.n_cycles, self.n_blocks, self.ct_cycles, self.tc_cycles
        )
        if (
            self.aggregation == "ratio"
            or self.forecast_column is not None
            or self.method in ("switchback_ratio_delta_t", "switchback_cuped_jackknife_t")
            or self.failure_code is not None
            or self.failure_context is not None
            or self.estimate is None
            or self.estimand == "retained_window_ratio_difference"
        ):
            if not self._check_metric_result(replicates):
                refuse(
                    _REFUSALS["estimation.contrast.reference_metadata"],
                    randomization_law=self.randomization_law,
                    method=self.method,
                    reference=self.reference,
                    dof=self.dof,
                    independent_replicates=replicates,
                )
            return self
        if self.method == "switchback_unit_variance_envelope":
            if self.residual_side is None:
                # The side cannot be rebuilt from the stored point and slope, which are rounded.
                refuse(
                    _REFUSALS["estimation.contrast.legacy_envelope"],
                    missing_fields=("residual_side",),
                    remedy="recompute_from_source",
                )
            expected_reference = (
                "residual_chebyshev" if self.alternative == "two-sided" else "residual_cantelli"
            )
            envelope = self.reference_spec
            valid = (
                self.randomization_law == "independent_bernoulli_order"
                and isinstance(envelope, UnitCycleVarianceEnvelope)
                and self.reference == expected_reference
                and self.dof is None
                and self.dof_unavailable_reason == "not_applicable"
                and self.mean_slope is not None
                and self.effective_alpha is not None
                and self.refusal_probability_upper is not None
                and self.residual_cutoff is not None
                and self.residual_p_value is not None
                # A zero residual, or one off a one-sided alternative, has p = 1.
                and (self.residual_side != "split" or self.residual_p_value == 1.0)
                and (
                    self.alternative == "two-sided"
                    or self.residual_side == self.alternative
                    or self.residual_p_value == 1.0
                )
                and self.ct_cycles is not None
                and self.tc_cycles is not None
                and self.ct_cycles + self.tc_cycles == self.n_cycles
                and self.minimum_cycles_per_unit == self.maximum_cycles_per_unit
                and self.minimum_cycles_per_unit == envelope.cycles_per_unit
                and self.metric == envelope.metric
                and self.control_group == envelope.control_group
                and self.treatment_group == envelope.treatment_group
                and self.aggregation == envelope.aggregation
                and self.estimand == envelope.estimand
                and self.n_cycles == self.n_units * envelope.cycles_per_unit
                and envelope.assignment.sequence.scheme == "independent_bernoulli_order"
                and self.probability_ct == envelope.assignment.sequence.probability_ct
                and self.washout_steps == envelope.assignment.window.washout_steps
                and self.observation_steps == envelope.assignment.window.observation_steps
                and self.carryover_order == envelope.assignment.window.carryover_order
                and self.estimate.alpha == self.alpha
                and (self.estimate.lb is not None or self.estimate.ub is not None)
                and self.provenance == envelope.provenance
                and self.response_meaning == envelope.response_meaning
                and self.estimate.open_side
                == (
                    "upper"
                    if self.alternative == "greater"
                    else "lower"
                    if self.alternative == "less"
                    else None
                )
                and (
                    (
                        self.standard_error is not None
                        and self.standard_error_unavailable_reason is None
                        and replicates >= 2
                    )
                    or (
                        self.standard_error is None
                        and self.standard_error_unavailable_reason == "insufficient_replicates"
                        and replicates == 1
                    )
                )
            )
            if valid:
                assert isinstance(envelope, UnitCycleVarianceEnvelope)
                cutoff = unit_cycle_envelope_cutoff(
                    envelope, n=self.n_units, alpha=self.alpha, alternative=self.alternative
                )
                # Outward-rounded display bounds need not exclude a significant null.
                valid = (
                    self.effective_alpha == cutoff.effective_alpha
                    and self.refusal_probability_upper == cutoff.refusal_probability_upper
                    and self.residual_cutoff == cutoff.value
                )
            if not valid:
                refuse(
                    _REFUSALS["estimation.contrast.reference_metadata"],
                    randomization_law=self.randomization_law,
                    method=self.method,
                    reference=self.reference,
                    dof=self.dof,
                    independent_replicates=replicates,
                )
            return self
        expected = (
            ("switchback_unit_t_approximation", "unit_t_approximation")
            if self.independence_grain == "unit_cycle"
            else ("switchback_block_t", "block_t")
        )
        if (
            (self.method, self.reference) != expected
            or replicates < 2
            or self.dof != replicates - 1
            or self.standard_error is None
            or self.dof_unavailable_reason is not None
            or not self._valid_t_residual()
            or (self.residual_p_value is not None and self.randomization_law != "fixed_count")
            or any(
                value is not None
                for value in (
                    self.standard_error_unavailable_reason,
                    self.effective_alpha,
                    self.refusal_probability_upper,
                    self.residual_cutoff,
                    self.provenance,
                    self.response_meaning,
                )
            )
            or (
                self.independence_grain == "unit_cycle"
                and not isinstance(self.reference_spec, UnitCycleTApproximation)
            )
            or (self.independence_grain == "shared_block" and self.reference_spec is not None)
        ):
            refuse(
                _REFUSALS["estimation.contrast.reference_metadata"],
                randomization_law=self.randomization_law,
                method=self.method,
                reference=self.reference,
                dof=self.dof,
                independent_replicates=replicates,
            )
        return self

    def __repr__(self) -> str:
        """Show the additive contrast identity and its interval, not its full evidence model."""
        estimate = self.estimate
        if estimate is None:
            interval = "unavailable"
        elif estimate.open_side == "lower" and estimate.ub is not None:
            interval = f"(-inf, {estimate.ub!r})"
        elif estimate.open_side == "upper" and estimate.lb is not None:
            interval = f"({estimate.lb!r}, +inf)"
        else:
            interval = f"({estimate.lb!r}, {estimate.ub!r})"
        return (
            f"ContrastResult(metric={self.metric!r}, control={self.control_group!r}, "
            f"treatment={self.treatment_group!r}, method={self.method!r}, "
            f"role={self.role!r}, estimand={self.estimand!r}, "
            f"population={self.analysis_population!r}, inference={self.inference!r}, "
            f"scale='absolute', estimate={None if estimate is None else estimate.value!r}, interval={interval}, "
            f"alternative={self.alternative!r}, null_abs={self.null_abs!r}, "
            f"randomization_law={self.randomization_law!r}, "
            f"independence_grain={self.independence_grain!r}, "
            f"carryover_order={self.carryover_order}, retained_steps={self.retained_steps}, "
            f"reference={self.reference!r}, decision_scope={self.decision_scope_complete!r})"
        )

    __str__ = __repr__

    def __repr_args__(self) -> list[tuple[str, object]]:
        """Give Rich a compact additive contrast summary."""
        return [
            ("metric", self.metric),
            ("control", self.control_group),
            ("treatment", self.treatment_group),
            ("population", self.analysis_population),
            ("scale", "absolute"),
            ("summary", self.__repr__()),
        ]


CONTRAST_FRAME_COLUMNS = (
    "metric",
    "control_group",
    "treatment_group",
    "method",
    "method_role",
    "role",
    "estimand",
    "aggregation",
    "probability_ct",
    "randomization_law",
    "independence_grain",
    "carryover_order",
    "observation_steps",
    "retained_steps",
    "identifying_assumption",
    "estimate",
    "lb",
    "ub",
    "standard_error",
    "alternative",
    "preferred_direction",
    "null_abs",
    "alpha",
    "n_units",
    "n_cycles",
    "n_blocks",
    "ct_cycles",
    "tc_cycles",
    "dof",
    "assignment",
    "inference",
    "reference",
    "open_side",
    "dof_unavailable_reason",
    "standard_error_unavailable_reason",
    "mean_slope",
    "minimum_cycles_per_unit",
    "maximum_cycles_per_unit",
    "washout_steps",
    "effective_alpha",
    "refusal_probability_upper",
    "residual_cutoff",
    "residual_p_value",
    "residual_side",
    "reference_spec",
    "provenance",
    "response_meaning",
    "value_scale",
    "analysis_population",
    "source_snapshot_id",
    "metric_type",
    "decision_scope_complete",
    "decision_scope_reason_code",
    "decision_scope_reason_context",
)


class ContrastResults(list[ContrastResult]):
    """Typed contrast results with immutable scope metadata and native frame output."""

    _model = ContrastResult

    def __init__(
        self,
        rows=(),
        *,
        metadata: ReadoutMetadata | None = None,
        source=None,
        sequential_snapshot=None,
        default_backend: Backend | None = None,
    ):
        super().__init__(rows)
        self._metadata = metadata
        self.source = source
        self.sequential_snapshot = sequential_snapshot
        self._default_backend = default_backend
        validate_collection(self, metadata)

    @property
    def metadata(self) -> ReadoutMetadata | None:
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> None:
        self._mutation("metadata_set")

    @metadata.deleter
    def metadata(self) -> None:
        self._mutation("metadata_delete")

    def __reduce_ex__(self, protocol):
        return _restore_collection, (
            type(self),
            tuple(self),
            self.metadata,
            self.source,
            self.sequential_snapshot,
            self._default_backend,
        )

    def as_cells(self, *, created_at=None):
        from datetime import UTC, datetime
        from importlib import import_module

        contracts = import_module("increment.kernel.contracts")
        # Upward: as_cells builds ResultCells, which restore these rows for portable reads.
        ResultCell = import_module("increment.result_envelope").ResultCell

        def cell_identity(row):
            identity = contracts.CellIdentity.from_row(row)
            return identity.model_copy(
                update={"key": identity.key.model_copy(update={"value_scale": "absolute"})}
            )

        stamp = created_at or datetime.now(UTC)
        return [
            ResultCell.from_report_payload(
                row_kind=row.method,
                identity=cell_identity(row),
                label=row.method,
                row=row,
                source_snapshot_id=row.source_snapshot_id,
                created_at=stamp,
            )
            for row in self
        ]

    def model_dump_json(self):
        return dump_collection(self)

    @overload
    def __getitem__(self, key: SupportsIndex) -> ContrastResult: ...

    @overload
    def __getitem__(self, key: slice) -> ContrastResults: ...

    def __getitem__(self, key: SupportsIndex | slice) -> ContrastResult | ContrastResults:
        if isinstance(key, slice):
            rows = super().__getitem__(key)
            return type(self)(
                rows,
                metadata=partial_metadata(self.metadata, rows, "slice"),
                source=self.source,
                sequential_snapshot=self.sequential_snapshot,
                default_backend=self._default_backend,
            )
        return super().__getitem__(index(key))

    def filter(self, predicate):
        rows = [row for row in self if predicate(row)]
        return type(self)(
            rows,
            metadata=partial_metadata(self.metadata, rows, "filter"),
            source=self.source,
            sequential_snapshot=self.sequential_snapshot,
            default_backend=self._default_backend,
        )

    def concat(self, other):
        return concat_collection(self, other)

    def __repr__(self) -> str:
        from importlib import import_module

        # Upward: __repr__ is the readout interpreter's rendering of these rows and their scope.
        interpretation_api = import_module("increment._interpretation")
        if self.metadata is None:
            return super().__repr__()
        return interpretation_api.render_readout_text(
            interpretation_api.interpret_readout(self, self.metadata)
        )

    __str__ = __repr__

    def __add__(self, other):
        return self.concat(other)

    def _mutation(self, operation):
        refuse_readout(
            "readout.collection.mutation_unsupported",
            operation=operation,
            model=type(self).__name__,
        )

    def append(self, value):
        self._mutation("append")

    def extend(self, values):
        self._mutation("extend")

    def insert(self, index, value):
        self._mutation("insert")

    def pop(self, index=-1):
        self._mutation("pop")

    def remove(self, value):
        self._mutation("remove")

    def clear(self):
        self._mutation("clear")

    def reverse(self):
        self._mutation("reverse")

    def sort(self, *args, **kwargs):
        self._mutation("sort")

    def __setitem__(self, key, value):
        self._mutation("item_set")

    def __delitem__(self, key):
        self._mutation("item_delete")

    def __iadd__(self, value):
        self._mutation("iadd")

    def __imul__(self, value):
        self._mutation("imul")

    def _frame_data(self) -> dict[str, list[object]]:
        columns = list(CONTRAST_FRAME_COLUMNS)
        fixed_count = any(row.randomization_law == "fixed_count" for row in self)
        if fixed_count:
            columns.extend(_FixedCountMetadata.model_fields)
            columns.append("stat_sig")
        if any(row.forecast_column is not None for row in self):
            columns.append("forecast_column")
        if fixed_count or any(row.aggregation == "ratio" or row.forecast_column for row in self):
            columns.append("p_value")
        data: dict[str, list[object]] = {name: [] for name in columns}
        for result in self:
            if "stat_sig" in data:
                data["stat_sig"].append(contrast_stat_sig(result))
            if "p_value" in data:
                data["p_value"].append(contrast_p_value(result))
            for name in ContrastResult.model_fields:
                if name not in data:
                    continue
                if name == "estimate":
                    estimate = result.estimate
                    data["estimate"].append(None if estimate is None else estimate.value)
                    data["lb"].append(None if estimate is None else estimate.lb)
                    data["ub"].append(None if estimate is None else estimate.ub)
                    data["open_side"].append(None if estimate is None else estimate.open_side)
                elif name in {"reference_spec", "provenance"}:
                    value = getattr(result, name)
                    data[name].append(
                        None
                        if value is None
                        else json.dumps(
                            value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                        )
                    )
                elif name == "decision_scope_reason_context":
                    value = getattr(result, name)
                    data[name].append(
                        None
                        if value is None
                        else json.dumps(thaw(value), sort_keys=True, separators=(",", ":"))
                    )
                else:
                    data[name].append(getattr(result, name))
        return data

    def to_frame(self, *, backend: Backend | None = None) -> IntoDataFrame:
        """Return rows as a native frame; pandas is preferred when available."""
        backend = default_frame_backend(backend or self._default_backend)
        data = self._frame_data()
        columns = list(data)

        schema = {
            name: nw.Boolean()
            if name in ("decision_scope_complete", "stat_sig")
            else nw.Float64()
            if name
            in {
                "estimate",
                "p_value",
                "lb",
                "ub",
                "standard_error",
                "probability_ct",
                "treatment_probability",
                "null_abs",
                "alpha",
                "dof",
                "mean_slope",
                "effective_alpha",
                "refusal_probability_upper",
                "residual_cutoff",
                "residual_p_value",
            }
            else nw.Int64()
            if name
            in {
                "n_units",
                "n_cycles",
                "n_blocks",
                "periods_per_cycle",
                "treated_periods",
                "ct_cycles",
                "tc_cycles",
                "carryover_order",
                "observation_steps",
                "retained_steps",
                "washout_steps",
                "minimum_cycles_per_unit",
                "maximum_cycles_per_unit",
            }
            else nw.String()
            for name in columns
        }
        nullable_ints = {
            name: data[name]
            for name in columns
            if name
            in {
                "n_blocks",
                "ct_cycles",
                "tc_cycles",
                "minimum_cycles_per_unit",
                "maximum_cycles_per_unit",
                "periods_per_cycle",
                "treated_periods",
            }
        }
        if backend == "pandas":
            for name in nullable_ints:
                data.pop(name)
                schema.pop(name)
        from importlib import import_module

        # Upward: to_frame appends the readout interpreter's evidence columns for these rows.
        interpretation_api = import_module("increment._interpretation")
        interpret_readout = interpretation_api.interpret_readout
        readout_evidence_columns = interpretation_api.readout_evidence_columns

        interpretation = interpret_readout(self, self.metadata)
        evidence = [
            readout_evidence_columns(interpretation, row, source_snapshot_id=row.source_snapshot_id)
            for row in self
        ]
        string_columns = {
            "source_snapshot_id",
            "analysis_population",
            "integrity_status",
            "integrity_code",
            "integrity_context",
            "roster_source",
            "posterior_reason_code",
            "posterior_reason_context",
            "failure_code",
            "failure_context",
            "failure_message",
            "unavailable_cells",
            "hidden_cells",
            "advisories",
            "multiplicity_status",
            "retained_family",
        }
        boolean_columns = {
            "decision_scope_complete",
            "roster_complete",
            "sampling_available",
            "point_available",
            "posterior_available",
            "view_partial",
            "provider_owned",
        }
        count_columns = {
            "required_cell_count",
            "available_cell_count",
            "source_cell_count",
            "unavailable_cell_count",
            "hidden_cell_count",
        }
        evidence_columns: dict[str, list[object]]
        if evidence:
            evidence_columns = {name: [row[name] for row in evidence] for name in evidence[0]}
        elif interpretation.groups and interpretation.groups[0].source_cells:
            group = interpretation.groups[0]
            empty_evidence = readout_evidence_columns(
                interpretation,
                group.source_cells[0].key,
                source_snapshot_id=group.source_snapshot_id,
            )
            evidence_columns = {name: [] for name in empty_evidence}
        else:
            evidence_columns = {}
        for name, values in evidence_columns.items():
            data[name] = values
            columns.append(name)
            schema[name] = (
                nw.String()
                if name in string_columns
                else nw.Boolean()
                if name in boolean_columns
                else nw.Int64()
                if name in count_columns
                else nw.String()
            )
        frame = nw.from_dict(data, schema=schema, backend=backend).to_native()
        if backend == "pandas":
            import pandas as pd

            for name, values in nullable_ints.items():
                frame.insert(columns.index(name), name, pd.array(values, dtype="Int64"))
            frame["preferred_direction"] = pd.array(data["preferred_direction"], dtype="string")
            if "stat_sig" in data:
                frame["stat_sig"] = pd.array(data["stat_sig"], dtype="boolean")
        frame = (
            nw.from_native(frame, eager_only=True)
            .with_columns(
                nw.lit(None if self.metadata is None else self.metadata.partial).alias(
                    "view_partial"
                )
            )
            .to_native()
        )
        if backend == "pandas":
            import pandas as pd

            for name in boolean_columns:
                if name in evidence_columns:
                    frame[name] = pd.array(evidence_columns[name], dtype="boolean")
            frame["view_partial"] = pd.array(
                [None if self.metadata is None else self.metadata.partial] * len(frame),
                dtype="boolean",
            )
        return frame


__all__ = ["ContrastResult", "ContrastResults"]
