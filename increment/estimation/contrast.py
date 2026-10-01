"""Additive switchback contrast evidence and centered reduction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.stats import t as _t

from increment._unit_cycle import (
    _float_outward,
    unit_cycle_admission_outcome,
    unit_cycle_envelope_cutoff,
    unit_cycle_report,
)
from increment._unit_cycle import (
    _refuse as _unit_refuse,
)
from increment.compatibility import Unsupported, refuse_unsupported
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.estimation._tails import student_t_isf, tail_isf
from increment.estimation.arm_contract import (
    CONTRAST_EVIDENCE_CONTRACT,
    ContrastCompatibilityRequest,
)
from increment.estimation.contrast_results import (
    ContrastResult,
    IndependenceGrain,
    RandomizationLaw,
    _validate_assignment_metadata,
    _validate_order_counts,
    _validate_replication_counts,
    _validate_retained_window,
)
from increment.estimation.decision_types import ContrastDecisionProcedure, FixedInference, NoFamily
from increment.estimation.results import Estimate
from increment.semantics.unit_cycle import UnitCycleVarianceEnvelope

if TYPE_CHECKING:
    from increment.decision import DecisionComputation


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.contrast.contrast_partition.contain_least_one": "contrast partition must contain at least one unit",
        "estimation.contrast.contrast_partition.unit_deltas_finite": "contrast partition unit deltas must be finite",
        "estimation.contrast.contrast_partition.cycles_mapping": "contrast partition cycles must be a mapping",
        "estimation.contrast.contrast_partition.cycles_positive_integers": "contrast partition cycles must be positive integers",
        "estimation.contrast.contrast_partition.unit_deltas_cycles": "unit_deltas, cycles_by_unit and ct_counts_by_unit must have identical keys",
        "estimation.contrast.contrast_partition.ct_counts": "CT counts must be a mapping of nonnegative integers bounded by order draws",
        "estimation.contrast.contrast_stats.control_group_treatment": "control_group and treatment_group must be distinct",
        "estimation.contrast.contrast_partition_centered": "contrast partition centered moments must be finite",
        "estimation.contrast.contrast_reduction_least": "contrast reduction requires at least one partition",
        "estimation.contrast.contrast_partition_does": "contrast partition {mismatch} does not match",
        "estimation.contrast.contrast_partitions_disjoint": "contrast partitions must be disjoint; unit overlap detected",
        "estimation.contrast.contrast_reduction_delta_mean_finite": "contrast reduction centered moments must be finite",
        "estimation.contrast.stats_contraststats": "stats must be ContrastStats, got {stats}",
        "estimation.contrast.procedure_contrastdecisionprocedure": "procedure must be ContrastDecisionProcedure, got {procedure}",
        "estimation.contrast.contrast_stats_metric": "contrast stats metric {stats!r} does not match procedure metric {procedure!r}",
        "estimation.contrast.contrast_stats_reference": "contrast stats reference, residual, and M2 must be finite",
        "estimation.contrast.contrast_stats_m2": "contrast stats M2 must be nonnegative",
        "estimation.contrast.switchback_unit_inference": "switchback unit-t inference requires at least two units",
        "estimation.contrast.switchback_block_inference": "switchback block-t inference requires at least two independent blocks",
        "estimation.contrast.shared_roster_size_mismatch": "shared-block partitions must share one fixed roster size across every block; found {roster_sizes!r}",
        "estimation.contrast.contrast_stats_n": "contrast stats n_cycles must be positive",
        "estimation.contrast.contrast_stats_reconstructed": "contrast stats reconstructed mean must be finite",
        "estimation.contrast.contrast_stats_standard": "contrast stats standard error must be finite",
        "estimation.contrast.procedure_alpha": "procedure alpha must be in (0, 1), got {procedure}",
        "estimation.contrast.one_sided_alpha": "one-sided alpha={procedure} doubles to alpha_eff={alpha_eff} >= 1: no displayable two-sided interval exists at that level",
        "estimation.contrast.contrast_interval_half": "contrast interval half-width must be finite",
    },
)
_raise = raiser(_REFUSALS)


def contrast_runtime_support(request: ContrastCompatibilityRequest):
    """Evaluate contrast compatibility before evidence reduction."""
    return CONTRAST_EVIDENCE_CONTRACT.runtime_support(request)


def _exact_sum(ratios: Iterable[tuple[int, int]]) -> Fraction:
    """Sum integer ratios exactly, reducing once at the end.

    Denominators accumulate through their least common multiple, so the result
    is the same rational as adding one Fraction per term.
    """
    numerator, denominator = 0, 1
    for term, scale in ratios:
        if denominator % scale == 0:
            numerator += term * (denominator // scale)
        elif scale % denominator == 0:
            numerator = numerator * (scale // denominator) + term
            denominator = scale
        else:
            common = math.gcd(denominator, scale)
            numerator = numerator * (scale // common) + term * (denominator // common)
            denominator *= scale // common
    return Fraction(numerator, denominator)


def _canonical_exact_ratio(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        _unit_refuse("unit_cycle.sufficient_state", reason="incomplete_unit_metadata")
    numerator, denominator = value
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        _unit_refuse("unit_cycle.sufficient_state", reason="incomplete_unit_metadata")
    common = math.gcd(numerator, denominator)
    return numerator // common, denominator // common


class ContrastPartition(CodedModel, BaseModel):
    """Float moments plus optional exact raw-row HT unit averages.

    Exact ratios use reduced integer pairs; mappings serialize in key order.
    Exact unit deltas and slopes require each other and realized order counts.
    Order counts also describe shared blocks without unit-only metadata.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", validate_default=True
    )

    metric: str = Field(min_length=1)
    aggregation: Literal["sum", "any"]
    probability_ct: float = Field(gt=0.0, lt=1.0)
    randomization_law: RandomizationLaw
    independence_grain: IndependenceGrain
    washout_steps: int = Field(default=0, ge=0, strict=True)
    carryover_order: int = Field(ge=0, strict=True)
    observation_steps: int = Field(ge=1, strict=True)
    retained_steps: int = Field(ge=1, strict=True)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    unit_deltas: Mapping[str, float]
    # Reduced (numerator, positive denominator) pairs from raw-row HT averages.
    exact_unit_deltas: Mapping[str, tuple[int, int]] = Field(default_factory=dict)
    cycles_by_unit: Mapping[str, int]
    unit_slopes: Mapping[str, float] = Field(default_factory=dict)
    ct_counts_by_unit: Mapping[str, int] = Field(default_factory=dict)

    @field_validator("unit_slopes")
    @classmethod
    def _slopes(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        if any(not math.isfinite(v) or v <= 0 for v in value.values()):
            _unit_refuse("unit_cycle.sufficient_state", reason="invalid_unit_slopes")
        return MappingProxyType(dict(value))

    @field_validator("exact_unit_deltas", mode="before")
    @classmethod
    def _exact_deltas(cls, value: object) -> dict[str, tuple[int, int]]:
        if not isinstance(value, Mapping):
            _unit_refuse("unit_cycle.sufficient_state", reason="unit_metadata_keys_or_counts")
        result = {}
        for key, ratio in value.items():
            if not isinstance(key, str):
                _unit_refuse("unit_cycle.sufficient_state", reason="unit_metadata_keys_or_counts")
            result[key] = _canonical_exact_ratio(ratio)
        return result

    @field_validator("exact_unit_deltas")
    @classmethod
    def _freeze_exact_deltas(
        cls, value: Mapping[str, tuple[int, int]]
    ) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType(dict(value))

    @field_serializer(
        "unit_deltas", "exact_unit_deltas", "cycles_by_unit", "unit_slopes", "ct_counts_by_unit"
    )
    def _serialize_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(sorted(value.items()))

    @field_validator("unit_deltas")
    @classmethod
    def _finite_deltas(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        copied = dict(value)
        if not copied:
            _raise("estimation.contrast.contrast_partition.contain_least_one")
        if any(not math.isfinite(delta) for delta in copied.values()):
            _raise("estimation.contrast.contrast_partition.unit_deltas_finite")
        return MappingProxyType(copied)

    @field_validator("cycles_by_unit", mode="before")
    @classmethod
    def _positive_cycles(cls, value: object) -> Mapping[str, int]:
        if not isinstance(value, Mapping):
            _raise("estimation.contrast.contrast_partition.cycles_mapping")
        copied = dict(value)
        if not copied:
            _raise("estimation.contrast.contrast_partition.contain_least_one")
        if any(
            isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1
            for cycles in copied.values()
        ):
            _raise("estimation.contrast.contrast_partition.cycles_positive_integers")
        return copied

    @field_validator("ct_counts_by_unit", mode="before")
    @classmethod
    def _nonnegative_ct_counts(cls, value: object) -> Mapping[str, int]:
        if not isinstance(value, Mapping):
            _raise("estimation.contrast.contrast_partition.ct_counts")
        copied = dict(value)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in copied.values()
        ):
            _raise("estimation.contrast.contrast_partition.ct_counts")
        return copied

    @field_validator("cycles_by_unit", "ct_counts_by_unit")
    @classmethod
    def _freeze_cycles(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def _matching_units(self) -> ContrastPartition:
        _validate_retained_window(self.observation_steps, self.retained_steps, self.carryover_order)
        _validate_assignment_metadata(self.randomization_law, self.independence_grain)
        if self.unit_deltas.keys() != self.cycles_by_unit.keys():
            _raise("estimation.contrast.contrast_partition.unit_deltas_cycles")
        shared = self.randomization_law == "shared_schedule"
        if (shared or self.ct_counts_by_unit) and (
            self.unit_deltas.keys() != self.ct_counts_by_unit.keys()
        ):
            _raise("estimation.contrast.contrast_partition.unit_deltas_cycles")
        if any(
            count > (1 if shared else self.cycles_by_unit[unit])
            for unit, count in self.ct_counts_by_unit.items()
        ):
            _raise("estimation.contrast.contrast_partition.ct_counts")
        if self.unit_slopes or self.exact_unit_deltas:
            if shared:
                _unit_refuse("unit_cycle.sufficient_state", reason="unit_metadata_on_shared_blocks")
            if not (
                self.unit_deltas.keys()
                == self.unit_slopes.keys()
                == self.exact_unit_deltas.keys()
                == self.ct_counts_by_unit.keys()
            ):
                _unit_refuse("unit_cycle.sufficient_state", reason="unit_metadata_keys_or_counts")
        if self.control_group == self.treatment_group:
            _raise("estimation.contrast.contrast_stats.control_group_treatment")
        return self


class ContrastStats(CodedModel, BaseModel):
    """Centered moments over the declared independent replicate grain.

    Unit-cycle envelope evidence additionally requires ``exact_delta_total``,
    the reduced integer numerator/denominator of the sum of raw-row HT unit
    averages. The pair serializes as a JSON array, or null when unavailable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")
    metric: str = Field(min_length=1)
    aggregation: Literal["sum", "any"]
    probability_ct: float = Field(gt=0.0, lt=1.0)
    randomization_law: RandomizationLaw
    independence_grain: IndependenceGrain
    washout_steps: int = Field(default=0, ge=0, strict=True)
    carryover_order: int = Field(ge=0, strict=True)
    observation_steps: int = Field(ge=1, strict=True)
    retained_steps: int = Field(ge=1, strict=True)
    identifying_assumption: Literal["no_residual_carryover_after_discarded_steps"] = (
        "no_residual_carryover_after_discarded_steps"
    )
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    n_units: int = Field(ge=1, strict=True)
    n_cycles: int = Field(ge=1, strict=True)
    n_blocks: int | None = Field(default=None, ge=1, strict=True)
    reference_delta: float = Field(allow_inf_nan=False)
    mean_residual: float = Field(allow_inf_nan=False)
    m2_delta: float = Field(ge=0.0, allow_inf_nan=False)
    mean_slope: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    ct_cycles: int | None = Field(default=None, ge=0, strict=True)
    tc_cycles: int | None = Field(default=None, ge=0, strict=True)
    minimum_cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    maximum_cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    # Exact sum of unit HT cycle averages; divide by n_units to recover mean A.
    exact_delta_total: tuple[int, int] | None = None

    @field_validator("exact_delta_total", mode="before")
    @classmethod
    def _exact_total(cls, value: object) -> tuple[int, int] | None:
        return None if value is None else _canonical_exact_ratio(value)

    @model_validator(mode="after")
    def _distinct_groups(self) -> ContrastStats:
        _validate_retained_window(self.observation_steps, self.retained_steps, self.carryover_order)
        _validate_assignment_metadata(self.randomization_law, self.independence_grain)
        _validate_replication_counts(
            self.randomization_law, self.n_units, self.n_cycles, self.n_blocks
        )
        _validate_order_counts(
            self.randomization_law, self.n_cycles, self.n_blocks, self.ct_cycles, self.tc_cycles
        )
        if self.control_group == self.treatment_group:
            _raise("estimation.contrast.contrast_stats.control_group_treatment")
        metadata = (
            self.mean_slope,
            self.minimum_cycles_per_unit,
            self.maximum_cycles_per_unit,
        )
        if any(v is not None for v in (*metadata, self.exact_delta_total)):
            if (
                self.randomization_law != "independent_bernoulli_order"
                or any(v is None for v in metadata)
                or self.ct_cycles is None
                or self.tc_cycles is None
            ):
                _unit_refuse("unit_cycle.sufficient_state", reason="incomplete_unit_metadata")
            assert self.ct_cycles is not None and self.tc_cycles is not None
            assert self.minimum_cycles_per_unit is not None
            assert self.maximum_cycles_per_unit is not None
            if (
                self.ct_cycles + self.tc_cycles != self.n_cycles
                or self.minimum_cycles_per_unit > self.maximum_cycles_per_unit
                or not self.n_units * self.minimum_cycles_per_unit
                <= self.n_cycles
                <= self.n_units * self.maximum_cycles_per_unit
            ):
                _unit_refuse("unit_cycle.sufficient_state", reason="inconsistent_unit_counts")
        return self


def _unit_centered_moments(values: Any) -> tuple[Any, Any, Any]:
    """Centered unit moments along the final axis, including experiment batches.

    Extended accumulation limits rounding in ordinary batches. Scalar runtime
    inputs retain fsum, including cancellation beyond extended precision.
    """
    array = np.asarray(values, dtype=float)
    reference = array[..., 0]
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = array - reference[..., None]
    if array.ndim == 1:
        mean = math.fsum(residuals) / len(array)
        m2 = math.fsum((float(value) - mean) ** 2 for value in residuals)
        return float(reference), mean, m2
    mean = np.sum(residuals, axis=-1, dtype=np.longdouble).astype(float) / array.shape[-1]
    squares = (residuals - mean[..., None]) ** 2
    m2 = np.sum(squares, axis=-1, dtype=np.longdouble).astype(float)
    return reference, mean, m2


def _partition_moments(part: ContrastPartition) -> tuple[int, int, float, float, float]:
    units = sorted(part.unit_deltas)
    n_units = len(units)
    try:
        if part.randomization_law == "independent_bernoulli_order":
            reference, mean_residual, m2_delta = _unit_centered_moments(
                [part.unit_deltas[unit] for unit in units]
            )
        else:
            reference = part.unit_deltas[units[0]]
            residuals = [part.unit_deltas[unit] - reference for unit in units]
            mean_residual = math.fsum(residuals) / n_units
            m2_delta = math.fsum((residual - mean_residual) ** 2 for residual in residuals)
    except (OverflowError, ValueError):
        _raise("estimation.contrast.contrast_partition_centered")
    n_cycles = sum(part.cycles_by_unit.values())
    values = (reference, mean_residual, m2_delta)
    if not all(math.isfinite(value) for value in values):
        _raise("estimation.contrast.contrast_partition_centered")
    return n_units, n_cycles, reference, mean_residual, m2_delta


def reduce_contrast_partitions(parts: Sequence[ContrastPartition]) -> ContrastStats:
    """Merge disjoint unit partitions with a shifted Chan reduction."""
    if not parts:
        _raise("estimation.contrast.contrast_reduction_least")

    parts = tuple(ContrastPartition.model_validate(part) for part in parts)
    first = parts[0]
    identity = (
        first.metric,
        first.aggregation,
        first.probability_ct,
        first.randomization_law,
        first.independence_grain,
        first.carryover_order,
        first.control_group,
        first.treatment_group,
        first.observation_steps,
        first.retained_steps,
        first.washout_steps,
    )
    shared = identity[3] == "shared_schedule"
    seen_units: set[str] = set()
    roster_sizes: set[int] = set()
    normalized: list[tuple[ContrastPartition, tuple[int, int, float, float, float]]] = []
    for part in parts:
        current = (
            part.metric,
            part.aggregation,
            part.probability_ct,
            part.randomization_law,
            part.independence_grain,
            part.carryover_order,
            part.control_group,
            part.treatment_group,
            part.observation_steps,
            part.retained_steps,
            part.washout_steps,
        )
        if current != identity:
            labels = (
                "metric",
                "aggregation",
                "probability_ct",
                "randomization_law",
                "independence_grain",
                "carryover_order",
                "control_group",
                "treatment_group",
                "observation_steps",
                "retained_steps",
                "washout_steps",
            )
            mismatch = next(
                label
                for label, expected, actual in zip(labels, identity, current, strict=True)
                if expected != actual
            )
            _raise("estimation.contrast.contrast_partition_does", mismatch=mismatch)
        units = set(part.unit_deltas)
        if seen_units.intersection(units):
            _raise("estimation.contrast.contrast_partitions_disjoint")
        seen_units.update(units)
        if shared:
            roster_sizes.update(part.cycles_by_unit.values())
        normalized.append((part, _partition_moments(part)))

    # A shared schedule is one fixed, complete roster averaged before
    # reduction (``_build_shared_stats``): every block must record that same
    # roster size, or the recovered ``n_units`` below would be invented.
    if shared and len(roster_sizes) > 1:
        _raise("estimation.contrast.shared_roster_size_mismatch", roster_sizes=sorted(roster_sizes))

    normalized.sort(key=lambda item: min(item[0].unit_deltas))
    n_a, cycles_a, ref_a, mean_a, m2_a = normalized[0][1]
    for _, state in normalized[1:]:
        n_b, cycles_b, ref_b, mean_b, m2_b = state
        n_total = n_a + n_b
        delta = math.fsum((ref_b, -ref_a, mean_b, -mean_a))
        cross_scale = math.sqrt(n_a) * math.sqrt(n_b / n_total)
        scaled_delta = delta * cross_scale
        cross = scaled_delta * scaled_delta
        merged_mean = math.fsum((mean_a, delta * (n_b / n_total)))
        if not all(math.isfinite(value) for value in (delta, scaled_delta, cross, merged_mean)):
            _raise("estimation.contrast.contrast_reduction_delta_mean_finite")
        merged_m2 = math.fsum((m2_a, m2_b, cross))
        if not math.isfinite(merged_m2):
            _raise("estimation.contrast.contrast_reduction_delta_mean_finite")
        n_a = n_total
        cycles_a += cycles_b
        mean_a = merged_mean
        m2_a = merged_m2

    # For ``independent_bernoulli_order``, each merged key is one unit, and
    # ``n_a`` is directly the unit count. For ``shared_schedule``, each
    # merged key is one block instead, and the uniform roster size validated
    # above (never invented via division) is the roster's ``n_units``.
    n_units = roster_sizes.pop() if shared else n_a
    n_blocks = n_a if shared else None
    ct_cycles = (
        sum(sum(part.ct_counts_by_unit.values()) for part in parts)
        if all(part.ct_counts_by_unit for part in parts)
        else None
    )
    tc_cycles = None if ct_cycles is None else (n_a if shared else cycles_a) - ct_cycles
    missing_metadata = shared or any(not part.exact_unit_deltas for part in parts)
    # Missing historical state stays unavailable; never reconstruct it from floats.
    exact_total = (
        None
        if missing_metadata
        else _exact_sum(ratio for part in parts for ratio in part.exact_unit_deltas.values())
    )
    return ContrastStats(
        metric=identity[0],
        aggregation=identity[1],
        probability_ct=identity[2],
        randomization_law=identity[3],
        independence_grain=identity[4],
        carryover_order=identity[5],
        control_group=identity[6],
        treatment_group=identity[7],
        observation_steps=identity[8],
        retained_steps=identity[9],
        washout_steps=identity[10],
        mean_slope=None
        if missing_metadata
        else float(
            _exact_sum(
                slope.as_integer_ratio() for part in parts for slope in part.unit_slopes.values()
            )
            / n_units
        ),
        exact_delta_total=None
        if exact_total is None
        else (exact_total.numerator, exact_total.denominator),
        minimum_cycles_per_unit=None
        if missing_metadata
        else min(c for part in parts for c in part.cycles_by_unit.values()),
        maximum_cycles_per_unit=None
        if missing_metadata
        else max(c for part in parts for c in part.cycles_by_unit.values()),
        n_units=n_units,
        n_cycles=cycles_a,
        n_blocks=n_blocks,
        ct_cycles=ct_cycles,
        tc_cycles=tc_cycles,
        reference_delta=ref_a,
        mean_residual=mean_a,
        m2_delta=m2_a,
    )


def _unit_t_reference(
    point: Any,
    standard_error: Any,
    dof: float,
    *,
    alpha: float,
    alternative: str,
    null_abs: float,
) -> tuple[Any, Any]:
    """Runtime unit-t arithmetic, also accepting arrays of independent experiments.

    Callers validate metadata and mark zero-SE evidence unavailable. This kernel
    supplies the existing reference; it does not establish finite-sample coverage.
    """
    tail = alpha / 2.0 if alternative == "two-sided" else alpha
    critical = tail_isf(student_t_isf, tail, dof, what="fixed unit-t critical value")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        half_width = critical * np.asarray(standard_error)
        statistic = (np.asarray(point) - null_abs) / standard_error
    if alternative == "greater":
        p_value = _t.sf(statistic, dof)
    elif alternative == "less":
        p_value = _t.cdf(statistic, dof)
    else:
        p_value = 2.0 * _t.sf(np.abs(statistic), dof)
    return half_width, p_value


def _envelope_result(
    stats: ContrastStats,
    procedure: ContrastDecisionProcedure,
    envelope: UnitCycleVarianceEnvelope,
) -> ContrastResult:
    window = envelope.assignment.window
    if (
        stats.randomization_law != "independent_bernoulli_order"
        or stats.aggregation != "sum"
        or stats.metric != envelope.metric
        or stats.control_group != envelope.control_group
        or stats.treatment_group != envelope.treatment_group
        or stats.probability_ct != envelope.assignment.sequence.probability_ct
        or stats.washout_steps != window.washout_steps
        or stats.observation_steps != window.observation_steps
        or stats.carryover_order != window.carryover_order
    ):
        _unit_refuse("unit_cycle.reference_mismatch", reason="envelope_identity")
    if (
        stats.mean_slope is None
        or stats.ct_cycles is None
        or stats.tc_cycles is None
        or stats.exact_delta_total is None
        or stats.minimum_cycles_per_unit is None
        or stats.maximum_cycles_per_unit is None
    ):
        _unit_refuse("unit_cycle.sufficient_state", reason="missing_slope_or_order_counts")
    if (
        stats.minimum_cycles_per_unit != envelope.cycles_per_unit
        or stats.maximum_cycles_per_unit != envelope.cycles_per_unit
        or stats.n_cycles != stats.n_units * envelope.cycles_per_unit
    ):
        _unit_refuse("unit_cycle.reference_mismatch", reason="cycles_per_unit")
    if not unit_cycle_admission_outcome(stats.n_cycles, stats.probability_ct, stats.ct_cycles):
        _unit_refuse(
            "unit_cycle.implausible_realized_split",
            reason="replayed_split_refused",
            ct_cycles=stats.ct_cycles,
            tc_cycles=stats.tc_cycles,
        )
    cutoff = unit_cycle_envelope_cutoff(
        envelope, n=stats.n_units, alpha=procedure.alpha, alternative=procedure.alternative
    )
    x = Fraction(*stats.exact_delta_total) / stats.n_units
    p = Fraction(stats.probability_ct)
    w_ct, w_tc = 1 / (2 * p), 1 / (2 * (1 - p))
    slope = (stats.ct_cycles * w_ct + stats.tc_cycles * w_tc) / stats.n_cycles
    alternative = procedure.alternative
    point, lower, upper, reported_slope = unit_cycle_report(x, slope, cutoff.value, alternative)
    estimate = Estimate(
        value=point,
        lb=lower,
        ub=upper,
        alpha=procedure.alpha,
        level=math.fsum((1.0, -procedure.alpha)),
        open_side="upper"
        if alternative == "greater"
        else "lower"
        if alternative == "less"
        else None,
    )
    residual = x - Fraction(procedure.null_abs) * slope
    favorable = -residual if alternative == "less" else residual
    v = Fraction(envelope.residual_variance_upper) / stats.n_units
    if residual == 0 or (alternative != "two-sided" and favorable <= 0):
        probability = Fraction(1)
    elif alternative == "two-sided":
        probability = min(Fraction(1), v / residual**2)
    else:
        probability = v / (v + residual**2)
    probability = min(Fraction(1), probability + Fraction(cutoff.refusal_probability_upper))
    # Evidence rounds the residual probability independently of the display bounds.
    p_value = _float_outward(probability, upper=True)
    standard_error = (
        math.sqrt(stats.m2_delta) / math.sqrt(stats.n_units - 1) / math.sqrt(stats.n_units)
        if stats.n_units > 1
        else None
    )
    return ContrastResult(
        metric=stats.metric,
        control_group=stats.control_group,
        treatment_group=stats.treatment_group,
        method="switchback_unit_variance_envelope",
        reference="residual_chebyshev" if alternative == "two-sided" else "residual_cantelli",
        role=procedure.role,
        estimand="mean_unit_retained_window_difference",
        aggregation=stats.aggregation,
        probability_ct=stats.probability_ct,
        randomization_law=stats.randomization_law,
        independence_grain=stats.independence_grain,
        carryover_order=stats.carryover_order,
        washout_steps=stats.washout_steps,
        observation_steps=stats.observation_steps,
        retained_steps=stats.retained_steps,
        identifying_assumption=stats.identifying_assumption,
        estimate=estimate,
        standard_error=standard_error,
        standard_error_unavailable_reason="insufficient_replicates"
        if standard_error is None
        else None,
        alternative=alternative,
        preferred_direction=procedure.preferred_direction,
        null_abs=procedure.null_abs,
        alpha=procedure.alpha,
        n_units=stats.n_units,
        n_cycles=stats.n_cycles,
        n_blocks=None,
        dof=None,
        dof_unavailable_reason="not_applicable",
        mean_slope=reported_slope,
        ct_cycles=stats.ct_cycles,
        tc_cycles=stats.tc_cycles,
        minimum_cycles_per_unit=stats.minimum_cycles_per_unit,
        maximum_cycles_per_unit=stats.maximum_cycles_per_unit,
        effective_alpha=cutoff.effective_alpha,
        refusal_probability_upper=cutoff.refusal_probability_upper,
        residual_cutoff=cutoff.value,
        residual_p_value=p_value,
        reference_spec=envelope,
        provenance=envelope.provenance,
        response_meaning=envelope.response_meaning,
    )


def _estimate_contrast_result(
    stats: ContrastStats, procedure: ContrastDecisionProcedure
) -> ContrastResult:
    """Estimate an HT contrast using its declared unit envelope or t reference.

    ``ContrastStats`` carries shifted deltas over independent units or blocks:
    ``reference_delta`` is one observed replicate value and ``mean_residual`` is
    the mean of all replicate values after subtracting that reference.  Reconstruct
    the t-reference point estimate with ``math.fsum`` so a large offset never
    gets subtracted from an unshifted online mean. Envelope evidence instead
    uses the exact raw-row HT total, rounding its mean once for reporting.
    """
    if not isinstance(stats, ContrastStats):
        _raise("estimation.contrast.stats_contraststats", stats=type(stats).__name__)
    if not isinstance(procedure, ContrastDecisionProcedure):
        _raise(
            "estimation.contrast.procedure_contrastdecisionprocedure",
            procedure=type(procedure).__name__,
        )

    # These contracts are deliberately narrow.  Check the discriminators
    # before revalidation so forged model instances fail with the supported
    # adaptation they attempted, rather than being silently coerced.
    if not isinstance(procedure.family, NoFamily) or procedure.family.kind != "none":
        refuse_unsupported(
            Unsupported("contrast.decision"),
            message=(
                "switchback contrasts do not support multiplicity or family-adjusted inference"
            ),
        )
    if not isinstance(procedure.inference, FixedInference) or procedure.inference.kind != "fixed":
        refuse_unsupported(
            Unsupported("contrast.inference"),
            message="switchback contrasts do not support sequential inference",
        )
    if stats.aggregation not in ("sum", "any"):
        refuse_unsupported(
            Unsupported("contrast.metric"),
            message=(
                "switchback contrasts support only additive sum or conversion any "
                "aggregation; ratio, quantile, and retention adaptations are unsupported"
            ),
        )

    # Both models are frozen, but model_construct/object.__setattr__ can still
    # create invalid instances.  Revalidation is therefore intentional here,
    # not redundant with the constructor.
    stats = ContrastStats.model_validate(stats)
    procedure = ContrastDecisionProcedure.model_validate(procedure)
    shared = stats.randomization_law == "shared_schedule"
    if not shared and procedure.reference is None:
        _unit_refuse(
            "unit_cycle.envelope_required",
            reason="unit_cycle_reference_required",
            metric=stats.metric,
        )
    if shared and procedure.reference is not None:
        _unit_refuse("unit_cycle.reference_mismatch", reason="unit_reference_on_shared_schedule")
    if stats.metric != procedure.metric:
        _raise(
            "estimation.contrast.contrast_stats_metric",
            procedure=procedure.metric,
            stats=stats.metric,
        )

    if isinstance(procedure.reference, UnitCycleVarianceEnvelope):
        return _envelope_result(stats, procedure, procedure.reference)

    return _t_reference_result(stats, procedure, shared=shared)


def _t_reference_result(
    stats: ContrastStats, procedure: ContrastDecisionProcedure, *, shared: bool
) -> ContrastResult:
    reference_delta = stats.reference_delta
    mean_residual = stats.mean_residual
    m2_delta = stats.m2_delta
    if not all(math.isfinite(value) for value in (reference_delta, mean_residual, m2_delta)):
        _raise("estimation.contrast.contrast_stats_reference")
    if m2_delta < 0.0:
        _raise("estimation.contrast.contrast_stats_m2")
    n_units = stats.n_units
    if shared:
        if stats.n_blocks is None or stats.n_blocks < 2:
            _raise("estimation.contrast.switchback_block_inference")
        replicates = stats.n_blocks
    else:
        if n_units < 2:
            _raise("estimation.contrast.switchback_unit_inference")
        replicates = n_units
    if stats.n_cycles < 1:
        _raise("estimation.contrast.contrast_stats_n")

    # This is the only point estimate reconstruction: never recover an
    # unshifted mean by subtracting two large raw totals.
    point = math.fsum((reference_delta, mean_residual))
    if not math.isfinite(point):
        _raise("estimation.contrast.contrast_stats_reconstructed")

    # M2 is the centered sum of squares over independent replicates (units,
    # or blocks for a shared schedule).  The replicate-level sample variance
    # is M2/(n-1), so the SE of its mean is sqrt(M2 / ((n-1)*n)); divide
    # sequentially to avoid integer overflow.
    dof = float(replicates - 1)
    standard_error = math.sqrt(m2_delta) / math.sqrt(dof) / math.sqrt(replicates)
    if not math.isfinite(standard_error):
        _raise("estimation.contrast.contrast_stats_standard")

    alpha_eff = procedure.alpha if procedure.alternative == "two-sided" else 2.0 * procedure.alpha
    if not 0.0 < procedure.alpha < 1.0:
        _raise("estimation.contrast.procedure_alpha", procedure=procedure.alpha)
    if alpha_eff >= 1.0:
        _raise(
            "estimation.contrast.one_sided_alpha",
            alpha_eff=alpha_eff,
            procedure=procedure.alpha,
        )
    # Pass the tail itself, never `1 - alpha_eff/2`: that complement rounds to 1.0 for valid
    # alpha (1e-20 under a family correction), making the quantile inf.
    level = math.fsum((1.0, -alpha_eff))
    if shared:
        critical = tail_isf(
            student_t_isf, alpha_eff / 2.0, dof, what="fixed block-t critical value"
        )
        half_width = critical * standard_error
    else:
        half_width, _ = _unit_t_reference(
            point,
            standard_error,
            dof,
            alpha=procedure.alpha,
            alternative=procedure.alternative,
            null_abs=procedure.null_abs,
        )
        half_width = float(half_width)
    if not math.isfinite(half_width):
        _raise("estimation.contrast.contrast_interval_half")

    # Equal realized replicate contributions (units, or blocks for a shared
    # schedule) do not establish zero population variance: the zero-SE
    # reference carries no interval, matching its unavailable evidence.
    interval_available = standard_error > 0.0
    estimate = Estimate(
        value=point,
        lb=point - half_width if interval_available else None,
        ub=point + half_width if interval_available else None,
        level=level if interval_available else None,
        # level rounds to exactly 1.0 for a small but valid alpha, which Estimate
        # refuses on its own precisely because the level would then misstate the
        # interval; alpha carries the requested rate through instead.
        alpha=alpha_eff if interval_available else None,
    )
    estimand = (
        "retained_window_total_difference"
        if stats.aggregation == "sum"
        else "retained_window_conversion_difference"
    )
    return ContrastResult(
        metric=stats.metric,
        control_group=stats.control_group,
        treatment_group=stats.treatment_group,
        method="switchback_block_t" if shared else "switchback_unit_t_approximation",
        reference="block_t" if shared else "unit_t_approximation",
        role=procedure.role,
        estimand=estimand,
        aggregation=stats.aggregation,
        probability_ct=stats.probability_ct,
        randomization_law=stats.randomization_law,
        independence_grain=stats.independence_grain,
        carryover_order=stats.carryover_order,
        observation_steps=stats.observation_steps,
        retained_steps=stats.retained_steps,
        identifying_assumption=stats.identifying_assumption,
        estimate=estimate,
        standard_error=standard_error,
        alternative=procedure.alternative,
        preferred_direction=procedure.preferred_direction,
        null_abs=procedure.null_abs,
        alpha=procedure.alpha,
        n_units=n_units,
        n_cycles=stats.n_cycles,
        n_blocks=stats.n_blocks,
        ct_cycles=stats.ct_cycles,
        tc_cycles=stats.tc_cycles,
        dof=dof,
        reference_spec=procedure.reference,
        washout_steps=stats.washout_steps,
        mean_slope=stats.mean_slope,
        minimum_cycles_per_unit=stats.minimum_cycles_per_unit,
        maximum_cycles_per_unit=stats.maximum_cycles_per_unit,
    )


def contrast_evidence_available(result: ContrastResult) -> bool:
    """Whether the declared reference supplies typed evidence.

    Envelope evidence does not depend on the descriptive sample standard error.
    """
    return (
        result.residual_p_value is not None
        if result.method == "switchback_unit_variance_envelope"
        else result.standard_error is not None and result.standard_error > 0.0
    )


def estimate_contrast(
    stats: ContrastStats, procedure: ContrastDecisionProcedure
) -> DecisionComputation[ContrastResult]:
    """Estimate one contrast and retain its typed decision evidence."""
    from increment.estimation.decision_types import (
        ContrastHypothesisKey,
        DecisionComputation,
        DecisionFailure,
        PValueEvidence,
    )

    result = _estimate_contrast_result(stats, procedure)
    hypothesis = ContrastHypothesisKey(result.metric, result.control_group, result.treatment_group)

    if not contrast_evidence_available(result):
        failure = DecisionFailure(
            hypothesis,
            "evidence.p_value.unavailable",
            {"metric": result.metric, "reason": "zero_standard_error"},
        )
        return DecisionComputation(results=(result,), evidence={}, failures={hypothesis: failure})
    if result.method == "switchback_unit_variance_envelope":
        assert result.residual_p_value is not None
        evidence = PValueEvidence(
            hypothesis, result.method, result.residual_p_value, result.reference
        )
        return DecisionComputation(results=(result,), evidence={hypothesis: evidence}, failures={})
    assert result.standard_error is not None and result.dof is not None
    statistic = (result.estimate.value - result.null_abs) / result.standard_error
    if result.randomization_law == "independent_bernoulli_order":
        _, p_value = _unit_t_reference(
            result.estimate.value,
            result.standard_error,
            result.dof,
            alpha=result.alpha,
            alternative=result.alternative,
            null_abs=result.null_abs,
        )
        p_value = float(p_value)
    elif result.alternative == "greater":
        p_value = float(_t.sf(statistic, result.dof))
    elif result.alternative == "less":
        p_value = float(_t.cdf(statistic, result.dof))
    else:
        p_value = float(2.0 * _t.sf(abs(statistic), result.dof))
    evidence = PValueEvidence(
        hypothesis,
        result.method,
        p_value,
        result.reference,
    )
    return DecisionComputation(results=(result,), evidence={hypothesis: evidence}, failures={})


compute_contrast = estimate_contrast


__all__ = [
    "ContrastPartition",
    "compute_contrast",
    "contrast_evidence_available",
    "contrast_runtime_support",
    "estimate_contrast",
    "reduce_contrast_partitions",
]
