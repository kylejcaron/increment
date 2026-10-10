"""Portable construction state for pooled upper-quantile mean inference."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Annotated, Literal, NoReturn, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, WrapValidator, model_validator
from pydantic_core.core_schema import ValidatorFunctionWrapHandler

from increment._winsor_errors import winsor_refuse as winsor_refuse
from increment.errors import MODEL_FIELD_REFUSALS, CodedModel, refuse

OutcomeStage = Literal["transformed", "raw"]

WinsorMethod = Literal[
    "pooled-size-route-v1",
    "positive-log-kernel-bootstrap-t-v1",
    "influence-normal-v1",
    "joint-rank-projection-v1",
]
ExecutedWinsorMethod = Literal[
    "positive-log-kernel-bootstrap-t-v1", "influence-normal-v1", "joint-rank-projection-v1"
]
_QUALIFICATIONS = {
    "pooled-size-route-v1": "pointwise_asymptotic_size_routed_v1",
    "positive-log-kernel-bootstrap-t-v1": "pointwise_asymptotic_model_conditioned_v1",
    "influence-normal-v1": "pointwise_asymptotic_influence_v1",
    "joint-rank-projection-v1": "uniform_support_conditioned_v1",
}
# Provisional size route: the analytic interval needs a large pool and a
# well-populated upper tail; smaller requests keep the full-procedure bootstrap.
ROUTE_POOL_MINIMUM = 20_000
ROUTE_TAIL_MINIMUM = 100


def _reuse_finite_float_tuple(value: object, handler: ValidatorFunctionWrapHandler) -> object:
    if isinstance(value, tuple) and len(value) >= 2 and all(type(item) is float for item in value):
        values = np.fromiter(value, dtype=np.float64, count=len(value))
        if np.isfinite(values).all():
            return value
    return handler(value)


_FiniteFloatTuple = Annotated[tuple[float, ...], WrapValidator(_reuse_finite_float_tuple)]


class _Frozen(CodedModel, BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class WinsorInferenceSpec(_Frozen):
    """Prospectively fixed inference law; stream is independent of outcome sampling.

    ``qualification`` is part of the persisted public contract. The default
    size route selects the analytic influence interval for large pools and the
    positive-log bootstrap otherwise; both are pointwise asymptotic candidates,
    not universal finite-sample guarantees. The executed method is recorded on
    the reference and the confidence set.
    """

    method: WinsorMethod = "pooled-size-route-v1"
    status: Literal["experimental"] = "experimental"
    qualification: (
        Literal[
            "pointwise_asymptotic_size_routed_v1",
            "pointwise_asymptotic_model_conditioned_v1",
            "pointwise_asymptotic_influence_v1",
            "uniform_support_conditioned_v1",
        ]
        | None
    ) = None
    replicates: Literal[1999] = 1999
    seed: int = Field(default=1729, ge=0, strict=True)
    stream: int = Field(default=0, ge=0, le=4294967295, strict=True)
    rng: Literal["PCG64DXSM-SeedSequence-v1"] = "PCG64DXSM-SeedSequence-v1"
    bandwidth: Literal["1.06-sample-log-sd-n^-1/5"] = "1.06-sample-log-sd-n^-1/5"
    alternative: Literal["two-sided"] = "two-sided"
    sampling: Literal["independent_iid_fixed_arms"] = "independent_iid_fixed_arms"
    numerical_policy: Literal["all-roots-or-unavailable-order-statistic-v1"] = (
        "all-roots-or-unavailable-order-statistic-v1"
    )

    @model_validator(mode="after")
    def _qualification_matches_method(self):
        expected = _QUALIFICATIONS[self.method]
        if self.qualification is None:
            object.__setattr__(self, "qualification", expected)
        elif self.qualification != expected:
            winsor_refuse(
                "invalid_state",
                "Winsor inference qualification contradicts the selected method.",
            )
        return self


class WinsorSupport(_Frozen):
    """External declarations, never inferred from observed extrema.

    A quantile upper bound restricts the population cutoff, not outcomes.
    Provenance must identify the justification independent of this sample.
    """

    lower: float
    upper: float | None = None
    quantile_upper: float | None = None
    control_mean_lower: float | None = Field(default=None, gt=0)
    provenance: str = Field(min_length=1)
    sampling: Literal["independent_iid_fixed_arms"] = "independent_iid_fixed_arms"

    @model_validator(mode="after")
    def _bounds(self):
        if not self.provenance.strip():
            winsor_refuse(
                "invalid_state", "Support provenance must identify an external justification."
            )
        if self.upper is not None and self.upper < self.lower:
            winsor_refuse("invalid_state", "Outcome upper support is below lower support.")
        if self.quantile_upper is not None and self.quantile_upper < self.lower:
            winsor_refuse("invalid_state", "Quantile upper bound is below outcome support.")
        if (
            self.control_mean_lower is not None
            and self.upper is not None
            and self.control_mean_lower > self.upper
        ):
            winsor_refuse("invalid_state", "Declared mean lower bound exceeds outcome support.")
        if (
            self.control_mean_lower is not None
            and self.quantile_upper is not None
            and self.control_mean_lower > self.quantile_upper
        ):
            winsor_refuse(
                "invalid_state", "Declared winsor mean lower bound exceeds cutoff upper bound."
            )
        return self


_EXPONENT_OFFSET = 1100
_PART_BITS = 18
_PART_MASK = (1 << _PART_BITS) - 1


def _exact_sum(values: np.ndarray) -> Fraction:
    """Exact rational sum of finite nonnegative doubles.

    Each double is ``mantissa * 2**(exponent - 53)`` with an integer mantissa
    below ``2**53``. Mantissas are split into three 18-bit parts and summed per
    exponent in float64, which is exact while every bucket stays below
    ``2**53`` (any pool below ``2**35`` units); the buckets then combine in
    integer arithmetic.
    """
    mantissas, exponents = np.frexp(values)
    integers = np.ldexp(mantissas, 53).astype(np.int64)
    buckets = exponents.astype(np.int64) + _EXPONENT_OFFSET
    size = int(buckets.max()) + 1
    bucket_sums = [
        np.bincount(
            buckets, weights=((integers >> shift) & _PART_MASK).astype(np.float64), minlength=size
        )
        for shift in (2 * _PART_BITS, _PART_BITS, 0)
    ]
    occupied = np.flatnonzero(bucket_sums[0] + bucket_sums[1] + bucket_sums[2])
    if not len(occupied):
        return Fraction()
    lowest = int(occupied[0])
    numerator = 0
    for bucket in occupied.tolist():
        mantissa_sum = (
            (int(bucket_sums[0][bucket]) << (2 * _PART_BITS))
            + (int(bucket_sums[1][bucket]) << _PART_BITS)
            + int(bucket_sums[2][bucket])
        )
        numerator += mantissa_sum << (bucket - lowest)
    scale = 53 + _EXPONENT_OFFSET - lowest
    return Fraction(numerator, 1 << scale) if scale >= 0 else Fraction(numerator << -scale)


def _refuse_outcomes(code: str, value: object, constraint: str = "") -> NoReturn:
    refuse(
        MODEL_FIELD_REFUSALS[code],
        model="RawArm",
        field="values",
        value=value,
        constraint=constraint,
    )


class _ValidatedOutcomes(tuple):
    """A nonempty tuple of finite floats in ascending order.

    The constructor establishes every invariant with the field's own refusal
    codes, so an instance is proof: tuples are immutable, and a revalidation
    of an arm whose outcomes already carry this type needs no scan.
    """

    __slots__ = ()

    def __new__(cls, values: object):
        if type(values) is cls:
            return values
        items = values if isinstance(values, tuple) else tuple(cast("Iterable[object]", values))
        if not items:
            _refuse_outcomes("model.field.length", items, "Tuple should have at least 1 item")
        kinds = set(map(type, items))
        if not kinds <= {float, int}:
            _refuse_outcomes("model.field.type", items, "a tuple of finite numbers")
        floats = cast("tuple[float, ...]", items)
        if kinds != {float}:
            floats = tuple(float(item) for item in floats)
        array = np.fromiter(floats, dtype=np.float64, count=len(floats))
        if not np.isfinite(array).all():
            _refuse_outcomes(
                "model.field.nonfinite", next(item for item in floats if not math.isfinite(item))
            )
        if len(floats) > 1 and not bool(np.all(array[1:] >= array[:-1])):
            floats = tuple(sorted(array.tolist()))
        return super().__new__(cls, floats)


def _validated_outcomes(value: object, handler: ValidatorFunctionWrapHandler) -> object:
    if type(value) is _ValidatedOutcomes:
        return value
    if isinstance(value, tuple) and value and set(map(type, value)) == {float}:
        return _ValidatedOutcomes(value)
    return _ValidatedOutcomes(handler(value))


_OutcomeTuple = Annotated[tuple[float, ...], WrapValidator(_validated_outcomes)]


class RawArm(_Frozen):
    group_id: str = Field(min_length=1)
    values: _OutcomeTuple = Field(min_length=1)


class WinsorRawState(_Frozen):
    """Lossless per-arm multisets; arrays serialize as ordered JSON lists."""

    version: Literal["pooled-winsor-raw-v1"] = "pooled-winsor-raw-v1"
    outcome_stage: Literal["raw"] = "raw"
    metric: str
    study_id: str
    population: Literal["assigned", "triggered"] = "assigned"
    missingness: str = Field(min_length=1)
    unit_grain: Literal["independent_unit"] = "independent_unit"
    quantile: float = Field(gt=0, lt=1)
    sample_quantile: Literal["linear"] = "linear"
    target: Literal["allocation_pooled_population_upper_winsor_mean"] = (
        "allocation_pooled_population_upper_winsor_mean"
    )
    support: WinsorSupport | None = None
    inference: WinsorInferenceSpec = Field(default_factory=WinsorInferenceSpec)
    arms: tuple[RawArm, ...] = Field(min_length=2)
    allocation: tuple[tuple[str, int, int], ...] = ()

    @model_validator(mode="after")
    def _validate_pool(self):
        labels = [a.group_id for a in self.arms]
        if len(set(labels)) != len(labels):
            winsor_refuse("pool_mismatch", "Duplicate cutoff-pool arm.")
        object.__setattr__(self, "arms", tuple(sorted(self.arms, key=lambda a: a.group_id)))
        total = sum(len(a.values) for a in self.arms)
        allocation = tuple((a.group_id, len(a.values), total) for a in self.arms)
        if self.allocation and self.allocation != allocation:
            winsor_refuse(
                "pool_mismatch", "Recorded rational allocations differ from the complete raw pool."
            )
        object.__setattr__(self, "allocation", allocation)
        if self.inference.method == "joint-rank-projection-v1" and self.support is None:
            winsor_refuse("support_required", "Rank inference requires declared lower support.")
        for arm in self.arms:
            if self.support is not None and (
                arm.values[0] < self.support.lower
                or (self.support.upper is not None and arm.values[-1] > self.support.upper)
            ):
                winsor_refuse(
                    "support_violation", f"Raw outcomes in {arm.group_id!r} violate support."
                )
        return self

    @property
    def counts(self) -> tuple[tuple[str, int], ...]:
        return tuple((a.group_id, len(a.values)) for a in self.arms)

    @property
    def weights(self) -> tuple[tuple[str, float], ...]:
        total = sum(n for _, n in self.counts)
        return tuple((g, n / total) for g, n in self.counts)

    @property
    def executed_method(self) -> ExecutedWinsorMethod:
        """Resolve the size route from the pool size and quantile alone.

        The expected count above the pooled cutoff is ``N (1 - q)`` in exact
        rational arithmetic; the route reads no outcome value.
        """
        method = self.inference.method
        if method != "pooled-size-route-v1":
            return method
        total = sum(n for _, n in self.counts)
        expected_above = Fraction(total) * (1 - Fraction(self.quantile))
        if total >= ROUTE_POOL_MINIMUM and expected_above >= ROUTE_TAIL_MINIMUM:
            return "influence-normal-v1"
        return "positive-log-kernel-bootstrap-t-v1"

    def _exact_point_summary(
        self,
    ) -> tuple[float, tuple[tuple[str, Fraction], ...]]:
        """Compute exact observed cutoff and clipped arm means.

        The two type-7 neighbours are order statistics, so a partition finds
        them; each clipped mean is an exact rational sum of binary fractions.
        """
        arrays = tuple(
            np.fromiter(arm.values, dtype=np.float64, count=len(arm.values)) for arm in self.arms
        )
        pooled = np.concatenate(arrays)
        rank = (len(pooled) - 1) * self.quantile
        left, right = math.floor(rank), math.ceil(rank)
        weight = Fraction(rank - left)
        neighbours = np.partition(pooled, (left, right))
        cutoff = float(
            (1 - weight) * Fraction(float(neighbours[left]))
            + weight * Fraction(float(neighbours[right]))
        )
        del pooled, neighbours
        means = tuple(
            (arm.group_id, _exact_sum(np.minimum(values, cutoff)) / len(arm.values))
            for arm, values in zip(self.arms, arrays, strict=True)
        )
        return cutoff, means

    def arm(self, group_id: str) -> RawArm:
        for arm in self.arms:
            if arm.group_id == group_id:
                return arm

        winsor_refuse("pool_mismatch", f"Arm {group_id!r} absent from raw cutoff pool.")


_POINT_SUMMARY_CONTEXT_TOKEN = object()


def _bootstrap_reference_context(
    raw: WinsorRawState,
) -> tuple[
    tuple[float, tuple[tuple[str, Fraction], ...]],
    dict[object, object],
]:
    summary = raw._exact_point_summary()
    return summary, {_POINT_SUMMARY_CONTEXT_TOKEN: summary}


def _reference_context(
    raw: WinsorRawState, validation_context: Mapping[object, object] | None
) -> tuple[
    tuple[float, tuple[tuple[str, Fraction], ...]],
    Mapping[object, object],
]:
    """Reuse a caller's exact point summary, or compute it once."""
    if validation_context is None:
        return _bootstrap_reference_context(raw)
    summary = validation_context[_POINT_SUMMARY_CONTEXT_TOKEN]
    return cast("tuple[float, tuple[tuple[str, Fraction], ...]]", summary), validation_context


class SetEndpoint(_Frozen):
    status: Literal["finite", "unbounded", "undefined"]
    value: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _status(self):
        if (self.status == "finite") != (self.value is not None):
            winsor_refuse("invalid_state", "Endpoint status contradicts its numeric value.")
        if self.status != "finite" and not self.reason:
            winsor_refuse("invalid_state", "Nonfinite endpoints require an explicit reason.")
        return self

    @classmethod
    def from_bound(cls, value: float, reason: str):
        if math.isnan(value):
            return cls(status="undefined", reason=reason)
        if math.isinf(value):
            return cls(status="unbounded", reason=reason)
        return cls(status="finite", value=value)


class SetInterval(_Frozen):
    lower: SetEndpoint
    upper: SetEndpoint

    @model_validator(mode="after")
    def _ordered(self):
        if (
            self.lower.value is not None
            and self.upper.value is not None
            and self.lower.value > self.upper.value
        ):
            winsor_refuse("invalid_state", "Confidence-set endpoints are inverted.")
        return self

    def excludes(self, null: float) -> bool:
        if "undefined" in (self.lower.status, self.upper.status):
            return False
        return (self.lower.value is not None and self.lower.value > null) or (
            self.upper.value is not None and self.upper.value < null
        )


class RankReference(_Frozen):
    method: Literal["joint-rank-projection-v1"] = "joint-rank-projection-v1"
    band_alpha: float = Field(ge=0, lt=1)
    cutoff_alpha: float = Field(ge=0, lt=1)
    band_identity: tuple[str, ...]


class WinsorPermutationTest(_Frozen):
    """Conditional sharp-null test, never a heterogeneous effect interval."""

    method: Literal["fixed-count-permutation-v1"] = "fixed-count-permutation-v1"
    null_kind: Literal["raw_distribution_exchangeability"] = "raw_distribution_exchangeability"
    raw: WinsorRawState
    control: str
    treatment: str
    exact: bool
    assignments: int = Field(gt=0)
    at_least_as_extreme: int = Field(ge=0)
    seed: int | None = None
    stream: int | None = None

    @model_validator(mode="after")
    def _counts(self):
        self.raw.arm(self.control)
        self.raw.arm(self.treatment)
        if self.control == self.treatment or self.at_least_as_extreme > self.assignments:
            winsor_refuse("invalid_state", "Invalid permutation contrast or count.")
        if self.exact:
            expected = math.comb(
                len(self.raw.arm(self.control).values) + len(self.raw.arm(self.treatment).values),
                len(self.raw.arm(self.control).values),
            )
            if self.assignments != expected or self.seed is not None or self.stream is not None:
                winsor_refuse(
                    "invalid_state",
                    "Exact permutation reference requires every fixed-count assignment.",
                )
        elif self.assignments != 1999 or self.seed is None or self.stream is None:
            winsor_refuse(
                "invalid_state",
                "Randomized permutation reference requires its fixed budget and stream.",
            )
        return self

    @property
    def p_value(self) -> float:
        if self.exact:
            return self.at_least_as_extreme / self.assignments
        return (1 + self.at_least_as_extreme) / (1 + self.assignments)


class PositiveLogPilot(_Frozen):
    """Arm-wise hurdle pilot: an atom at zero plus a lognormal kernel mixture.

    ``log_centers`` are the logs of the arm's positive outcomes; ``zero_count``
    is the size of the atom, resampled with the same probability as any
    positive centre. A positive-only arm has ``zero_count == 0``.
    """

    group_id: str
    log_centers: _FiniteFloatTuple = Field(min_length=2)
    bandwidth: float = Field(gt=0)
    zero_count: int = Field(default=0, ge=0, strict=True)


def raw_arm_zero_counts(raw: WinsorRawState) -> tuple[int, ...]:
    """Apply the log-kernel pilot's admissibility rules to every pool arm.

    Outcomes are sorted, so the atom size is a bisection and the positive
    part's exact log variance vanishes precisely when its smallest and largest
    logs coincide. Fresh construction and stored-reference validation share
    these refusals.
    """
    counts = []
    for arm in raw.arms:
        if arm.values[0] < 0:
            winsor_refuse(
                "pilot_negative_outcome",
                "Log-kernel inference requires nonnegative raw outcomes.",
            )
        zero_count = bisect_right(arm.values, 0.0)
        if len(arm.values) - zero_count < 2:
            winsor_refuse(
                "pilot_degenerate",
                "Log-kernel fitting requires at least two positive observations per arm.",
            )
        if math.log(arm.values[zero_count]) == math.log(arm.values[-1]):
            winsor_refuse("pilot_degenerate", "Each arm must have nonzero finite log variance.")
        counts.append(zero_count)
    return tuple(counts)


def refuse_cutoff_in_zero_atom(raw: WinsorRawState, zero_count: int) -> None:
    """Refuse a pooled percentile whose lower type-7 neighbour is a zero.

    The cutoff is then zero (zero share at or above the quantile, a degenerate
    estimand) or an interpolation artefact below every positive outcome, which
    no smooth-cutoff construction describes.
    """
    total = sum(n for _, n in raw.counts)
    if math.floor((total - 1) * raw.quantile) < zero_count:
        winsor_refuse(
            "cutoff_in_zero_atom",
            f"The pooled {raw.quantile:g} cutoff lies in the zero atom ({zero_count} of {total} "
            "outcomes are zero); raise upper_percentile above the zero share or use a fixed "
            "upper_value.",
        )


def _check_raw_applicability(raw: WinsorRawState) -> tuple[int, ...]:
    counts = raw_arm_zero_counts(raw)
    refuse_cutoff_in_zero_atom(raw, sum(counts))
    return counts


def _check_executed_method(reference, executed: str, label: str) -> None:
    spec, raw = reference.spec, reference.raw
    if spec != raw.inference or spec.method not in (executed, "pooled-size-route-v1"):
        winsor_refuse("pool_mismatch", f"{label} specification differs from raw state.")
    if raw.executed_method != executed:
        winsor_refuse("pool_mismatch", f"{label} reference contradicts the resolved size route.")


def _check_observed_points(reference, info: ValidationInfo, label: str) -> None:
    """Stored cutoff and points must equal the exact type-7 raw procedure."""
    context = info.context
    summary = context.get(_POINT_SUMMARY_CONTEXT_TOKEN) if isinstance(context, dict) else None
    if summary is None:
        cutoff, exact_means = reference.raw._exact_point_summary()
    else:
        cutoff, exact_means = summary
    means = dict(exact_means)
    mc, mt = means[reference.control], means[reference.treatment]
    try:
        point = float((mt - mc) / mc)
    except OverflowError:
        point = math.inf
    log_point = (
        math.log1p(point)
        if math.isfinite(point) and point > -1
        else math.log(float(mt)) - math.log(float(mc))
    )
    if (
        reference.observed_cutoff != cutoff
        or reference.log_relative.point != log_point
        or reference.additive.point != float(mt - mc)
    ):
        winsor_refuse(
            "pool_mismatch", f"Stored {label} statistics differ from the raw type-7 procedure."
        )


def positive_log_bandwidth(log_centers: tuple[float, ...] | np.ndarray) -> float:
    return 1.06 * statistics.stdev(log_centers) * len(log_centers) ** -0.2


class BootstrapRoots(_Frozen):
    point: float
    pilot_target: float
    se: float = Field(gt=0)
    roots: tuple[float | None, ...]


class BootstrapReference(_Frozen):
    method: Literal["positive-log-kernel-bootstrap-t-v1"] = "positive-log-kernel-bootstrap-t-v1"
    spec: WinsorInferenceSpec
    raw: WinsorRawState
    control: str
    treatment: str
    pilots: tuple[PositiveLogPilot, ...]
    pilot_cutoff: float = Field(gt=0)
    observed_cutoff: float = Field(gt=0)
    log_relative: BootstrapRoots
    additive: BootstrapRoots
    centering: Literal["pilot_population_pooled_quantile_target"] = (
        "pilot_population_pooled_quantile_target"
    )
    failure_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _identity(self, info: ValidationInfo):
        _check_executed_method(self, self.method, "Bootstrap")
        zero_counts = _check_raw_applicability(self.raw)
        self.raw.arm(self.control)
        self.raw.arm(self.treatment)
        if (
            self.control == self.treatment
            or tuple((p.group_id, p.zero_count + len(p.log_centers)) for p in self.pilots)
            != self.raw.counts
        ):
            winsor_refuse("pool_mismatch", "Bootstrap pilot differs from the all-arm allocation.")
        if tuple(p.zero_count for p in self.pilots) != zero_counts:
            winsor_refuse("pool_mismatch", "Pilot atom differs from the zero raw outcomes.")
        for pilot, arm in zip(self.pilots, self.raw.arms, strict=True):
            if pilot.bandwidth != positive_log_bandwidth(pilot.log_centers):
                winsor_refuse(
                    "invalid_state",
                    "Pilot bandwidth differs from the versioned sample log-SD rule.",
                )
            positives = arm.values[pilot.zero_count :]
            if len(positives) != len(pilot.log_centers) or any(
                center != math.log(y)
                for center, y in zip(pilot.log_centers, positives, strict=True)
            ):
                winsor_refuse("pool_mismatch", "Pilot centers differ from positive raw outcomes.")
        _check_observed_points(self, info, "observed")
        if (
            len(self.log_relative.roots) != self.spec.replicates
            or len(self.additive.roots) != self.spec.replicates
        ):
            winsor_refuse("invalid_state", "Bootstrap reference must retain every scheduled root.")
        failures = tuple(
            i
            for i, (a, b) in enumerate(
                zip(self.log_relative.roots, self.additive.roots, strict=True)
            )
            if a is None or b is None
        )
        if self.failure_indices != failures:
            winsor_refuse("invalid_state", "Bootstrap failure indices contradict stored roots.")
        return self


class InfluenceSeries(_Frozen):
    point: float
    se: float = Field(gt=0)


class InfluenceReference(_Frozen):
    """Analytic influence-function interval with normal quantiles.

    ``scaled_density`` is the pooled outcome density at the observed cutoff
    multiplied by the cutoff: the arm-wise log kernel on positive outcomes,
    weighted by each arm's positive share, in the dimensionless form the
    studentization consumes, so it stays representable at any outcome scale.
    Both standard errors are the centred empirical influence variance over
    every pool arm, including the estimated cutoff, the estimated zero shares
    and their cross term.
    """

    method: Literal["influence-normal-v1"] = "influence-normal-v1"
    spec: WinsorInferenceSpec
    raw: WinsorRawState
    control: str
    treatment: str
    observed_cutoff: float = Field(gt=0)
    scaled_density: float = Field(gt=0)
    log_relative: InfluenceSeries
    additive: InfluenceSeries
    studentization: Literal["empirical_influence_all_pool_arms_v1"] = (
        "empirical_influence_all_pool_arms_v1"
    )

    @model_validator(mode="after")
    def _identity(self, info: ValidationInfo):
        _check_executed_method(self, self.method, "Influence")
        _check_raw_applicability(self.raw)
        self.raw.arm(self.control)
        self.raw.arm(self.treatment)
        if self.control == self.treatment:
            winsor_refuse("pool_mismatch", "Influence reference requires distinct contrast arms.")
        _check_observed_points(self, info, "influence")
        return self


WinsorReference = Annotated[
    RankReference | BootstrapReference | InfluenceReference, Field(discriminator="method")
]


def _bounds_interval(bounds: tuple[float, float], *, relative: bool) -> SetInterval:
    endpoints = []
    for bound in bounds:
        try:
            value = math.expm1(bound) if relative else bound
        except OverflowError:
            value = math.inf
        endpoints.append(
            SetEndpoint(status="finite", value=value)
            if math.isfinite(value) and (not relative or value > -1)
            else SetEndpoint(status="undefined", reason="endpoint_unrepresentable")
        )
    return SetInterval(lower=endpoints[0], upper=endpoints[1])


def bootstrap_root_interval(
    reference: BootstrapReference, alpha: float, *, relative: bool
) -> SetInterval:
    """Equal tails from order statistics, using the upper tail directly."""
    series = reference.log_relative if relative else reference.additive
    if reference.failure_indices:
        endpoint = SetEndpoint(status="undefined", reason="bootstrap_replicate_failure")
        return SetInterval(lower=endpoint, upper=endpoint)
    # A tail rank of zero cannot be resolved by this finite root sample.
    rank = (Fraction(alpha) * (reference.spec.replicates + 1) / 2).__floor__()
    if rank < 1:
        return SetInterval(
            lower=SetEndpoint(status="finite", value=-1)
            if relative
            else SetEndpoint(status="unbounded", reason="bootstrap_tail_unresolved"),
            upper=SetEndpoint(status="unbounded", reason="bootstrap_tail_unresolved"),
        )
    roots = sorted(x for x in series.roots if x is not None)
    bounds = (series.point - roots[-rank] * series.se, series.point - roots[rank - 1] * series.se)
    return _bounds_interval(bounds, relative=relative)


def influence_interval(
    reference: InfluenceReference, alpha: float, *, relative: bool
) -> SetInterval:
    """Symmetric normal interval on the log-ratio or difference scale."""
    from scipy.special import ndtri_exp

    series = reference.log_relative if relative else reference.additive
    # Keep alpha/2 in log space: the smallest positive float underflows when halved.
    z = -float(ndtri_exp(math.log(alpha) - math.log(2)))
    return _bounds_interval(
        (series.point - z * series.se, series.point + z * series.se), relative=relative
    )


def influence_p_value(reference: InfluenceReference, null: float, *, relative: bool) -> float:
    """Two-sided normal p-value from the survival function of the pivot."""
    from scipy.special import ndtr

    if relative and null <= -1:
        return 0.0
    series = reference.log_relative if relative else reference.additive
    pivot = (series.point - (math.log1p(null) if relative else null)) / series.se
    return min(1.0, 2 * float(ndtr(-abs(pivot))))


def bootstrap_point(reference: BootstrapReference | InfluenceReference) -> float | None:
    try:
        point = math.expm1(reference.log_relative.point)
    except OverflowError:
        return None
    return point if math.isfinite(point) and point > -1 else None


class WinsorConfidenceSet(_Frozen):
    """A confidence region, with an explicit qualified reference contract.

    Bootstrap regions are pointwise/model-conditioned asymptotic candidates,
    influence regions are pointwise normal-approximation candidates, and rank
    regions are support-conditioned. ``method`` names the construction that
    ran, which a size-routed request only fixes after the pool is known.
    """

    null_kind: Literal["pooled_winsor_population_effect"] = "pooled_winsor_population_effect"
    raw: WinsorRawState
    control: str
    treatment: str
    alpha: float = Field(gt=0, lt=1)
    reference: WinsorReference
    relative: SetInterval
    additive: SetInterval
    cutoff: SetInterval | None = None
    point: float | None = None
    additive_point: float | None

    @model_validator(mode="after")
    def _contract(self):
        self.raw.arm(self.control)
        self.raw.arm(self.treatment)
        if self.control == self.treatment or self.reference.method != self.raw.executed_method:
            winsor_refuse("pool_mismatch", "Confidence set has inconsistent pool identity.")
        ref = self.reference
        if isinstance(ref, RankReference):
            if len(ref.band_identity) != len(self.raw.arms) or self.cutoff is None:
                winsor_refuse(
                    "pool_mismatch", "Rank reference requires all arm bands and cutoff region."
                )
            if Fraction(ref.band_alpha) + Fraction(ref.cutoff_alpha) > Fraction(self.alpha):
                winsor_refuse("invalid_state", "Confidence-set error allocation exceeds alpha.")
            expected = _rank_region_fields(self.raw, self.control, self.treatment, self.alpha)
            actual = (
                ref,
                self.relative,
                self.additive,
                self.cutoff,
                self.point,
                self.additive_point,
            )
            if actual != expected:
                winsor_refuse("invalid_state", "Rank region contradicts its construction state.")
            return self
        if ref.raw != self.raw or (ref.control, ref.treatment) != (self.control, self.treatment):
            winsor_refuse(
                "pool_mismatch", "Stored reference refers to a different population or contrast."
            )
        if isinstance(ref, BootstrapReference):
            expected_relative = bootstrap_root_interval(ref, self.alpha, relative=True)
            expected_additive = bootstrap_root_interval(ref, self.alpha, relative=False)
        else:
            expected_relative = influence_interval(ref, self.alpha, relative=True)
            expected_additive = influence_interval(ref, self.alpha, relative=False)
        if (
            self.cutoff is not None
            or self.relative != expected_relative
            or self.additive != expected_additive
        ):
            winsor_refuse("invalid_state", "Endpoints contradict the stored reference.")
        if self.point != bootstrap_point(ref) or self.additive_point != ref.additive.point:
            winsor_refuse("invalid_state", "Points contradict the stored reference statistics.")
        return self

    @property
    def method(self) -> str:
        return self.reference.method

    @property
    def qualification(self) -> str:
        """Return the explicit validity scope carried by this reference."""
        return _QUALIFICATIONS[self.reference.method]

    @property
    def lower(self) -> float | None:
        return self.relative.lower.value

    @property
    def upper(self) -> float | None:
        return self.relative.upper.value

    @property
    def level(self) -> float:
        return math.fsum((1.0, -self.alpha))


def _rank_interval(value):
    return SetInterval(
        lower=SetEndpoint.from_bound(value.lower, value.reason),
        upper=SetEndpoint.from_bound(value.upper, value.reason),
    )


def _rank_region_fields(raw: WinsorRawState, control: str, treatment: str, alpha: float):
    """Convert the shared numerical construction to portable, immutable fields."""
    from increment._winsor_rank import _rank_construction

    metadata, relative, additive, cutoff, point, additive_point = _rank_construction(
        raw, control, treatment, alpha
    )
    reference = RankReference(
        band_alpha=metadata[0], cutoff_alpha=metadata[1], band_identity=metadata[2]
    )
    return (
        reference,
        _rank_interval(relative),
        _rank_interval(additive),
        _rank_interval(cutoff),
        point,
        additive_point,
    )
