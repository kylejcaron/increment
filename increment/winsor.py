"""Portable construction state for pooled upper-quantile mean inference."""

from __future__ import annotations

import math
import statistics
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._winsor_errors import winsor_refuse as winsor_refuse
from increment.errors import CodedModel

OutcomeStage = Literal["transformed", "raw"]


class _Frozen(CodedModel, BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class WinsorInferenceSpec(_Frozen):
    """Prospectively fixed inference law; stream is independent of outcome sampling.

    ``qualification`` is part of the persisted public contract. In particular,
    the positive-log bootstrap is a pointwise, model-conditioned asymptotic
    candidate; it is not a universal finite-sample guarantee.
    """

    method: Literal["positive-log-kernel-bootstrap-t-v1", "joint-rank-projection-v1"] = (
        "positive-log-kernel-bootstrap-t-v1"
    )
    status: Literal["experimental"] = "experimental"
    qualification: (
        Literal["pointwise_asymptotic_model_conditioned_v1", "uniform_support_conditioned_v1"]
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
        expected = (
            "uniform_support_conditioned_v1"
            if self.method == "joint-rank-projection-v1"
            else "pointwise_asymptotic_model_conditioned_v1"
        )
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


class RawArm(_Frozen):
    group_id: str = Field(min_length=1)
    values: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical(self):
        object.__setattr__(self, "values", tuple(sorted(self.values)))
        return self


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

    def arm(self, group_id: str) -> RawArm:
        for arm in self.arms:
            if arm.group_id == group_id:
                return arm
        winsor_refuse("pool_mismatch", f"Arm {group_id!r} absent from raw cutoff pool.")


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
    group_id: str
    log_centers: tuple[float, ...] = Field(min_length=2)
    bandwidth: float = Field(gt=0)


def positive_log_bandwidth(log_centers: tuple[float, ...]) -> float:
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
    def _identity(self):
        if self.spec != self.raw.inference or self.spec.method != self.method:
            winsor_refuse("pool_mismatch", "Bootstrap specification differs from raw state.")
        self.raw.arm(self.control)
        self.raw.arm(self.treatment)
        if (
            self.control == self.treatment
            or tuple((p.group_id, len(p.log_centers)) for p in self.pilots) != self.raw.counts
        ):
            winsor_refuse("pool_mismatch", "Bootstrap pilot differs from the all-arm allocation.")
        for pilot, arm in zip(self.pilots, self.raw.arms, strict=True):
            if pilot.bandwidth != positive_log_bandwidth(pilot.log_centers):
                winsor_refuse(
                    "invalid_state",
                    "Pilot bandwidth differs from the versioned sample log-SD rule.",
                )
            if any(y <= 0 for y in arm.values) or pilot.log_centers != tuple(
                math.log(y) for y in arm.values
            ):
                winsor_refuse("pool_mismatch", "Pilot centers differ from positive raw outcomes.")
        values = sorted(y for arm in self.raw.arms for y in arm.values)
        rank = (len(values) - 1) * self.raw.quantile
        left, right = math.floor(rank), math.ceil(rank)
        weight = Fraction(rank - left)
        cutoff = float((1 - weight) * Fraction(values[left]) + weight * Fraction(values[right]))
        means = {
            g: sum((Fraction(min(y, cutoff)) for y in self.raw.arm(g).values), Fraction())
            / len(self.raw.arm(g).values)
            for g in (self.control, self.treatment)
        }
        mc, mt = means[self.control], means[self.treatment]
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
            self.observed_cutoff != cutoff
            or self.log_relative.point != log_point
            or self.additive.point != float(mt - mc)
        ):
            winsor_refuse(
                "pool_mismatch", "Stored observed statistics differ from the raw type-7 procedure."
            )
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


WinsorReference = Annotated[RankReference | BootstrapReference, Field(discriminator="method")]


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


def bootstrap_point(reference: BootstrapReference) -> float | None:
    try:
        point = math.expm1(reference.log_relative.point)
    except OverflowError:
        return None
    return point if math.isfinite(point) and point > -1 else None


class WinsorConfidenceSet(_Frozen):
    """A confidence region, with an explicit qualified reference contract.

    Bootstrap regions are pointwise/model-conditioned asymptotic candidates;
    rank regions are support-conditioned. Neither label is a universal
    finite-sample promise.
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
        if self.control == self.treatment or self.reference.method != self.raw.inference.method:
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
        else:
            if ref.raw != self.raw or (ref.control, ref.treatment) != (
                self.control,
                self.treatment,
            ):
                winsor_refuse(
                    "pool_mismatch", "Stored roots refer to a different population or contrast."
                )
            if (
                self.cutoff is not None
                or self.relative != bootstrap_root_interval(ref, self.alpha, relative=True)
                or self.additive != bootstrap_root_interval(ref, self.alpha, relative=False)
            ):
                winsor_refuse("invalid_state", "Endpoints contradict stored bootstrap roots.")
            if self.point != bootstrap_point(ref) or self.additive_point != ref.additive.point:
                winsor_refuse("invalid_state", "Points contradict stored bootstrap statistics.")
        return self

    @property
    def method(self) -> str:
        return self.reference.method

    @property
    def qualification(self) -> str:
        """Return the explicit validity scope carried by this reference."""
        if isinstance(self.reference, BootstrapReference):
            return "pointwise_asymptotic_model_conditioned_v1"
        return "uniform_support_conditioned_v1"

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
