"""Effect-dependent switchback planning over independent units or shared blocks.

The noncentral-t reference is a planning approximation, not an exact finite
branch randomization law. Summaries already include cycle dependence or the
shared roster reduction; neither cycles nor roster size rescale N or variance.

Only constant additive shifts of the retained-window effect are modeled.
Binary outcomes need a separately derived bounded model and are refused.
The solvers do not infer an outcome transformation from a metric name.
"""

from __future__ import annotations

import math
import struct
import sys
import warnings
from collections.abc import Mapping
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from scipy.optimize import brentq

from increment.compatibility import _conservative_divide
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    _safe_error_value,
    refuse,
)
from increment.estimation.contrast_results import IndependenceGrain, RandomizationLaw
from increment.estimation.decision_types import ContrastDecisionProcedure
from increment.power._noncentral_t import _scalar_power_from_nc
from increment.power._search import (
    _noncentrality,
    bisect_first_true,
    float_from_ordinal,
    float_ordinal,
    require_reason_when_null,
)
from increment.semantics.assignment import SwitchbackAssignment

UnavailableReason = Literal[
    "not_requested",
    "metadata_mismatch",
    "effect_outside_domain",
    "degenerate_zero_variance",
    "non_favorable_effect",
    "unattained",
    "unrepresentable",
    "numerical_resolution",
]

_REFUSALS = {
    f"power.switchback.{name}": RefusalSpec(
        f"power.switchback.{name}",
        InvalidRequestError,
        template="{message}",
        keys=frozenset({"field", "value", "constraint", "route"}),
    )
    for name in (
        "baseline",
        "roster",
        "procedure",
        "n",
        "target_power",
        "delta",
        "result",
        "bounded_model_required",
    )
}


_ROUTES = {
    "baseline": "Correct the switchback baseline declaration.",
    "roster": "Correct the shared roster declaration.",
    "procedure": "Use a procedure compatible with the switchback baseline.",
    "n": "Provide a valid independent unit or block count.",
    "target_power": "Provide a finite target power strictly between zero and one.",
    "delta": "Provide a finite effect value within the declared effect domain.",
    "result": "Construct or serialize a result satisfying its declared invariants.",
    "bounded_model_required": "Use aggregation='sum' with the additive contribution model.",
}


def _invalid(
    name: str,
    message: str,
    *,
    field: str | None = None,
    value: object = None,
    constraint: str | None = None,
    route: str | None = None,
) -> NoReturn:
    refuse(
        _REFUSALS[f"power.switchback.{name}"],
        message=message,
        field=field or name,
        value=(
            tuple(_safe_error_value(item) for item in value)
            if isinstance(value, (tuple, list))
            else _safe_error_value(value)
        ),
        constraint=constraint or message,
        route=route or _ROUTES[name],
    )


class SwitchbackBaseline(CodedModel, BaseModel):
    """Centered (A, G) SD/correlation summary at the independent planning grain.

    ``A`` is the contribution at ``delta_ref``; ``G`` is its additive-shift
    slope, with E[G]=1. ``sd_a**2``, ``sd_g**2``, and ``rho*sd_a*sd_g`` encode
    VarA, VarG, CovAG exactly. PSD, including rho=+/-1, is valid. A zero SD
    requires rho=0 because correlation is then unidentified.

    Independent-order summaries average a fixed ``cycles_per_unit`` within
    each unit. Shared-schedule summaries average the fixed ``shared_roster``
    within each block. These fields are mutually exclusive. Changing assignment,
    window, cycles, roster, or effect model requires newly derived summaries.

    Pilot-estimated moments and reference effects are plug-in estimates.
    Planning conditions on that fitted baseline, not known population moments.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    assignment: SwitchbackAssignment
    metric: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    aggregation: Literal["sum", "any"]
    estimand: Literal["retained_window_total_difference", "retained_window_conversion_difference"]
    identifying_assumption: Literal["no_residual_carryover_after_discarded_steps"] = (
        "no_residual_carryover_after_discarded_steps"
    )
    cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    shared_roster: tuple[str, ...] | None = None
    delta_ref: float = Field(allow_inf_nan=False)
    sd_a: float = Field(ge=0, allow_inf_nan=False)
    sd_g: float = Field(ge=0, allow_inf_nan=False)
    rho: float = Field(ge=-1, le=1, allow_inf_nan=False)
    moment_source: Literal["specified", "pilot_estimated"] = "specified"
    effect_model: Literal["constant_additive"] = "constant_additive"
    effect_bounds: tuple[float, float] | None = None

    @field_validator("assignment", mode="before")
    @classmethod
    def _revalidate_assignment(cls, value: object) -> object:
        # Assignment models do not revalidate instances, so validate their
        # declarations as mappings, including the nested sequence and window.
        if isinstance(value, BaseModel):
            value = dict(value)
        if isinstance(value, Mapping):
            return {key: cls._revalidate_assignment(item) for key, item in value.items()}
        return value

    @model_validator(mode="after")
    def _consistent(self) -> SwitchbackBaseline:
        if self.aggregation == "any":
            _invalid(
                "bounded_model_required",
                "binary outcomes do not satisfy the constant-additive contribution model",
                field="aggregation",
                value=self.aggregation,
                constraint="constant-additive model requires aggregation='sum'",
            )
        if self.control_group == self.treatment_group:
            _invalid(
                "baseline",
                "control_group and treatment_group must be distinct",
                field="control_group/treatment_group",
                value=(self.control_group, self.treatment_group),
                constraint="distinct nonempty arm identifiers",
            )
        expected = (
            "retained_window_total_difference"
            if self.aggregation == "sum"
            else "retained_window_conversion_difference"
        )
        if self.estimand != expected:
            _invalid(
                "baseline",
                "aggregation and estimand must match",
                field="aggregation/estimand",
                value=(self.aggregation, self.estimand),
                constraint=f"estimand={expected!r} for this aggregation",
            )
        if (self.sd_a == 0 or self.sd_g == 0) and self.rho != 0:
            _invalid(
                "baseline",
                "zero SD requires rho=0",
                field="sd_a/sd_g/rho",
                value=(self.sd_a, self.sd_g, self.rho),
                constraint="rho=0 whenever either standard deviation is zero",
            )
        if self.randomization_law == "independent_bernoulli_order":
            if self.cycles_per_unit is None or self.shared_roster is not None:
                _invalid(
                    "baseline",
                    "independent orders require cycles_per_unit only",
                    field="cycles_per_unit/shared_roster",
                    value=(self.cycles_per_unit, self.shared_roster),
                    constraint="independent orders require cycles_per_unit and no shared_roster",
                )
        elif self.cycles_per_unit is not None or self.shared_roster is None:
            _invalid(
                "baseline",
                "shared schedules require shared_roster only",
                field="cycles_per_unit/shared_roster",
                value=(self.cycles_per_unit, self.shared_roster),
                constraint="shared schedules require shared_roster and no cycles_per_unit",
            )
        if self.shared_roster is not None and (
            not self.shared_roster
            or any(not key.strip() for key in self.shared_roster)
            or len(set(self.shared_roster)) != len(self.shared_roster)
        ):
            _invalid(
                "roster",
                "shared_roster must contain distinct nonempty unit identifiers",
                field="shared_roster",
                value=self.shared_roster,
                constraint="nonempty distinct identifiers",
            )
        if self.effect_bounds is not None:
            lo, hi = self.effect_bounds
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                _invalid(
                    "baseline",
                    "effect_bounds must be a finite increasing closed interval",
                    field="effect_bounds",
                    value=self.effect_bounds,
                    constraint="finite increasing closed interval",
                )
            if not lo <= self.delta_ref <= hi:
                _invalid(
                    "baseline",
                    "delta_ref must belong to effect_bounds",
                    field="delta_ref/effect_bounds",
                    value=(self.delta_ref, self.effect_bounds),
                    constraint="delta_ref within effect_bounds",
                )
        return self

    @property
    def randomization_law(self) -> RandomizationLaw:
        return self.assignment.sequence.scheme

    @property
    def independence_grain(self) -> IndependenceGrain:
        """Assignment grain; planning N still counts units for unit_cycle."""
        return self.assignment.sequence.independence_unit

    @property
    def planning_grain(self) -> Literal["unit", "shared_block"]:
        return "unit" if self.randomization_law == "independent_bernoulli_order" else "shared_block"


class SwitchbackPowerResult(CodedModel, BaseModel):
    """Immutable power result with a separate reason for every numeric null.

    ``mde_abs`` is the absolute favorable effect delta, not a standardized or
    relative lift. Achieved-power and required-N leave it null with
    ``mde_unavailable_reason='not_requested'``; there is no default target.
    Full assignment/window, labels, assumptions and decision metadata are
    retained in ``baseline`` and ``procedure`` and in their JSON serialization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    power_kind: Literal["moment_t_approximation"] = "moment_t_approximation"
    baseline: SwitchbackBaseline
    procedure: ContrastDecisionProcedure
    n: int | None = Field(ge=2, strict=True)
    df: int | None = Field(ge=1, strict=True)
    delta: float | None = Field(allow_inf_nan=False)
    standard_error: float | None = Field(gt=0, allow_inf_nan=False)
    power: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    mde_abs: float | None = Field(allow_inf_nan=False)
    admission_probability: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    n_unavailable_reason: UnavailableReason | None = None
    df_unavailable_reason: UnavailableReason | None = None
    delta_unavailable_reason: UnavailableReason | None = None
    standard_error_unavailable_reason: UnavailableReason | None = None
    power_unavailable_reason: UnavailableReason | None = None
    mde_unavailable_reason: UnavailableReason | None = None
    admission_probability_unavailable_reason: UnavailableReason | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SwitchbackPowerResult:
        for field in (
            "n",
            "df",
            "delta",
            "standard_error",
            "power",
            "mde_abs",
            "admission_probability",
        ):
            reason = (
                "mde_unavailable_reason" if field == "mde_abs" else f"{field}_unavailable_reason"
            )
            require_reason_when_null(
                getattr(self, field),
                getattr(self, reason),
                name=field,
                invalid=lambda message, field=field, reason=reason: _invalid(
                    "result",
                    message,
                    field=field,
                    value=(getattr(self, field), getattr(self, reason)),
                    constraint=message,
                    route="Set an unavailable reason exactly when the corresponding value is null.",
                ),
            )
        if (self.n is None) != (self.df is None) or (self.n is not None and self.df != self.n - 1):
            _invalid(
                "result",
                "df must equal n - 1 at the independent planning grain",
                field="n/df",
                value=(self.n, self.df),
                constraint="df=n-1 when n is provided, or both null",
            )
        if self.mde_abs is not None and self.mde_abs != self.delta:
            _invalid(
                "result",
                "mde_abs must equal the returned absolute effect delta",
                field="mde_abs/delta",
                value=(self.mde_abs, self.delta),
                constraint="mde_abs equals delta",
            )
        return self

    @property
    def planning_grain(self) -> Literal["unit", "shared_block"]:
        return self.baseline.planning_grain


def _result(
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
    *,
    n: int | None,
    delta: float | None,
    reason: UnavailableReason = "numerical_resolution",
    se: float | None = None,
    power: float | None = None,
    mde: float | None = None,
    mde_reason: UnavailableReason | None = "not_requested",
    se_reason: UnavailableReason | None = None,
) -> SwitchbackPowerResult:
    admission_probability = _admission_probability(n, baseline) if n is not None else None
    return SwitchbackPowerResult(
        baseline=baseline,
        procedure=procedure,
        n=n,
        df=None if n is None else n - 1,
        delta=delta,
        standard_error=se,
        power=power,
        mde_abs=mde,
        admission_probability=admission_probability,
        n_unavailable_reason=reason if n is None else None,
        df_unavailable_reason=reason if n is None else None,
        admission_probability_unavailable_reason=reason if n is None else None,
        delta_unavailable_reason=reason if delta is None else None,
        standard_error_unavailable_reason=(se_reason or reason) if se is None else None,
        power_unavailable_reason=reason if power is None else None,
        mde_unavailable_reason=mde_reason if mde is None else None,
    )


def _prepare(
    baseline: SwitchbackBaseline, procedure: ContrastDecisionProcedure
) -> tuple[SwitchbackBaseline, ContrastDecisionProcedure, UnavailableReason | None]:
    if not isinstance(baseline, SwitchbackBaseline):
        _invalid(
            "baseline",
            "baseline must be SwitchbackBaseline",
            field="baseline",
            value=baseline,
            constraint="SwitchbackBaseline instance",
        )
    if not isinstance(procedure, ContrastDecisionProcedure):
        _invalid(
            "procedure",
            "procedure must be ContrastDecisionProcedure",
            field="procedure",
            value=procedure,
            constraint="ContrastDecisionProcedure instance",
        )
    try:
        baseline = SwitchbackBaseline.model_validate(baseline)
    except InvalidRequestError:
        raise
    except ValidationError as exc:
        _invalid(
            "baseline",
            str(exc),
            field="baseline",
            value=baseline,
            constraint="valid SwitchbackBaseline model",
        )
    try:
        procedure = ContrastDecisionProcedure.model_validate(procedure)
    except InvalidRequestError:
        raise
    except ValidationError as exc:
        _invalid(
            "procedure",
            str(exc),
            field="procedure",
            value=procedure,
            constraint="valid ContrastDecisionProcedure model",
        )
    if procedure.reference is not None and procedure.reference.kind != "unit_t_approximation":
        _invalid(
            "procedure",
            "variance-envelope procedures require unit_cycle_model_power",
            field="procedure.reference.kind",
            value=procedure.reference.kind,
            constraint="unit_t_approximation for switchback planning",
            route="Use unit_cycle_model_power for a variance-envelope procedure.",
        )
    reason: UnavailableReason | None = (
        "metadata_mismatch" if baseline.metric != procedure.metric else None
    )
    return baseline, procedure, reason


def _check_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 2**53:
        _invalid(
            "n",
            "n must be an independent unit/block integer in [2, 2**53]",
            field="n",
            value=n,
            constraint="integer in [2, 2**53], excluding bool",
        )


def _check_target(target: float) -> None:
    if not math.isfinite(target) or not 0 < target < 1:
        _invalid(
            "target_power",
            "target_power must be finite and strictly between zero and one",
            field="target_power",
            value=target,
            constraint="finite value strictly between 0 and 1",
        )


def _admission_probability(n: int, baseline: SwitchbackBaseline) -> float:
    """Probability a declared shared-schedule design realizes both cycle
    orders across its n blocks (the increment/switchback.py positivity
    refusal); 1.0 for an independent-order baseline, where that refusal
    does not apply.

    p**n + (1-p)**n is the probability every block lands the same
    order (all-CT or all-TC); its complement is the probability the
    shared-schedule construction admits at all. Strictly in (0, 1) for
    n >= 2 and p in (0, 1) (both enforced by SharedScheduleOrder's own
    validators and _check_n), so this never divides by zero.
    """
    if baseline.planning_grain != "shared_block":
        return 1.0
    p = baseline.assignment.sequence.probability_ct
    return 1.0 - p**n - (1.0 - p) ** n


def _variance(delta: Fraction, baseline: SwitchbackBaseline) -> Fraction:
    """Exact Cholesky squares for binary64 inputs, including exact zero tests.

    Rational arithmetic also protects delta-delta_ref, intermediate squares,
    and near cancellation. The residual uses (1-|rho|)(1+|rho|), never 1-rho².
    """
    sa, sg, rho = map(Fraction, (baseline.sd_a, baseline.sd_g, baseline.rho))
    dsg = (delta - Fraction(baseline.delta_ref)) * sg
    return (sa + rho * dsg) ** 2 + dsg**2 * (1 - abs(rho)) * (1 + abs(rho))


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _sqrt(value: Fraction) -> float:
    with localcontext() as ctx:
        ctx.prec = 100
        result = float(_decimal(value).sqrt())
    if not math.isfinite(result):
        return result
    # Decimal-to-float rounding can straddle an exact binary midpoint.
    # Compare squared rational midpoints, including half the least subnormal.
    even = struct.unpack(">Q", struct.pack(">d", result))[0] % 2 == 0
    if result > 0:
        previous = math.nextafter(result, 0.0)
        midpoint = (Fraction(previous) + Fraction(result)) / 2
        if value < midpoint**2 or (value == midpoint**2 and not even):
            return previous
    following = math.nextafter(result, math.inf)
    if math.isfinite(following):
        midpoint = (Fraction(result) + Fraction(following)) / 2
        if value > midpoint**2 or (value == midpoint**2 and not even):
            return following
    return result


def _nc(distance: Fraction, variance: Fraction, n: int) -> float:
    if distance == 0:
        return 0.0
    # A dimensionless variance avoids cancelling huge logarithms. The existing
    # helper supplies its signed overflow limit without raw float squares.
    ratio = variance / (n * distance**2)
    with localcontext() as ctx:
        ctx.prec = 80
        log_ratio = float(_decimal(ratio).ln())
    return _noncentrality(1.0 if distance > 0 else -1.0, log_ratio)


def _power(nc: float, n: int, procedure: ContrastDecisionProcedure) -> float | None:
    try:
        tail = (
            _conservative_divide(procedure.alpha, 2)
            if procedure.alternative == "two-sided"
            else procedure.alpha
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = _scalar_power_from_nc(
                nc,
                alternative=procedure.alternative,
                tail_alpha=tail,
                dof=n - 1,
            )
    except (InvalidRequestError, OverflowError, FloatingPointError, RuntimeWarning):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def _inside(delta: float, baseline: SwitchbackBaseline) -> bool:
    return (
        baseline.effect_bounds is None
        or baseline.effect_bounds[0] <= delta <= baseline.effect_bounds[1]
    )


def switchback_achieved_power(
    n: int,
    delta: float,
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
) -> SwitchbackPowerResult:
    """Noncentral-t planning power at SE=sqrt(V(delta)/n), df=n-1.

    A true zero variance is unavailable. Positive variance below float range
    remains positive for power calculation; only an unrepresentable SE is null.
    Effects are normalized to binary64 before calculation and recording.
    """
    _check_n(n)
    try:
        delta = float(delta)
    except (TypeError, ValueError, OverflowError):
        _invalid(
            "delta",
            "delta must be a finite binary64 value",
            field="delta",
            value=delta,
            constraint="finite binary64 value",
        )
    if not math.isfinite(delta):
        _invalid(
            "delta",
            "delta must be finite",
            field="delta",
            value=delta,
            constraint="finite real value",
        )
    baseline, procedure, reason = _prepare(baseline, procedure)
    if reason is not None:
        return _result(baseline, procedure, n=n, delta=delta, reason=reason)
    if not _inside(delta, baseline):
        return _result(baseline, procedure, n=n, delta=delta, reason="effect_outside_domain")
    variance = _variance(Fraction(delta), baseline)
    if variance == 0:
        return _result(baseline, procedure, n=n, delta=delta, reason="degenerate_zero_variance")
    se = _sqrt(variance / n)
    power_given_admissible = _power(
        _nc(Fraction(delta) - Fraction(procedure.null_abs), variance, n), n, procedure
    )
    power = (
        None
        if power_given_admissible is None
        else power_given_admissible * _admission_probability(n, baseline)
    )
    return _result(
        baseline,
        procedure,
        n=n,
        delta=delta,
        power=power,
        se=se if 0 < se < math.inf else None,
        se_reason="unrepresentable",
    )


def _coefficients(
    baseline: SwitchbackBaseline, null: float, sign: int
) -> tuple[Fraction, Fraction, Fraction]:
    sa, sg, rho = map(Fraction, (baseline.sd_a, baseline.sd_g, baseline.rho))
    d0 = Fraction(null) - Fraction(baseline.delta_ref)
    a = sg**2
    b = sign * (sa * sg * rho + d0 * a)
    return a, b, _variance(Fraction(null), baseline)


def _quadratic_roots(
    n: int,
    level: float,
    a: Fraction,
    b: Fraction,
    c: Fraction,
) -> tuple[Decimal, ...]:
    """Solve (signal_scale-L²a)x²-2L²bx-L²c=0; signal_scale=n.

    All discriminant arithmetic is exact. One root uses the large-magnitude
    sum, the other the product of roots, so neither loses a small root.
    """
    l2 = Fraction(level) ** 2
    leading, half_linear, constant = Fraction(n) - l2 * a, -l2 * b, -l2 * c
    if leading == 0:
        return () if half_linear == 0 else (_decimal(-constant / (2 * half_linear)),)
    discriminant = half_linear**2 - leading * constant
    if discriminant < 0:
        return ()
    with localcontext() as ctx:
        ctx.prec = 100
        radical = _decimal(discriminant).sqrt()
        q = -_decimal(half_linear) - (radical if half_linear >= 0 else -radical)
        if q == 0:
            return (Decimal(0),)
        roots = (q / _decimal(leading), _decimal(constant) / q)
        return tuple(sorted(root for root in roots if root >= 0))


def _effect(null: float, sign: int, distance: Fraction | Decimal) -> float:
    with localcontext() as ctx:
        ctx.prec = 100
        x = _decimal(distance) if isinstance(distance, Fraction) else distance
        return float(Decimal.from_float(null) + sign * x)


def _favorable_power(nc: float, n: int, procedure: ContrastDecisionProcedure) -> float | None:
    return _power(-nc if procedure.alternative == "less" else nc, n, procedure)


def _required_nc(n: int, procedure: ContrastDecisionProcedure, target: float) -> float | None:
    """Invert only the scalar reference distribution, once per MDE query."""
    upper = 1.0
    for _ in range(1024):
        value = _favorable_power(upper, n, procedure)
        if value is None:
            return None
        if value >= target:
            break
        upper *= 2
    else:
        return None

    def objective(nc: float) -> float:
        value = _favorable_power(nc, n, procedure)
        return math.nan if value is None else value - target

    try:
        return float(
            brentq(objective, 0, upper, xtol=math.ulp(0.0), rtol=4 * sys.float_info.epsilon)
        )
    except (ValueError, RuntimeError):
        return None


def _search_sign(procedure: ContrastDecisionProcedure) -> int:
    """Return the signed effect direction used by inverse planning."""
    if procedure.alternative == "less" or (
        procedure.alternative == "two-sided" and procedure.preferred_direction == "decrease"
    ):
        return -1
    return 1


def _certify_effect(
    candidate: float,
    start: float,
    top: float,
    n: int,
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
    target: float,
) -> SwitchbackPowerResult:
    """Round an analytic root to the first feasible public float on its branch.

    Float-key refinement is secondary to the analytic inversion and restricted
    to the proven ascending branch. Both the answer and predecessor are checked
    through the public achieved-power calculation, including shifted-null rounding.
    """
    sign = _search_sign(procedure)

    def evaluate(t: float) -> SwitchbackPowerResult:
        return switchback_achieved_power(n, sign * t, baseline, procedure)

    def finish(row: SwitchbackPowerResult) -> SwitchbackPowerResult:
        return row.model_copy(update={"mde_abs": row.delta, "mde_unavailable_reason": None})

    def failed(reason: UnavailableReason) -> SwitchbackPowerResult:
        return _result(baseline, procedure, n=n, delta=None, reason=reason, mde_reason=reason)

    lower, upper = sign * start, sign * top
    t = min(upper, max(lower, sign * candidate))
    # Usually rounding the analytic root needs fewer than ten neighboring floats.
    for _ in range(32):
        row = evaluate(t)
        if row.power is None:
            return failed("numerical_resolution")
        if row.power >= target:
            prev = math.nextafter(t, -math.inf)
            if prev < lower:
                return finish(row)
            previous = evaluate(prev)
            if previous.power is None:
                return failed("numerical_resolution")
            if previous.power < target:
                return finish(row)
            upper, t = t, prev
        else:
            lower = math.nextafter(t, math.inf)
            if lower > upper:
                return failed("unrepresentable")
            t = lower
    # For a large offset or flat peak, refine float rounding, not the quadratic.
    high_row = evaluate(upper)
    if high_row.power is None:
        return failed("numerical_resolution")
    if high_row.power < target:
        return failed("numerical_resolution")
    lo_ordinal, hi_ordinal = float_ordinal(lower), float_ordinal(upper)
    while lo_ordinal < hi_ordinal:
        mid = (lo_ordinal + hi_ordinal) // 2
        row = evaluate(float_from_ordinal(mid))
        if row.power is None:
            return failed("numerical_resolution")
        if row.power >= target:
            hi_ordinal = mid
        else:
            lo_ordinal = mid + 1
    row = evaluate(float_from_ordinal(lo_ordinal))
    previous_t = math.nextafter(float_from_ordinal(lo_ordinal), -math.inf)
    if previous_t >= sign * start:
        previous = evaluate(previous_t)
        if previous.power is None or previous.power >= target:
            return failed("numerical_resolution")
    return finish(row)


def _peak_bounds(
    n: int,
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
    target: float,
    start_t: float,
    top_t: float,
) -> tuple[float, float | None, UnavailableReason | None]:
    """Limit the ascending branch and retain the first descending public float."""
    sign = _search_sign(procedure)
    null = procedure.null_abs
    a, b, c = _coefficients(baseline, null, sign)
    descending_start: float | None = None
    if b < 0:
        peak = -c / b
        peak_variance = a * peak**2 + 2 * b * peak + c
        if peak_variance:
            peak_power = _favorable_power(_nc(peak, peak_variance, n), n, procedure)
            if peak_power is not None and peak_power < target:
                return top_t, None, "unattained"
            # A huge peak NC can exceed the tail backend's range while the
            # ascending root is resolvable. The quadratic still identifies it.
        exact_t = Fraction(sign) * Fraction(null) + peak
        try:
            peak_t = float(exact_t)
        except OverflowError:
            peak_t = math.inf
        # There may be no feasible ascending float, but the first descending
        # float can still lie between the roots. Keep that candidate too.
        if math.isfinite(peak_t):
            descending_t = peak_t
            if Fraction(descending_t) < exact_t or (
                peak_variance == 0 and Fraction(descending_t) == exact_t
            ):
                descending_t = math.nextafter(descending_t, math.inf)
            if start_t <= descending_t <= top_t:
                descending_start = sign * descending_t
            if Fraction(peak_t) > exact_t or (peak_variance == 0 and Fraction(peak_t) == exact_t):
                peak_t = math.nextafter(peak_t, -math.inf)
            top_t = min(top_t, peak_t)
    return top_t, descending_start, None


def _finish_mde(
    result: SwitchbackPowerResult,
    descending_start: float | None,
    top_t: float,
    target: float,
) -> SwitchbackPowerResult:
    """Check the descending candidate only after excluding every ascending float."""
    n, baseline, procedure = result.n, result.baseline, result.procedure
    assert n is not None
    sign = _search_sign(procedure)

    def failed(reason: UnavailableReason) -> SwitchbackPowerResult:
        return _result(baseline, procedure, n=n, delta=None, reason=reason, mde_reason=reason)

    if result.mde_unavailable_reason == "unrepresentable" and descending_start is not None:
        descending = switchback_achieved_power(n, descending_start, baseline, procedure)
        if descending.power is None:
            return failed("numerical_resolution")
        if descending.power >= target:
            previous_delta = sign * math.nextafter(sign * descending_start, -math.inf)
            previous = switchback_achieved_power(n, previous_delta, baseline, procedure)
            if (previous.power is not None and previous.power < target) or (
                previous.power_unavailable_reason == "degenerate_zero_variance"
            ):
                return descending.model_copy(
                    update={
                        "mde_abs": descending_start,
                        "mde_unavailable_reason": None,
                    }
                )
            return failed("numerical_resolution")
    if result.mde_unavailable_reason == "unrepresentable" and baseline.effect_bounds is not None:
        endpoint = sign * (baseline.effect_bounds[0] if sign < 0 else baseline.effect_bounds[1])
        if top_t == endpoint:
            return failed("unattained")
    return result


def _constant_mde(
    n: int,
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
    target_power: float,
    target_power_given_admissible: float,
    start_t: float,
    top_t: float,
) -> SwitchbackPowerResult:
    """Handle an excluded zero-variance null and a constant positive half-line."""
    sign = _search_sign(procedure)
    a = Fraction(baseline.sd_g) ** 2

    def failed(reason: UnavailableReason) -> SwitchbackPowerResult:
        return _result(baseline, procedure, n=n, delta=None, reason=reason, mde_reason=reason)

    constant = _favorable_power(_nc(Fraction(1), a, n), n, procedure)
    if constant is None:
        return failed("numerical_resolution")
    if constant < target_power_given_admissible:
        return failed("unattained")
    start_t = max(start_t, math.nextafter(sign * procedure.null_abs, math.inf))
    if start_t > top_t or not math.isfinite(start_t):
        return failed("unrepresentable")
    return _certify_effect(
        sign * start_t, sign * start_t, sign * top_t, n, baseline, procedure, target_power
    )


def switchback_minimum_detectable_effect(
    n: int,
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
    *,
    target_power: float,
) -> SwitchbackPowerResult:
    """First favorable absolute effect attaining target_power, analytically inverted.

    Write delta=null_abs+sign*x, x>=0, and V=a*x²+2*b*x+c. The derivative
    of x/sqrt(V) has sign b*x+c. For b<0 the maximum is at -c/b;
    otherwise the curve increases to sqrt(n/a) (or infinity when a=0).
    A zero-variance maximum is excluded. c=0 is a constant positive
    half-line; a=c=0 is wholly degenerate. Asymptote equality is unattained.
    """
    _check_n(n)
    _check_target(target_power)

    def failed(reason: UnavailableReason) -> SwitchbackPowerResult:
        return _result(baseline, procedure, n=n, delta=None, reason=reason, mde_reason=reason)

    baseline, procedure, reason = _prepare(baseline, procedure)
    if reason is not None:
        return failed(reason)
    admission_probability = _admission_probability(n, baseline)
    if admission_probability <= target_power:
        # Conditional power is a survival probability below 1 at every finite delta, so a
        # rescaled target >= 1 is unattainable (equality included, as at the asymptote). Report
        # "unattained" before _required_nc sees it: with a=0 it would return None and misreport
        # "numerical_resolution". More blocks or probability_ct nearer 0.5 raise admission.
        return failed("unattained")
    target_power_given_admissible = target_power / admission_probability
    sign = _search_sign(procedure)
    null = procedure.null_abs
    a, b, c = _coefficients(baseline, null, sign)
    start_t = sign * null
    top_t = sys.float_info.max
    if baseline.effect_bounds is not None:
        bounds = sorted(sign * value for value in baseline.effect_bounds)
        start_t, top_t = max(start_t, bounds[0]), bounds[1]
    if start_t > top_t:
        return failed("effect_outside_domain")
    start = sign * start_t
    initial = switchback_achieved_power(n, start, baseline, procedure)
    if start == null and initial.power is not None and initial.power >= target_power:
        return initial.model_copy(update={"mde_abs": start, "mde_unavailable_reason": None})
    if initial.power is None and initial.power_unavailable_reason != "degenerate_zero_variance":
        return failed("numerical_resolution")
    if a == 0 and c == 0:
        return failed("degenerate_zero_variance")
    if c == 0:
        return _constant_mde(
            n, baseline, procedure, target_power, target_power_given_admissible, start_t, top_t
        )

    if a > 0 and b >= 0:
        asymptote = _favorable_power(_nc(Fraction(1), a, n), n, procedure)
        if asymptote is None:
            return failed("numerical_resolution")
        if asymptote <= target_power_given_admissible:
            return failed("unattained")

    if initial.power is not None and initial.power >= target_power:
        return initial.model_copy(update={"mde_abs": start, "mde_unavailable_reason": None})
    top_t, descending_start, reason = _peak_bounds(
        n,
        baseline,
        procedure,
        target_power_given_admissible,
        start_t,
        top_t,
    )
    if reason is not None:
        return failed(reason)
    if start_t > top_t:
        if descending_start is not None:
            return _finish_mde(failed("unrepresentable"), descending_start, top_t, target_power)
        return failed("unattained")
    if _variance(Fraction(sign * start_t), baseline) == 0:
        start_t = math.nextafter(start_t, math.inf)
    if start_t > top_t:
        return failed("unrepresentable")
    null_power = _favorable_power(0, n, procedure)
    if null_power is None:
        return failed("numerical_resolution")
    if null_power >= target_power_given_admissible:
        return _certify_effect(
            sign * start_t, sign * start_t, sign * top_t, n, baseline, procedure, target_power
        )
    level = _required_nc(n, procedure, target_power_given_admissible)
    if level is None:
        return failed("numerical_resolution")
    roots = _quadratic_roots(n, level, a, b, c)
    # Near tangency the reference inversion can round L just above the maximum.
    # The directly evaluated peak is the feasible rounding bracket in that case.
    candidate = _effect(null, sign, roots[0]) if roots else sign * top_t
    result = _certify_effect(
        candidate, sign * start_t, sign * top_t, n, baseline, procedure, target_power
    )
    return _finish_mde(result, descending_start, top_t, target_power)


def switchback_required_blocks_or_units(
    delta: float,
    baseline: SwitchbackBaseline,
    procedure: ContrastDecisionProcedure,
    *,
    target_power: float,
) -> SwitchbackPowerResult:
    """Smallest integer independent N>=2 meeting target, with df=N-1 each time.

    For fixed delta, V(delta)>0 is fixed and the reference is the one-sample
    Gaussian t experiment with noncentrality sqrt(N)*(delta-null)/sqrt(V).
    Its one-sided invariant test (two-sided symmetric unbiased test) is most
    powerful in its class. Ignoring the added observation gives an admissible
    test at N+1 with the old power; optimality therefore proves nondecreasing
    power. This argument is for the planning reference, not the branch law.
    Doubling and integer bisection use that property only for favorable effects.
    The exact-integer float reference range is capped at 2**53.
    """
    _check_target(target_power)
    first = switchback_achieved_power(2, delta, baseline, procedure)
    assert first.delta is not None
    delta = first.delta
    baseline, procedure = first.baseline, first.procedure

    def failed(reason: UnavailableReason) -> SwitchbackPowerResult:
        return _result(baseline, procedure, n=None, delta=delta, reason=reason)

    if first.power is None:
        return failed(first.power_unavailable_reason or "numerical_resolution")
    distance = Fraction(delta) - Fraction(procedure.null_abs)
    if (
        distance == 0
        or (procedure.alternative == "greater" and distance < 0)
        or (procedure.alternative == "less" and distance > 0)
    ):
        return failed("non_favorable_effect")
    if first.power >= target_power:
        return first
    low, high = 2, 4
    while True:
        row = switchback_achieved_power(high, delta, baseline, procedure)
        if row.power is None:
            return failed(row.power_unavailable_reason or "numerical_resolution")
        if row.power >= target_power:
            break
        if high == 2**53:
            return failed("unrepresentable")
        low, high = high, min(2**53, high * 2)

    class _BisectionUnavailable(Exception):
        def __init__(self, reason: UnavailableReason) -> None:
            self.reason = reason

    def accepts(n: int) -> bool:
        row = switchback_achieved_power(n, delta, baseline, procedure)
        if row.power is None:
            raise _BisectionUnavailable(row.power_unavailable_reason or "numerical_resolution")
        return row.power >= target_power

    try:
        high = bisect_first_true(low + 1, high, accepts)
    except _BisectionUnavailable as error:
        return failed(error.reason)
    result = switchback_achieved_power(high, delta, baseline, procedure)
    predecessor = switchback_achieved_power(high - 1, delta, baseline, procedure)
    if (
        result.power is None
        or result.power < target_power
        or predecessor.power is None
        or predecessor.power >= target_power
    ):
        return failed("numerical_resolution")
    return result


__all__ = [
    "SwitchbackBaseline",
    "SwitchbackPowerResult",
    "switchback_achieved_power",
    "switchback_minimum_detectable_effect",
    "switchback_required_blocks_or_units",
]
