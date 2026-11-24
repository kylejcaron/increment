"""Model-conditioned rejection probabilities for fixed-horizon switchbacks.

With m of K positions treated, a replicate contributes the treated mean minus
control mean, averaged over all its cycles (unit) or roster members (block).
A constant shift adds exactly delta to this contrast. Resample complete
untreated profiles, fitted as Yobs - Z * delta_ref, then redraw the schedule.
For conversion instead draw independent retained-window indicators given arm,
with risks p0 and p0 + delta in [0, 1], using the shortest round-trip decimal
representations of the declared risk and effect. Paired laws keep their order weights.
Neither model corrects residual carryover. In particular the Bernoulli model
ignores serial dependence and common shocks: use external design simulation if
these matter. Confidence bounds cover Monte Carlo error conditional on the
model, not pilot fitting error, test size, or a continuous-grid minimum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.special import betainccinv, betaincinv
from scipy.stats import binomtest

from increment._unit_cycle import (
    _fixed_count_moments,
    _fixed_count_t_test,
    _float_outward,
    unit_cycle_admission_outcome,
)
from increment.errors import CodedModel, InvalidRequestError
from increment.estimation._tails import student_t_isf, tail_isf
from increment.estimation.decision_types import ContrastDecisionProcedure, exact_fraction
from increment.power._validation import _as_count
from increment.power.switchback import _invalid
from increment.semantics.assignment import SwitchbackAssignment
from increment.semantics.unit_cycle import UnitCycleTApproximation

# Bound simulated window arrays independently of the requested number of draws.
_MAX_WINDOWS = 262_144
_MAX_CELLS = 4096


class _Frozen(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class _ProfileBaseline(_Frozen):
    assignment: SwitchbackAssignment
    metric: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    roster: tuple[str, ...] = Field(min_length=1)
    cycles_per_unit: int = Field(ge=1, strict=True)
    delta_ref: float = Field(default=0.0, allow_inf_nan=False)
    identifying_assumption: Literal["no_residual_carryover_after_discarded_steps"] = (
        "no_residual_carryover_after_discarded_steps"
    )

    @field_validator("assignment", mode="before")
    @classmethod
    def _copy_assignment(cls, value: object) -> object:
        return value.model_dump() if isinstance(value, SwitchbackAssignment) else value

    @model_validator(mode="after")
    def _metadata(self):
        if self.control_group == self.treatment_group:
            _invalid("baseline", "Control and treatment must have distinct names.")
        if any(not key.strip() for key in self.roster) or self.roster != tuple(
            sorted(set(self.roster))
        ):
            _invalid("roster", "List distinct nonempty roster names in sorted order.")
        return self

    @property
    def planning_grain(self) -> Literal["unit", "shared_block"]:
        return (
            "unit" if self.assignment.sequence.independence_unit == "unit_cycle" else "shared_block"
        )


@dataclass(frozen=True, slots=True)
class _Pilot:
    roster: tuple[str, ...]
    y: np.ndarray
    z: np.ndarray


def _total(value: Any) -> Fraction:
    # Fraction's Rational path keeps a NumPy integer's fixed-width numerator.
    # Convert native array scalars before doing any exact arithmetic.
    return Fraction(value.item() if isinstance(value, np.generic) else value)


def _pilot(
    assignment: SwitchbackAssignment, roster: Sequence[str], observed: ArrayLike, treated: ArrayLike
) -> _Pilot:
    if not isinstance(assignment, SwitchbackAssignment):
        _invalid("baseline", "Provide a switchback assignment with its window.")
    assignment = SwitchbackAssignment.model_validate(assignment.model_dump())
    try:
        # Object dtype preserves integer totals and exact rationals across heterogeneous rows.
        y = np.asarray(observed, dtype=object)
        z = np.asarray(treated)
    except (TypeError, ValueError, OverflowError):
        _invalid("baseline", "Provide complete rectangular unit by cycle by position arrays.")
    seq = assignment.sequence
    if (
        y.shape != z.shape
        or y.ndim != 3
        or min(y.shape) < 1
        or y.shape[2] != assignment.periods_per_cycle
    ):
        _invalid("baseline", "Pilot arrays must match on unit, cycle and every assigned position.")
    if (
        isinstance(roster, str)
        or len(roster) != len(y)
        or any(not isinstance(r, str) for r in roster)
    ):
        _invalid("roster", "Give one unit name per pilot trajectory in the same order.")
    if not np.isin(z, (0, 1)).all():
        _invalid("baseline", "Pilot assignments must be complete zero/one indicators.")
    z = z.astype(bool)
    m = seq.treated_periods if seq.scheme == "fixed_count" else 1
    if not (z.sum(axis=2) == m).all():
        _invalid("baseline", "Pilot treatment counts differ from the declared schedule.")
    shared = seq.independence_unit == "shared_block"
    if shared and not (z == z[0]).all():
        _invalid("baseline", "All pilot roster members must follow the shared block schedule.")
    if (y.shape[1] if shared else len(y)) < 2:
        _invalid("baseline", "The pilot must contain at least two independent units or blocks.")
    draws = z[0] if shared else z.reshape(-1, y.shape[2])
    if seq.scheme == "fixed_count":
        if not (draws.any(axis=0) & ~draws.all(axis=0)).all():
            _invalid("baseline", "Pilot assignments must expose each position to both arms.")
    elif not _paired_admission(np.sum(draws[:, 1]), len(draws), seq.probability_ct, shared=shared):
        _invalid("baseline", "The pilot order split is not admitted by the paired schedule.")
    order = sorted(range(len(roster)), key=roster.__getitem__)
    return _Pilot(tuple(roster[i] for i in order), y[order], z[order])


class SwitchbackTrajectoryBaseline(_ProfileBaseline):
    """Whole untreated trajectories on a uniform fixed-count schedule.

    Profiles are unit × cycle × position for independent orders, or block ×
    sorted roster × position for shared orders. Rational window totals preserve
    small changes at large offsets in the portable declaration.
    """

    aggregation: Literal["sum"] = "sum"
    estimand: Literal["retained_window_total_difference"] = "retained_window_total_difference"
    effect_model: Literal["constant_additive_whole_profiles"] = "constant_additive_whole_profiles"
    profile_source: Literal["specified", "pilot_estimated"] = "specified"
    profiles: tuple[tuple[tuple[Fraction, ...], ...], ...]

    @model_validator(mode="after")
    def _shape(self):
        if self.assignment.sequence.scheme != "fixed_count":
            _invalid("baseline", "Use the paired moment planner for paired mean outcomes.")
        a, b = len(self.roster), self.cycles_per_unit
        if self.planning_grain == "shared_block":
            a, b = b, a
        if len(self.profiles) != a or any(
            len(rep) != b or any(len(row) != self.assignment.periods_per_cycle for row in rep)
            for rep in self.profiles
        ):
            _invalid(
                "baseline", "Complete profiles must match the roster, cycle and position counts."
            )
        return self

    @classmethod
    def from_pilot(
        cls,
        *,
        assignment: SwitchbackAssignment,
        metric: str,
        control_group: str,
        treatment_group: str,
        roster: Sequence[str],
        observed: ArrayLike,
        treated: ArrayLike,
        delta_ref: float,
    ) -> SwitchbackTrajectoryBaseline:
        """Remove the fitted constant shift from complete assigned period totals."""
        data = _pilot(assignment, roster, observed, treated)
        shift = Fraction(_finite(delta_ref, "delta_ref"))
        rows = data.y
        flags = data.z
        if assignment.sequence.independence_unit == "shared_block":
            rows, flags = rows.swapaxes(0, 1), flags.swapaxes(0, 1)
        try:
            profiles = tuple(
                tuple(
                    tuple(_total(v) - (shift if z else 0) for v, z in zip(row, mask, strict=True))
                    for row, mask in zip(rep, arm, strict=True)
                )
                for rep, arm in zip(rows, flags, strict=True)
            )
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            _invalid("baseline", "Pilot retained totals must be finite numbers.")
        return cls(
            assignment=assignment,
            metric=metric,
            control_group=control_group,
            treatment_group=treatment_group,
            roster=data.roster,
            cycles_per_unit=data.y.shape[1],
            delta_ref=delta_ref,
            profiles=profiles,
            profile_source="pilot_estimated",
        )


class SwitchbackConversionBaseline(_ProfileBaseline):
    """Bernoulli risk of any event in a retained window (not per observation step)."""

    aggregation: Literal["any"] = "any"
    estimand: Literal["retained_window_conversion_difference"] = (
        "retained_window_conversion_difference"
    )
    effect_model: Literal["bernoulli_retained_window"] = "bernoulli_retained_window"
    dependence: Literal["independent_windows_and_roster_units_given_arm"] = (
        "independent_windows_and_roster_units_given_arm"
    )
    untreated_risk: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_source: Literal["specified", "pilot_estimated"] = "specified"

    @classmethod
    def from_pilot(
        cls,
        *,
        assignment: SwitchbackAssignment,
        metric: str,
        control_group: str,
        treatment_group: str,
        roster: Sequence[str],
        observed: ArrayLike,
        treated: ArrayLike,
        delta_ref: float,
    ) -> SwitchbackConversionBaseline:
        """Estimate p0 as sum(I_control Y / pi_control) / sum(I_control / pi_control)."""
        data = _pilot(assignment, roster, observed, treated)
        if not np.isin(data.y, (0, 1)).all():
            _invalid(
                "baseline", "Conversion observations must be retained-window binary indicators."
            )
        hits = data.y.astype(bool) & ~data.z
        seq = assignment.sequence
        if seq.scheme == "fixed_count":
            risk = int(hits.sum()) / int((~data.z).sum())
        else:
            p = Fraction(seq.probability_ct)
            control_counts = (~data.z).sum(axis=(0, 1)).tolist()
            successes = hits.sum(axis=(0, 1)).tolist()
            # The control probabilities at the first and second positions are p and 1-p.
            q = 1 - p
            risk = float(
                (successes[0] * q + successes[1] * p)
                / (control_counts[0] * q + control_counts[1] * p)
            )
        return cls(
            assignment=assignment,
            metric=metric,
            control_group=control_group,
            treatment_group=treatment_group,
            roster=data.roster,
            cycles_per_unit=data.y.shape[1],
            delta_ref=delta_ref,
            untreated_risk=risk,
            risk_source="pilot_estimated",
        )


type _Baseline = Annotated[
    SwitchbackTrajectoryBaseline | SwitchbackConversionBaseline, Field(discriminator="effect_model")
]


def _finite(value: object, field: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        _invalid("delta", "Provide a finite numeric value.", field=field, value=value)
    try:
        result = float(value)
    except OverflowError:
        _invalid("delta", "The value is outside binary64 range.", field=field, value=value)
    if not math.isfinite(result):
        _invalid("delta", "Provide a finite numeric value.", field=field, value=value)
    return result


def _count(value: object, field: str, minimum: int) -> int:
    result = _as_count(value)
    if result is None or result < minimum:
        _invalid(
            "n", f"{field} must be an integer of at least {minimum}.", field=field, value=value
        )
    return result


def _check_binding(b: _Baseline, p: ContrastDecisionProcedure) -> None:
    if b.metric != p.metric or p.forecast_column is not None:
        _invalid("procedure", "Use the same unadjusted metric as the pilot.")
    if b.planning_grain == "shared_block":
        if p.reference is not None:
            _invalid(
                "procedure", "A shared block uses its block-t reference, not a unit reference."
            )
    elif not isinstance(p.reference, UnitCycleTApproximation):
        _invalid("procedure", "Independent units require the unit-t approximation for this model.")


def _risks(b: SwitchbackConversionBaseline, delta: float) -> tuple[Fraction, Fraction]:
    p0 = exact_fraction(b.untreated_risk)
    p1 = p0 + exact_fraction(delta)
    if not 0 <= p1 <= 1:
        _invalid(
            "delta",
            "The untreated risk plus the proposed risk difference must be in [0, 1].",
            field="untreated_risk/delta",
            value=(b.untreated_risk, delta),
        )
    return p0, p1


@lru_cache(maxsize=2048)
def _paired_count(count: int, draws: int, p: float, *, shared: bool) -> bool:
    if shared:
        if count == 0 or count == draws:
            return False
        p_value = float(binomtest(count, draws, p).pvalue)
        return math.isfinite(p_value) and p_value >= 1e-6
    try:
        return unit_cycle_admission_outcome(draws, p, count)
    except InvalidRequestError as exc:
        if exc.code != "unit_cycle.numerical":
            raise
        return False


def _paired_admission(counts: Any, draws: int, p: float, *, shared: bool) -> np.ndarray:
    values, inverse = np.unique(counts, return_inverse=True)
    admitted = np.array([_paired_count(int(k), draws, p, shared=shared) for k in values])
    return admitted[inverse].reshape(np.shape(counts))


def _admission(b: _Baseline, z: np.ndarray) -> np.ndarray:
    # z is attempts × independent replicates × assignment draws × positions.
    seq = b.assignment.sequence
    if seq.scheme == "fixed_count":
        return np.all(z.any(axis=(1, 2)) & ~z.all(axis=(1, 2)), axis=-1)
    return _paired_admission(
        z[..., 1].sum(axis=(1, 2)),
        z.shape[1] * z.shape[2],
        seq.probability_ct,
        shared=b.planning_grain == "shared_block",
    )


def _bernoulli(rng: np.random.Generator, p: Fraction, size: int) -> np.ndarray:
    """Compare uniform 64-bit limbs to p; refine only an unresolved equal prefix."""
    if p in (0, 1):
        return np.full(size, bool(p))
    numerator, denominator = p.as_integer_ratio()
    threshold, remainder = divmod(numerator << 64, denominator)
    bits = rng.bit_generator.random_raw(size)
    result = bits < np.uint64(threshold)
    if remainder:
        ties = bits == np.uint64(threshold)
        count = int(ties.sum())
        if count:
            result[ties] = _bernoulli(rng, Fraction(remainder, denominator), count)
    return result


def _assignments(b: _Baseline, rng: np.random.Generator, batch: int, n: int) -> np.ndarray:
    g = b.cycles_per_unit if b.planning_grain == "unit" else 1
    k = b.assignment.periods_per_cycle
    shape = (batch, n, g, k)
    seq = b.assignment.sequence
    if seq.scheme == "fixed_count":
        # Generator.permuted independently shuffles each slice, without floating rank/tie bias.
        return rng.permuted(np.broadcast_to(np.arange(k) < seq.treated_periods, shape), axis=-1)
    ct = _bernoulli(rng, Fraction(seq.probability_ct), batch * n * g).reshape(shape[:-1])
    return np.stack((~ct, ct), axis=-1)


@dataclass(frozen=True, slots=True)
class _Draws:
    reason: np.ndarray
    rejected: np.ndarray


@lru_cache(maxsize=128)
def _critical(df: int, tail: float) -> float:
    return tail_isf(student_t_isf, tail, df, what="switchback t reference")


def _fixed_count_draws(
    numerator: np.ndarray,
    denominator: int,
    shift: Fraction,
    admitted: np.ndarray,
    procedure: ContrastDecisionProcedure,
) -> _Draws:
    """Apply the reported fixed-count test to each admitted schedule."""
    batch, n = numerator.shape
    df = n - 1
    root_df, root_n = math.sqrt(df), math.sqrt(n)
    reason = np.full(batch, None, dtype=object)
    reason[~admitted] = "missing_position_support"
    rejected = np.zeros(batch, dtype=bool)
    tail = procedure.alpha / 2 if procedure.alternative == "two-sided" else procedure.alpha
    try:
        critical = _critical(df, tail)
    except InvalidRequestError:
        reason[admitted] = "decision_failure"
        return _Draws(reason, rejected)
    for i in np.flatnonzero(admitted):
        try:
            row = numerator[i]
            # Conversion counts are fixed-width integers; trajectories are already rational.
            total, m2 = _fixed_count_moments(
                row if numerator.dtype.kind == "O" else map(int, row), n, denominator
            )
            center = total / n + shift
            if not m2:
                reason[i] = "zero_se" if math.isfinite(_as_float(center)) else "decision_failure"
            else:
                se = math.sqrt(m2) / root_df / root_n
                _, _, p, _ = _fixed_count_t_test(
                    center,
                    se,
                    df,
                    critical,
                    null_abs=procedure.null_abs,
                    alternative=procedure.alternative,
                )
                rejected[i] = p < procedure.alpha
        except (InvalidRequestError, OverflowError, ValueError):
            reason[i] = "decision_failure"
    return _Draws(reason, rejected)


def _paired_decisions(
    x: np.ndarray, admitted: np.ndarray, procedure: ContrastDecisionProcedure
) -> _Draws:
    batch, n = x.shape
    df = n - 1
    point = np.full(batch, np.nan)
    se = np.full(batch, np.nan)
    reason = np.full(batch, None, dtype=object)
    reason[~admitted] = "paired_order_split"
    for i in np.flatnonzero(admitted):
        try:
            reference = float(x[i, 0])
            residual = [float(value) - reference for value in x[i]]
            mean = math.fsum(residual) / n
            m2 = math.fsum((value - mean) ** 2 for value in residual)
            point[i] = math.fsum((reference, mean))
            se[i] = math.sqrt(m2) / math.sqrt(df) / math.sqrt(n)
        except (OverflowError, ValueError):
            reason[i] = "decision_failure"
    tail = procedure.alpha / 2 if procedure.alternative == "two-sided" else procedure.alpha
    try:
        critical = _critical(df, tail)
    except InvalidRequestError:
        reason[admitted] = "decision_failure"
        return _Draws(reason, np.zeros(batch, dtype=bool))
    with np.errstate(over="ignore", invalid="ignore"):
        half = critical * se
        lower, upper = point - half, point + half
    finite = np.isfinite(lower) & np.isfinite(upper)
    reason[admitted & ~finite] = "decision_failure"
    reason[admitted & finite & (se == 0)] = "zero_se"
    reject = np.zeros(batch, dtype=bool)
    if procedure.alternative != "less":
        reject |= lower > procedure.null_abs
    if procedure.alternative != "greater":
        reject |= upper < procedure.null_abs
    return _Draws(reason, reject & admitted & finite & (se > 0))


def _as_float(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


@dataclass(frozen=True, slots=True)
class _TrajectoryTable:
    untreated: np.ndarray
    total: np.ndarray


def _trajectory_table(b: SwitchbackTrajectoryBaseline) -> _TrajectoryTable:
    raw = np.asarray(b.profiles, dtype=object)
    # Only centered positions enter the contrast: subtract the first level exactly.
    raw = raw - raw[..., :1]
    if b.planning_grain == "shared_block":
        raw = raw.sum(axis=1, keepdims=True) / len(b.roster)
    return _TrajectoryTable(raw, raw.sum(axis=(1, 2)))


def _trajectory_draws(
    b: SwitchbackTrajectoryBaseline,
    p: ContrastDecisionProcedure,
    delta: float,
    indices: np.ndarray,
    z: np.ndarray,
    table: _TrajectoryTable,
) -> _Draws:
    seq = b.assignment.sequence
    assert seq.scheme == "fixed_count"
    k, m = b.assignment.periods_per_cycle, seq.treated_periods
    selected = np.where(z, table.untreated[indices], 0).sum(axis=(2, 3))
    # X = delta + [K sum_T Y0 - m sum_all Y0] / [g m (K-m)].
    numerator = k * selected - m * table.total[indices]
    denom = z.shape[2] * m * (k - m)
    return _fixed_count_draws(numerator, denom, Fraction(delta), _admission(b, z), p)


def _conversion_draws(
    b: SwitchbackConversionBaseline,
    proc: ContrastDecisionProcedure,
    risks: tuple[Fraction, Fraction],
    z: np.ndarray,
    rng: np.random.Generator,
) -> _Draws:
    shape = (
        (*z.shape[:2], len(b.roster), z.shape[3]) if b.planning_grain == "shared_block" else z.shape
    )
    arm = np.broadcast_to(z, shape)
    nt = int(arm.sum())
    y = np.empty(shape, dtype=bool)
    p0, p1 = risks
    y[arm] = _bernoulli(rng, p1, nt)
    y[~arm] = _bernoulli(rng, p0, arm.size - nt)
    seq = b.assignment.sequence
    g, k = shape[2:]
    if seq.scheme == "fixed_count":
        m = seq.treated_periods
        numerator = k * (arm & y).sum(axis=(2, 3)) - m * y.sum(axis=(2, 3))
        return _fixed_count_draws(numerator, m * (k - m) * g, Fraction(0), _admission(b, z), proc)
    # Exact counts in each branch, then round the weighted replicate only once.
    ct = arm[..., 1]
    diff = y[..., 1].astype(np.int8) - y[..., 0]
    sums = np.stack(((ct * diff).sum(axis=2), ((~ct) * -diff).sum(axis=2)), axis=-1)
    pairs, inverse = np.unique(sums.reshape(-1, 2), axis=0, return_inverse=True)
    prob = Fraction(seq.probability_ct)
    wct, wtc = 1 / (2 * g * prob), 1 / (2 * g * (1 - prob))
    values = np.array([_as_float(int(a) * wct + int(c) * wtc) for a, c in pairs])
    x = values[inverse].reshape(shape[:2])
    # Runtime also requires the realized inverse order weights to be representable.
    for branch, weight in ((ct, wct), (~ct, wtc)):
        if not math.isfinite(_as_float(weight * g)):
            x[branch.any(axis=2)] = np.nan
    return _paired_decisions(x, _admission(b, z), proc)


def _tail_allocation(mc_error: float, cells: int) -> float:
    tail = _float_outward(Fraction(mc_error) / (2 * cells), upper=False)
    if tail == 0:
        _invalid(
            "result", "Monte Carlo error per grid tail is below binary64 range.", field="mc_error"
        )
    return tail


@lru_cache(maxsize=256)
def _bounds(rejected: int, attempts: int, tail: float) -> tuple[float, float]:
    """Clopper-Pearson bounds via the two beta tails, including boundary atoms."""
    hits, misses = rejected, attempts - rejected
    lo = float(betaincinv(hits, misses + 1, tail)) if hits else 0.0
    hi = float(betainccinv(hits + 1, misses, tail)) if misses else 1.0
    if not (0 <= lo <= hits / attempts <= hi <= 1):
        _invalid("result", "Binomial Monte Carlo limits could not be resolved.", field="mc_error")
    return max(0.0, math.nextafter(lo, -math.inf)), min(1.0, math.nextafter(hi, math.inf))


class _Point(_Frozen):
    n: int = Field(ge=2, strict=True)
    delta: float = Field(allow_inf_nan=False)
    attempts: int = Field(ge=1, strict=True)
    admission_failures: int = Field(ge=0, strict=True)
    zero_se: int = Field(ge=0, strict=True)
    decision_failures: int = Field(ge=0, strict=True)
    rejected: int = Field(ge=0, strict=True)
    power: float = Field(ge=0, le=1)
    power_lower: float = Field(ge=0, le=1)
    power_upper: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _counts(self):
        if (
            self.admission_failures + self.zero_se + self.decision_failures + self.rejected
            > self.attempts
            or self.power != self.rejected / self.attempts
        ):
            _invalid("result", "Counts and rejection probability must use all attempts.")
        return self


class _Selection(_Frozen):
    status: Literal["attained", "not_attained", "undecided"]
    value: int | float | None


class SwitchbackProfilePlan(_Frozen):
    """Attempt counts and simultaneous power bounds on the declared finite grid only."""

    power_kind: Literal["model_based_joint_admission_and_rejection"] = (
        "model_based_joint_admission_and_rejection"
    )
    interval_method: Literal["simultaneous_bonferroni_clopper_pearson"] = (
        "simultaneous_bonferroni_clopper_pearson"
    )
    baseline: _Baseline
    procedure: ContrastDecisionProcedure
    reference: Literal["unit_t_approximation", "block_t"]
    sizes: tuple[int, ...] = Field(min_length=1)
    effects: tuple[float, ...] = Field(min_length=1)
    repetitions: int = Field(ge=1, strict=True)
    seed: int = Field(ge=0, strict=True)
    mc_error: float = Field(gt=0, lt=1)
    target_power: float = Field(gt=0, lt=1)
    points: tuple[_Point, ...]

    @model_validator(mode="after")
    def _consistent(self):
        _check_binding(self.baseline, self.procedure)
        if self.reference != (
            "block_t" if self.baseline.planning_grain == "shared_block" else "unit_t_approximation"
        ):
            _invalid("result", "Reference must match the independent planning grain.")
        cells = len(self.sizes) * len(self.effects)
        if (
            len(self.points) != cells
            or cells > _MAX_CELLS
            or self.sizes != tuple(sorted(set(self.sizes)))
            or len(set(self.effects)) != len(self.effects)
        ):
            _invalid("result", "Provide each distinct size and effect pair once.")
        tail = _tail_allocation(self.mc_error, cells)
        for row, cell in zip(self.points, product(self.sizes, self.effects), strict=True):
            if isinstance(self.baseline, SwitchbackConversionBaseline):
                _risks(self.baseline, row.delta)
            if (
                row.attempts != self.repetitions
                or (row.n, row.delta) != cell
                or (row.power_lower, row.power_upper) != _bounds(row.rejected, row.attempts, tail)
            ):
                _invalid("result", "Power limits must match the counts and the full declared grid.")
        return self

    def _first(self, candidates: Sequence[_Point], field: Literal["n", "delta"]) -> _Selection:
        for point in candidates:
            if point.power_lower >= self.target_power:
                return _Selection(status="attained", value=getattr(point, field))
            if point.power_upper >= self.target_power:
                return _Selection(status="undecided", value=None)
        return _Selection(status="not_attained", value=None)

    def grid_mde(self, n: int) -> _Selection:
        """First resolved signed effect on the favorable side, ordered outward from the null."""
        if n not in self.sizes:
            _invalid("n", "Select a sample size on the declared grid.", value=n)
        p = self.procedure
        decreasing = p.alternative == "less" or (
            p.alternative == "two-sided" and p.preferred_direction == "decrease"
        )
        rows = [
            r
            for r in self.points
            if r.n == n and (r.delta <= p.null_abs if decreasing else r.delta >= p.null_abs)
        ]
        return self._first(sorted(rows, key=lambda r: r.delta, reverse=decreasing), "delta")

    def grid_size(self, delta: float) -> _Selection:
        """First resolved independent-replicate count among the supplied sizes only."""
        if delta not in self.effects:
            _invalid("delta", "Select an effect on the declared grid.", value=delta)
        return self._first([r for r in self.points if r.delta == delta], "n")


def _windows_per_replicate(b: _Baseline) -> int:
    # A trajectory block is simulated as its roster average; conversion draws every member.
    shared = len(b.roster) if isinstance(b, SwitchbackConversionBaseline) else 1
    rows = b.cycles_per_unit if b.planning_grain == "unit" else shared
    return rows * b.assignment.periods_per_cycle


def _point(
    b: _Baseline,
    proc: ContrastDecisionProcedure,
    n: int,
    delta: float,
    reps: int,
    rng: np.random.Generator,
    tail: float,
    table: _TrajectoryTable | None,
) -> _Point:
    batch_size = min(512, _MAX_WINDOWS // (n * _windows_per_replicate(b)))
    risks = _risks(b, delta) if isinstance(b, SwitchbackConversionBaseline) else None
    admission = zero = failed = rejected = 0
    for start in range(0, reps, batch_size):
        batch = min(batch_size, reps - start)
        z = _assignments(b, rng, batch, n)
        if isinstance(b, SwitchbackTrajectoryBaseline):
            assert table is not None
            indices = rng.integers(len(b.profiles), size=(batch, n))
            draws = _trajectory_draws(b, proc, delta, indices, z, table)
        else:
            assert risks is not None
            draws = _conversion_draws(b, proc, risks, z, rng)
        admission += int(
            np.isin(draws.reason, ("missing_position_support", "paired_order_split")).sum()
        )
        zero += int(np.count_nonzero(draws.reason == "zero_se"))
        failed += int(np.count_nonzero(draws.reason == "decision_failure"))
        rejected += int(draws.rejected.sum())
    lo, hi = _bounds(rejected, reps, tail)
    return _Point(
        n=n,
        delta=delta,
        attempts=reps,
        admission_failures=admission,
        zero_se=zero,
        decision_failures=failed,
        rejected=rejected,
        power=rejected / reps,
        power_lower=lo,
        power_upper=hi,
    )


def switchback_profile_plan(
    baseline: _Baseline,
    procedure: ContrastDecisionProcedure,
    *,
    sizes: Sequence[int],
    effects: Sequence[float],
    target_power: float = 0.8,
    repetitions: int = 4096,
    seed: int = 0,
    mc_error: float = 0.05,
) -> SwitchbackProfilePlan:
    """Estimate fitted-model power over candidate independent sizes and signed effects.

    Every attempted experiment counts, including refused schedules and zero
    standard errors. The simultaneous binomial limits condition on the pilot
    and the prespecified grid. They do not establish type-I control.
    """
    if not isinstance(baseline, (SwitchbackTrajectoryBaseline, SwitchbackConversionBaseline)):
        _invalid("baseline", "Use a complete trajectory or retained-window conversion baseline.")
    baseline = type(baseline).model_validate(baseline)
    if not isinstance(procedure, ContrastDecisionProcedure):
        _invalid("procedure", "Provide a fixed-horizon switchback decision procedure.")
    procedure = ContrastDecisionProcedure.model_validate(procedure)
    _check_binding(baseline, procedure)
    targets = _finite(target_power, "target_power"), _finite(mc_error, "mc_error")
    if not all(0 < p < 1 for p in targets):
        _invalid(
            "target_power",
            "Target power and Monte Carlo error must be strictly between zero and one.",
        )
    repetitions, seed = _count(repetitions, "repetitions", 1), _count(seed, "seed", 0)
    try:
        cell_count = len(sizes) * len(effects)
    except (OverflowError, TypeError):
        _invalid("n", "Sizes and effects must be finite sequences.")
    if not 1 <= cell_count <= _MAX_CELLS:
        _invalid("n", f"Supply a nonempty size by effect grid of at most {_MAX_CELLS} points.")
    ns = tuple(sorted(_count(n, "sizes", 2) for n in sizes))
    ds = tuple(_finite(d, "effects") for d in effects)
    if ns[-1] * _windows_per_replicate(baseline) > _MAX_WINDOWS:
        _invalid(
            "n",
            f"One simulated experiment exceeds {_MAX_WINDOWS} retained windows.",
            value=ns[-1],
            route="Reduce the candidate size or use an external design simulation.",
        )
    if repetitions > 2**53:
        _invalid(
            "n",
            "Repetitions must be representable as an exact binary64 count.",
            field="repetitions",
        )
    if len(set(ns)) != len(ns) or len(set(ds)) != len(ds):
        _invalid("result", "Every sample size and effect must be distinct on its own axis.")
    if isinstance(baseline, SwitchbackConversionBaseline):
        for d in ds:
            _risks(baseline, d)
    tail = _tail_allocation(targets[1], len(ns) * len(ds))
    rng = np.random.Generator(np.random.PCG64(seed))
    table = (
        _trajectory_table(baseline) if isinstance(baseline, SwitchbackTrajectoryBaseline) else None
    )
    rows: dict[tuple[int, float], _Point] = {}
    for d in ds:
        for n in ns:
            rows[n, d] = _point(baseline, procedure, n, d, repetitions, rng, tail, table)
    points = tuple(rows[cell] for cell in product(ns, ds))
    return SwitchbackProfilePlan(
        baseline=baseline,
        procedure=procedure,
        reference="block_t"
        if baseline.planning_grain == "shared_block"
        else "unit_t_approximation",
        sizes=ns,
        effects=ds,
        repetitions=repetitions,
        seed=seed,
        mc_error=targets[1],
        target_power=targets[0],
        points=points,
    )


__all__ = [
    "SwitchbackConversionBaseline",
    "SwitchbackProfilePlan",
    "SwitchbackTrajectoryBaseline",
    "switchback_profile_plan",
]
