"""Eager narwhals switchback source construction.

A switchback frame is deliberately narrower than the ordinary frame sources:
its rows are one unit x cycle x period x step cell, and its only estimand is a
fixed additive contrast over a declared retained window. This module never
imports a warehouse adapter or a dataframe backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import narwhals as nw
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._study import (
    SWITCHBACK_IDENTIFICATION,
    SwitchbackStudyEnvelope,
    allocation_defect,
    refuse_allocation_defect,
)
from increment._unit_cycle import (
    _refuse as _unit_refuse,
)
from increment._unit_cycle import (
    unit_cycle_admission_outcome,
    unit_cycle_envelope_cutoff,
)
from increment.decision import ContrastContext, compile_contrast_procedures
from increment.errors import (
    CapabilityError,
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    refuse,
)
from increment.estimation.contrast import (
    ContrastPartition,
    ContrastStats,
    _exact_sum,
    reduce_contrast_partitions,
)
from increment.estimation.contrast_results import (
    IndependenceGrain,
    RandomizationLaw,
    _validate_assignment_metadata,
    _validate_order_counts,
    _validate_replication_counts,
    _validate_retained_window,
)
from increment.semantics.assignment import SwitchbackAssignment
from increment.semantics.design import Randomized
from increment.semantics.unit_cycle import (
    UnitCycleReference,
    UnitCycleTApproximation,
    UnitCycleVarianceEnvelope,
    _copy_references,
)

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._frame_validation import SwitchbackSchedule
    from increment._metric_specs import MetricsArg, MetricSpec
    from increment.power.switchback import SwitchbackBaseline
    from increment.semantics.models import AnalysisPlan, Metric


def _render_message(*, message: str, **_: object) -> str:
    return message


_SWITCHBACK_ASSIGNMENT = RefusalSpec(
    "source.frame.switchback.assignment",
    InvalidRequestError,
    _render_message,
)
_SWITCHBACK_METRIC = RefusalSpec(
    "source.frame.switchback.metric",
    CapabilityError,
    _render_message,
)
_SWITCHBACK_COLUMNS = RefusalSpec(
    "source.frame.switchback.columns",
    InvalidRequestError,
    _render_message,
)
_SWITCHBACK_DOMAIN = RefusalSpec(
    "source.frame.switchback.domain",
    InvalidRequestError,
    _render_message,
)
_SWITCHBACK_SCHEDULE = RefusalSpec(
    "source.frame.switchback.schedule",
    InvalidRequestError,
    _render_message,
)
_SWITCHBACK_MISSINGNESS = RefusalSpec(
    "source.frame.switchback.missingness",
    InvalidRequestError,
    _render_message,
)
_SWITCHBACK_NUMERIC = RefusalSpec(
    "source.frame.switchback.numeric",
    InvalidRequestError,
    _render_message,
)
_SWITCHBACK_UNITS = RefusalSpec(
    "source.frame.switchback.units",
    InvalidRequestError,
    _render_message,
)


def _reject(
    spec: RefusalSpec,
    *,
    message: str,
    **context: object,
) -> NoReturn:
    refuse(spec, message=message, **context)


def _reject_assignment(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_ASSIGNMENT, message=message, **context)


def _reject_identification(*, message: str, **context: object) -> NoReturn:
    _reject(SWITCHBACK_IDENTIFICATION, message=message, **context)


def _reject_metric(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_METRIC, message=message, **context)


def _reject_columns(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_COLUMNS, message=message, **context)


def _reject_domain(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_DOMAIN, message=message, **context)


def _reject_schedule(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_SCHEDULE, message=message, **context)


def _reject_missingness(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_MISSINGNESS, message=message, **context)


def _reject_numeric(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_NUMERIC, message=message, **context)


def _reject_units(*, message: str, **context: object) -> NoReturn:
    _reject(_SWITCHBACK_UNITS, message=message, **context)


class SwitchbackAssignmentDiagnostic(CodedModel, BaseModel):
    """Construction-time schedule and assignment integrity evidence.

    ``ct_cycles``/``tc_cycles`` count the realized independent CT/TC order
    draws: one per unit-cycle under ``independent_bernoulli_order``, or one
    per two-period block (shared by the whole roster) under
    ``shared_schedule``. ``n_blocks`` is ``None`` for the unit-cycle law and
    the declared block count for a shared schedule. ``n_units`` is always
    the fixed roster size, never the inferential replicate count for a
    shared law.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    probability_ct: float = Field(gt=0.0, lt=1.0)
    randomization_law: RandomizationLaw
    independence_grain: IndependenceGrain
    schedule_complete: bool
    n_units: int = Field(ge=1)
    n_cycles: int = Field(ge=1)
    ct_cycles: int = Field(ge=0, strict=True)
    tc_cycles: int = Field(ge=0, strict=True)
    n_blocks: int | None = Field(default=None, ge=1, strict=True)
    washout_steps: int = Field(ge=0)
    observation_steps: int = Field(ge=1)
    retained_steps: int = Field(ge=1)
    washout_rows: int = Field(ge=0)
    observation_rows: int = Field(ge=1)
    retained_rows: int = Field(ge=1)
    carryover_order: int = Field(ge=0)
    identifying_assumption: Literal["no_residual_carryover_after_discarded_steps"] = (
        "no_residual_carryover_after_discarded_steps"
    )
    integrity_failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent_design(self) -> SwitchbackAssignmentDiagnostic:
        _validate_assignment_metadata(self.randomization_law, self.independence_grain)
        _validate_replication_counts(
            self.randomization_law, self.n_units, self.n_cycles, self.n_blocks
        )
        _validate_order_counts(
            self.randomization_law, self.n_cycles, self.n_blocks, self.ct_cycles, self.tc_cycles
        )
        _validate_retained_window(self.observation_steps, self.retained_steps, self.carryover_order)
        return self

    def __getitem__(self, name: str) -> object:
        """Permit the compact ``diagnostics["field"]`` inspection style."""
        return getattr(self, name)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()

    @property
    def ct_count(self) -> int:
        return self.ct_cycles

    @property
    def tc_count(self) -> int:
        return self.tc_cycles


@dataclass(frozen=True, slots=True)
class _PlanningMoments:
    sd_a: float
    sd_g: float
    rho: float


def _planning_moments(
    partition: ContrastPartition, stats: ContrastStats, slopes: Mapping[str, float]
) -> _PlanningMoments | None:
    keys = sorted(partition.unit_deltas)
    if len(keys) < 2:
        return None
    reference_g = slopes[keys[0]]
    offsets_g = [slopes[key] - reference_g for key in keys]
    mean_g = math.fsum(offsets_g) / len(keys)
    residuals_g = [value - mean_g for value in offsets_g]
    norm_g = math.hypot(*residuals_g)
    norm_a = math.sqrt(stats.m2_delta)
    divisor = math.sqrt(len(keys) - 1)
    rho = 0.0
    if norm_a and norm_g:
        rho = math.fsum(
            ((partition.unit_deltas[key] - stats.reference_delta - stats.mean_residual) / norm_a)
            * (residual_g / norm_g)
            for key, residual_g in zip(keys, residuals_g, strict=True)
        )
        # A unit-vector dot product can round just outside its mathematical range.
        rho = max(-1.0, min(1.0, rho))
    return _PlanningMoments(norm_a / divisor, norm_g / divisor, rho)


class FrameSwitchbackSource:
    """A validated eager frame for fixed switchback contrasts.

    The source stores reduced unit or block contrast contributions, immutable
    contrast context, and construction diagnostics. Raw panel rows are
    released after eager validation and reduction.
    """

    def __init__(
        self,
        *,
        context: ContrastContext,
        diagnostics: SwitchbackAssignmentDiagnostic,
        stats: Mapping[str, ContrastStats],
        planning: Mapping[str, _PlanningMoments | None],
        roster: Sequence[str],
    ) -> None:
        self._context = context
        self._diagnostics = diagnostics
        self._stats = dict(stats)
        self._planning = dict(planning)
        self._roster = tuple(roster)

    @property
    def context(self) -> ContrastContext:
        return self._context

    @property
    def contrast_references(self) -> Mapping[str, UnitCycleReference]:
        return self._context.contrast_references

    @property
    def diagnostics(self) -> SwitchbackAssignmentDiagnostic:
        return self._diagnostics

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return self._context.metrics

    def contrast_stats(self, metric: Metric) -> ContrastStats:
        """Return contrast statistics reduced over independent units or shared blocks."""
        try:
            return self._stats[metric.name]
        except KeyError:
            declared = sorted(self._stats)
            _reject_metric(
                message=(
                    f"metric {metric.name!r} was not declared on this switchback source; "
                    f"declared metrics: {declared!r}"
                ),
                metric=metric.name,
                declared=declared,
                reason="undeclared_metric",
            )

    def planning_baseline(
        self, metric: Metric, *, delta_ref: float | None = None
    ) -> SwitchbackBaseline:
        """Estimate a pilot-fitted additive-shift planning baseline.

        When ``delta_ref`` is omitted, the observed pilot mean is used as the
        reference effect.  This is a model-conditioned planning approximation:
        the centered pilot covariance is retained at the independent unit/block
        grain, while the fitted reference is not treated as a population
        certificate.
        """
        from increment.power.switchback import SwitchbackBaseline

        stats = self.contrast_stats(metric)
        moments = self._planning[metric.name]
        shared = stats.randomization_law == "shared_schedule"
        if moments is None:
            _reject_units(
                message="switchback pilot planning requires at least two independent replicates",
                n_replicates=stats.n_blocks if shared else stats.n_units,
                reason="planning_requires_replicates",
            )
        assert moments is not None
        fitted_reference = (
            float(math.fsum((stats.reference_delta, stats.mean_residual)))
            if delta_ref is None
            else delta_ref
        )
        if not math.isfinite(fitted_reference):
            _reject_numeric(
                message="switchback pilot reference effect must be finite",
                metric=metric.name,
                reason="nonfinite_reference_effect",
            )
        return SwitchbackBaseline(
            assignment=self.context.study.assignment,
            metric=stats.metric,
            control_group=stats.control_group,
            treatment_group=stats.treatment_group,
            aggregation=stats.aggregation,
            estimand=(
                "retained_window_total_difference"
                if stats.aggregation == "sum"
                else "retained_window_conversion_difference"
            ),
            cycles_per_unit=None if shared else stats.n_cycles // stats.n_units,
            shared_roster=self._roster if shared else None,
            delta_ref=fitted_reference,
            sd_a=moments.sd_a,
            sd_g=moments.sd_g,
            moment_source="pilot_estimated",
            rho=moments.rho,
        )

    def diagnostic_report(self) -> dict[str, object]:
        """Return diagnostics as a plain immutable-state copy for reporting."""
        return self._diagnostics.as_dict()

    def close(self) -> None:
        pass


def _exact_signed_total(blocks: Sequence[tuple[int, Iterable[Any]]]) -> Fraction:
    """Accumulate signed scalar blocks without allocating a Fraction per scalar.

    Each block contributes ``sign`` times every one of its scalars. Signing the
    integer numerator keeps the accumulation exact: a rational sum does not
    depend on how its terms are grouped, so one signed accumulation replaces a
    per-term Fraction without moving a single bit of the reduced result.
    """

    def ratios() -> Iterator[tuple[int, int]]:
        for sign, values in blocks:
            for value in values:
                if type(value) in (int, float):
                    numerator, denominator = value.as_integer_ratio()
                else:
                    try:
                        exact = Fraction(value)
                    except (TypeError, ValueError, OverflowError):
                        exact = Fraction(float(value))
                    numerator, denominator = int(exact.numerator), int(exact.denominator)
                yield (-numerator if sign < 0 else numerator, denominator)

    return _exact_sum(ratios())


def _exact_group_sums(
    values: np.ndarray, signs: np.ndarray, groups: np.ndarray, n_groups: int
) -> list[Fraction]:
    """Exact signed sums of numeric array scalars, one rational per group.

    Every finite double is an integer mantissa times a power of two, and every
    integer is its own mantissa. Mantissas are summed per group and exponent in
    32-bit halves, which cannot overflow, and the buckets are combined with
    Python integers: a rational sum does not depend on how its terms are grouped.
    """
    if values.dtype.kind == "f":
        fraction, binary_exponent = np.frexp(values.astype(np.float64, copy=False))
        mantissa = np.ldexp(fraction, 53).astype(np.int64)
        exponent = binary_exponent.astype(np.int64) - 53
    elif values.dtype == np.uint64:
        # Two 32-bit halves per value both fit a signed mantissa.
        mantissa = np.concatenate(
            (
                (values >> np.uint64(32)).astype(np.int64),
                (values & np.uint64(0xFFFFFFFF)).astype(np.int64),
            )
        )
        exponent = np.repeat(np.array([32, 0], dtype=np.int64), values.size)
        signs = np.concatenate((signs, signs))
        groups = np.concatenate((groups, groups))
    else:
        mantissa = values.astype(np.int64)
        exponent = np.zeros(values.size, dtype=np.int64)
    if mantissa.size == 0:
        return [Fraction(0)] * n_groups
    high = (mantissa >> 32) * signs
    low = (mantissa & 0xFFFFFFFF) * signs
    e_min = int(exponent.min())
    span = int(exponent.max()) - e_min + 1
    buckets, inverse = np.unique(
        groups.astype(np.int64) * span + (exponent - e_min), return_inverse=True
    )
    high_sums = np.zeros(buckets.size, dtype=np.int64)
    low_sums = np.zeros(buckets.size, dtype=np.int64)
    np.add.at(high_sums, inverse, high)
    np.add.at(low_sums, inverse, low)
    totals = [0] * n_groups
    for bucket, high_sum, low_sum in zip(
        buckets.tolist(), high_sums.tolist(), low_sums.tolist(), strict=True
    ):
        group, shift = divmod(bucket, span)
        totals[group] += ((high_sum << 32) + low_sum) << shift
    if e_min < 0:
        return [Fraction(total, 1 << -e_min) for total in totals]
    return [Fraction(total << e_min) for total in totals]


def _exact_branch_sums(
    values: np.ndarray | Sequence[Any], control_first: np.ndarray, rows: int
) -> tuple[list[Fraction], list[Fraction]]:
    """Exact per-unit signed sums of retained scalars for the CT and TC branches.

    ``values`` holds every retained scalar in block order: unit, cycle, period,
    then step. Under a CT order the second period gains and the first loses;
    under TC the reverse. Each branch sum is the exact rational a unit's HT
    contribution is built from, before the fixed branch weight is applied.
    """
    n_units, n_cycles = control_first.shape
    if isinstance(values, np.ndarray):
        shape = (n_units, n_cycles, 2, rows)
        gains = control_first[:, :, None, None] == np.array([False, True])[None, None, :, None]
        branch = np.where(control_first, 0, 1)[:, :, None, None]
        group = np.arange(n_units)[:, None, None, None] * 2 + branch
        sums = _exact_group_sums(
            np.reshape(values, shape).ravel(),
            np.broadcast_to(np.where(gains, 1, -1), shape).ravel(),
            np.broadcast_to(group, shape).ravel(),
            2 * n_units,
        )
        return sums[0::2], sums[1::2]
    ct_sums: list[Fraction] = []
    tc_sums: list[Fraction] = []
    for index in range(n_units):
        ct_gain: list[Any] = []
        ct_loss: list[Any] = []
        tc_gain: list[Any] = []
        tc_loss: list[Any] = []
        for offset in range(n_cycles):
            block = (index * n_cycles + offset) * 2
            first = values[block * rows : (block + 1) * rows]
            second = values[(block + 1) * rows : (block + 2) * rows]
            if control_first[index, offset]:
                ct_gain += second
                ct_loss += first
            else:
                tc_gain += first
                tc_loss += second
        ct_sums.append(_exact_signed_total(((1, ct_gain), (-1, ct_loss))))
        tc_sums.append(_exact_signed_total(((1, tc_gain), (-1, tc_loss))))
    return ct_sums, tc_sums


@dataclass(frozen=True, slots=True)
class _RetainedBlocks:
    """Retained observation rows in unit-cycle-period block order.

    ``units`` lists the distinct unit labels in frame-engine sort order; blocks
    follow that order, then ascending cycle, then period, and every block holds
    ``rows`` retained steps in their original frame order because the schedule
    was validated complete. ``values`` carries each metric's retained scalars in
    that block order: a numeric array for float and integer columns, otherwise
    the raw Python scalars. ``numeric`` names the metrics whose source column is
    already numeric and therefore needs no scalar coercion.
    """

    units: tuple[object, ...]
    cycles: Sequence[int]
    rows: int
    values: dict[str, np.ndarray | list[Any]]
    numeric: frozenset[str]

    def key(self, block: int) -> tuple[object, int, int]:
        """The unit, cycle and period of one block, for refusal context."""
        index, rest = divmod(block, len(self.cycles) * 2)
        offset, period = divmod(rest, 2)
        return self.units[index], int(self.cycles[offset]), period


def _retained_blocks(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    cycle: str,
    period: str,
    step: str,
    specs: Sequence[MetricSpec],
    washout_steps: int,
    observation_steps: int,
    carryover_order: int,
    cycles: Sequence[int],
) -> _RetainedBlocks:
    """Group retained observation steps by unit-cycle-period.

    Retained steps satisfy ``step >= washout_steps + carryover_order``: the
    declared washout, plus any additional post-washout steps still assumed
    contaminated, are both discarded before grouping.

    Discarding and ordering run in the frame engine over whole columns.
    Ordering by the key columns and then by the row index leaves every block
    holding exactly the rows it would hold under a row-wise scan, in the same
    order, so the reductions built on top of these blocks are unchanged.
    """
    keys = [unit, cycle, period]
    columns: list[str] = []
    for column in (*keys, step, *(spec.y_column for spec in specs)):
        if column not in columns:
            columns.append(column)
    schema = frame.schema
    selected = frame.select(*columns)
    retain_from_step = washout_steps + carryover_order
    if retain_from_step:
        selected = selected.filter(nw.col(step) >= retain_from_step)
    ordered = selected.with_row_index("__row_order__").sort(*keys, "__row_order__")
    rows = observation_steps - carryover_order
    per_unit = len(cycles) * 2 * rows
    # Blocks are uniform: _validate_switchback_domains fixed the step domain and
    # _validate_switchback_schedule required every unit-cycle-period cell once.
    units = tuple(ordered[unit].gather_every(per_unit).to_list())
    extracted: dict[str, np.ndarray | list[Any]] = {}
    values: dict[str, np.ndarray | list[Any]] = {}
    for spec in specs:
        column = spec.y_column
        if column not in extracted:
            dtype = schema[column]
            series = ordered[column]
            extracted[column] = (
                series.to_numpy() if dtype.is_float() or dtype.is_integer() else series.to_list()
            )
        values[spec.name] = extracted[column]
    return _RetainedBlocks(
        units=units,
        cycles=cycles,
        rows=rows,
        values=values,
        numeric=frozenset(spec.name for spec in specs if schema[spec.y_column].is_numeric()),
    )


def _reduce_blocks(
    blocks: _RetainedBlocks,
    specs: Sequence[MetricSpec],
    reject: Callable[..., NoReturn],
) -> dict[str, list[float | bool]]:
    """Collapse each retained block to one value per metric, in block order."""
    rows = blocks.rows
    reduced: dict[str, list[float | bool]] = {}
    for spec in specs:
        observations = blocks.values[spec.name]
        if isinstance(observations, np.ndarray):
            table = np.reshape(observations, (-1, rows))
            if spec.type == "conversion":
                reduced[spec.name] = table.astype(bool).any(axis=1).tolist()
                continue
            groups: list[list[Any]] = table.tolist()
        else:
            groups = [
                list(observations[start : start + rows])
                for start in range(0, len(observations), rows)
            ]
            if spec.type == "conversion":
                reduced[spec.name] = [any(bool(value) for value in group) for group in groups]
                continue
            if spec.name not in blocks.numeric:
                groups = [[float(value) for value in group] for group in groups]
        try:
            totals: list[float] | None = list(map(math.fsum, groups))
        except OverflowError:
            totals = None
        if totals is None or not all(map(math.isfinite, totals)):
            # Walk the blocks in order so the refusal names the first offender.
            for block, group in enumerate(groups):
                try:
                    total = math.fsum(group)
                except OverflowError:
                    unit_value, cycle_value, period_value = blocks.key(block)
                    reject(
                        message=f"metric {spec.name!r}: aggregated switchback outcome overflowed",
                        metric=spec.name,
                        unit=unit_value,
                        cycle=cycle_value,
                        period=period_value,
                        reason="aggregated_outcome_overflow",
                    )
                if not math.isfinite(total):
                    unit_value, cycle_value, period_value = blocks.key(block)
                    reject(
                        message=f"metric {spec.name!r}: aggregated switchback outcome is not finite",
                        metric=spec.name,
                        unit=unit_value,
                        cycle=cycle_value,
                        period=period_value,
                        reason="aggregated_outcome_nonfinite",
                    )
        assert totals is not None
        reduced[spec.name] = list(totals)
    return reduced


def _unit_cycle_contribution(
    first: Any, second: Any, ct: Any, probability_ct: float
) -> tuple[Any, Any]:
    """HT contribution and additive-shift slope, scalar or broadcast arrays.

    Inputs are already retained period totals. Average cycles within each unit
    before reducing independent units; the arrays do not declare extra N.
    """
    if (
        isinstance(ct, (bool, np.bool_))
        and type(first) is float
        and type(second) is float
        and 0.0 < probability_ct < 1.0
    ):
        # Retained period totals are Python floats; keep array/scalar coercion
        # and exceptional-input behavior in the NumPy path below.
        weight = 2.0 * probability_ct if ct else 2.0 * (1.0 - probability_ct)
        difference = second - first if ct else first - second
        return difference / weight, 1.0 / weight
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weight = np.where(ct, 2.0 * probability_ct, 2.0 * (1.0 - probability_ct))
        difference = np.where(ct, np.asarray(second) - first, np.asarray(first) - second)
        return difference / weight, 1.0 / weight


def _unit_cycle_average(contributions: Sequence[float]) -> float:
    """Correctly rounded mean of one unit's cycle contributions.

    Partial sums that overflow are accumulated exactly instead, so only a mean
    that is itself unrepresentable raises OverflowError.
    """
    try:
        return math.fsum(contributions) / len(contributions)
    except OverflowError:
        return float(sum(map(Fraction, contributions), Fraction(0)) / len(contributions))


def _weighted_exact_average(
    ct_sum: Fraction, tc_sum: Fraction, w_ct: Fraction, w_tc: Fraction, n_cycles: int
) -> Fraction:
    """``(ct_sum * w_ct + tc_sum * w_tc) / n_cycles`` reduced once, not per operation."""
    ct_denominator = ct_sum.denominator * w_ct.denominator
    tc_denominator = tc_sum.denominator * w_tc.denominator
    return Fraction(
        ct_sum.numerator * w_ct.numerator * tc_denominator
        + tc_sum.numerator * w_tc.numerator * ct_denominator,
        ct_denominator * tc_denominator * n_cycles,
    )


def _unit_order(units: Sequence[object], reject_units: Callable[..., NoReturn]) -> list[int]:
    """Indices of ``units`` in string-token order, refusing colliding tokens."""
    order = sorted(range(len(units)), key=lambda index: str(units[index]))
    unit_tokens: dict[str, object] = {}
    for index in order:
        value = units[index]
        token = str(value)
        prior = unit_tokens.get(token)
        if prior is not None and prior != value:
            reject_units(
                message=(
                    f"switchback unit identities are unstable after string conversion: {value!r}"
                ),
                unit=value,
                reason="unstable_unit_identity",
            )
        unit_tokens[token] = value
    return order


def _switchback_groups(identification: Randomized) -> tuple[str, str]:
    allocation = identification.allocation
    control = str(identification.control_group)
    treatment_arms = [arm for arm in allocation if arm != control] if allocation is not None else []
    if allocation is None or control not in allocation or len(treatment_arms) != 1:
        _reject_identification(
            message=(
                "switchback identification allocation must contain exactly "
                "its control and one treatment weight"
            ),
            allocation=allocation,
            control_group=control,
            reason="invalid_allocation",
        )
    treatment = str(treatment_arms[0])
    defect = allocation_defect(allocation[control], allocation[treatment])
    if defect is not None:
        refuse_allocation_defect(
            defect, allocation=allocation, control_group=control, treatment_group=treatment
        )
    return control, treatment


def _validated_unit_cycle_references(
    references: Mapping[str, UnitCycleReference] | None,
    specs: Sequence[MetricSpec],
    assignment: SwitchbackAssignment,
    control: str,
    treatment: str,
) -> Mapping[str, UnitCycleReference]:
    copied = _copy_references(references)
    if not copied and assignment.sequence.scheme == "independent_bernoulli_order":
        # Ordinary callers receive a qualified unit-t approximation from the
        # declared design; complete envelopes remain research inputs.
        copied = {spec.name: UnitCycleTApproximation() for spec in specs}
    declared_metrics = {spec.name: spec for spec in specs}
    for name, reference in copied.items():
        if name not in declared_metrics:
            _unit_refuse("unit_cycle.reference_mismatch", reason="undeclared_metric", metric=name)
        if assignment.sequence.scheme != "independent_bernoulli_order":
            _unit_refuse(
                "unit_cycle.reference_mismatch", reason="unit_reference_on_shared_schedule"
            )
        if isinstance(reference, UnitCycleVarianceEnvelope):
            if (
                reference.assignment != assignment
                or reference.control_group != control
                or reference.treatment_group != treatment
            ):
                _unit_refuse("unit_cycle.reference_mismatch", reason="assignment_or_groups")
            if declared_metrics[name].type != "mean":
                _unit_refuse(
                    "unit_cycle.reference_mismatch", reason="additive_conversion_unsupported"
                )
    return copied


def _aligned_orders(schedule: SwitchbackSchedule, units: Sequence[object]) -> np.ndarray:
    """Realized orders re-indexed to the retained blocks' unit order."""
    position = {value: index for index, value in enumerate(schedule.units)}
    return schedule.control_first[[position[value] for value in units]]


def _build_stats(  # noqa: PLR0913
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    cycle: str,
    period: str,
    step: str,
    specs: Sequence[MetricSpec],
    control: str,
    treatment: str,
    probability_ct: float,
    randomization_law: RandomizationLaw,
    independence_grain: IndependenceGrain,
    carryover_order: int,
    observation_steps: int,
    washout_steps: int,
    cycles: Sequence[int],
    schedule: SwitchbackSchedule,
    reject_numeric: Callable[..., NoReturn],
    reject_units: Callable[..., NoReturn],
) -> tuple[dict[str, ContrastStats], dict[str, _PlanningMoments | None]]:
    blocks = _retained_blocks(
        frame,
        unit=unit,
        cycle=cycle,
        period=period,
        step=step,
        specs=specs,
        washout_steps=washout_steps,
        observation_steps=observation_steps,
        carryover_order=carryover_order,
        cycles=cycles,
    )
    reduced = _reduce_blocks(blocks, specs, reject_numeric)
    n_units, n_cycles = len(blocks.units), len(cycles)
    ct = _aligned_orders(schedule, blocks.units)
    order = _unit_order(blocks.units, reject_units)
    tokens = [str(value) for value in blocks.units]
    planning: dict[str, _PlanningMoments | None] = {}

    # The realized order is a property of the schedule, not of any one metric.
    control_slope = 1.0 / (2.0 * probability_ct)
    treatment_slope = 1.0 / (2.0 * (1.0 - probability_ct))
    slope_terms = np.where(ct, control_slope, treatment_slope).tolist()
    slopes: dict[str, float] = {}
    for index in order:
        try:
            slope = _unit_cycle_average(slope_terms[index])
        except (OverflowError, ValueError):
            reject_numeric(
                message="unit-cycle slope cannot be represented",
                unit=blocks.units[index],
                reason="unit_slope_nonfinite",
            )
        if not math.isfinite(slope) or slope <= 0:
            reject_numeric(
                message="unit-cycle slope cannot be represented",
                unit=blocks.units[index],
                reason="unit_slope_nonfinite",
            )
        slopes[tokens[index]] = slope
    ct_counts_by_unit = {tokens[index]: int(ct[index].sum()) for index in order}

    p = Fraction(probability_ct)
    w_ct, w_tc = 1 / (2 * p), 1 / (2 * (1 - p))
    stats: dict[str, ContrastStats] = {}
    for spec in specs:
        block_totals = reduced[spec.name]
        totals = np.asarray(block_totals, dtype=float).reshape(n_units, n_cycles, 2)
        contributions, _ = _unit_cycle_contribution(
            totals[..., 0], totals[..., 1], ct, probability_ct
        )
        finite = np.isfinite(contributions)
        if not finite.all():
            index = next(index for index in order if not finite[index].all())
            reject_numeric(
                message=f"metric {spec.name!r}: inverse-probability contribution is not finite",
                metric=spec.name,
                unit=blocks.units[index],
                cycle=cycles[int(np.argmin(finite[index]))],
                reason="inverse_probability_nonfinite",
            )
        if spec.type == "conversion":
            exact_values: np.ndarray | Sequence[Any] = np.asarray(block_totals, dtype=np.int64)
            exact_rows = 1
        else:
            exact_values = blocks.values[spec.name]
            exact_rows = blocks.rows
        ct_sums, tc_sums = _exact_branch_sums(exact_values, ct, exact_rows)
        deltas: dict[str, float] = {}
        exact_deltas: dict[str, tuple[int, int]] = {}
        cycles_by_unit: dict[str, int] = {}
        for index in order:
            token = tokens[index]
            try:
                # Round the exact weighted contrast once, after cancellation.
                exact_average = _weighted_exact_average(
                    ct_sums[index], tc_sums[index], w_ct, w_tc, n_cycles
                )
                cycle_average = float(exact_average)
            except OverflowError:
                reject_numeric(
                    message=(
                        f"metric {spec.name!r}: cycle contributions overflowed during aggregation"
                    ),
                    metric=spec.name,
                    unit=blocks.units[index],
                    reason="cycle_contribution_overflow",
                )
            deltas[token] = cycle_average
            exact_deltas[token] = (exact_average.numerator, exact_average.denominator)
            cycles_by_unit[token] = n_cycles

        partition = ContrastPartition(
            metric=spec.name,
            aggregation="any" if spec.type == "conversion" else "sum",
            probability_ct=probability_ct,
            randomization_law=randomization_law,
            independence_grain=independence_grain,
            carryover_order=carryover_order,
            observation_steps=observation_steps,
            retained_steps=observation_steps - carryover_order,
            control_group=control,
            treatment_group=treatment,
            unit_deltas=deltas,
            exact_unit_deltas=exact_deltas,
            cycles_by_unit=cycles_by_unit,
            unit_slopes=slopes,
            ct_counts_by_unit=ct_counts_by_unit,
            washout_steps=washout_steps,
        )
        try:
            stats[spec.name] = reduce_contrast_partitions([partition])
            planning[spec.name] = _planning_moments(partition, stats[spec.name], slopes)
        except (OverflowError, ValueError):
            reject_numeric(
                message=(
                    f"metric {spec.name!r}: centered contrast reduction overflowed "
                    "or produced non-finite moments"
                ),
                metric=spec.name,
                reason="centered_contrast_reduction_overflow",
            )
    return stats, planning


def _validate_shared_block_orders(
    schedule: SwitchbackSchedule,
    *,
    control: str,
    treatment: str,
    cycles: Sequence[int],
    reject_schedule: Callable[..., NoReturn],
) -> np.ndarray:
    """Require one identical realized CT/TC order per two-period block.

    A shared schedule draws one order per block for the whole fixed roster.
    ``_validate_switchback_schedule`` already confirmed every unit-cycle-period
    cell holds exactly one of ``control``/``treatment`` with no third arm;
    this additionally requires that order to agree across every unit in the
    roster for a given block, and returns the agreed CT flag per block.
    """
    control_first = schedule.control_first
    agreed = control_first.all(axis=0) | (~control_first).all(axis=0)
    if not agreed.all():
        current_cycle = cycles[int(np.argmin(agreed))]
        sequences = {(control, treatment), (treatment, control)}
        reject_schedule(
            message=(
                "shared switchback schedule requires the identical realized "
                f"CT/TC order for every unit in block (cycle) {current_cycle}; "
                f"observed {sorted(str(seq) for seq in sequences)!r}"
            ),
            cycle=current_cycle,
            reason="shared_block_order_mismatch",
        )
    return control_first[0]


# Shared blocks reduce their complete roster before independent-block inference.
def _build_shared_stats(  # noqa: PLR0913
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    cycle: str,
    period: str,
    step: str,
    specs: Sequence[MetricSpec],
    control: str,
    treatment: str,
    probability_ct: float,
    randomization_law: RandomizationLaw,
    independence_grain: IndependenceGrain,
    carryover_order: int,
    observation_steps: int,
    washout_steps: int,
    cycles: Sequence[int],
    block_control_first: np.ndarray,
    reject_numeric: Callable[..., NoReturn],
) -> tuple[dict[str, ContrastStats], dict[str, _PlanningMoments | None]]:
    """Average the fixed, complete unit roster before the shared HT reduction.

    Each block's ``X_b`` is the roster-mean-unit retained-window additive
    treatment effect: ``Z_b * D_CT / (2p) + (1 - Z_b) * D_TC / (2 * (1 - p))``,
    with ``D_CT = M2 - M1`` and ``D_TC = M1 - M2``. Cloning the roster
    changes neither ``X_b`` nor the number of independent blocks.

    Signed period levels are accumulated together before roster/IPW scaling;
    exact accumulation handles otherwise avoidable intermediate overflow.
    """
    blocks = _retained_blocks(
        frame,
        unit=unit,
        cycle=cycle,
        period=period,
        step=step,
        specs=specs,
        washout_steps=washout_steps,
        observation_steps=observation_steps,
        carryover_order=carryover_order,
        cycles=cycles,
    )
    reduced = _reduce_blocks(blocks, specs, reject_numeric)
    n_units, n_cycles = len(blocks.units), len(cycles)
    roster = sorted(range(n_units), key=lambda index: str(blocks.units[index]))
    block_ct = block_control_first.tolist()
    slopes = {
        str(current_cycle): 1.0 / (2.0 * (probability_ct if is_ct else 1.0 - probability_ct))
        for current_cycle, is_ct in zip(cycles, block_ct, strict=True)
    }
    stats: dict[str, ContrastStats] = {}
    planning: dict[str, _PlanningMoments | None] = {}
    exact_probability = Fraction(probability_ct)
    exact_weights = [
        2 * exact_probability if is_ct else -2 * (1 - exact_probability) for is_ct in block_ct
    ]
    for spec in specs:
        block_totals = reduced[spec.name]
        observations: np.ndarray | Sequence[Any]
        rows: int
        if spec.type == "conversion":
            observations = np.asarray(block_totals, dtype=np.int64)
            rows = 1
        else:
            observations = blocks.values[spec.name]
            rows = blocks.rows
        block_deltas: dict[str, float] = {}
        block_sizes: dict[str, int] = {}
        for offset, current_cycle in enumerate(cycles):
            try:
                exact_total = Fraction(0)
                for index in roster:
                    block = (index * n_cycles + offset) * 2
                    first = observations[block * rows : (block + 1) * rows]
                    second = observations[(block + 1) * rows : (block + 2) * rows]
                    exact_total += _exact_signed_total(((1, second), (-1, first)))
                exact_delta = exact_total / (n_units * exact_weights[offset])
                x_b = float(exact_delta)
            except OverflowError:
                reject_numeric(
                    message=f"metric {spec.name!r}: shared-block contribution overflowed",
                    metric=spec.name,
                    cycle=current_cycle,
                    reason="inverse_probability_nonfinite",
                )
            if not math.isfinite(x_b):
                reject_numeric(
                    message=(
                        f"metric {spec.name!r}: shared-block inverse-probability "
                        "contribution is not finite"
                    ),
                    metric=spec.name,
                    cycle=current_cycle,
                    reason="inverse_probability_nonfinite",
                )
            token = str(current_cycle)
            block_deltas[token] = x_b
            block_sizes[token] = n_units

        partition = ContrastPartition(
            metric=spec.name,
            aggregation="any" if spec.type == "conversion" else "sum",
            probability_ct=probability_ct,
            randomization_law=randomization_law,
            independence_grain=independence_grain,
            carryover_order=carryover_order,
            observation_steps=observation_steps,
            retained_steps=observation_steps - carryover_order,
            control_group=control,
            treatment_group=treatment,
            washout_steps=washout_steps,
            unit_deltas=block_deltas,
            cycles_by_unit=block_sizes,
            ct_counts_by_unit={
                str(current_cycle): int(is_ct)
                for current_cycle, is_ct in zip(cycles, block_ct, strict=True)
            },
        )
        try:
            stats[spec.name] = reduce_contrast_partitions([partition])
            planning[spec.name] = _planning_moments(partition, stats[spec.name], slopes)
        except (OverflowError, ValueError):
            reject_numeric(
                message=(
                    f"metric {spec.name!r}: centered contrast reduction overflowed "
                    "or produced non-finite moments"
                ),
                metric=spec.name,
                reason="centered_contrast_reduction_overflow",
            )
    return stats, planning


def from_switchback_panel(  # noqa: PLR0915
    frame: IntoDataFrame,
    *,
    unit: str,
    cycle: str,
    period: str,
    step: str,
    group: str,
    metrics: MetricsArg,
    identification: Randomized,
    assignment: SwitchbackAssignment,
    experiment_id: str = "frame",
    plan: AnalysisPlan | None = None,
    contrast_references: Mapping[str, UnitCycleReference] | None = None,
) -> FrameSwitchbackSource:
    """Build a validated eager switchback source from a complete schedule.

    ``unit`` names the roster member, which can be a market or geo with metrics
    pre-aggregated to geo x cycle x period x step. ``n_units`` then counts geos.
    Under independent unit-cycle draws, the sample size is that count. Under a shared schedule they follow the independent block count.

    Each unit draws its order independently under the declared
    ``independent_bernoulli_order`` mechanism, which makes the unit the
    independent replicate.  Ordinary inference selects a qualified
    unit-level t approximation from the design; a prospective variance
    envelope remains an optional research/reference declaration.  A declared
    ``shared_schedule`` sequence instead draws one order per two-period
    block, shared by a fixed, complete unit roster; there the block is the
    independent replicate, and inference uses a block-level Student-t
    reference over ``n_blocks`` instead.
    Both laws estimate the mean-unit retained-window additive effect. Each period
    retains ``observation_steps - carryover_order`` steps starting at
    ``washout_steps + carryover_order``; no residual carryover after those discarded
    steps is an assumption. Declared and retained lengths survive into result JSON.
    """
    from increment._frame_validation import (
        _validate_columns,
        _validate_switchback_domains,
        _validate_switchback_labels,
        _validate_switchback_metric_values,
        _validate_switchback_schedule,
    )
    from increment._metric_specs import coerce_metrics, synthesise_metric

    if not isinstance(identification, Randomized):
        _reject_identification(
            message=(
                "from_switchback_panel requires Randomized identification; "
                "restricted, observational, and encouragement designs are unsupported"
            ),
            identification_type=type(identification).__name__,
            expected="Randomized",
            reason="unsupported_identification",
        )
    if not isinstance(assignment, SwitchbackAssignment):
        _reject_assignment(
            message=(
                "from_switchback_panel requires a SwitchbackAssignment; "
                "ParallelAssignment is a mixed parallel assignment"
            ),
            assignment_type=type(assignment).__name__,
            expected="SwitchbackAssignment",
            reason="unsupported_assignment",
        )
    if assignment.periods_per_cycle != 2:
        # The discriminated ``AssignmentSequence`` union already guarantees
        # ``scheme`` is one of the two known mechanisms; this guards only
        # the (schema-fixed) period count for defense in depth.
        _reject_assignment(
            message="switchback assignment must declare exactly two periods per cycle",
            assignment_type=type(assignment).__name__,
            periods_per_cycle=assignment.periods_per_cycle,
            reason="unsupported_assignment_mechanism",
        )
    control, treatment = _switchback_groups(identification)

    specs = coerce_metrics(metrics)
    unsupported = [spec.name for spec in specs if spec.type not in ("mean", "conversion")]
    if unsupported:
        _reject_metric(
            message=(
                "switchback metrics must be type='mean' or type='conversion'; "
                f"unsupported metrics: {unsupported!r}"
            ),
            metrics=unsupported,
            reason="unsupported_metric_type",
        )
    for spec in specs:
        if spec.winsorization is not None:
            _reject_metric(
                message=(
                    f"switchback metric {spec.name!r} declares winsorization, "
                    "which switchback aggregation does not apply"
                ),
                metric=spec.name,
                field="winsorization",
                reason="ignored_metric_option",
            )
        if spec.missing != "error":
            _reject_metric(
                message=(
                    f"switchback metric {spec.name!r} declares missing={spec.missing!r}, "
                    "which switchback aggregation does not apply"
                ),
                metric=spec.name,
                field="missing",
                policy=spec.missing,
                reason="ignored_metric_option",
            )
        if spec.decision_method is not None or spec.sensitivity_methods:
            offending_field = (
                "decision_method" if spec.decision_method is not None else "sensitivity_methods"
            )
            _reject_metric(
                message=(
                    f"switchback metric {spec.name!r} declares {offending_field}, "
                    "which switchback procedures do not apply"
                ),
                metric=spec.name,
                field=offending_field,
                reason="ignored_metric_option",
            )
        if spec.prior is not None:
            _reject_metric(
                message=(
                    f"switchback metric {spec.name!r} declares prior, which "
                    "switchback procedures do not apply"
                ),
                metric=spec.name,
                field="prior",
                reason="ignored_metric_option",
            )
    if any(spec.covariate is not None for spec in specs):
        _reject_metric(
            message="switchback metrics do not support covariate adjustment",
            field="covariate",
            reason="unsupported_metric_option",
        )
    if any(spec.window_days is not None for spec in specs):
        _reject_metric(
            message="switchback metrics do not support window_days",
            field="window_days",
            reason="unsupported_metric_option",
        )

    synthesised = [synthesise_metric(spec) for spec in specs]
    study = SwitchbackStudyEnvelope(identification=identification, assignment=assignment)
    references = _validated_unit_cycle_references(
        contrast_references, specs, assignment, control, treatment
    )
    procedures = compile_contrast_procedures(
        plan, synthesised, path="frame/contrast", contrast_references=references
    )
    context = ContrastContext(
        study_id=experiment_id,
        study=study,
        metrics=tuple(synthesised),
        procedures=procedures,
        contrast_references=references,
    )

    window = assignment.window
    total_steps = window.washout_steps + window.observation_steps

    nwf = nw.from_native(frame, eager_only=True)
    roles = [
        ("unit", unit),
        ("cycle", cycle),
        ("period", period),
        ("step", step),
        ("group", group),
    ]
    _validate_columns(nwf, roles=roles, metrics=specs, reject=_reject_columns)
    _validate_switchback_labels(
        nwf,
        unit=unit,
        group=group,
        reject_unit=_reject_domain,
        reject_group=_reject_columns,
    )
    cycles, _, _ = _validate_switchback_domains(
        nwf,
        unit=unit,
        cycle=cycle,
        period=period,
        step=step,
        total_steps=total_steps,
        reject_missingness=_reject_missingness,
        reject_domain=_reject_domain,
    )
    for reference in references.values():
        if isinstance(reference, UnitCycleVarianceEnvelope) and reference.cycles_per_unit != len(
            cycles
        ):
            _unit_refuse("unit_cycle.reference_mismatch", reason="cycles_per_unit")

    _validate_switchback_metric_values(
        nwf,
        specs,
        reject_missingness=_reject_missingness,
        reject_numeric=_reject_numeric,
    )
    units = set(nwf[unit].unique().to_list())
    is_shared = assignment.sequence.scheme == "shared_schedule"
    has_envelopes = all(
        isinstance(references.get(spec.name), UnitCycleVarianceEnvelope) for spec in specs
    )
    if len(units) < 2 and not is_shared and not has_envelopes:
        _reject_units(
            message=(
                f"switchback contrast requires at least two independent units; found {len(units)}"
            ),
            column=unit,
            unit_count=len(units),
            reason="insufficient_independent_units",
        )
    schedule = _validate_switchback_schedule(
        nwf,
        unit=unit,
        cycle=cycle,
        period=period,
        step=step,
        group=group,
        control=control,
        treatment=treatment,
        cycles=cycles,
        total_steps=total_steps,
        reject_missingness=_reject_missingness,
        reject_schedule=_reject_schedule,
    )
    ct_cycles, tc_cycles = schedule.ct_cycles, schedule.tc_cycles

    n_blocks: int | None = None
    block_orders: np.ndarray | None = None
    if is_shared:
        block_orders = _validate_shared_block_orders(
            schedule,
            control=control,
            treatment=treatment,
            cycles=cycles,
            reject_schedule=_reject_schedule,
        )
        # A shared schedule draws one independent Bernoulli order per
        # block, not per unit-cycle: unit-cycle-inflated counts would
        # misstate both the integrity check below and the diagnostic
        # evidence, since the whole roster shares a single realized order.
        ct_cycles = int(block_orders.sum())
        tc_cycles = len(cycles) - ct_cycles
        n_blocks = len(cycles)

    p_ct = assignment.sequence.probability_ct
    drawn = ct_cycles + tc_cycles
    if is_shared:
        from scipy.stats import binomtest

        split_p_value = float(binomtest(ct_cycles, drawn, p_ct).pvalue)
        if not math.isfinite(split_p_value):
            _unit_refuse("unit_cycle.numerical", reason="binomial_admission_nonfinite")
        mechanism_rejected = split_p_value < 1e-6
    else:
        mechanism_rejected = not unit_cycle_admission_outcome(drawn, p_ct, ct_cycles)
    # With a shared schedule, the between-block variance estimates a treatment
    # contrast only if both cycle orders appear. A missing order is a positivity
    # failure and refuses even when the binomial test passes (e.g. 6/6 blocks at
    # probability_ct=0.8, p=0.26).
    missing_order = is_shared and drawn > 0 and (ct_cycles == 0 or tc_cycles == 0)
    if drawn and (mechanism_rejected or missing_order):
        # Construction-time evidence the declared mechanism is wrong, or
        # that the realized split cannot support inference at all: do not
        # infer the true law from this test, and do not defer this to a
        # soft diagnostic surfaced only after results already exist.
        _reject_schedule(
            message=(
                f"declared probability_ct={p_ct} is incompatible with the realized "
                f"{ct_cycles}/{tc_cycles} CT/TC split"
            )
            if mechanism_rejected
            else (
                f"shared-schedule block-t requires both cycle orders to occur across "
                f"the {n_blocks} declared block(s); the realized split is "
                f"{ct_cycles} CT / {tc_cycles} TC, so between-block variance would "
                f"measure only observation noise, not the treatment contrast"
            ),
            probability_ct=p_ct,
            ct_cycles=ct_cycles,
            tc_cycles=tc_cycles,
            n_blocks=n_blocks,
            route=(
                "confirm the declared probability_ct matches how the panel was actually "
                "assigned, or re-check the shared-schedule declaration"
                if mechanism_rejected
                else (
                    "declare more blocks, or a probability_ct closer to 0.5, so both cycle "
                    "orders are likely to occur -- the realized order is known before launch, "
                    "so this can be checked before collecting outcomes"
                )
            ),
            reason=(
                "implausible_realized_split"
                if mechanism_rejected
                else "shared_schedule_missing_cycle_order"
            ),
        )

    for name, reference in references.items():
        if isinstance(reference, UnitCycleVarianceEnvelope):
            procedure = procedures[name]
            unit_cycle_envelope_cutoff(
                reference, n=len(units), alpha=procedure.alpha, alternative=procedure.alternative
            )

    if is_shared:
        assert block_orders is not None
        stats, planning = _build_shared_stats(
            nwf,
            unit=unit,
            cycle=cycle,
            period=period,
            step=step,
            specs=specs,
            control=control,
            treatment=treatment,
            probability_ct=assignment.sequence.probability_ct,
            randomization_law=assignment.sequence.scheme,
            independence_grain=assignment.sequence.independence_unit,
            carryover_order=window.carryover_order,
            observation_steps=window.observation_steps,
            washout_steps=window.washout_steps,
            cycles=cycles,
            block_control_first=block_orders,
            reject_numeric=_reject_numeric,
        )
    else:
        stats, planning = _build_stats(
            nwf,
            unit=unit,
            cycle=cycle,
            period=period,
            step=step,
            specs=specs,
            control=control,
            treatment=treatment,
            probability_ct=assignment.sequence.probability_ct,
            randomization_law=assignment.sequence.scheme,
            independence_grain=assignment.sequence.independence_unit,
            carryover_order=window.carryover_order,
            observation_steps=window.observation_steps,
            washout_steps=window.washout_steps,
            cycles=cycles,
            schedule=schedule,
            reject_numeric=_reject_numeric,
            reject_units=_reject_units,
        )

    diagnostics = SwitchbackAssignmentDiagnostic(
        probability_ct=p_ct,
        randomization_law=assignment.sequence.scheme,
        independence_grain=assignment.sequence.independence_unit,
        schedule_complete=True,
        n_units=len(units),
        n_cycles=len(units) * len(cycles),
        ct_cycles=ct_cycles,
        tc_cycles=tc_cycles,
        n_blocks=n_blocks,
        washout_steps=window.washout_steps,
        observation_steps=window.observation_steps,
        retained_steps=window.observation_steps - window.carryover_order,
        washout_rows=int(nwf.filter(nw.col(step) < window.washout_steps).shape[0]),
        observation_rows=int(nwf.filter(nw.col(step) >= window.washout_steps).shape[0]),
        retained_rows=int(
            nwf.filter(nw.col(step) >= window.washout_steps + window.carryover_order).shape[0]
        ),
        carryover_order=window.carryover_order,
        integrity_failures=(),
    )
    return FrameSwitchbackSource(
        context=context,
        diagnostics=diagnostics,
        stats=stats,
        planning=planning,
        roster=sorted(str(value) for value in units),
    )


__all__ = [
    "FrameSwitchbackSource",
    "SwitchbackAssignmentDiagnostic",
    "from_switchback_panel",
]
