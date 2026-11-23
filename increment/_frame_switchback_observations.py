"""Align switchback responses to an independently recorded assignment schedule."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn

import narwhals as nw

from increment._frame_validation import (
    _is_missing,
    _switchback_integral_domain,
    _validate_columns,
    eager_frame,
    require_eager_frame,
)
from increment._study import SwitchbackObservationDesign
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._metric_specs import MetricSpec
    from increment.semantics.assignment import SwitchbackAssignment


_INVALID_DESIGN = RefusalSpec(
    "source.frame.switchback.observation_design",
    InvalidRequestError,
    template="Invalid switchback observation schedule: {reason} ({field}).",
)
_ASSIGNMENT_UNRECORDED = RefusalSpec(
    "source.frame.switchback.assignment_unrecorded",
    InvalidRequestError,
    template=(
        "No assigned arm was recorded for unit {unit!r}, cycle {cycle}, period {period}. "
        "Supply the assignment log for the complete roster and cycle horizon."
    ),
    keys=frozenset({"missing_cells"}),
)
_UNOBSERVED_CODE: Final = "source.frame.switchback.response_unobserved"
_RESPONSE_UNOBSERVED = RefusalSpec(
    _UNOBSERVED_CODE,
    CapabilityError,
    template=(
        "Metric {metric!r} cannot supply a complete planning pilot: {missing_count} retained "
        "steps lack a measurement. The first is {first_missing_key!r}, assigned to {arm!r}."
    ),
    keys=frozenset({"reason"}),
)
_EVENT_COVERAGE = RefusalSpec(
    "source.frame.switchback.event_coverage",
    CapabilityError,
    template=(
        "Metric {metric!r}: counting an absent event as zero requires both missing='zero' "
        "and a complete event feed for that metric over every scheduled retained step."
    ),
    keys=frozenset({"reason"}),
)


def _names(values: Sequence[str], name: str, *, required: bool) -> tuple[str, ...]:
    if (
        isinstance(values, str | bytes)
        or not isinstance(values, Sequence)
        or (required and not values)
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        refuse(_INVALID_DESIGN, field=name, reason="supply distinct nonempty names")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True, eq=False)
class SwitchbackObservationSchedule:
    """One assigned arm per roster member, cycle and period, independent of responses.

    ``units`` and ``cycles`` declare the expected roster and number of cycles starting
    at zero, even if a unit or the last cycle is absent from ``assignments``. The log
    uses the builder's unit, cycle, period and group columns. An observation need not
    repeat the arm; if it does, the two labels must agree.

    ``complete_event_metrics`` names sparse feeds known to cover every retained step.
    For those metrics only, ``missing='zero'`` means an absent event row contributes
    zero. A recorded null is still an unknown measurement, as is an absent denominator.
    """

    assignments: IntoDataFrame = field(repr=False)
    units: tuple[str, ...]
    cycles: int
    complete_event_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.cycles, bool)
            or not isinstance(self.cycles, Integral)
            or self.cycles < 1
        ):
            refuse(_INVALID_DESIGN, field="cycles", reason="supply a positive integer")
        object.__setattr__(self, "cycles", int(self.cycles))
        object.__setattr__(self, "units", _names(self.units, "units", required=True))
        object.__setattr__(
            self,
            "complete_event_metrics",
            _names(self.complete_event_metrics, "complete_event_metrics", required=False),
        )

    def validate_metrics(self, specs: Sequence[MetricSpec]) -> None:
        """Require an independent complete-feed assertion for each zero policy."""
        zero = {spec.name for spec in specs if spec.missing == "zero"}
        different = zero.symmetric_difference(self.complete_event_metrics)
        if different:
            name = min(different)
            refuse(
                _EVENT_COVERAGE,
                metric=name,
                reason="coverage_not_declared" if name in zero else "zero_policy_not_declared",
            )

    def declaration(self) -> SwitchbackObservationDesign:
        """Portable roster, horizon and event policies, without measured responses."""
        return SwitchbackObservationDesign(
            units=self.units, cycles=self.cycles, complete_event_metrics=self.complete_event_metrics
        )


type _Period = tuple[str, int, int]
type _Step = tuple[str, int, int, int]
type _Reject = Callable[..., NoReturn]


@dataclass(frozen=True, slots=True)
class MissingSwitchbackResponse:
    """One metric's unmeasured scheduled steps, before any no-event zeros are filled."""

    metric: str
    missing_count: int
    first_missing_key: _Step
    arm: str

    @property
    def code(self) -> Literal["source.frame.switchback.response_unobserved"]:
        return _UNOBSERVED_CODE

    def context(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "reason": "unobserved_response",
            "missing_count": self.missing_count,
            "first_missing_key": self.first_missing_key,
            "arm": self.arm,
        }

    def refuse_planning(self) -> NoReturn:
        refuse(_RESPONSE_UNOBSERVED, **self.context())


@dataclass(frozen=True, slots=True)
class _Axes:
    unit: str
    cycle: str
    period: str
    step: str
    group: str

    @property
    def keys(self) -> tuple[str, str, str, str]:
        return self.unit, self.cycle, self.period, self.step

    @property
    def roles(self) -> tuple[tuple[str, str], ...]:
        return (
            ("unit", self.unit),
            ("cycle", self.cycle),
            ("period", self.period),
            ("step", self.step),
            ("group", self.group),
        )


def _check_axes(
    frame: nw.DataFrame[Any], roles: Sequence[tuple[str, str]], reject: _Reject
) -> None:
    if not frame.is_empty():
        for role, name in roles:
            if role in ("cycle", "period", "step"):
                _switchback_integral_domain(frame, column=name, label=role, reject=reject)


def _logged_periods(
    schedule: SwitchbackObservationSchedule,
    axes: _Axes,
    periods: int,
    reject_columns: _Reject,
    reject_domain: _Reject,
    reject_schedule: _Reject,
) -> dict[_Period, str]:
    require_eager_frame(schedule.assignments, constructor="from_switchback_panel")
    log = eager_frame(schedule.assignments)
    roles = [(role, name) for role, name in axes.roles if role != "step"]
    _validate_columns(log, roles=roles, metrics=(), reject=reject_columns)
    _check_axes(log, roles, reject_domain)
    roster = set(schedule.units)
    arms: dict[_Period, str] = {}
    nulls: set[_Period] = set()
    rows = log.select(*(name for _, name in roles)).iter_rows()
    missing_arms = log.select(_is_missing(log, axes.group)).iter_rows()
    for (u, c, p, arm), (missing,) in zip(rows, missing_arms, strict=True):
        if (
            not isinstance(u, str)
            or u not in roster
            or not 0 <= c < schedule.cycles
            or not 0 <= p < periods
        ):
            reject_schedule(
                message="An assignment lies outside the declared roster, cycles or periods.",
                unit=u,
                cycle=c,
                period=p,
                reason="off_schedule_assignment",
            )
        key = (u, c, p)
        if key in arms or key in nulls:
            reject_schedule(
                message="More than one assignment was logged for the same unit, cycle and period.",
                unit=u,
                cycle=c,
                period=p,
                reason="duplicate_assignment_cells",
            )
        if missing:
            nulls.add(key)
        elif not isinstance(arm, str) or not arm:
            reject_columns(
                message="Logged arms must have nonempty string labels.",
                column=axes.group,
                field="group",
                values=[arm],
                reason="empty_label" if arm == "" else "non_string_label",
            )
        else:
            arms[key] = arm
    expected = len(roster) * schedule.cycles * periods
    if len(arms) != expected:
        # Scan only to the first gap; do not allocate a Cartesian product from unchecked counts.
        u, c, p = next(
            (u, c, p)
            for u in schedule.units
            for c in range(schedule.cycles)
            for p in range(periods)
            if (u, c, p) not in arms
        )
        refuse(
            _ASSIGNMENT_UNRECORDED, unit=u, cycle=c, period=p, missing_cells=expected - len(arms)
        )
    return arms


def _observation_indices(
    frame: nw.DataFrame[Any],
    axes: _Axes,
    arms: dict[_Period, str],
    total: int,
    reject_domain: _Reject,
    reject_schedule: _Reject,
) -> dict[_Step, int]:
    _check_axes(frame, axes.roles, reject_domain)
    labels = axes.group in frame.columns
    cols = (*axes.keys, axes.group) if labels else axes.keys
    indices: dict[_Step, int] = {}
    duplicates: set[_Step] = set()
    for i, (u, c, p, s, *reported) in enumerate(frame.select(*cols).iter_rows()):
        if not isinstance(u, str) or (u, c, p) not in arms or not 0 <= s < total:
            reject_schedule(
                message="A response lies outside the logged assignments or declared step range.",
                unit=u,
                cycle=c,
                period=p,
                step=s,
                reason="off_schedule_observation",
            )
        arm = arms[u, c, p]
        if reported and (not isinstance(reported[0], str) or arm != reported[0]):
            reject_schedule(
                message="The observation's arm label disagrees with the assignment log.",
                unit=u,
                cycle=c,
                period=p,
                step=s,
                expected=arm,
                observed=reported[0],
                reason="observation_arm_mismatch",
            )
        key = (u, c, p, s)
        if key in indices:
            duplicates.add(key)
        indices[key] = i
    if duplicates:
        reject_schedule(
            message="More than one response row was supplied for the same scheduled step.",
            unit=axes.unit,
            cycle=axes.cycle,
            period=axes.period,
            step=axes.step,
            duplicate_cells=len(duplicates),
            reason="duplicate_schedule_cells",
        )
    return indices


def _missing_responses(
    frame: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    axes: _Axes,
    arms: dict[_Period, str],
    indices: dict[_Step, int],
    retained: range,
) -> dict[str, MissingSwitchbackResponse]:
    n_retained = sum(key[3] >= retained.start for key in indices)
    n_absent = len(arms) * (retained.stop - retained.start) - n_retained
    first_absent = (
        next(
            (u, c, p, s)
            for u, c, p in sorted(arms)
            for s in retained
            if (u, c, p, s) not in indices
        )
        if n_absent
        else None
    )
    recorded = frame.filter(nw.col(axes.step) >= retained.start)
    missing: dict[str, MissingSwitchbackResponse] = {}
    for spec in specs:
        # A ratio needs its measured denominator even when an absent numerator means zero.
        absent_unknown = spec.missing != "zero" or spec.denominator is not None
        count = n_absent if absent_unknown else 0
        key = first_absent if absent_unknown else None
        unknown = _is_missing(recorded, spec.y_column)
        if spec.denominator is not None:
            unknown |= _is_missing(recorded, spec.denominator)
        nulls = recorded.filter(unknown).select(*axes.keys)
        if not nulls.is_empty():
            count += len(nulls)
            u, c, p, s = min(nulls.iter_rows())
            null_key = (u, c, p, s)
            if key is None or null_key < key:
                key = null_key
        if key is not None:
            u, c, p, _ = key
            missing[spec.name] = MissingSwitchbackResponse(spec.name, count, key, arms[u, c, p])
    return missing


def _align(
    frame: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    axes: _Axes,
    arms: dict[_Period, str],
    indices: dict[_Step, int],
    retained: range,
) -> nw.DataFrame[Any]:
    expected_rows = len(arms) * (retained.stop - retained.start)
    observed = frame.filter(nw.col(axes.step) >= retained.start) if retained.start else frame
    if len(observed) == expected_rows:
        if axes.group in observed.columns:
            return observed
        return observed.with_columns(
            nw.new_series(
                axes.group,
                [arms[u, c, p] for u, c, p in observed.select(*axes.keys[:3]).iter_rows()],
                backend=frame.implementation,
            )
        )
    if expected_rows > sys.maxsize // 8:
        refuse(
            _INVALID_DESIGN, field="steps", reason="the scheduled panel exceeds the row count limit"
        )
    columns = dict.fromkeys(
        col for spec in specs for col in (spec.y_column, spec.denominator) if col is not None
    )
    payload = frame.select(*columns)
    if frame.implementation.is_pandas():
        # A pandas left join pads plain integers with float NaN, rounding integers above 2**53.
        nullable = {
            col: "boolean" if dtype == nw.Boolean else str(dtype)
            for col, dtype in payload.schema.items()
            if dtype.is_integer() or dtype == nw.Boolean
        }
        if nullable:
            payload = eager_frame(payload.to_native().astype(nullable))
    ordered = sorted(arms)
    backend = frame.implementation
    cells = nw.from_dict(
        {
            axes.unit: [u for u, _, _ in ordered],
            axes.cycle: [c for _, c, _ in ordered],
            axes.period: [p for _, _, p in ordered],
            axes.group: [arms[key] for key in ordered],
        },
        backend=backend,
    )
    grid = cells.join(nw.from_dict({axes.step: retained}, backend=backend), how="cross")
    idx = "__observation_row"
    while idx in frame.columns or idx in grid.columns:
        idx += "_"
    rows = [indices.get((u, c, p, s), -1) for u, c, p, s in grid.select(*axes.keys).iter_rows()]
    grid = grid.with_columns(nw.new_series(idx, rows, dtype=nw.Int64, backend=backend))
    if backend.is_pyarrow():
        # A null take index pads every payload type, including Arrow's all-null type.
        positions = grid.select(nw.when(nw.col(idx) >= 0).then(nw.col(idx)).alias(idx))[idx]
        aligned = nw.concat([grid, payload[positions, :]], how="horizontal")
    else:
        payload = payload.with_row_index(idx).with_columns(nw.col(idx).cast(nw.Int64))
        aligned = grid.join(payload, on=idx, how="left")
    absent = nw.col(idx) == -1
    zeros = []
    schema = frame.schema
    for col in dict.fromkeys(spec.y_column for spec in specs if spec.missing == "zero"):
        dtype = schema[col]
        zero = nw.lit(
            False if dtype == nw.Boolean else 0, dtype=dtype if dtype.is_numeric() else None
        )
        # Let the measured column, not a signed zero literal, own the result dtype.
        zeros.append(nw.when(~absent).then(nw.col(col)).otherwise(zero).alias(col))
    return aligned.with_columns(*zeros).drop(idx)


def reconcile_switchback_observations(
    frame: nw.DataFrame[Any],
    *,
    schedule: SwitchbackObservationSchedule | None,
    assignment: SwitchbackAssignment,
    specs: Sequence[MetricSpec],
    unit: str,
    cycle: str,
    period: str,
    step: str,
    group: str,
    reject_columns: _Reject,
    reject_domain: _Reject,
    reject_schedule: _Reject,
) -> tuple[nw.DataFrame[Any], dict[str, MissingSwitchbackResponse]]:
    """Validate response columns and align retained measurements to a log when supplied."""
    axes = _Axes(unit, cycle, period, step, group)
    identities = {*axes.keys, group}
    roles = axes.roles
    arms = None
    if schedule is not None:
        if len(identities) != 5:
            reject_columns(
                message="Each switchback identity role needs a distinct column.",
                reason="overlapping_roles",
            )
        arms = _logged_periods(
            schedule,
            axes,
            assignment.periods_per_cycle,
            reject_columns,
            reject_domain,
            reject_schedule,
        )
        if group not in frame.columns:
            roles = tuple((role, name) for role, name in roles if role != "group")
    # Pre-assignment forecasts are a separate input, never inferred from outcome rows.
    outcomes = tuple(
        spec.model_copy(update={"decision_method": None, "sensitivity_methods": ()})
        if spec._methods
        else spec
        for spec in specs
    )
    _validate_columns(frame, roles=roles, metrics=outcomes, reject=reject_columns)
    for spec in specs:
        for col in (spec.y_column, spec.denominator):
            if col in identities:
                reject_columns(
                    message="An observation identity column is not a measured response.",
                    name=col,
                    metric=spec.name,
                    reason="metric_overlaps_column",
                )
    if arms is None:
        return frame, {}
    window = assignment.window
    retained = range(
        window.washout_steps + window.carryover_order,
        window.washout_steps + window.observation_steps,
    )
    indices = _observation_indices(frame, axes, arms, retained.stop, reject_domain, reject_schedule)
    missing = _missing_responses(frame, specs, axes, arms, indices, retained)
    return _align(frame, specs, axes, arms, indices, retained), missing
