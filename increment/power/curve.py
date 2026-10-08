"""Typed rows and grid evaluation for power-analysis curves."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    SupportsIndex,
    cast,
    get_args,
    overload,
)

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel, ConfigDict, model_validator

from increment.decision import FixedInference
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.power.core import (
    Baseline,
    MdeUnavailableReason,
    PowerBasis,
    PowerDesign,
    PowerResult,
    _GridQuery,
    _planned_looks,
    _prepare_solver,
    _require_companion_mde,
    _runtime_binomial,
    _solve_together,
)

if TYPE_CHECKING:
    pass

Backend = Literal["pandas", "polars", "pyarrow"]
SolveFor = Literal["power", "mde"]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "power.scalar_numeric_sequence": "{name} must be a scalar or numeric sequence",
        "power.empty": "{name} must not be empty",
        "power.curve.n_per_arm_int": "n_per_arm must contain integers, got {value!r}",
        "power.curve.n_per_arm_min": "n_per_arm must be >= 2, got {value}",
        "power.units_per_week_finite_positive": "units_per_week must be a finite positive number",
        "power.max_workers_positive": "max_workers must be a positive integer or None",
        "power.max_workers": "max_workers must be >= 1",
        "power.exactly_one_relative": "exactly one of relative_lift and target_power must be provided",
    },
)
_raise = raiser(_REFUSALS)


class PowerCurvePoint(CodedModel, BaseModel):
    """One evaluated point in a power or MDE curve.

    ``mde_relative`` is ``None`` with ``mde_unavailable_reason`` set when
    no minimum detectable effect exists at that row's size and target
    (see ``PowerResult``); frames keep the column numeric with a null.
    ``power_basis`` names the planning model behind ``power`` and
    ``mde_relative``, as on ``PowerResult``. ``numerical_qualification`` is
    copied from that result and records the arithmetic scope of the reported
    model, not its statistical guarantee.
    """

    model_config = ConfigDict(frozen=True)

    solve_for: SolveFor
    n_per_arm: int
    n_total: int
    relative_lift: float | None
    target_power: float
    alpha: float
    power: float
    power_basis: PowerBasis
    numerical_qualification: Literal[
        "closed_form_model_only_v1",
        "sequential_crossing_quadrature_v1",
        "scipy_special_function_error_model_conditional_v1",
        "unclaimed_approximation_diagnostic_v1",
    ]
    mde_relative: float | None
    mde_unavailable_reason: MdeUnavailableReason | None = None
    effective_var: float
    n_clusters_per_arm: int | None = None
    n_clusters_total: int | None = None
    n_triggered_per_arm: int | None = None  # None = no trigger rate declared
    n_triggered_total: int | None = None
    duration_days: int | None = None
    # Both None for a fixed-horizon row (no inference spec); see
    # PowerResult.expected_n_total for the early-stopping accounting.
    expected_n_total: int | None = None
    expected_duration_days: int | None = None

    @model_validator(mode="after")
    def _check(self) -> PowerCurvePoint:
        _require_companion_mde(self.mde_relative, self.mde_unavailable_reason)
        return self


def _dtype_base(annotation: Any) -> Any:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return args[0] if args else annotation


def _frame_dtype(annotation: Any) -> nw.dtypes.DType:
    base = _dtype_base(annotation)
    if base is int:
        return nw.Float64() if type(None) in get_args(annotation) else nw.Int64()
    if base is float:
        return nw.Float64()
    return nw.String()


class PowerCurve(list[PowerCurvePoint]):
    """List-like power-curve result with dict and dataframe conversion."""

    def to_dicts(self) -> list[dict[str, object]]:
        return [row.model_dump() for row in self]

    def to_frame(self, backend: Backend = "pandas") -> IntoDataFrame:

        fields = PowerCurvePoint.model_fields
        names = list(fields)
        names.insert(names.index("power_basis") + 1, "numerical_qualification")
        data = {
            name: (
                [row.numerical_qualification for row in self]
                if name == "numerical_qualification"
                else [getattr(row, name) for row in self]
            )
            for name in names
        }
        schema = {
            name: (
                nw.String()
                if name == "numerical_qualification"
                else _frame_dtype(fields[name].annotation)
            )
            for name in names
        }
        frame = nw.from_dict(data, schema=schema, backend=backend).to_native()

        nullable_strings = [
            name
            for name, field in fields.items()
            if type(None) in get_args(field.annotation)
            and _dtype_base(field.annotation) not in (int, float)
        ]
        if backend == "pandas" and nullable_strings:
            # Pandas can coerce nullable strings to object dtype during the
            # Narwhals conversion; restore pandas' nullable StringDtype.
            import pandas as pd

            for name in nullable_strings:
                frame[name] = pd.array(data[name], dtype="string")
        return frame

    @overload
    def __getitem__(self, key: SupportsIndex) -> PowerCurvePoint: ...

    @overload
    def __getitem__(self, key: slice) -> PowerCurve: ...

    def __getitem__(self, key: SupportsIndex | slice) -> PowerCurvePoint | PowerCurve:
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return cast("PowerCurve", type(self)(cast("list[PowerCurvePoint]", value)))
        return cast("PowerCurvePoint", value)

    def __add__(self, other: list[PowerCurvePoint]) -> PowerCurve:
        return type(self)(super().__add__(other))

    def __iadd__(self, other: Iterable[PowerCurvePoint]) -> Self:
        super().__iadd__(other)
        return self


@dataclass(frozen=True, slots=True)
class _CurveJob:
    solve_for: SolveFor
    n_per_arm: int
    relative_lift: float | None
    procedure: ArmPlanningProcedure
    design: PowerDesign


def _as_tuple[T](name: str, value: T | Sequence[T]) -> tuple[T, ...]:
    if isinstance(value, (str, bytes)):
        _raise("power.scalar_numeric_sequence", name=name)
    values = tuple(cast("Sequence[T]", value)) if isinstance(value, Sequence) else (value,)
    if not values:
        _raise("power.empty", name=name)
    return values


def _validated_design(base: PowerDesign, *, power: float) -> PowerDesign:
    values = base.model_dump()
    values.update(power=power)
    return PowerDesign.model_validate(values)


def _validate_sample_sizes(values: tuple[int, ...]) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            _raise("power.curve.n_per_arm_int", value=value)
        if value < 2:
            _raise("power.curve.n_per_arm_min", value=value)


def _validate_units_per_week(units_per_week: float | None) -> None:
    if units_per_week is None:
        return
    if isinstance(units_per_week, bool) or not isinstance(units_per_week, (int, float)):
        _raise("power.units_per_week_finite_positive")
    if not math.isfinite(units_per_week) or units_per_week <= 0:
        _raise("power.units_per_week_finite_positive")


def _point_from_result(
    job: _CurveJob,
    result: PowerResult,
    units_per_week: float | None,
) -> PowerCurvePoint:
    duration_days = (
        None if units_per_week is None else math.ceil(7 * result.n_total / units_per_week)
    )
    expected_duration_days = (
        None
        if units_per_week is None or result.expected_n_total is None
        else math.ceil(7 * result.expected_n_total / units_per_week)
    )
    return PowerCurvePoint(
        solve_for=job.solve_for,
        n_per_arm=result.n_per_arm,
        n_total=result.n_total,
        relative_lift=job.relative_lift,
        target_power=job.design.power,
        alpha=job.procedure.compiled_alpha,
        power=result.power,
        power_basis=result.power_basis,
        numerical_qualification=result.numerical_qualification,
        mde_relative=result.mde_relative,
        mde_unavailable_reason=result.mde_unavailable_reason,
        effective_var=result.effective_var,
        n_clusters_per_arm=result.n_clusters_per_arm,
        n_clusters_total=result.n_clusters_total,
        n_triggered_per_arm=result.n_triggered_per_arm,
        n_triggered_total=result.n_triggered_total,
        duration_days=duration_days,
        expected_n_total=result.expected_n_total,
        expected_duration_days=expected_duration_days,
    )


def _evaluate_group(
    jobs: list[_CurveJob],
    *,
    baseline: Baseline,
    planned_looks: int | None,
    units_per_week: float | None,
) -> list[PowerCurvePoint]:
    """Evaluate jobs sharing one size and procedure through the core grid
    solve, which reuses a binomial plan's rejection geometry across them."""
    queries = [
        _GridQuery(
            job.n_per_arm,
            job.relative_lift if job.solve_for == "power" else None,
            job.procedure,
            job.design,
        )
        for job in jobs
    ]
    results = _solve_together(queries, baseline=baseline, planned_looks=planned_looks)
    return [
        _point_from_result(job, result, units_per_week)
        for job, result in zip(jobs, results, strict=True)
    ]


def _validate_max_workers(max_workers: int | None) -> None:
    if max_workers is None:
        return
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        _raise("power.max_workers_positive")
    if max_workers < 1:
        _raise("power.max_workers")


def _effective_workers(
    *,
    grid_size: int,
    sequential: bool,
    max_workers: int | None,
) -> int:
    if not sequential or grid_size < 4 or max_workers == 1:
        return 1
    cpu_count = os.cpu_count() or 1
    limit = 4 if max_workers is None else max_workers
    return min(limit, cpu_count, grid_size)


def _evaluate_jobs(
    jobs: list[_CurveJob],
    *,
    baseline: Baseline,
    planned_looks: int | None,
    units_per_week: float | None,
    max_workers: int | None,
) -> PowerCurve:
    # Jobs of one runtime-binomial decision share its rejection geometry, so
    # they run together; every other job keeps its own slot for the workers.
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, job in enumerate(jobs):
        key = (
            (job.n_per_arm, id(job.procedure))
            if _runtime_binomial(job.procedure)
            else ("job", index)
        )
        groups.setdefault(key, []).append(index)
    evaluate = partial(
        _evaluate_group,
        baseline=baseline,
        planned_looks=planned_looks,
        units_per_week=units_per_week,
    )
    batches = [[jobs[index] for index in members] for members in groups.values()]
    workers = _effective_workers(
        grid_size=len(batches),
        sequential=any(not isinstance(job.procedure.inference, FixedInference) for job in jobs),
        max_workers=max_workers,
    )
    if workers == 1:
        results = [evaluate(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(evaluate, batches))
    points: list[PowerCurvePoint] = []
    order: list[int] = []
    for members, batch_points in zip(groups.values(), results, strict=True):
        order.extend(members)
        points.extend(batch_points)
    ranked = sorted(zip(order, points, strict=True), key=lambda item: item[0])
    return PowerCurve(point for _, point in ranked)


def power_curve(
    *,
    n_per_arm: int | Sequence[int],
    baseline: Baseline,
    procedure: ArmPlanningProcedure | Sequence[ArmPlanningProcedure],
    relative_lift: float | Sequence[float] | None = None,
    target_power: float | Sequence[float] | None = None,
    design: PowerDesign | None = None,
    units_per_week: float | None = None,
    planned_looks: int | None = None,
    max_workers: int | None = None,
) -> PowerCurve:
    """Evaluate an ordered fixed-horizon or sequential power or MDE grid.

    Provide exactly one of ``relative_lift`` and ``target_power``; the omitted
    quantity is solved per row. ``procedure`` may be a sequence to evaluate
    multiple compiled decision policies while retaining product order.
    ``PowerCurvePoint.alpha`` records each procedure's derived decision alpha.
    Each sequential procedure's look schedule resolves as in
    ``required_sample_size`` and is validated before the grid is built.
    """
    if (relative_lift is None) == (target_power is None):
        _raise("power.exactly_one_relative")

    procedures_raw = _as_tuple("procedure", procedure)
    procedures: tuple[ArmPlanningProcedure, ...] = ()
    for item in procedures_raw:
        validated_procedure, _ = _prepare_solver(item, baseline)
        _planned_looks(validated_procedure, planned_looks)
        procedures += (validated_procedure,)

    base = PowerDesign() if design is None else PowerDesign.model_validate(design)
    sample_sizes = _as_tuple("n_per_arm", n_per_arm)
    _validate_sample_sizes(sample_sizes)
    _validate_units_per_week(units_per_week)
    _validate_max_workers(max_workers)

    jobs: list[_CurveJob] = []
    if relative_lift is not None:
        lifts = _as_tuple("relative_lift", relative_lift)
        for n, lift, proc in product(sample_sizes, lifts, procedures):
            jobs.append(
                _CurveJob(
                    solve_for="power",
                    n_per_arm=n,
                    relative_lift=lift,
                    procedure=proc,
                    design=_validated_design(base, power=base.power),
                )
            )
    else:
        assert target_power is not None
        targets = _as_tuple("target_power", target_power)
        for n, target, proc in product(sample_sizes, targets, procedures):
            jobs.append(
                _CurveJob(
                    solve_for="mde",
                    n_per_arm=n,
                    relative_lift=None,
                    procedure=proc,
                    design=_validated_design(base, power=target),
                )
            )

    return _evaluate_jobs(
        jobs,
        baseline=baseline,
        planned_looks=planned_looks,
        units_per_week=units_per_week,
        max_workers=max_workers,
    )
