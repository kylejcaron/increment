from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from increment.semantics.models import Breakout, Metric

if TYPE_CHECKING:
    from collections.abc import Mapping

    from increment.errors import CodedError, RefusalSpec
    from increment.semantics.design import Encouragement


@dataclass(frozen=True, slots=True)
class _WarehouseCupedFitEvidence:
    fit: Any
    arm_order: tuple[str, ...]
    withheld: tuple[tuple[str, CodedError], ...] = ()


def warehouse_cuped_fit_evidence(
    fit: Any, arm_order: tuple[str, ...], withheld: tuple[tuple[str, CodedError], ...] = ()
) -> _WarehouseCupedFitEvidence:
    """Build the single warehouse-fit envelope used by live and fitted sources."""
    return _WarehouseCupedFitEvidence(fit, tuple(arm_order), tuple(withheld))


@dataclass(frozen=True, slots=True)
class BreakoutIdentity:
    """The declared identity of a breakout dimension and its source."""

    property: str
    source: str | None


@dataclass(frozen=True, slots=True)
class BreakoutSlicesRequest:
    """Acquire source views for a batch of declared breakout identities."""

    breakouts: tuple[BreakoutIdentity, ...]
    breakout_definitions: tuple[Breakout, ...] = ()
    metrics: tuple[Metric, ...] = ()
    population: Literal["assigned", "triggered"] = "assigned"
    configs: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class BreakoutSlice:
    identity: BreakoutIdentity
    source: Any
    source_name: str | None
    stamp_source: str | None


@dataclass(frozen=True, slots=True)
class BreakoutSlices:
    route: Literal["artifact", "native", "panel", "unsupported"]
    slices: tuple[BreakoutSlice, ...]


@dataclass(frozen=True, slots=True)
class BreakoutUnitRowsRequest:
    """Acquire unit-frame evidence for one breakout/metric pair."""

    metric: Metric
    breakout: BreakoutIdentity
    covariates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BreakoutUnitRows:
    identity: BreakoutIdentity
    route: Literal["artifact", "native", "panel", "unsupported"]
    frame: Any
    unit_frame_has_dimension: bool


@dataclass(frozen=True, slots=True)
class DayAxisRequest:
    caller: Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"]
    grain: Literal["daily", "asof"]
    metrics: tuple[Metric, ...]
    dimension: str | None
    completed_windows_only: bool = False
    estimands: tuple[str, ...] | None = None
    population: Literal["assigned", "triggered"] = "assigned"
    experiment: Any | None = None
    covariate_metrics: frozenset[str] = frozenset()
    by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DayAxisRows:
    """Acquired daily/as-of row slices and the source view used to read them."""

    source: Any
    route: Literal["artifact", "moments", "native"]
    slices: Iterator[Any]


@dataclass(frozen=True, slots=True)
class ComplianceRequest:
    design: Encouragement
    as_of: object | None = None
    completed_windows_only: bool = False


@dataclass(frozen=True, slots=True)
class ComplianceSeriesRequest:
    design: Encouragement
    completed_windows_only: bool = False
    population: Literal["assigned", "triggered"] = "assigned"


@dataclass(frozen=True, slots=True)
class MetricEvidenceRequest:
    metric: Metric
    control_group: str
    by: tuple[str, ...] = ()
    methods: tuple[Any, ...] = ()
    covariates: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """One metric's aggregate rows and/or per-unit estimator evidence."""

    rows: tuple[Mapping[str, Any], ...]
    unit_frame: Any | None
    observed_arms: frozenset[str]
    n_treatment_arms: int
    evidence_kind: str
    evidence_sha256: str
    evidence_count: int
    decision_evidence_components: tuple[tuple[str, str, int], ...]
    cluster: str | None
    control_group: str
    evidence_components: tuple[tuple[str, str, int], ...] = ()
    covariate_columns: tuple[tuple[str, str], ...] = ()
    method_rows: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...] = ()
    method_refusals: tuple[tuple[str, RefusalSpec, Mapping[str, object]], ...] = ()
    method_cell_refusals: tuple[tuple[str, str, str, RefusalSpec, Mapping[str, object]], ...] = ()
    categorical_evidence: tuple[tuple[str, Any], ...] = ()
    warehouse_cuped_fits: tuple[tuple[str, Any], ...] = ()
    warehouse_cuped_breakout_fits: tuple[tuple[str, str, str, Any, tuple[object, ...]], ...] = ()

    def categorical_for(self, method: Any) -> Any | None:
        """Retained joint categorical evidence for one multicovariate method."""
        return next(
            (evidence for name, evidence in self.categorical_evidence if name == method.name),
            None,
        )

    def warehouse_fit_for(self, method: Any) -> Any | None:
        """Return this request's finalized warehouse fit, if it was acquired."""
        return next(
            (fit for name, fit in self.warehouse_cuped_fits if name == method.name),
            None,
        )

    def warehouse_breakout_fit_for(
        self, method: Any, *, dimension: str, dimension_value: str
    ) -> Any | None:
        return next(
            (
                fit
                for name, fit_dimension, value, fit, _identity in self.warehouse_cuped_breakout_fits
                if name == method.name and fit_dimension == dimension and value == dimension_value
            ),
            None,
        )

    def rows_for_method(
        self,
        method: Any,
        *,
        dimension: str | None = None,
        dimension_value: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return the produced input family, replaying only its own cell refusal."""
        from increment.errors import refuse

        for name, spec, context in self.method_refusals:
            if name == method.name:
                refuse(spec, **context)
        for name, cell_dimension, value, spec, context in self.method_cell_refusals:
            if name != method.name:
                continue
            if dimension is None:
                raise AssertionError("Partitioned scalar evidence requires a segment.")
            if cell_dimension == dimension and value == dimension_value:
                refuse(spec, **context)
        for name, rows in self.method_rows:
            if name == method.name:
                if dimension is None:
                    return rows
                return tuple(
                    row
                    for row in rows
                    if ("__null__" if row.get(dimension) is None else str(row[dimension]))
                    == dimension_value
                )
        raise AssertionError(f"Scalar evidence for method {method.name!r} was not acquired.")
