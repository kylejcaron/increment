from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, Literal

from increment.semantics.models import Breakout, Metric

BreakoutRoute = Literal["artifact", "native", "panel", "unsupported"]
DayAxisRoute = Literal["artifact", "moments", "native"]


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
class BreakoutCell:
    """One requested cell within a declared breakout identity."""

    breakout: BreakoutIdentity
    value: object
    is_null: bool


@dataclass(frozen=True, slots=True)
class MethodFitRequest:
    """Request method-specific CUPED evidence for a metric or breakout cell."""

    metric: Metric
    method: Any
    control: str
    arm_order: tuple[str, ...]
    population: Literal["assigned", "triggered"]
    cell: BreakoutCell | None = None


@dataclass(frozen=True, slots=True)
class MethodFitAnswer:
    """Retained or warehouse method-fit evidence, or its coded acquisition failure."""

    origin: Literal["warehouse", "retained_fitted", "retained_categorical", "unavailable"]
    envelope: Any | None = None
    cells: Any | None = None
    failure: CodedError | None = None
    fit_identity: tuple[object, ...] = ()
    evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedMulticovariateFit:
    """Selected retained, warehouse, or local input for a mixed-CUPED fit."""

    origin: Literal["warehouse", "retained_fitted", "retained_categorical", "local"]
    envelope: Any | None = None
    cells: Any | None = None
    failure: CodedError | None = None
    fit_identity: tuple[object, ...] = ()


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
    route: BreakoutRoute
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
    route: BreakoutRoute
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
    route: DayAxisRoute
    slices: Iterator[Any]


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    """The source views and readout lifetime required by one evidence consumer."""

    metrics: tuple[Metric, ...] = ()
    population: Literal["assigned", "triggered"] = "assigned"
    grain: Literal["total", "daily", "asof"] | None = None
    breakouts: tuple[BreakoutIdentity, ...] = ()
    configs: tuple[Any, ...] | None = None
    uptake_facts: tuple[str, ...] = ()
    snapshot: bool = False
    execution: bool = False
    include_breakouts: bool = False


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
    method_fit_answers: tuple[tuple[MethodFitRequest, MethodFitAnswer], ...] = ()

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


@dataclass(frozen=True, slots=True)
class MomentsRequest:
    """Select aggregate metric evidence at one time grain."""

    metric: Metric
    grain: Literal["total", "daily", "asof"] = "total"
    by: tuple[str, ...] = ()
    method: Any | None = None
    include_covariate: bool = False
    completed_windows_only: bool = False
    population: Literal["assigned", "triggered"] = "assigned"


MomentRows = tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class BreakoutMethodMomentsRequest:
    """Select method-specific moment rows and refusals for one breakout."""

    metric: Metric
    method: Any
    breakout: BreakoutIdentity
    population: Literal["assigned", "triggered"] = "assigned"
    completed_windows_only: bool = False


@dataclass(frozen=True, slots=True)
class ScalarBreakoutRows:
    """Method-specific moments and scalar refusals indexed by segment."""

    rows: tuple[Mapping[str, Any], ...]
    refusals: tuple[tuple[str, str, RefusalSpec, Mapping[str, object]], ...] = ()


@dataclass(frozen=True, slots=True)
class UnitRowsRequest:
    """Select per-unit observations needed for an estimator."""

    metric: Metric
    covariates: tuple[str, ...] = ()
    outcome_stage: Literal["transformed", "raw"] = "transformed"
    population: Literal["assigned", "triggered"] = "assigned"


UnitRows = Any


@dataclass(frozen=True, slots=True)
class CountsRequest:
    """Select enrolled-unit, cluster, assignment, or triggered counts."""

    kind: Literal["unit", "cluster", "assignment", "triggered"]
    population: Literal["assigned", "triggered"] = "assigned"


@dataclass(frozen=True, slots=True)
class CountEvidence:
    """Counts for the requested population and randomization grain."""

    population: str
    grain: str
    counts: Mapping[str, int]
    unit_counts: Mapping[str, int] = dataclass_field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequentialCaptureRequest:
    """Capture a registered finalized prefix of sequential data."""

    finalized: bool
    as_of: object | None = None
    previous: Any | None = None
    freeze: tuple[str, ...] = ()


SequentialCapture = Any


@dataclass(frozen=True, slots=True)
class EvidenceScopeResult:
    """A source view kept open for the lifetime of one analysis."""

    source: Any
    scope: EvidenceScope
