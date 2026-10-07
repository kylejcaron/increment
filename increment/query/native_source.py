"""DefinitionsMomentSource: the warehouse `MomentSource` over the
definitions/ibis pipeline - the raw-events sibling of `SqlPanelSource`,
`FrameTotalsSource`, `FramePanelSource`, `MomentsSource`. The only
substrate offering every grain plus SQL introspection, because it
alone retains raw fact tables.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import ibis
import ibis.expr.types as ir
from ibis import Table
from ibis import to_sql as ibis_to_sql

from increment._analysis_config import effective_methods, overlay_configs, resolve_configs
from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW
from increment._source_operations import DashboardGroupData
from increment._window import NO_DATA_SIGNAL, resolve_window_days
from increment.errors import (
    CapabilityError,
    CodedError,
    InvalidRequestError,
    RefusalSpec,
    _safe_error_value,
)
from increment.errors import refuse as _refuse
from increment.estimation._readout_refusals import refuse_observational_quantile
from increment.estimation.armstats import ArmStats
from increment.query.artifact_contract import ArtifactStore
from increment.query.artifact_publish import ArtifactPublisher
from increment.query.builders import (
    _apply_filter,
    _attach_uptake_flag,
    _censor_to_observable_window,
    _dense_unit_days,
    _final_maturity_day,
    _local_date,
    _observable_window_flags,
    _scope_exposure_events,
    _uptake_events_in_elapsed_window,
    _windowed_fact_sum,
    asof_group_summary,
    breakout_property_table,
    cluster_exposure_counts,
    cohort_group_summary,
    compliance_event_horizon,
    daily_exposure_counts,
    daily_group_summary,
    first_exposures,
    group_summary,
    join_breakout_dimension,
    join_pre_period_covariate,
    join_unit_covariates,
    metric_events,
    panel_spine,
    post_exposure_stats,
    pre_period_stats,
    resolved_measure_key,
    site_volume,
    triggered_population,
    union_event_horizon,
    unit_day_panel,
    unit_day_spine_stats,
    unit_totals,
    validate_site_volume_metric,
    window_bound_stats,
    winsorize_unit_totals,
)
from increment.query.fact_resolution import (
    _find_fact_source,
    _rename_to_builder_cols,
    _resolve_breakout_fact_source,
    _resolve_value_column,
)
from increment.query.integrity import (
    arm_counts,
    enforce_mixed_assignments,
    mixed_assignment_snapshot,
    validate_cluster_labels,
    validate_cluster_uniqueness,
)
from increment.query.native_contract import (
    _NATIVE_COVARIATE_RESERVED,
    DayEvidenceSource,
    DimensionedDayEvidence,
    SitewideEvidence,
)
from increment.query.session import SourceScope, WarehouseSession
from increment.semantics.artifact import UnitDayArtifactRef
from increment.semantics.design import Encouragement
from increment.semantics.models import (
    Breakout,
    ConversionMetric,
    Definitions,
    Experiment,
    Factor,
    FactSource,
    Metric,
    ObservationalDeclaration,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
)
from increment.sequential_source import SequentialSourceMixin
from increment.sources import (
    ASSIGNMENT_COUNTS_FIELD,
    COMPLIANCE_SUMMARY_FIELD,
    MIXED_ASSIGNMENT_LABEL,
    UNASSIGNED_LABEL,
    BreakoutMomentsSource,
    ComplianceArm,
    ComplianceSummary,
    MomentSource,
    SourceContext,
    SourceOperation,
)

_NATIVE_GRAIN = RefusalSpec(
    "source.native.grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_NATIVE_COVARIATE_UNRESOLVED = RefusalSpec(
    "source.native.covariate_unresolved",
    CapabilityError,
    lambda *, covariate, unit, source: (
        f"unit_frame: covariate {covariate!r} could not be resolved to a property "
        + (f"on fact source {source!r}" if source else "on any fact source")
        + f" with unit {unit!r} as an entity -- declare it as a Property on the "
        "fact source that carries it."
    ),
)
_NATIVE_COVARIATE_AMBIGUOUS = RefusalSpec(
    "source.native.covariate_ambiguous",
    CapabilityError,
    lambda *, covariate, candidates: (
        f"unit_frame: covariate {covariate!r} matches {len(candidates)} fact "
        f"sources ({', '.join(candidates)}) -- rename the property on one source, "
        "or declare the covariate under the experiment's observational design "
        "with 'source: <name>' to disambiguate."
    ),
)
_NATIVE_COVARIATE_AS_OF = RefusalSpec(
    "source.native.covariate_as_of",
    CapabilityError,
    template="unit_frame: covariate {covariate!r} (source {source!r}) has as_of={as_of!r} -- conditioning on a value measured at/after exposure biases the adjustment. Declare as_of='pre_exposure' or 'static' on the property.",
)
_NATIVE_COVARIATE_DTYPE = RefusalSpec(
    "source.native.covariate_dtype",
    CapabilityError,
    template="unit_frame: covariate {covariate!r} (source {source!r}) has dtype {dtype!r}; covariate adjustment needs a numeric (int/float/bool) or categorical (string) column -- a date carries no adjustment meaning. Derive a numeric or categorical pre-exposure property.",
)
_NATIVE_UNIT_GRAIN = RefusalSpec(
    "source.native.unit_grain",
    CapabilityError,
    template="'unit_frame' needs unit-grain rows, which this source does not retain -- only aggregated moments are available here. unit_frame() unavailable; use a frame-backed source (from_unit_summary / from_unit_panel) for unit-grain estimators.",
    keys=frozenset({"method"}),
)
_NATIVE_CLUSTER = RefusalSpec(
    "source.native.cluster_grain",
    CapabilityError,
    template="cluster_counts() is unavailable on {source}: day-axis views carry no cluster-robust variance -- a clustered experiment is refused before this source is ever constructed (Analysis._refuse_clustered_day_axis). The randomization-grain count needs a declared cluster column; build the source with from_unit_summary(..., cluster=...) or analyse from definitions declaring Experiment.cluster.",
    keys=frozenset({"because"}),
)
_NATIVE_WAREHOUSE_CLUSTER = RefusalSpec(
    "source.native.warehouse_cluster_grain",
    CapabilityError,
    template="cluster_counts() is unavailable on {source}: {because}. The randomization-grain count needs a declared cluster column; build the source with from_unit_summary(..., cluster=...) or analyse from definitions declaring Experiment.cluster.",
)
_NATIVE_SQL_GRAIN = RefusalSpec(
    "source.native_sql_grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_NATIVE_OPERATION = RefusalSpec(
    "source.native.operation",
    CapabilityError,
    template=(
        "native source cannot perform {operation!r} for {request!r}; "
        "available capability is {offered!r}. {route}"
    ),
)
_NATIVE_COMPLIANCE_DESIGN_MISMATCH = RefusalSpec(
    "source.native.compliance_design_mismatch",
    InvalidRequestError,
    template=(
        "compliance_summary design mismatch: expected source design {expected!r}, "
        "received {received!r}; rebuild the source for the requested design"
    ),
)


def _bounded_context_value(value: object) -> object:
    """One bounded, pickle-safe context value; containers one level deep."""
    if isinstance(value, Mapping):
        return {str(key): _safe_error_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_safe_error_value(item) for item in value)
    return _safe_error_value(value)


def _refuse_operation(
    *,
    operation: str,
    request: Mapping[str, _ContextValue],
    offered: _ContextValue,
    route: str,
) -> NoReturn:
    """Refuse one native operation with its actual request and served capability."""
    _refuse(
        _NATIVE_OPERATION,
        operation=operation,
        request={key: _bounded_context_value(value) for key, value in request.items()},
        offered=_bounded_context_value(offered),
        route=route,
    )


def _winsorization_bounds(winsorization: Winsorization) -> dict[str, float | None]:
    """The declared winsorization bounds as plain context values."""
    return {
        "lower_percentile": winsorization.lower_percentile,
        "upper_percentile": winsorization.upper_percentile,
        "lower_value": winsorization.lower_value,
        "upper_value": winsorization.upper_value,
    }


def _percentile_winsorization(metric: Metric) -> Winsorization | None:
    """The metric's winsorization when any declared bound is a percentile."""
    config = getattr(metric, "winsorization", None)
    return config if config is not None and config.has_percentile else None


def _refuse_unsupported_day_metric(metric: Metric, *, operation: str) -> None:
    """Reject metrics whose declared transformation has no day moments."""
    if metric.type == "quantile":
        _refuse_operation(
            operation=operation,
            request={"metric": metric.name, "metric_type": metric.type},
            offered=tuple(sorted(_DaySource.capabilities)),
            route=(
                "estimate the quantile with run(), which reads retained per-unit rows; "
                "a quantile has no moments representation"
            ),
        )
    winsorization = getattr(metric, "winsorization", None)
    if winsorization is not None:
        _refuse_operation(
            operation=operation,
            request={"metric": metric.name, "winsorization": _winsorization_bounds(winsorization)},
            offered=tuple(sorted(_DaySource.capabilities)),
            route=(
                "read the winsorized metric through run(), or drop winsorization for "
                "daily/as-of evidence; day-axis moments cannot apply a whole-window "
                "transformation"
            ),
        )


def _refuse_missing_pre_period(
    experiment: Experiment, metrics: Sequence[Metric], *, operation: str
) -> None:
    """Refuse a CUPED request the experiment declares no pre-period for.

    A ratio metric refuses under its own code: its warehouse covariate is
    the numerator's pre-period total, so ``n_pre_periods`` is its only switch.
    """
    if experiment.n_pre_periods > 0:
        return
    ratio = next((m for m in metrics if isinstance(m, RatioMetric)), None)
    if ratio is not None:
        from increment.estimation.engine import _refuse as _refuse_engine

        _refuse_engine(
            "estimation.engine.ratio.cuped", metric=ratio.name, experiment=experiment.name
        )
    _refuse_operation(
        operation=operation,
        request={
            "experiment": experiment.name,
            "metrics": tuple(metric.name for metric in metrics),
            "include_covariate": True,
        },
        offered={"n_pre_periods": experiment.n_pre_periods},
        route=(
            "declare n_pre_periods > 0 on the experiment, or request day-axis evidence "
            "without CUPED"
        ),
    )


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

    import pyarrow as pa

    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Observational, Randomized
    from increment.semantics.models import Winsorization
    from increment.sequential_state import SequentialCheckpoint, SequentialSnapshot
    from increment.sources import Grain, MomentSource

    _ContextScalar = str | int | float | bool | None
    _ContextValue = _ContextScalar | tuple[_ContextScalar, ...] | Mapping[str, _ContextScalar]


class _DaySource:
    """`MomentSource` over the definitions pipeline's per-metric day-axis
    panels - lazy: constructing it issues no query, and each
    `moments()` call builds and reduces only the metric/grain it is
    asked for, reusing `DefinitionsMomentSource._build_panel_for_metric`/
    `_build_asof_panels` as injected callables.

    `x` is a fixed pre-exposure unit covariate and `y` is daily,
    retention-cohort, or cumulative as-of. CUPED fields are populated only
    when the internal source request sets ``include_covariate=True`` and
    native pre-period moments are available; callers select a CUPED method,
    not this source-only flag.

    `capabilities` is `{"daily", "asof"}`, never `"total"`; dimensioned
    reads go through the sibling `breakout_moments()`, not a `by=` here.

    `grain="daily"` dispatches supported additive metrics by type: a
    `RetentionMetric` routes through `cohort_group_summary` off
    `build_panel`'s `spine`/`stats` legs; other supported metrics route
    through `daily_group_summary` off `build_asof_panel`'s
    `(panel, den_panel)` pair with
    `window_bound_stats` applied here (the as-of panel is deliberately
    left unbound so its own window masking still works).

    `grain="asof"` sends every supported additive metric, including
    retention, through `asof_group_summary` off `build_asof_panel`'s full
    tuple. At either grain, quantile metrics have no moments representation,
    and winsorized metrics cannot apply their whole-window transformation,
    so both are refused before dispatch. That tuple's uptake fields are
    populated only when `self._design` is an `Encouragement`;
    `build_asof_panel` resolves the uptake fact unconditionally whenever
    that's true, so a caller
    that ever sets an `Encouragement` design would make
    `grain="daily"` depend on that resolution too, even though it
    discards the result.
    """

    capabilities: frozenset[Grain] = frozenset({"daily", "asof"})
    operations: frozenset[SourceOperation] = frozenset()
    shape: Literal["unit_summary", "unit_panel"] | None = None

    # Keep source callbacks together: construction and cache state form one adapter.
    def __init__(  # noqa: PLR0913
        self,
        experiment: Experiment,
        context: SourceContext,
        get_exposures: Callable[[], ir.Table],
        build_panel: Callable[
            [ir.Table, Metric, tuple[Metric, ...]],
            tuple[
                ir.Table,
                ir.Table,
                ir.Table,
                ir.Table,
                ir.Table | None,
                ir.Table,
                str | None,
                ir.Scalar,
            ],
        ],
        build_asof_panel: Callable[
            [ir.Table, Metric, tuple[Metric, ...], bool, bool],
            tuple[ir.Table, ir.Table | None, ir.Table | None, int | None, ir.Scalar | None],
        ],
        breakouts: tuple[str, ...] = (),
        *,
        build_breakout_properties: Callable[[FactSource, str, ir.Table], ir.Table],
        defs: Definitions,
        preflight_breakout: Callable[..., FactSource],
        preflight_day_request: Callable[..., None],
        note_reduction: Callable[[], None],
        horizon_metrics: tuple[Metric, ...],
        legacy_facade: bool = False,
        compliance_provider: Callable[..., ComplianceSummary] | None = None,
        compliance_dates_provider: Callable[[], Sequence[object]] | None = None,
        validated_assignments: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        self._experiment = experiment
        self._context = context
        self._get_exposures = get_exposures
        self._exposures: ir.Table | None = None
        self._build_panel = build_panel
        self._build_asof_panel = build_asof_panel
        self._horizon_metrics = horizon_metrics
        self.breakouts: tuple[str, ...] = breakouts
        self._build_breakout_properties = build_breakout_properties
        self._defs = defs
        self._preflight_breakout = preflight_breakout
        self._preflight_day_request = preflight_day_request
        self._note_reduction = note_reduction
        self._legacy_facade = legacy_facade
        self._compliance_provider = compliance_provider
        self._compliance_dates_provider = compliance_dates_provider
        self._validated_assignments_scope = validated_assignments
        self._ready = False
        self._asof_panel_cache: dict[
            tuple[Metric, tuple[Metric, ...], bool, bool],
            tuple[ir.Table, ir.Table | None, ir.Table | None, int | None, ir.Scalar | None],
        ] = {}
        self._breakout_properties_cache: dict[tuple[str, str], ir.Table] = {}

    def compliance_dates(self) -> Sequence[object]:
        if self._compliance_dates_provider is None:
            from increment._source_types import raise_legacy_compliance_state

            raise_legacy_compliance_state(
                study_id=self.context.study_id, reason="day source has no enrollment day axis"
            )
        return self._compliance_dates_provider()

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        if self._compliance_provider is None:
            from increment._source_types import raise_legacy_compliance_state

            raise_legacy_compliance_state(
                study_id=self.context.study_id,
                reason="day source has no enrollment uptake provider",
            )
        return self._compliance_provider(
            design, as_of=as_of, completed_windows_only=completed_windows_only
        )

    def _validated_assignments(self) -> AbstractContextManager[None]:
        """The parent's once-per-series assignment validation, when it has one."""
        if self._validated_assignments_scope is None:
            return nullcontext()
        return self._validated_assignments_scope()

    def _ensure_ready(
        self,
        metric: Metric,
        *,
        dimension: str | None,
        include_covariate: bool,
        operation: str,
    ) -> ir.Table:
        """Run parent preflights, then start one lazy day-axis reduction."""
        self._preflight_day_request(
            (metric,),
            dimension=dimension,
            include_covariate=include_covariate,
            require_uptake=True,
            operation=operation,
        )
        if not self._ready:
            if not self._legacy_facade:
                self._note_reduction()
            self._exposures = self._get_exposures()
            self._ready = True
        assert self._exposures is not None
        return self._exposures

    def _effective_horizon(self, metric: Metric) -> tuple[Metric, ...]:
        """Include the metric requested after source construction."""
        if metric in self._horizon_metrics:
            return self._horizon_metrics
        return (*self._horizon_metrics, metric)

    def _cached_asof_panel(
        self,
        exposures: ir.Table,
        metric: Metric,
        # Internal covariate-inclusion flag for cached panel construction.
        include_covariate: bool,  # noqa: FBT001
        *,
        include_uptake_horizon: bool = False,
        completed_windows_only: bool = False,
    ) -> tuple[ir.Table, ir.Table | None, ir.Table | None, int | None, ir.Scalar | None]:
        horizon = self._effective_horizon(metric)
        key = (metric, horizon, include_covariate, include_uptake_horizon)
        cached = self._asof_panel_cache.get(key)
        if cached is None:
            cached = self._build_asof_panel(
                exposures, metric, horizon, include_covariate, include_uptake_horizon
            )
            self._asof_panel_cache[key] = cached
        if (
            completed_windows_only
            and (isinstance(metric, RatioMetric) or isinstance(self._context.design, Encouragement))
            and _final_maturity_day(metric) is not None
        ):
            panel, den_panel, uptake_panel, uptake_window_days, outcome_edge = cached
            data_as_of = self._build_panel(exposures, metric, horizon)[-1]
            panel = _censor_to_observable_window(
                panel, metric, self._experiment, data_as_of, warn_on_censoring=False
            )
            return panel, den_panel, uptake_panel, uptake_window_days, outcome_edge
        return cached

    def _breakout_properties(
        self, fs: FactSource, property_name: str, exposures: ir.Table
    ) -> ir.Table:
        key = (fs.name, property_name)
        cached = self._breakout_properties_cache.get(key)
        if cached is None:
            cached = self._build_breakout_properties(fs, property_name, exposures)
            self._breakout_properties_cache[key] = cached
        return cached

    def _build_pre_stats(
        self,
        exposures: ir.Table,
        metric: Metric,
        fact_tbl: ir.Table,
        value_col: str | None,
        *,
        include_covariate: bool,
    ) -> ir.Table | None:
        """Build fixed pre-period totals only for covariate consumers.

        ``include_covariate`` is an internal source requirement derived from
        requested methods; it is not a caller-facing statistical option.
        """
        if not include_covariate or self._experiment.n_pre_periods <= 0:
            return None
        pre_events = metric_events(fact_tbl, metric, value_column=value_col)
        return pre_period_stats(
            pre_events,
            exposures,
            self._experiment,
            source_key=f"{metric.name}:pre",
        )

    def _preflight_covariate(
        self,
        metric: Metric,
        # Internal covariate-inclusion flag used by preflight validation.
        include_covariate: bool,  # noqa: FBT001
        *,
        operation: str,
    ) -> None:
        if include_covariate:
            _refuse_missing_pre_period(self._experiment, (metric,), operation=operation)

    @property
    def context(self) -> SourceContext:
        return self._context

    def day_axis_source(self, *, selected: Sequence[Metric] | None = None) -> _DaySource:
        return self

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "daily",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        _refuse_unsupported_day_metric(metric, operation="moments")
        if grain not in self.capabilities:
            _refuse(
                _NATIVE_GRAIN,
                grain=grain,
                offered=self.capabilities,
            )
        if by:
            _refuse_operation(
                operation="moments",
                request={"metric": metric.name, "grain": grain, "by": tuple(by)},
                offered=tuple(sorted(self.capabilities)),
                route=(
                    "use breakout_moments(metric, breakout) for a declared breakout; "
                    "moments() serves only the undimensioned day axis"
                ),
            )
        self._preflight_covariate(metric, include_covariate, operation="moments")
        exposures = self._ensure_ready(
            metric,
            dimension=None,
            include_covariate=include_covariate,
            operation="moments",
        )

        if grain == "asof":
            panel, den_panel, uptake_panel, uptake_window_days, outcome_edge = (
                self._cached_asof_panel(
                    exposures,
                    metric,
                    include_covariate,
                    include_uptake_horizon=True,
                    completed_windows_only=completed_windows_only,
                )
            )
            asof = asof_group_summary(
                panel,
                metric,
                den_panel=den_panel,
                uptake_panel=uptake_panel,
                uptake_window_days=uptake_window_days,
                _uptake_elapsed_windowed=uptake_panel is not None,
                completed_windows_only=completed_windows_only,
                _outcome_observation_end=outcome_edge,
            )
            return cast("list[dict[str, Any]]", asof.to_pyarrow().to_pylist())

        if isinstance(metric, RetentionMetric):
            _panel, spine, stats, _events, _den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel(exposures, metric, self._effective_horizon(metric))
            )
            pre_stats = self._build_pre_stats(
                exposures,
                metric,
                fact_tbl,
                value_col,
                include_covariate=include_covariate,
            )
            cohorts = cohort_group_summary(
                spine,
                stats,
                metric,
                self._experiment,
                pre_stats=pre_stats,
                data_as_of=data_as_of,
            )
            return cast("list[dict[str, Any]]", cohorts.to_pyarrow().to_pylist())

        panel, den_panel, _uptake_panel, _uptake_window_days, _edge = self._cached_asof_panel(
            exposures, metric, include_covariate
        )
        panel = window_bound_stats(panel, metric)
        if den_panel is not None:
            den_panel = window_bound_stats(den_panel, metric)
        daily = daily_group_summary(panel, metric=metric, den_panel=den_panel)
        return cast("list[dict[str, Any]]", daily.to_pyarrow().to_pylist())

    def breakout_moments(
        self,
        metric: Metric,
        breakout: Breakout,
        *,
        grain: Grain = "daily",
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> DimensionedDayEvidence:
        """One declared breakout's dimensioned daily/as-of evidence.

        The result carries only the resolved source name and materialized rows;
        it never exposes the definitions-layer ``FactSource`` recipe.
        """
        _refuse_unsupported_day_metric(metric, operation="breakout_moments")
        if grain not in self.capabilities:
            _refuse(
                _NATIVE_GRAIN,
                grain=grain,
                offered=self.capabilities,
            )
        self._preflight_covariate(metric, include_covariate, operation="breakout_moments")
        fs = self._preflight_breakout(breakout, operation="breakout_moments")
        exposures = self._ensure_ready(
            metric,
            dimension=breakout.property,
            include_covariate=include_covariate,
            operation="breakout_moments",
        )
        properties_table = self._breakout_properties(fs, breakout.property, exposures)
        by = [breakout.property]
        if grain == "asof":
            panel, den_panel, uptake_panel, uptake_window_days, outcome_edge = (
                self._cached_asof_panel(
                    exposures,
                    metric,
                    include_covariate,
                    include_uptake_horizon=True,
                    completed_windows_only=completed_windows_only,
                )
            )
            dim_panel = join_breakout_dimension(panel, properties_table, by)
            asof = asof_group_summary(
                dim_panel,
                metric,
                by=by,
                den_panel=den_panel,
                uptake_panel=uptake_panel,
                uptake_window_days=uptake_window_days,
                _uptake_elapsed_windowed=uptake_panel is not None,
                completed_windows_only=completed_windows_only,
                _outcome_observation_end=outcome_edge,
            )
            return DimensionedDayEvidence(
                source_name=fs.name,
                rows=cast("list[dict[str, Any]]", asof.to_pyarrow().to_pylist()),
            )

        if isinstance(metric, RetentionMetric):
            _panel, spine, stats, _events, _den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel(exposures, metric, self._effective_horizon(metric))
            )
            pre_stats = self._build_pre_stats(
                exposures,
                metric,
                fact_tbl,
                value_col,
                include_covariate=include_covariate,
            )
            cohorts = cohort_group_summary(
                spine,
                stats,
                metric,
                self._experiment,
                pre_stats=pre_stats,
                by=by,
                properties_table=properties_table,
                data_as_of=data_as_of,
            )
            return DimensionedDayEvidence(
                source_name=fs.name,
                rows=cast("list[dict[str, Any]]", cohorts.to_pyarrow().to_pylist()),
            )

        panel, den_panel, _uptake_panel, _uptake_window_days, _edge = self._cached_asof_panel(
            exposures, metric, include_covariate
        )
        panel = window_bound_stats(panel, metric)
        if den_panel is not None:
            den_panel = window_bound_stats(den_panel, metric)
        dim_panel = join_breakout_dimension(panel, properties_table, by)
        daily = daily_group_summary(dim_panel, metric=metric, by=by, den_panel=den_panel)
        return DimensionedDayEvidence(
            source_name=fs.name,
            rows=cast("list[dict[str, Any]]", daily.to_pyarrow().to_pylist()),
        )

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> NoReturn:
        if outcome_stage == "raw":
            from increment.winsor import winsor_refuse

            winsor_refuse(
                "raw_state_required", "This source does not retain exact pre-winsor unit outcomes."
            )
        _refuse(_NATIVE_UNIT_GRAIN, method="unit_frame")

    def unit_counts(self) -> dict[str, int]:
        _refuse_operation(
            operation="unit_counts",
            request={"grain": "total"},
            offered=tuple(sorted(self.capabilities)),
            route=(
                "read unit counts from DefinitionsMomentSource.unit_counts(); day-axis "
                "views carry no total-grain moments"
            ),
        )

    def cluster_counts(self) -> dict[str, int]:
        _refuse(
            _NATIVE_CLUSTER,
            source="a definitions day-axis source",
            because=(
                "day-axis views carry no cluster-robust variance -- a "
                "randomization-grain count would silently be unit-grain"
            ),
        )

    def sql(self, *, grain: Grain = "daily") -> dict[str, str]:
        _refuse_operation(
            operation="sql",
            request={"grain": grain},
            offered=tuple(sorted(self.capabilities)),
            route=(
                "read day-axis rows through moments(); each call issues its own lazy "
                "query, so there is no single summary expression to render"
            ),
        )

    def close(self) -> None:
        self._exposures = None
        self._ready = False
        self._asof_panel_cache.clear()
        self._breakout_properties_cache.clear()


class TriggeredPopulationSource:
    """MomentSource view that narrows a native source to triggered units."""

    operations: frozenset[SourceOperation] = frozenset()
    shape: Literal["unit_summary", "unit_panel"] | None = None
    _population: Literal["triggered"] = "triggered"

    def __init__(self, source: Any) -> None:
        source._validate_trigger_capability(operation="triggered_source")
        self._source = source
        self.capabilities: frozenset[Grain] = frozenset({"total"})
        self.breakouts = source.breakouts
        self.shape = source.shape

    @property
    def context(self) -> SourceContext:
        return self._source.context

    @property
    def source_name(self) -> str | None:
        return getattr(self._source, "source_name", None)

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        return self._source.moments(
            metric,
            grain=grain,
            by=by,
            completed_windows_only=completed_windows_only,
            include_covariate=include_covariate,
            population="triggered",
        )

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> pa.Table:
        return self._source.unit_frame(
            metric,
            covariates=covariates,
            population="triggered",
            outcome_stage=outcome_stage,
        )

    def unit_counts(self) -> dict[str, int]:
        grain, counts, unit_counts = self._source.triggered_counts()
        return unit_counts if grain == "cluster" else counts

    def cluster_counts(self) -> dict[str, int]:
        grain, counts, _unit_counts = self._source.triggered_counts()
        if grain != "cluster":
            return self._source.cluster_counts()
        return counts

    def compliance_dates(self) -> Sequence[object]:
        _refuse_operation(
            operation="compliance_dates",
            request={"population": self._population, "grain": "asof"},
            offered=tuple(sorted(self.capabilities)),
            route="read as-of compliance from the assigned-population source",
        )

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        return self._source.compliance_summary(
            design,
            as_of=as_of,
            completed_windows_only=completed_windows_only,
            population="triggered",
        )

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        _refuse_operation(
            operation="sql",
            request={"grain": grain, "population": self._population},
            offered=tuple(sorted(self.capabilities)),
            route=(
                "read triggered moments through moments(); summary SQL describes the "
                "assigned population, not the triggered one"
            ),
        )

    def close(self) -> None:
        self._source.close()

    @contextmanager
    def readout_snapshot(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"],
        uptake_facts: Sequence[str] = (),
        include_breakouts: bool = False,
    ) -> Iterator[MomentSource]:
        with self._source.readout_snapshot(
            metrics=metrics,
            population="assigned",
            uptake_facts=uptake_facts,
            include_breakouts=include_breakouts,
        ) as pinned:
            yield self if pinned is self._source else pinned.triggered_source()


def _dashboard_value(value, name: str, unavailable: dict[str, str]) -> float | None:
    if value is None:
        unavailable.setdefault(name, "not applicable to this metric or retained evidence")
        return None
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        unavailable[name] = "outside the finite numeric range"
        return None
    return result


def _retained_dashboard_groups(
    metrics: Sequence[Metric],
    checkpoints: Mapping[str, SequentialCheckpoint],
    snapshot: SequentialSnapshot,
) -> tuple[DashboardGroupData, ...]:
    """Project retained observations without consulting current warehouse data."""
    if set(checkpoints) != {metric.name for metric in metrics}:
        _refuse_operation(
            operation="dashboard_group_data",
            request={"metrics": tuple(metric.name for metric in metrics)},
            offered=tuple(checkpoints),
            route="supply each displayed metric's retained checkpoint",
        )
    rows: list[DashboardGroupData] = []
    for metric in metrics:
        checkpoint = checkpoints[metric.name]
        if (
            checkpoint.cell.metric != metric.name
            or checkpoint.model.observable != "outcome"
            or checkpoint.cell.estimand != "itt"
            or checkpoint.cell.segment
        ):
            _refuse_operation(
                operation="dashboard_group_data",
                request={"metric": metric.name},
                offered=(checkpoint.cell.metric,),
                route="supply this metric's unsegmented outcome checkpoint",
            )
        checkpoint.verify_snapshot(snapshot)
        transformed = (
            getattr(checkpoint.model, "adjustment", None) is not None
            or getattr(metric, "winsorization", None) is not None
        )
        window = (
            metric.band if isinstance(metric, RetentionMetric) else (0, resolve_window_days(metric))
        )
        for state in (checkpoint.control, checkpoint.treatment):
            unavailable = dict.fromkeys(
                (
                    "assigned_units",
                    "event_count",
                    "excluded_not_mature",
                    "excluded_no_observed_day",
                    "excluded_other",
                    "observation_end",
                ),
                "historical accounting was not retained in this checkpoint",
            )
            observed = analysis_input = total = numerator = denominator = None
            retained_units = None
            if state.n:
                if state.law == "bernoulli":
                    assert state.successes is not None
                    retained_units = total = state.successes
                    observed = Fraction(state.successes, state.n)
                else:
                    first = state.mean[0]
                    is_ratio = state.law in ("ratio_mean", "adjusted_ratio_mean", "gaussian_ratio")
                    value = (
                        first / state.mean[1]
                        if is_ratio and state.mean[1]
                        else (None if is_ratio else first)
                    )
                    if transformed:
                        analysis_input = value
                        for name in ("observed_value", "sum_value", "numerator", "denominator"):
                            unavailable[name] = "pre-transform outcomes were not retained"
                    else:
                        observed, total = value, first * state.n
                        if is_ratio:
                            numerator, denominator = total, state.mean[1] * state.n
                            if state.mean[1] == 0:
                                unavailable["observed_value"] = "retained denominator is zero"
                        if isinstance(metric, (ConversionMetric, RetentionMetric)):
                            retained_units = int(total)
            else:
                for name in (
                    "observed_value",
                    "analysis_input_value",
                    "sum_value",
                    "numerator",
                    "denominator",
                ):
                    unavailable[name] = "no eligible units in the retained checkpoint"
                if isinstance(metric, (ConversionMetric, RetentionMetric)):
                    retained_units = 0
            if retained_units is None:
                unavailable["retained_units"] = "this metric is not a binary unit outcome"
            if window[1] is None:
                unavailable["window_end_days"] = "the metric has no fixed end day"
            rows.append(
                DashboardGroupData(
                    metric=metric.name,
                    group_id=state.group_id,
                    assigned_units=None,
                    eligible_units=state.n,
                    observed_value=_dashboard_value(observed, "observed_value", unavailable),
                    analysis_input_value=_dashboard_value(
                        analysis_input, "analysis_input_value", unavailable
                    ),
                    sum_value=_dashboard_value(total, "sum_value", unavailable),
                    event_count=None,
                    retained_units=retained_units,
                    numerator=_dashboard_value(numerator, "numerator", unavailable),
                    denominator=_dashboard_value(denominator, "denominator", unavailable),
                    excluded_not_mature=None,
                    excluded_no_observed_day=None,
                    excluded_other=None,
                    observation_end=None,
                    window_start_days=window[0],
                    window_end_days=window[1],
                    source_kind="retained_checkpoint",
                    prefix_id=checkpoint.prefix_id,
                    unavailable=unavailable,
                )
            )
    return tuple(rows)


def _dashboard_quantiles(totals: ir.Table, quantile: float) -> ir.Table:
    """Reduce linear sample quantiles using portable ordered ranks."""
    observed = totals.select("group_id", "y")
    observed = observed.filter(observed.y.notnull())
    ranked = observed.mutate(
        _rank=ibis.row_number().over(ibis.window(group_by=observed.group_id, order_by=observed.y)),
        _size=observed.y.count().over(ibis.window(group_by=observed.group_id)),
    )
    position = (ranked._size - 1) * quantile
    bounds = ranked.group_by("group_id").aggregate(
        lower=ranked.y.max(where=ranked._rank == position.floor()),
        upper=ranked.y.max(where=ranked._rank == position.ceil()),
        weight=(position - position.floor()).max(),
    )
    return bounds.select(
        "group_id",
        quantile_value=(1 - bounds.weight) * bounds.lower + bounds.weight * bounds.upper,
    )


def _observed_dashboard_group(
    metric: Metric,
    record: Mapping[str, Any],
    window: tuple[int | None, int | None],
) -> DashboardGroupData:
    """Decode one bounded warehouse aggregate into observed group evidence."""
    eligible = int(record["n"] or 0)
    assigned, no_observed, not_mature = (
        int(record[name]) for name in ("assigned", "no_observed", "not_mature")
    )
    unavailable: dict[str, str] = {}
    observed = total = numerator = denominator = None
    retained_units = 0 if isinstance(metric, (ConversionMetric, RetentionMetric)) else None
    if retained_units is None:
        unavailable["retained_units"] = "this metric is not a binary unit outcome"
    if window[1] is None:
        unavailable["window_end_days"] = "the metric has no fixed end day"
    if record["observation_end"] is None:
        unavailable["observation_end"] = "the running source has no observed metric days"
    unavailable["prefix_id"] = "pinned warehouse evidence is not a retained checkpoint"
    if eligible:
        arm = ArmStats.model_validate(
            {
                **{name: record[name] for name in ArmStats.model_fields if name in record},
                "study_id": record["experiment_id"],
            }
        )
        mean = arm.mean_y() * record["scale_y"]
        total = mean * eligible
        if isinstance(metric, RatioMetric):
            den_mean = arm.mean_den() * record["scale_y_den"]
            numerator, denominator = total, den_mean * eligible
            observed = mean / den_mean if den_mean else None
            if den_mean == 0:
                unavailable["observed_value"] = "group denominator is zero"
        elif isinstance(metric, QuantileMetric):
            observed = record["quantile_value"]
        else:
            observed = mean
            if retained_units is not None:
                retained_units = round(total)
    else:
        unavailable["observed_value"] = "no eligible observed units"
        unavailable["sum_value"] = "no eligible observed units"
    return DashboardGroupData(
        metric=metric.name,
        group_id=str(record["group_id"]),
        assigned_units=assigned,
        eligible_units=eligible,
        observed_value=_dashboard_value(observed, "observed_value", unavailable),
        analysis_input_value=_dashboard_value(None, "analysis_input_value", unavailable),
        sum_value=_dashboard_value(total, "sum_value", unavailable),
        event_count=int(record["event_count"] or 0),
        retained_units=retained_units,
        numerator=_dashboard_value(numerator, "numerator", unavailable),
        denominator=_dashboard_value(denominator, "denominator", unavailable),
        excluded_not_mature=not_mature,
        excluded_no_observed_day=no_observed,
        excluded_other=assigned - eligible - no_observed - not_mature,
        observation_end=record["observation_end"],
        window_start_days=window[0],
        window_end_days=window[1],
        source_kind="pinned_warehouse",
        prefix_id=None,
        unavailable=unavailable,
    )


class DefinitionsMomentSource(SequentialSourceMixin):
    """MomentSource over the definitions/ibis pipeline - the warehouse
    sibling of SqlPanelSource, FrameTotalsSource, FramePanelSource,
    MomentsSource. The one substrate offering every grain plus SQL
    introspection, because it alone retains raw fact tables.

    Public ``design``/``plan`` read-only properties mirror the frame
    sources' convention (``FrameTotalsSource.design``/``.plan``,
    ``FramePanelSource.design``/``.plan``): the identification mechanism
    and the fully-resolved ``AnalysisPlan`` this instance was constructed
    under. ``plan`` is a required constructor argument, so it is always
    set (never ``None``). ``design`` defaults to ``None`` for direct source
    construction, while ``Analysis.__init__`` resolves and passes the
    declared design on the native facade path. This instance has no
    ``design`` setter; callers that need a different design should build a
    new source through the appropriate constructor.

    """

    capabilities: frozenset[Grain] = frozenset({"total", "daily", "asof"})
    operations: frozenset[SourceOperation] = frozenset(
        {
            "allocation_history",
            "readout_snapshot",
            "dashboard_group_data",
            "materialize",
            "triggered_counts",
            "triggered_source",
            "sitewide_evidence",
            "export_moments",
            "moments_source",
            "panel_sql",
            "summary_sql",
            "breakout_summaries",
            "factor_summaries",
            "breakout_source",
            "breakout_sources",
            "day_source",
            "exploratory_source",
        }
    )
    shape: Literal["unit_summary", "unit_panel"] | None = None

    def __init__(
        self,
        session: WarehouseSession,
        experiment: Experiment,
        metrics: Sequence[Metric],
        *,
        store: Literal["auto", "always", "none"],
        on_mixed_assignment: Literal["error", "warn", "exclude"],
        backend: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: CompiledDecisionPlan,
    ) -> None:
        self._session = session
        self._con = session.con
        self._defs: Definitions = session.defs
        self._experiment = experiment
        self._experiment_name = experiment.name
        self.breakouts: Sequence[str] = tuple(sorted({b.property for b in experiment.breakouts}))
        self._metrics: list[Metric] = list(metrics)
        self._horizon_metrics = tuple(self._metrics)
        self._store: Literal["auto", "always", "none"] = store
        self._on_mixed_assignment: Literal["error", "warn", "exclude"] = on_mixed_assignment
        self._backend = backend
        from increment.plan import refuse_observational_relative_margin

        refuse_observational_relative_margin(design, plan)

        self._context = SourceContext(
            study_id=self._experiment_name,
            design=design,
            plan=plan,
            metrics=self._horizon_metrics,
            configs=resolve_configs(
                self._metrics,
                bindings=self._experiment.bindings,
                specs=None,
                methods=None,
                prior=None,
            ),
            cluster=self._experiment.cluster,
            intervention_grain=self._experiment.intervention_grain,
        )
        self._exposures_cache: Table | None = None
        from increment.sequential_source import validate_source_mapping

        validate_source_mapping(self._context, self._sequential_observation_mapping())
        self._trigger_cache: Table | None = None
        self._mixed_assignment_units: int | None = None
        self._unassigned_assignment_units: int | None = None
        self._mixed_assignment_fingerprint: int | None = None
        self._mixed_assignment_warned = False
        self._union_horizon_cache: ir.Scalar | None = None
        self._union_horizon_metric_cache: dict[tuple[Metric, ...], ir.Scalar] = {}
        self._panel_cache: dict[
            tuple[str, Metric] | tuple[str, Metric, tuple[Metric, ...]],
            tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar],
        ] = {}
        self._reduction_calls = 0
        self._reduction_batch_depth = 0
        self._assignment_validation_depth = 0
        self._materialized = False
        self._exp_lookup_exposures = {e.name: e for e in self._defs.exposures}

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def design(self):
        """The identification design captured at construction."""
        return self._context.design

    @property
    def plan(self) -> CompiledDecisionPlan:
        """The compiled decision plan captured at construction."""
        return self._context.plan

    def _get_exposures(self) -> Table:
        """Return the cached lazy exposure expression without executing it."""
        if self._exposures_cache is None:
            exposure_events_table = self._build_exposure_events_table()
            self._exposures_cache = first_exposures(exposure_events_table, self._experiment)
        return self._exposures_cache

    def _get_trigger_population(self, *, operation: str) -> Table:
        """The enrolled units that also produced the declared trigger.

        Resolved through the same first-occurrence machinery as
        enrollment, then semi-joined onto it. Cached separately from
        ``_get_exposures``, which must keep serving the unnarrowed set.
        """
        if self._trigger_cache is None:
            self._validate_trigger_capability(operation=operation)
            experiment = self._experiment
            trigger_events = self._build_trigger_events_table()
            triggers = first_exposures(trigger_events, experiment)
            self._trigger_cache = triggered_population(self._get_exposures(), triggers)
        return self._trigger_cache

    def _validate_trigger_capability(self, *, operation: str) -> None:
        """Refuse a triggered population before resolving warehouse tables."""
        experiment = self._experiment
        if experiment.trigger is None:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": experiment.name,
                    "population": "triggered",
                    "trigger": experiment.trigger,
                },
                offered=("assigned",),
                route="declare `trigger: <exposure name>` on the experiment",
            )

    def _invalidate_materialization(self) -> None:
        """Drop the previous operation's TEMP tables and cached panel expressions."""
        if not self._materialized:
            return
        self._session.drop_materialized()
        self._materialized = False
        self._panel_cache = {}

    def _mixed_assignment_snapshot(self, exposure_events: Table) -> tuple[int, int, int]:
        """Count mixed and NULL assignments with one bounded aggregate."""
        return mixed_assignment_snapshot(self._con, exposure_events, self._experiment)

    def _mixed_assignment_count(self, exposure_events: Table) -> int:
        """Count mixed units with one bounded aggregate result."""
        return self._mixed_assignment_snapshot(exposure_events)[0]

    def _validate_mixed_assignments(self) -> None:
        """Refresh and enforce assignment integrity for one readout."""
        if self._assignment_validation_depth:
            return
        exposure_events_table = self._build_exposure_events_table()
        mixed, unassigned, fingerprint = self._mixed_assignment_snapshot(exposure_events_table)
        previous_fingerprint = getattr(self, "_mixed_assignment_fingerprint", None)
        self._mixed_assignment_units = mixed
        self._unassigned_assignment_units = unassigned
        self._mixed_assignment_fingerprint = fingerprint
        if previous_fingerprint is not None and previous_fingerprint != fingerprint:
            self._invalidate_materialization()
        self._mixed_assignment_warned = enforce_mixed_assignments(
            mixed,
            getattr(self, "_on_mixed_assignment", "error"),
            unassigned_count=unassigned,
            already_warned=getattr(self, "_mixed_assignment_warned", False),
        )

    @contextmanager
    def _validated_assignments(self) -> Iterator[None]:
        """Validate assignments once for a series of per-date reads."""
        self._validate_mixed_assignments()
        self._assignment_validation_depth += 1
        try:
            yield
        finally:
            self._assignment_validation_depth -= 1

    def _get_fact_table(self, fact_source: FactSource) -> Table:
        """Memoized fact table for *fact_source*, via the shared warehouse session."""
        return self._session.fact_table(fact_source, self._experiment.unit)

    def _site_volume_scope_names(
        self, site_volume_metrics: frozenset[str], sources: dict[str, FactSource]
    ) -> tuple[set[str], set[str]]:
        """Resolve which fact sources (and the dims they join) back a
        requested site_volume metric, so `_pinned_source` can drop their
        unit hint without touching the window hint.

        A site_volume request reads whole-population events in [start, end]
        (artifact_publish.py's own _relevant_events site branch), never
        scoped to enrolled units -- resolve which fact source(s) back each
        requested site-volume metric name the same way add_fact already
        resolves an ordinary metric's fact. A dim a site_volume-referenced
        fact source joins (e.g. a property the metric filters on) must not
        keep its own unit filter either: the fact source's rows now include
        non-enrolled units, but a dim scoped to enrolled units only would
        left-join those rows to a NULL property and silently fail any
        filter on it, re-shrinking the "whole site" count back to roughly
        the enrolled population through the join instead of through the
        fact source directly.
        """
        site_volume_source_names: set[str] = set()
        if site_volume_metrics:
            for metric in self._metrics:
                if metric.name not in site_volume_metrics:
                    continue
                fact_names = (
                    (metric.numerator.fact, metric.denominator.fact)
                    if isinstance(metric, RatioMetric)
                    else (metric.fact,)
                )
                for fact_name in fact_names:
                    fact_source, _ = _find_fact_source(self._defs, fact_name)
                    site_volume_source_names.add(fact_source.name)
        site_volume_dim_names = {
            dim_name
            for source in sources.values()
            if source.name in site_volume_source_names
            for dim_name in source.dims
        }
        return site_volume_source_names, site_volume_dim_names

    def _scope_enrolled_units(self) -> Table:
        """Distinct enrolled ``(unit_id, experiment_id)`` pairs, used only to
        narrow the pinned snapshot's non-exposure branches (`SourceScope`).

        Reads the exposure's raw rows directly (``dim_joined=False``)
        rather than through `_get_exposures`'s dim-joined, first-exposure
        machinery: population membership never depends on a joined
        dimension property or on picking one row per unit -- being
        over-inclusive here only means a scoped branch copies a few extra
        rows, never that it drops one (see the margins note below). Reusing
        `_get_exposures` would force a second dim-joined build of the same
        fact source `_assemble_publication` builds again from the pinned
        snapshot.
        """
        events = self._build_events_table_for_exposure(self._experiment.exposure, dim_joined=False)
        return events.select("unit_id", "experiment_id").distinct()

    def _pinned_source(
        self,
        *,
        metrics: Sequence[Metric] | None = None,
        extra_sources: Sequence[str] = (),
        uptake_facts: Sequence[str] = (),
        site_volume_metrics: frozenset[str] = frozenset(),
    ) -> DefinitionsMomentSource:
        """Capture the experiment's input streams before any data validation."""
        if self._session._source_tables is not None:
            return self
        sources: dict[str, FactSource] = {}
        queries: list[str] = []
        column_hints: dict[str, tuple[str | None, str | None]] = {}
        # Exposure/trigger SQL stays unscoped: it defines enrollment and the
        # observation window, so filtering it by those same values is circular.
        # Shared fact sources also remain unscoped.
        unscoped_queries: set[str] = set()

        def add_fact(fact: str, *, unscoped: bool = False) -> str:
            source, _ = _find_fact_source(self._defs, fact)
            sources[source.name] = source
            if unscoped:
                unscoped_queries.add(source.sql)
            return source.name

        # Sources `_data_as_of` reads (declared/selected metrics' own
        # numerator/denominator facts): freshness must see events for every
        # unit, so these never get the enrolled-units semi-join, unlike
        # other fact sources this snapshot also captures.
        freshness_source_names: set[str] = set()

        for metric in self._metrics if metrics is None else metrics:
            if isinstance(metric, RatioMetric):
                freshness_source_names.add(add_fact(metric.numerator.fact))
                freshness_source_names.add(add_fact(metric.denominator.fact))
            else:
                freshness_source_names.add(add_fact(metric.fact))

        for name in (self._experiment.exposure, self._experiment.trigger):
            if name is None:
                continue
            exposure = self._exp_lookup_exposures[name]
            if exposure.sql is not None:
                queries.append(exposure.sql)
                unscoped_queries.add(exposure.sql)
            else:
                assert exposure.fact is not None
                add_fact(exposure.fact, unscoped=True)
        if metrics is None:
            uptake = getattr(self._context.design, "uptake", None)
            if uptake is not None:
                add_fact(uptake.fact)
        for fact in uptake_facts:
            add_fact(fact)
        # Sources read only for a breakout/factor/covariate property lookup:
        # `_build_breakout_properties_table` takes the latest value over the
        # whole source (static) or strictly before exposure (pre_exposure),
        # so these never get the window's lower time bound either.
        property_source_names: set[str] = set(extra_sources)
        for source_name in extra_sources:
            source = next(fs for fs in self._defs.fact_sources if fs.name == source_name)
            sources[source.name] = source
        # An observational design's covariates are read the same way a
        # breakout property is (see below): pin them unconditionally, since
        # every readout of this design needs `unit_frame`'s covariate join
        # to resolve, not only an explicit publish request.
        self._add_observational_covariate_sources(sources, property_source_names)
        site_volume_source_names, site_volume_dim_names = self._site_volume_scope_names(
            site_volume_metrics, sources
        )
        freshness_dim_names = {
            dim_name
            for source in sources.values()
            if source.name in freshness_source_names
            for dim_name in source.dims
        }
        dimensions = {name for source in sources.values() for name in source.dims}
        for source in sources.values():
            queries.append(source.sql)
            column_hints[source.sql] = self._source_column_hint(
                source,
                unscoped_queries=unscoped_queries,
                site_volume_source_names=site_volume_source_names,
                freshness_source_names=freshness_source_names,
                property_source_names=property_source_names,
            )
        dim_by_name = {d.name: d for d in self._defs.dim_sources}
        for dim_name in dimensions:
            dim = dim_by_name[dim_name]
            queries.append(dim.sql)
            if dim_name in site_volume_dim_names or dim_name in freshness_dim_names:
                column_hints[dim.sql] = (None, None)
            else:
                unit_col = dim.entity if dim.entity == self._experiment.unit else None
                column_hints[dim.sql] = (unit_col, None)
        enrolled = self._scope_enrolled_units()
        # The generous lower margin limits only the snapshot copy, not builder
        # windows, and absorbs day-boundary edge cases. Add no upper ts bound:
        # `_data_as_of` needs later events to prove freshness; a horizon cap would
        # stop it at the last in-window event and censor every enrolled unit.
        window_start = self._experiment.start - dt.timedelta(
            days=self._experiment.n_pre_periods + 1
        )
        scope = SourceScope(enrolled_units=enrolled, window_start=window_start)
        pinned = DefinitionsMomentSource(
            self._session.pin_sources(tuple(queries), column_hints=column_hints, scope=scope),
            self._experiment,
            self._metrics,
            store="none",
            on_mixed_assignment=self._on_mixed_assignment,
            backend=self._backend,
            design=self._context.design,
            plan=self._context.plan,
        )
        # Preserve registration metadata, but derive freshness only from captured streams.
        pinned._horizon_metrics = tuple(self._metrics if metrics is None else metrics)
        if self._sequential_snapshot is not None:
            pinned.adopt_sequential_snapshot(self._sequential_snapshot)
        return pinned

    def _add_observational_covariate_sources(
        self, sources: dict[str, FactSource], property_source_names: set[str]
    ) -> None:
        """Pin every source an observational design's declared covariates
        resolve to, the same way a breakout property source is pinned."""
        design = self._experiment.design
        if not isinstance(design, ObservationalDeclaration):
            return
        for covariate in design.covariates:
            cov_source = self._resolve_unit_covariate_source(covariate.property)
            sources[cov_source.name] = cov_source
            property_source_names.add(cov_source.name)

    def _source_column_hint(
        self,
        source: FactSource,
        *,
        unscoped_queries: set[str],
        site_volume_source_names: set[str],
        freshness_source_names: set[str],
        property_source_names: set[str],
    ) -> tuple[str | None, str | None]:
        """The `(unit_column, timestamp_column)` hint `pin_sources` scopes
        *source* by: never for an unscoped (exposure/trigger) or site-volume
        source; a freshness-bearing source (declared/selected metrics'
        numerator/denominator) never by unit, since `_data_as_of` must see
        every unit; a property-bearing source (breakout/factor/covariate)
        never by the window's lower time bound, since it takes the latest
        value over the whole source or strictly before exposure."""
        if source.sql in unscoped_queries:
            return None, None
        if source.name in site_volume_source_names:
            return None, source.timestamp_column
        unit_col = (
            self._experiment.unit
            if self._experiment.unit in source.entities
            and source.name not in freshness_source_names
            else None
        )
        ts_col = None if source.name in property_source_names else source.timestamp_column
        return unit_col, ts_col

    @contextmanager
    def _pinned_source_execution(
        self,
        *,
        metrics: Sequence[Metric] | None = None,
        extra_sources: Sequence[str] = (),
        uptake_facts: Sequence[str] = (),
        site_volume_metrics: frozenset[str] = frozenset(),
    ):
        """Bind assignment, outcomes, freshness and extensions to one execution."""
        pinned = self._pinned_source(
            metrics=metrics,
            extra_sources=extra_sources,
            uptake_facts=uptake_facts,
            site_volume_metrics=site_volume_metrics,
        )
        try:
            yield pinned
        finally:
            if pinned is not self:
                pinned.close()

    def _declared_breakout_sources(self) -> tuple[str, ...]:
        """Fact sources backing the declared breakouts, in declaration order.

        A breakout whose property cannot be resolved is skipped here: the readout that needs it
        refuses with the resolver's own coded error, so pinning must not pre-empt that refusal.
        """
        names: dict[str, None] = {}
        for breakout in self._experiment.breakouts:
            try:
                source = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
                names[source.name] = None
            except CodedError:
                continue
        return tuple(names)

    @contextmanager
    def readout_snapshot(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"],
        uptake_facts: Sequence[str] = (),
        include_breakouts: bool = False,
    ) -> Iterator[MomentSource]:
        """Pin selected metrics, requested uptake streams, and optionally breakout properties.

        ``include_breakouts`` also captures each declared breakout's property stream, so a
        segment or dimensioned readout of the pinned source reads the same execution snapshot
        as the headline instead of failing on, or silently re-reading, an unpinned stream.
        """
        with self._pinned_source_execution(
            metrics=metrics,
            uptake_facts=uptake_facts,
            extra_sources=self._declared_breakout_sources() if include_breakouts else (),
        ) as pinned:
            yield pinned if population == "assigned" else pinned.triggered_source()

    def exploratory_source(self, *, metrics: Sequence[Metric]) -> DefinitionsMomentSource:
        """A sibling source over the declared metrics plus *metrics*, in this session.

        Each added metric compiles the default unassigned procedure at this plan's alpha
        (:func:`increment.plan.with_unassigned_procedures`), so no declared procedure,
        family, or configuration changes. It shares this source's session, so a pinned
        snapshot serves it from the same pin; it never materializes (``store="none"``)
        and holds nothing to close.
        """
        from increment.plan import with_unassigned_procedures

        added = tuple(metrics)
        plan = with_unassigned_procedures(self._context.plan, added, design=self._context.design)
        sibling = DefinitionsMomentSource(
            self._session,
            self._experiment,
            [*self._metrics, *added],
            store="none",
            on_mixed_assignment=self._on_mixed_assignment,
            backend=self._backend,
            design=self._context.design,
            plan=plan,
        )
        sibling._horizon_metrics = (
            *self._horizon_metrics,
            *(metric for metric in added if metric not in self._horizon_metrics),
        )
        return sibling

    def _metric_primary_events(self, metric: Metric) -> Table:
        """This metric's own primary events table (numerator events for a
        ``RatioMetric``).

        Reuses ``_get_fact_table``'s cache, so calling this per declared
        metric (as ``_union_event_horizon`` does) never re-scans a fact
        source already touched by this readout.
        """
        if isinstance(metric, RatioMetric):
            num_fs, num_fact_def = _find_fact_source(self._defs, metric.numerator.fact)
            num_fact_tbl = self._get_fact_table(num_fs)
            num_value_col = _resolve_value_column(num_fs, num_fact_def)
            return metric_events(num_fact_tbl, metric, value_column=num_value_col, part="numerator")
        fs, fact_def = _find_fact_source(self._defs, metric.fact)
        fact_tbl = self._get_fact_table(fs)
        value_col = _resolve_value_column(fs, fact_def)
        return metric_events(fact_tbl, metric, value_column=value_col)

    def _union_event_horizon(self, metrics: Sequence[Metric] | None = None) -> ir.Scalar:
        """Memoized union of declared and any caller-selected metrics.

        Day-axis callers can supply an effective metric variant (including
        an ad-hoc fact) without changing this source's declared context. The
        selected streams must still extend the shared spine, or late events
        silently disappear from daily/as-of reductions.
        """
        if metrics is None:
            horizon_metrics = self._horizon_metrics
        else:
            horizon_metrics_list = list(self._horizon_metrics)
            for metric in metrics:
                if metric not in horizon_metrics_list:
                    horizon_metrics_list.append(metric)
            horizon_metrics = tuple(horizon_metrics_list)

        if horizon_metrics == self._horizon_metrics:
            if self._union_horizon_cache is None:
                self._union_horizon_cache = self._union_event_horizon_for(horizon_metrics)
            return self._union_horizon_cache

        cached = self._union_horizon_metric_cache.get(horizon_metrics)
        if cached is None:
            cached = self._union_event_horizon_for(horizon_metrics)
            self._union_horizon_metric_cache[horizon_metrics] = cached
        return cached

    def _union_event_horizon_for(self, metrics: Sequence[Metric]) -> ir.Scalar:
        """Build one event horizon for an explicit effective metric set."""
        event_tables = [self._metric_primary_events(m) for m in metrics]
        for metric in metrics:
            if isinstance(metric, RatioMetric):
                den_fs, den_fact_def = _find_fact_source(self._defs, metric.denominator.fact)
                den_tbl = self._get_fact_table(den_fs)
                den_col = _resolve_value_column(den_fs, den_fact_def)
                event_tables.append(
                    metric_events(den_tbl, metric, value_column=den_col, part="denominator")
                )
        return union_event_horizon(event_tables, self._experiment)

    def materialize(self) -> None:
        """Rebuild materialized tables from current warehouse state.

        Each explicit call rescans the source; reuse is confined to one
        operation. No materialization is needed for store='none' or no metrics.
        """
        if self._store == "none" or not self._metrics:
            return
        self._validate_mixed_assignments()
        self._invalidate_materialization()
        self._materialize()
        self._reduction_calls += 1

    def _note_reduction(self) -> None:
        """Called once per top-level readout call (never for panel_sql/summary_sql).

        Each call past the store threshold rebuilds from current warehouse
        state. Nested reductions share that operation's materialization.
        """
        self._validate_mixed_assignments()
        self._reduction_calls += 1
        if self._store == "none":
            return
        threshold = 1 if self._store == "always" else 2
        if self._reduction_calls >= threshold:
            self._invalidate_materialization()
            self._materialize()

    @property
    def _publisher(self) -> ArtifactPublisher:
        # Built per call: bound accessors and audit counters must stay live.
        return ArtifactPublisher(
            experiment=self._experiment,
            definitions=self._defs,
            connection=self._con,
            metrics=self._metrics,
            on_mixed_assignment=self._on_mixed_assignment,
            no_data_signal=self._NO_DATA_SIGNAL,
            get_exposures=self._get_exposures,
            get_fact_table=self._get_fact_table,
            build_breakout_properties_table=self._build_breakout_properties_table,
            counts_for_population=partial(
                self._counts_for_population, operation="publish_unit_day_artifact"
            ),
            get_trigger_population=partial(
                self._get_trigger_population, operation="publish_unit_day_artifact"
            ),
            validate_cluster_labels=self._validate_cluster_labels,
            validate_mixed_assignments=self._validate_mixed_assignments,
            data_as_of=self._data_as_of,
        )

    def publish_unit_day_artifact(
        self,
        store: ArtifactStore,
        *,
        extensions: Sequence[Any] = (),
        refresh_of: UnitDayArtifactRef | None = None,
    ) -> UnitDayArtifactRef:
        """Publish an immutable unit-day artifact; failures are not atomic.
        Inspect visible manifests and verify backing relations or drop the
        affected generation before retrying; see the data model guide.
        """
        return self._publisher.publish(
            store, extensions=extensions, source=self, refresh_of=refresh_of
        )

    @contextmanager
    def _reduction_batch(self):
        """Count one source-owned reduction for a nested batch of views."""
        outer = self._reduction_batch_depth == 0
        if outer:
            self._note_reduction()
        self._reduction_batch_depth += 1
        try:
            yield
        finally:
            self._reduction_batch_depth -= 1

    def _temp_name(self, *parts: str) -> str:
        """Backend-portable temp-table name, via the shared warehouse session."""
        return self._session.temp_name(*parts)

    def _materialize_table(self, name: str, expr: ir.Table) -> ir.Table:
        """``CREATE TEMP TABLE``, degrading to unmaterialized on failure, via the session."""
        return self._session.materialize_table(name, expr)

    def _materialize(self) -> None:
        """Materialize the shared spine and one stats table per distinct measure.

        Within one operation, several declared metrics can resolve to
        the same underlying measure (same fact, filters, and value
        column); their ``stats`` tables agree in every load-bearing
        column, so this keys materialized stats by
        ``resolved_measure_key`` rather than metric name - N metrics
        sharing one measure get one ``CREATE TEMP TABLE`` instead of N.
        Uses ``_union_event_horizon``, the same memoized bound
        ``_build_panel_for_metric_impl`` threads into every spine, so the
        materialized spine is numerically identical to a fused readout's.
        """
        exposures = self._get_exposures()
        raw: dict[
            str, tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar]
        ] = {}
        for metric in self._metrics:
            raw[metric.name] = self._build_panel_for_metric_impl(exposures, metric)

        spine_name = self._temp_name(self._experiment_name, "spine")
        spine_expr = panel_spine(exposures, self._experiment, end_date=self._union_event_horizon())
        spine = self._materialize_table(spine_name, spine_expr)

        # Metrics sharing a measure (see `resolved_measure_key`) share one
        # materialized stats table instead of each re-scanning the same events.
        measure_tables: dict[
            tuple[
                str, tuple[tuple[str, str, tuple[str | int | float | bool, ...]], ...], str | None
            ],
            Table,
        ] = {}
        for metric in self._metrics:
            panel, _spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = raw[
                metric.name
            ]
            measure_ref = metric.numerator if isinstance(metric, RatioMetric) else metric
            measure_key = resolved_measure_key(measure_ref, value_column=value_col)
            materialized_stats = measure_tables.get(measure_key)
            if materialized_stats is None:
                stats_name = self._temp_name(self._experiment_name, "stats", metric.name)
                materialized_stats = self._materialize_table(stats_name, stats)
                measure_tables[measure_key] = materialized_stats
            self._panel_cache[("assigned", metric)] = (
                panel,
                spine,
                materialized_stats,
                raw[metric.name][3],
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
            )
        self._materialized = True

    def _build_breakout_properties_table(
        self, fact_source: FactSource, property_name: str, exposures: Table
    ) -> Table:
        """Build a deduplicated ``(unit_id, <property_name>)`` table for a
        breakout dimension or a unit covariate.

        One row per unit. ``Property.as_of == "static"`` takes the
        latest value over the whole source; ``"pre_exposure"`` takes the
        latest value strictly before the unit's ``first_exposure_ts``, so
        a segment never conditions on a post-exposure value.
        ``"event_time"`` never reaches here - it is rejected at
        definitions-load time. *exposures* is required so the
        ``pre_exposure`` branch is always exercisable; a unit with no
        qualifying value is absent from the result and lands in the
        ``"__null__"`` bin via the caller's left join.
        """
        raw = self._get_fact_table(fact_source)
        prop_def = next(p for p in self._defs.properties_of(fact_source) if p.name == property_name)
        scoped_exposures = exposures if prop_def.as_of == "pre_exposure" else None
        return breakout_property_table(raw, property_name, scoped_exposures)

    def _resolve_unit_covariate_source(self, name: str) -> FactSource:
        """Resolve a covariate the way a breakout resolves: an explicit
        ``source:`` on a declared observational covariate wins; otherwise
        exactly one fact source with the unit must carry the property. The
        property must be pre-exposure/static and numeric (int/float/bool)
        or categorical (string)."""
        unit = self._experiment.unit
        design = self._experiment.design
        declared = (
            next((c for c in design.covariates if c.property == name), None)
            if isinstance(design, ObservationalDeclaration)
            else None
        )
        if declared is not None and declared.source is not None:
            candidates = [
                fs
                for fs in self._defs.fact_sources
                if fs.name == declared.source
                and unit in fs.entities
                and any(p.name == name for p in self._defs.properties_of(fs))
            ]
            if not candidates:
                _refuse(
                    _NATIVE_COVARIATE_UNRESOLVED,
                    covariate=name,
                    unit=unit,
                    source=declared.source,
                )
        else:
            candidates = [
                fs
                for fs in self._defs.fact_sources
                if unit in fs.entities and any(p.name == name for p in self._defs.properties_of(fs))
            ]
            if not candidates:
                _refuse(_NATIVE_COVARIATE_UNRESOLVED, covariate=name, unit=unit, source=None)
            if len(candidates) > 1:
                _refuse(
                    _NATIVE_COVARIATE_AMBIGUOUS,
                    covariate=name,
                    candidates=tuple(sorted(fs.name for fs in candidates)),
                )
        fact_source = candidates[0]
        prop = next(p for p in self._defs.properties_of(fact_source) if p.name == name)
        if prop.as_of == "event_time":
            _refuse(
                _NATIVE_COVARIATE_AS_OF,
                covariate=name,
                source=fact_source.name,
                as_of=prop.as_of,
            )
        if prop.dtype not in ("int", "float", "bool", "string"):
            _refuse(
                _NATIVE_COVARIATE_DTYPE,
                covariate=name,
                source=fact_source.name,
                dtype=prop.dtype,
            )
        return fact_source

    def _build_exposure_events_table(self) -> Table:
        """Build the raw exposure-events table from the experiment's exposure.

        Scoped to this experiment's ``experiment_id``: two experiments
        can share one fact-based exposure definition (e.g. both trigger
        on "first page view"), so without this filter ``first_exposures``
        would silently mix cohorts whenever an ``experiment_id`` value
        collides across experiments sharing an exposure.
        """
        return self._build_events_table_for_exposure(self._experiment.exposure)

    def _build_trigger_events_table(self) -> Table:
        """Same as :meth:`_build_exposure_events_table`, for the declared
        trigger's own exposure definition rather than the assignment one.
        """
        assert self._experiment.trigger is not None
        return self._build_events_table_for_exposure(self._experiment.trigger)

    def _build_events_table_for_exposure(
        self, exposure_name: str, *, dim_joined: bool = True
    ) -> Table:
        """Shared body: raw events for the named exposure, scoped to this
        experiment's ``experiment_id`` (see :meth:`_build_exposure_events_table`).
        """
        exposure_def = self._exp_lookup_exposures[exposure_name]
        if exposure_def.sql is not None:
            tbl = self._session.source_sql(exposure_def.sql)
            # first_exposures() requires unit_id/ts/group_id; a documented
            # (entity_id, first_exposure_ts)-only result would otherwise
            # surface as an opaque AttributeError deep inside first_exposures.
            required = frozenset({"unit_id", "ts", "group_id"})
            missing = sorted(required - set(tbl.columns))
            if missing:
                _refuse_operation(
                    operation="exposure_events",
                    request={
                        "experiment": self._experiment.name,
                        "exposure": exposure_name,
                        "required": tuple(sorted(required)),
                        "missing": tuple(missing),
                    },
                    offered=tuple(sorted(tbl.columns)),
                    route=(
                        "select unit_id, ts, and group_id (plus an optional experiment_id) "
                        "in the exposure sql; enrollment and arm assignment read them"
                    ),
                )
            if "experiment_id" in tbl.columns:
                tbl = tbl.filter(tbl.experiment_id == self._experiment.name)
            else:
                # The column is documented as optional, and downstream reductions
                # read it unconditionally, so supply it rather than crashing on a
                # query that returned exactly the required three columns.
                tbl = tbl.mutate(experiment_id=ibis.literal(self._experiment.name))
            return tbl

        # Exposure._exactly_one_source enforces sql XOR fact, so fact is
        # guaranteed non-None once sql is None.
        assert exposure_def.fact is not None
        # Fact-based exposure: check raw columns for experiment_id first. On a sourceless
        # fact table, _get_fact_table synthesizes an all-NULL one, so a post-rename check
        # would always pass and silently drop every row.
        fs, _fact = _find_fact_source(self._defs, exposure_def.fact)
        raw_has_experiment_id = "experiment_id" in self._session.source_sql(fs.sql).columns
        # Scoping before pinning (`dim_joined=False`) skips the dim join when the
        # source declares no dims: scoping reads no joined property, and
        # `_assemble_publication` rebuilds the joined table from the snapshot.
        # Declared dims still join because `exposure_def.filters` may read them.
        if dim_joined or fs.dims:
            tbl = self._get_fact_table(fs)
        else:
            tbl = _rename_to_builder_cols(
                self._session.source_sql(fs.sql), fs, self._experiment.unit
            )
        tbl = tbl.filter(tbl.event == exposure_def.fact)
        if raw_has_experiment_id:
            # NULL experiment_id can mark unrelated rows sharing this fact, such as
            # pre-period events, and those stay excluded. An all-NULL column carries
            # no signal, so coalesce it like a missing column rather than enrolling
            # nobody; one windowed aggregate detects that case.
            column_unused = ~tbl.experiment_id.notnull().any()
            tbl = tbl.mutate(
                experiment_id=ibis.ifelse(column_unused, self._experiment.name, tbl.experiment_id)
            )
            tbl = tbl.filter(tbl.experiment_id == self._experiment.name)
        # Declared Exposure.filters (fact-based exposures only; validated at load
        # time) narrow enrollment.
        for flt in exposure_def.filters:
            tbl = _apply_filter(tbl, flt)
        return tbl

    def _build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        population: Literal["assigned", "triggered"] = "assigned",
        *,
        horizon_metrics: Sequence[Metric] | None = None,
        use_cache: bool = True,
    ) -> tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar]:
        """Cached wrapper over ``_build_panel_for_metric_impl``.

        The cache is keyed by the effective metric object, not just its
        name, so a same-name caller variant cannot reuse a declared panel.
        Expanded horizons use a horizon-aware key, keeping them separate
        from the default entry with its shorter spine.
        Public lazy queries bypass this operation-owned cache so subsequent
        readouts cannot invalidate their relations by dropping TEMP tables.
        """
        effective_horizon = list(self._horizon_metrics)
        for candidate in (*tuple(horizon_metrics or ()), metric):
            if candidate not in effective_horizon:
                effective_horizon.append(candidate)
        horizon_key = tuple(effective_horizon)
        if horizon_key == self._horizon_metrics:
            key: tuple[str, Metric] | tuple[str, Metric, tuple[Metric, ...]] = (
                population,
                metric,
            )
            horizon_argument = None
        else:
            key = (population, metric, horizon_key)
            horizon_argument = horizon_key
        if not use_cache:
            return self._build_panel_for_metric_impl(
                exposures, metric, horizon_metrics=horizon_argument
            )
        cached = self._panel_cache.get(key)
        if cached is None:
            cached = self._build_panel_for_metric_impl(
                exposures,
                metric,
                horizon_metrics=horizon_argument,
            )
            self._panel_cache[key] = cached
        return cached

    def _build_panel_for_metric_impl(
        self,
        exposures: Table,
        metric: Metric,
        *,
        horizon_metrics: Sequence[Metric] | None = None,
    ) -> tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar]:
        """Build the dense unit-day panel and the spine + sufficient-stats
        pair for one metric.

        Shared by the un-broken-out pipelines (``_panel_sql_for_metrics``,
        ``_summary_sql_for_metrics``, ``_moments_for_metrics``) and the
        breakout pipeline (``breakout_summaries``), which all
        need the same panel/spine/stats/covariate-source tables per
        metric, built independently from the same underlying events.

        Returns ``(panel, spine, stats, events, den_events, fact_tbl,
        value_col, data_as_of)``: ``events`` is the raw events table
        ``spine``/``stats`` were derived from. ``spine``'s right edge is
        ``_union_event_horizon`` over the declared metrics plus any
        caller-selected effective metric, so a metric's numeric answer never
        depends on whether this is a fused or a later materialized readout.
        ``den_events`` is the raw denominator events for a ``RatioMetric``
        (else ``None``); ``fact_tbl``/``value_col`` carry the CUPED
        pre-period covariate through. ``data_as_of`` is a lazy scalar
        (never ``None`` - an absent signal is the coalesced sentinel
        instead), the earlier of numerator/denominator freshness for a ratio
        metric.
        """
        den_events: Table | None = None
        if isinstance(metric, RatioMetric):
            num_fs, num_fact_def = _find_fact_source(self._defs, metric.numerator.fact)
            num_fact_tbl = self._get_fact_table(num_fs)
            num_value_col = _resolve_value_column(num_fs, num_fact_def)
            events = metric_events(
                num_fact_tbl, metric, value_column=num_value_col, part="numerator"
            )

            den_fs, den_fact_def = _find_fact_source(self._defs, metric.denominator.fact)
            den_fact_tbl = self._get_fact_table(den_fs)
            den_value_col = _resolve_value_column(den_fs, den_fact_def)
            den_events = metric_events(
                den_fact_tbl, metric, value_column=den_value_col, part="denominator"
            )
            fact_tbl = num_fact_tbl  # the numerator is the pre-period covariate's source
            value_col = num_value_col
            data_as_of = cast(
                ir.Scalar,
                ibis.least(
                    self._data_as_of(num_fact_tbl, metric.numerator.fact),
                    self._data_as_of(den_fact_tbl, metric.denominator.fact),
                ),
            )
        else:
            fs, fact_def = _find_fact_source(self._defs, metric.fact)
            fact_tbl = self._get_fact_table(fs)
            value_col = _resolve_value_column(fs, fact_def)
            events = metric_events(fact_tbl, metric, value_column=value_col)
            data_as_of = self._data_as_of(fact_tbl, metric.fact)

        horizon = self._union_event_horizon(horizon_metrics)
        panel = unit_day_panel(
            exposures,
            events,
            self._experiment,
            metric_name=metric.name,
            end_date=horizon,
        )
        spine, stats = unit_day_spine_stats(
            exposures, events, self._experiment, metric.name, end_date=horizon
        )
        return panel, spine, stats, events, den_events, fact_tbl, value_col, data_as_of

    _NO_DATA_SIGNAL = NO_DATA_SIGNAL

    def _data_as_of(self, fact_tbl: Table, fact: str) -> ir.Scalar:
        """Lazy scalar: the latest date *fact_tbl* has loaded *fact* through.

        A max(ts) lower bound on true pipeline freshness, not an exact
        watermark. Scoped to rows where ``event == fact``, since a fact
        source commonly backs several facts with different real-world
        latency, and an unfiltered max would let a fresher sibling fact
        mask a lagging one. The day is bucketed via ``_local_date``
        under the experiment's ``day_boundary`` so freshness censors in
        the same calendar the panel's day buckets live in.

        Stays lazy - composed into the same ibis expression
        ``unit_totals`` builds rather than resolved here, so
        ``panel_sql``/``summary_sql`` never trigger a warehouse
        round-trip just to compute it. Coalesced against a far-future
        sentinel so a fact with zero rows doesn't propagate SQL NULL
        through ``ibis.least`` and wrongly cap censoring to nothing.
        """
        scoped = fact_tbl.filter(fact_tbl.event == fact)
        return cast(
            "ir.Scalar",
            ibis.coalesce(
                _local_date(scoped.ts, self._experiment).max(),
                ibis.literal(self._NO_DATA_SIGNAL),
            ),
        )

    def _resolve_breakout_props(
        self, breakouts: Sequence[Breakout | Factor], exposures: Table
    ) -> list[tuple[Breakout | Factor, FactSource, Table]]:
        """Resolve and build each breakout's/factor's deduplicated properties table.

        Shared by ``_panel_sql_for_metrics``/``_summary_sql_for_metrics``,
        ``breakout_summaries``, and
        ``factor_summaries``, which all need the same (breakout, fact
        source, properties table) triples, resolved once regardless of
        which metric is being processed. ``Factor`` shares ``Breakout``'s
        ``property``/``source`` shape, so this resolver serves both.

        The resolved ``FactSource`` travels with the properties table so
        callers can qualify per-breakout keys by source name: two
        breakouts can share a ``property`` name while resolving to
        different sources, so the source name keeps their keys from
        colliding. *exposures* is required (not defaulted to ``None``)
        since every caller already holds it.
        """
        result: list[tuple[Breakout | Factor, FactSource, Table]] = []
        for b in breakouts:
            fs = _resolve_breakout_fact_source(self._defs, self._experiment, b)
            props = self._build_breakout_properties_table(fs, b.property, exposures)
            result.append((b, fs, props))
        return result

    def _build_pre_events(
        self, fact_tbl: Table, metric: Metric, value_col: str | None
    ) -> Table | None:
        """Build the CUPED pre-period covariate events table, or None.

        Shared by ``_summary_sql_for_metrics``/``_moments_for_metrics`` and
        ``breakout_summaries``, which all build the pre-period events from the same
        fact/value_column/aggregation as the outcome, gated on
        ``Experiment.n_pre_periods``.

        A ratio metric's covariate is its NUMERATOR's pre-period total:
        ``fact_tbl``/``value_col`` are the numerator's here and
        ``metric_events`` defaults to the numerator part, so the same one
        covariate feeds both component slopes (docs/guides/cuped.md).
        """
        if self._experiment.n_pre_periods > 0:
            return metric_events(fact_tbl, metric, value_column=value_col)
        return None

    def _validate_cluster_labels(self, exposures: Table, cluster: str) -> None:
        """Loud data-quality gates on the declared cluster column.

        Checks the raw (undeduplicated) admitted exposure rows for a
        conflicting label first: `first_exposures` picks one label per
        unit (argmin on ts), which would otherwise hide two distinct
        non-null labels for the same unit behind a clean-looking result.
        """
        raw = _scope_exposure_events(self._build_exposure_events_table(), self._experiment)
        raw = raw.semi_join(
            exposures.select("unit_id", "experiment_id").distinct(),
            ["unit_id", "experiment_id"],
        )
        validate_cluster_uniqueness(raw, cluster, self._experiment.name, con=self._con)
        validate_cluster_labels(
            exposures,
            cluster,
            self._experiment.name,
            enforce_purity=getattr(self._context.design, "mechanism", None) != "observational",
            con=self._con,
        )

    def _resolve_uptake(
        self, cluster: str | None, *, operation: str
    ) -> tuple[Table | None, int | None]:
        """An Encouragement design's uptake fact table and window, or
        ``(None, None)``.

        Resolved once per readout and shared across every metric's
        ``unit_totals`` call; depends only on ``self._design``, never on
        which metric is being processed. A declared ``cluster`` first runs
        :meth:`_refuse_clustered_encouragement`.
        """
        from increment.semantics.design import Encouragement as _Encouragement

        _design = self._context.design
        if isinstance(_design, _Encouragement) and cluster is not None:
            self._refuse_clustered_encouragement(_design, cluster, operation=operation)
        if isinstance(_design, _Encouragement):
            uptake_fs, _uptake_fact = _find_fact_source(self._defs, _design.uptake.fact)
            uptake_fact_tbl = self._get_fact_table(uptake_fs)
            uptake_events = uptake_fact_tbl.filter(uptake_fact_tbl.event == _design.uptake.fact)
            return uptake_events, _design.uptake.window_days
        return None, None

    def _refuse_clustered_encouragement(
        self, design: Encouragement, cluster: str, *, operation: str
    ) -> None:
        """Refuse what a clustered LATE reduction cannot carry.

        A declared ``cluster`` leaves no per-unit CUPED covariate moments to
        adjust with, and the clustered denominator family is already spoken
        for by the LATE first stage's cluster sizes, so a ratio/quantile
        metric's own denominator has nowhere to live under this design.
        Both checks read only declarations, never warehouse state.
        """
        n_pre_periods = self._experiment.n_pre_periods
        if n_pre_periods and n_pre_periods > 0:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "design": design.mechanism,
                    "cluster": cluster,
                    "n_pre_periods": n_pre_periods,
                },
                offered={"n_pre_periods": 0},
                route=(
                    "drop the cluster declaration or n_pre_periods; the clustered LATE "
                    "reduction has no per-unit covariate moments to adjust with"
                ),
            )
        unsupported = tuple(m for m in self._metrics if m.type in ("ratio", "quantile"))
        if unsupported:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "design": design.mechanism,
                    "cluster": cluster,
                    "metrics": tuple(m.name for m in unsupported),
                    "metric_types": tuple(m.type for m in unsupported),
                },
                offered=tuple(m.name for m in self._metrics if m.type not in ("ratio", "quantile")),
                route=(
                    "remove the ratio and quantile metrics from this clustered "
                    "encouragement experiment; the LATE first stage reads cluster sizes "
                    "from the denominator family, and LATE's Wald numerator is undefined "
                    "for a ratio metric"
                ),
            )

    def compliance_dates(self) -> Sequence[object]:
        """Enrollment spine before outcome missingness or completion filters."""
        self._validate_mixed_assignments()
        design = cast(Encouragement, self._context.design)
        uptake_fs, _ = _find_fact_source(self._defs, design.uptake.fact)
        table = self._get_fact_table(uptake_fs)
        uptake = table.filter(table.event == design.uptake.fact)
        exposures = self._get_exposures()
        spine = panel_spine(
            exposures,
            self._experiment,
            end_date=compliance_event_horizon(exposures, uptake, self._experiment),
        )
        days = spine.filter(spine.ds.notnull()).select("ds").distinct().order_by("ds")
        return [row["ds"] for row in self._con.to_pyarrow(days).to_pylist()]

    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
    ) -> tuple[DashboardGroupData, ...]:
        """Read group aggregates, or decode the displayed retained checkpoints."""
        if checkpoints is not None:
            return _retained_dashboard_groups(metrics, checkpoints, self.sequential_snapshot())
        rows: list[DashboardGroupData] = []

        self._validate_mixed_assignments()
        with self._reduction_batch():
            exposures = self._get_exposures()
            for metric in metrics:
                _panel, spine, stats, _events, den_events, _facts, _value, data_as_of = (
                    self._build_panel_for_metric(exposures, metric)
                )
                den_stats = (
                    post_exposure_stats(
                        den_events,
                        exposures,
                        source_key=f"{metric.name}:den",
                        experiment=self._experiment,
                    )
                    if den_events is not None
                    else None
                )
                raw_totals = unit_totals(
                    spine,
                    stats,
                    metric,
                    self._experiment,
                    den_stats=den_stats,
                    data_as_of=data_as_of,
                    warn_on_censoring=False,
                )
                dense = _dense_unit_days(spine, stats)
                has_observed_day, window_complete, observation_end = _observable_window_flags(
                    dense, metric, self._experiment, data_as_of
                )
                if self._experiment.observation_horizon is None:
                    observation_end = (
                        dense.ds.max()
                        .as_scalar()
                        .notnull()
                        .ifelse(observation_end, ibis.null().cast("date"))
                    )
                flagged = dense.mutate(
                    _has_observed_day=has_observed_day,
                    _window_complete=window_complete,
                    _observation_end=observation_end,
                )
                accounting = flagged.group_by("group_id").aggregate(
                    assigned=flagged.unit_id.nunique(),
                    no_observed=flagged.unit_id.nunique(where=~flagged._has_observed_day),
                    not_mature=flagged.unit_id.nunique(
                        where=flagged._has_observed_day & ~flagged._window_complete
                    ),
                    observation_end=flagged._observation_end.max(),
                )
                eligible_days = flagged.semi_join(raw_totals.select("unit_id"), "unit_id")
                event_dense = window_bound_stats(eligible_days, metric)
                if isinstance(metric, RetentionMetric):
                    event_dense = event_dense.filter(
                        event_dense.ds
                        >= event_dense.first_exposure_date + ibis.interval(days=metric.band[0])
                    )
                event_counts = event_dense.group_by("group_id").aggregate(
                    event_count=event_dense.n_events.sum()
                )
                # Normalize descriptive moments so finite means survive overflowing totals.
                # Raw unit values still supply the quantile and eligibility.
                group_window = ibis.window(group_by=["group_id"])
                value_columns = ("y", "y_den") if isinstance(metric, RatioMetric) else ("y",)
                scales = {
                    f"scale_{name}": raw_totals[name].abs().max().over(group_window)
                    for name in value_columns
                }
                scaled = raw_totals.mutate(**scales)
                normalized = scaled.mutate(
                    **{
                        name: scaled[name]
                        / (scaled[f"scale_{name}"] > 0).ifelse(scaled[f"scale_{name}"], 1.0)
                        for name in value_columns
                    }
                )
                scale_rows = scaled.group_by("group_id").aggregate(
                    **{name: scaled[name].max() for name in scales}
                )
                aggregates = group_summary(normalized).left_join(
                    scale_rows, "group_id", rname="{name}_scale"
                )
                if isinstance(metric, QuantileMetric):
                    quantiles = _dashboard_quantiles(raw_totals, metric.quantile)
                    aggregates = aggregates.left_join(quantiles, "group_id")
                result = accounting.left_join(
                    aggregates, "group_id", rname="{name}_summary"
                ).left_join(event_counts, "group_id", rname="{name}_events")
                window = (
                    metric.band
                    if isinstance(metric, RetentionMetric)
                    else (0, resolve_window_days(metric))
                )
                for record in self._con.to_pyarrow(result).to_pylist():
                    rows.append(_observed_dashboard_group(metric, record, window))
        return tuple(rows)

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> ComplianceSummary:
        """Design-level uptake sufficient state off the warehouse's own
        declared uptake fact -- independent of any outcome metric's
        maturity gate, missingness, or declaration order.

        Resolves the uptake fact directly (not through :meth:`_resolve_uptake`,
        whose CUPED/ratio guards are scoped to this source's whole declared
        metric set and would spuriously fire on a metric unrelated to
        compliance). Reuses
        :func:`~increment.query.builders._attach_uptake_flag`'s
        metric-independent uptake-window attachment and
        :func:`~increment.query.builders.group_summary`'s cluster bivariate
        collapse -- the same primitives a clustered metric's own
        compliance-relevant ``x`` family already goes through.
        """
        if completed_windows_only:
            from increment._source_types import validate_compliance_completion

            validate_compliance_completion(design, as_of=as_of)
        if design != self._context.design:
            expected = self._context.design
            _refuse(
                _NATIVE_COMPLIANCE_DESIGN_MISMATCH,
                expected=None if expected is None else expected.model_dump(mode="json"),
                received=design.model_dump(mode="json"),
            )
        if population == "triggered":
            self._validate_trigger_capability(operation="compliance_summary")
        self._validate_mixed_assignments()
        cluster = self._context.cluster
        uptake_fs, _uptake_fact = _find_fact_source(self._defs, design.uptake.fact)
        uptake_fact_tbl = self._get_fact_table(uptake_fs)
        uptake_events = uptake_fact_tbl.filter(uptake_fact_tbl.event == design.uptake.fact)
        window_days = design.uptake.window_days
        exposures = (
            self._get_trigger_population(operation="compliance_summary")
            if population == "triggered"
            else self._get_exposures()
        )
        if cluster is not None:
            self._validate_cluster_labels(exposures, cluster)
        if as_of is not None:
            exposures = exposures.filter(
                _local_date(exposures.first_exposure_ts, self._experiment) <= ibis.literal(as_of)
            )
            uptake_events = uptake_events.filter(
                _local_date(uptake_events.ts, self._experiment) <= ibis.literal(as_of)
            )
        if completed_windows_only:
            exposures = exposures.filter(
                _local_date(exposures.first_exposure_ts, self._experiment)
                + ibis.interval(days=window_days)
                <= ibis.literal(as_of)
            )
        totals = _attach_uptake_flag(exposures, exposures, uptake_events, window_days)
        totals = totals.mutate(y=totals.d, metric=ibis.literal("uptake"))
        base_rows = self._con.to_pyarrow(
            totals.group_by("group_id").agg(
                n_units=totals.count(),
                uptake_total=totals.d.sum(),
            )
        ).to_pylist()
        rows: dict[str, dict[str, Any]] = {
            str(r["group_id"]): {
                "group_id": str(r["group_id"]),
                "n_units": int(r["n_units"]),
                "uptake_total": float(r["uptake_total"]),
            }
            for r in base_rows
        }
        if cluster is not None:
            cluster_rows = self._con.to_pyarrow(
                group_summary(totals, cluster=cluster, uptake=True)
            ).to_pylist()
            for r in cluster_rows:
                group_id = str(r["group_id"])
                rows[group_id].update(
                    {
                        name: int(r[field]) if name == "n_clusters" else float(r[field])
                        for name, field in COMPLIANCE_ARM_FROM_CLUSTER_ROW.items()
                    }
                )
        return ComplianceSummary(
            study_id=self._experiment_name,
            control_group=str(design.control_group),
            cohort=design.uptake.fact,
            window_days=design.uptake.window_days,
            one_sided=design.one_sided,
            cluster=cluster,
            as_of=as_of,
            arms=tuple(ComplianceArm(**row) for row in rows.values()),
        )

    # Keep metric summary assembly cohesive so all derived tables share one reduction.
    def _build_metric_summary(  # noqa: PLR0913
        self,
        metric: Metric,
        exposures: Table,
        spine: Table,
        stats: Table,
        den_events: Table | None,
        fact_tbl: Table,
        value_col: str | None,
        data_as_of: ir.Scalar,
        *,
        cluster: str | None,
        uptake_events: Table | None,
        uptake_window_days: int | None,
        warn_on_censoring: bool,
        want_cuped: bool = True,
        by: Sequence[str] = (),
        properties_table: Table | None = None,
    ) -> tuple[Table, Table | None, Table | None, Table]:
        """Build summary, pre-period stats, denominator stats, and unit totals.
        Retention cohorts reuse pre-period stats; quantiles use the unit totals.
        Disable want_cuped when no requested method needs pre-period covariates.
        """
        # CUPED covariate: built from the same fact/value_column/aggregation as
        # the outcome (a ratio's numerator); docs/guides/cuped.md.
        pre_events = self._build_pre_events(fact_tbl, metric, value_col) if want_cuped else None
        pre_stats = (
            pre_period_stats(
                pre_events, exposures, self._experiment, source_key=f"{metric.name}:pre"
            )
            if pre_events is not None
            else None
        )
        den_stats = (
            post_exposure_stats(
                den_events,
                exposures,
                source_key=f"{metric.name}:den",
                experiment=self._experiment,
            )
            if den_events is not None
            else None
        )
        totals = unit_totals(
            spine,
            stats,
            metric,
            self._experiment,
            pre_stats=pre_stats,
            den_stats=den_stats,
            uptake_events=uptake_events,
            uptake_window_days=uptake_window_days,
            by=list(by) or None,
            properties_table=properties_table,
            data_as_of=data_as_of,
            warn_on_censoring=(
                (lambda query: self._con.to_pyarrow(query).to_pylist())
                if warn_on_censoring
                else False
            ),
        )
        totals = winsorize_unit_totals(totals, metric)
        # The metric type fixes the denominator family: a clustered ratio row carries
        # its own per-cluster denominator total instead of the cluster size.
        summary = group_summary(
            totals,
            by=list(by) or None,
            cluster=cluster,
            ratio_metrics=[metric.name] if metric.type == "ratio" else None,
            uptake=uptake_events is not None,
        )
        return summary, pre_stats, den_stats, totals

    def _preflight_breakout(self, breakout: Breakout, *, operation: str) -> FactSource:
        """Resolve and validate one breakout without touching warehouse state.

        A request is declared only when both its property and resolved
        ``FactSource`` identity match an experiment breakout. Inferred and
        explicit source forms therefore share the same declaration check.
        """
        property_name = getattr(breakout, "property", None)
        requested_source = getattr(breakout, "source", None)
        declared = tuple(b for b in self._experiment.breakouts if b.property == property_name)
        if not declared:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "dimension": property_name,
                    "source": requested_source,
                },
                offered=tuple(sorted({b.property for b in self._experiment.breakouts})),
                route="declare the breakout on the experiment or request a declared property",
            )

        declared_sources = {
            _resolve_breakout_fact_source(self._defs, self._experiment, item).name
            for item in declared
        }
        try:
            resolved = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
        except ValueError:
            if requested_source is not None:
                _refuse_operation(
                    operation=operation,
                    request={
                        "experiment": self._experiment.name,
                        "dimension": property_name,
                        "source": requested_source,
                    },
                    offered=tuple(sorted(declared_sources)),
                    route="request the breakout with one of its declared sources",
                )
            raise
        if resolved.name not in declared_sources:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "dimension": property_name,
                    "source": requested_source,
                    "resolved_source": resolved.name,
                },
                offered=tuple(sorted(declared_sources)),
                route="request the breakout with one of its declared sources",
            )
        return resolved

    def _preflight_day_request(
        self,
        metrics: Sequence[Metric],
        *,
        dimension: str | None,
        include_covariate: bool,
        operation: str,
        require_uptake: bool = False,
    ) -> None:
        """Reject unsupported day evidence before any panel is built."""
        if dimension is not None:
            declared = sorted({b.property for b in self._experiment.breakouts})
            if dimension not in declared:
                _refuse_operation(
                    operation=operation,
                    request={"experiment": self._experiment.name, "dimension": dimension},
                    offered=tuple(declared),
                    route="declare the breakout on the experiment or request a declared property",
                )
        if include_covariate:
            _refuse_missing_pre_period(self._experiment, metrics, operation=operation)
        cluster = self._experiment.cluster
        if cluster is not None:
            # Clustered day-axis evidence is unsupported.  Keep the
            # encouragement-specific compatibility refusals local and pure so
            # an unsupported request never resolves the uptake fact first.
            from increment.semantics.design import Encouragement as _Encouragement

            design = self._context.design
            if isinstance(design, _Encouragement):
                self._refuse_clustered_encouragement(design, cluster, operation=operation)
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "cluster": cluster,
                    "metrics": tuple(metric.name for metric in metrics),
                    "dimension": dimension,
                },
                offered=("total",),
                route="request total-grain readouts; cluster-robust inference is total-grain only",
            )
        if require_uptake:
            self._resolve_uptake(None, operation=operation)

    def _refuse_observational_quantile(self, metric: Metric) -> None:
        """An observational design has no quantile estimator on any ingress path."""
        if metric.type == "quantile" and getattr(self._context.design, "mechanism", None) == (
            "observational"
        ):
            refuse_observational_quantile(metric)

    def _refuse_quantile_metrics(self, effective: Sequence[Metric], *, operation: str) -> None:
        """A quantile has no moments representation: there is no summary
        SQL to introspect and nothing an exported moments cube could carry.
        """
        non_additive = [m.name for m in effective if m.type == "quantile"]
        if operation == "moments":
            for metric in effective:
                self._refuse_observational_quantile(metric)
        if non_additive:
            _refuse_operation(
                operation=operation,
                request={"metrics": tuple(non_additive), "metric_type": "quantile"},
                offered=tuple(m.name for m in effective if m.type != "quantile"),
                route=(
                    "estimate quantiles with run(), which reads retained per-unit rows; "
                    "a quantile has no moments representation"
                ),
            )

    def _refuse_percentile_winsorized(self, metrics: Sequence[Metric], operation: str) -> None:
        """Percentile winsorization needs an executed quantile, which the
        totals-grain breakout/factor summary paths never compute."""
        for metric in metrics:
            config = _percentile_winsorization(metric)
            if config is not None:
                _refuse_operation(
                    operation=operation,
                    request={"metric": metric.name, "winsorization": _winsorization_bounds(config)},
                    offered=tuple(m.name for m in metrics if _percentile_winsorization(m) is None),
                    route=(
                        "use fixed-value winsorization (lower_value/upper_value); breakout "
                        "and factor summaries never compute an executed quantile"
                    ),
                )

    def _refuse_declared_trigger(self, operation: str) -> None:
        """Reject views that cannot preserve the declared population."""
        trigger = self._experiment.trigger
        if trigger is not None:
            _refuse_operation(
                operation=operation,
                request={"experiment": self._experiment.name, "trigger": trigger},
                offered=("assigned",),
                route=(
                    "use run() for the triggered readout, or analyze an experiment with no "
                    "declared trigger"
                ),
            )

    def _panel_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]:
        """Per-metric panel SQL, keyed by metric name, plus one dimensioned
        entry per requested breakout, keyed
        ``f"{metric.name}:{breakout.property}:{source_name}"`` (qualified by
        source name so two breakouts sharing a property but resolving to
        different sources don't collide).

        *selected*, when given, narrows the metric set this call builds
        for; ``None`` (default) uses ``self._metrics`` (every declared
        metric).
        """
        effective = self._metrics if selected is None else list(selected)
        self._refuse_quantile_metrics(effective, operation="panel_sql")
        if not effective:
            return {}
        exposures = (
            self._get_trigger_population(operation="panel_sql")
            if population == "triggered"
            else self._get_exposures()
        )
        breakout_props = self._resolve_breakout_props(breakouts or [], exposures)
        # Uptake fact (Encouragement only): unused here, but resolving it also
        # runs the cluster+CUPED/ratio-quantile refusal gate every stage shares.
        self._resolve_uptake(self._experiment.cluster, operation="panel_sql")
        sql_results: dict[str, str] = {}
        for metric in effective:
            panel, *_rest = self._build_panel_for_metric(exposures, metric, population)
            sql_results[metric.name] = ibis_to_sql(panel, dialect=self._backend or self._con.name)
            for breakout, fs, properties_table in breakout_props:
                dim_panel = join_breakout_dimension(panel, properties_table, [breakout.property])
                key = f"{metric.name}:{breakout.property}:{fs.name}"
                sql_results[key] = ibis_to_sql(dim_panel, dialect=self._backend or self._con.name)
        return sql_results

    def _summary_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]:
        """Per-metric ``group_summary`` SQL, keyed by metric name, plus one
        dimensioned entry per requested breakout (same keying as
        `_panel_sql_for_metrics`).

        *selected* narrows the metric set the same way as
        `_panel_sql_for_metrics`.
        """
        effective = self._metrics if selected is None else list(selected)
        self._refuse_quantile_metrics(effective, operation="summary_sql")
        if not effective:
            return {}
        exposures = (
            self._get_trigger_population(operation="summary_sql")
            if population == "triggered"
            else self._get_exposures()
        )
        cluster = self._experiment.cluster
        breakout_props = self._resolve_breakout_props(breakouts or [], exposures)
        uptake_events, uptake_window_days = self._resolve_uptake(cluster, operation="summary_sql")
        sql_results: dict[str, str] = {}
        for metric in effective:
            panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population, use_cache=False)
            )
            summary, _pre_stats, _den_stats, _totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=False,
            )
            sql_results[metric.name] = ibis_to_sql(summary, dialect=self._backend or self._con.name)
            for breakout, fs, properties_table in breakout_props:
                summary_by, _pre_stats_by, _den_stats_by, _totals_by = self._build_metric_summary(
                    metric,
                    exposures,
                    spine,
                    stats,
                    den_events,
                    fact_tbl,
                    value_col,
                    data_as_of,
                    cluster=None,
                    uptake_events=uptake_events,
                    uptake_window_days=uptake_window_days,
                    warn_on_censoring=False,
                    by=[breakout.property],
                    properties_table=properties_table,
                )
                key = f"{metric.name}:{breakout.property}:{fs.name}"
                sql_results[key] = ibis_to_sql(summary_by, dialect=self._backend or self._con.name)
        return sql_results

    def _moments_for_metrics(
        self,
        methods: list[Any] | None = None,
        selected: Sequence[Metric] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        # Internal flag controlling whether CUPED narrowing is applied.
        narrow_cuped: bool = False,  # noqa: FBT001, FBT002
    ) -> pa.Table:
        """Every selected metric's ``group_summary`` moments, unioned into
        one arrow table -- the shared substrate ``export()``/``moments()``/
        ``DefinitionsMomentSource.moments_source()`` all serve from.

        *methods*/*prior* are opaque here (typed loosely to keep this
        module's own imports SQL/query-only, never reaching into the
        estimator layer): forwarded verbatim to
        `_analysis_config.resolve_configs`, which types and interprets
        them, and never inspected in this module.

        *narrow_cuped* (default ``False``) narrows "want cuped" to only
        the metrics whose resolved *methods* request it, instead of the
        unconditional default (an exported moments cube must carry the
        covariate whenever ``n_pre_periods>0`` allows it). Set only by
        ``DefinitionsMomentSource.moments_source``, whose cube is consumed immediately by
        the same call that resolved *methods*.
        """
        # Import lazily because pyarrow is provided by live ibis connections.
        import pyarrow as pa

        effective = self._metrics if selected is None else list(selected)
        self._refuse_quantile_metrics(effective, operation="moments")
        if not effective:
            return pa.table({})
        configs = overlay_configs(
            effective,
            self._context.configs,
            methods=methods,
            prior=prior,
        )
        self._note_reduction()
        # Build exposure events once (shared across all metrics)
        exposures = (
            self._get_trigger_population(operation="moments")
            if population == "triggered"
            else self._get_exposures()
        )

        cluster = self._experiment.cluster
        if cluster is not None:
            self._validate_cluster_labels(exposures, cluster)

        # Uptake fact (Encouragement only): resolved once and shared like
        # `exposures`; the cluster+CUPED/ratio-quantile refusals live in _resolve_uptake.
        uptake_events, uptake_window_days = self._resolve_uptake(cluster, operation="moments")

        tables: list[pa.Table] = []
        for metric, config in zip(effective, configs, strict=True):
            panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population)
            )
            resolved_cuped = any(
                method.variance_reduction == "cuped"
                for method in effective_methods(config, design=self.context.design)
            )
            want_cuped = resolved_cuped if narrow_cuped else True
            summary, _pre_stats, _den_stats, _totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
                want_cuped=want_cuped,
            )
            tables.append(summary.to_pyarrow())

        return pa.concat_tables(tables)

    def _windowed_metric_sum(
        self,
        fact_tbl: Table,
        metric: Metric,
        value_col: str | None,
        part: Literal["numerator", "denominator"],
    ) -> Table:
        """One ratio part's whole-site windowed sum over *fact_tbl*.

        Delegates to ``builders._windowed_fact_sum`` -- shared with
        ``site_volume`` -- because ``_site_volume_row`` needs two
        independently-resolved fact tables rather than one shared table.
        A missing-rows sum is a typed 0.0, never SQL NULL, matching
        ``site_volume``'s convention, so an empty window cannot reach
        ``sitewide_evidence``'s ``float(row["y"])`` as NULL.
        """
        return _windowed_fact_sum(
            fact_tbl, metric, self._experiment, value_column=value_col, part=part
        )

    def _site_volume_row(self, metric: Metric) -> dict[str, Any]:
        """Whole-site ``{"metric", "y", "y_den"}`` for *metric*.

        ``site_volume`` takes a single fact table and reuses it for both
        a ratio metric's numerator and denominator, correct only when
        the two share one physical fact source. A ratio metric's
        numerator and denominator may resolve to different fact
        sources, so this always resolves them independently via two
        ``_windowed_metric_sum`` calls; a non-ratio metric has only one
        fact table and calls ``site_volume`` directly.
        """
        if isinstance(metric, RatioMetric):
            num_fs, num_fact_def = _find_fact_source(self._defs, metric.numerator.fact)
            num_fact_tbl = self._get_fact_table(num_fs)
            num_value_col = _resolve_value_column(num_fs, num_fact_def)
            den_fs, den_fact_def = _find_fact_source(self._defs, metric.denominator.fact)
            den_fact_tbl = self._get_fact_table(den_fs)
            den_value_col = _resolve_value_column(den_fs, den_fact_def)
            num_row = self._con.to_pyarrow(
                self._windowed_metric_sum(num_fact_tbl, metric, num_value_col, "numerator")
            ).to_pylist()[0]
            den_row = self._con.to_pyarrow(
                self._windowed_metric_sum(den_fact_tbl, metric, den_value_col, "denominator")
            ).to_pylist()[0]
            return {"metric": metric.name, "y": num_row["y"], "y_den": den_row["y"]}
        fs, fact_def = _find_fact_source(self._defs, metric.fact)
        fact_tbl = self._get_fact_table(fs)
        value_col = _resolve_value_column(fs, fact_def)
        volume = site_volume(fact_tbl, metric, self._experiment, value_column=value_col)
        return self._con.to_pyarrow(volume).to_pylist()[0]

    def sitewide_evidence(self, metric: Metric, *, include_ratio: bool = False) -> SitewideEvidence:
        """Prepare typed, exposure-independent evidence for ``sitewide``."""
        self._refuse_declared_trigger("sitewide_evidence")
        if isinstance(metric, RatioMetric) and not include_ratio:
            _refuse_operation(
                operation="sitewide_evidence",
                request={
                    "metric": metric.name,
                    "metric_type": metric.type,
                    "include_ratio": include_ratio,
                },
                offered={"include_ratio": True},
                route=(
                    "request the ratio reducer with include_ratio=True; the named "
                    "evidence carries one site total"
                ),
            )
        winsorization = getattr(metric, "winsorization", None)
        if winsorization is not None:
            _refuse_operation(
                operation="sitewide_evidence",
                request={
                    "metric": metric.name,
                    "winsorization": _winsorization_bounds(winsorization),
                },
                offered={"winsorization": None},
                route=(
                    "use the unwinsorized metric; whole-site outcomes cannot reuse the "
                    "per-unit transformed experiment metric"
                ),
            )
        validate_site_volume_metric(metric)
        self._note_reduction()
        exposures = self._get_exposures()
        cluster = self._experiment.cluster
        cluster_counts: dict[str, int] | None = None
        unit_counts: dict[str, int] | None = None
        if cluster is not None:
            self._validate_cluster_labels(exposures, cluster)
            count_rows = self._con.to_pyarrow(
                cluster_exposure_counts(exposures, self._experiment)
            ).to_pylist()
            cluster_counts = {str(r["group_id"]): int(r["n_clusters"]) for r in count_rows}
            unit_counts = {str(r["group_id"]): int(r["n_units"]) for r in count_rows}
        uptake_events, uptake_window_days = self._resolve_uptake(
            cluster, operation="sitewide_evidence"
        )
        _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
            self._build_panel_for_metric(exposures, metric)
        )
        summary, _pre_stats, _den_stats, _totals = self._build_metric_summary(
            metric,
            exposures,
            spine,
            stats,
            den_events,
            fact_tbl,
            value_col,
            data_as_of,
            cluster=cluster,
            uptake_events=uptake_events,
            uptake_window_days=uptake_window_days,
            warn_on_censoring=True,
        )
        from increment.estimation.engine import _df_to_arms

        row = self._site_volume_row(metric)
        return SitewideEvidence(
            metric=metric,
            site_total=float(row["y"]),
            arm_stats=tuple(_df_to_arms(summary.to_pyarrow())),
            control_group=self._experiment.control_group,
            cluster=cluster,
            site_total_denominator=(
                float(row["y_den"]) if isinstance(metric, RatioMetric) else None
            ),
            cluster_counts=cluster_counts,
            unit_counts=unit_counts,
        )

    def _day_axis_source(
        self,
        *,
        selected: Sequence[Metric] | None = None,
        _legacy_facade: bool = True,
    ) -> _DaySource:
        """This experiment's day-axis panels, served as a `MomentSource`.

        The private adapter defaults to the legacy facade contract: unchanged
        ``Analysis`` day/as-of methods pre-note their top-level reduction.
        Direct native callers use ``day_source()`` or ``day_axis_source()``,
        which explicitly opt into source-owned lazy reduction accounting.
        """
        self._refuse_declared_trigger("day_axis_source")
        if selected is None:
            context = self._context
        else:
            selected_metrics = tuple(selected)
            configs_by_name = {config.metric.name: config for config in self._context.configs}
            missing = [metric for metric in selected_metrics if metric.name not in configs_by_name]
            if missing:
                configs_by_name.update(
                    {
                        config.metric.name: config
                        for config in resolve_configs(
                            missing,
                            bindings=None,
                            specs=None,
                            methods=None,
                            prior=None,
                        )
                    }
                )
            procedures = {
                name: procedure
                for name, procedure in self._context.plan.procedures.items()
                if name in {metric.name for metric in selected_metrics}
            }
            if missing:
                from increment.plan import with_unassigned_procedures

                defaults = with_unassigned_procedures(self._context.plan, missing, design=None)
                procedures.update(
                    {metric.name: defaults.procedures[metric.name] for metric in missing}
                )
            context_plan = type(self._context.plan)(
                declared=self._context.plan.declared,
                alpha=self._context.plan.alpha,
                q=self._context.plan.q,
                path=self._context.plan.path,
                inference=self._context.plan.inference,
                procedures=procedures,
                view_policies=self._context.plan.view_policies,
            )
            context = replace(
                self._context,
                metrics=selected_metrics,
                configs=tuple(configs_by_name[metric.name] for metric in selected_metrics),
                plan=context_plan,
            )
        selected_metrics = tuple(self._metrics if selected is None else selected)
        horizon_metrics = list(self._horizon_metrics)
        for metric in selected_metrics:
            if metric not in horizon_metrics:
                horizon_metrics.append(metric)
        effective_horizon = tuple(horizon_metrics)

        def build_panel(exposures: Table, metric: Metric, horizon_metrics: tuple[Metric, ...]):
            return self._build_panel_for_metric(exposures, metric, horizon_metrics=horizon_metrics)

        def build_asof_panel(
            exposures: Table,
            metric: Metric,
            horizon_metrics: tuple[Metric, ...],
            # Internal flag controlling whether the panel includes a covariate.
            include_covariate: bool = False,  # noqa: FBT001, FBT002
            include_uptake_horizon: bool = False,  # noqa: FBT001, FBT002
        ):
            return self._build_asof_panels(
                exposures,
                metric,
                horizon_metrics=horizon_metrics,
                include_covariate=include_covariate,
                include_uptake_horizon=include_uptake_horizon,
            )

        return _DaySource(
            self._experiment,
            context,
            self._get_exposures,
            build_panel,
            build_asof_panel,
            breakouts=tuple(sorted({b.property for b in self._experiment.breakouts})),
            build_breakout_properties=self._build_breakout_properties_table,
            defs=self._defs,
            preflight_breakout=self._preflight_breakout,
            legacy_facade=_legacy_facade,
            preflight_day_request=self._preflight_day_request,
            note_reduction=self._note_reduction,
            horizon_metrics=effective_horizon,
            compliance_provider=self.compliance_summary,
            compliance_dates_provider=self.compliance_dates,
            validated_assignments=self._validated_assignments,
        )

    def _breakout_moments_source(
        self, breakout: Breakout, *, metrics: Sequence[Metric]
    ) -> BreakoutMomentsSource:
        """One declared `Breakout`'s totals-grain moments, served as a
        `MomentSource` scoped to exactly this breakout.

        Reuses the same ``pre_stats``/``den_stats`` construction
        ``breakout_summaries()`` performs per metric, so CUPED stays
        correct by construction. Builds only the totals-grain half
        ``breakout_summaries()`` also builds, skipping its day-axis half
        entirely since ``run_breakout`` never reads it.

        Passes ``cluster=None`` to ``_build_metric_summary``, matching
        ``breakout_summaries()``'s own totals-grain call; ``ratio_metrics=``
        still flows through underneath but stays inert at this grain, since
        ``group_summary`` only reads it under a declared ``cluster``. Under
        the declared uptake fact is joined at unit grain before this
        breakout-scoped reduction so its additive moments stay available.
        """
        exposures = self._get_exposures()
        breakout_source = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
        properties_table = self._build_breakout_properties_table(
            breakout_source,
            breakout.property,
            exposures,
        )
        by = [breakout.property]
        effective_horizon = list(self._horizon_metrics)
        for requested in metrics:
            if requested not in effective_horizon:
                effective_horizon.append(requested)
        effective_horizon = tuple(effective_horizon)
        uptake_events, uptake_window_days = self._resolve_uptake(
            self._experiment.cluster, operation="breakout_moments"
        )

        rows: list[dict[str, object]] = []
        for metric in metrics:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(
                    exposures,
                    metric,
                    horizon_metrics=effective_horizon,
                )
            )
            summary_by, _pre_stats, _den_stats, _totals_by = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=None,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
                by=by,
                properties_table=properties_table,
            )
            # Row dicts, never pa.concat_tables: differently winsorized metrics get
            # different winsor-column dtypes (null vs double), which it rejects.
            rows.extend(summary_by.to_pyarrow().to_pylist())

        return BreakoutMomentsSource(
            rows,
            dimension=breakout.property,
            metrics=list(metrics),
            study_id=self._experiment_name,
            source_name=breakout_source.name,
            design=self._context.design,
            plan=self._context.plan,
            configs=self._context.configs,
        )

    def breakout_summaries(self, *, metrics: Sequence[Metric]) -> dict[str, dict[str, pa.Table]]:
        """Build totals and activity-day or retention-cohort breakout summaries."""

        from increment.breakout.estimates import reject_retention_metrics

        self._refuse_declared_trigger("breakout_summaries")

        breakouts = self._experiment.breakouts
        if not breakouts or not metrics:
            return {}
        self._refuse_quantile_metrics(metrics, operation="breakout_summaries")
        self._refuse_percentile_winsorized(metrics, "breakout_summaries")
        reject_retention_metrics(metrics, "breakout_summaries", view="cohort")
        self._note_reduction()
        exposures = self._get_exposures()
        breakout_props = self._resolve_breakout_props(breakouts, exposures)
        effective_horizon = list(self._horizon_metrics)
        for requested in metrics:
            if requested not in effective_horizon:
                effective_horizon.append(requested)
        effective_horizon = tuple(effective_horizon)
        uptake_events, uptake_window_days = self._resolve_uptake(
            self._experiment.cluster, operation="breakout_summaries"
        )
        results: dict[str, dict[str, pa.Table]] = {}
        for metric in metrics:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(
                    exposures,
                    metric,
                    horizon_metrics=effective_horizon,
                )
            )
            den_panel: Table | None = None
            daily_panel: Table | None = None
            if not isinstance(metric, RetentionMetric):
                if den_events is not None:
                    den_panel = unit_day_panel(
                        exposures,
                        den_events,
                        self._experiment,
                        metric_name=metric.name,
                        end_date=self._union_event_horizon(effective_horizon),
                    )
                    den_panel = window_bound_stats(den_panel, metric)
                daily_panel = window_bound_stats(_panel, metric)
            for breakout, fs, properties_table in breakout_props:
                by = [breakout.property]
                summary_by, pre_stats_by, _den_stats_by, _totals_by = self._build_metric_summary(
                    metric,
                    exposures,
                    spine,
                    stats,
                    den_events,
                    fact_tbl,
                    value_col,
                    data_as_of,
                    cluster=None,
                    uptake_events=uptake_events,
                    uptake_window_days=uptake_window_days,
                    warn_on_censoring=True,
                    by=by,
                    properties_table=properties_table,
                )
                if isinstance(metric, RetentionMetric):
                    daily_by = cohort_group_summary(
                        spine,
                        stats,
                        metric,
                        self._experiment,
                        pre_stats=pre_stats_by,
                        by=by,
                        properties_table=properties_table,
                        data_as_of=data_as_of,
                        warn_on_censoring=False,
                    )
                else:
                    assert daily_panel is not None
                    dim_panel = join_breakout_dimension(daily_panel, properties_table, by)
                    daily_by = daily_group_summary(
                        dim_panel, metric=metric, by=by, den_panel=den_panel
                    )
                key = f"{metric.name}:{breakout.property}:{fs.name}"
                results[key] = {
                    "group_summary": summary_by.to_pyarrow(),
                    "daily_group_summary": daily_by.to_pyarrow(),
                }
        return results

    def factor_summaries(self, *, metrics: Sequence[Metric]) -> dict[str, pa.Table]:
        """Build whole-window summaries for every declared absorption factor."""
        self._refuse_declared_trigger("factor_summaries")
        factors = self._experiment.factors
        if not factors or not metrics:
            return {}
        self._refuse_quantile_metrics(metrics, operation="factor_summaries")
        self._refuse_percentile_winsorized(metrics, "factor_summaries")
        self._note_reduction()
        exposures = self._get_exposures()
        factor_props = self._resolve_breakout_props(factors, exposures)
        uptake_events, uptake_window_days = self._resolve_uptake(
            self._experiment.cluster, operation="factor_summaries"
        )
        results: dict[str, pa.Table] = {}
        for metric in metrics:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric)
            )
            for factor, fs, properties_table in factor_props:
                by = [factor.property]
                summary_by, _pre_stats_by, _den_stats_by, _totals_by = self._build_metric_summary(
                    metric,
                    exposures,
                    spine,
                    stats,
                    den_events,
                    fact_tbl,
                    value_col,
                    data_as_of,
                    cluster=None,
                    uptake_events=uptake_events,
                    uptake_window_days=uptake_window_days,
                    warn_on_censoring=True,
                    by=by,
                    properties_table=properties_table,
                )
                key = f"{metric.name}:{factor.property}:{fs.name}"
                results[key] = summary_by.to_pyarrow()
        return results

    def breakout_source(
        self, breakout: Breakout, *, metrics: Sequence[Metric]
    ) -> BreakoutMomentsSource:
        """Return one breakout-scoped moments source."""
        self._refuse_declared_trigger("breakout_source")
        self._refuse_quantile_metrics(metrics, operation="breakout_source")
        self._preflight_breakout(breakout, operation="breakout_source")
        with self._reduction_batch():
            return self._breakout_moments_source(breakout, metrics=metrics)

    def breakout_sources(
        self, breakouts: Sequence[Breakout], *, metrics: Sequence[Metric]
    ) -> tuple[BreakoutMomentsSource, ...]:
        """Return every breakout view in one reduction batch."""
        self._refuse_declared_trigger("breakout_sources")
        self._refuse_quantile_metrics(metrics, operation="breakout_sources")
        ordered = tuple(breakouts)
        for breakout in ordered:
            self._preflight_breakout(breakout, operation="breakout_sources")
        with self._reduction_batch():
            return tuple(
                self._breakout_moments_source(breakout, metrics=metrics) for breakout in ordered
            )

    def day_source(self, *, metrics: Sequence[Metric]) -> DayEvidenceSource:
        """Return the source-owned daily/as-of evidence view."""
        self._refuse_declared_trigger("day_source")
        self._refuse_quantile_metrics(metrics, operation="day_source")
        for metric in metrics:
            _refuse_unsupported_day_metric(metric, operation="day_source")
        return self._day_axis_source(selected=metrics, _legacy_facade=False)

    def _build_asof_panels(
        self,
        exposures: Table,
        metric: Metric,
        *,
        horizon_metrics: Sequence[Metric] | None = None,
        include_covariate: bool = False,
        include_uptake_horizon: bool = False,
    ) -> tuple[Table, Table | None, Table | None, int | None, ir.Scalar | None]:
        """Build outcome/uptake panels, the uptake window and outcome coverage edge.

        Shared by ``run_asof`` and ``run_asof_lift``, both un-dimensioned
        and per-segment: all four paths need the identical
        ``(panel, den_panel)`` pair and deliberately omit
        ``window_bound_stats`` on either side, since
        ``asof_group_summary`` does its own per-unit window masking -
        binding first would filter post-window rows away before the
        builder ever saw them.

        ``include_covariate`` attaches one fixed, zero-filled pre-period
        total per unit before the as-of reduction. It is requested only by
        CUPED lift readouts; value-only paths leave the panel unchanged.

        ``den_panel`` is ``None`` for every non-ratio metric. On the
        per-segment paths it is never dimension-joined: the builder
        joins it to the already-dimensioned numerator panel at
        ``(unit_id, ds)`` and reads every grouping column off that
        side, so joining the dimension onto the denominator too would
        produce a column nothing reads.

        ``uptake_panel``/``uptake_window_days`` mirror
        ``_summary_sql_for_metrics``/``_moments_for_metrics``'s own uptake resolution and are built
        whenever ``self._design`` is an ``Encouragement`` design,
        regardless of caller. ``uptake_panel`` is dense, like
        ``den_panel``; both are ``None`` for every non-``Encouragement``
        design, which ``asof_group_summary`` reads as "no uptake data,
        leave sum_d/cyd/cy2d null". ``run_asof`` ignores both return
        values; ``run_asof_lift`` passes them straight through.
        """
        (
            panel,
            _spine,
            stats,
            _events,
            den_events,
            fact_tbl,
            value_col,
            _data_as_of,
        ) = self._build_panel_for_metric(exposures, metric, horizon_metrics=horizon_metrics)
        horizon = self._union_event_horizon(horizon_metrics)
        outcome_edge: ir.Scalar | None = None

        uptake_panel: Table | None = None
        uptake_window_days: int | None = None
        _design = self._context.design
        if isinstance(_design, Encouragement):
            uptake_fs, _uptake_fact = _find_fact_source(self._defs, _design.uptake.fact)
            uptake_fact_tbl = self._get_fact_table(uptake_fs)
            # unit_day_panel needs metric_events's (unit_id, ts, metric, value) shape;
            # a synthetic ConversionMetric lets metric_events filter and stamp rows.
            uptake_events = metric_events(
                uptake_fact_tbl,
                ConversionMetric(
                    name="__uptake", entity=self._experiment.unit, fact=_design.uptake.fact
                ),
            )
            if include_uptake_horizon and self._experiment.observation_horizon is None:
                outcome_edge = horizon
                uptake_edge = compliance_event_horizon(exposures, uptake_events, self._experiment)
                horizon = cast(
                    "ir.Scalar",
                    ibis.coalesce(ibis.greatest(horizon, uptake_edge), horizon, uptake_edge),
                )
                spine = panel_spine(exposures, self._experiment, end_date=horizon)
                panel = _dense_unit_days(spine.filter(spine.ds.notnull()), stats).mutate(
                    metric=ibis.literal(metric.name)
                )
            uptake_events = _uptake_events_in_elapsed_window(
                exposures, uptake_events, _design.uptake.window_days
            )
            uptake_panel = unit_day_panel(
                exposures,
                uptake_events,
                self._experiment,
                metric_name="__uptake",
                end_date=horizon,
                include_exposure=True,
            )
            uptake_window_days = _design.uptake.window_days
        if include_covariate and self._experiment.n_pre_periods > 0:
            pre_events = self._build_pre_events(fact_tbl, metric, value_col)
            if pre_events is not None:
                pre_stats = pre_period_stats(
                    pre_events,
                    exposures,
                    self._experiment,
                    source_key=f"{metric.name}:pre",
                )
                panel = join_pre_period_covariate(panel, pre_stats)
        den_panel: Table | None = None
        if den_events is not None:
            den_panel = unit_day_panel(
                exposures,
                den_events,
                self._experiment,
                metric_name=metric.name,
                end_date=horizon,
            )

        return panel, den_panel, uptake_panel, uptake_window_days, outcome_edge

    def triggered_source(self) -> MomentSource:
        """Return a source-owned view of the triggered population."""
        return TriggeredPopulationSource(self)

    def _compliance_wire_payload(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> str | None:
        """JSON ``compliance_summary()`` payload for the moments-cube wire
        format (mirrors ``ASSIGNMENT_COUNTS_FIELD``/``DECISION_PLAN_FIELD``),
        or ``None`` when the design has no declared uptake cohort (only an
        Encouragement design does). Clustered payloads retain the cluster
        identity, member counts, and full bivariate uptake/size moments.
        """
        from increment.semantics.design import Encouragement as _Encouragement

        design = self._context.design
        if not isinstance(design, _Encouragement):
            return None
        summary = self.compliance_summary(design, population=population)
        return json.dumps(summary.to_wire(), sort_keys=True, separators=(",", ":"))

    def moments_source(
        self,
        *,
        metrics: Sequence[Metric],
        methods: list[Any] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        narrow_cuped: bool = False,
    ) -> MomentSource:
        """Build an in-process moments source for *metrics*.

        This is the source-owned counterpart to the export/from-moments wire
        boundary. The cube is reduced once, stamped with the same format and
        source-level assignment counts as an export, then rehydrated through
        :class:`MomentsSource` without exposing warehouse builders or caches.
        """
        cluster = self._context.cluster
        if cluster is not None and not isinstance(self._context.design, Encouragement):
            from increment.sources import refuse_cluster_grain_transport

            refuse_cluster_grain_transport(
                operation="moments_source",
                source="native",
                cluster=cluster,
                design=self._context.design,
                route_forward=(
                    "analyse the clustered source directly; direct clustered inference "
                    "preserves cluster-grain degrees of freedom and counts"
                ),
            )
        import pyarrow as pa

        from increment.decision_wire import compiled_plan_to_json
        from increment.sources import ASSIGNMENT_COUNTS_FIELD, MOMENTS_FORMAT

        selected = list(metrics)
        cube = self._moments_for_metrics(
            methods=methods,
            selected=selected,
            prior=prior,
            population=population,
            narrow_cuped=narrow_cuped,
        )
        cube = cube.append_column(
            "moments_format",
            pa.array([MOMENTS_FORMAT] * cube.num_rows, type=pa.int64()),
        )
        _grain, assignment_counts, unit_counts = self._counts_for_population(
            population, operation="moments_source"
        )
        assignment_counts.update(unit_counts)
        cube = cube.append_column(
            ASSIGNMENT_COUNTS_FIELD,
            pa.array(
                [json.dumps(assignment_counts, sort_keys=True, separators=(",", ":"))]
                * cube.num_rows,
                type=pa.string(),
            ),
        )
        compliance_payload = self._compliance_wire_payload(population=population)
        if compliance_payload is not None:
            cube = cube.append_column(
                COMPLIANCE_SUMMARY_FIELD,
                pa.array([compliance_payload] * cube.num_rows, type=pa.string()),
            )
        if cube.num_rows == 0 and not self._metrics:
            from increment.sources import MomentsSource, _design_summary_row

            row = _design_summary_row(
                self._experiment_name,
                compiled_plan_to_json(self._context.plan),
                json.dumps(assignment_counts, sort_keys=True, separators=(",", ":")),
                compliance_payload,
            )
            return MomentsSource(
                [row],
                metrics=self._metrics,
                study_id=self._experiment_name,
                bindings_by_name=self._experiment.bindings,
                design=self._context.design,
                plan=self._context.plan,
                path="warehouse",
            )
        from increment.sources import MomentsSource

        return MomentsSource(
            cube.to_pylist(),
            metrics=self._metrics,
            study_id=self._experiment_name,
            bindings_by_name=self._experiment.bindings,
            design=self._context.design,
            plan=self._context.plan,
            path="warehouse",
        )

    def build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
        horizon_metrics: Sequence[Metric] | None = None,
    ) -> Any:
        """Return live panel relations independent of operation-owned TEMP tables."""
        return self._build_panel_for_metric(
            exposures,
            metric,
            population,
            horizon_metrics=horizon_metrics,
            use_cache=False,
        )

    def panel_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]:
        """Return panel SQL for every declared metric and requested breakout."""
        return self._panel_sql_for_metrics(breakouts=list(breakouts))

    def summary_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]:
        """Return group-summary SQL for every declared metric and breakout."""
        return self._summary_sql_for_metrics(breakouts=list(breakouts))

    def export_moments(self, path: str | Path) -> None:
        """Write this experiment's moments (per-metric ``group_summary``
        rows, unioned) to a parquet file at *path*.

        The native family's transport format for
        ``Analysis.from_moments`` - see that facade method's docstring
        for the wire-format contract.

        Fixed-horizon exports carry ``group_summary`` rows, or one metric-free
        ``design_summary`` envelope, stamped ``moments_format=8``. Registered
        sequential exports instead carry one
        typed ``sequential_checkpoint`` envelope stamped ``moments_format=9``;
        they are not ordinary fixed moments rows. ``from_moments`` validates
        and strips these transport columns before constructing its immutable
        context. Formats 1-6 and missing or partial decision-plan
        payloads are refused; all supported inference variants and view
        policies round-trip without fallback.
        """
        if getattr(self.context.plan.inference, "registration", None) is not None:
            from increment.sources import export_source_moments

            return export_source_moments(
                self, path, observational_refusal=self._refuse_observational_quantile
            )

        if self._experiment.cluster is not None and not isinstance(
            self._context.design, Encouragement
        ):
            from increment.sources import refuse_cluster_grain_transport

            refuse_cluster_grain_transport(
                operation="export_moments",
                source="native",
                cluster=self._experiment.cluster,
                design=self._context.design,
                route_forward=(
                    "analyse the clustered source directly; direct clustered inference "
                    "preserves cluster-grain degrees of freedom and counts"
                ),
            )
        from increment.sources import refuse_quantile_moments_export

        refuse_quantile_moments_export(
            self._metrics,
            design=self._context.design,
            observational_refusal=self._refuse_observational_quantile,
        )
        if not self._metrics:
            from increment.sources import export_source_moments

            return export_source_moments(
                self, path, observational_refusal=self._refuse_observational_quantile
            )
        # Import lazily because parquet export requires the live ibis session.
        import pyarrow as pa
        import pyarrow.parquet as pq

        from increment.decision_wire import compiled_plan_to_json
        from increment.sources import DECISION_PLAN_FIELD, MOMENTS_FORMAT

        moments = self._moments_for_metrics()
        moments = moments.append_column(
            "moments_format", pa.array([MOMENTS_FORMAT] * moments.num_rows, type=pa.int64())
        )
        assignment_counts = self.unit_counts()
        moments = moments.append_column(
            ASSIGNMENT_COUNTS_FIELD,
            pa.array(
                [json.dumps(assignment_counts, sort_keys=True, separators=(",", ":"))]
                * moments.num_rows,
                type=pa.string(),
            ),
        )
        moments = moments.append_column(
            DECISION_PLAN_FIELD,
            pa.array(
                [compiled_plan_to_json(self._context.plan)] * moments.num_rows,
                type=pa.string(),
            ),
        )
        compliance_payload = self._compliance_wire_payload()
        if compliance_payload is not None:
            moments = moments.append_column(
                COMPLIANCE_SUMMARY_FIELD,
                pa.array([compliance_payload] * moments.num_rows, type=pa.string()),
            )
        moments = moments.replace_schema_metadata(
            {
                **(moments.schema.metadata or {}),
                b"increment.moments_format": str(MOMENTS_FORMAT).encode(),
            }
        )
        pq.write_table(moments, str(path))

    # - integrity/counting helpers ------------------------------------

    def _arm_counts(self, population: Table) -> dict[str, int]:
        """Unit counts per arm off an already-narrowed population table."""
        return arm_counts(self._con, population)

    def trigger_rates(self) -> dict[str, float]:
        """Observed triggered share per arm."""
        self._validate_mixed_assignments()
        assigned = self._arm_counts(self._get_exposures())
        triggered = self._arm_counts(self._get_trigger_population(operation="trigger_rates"))
        return {arm: (triggered.get(arm, 0) / n if n else 0.0) for arm, n in assigned.items()}

    def _counts_for_population(
        self, population: Literal["assigned", "triggered"], *, operation: str
    ) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]:
        """Return assignment counts for one source-owned population."""
        if population == "triggered":
            self._validate_trigger_capability(operation=operation)
        self._validate_mixed_assignments()
        exposures = (
            self._get_trigger_population(operation=operation)
            if population == "triggered"
            else self._get_exposures()
        )
        cluster = self._experiment.cluster
        if cluster is None:
            counts = self._arm_counts(exposures)
            unit_counts: dict[str, int] = {}
            grain: Literal["unit", "cluster"] = "unit"
        else:
            self._validate_cluster_labels(exposures, cluster)
            rows = self._con.to_pyarrow(
                cluster_exposure_counts(exposures, self._experiment)
            ).to_pylist()
            counts = {str(r["group_id"]): int(r["n_clusters"]) for r in rows}
            unit_counts = {str(r["group_id"]): int(r["n_units"]) for r in rows}
            grain = "cluster"
        mixed = self._mixed_assignment_units or 0
        unassigned = self._unassigned_assignment_units or 0
        if mixed:
            counts[MIXED_ASSIGNMENT_LABEL] = mixed
        if unassigned:
            counts[UNASSIGNED_LABEL] = unassigned
        return grain, counts, unit_counts

    def triggered_counts(
        self,
    ) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]:
        """Return assignment counts for the source's triggered population."""
        return self._counts_for_population("triggered", operation="triggered_counts")

    def assignment_counts(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> dict[str, int]:
        """Source-level enrolled counts independent of metric maturity."""
        return self._counts_for_population(population, operation="assignment_counts")[1]

    def allocation_history(self) -> pa.Table:
        """Return assigned-unit enrollment by declared day boundary and arm."""
        cluster = self._experiment.cluster
        if cluster is not None:
            _refuse_operation(
                operation="allocation_history",
                request={"experiment": self._experiment.name, "cluster": cluster},
                offered=("unit",),
                route="use srm() or triggered_counts() for the cluster-grain randomization check",
            )
        self._validate_mixed_assignments()
        exposures = self._get_exposures()
        history = daily_exposure_counts(exposures, self._experiment).order_by(["ds", "group_id"])
        return cast("pa.Table", self._con.to_pyarrow(history))

    # - MomentSource protocol -------------------------------------------

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> list[dict[str, Any]]:
        if grain != "total" and population != "assigned":
            _refuse_operation(
                operation="moments",
                request={"metric": metric.name, "grain": grain, "population": population},
                offered=("total",),
                route="request grain='total' for the triggered population",
            )
        if grain != "total":
            # Preserve direct MomentSource calls as source-owned lazy reads.
            return self._day_axis_source(selected=[metric], _legacy_facade=False).moments(
                metric,
                grain=grain,
                by=by,
                completed_windows_only=completed_windows_only,
                include_covariate=include_covariate,
            )
        if by:
            _refuse_operation(
                operation="moments",
                request={"metric": metric.name, "grain": grain, "by": tuple(by)},
                offered=tuple(sorted(self.capabilities)),
                route="use breakout_moments() for a dimensioned totals view",
            )
        combined = self._moments_for_metrics(selected=[metric], population=population)
        return cast("list[dict[str, Any]]", combined.to_pylist())

    def _sequential_observation_mapping(self):
        from increment.sequential_source import native_observation_mapping

        return native_observation_mapping(
            self._defs, self._experiment, on_mixed_assignment=self._on_mixed_assignment
        )

    def capture_sequential(self, *, finalized: bool, as_of: dt.date, previous=None):
        """Capture only units finalized through the explicit common horizon."""
        from increment.query.artifact_publish import artifact_context
        from increment.query.sequential_capture import (
            capture_relations,
            record_batches,
            validate_relational_capture,
        )
        from increment.sequential_state import sequential_refuse

        recipe = artifact_context(
            self._defs,
            self._experiment,
            self._on_mixed_assignment,
            metrics=self.context.metrics,
            encouragement_uptake=self.context.design
            if isinstance(self.context.design, Encouragement)
            else None,
        )
        previous = previous if previous is not None else getattr(self, "_sequential_snapshot", None)
        covariate = self._experiment.n_pre_periods > 0
        registration, _ = validate_relational_capture(
            self, finalized=finalized, as_of=as_of, previous=previous, covariate=covariate
        )
        modeled = {model.metric for model in registration.models if model.observable == "outcome"}
        uptake_facts = (
            (self.context.design.uptake.fact,)
            if isinstance(self.context.design, Encouragement)
            and any(model.observable == "uptake" for model in registration.models)
            else ()
        )
        with self._pinned_source_execution(
            metrics=[metric for metric in self.context.metrics if metric.name in modeled],
            uptake_facts=uptake_facts,
        ) as pinned:
            pinned._validate_mixed_assignments()
            exposures = pinned._get_exposures()
            cohort = exposures.filter(
                _local_date(exposures.first_exposure_ts, pinned._experiment)
                + ibis.interval(days=max(0, registration.reveal.longest_window_days - 1))
                <= ibis.literal(as_of)
            )

            def outcome(metric):
                if isinstance(metric, RatioMetric):
                    fs, fact = _find_fact_source(pinned._defs, metric.numerator.fact)
                    table = pinned._get_fact_table(fs)
                    events = metric_events(
                        table,
                        metric,
                        value_column=_resolve_value_column(fs, fact),
                        part="numerator",
                    )
                    dfs, dfact = _find_fact_source(pinned._defs, metric.denominator.fact)
                    den_events = metric_events(
                        pinned._get_fact_table(dfs),
                        metric,
                        value_column=_resolve_value_column(dfs, dfact),
                        part="denominator",
                    )
                    den_stats = post_exposure_stats(
                        den_events,
                        cohort,
                        source_key=f"{metric.name}:den",
                        experiment=pinned._experiment,
                    )
                else:
                    fs, fact = _find_fact_source(pinned._defs, metric.fact)
                    events = metric_events(
                        pinned._get_fact_table(fs),
                        metric,
                        value_column=_resolve_value_column(fs, fact),
                    )
                    den_stats = None
                spine, stats = unit_day_spine_stats(
                    cohort, events, pinned._experiment, metric.name, end_date=ibis.literal(as_of)
                )
                # The same zero-filled pre-period total fixed-horizon CUPED reads.
                pre_stats = (
                    pre_period_stats(
                        events, cohort, pinned._experiment, source_key=f"{metric.name}:pre"
                    )
                    if covariate
                    else None
                )
                totals = unit_totals(
                    spine,
                    stats,
                    metric,
                    pinned._experiment,
                    pre_stats=pre_stats,
                    den_stats=den_stats,
                    data_as_of=as_of,
                    warn_on_censoring=False,
                )
                # A fixed threshold clips each unit's finalized total as it is revealed.
                return winsorize_unit_totals(totals, metric)

            def uptake():
                design = pinned.context.design
                if not isinstance(design, Encouragement):
                    sequential_refuse("source.invalid", "uptake requires encouragement assignment")
                fs, _ = _find_fact_source(pinned._defs, design.uptake.fact)
                table = pinned._get_fact_table(fs)
                events = table.filter(table.event == design.uptake.fact)
                return _attach_uptake_flag(cohort, cohort, events, design.uptake.window_days)

            snapshot = capture_relations(
                pinned,
                outcome,
                uptake,
                cohort,
                lambda relation: record_batches(pinned._con, relation),
                recipe_id=recipe.sha256,
                finalized=finalized,
                as_of=as_of,
                previous=previous,
                covariate=covariate,
            )
        self._sequential_snapshot = snapshot
        return snapshot

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        population: Literal["assigned", "triggered"] = "assigned",
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> pa.Table:
        if outcome_stage not in ("raw", "transformed"):
            from increment.winsor import winsor_refuse

            winsor_refuse("invalid_state", "Unknown unit outcome stage.")
        cluster = self._experiment.cluster
        if "cluster_id" in covariates and cluster != "cluster_id":
            _refuse(_NATIVE_COVARIATE_RESERVED, covariate="cluster_id", cluster=cluster)
        # A requested declared cluster column reuses the attached metadata column.
        covariate_sources = {
            name: self._resolve_unit_covariate_source(name)
            for name in dict.fromkeys(covariates)
            if name != "cluster_id"
        }
        if outcome_stage == "raw" and self._session._source_tables is None:
            with self._pinned_source_execution(metrics=(metric,)) as pinned:
                return pinned.unit_frame(
                    metric,
                    covariates=covariates,
                    population=population,
                    outcome_stage=outcome_stage,
                )
        with self._reduction_batch():
            exposures = (
                self._get_trigger_population(operation="unit_frame")
                if population == "triggered"
                else self._get_exposures()
            )
            # Covariates are scoped to before assignment, whichever population is read.
            assigned = exposures if population == "assigned" else self._get_exposures()
            if cluster is not None:
                self._validate_cluster_labels(exposures, cluster)
            uptake_events, uptake_window_days = self._resolve_uptake(
                cluster, operation="unit_frame"
            )
            panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population)
            )
            _summary, _pre_stats, _den_stats, totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
                want_cuped=False,
            )
            if (
                outcome_stage == "raw"
                and getattr(metric, "winsorization", None) is not None
                and "y_raw" not in totals.columns
            ):
                from increment.winsor import winsor_refuse

                winsor_refuse(
                    "raw_state_required", "Warehouse reduction did not retain pre-winsor outcomes."
                )
            for name, fact_source in covariate_sources.items():
                properties = self._build_breakout_properties_table(fact_source, name, assigned)
                prop = next(p for p in self._defs.properties_of(fact_source) if p.name == name)
                totals = join_unit_covariates(
                    totals, properties, name, categorical=prop.dtype == "string"
                )
            select_cols: dict[str, Any] = {
                "unit_id": totals["unit_id"],
                "group_id": totals["group_id"],
                "y": totals["y_raw"]
                if outcome_stage == "raw" and "y_raw" in totals.columns
                else totals["y"],
            }
            if isinstance(metric, RatioMetric):
                select_cols["y_den"] = totals["y_den"]
            for name in covariate_sources:
                select_cols[name] = totals[name]
            if cluster is not None:
                select_cols["cluster_id"] = totals[cluster].cast("string")
            return cast("pa.Table", self._con.to_pyarrow(totals.select(**select_cols)))

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        if grain != "total":
            _refuse(
                _NATIVE_SQL_GRAIN,
                grain=grain,
                offered=self.capabilities,
            )
        return self._summary_sql_for_metrics()

    def unit_counts(self) -> dict[str, int]:
        """Per-arm enrolled unit counts, with the mixed-assignment
        accounting entry folded in - matches every other native
        readout's "refresh the integrity policy, then count" contract.
        """
        self._validate_mixed_assignments()
        counts = self._arm_counts(self._get_exposures())
        mixed = self._mixed_assignment_units or 0
        unassigned = self._unassigned_assignment_units or 0
        if mixed:
            counts[MIXED_ASSIGNMENT_LABEL] = mixed
        if unassigned:
            counts[UNASSIGNED_LABEL] = unassigned
        return counts

    def cluster_counts(self) -> dict[str, int]:
        cluster = self._experiment.cluster
        if cluster is None:
            _refuse(
                _NATIVE_WAREHOUSE_CLUSTER,
                source="this warehouse source",
                because=f"experiment {self._experiment.name!r} declares no cluster",
            )
        exposures = self._get_exposures()
        self._validate_cluster_labels(exposures, cluster)
        rows = self._con.to_pyarrow(
            cluster_exposure_counts(exposures, self._experiment)
        ).to_pylist()
        return {str(r["group_id"]): int(r["n_clusters"]) for r in rows}

    def close(self) -> None:
        self._session.drop_materialized()
        self._exposures_cache = None
        self._trigger_cache = None
        self._mixed_assignment_units = None
        self._unassigned_assignment_units = None
        self._mixed_assignment_warned = False
        self._mixed_assignment_fingerprint = None
        self._union_horizon_cache = None
        self._union_horizon_metric_cache = {}
        self._panel_cache = {}
        self._reduction_calls = 0
        self._materialized = False
        self._reduction_batch_depth = 0
        self._assignment_validation_depth = 0

    # - public aliases for the native-only capability names -------------

    sitewide_volume = _site_volume_row

    def day_axis_source(self, *, selected: Sequence[Metric] | None = None) -> _DaySource:
        """Public direct adapter with source-owned reduction accounting."""
        return self._day_axis_source(selected=selected, _legacy_facade=False)

    breakout_moments = _breakout_moments_source
