from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import ibis.expr.types as ir

from increment.errors import CapabilityError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.query._native_refusals import _refuse_operation, _winsorization_bounds
from increment.query.builders import (
    _censor_to_observable_window,
    _final_maturity_day,
    asof_group_summary,
    cohort_group_summary,
    daily_group_summary,
    join_breakout_dimension,
    metric_events,
    pre_period_stats,
    window_bound_stats,
)
from increment.query.native_contract import DimensionedDayEvidence
from increment.semantics.design import Encouragement
from increment.semantics.models import (
    Breakout,
    Definitions,
    Experiment,
    FactSource,
    Metric,
    RatioMetric,
    RetentionMetric,
)
from increment.sources import ComplianceSummary, SourceContext, SourceOperation

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from increment.sources import Grain


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
            tuple[
                ir.Table,
                ir.Table | None,
                ir.Table | None,
                int | None,
                ir.Scalar | None,
                dt.date | None,
            ],
        ],
        breakouts: tuple[str, ...] = (),
        *,
        triggered_observation_edges: Callable[[Metric], tuple[dt.date, dt.date | None]],
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
        self._triggered_observation_edges = triggered_observation_edges
        self._ready = False
        self._asof_panel_cache: dict[
            tuple[Metric, tuple[Metric, ...], bool, bool],
            tuple[
                ir.Table,
                ir.Table | None,
                ir.Table | None,
                int | None,
                ir.Scalar | None,
                dt.date | None,
            ],
        ] = {}
        self._breakout_properties_cache: dict[tuple[str, str], ir.Table] = {}

    def triggered_observation_edges(self, metric: Metric) -> tuple[dt.date, dt.date | None]:
        """Return the effective observed edge and optional conservative certification edge."""
        return self._triggered_observation_edges(metric)

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
    ) -> tuple[
        ir.Table,
        ir.Table | None,
        ir.Table | None,
        int | None,
        ir.Scalar | None,
        dt.date | None,
    ]:
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
            panel, den_panel, uptake_panel, uptake_window_days, outcome_edge, uptake_edge = cached
            data_as_of = self._build_panel(exposures, metric, horizon)[-1]
            panel = _censor_to_observable_window(
                panel, metric, self._experiment, data_as_of, warn_on_censoring=False
            )
            return panel, den_panel, uptake_panel, uptake_window_days, outcome_edge, uptake_edge
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
            (
                panel,
                den_panel,
                uptake_panel,
                uptake_window_days,
                outcome_edge,
                uptake_certified_edge,
            ) = self._cached_asof_panel(
                exposures,
                metric,
                include_covariate,
                include_uptake_horizon=True,
                completed_windows_only=completed_windows_only,
            )
            asof = asof_group_summary(
                panel,
                metric,
                den_panel=den_panel,
                uptake_panel=uptake_panel,
                uptake_window_days=uptake_window_days,
                _uptake_elapsed_windowed=uptake_panel is not None,
                _uptake_certified_edge=uptake_certified_edge,
                _uptake_day_boundary_offset=self._experiment.day_boundary_offset,
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

        panel, den_panel, _uptake_panel, _uptake_window_days, _edge, _uptake_edge = (
            self._cached_asof_panel(exposures, metric, include_covariate)
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
            (
                panel,
                den_panel,
                uptake_panel,
                uptake_window_days,
                outcome_edge,
                uptake_certified_edge,
            ) = self._cached_asof_panel(
                exposures,
                metric,
                include_covariate,
                include_uptake_horizon=True,
                completed_windows_only=completed_windows_only,
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
                _uptake_certified_edge=uptake_certified_edge,
                _uptake_day_boundary_offset=self._experiment.day_boundary_offset,
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

        panel, den_panel, _uptake_panel, _uptake_window_days, _edge, _uptake_edge = (
            self._cached_asof_panel(exposures, metric, include_covariate)
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
