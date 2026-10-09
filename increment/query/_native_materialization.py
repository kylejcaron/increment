"""Source-owned materialization and shared native metric reduction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import ibis.expr.types as ir
from ibis import Table

from increment.query.builders import (
    declared_binary_metrics,
    group_summary,
    panel_spine,
    post_exposure_stats,
    pre_period_stats,
    resolved_measure_key,
    unit_totals,
    winsorize_unit_totals,
)
from increment.semantics.models import RatioMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from increment.semantics.models import Experiment, Metric


class _MaterializationSource(Protocol):
    _store: Literal["auto", "always", "none"]

    @property
    def _metrics(self) -> Sequence[Metric]: ...

    _reduction_calls: int
    _reduction_batch_depth: int
    _materialized: bool
    _panel_cache: Any
    _experiment_name: str
    _experiment: Experiment
    _con: Any
    _session: Any

    def _validate_mixed_assignments(self) -> None: ...

    def _invalidate_materialization(self) -> None: ...

    def _get_exposures(self) -> Table: ...

    def _build_panel_for_metric_impl(self, exposures: Table, metric: Metric) -> Any: ...

    def _union_event_horizon(self) -> ir.Scalar: ...

    def _build_pre_events(
        self, fact_tbl: Table, metric: Metric, value_col: str | None
    ) -> Table | None: ...


class _NativeMaterializationMixin:
    """Manage the source's one-operation shared materialized reduction state."""

    def materialize(self: _MaterializationSource) -> None:
        """Rebuild materialized tables from current warehouse state.

        Each explicit call rescans the source; reuse is confined to one
        operation. No materialization is needed for store='none' or no metrics.
        """
        if self._store == "none" or not self._metrics:
            return
        self._validate_mixed_assignments()
        self._invalidate_materialization()
        cast(Any, self)._materialize()
        self._reduction_calls += 1

    def _note_reduction(self: _MaterializationSource) -> None:
        """Count one top-level reduction call, with store-threshold materialization.

        Nested batches share the current operation's cached reduction; later calls
        rebuild from current warehouse state once the selected threshold is reached.
        """
        self._validate_mixed_assignments()
        self._reduction_calls += 1
        if self._store == "none":
            return
        threshold = 1 if self._store == "always" else 2
        if self._reduction_calls >= threshold:
            self._invalidate_materialization()
            cast(Any, self)._materialize()

    @contextmanager
    def _reduction_batch(self: _MaterializationSource) -> Iterator[None]:
        """Count one source-owned reduction for a nested batch of views."""
        outer = self._reduction_batch_depth == 0
        if outer:
            cast(Any, self)._note_reduction()
        self._reduction_batch_depth += 1
        try:
            yield
        finally:
            self._reduction_batch_depth -= 1

    def _temp_name(self: _MaterializationSource, *parts: str) -> str:
        """Return a backend-portable temporary-table name."""
        return self._session.temp_name(*parts)

    def _materialize_table(self: _MaterializationSource, name: str, expr: ir.Table) -> ir.Table:
        """``CREATE TEMP TABLE``, degrading to an unmaterialized expression on failure."""
        return self._session.materialize_table(name, expr)

    def _materialize(self: _MaterializationSource) -> None:
        """Materialize the shared spine and one stats table per distinct measure.

        Metrics with the same resolved measure share a stats table. The spine
        uses the same memoized event horizon threaded through panel construction,
        so it is identical to the corresponding fused readout's spine.
        """
        exposures = self._get_exposures()
        raw: dict[
            str, tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar]
        ] = {}
        for metric in self._metrics:
            raw[metric.name] = self._build_panel_for_metric_impl(exposures, metric)

        spine_name = cast(Any, self)._temp_name(self._experiment_name, "spine")
        spine_expr = panel_spine(exposures, self._experiment, end_date=self._union_event_horizon())
        spine = cast(Any, self)._materialize_table(spine_name, spine_expr)

        measure_tables: dict[
            tuple[
                str, tuple[tuple[str, str, tuple[str | int | float | bool, ...]], ...], str | None
            ],
            Table,
        ] = {}

        # Metrics sharing a measure use one stats table instead of rescanning events.
        for metric in self._metrics:
            panel, _spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = raw[
                metric.name
            ]
            measure_ref = metric.numerator if isinstance(metric, RatioMetric) else metric
            measure_key = resolved_measure_key(measure_ref, value_column=value_col)
            materialized_stats = measure_tables.get(measure_key)
            if materialized_stats is None:
                stats_name = cast(Any, self)._temp_name(self._experiment_name, "stats", metric.name)
                materialized_stats = cast(Any, self)._materialize_table(stats_name, stats)
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

    def _build_metric_summary(  # noqa: PLR0913
        self: _MaterializationSource,
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

        Retention cohorts reuse pre-period stats; quantiles use unit totals.
        Disable CUPED work when no requested method needs a pre-period covariate.
        """
        # Build CUPED covariates from the same fact/value-column/aggregation as the outcome.
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
            uptake_exposures=exposures,
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
        # Clustered ratio rows carry their own per-cluster denominator total.
        summary = group_summary(
            totals,
            by=list(by) or None,
            cluster=cluster,
            ratio_metrics=[metric.name] if metric.type == "ratio" else None,
            uptake=uptake_events is not None,
            binary_metrics=declared_binary_metrics([metric]),
        )
        return summary, pre_stats, den_stats, totals
