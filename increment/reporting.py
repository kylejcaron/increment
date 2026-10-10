"""Report - the public facade for calendar metric trends.

The report twin of `Analysis`: same semantic layer, same fact sources, no
experiment. A readout renders an experiment's results; a report evaluates
metrics over calendar time.

This module is pure stats/query and must not import any visualization
library.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import ibis
import ibis.expr.types as ir
from ibis import to_sql as ibis_to_sql

from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.query.builders import (
    _local_date_at_offset,
    day_boundary_offset,
    metric_events,
    unit_day_stats,
)
from increment.query.calendar import (
    PeriodGrain,
    WeekStart,
    active_period_values,
    asof_property_table,
    calendar_periods,
    period_moments,
    period_population,
    period_unit_values,
    ratio_period_values,
    report_horizon_relation,
    rolling_active_values,
    rolling_total_values,
    total_period_values,
    validate_report_population,
)

# Shared column-normalization contract with the Analysis facade: both
# facades must feed the query builders identically shaped tables.
from increment.query.fact_resolution import _find_fact_source
from increment.query.session import WarehouseSession
from increment.semantics.loader import admit_read_only_sql, load, resolve_execution_dialect
from increment.semantics.models import (
    ActiveMetric,
    ConversionMetric,
    Definitions,
    FactSource,
    MeanMetric,
    RatioMetric,
    TotalMetric,
)

__all__ = ["MetricTrend", "Report"]


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ibis.backends.sql import SQLBackend

_REFUSED_TYPES: dict[str, str] = {
    "retention": (
        "not supported in reports: retention needs a cohort anchor, "
        "which reports do not have -- use the experiment path"
    ),
    "quantile": (
        "quantile metrics have no moments representation and the report "
        "path carries no per-unit rows -- not supported in reports; use the experiment path"
    ),
}

_REPORT_REFUSAL = RefusalSpec(
    "report.metric.unsupported",
    CapabilityError,
    lambda *, errors: "report cannot serve:\n  - " + "\n  - ".join(errors),
)

# Aggregations whose "no events" value is a true zero for a population unit.
_ZERO_FILL_AGGS = frozenset({"conversion", "sum", "count", "count_distinct", "avg_calendar_day"})


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "reporting.alpha_finite": "alpha must be finite and in (0, 1); got {alpha!r}",
        "reporting.week_start_monday": "week_start must be 'monday' or 'sunday'; got {week_start!r}",
        "reporting.metric_trend.unknown_backend": "unknown backend {backend!r}",
    },
)
_raise = raiser(_REFUSALS)


def _validate_report_options(alpha: float, week_start: str) -> None:
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        _raise("reporting.alpha_finite", alpha=alpha)
    if week_start not in ("monday", "sunday"):
        _raise("reporting.week_start_monday", week_start=week_start)


class MetricTrend:
    """One metric's calendar trend: a lazy ibis query plus metadata.

    When ``Report.metric`` omits ``end``, it resolves the selected metric's
    fact horizon through a bounded aggregate over the bound warehouse
    relation at construction; no in-memory row is registered. The returned
    trend remains lazy and its default edge is a snapshot.

    Columns (order not guaranteed, only names): `metric`, `grain`, `window`,
    `period`, `[dims...]`, `period_complete`, `n`, `value`, `ci_lb`, `ci_ub`.
    `n` is the per-period unit denominator for entity-scoped metrics and
    NULL for total/active (an `active` metric's count is its `value`).
    `ci_lb`/`ci_ub` are NULL for ratio (a point estimate only) and for
    total/active (no variance concept). `grain` keeps materialized rows
    self-describing across grains. `window` is the rolling trailing-day
    size (total/active only), NULL for every calendar-bucket trend.

    `period_complete` is true once the metric's own fact has any event at
    or past the period's end (`period_end <= max(fact ts)`), so a period
    can be marked complete up to one partial day early. Fine for date-grain
    horizons; a consumer gating on the last complete period of a
    still-loading warehouse should ignore it or wait for the next load.
    """

    def __init__(
        self,
        table: ir.Table,
        *,
        metric: str,
        grain: str,
        window: int | None,
        week_start: str,
        denominator: Literal["active", "population", "none"],
        start: dt.date,
        end: dt.date | None,
        by: Sequence[str],
        alpha: float,
    ) -> None:
        self._table = table
        self.metric = metric
        self.grain = grain
        self.window = window
        self.week_start = week_start
        self.denominator = denominator
        self.start = start
        self.end = end
        self.by = tuple(by)
        self.alpha = alpha

    def to_table(self) -> ir.Table:
        return self._table

    def sql(self) -> str:
        return str(ibis_to_sql(self._table))

    def to_frame(self, backend: str = "pandas"):
        import narwhals as nw

        frame = nw.from_native(self._table.to_pyarrow(), eager_only=True)
        if backend == "pyarrow":
            return frame.to_native()
        if backend == "pandas":
            return frame.to_pandas()
        if backend == "polars":
            return frame.to_polars()
        _raise("reporting.metric_trend.unknown_backend", backend=backend)


class Report:
    """Evaluate governed metrics over calendar time - no experiment.

    An omitted end date resolves the selected metric's own fact horizon
    through a bounded aggregate when the trend is constructed.
    ``Report.metrics`` resolves each selected metric independently. The
    default edge is therefore a snapshot, and ranges beyond the ten-year
    calendar capacity refuse before a trend is returned. Population SQL and
    the required ``unit_id`` key are admitted before this data query.

    Usage::

        report = Report.from_definitions("definitions/", con)
        trend = report.metric("purchase_rate", grain="week",
                              start=dt.date(2026, 1, 1))
        trend.to_frame()
    """

    def __init__(self, defs: Definitions, con: SQLBackend) -> None:
        snapshot = Definitions.model_validate(defs.model_dump())
        dialect = resolve_execution_dialect(snapshot, con)
        for fact_source in snapshot.fact_sources:
            if fact_source.sql.strip():
                admit_read_only_sql(
                    fact_source.sql,
                    dialect=dialect,
                    label=f"fact source '{fact_source.name}'",
                )
        for dim_source in snapshot.dim_sources:
            if dim_source.sql.strip():
                admit_read_only_sql(
                    dim_source.sql,
                    dialect=dialect,
                    label=f"dim source '{dim_source.name}'",
                )
        for exposure in snapshot.exposures:
            if exposure.sql is not None and exposure.sql.strip():
                admit_read_only_sql(
                    exposure.sql,
                    dialect=dialect,
                    label=f"exposure '{exposure.name}'",
                )
        self._defs = snapshot
        self._con = con
        self._session = WarehouseSession(con, snapshot)

    @classmethod
    def from_definitions(cls, defs: Definitions | str | Path, con: SQLBackend) -> Report:
        if not isinstance(defs, Definitions):
            defs = load(defs)
        return cls(defs, con)

    # ── public API ─────────────────────────────────────────────────────

    def metric(
        self,
        name: str,
        *,
        grain: PeriodGrain = "day",
        by: Sequence[str] | None = None,
        start: dt.date,
        end: dt.date | None = None,
        week_start: WeekStart | None = None,
        population: str | ir.Table | None = None,
        alpha: float = 0.05,
        window: int | None = None,
    ) -> MetricTrend:
        week_start = self._defs.week_start if week_start is None else week_start
        _validate_report_options(alpha, week_start)
        self._validate([name], by=by, population=population, window=window)
        return self._build(
            name,
            grain=grain,
            by=by,
            start=start,
            end=end,
            week_start=week_start,
            population=population,
            alpha=alpha,
            window=window,
        )

    def metrics(
        self,
        names: Sequence[str],
        *,
        grain: PeriodGrain = "day",
        by: Sequence[str] | None = None,
        start: dt.date,
        end: dt.date | None = None,
        week_start: WeekStart | None = None,
        population: str | ir.Table | None = None,
        alpha: float = 0.05,
        window: int | None = None,
    ) -> dict[str, MetricTrend]:
        week_start = self._defs.week_start if week_start is None else week_start
        _validate_report_options(alpha, week_start)
        self._validate(names, by=by, population=population, window=window)  # all offenders
        return {
            n: self._build(
                n,
                grain=grain,
                by=by,
                start=start,
                end=end,
                week_start=week_start,
                population=population,
                alpha=alpha,
                window=window,
            )
            for n in names
        }

    # ── validation (static: never touches the warehouse) ──────────────

    def _validate(
        self,
        names: Sequence[str],
        *,
        by: Sequence[str] | None,
        population: str | ir.Table | None,
        window: int | None,
    ) -> None:
        errors: list[str] = []
        by = list(by or [])
        seen: set[str] = set()
        for name in names:
            if name in seen:
                errors.append(f"'{name}': appears more than once in the metrics list")
                continue
            seen.add(name)
            m = self._defs.metric(name)
            if m is None:
                errors.append(f"'{name}': not defined in the definitions")
                continue
            if m.type in _REFUSED_TYPES:
                errors.append(f"'{name}': {_REFUSED_TYPES[m.type]}")
                continue
            if by:
                if m.type == "ratio":
                    errors.append(
                        f"'{name}': by= is not supported for ratio metrics in "
                        f"v1 -- a segmented ratio needs the as-of dimension "
                        f"joined through both parts"
                    )
                    continue
                if m.type in ("mean", "conversion") and len(by) > 1:
                    errors.append(
                        f"'{name}': entity-scoped report metrics support one "
                        f"dimension per call (got {by})"
                    )
                fs = _find_fact_source(self._defs, m.fact)[0]
                props = {p.name for p in self._defs.properties_of(fs)}
                missing = [d for d in by if d not in props]
                if missing:
                    errors.append(
                        f"'{name}': dimension(s) {missing} are not properties "
                        f"of fact source '{fs.name if fs else '?'}' "
                        f"(available: {sorted(props)})"
                    )
            if window is not None:
                if window < 1:
                    errors.append(f"'{name}': window must be >= 1 day (got {window})")
                elif m.type not in ("total", "active"):
                    errors.append(
                        f"'{name}': rolling window= applies to total/active "
                        f"metrics only -- {m.type} metrics have no overlapping-window "
                        f"denominator in a report"
                    )
            if population is not None and m.type in ("total", "active"):
                errors.append(
                    f"'{name}': population= has no unit denominator to scope on a "
                    f"{m.type} metric -- drop it, or use an entity-scoped metric"
                )
            if isinstance(m, ConversionMetric) and not m.filters and population is None:
                fs = _find_fact_source(self._defs, m.fact)[0]
                if len(fs.facts) == 1:
                    errors.append(
                        f"'{name}': the default denominator (units active on "
                        f"fact source '{fs.name}') is degenerate -- "
                        f"'{m.fact}' is the source's only fact and the metric "
                        f"is filterless, so every active unit trivially "
                        f"qualifies (rate == 100%). Supply population=."
                    )
        if errors:
            refuse(_REPORT_REFUSAL, errors=errors)

    # ── query assembly ─────────────────────────────────────────────────

    def _resolve_population(self, population: str | ir.Table | None) -> ir.Table | None:
        if population is None or isinstance(population, ir.Table):
            return population
        dialect = resolve_execution_dialect(self._defs, self._con)
        admit_read_only_sql(
            population,
            dialect=dialect,
            label="report population",
        )
        return self._con.sql(population, dialect=self._defs.dialect)

    def _resolve_dims(self, fs: FactSource, by: Sequence[str]) -> list[tuple[str, str]]:
        """Map dim names to (property_name, table_column) pairs; names were
        already validated in `_validate`.

        The table column is always the property name (never
        `Property.column`) since `WarehouseSession.fact_table` renames every
        property to its logical name first. Returned as pairs, not a bare
        list, because callers key downstream renames and selects off it.
        """
        props = {p.name for p in self._defs.properties_of(fs)}
        return [(d, d) for d in by if d in props]

    @staticmethod
    def _finalize(trend: ir.Table, dims: Sequence[str]) -> ir.Table:
        """Fix the column set (order is non-contractual, names are)."""
        return trend.select(
            "metric",
            "grain",
            "window",
            "period",
            *dims,
            "period_complete",
            "n",
            "value",
            "ci_lb",
            "ci_ub",
        )

    def _build(  # noqa: PLR0915
        self,
        name: str,
        *,
        grain: PeriodGrain,
        by: Sequence[str] | None,
        start: dt.date,
        end: dt.date | None,
        week_start: WeekStart,
        population: str | ir.Table | None,
        alpha: float,
        window: int | None,
    ) -> MetricTrend:
        from scipy.stats import norm

        from increment.estimation._tails import two_sided_critical_value

        m = self._defs.metric(name)
        assert m is not None  # _validate ran
        by = list(by or [])
        z = two_sided_critical_value(norm.isf, alpha, what="report metric confidence interval")
        # Admit and validate the caller population before any data query.
        pop_tbl = self._resolve_population(population)
        validate_report_population(pop_tbl)
        # Report/calendar path carries only Definitions.day_boundary (a raw
        # string), never an Experiment -- see builders.day_boundary_offset.
        offset = day_boundary_offset(self._defs.day_boundary)

        # ── resolve sources and the completeness horizon ───────────────

        # A no-data fact's max(ts) is NULL, so its sentinel is one day
        # before `start`, keeping every period incomplete honestly.
        no_data_date = start - dt.timedelta(days=1)
        no_data_sentinel = ibis.literal(no_data_date)
        if isinstance(m, RatioMetric):
            num_fs, num_fact = _find_fact_source(self._defs, m.numerator.fact)
            den_fs, den_fact = _find_fact_source(self._defs, m.denominator.fact)
            num_tbl = self._session.fact_table(num_fs, m.entity)
            den_tbl = self._session.fact_table(den_fs, m.entity)
            # Horizon scoped to each part's own fact, not the whole source,
            # so a fresher sibling fact can't mask this one still loading.
            num_horizon = ibis.coalesce(
                _local_date_at_offset(
                    num_tbl.filter(num_tbl.event == m.numerator.fact).ts.max(), offset
                ),
                no_data_sentinel,
            )
            den_horizon = ibis.coalesce(
                _local_date_at_offset(
                    den_tbl.filter(den_tbl.event == m.denominator.fact).ts.max(), offset
                ),
                no_data_sentinel,
            )
            horizon = cast("ir.Scalar", ibis.least(num_horizon, den_horizon))
            horizon_facts = [(num_tbl, m.numerator.fact), (den_tbl, m.denominator.fact)]
        else:
            fs, fact_def = _find_fact_source(self._defs, m.fact)
            unit = fs.entities[0] if isinstance(m, TotalMetric) else m.entity
            tbl = self._session.fact_table(fs, unit)
            # Horizon scoped to the metric's own fact.
            horizon = cast(
                "ir.Scalar",
                ibis.coalesce(
                    _local_date_at_offset(tbl.filter(tbl.event == m.fact).ts.max(), offset),
                    no_data_sentinel,
                ),
            )
            horizon_facts = [(tbl, m.fact)]

        if end is None:
            # Execute only a bounded aggregate over the already-bound fact
            # relations; no client-side one-row table is registered.
            resolved_horizon = self._con.execute(
                report_horizon_relation(
                    horizon_facts,
                    day_boundary_offset=offset,
                    no_data_sentinel=no_data_date,
                )
            ).iloc[0, 0]
            if isinstance(resolved_horizon, dt.datetime):
                resolved_horizon = resolved_horizon.date()
            edge = cast("dt.date", resolved_horizon)
        else:
            edge = end
        periods = calendar_periods(
            start,
            end_edge=edge,
            data_horizon=horizon,
            grain=grain,
            week_start=week_start,
        )
        null_f64 = ibis.null().cast("float64")
        window_lit = (
            ibis.literal(window, type="int64") if window is not None else ibis.null().cast("int64")
        )

        # ── total / active: no unit layer ──────────────────────────────
        if isinstance(m, TotalMetric | ActiveMetric):
            dims = self._resolve_dims(fs, by)
            keep = [c for (_, c) in dims]
            events = metric_events(
                tbl,
                m,
                value_column=fact_def.column if isinstance(m, TotalMetric) else None,
                keep=keep,
            )
            if dims:
                events = events.rename(**dict(dims))
            dim_names = [d for (d, _) in dims]
            if isinstance(m, TotalMetric):
                if window is not None:
                    out = rolling_total_values(
                        events,
                        periods,
                        window=window,
                        aggregation=m.aggregation,
                        start=start,
                        end_edge=edge,
                        by=dim_names,
                        day_boundary_offset=offset,
                    )
                else:
                    out = total_period_values(
                        events,
                        periods,
                        aggregation=m.aggregation,
                        grain=grain,
                        week_start=week_start,
                        start=start,
                        end_edge=edge,
                        by=dim_names,
                        day_boundary_offset=offset,
                    )
            else:
                if window is not None:
                    out = rolling_active_values(
                        events,
                        periods,
                        window=window,
                        start=start,
                        end_edge=edge,
                        by=dim_names,
                        day_boundary_offset=offset,
                    )
                else:
                    out = active_period_values(
                        events,
                        periods,
                        grain=grain,
                        week_start=week_start,
                        start=start,
                        end_edge=edge,
                        by=dim_names,
                        day_boundary_offset=offset,
                    )
            out = out.mutate(
                metric=ibis.literal(name),
                grain=ibis.literal(grain),
                window=window_lit,
                n=ibis.null().cast("int64"),
                ci_lb=null_f64,
                ci_ub=null_f64,
            )
            return MetricTrend(
                self._finalize(out, dim_names),
                metric=name,
                grain=grain,
                window=window,
                week_start=week_start,
                denominator="none",
                start=start,
                end=end,
                by=by,
                alpha=alpha,
            )

        denominator: Literal["active", "population"] = (
            "population" if pop_tbl is not None else "active"
        )

        # ── ratio: point estimate over the union population ────────────
        if isinstance(m, RatioMetric):
            if pop_tbl is not None:
                pop = period_population(
                    num_tbl,
                    grain=grain,
                    week_start=week_start,
                    start=start,
                    end_edge=edge,
                    periods=periods,
                    population=pop_tbl,
                    day_boundary_offset=offset,
                )
            else:
                # Units active on either source: a denominator-only unit
                # must not be dropped from the join.
                pop = (
                    period_population(
                        num_tbl,
                        grain=grain,
                        week_start=week_start,
                        start=start,
                        end_edge=edge,
                        periods=periods,
                        day_boundary_offset=offset,
                    )
                    .union(
                        period_population(
                            den_tbl,
                            grain=grain,
                            week_start=week_start,
                            start=start,
                            end_edge=edge,
                            periods=periods,
                            day_boundary_offset=offset,
                        )
                    )
                    .distinct()
                )
            num_e = metric_events(num_tbl, m, value_column=num_fact.column, part="numerator")
            den_e = metric_events(den_tbl, m, value_column=den_fact.column, part="denominator")
            num = period_unit_values(
                unit_day_stats(num_e, source_key=num_fs.name, day_boundary_offset=offset),
                aggregation=m.numerator.aggregation,
                grain=grain,
                week_start=week_start,
                start=start,
                end_edge=edge,
            )
            den = period_unit_values(
                unit_day_stats(den_e, source_key=den_fs.name, day_boundary_offset=offset),
                aggregation=m.denominator.aggregation,
                grain=grain,
                week_start=week_start,
                start=start,
                end_edge=edge,
            )
            out = ratio_period_values(num, den, pop, periods)
            out = out.mutate(
                metric=ibis.literal(name),
                grain=ibis.literal(grain),
                window=window_lit,
                ci_lb=null_f64,
                ci_ub=null_f64,
            )
            return MetricTrend(
                self._finalize(out, []),
                metric=name,
                grain=grain,
                window=window,
                week_start=week_start,
                denominator=denominator,
                start=start,
                end=end,
                by=by,
                alpha=alpha,
            )

        # ── mean / conversion: centered moments -> derived stats ───────

        # RetentionMetric is refused earlier; MeanMetric narrowing keeps
        # `.aggregation` type-checked without an m.type string compare.
        is_conversion = not isinstance(m, MeanMetric)
        aggregation = m.aggregation if isinstance(m, MeanMetric) else "conversion"
        events = metric_events(tbl, m, value_column=None if is_conversion else fact_def.column)
        values = period_unit_values(
            unit_day_stats(events, source_key=fs.name, day_boundary_offset=offset),
            aggregation=aggregation,
            grain=grain,
            week_start=week_start,
            start=start,
            end_edge=edge,
        )
        pop = period_population(
            tbl,
            grain=grain,
            week_start=week_start,
            start=start,
            end_edge=edge,
            periods=periods,
            population=pop_tbl,
            day_boundary_offset=offset,
        )
        by_table = None
        dims = self._resolve_dims(fs, by)
        if dims:
            dim_name, dim_col = dims[0]
            prop_events = tbl.select("unit_id", "ts", **{dim_name: tbl[dim_col]})
            by_table = asof_property_table(
                prop_events, periods, dim=dim_name, day_boundary_offset=offset
            )
        moments = period_moments(
            values,
            pop,
            periods,
            zero_fill=aggregation in _ZERO_FILL_AGGS,
            by=by_table,
        )
        n_pos = moments.n.nullif(0).cast("float64")
        # First moment is exactly recoverable: sum(y) == n*ref_y + cy1.
        mean = moments.ref_y + moments.cy1 / n_pos
        if is_conversion:
            # Wilson: honest at p-hat in {0, 1} and bounded to [0, 1].
            denom_w = 1 + z * z / n_pos
            center = (mean + z * z / (2 * n_pos)) / denom_w
            half = z * (mean * (1 - mean) / n_pos + z * z / (4 * n_pos * n_pos)).sqrt() / denom_w
            ci_lb, ci_ub = center - half, center + half
        else:
            # ddof=1 variance, centered form: S(y-ybar)**2 == cy2 - cy1**2/n
            # (cy1 ~ 0); clamped at zero since a constant period can round negative.
            ss = ibis.greatest(moments.cy2 - moments.cy1 * moments.cy1 / n_pos, 0.0)
            var = ss / ((moments.n - 1).nullif(0).cast("float64"))
            se = (var / n_pos).sqrt()
            ci_lb, ci_ub = mean - z * se, mean + z * se
        out = moments.mutate(
            metric=ibis.literal(name),
            grain=ibis.literal(grain),
            window=window_lit,
            value=mean,
            ci_lb=ci_lb,
            ci_ub=ci_ub,
        )
        return MetricTrend(
            self._finalize(out, [d for (d, _) in dims]),
            metric=name,
            grain=grain,
            window=window,
            week_start=week_start,
            denominator=denominator,
            start=start,
            end=end,
            by=by,
            alpha=alpha,
        )
