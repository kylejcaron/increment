"""``PARITY_CASES``: the parity harness's public, appendable contract.

Downstream lanes append a ``ParityCase`` here in the same change that lands
a new capability (AGENTS.md "Ingress paths and method scope": "Every
capability adds a row to the enumerated parity harness"). Do not rename
``ParityCase``/``ParityDataset`` fields; other lanes' plans reference them
by name.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pandas as pd
import pyarrow.parquet as pq
import pytest

from increment import SequentialCell
from increment._analysis_config import UNSET, _Unset
from increment.analysis import Analysis
from increment.errors import CodedError, IncrementWarning
from increment.estimation.inference import Normal, Prior
from increment.estimation.priors import MixturePrior, StudentTPrior
from increment.frame import MetricSpec
from increment.plan import bind_automatic_sequential_plan
from increment.query.artifact_contract import unit_day_artifact_extension_catalog
from increment.query.session import WarehouseArtifactStore
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    Randomized,
    UptakeSpec,
)
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    ExperimentMetric,
    InferenceSpec,
    NormalPriorSpec,
)
from increment.semantics.sequential import SequentialCompliancePolicy
from increment.sequential_source import native_observation_mapping
from tests.analysis_factory import (
    _moment_source,
    _native_source,
    make_analysis,
    make_analysis_like,
    native_connection,
)
from tests.estimation._binomial_endpoint_reference import assert_set_contains_finer_reference

from . import dataset as ds

CONSTRUCTORS = (
    "from_definitions",
    "from_unit_day_artifact",
    "from_unit_summary",
    "from_unit_panel",
    "from_switchback_panel",
    "from_moments",
)

# Cluster identity is required by clustered cases and a no-op elsewhere.
_ARTIFACT_EXTENSION_KINDS = (
    "cuped_preperiod",
    "breakout_dimension",
    "cluster_identity",
    "site_volume",
)

# Every case waives from_switchback_panel, which needs a switchback-shaped
# Randomized frame no case here builds; `runner.assert_parity` requires the reason.
_SWITCHBACK_WAIVE = {
    "from_switchback_panel": (
        "SOURCE: from_switchback_panel needs a switchback-shaped frame "
        "(unit/cycle/period/step columns) and only accepts a Randomized "
        "identification -- no case in this module builds that shape."
    )
}


@dataclass(frozen=True)
class ParityDataset:
    """One case's own copy of the shared per-unit(-day) truth."""

    event_rows: list[dict[str, Any]]
    definitions: dict[str, Any]


@dataclass(frozen=True)
class Absence:
    """The exception a constructor raises because its signature or schema cannot express the
    request, and the unsupported field or keyword that exception must name."""

    error: type[Exception]
    field: str


@dataclass(frozen=True)
class ParityCase:
    """One capability, driven through every applicable constructor.

    ``build`` maps a constructor name (see ``CONSTRUCTORS``) to a zero-arg
    builder returning an ``Analysis``. Every one of the six constructor
    names MUST appear in ``build`` or in ``waive`` -- ``runner.assert_parity``
    enforces ``set(build) | set(waive) == set(CONSTRUCTORS)`` and fails a
    case that leaves any constructor unaccounted for. A name in ``waive``
    but ABSENT from ``build`` is not attempted at all (a SOURCE reason
    only, no code -- e.g. a switchback-only constructor for a non-
    switchback dataset). A name present in BOTH ``build`` and ``waive`` is
    attempted and MUST either raise a ``CodedError`` whose ``.code`` equals
    ``waived_refusal_codes[name]`` or, when listed in ``expected_absence``, fail to
    construct as described below; the runner asserts this. A name in
    ``waived_refusal_codes`` without a matching ``build`` entry, or a name
    in ``build`` with a reason-only ``waive`` entry and no code or absence, is a
    contract error the runner also rejects.

    ``view`` reads one day-axis method (``run_daily``, ``run_daily_lift``,
    ``run_asof`` or ``run_asof_lift``) instead of ``run``/``run_breakout``;
    ``breakout_dimension`` then names the day-axis ``dimension``. ``refusal_only`` lets a
    case in which EVERY attempted ingress raises its recorded code, or is declared absent,
    pass (no ingress is compared). A name in ``expected_absence`` is attempted and MUST
    raise exactly that ``Absence.error`` type naming ``Absence.field``: the constructor
    signature or schema cannot express the request (an unsupported keyword is a
    ``TypeError``, an unsupported schema field a validation error), so no refusal code
    exists to record.
    """

    id: str
    build: Mapping[str, Callable[[], Analysis]]
    estimands: tuple[str, ...] | None = None
    metrics: tuple[str, ...] | None = None
    breakout_dimension: str | None = None
    sequential: bool = False
    # Optional end-to-end controls for a sequential case.  The runner invokes
    # this after ordinary constructor parity; it is deliberately separate from
    # the native/calendar builders so structured-label probes cannot change the
    # public row comparison.
    sequential_probe: Callable[[], None] | None = None
    waive: Mapping[str, str] = field(default_factory=dict)
    waived_refusal_codes: Mapping[str, str] = field(default_factory=dict)
    slow: bool = False
    prior: Prior | None | _Unset = UNSET
    # A family case whose paths could all select nothing and still agree must
    # show at least one selected and one unselected row.
    require_selection: bool = False
    readout_probe: Callable[[Any], None] | None = None
    source_probe: Callable[[str, Analysis], None] | None = None
    expected_warning_codes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    view: Literal["daily", "daily_lift", "asof", "asof_lift"] | None = None
    refusal_only: bool = False
    expected_absence: Mapping[str, Absence] = field(default_factory=dict)


class _Warehouse(NamedTuple):
    """The warehouse a case's native and artifact paths run on."""

    connect: Callable[[list[dict[str, Any]]], Any]
    dialect: str


_DUCKDB = _Warehouse(ds.duckdb_connection, "duckdb")


def _track_connection(analysis: Analysis, con: Any) -> Analysis:
    """Record `con` on `analysis` so `runner.run_case`'s `finally` block
    disconnects it once this constructor's run is done. `Analysis.close()`
    deliberately leaves caller-owned connections open (its own docstring:
    "without closing caller connections"), and every builder in this
    module owns the connection it opens -- the builder is the one place
    that knows the connection is no longer needed once its Analysis's
    lifetime ends, so it records that here rather than leaving it to leak
    until the process exits. Multiple connections can stack (a case with
    more than one still-open connection per Analysis is not expected
    today, but nothing here assumes exactly one)."""
    existing = getattr(analysis, "_parity_connections", ())
    analysis._parity_connections = (*existing, con)  # ty: ignore[unresolved-attribute]
    return analysis


def _close_parity_analysis(analysis: Analysis) -> None:
    """Close readers, erase owned generations and receipts, then disconnect."""
    from integration.warehouse_execution._suite import _drop_probe_generation

    with ExitStack() as cleanup:
        for connection in getattr(analysis, "_parity_connections", ()):
            cleanup.callback(connection.disconnect)
        for connection, store, ref in getattr(analysis, "_parity_artifacts", ()):
            cleanup.callback(
                _drop_probe_generation,
                connection,
                store,
                ref,
                connection.name,
                schema_name="artifacts",
            )
        cleanup.callback(analysis.close)


def _publish_and_adopt(
    con: Any, native: Analysis, kinds: tuple[str, ...] = _ARTIFACT_EXTENSION_KINDS
) -> Analysis:
    """Publish `native`'s unit-day artifact and adopt it as a fresh
    Analysis. `native` is a throwaway intermediate that exists only to
    publish -- closed here, right after the adopted Analysis exists, so it
    is never left dangling for the runner to notice. `con` is NOT
    closeable yet: the adopted Analysis reads from `con`-backed tables for
    the rest of its life (through every later `run()`/`run_breakout()`/
    `capture_sequential()` call), so it is tracked via `_track_connection`
    for `runner.run_case`'s `finally` block to disconnect once that
    lifetime ends.
    """
    context = native._artifact_context(
        native._defs,  # ty: ignore[invalid-argument-type]
        native.experiment,
        native._on_mixed_assignment,
        design=native._design,
    )
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    requests = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in kinds
    ]
    _track_connection(native, con)
    try:
        ref = native.publish_unit_day_artifact(store, extensions=requests)
        native._parity_artifacts = ((con, store, ref),)  # ty: ignore[unresolved-attribute]
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        adopted._parity_artifacts = ((con, store, ref),)  # ty: ignore[unresolved-attribute]
        native.close()
        return _track_connection(adopted, con)
    except BaseException:
        _close_parity_analysis(native)
        raise


def _export_and_replay(
    analysis: Analysis, metrics: list[MetricSpec], *, plan: AnalysisPlan | None = None
) -> Analysis:
    """Export `analysis`'s moments and replay them through `from_moments`.
    `analysis` (the caller's already-built source, e.g. a from_unit_summary
    Analysis) is a throwaway intermediate once its export completes --
    closed here, right after the derived from_moments Analysis is built,
    rather than left for the caller to remember."""
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            analysis.export(path)
            rows = pq.read_table(path).to_pylist()
        return Analysis.from_moments(rows, control="control", metrics=metrics, plan=plan)
    finally:
        _close_parity_analysis(analysis)


# Definitions/artifact put the CUPED override on the plan's ExperimentMetric;
# the frame path refuses plan-level sensitivity_methods overrides and takes it on
# the MetricSpec. Both plans share role/q/alpha; dataset.py's docstring explains
# why they are separate objects with matching rows.
_DEFINITIONS_PLAN = AnalysisPlan(
    q=0.10,
    secondaries=(
        "revenue",
        "purchase_rate",
        "rps",
        ExperimentMetric(
            metric="revenue_cuped",
            sensitivity_methods=({"name": "cuped", "variance_reduction": "cuped"},),
        ),
    ),
)
_FRAME_PLAN = AnalysisPlan(q=0.10, secondaries=("revenue", "purchase_rate", "rps", "revenue_cuped"))
# CUPED-free plans for `_breakout_case` (`include_cuped=False`): breakout is
# orthogonal to CUPED, so dropping the covariate metric lets from_unit_panel's
# BH-breakout row be compared with the other four paths rather than waived.
_DEFINITIONS_PLAN_NO_CUPED = AnalysisPlan(q=0.10, secondaries=("revenue", "purchase_rate", "rps"))
_FRAME_PLAN_NO_CUPED = AnalysisPlan(q=0.10, secondaries=("revenue", "purchase_rate", "rps"))


def _fixed_horizon_plans(
    include_cuped: bool, ratio_cuped: bool
) -> tuple[AnalysisPlan, AnalysisPlan]:
    if not include_cuped:
        return _DEFINITIONS_PLAN_NO_CUPED, _FRAME_PLAN_NO_CUPED
    if not ratio_cuped:
        return _DEFINITIONS_PLAN, _FRAME_PLAN
    ratio = ExperimentMetric(
        metric="rps_cuped", sensitivity_methods=({"name": "cuped", "variance_reduction": "cuped"},)
    )
    return (
        _DEFINITIONS_PLAN.model_copy(
            update={"secondaries": (*(_DEFINITIONS_PLAN.secondaries or ()), ratio)}
        ),
        _FRAME_PLAN.model_copy(
            update={"secondaries": (*(_FRAME_PLAN.secondaries or ()), "rps_cuped")}
        ),
    )


def _ratio_cuped_parity_spec(*, replay: bool = False) -> MetricSpec:
    return MetricSpec(
        name="rps_cuped",
        type="ratio",
        numerator="revenue",
        denominator="sessions",
        covariate=None if replay else "pre_revenue",
        missing="zero",
        preferred_direction="increase",
        sensitivity_methods=() if replay else ({"name": "cuped", "variance_reduction": "cuped"},),
    )


def _rounded_event_rows_for_parity() -> list[dict[str, Any]]:
    return [
        {**row, "revenue": 0.1} if rounded else row
        for row in ds.event_rows()
        for rounded in (
            [True] * (10 if int(row["user_id"][1:]) % 2 else 100)
            if row["event"] == "purchase" and row["revenue"] and row["event_at"].day == 10
            else [False]
        )
    ]


def _covariate_rescaled_event_rows() -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in ds.event_rows()
        for _ in range(1 + int(row["user_id"][1:]) % 4 if row["sess"] == 1 else 1)
    ]
    for row in rows:
        if row["revenue"] is not None:
            row["revenue"] *= 1e150 if row["event_at"] < ds._EXPOSURE_AT else 1e-50
    return rows


def _publish_rounded_artifact(con: Any, native: Analysis) -> Analysis:
    from increment.estimation.sitewide import SitewideImpact

    expected = native.sitewide("revenue")
    adopted = _publish_and_adopt(con, native)
    actual = adopted.sitewide("revenue")
    assert isinstance(expected, SitewideImpact) and isinstance(actual, SitewideImpact)
    assert expected.site_total_volume > 0
    assert actual.site_total_volume == pytest.approx(expected.site_total_volume)
    assert actual.delta == pytest.approx(expected.delta)
    return adopted


def _neighboring_ratio_event_rows() -> list[dict[str, Any]]:
    rows = [
        row
        for row in ds.event_rows()
        if not (row["event"] == "purchase" and row["event_at"] == ds._PURCHASE_AT)
    ]
    for index in range(len(rows)):
        row = rows[index]
        if row["event"] == "exposure":
            noise = (-2.0, -1.0, 1.0, 2.0)[int(row["user_id"][1:]) % 4]
            rows.append(
                ds._row(
                    row["user_id"],
                    ds._PURCHASE_AT,
                    "purchase",
                    store_id=row["store_id"],
                    revenue=1e15 + noise + (row["group_id"] == "treatment"),
                )
            )
    return rows


def _assert_neighboring_ratio_rows(results: Any) -> None:
    rows = {(row.metric, row.method): row for row in results}
    for (metric, method), row in rows.items():
        if metric not in ("rps", "rps_cuped"):
            continue
        lift = row.require_lift()
        assert lift.value == pytest.approx(1e-15, rel=1e-12, abs=0)
        mean = rows["revenue" if metric == "rps" else "revenue_cuped", method]
        for field_name in ("value", "lb", "ub", "log_se"):
            assert getattr(lift, field_name) == pytest.approx(
                getattr(mean.require_lift(), field_name), rel=1e-10, abs=0
            )
        assert row.stat_sig() == mean.stat_sig()


def _fixed_horizon_frame_metrics(
    *,
    include_cuped: bool,
    unselected_winsorized_sibling: bool,
    clear_bound_prior: bool,
    ratio_numerics: Literal["rescaled", "neighboring"] | None,
) -> list[MetricSpec]:
    """The frame constructors' metric set for `_fixed_horizon_case`."""
    metrics = [
        MetricSpec(
            name="revenue",
            type="mean",
            missing="zero",
            preferred_direction="increase",
            prior=Normal(mu=0.0, sigma=0.1) if clear_bound_prior else None,
        ),
        MetricSpec(
            name="purchase_rate",
            type="conversion",
            value_column="converted",
            preferred_direction="increase",
        ),
        MetricSpec(
            name="rps",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            missing="zero",
            preferred_direction="increase",
        ),
        *([_ratio_cuped_parity_spec()] if ratio_numerics is not None else []),
    ]
    if unselected_winsorized_sibling:
        metrics.append(
            MetricSpec(
                name="capped_revenue",
                value_column="revenue",
                missing="zero",
                winsorization={"upper_value": 12.0},
                preferred_direction="increase",
            )
        )
    if include_cuped:
        metrics.append(
            MetricSpec(
                name="revenue_cuped",
                type="mean",
                value_column="revenue",
                covariate="pre_revenue",
                missing="zero",
                preferred_direction="increase",
                sensitivity_methods=({"name": "cuped", "variance_reduction": "cuped"},),
            )
        )

    return metrics


@dataclass(frozen=True)
class _Fixture:
    """Optional fixture variations for `_fixed_horizon_case`.

    ``boolean_store`` makes ``store`` a genuine BOOLEAN warehouse column
    (True/False/NULL) declared ``dtype: bool``, so the native path, the published
    artifact and the nullable Boolean panel column all read out
    ``true``/``false``/``__null__``; a string-typed column could not observe the
    artifact's Boolean label spelling. ``window_by_route`` re-spells the declared
    experiment start/end per warehouse route at ``day_boundary`` UTC-05:00 and adds
    units exposed at each window edge.
    """

    boolean_store: bool = False
    window_by_route: Mapping[str, tuple[str, str]] | None = None

    def declare_store_dtype(self, defs_dict: dict[str, Any]) -> None:
        if self.boolean_store:
            store_property = next(
                prop
                for prop in defs_dict["fact_sources"][0]["properties"]
                if prop["name"] == "store"
            )
            store_property["dtype"] = "bool"

    def definitions(self, defs_dict: dict[str, Any], route: str) -> Definitions:
        payload = copy.deepcopy(defs_dict)
        if self.window_by_route is not None:
            start, end = self.window_by_route[route]
            payload["experiments"][0].update(start=start, end=end, day_boundary="UTC-05:00")
        return Definitions.model_validate(payload)


_NO_FIXTURE = _Fixture()


def _fixture_rows(rows: list[dict[str, Any]], fixture: _Fixture) -> list[dict[str, Any]]:
    if fixture.boolean_store:
        rows = [{**row, "store_id": _BOOLEAN_STORE_VALUE[row["store_id"]]} for row in rows]
    if fixture.window_by_route is not None:
        rows = [*rows, *_window_edge_rows()]
    return rows


def _fixed_horizon_case(
    *,
    id: str,
    breakout: bool,
    waive: Mapping[str, str],
    waived_refusal_codes: Mapping[str, str],
    include_cuped: bool = True,
    unselected_winsorized_sibling: bool = False,
    clear_bound_prior: bool = False,
    rounded_events: bool = False,
    ratio_numerics: Literal["rescaled", "neighboring"] | None = None,
    prior: Prior | None | _Unset = UNSET,
    warehouse: _Warehouse = _DUCKDB,
    fixture: _Fixture = _NO_FIXTURE,
) -> ParityCase:
    """Mean/ratio/conversion(/CUPED-mean, when *include_cuped*), fixed
    horizon, under a real two-secondary(-plus-CUPED-sensitivity) plan
    (q=0.10): supported on from_definitions, from_unit_day_artifact,
    from_unit_summary, from_unit_panel and from_moments today -- the
    CUPED covariate is a genuine pre-period value replicated across a
    unit's panel rows, so from_unit_panel resolves it the same way
    from_unit_summary's frame already carries it. `_breakout_case` passes
    `include_cuped=False`: the breakout capability itself has nothing to
    do with CUPED, and dropping the covariate metric keeps that case's
    row set focused on breakouts alone.
    """
    rows = (
        _neighboring_ratio_event_rows()
        if ratio_numerics == "neighboring"
        else _rounded_event_rows_for_parity()
        if rounded_events
        else _covariate_rescaled_event_rows()
        if ratio_numerics == "rescaled"
        else ds.event_rows()
    )
    rows = _fixture_rows(rows, fixture)
    definitions_plan, frame_plan = _fixed_horizon_plans(include_cuped, ratio_numerics is not None)
    if clear_bound_prior:
        treated_units = {row["user_id"] for row in rows if row["group_id"] == "treatment"}
        for row in rows:
            if row["user_id"] in treated_units and row["revenue"] is not None:
                row["revenue"] *= 2
        bindings = definitions_plan.secondaries
        assert bindings is not None
        definitions_plan = definitions_plan.model_copy(
            update={
                "secondaries": tuple(
                    ExperimentMetric(metric="revenue", prior=NormalPriorSpec(mu=0.0, sigma=0.1))
                    if binding == "revenue"
                    else binding
                    for binding in bindings
                )
            }
        )
    if unselected_winsorized_sibling:
        definitions_plan = frame_plan = AnalysisPlan(primary="revenue")
    defs_dict = ds.definitions_dict(plan=definitions_plan, breakout=breakout)
    fixture.declare_store_dtype(defs_dict)
    defs_dict["dialect"] = warehouse.dialect
    defs_dict["metrics"].extend(
        [
            {
                **next(metric for metric in defs_dict["metrics"] if metric["name"] == "rps"),
                "name": "rps_cuped",
            }
        ]
        if ratio_numerics is not None
        else []
    )
    if unselected_winsorized_sibling:
        revenue = next(metric for metric in defs_dict["metrics"] if metric["name"] == "revenue")
        defs_dict["metrics"].append(
            {**revenue, "name": "capped_revenue", "winsorization": {"upper_value": 12.0}}
        )

    def route_defs(route: str) -> Definitions:
        return fixture.definitions(defs_dict, route)

    def build_definitions() -> Analysis:
        con = warehouse.connect(rows)
        analysis = make_analysis(con, route_defs("from_definitions"), experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse.connect(rows)
        native = make_analysis(con, route_defs("from_unit_day_artifact"), experiment="exp")
        return (_publish_rounded_artifact if rounded_events else _publish_and_adopt)(con, native)

    summary_metrics = _fixed_horizon_frame_metrics(
        include_cuped=include_cuped,
        unselected_winsorized_sibling=unselected_winsorized_sibling,
        clear_bound_prior=clear_bound_prior,
        ratio_numerics=ratio_numerics,
    )

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.unit_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=summary_metrics,
            plan=frame_plan,
        )

    panel_metrics = _fixed_horizon_frame_metrics(
        include_cuped=include_cuped,
        unselected_winsorized_sibling=unselected_winsorized_sibling,
        clear_bound_prior=clear_bound_prior,
        ratio_numerics=ratio_numerics,
    )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.unit_summary_frame(con)
        con.disconnect()  # `summary`/`panel` are already materialized below
        panel = ds.unit_panel_frame(summary)
        if fixture.boolean_store:
            panel["store"] = pd.array(panel["store"].tolist(), dtype="boolean")
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            control="control",
            metrics=panel_metrics,
            plan=frame_plan,
            breakouts=("store",) if breakout else (),
        )

    def build_moments() -> Analysis:
        replay_metrics = [
            MetricSpec(
                name="revenue",
                type="mean",
                preferred_direction="increase",
                prior=Normal(mu=0.0, sigma=0.1) if clear_bound_prior else None,
            ),
            MetricSpec(name="purchase_rate", type="conversion", preferred_direction="increase"),
            MetricSpec(
                name="rps",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
                preferred_direction="increase",
            ),
            *([_ratio_cuped_parity_spec(replay=True)] if ratio_numerics is not None else []),
        ]
        if include_cuped:
            replay_metrics.append(
                MetricSpec(name="revenue_cuped", type="mean", preferred_direction="increase")
            )
        if unselected_winsorized_sibling:
            replay_metrics = [metric for metric in replay_metrics if metric.name == "revenue"]
        if clear_bound_prior:
            native = build_definitions()
            try:
                return _export_and_replay(native, replay_metrics)
            finally:
                native.close()
                for connection in getattr(native, "_parity_connections", ()):
                    connection.disconnect()
        con = ds.duckdb_connection(rows)
        frame = ds.unit_summary_frame(con)
        con.disconnect()
        summary = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=summary_metrics,
            plan=frame_plan,
        )
        return _export_and_replay(summary, replay_metrics)

    return ParityCase(
        id=id,
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        metrics=("revenue",) if unselected_winsorized_sibling else None,
        breakout_dimension="store" if breakout else None,
        waive={**waive, **_SWITCHBACK_WAIVE},
        waived_refusal_codes=waived_refusal_codes,
        prior=None if clear_bound_prior else prior,
        require_selection=clear_bound_prior,
        readout_probe=_assert_neighboring_ratio_rows if ratio_numerics == "neighboring" else None,
        # Publishing, adopting and replaying an artifact on every run stays under
        # 2s, far above this module's ~40ms ordinary-case budget.
        slow=True,
    )


def _mean_ratio_conversion_cuped_case() -> ParityCase:
    return _fixed_horizon_case(
        id="mean_ratio_conversion_cuped_fixed_horizon",
        breakout=False,
        waive={},
        waived_refusal_codes={},
    )


def _dashboard_group_data_case() -> ParityCase:
    def probe(path: str, analysis: Analysis) -> None:
        if path != "from_definitions":
            with pytest.raises(CodedError) as caught:
                analysis.dashboard_group_data(metrics=analysis.metrics)
            assert caught.value.code == "facade.analysis.operation"
            return
        connection = ds.duckdb_connection(ds.event_rows())
        try:
            units = ds.unit_summary_frame(connection).to_pylist()
        finally:
            connection.disconnect()
        expected = {}
        for arm in ("control", "treatment"):
            selected = [row for row in units if row["variant"] == arm]
            revenue = sum(row["revenue"] or 0.0 for row in selected)
            values = {
                "revenue": revenue / len(selected),
                "revenue_cuped": revenue / len(selected),
                "purchase_rate": sum(row["converted"] for row in selected) / len(selected),
                "rps": revenue / sum(row["sessions"] for row in selected),
            }
            expected.update(
                ((metric, arm), (value, len(selected))) for metric, value in values.items()
            )
        observed = analysis.dashboard_group_data(metrics=analysis.metrics)
        assert {(row.metric, row.group_id) for row in observed} == set(expected)
        for row in observed:
            value, count = expected[row.metric, row.group_id]
            assert row.observed_value == pytest.approx(value, rel=1e-9, abs=1e-9)
            assert row.assigned_units == row.eligible_units == count

    return replace(
        _mean_ratio_conversion_cuped_case(),
        id="dashboard-group-data",
        source_probe=probe,
    )


def _prior_reset_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    return _fixed_horizon_case(
        id="audit-prior-reset-family",
        breakout=False,
        waive={},
        waived_refusal_codes={},
        include_cuped=False,
        clear_bound_prior=True,
        warehouse=_Warehouse(warehouse_connection, dialect),
    )


def _lift_prior_case(
    kind: Literal["normal", "student_t", "mixture"],
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    priors: dict[str, Prior] = {
        "normal": Normal(mu=0.1, sigma=0.05),
        "student_t": StudentTPrior(nu=4, scale=0.05),
        "mixture": MixturePrior(weights=(0.7, 0.3), means=(0.0, 0.2), sigmas=(0.03, 0.1)),
    }
    return _fixed_horizon_case(
        id=f"lift_prior_{kind}_mean_ratio_conversion_cuped",
        breakout=False,
        waive={},
        waived_refusal_codes={},
        prior=priors[kind],
        warehouse=_Warehouse(warehouse_connection, dialect),
    )


def _breakout_case(*, boolean: bool = False) -> ParityCase:
    """Same three metrics and plan (no CUPED -- see `include_cuped=False`
    below), plus a declared `store` breakout (3 segments, both arms
    present in every segment). Supported on from_definitions,
    from_unit_day_artifact and from_unit_panel today. from_unit_summary
    has no `breakouts=` construction parameter at all (SOURCE cannot
    supply a declared breakout catalog on a one-row-per-unit frame) and
    from_moments explicitly refuses run_breakout by its own docstring;
    both raise `facade.analysis.operation`.

    from_unit_panel used to be waived here too, blamed on the shared
    metric set's `revenue_cuped` covariate (from_unit_panel does not yet
    support CUPED at all -- `_mean_ratio_conversion_cuped_case` waives the
    identical construction gap) -- an unrelated fixture choice, not a
    breakout limitation: `run_breakout` names `from_unit_panel` as one of
    only two constructors it supports at all
    (`_breakout_readouts.py`), and `test_analysis_breakout.py`'s own
    `test_run_breakout_bh_family_survives_*` tests already run BH-corrected
    breakouts through it. `include_cuped=False` drops the covariate metric
    from every constructor's plan/metric set (not just from_unit_panel's),
    so all five attempted paths emit the identical (metric, arm, segment)
    row set and from_unit_panel's own breakout row is genuinely compared,
    not waived.

    An earlier draft of this case mis-attributed the refusal to breakout
    sample size and waived it as "deferred" without ever building
    `build_unit_panel` to check; that draft also carried a real bug in
    `event_rows`' original store formula (`f"s{i % n_stores}"` with
    `n_stores=3` exactly aliased the exact-binomial skip condition `i % 3
    != 0`, so segment `s0`'s `revenue`/`rps` mean was identically zero in
    every arm -- confirmed live, and confirmed still failing at 4x this
    seed's `n_per_arm`, so no fixture size fixes it) -- corrected now:
    `event_rows` decorrelates store from the skip condition.

    ``boolean=True`` makes ``store`` a genuine BOOLEAN warehouse column (declared
    ``dtype: bool``) feeding the native path and the published artifact, and a
    nullable Boolean column on the panel frame: native DuckDB is the oracle, every
    path reads out as ``true``/``false``/``__null__``, and the same source
    refusals apply.
    """
    case = _fixed_horizon_case(
        id="breakout_boolean_store_dimension" if boolean else "breakout_store_dimension",
        breakout=True,
        include_cuped=False,
        fixture=_Fixture(boolean_store=boolean),
        waive={
            "from_unit_summary": (
                "SOURCE: Analysis.from_unit_summary has no breakouts= constructor "
                "parameter, so it carries no declared Breakout catalog for "
                "run_breakout to read; run_breakout raises facade.analysis.operation."
            ),
            "from_moments": (
                "SOURCE: Analysis.from_moments's own docstring names run_breakout "
                "among the methods it refuses -- a moments cube enters at moment "
                "grain and carries no per-unit breakout dimension."
            ),
        },
        waived_refusal_codes={
            "from_unit_summary": "facade.analysis.operation",
            "from_moments": "facade.analysis.operation",
        },
    )
    return case


def _dashboard_breakout_reads_case() -> ParityCase:
    from increment.semantics.models import Breakout

    def outcome(read: Callable[[], Any]) -> Any:
        try:
            rows = read()
        except CodedError as exc:
            return exc.code
        return sorted(row.model_dump_json() for row in rows)

    def probe(path: str, analysis: Analysis) -> None:
        if path != "from_definitions":
            with pytest.raises(CodedError) as caught:
                analysis.dashboard_breakout_reads(Breakout(property="store"))
            assert caught.value.code == "facade.analysis.operation"
            return
        (declared,) = analysis.experiment.breakouts
        with pytest.raises(CodedError) as caught:
            analysis.dashboard_breakout_reads(declared.model_copy(update={"property": "nope"}))
        assert caught.value.code == "facade.analysis.undeclared_breakout"
        reads = analysis.dashboard_breakout_reads(declared)
        names = [metric.name for metric in analysis.metrics]
        dimension = declared.property
        observed = []
        for scoped, whole in (
            (
                lambda: reads.run_asof_lift(metrics=names),
                lambda: analysis.run_asof_lift(metrics=names, dimension=dimension),
            ),
            (
                lambda: reads.run_asof(metrics=names, completed_windows_only=True),
                lambda: analysis.run_asof(
                    metrics=names, completed_windows_only=True, dimension=dimension
                ),
            ),
            (
                lambda: reads.run_daily(metrics=names),
                lambda: analysis.run_daily(metrics=names, dimension=dimension),
            ),
            (
                lambda: reads.run_breakout(metrics=names),
                lambda: analysis.run_breakout(metrics=names),
            ),
        ):
            observed.append(outcome(scoped))
            assert observed[-1] == outcome(whole)
        assert all(isinstance(rows, list) and rows for rows in observed), observed

    return replace(_breakout_case(), id="dashboard-breakout-reads", source_probe=probe)


def _window_edge_rows() -> list[dict[str, Any]]:
    """Units exposed just inside each edge of the declared window, in the
    UTC-05:00 day the window names: 2025-01-10T03:00Z is 2025-01-09 22:00 local
    (the first window day) and 2025-01-20T02:00Z is 2025-01-19 21:00 local (the
    last). A window day read from the spelling's leading date instead of the
    instant at the boundary would drop or admit exactly these units."""
    rows: list[dict[str, Any]] = []
    for edge, exposed_at in (
        ("first", dt.datetime(2025, 1, 10, 3)),
        ("last", dt.datetime(2025, 1, 20, 2)),
    ):
        for arm, offset in (("control", 0.0), ("treatment", 1.0)):
            for i in range(8):
                unit = f"edge-{edge}-{arm}-{i}"
                rows.append(
                    ds._row(unit, exposed_at, "exposure", group_id=arm, experiment_id="exp")
                )
                rows.append(
                    ds._row(
                        unit,
                        exposed_at + dt.timedelta(hours=1),
                        "purchase",
                        revenue=2.0 + i + offset,
                    )
                )
                rows.append(
                    ds._row(unit, exposed_at + dt.timedelta(hours=2), "session_end", sess=1)
                )
    return rows


def _window_spelling_case(*, native_spelling: str) -> ParityCase:
    """One declared instant spelled two ways is one experiment window.

    The window is declared at ``day_boundary`` UTC-05:00 as either
    ``2025-01-10T00:00:00Z`` (``zulu``) or ``2025-01-09T19:00:00-05:00``
    (``offset``). The artifact route always reads the ``offset`` spelling;
    ``from_definitions`` reads *native_spelling*. The two rows (native ``zulu``
    and native ``offset``) together pin native(zulu) == artifact(offset) ==
    native(offset): emitted rows, arm spine and estimates agree across both
    routes and both spellings, and any spelling-dependent window day would
    break one of them.

    The frame constructors and ``from_moments`` carry no window lever
    (``tests/test_frame_window_parity.py``: a non-UTC boundary is the
    caller's pre-bucketing of ``ds``/``exposure_date``, and a moments cube has
    no experiment window), and ``from_switchback_panel`` needs a switchback
    frame, so none is attempted.
    """
    spellings = {
        "zulu": ("2025-01-10T00:00:00Z", "2025-01-20T00:00:00Z"),
        "offset": ("2025-01-09T19:00:00-05:00", "2025-01-19T19:00:00-05:00"),
    }
    case = _fixed_horizon_case(
        id=f"window_spelling_{native_spelling}_vs_offset",
        breakout=False,
        include_cuped=False,
        waive={},
        waived_refusal_codes={},
        fixture=_Fixture(
            window_by_route={
                "from_definitions": spellings[native_spelling],
                "from_unit_day_artifact": spellings["offset"],
            }
        ),
    )
    no_lever = (
        "SOURCE: {name} has no experiment-window lever -- the day boundary is "
        "applied by the caller pre-bucketing ds/exposure_date (see "
        "tests/test_frame_window_parity.py), so the window spelling is not an input."
    )
    edge_units = {
        (edge, arm): {
            row["user_id"]
            for row in _window_edge_rows()
            if row["user_id"].startswith(f"edge-{edge}-{arm}-")
        }
        for edge in ("first", "last")
        for arm in ("control", "treatment")
    }
    edge_days = {"first": dt.date(2025, 1, 9), "last": dt.date(2025, 1, 19)}

    def edges_admitted(name: str, analysis: Analysis) -> None:
        """Two routes that both dropped an edge cohort would still match, so each
        attempted route must show the first- and last-local-day cohorts itself."""
        daily = {
            (row.ds, row.group_id): row.n
            for row in analysis.run_daily(metrics=["revenue"])
            if row.dimension is None
        }
        for (edge, arm), units in edge_units.items():
            assert units, (edge, arm)
            assert daily.get((edge_days[edge], arm)) == len(units), (
                name,
                edge,
                arm,
                daily.get((edge_days[edge], arm)),
            )

    return replace(
        case,
        source_probe=edges_admitted,
        build={
            name: builder
            for name, builder in case.build.items()
            if name in ("from_definitions", "from_unit_day_artifact")
        },
        waive={
            **case.waive,
            "from_unit_summary": no_lever.format(name="from_unit_summary"),
            "from_unit_panel": no_lever.format(name="from_unit_panel"),
            "from_moments": (
                "SOURCE: from_moments replays an exported moment cube; it carries no "
                "declared experiment window to spell."
            ),
        },
    )


def _sequential_structured_order_probe() -> None:
    """Exercise empty-freeze replay and d1/d2/d10 append ordering.

    This is intentionally a probe, not another parity constructor: the public
    parity row continues to use the ordinary/native calendar mapping above.
    """
    from increment import snapshot_from_json
    from increment.sequential_state import declare_sequential_freeze_cells

    rows = ds.event_rows()
    plan = AnalysisPlan(
        primary="revenue",
        inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
    )
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    con = ds.duckdb_connection(rows)
    panel = ds.sequential_unit_panel_frame(con).copy()
    con.disconnect()
    labels = {
        dt.date(2025, 1, 5): "d1",
        dt.date(2025, 1, 10): "d2",
        dt.date(2025, 1, 19): "d10",
    }
    panel["date"] = panel["date"].map(labels)
    # Make d10 a genuine append of newly exposed units.  Existing d2 units
    # therefore retain exactly the same observed rows across the extension;
    # the later label tests ordering rather than a same-unit correction.
    late_units = set(sorted(panel["user_id"].unique())[len(panel["user_id"].unique()) // 2 :])
    panel = panel[(panel["date"] != "d10") | panel["user_id"].isin(late_units)].copy()
    panel["exposure_date"] = panel["user_id"].map(lambda uid: "d10" if uid in late_units else "d2")

    def build(frame: pd.DataFrame) -> Analysis:
        return Analysis.from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    analysis = build(panel)
    try:
        first = analysis.capture_sequential(finalized=True, as_of="d2")
        empty = declare_sequential_freeze_cells(first, ())
        assert empty == first, "an empty requested-cell freeze must preserve the snapshot"
        replayed = snapshot_from_json(empty.model_dump_json())
        assert replayed == first, "empty-freeze JSON replay changed retained rows/state"
        extended = analysis.capture_sequential(finalized=True, previous=replayed, as_of="d10")
        independent = build(panel)
        try:
            fresh = independent.capture_sequential(finalized=True, as_of="d10")
            assert extended.records == fresh.records
            assert extended.states == fresh.states
        finally:
            independent.close()

        mutated = panel.copy()
        target = (
            (mutated["date"] == "d2")
            & ~mutated["user_id"].isin(late_units)
            & mutated["revenue"].notna()
        )
        index = mutated.index[target][0]
        mutated.loc[index, "revenue"] = float(mutated.loc[index, "revenue"]) + 1.0
        rewritten = build(mutated)
        try:
            rewritten.capture_sequential(finalized=True, previous=replayed, as_of="d10")
        except CodedError as exc:
            assert exc.code == "sequential.continuation.rewrite"
        else:
            raise AssertionError("mutating a finalized d2 observation was accepted")
        finally:
            rewritten.close()
    finally:
        analysis.close()


def _sequential_asymptotic_mean_case() -> ParityCase:
    """A plain (non-CUPED) mean under registered asymptotic_mean
    monitoring. Supported and matching, byte-for-byte on the point/interval,
    on all five reachable constructors today, including from_unit_panel:
    `capture_sequential` on a panel source needs an explicit
    `exposure_date=` column AND `MetricSpec.window_days` resolved against
    each row's OWN calendar day, which `dataset.sequential_unit_panel_frame`
    supplies (see its own docstring for why the OTHER cases' `unit_panel_frame`
    -- a synthetic day-0-zero/day-1-real split -- does not work here: it
    produces `note='zero_arm_variance'` because the declared window
    excludes the day actually carrying the value). from_switchback_panel
    has no sequential construction at all (a genuine, permanent SOURCE
    limitation already documented in docs/reference/capabilities-by-entry-point.md), waived, not
    attempted.
    """
    rows = ds.event_rows()
    sequential_plan = AnalysisPlan(
        primary="revenue",
        inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
    )
    defs_dict = ds.definitions_dict(
        plan=sequential_plan, allocation={"control": 0.5, "treatment": 0.5}
    )
    defs = Definitions.model_validate(defs_dict)
    exp = defs.experiment("exp")
    assert exp is not None
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    as_of = ds._EXPERIMENT_END - dt.timedelta(days=1)

    def _native_bound_plan() -> AnalysisPlan:
        # Analysis.__init__ binds an automatic sequential plan's registration
        # before building the source; make_analysis compiles its plan as given and
        # refuses an unbound one (`sequential.registration.invalid`), so bind here.
        assert exp is not None
        bound = bind_automatic_sequential_plan(
            exp.plan,
            [m for m in defs.metrics if m.name in exp.metric_names],
            design=design,
            source_id=exp.name,
            source_mapping=native_observation_mapping(defs, exp, on_mixed_assignment="error"),
            pre_period_covariate=exp.n_pre_periods > 0,
        )
        assert bound is not None
        return bound

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]  # consumed by the runner's capture_sequential call
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        adopted = _publish_and_adopt(con, native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.unit_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            design=design,
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero")],
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()  # `panel` is already materialized; nothing below needs `con` alive
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        analysis._sequential_as_of = dt.date(2025, 1, 11)  # ty: ignore[unresolved-attribute]  # matches _EXPOSURE_AT's day + window_days=1
        return analysis

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        return _export_and_replay(summary, [MetricSpec(name="revenue", type="mean")])

    return ParityCase(
        id="sequential_asymptotic_mean_revenue",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        sequential=True,
        sequential_probe=_sequential_structured_order_probe,
        waive=dict(_SWITCHBACK_WAIVE),
        # Publishes/adopts a unit-day artifact, exports/replays moments,
        # and captures a finalized sequential checkpoint on every
        # constructor -- see _fixed_horizon_case's slow=True for why.
        slow=True,
    )


def _source_horizon_case(*, pinned_dimension: bool = False) -> ParityCase:
    """Match natural panel bounds and pinned dimension freshness to unit totals."""
    start = dt.datetime(2025, 1, 1)
    days = (1,) if pinned_dimension else (0, 9, 10)
    window = 7 if pinned_dimension else 11
    plan = AnalysisPlan(primary="revenue")
    records, events = [], []
    units = []
    for arm, multiplier in (("control", 1), ("treatment", 2)):
        for i in range(3):
            unit = f"{arm}-{i}"
            units.append(unit)
            events.append(ds._row(unit, start, "exposure", group_id=arm, experiment_id="exp"))
            for day in days:
                value = float(multiplier * (i + 1))
                events.append(
                    ds._row(
                        unit, start + dt.timedelta(days=day, hours=12), "purchase", revenue=value
                    )
                )
                records.append(
                    {
                        "user_id": unit,
                        "group_id": arm,
                        "day": f"d{day}",
                        "exposure": "d0",
                        "revenue": value,
                    }
                )
    definition = ds.definitions_dict(plan=plan)
    metric_definition: dict[str, Any] = {
        "name": "revenue",
        "type": "mean",
        "entity": "user_id",
        "fact": "purchase",
        "aggregation": "sum",
        "window_days": window,
        "preferred_direction": "increase",
    }
    definition["metrics"] = [metric_definition]
    definition["experiments"][0].update(
        start=start.isoformat(),
        end=None,
        n_pre_periods=0,
    )
    if pinned_dimension:
        events.append(
            ds._row("outside", start + dt.timedelta(days=9, hours=12), "purchase", revenue=1.0)
        )
        definition["dim_sources"] = [
            {
                "name": "users",
                "sql": "SELECT * FROM users",
                "entity": "user_id",
                "properties": [{"name": "country", "column": "country", "as_of": "static"}],
            }
        ]
        definition["fact_sources"][0]["dims"] = ["users"]
        metric_definition["filters"] = [{"property": "country", "op": "equals", "values": ["UK"]}]
    definitions = Definitions.model_validate(definition)
    spec = MetricSpec(name="revenue", preferred_direction="increase")

    def native(*, artifact: bool = False) -> Analysis:
        con = ds.duckdb_connection(events)
        if pinned_dimension:
            con.create_table(
                "users", pd.DataFrame({"user_id": [*units, "outside"], "country": ["UK"] * 7})
            )
        analysis = make_analysis(con, definitions, experiment="exp")
        if artifact:
            return _publish_and_adopt(con, analysis)
        if pinned_dimension:
            pinned = _native_source(analysis)._pinned_source()
            analysis.close()
            analysis = Analysis._from_source(
                pinned,
                defs=definitions,
                experiment=definitions.experiments[0],
                con=native_connection(pinned),
            )
        return _track_connection(analysis, con)

    def summary() -> Analysis:
        totals = (
            pd.DataFrame(records).groupby(["user_id", "group_id"], as_index=False)["revenue"].sum()
        )
        return Analysis.from_unit_summary(
            totals,
            unit="user_id",
            group="group_id",
            control="control",
            metrics=[spec],
            plan=plan,
        )

    def panel() -> Analysis:
        return Analysis.from_unit_panel(
            pd.DataFrame(records[::-1]),
            unit="user_id",
            group="group_id",
            control="control",
            date="day",
            exposure_date="exposure",
            observation_end="d9" if pinned_dimension else None,
            metrics=[spec.model_copy(update={"window_days": window})],
            plan=plan,
        )

    def probe(results: Any) -> None:
        (row,) = results
        assert row.lift is not None and row.lift.value == pytest.approx(1.0)
        assert row.lift.lb is not None and row.lift.ub is not None

    return ParityCase(
        id="audit-pinned-dimension-freshness"
        if pinned_dimension
        else "audit-panel-component-horizon",
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": summary,
            "from_unit_panel": panel,
            "from_moments": lambda: _export_and_replay(summary(), [spec]),
        },
        readout_probe=probe,
        waive=dict(_SWITCHBACK_WAIVE),
        slow=True,
    )


def _ratio_component_horizon_case(
    *,
    complete: bool = False,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """Keep only cohorts whose numerator and denominator streams are mature."""
    start = dt.datetime(2025, 1, 1)
    plan = AnalysisPlan(primary="ratio")
    records, events, totals = [], [], []
    for arm, multiplier in (("control", 1), ("treatment", 2)):
        for cohort, exposure_day in (("old", 0), ("new", 2)):
            for i in range(3):
                unit = f"{arm}-{cohort}-{i}"
                value = float(multiplier * ((2, 2.5, 3)[i] if cohort == "old" else 3 * (i + 1)))
                events.append(
                    ds._row(
                        unit,
                        start + dt.timedelta(days=exposure_day),
                        "exposure",
                        group_id=arm,
                        experiment_id="exp",
                    )
                )
                for offset in (0, 2):
                    day = exposure_day + offset
                    denominator = 1.0 if cohort == "old" or offset == 0 or complete else None
                    events.append(
                        ds._row(
                            unit,
                            start + dt.timedelta(days=day, hours=12),
                            "purchase",
                            revenue=value,
                        )
                    )
                    if denominator is not None:
                        events.append(
                            ds._row(
                                unit,
                                start + dt.timedelta(days=day, hours=13),
                                "session_end",
                                sess=1,
                            )
                        )
                    records.append(
                        {
                            "unit": unit,
                            "group": arm,
                            "day": f"d{day}",
                            "exposure": f"d{exposure_day}",
                            "num": value,
                            "den": denominator,
                        }
                    )
                if cohort == "old" or complete:
                    totals.append({"unit": unit, "group": arm, "num": 2 * value, "den": 2.0})
    events.append(
        ds._row("control-old-0", start + dt.timedelta(days=5, hours=12), "purchase", revenue=1000.0)
    )
    records.append(
        {
            "unit": "control-old-0",
            "group": "control",
            "day": "d5",
            "exposure": "d0",
            "num": 1000.0,
            "den": None,
        }
    )
    definition = ds.definitions_dict(plan=plan)
    definition["dialect"] = dialect
    definition["experiments"][0].update(start=start.isoformat(), end=None, n_pre_periods=0)
    definition["metrics"] = [
        {
            "name": "ratio",
            "type": "ratio",
            "entity": "user_id",
            "preferred_direction": "increase",
            "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 3},
            "denominator": {"fact": "session_end", "aggregation": "count", "window_days": 3},
        }
    ]
    definitions = Definitions.model_validate(definition)
    spec = MetricSpec(
        name="ratio",
        type="ratio",
        numerator="num",
        denominator="den",
        preferred_direction="increase",
    )

    def native(*, artifact: bool = False) -> Analysis:
        con = warehouse_connection(events)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)

    def summary() -> Analysis:
        return Analysis.from_unit_summary(
            pd.DataFrame(totals),
            unit="unit",
            group="group",
            control="control",
            metrics=[spec],
            plan=plan,
        )

    def panel() -> Analysis:
        return Analysis.from_unit_panel(
            pd.DataFrame(records[::-1]),
            unit="unit",
            group="group",
            control="control",
            date="day",
            exposure_date="exposure",
            metrics=[spec.model_copy(update={"window_days": 3, "missing": "zero"})],
            plan=plan,
        )

    def assert_moments(rows: Iterable[Mapping[str, Any]]) -> None:
        moments = {row["group_id"]: row for row in rows}
        assert set(moments) == {"control", "treatment"}
        for arm, multiplier in (("control", 1), ("treatment", 2)):
            row = moments[arm]
            n = 6 if complete else 3
            assert row["n"] == n
            assert n * row["ref_y"] + row["cy1"] == pytest.approx(
                multiplier * (51.0 if complete else 15.0)
            )
            assert n * row["ref_den"] + row["cden1"] == pytest.approx(2.0 * n)
            assert n * row["ref_y"] ** 2 + 2 * row["ref_y"] * row["cy1"] + row[
                "cy2"
            ] == pytest.approx(multiplier**2 * (581.0 if complete else 77.0))

    def moment_probe(constructor: str, analysis: Analysis) -> None:
        source = _moment_source(analysis)
        if constructor == "from_moments":
            assert_moments(source.moments(source.context.metrics[0]))
        else:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "moments.parquet"
                analysis.export(path)
                assert_moments(pq.read_table(path).to_pylist())
        if constructor in {"from_definitions", "from_unit_day_artifact", "from_unit_panel"}:
            rows = source.moments(
                source.context.metrics[0], grain="asof", completed_windows_only=True
            )
            last = max(row["ds"] for row in rows)
            assert_moments(row for row in rows if row["ds"] == last)

    def probe(results: Any) -> None:
        (row,) = results
        assert row.lift is not None and row.lift.value == pytest.approx(1.0)
        assert row.lift.lb is not None and row.lift.ub is not None

    return ParityCase(
        id="audit-panel-component-horizon-ratio-complete"
        if complete
        else "audit-panel-component-horizon-ratio",
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": summary,
            "from_unit_panel": panel,
            "from_moments": lambda: _export_and_replay(summary(), [spec]),
        },
        readout_probe=probe,
        source_probe=moment_probe,
        expected_warning_codes={}
        if complete
        else dict.fromkeys(
            ("from_definitions", "from_unit_panel"),
            ("frame.censoring.dropped_units",),
        ),
        waive=dict(_SWITCHBACK_WAIVE),
        slow=True,
    )


def _adjusted_ratio_anchor_case(*, separated: bool = False) -> ParityCase:
    """Retain the same pooled-anchor denominator state on every reachable source."""
    method = {"name": "cuped", "variance_reduction": "cuped"}
    inference = InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=100)
    frame_plan = AnalysisPlan(primary="ratio", inference=inference)
    native_plan = AnalysisPlan(
        primary=ExperimentMetric(metric="ratio", decision_method=method), inference=inference
    )
    allocation = {"control": 0.5, "treatment": 0.5}
    design = Randomized(control_group="control", allocation=allocation)
    spec = MetricSpec(
        name="ratio",
        type="ratio",
        numerator="y",
        denominator="den",
        covariate="x",
        decision_method=method,
    )
    events: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for arm, base, count in (
        ("control", 1, 17 if separated else 20),
        ("treatment", 4 if separated else 3, 23 if separated else 20),
    ):
        for i in range(count):
            unit = f"{arm}-{i:03d}"
            numerator = base + (i % 3 if separated else i % 4) / 10
            denominator = 10 + i % 4 / 10 if separated else int(i < 2) + i % 2 / 100
            covariate = i % 2 if separated else int(i < 2)
            events.extend(
                [
                    ds._row(unit, ds._EXPOSURE_AT, "exposure", group_id=arm, experiment_id="exp"),
                    ds._row(unit, ds._PRE_PURCHASE_AT, "purchase", revenue=float(covariate)),
                    ds._row(unit, ds._PURCHASE_AT, "purchase", revenue=numerator),
                    ds._row(unit, ds._SESSION_AT, "denominator", latency=denominator),
                ]
            )
            units.append(
                {
                    "unit": unit,
                    "arm": arm,
                    "exposure": ds._EXPOSURE_AT,
                    "day": ds._EXPOSURE_AT.date(),
                    "y": numerator,
                    "den": denominator,
                    "x": covariate,
                }
            )
    events.extend(
        [
            ds._row("freshness", ds._FRESHNESS_PAD_AT, "purchase", revenue=0.0),
            ds._row("freshness", ds._FRESHNESS_PAD_AT, "denominator", latency=0.0),
        ]
    )
    definition = ds.definitions_dict(plan=native_plan, allocation=allocation)
    definition["fact_sources"][0]["facts"].append({"name": "denominator", "column": "latency"})
    definition["metrics"] = [
        {
            "name": "ratio",
            "type": "ratio",
            "entity": "user_id",
            "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 1},
            "denominator": {"fact": "denominator", "aggregation": "sum", "window_days": 1},
        }
    ]
    definitions = Definitions.model_validate(definition)
    experiment = definitions.experiment("exp")
    assert experiment is not None
    bound = bind_automatic_sequential_plan(
        native_plan,
        definitions.metrics,
        design=design,
        source_id="exp",
        source_mapping=native_observation_mapping(definitions, experiment),
        pre_period_covariate=True,
    )
    as_of = ds._EXPERIMENT_END - dt.timedelta(days=1)

    def native(*, artifact: bool = False) -> Analysis:
        con = ds.duckdb_connection(events)
        analysis = make_analysis(con, definitions, experiment="exp", plan=bound)
        analysis = (
            _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)
        )
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return analysis

    def summary() -> Analysis:
        return Analysis.from_unit_summary(
            pd.DataFrame(units),
            unit="unit",
            group="arm",
            exposure_date="exposure",
            design=design,
            metrics=[spec],
            plan=frame_plan,
            experiment_id="exp",
        )

    def panel() -> Analysis:
        analysis = Analysis.from_unit_panel(
            pd.DataFrame(units),
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposure",
            design=design,
            metrics=[spec.model_copy(update={"window_days": 1})],
            plan=frame_plan,
            experiment_id="exp",
        )
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return analysis

    def check_availability(readout: Any) -> None:
        (row,) = readout
        result = row.require_asymptotic_sequential_result()
        assert result.bounds.available is separated
        if not separated:
            assert result.bounds.reason == "denominator_near_zero"
            assert not row.stat_sig()

    return ParityCase(
        id="audit-adjusted-ratio-anchor-resolved" if separated else "audit-adjusted-ratio-anchor",
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": summary,
            "from_unit_panel": panel,
            "from_moments": lambda: _export_and_replay(summary(), [spec]),
        },
        sequential=True,
        readout_probe=check_availability,
        waive={
            **_SWITCHBACK_WAIVE,
            "from_unit_panel": (
                "SOURCE: registered panel capture cannot supply the per-unit pre-period "
                "covariate for adjusted_ratio_mean; use a unit-summary frame."
            ),
        },
        waived_refusal_codes={"from_unit_panel": "frame.validation.from_unit_panel"},
        slow=True,
    )


def _sequential_missing_zero_mean_ratio_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """Same mean/ratio reveals from absent facts and zero-filled null/NaN outcomes."""
    rows = ds.event_rows()
    sequential_plan = AnalysisPlan(
        primary="revenue",
        secondaries=("rps",),
        inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
    )
    defs_dict = ds.definitions_dict(
        plan=sequential_plan, allocation={"control": 0.5, "treatment": 0.5}
    )
    defs_dict["dialect"] = dialect
    defs = Definitions.model_validate(defs_dict)
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    as_of = ds._EXPERIMENT_END - dt.timedelta(days=1)

    def _native_bound_plan() -> AnalysisPlan:
        exp = defs.experiment("exp")
        assert exp is not None
        bound = bind_automatic_sequential_plan(
            exp.plan,
            [m for m in defs.metrics if m.name in exp.metric_names],
            design=design,
            source_id=exp.name,
            source_mapping=native_observation_mapping(defs, exp, on_mixed_assignment="error"),
            pre_period_covariate=exp.n_pre_periods > 0,
        )
        assert bound is not None
        return bound

    def build_definitions() -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        adopted = _publish_and_adopt(con, native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    summary_metrics = [
        MetricSpec(name="revenue", type="mean", missing="zero", preferred_direction="increase"),
        MetricSpec(
            name="rps",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            missing="zero",
            preferred_direction="increase",
        ),
    ]

    def _summary_frame() -> pd.DataFrame:
        con = ds.duckdb_connection(rows)
        summary = ds.unit_summary_frame(con)
        con.disconnect()
        frame = pd.DataFrame(summary.to_pylist())
        missing = frame.index[frame["revenue"].isna()].tolist()
        assert len(missing) >= 2
        frame["revenue"] = frame["revenue"].astype(object)
        frame.loc[missing[0], "revenue"] = float("nan")
        # Keep the next missing unit as an actual null; both feed the same
        # zero-equivalent absent-purchase cohort used by warehouse routes.
        frame.loc[missing[1], "revenue"] = None
        return frame

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            _summary_frame(),
            unit="user_id",
            group="variant",
            design=design,
            metrics=summary_metrics,
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_unit_panel() -> Analysis:
        summary = _summary_frame()
        panel = summary[["user_id", "variant", "exposure_date", "revenue", "sessions"]].copy()
        panel["date"] = ds._EXPOSURE_AT.date()
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    missing="zero",
                    window_days=1,
                    preferred_direction="increase",
                ),
                MetricSpec(
                    name="rps",
                    type="ratio",
                    numerator="revenue",
                    denominator="sessions",
                    missing="zero",
                    window_days=1,
                    preferred_direction="increase",
                ),
            ],
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        analysis._sequential_as_of = ds._EXPOSURE_AT.date() + dt.timedelta(days=1)  # ty: ignore[unresolved-attribute]
        return analysis

    def build_moments() -> Analysis:
        return _export_and_replay(
            build_unit_summary(),
            [
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions"),
            ],
        )

    return ParityCase(
        id="audit-sequential-missing-zero",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        sequential=True,
        waive=dict(_SWITCHBACK_WAIVE),
        slow=True,
    )


def _sequential_composed_itt_and_uptake_case(
    *,
    metric_free: bool = False,
    exclusion_declared: bool = True,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """Compare automatic uptake monitoring, with or without outcome metrics."""
    rows = ds.event_rows()
    sequential_plan = AnalysisPlan(
        primary=None if metric_free else "revenue",
        inference={"kind": "always_valid", "baseline_rate": 0.25}
        if metric_free
        else {"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
        compliance=SequentialCompliancePolicy(alpha=0.05),
    )
    defs_dict = ds.definitions_dict(
        plan=sequential_plan, allocation={"control": 0.5, "treatment": 0.5}
    )
    defs_dict["dialect"] = dialect
    if metric_free:
        defs_dict["metrics"] = []
    summary_specs = [] if metric_free else [MetricSpec(name="revenue", missing="zero")]
    panel_specs = [] if metric_free else [MetricSpec(name="revenue", missing="zero", window_days=1)]
    frame_design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="converted", window_days=1),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="the in-window purchase mediates the revenue effect"
        )
        if exclusion_declared
        else None,
        allocation={"control": 0.5, "treatment": 0.5},
    )
    # The warehouse has no "converted" fact (a frame column: in-window purchase
    # revenue > 0). In-window "purchase" occurrence is the same boolean here:
    # every in-window purchase has positive revenue, and dataset.py's
    # freshness-padding purchase falls outside window_days=1.
    native_design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="purchase", window_days=1),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="the in-window purchase mediates the revenue effect"
        )
        if exclusion_declared
        else None,
        allocation={"control": 0.5, "treatment": 0.5},
    )
    defs_dict["experiments"][0]["design"] = {
        "mechanism": "encouragement",
        "uptake": {"fact": "purchase", "window_days": 1},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "the in-window purchase mediates the revenue effect",
        },
    }
    if not exclusion_declared:
        del defs_dict["experiments"][0]["design"]["exclusion_restriction"]
    defs = Definitions.model_validate(defs_dict)
    exp = defs.experiment("exp")
    assert exp is not None
    as_of = ds._EXPOSURE_AT.date() + dt.timedelta(days=1)

    def _native_bound_plan() -> AnalysisPlan:
        assert exp is not None
        bound = bind_automatic_sequential_plan(
            exp.plan,
            [m for m in defs.metrics if m.name in exp.metric_names],
            design=native_design,
            source_id=exp.name,
            source_mapping=native_observation_mapping(defs, exp, on_mixed_assignment="error"),
            pre_period_covariate=exp.n_pre_periods > 0,
        )
        assert bound is not None
        return bound

    def build_definitions() -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]  # consumed by the runner's capture_sequential call
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        adopted = _publish_and_adopt(con, native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.unit_summary_frame(con)
        con.disconnect()  # `summary` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            summary,
            unit="user_id",
            group="variant",
            design=frame_design,
            metrics=summary_specs,
            uptake="converted",
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()  # `panel` is already materialized; nothing below needs `con` alive
        panel["uptake"] = (panel["revenue"] > 0).astype(int)
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=frame_design,
            metrics=panel_specs,
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
            uptake="uptake",
        )
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]  # matches _EXPOSURE_AT's day + window_days=1
        return analysis

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        summary.capture_sequential(finalized=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(replay_rows, metrics=summary_specs, design=frame_design)
        summary.close()
        return replayed

    return ParityCase(
        id="sequential_encouragement_optional_exclusion"
        if not exclusion_declared
        else "audit-compliance-only-sequential-empty-catalog"
        if metric_free
        else "sequential_composed_itt_and_uptake",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=("compliance",) if metric_free else ("itt", "compliance"),
        sequential=True,
        source_probe=_assert_missing_exclusion_precedes_data if not exclusion_declared else None,
        waive=dict(_SWITCHBACK_WAIVE),
        # Exports/replays moments and captures a finalized sequential
        # checkpoint on every constructor -- see _fixed_horizon_case's
        # slow=True for why.
        slow=True,
    )


def _retained_catalog_probes() -> tuple[Callable[[str, Analysis], None], Callable[[], None]]:
    """The per-constructor probe and the cross-constructor agreement check for
    `_sequential_unbounded_retention_catalog_case`."""
    checkpoints: dict[str, list[tuple[Any, ...]]] = {}

    def probe_retained_catalog(constructor: str, analysis: Analysis) -> None:
        # An empty catalog would make the exemption vacuous, so the retained
        # catalog is asserted, then the compliance-only checkpoint recorded.
        assert [m.name for m in analysis.metrics] == ["returned"], constructor
        checkpoint_rows = analysis.run_asof_lift(
            estimands=("compliance",), completed_windows_only=True
        )
        # `prefix_id` depends on reveal order, so compare the evidence itself.
        checkpoints[constructor] = []
        for row in checkpoint_rows:
            assert row.sequential_result is not None
            checkpoints[constructor].append(
                (
                    row.metric,
                    row.group_id,
                    row.estimand,
                    row.ds,
                    row.lift.model_dump() if row.lift is not None else None,
                    row.sequential_result.log_e,
                    row.sequential_result.decision_alpha,
                )
            )
        assert len(checkpoints[constructor]) == 1, constructor
        for completed_windows_only in (False, True):
            with pytest.raises(CodedError) as refused:
                analysis.run_asof_lift(
                    estimands=("itt",), completed_windows_only=completed_windows_only
                )
            assert refused.value.code == "breakout.retention.encouragement", (
                constructor,
                completed_windows_only,
            )

    def assert_checkpoints_agree() -> None:
        from .comparison import nested_close

        assert set(checkpoints) == {"from_definitions", "from_unit_day_artifact", "from_moments"}
        for constructor, observed in checkpoints.items():
            assert nested_close(checkpoints["from_definitions"], observed), constructor

    return probe_retained_catalog, assert_checkpoints_agree


def _sequential_unbounded_retention_catalog_case() -> ParityCase:
    """A registered compliance-only checkpoint that keeps an unbounded retention
    metric in its outcome catalog.

    The checkpoint reads uptake and never that outcome, so the catalog must
    survive each ingress unchanged; the empty-catalog case above says nothing
    about a nonempty one. An automatic registration monitors every catalog
    metric and cannot reveal an unbounded band, so the uptake-only registration
    is declared explicitly over the full catalog, in the definitions an
    artifact adopts. `from_unit_summary` cannot declare a retention metric and
    `from_unit_panel` refuses one under an Encouragement design at construction,
    so neither carries the catalog; `from_moments` replays the definitions
    export with the metric kept in `metrics=`. Outcome-consuming requests keep
    refusing the retention metric on every path that carries it.
    """
    from increment.semantics.sequential import SequentialRegistration
    from increment.sequential_source import sequential_definition_id

    # Extra treatment-arm uptake gives the compliance checkpoint a nonzero effect.
    rows = ds.event_rows() + _extra_treatment_purchases(11)
    returned = MetricSpec(name="returned", type="retention", threshold_days=1)
    plan = AnalysisPlan(
        primary=None,
        secondaries=["returned"],
        inference={"kind": "always_valid", "baseline_rate": 0.25},
        compliance=SequentialCompliancePolicy(alpha=0.05),
    )
    justification = "the in-window purchase mediates the revenue effect"
    defs_dict = ds.definitions_dict(plan=plan, allocation={"control": 0.5, "treatment": 0.5})
    defs_dict["metrics"] = [
        {
            "type": "retention",
            "name": "returned",
            "entity": "user_id",
            "fact": "purchase",
            "threshold_days": 1,
            "preferred_direction": "increase",
        }
    ]
    defs_dict["experiments"][0]["design"] = {
        "mechanism": "encouragement",
        "uptake": {"fact": "purchase", "window_days": 1},
        "exclusion_restriction": {"acknowledged": True, "justification": justification},
    }
    declared = Definitions.model_validate(defs_dict)
    exp = declared.experiment("exp")
    assert exp is not None
    exclusion = ExclusionRestriction(acknowledged=True, justification=justification)
    native_design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="purchase", window_days=1),
        exclusion_restriction=exclusion,
        allocation={"control": 0.5, "treatment": 0.5},
    )
    frame_design = native_design.model_copy(
        update={"uptake": UptakeSpec(fact="converted", window_days=1)}
    )
    mapping = native_observation_mapping(declared, exp, on_mixed_assignment="error")
    automatic = bind_automatic_sequential_plan(
        exp.plan.model_copy(update={"secondaries": ()}),
        [],
        design=native_design,
        source_id=exp.name,
        source_mapping=mapping,
        pre_period_covariate=exp.n_pre_periods > 0,
    )
    assert automatic is not None and automatic.inference.registration is not None  # ty: ignore[unresolved-attribute]
    registration = SequentialRegistration.model_validate(
        {
            **automatic.inference.registration.model_dump(),  # ty: ignore[unresolved-attribute]
            "definitions_id": sequential_definition_id(
                list(declared.metrics), native_design, source_mapping=mapping
            ),
        }
    )
    registered = exp.plan.model_copy(
        update={"inference": InferenceSpec(kind="always_valid", registration=registration)}
    )
    defs = declared.model_copy(
        update={"experiments": (exp.model_copy(update={"plan": registered}),)}
    )
    as_of = ds._EXPOSURE_AT.date() + dt.timedelta(days=1)

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp", plan=registered)
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        adopted = _publish_and_adopt(
            con, make_analysis(con, defs, experiment="exp", plan=registered)
        )
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.unit_summary_frame(con)
        con.disconnect()
        return Analysis.from_unit_summary(
            summary,
            unit="user_id",
            group="variant",
            design=frame_design,
            metrics=[returned],
            uptake="converted",
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()
        panel["uptake"] = (panel["revenue"] > 0).astype(int)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=frame_design,
            metrics=[returned],
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
            uptake="uptake",
        )

    def build_moments() -> Analysis:
        source = build_definitions()
        try:
            source.capture_sequential(finalized=True, as_of=as_of)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "moments.parquet"
                source.export(path)
                replay_rows = pq.read_table(path).to_pylist()
            return Analysis.from_moments(replay_rows, metrics=[returned], design=native_design)
        finally:
            _close_parity_analysis(source)

    probe_retained_catalog, assert_checkpoints_agree = _retained_catalog_probes()

    return ParityCase(
        id="audit-compliance-only-sequential-unbounded-retention-catalog",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=("compliance",),
        sequential=True,
        source_probe=probe_retained_catalog,
        sequential_probe=assert_checkpoints_agree,
        waive={
            **_SWITCHBACK_WAIVE,
            "from_unit_summary": (
                "SOURCE: a one-row-per-unit summary has no per-unit dates to resolve "
                "a retention band against, so the catalog cannot be declared."
            ),
            "from_unit_panel": (
                "COMBINATION: a unit panel refuses a retention metric under an "
                "Encouragement design at construction, before any request is read."
            ),
        },
        waived_refusal_codes={
            "from_unit_summary": "source.frame.constructor",
            "from_unit_panel": "readout.encouragement.retention",
        },
        slow=True,
    )


def _extra_treatment_purchases(n: int) -> list[dict[str, Any]]:
    """`n` additional in-window purchase events for treatment units
    `event_rows` otherwise leaves as genuine non-purchasers (`i % 3 == 0`)
    -- the only way to give a real, non-trivial treatment effect to
    `_sequential_composed_family_ebh_case`'s e-BH family without touching
    the shared `event_rows` every other parity case also reads."""
    candidates = [i for i in range(ds._N_PER_ARM) if i % 3 == 0]
    assert n <= len(candidates)
    return [
        ds._row(
            f"t{i}",
            ds._PURCHASE_AT,
            "purchase",
            store_id=ds.store_for(i),
            revenue=5.0 + 1 + (i % 5),
        )
        for i in candidates[:n]
    ]


def _sequential_composed_family_ebh_case() -> ParityCase:
    """The same composed ITT + Bernoulli-uptake compliance registration as
    `_sequential_composed_itt_and_uptake_case`, but with two genuinely
    in-family secondaries (`purchase_rate`, `rps`) declared on the plan
    (`secondaries=`/`q=`) and `SequentialCompliancePolicy(family=True)`, so
    the compliance cell joins the SAME e-BH family as the secondaries
    (`sequential_family_size`) and `select_sequential_family` actually
    runs -- `_sequential_composed_itt_and_uptake_case`'s
    `SequentialCompliancePolicy` defaults `family=False`, so that case
    never exercises e-BH selection at all, the gap `docs/reference/capabilities-by-entry-point.md`
    and `docs/guides/encouragement.md` both claim is covered ("e-BH
    selected" across five constructors). `_extra_treatment_purchases(11)`
    converts 11 of the treatment arm's genuine non-purchasers into
    purchasers, a real (not tolerance-tuned) effect that gives this family
    exactly one selected member at every constructor: confirmed live,
    `rps` discovers (`log_e` ~4.26, reinverted `decision_alpha` 0.0267 --
    below the 0.05 nominal cap, since `family_threshold = q*1/3 < 0.05`),
    while `purchase_rate` (`log_e` ~1.39) and the uptake compliance cell
    (`log_e` ~-1.25) both stay unselected. A path that selected a
    different member, selected none, or reinverted at a different alpha
    would fail this case where it used to pass trivially at R=0. Reuses
    `dataset.sequential_unit_panel_frame`'s `sessions` column (added
    alongside `revenue` for exactly this ratio secondary) for `rps`'s
    per-day denominator on `from_unit_panel`. Every constructor agrees,
    including every family-selection field the runner compares
    (`discovery`, `family_q`, `family_threshold`, `family_guarantee`,
    `family_nominal_alpha`) on every row -- primary, both secondaries, and
    the uptake cell. from_switchback_panel has no sequential construction
    at all (the same permanent SOURCE limitation the other two sequential
    cases already waive).
    """
    rows = ds.event_rows() + _extra_treatment_purchases(11)
    sequential_plan = AnalysisPlan(
        primary="revenue",
        secondaries=["purchase_rate", "rps"],
        q=0.08,
        inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
        compliance=SequentialCompliancePolicy(alpha=0.05, family=True),
    )
    defs_dict = ds.definitions_dict(
        plan=sequential_plan, allocation={"control": 0.5, "treatment": 0.5}
    )
    frame_design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="converted", window_days=1),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="the in-window purchase mediates the revenue effect"
        ),
        allocation={"control": 0.5, "treatment": 0.5},
    )
    # Same warehouse/dataframe uptake-fact substitution as
    # `_sequential_composed_itt_and_uptake_case`: "converted" is a
    # dataframe-only derived column, "purchase" is the identical
    # warehouse-native signal (see that case's own comment).
    native_design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="purchase", window_days=1),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="the in-window purchase mediates the revenue effect"
        ),
        allocation={"control": 0.5, "treatment": 0.5},
    )
    defs_dict["experiments"][0]["design"] = {
        "mechanism": "encouragement",
        "uptake": {"fact": "purchase", "window_days": 1},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "the in-window purchase mediates the revenue effect",
        },
    }
    defs = Definitions.model_validate(defs_dict)
    exp = defs.experiment("exp")
    assert exp is not None
    as_of = ds._EXPOSURE_AT.date() + dt.timedelta(days=1)

    def _native_bound_plan() -> AnalysisPlan:
        assert exp is not None
        bound = bind_automatic_sequential_plan(
            exp.plan,
            [m for m in defs.metrics if m.name in exp.metric_names],
            design=native_design,
            source_id=exp.name,
            source_mapping=native_observation_mapping(defs, exp, on_mixed_assignment="error"),
            pre_period_covariate=exp.n_pre_periods > 0,
        )
        assert bound is not None
        return bound

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]  # consumed by the runner's capture_sequential call
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        adopted = _publish_and_adopt(con, native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    summary_metrics = [
        MetricSpec(name="revenue", type="mean", missing="zero"),
        MetricSpec(
            name="purchase_rate", type="conversion", value_column="converted", missing="zero"
        ),
        MetricSpec(
            name="rps", type="ratio", numerator="revenue", denominator="sessions", missing="zero"
        ),
    ]

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.unit_summary_frame(con)
        con.disconnect()  # `summary` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            summary,
            unit="user_id",
            group="variant",
            design=frame_design,
            metrics=summary_metrics,
            uptake="converted",
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()  # `panel` is already materialized; nothing below needs `con` alive
        panel["uptake"] = (panel["revenue"] > 0).astype(int)
        panel["purchase_rate"] = (panel["revenue"] > 0).astype(int)
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=frame_design,
            metrics=[
                MetricSpec(name="revenue", type="mean", missing="zero", window_days=1),
                MetricSpec(name="purchase_rate", type="conversion", missing="zero", window_days=1),
                MetricSpec(
                    name="rps",
                    type="ratio",
                    numerator="revenue",
                    denominator="sessions",
                    missing="zero",
                    window_days=1,
                ),
            ],
            plan=sequential_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
            uptake="uptake",
        )
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]  # matches _EXPOSURE_AT's day + window_days=1
        return analysis

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        summary.capture_sequential(finalized=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(
            replay_rows,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="purchase_rate", type="conversion"),
                MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions"),
            ],
            design=frame_design,
        )
        summary.close()
        return replayed

    return ParityCase(
        id="sequential_composed_family_ebh",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=("itt", "compliance"),
        sequential=True,
        require_selection=True,
        waive=dict(_SWITCHBACK_WAIVE),
        # Exports/replays moments and captures a finalized sequential
        # checkpoint on every constructor -- see _fixed_horizon_case's
        # slow=True for why.
        slow=True,
    )


_SEGMENTED_BREAKOUT_STORES = ("s0", "s1", "s2")


def _segmented_roster(
    metric: str, base_alpha: Fraction, stores: tuple[str, ...] = _SEGMENTED_BREAKOUT_STORES
) -> tuple[SequentialCell, ...]:
    """A registered per-segment roster at a fair share of the automatic
    registration's own per-cell allocation -- `validate_breakout_registration`
    refuses a cell whose alpha exceeds `alpha_by_metric / divisor`
    (`sequential_source.py`), `divisor` being the segment count for a
    Bonferroni family."""
    alpha = base_alpha / len(stores)
    return tuple(
        SequentialCell(
            metric=metric, group_id="treatment", family=True, alpha=alpha, segment=(("store", s),)
        )
        for s in stores
    )


def _segmented_plan_from(
    metric: str,
    kind: Literal["always_valid", "asymptotic_mean"],
    base_reg: Any,
    stores: tuple[str, ...] = _SEGMENTED_BREAKOUT_STORES,
) -> AnalysisPlan:
    """A per-segment roster derived from an existing automatic registration's
    own per-cell allocation -- shared by every builder in
    `_registered_segmented_breakout_case` that needs to turn an automatic
    registration into a segmented one."""
    assert base_reg is not None
    roster = _segmented_roster(metric, base_reg.roster[0].alpha, stores)
    return AnalysisPlan(
        primary=metric,
        inference=InferenceSpec(
            kind=kind, registration=base_reg.model_copy(update={"roster": roster})
        ),
    )


_SEGMENTED_BREAKOUT_BOOST_N_PER_ARM = 150
_SEGMENTED_BREAKOUT_BOOST_STORE = "s0"


def _segmented_breakout_boost_rows(
    kind: Literal["always_valid", "asymptotic_mean"], n_per_arm: int, store: str
) -> list[dict[str, Any]]:
    """Extra treatment-arm purchase events confined to one breakout segment
    -- gives `_registered_segmented_breakout_case`'s registered family a
    real, non-trivial selected member in `store` while every other segment
    stays at `event_rows`'s built-in zero true effect (`purchase_rate`'s
    in-window-purchase cohort and `revenue`'s treatment bump are otherwise
    identical across segments). Discrete (`purchase_rate`) converts every
    genuine non-purchaser (`i % 3 == 0`) in the segment into a purchaser --
    confirmed live, this alone crosses the registered e-BH family's
    selection threshold (`log_e` ~10.7 at `store`'s full 20-candidate
    segment, comfortably above the `m/(q*R)` cutoff) while the other two
    segments stay unselected at their fixed per-cell alpha. Continuous
    (`revenue`) adds one flat bonus purchase to every unit in the segment
    -- confirmed live, this alone crosses Bonferroni's fixed per-cell
    threshold there (`log_e` ~21) while the other two segments stay
    unselected.
    """
    candidates = [i for i in range(n_per_arm) if ds.store_for(i) == store]
    if kind == "always_valid":
        return [
            ds._row(f"t{i}", ds._PURCHASE_AT, "purchase", store_id=store, revenue=5.0 + 1 + (i % 5))
            for i in candidates
            if i % 3 == 0
        ]
    return [
        ds._row(f"t{i}", ds._PURCHASE_AT, "purchase", store_id=store, revenue=5.0)
        for i in candidates
    ]


def _unit_summary_from_rows(rows: list[dict[str, Any]]):
    con = ds.duckdb_connection(rows)
    try:
        return ds.unit_summary_frame(con)
    finally:
        con.disconnect()


# The three store segments relabelled as a nullable Boolean dimension: the
# canonical strings are the oracle labels a typed Boolean/null column must reach.
_BOOLEAN_SEGMENT_LABEL = {"s0": "true", "s1": "false", "s2": "__null__"}
_BOOLEAN_SEGMENT_VALUE = {"true": True, "false": False, "__null__": None}
_BOOLEAN_SEGMENT_STORES = tuple(_BOOLEAN_SEGMENT_LABEL.values())
# Warehouse values of the same segments: a genuine BOOLEAN column, a missing value is NULL.
_BOOLEAN_STORE_VALUE = {"s0": True, "s1": False, "s2": None}


def _typed_boolean_store(frame: Any, kind: Literal["pandas", "polars", "arrow"]) -> Any:
    """*frame* (pandas or Arrow, ``store`` holding canonical labels) with ``store``
    rebuilt as the nullable Boolean column those labels name, in *kind*'s dtype."""
    import pyarrow as pa

    if isinstance(frame, pd.DataFrame):
        labels = frame["store"].tolist()
        base = frame.drop(columns="store")
    else:
        labels = frame["store"].to_pylist()
        base = frame.to_pandas().drop(columns="store")
    values = [_BOOLEAN_SEGMENT_VALUE[label] for label in labels]
    if kind == "pandas":
        return base.assign(store=pd.array(values, dtype="boolean"))
    if kind == "polars":
        import polars as pl

        return pl.from_pandas(base).with_columns(pl.Series("store", values, dtype=pl.Boolean))
    return pa.Table.from_pandas(base, preserve_index=False).append_column(
        "store", pa.array(values, pa.bool_())
    )


def _relabel_summary_store(table: Any, typed: Literal["pandas", "polars", "arrow"] | None) -> Any:
    """The summary table with ``store`` relabelled ``true``/``false``/``__null__``
    (canonical strings), or as the nullable Boolean column those labels name."""
    import pyarrow as pa

    labels = [_BOOLEAN_SEGMENT_LABEL[s] for s in table["store"].to_pylist()]
    relabelled = table.set_column(table.schema.get_field_index("store"), "store", pa.array(labels))
    return relabelled if typed is None else _typed_boolean_store(relabelled, typed)


def _segmented_panel(
    rows: list[dict[str, Any]], store_by_unit: dict[str, str], metric: str, boolean: bool
) -> pd.DataFrame:
    panel = ds.sequential_unit_panel_frame(ds.duckdb_connection(rows))
    panel["store"] = panel["user_id"].map(store_by_unit)
    if metric == "purchase_rate":
        panel["purchase_rate"] = (panel["revenue"] > 0).astype(int)
    if boolean:
        panel["store"] = panel["store"].map(_BOOLEAN_SEGMENT_LABEL)
        panel = _typed_boolean_store(panel, "pandas")
    return panel


def _assert_typed_summaries_match_oracle(
    plan: AnalysisPlan,
    summary_frame: Callable[..., Any],
    summary_analysis: Callable[[Any, AnalysisPlan], Analysis],
) -> None:
    """Typed Boolean/null summaries in every dataframe dtype retain the
    canonical-string oracle's state and read out its rows exactly."""
    oracle = summary_analysis(summary_frame(), plan)
    try:
        expected = oracle.capture_sequential(finalized=True)
        expected_rows = [row.model_dump() for row in oracle.run_breakout()]
        for typed in ("pandas", "polars", "arrow"):
            candidate = summary_analysis(summary_frame(typed), plan)
            try:
                assert candidate.capture_sequential(finalized=True) == expected
                assert [row.model_dump() for row in candidate.run_breakout()] == expected_rows
            finally:
                candidate.close()
    finally:
        oracle.close()


def _registered_segmented_breakout_case(
    *,
    id: str,
    metric: str,
    spec_summary: MetricSpec,
    spec_panel: MetricSpec,
    kind: Literal["always_valid", "asymptotic_mean"],
    automatic: bool = False,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
    boolean_segments: bool = False,
) -> ParityCase:
    """A registered (explicit, not automatic) segmented sequential family:
    `SequentialRegistration.roster` names one segment cell per `store`
    value, so `select_sequential_family`/`validate_breakout_registration`
    treat this as the "registered segmented sequential" route
    `docs/reference/capabilities-by-entry-point.md` describes -- discrete (`kind="always_valid"`,
    Bernoulli law) registers `correction="bh"`; continuous
    (`kind="asymptotic_mean"`, scalar-mean law) registers
    `correction="bonferroni"` -- both derived automatically by
    `plan._compile_view_policy` from `kind`, not declared on the plan.

    Supported and matching on `from_unit_summary` and `from_unit_panel`,
    the two constructors whose sources can attach a caller-built
    `SequentialRegistration` at all. `from_definitions` IS given the same
    segmented registration (derived from its own automatic native-bound
    plan, the same way `from_unit_summary`'s is derived): confirmed live,
    `validate_relational_capture` (`increment/query/sequential_capture.py`,
    shared by `native_source.py` and `artifact_reader.py`) refuses that
    segmented roster at `capture_sequential` (`sequential.route.unsupported`,
    "relational segment capture needs a registered immutable property
    relation; use finalized joint records or frame segments"). Separately,
    `from_unit_day_artifact`'s adoption always re-derives its own
    automatic (unsegmented) registration from the experiment's declared
    plan (`artifact_context`), so a caller-built segmented registration
    can never reach it in the first place -- confirmed live with the
    automatic plan alone, `run_breakout` refuses
    `sequential.route.unsupported` ("a breakout readout requires one
    registered segment dimension") for that reason: an automatic
    registration carries no segment cells. `from_moments` IS given a
    replay of the same segmented `from_unit_summary` checkpoint (export
    survives the segmented roster; confirmed live, the replayed roster
    still names all three `store` cells), but `MomentsSource` declares no
    breakout catalog at all, so `run_breakout` refuses
    `readout.source.dimension` ("MomentsSource does not support breakout
    dimension 'store'; declared breakouts: []"), confirmed live.
    `from_switchback_panel` has no sequential construction at all, the
    same permanent SOURCE limitation every other sequential case here
    waives.

    `_segmented_breakout_boost_rows` gives segment `s0` a real treatment
    effect the other two segments do not have, so this family selects
    exactly one of its three members (see that function's docstring for
    the confirmed live evidence) -- `s0` discovers, `s1`/`s2` stay
    unselected at their fixed per-cell alpha. A path that selected a
    different segment, selected none, or selected all three would fail
    this case where it used to pass trivially at R=0.

    ``automatic=True`` declares the same family through the plan's predeclared
    ``InferenceSpec.segments`` (``store`` and its three levels) instead of a
    caller-built roster: every constructor binds the segmented registration
    itself, and the same source refusals apply for the same reasons.
    """
    stores = _BOOLEAN_SEGMENT_STORES if boolean_segments else _SEGMENTED_BREAKOUT_STORES
    rows = ds.event_rows(
        n_per_arm=_SEGMENTED_BREAKOUT_BOOST_N_PER_ARM
    ) + _segmented_breakout_boost_rows(
        kind, _SEGMENTED_BREAKOUT_BOOST_N_PER_ARM, _SEGMENTED_BREAKOUT_BOOST_STORE
    )
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    inference: dict[str, Any] = (
        {"kind": kind}
        if kind == "always_valid"
        else {"kind": kind, "expected_decision_sample_size": 100}
    )
    if automatic:
        inference["segments"] = {"store": stores}
    automatic_plan = AnalysisPlan(primary=metric, inference=inference)
    defs_dict = ds.definitions_dict(
        plan=automatic_plan, breakout=True, allocation={"control": 0.5, "treatment": 0.5}
    )
    defs_dict["dialect"] = dialect
    defs = Definitions.model_validate(defs_dict)
    exp = defs.experiment("exp")
    assert exp is not None
    as_of = ds._EXPERIMENT_END - dt.timedelta(days=1)
    panel_as_of = ds._EXPOSURE_AT.date() + dt.timedelta(days=1)

    def _native_bound_plan() -> AnalysisPlan:
        return _automatic_bound_plan(defs, exp, design)

    def _summary_frame(typed: Literal["pandas", "polars", "arrow"] | None = None) -> Any:
        table = _unit_summary_from_rows(rows)
        return _relabel_summary_store(table, typed) if boolean_segments else table

    def _summary_analysis(frame: Any, plan: AnalysisPlan) -> Analysis:
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            design=design,
            metrics=[spec_summary],
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_definitions() -> Analysis:
        # Segmented registration, derived from this source's own automatic
        # native-bound plan (or bound from the predeclared family) -- see
        # docstring: a relational source refuses this roster outright at
        # capture_sequential.
        con = warehouse_connection(rows)
        bound = _native_bound_plan()
        if not automatic:
            base_reg = bound.inference.registration  # ty: ignore[unresolved-attribute]
            bound = _segmented_plan_from(metric, kind, base_reg, stores)
        analysis = make_analysis(con, defs, experiment="exp", plan=bound)
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, defs, experiment="exp", plan=_native_bound_plan())
        adopted = _publish_and_adopt(con, native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    def _segmented_summary_plan() -> AnalysisPlan:
        if automatic:
            return automatic_plan
        summary = _unit_summary_from_rows(rows)
        base = Analysis.from_unit_summary(
            summary,
            unit="user_id",
            group="variant",
            design=design,
            metrics=[spec_summary],
            plan=automatic_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        base_reg = base.capture_sequential(finalized=True).registration
        base.close()
        return _segmented_plan_from(metric, kind, base_reg, stores)

    def build_unit_summary() -> Analysis:
        return _summary_analysis(_summary_frame(), _segmented_summary_plan())

    def build_unit_panel() -> Analysis:
        summary = _unit_summary_from_rows(rows)
        store_by_unit = {r["user_id"]: r["store"] for r in summary.to_pylist()}
        panel = _segmented_panel(rows, store_by_unit, metric, boolean_segments)
        segmented_plan = automatic_plan
        if not automatic:
            base = Analysis.from_unit_panel(
                panel,
                unit="user_id",
                group="variant",
                date="date",
                design=design,
                metrics=[spec_panel],
                plan=automatic_plan,
                experiment_id="exp",
                exposure_date="exposure_date",
            )
            base_reg = base.capture_sequential(finalized=True, as_of=panel_as_of).registration
            base.close()
            segmented_plan = _segmented_plan_from(metric, kind, base_reg, stores)
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=[spec_panel],
            plan=segmented_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        analysis._sequential_as_of = panel_as_of  # ty: ignore[unresolved-attribute]
        return analysis

    def build_moments() -> Analysis:
        # The exported segmented registration keeps the roster, so replayed
        # moments reach run_breakout and refuse there: MomentsSource declares no
        # breakout catalog.
        base = build_unit_summary()
        base.capture_sequential(finalized=True)
        return _export_and_replay(base, [MetricSpec(name=metric, type=spec_summary.type)])

    def typed_summary_probe() -> None:
        _assert_typed_summaries_match_oracle(
            _segmented_summary_plan(), _summary_frame, _summary_analysis
        )

    return ParityCase(
        id=id,
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        breakout_dimension="store",
        sequential=True,
        sequential_probe=typed_summary_probe if boolean_segments else None,
        waive={
            "from_definitions": (
                "SOURCE: given the same segmented registration as from_unit_summary, "
                "a relational source (validate_relational_capture, shared by "
                "native_source.py and artifact_reader.py) refuses that segmented "
                "sequential roster outright -- 'relational segment capture needs a "
                "registered immutable property relation; use finalized joint records "
                "or frame segments'."
            ),
            "from_unit_day_artifact": (
                "SOURCE: publish/adopt re-derives the experiment's declared plan's own "
                "automatic registration -- segmented from the predeclared family, so the "
                "relational capture check above refuses it for the same reason."
                if automatic
                else "SOURCE: publish/adopt always re-derives an automatic (unsegmented) "
                "registration from the experiment's declared plan -- a caller-built "
                "segmented registration never reaches it -- and the relational "
                "capture check above would refuse it regardless."
            ),
            "from_moments": (
                "SOURCE: MomentsSource declares no breakout catalog at all -- a "
                "replayed checkpoint carries the segmented registration intact "
                "(export survives it), but run_breakout still refuses because a "
                "moments cube has no breakout dimension to serve."
            ),
            **_SWITCHBACK_WAIVE,
        },
        waived_refusal_codes={
            "from_definitions": "sequential.route.unsupported",
            "from_unit_day_artifact": "sequential.route.unsupported",
            "from_moments": "readout.source.dimension",
        },
        # Captures a finalized sequential checkpoint twice (once to derive
        # the automatic registration, once segmented) on the dataframe
        # paths, and publishes/adopts a unit-day artifact plus exports/
        # replays moments -- see _fixed_horizon_case's slow=True for why.
        slow=True,
        require_selection=True,
    )


def _sequential_registered_breakout_discrete_case() -> ParityCase:
    return _registered_segmented_breakout_case(
        id="sequential_registered_breakout_discrete_bh",
        metric="purchase_rate",
        spec_summary=MetricSpec(name="purchase_rate", type="conversion", value_column="converted"),
        spec_panel=MetricSpec(name="purchase_rate", type="conversion", window_days=1),
        kind="always_valid",
    )


def _sequential_registered_breakout_continuous_case() -> ParityCase:
    return _registered_segmented_breakout_case(
        id="sequential_registered_breakout_continuous_bonferroni",
        metric="revenue",
        spec_summary=MetricSpec(name="revenue", type="mean", missing="zero"),
        spec_panel=MetricSpec(name="revenue", type="mean", missing="zero", window_days=1),
        kind="asymptotic_mean",
    )


def _sequential_automatic_breakout_discrete_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """`_sequential_registered_breakout_discrete_case` with the family predeclared
    on the plan (`InferenceSpec.segments`) instead of a caller-built roster; the
    warehouse constructors keep their relational segment-capture refusal."""
    return _registered_segmented_breakout_case(
        id="sequential_automatic_breakout_discrete_bh",
        metric="purchase_rate",
        spec_summary=MetricSpec(name="purchase_rate", type="conversion", value_column="converted"),
        spec_panel=MetricSpec(name="purchase_rate", type="conversion", window_days=1),
        kind="always_valid",
        automatic=True,
        warehouse_connection=warehouse_connection,
        dialect=dialect,
    )


def _sequential_automatic_breakout_continuous_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """`_sequential_registered_breakout_continuous_case` with the family predeclared
    on the plan; the asymptotic family keeps its fixed-roster Bonferroni
    construction and the warehouse constructors their segment-capture refusal."""
    return _registered_segmented_breakout_case(
        id="sequential_automatic_breakout_continuous_bonferroni",
        metric="revenue",
        spec_summary=MetricSpec(name="revenue", type="mean", missing="zero"),
        spec_panel=MetricSpec(name="revenue", type="mean", missing="zero", window_days=1),
        kind="asymptotic_mean",
        automatic=True,
        warehouse_connection=warehouse_connection,
        dialect=dialect,
    )


def _sequential_boolean_segment_case(
    kind: Literal["always_valid", "asymptotic_mean"], *, automatic: bool
) -> ParityCase:
    """The segmented breakout family over a typed Boolean/null dimension: the
    summary source carries canonical ``true``/``false``/``__null__`` strings (the
    oracle) while the panel source carries a nullable Boolean column, and a probe
    replays the summary in pandas, polars and Arrow dtypes. Raw snapshot identity
    is not compared across the differing source mappings; rows, counts, point
    estimates, interval geometry, selection and retained arm state are."""
    discrete = kind == "always_valid"
    metric = "purchase_rate" if discrete else "revenue"
    return _registered_segmented_breakout_case(
        id=(
            f"sequential_boolean_segments_{'automatic' if automatic else 'registered'}_"
            f"{'exact_bernoulli' if discrete else 'asymptotic_mean'}"
        ),
        metric=metric,
        spec_summary=(
            MetricSpec(name=metric, type="conversion", value_column="converted")
            if discrete
            else MetricSpec(name=metric, type="mean", missing="zero")
        ),
        spec_panel=(
            MetricSpec(name=metric, type="conversion", window_days=1)
            if discrete
            else MetricSpec(name=metric, type="mean", missing="zero", window_days=1)
        ),
        kind=kind,
        automatic=automatic,
        boolean_segments=True,
    )


def _automatic_bound_plan(defs: Definitions, exp: Any, design: Randomized) -> AnalysisPlan:
    """The experiment's plan with its automatic sequential registration bound to
    the native source mapping, as `Analysis.__init__` binds it."""
    bound = bind_automatic_sequential_plan(
        exp.plan,
        [m for m in defs.metrics if m.name in exp.metric_names],
        design=design,
        source_id=exp.name,
        source_mapping=native_observation_mapping(defs, exp, on_mixed_assignment="error"),
        pre_period_covariate=exp.n_pre_periods > 0,
    )
    assert bound is not None
    return bound


def _declared_registration_plans(
    defs: Definitions, exp: Any, plan: AnalysisPlan, design: Randomized, *, explicit: bool
) -> tuple[AnalysisPlan, Definitions]:
    """`plan` carrying a pre-data registration bound to the native source mapping
    (rewritten with a caller-computed definition id when *explicit*), and the
    definitions declaring it."""
    from increment.semantics.sequential import SequentialRegistration
    from increment.sequential_source import sequential_definition_id

    metrics = [m for m in defs.metrics if m.name in exp.metric_names]
    mapping = native_observation_mapping(defs, exp, on_mixed_assignment="error")
    registration = _automatic_bound_plan(defs, exp, design).inference.registration  # ty: ignore[unresolved-attribute]
    assert registration is not None
    if explicit:
        registration = SequentialRegistration.model_validate(
            {
                **registration.model_dump(),
                "definitions_id": sequential_definition_id(metrics, design, source_mapping=mapping),
            }
        )
    declared = plan.model_copy(
        update={"inference": InferenceSpec(kind="always_valid", registration=registration)}
    )
    declared_defs = defs.model_copy(
        update={"experiments": (exp.model_copy(update={"plan": declared}),)}
    )
    return declared, declared_defs


def _assert_shared_native_artifact_identity(
    con: Any, defs: Definitions, plan: AnalysisPlan, as_of: dt.date
) -> None:
    """A fresh native capture and an adopted artifact's capture share registration,
    arm state and record count (prefix id and record order depend on accumulation
    order, not identity)."""
    native = make_analysis(con, defs, experiment="exp", plan=plan)
    adopted = _publish_and_adopt(con, native)
    try:
        fresh = make_analysis(con, defs, experiment="exp", plan=plan)
        try:
            left = fresh.capture_sequential(finalized=True, as_of=as_of)
        finally:
            fresh.close()
        right = adopted.capture_sequential(finalized=True, as_of=as_of)
        assert left.registration == right.registration
        assert left.states == right.states
        assert len(left.records) == len(right.records)
    finally:
        _close_parity_analysis(adopted)


def _sequential_automatic_multi_arm_exact_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
    declared_registration: Literal["registered", "explicit"] | None = None,
) -> ParityCase:
    """Exact Bernoulli monitoring of a three-arm primary bound automatically from
    the declared allocation: one retained cell per treatment arm at the primary's
    level split across the arms (`alpha / 2`), the same registration on every
    constructor that reaches a sequential checkpoint. `from_switchback_panel`
    has no sequential construction (waived as everywhere).

    ``declared_registration`` instead puts a pre-data registration into the
    experiment's declared plan: ``"registered"`` is the automatically bound
    registration itself, ``"explicit"`` the same cells rewritten by the caller
    with a definition id it computed. Such a registration names the native
    source mapping, so the warehouse and artifact paths must capture identical
    registration and arm state, and the replay path independently replays the
    checkpoint exported from the native source, while the dataframe constructors
    refuse it with ``sequential.source.invalid`` rather than adopt a registration
    bound to another source.
    """
    rows = ds.multiplicity_event_rows()
    plan = AnalysisPlan(primary="purchase_rate", inference={"kind": "always_valid"})
    defs_dict = ds.multiplicity_definitions_dict(plan=plan)
    defs_dict["metrics"] = [m for m in defs_dict["metrics"] if m["name"] == "purchase_rate"]
    defs_dict["dialect"] = dialect
    defs = Definitions.model_validate(defs_dict)
    exp = defs.experiment("exp")
    assert exp is not None
    design = Randomized(control_group="control", allocation=ds.multiplicity_allocation())
    as_of = ds._EXPERIMENT_END - dt.timedelta(days=1)
    exposure_day = ds._EXPOSURE_AT.date()
    spec_summary = MetricSpec(name="purchase_rate", type="conversion", value_column="converted")
    spec_panel = MetricSpec(name="purchase_rate", type="conversion", window_days=1)
    native_defs = defs
    frame_plan = plan

    if declared_registration is not None:
        frame_plan, native_defs = _declared_registration_plans(
            defs, exp, plan, design, explicit=declared_registration == "explicit"
        )

    def _native_bound_plan() -> AnalysisPlan:
        if declared_registration is not None:
            return frame_plan
        return _automatic_bound_plan(defs, exp, design)

    def _assert_native_and_artifact_share_identity() -> None:
        con = warehouse_connection(rows)
        _assert_shared_native_artifact_identity(con, native_defs, _native_bound_plan(), as_of)

    def build_definitions() -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, native_defs, experiment="exp", plan=_native_bound_plan())
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, native_defs, experiment="exp", plan=_native_bound_plan())
        adopted = _publish_and_adopt(con, native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    def _summary() -> pd.DataFrame:
        con = ds.duckdb_connection(rows)
        frame = ds.multiplicity_summary_frame(con).to_pandas()
        con.disconnect()
        frame["exposure_date"] = exposure_day
        return frame

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            _summary(),
            unit="user_id",
            group="variant",
            design=design,
            metrics=[spec_summary],
            plan=frame_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_unit_panel() -> Analysis:
        # The conversion lands on the exposure day, inside window_days=1.
        summary = _summary()
        panel = pd.DataFrame(
            [
                {
                    "user_id": r["user_id"],
                    "variant": r["variant"],
                    "exposure_date": exposure_day,
                    "date": exposure_day + dt.timedelta(days=offset),
                    "purchase_rate": int(r["converted"]) if offset == 0 else 0,
                }
                for r in summary.to_dict("records")
                for offset in (0, 1)
            ]
        )
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=[spec_panel],
            plan=frame_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        analysis._sequential_as_of = exposure_day + dt.timedelta(days=1)  # ty: ignore[unresolved-attribute]
        return analysis

    def build_moments() -> Analysis:
        if declared_registration is None:
            return _export_and_replay(
                build_unit_summary(), [MetricSpec(name="purchase_rate", type="conversion")]
            )
        # The registration is bound to the native source, so the portable cube
        # is exported from that source and replayed independently of the frames.
        native = build_definitions()
        try:
            native.capture_sequential(finalized=True, as_of=as_of)
            return _export_and_replay(native, [MetricSpec(name="purchase_rate", type="conversion")])
        finally:
            native.close()
            for connection in getattr(native, "_parity_connections", ()):
                connection.disconnect()

    frame_refusal = dict.fromkeys(
        ("from_unit_summary", "from_unit_panel"), "sequential.source.invalid"
    )
    return ParityCase(
        id=(
            "sequential_automatic_multi_arm_exact"
            if declared_registration is None
            else f"sequential_{declared_registration}_registration_native_artifact_identity"
        ),
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        sequential=True,
        sequential_probe=(
            None if declared_registration is None else _assert_native_and_artifact_share_identity
        ),
        waive={
            **_SWITCHBACK_WAIVE,
            **(
                dict.fromkeys(
                    frame_refusal,
                    "SOURCE: a registration declared against the native source mapping "
                    "is bound to that source's identity and is never adopted by a frame source",
                )
                if declared_registration is not None
                else {}
            ),
        },
        waived_refusal_codes=frame_refusal if declared_registration is not None else {},
        # Publishes/adopts a unit-day artifact, exports/replays moments and
        # captures a finalized sequential checkpoint on every constructor.
        slow=True,
    )


_ENCOURAGEMENT_EXPOSURE_AT = dt.datetime(2025, 1, 10, 9)
_ENCOURAGEMENT_CLICK_AT = dt.datetime(2025, 1, 10, 12)
_ENCOURAGEMENT_PURCHASE_AT = dt.datetime(2025, 1, 10, 15)
# Same freshness-padding technique as dataset.py's own `_FRESHNESS_PAD_AT`:
# `_censor_to_observable_window` bases a metric's freshness bound on the
# fact's OWN latest observed timestamp, not the declared `end`.
_ENCOURAGEMENT_FRESHNESS_PAD_AT = dt.datetime(2025, 1, 19, 9)
_ENCOURAGEMENT_EXPERIMENT_END = dt.date(2025, 1, 20)
_ENCOURAGEMENT_N_PER_ARM = 20


def _encouragement_revenue(i: int, *, clicked: bool) -> float:
    # Genuine within-arm variance: a constant control arm makes
    # family.evidence.incomplete refuse instead of exercising the row.
    return 8.0 + (i % 4) + (5.0 if clicked else 0.0)


def _encouragement_rows_for_parity(
    n_per_arm: int = _ENCOURAGEMENT_N_PER_ARM,
) -> list[dict[str, Any]]:
    """A randomized-encouragement dataset on `dataset.py`'s shared `_row`/
    `duckdb_connection` shape: exposure, an uptake ("clicked") event for
    half of treatment, and an in-window purchase for every unit."""
    control = [f"c{i}" for i in range(1, n_per_arm + 1)]
    treat = [f"t{i}" for i in range(1, n_per_arm + 1)]
    clickers = {f"t{i}" for i in range(1, n_per_arm // 2 + 1)}
    rows: list[dict[str, Any]] = [
        ds._row(u, _ENCOURAGEMENT_EXPOSURE_AT, "exposure", group_id="control", experiment_id="exp")
        for u in control
    ]
    rows += [
        ds._row(
            u, _ENCOURAGEMENT_EXPOSURE_AT, "exposure", group_id="treatment", experiment_id="exp"
        )
        for u in treat
    ]
    rows += [ds._row(u, _ENCOURAGEMENT_CLICK_AT, "clicked") for u in clickers]
    for i, u in enumerate(control + treat, start=1):
        clicked = u in clickers
        rows.append(
            ds._row(
                u,
                _ENCOURAGEMENT_PURCHASE_AT,
                "purchase",
                revenue=_encouragement_revenue(i, clicked=clicked),
            )
        )
        rows.append(ds._row(u, _ENCOURAGEMENT_FRESHNESS_PAD_AT, "purchase", revenue=0.0))
    return rows


def _encouragement_defs_dict(plan: AnalysisPlan) -> dict[str, Any]:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "clicked", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "increase",
            }
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2025-01-10",
                "end": _ENCOURAGEMENT_EXPERIMENT_END.isoformat(),
                "control_group": "control",
                "allocation": {"control": 0.5, "treatment": 0.5},
                "plan": plan.model_dump(mode="json"),
                "design": {
                    "mechanism": "encouragement",
                    "uptake": {"fact": "clicked"},
                    "one_sided": True,
                    "exclusion_restriction": {
                        "acknowledged": True,
                        "justification": "assignment moves revenue only via uptake",
                    },
                },
            }
        ],
    }


def _encouragement_oracle_frame_for_parity(*, missing_treatment_outcomes: bool = False) -> Any:
    import pandas as pd

    control = [f"c{i}" for i in range(1, _ENCOURAGEMENT_N_PER_ARM + 1)]
    treat = [f"t{i}" for i in range(1, _ENCOURAGEMENT_N_PER_ARM + 1)]
    clickers = {f"t{i}" for i in range(1, _ENCOURAGEMENT_N_PER_ARM // 2 + 1)}
    records = []
    for i, u in enumerate(control + treat, start=1):
        clicked = u in clickers
        records.append(
            {
                "user_id": u,
                "group": "control" if u.startswith("c") else "treatment",
                "clicked": int(clicked),
                "revenue": None
                if missing_treatment_outcomes and u in treat
                else _encouragement_revenue(i, clicked=clicked),
            }
        )
    return pd.DataFrame(records)


def _encouragement_oracle_panel_frame_for_parity(
    *, missing_treatment_outcomes: bool = False
) -> Any:
    frame = _encouragement_oracle_frame_for_parity(
        missing_treatment_outcomes=missing_treatment_outcomes
    )
    frame["day"] = _ENCOURAGEMENT_PURCHASE_AT.date()
    return frame


def _uptake_timestamp_boundaries_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """Match raw timestamp uptake to its elapsed-day and unit-level reductions."""
    plan = AnalysisPlan(secondaries=["revenue"])
    start = _ENCOURAGEMENT_EXPOSURE_AT.replace(hour=12)
    events = _encouragement_rows_for_parity()
    for row in events:
        row["event_at"] += dt.timedelta(hours=3)
        if row["event"] == "clicked":
            index = int(row["user_id"][1:])
            row["event_at"] = start + dt.timedelta(hours=0 if index <= 5 else 21)
    events.extend(ds._row(f"t{i}", start + dt.timedelta(days=1), "clicked") for i in range(11, 16))
    events.append(ds._row("t16", start - dt.timedelta(minutes=1), "clicked"))
    definition = _encouragement_defs_dict(plan)
    definition["dialect"] = dialect
    definition["experiments"][0]["design"]["uptake"]["window_days"] = 1
    definitions = Definitions.model_validate(definition)
    design = definitions.experiments[0].resolved_design()
    spec = MetricSpec(name="revenue", preferred_direction="increase")
    frame = _encouragement_oracle_frame_for_parity()

    def native(*, artifact: bool = False) -> Analysis:
        con = warehouse_connection(events)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)

    def summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="group",
            metrics=[spec],
            design=design,
            uptake="clicked",
            plan=plan,
        )

    def panel() -> Analysis:
        records = []
        closing_units = {f"t{i}" for i in range(11, 16)}
        for row in frame.to_dict("records"):
            for day in range(3):
                # The frame input is reduced by elapsed exposure day, not calendar date.
                records.append(
                    {
                        **row,
                        "day": day,
                        "exposure": 0,
                        "revenue": row["revenue"] if day == 0 else 0.0,
                        "clicked": (
                            row["clicked"]
                            if day == 0
                            else float(day == 1 and row["user_id"] in closing_units)
                        ),
                    }
                )
        return Analysis.from_unit_panel(
            pd.DataFrame(records),
            unit="user_id",
            group="group",
            date="day",
            exposure_date="exposure",
            observation_end=2,
            metrics=[spec],
            design=design,
            uptake="clicked",
            plan=plan,
        )

    def portable() -> Analysis:
        with summary() as analysis, tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            analysis.export(path)
            return Analysis.from_moments(
                pq.read_table(path).to_pylist(), metrics=[spec], design=design, plan=plan
            )

    def probe(results: Any) -> None:
        rows = {row.estimand: row for row in results}
        assert set(rows) == {"itt", "compliance", "late"}
        for estimand, value in (("itt", 2.5 / 9.5), ("compliance", 0.5), ("late", 5.0 / 9.5)):
            assert rows[estimand].lift is not None
            assert rows[estimand].lift.value == pytest.approx(value)
            assert rows[estimand].lift.lb is not None
            assert rows[estimand].lift.ub is not None

    def asof_probe(constructor: str, analysis: Analysis) -> None:
        if constructor not in {"from_definitions", "from_unit_day_artifact", "from_unit_panel"}:
            return
        results = analysis.run_asof_lift(estimands=("itt", "compliance", "late"))
        last = max(row.ds for row in results)
        probe([row for row in results if row.ds == last])

    return ParityCase(
        id="audit-uptake-timestamp-boundaries",
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": summary,
            "from_unit_panel": panel,
            "from_moments": portable,
        },
        estimands=("itt", "compliance", "late"),
        readout_probe=probe,
        source_probe=asof_probe,
        waive=dict(_SWITCHBACK_WAIVE),
        slow=True,
    )


def _assert_missing_exclusion_precedes_data(_constructor: str, analysis: Analysis) -> None:
    from unittest.mock import patch

    from increment import IdentificationError

    def unread(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A missing LATE assumption must refuse before reading data")

    source_type = type(_moment_source(analysis))
    with (
        patch.object(source_type, "moments", unread),
        patch.object(source_type, "compliance_summary", unread),
    ):
        for estimands in (None, ("late",), ("itt", "late")):
            with pytest.raises(IdentificationError) as raised:
                if estimands is None:
                    analysis.run()
                else:
                    analysis.run(estimands=estimands)
            assert raised.value.code == "identification.encouragement.exclusion_required"
            assert raised.value.context["estimands"] == (estimands or ("itt", "compliance", "late"))


def _encouragement_declared_definitions_case(
    *,
    missing_treatment_outcomes: bool = False,
    metric_free: bool = False,
    exclusion_declared: bool = True,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
    supplied_design: bool = False,
) -> ParityCase:
    """Compare declared encouragement, including outcome-independent compliance.

    ``supplied_design`` leaves the experiment without a declared design and hands
    the same encouragement design to the native source separately (the accepted
    ``_design`` form); the published artifact must carry it whole, so every
    ITT/compliance/LATE row and interval matches the declared form.
    """
    rows = _encouragement_rows_for_parity()
    plan = AnalysisPlan(secondaries=[] if metric_free else ["revenue"])
    defs_dict = _encouragement_defs_dict(plan)
    defs_dict["dialect"] = dialect
    if not exclusion_declared:
        del defs_dict["experiments"][0]["design"]["exclusion_restriction"]
    if supplied_design:
        del defs_dict["experiments"][0]["design"]
    if missing_treatment_outcomes:
        treated = {row["user_id"] for row in rows if row["group_id"] == "treatment"}
        rows = [
            row for row in rows if not (row["user_id"] in treated and row["event"] == "purchase")
        ]
        defs_dict["metrics"][0]["aggregation"] = "avg_event"
    specs = [
        MetricSpec(name="revenue", type="mean", missing="drop", preferred_direction="increase")
        if missing_treatment_outcomes
        else MetricSpec(name="revenue", type="mean", preferred_direction="increase")
    ]
    if metric_free:
        defs_dict["metrics"] = []
        specs = []
    defs = Definitions.model_validate(defs_dict)

    def supplied() -> dict[str, Any]:
        return {"_design": _design()} if supplied_design else {}

    def build_definitions() -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp", **supplied())
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, defs, experiment="exp", **supplied())
        return _publish_and_adopt(con, native)

    def _design() -> Encouragement:
        return Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment moves revenue only via uptake"
            )
            if exclusion_declared
            else None,
            one_sided=True,
            allocation={"control": 0.5, "treatment": 0.5},
        )

    def build_unit_summary() -> Analysis:
        needs_drop = missing_treatment_outcomes and not metric_free
        with pytest.warns(IncrementWarning) if needs_drop else nullcontext():
            return Analysis.from_unit_summary(
                _encouragement_oracle_frame_for_parity(
                    missing_treatment_outcomes=missing_treatment_outcomes
                ),
                unit="user_id",
                group="group",
                metrics=specs,
                design=_design(),
                uptake="clicked",
                plan=plan,
            )

    def build_unit_panel() -> Analysis:
        return Analysis.from_unit_panel(
            _encouragement_oracle_panel_frame_for_parity(
                missing_treatment_outcomes=missing_treatment_outcomes
            ),
            unit="user_id",
            group="group",
            date="day",
            metrics=specs,
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_moments() -> Analysis:
        # `_export_and_replay` reimports with `control="control"` (a plain
        # Randomized design) -- wrong here, where the point is preserving
        # the declared Encouragement design through export/reimport.
        con = warehouse_connection(rows)
        native = make_analysis(con, defs, experiment="exp", **supplied())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            native.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(replay_rows, metrics=specs, design=_design(), plan=plan)
        native.close()
        con.disconnect()
        return replayed

    return ParityCase(
        id="encouragement_supplied_design_definitions"
        if supplied_design
        else "encouragement_optional_exclusion"
        if not exclusion_declared
        else "audit-compliance-only-empty-catalog"
        if metric_free
        else "audit-compliance-only-outcome-missingness"
        if missing_treatment_outcomes
        else "encouragement_declared_definitions",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=("compliance",)
        if missing_treatment_outcomes or metric_free
        else ("itt", "compliance")
        if not exclusion_declared
        else ("itt", "compliance", "late"),
        source_probe=_assert_missing_exclusion_precedes_data if not exclusion_declared else None,
        waive={
            "from_switchback_panel": "COMBINATION: no switchback schedule in this dataset",
            **(
                {
                    "from_unit_panel": "CONSTRUCTION: panel densification has no complete-case "
                    "missing='drop' path; zero-filling would change the declared outcome policy"
                }
                if missing_treatment_outcomes and not metric_free
                else {}
            ),
        },
        waived_refusal_codes={"from_unit_panel": "frame.missing_policy.panel_drop"}
        if missing_treatment_outcomes and not metric_free
        else {},
        slow=True,
    )


def _guardrail_latency(i: int) -> float:
    # Within-arm variance on both arms, so the guardrail cell is estimable.
    return 100.0 + (i % 3)


def _encouragement_guardrail_case(
    *, id: str, estimands: tuple[str, ...], margin_abs: float | None = None
) -> ParityCase:
    """A secondary plus a `latency` guardrail under a declared Encouragement
    design: the guardrail reports exactly the requested estimands, and the
    same rows, on every constructor. `margin_abs`, when given, declares a
    non-inferiority margin on the guardrail -- its verdict rides the ITT
    row (`_encouragement_margin_guardrail_case`); without it (the plain
    `latency` string form) the guardrail is an ordinary unmargined test."""
    rows = _encouragement_rows_for_parity()
    control = [f"c{i}" for i in range(1, _ENCOURAGEMENT_N_PER_ARM + 1)]
    treat = [f"t{i}" for i in range(1, _ENCOURAGEMENT_N_PER_ARM + 1)]
    for i, u in enumerate(control + treat, start=1):
        rows.append(
            ds._row(u, _ENCOURAGEMENT_PURCHASE_AT, "latency_sample", latency=_guardrail_latency(i))
        )
        rows.append(ds._row(u, _ENCOURAGEMENT_FRESHNESS_PAD_AT, "latency_sample", latency=0.0))
    guardrail = (
        ExperimentMetric(metric="latency", margin_abs=margin_abs)
        if margin_abs is not None
        else "latency"
    )
    plan = AnalysisPlan(secondaries=["revenue"], guardrails=[guardrail])
    defs_dict = _encouragement_defs_dict(plan)
    defs_dict["fact_sources"][0]["facts"].append({"name": "latency_sample", "column": "latency"})
    defs_dict["metrics"].append(
        {
            "type": "mean",
            "name": "latency",
            "entity": "user_id",
            "fact": "latency_sample",
            "aggregation": "sum",
            "window_days": 1,
            "preferred_direction": "decrease",
        }
    )
    defs = Definitions.model_validate(defs_dict)

    def frame() -> Any:
        summary = _encouragement_oracle_frame_for_parity()
        summary["latency"] = [
            _guardrail_latency(i) for i in range(1, 2 * _ENCOURAGEMENT_N_PER_ARM + 1)
        ]
        return summary

    def panel_frame() -> Any:
        panel = frame()
        panel["day"] = _ENCOURAGEMENT_PURCHASE_AT.date()
        return panel

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp")
        return _publish_and_adopt(con, native)

    def _design() -> Encouragement:
        return Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment moves revenue only via uptake"
            ),
            one_sided=True,
            allocation={"control": 0.5, "treatment": 0.5},
        )

    specs = [
        MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
        MetricSpec(name="latency", type="mean", preferred_direction="decrease"),
    ]

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame(),
            unit="user_id",
            group="group",
            metrics=specs,
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_unit_panel() -> Analysis:
        return Analysis.from_unit_panel(
            panel_frame(),
            unit="user_id",
            group="group",
            date="day",
            metrics=specs,
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_moments() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            native.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(replay_rows, metrics=specs, design=_design(), plan=plan)
        native.close()
        con.disconnect()
        return replayed

    return ParityCase(
        id=id,
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=estimands,
        waive={"from_switchback_panel": "COMBINATION: no switchback schedule in this dataset"},
        slow=True,
    )


def _encouragement_guardrail_late_rows_case() -> ParityCase:
    return _encouragement_guardrail_case(
        id="encouragement_guardrail_late_rows", estimands=("itt", "compliance", "late")
    )


def _encouragement_guardrail_late_only_case() -> ParityCase:
    return _encouragement_guardrail_case(
        id="encouragement_guardrail_late_only_request", estimands=("late",)
    )


def _encouragement_margin_guardrail_case() -> ParityCase:
    """The margined-guardrail cell the capabilities-by-entry-point table actually claims: `latency`
    declares `margin_abs=5.0`, and a full-estimands request (itt, compliance,
    late) succeeds identically on every reachable path, with the margin
    verdict riding the ITT row. The late-only refusal this margin also
    produces (`readout.encouragement.margin`) needs a request excluding
    itt, which leaves nothing left to compare on any path -- covered instead
    by `test_encouragement_margin_guardrail_refuses_late_only_on_every_constructor`
    in tests/test_readouts_encouragement.py, not a second ParityCase."""
    return _encouragement_guardrail_case(
        id="encouragement_margin_guardrail",
        estimands=("itt", "compliance", "late"),
        margin_abs=5.0,
    )


def _retention_breakout_cohorts_case(
    *,
    single_segment: bool = False,
    late_units: int = 2,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """Mature binary retention, repeated returns, and excluded late cohorts."""
    stores = ("s0",) if single_segment else ("s0", "s1")
    rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for arm in ("control", "treatment"):
        for store_index, store in enumerate(stores):
            for index in range(20 + late_units):
                unit = f"{arm}-{store_index}-{index}"
                exposed = dt.datetime(2025, 1, 10 if index < 20 else 15, 9)
                returns = index % 4 < (1 if arm == "control" else 2)
                rows.append(
                    ds._row(
                        unit,
                        exposed,
                        "exposure",
                        group_id=arm,
                        experiment_id="exp",
                        store_id=store,
                    )
                )
                for day in range(10):
                    occurrences = 2 if day == 7 and returns else int(day in (6, 9))
                    for occurrence in range(occurrences):
                        rows.append(
                            ds._row(
                                unit,
                                exposed + dt.timedelta(days=day, hours=occurrence),
                                "purchase",
                                store_id=store,
                                revenue=1.0,
                            )
                        )
                    panel_rows.append(
                        {
                            "user_id": unit,
                            "group": arm,
                            "store": store,
                            "day": exposed.date() + dt.timedelta(days=day),
                            "exposed_on": exposed.date(),
                            "returned": float(occurrences),
                        }
                    )
                summary_rows.append(
                    {
                        "user_id": unit,
                        "group": arm,
                        "returned": float(returns),
                    }
                )

    plan = AnalysisPlan(secondaries=["d7_retention"])
    defs_dict = ds.definitions_dict(plan=plan, breakout=True)
    defs_dict["dialect"] = dialect
    defs_dict["experiments"][0]["end"] = "2025-01-16"
    defs_dict["experiments"][0]["observation_end"] = "2025-01-20"
    defs_dict["metrics"] = [
        {
            "type": "retention",
            "name": "d7_retention",
            "entity": "user_id",
            "fact": "purchase",
            "threshold_days": [7, 8],
            "preferred_direction": "increase",
        }
    ]
    definitions = Definitions.model_validate(defs_dict)
    metric = MetricSpec(
        name="d7_retention",
        type="retention",
        value_column="returned",
        threshold_days=(7, 8),
        preferred_direction="increase",
    )

    def native(*, artifact: bool = False) -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)

    def panel() -> Analysis:
        return Analysis.from_unit_panel(
            pd.DataFrame(panel_rows),
            unit="user_id",
            group="group",
            date="day",
            exposure_date="exposed_on",
            observation_end=dt.date(2025, 1, 20),
            metrics=[metric],
            breakouts=["store"],
            plan=plan,
            control="control",
        )

    def summary() -> Analysis:
        return Analysis.from_unit_summary(
            pd.DataFrame(summary_rows),
            unit="user_id",
            group="group",
            metrics=[metric],
            plan=plan,
            control="control",
        )

    def replay() -> Analysis:
        return _export_and_replay(native(), [metric])

    def probe(results: Any) -> None:
        assert {(row.metric, row.dimension_value) for row in results} == {
            ("d7_retention", store) for store in stores
        }
        for row in results:
            assert row.lift is not None
            assert row.lift.value == pytest.approx(1.0)
            assert row.lift.lb is not None and row.lift.ub is not None

    def source_probe(name: str, analysis: Analysis) -> None:
        if name not in {"from_definitions", "from_unit_day_artifact"}:
            return
        tables = analysis.breakout_summaries(metrics=["d7_retention"])
        key = "d7_retention:store:events"
        assert set(tables) == {key}
        expected = {
            (arm, store): (20, 5 if arm == "control" else 10)
            for arm in ("control", "treatment")
            for store in stores
        }
        for table in ("group_summary", "daily_group_summary"):
            records = tables[key][table].to_pylist()
            assert {(row["group_id"], row["store"]) for row in records} == set(expected)
            for row in records:
                n, successes = expected[(row["group_id"], row["store"])]
                assert row["n"] == n
                assert n * row["ref_y"] + row["cy1"] == pytest.approx(successes)
                second = row["cy2"] + 2 * row["ref_y"] * row["cy1"] + n * row["ref_y"] ** 2
                assert second == pytest.approx(successes)
            if table == "daily_group_summary":
                assert {row["ds"] for row in records} == {dt.date(2025, 1, 10)}
        if single_segment:
            from increment.estimation.results import LiftEstimate

            pooled = analysis.run(metrics=["d7_retention"])
            segmented = analysis.run_breakout()
            for segment, total in zip(segmented, pooled, strict=True):
                assert isinstance(total, LiftEstimate)
                assert segment.group_id == total.group_id
                assert segment.lift is not None and total.lift is not None
                assert segment.lift.value == pytest.approx(total.lift.value)
                assert segment.lift.lb == pytest.approx(total.lift.lb)
                assert segment.lift.ub == pytest.approx(total.lift.ub)

    return ParityCase(
        id="audit-retention-breakout-cohorts" + ("-single-segment" if single_segment else ""),
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_panel": panel,
            "from_unit_summary": summary,
            "from_moments": replay,
        },
        breakout_dimension="store",
        waive={
            **_SWITCHBACK_WAIVE,
            "from_unit_summary": "SOURCE: unit summaries cannot supply retention exposure cohorts.",
            "from_moments": "SOURCE: portable moments retain no breakout operation or cohort identity.",
        },
        waived_refusal_codes={
            "from_unit_summary": "source.frame.constructor",
            "from_moments": "facade.analysis.operation",
        },
        readout_probe=probe,
        source_probe=source_probe,
        slow=True,
    )


def _encouragement_retention_case() -> ParityCase:
    """A retention metric declared as a secondary under an Encouragement
    design: not a matching ParityCase (retention under encouragement is
    never estimable on any reachable path -- see docstring below), but a
    builder every reachable constructor can attempt, reused by
    `test_encouragement_retention_refuses_on_every_reachable_constructor`
    in tests/test_readouts_encouragement.py the same way
    `_encouragement_guardrail_case` is reused by that file's late-only
    margin probe.

    Confirmed live, every reachable constructor refuses, and
    `from_definitions`, `from_unit_day_artifact`, `from_moments` and
    `from_unit_panel` all share `readout.encouragement.retention`
    (retention's maturity gate): the first three check it in
    `validate_readout_encouragement`, the seam every view of `.run()`
    shares regardless of source; `from_unit_panel` reaches the same
    shared refusal spec earlier, at CONSTRUCTION, in
    `FramePanelSource.from_frame`'s own gate on the raw declared metrics,
    before an Encouragement-designed source is ever built -- same
    hazard, same code, different entry point, matching AGENTS.md's
    "Refusals" contract. `from_unit_summary` refuses for a genuinely
    unrelated reason regardless of design (`source.frame.constructor` --
    retention needs `threshold_days` resolved against real per-unit
    dates, which a one-row-per-unit summary never has, encouragement or
    not); that divergence is inherent, not a gap."""
    rows = _encouragement_rows_for_parity()
    plan = AnalysisPlan(secondaries=["revenue", "returned"])
    defs_dict = _encouragement_defs_dict(plan)
    defs_dict["metrics"].append(
        {
            "type": "retention",
            "name": "returned",
            "entity": "user_id",
            "fact": "purchase",
            "threshold_days": 1,
        }
    )
    defs = Definitions.model_validate(defs_dict)

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp")
        return _publish_and_adopt(con, native)

    def _design() -> Encouragement:
        return Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment moves revenue only via uptake"
            ),
            one_sided=True,
            allocation={"control": 0.5, "treatment": 0.5},
        )

    def build_unit_summary() -> Analysis:
        summary = _encouragement_oracle_frame_for_parity()
        return Analysis.from_unit_summary(
            summary,
            unit="user_id",
            group="group",
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="returned", type="retention", threshold_days=1),
            ],
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_unit_panel() -> Analysis:
        panel = _encouragement_oracle_panel_frame_for_parity()
        panel["returned"] = (panel["revenue"] > 0).astype(int)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="group",
            date="day",
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="returned", type="retention", threshold_days=1),
            ],
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_moments() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            native.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(
            replay_rows,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="returned", type="retention", threshold_days=1),
            ],
            design=_design(),
            plan=plan,
        )
        native.close()
        con.disconnect()
        return replayed

    return ParityCase(
        id="encouragement_retention_probe",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive={"from_switchback_panel": "COMBINATION: no switchback schedule in this dataset"},
        slow=True,
    )


_FCR_N_PER_ARM = 30


def _fcr_revenue(i: int, *, arm: str) -> float:
    # Within-arm variance (i % 3) plus a genuine arm-level shift for both
    # treatment arms. errors (below) gets no shift, so only revenue is
    # selected -- this is the smallest shape where BH's corrected level
    # differs from nominal.
    return 8.0 + (i % 3) + (5.0 if arm in ("a", "b") else 0.0)


def _fcr_errors(i: int) -> float:
    # No arm effect: errors must stay unselected, at nominal level, while
    # revenue moves to the FCR-corrected one.
    return 1.0 if i % 3 == 0 else 0.0


def _fcr_row(
    user_id: str,
    event_at: dt.datetime,
    event: str,
    *,
    group_id: str | None = None,
    revenue: float | None = None,
    errors: float | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "event_at": event_at,
        "event": event,
        "group_id": group_id,
        "experiment_id": "exp",
        "revenue": revenue,
        "errors": errors,
    }


def _fcr_clickers() -> set[str]:
    return {f"{arm}{i}" for arm in ("a", "b") for i in range(1, (2 * _FCR_N_PER_ARM // 3) + 1)}


def _fcr_reestimation_rows() -> list[dict[str, Any]]:
    """Three arms (control/a/b), two secondaries -- revenue moves (and is
    selected) in both treatment arms, errors moves in neither -- the
    smallest shape where BH's corrected level (0.985) differs from
    nominal (0.95), which the single-arm, single-secondary D1/D2 rows
    never exercise (BH's m=1 never differs from nominal)."""
    clickers = _fcr_clickers()
    rows: list[dict[str, Any]] = []
    for arm in ("control", "a", "b"):
        for i in range(1, _FCR_N_PER_ARM + 1):
            uid = f"{arm}{i}"
            rows.append(_fcr_row(uid, _ENCOURAGEMENT_EXPOSURE_AT, "exposure", group_id=arm))
            if uid in clickers:
                rows.append(_fcr_row(uid, _ENCOURAGEMENT_CLICK_AT, "clicked"))
            rows.append(
                _fcr_row(
                    uid,
                    _ENCOURAGEMENT_PURCHASE_AT,
                    "purchase",
                    revenue=_fcr_revenue(i, arm=arm),
                )
            )
            rows.append(
                _fcr_row(uid, _ENCOURAGEMENT_PURCHASE_AT, "error_event", errors=_fcr_errors(i))
            )
            rows.append(_fcr_row(uid, _ENCOURAGEMENT_FRESHNESS_PAD_AT, "purchase", revenue=0.0))
            rows.append(_fcr_row(uid, _ENCOURAGEMENT_FRESHNESS_PAD_AT, "error_event", errors=0.0))
    return rows


def _fcr_reestimation_defs_dict(plan: AnalysisPlan) -> dict[str, Any]:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "clicked", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "error_event", "column": "errors"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "increase",
            },
            {
                "type": "mean",
                "name": "errors",
                "entity": "user_id",
                "fact": "error_event",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "decrease",
            },
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2025-01-10",
                "end": _ENCOURAGEMENT_EXPERIMENT_END.isoformat(),
                "control_group": "control",
                "allocation": {"control": 1 / 3, "a": 1 / 3, "b": 1 / 3},
                "plan": plan.model_dump(mode="json"),
                "design": {
                    "mechanism": "encouragement",
                    "uptake": {"fact": "clicked"},
                    "one_sided": True,
                    "exclusion_restriction": {
                        "acknowledged": True,
                        "justification": "assignment moves revenue only via uptake",
                    },
                },
            }
        ],
    }


def _fcr_oracle_frame_for_parity() -> Any:
    import pandas as pd

    clickers = _fcr_clickers()
    records = []
    for arm in ("control", "a", "b"):
        for i in range(1, _FCR_N_PER_ARM + 1):
            uid = f"{arm}{i}"
            records.append(
                {
                    "user_id": uid,
                    "group": arm,
                    "clicked": int(uid in clickers),
                    "revenue": _fcr_revenue(i, arm=arm),
                    "errors": _fcr_errors(i),
                }
            )
    return pd.DataFrame(records)


def _encouragement_fcr_reestimation_case() -> ParityCase:
    """A selected secondary's LATE row must carry the same FCR-corrected
    level as its ITT row, and both must agree across every constructor --
    the D1/D2 rows' single-arm, single-secondary shape never exercises BH
    selection (m=1 never differs from nominal)."""
    rows = _fcr_reestimation_rows()
    plan = AnalysisPlan(secondaries=["revenue", "errors"], q=0.03)
    defs_dict = _fcr_reestimation_defs_dict(plan)
    defs = Definitions.model_validate(defs_dict)

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, defs, experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp")
        return _publish_and_adopt(con, native)

    def _design() -> Encouragement:
        return Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment moves revenue only via uptake"
            ),
            one_sided=True,
            allocation={"control": 1 / 3, "a": 1 / 3, "b": 1 / 3},
        )

    specs = [
        MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
        MetricSpec(name="errors", type="mean", preferred_direction="decrease"),
    ]

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            _fcr_oracle_frame_for_parity(),
            unit="user_id",
            group="group",
            metrics=specs,
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_unit_panel() -> Analysis:
        panel = _fcr_oracle_frame_for_parity()
        panel["day"] = _ENCOURAGEMENT_PURCHASE_AT.date()
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="group",
            date="day",
            metrics=specs,
            design=_design(),
            uptake="clicked",
            plan=plan,
        )

    def build_moments() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, defs, experiment="exp")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            native.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(replay_rows, metrics=specs, design=_design(), plan=plan)
        native.close()
        con.disconnect()
        return replayed

    return ParityCase(
        id="encouragement_fcr_reestimation",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=("itt", "compliance", "late"),
        waive={"from_switchback_panel": "COMBINATION: no switchback schedule in this dataset"},
        slow=True,
    )


_MULTIPLICITY_PLAN = AnalysisPlan(
    alpha=0.05,
    q=0.30,
    primary="revenue",
    secondaries=("purchase_rate", "rps"),
    guardrails=(ExperimentMetric(metric="latency", margin=0.5),),
)


def _multiplicity_roles_case() -> ParityCase:
    """Capability rows must include multiplicity-bearing cases: >=2
    secondaries with at least one selected, a guardrail, multi-arm
    primary. One case exercises all three axes
    together against three arms (`control`, `treatment_a`, `treatment_b`):
    `revenue` is the multi-arm primary (Bonferroni-split across both
    non-control arms), `purchase_rate`/`rps` are secondaries under a loose
    q=0.30 BH family (`purchase_rate`'s and `rps`'s `treatment_a` cells
    clear the threshold, `purchase_rate`'s `treatment_b` cell does not --
    both a discovered and an undiscovered secondary row are present), and
    `latency` is a guardrail (`preferred_direction="decrease"`, so its
    `alternative` reads "less"). Verified with a bounded probe before
    writing this file: role/discovery/value agree exactly across
    from_definitions, from_unit_day_artifact, from_unit_summary,
    from_unit_panel and from_moments; `from_switchback_panel` needs an
    unrelated schedule shape, so it is recorded in `waive` rather than
    attempted. No metric here declares a covariate, so from_unit_panel needs
    no CUPED waiver -- every one of the six constructors is either built or
    waived, with no silent gap."""
    rows = ds.multiplicity_event_rows()
    defs_dict = ds.multiplicity_definitions_dict(plan=_MULTIPLICITY_PLAN)
    design = Randomized(control_group="control", allocation=ds.multiplicity_allocation())

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        return _publish_and_adopt(con, native)

    metrics = [
        MetricSpec(name="revenue", type="mean", missing="zero", preferred_direction="increase"),
        MetricSpec(
            name="purchase_rate",
            type="conversion",
            value_column="converted",
            preferred_direction="increase",
        ),
        MetricSpec(
            name="rps",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            missing="zero",
            preferred_direction="increase",
        ),
        MetricSpec(name="latency", type="mean", missing="zero", preferred_direction="decrease"),
    ]

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.multiplicity_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            design=design,
            metrics=metrics,
            plan=_MULTIPLICITY_PLAN,
        )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.multiplicity_summary_frame(con)
        con.disconnect()  # `summary`/`panel` are already materialized below
        panel = ds.multiplicity_panel_frame(summary)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=metrics,
            plan=_MULTIPLICITY_PLAN,
        )

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        return _export_and_replay(
            summary,
            [
                MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
                MetricSpec(name="purchase_rate", type="conversion", preferred_direction="increase"),
                MetricSpec(
                    name="rps",
                    type="ratio",
                    numerator="revenue",
                    denominator="sessions",
                    preferred_direction="increase",
                ),
                MetricSpec(name="latency", type="mean", preferred_direction="decrease"),
            ],
        )

    return ParityCase(
        id="multiplicity_primary_secondaries_guardrail",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive=dict(_SWITCHBACK_WAIVE),
        # Publishes/adopts a unit-day artifact and exports/replays moments
        # on every run -- see _fixed_horizon_case's slow=True for why.
        slow=True,
    )


_OBSERVATIONAL_PLAN = AnalysisPlan(primary="revenue", secondaries=("errors", "signups"))


def _observational_covariate_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Three arms, each unit with a pre-exposure numeric ``tenure`` that
    drives both its arm and its outcomes; returns event rows plus per-unit truth."""
    import numpy as np

    rng = np.random.default_rng(11)
    exposure_at = dt.datetime(2025, 1, 10, 9)
    rows: list[dict[str, Any]] = []
    units: dict[str, dict[str, Any]] = {}
    for i in range(90):
        uid = f"u{i}"
        tenure = float(rng.normal(100, 15))
        arm = ("control", "treat_a", "treat_b")[(i + (tenure > 105) + (tenure > 115)) % 3]
        treated = 0.0 if arm == "control" else 1.0
        unit = {
            "variant": arm,
            "tenure": tenure,
            "revenue": 20.0 + 0.05 * tenure + 2.0 * treated + float(rng.normal(0, 1)),
            "errors": 1.0 + 0.01 * tenure + 0.1 * treated + float(rng.normal(0, 0.5)),
            "signups": 0.5 + 0.002 * tenure + 0.3 * treated + float(rng.normal(0, 0.4)),
        }
        units[uid] = unit
        blank = {"revenue": None, "errors": None, "signups": None, "tenure": None}
        rows.append(
            {
                "user_id": uid,
                "event_at": exposure_at,
                "event": "exposure",
                "experiment_id": "obs_exp",
                "group_id": arm,
                **blank,
            }
        )
        rows.append(
            {
                "user_id": uid,
                "event_at": exposure_at - dt.timedelta(days=14),
                "event": "profile",
                "experiment_id": None,
                "group_id": None,
                **{**blank, "tenure": tenure},
            }
        )
        rows.extend(
            {
                "user_id": uid,
                "event_at": exposure_at + dt.timedelta(days=days),
                "event": "profile",
                "experiment_id": None,
                "group_id": None,
                **{**blank, "tenure": other_tenure},
            }
            for days, other_tenure in ((-21, float(50 + i * 17 % 37)), (1, float(500 + i * 7 % 53)))
        )
        rows.extend(
            {
                "user_id": uid,
                "event_at": exposure_at + dt.timedelta(hours=6),
                "event": f"purchase_{name}",
                "experiment_id": None,
                "group_id": None,
                **{**blank, name: unit[name]},
            }
            for name in ("revenue", "errors", "signups")
        )
        # After the declared end, so every unit's window is observed.
        rows.append(
            {
                "user_id": uid,
                "event_at": dt.datetime(2025, 1, 25),
                "event": "profile",
                "experiment_id": None,
                "group_id": None,
                **blank,
            }
        )
    return rows, units


def _observational_covariate_case() -> ParityCase:
    """IPTW adjustment on a declared numeric pre-exposure covariate with a
    two-treatment-arm primary and two secondaries: from_definitions,
    from_unit_day_artifact and from_unit_panel must all reproduce the
    dataframe oracle (from_unit_summary) row for row. from_unit_panel
    collapses to a per-unit total for this unwindowed metric and reads the
    covariate (constant within each unit); from_moments is attempted and
    refuses by its own code -- a moments cube has no per-unit rows to
    attach a covariate to."""
    rows, units = _observational_covariate_rows()
    metric_def = {"type": "mean", "entity": "user_id", "aggregation": "sum"}
    defs_dict: dict[str, Any] = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase_revenue", "column": "revenue"},
                    {"name": "purchase_errors", "column": "errors"},
                    {"name": "purchase_signups", "column": "signups"},
                ],
                "properties": [
                    {
                        "name": "tenure",
                        "column": "tenure",
                        "dtype": "float",
                        "as_of": "pre_exposure",
                    }
                ],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "exposure"}],
        "metrics": [
            {"name": name, "fact": f"purchase_{name}", **metric_def}
            for name in ("revenue", "errors", "signups")
        ],
        "experiments": [
            {
                "name": "obs_exp",
                "exposure": "enrolled",
                "unit": "user_id",
                "control_group": "control",
                "start": "2025-01-01",
                "end": "2025-01-12",
                "plan": {"primary": "revenue", "secondaries": ["errors", "signups"]},
                "design": {
                    "mechanism": "observational",
                    "covariates": [{"property": "tenure", "source": "events"}],
                },
            }
        ],
    }
    specs = [MetricSpec(name=name, type="mean") for name in ("revenue", "errors", "signups")]

    def design() -> Observational:
        return Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        )

    def frame() -> Any:
        import pandas as pd

        return pd.DataFrame([{"user_id": uid, **unit} for uid, unit in units.items()])

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _publish_and_adopt(con, native, kinds=("unit_covariate",))

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame(),
            unit="user_id",
            group="variant",
            metrics=specs,
            design=design(),
            plan=_OBSERVATIONAL_PLAN,
        )

    def build_unit_panel() -> Analysis:
        panel = frame()
        panel["date"] = dt.date(2025, 1, 10)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            metrics=specs,
            design=design(),
            plan=_OBSERVATIONAL_PLAN,
        )

    def build_moments() -> Analysis:
        # `_export_and_replay` reimports as Randomized; the observational
        # design must survive the replay to reach the covariate read.
        summary = build_unit_summary()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        summary.close()
        return Analysis.from_moments(
            replay_rows, metrics=specs, design=design(), plan=_OBSERVATIONAL_PLAN
        )

    return ParityCase(
        id="observational_iptw_covariate_adjusted_ate",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive={
            "from_moments": (
                "SOURCE: a moments cube holds no per-unit rows to attach a covariate to."
            ),
            "from_switchback_panel": (
                "SOURCE: a per-unit adjustment set has no analogue on a switchback "
                "block/period schedule, and this dataset has none."
            ),
        },
        waived_refusal_codes={
            "from_moments": "source.moments.covariate_unavailable",
        },
        # Publishes/adopts a unit-day artifact on every run.
        slow=True,
    )


_OBSERVATIONAL_QUANTILE_PLAN = AnalysisPlan(primary="revenue")


def _observational_quantile_case() -> ParityCase:
    """The observational quantile estimator refuses before reading outcomes on every
    ingress that can declare it. The portable case declares a quantile over a real scalar
    moments cube: those moments cannot estimate a quantile, and must not be used as one."""
    rows, units = _observational_covariate_rows()
    defs_dict: dict[str, Any] = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase_revenue", "column": "revenue"},
                ],
                "properties": [
                    {
                        "name": "tenure",
                        "column": "tenure",
                        "dtype": "float",
                        "as_of": "pre_exposure",
                    }
                ],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "exposure"}],
        "metrics": [
            {
                "name": "revenue",
                "type": "quantile",
                "entity": "user_id",
                "fact": "purchase_revenue",
                "aggregation": "sum",
                "quantile": 0.5,
            }
        ],
        "experiments": [
            {
                "name": "obs_exp",
                "exposure": "enrolled",
                "unit": "user_id",
                "control_group": "control",
                "start": "2025-01-01",
                "end": "2025-01-12",
                "plan": {"primary": "revenue"},
                "design": {
                    "mechanism": "observational",
                    "covariates": [{"property": "tenure", "source": "events"}],
                },
            }
        ],
    }
    specs = [MetricSpec(name="revenue", type="quantile", quantile=0.5)]

    def design() -> Observational:
        return Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        )

    def frame() -> Any:
        return pd.DataFrame([{"user_id": uid, **unit} for uid, unit in units.items()])

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _publish_and_adopt(con, native, kinds=("unit_covariate",))

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame(),
            unit="user_id",
            group="variant",
            metrics=specs,
            design=design(),
            plan=_OBSERVATIONAL_QUANTILE_PLAN,
        )

    def build_unit_panel() -> Analysis:
        panel = frame()
        panel["date"] = dt.date(2025, 1, 10)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            metrics=specs,
            design=design(),
            plan=_OBSERVATIONAL_QUANTILE_PLAN,
        )

    def build_moments() -> Analysis:
        # Export scalar moments, then ask for the unsupported observational quantile.
        summary = Analysis.from_unit_summary(
            frame(),
            unit="user_id",
            group="variant",
            metrics=[MetricSpec(name="revenue", type="mean")],
            design=Randomized(control_group="control"),
        )
        try:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "moments.parquet"
                summary.export(path)
                replay_rows = pq.read_table(path).to_pylist()
        finally:
            summary.close()
        return Analysis.from_moments(
            replay_rows, metrics=specs, design=design(), plan=_OBSERVATIONAL_QUANTILE_PLAN
        )

    seam = "readout.observational.quantile"
    return ParityCase(
        id="observational_quantile_refused_at_the_readout_seam",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive={
            "from_definitions": f"CONSTRUCTION: an observational design has no quantile estimator ({seam})",
            "from_unit_day_artifact": f"CONSTRUCTION: no quantile estimator under an observational design ({seam})",
            "from_unit_summary": f"CONSTRUCTION: no quantile estimator under an observational design ({seam})",
            "from_unit_panel": f"CONSTRUCTION: no quantile estimator under an observational design ({seam})",
            "from_moments": (
                f"CONSTRUCTION: the quantile declaration over scalar moments must refuse "
                f"its absent observational estimator before using those moments ({seam})."
            ),
            "from_switchback_panel": (
                "SOURCE: no design= parameter, and a quantile metric is refused at the "
                "switchback metric-type gate (source.frame.switchback.metric)."
            ),
        },
        waived_refusal_codes={
            "from_definitions": seam,
            "from_unit_day_artifact": seam,
            "from_unit_summary": seam,
            "from_unit_panel": seam,
            "from_moments": seam,
        },
        refusal_only=True,
        # Publishes/adopts a unit-day artifact on every run.
        slow=True,
    )


_OBSERVATIONAL_MULTIPLICITY_PLAN = AnalysisPlan(
    primary="revenue", secondaries=["signups"], guardrails=["errors"]
)


def _observational_multiplicity_case() -> ParityCase:
    """Observational role multiplicity: `revenue` primary, `signups`
    secondary, `errors` guardrail (`preferred_direction="decrease"`, full
    plan alpha, no family correction) under a declared IPTW-adjusted
    Observational design -- reuses `_observational_covariate_case`'s own
    dataset (`_observational_covariate_rows`) with `errors` promoted from
    secondary to guardrail. Moved here from `scripts/probe_capability_table.py`
    (`_row_observational_multiplicity`) so a regression on any attempted
    path fails a real test tier, not just a manually-run probe."""
    rows, units = _observational_covariate_rows()
    metric_def = {"type": "mean", "entity": "user_id", "aggregation": "sum"}
    defs_dict: dict[str, Any] = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase_revenue", "column": "revenue"},
                    {"name": "purchase_errors", "column": "errors"},
                    {"name": "purchase_signups", "column": "signups"},
                ],
                "properties": [
                    {
                        "name": "tenure",
                        "column": "tenure",
                        "dtype": "float",
                        "as_of": "pre_exposure",
                    }
                ],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "exposure"}],
        "metrics": [
            {
                "name": "revenue",
                "fact": "purchase_revenue",
                "preferred_direction": "increase",
                **metric_def,
            },
            {
                "name": "signups",
                "fact": "purchase_signups",
                "preferred_direction": "increase",
                **metric_def,
            },
            {
                "name": "errors",
                "fact": "purchase_errors",
                "preferred_direction": "decrease",
                **metric_def,
            },
        ],
        "experiments": [
            {
                "name": "obs_exp",
                "exposure": "enrolled",
                "unit": "user_id",
                "control_group": "control",
                "start": "2025-01-01",
                "end": "2025-01-12",
                "plan": {
                    "primary": "revenue",
                    "secondaries": ["signups"],
                    "guardrails": ["errors"],
                },
                "design": {
                    "mechanism": "observational",
                    "covariates": [{"property": "tenure", "source": "events"}],
                },
            }
        ],
    }
    specs = [
        MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
        MetricSpec(name="signups", type="mean", preferred_direction="increase"),
        MetricSpec(name="errors", type="mean", preferred_direction="decrease"),
    ]

    def design() -> Observational:
        return Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        )

    def frame() -> Any:
        return pd.DataFrame([{"user_id": uid, **unit} for uid, unit in units.items()])

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _publish_and_adopt(con, native, kinds=("unit_covariate",))

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame(),
            unit="user_id",
            group="variant",
            metrics=specs,
            design=design(),
            plan=_OBSERVATIONAL_MULTIPLICITY_PLAN,
        )

    def build_unit_panel() -> Analysis:
        panel = frame()
        panel["date"] = dt.date(2025, 1, 10)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            metrics=specs,
            design=design(),
            plan=_OBSERVATIONAL_MULTIPLICITY_PLAN,
        )

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        summary.close()
        return Analysis.from_moments(
            replay_rows, metrics=specs, design=design(), plan=_OBSERVATIONAL_MULTIPLICITY_PLAN
        )

    return ParityCase(
        id="observational_multiplicity_primary_secondary_guardrail",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive={
            "from_moments": "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
            "from_switchback_panel": (
                "SOURCE: a per-unit adjustment set has no analogue on a switchback "
                "block/period schedule, and this dataset has none."
            ),
        },
        waived_refusal_codes={"from_moments": "source.moments.covariate_unavailable"},
        # Publishes/adopts a unit-day artifact on every run.
        slow=True,
    )


def _observational_aipw_dml_case(
    *,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
    default_sensitivity: bool = False,
) -> ParityCase:
    """AIPW and DML both depend on cross-fit fold assignment and unit-frame
    row order, unlike the default IPTW path `_observational_covariate_case`
    already covers -- exactly the methods whose cross-path agreement needs
    proving (AGENTS.md: 'Prove path parity, not path presence'). Reuses
    `_observational_covariate_case`'s dataset with `revenue` (primary)
    declared `decision_method=aipw` and `errors` (secondary) declared
    `decision_method=dml`; `signups` (secondary) stays default IPTW so a
    mixed-method row set is exercised too. from_moments is attempted and
    refuses by its own code, same as the sibling IPTW case."""
    rows, units = _observational_covariate_rows()
    revenue_methods = (
        {"sensitivity_methods": [{"name": "unadjusted"}]}
        if default_sensitivity
        else {"decision_method": {"name": "aipw"}}
    )
    metric_def = {"type": "mean", "entity": "user_id", "aggregation": "sum"}
    defs_dict: dict[str, Any] = {
        "dialect": dialect,
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase_revenue", "column": "revenue"},
                    {"name": "purchase_errors", "column": "errors"},
                    {"name": "purchase_signups", "column": "signups"},
                ],
                "properties": [
                    {
                        "name": "tenure",
                        "column": "tenure",
                        "dtype": "float",
                        "as_of": "pre_exposure",
                    }
                ],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "exposure"}],
        "metrics": [
            {"name": name, "fact": f"purchase_{name}", **metric_def}
            for name in ("revenue", "errors", "signups")
        ],
        "experiments": [
            {
                "name": "obs_exp",
                "exposure": "enrolled",
                "unit": "user_id",
                "control_group": "control",
                "start": "2025-01-01",
                "end": "2025-01-12",
                "plan": {
                    "primary": {"metric": "revenue", **revenue_methods},
                    "secondaries": [
                        {"metric": "errors", "decision_method": {"name": "dml"}},
                        "signups",
                    ],
                },
                "design": {
                    "mechanism": "observational",
                    "covariates": [{"property": "tenure", "source": "events"}],
                },
            }
        ],
    }
    specs = [
        MetricSpec.model_validate({"name": "revenue", "type": "mean", **revenue_methods}),
        MetricSpec(name="errors", type="mean", decision_method={"name": "dml"}),
        MetricSpec(name="signups", type="mean"),
    ]
    plan = AnalysisPlan(primary="revenue", secondaries=("errors", "signups"))

    def design() -> Observational:
        return Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        )

    def frame() -> Any:
        return pd.DataFrame([{"user_id": uid, **unit} for uid, unit in units.items()])

    def build_definitions() -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_exp")
        return _publish_and_adopt(con, native, kinds=("unit_covariate",))

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame(), unit="user_id", group="variant", metrics=specs, design=design(), plan=plan
        )

    def build_unit_panel() -> Analysis:
        panel = frame()
        panel["date"] = dt.date(2025, 1, 10)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            metrics=specs,
            design=design(),
            plan=plan,
        )

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        summary.close()
        return Analysis.from_moments(replay_rows, metrics=specs, design=design(), plan=plan)

    return ParityCase(
        id=(
            "observational_default_with_unadjusted_sensitivity"
            if default_sensitivity
            else "observational_aipw_dml_covariate_adjusted"
        ),
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive={
            "from_moments": "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
            "from_switchback_panel": (
                "SOURCE: a per-unit adjustment set has no analogue on a switchback "
                "block/period schedule, and this dataset has none."
            ),
        },
        waived_refusal_codes={"from_moments": "source.moments.covariate_unavailable"},
        # Publishes/adopts a unit-day artifact on every run.
        slow=True,
    )


_CATEGORICAL_LEVELS = ("east", "west", "north")


def _observational_categorical_rows(
    *, missing_levels: bool = False
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Two arms; every unit carries a pre-exposure numeric ``tenure`` and a
    categorical ``region`` (modal ``east``) that both nudge its arm and its
    outcomes. Decoy ``region`` records before and after the pre-exposure one
    carry a level (``south``) absent from the truth, so a path resolving the
    wrong record would emit different rows. With *missing_levels*, the first
    five units of each arm have no pre-exposure ``region`` (NULL, never a
    level of its own); the unit is still enrolled and keeps its outcomes.
    Returns event rows plus per-unit truth."""
    import numpy as np

    rng = np.random.default_rng(23)
    exposure_at = dt.datetime(2025, 1, 10, 9)
    rows: list[dict[str, Any]] = []
    units: dict[str, dict[str, Any]] = {}
    seen: dict[str, int] = {"control": 0, "treatment": 0}
    for i in range(200):
        uid = f"u{i}"
        tenure = float(rng.normal(100, 15))
        region = _CATEGORICAL_LEVELS[int(rng.choice(3, p=(0.55, 0.30, 0.15)))]
        west, north = float(region == "west"), float(region == "north")
        score = 0.02 * (tenure - 100) + 0.5 * west - 0.4 * north + float(rng.normal(0, 1.0))
        arm = "treatment" if score > 0 else "control"
        treated = 1.0 if arm == "treatment" else 0.0
        missing = missing_levels and seen[arm] < 5
        seen[arm] += 1
        unit = {
            "variant": arm,
            "tenure": tenure,
            "region": None if missing else region,
            "revenue": 20.0
            + 0.05 * tenure
            + 1.5 * west
            - 0.8 * north
            + 2.0 * treated
            + float(rng.normal(0, 1)),
            "errors": 1.0
            + 0.01 * tenure
            + 0.2 * west
            - 0.1 * north
            + 0.1 * treated
            + float(rng.normal(0, 0.5)),
            "signups": 0.5
            + 0.002 * tenure
            + 0.1 * west
            + 0.3 * treated
            + float(rng.normal(0, 0.4)),
        }
        units[uid] = unit
        blank = {"revenue": None, "errors": None, "signups": None, "tenure": None, "region": None}
        rows.append(
            {
                "user_id": uid,
                "event_at": exposure_at,
                "event": "exposure",
                "experiment_id": "obs_cat",
                "group_id": arm,
                **blank,
            }
        )
        rows.append(
            {
                "user_id": uid,
                "event_at": exposure_at - dt.timedelta(days=14),
                "event": "profile",
                "experiment_id": None,
                "group_id": None,
                **{**blank, "tenure": tenure, "region": unit["region"]},
            }
        )
        rows.extend(
            {
                "user_id": uid,
                "event_at": exposure_at + dt.timedelta(days=days),
                "event": "profile",
                "experiment_id": None,
                "group_id": None,
                **{**blank, "tenure": other_tenure, "region": "south"},
            }
            for days, other_tenure in ((-21, float(50 + i * 17 % 37)), (1, float(500 + i * 7 % 53)))
        )
        rows.extend(
            {
                "user_id": uid,
                "event_at": exposure_at + dt.timedelta(hours=6),
                "event": f"purchase_{name}",
                "experiment_id": None,
                "group_id": None,
                **{**blank, name: unit[name]},
            }
            for name in ("revenue", "errors", "signups")
        )
        # After the declared end, so every unit's window is observed.
        rows.append(
            {
                "user_id": uid,
                "event_at": dt.datetime(2025, 1, 25),
                "event": "profile",
                "experiment_id": None,
                "group_id": None,
                **blank,
            }
        )
    return rows, units


def _observational_categorical_case(
    *,
    missing_levels: bool = False,
    warehouse_connection: Callable[[list[dict[str, Any]]], Any] = ds.duckdb_connection,
    dialect: str = "duckdb",
) -> ParityCase:
    """A categorical (string) adjustment covariate declared beside a numeric
    one on the ordinary `AdjustmentSet`, encoded inside every nuisance fit:
    `revenue` (primary) on the default IPTW path, `errors` (secondary) on
    AIPW and `signups` (secondary) on DML. from_definitions joins the string
    property as a string column, from_unit_day_artifact publishes it on the
    `unit_covariate_level` relation, and both must reproduce the dataframe
    oracle (from_unit_summary) and from_unit_panel row for row; from_moments
    is attempted and refuses by its own code, as in the numeric siblings.

    With *missing_levels*, ten units carry a NULL region. The frame
    constructors declare ``missing="impute-indicator"`` and keep every
    unit; the definitions grammar declares no missing policy, so
    from_definitions and from_unit_day_artifact resolve the default
    ``missing="refuse"`` and must refuse by the identification code that
    names the null -- proving the NULL reached them as a missing value, not
    as a level (a path serving ``"None"`` or dropping the unit would emit
    rows instead and fail the waiver)."""
    rows, units = _observational_categorical_rows(missing_levels=missing_levels)
    metric_def = {"type": "mean", "entity": "user_id", "aggregation": "sum"}
    defs_dict: dict[str, Any] = {
        "dialect": dialect,
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase_revenue", "column": "revenue"},
                    {"name": "purchase_errors", "column": "errors"},
                    {"name": "purchase_signups", "column": "signups"},
                ],
                "properties": [
                    {
                        "name": "tenure",
                        "column": "tenure",
                        "dtype": "float",
                        "as_of": "pre_exposure",
                    },
                    {
                        "name": "region",
                        "column": "region",
                        "dtype": "string",
                        "as_of": "pre_exposure",
                    },
                ],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "exposure"}],
        "metrics": [
            {"name": name, "fact": f"purchase_{name}", **metric_def}
            for name in ("revenue", "errors", "signups")
        ],
        "experiments": [
            {
                "name": "obs_cat",
                "exposure": "enrolled",
                "unit": "user_id",
                "control_group": "control",
                "start": "2025-01-01",
                "end": "2025-01-12",
                "plan": {
                    "primary": "revenue",
                    "secondaries": [
                        {"metric": "errors", "decision_method": {"name": "aipw"}},
                        {"metric": "signups", "decision_method": {"name": "dml"}},
                    ],
                },
                "design": {
                    "mechanism": "observational",
                    "covariates": [
                        {"property": "tenure", "source": "events"},
                        {"property": "region", "source": "events"},
                    ],
                },
            }
        ],
    }
    specs = [
        MetricSpec(name="revenue", type="mean"),
        MetricSpec(name="errors", type="mean", decision_method={"name": "aipw"}),
        MetricSpec(name="signups", type="mean", decision_method={"name": "dml"}),
    ]
    plan = AnalysisPlan(primary="revenue", secondaries=("errors", "signups"))

    def design() -> Observational:
        return Observational(
            control_group="control",
            adjustment=AdjustmentSet(
                covariates=("tenure", "region"),
                missing="impute-indicator" if missing_levels else "refuse",
            ),
        )

    def frame() -> Any:
        return pd.DataFrame([{"user_id": uid, **unit} for uid, unit in units.items()])

    def build_definitions() -> Analysis:
        con = warehouse_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_cat")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = warehouse_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="obs_cat")
        return _publish_and_adopt(con, native, kinds=("unit_covariate", "unit_covariate_level"))

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            frame(), unit="user_id", group="variant", metrics=specs, design=design(), plan=plan
        )

    def build_unit_panel() -> Analysis:
        panel = frame()
        panel["date"] = dt.date(2025, 1, 10)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            metrics=specs,
            design=design(),
            plan=plan,
        )

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            replay_rows = pq.read_table(path).to_pylist()
        summary.close()
        return Analysis.from_moments(replay_rows, metrics=specs, design=design(), plan=plan)

    waive = {
        "from_moments": "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
        "from_switchback_panel": (
            "SOURCE: a per-unit adjustment set has no analogue on a switchback "
            "block/period schedule, and this dataset has none."
        ),
    }
    waived_refusal_codes = {"from_moments": "source.moments.covariate_unavailable"}
    if missing_levels:
        reason = (
            "SOURCE: the definitions grammar declares no missing-covariate policy, so "
            "this path resolves the default missing='refuse' and must name the NULL level."
        )
        for name in ("from_definitions", "from_unit_day_artifact"):
            waive[name] = reason
            waived_refusal_codes[name] = "adjust.identification.missing_covariates"
    return ParityCase(
        id=(
            "observational_categorical_covariate_missing_levels"
            if missing_levels
            else "observational_categorical_covariate_adjusted"
        ),
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive=waive,
        waived_refusal_codes=waived_refusal_codes,
        # Publishes/adopts a unit-day artifact on every run.
        slow=True,
    )


_NONPOSITIVE_MEAN_PLAN = AnalysisPlan(primary=["revenue"], secondaries=["refunds", "converted"])


def _nonpositive_mean_case() -> ParityCase:
    """`refunds` (secondary) has no `refund` events at all in treatment --
    the additive-only-row hazard `engine.py`'s `_nonpositive_mean_additive_row`
    covers: `lift=None`, `relative_unavailable_reason='nonpositive_arm_mean'`,
    a real `abs_diff`/`abs_lb`/`abs_ub`. `revenue` (primary) and `converted`
    (the other secondary) stay ordinary and positive in both arms. No CUPED
    covariate is declared, so -- unlike `_mean_ratio_conversion_cuped_case` --
    every non-switchback constructor is reachable and must match the
    additive row's full payload identically, not merely its presence."""
    rows = ds.nonpositive_mean_event_rows()
    defs_dict = ds.nonpositive_mean_definitions_dict(plan=_NONPOSITIVE_MEAN_PLAN)

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        return _publish_and_adopt(con, native)

    summary_metrics = [
        MetricSpec(name="revenue", type="mean", missing="zero", preferred_direction="increase"),
        MetricSpec(name="refunds", type="mean", missing="zero", preferred_direction="decrease"),
        MetricSpec(
            name="converted",
            type="conversion",
            preferred_direction="increase",
        ),
    ]

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.nonpositive_mean_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=summary_metrics,
            plan=_NONPOSITIVE_MEAN_PLAN,
        )

    def build_unit_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.nonpositive_mean_summary_frame(con)
        con.disconnect()  # `summary`/`panel` are already materialized below
        panel = ds.nonpositive_mean_panel_frame(summary)
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            control="control",
            metrics=summary_metrics,
            plan=_NONPOSITIVE_MEAN_PLAN,
        )

    def build_moments() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.nonpositive_mean_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        summary = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=summary_metrics,
            plan=_NONPOSITIVE_MEAN_PLAN,
        )
        return _export_and_replay(
            summary,
            [
                MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
                MetricSpec(name="refunds", type="mean", preferred_direction="decrease"),
                MetricSpec(name="converted", type="conversion", preferred_direction="increase"),
            ],
        )

    return ParityCase(
        id="nonpositive_mean_additive_row_matches_across_sources",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive=dict(_SWITCHBACK_WAIVE),
        # Publishes/adopts a unit-day artifact and exports/replays moments
        # on every run -- see _fixed_horizon_case's slow=True for why.
        slow=True,
    )


_CLUSTERED_NEGATIVE_MEAN_PLAN = AnalysisPlan(primary=["net_revenue"])


def _clustered_negative_mean_case(*, zero_treatment: bool = False) -> ParityCase:
    """A clustered, signed-mean ratio contrast -- the capability restored
    by moving the ratio positivity guard off the shared `ratio_moments`
    reducer and onto the log-scale `RatioVarianceModel.log_mean_se`
    (`increment/estimation/variance.py`): the clustered Fieller route
    never took a log in the first place, so it never needed that guard.
    Every control-arm store's `net_revenue` numerator total is negative;
    treatment-arm totals are positive unless testing zero relative variance.
    The point remains defined when its relative uncertainty is unavailable.

    `from_moments` is attempted and MUST raise, coded: exporting a
    clustered, non-Encouragement source's moments is a real, pre-existing
    SOURCE gap (`increment/sources.py`'s `export_source_moments`: "cluster
    transport requires a complete encouragement compliance payload") --
    verified live on this checkout, independent of ratio metrics or this
    fix. `from_unit_panel` is waived outright (not attempted): a
    per-day panel is collapsed to one row per unit before clustering could
    apply, and `Analysis.from_unit_panel(cluster=...)` refuses
    unconditionally with `source.frame_panel.cluster_grain`
    (`increment/frame.py`) -- the same pre-existing, structural gap the
    non-positive-arm-mean row above already waives for the same reason."""
    rows = ds.clustered_negative_mean_event_rows(zero_treatment=zero_treatment)
    plan = (
        AnalysisPlan(secondaries=["net_revenue"])
        if zero_treatment
        else _CLUSTERED_NEGATIVE_MEAN_PLAN
    )
    defs_dict = ds.clustered_negative_mean_definitions_dict(plan=plan)

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        return _publish_and_adopt(con, native)

    summary_metrics = [
        MetricSpec(
            name="net_revenue",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            preferred_direction="increase",
        ),
    ]

    def build_unit_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.clustered_negative_mean_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=summary_metrics,
            cluster="cluster_id",
            plan=plan,
        )

    def build_moments() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.clustered_negative_mean_summary_frame(con)
        con.disconnect()  # `frame` is already materialized; nothing below needs `con` alive
        summary = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=summary_metrics,
            cluster="cluster_id",
            plan=plan,
        )
        return _export_and_replay(
            summary,
            [
                MetricSpec(
                    name="net_revenue", type="ratio", numerator="revenue", denominator="sessions"
                )
            ],
        )

    return ParityCase(
        id=(
            "clustered_zero_relative_variance_preserves_point_and_additive_interval"
            if zero_treatment
            else "clustered_negative_control_mean_produces_a_fieller_set"
        ),
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_moments": build_moments,
        },
        waive={
            "from_moments": (
                "SOURCE: export_source_moments refuses a clustered, "
                "non-Encouragement source ('cluster transport requires a "
                "complete encouragement compliance payload') -- the moments "
                "wire format carries no cluster marker, so from_moments "
                "would rehydrate the cluster rows as unit-grain ratio "
                "moments and silently misread them."
            ),
            "from_unit_panel": (
                "SOURCE cannot supply a clustered arm-lift ratio readout: "
                "Analysis.from_unit_panel(cluster=...) refuses "
                "unconditionally with source.frame_panel.cluster_grain -- "
                "the panel is collapsed to one row per unit before "
                "clustering could apply, and the collapse is ambiguous on "
                "this shape (increment/frame.py); build the clustered arm "
                "totals with from_unit_summary(cluster=...) instead."
            ),
            "from_switchback_panel": (
                "SOURCE cannot supply a clustered arm-lift ratio readout: a "
                "switchback panel declares a shared-schedule contrast "
                "(ContrastResults), a structurally different estimand "
                "shape from the LiftEstimate rows this case compares."
            ),
        },
        waived_refusal_codes={"from_moments": "source.moments.cluster_grain"},
        # Publishes/adopts a unit-day artifact and exports/replays moments
        # on every run -- see _fixed_horizon_case's slow=True for why.
        slow=True,
    )


def _cluster_export_refusal_case() -> ParityCase:
    """Keep direct clustered inference after every lossy export refuses."""
    base = _clustered_negative_mean_case()
    routes = {
        "from_definitions": "native",
        "from_unit_day_artifact": "artifact",
        "from_unit_summary": "frame",
    }

    def checked(build: Callable[[], Analysis], source: str) -> Callable[[], Analysis]:
        def construct() -> Analysis:
            analysis = build()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / "existing.parquet"
                    original = b"preserve the existing destination"
                    destination.write_bytes(original)
                    with pytest.raises(CodedError) as refused:
                        analysis.export(destination)
                    assert refused.value.code == "source.moments.cluster_grain"
                    assert refused.value.context["operation"] == "export_moments"
                    assert refused.value.context["source"] == source
                    assert destination.read_bytes() == original
            except BaseException:
                analysis.close()
                for connection in getattr(analysis, "_parity_connections", ()):
                    connection.disconnect()
                raise
            return analysis

        return construct

    return replace(
        base,
        id="audit-cluster-export-refusal",
        build={
            name: checked(build, routes[name]) if name in routes else build
            for name, build in base.build.items()
        },
    )


_ENCOURAGEMENT_ITT_DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="uptake"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="uptake does not gate the conversion outcome"
    ),
)
_ENCOURAGEMENT_ITT_N_PER_ARM = 90


def _encouragement_itt_metrics() -> list[MetricSpec]:
    return [MetricSpec(name="converted", type="conversion", preferred_direction="increase")]


def _encouragement_itt_rows() -> list[dict[str, Any]]:
    """Deterministic per-unit conversion/uptake rows. Control never takes
    up the encouragement (`uptake=0` for every control unit, matching a
    real encouragement design where only treatment is offered); treatment
    has a mixed uptake cohort. `converted` differs by arm so the ITT is
    non-degenerate. Neither column depends on the other's value per unit,
    keeping `uptake` a genuinely different random variable over the same
    units from `converted`."""
    rows: list[dict[str, Any]] = []
    for i in range(_ENCOURAGEMENT_ITT_N_PER_ARM):
        rows.append(
            {
                "unit_id": f"c{i}",
                "group_id": "control",
                "converted": 0 if i % 3 == 0 else 1,
                "uptake": 0,
            }
        )
        rows.append(
            {
                "unit_id": f"t{i}",
                "group_id": "treatment",
                "converted": 0 if i % 2 == 0 else 1,
                "uptake": 1 if i % 3 == 0 else 0,
            }
        )
    return rows


def _encouragement_itt_frame() -> pd.DataFrame:
    return pd.DataFrame(_encouragement_itt_rows())


def _encouragement_itt_panel_frame() -> pd.DataFrame:
    """A two-day panel (pre-exposure day zeroed; exposure day carrying the
    same per-unit totals `_encouragement_itt_frame` computed) built FROM
    the unit-summary rows, so panel and summary agree by construction --
    mirrors `dataset.unit_panel_frame`'s shape."""
    rows: list[dict[str, Any]] = []
    for r in _encouragement_itt_rows():
        rows.append(
            {
                "unit_id": r["unit_id"],
                "group_id": r["group_id"],
                "date": dt.date(2025, 1, 10),
                "converted": 0,
                "uptake": 0,
            }
        )
        rows.append(
            {
                "unit_id": r["unit_id"],
                "group_id": r["group_id"],
                "date": dt.date(2025, 1, 11),
                "converted": r["converted"],
                "uptake": r["uptake"],
            }
        )
    return pd.DataFrame(rows)


def _encouragement_itt_event_rows() -> list[dict[str, Any]]:
    """Event-log shape of `_encouragement_itt_rows`, on `dataset.py`'s
    shared `_row`/`duckdb_connection` convention: an exposure per unit, a
    `clicked` event for every unit with `uptake == 1`, and a `purchase`
    event for every unit with `converted == 1` (the `converted` metric is
    the presence of `purchase` in-window, same convention as
    `dataset.nonpositive_mean_definitions_dict`'s `converted` metric).
    Freshness padding on `purchase` mirrors `_encouragement_rows_for_parity`
    so the freshness bound reaches the declared experiment `end`."""
    rows: list[dict[str, Any]] = []
    for r in _encouragement_itt_rows():
        rows.append(
            ds._row(
                r["unit_id"],
                _ENCOURAGEMENT_EXPOSURE_AT,
                "exposure",
                group_id=r["group_id"],
                experiment_id="exp",
            )
        )
        if r["uptake"]:
            rows.append(ds._row(r["unit_id"], _ENCOURAGEMENT_CLICK_AT, "clicked"))
        if r["converted"]:
            rows.append(ds._row(r["unit_id"], _ENCOURAGEMENT_PURCHASE_AT, "purchase"))
        rows.append(ds._row(r["unit_id"], _ENCOURAGEMENT_FRESHNESS_PAD_AT, "purchase", revenue=0.0))
    return rows


def _encouragement_itt_defs_dict() -> dict[str, Any]:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "clicked", "column": None},
                    {"name": "purchase", "column": None},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "conversion",
                "name": "converted",
                "entity": "user_id",
                "fact": "purchase",
                "window_days": 1,
                "preferred_direction": "increase",
            }
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2025-01-10",
                "end": _ENCOURAGEMENT_EXPERIMENT_END.isoformat(),
                "control_group": "control",
                "allocation": {"control": 0.5, "treatment": 0.5},
                "plan": AnalysisPlan(primary="converted").model_dump(mode="json"),
                "design": {
                    "mechanism": "encouragement",
                    "uptake": {"fact": "clicked"},
                    "exclusion_restriction": {
                        "acknowledged": True,
                        "justification": "uptake does not gate the conversion outcome",
                    },
                },
            }
        ],
    }


def _encouragement_itt_case() -> ParityCase:
    """The capability restored by dropping `_binomial_eligible`'s
    `arm.sum_d is None` clause and stripping uptake moments before
    `_infer_binomial_lift_result` reads the arm
    (`_without_unused_binomial_uptake`, `increment/estimation/engine.py`):
    an Encouragement design's ITT on a conversion metric must keep the
    exact independent-binomial route, identically across every applicable
    constructor. `from_switchback_panel` has no `design=` parameter at
    all."""
    itt_defs = Definitions.model_validate(_encouragement_itt_defs_dict())
    itt_plan = AnalysisPlan(primary="converted")

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(_encouragement_itt_event_rows())
        analysis = make_analysis(con, itt_defs, experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(_encouragement_itt_event_rows())
        native = make_analysis(con, itt_defs, experiment="exp")
        return _publish_and_adopt(con, native)

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            _encouragement_itt_frame(),
            unit="unit_id",
            group="group_id",
            metrics=_encouragement_itt_metrics(),
            design=_ENCOURAGEMENT_ITT_DESIGN,
            uptake="uptake",
            plan=itt_plan,
        )

    def build_unit_panel() -> Analysis:
        return Analysis.from_unit_panel(
            _encouragement_itt_panel_frame(),
            unit="unit_id",
            group="group_id",
            date="date",
            metrics=_encouragement_itt_metrics(),
            design=_ENCOURAGEMENT_ITT_DESIGN,
            uptake="uptake",
            plan=itt_plan,
        )

    def build_moments() -> Analysis:
        summary = build_unit_summary()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            summary.export(path)
            rows = pq.read_table(path).to_pylist()
        replayed = Analysis.from_moments(
            rows,
            metrics=_encouragement_itt_metrics(),
            design=_ENCOURAGEMENT_ITT_DESIGN,
            plan=itt_plan,
        )
        summary.close()
        return replayed

    return ParityCase(
        id="encouragement_itt_keeps_binomial_reference_kind",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        estimands=("itt",),
        waive=dict(_SWITCHBACK_WAIVE),
        # Exports/replays moments on every run -- see _fixed_horizon_case's
        # slow=True for why.
        slow=True,
    )


def _binary_breakout_ancillary_uptake_case(*, include_fact_only_unit: bool = False) -> ParityCase:
    """Preserve set-only and finite binary breakout rows beside uptake moments."""
    records, events = [], []
    for segment in ("zero_control", "positive_control"):
        for arm in ("control", "treatment"):
            for i in range(50):
                unit = f"{segment}-{arm}-{i}"
                converted = i < (
                    25 if arm == "treatment" else 10 if segment == "positive_control" else 0
                )
                uptake = arm == "treatment" and i < 25
                records.append(
                    {
                        "unit_id": unit,
                        "group_id": arm,
                        "store": segment,
                        "converted": int(converted),
                        "uptake": int(uptake),
                        "date": _ENCOURAGEMENT_PURCHASE_AT.date(),
                    }
                )
                events.append(
                    ds._row(
                        unit,
                        _ENCOURAGEMENT_EXPOSURE_AT,
                        "exposure",
                        group_id=arm,
                        experiment_id="exp",
                        store_id=segment,
                    )
                )
                if uptake:
                    events.append(
                        ds._row(unit, _ENCOURAGEMENT_CLICK_AT, "clicked", store_id=segment)
                    )
                if converted:
                    events.append(
                        ds._row(unit, _ENCOURAGEMENT_PURCHASE_AT, "purchase", store_id=segment)
                    )
    events.append(
        ds._row(
            "positive_control-treatment-49",
            _ENCOURAGEMENT_FRESHNESS_PAD_AT,
            "purchase",
            store_id="positive_control",
        )
    )
    if include_fact_only_unit:
        events.append(ds._row("freshness-only", _ENCOURAGEMENT_FRESHNESS_PAD_AT, "purchase"))
    definition = _encouragement_itt_defs_dict()
    definition["fact_sources"][0]["properties"] = [
        {"name": "store", "column": "store_id", "dtype": "string", "as_of": "static"}
    ]
    definition["experiments"][0]["breakouts"] = [{"property": "store"}]
    definitions = Definitions.model_validate(definition)
    plan = AnalysisPlan(primary="converted")
    specs = _encouragement_itt_metrics()

    def native(*, artifact: bool = False) -> Analysis:
        con = ds.duckdb_connection(events)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)

    def summary() -> Analysis:
        return Analysis.from_unit_summary(
            pd.DataFrame(records),
            unit="unit_id",
            group="group_id",
            metrics=specs,
            design=_ENCOURAGEMENT_ITT_DESIGN,
            uptake="uptake",
            plan=plan,
        )

    def panel() -> Analysis:
        return Analysis.from_unit_panel(
            pd.DataFrame(records),
            unit="unit_id",
            group="group_id",
            date="date",
            metrics=specs,
            design=_ENCOURAGEMENT_ITT_DESIGN,
            uptake="uptake",
            plan=plan,
            breakouts=["store"],
        )

    def replay() -> Analysis:
        analysis = native()
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "moments.parquet"
                analysis.export(destination)
                rows = pq.read_table(destination).to_pylist()
            return Analysis.from_moments(
                rows,
                metrics=specs,
                design=definitions.experiments[0].resolved_design(),
                plan=plan,
            )
        finally:
            analysis.close()
            for con in getattr(analysis, "_parity_connections", ()):
                con.disconnect()

    def probe(results: Any) -> None:
        by_segment = {row.dimension_value: row for row in results if row.estimand == "itt"}
        assert set(by_segment) == {"zero_control", "positive_control"}
        for row in by_segment.values():
            assert row.excluded is None and row.reference_kind == "binomial"
            assert row.binomial_set is not None
        zero = by_segment["zero_control"]
        assert zero.lift is None
        assert_set_contains_finer_reference(zero.binomial_set)
        assert zero.binomial_set.upper is None
        positive = by_segment["positive_control"]
        assert positive.lift is not None and positive.lift.value == pytest.approx(1.5)

    return ParityCase(
        id=(
            "artifact-breakout-exposure-coverage"
            if include_fact_only_unit
            else "audit-binary-breakout-ancillary-uptake"
        ),
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": summary,
            "from_unit_panel": panel,
            "from_moments": replay,
        },
        breakout_dimension="store",
        readout_probe=probe,
        waive={
            **_SWITCHBACK_WAIVE,
            "from_unit_summary": "SOURCE: unit summaries declare no breakout dimension",
            "from_moments": "SOURCE: portable moments retain no per-unit breakout dimension",
        },
        waived_refusal_codes={
            "from_unit_summary": "facade.analysis.operation",
            "from_moments": "facade.analysis.operation",
        },
        slow=True,
    )


_QUANTILE_TIES_METRIC = MetricSpec(
    name="p90_latency", value_column="latency", type="quantile", quantile=0.9
)

_QUANTILE_TIES_DEFS_YAML = """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: latency
        column: value
      - name: enrolled
        column: null
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2024-01-01T00:00:00
    end: 2024-01-08T00:00:00
    control_group: C
    plan: {secondaries: [p90_latency]}
"""


def _quantile_ties_event_table(n_per_arm: int = 200, seed: int = 3, *, family: bool = False) -> Any:
    import numpy as np
    import pyarrow as pa

    if family:
        baseline = np.arange(1.0, n_per_arm + 1.0)
        values = np.concatenate([baseline, baseline + 10.0, baseline])
        groups = ["C"] * n_per_arm + ["T"] * n_per_arm + ["T2"] * n_per_arm
        users = [f"{group}-{i}" for group in ("C", "T", "T2") for i in range(n_per_arm)]
    else:
        rng = np.random.default_rng(seed)
        # Repeated rounded values exercise tied order-statistic brackets.
        control = np.round(rng.lognormal(0.0, 0.5, n_per_arm) * 100) / 100
        treatment = np.round(rng.lognormal(0.0, 0.5, n_per_arm) * 100 * 1.2) / 100
        users = [f"c{i}" for i in range(n_per_arm)] + [f"t{i}" for i in range(n_per_arm)]
        groups = ["C"] * n_per_arm + ["T"] * n_per_arm
        values = np.concatenate([control, treatment])
    enroll_ts = np.datetime64("2024-01-02T00:00:00")
    metric_ts = np.datetime64("2024-01-08T01:00:00")
    rows: list[dict] = []
    for user, group, value in zip(users, groups, values, strict=True):
        common = {"user_id": user, "experiment_id": "exp", "group_id": group}
        rows.append(
            {
                **common,
                "ts": enroll_ts - np.timedelta64(1, "D"),
                "event": "latency",
                "value": 1000.0 if group == "T" else 100.0,
            }
        )
        rows.append({**common, "ts": enroll_ts, "event": "enrolled", "value": None})
        rows.append({**common, "ts": metric_ts, "event": "latency", "value": float(value)})
    return pa.Table.from_pylist(rows)


def _mixed_quantile_checkpoint_probes(
    summary_specs: list[MetricSpec],
    design: Randomized,
    checked_declaration: Callable[[Callable[[], Analysis]], Analysis],
) -> tuple[Callable[[Any], None], Callable[[str, Analysis], None]]:
    def check_rows(readout: Any) -> None:
        (row,) = readout
        assert (row.metric, row.group_id, row.estimand) == ("purchase_rate", "treatment", "itt")
        lift = row.require_lift()
        assert lift.value is not None
        assert lift.value == pytest.approx(0.275)
        assert lift.lb is not None and lift.ub is not None
        assert lift.lb <= lift.value <= lift.ub
        assert row.sequential_result is not None

    def probe_checkpoint(constructor: str, analysis: Analysis) -> None:
        from .comparison import nested_close

        def check_catalog(source: Analysis) -> None:
            assert [(m.name, m.type) for m in source.metrics] == [
                ("purchase_rate", "conversion"),
                ("p90_revenue", "quantile"),
            ], constructor
            assert getattr(source.metrics[1], "quantile", None) == 0.9
            with pytest.raises(CodedError) as refused:
                source.run(metrics=["p90_revenue"])
            assert refused.value.code == "sequential.route.unsupported", constructor

        check_catalog(analysis)
        snapshot = analysis.sequential_snapshot()
        assert [(m.metric, m.law) for m in snapshot.registration.models] == [
            ("purchase_rate", "bernoulli")
        ]
        assert sorted((s.metric, s.group_id, s.n, s.successes) for s in snapshot.states) == [
            ("purchase_rate", "control", 60, 40),
            ("purchase_rate", "treatment", 60, 51),
        ], constructor
        expected = analysis.run(metrics=["purchase_rate"])
        check_rows(expected)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.parquet"
            if constructor == "from_moments":
                with pytest.raises(CodedError) as refused:
                    analysis.export(path)
                assert refused.value.code == "facade.analysis.operation"
                assert refused.value.context["operation"] == "export_moments"
                assert not path.exists()
                return
            analysis.export(path)
            (envelope,) = pq.read_table(path).to_pylist()
        assert envelope["record_kind"] == "sequential_checkpoint"
        assert envelope["moments_format"] == 9
        assert not {"metric", "group_id", "n", "sum", "sum_sq", "ref", "ref_den"} & envelope.keys()
        replay = checked_declaration(
            lambda: Analysis.from_moments([envelope], metrics=summary_specs, design=design)
        )
        try:
            check_catalog(replay)
            restored = replay.sequential_snapshot()
            assert restored == snapshot, constructor
            actual = replay.run(metrics=["purchase_rate"])
            check_rows(actual)
            assert nested_close(
                [row.model_dump() for row in expected], [row.model_dump() for row in actual]
            ), constructor
        finally:
            _close_parity_analysis(replay)

    return check_rows, probe_checkpoint


def _mixed_quantile_declaration(build: Callable[[], Analysis]) -> Analysis:
    import warnings

    from tests.warning_codes import warning_codes

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IncrementWarning)
        analysis = build()
    unexpected = set(warning_codes(caught)) - {"decision.family.quantile_always_valid_excluded"}
    if unexpected:
        _close_parity_analysis(analysis)
        pytest.fail(f"unexpected declaration warning codes: {unexpected}")
    return analysis


def _sequential_mixed_quantile_catalog_case() -> ParityCase:
    """A conversion checkpoint retains an unmodeled quantile, never quantile moments."""
    from increment.frame import synthesise_metric
    from increment.semantics.sequential import SequentialRegistration
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id

    rows = ds.event_rows() + _extra_treatment_purchases(11)
    plan = AnalysisPlan(
        primary="purchase_rate",
        secondaries=["p90_revenue"],
        inference={"kind": "always_valid"},
    )
    allocation = {"control": 0.5, "treatment": 0.5}
    design = Randomized(control_group="control", allocation=allocation)
    defs_dict = ds.definitions_dict(plan=plan, allocation=allocation)
    conversion = next(m for m in defs_dict["metrics"] if m["name"] == "purchase_rate")
    revenue = next(m for m in defs_dict["metrics"] if m["name"] == "revenue")
    defs_dict["metrics"] = [
        conversion,
        {
            **revenue,
            "name": "p90_revenue",
            "type": "quantile",
            "quantile": 0.9,
            "window_days": None,
        },
    ]
    defs_dict["experiments"][0]["n_pre_periods"] = 0
    declared = Definitions.model_validate(defs_dict)
    exp = declared.experiment("exp")
    assert exp is not None
    as_of = ds._EXPOSURE_AT.date() + dt.timedelta(days=1)
    summary_specs = [
        MetricSpec(
            name="purchase_rate",
            type="conversion",
            value_column="converted",
            preferred_direction="increase",
        ),
        MetricSpec(
            name="p90_revenue",
            type="quantile",
            quantile=0.9,
            value_column="revenue",
            missing="zero",
            preferred_direction="increase",
        ),
    ]
    panel_specs = [
        summary_specs[0].model_copy(update={"value_column": "purchase_rate", "window_days": 1}),
        summary_specs[1],
    ]

    def registered_plan(
        metrics: list[Any], mapping: Mapping[str, object], *, specs=()
    ) -> AnalysisPlan:
        # Automatic registration refuses a quantile catalog. Declare the conversion
        # law alone, then bind that immutable registration to the full source catalog.
        automatic = bind_automatic_sequential_plan(
            plan.model_copy(update={"secondaries": ()}),
            [metrics[0]],
            design=design,
            source_id="exp",
            source_mapping=mapping,
            transformations=specs[:1],
            path="frame" if specs else "warehouse",
        )
        assert automatic is not None and automatic.inference is not None
        assert automatic.inference.registration is not None
        registration = SequentialRegistration.model_validate(
            {
                **automatic.inference.registration.model_dump(),
                "definitions_id": sequential_definition_id(
                    metrics, design, source_mapping=mapping, transformations=specs
                ),
            }
        )
        return plan.model_copy(
            update={"inference": InferenceSpec(kind="always_valid", registration=registration)}
        )

    native_plan = registered_plan(
        list(declared.metrics),
        native_observation_mapping(declared, exp, on_mixed_assignment="error"),
    )
    defs = declared.model_copy(
        update={"experiments": (exp.model_copy(update={"plan": native_plan}),)}
    )

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        try:
            analysis = make_analysis(con, defs, experiment="exp", plan=native_plan)
        except BaseException:
            con.disconnect()
            raise
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        native = build_definitions()
        adopted = _publish_and_adopt(native_connection(_native_source(native)), native)
        adopted._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return adopted

    def build_frame(*, panel: bool) -> Analysis:
        con = ds.duckdb_connection(rows)
        try:
            frame: Any = (
                ds.sequential_unit_panel_frame(con) if panel else ds.unit_summary_frame(con)
            )
        finally:
            con.disconnect()
        specs = panel_specs if panel else summary_specs
        if panel:
            frame["purchase_rate"] = (frame["revenue"] > 0).astype(int)
        frame_plan = registered_plan(
            [synthesise_metric(spec) for spec in specs],
            frame_observation_mapping(
                unit="user_id",
                group="variant",
                date="date" if panel else None,
                exposure_date="exposure_date",
            ),
            specs=specs,
        )
        builder = Analysis.from_unit_panel if panel else Analysis.from_unit_summary
        analysis = builder(
            frame,
            unit="user_id",
            group="variant",
            design=design,
            metrics=specs,
            plan=frame_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
            **({"date": "date"} if panel else {}),
        )
        if panel:
            analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return analysis

    def build_moments() -> Analysis:
        summary = build_frame(panel=False)
        return _export_and_replay(summary, summary_specs)

    check_rows, probe_checkpoint = _mixed_quantile_checkpoint_probes(
        summary_specs, design, _mixed_quantile_declaration
    )

    return ParityCase(
        id="sequential-mixed-quantile-catalog-checkpoint",
        build={
            name: lambda build=build: _mixed_quantile_declaration(build)
            for name, build in {
                "from_definitions": build_definitions,
                "from_unit_day_artifact": build_artifact,
                "from_unit_summary": lambda: build_frame(panel=False),
                "from_unit_panel": lambda: build_frame(panel=True),
                "from_moments": build_moments,
            }.items()
        },
        metrics=("purchase_rate",),
        sequential=True,
        readout_probe=check_rows,
        source_probe=probe_checkpoint,
        waive={
            "from_switchback_panel": (
                "SOURCE: switchback accepts only mean/conversion metrics and refuses a "
                "quantile catalog (source.frame.switchback.metric); it also has no "
                "registered sequential construction. This dataset has no switchback schedule."
            )
        },
        slow=True,
    )


def _quantile_ties_case(
    *, family: bool = False, unselected_conversion_sibling: bool = False
) -> ParityCase:
    """Compare tied quantiles and threshold-sensitive families across raw-data paths."""
    import ibis
    import pyarrow as pa
    import yaml

    table = _quantile_ties_event_table(n_per_arm=100 if family else 200, family=family)
    metric = (
        MetricSpec(name="median", value_column="latency", type="quantile", quantile=0.5)
        if family
        else _QUANTILE_TIES_METRIC
    )
    plan = AnalysisPlan(secondaries=[metric.name], q=math.nextafter(1.0, 0.0) if family else 0.10)
    definition = yaml.safe_load(_QUANTILE_TIES_DEFS_YAML)
    definition["metrics"][0].update(name=metric.name, quantile=metric.quantile)
    metrics = [metric]
    if unselected_conversion_sibling:
        plan = AnalysisPlan(primary="mean_latency", secondaries=[metric.name, "converted"])
        definition["metrics"].extend(
            [
                {
                    "name": "mean_latency",
                    "type": "mean",
                    "entity": "user_id",
                    "fact": "latency",
                    "aggregation": "sum",
                },
                {
                    "name": "converted",
                    "type": "conversion",
                    "entity": "user_id",
                    "fact": "latency",
                },
            ]
        )
        metrics.extend(
            [
                MetricSpec(name="mean_latency", value_column="latency"),
                MetricSpec(name="converted", type="conversion"),
            ]
        )
    definition["experiments"][0]["plan"] = plan.model_dump(mode="json")
    definitions = Definitions.model_validate(definition)

    def build_definitions() -> Analysis:
        con = ibis.duckdb.connect()
        con.create_table("events", table)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _track_connection(analysis, con)

    def build_artifact() -> Analysis:
        con = ibis.duckdb.connect()
        con.create_table("events", table)
        native = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, native)

    def _per_unit_frame() -> Any:
        import pyarrow.compute as pc

        latency_rows = table.filter(
            (pc.field("event") == "latency") & (pc.field("ts") >= dt.datetime(2024, 1, 2))
        )
        frame = latency_rows.select(["user_id", "group_id", "value"]).rename_columns(
            ["user_id", "group_id", "latency"]
        )
        if unselected_conversion_sibling:
            frame = frame.append_column("converted", pa.array([1] * len(frame)))
        return frame

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            _per_unit_frame(),
            unit="user_id",
            group="group_id",
            control="C",
            metrics=metrics,
            plan=plan,
        )

    def build_unit_panel() -> Analysis:
        per_unit = _per_unit_frame().to_pylist()
        panel_rows: list[dict] = []
        for r in per_unit:
            panel_rows.append(
                {
                    "user_id": r["user_id"],
                    "variant": r["group_id"],
                    "date": dt.date(2024, 1, 1),
                    "latency": 1000.0 if r["group_id"] == "T" else 100.0,
                }
            )
            panel_rows.append(
                {
                    "user_id": r["user_id"],
                    "variant": r["group_id"],
                    "date": dt.date(2024, 1, 2),
                    "latency": 0.0,
                }
            )
            panel_rows.append(
                {
                    "user_id": r["user_id"],
                    "variant": r["group_id"],
                    "date": dt.date(2024, 1, 8),
                    "latency": r["latency"],
                }
            )
        panel = pd.DataFrame(panel_rows)
        panel["enrolled_on"] = dt.date(2024, 1, 2)
        if unselected_conversion_sibling:
            panel["converted"] = 1
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            exposure_date="enrolled_on",
            control="C",
            metrics=metrics,
            plan=plan,
        )

    def build_moments() -> Analysis:
        # A quantile has no moments representation: `export()` itself
        # refuses before a from_moments cube could ever exist.
        with build_unit_summary() as summary, tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.parquet"
            summary.export(path)
        raise AssertionError("export should have refused before reaching this line")

    def check_selected_rows(readout: Any) -> None:
        assert {row.metric for row in readout} == {metric.name, "mean_latency"}
        for row in readout:
            assert row.require_lift().value is not None

    return ParityCase(
        id="audit-panel-pre-exposure-and-selection"
        if unselected_conversion_sibling
        else "audit-quantile-family"
        if family
        else "quantile_tied_rounded_outcomes",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        metrics=(metric.name, "mean_latency") if unselected_conversion_sibling else None,
        readout_probe=check_selected_rows if unselected_conversion_sibling else None,
        waive={
            "from_moments": (
                "SOURCE: a quantile has no moments representation -- export() "
                "itself refuses before a from_moments cube could ever exist."
            ),
            "from_switchback_panel": (
                "SOURCE: switchback metrics must be type='mean' or "
                "type='conversion'; quantile is refused before any frame is read "
                "(confirmed live: source.frame.switchback.metric)."
            ),
        },
        waived_refusal_codes={"from_moments": "source.frame.quantile_no_moments"},
        require_selection=family,
        slow=family or unselected_conversion_sibling,
    )


def _inferred_null_metric_case() -> ParityCase:
    import pyarrow as pa

    rows = [
        row
        for row in ds.event_rows()
        if not (row["event"] == "purchase" and row["event_at"] == ds._PURCHASE_AT)
    ]
    plan = AnalysisPlan(primary="purchase_rate")
    definition = ds.definitions_dict(plan=plan)
    definition["metrics"] = [
        metric for metric in definition["metrics"] if metric["name"] == "purchase_rate"
    ]
    definitions = Definitions.model_validate(definition)
    specs = [
        MetricSpec(
            name="purchase_rate",
            type="conversion",
            value_column="converted",
            missing="zero",
            preferred_direction="increase",
        )
    ]

    def native(*, artifact: bool = False) -> Analysis:
        con = ds.duckdb_connection(rows)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)

    def frame(*, panel: bool = False) -> Analysis:
        con = ds.duckdb_connection(rows)
        summary = ds.unit_summary_frame(con)
        con.disconnect()
        if panel:
            data = pa.Table.from_pandas(
                ds.unit_panel_frame(summary)[["user_id", "variant", "date"]], preserve_index=False
            )
        else:
            data = summary.select(["user_id", "variant"])
        data = data.append_column("converted", pa.nulls(len(data)))
        constructor = Analysis.from_unit_panel if panel else Analysis.from_unit_summary
        return constructor(
            data,
            unit="user_id",
            group="variant",
            control="control",
            metrics=specs,
            plan=plan,
            **({"date": "date"} if panel else {}),
        )

    def replay() -> Analysis:
        analysis = native()
        try:
            return _export_and_replay(
                analysis,
                [
                    MetricSpec(
                        name="purchase_rate", type="conversion", preferred_direction="increase"
                    )
                ],
            )
        finally:
            analysis.close()
            for con in getattr(analysis, "_parity_connections", ()):
                con.disconnect()

    def probe(results: Any) -> None:
        (row,) = results
        assert row.metric == "purchase_rate" and row.abs_diff == 0.0

    return ParityCase(
        id="audit-panel-missing-zero-inferred",
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": frame,
            "from_unit_panel": lambda: frame(panel=True),
            "from_moments": replay,
        },
        waive=_SWITCHBACK_WAIVE,
        readout_probe=probe,
        slow=True,
    )


def _neighboring_integer_switchback_analysis() -> Analysis:
    import polars as pl

    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    rows = [
        {
            "unit": unit,
            "cycle": 0,
            "period": period,
            "step": 0,
            "arm": "control" if period == 0 else "treatment",
            "y": 2**53 + period * delta,
        }
        for unit, delta in (("a", 1), ("b", 2))
        for period in (0, 1)
    ]
    return Analysis.from_switchback_panel(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="arm",
        metrics={"y": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
        contrast_references={"y": UnitCycleTApproximation()},
    )


def _switchback_neighboring_integer_case() -> ParityCase:
    from scipy.stats import t

    def probe(results: Any) -> None:
        (row,) = results
        assert row.estimate.value == pytest.approx(1.5)
        assert row.standard_error == pytest.approx(0.5)
        radius = 0.5 * t.isf(row.alpha / 2, 1)
        assert row.estimate.lb == pytest.approx(1.5 - radius)
        assert row.estimate.ub == pytest.approx(1.5 + radius)

    return ParityCase(
        id="audit-switchback-neighboring-integers",
        build={"from_switchback_panel": _neighboring_integer_switchback_analysis},
        waive={
            name: "SOURCE: parallel-arm and portable-moment constructors carry no switchback schedule"
            for name in CONSTRUCTORS
            if name != "from_switchback_panel"
        },
        readout_probe=probe,
    )


_EXPLORATORY_SOURCE_LIMITED = "facade.analysis.exploratory_metrics_source_limited"
_EXPLORATORY_READS: dict[str, Callable[..., Any]] = {
    "run": lambda analysis, names: analysis.run(exploratory_metrics=names),
    "run_breakout": lambda analysis, names: analysis.run_breakout(exploratory_metrics=names),
    "run_asof_lift": lambda analysis, names: analysis.run_asof_lift(exploratory_metrics=names),
    "run_asof": lambda analysis, names: analysis.run_asof(exploratory_metrics=names),
    "run_daily": lambda analysis, names: analysis.run_daily(exploratory_metrics=names),
}


def _exploratory_metrics_case() -> ParityCase:
    """Added metrics are supported on ``from_definitions`` alone (the only ingress holding
    saved definitions to add from); every other ingress refuses with one shared code. The
    switchback ingress builds no case-wide fixture, so its refusal is probed beside the
    native one."""

    def refused(analysis: Analysis) -> None:
        for read in _EXPLORATORY_READS.values():
            with pytest.raises(CodedError) as caught:
                read(analysis, ["purchase_rate"])
            assert caught.value.code == _EXPLORATORY_SOURCE_LIMITED
        with pytest.raises(CodedError) as caught:
            analysis.available_metrics  # noqa: B018
        assert caught.value.code == _EXPLORATORY_SOURCE_LIMITED

    def probe(path: str, analysis: Analysis) -> None:
        if path != "from_definitions":
            refused(analysis)
            return
        refused(_neighboring_integer_switchback_analysis())
        revenue = [metric for metric in analysis.metrics if metric.name == "revenue"]
        narrowed = make_analysis_like(
            analysis,
            metrics=revenue,
            experiment=analysis.experiment.model_copy(
                update={"plan": AnalysisPlan(secondaries=["revenue"])}
            ),
        )
        assert [metric.name for metric in narrowed.available_metrics] == [
            "purchase_rate",
            "rps",
            "revenue_cuped",
        ]
        rows = narrowed.run(metrics=[], exploratory_metrics=["purchase_rate", "rps"])
        assert [row.metric for row in rows] == ["purchase_rate", "rps"]
        assert {row.role for row in rows} == {"exploratory"}
        assert [row.metric for row in narrowed.run(exploratory_metrics=["rps"])] == [
            "revenue",
            "rps",
        ]
        with pytest.raises(CodedError) as caught:
            narrowed.run(exploratory_metrics=["revenue"])
        assert caught.value.code == "facade.analysis_config.exploratory_metric_unavailable"

    return replace(
        _mean_ratio_conversion_cuped_case(), id="exploratory-metrics", source_probe=probe
    )


def _encouragement_multi_metric_breakout_case() -> ParityCase:
    from increment.breakout.estimates import BreakoutEstimates
    from increment.breakout.heterogeneity import segment_heterogeneity
    from increment.breakout.rollout import segment_rollout_recommendation

    records, events = [], []
    for arm in ("control", "treatment"):
        for i in range(1, 41):
            unit = f"{arm}-{i}"
            store = "A" if i % 2 else "B"
            clicked = arm == "treatment" and i <= (16 if i % 2 else 24)
            revenue = _encouragement_revenue(i, clicked=clicked)
            records.append(
                {
                    "user_id": unit,
                    "group": arm,
                    "store": store,
                    "clicked": int(clicked),
                    "revenue": revenue,
                    "sales": 2 * revenue,
                    "day": _ENCOURAGEMENT_PURCHASE_AT.date(),
                }
            )
            events.extend(
                [
                    ds._row(
                        unit,
                        _ENCOURAGEMENT_EXPOSURE_AT,
                        "exposure",
                        group_id=arm,
                        experiment_id="exp",
                        store_id=store,
                    ),
                    ds._row(
                        unit,
                        _ENCOURAGEMENT_PURCHASE_AT,
                        "purchase",
                        revenue=revenue,
                        store_id=store,
                    ),
                    ds._row(
                        unit,
                        _ENCOURAGEMENT_PURCHASE_AT,
                        "sales",
                        latency=2 * revenue,
                        store_id=store,
                    ),
                    ds._row(
                        unit,
                        _ENCOURAGEMENT_FRESHNESS_PAD_AT,
                        "purchase",
                        revenue=0.0,
                        store_id=store,
                    ),
                    ds._row(
                        unit, _ENCOURAGEMENT_FRESHNESS_PAD_AT, "sales", latency=0.0, store_id=store
                    ),
                ]
            )
            if clicked:
                events.append(ds._row(unit, _ENCOURAGEMENT_CLICK_AT, "clicked", store_id=store))
    plan = AnalysisPlan(secondaries=["revenue", "sales"])
    definition = _encouragement_defs_dict(plan)
    definition["fact_sources"][0]["facts"].append({"name": "sales", "column": "latency"})
    definition["fact_sources"][0]["properties"] = [
        {"name": "store", "column": "store_id", "dtype": "string", "as_of": "static"}
    ]
    definition["metrics"].append({**definition["metrics"][0], "name": "sales", "fact": "sales"})
    definition["experiments"][0]["breakouts"] = [{"property": "store"}]
    definitions = Definitions.model_validate(definition)
    experiment = definitions.experiment("exp")
    assert experiment is not None
    design = experiment.resolved_design()
    specs = [MetricSpec(name=name, preferred_direction="increase") for name in ("revenue", "sales")]

    def native(*, artifact: bool = False) -> Analysis:
        con = ds.duckdb_connection(events)
        analysis = make_analysis(con, definitions, experiment="exp")
        return _publish_and_adopt(con, analysis) if artifact else _track_connection(analysis, con)

    def summary() -> Analysis:
        return Analysis.from_unit_summary(
            pd.DataFrame(records),
            unit="user_id",
            group="group",
            metrics=specs,
            design=design,
            uptake="clicked",
            plan=plan,
        )

    def panel() -> Analysis:
        return Analysis.from_unit_panel(
            pd.DataFrame(records),
            unit="user_id",
            group="group",
            date="day",
            metrics=specs,
            design=design,
            uptake="clicked",
            plan=plan,
            breakouts=["store"],
        )

    def replay() -> Analysis:
        analysis = native()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "moments.parquet"
                analysis.export(path)
                rows = pq.read_table(path).to_pylist()
            return Analysis.from_moments(rows, metrics=specs, design=design, plan=plan)
        finally:
            analysis.close()
            for con in getattr(analysis, "_parity_connections", ()):
                con.disconnect()

    def probe(results: Any) -> None:
        assert {row.metric for row in results} == {"revenue", "sales"}
        pooled, _ = segment_heterogeneity(results)
        compliance = [
            row for row in pooled if row.estimand == "compliance" and row.scale == "absolute"
        ]
        assert {row.metric for row in compliance} == {"revenue", "sales"}
        for row in compliance:
            assert row.k == 2 and row.pooled.value == pytest.approx(0.5)
        assert any(row.estimand == "itt" and row.scale == "relative" for row in pooled)
        absolute = BreakoutEstimates([row for row in results if row.value_scale == "absolute"])
        recommendations, segments = segment_rollout_recommendation(absolute)
        assert not recommendations and not segments

    return ParityCase(
        id="audit-encouragement-multi-metric-breakout",
        build={
            "from_definitions": native,
            "from_unit_day_artifact": lambda: native(artifact=True),
            "from_unit_summary": summary,
            "from_unit_panel": panel,
            "from_moments": replay,
        },
        breakout_dimension="store",
        waive={
            **_SWITCHBACK_WAIVE,
            "from_unit_summary": "SOURCE: unit summaries declare no breakout dimension",
            "from_moments": "SOURCE: portable moments retain no per-unit breakout dimension",
        },
        waived_refusal_codes={
            "from_unit_summary": "facade.analysis.operation",
            "from_moments": "facade.analysis.operation",
        },
        readout_probe=probe,
        slow=True,
    )


# A control arm above the 4,000,000 units the exact binomial route once refused, against a
# small treatment arm: from there `armstats.binary_counts` checks the Bernoulli second moment
# against the rounding bound of the arm's units, and each ingress builds that moment its own
# way. One large arm keeps the build within bounded memory and time.
_CEILING_CONTROL_UNITS = 4_000_100
_CEILING_TREATMENT_UNITS = 4_000
_CEILING_CONTROL_CONVERSIONS = 4_000
_CEILING_TREATMENT_CONVERSIONS = 80

_CEILING_DEFS_YAML = """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: converted
        column: value
      - name: enrolled
        column: null
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: conversion
    type: conversion
    entity: user_id
    fact: converted
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2024-01-01T00:00:00
    end: 2024-01-20T00:00:00
    control_group: control
    plan: {secondaries: [conversion]}
"""


def _ceiling_scale_connection() -> Any:
    """DuckDB holding one enrollment per unit and one conversion event per converter, generated
    inside the database so no row exists in Python. A memory limit and a private spill directory
    bound a table this size; a conversion event past the window pads the data's freshness."""
    import atexit

    import ibis

    spill = tempfile.TemporaryDirectory(prefix="increment-parity-spill-")
    atexit.register(spill.cleanup)
    con = ibis.duckdb.connect(memory_limit="6GB", threads=2, temp_directory=spill.name)
    n_c, n_t = _CEILING_CONTROL_UNITS, _CEILING_TREATMENT_UNITS
    c_conv, t_conv = _CEILING_CONTROL_CONVERSIONS, _CEILING_TREATMENT_CONVERSIONS
    con.raw_sql(
        f"""
        CREATE TABLE events AS
        SELECT i AS user_id, 'exp' AS experiment_id, 'control' AS group_id,
               TIMESTAMP '2024-01-02 00:00:00' AS ts, 'enrolled' AS event,
               CAST(NULL AS DOUBLE) AS value
        FROM range({n_c}) t(i)
        UNION ALL
        SELECT i + {n_c}, 'exp', 'treatment', TIMESTAMP '2024-01-02 00:00:00', 'enrolled',
               CAST(NULL AS DOUBLE)
        FROM range({n_t}) t(i)
        UNION ALL
        SELECT i, 'exp', 'control', TIMESTAMP '2024-01-03 00:00:00', 'converted', 1.0
        FROM range({c_conv}) t(i)
        UNION ALL
        SELECT i + {n_c}, 'exp', 'treatment', TIMESTAMP '2024-01-03 00:00:00', 'converted', 1.0
        FROM range({t_conv}) t(i)
        UNION ALL
        SELECT 0, 'exp', 'control', TIMESTAMP '2024-01-19 00:00:00', 'converted', 0.0
        """
    )
    return con


def _ceiling_scale_frame() -> Any:
    import numpy as np
    import pyarrow as pa

    n_c, n_t = _CEILING_CONTROL_UNITS, _CEILING_TREATMENT_UNITS
    units = np.arange(n_c + n_t, dtype=np.int64)
    converted = np.zeros(units.size, np.int8)
    converted[:_CEILING_CONTROL_CONVERSIONS] = 1
    converted[n_c : n_c + _CEILING_TREATMENT_CONVERSIONS] = 1
    group = pa.array(np.where(units < n_c, "control", "treatment"))
    return pa.table({"user_id": units, "group_id": group, "conversion": converted})


def _exact_binomial_beyond_the_former_arm_ceiling_case() -> ParityCase:
    """The exact binomial route on a control arm of 4,000,100 units reaches the same counts, and
    so the same interval, through every ingress that builds its moments differently: a
    warehouse aggregation (definitions, unit-day artifact), a per-unit frame (summary, panel)
    and an exported moments cube."""
    import numpy as np
    import pyarrow as pa
    import yaml

    metric = MetricSpec(name="conversion", type="conversion")
    plan = AnalysisPlan(secondaries=["conversion"])
    definition = yaml.safe_load(_CEILING_DEFS_YAML)
    definition["experiments"][0]["plan"] = plan.model_dump(mode="json")
    definitions = Definitions.model_validate(definition)

    def build_definitions() -> Analysis:
        con = _ceiling_scale_connection()
        return _track_connection(make_analysis(con, definitions, experiment="exp"), con)

    def build_artifact() -> Analysis:
        con = _ceiling_scale_connection()
        return _publish_and_adopt(con, make_analysis(con, definitions, experiment="exp"))

    def build_unit_summary() -> Analysis:
        return Analysis.from_unit_summary(
            _ceiling_scale_frame(),
            unit="user_id",
            group="group_id",
            control="control",
            metrics=[metric],
            plan=plan,
        )

    def build_unit_panel() -> Analysis:
        frame = _ceiling_scale_frame()
        days = np.full(frame.num_rows, (dt.date(2024, 1, 3) - dt.date(1970, 1, 1)).days, np.int32)
        panel = frame.append_column("date", pa.array(days, pa.date32()))
        panel = panel.append_column("enrolled_on", pa.array(days - 1, pa.date32()))
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="group_id",
            date="date",
            exposure_date="enrolled_on",
            control="control",
            metrics=[metric],
            plan=plan,
        )

    def build_moments() -> Analysis:
        return _export_and_replay(build_unit_summary(), [metric])

    def probe(results: Any) -> None:
        (row,) = [r for r in results if r.metric == "conversion"]
        assert row.reference_kind == "binomial"
        assert row.binomial_set is not None
        counts = (row.binomial_set.x_c, row.binomial_set.n_c)
        assert counts == (_CEILING_CONTROL_CONVERSIONS, _CEILING_CONTROL_UNITS)
        counts = (row.binomial_set.x_t, row.binomial_set.n_t)
        assert counts == (_CEILING_TREATMENT_CONVERSIONS, _CEILING_TREATMENT_UNITS)

    return ParityCase(
        id="exact_binomial_beyond_the_former_arm_ceiling",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_unit_summary,
            "from_unit_panel": build_unit_panel,
            "from_moments": build_moments,
        },
        waive=_SWITCHBACK_WAIVE,
        readout_probe=probe,
        slow=True,
    )


PARITY_CASES: tuple[ParityCase, ...] = (
    _encouragement_multi_metric_breakout_case(),
    _inferred_null_metric_case(),
    _switchback_neighboring_integer_case(),
    _mean_ratio_conversion_cuped_case(),
    _dashboard_group_data_case(),
    _prior_reset_case(),
    _fixed_horizon_case(
        id="portable_subset_with_winsorized_sibling",
        breakout=False,
        waive={},
        waived_refusal_codes={},
        include_cuped=False,
        unselected_winsorized_sibling=True,
    ),
    _fixed_horizon_case(
        id="audit-artifact-rounded-aggregates",
        breakout=False,
        waive={},
        waived_refusal_codes={},
        include_cuped=False,
        rounded_events=True,
    ),
    _retention_breakout_cohorts_case(),
    _fixed_horizon_case(
        id="audit-cuped-covariate-rescaling",
        breakout=False,
        waive={},
        waived_refusal_codes={},
        ratio_numerics="rescaled",
    ),
    _fixed_horizon_case(
        id="audit-ratio-neighboring-means",
        breakout=False,
        waive={},
        waived_refusal_codes={},
        ratio_numerics="neighboring",
    ),
    _lift_prior_case("normal"),
    _lift_prior_case("student_t"),
    _lift_prior_case("mixture"),
    _breakout_case(),
    _breakout_case(boolean=True),
    _dashboard_breakout_reads_case(),
    _exploratory_metrics_case(),
    _window_spelling_case(native_spelling="zulu"),
    _window_spelling_case(native_spelling="offset"),
    _sequential_missing_zero_mean_ratio_case(),
    _sequential_asymptotic_mean_case(),
    _source_horizon_case(),
    _source_horizon_case(pinned_dimension=True),
    _ratio_component_horizon_case(),
    _ratio_component_horizon_case(complete=True),
    _adjusted_ratio_anchor_case(),
    _adjusted_ratio_anchor_case(separated=True),
    _sequential_composed_itt_and_uptake_case(),
    _sequential_composed_itt_and_uptake_case(exclusion_declared=False),
    _sequential_composed_itt_and_uptake_case(metric_free=True),
    _sequential_unbounded_retention_catalog_case(),
    _sequential_composed_family_ebh_case(),
    _sequential_registered_breakout_discrete_case(),
    _sequential_registered_breakout_continuous_case(),
    _sequential_automatic_breakout_discrete_case(),
    _sequential_automatic_breakout_continuous_case(),
    _sequential_automatic_multi_arm_exact_case(),
    _sequential_automatic_multi_arm_exact_case(declared_registration="registered"),
    _sequential_automatic_multi_arm_exact_case(declared_registration="explicit"),
    _sequential_boolean_segment_case("always_valid", automatic=True),
    _sequential_boolean_segment_case("always_valid", automatic=False),
    _sequential_boolean_segment_case("asymptotic_mean", automatic=True),
    _sequential_boolean_segment_case("asymptotic_mean", automatic=False),
    _multiplicity_roles_case(),
    _nonpositive_mean_case(),
    _clustered_negative_mean_case(),
    _clustered_negative_mean_case(zero_treatment=True),
    _cluster_export_refusal_case(),
    _encouragement_itt_case(),
    _binary_breakout_ancillary_uptake_case(),
    _binary_breakout_ancillary_uptake_case(include_fact_only_unit=True),
    _encouragement_declared_definitions_case(),
    _encouragement_declared_definitions_case(exclusion_declared=False),
    _encouragement_declared_definitions_case(supplied_design=True),
    _uptake_timestamp_boundaries_case(),
    _encouragement_declared_definitions_case(missing_treatment_outcomes=True),
    _encouragement_declared_definitions_case(metric_free=True),
    _encouragement_guardrail_late_rows_case(),
    _encouragement_guardrail_late_only_case(),
    _encouragement_margin_guardrail_case(),
    _encouragement_fcr_reestimation_case(),
    _observational_covariate_case(),
    _observational_multiplicity_case(),
    _observational_quantile_case(),
    _observational_aipw_dml_case(),
    _observational_aipw_dml_case(default_sensitivity=True),
    _observational_categorical_case(),
    _observational_categorical_case(missing_levels=True),
    _sequential_mixed_quantile_catalog_case(),
    _quantile_ties_case(),
    _quantile_ties_case(family=True),
    _quantile_ties_case(unselected_conversion_sibling=True),
    _exact_binomial_beyond_the_former_arm_ceiling_case(),
)
