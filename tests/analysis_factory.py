"""Shared constructor for test-only :class:`Analysis` instances.

Definitions are built in memory, so this helper resolves the experiment,
constructs its warehouse source, and routes both paths through the same
immutable Analysis construction state used by production constructors.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

import ibis
import pyarrow as pa
from ibis.backends.sql import SQLBackend

from increment.analysis import Analysis
from increment.breakout.estimates import LiftEstimates
from increment.estimation.contrast_results import ContrastResults
from increment.query.native_source import DefinitionsMomentSource
from increment.query.session import SourceSnapshotEvidence, WarehouseSession

if TYPE_CHECKING:
    from increment.semantics.models import Definitions, Experiment, Metric
    from increment.sources import MomentSource


def _moment_source(analysis: Analysis) -> MomentSource:
    """Access an arm-analysis source for numeric moment probes."""
    return cast("MomentSource", analysis._src)


def _native_source(analysis: Analysis) -> DefinitionsMomentSource:
    return cast("DefinitionsMomentSource", _moment_source(analysis))


def native_connection(source: DefinitionsMomentSource) -> Any:
    """Expose the fixture's connection for deliberate upstream mutation tests."""
    return source._con


def lift_rows(results: LiftEstimates | ContrastResults) -> LiftEstimates:
    """Narrow the arm-analysis result union for tests of lift rows."""
    assert isinstance(results, LiftEstimates)
    return results


def contrast_rows(results: LiftEstimates | ContrastResults) -> ContrastResults:
    """Narrow Analysis.run() to switchback contrast results in tests."""
    assert isinstance(results, ContrastResults)
    return results


def make_analysis(
    con: Any = None,
    defs: Definitions | None = None,
    *,
    experiment: Experiment | str | None = None,
    store: Literal["auto", "always", "none"] = "none",
    backend: str | None = None,
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    metrics: list[Metric] | None = None,
    plan: Any = None,
    source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    **overrides: Any,
) -> Analysis:
    """Build a test analysis through the same source/context constructor."""
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.sources import MomentsSource

    design = overrides.pop("_design", None)
    if defs is None or con is None:
        metric_catalog = list(metrics or [])
        design = design or Randomized(control_group="control")
        src = MomentsSource(
            [],
            metrics=metric_catalog,
            study_id="factory",
            design=design,
            plan=plan if hasattr(plan, "alpha") else None,
        )
        return Analysis._from_source(
            src,
            design,
            defs=defs,
            con=con,
            experiment=None,
            session=None,
            experiment_name="factory",
            backend=backend,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
        )

    if experiment is None:
        exp = defs.experiments[0]
    elif isinstance(experiment, str):
        exp = defs.experiment(experiment)
        if exp is None:
            raise KeyError(f"no experiment named {experiment!r} in these definitions")
    else:
        exp = experiment
    metric_lookup = {m.name: m for m in defs.metrics}
    declared_metrics = [metric_lookup[name] for name in exp.metric_names if name in metric_lookup]
    metric_catalog = declared_metrics if metrics is None else list(metrics)
    design = design or exp.resolved_design()
    from increment.decision import CompiledDecisionPlan

    if isinstance(plan, CompiledDecisionPlan):
        compiled_plan = plan
    else:
        from increment.plan import bind_automatic_sequential_plan
        from increment.sequential_source import native_observation_mapping

        declared_plan = bind_automatic_sequential_plan(
            exp.plan if plan is None else plan,
            metric_catalog,
            design=design,
            source_id=exp.name,
            source_mapping=native_observation_mapping(
                defs, exp, on_mixed_assignment=on_mixed_assignment
            ),
            pre_period_covariate=exp.n_pre_periods > 0,
            trigger=exp.trigger,
        )
        compiled_plan = compile_decision_plan(
            declared_plan,
            metric_catalog,
            path="warehouse",
            design=design,
        )
    session = WarehouseSession(con, defs, source_snapshot_evidence=source_snapshot_evidence)
    session.drop_materialized()
    src = DefinitionsMomentSource(
        session,
        exp,
        metric_catalog,
        store=store,
        on_mixed_assignment=on_mixed_assignment,
        backend=backend,
        plan=compiled_plan,
        design=design,
    )
    return Analysis._from_source(
        src,
        design,
        defs=defs,
        experiment=exp,
        con=con,
        session=session,
        experiment_name=exp.name,
        backend=backend,
        store=store,
        on_mixed_assignment=on_mixed_assignment,
    )


def make_analysis_like(
    analysis: Analysis,
    metrics: list[Metric] | None = None,
    *,
    design: Any = None,
    plan: Any = None,
    experiment: Any = None,
) -> Analysis:
    """Rebuild a native test analysis through the in-memory constructor.

    Reads `Analysis`'s construction state directly: the facade exposes only
    ``experiment``/``metrics`` publicly, and callers hold no other input to
    re-derive a con/defs/store/plan from.
    """
    current_experiment = analysis.experiment
    next_experiment = current_experiment if experiment is None else experiment
    if design is None and experiment is not None:
        declared = current_experiment.resolved_design()
        design = (
            next_experiment.resolved_design() if analysis._design == declared else analysis._design
        )
    return make_analysis(
        analysis._con,
        analysis._defs,
        experiment=next_experiment,
        store=analysis._store,
        backend=analysis._backend,
        plan=(
            plan
            if plan is not None
            else None
            if metrics is not None or experiment is not None
            else analysis._plan
        ),
        on_mixed_assignment=analysis._on_mixed_assignment,
        metrics=list(analysis.metrics) if metrics is None else metrics,
        source_snapshot_evidence=getattr(analysis._session, "source_snapshot_evidence", None),
        _design=analysis._design if design is None else design,
    )


# Shared test fixtures and event-log factories.


def _recovered_sum(n, ref, resid):
    """A producer row's raw first moment, recovered exactly from its centered
    moments: ``sum(v) == n * ref_v + cv1``."""
    return n * ref + resid


_PRICING_TIER_UNITS_PER_ARM = 20


def _pricing_tier_test_rows() -> list[dict]:
    """Rows for ``pricing_tier_test`` - the experiment that exercises a
    ``RatioMetric`` (``revenue_per_session``) through the facade.

    Every unit is exposed 2025-03-05 and has exactly one session that day,
    so each arm's ratio denominator is its unit count and every readout is
    hand-computable - control, then treatment:

    * ``purchase_rate``        10/20 = 0.5 | 14/20 = 0.7
    * ``revenue_per_session``  10 x 20.0 / 20 = 10.0 | 14 x 40.0 / 20 = 28.0
    * ``d7_retention``         8/20 = 0.4 | 12/20 = 0.6
    * ``avg_session_duration`` mean(100, 102, .. 138) | mean(95, 97, .. 133)

    Converters take the low indices of each arm and so do returners, so a
    converter is more likely to also be a returner - outcomes correlate
    within a unit, as they do in a real population, instead of behaving
    like independent draws.
    """
    rows: list[dict] = []
    arms = (
        # group_id, converters, returners, first session duration
        ("control", 10, 8, 100.0),
        ("treatment", 14, 12, 95.0),
    )
    for group_id, n_converting, n_returning, base_duration in arms:
        revenue = 20.0 if group_id == "control" else 40.0
        for i in range(_PRICING_TIER_UNITS_PER_ARM):
            uid = f"{group_id[0]}p{i:02d}"
            base = {
                "user_id": uid,
                "session_id": f"s_{uid}",
                "experiment_id": None,
                "group_id": None,
                "revenue": None,
                "duration_s": None,
                "country_code": "US",
                "device_type": "web",
                "plan": "free",
            }
            # Exposure: staggered within the hour so first_exposure is
            # unambiguous per unit but every unit shares the same day.
            rows.append(
                base
                | {
                    "event_at": datetime(2025, 3, 5, 9, i % 60, 0),
                    "event": "page_view",
                    "experiment_id": "pricing_tier_test",
                    "group_id": group_id,
                }
            )
            # Day-0 session: revenue_per_session's denominator and
            # avg_session_duration's value.
            rows.append(
                base
                | {
                    "event_at": datetime(2025, 3, 5, 11, i % 60, 0),
                    "event": "session_end",
                    "duration_s": base_duration + 2.0 * i,
                }
            )
            if i < n_converting:
                # Day 1: inside purchase_rate's 14-day window and
                # revenue_per_session's 7-day numerator window.
                rows.append(
                    base
                    | {
                        "event_at": datetime(2025, 3, 6, 14, i % 60, 0),
                        "event": "purchase",
                        "revenue": revenue,
                    }
                )
            if i < n_returning:
                # Day 8: inside d7_retention's [7, 14) observation band.
                rows.append(
                    base
                    | {
                        "event_at": datetime(2025, 3, 13, 9, i % 60, 0),
                        "event": "page_view",
                    }
                )
    return rows


def _make_event_log_table(con, extra_rows: list[dict] | None = None):
    """Create an ``analytics.event_log`` table in DuckDB.

    The fact source SQL in examples/definitions/ references this table.
    We create it with columns matching the fact source's facts + properties.
    """
    rows = [
        # - Exposure events (page_view) -------------------------------------
        {
            "event_at": datetime(2025, 1, 20, 10, 0, 0),
            "user_id": "u1",
            "session_id": "s1",
            "event": "page_view",
            "experiment_id": "new_onboarding_v2",
            "group_id": "control",
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 20, 10, 5, 0),
            "user_id": "u2",
            "session_id": "s2",
            "event": "page_view",
            "experiment_id": "new_onboarding_v2",
            "group_id": "control",
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 20, 11, 0, 0),
            "user_id": "u3",
            "session_id": "s3",
            "event": "page_view",
            "experiment_id": "new_onboarding_v2",
            "group_id": "treatment",
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 20, 11, 5, 0),
            "user_id": "u4",
            "session_id": "s4",
            "event": "page_view",
            "experiment_id": "new_onboarding_v2",
            "group_id": "treatment",
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # Pre-exposure purchase events (CUPED covariate for purchase_rate; window [Jan 6, Jan 20)): u1/u3 have a pre-period purchase, u2/u4 don't (nonzero-variance covariate).
        {
            "event_at": datetime(2025, 1, 10, 12, 0, 0),
            "user_id": "u1",
            "session_id": "s1",
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 9.99,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 11, 12, 0, 0),
            "user_id": "u3",
            "session_id": "s3",
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 9.99,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # - Purchase events (purchase_rate metric) -------------------------
        {
            "event_at": datetime(2025, 1, 22, 14, 0, 0),
            "user_id": "u1",
            "session_id": "s1",
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 49.99,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 23, 9, 0, 0),
            "user_id": "u2",
            "session_id": "s2",
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 29.99,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 22, 15, 0, 0),
            "user_id": "u3",
            "session_id": "s3",
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 19.99,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # - PRE-exposure session-end events (CUPED covariate for
        #    avg_session_duration; window [Jan 6, Jan 20)) -------------------
        {
            "event_at": datetime(2025, 1, 12, 9, 0, 0),
            "user_id": "u1",
            "session_id": "s1",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 200.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 13, 9, 0, 0),
            "user_id": "u2",
            "session_id": "s2",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 210.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 14, 9, 0, 0),
            "user_id": "u3",
            "session_id": "s3",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 220.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 15, 9, 0, 0),
            "user_id": "u4",
            "session_id": "s4",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 230.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # - Session-end events (avg_session_duration) -----------------------
        {
            "event_at": datetime(2025, 1, 21, 9, 0, 0),
            "user_id": "u1",
            "session_id": "s1",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 120.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 21, 10, 0, 0),
            "user_id": "u2",
            "session_id": "s2",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 90.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 21, 12, 0, 0),
            "user_id": "u3",
            "session_id": "s3",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 150.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 1, 21, 13, 0, 0),
            "user_id": "u4",
            "session_id": "s4",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 80.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # - Page-view events for d7_retention ------------------------------
        # u1: day 7 (>= threshold) -> retention=1
        {
            "event_at": datetime(2025, 1, 27, 10, 0, 0),
            "user_id": "u1",
            "session_id": "s1",
            "event": "page_view",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # u2: day 6 (< threshold) -> retention=0
        {
            "event_at": datetime(2025, 1, 26, 10, 0, 0),
            "user_id": "u2",
            "session_id": "s2",
            "event": "page_view",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # u3: day 8 (>= threshold) -> retention=1
        {
            "event_at": datetime(2025, 1, 28, 10, 0, 0),
            "user_id": "u3",
            "session_id": "s3",
            "event": "page_view",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # u4: day 9 (>= threshold) -> retention=1
        {
            "event_at": datetime(2025, 1, 29, 10, 0, 0),
            "user_id": "u4",
            "session_id": "s4",
            "event": "page_view",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        # Keepalive events (unrelated dummy unit, no exposure row) so pricing_tier_test's windowed metrics have observed data reaching their maturity date - without these every unit gets censored as not-yet-observable. These rows never join to any exposed unit, so they don't change any assertion.
        {
            "event_at": datetime(2025, 3, 25, 0, 0, 0),
            "user_id": "u_keepalive",
            "session_id": "s_keepalive",
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 0.0,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 3, 25, 0, 0, 0),
            "user_id": "u_keepalive",
            "session_id": "s_keepalive",
            "event": "session_end",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": 0.0,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
        {
            "event_at": datetime(2025, 3, 25, 0, 0, 0),
            "user_id": "u_keepalive",
            "session_id": "s_keepalive",
            "event": "page_view",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        },
    ]
    rows += _pricing_tier_test_rows()
    if extra_rows:
        rows += extra_rows

    tbl = pa.Table.from_pylist(rows)
    con.raw_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    col_defs = []
    for field in tbl.schema:
        dtype = str(field.type).upper()
        if "TIMESTAMP" in dtype or "DATE" in dtype:
            col_defs.append(f'"{field.name}" TIMESTAMP')
        elif "INT" in dtype or "INT64" in dtype:
            col_defs.append(f'"{field.name}" BIGINT')
        elif "FLOAT" in dtype or "DOUBLE" in dtype:
            col_defs.append(f'"{field.name}" DOUBLE')
        elif "BOOL" in dtype:
            col_defs.append(f'"{field.name}" BOOLEAN')
        else:
            col_defs.append(f'"{field.name}" VARCHAR')
    con.raw_sql(f"CREATE TABLE analytics.event_log ({', '.join(col_defs)})")
    # Name the columns instead of relying on each dict's key order: rows come from several builders and a positional INSERT would bind a differently-ordered dict's values to the wrong columns.
    names = tbl.schema.names
    cols = ", ".join(f'"{name}"' for name in names)
    ph = ", ".join(["?"] * len(names))
    for row in rows:
        con.raw_sql(
            f"INSERT INTO analytics.event_log ({cols}) VALUES ({ph})",
            parameters=[row[name] for name in names],
        )


def _defs_and_con_for_session_tests() -> tuple[Definitions, SQLBackend]:
    """Smallest real ``(Definitions, con)`` pair for `WarehouseSession` tests
    loads the same ``examples/definitions/`` YAML `Analysis` uses, against a
    fresh event-log connection built by :func:`_make_event_log_table`.
    """
    from increment.semantics.loader import load

    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    return load("examples/definitions/"), con
