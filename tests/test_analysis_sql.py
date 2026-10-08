"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime

import ibis
import pytest

from increment import Analysis
from increment.errors import CapabilityError, IncrementWarning, InvalidRequestError
from increment.estimation.diagnostics import SRMResult
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.plan import compile_decision_plan
from increment.semantics.design import AdjustmentSet, Observational
from increment.semantics.loader import load
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    Experiment,
    Exposure,
    Fact,
    FactSource,
    MeanMetric,
    Winsorization,
)
from tests.analysis_factory import _recovered_sum, make_analysis, make_analysis_like


def _srm_result(result: object) -> SRMResult:
    assert isinstance(result, SRMResult)
    return result


def _union_horizon_defs():
    """One experiment declaring BOTH ``early_avg`` (an unbounded - no
    ``window_days`` - ``avg`` :class:`MeanMetric`, whose own last event
    is day 2) and ``late_conv`` (a :class:`ConversionMetric` whose own
    last event is day 14), still running (no ``end``/``observation_end``).

    ``early_avg``'s spine right edge is the union of every metric
    DECLARED metric collection's own event horizon - day 14 when
    ``late_conv`` is also in play, day 2 when ``early_avg`` is analysed
    alone (see :func:`_union_horizon_analysis`'s ``metrics=`` override).
    An avg metric with no ``window_days`` divides by the spine's day
    count (``dense.sum_value.sum() / dense.count()``), so these two
    spines produce genuinely different numeric answers for the identical
    metric over the identical units - the two tests below both lean on
    that.
    """
    con = ibis.duckdb.connect()
    exposure_rows = [
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 1, 9, 0, 0),
            "event": "page_view",
            "group_id": g,
            "experiment_id": "union_horizon_exp",
            "value": None,
        }
        for g, uids in [("control", ["c0", "c1"]), ("treatment", ["t0", "t1"])]
        for uid in uids
    ]
    early_values = {"c0": 10.0, "c1": 12.0, "t0": 20.0, "t1": 24.0}
    early_rows = [
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 2, 10, 0, 0),
            "event": "early_event",
            "group_id": None,
            "experiment_id": None,
            "value": v,
        }
        for uid, v in early_values.items()
    ]
    # Only c0/t0 convert in-window (day 3, well inside window_days=7); c1/t1 stay unconverted, so each arm has nonzero variance. Censoring cares only about maturity, not whether an occurrence sits inside or outside the window.
    late_rows = [
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 3, 10, 0, 0),
            "event": "late_event",
            "group_id": None,
            "experiment_id": None,
            "value": None,
        }
        for uid in ("c0", "t0")
    ] + [
        # A second occurrence per unit, on day 14, outside window_days=7 (doesn't count toward conversion) but still part of the raw event stream union_event_horizon reads - pushes early_avg's shared spine to day 14 while late_conv's own maturity (day 8) still clears it, so no censoring.
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 14, 10, 0, 0),
            "event": "late_event",
            "group_id": None,
            "experiment_id": None,
            "value": None,
        }
        for uid in early_values
    ]
    con.create_table("union_horizon_events", obj=exposure_rows + early_rows + late_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM union_horizon_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "early_event", "column": "value"},
                        {"name": "late_event", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "early_avg",
                    "entity": "user_id",
                    "fact": "early_event",
                    "aggregation": "avg_calendar_day",
                    # window_days omitted: unbounded per-unit window -
                    # the branch that divides by the spine's day count.
                },
                {
                    "type": "conversion",
                    "name": "late_conv",
                    "entity": "user_id",
                    "fact": "late_event",
                    "window_days": 7,
                },
            ],
            "experiments": [
                {
                    "name": "union_horizon_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-08-01",
                    # end/observation_end both omitted - still running
                    "control_group": "control",
                    "allocation_scheme": "independent",
                    "plan": {"secondaries": ["early_avg", "late_conv"]},
                },
            ],
        }
    )
    return con, defs


def _union_horizon_analysis(con, defs, *, store, metrics=None):
    """Build an ``Analysis`` for ``union_horizon_exp``.

    *metrics*, when given, is threaded through ``make_analysis``'s
    ``metrics=`` kwarg, which wires the selected metric list directly while
    building the instance. This scopes which metrics are declared for the
    instance without needing a second experiment or exposure population.
    """
    experiment = defs.experiment("union_horizon_exp")
    analysis = make_analysis(
        con,
        defs,
        experiment=experiment,
        store=store,
        metrics=metrics,
    )
    return analysis


def test_constructor_with_new_experiment_refreshes_the_declared_design_control_group():
    """Building from a new experiment refreshes the declared design control
    group while preserving explicitly pinned designs."""
    con, defs = _union_horizon_defs()
    a = _union_horizon_analysis(con, defs, store="none")
    assert a.experiment.control_group == "control"

    a = make_analysis_like(
        a, experiment=a.experiment.model_copy(update={"control_group": "treatment"})
    )
    assert a.experiment.control_group == "treatment"

    # A hand-pinned Encouragement test double owns its own control_group
    # and must be left untouched when rebuilding with a new experiment.
    from increment.semantics.design import (
        Encouragement,
        ExclusionRestriction,
        Randomized,
        UptakeSpec,
    )

    encouragement = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="page_view"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="test double, not a real design"
        ),
    )
    a = make_analysis_like(a, design=encouragement)
    a = make_analysis_like(
        a, experiment=a.experiment.model_copy(update={"control_group": "control"})
    )
    late = a.run(estimands=("late",))
    assert late

    # A hand-pinned Randomized design with a non-default allocation is a
    # deliberate customization and must survive a new experiment build.
    custom_randomized = Randomized(
        control_group="control",
        allocation={"control": 0.3, "treatment": 0.7},
        allocation_scheme="independent",
    )
    a = make_analysis_like(a, design=custom_randomized)
    a = make_analysis_like(
        a, experiment=a.experiment.model_copy(update={"control_group": "treatment"})
    )
    assert _srm_result(a.srm()).expected == {"control": 0.3, "treatment": 0.7}

    # A hand-pinned Observational design is likewise never the plain
    # declared default and must survive untouched too.
    observational = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("x",))
    )
    a = make_analysis_like(a, design=observational)
    a = make_analysis_like(
        a, experiment=a.experiment.model_copy(update={"control_group": "treatment"})
    )
    # The pinned observational design routes the readout to the per-unit
    # frame, which then refuses the undeclared covariate "x" by name.
    with pytest.raises(CapabilityError) as raised:
        a.run()
    assert raised.value.code == "source.native.covariate_unresolved"


@pytest.mark.slow
def test_experiment_analysis_from_defs(seeded_pre_period_con, seeded_defs):
    """Analysis.run() returns one LiftEstimate per experiment metric.

    Validates the full pipeline from definitions through builders to
    estimation with the pre-period covariate materialized: new_onboarding_v2
    declares n_pre_periods=14, and the fixture actually carries pre-exposure
    events, so every metric's covariate moments are built from real data
    rather than an empty lookback. `n_pre_periods` is a materialization
    switch, not a reporting one - `run()` defaults to unadjusted, so the
    ADJUSTMENT itself is pinned by
    test_cuped_covariate_is_per_metric_not_metric_zero, which requests the
    cuped method explicitly.

    Runs on the seeded population, not the 4-unit `con` fixture: two units
    per arm cannot support inference on a binary metric at all (a saturated
    or an all-equal arm has zero variance, and a 50/50 split at n=2 blows
    past infer_lift's log-scale SE ceiling), so `con` would exercise the
    refusal path instead of the pipeline this test is about.
    """
    analysis = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path=seeded_defs,
        con=seeded_pre_period_con,
    )
    assert analysis.experiment.n_pre_periods == 14, (
        "test fixture drifted -- this test intends to materialize a covariate"
    )
    results = analysis.run()

    assert isinstance(results, list)
    assert len(results) > 0, "Expected at least one LiftEstimate"

    expected_metrics = {"purchase_rate", "avg_session_duration", "d7_retention"}
    result_metrics = {r.metric for r in results}

    for m in expected_metrics:
        assert m in result_metrics, f"Missing metric '{m}' in results"

    for r in results:
        assert isinstance(r, LiftEstimate)
        assert isinstance(r.require_lift().value, float)
        assert r.group_id in ("control", "treatment")

    # Check SQL introspection
    panel_sql = analysis.panel_sql()
    assert isinstance(panel_sql, dict)
    for m in expected_metrics:
        assert m in panel_sql, f"Missing metric '{m}' in panel_sql"
        assert isinstance(panel_sql[m], str) and panel_sql[m], f"Empty SQL for {m}"

    summary_sql = analysis.summary_sql()
    assert isinstance(summary_sql, dict)
    for m in expected_metrics:
        assert m in summary_sql, f"Missing metric '{m}' in summary_sql"
        assert isinstance(summary_sql[m], str) and summary_sql[m], f"Empty SQL for {m}"


def test_ratio_metric_with_n_pre_periods_never_builds_covariate(
    seeded_defs, seeded_con, monkeypatch
):
    """A ratio metric with an unadjusted binding must not build covariate
    evidence even when the experiment declares a pre-period."""
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    assert analysis.experiment.n_pre_periods == 14
    ratio_metric = next(m for m in load(seeded_defs).metrics if m.name == "revenue_per_session")
    analysis = make_analysis_like(analysis, [*analysis.metrics, ratio_metric])
    queries: list[str] = []
    execute = seeded_con.to_pyarrow

    def record_query(expr, *args, **kwargs):
        queries.append(ibis.to_sql(expr))
        return execute(expr, *args, **kwargs)

    monkeypatch.setattr(seeded_con, "to_pyarrow", record_query)
    results = analysis.run()
    ratio_rows = [r for r in results if r.metric == "revenue_per_session"]
    assert ratio_rows
    assert all(r.method == "unadjusted" for r in ratio_rows)
    assert not any(
        "ts <" in query.lower().replace('"ts"', "ts").replace("`ts`", "ts") for query in queries
    )


def test_run_rejects_unknown_estimand_loudly(con):
    """A typo'd ``estimands`` value must refuse loudly on the native
    (non-frame) path, not silently drop every non-guardrail metric's rows:
    pre-fix, ``run(estimands=("typo",))`` on an experiment with a declared
    guardrail (``d7_retention``) forced ``itt`` back onto the guardrail
    (which always reports it regardless of ``estimands``) while every
    OTHER metric's rows - which also carry ``estimand="itt"`` - got
    filtered out as "not requested", leaving one silent guardrail-only
    row and no error."""
    analysis = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
    )
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(estimands=("typo",))
    assert raised.value.code == "readout.estimands.unknown"


def test_run_rejects_unidentifiable_estimand_on_native_randomized_path(con):
    """A known-but-unidentifiable ``estimands`` value (``"late"``/
    ``"compliance"``) must refuse loudly on the native (non-frame) path
    for a design that isn't an Encouragement, same as it already does on
    the frame path (``increment.readouts``), rather than returning itt rows the
    caller never requested."""
    analysis = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
    )
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(estimands=("late",))
    assert raised.value.code == "readout.assignment.estimands"


def test_run_itt_estimand_works_on_native_randomized_path(seeded_defs, seeded_con):
    """itt (the only estimand a Randomized design identifies) still works
    on the native path. Runs on the seeded warehouse: the 4-unit micro
    fixture's purchase_rate control arm is saturated (every unit
    purchased) and infer_lift's degenerate-arm guard refuses it, aborting
    run() before the estimands plumbing under test here is reached."""
    results = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        estimands=("itt",)
    )
    assert results


@pytest.mark.slow
def test_experiment_analysis_with_ratio_metric(con):
    """Analysis.run() handles a RatioMetric through the facade.

    Regression test: RatioMetric has no top-level `.fact` (only
    `.numerator.fact`/`.denominator.fact`), and the facade's metric loop
    used to call `_find_fact_source(defs, metric.fact)` unconditionally,
    crashing with AttributeError on any ratio metric. pricing_tier_test
    (n_pre_periods=0) includes revenue_per_session, a RatioMetric.
    """
    analysis = Analysis(
        experiment_name="pricing_tier_test",
        definitions_path="examples/definitions/",
        con=con,
    )
    results = analysis.run()

    assert isinstance(results, list)
    result_metrics = {r.metric for r in results}
    assert "revenue_per_session" in result_metrics, (
        f"Missing ratio metric in results: {result_metrics}"
    )

    ratio_results = [r for r in results if r.metric == "revenue_per_session"]
    assert len(ratio_results) > 0
    for r in ratio_results:
        assert isinstance(r, LiftEstimate)
        assert isinstance(r.require_lift().value, float)
        # Treatment 14 x 40.0/20 sessions = 28.0, control 10 x 20.0/20 = 10.0 - pin the ratio of sums, not just "non-crashing": a numerator/denominator scoped to the wrong window still lands on the correct side of zero.
        assert r.require_lift().value == pytest.approx(28.0 / 10.0 - 1.0, rel=1e-12)
        assert r.group_id == "treatment"

    # SQL introspection must also work for the ratio metric
    panel_sql = analysis.panel_sql()
    assert "revenue_per_session" in panel_sql
    summary_sql = analysis.summary_sql()
    assert "revenue_per_session" in summary_sql


def _analysis_with_ratio_metric_material_censoring(con):
    """Build an ``Analysis`` (no breakout) whose only metric is
    a :class:`RatioMetric` (``revenue_per_session`` = sum(purchase.revenue)
    / count(session_end), ``numerator.window_days=7``) sized so that
    censoring drops a material (>10%) share of enrolled units, forcing
    the warning-attribution branch to resolve ``data_as_of`` as a real
    value.

    Regression fixture for a real crash: a ``RatioMetric``'s
    ``data_as_of`` is composed via ``ibis.least(numerator_as_of,
    denominator_as_of)`` over two SEPARATELY built (and here, since
    numerator/denominator share one physical fact source but are
    queried independently, genuinely distinct) ``ir.Scalar`` expressions
    - a scalar spanning two base table references, which
    ``ir.Scalar.execute()`` cannot resolve directly (it calls
    ``.as_table()`` internally, which raises
    ``ibis.common.exceptions.RelationError`` for a multi-relation
    scalar). This crashed ``Analysis.run()`` for ANY
    RatioMetric whose censoring warning fired - not an edge case.

    2 early units per arm (c_early0/c_early1, t_early0/t_early1, exposed
    2025-08-01) mature within the numerator's 7-day window (real
    purchase/session_end events on 2025-08-02); 4 late units per arm
    (exposed 2025-08-15) do not - their windows close well after
    ``end=2025-08-20``. 8 of 12 enrolled units (67%) get censored, well
    past the 10% warning threshold.

    A ``freshness_anchor`` unit_id (never exposed, so it never joins
    into any real unit's panel - ``unit_day_panel`` only joins events
    strictly post-exposure) contributes a purchase AND a session_end
    event on 2025-08-10, purely to push the raw fact table's observed
    ``max(ts)`` for both facts past the early units' maturity_bound
    (2025-08-08) - without it, ``data_as_of`` itself would (correctly,
    if confusingly for this fixture's purpose) also censor the early
    units, leaving no data to compute a lift estimate from at all.
    """
    if "ratio_censoring_events" not in con.list_tables():
        exposure_rows = []
        for group_id, early_uids, late_uids in [
            ("control", ["c_early0", "c_early1"], [f"c_late{i}" for i in range(4)]),
            ("treatment", ["t_early0", "t_early1"], [f"t_late{i}" for i in range(4)]),
        ]:
            for uid in early_uids:
                exposure_rows.append(
                    {
                        "user_id": uid,
                        "ts": datetime(2025, 8, 1, 9, 0, 0),
                        "event": "page_view",
                        "group_id": group_id,
                        "revenue": None,
                        "experiment_id": "ratio_censoring_exp",
                    }
                )
            for uid in late_uids:
                exposure_rows.append(
                    {
                        "user_id": uid,
                        "ts": datetime(2025, 8, 15, 9, 0, 0),
                        "event": "page_view",
                        "group_id": group_id,
                        "revenue": None,
                        "experiment_id": "ratio_censoring_exp",
                    }
                )
        purchase_amounts = {
            "c_early0": 10.0,
            "c_early1": 12.0,
            "t_early0": 15.0,
            "t_early1": 20.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 8, 2, 10, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in purchase_amounts.items()
        ]
        session_end_counts = {"c_early0": 2, "c_early1": 3, "t_early0": 1, "t_early1": 2}
        session_end_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 8, 2, 10, i, 0),
                "event": "session_end",
                "group_id": None,
                "revenue": None,
                "experiment_id": None,
            }
            for uid, count in session_end_counts.items()
            for i in range(count)
        ]
        freshness_anchor_rows = [
            {
                "user_id": "freshness_anchor",
                "ts": datetime(2025, 8, 10, 12, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": 0.0,
                "experiment_id": None,
            },
            {
                "user_id": "freshness_anchor",
                "ts": datetime(2025, 8, 10, 12, 0, 0),
                "event": "session_end",
                "group_id": None,
                "revenue": None,
                "experiment_id": None,
            },
        ]
        con.create_table(
            "ratio_censoring_events",
            obj=exposure_rows + purchase_rows + session_end_rows + freshness_anchor_rows,
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM ratio_censoring_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "session_end", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "revenue_per_session",
                    "entity": "user_id",
                    "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 7},
                    "denominator": {"fact": "session_end", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "ratio_censoring_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-08-01",
                    "end": "2025-08-20",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue_per_session"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_ratio_metric_with_material_censoring_does_not_crash(con):
    """``Analysis.run()`` on a RatioMetric whose censoring
    warning fires must not crash.

    Regression: the warning's attribution branch resolves a lazy
    ``data_as_of`` via ``.execute()`` - for a RatioMetric this scalar is
    ``ibis.least(numerator_as_of, denominator_as_of)``, spanning two base
    table references, which raised
    ``ibis.common.exceptions.RelationError: ... involves multiple base
    table references`` before the fix (anchoring the resolution in a
    throwaway single-row memtable).
    """
    analysis = _analysis_with_ratio_metric_material_censoring(con)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = analysis.run()

    censoring_warnings = [
        w
        for w in record
        if isinstance(w.message, IncrementWarning)
        and w.message.code == "frame.censoring.dropped_units"
        and "revenue_per_session" in str(w.message)
    ]
    assert len(censoring_warnings) == 1, [str(w.message) for w in record]
    message = str(censoring_warnings[0].message)
    assert "8 of 12" in message, message
    assert "data_as_of" in message, message
    assert "2025-08-10" in message, message  # resolved from the multi-relation scalar

    ratio_results = [r for r in results if r.metric == "revenue_per_session"]
    assert len(ratio_results) == 1
    assert math.isfinite(ratio_results[0].require_lift().value)


@pytest.mark.slow
def test_cuped_covariate_is_per_metric_not_metric_zero(con, seeded_pre_period_con, seeded_defs):
    """CUPED's pre-period covariate for each metric comes from THAT metric's
    own fact, not the first metric's.

    Regression test for a bug where pre_events was built once outside the
    per-metric loop from the first configured metric (purchase_rate) and reused for
    every metric - avg_session_duration would have gotten purchase-rate's
    occurrence data as its "covariate" instead of its own pre-period
    session-duration sums, silently violating CUPED's "same aggregation
    for X and Y" rule with no error.

    Verifies numerically: execute avg_session_duration's summary SQL and
    confirm sum_x matches the HAND-COMPUTED pre-period session_end sum
    (group C: u1=200+u2=210=410; group T: u3=220+u4=230=450) - these
    values are on a completely different scale from purchase revenue
    (19.99-49.99) or purchase-occurrence counts (0/1), so a wrong-metric
    covariate would produce a numerically distinguishable (and wrong)
    sum_x.
    """

    analysis = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
    )

    summary_sql = analysis.summary_sql()
    df = con.sql(summary_sql["avg_session_duration"]).execute()
    df = df.set_index("group_id")

    sum_x = _recovered_sum(df["n"], df["ref_x"], df["cx1"])
    assert sum_x.loc["control"] == pytest.approx(410.0), (
        f"group control sum_x should be 410.0 (200+210 from session_end pre-period), "
        f"got {sum_x.loc['control']} -- covariate may be sourced from the wrong metric"
    )
    assert sum_x.loc["treatment"] == pytest.approx(450.0), (
        f"group treatment sum_x should be 450.0 (220+230 from session_end pre-period), "
        f"got {sum_x.loc['treatment']} -- covariate may be sourced from the wrong metric"
    )

    # The cuped method must be reachable via the public API on data that supports it: runs on the seeded warehouse (with a pre-period) rather than the hand-written fixture, scoped to the two metrics with a genuine nonzero-variance pre-period covariate (d7_retention shares its fact with the exposure definition, so has none).
    scoped = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path=seeded_defs,
        con=seeded_pre_period_con,
    )
    results = scoped.run(
        decision_method=Method(name="unadjusted"),
        sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
        metrics=["purchase_rate", "avg_session_duration"],
    )
    methods_seen = {r.method for r in results}
    assert "cuped" in methods_seen, f"cuped method not in results: {methods_seen}"
    assert "unadjusted" in methods_seen


def test_exposure_events_not_dropped_when_fact_source_lacks_experiment_id():
    """A fact-based exposure whose source has no `experiment_id` column
    must NOT be silently filtered to zero rows.

    Regression test: _build_exposure_events_table's experiment-scoping
    filter used to check "experiment_id" in tbl.columns AFTER
    _get_fact_table/_rename_to_builder_cols, which unconditionally
    synthesizes an all-NULL experiment_id column when the source lacks
    one - so the presence check always passed, and `NULL == name` is
    never true in SQL, silently filtering out every row. Fixed by
    checking the RAW (pre-synthesis) table's columns instead.
    """
    con = ibis.duckdb.connect()
    con.create_table(
        "raw_events",
        obj=[
            {
                "unit_id": "u1",
                "event_at": datetime(2025, 1, 1),
                "event": "signup",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "event_at": datetime(2025, 1, 2),
                "event": "signup",
                "group_id": "treatment",
            },
        ],
    )

    fs = FactSource(
        name="raw",
        sql="SELECT * FROM raw_events",  # deliberately no experiment_id column
        timestamp_column="event_at",
        entities=["unit_id"],
        facts=[Fact(name="signup", column=None)],
    )
    exposure = Exposure(name="on_signup", fact="signup")
    experiment = Experiment(
        name="no_id_experiment",
        exposure="on_signup",
        unit="unit_id",
        start=datetime(2025, 1, 1),
        control_group="control",
        allocation_scheme="independent",
        plan=AnalysisPlan(),
    )
    defs = Definitions(fact_sources=[fs], exposures=[exposure], experiments=[experiment])

    analysis = make_analysis(
        con,
        defs,
        experiment=experiment,
        backend="duckdb",
        metrics=[],
        _exp_lookup_exposures={exposure.name: exposure},
    )

    result = _srm_result(analysis.srm(expected={"control": 0.5, "treatment": 0.5}))
    assert sum(result.observed.values()) == 2, (
        f"expected both signup events retained (no experiment_id to filter on), got {result.observed}"
    )


def test_enrollment_not_empty_when_fact_source_lacks_experiment_id():
    """Enrollment survives a source with no `experiment_id` column.

    Regression: the events table kept its rows (test above) but
    `first_exposures` then semi-joined on the synthesized all-NULL
    `experiment_id`, and `NULL = NULL` is never true - so exposures came
    back empty and every readout reported zero enrolled units, or failed
    far away with "control_group not found in arms".
    """
    con = ibis.duckdb.connect()
    con.create_table(
        "raw_group_events",
        obj=[
            {
                "unit_id": "u1",
                "event_at": datetime(2025, 1, 1),
                "event": "signup",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "event_at": datetime(2025, 1, 2),
                "event": "signup",
                "group_id": "treatment",
            },
        ],
    )

    fs = FactSource(
        name="raw",
        sql="SELECT * FROM raw_group_events",  # deliberately no experiment_id column
        timestamp_column="event_at",
        entities=["unit_id"],
        facts=[Fact(name="signup", column=None)],
    )
    exposure = Exposure(name="on_signup", fact="signup")
    experiment = Experiment(
        name="no_id_experiment",
        exposure="on_signup",
        unit="unit_id",
        start=datetime(2025, 1, 1),
        control_group="control",
        allocation_scheme="independent",
        plan=AnalysisPlan(),
    )
    defs = Definitions(fact_sources=[fs], exposures=[exposure], experiments=[experiment])

    analysis = make_analysis(
        con,
        defs,
        experiment=experiment,
        backend="duckdb",
        metrics=[],
        _exp_lookup_exposures={exposure.name: exposure},
    )

    result = _srm_result(analysis.srm(expected={"control": 0.5, "treatment": 0.5}))
    assert result.observed == {"control": 1, "treatment": 1}


@pytest.mark.slow
def test_panel_and_summary_sql_dialect_follows_connection_name(con, monkeypatch):
    """With no explicit ``backend=``, the SQL surfaces take their dialect from
    the connection: a BigQuery-named connection gets BigQuery's backtick
    identifier quoting, not the DuckDB fixture connection's double quotes.

    The connection itself stays DuckDB throughout; only its reported name
    changes, which is what the renderer boundary reads.
    """
    analysis = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
    )
    assert con.name == "duckdb"

    expected_keys = {"purchase_rate", "avg_session_duration", "d7_retention"}

    panel_duckdb = analysis.panel_sql()
    summary_duckdb = analysis.summary_sql()
    assert set(panel_duckdb) == expected_keys
    assert set(summary_duckdb) == expected_keys
    duckdb_sql = [*panel_duckdb.values(), *summary_duckdb.values()]
    assert all(isinstance(sql, str) and sql for sql in duckdb_sql)
    assert not any("`" in sql for sql in duckdb_sql)

    monkeypatch.setattr(con, "name", "bigquery")

    panel_bigquery = analysis.panel_sql()
    summary_bigquery = analysis.summary_sql()
    assert set(panel_bigquery) == expected_keys
    assert set(summary_bigquery) == expected_keys
    bigquery_sql = [*panel_bigquery.values(), *summary_bigquery.values()]
    assert all(isinstance(sql, str) and sql for sql in bigquery_sql)
    assert all("`" in sql for sql in bigquery_sql)


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery", "postgres"])
def test_panel_and_summary_sql_compile_for_advertised_dialects(con, dialect):
    with Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
        backend=dialect,
        store="none",
    ) as analysis:
        panel = analysis.panel_sql()
        summary = analysis.summary_sql()
    expected = {"purchase_rate", "avg_session_duration", "d7_retention"}
    assert set(panel) == expected
    assert set(summary) == expected
    assert all(sql.strip() for sql in [*panel.values(), *summary.values()])


def test_explicit_backend_overrides_connection_name(con):
    """An explicit ``backend=`` constructor argument reaches the SQL renderer:
    the emitted SQL is compiled for that dialect, not the connection's."""
    analysis = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
        backend="bigquery",
    )
    assert con.name == "duckdb"  # the connection itself is still duckdb

    panel_sql = analysis.panel_sql()
    summary_sql = analysis.summary_sql()

    expected_keys = {"purchase_rate", "avg_session_duration", "d7_retention"}
    assert set(panel_sql) == expected_keys
    assert set(summary_sql) == expected_keys
    emitted = [*panel_sql.values(), *summary_sql.values()]
    assert all(isinstance(sql, str) and sql for sql in emitted)
    assert all("`" in sql for sql in emitted)  # BigQuery identifier quoting

    default = Analysis(
        experiment_name="new_onboarding_v2",
        definitions_path="examples/definitions/",
        con=con,
    )
    duckdb_sql = [*default.panel_sql().values(), *default.summary_sql().values()]
    assert not any("`" in sql for sql in duckdb_sql)


@pytest.mark.parametrize("method", ["run", "run_breakout"])
def test_native_analysis_rejects_percentile_winsorization_before_reduction(
    con, monkeypatch, method
):
    analysis = Analysis.from_definitions(
        "new_onboarding_v2", "examples/definitions", con, store="none"
    )
    revenue_metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        winsorization=Winsorization(
            upper_percentile=0.99, inference={"method": "joint-rank-projection-v1"}
        ),
    )
    # Pin a revenue-only plan so the ad-hoc metric reaches the
    # percentile-winsorization guard without re-resolving new_onboarding_v2's
    # declared plan against metrics it never named.
    analysis = make_analysis_like(
        analysis,
        [revenue_metric],
        plan=compile_decision_plan(None, [revenue_metric], path="warehouse"),
    )

    monkeypatch.setattr(
        con,
        "to_pyarrow",
        lambda *_args, **_kwargs: pytest.fail("percentile refusal reached reduction"),
    )

    with pytest.raises(CapabilityError) as raised:
        getattr(analysis, method)()
    expected = (
        "estimation.winsor.support_required"
        if method == "run"
        else "readout.metric.percentile_winsorization"
    )
    assert getattr(raised.value, "code", None) == expected


def test_zero_metrics_returns_empty_list(con):
    """analysis.run() with no metrics/guardrails returns [] instead of crashing."""
    analysis = Analysis("pricing_tier_test", definitions_path="examples/definitions/", con=con)
    analysis = make_analysis_like(analysis, [], plan=AnalysisPlan())

    result = analysis.run()
    assert result == []


def test_zero_metrics_panel_sql_returns_empty_dict(con):
    """analysis.panel_sql() with no metrics returns {}."""
    analysis = Analysis("pricing_tier_test", definitions_path="examples/definitions/", con=con)
    analysis = make_analysis_like(analysis, [], plan=AnalysisPlan())

    result = analysis.panel_sql()
    assert result == {}


def test_zero_metrics_summary_sql_returns_empty_dict(con):
    """analysis.summary_sql() with no metrics returns {}."""
    analysis = Analysis("pricing_tier_test", definitions_path="examples/definitions/", con=con)
    analysis = make_analysis_like(analysis, [], plan=AnalysisPlan())

    result = analysis.summary_sql()
    assert result == {}


def test_panel_sql_and_summary_sql_never_execute_even_when_censoring_would_warn(con, monkeypatch):
    """``panel_sql``/``summary_sql`` promise SQL text without executing
    anything (see their own docstrings) - proven with warnings-as-errors:
    ``_analysis_with_stale_retention_data``'s fixture censors its whole
    cohort (a 100% drop, well past the material-censoring threshold) once
    actually executed, via a real ``data_as_of`` computed from the fact
    table. If either ``_data_as_of`` eagerly executed, or the SQL-preview
    stages let the censoring warning fire, this would raise - since
    neither happens, no warning is raised and both calls still return
    real, non-empty SQL text.
    """
    analysis = _analysis_with_stale_retention_data(con)
    monkeypatch.setattr(
        con,
        "to_pyarrow",
        lambda *_args, **_kwargs: pytest.fail("SQL preview executed a warehouse query"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        panel_sql = analysis.panel_sql()
        summary_sql = analysis.summary_sql()

    assert panel_sql["returned"]
    assert summary_sql["returned"]


def _analysis_with_stale_retention_data(con):
    """Build an ``Analysis`` (no breakout) with one bounded
    RetentionMetric where the fact source's real data stops well before
    the cohort's maturity date, but ``observation_end`` is declared far
    enough out that - WITHOUT the ``data_as_of`` cap threaded from the
    real fact table - the cohort would wrongly be treated as matured.

    4 units all exposed 2025-08-01 (2 per arm). c1/t1 return via an
    ``app_open`` event on 2025-08-02 (a real, already-loaded event); c2/t2
    have no ``app_open`` row at all - indistinguishable, in the raw
    event data, between "did not return" and "data for that unit hasn't
    loaded yet". The fact source's freshness bound comes from the metric's
    own fact (``app_open``, filtered via ``event == fact`` - see
    :meth:`Analysis.data_as_of`, whose last event is
    2025-08-02, long before the cohort's maturity date (exposure + window_days=10 =
    2025-08-11). ``observation_end=2025-12-01`` is declared far past that
    maturity date, so only the ``data_as_of`` cap can prevent the cohort
    from being (wrongly) admitted.
    """
    if "stale_retention_events" not in con.list_tables():
        exposure_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "experiment_id": "stale_retention_exp",
            }
            for group_id, units in {
                "control": ["c1", "c2"],
                "treatment": ["t1", "t2"],
            }.items()
            for uid in units
        ]
        return_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 2, 10, 0, 0),
                "event": "app_open",
                "group_id": None,
                "experiment_id": None,
            }
            for uid in ["c1", "t1"]
        ]
        con.create_table("stale_retention_events", obj=exposure_rows + return_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM stale_retention_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "app_open", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "retention",
                    "name": "returned",
                    "entity": "unit_id",
                    "fact": "app_open",
                    "threshold_days": [1, 10],
                }
            ],
            "experiments": [
                {
                    "name": "stale_retention_exp",
                    "exposure": "e",
                    "unit": "unit_id",
                    "start": "2025-08-01",
                    "end": "2025-08-01",
                    "observation_end": "2025-12-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["returned"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_public_facade_uses_experiment_unit_when_entities_are_reordered():
    """The public facade keys assignment counts by Experiment.unit, not
    the first entity declared by a fact source."""
    con = ibis.duckdb.connect()
    con.create_table(
        "reordered_entities",
        obj=[
            {
                "event_at": datetime(2025, 1, 1),
                "session_id": "s1",
                "user_id": "u1",
                "event": "assignment",
                "group_id": "control",
            },
            {
                "event_at": datetime(2025, 1, 1),
                "session_id": "s1",
                "user_id": "u2",
                "event": "assignment",
                "group_id": "treatment",
            },
        ],
    )
    fs = FactSource(
        name="events",
        sql="SELECT * FROM reordered_entities",
        timestamp_column="event_at",
        entities=["session_id", "user_id"],
        facts=[Fact(name="assignment", column=None)],
    )
    exposure = Exposure(name="on_assignment", fact="assignment")
    experiment = Experiment(
        name="reordered_unit_exp",
        exposure="on_assignment",
        unit="user_id",
        start=datetime(2025, 1, 1),
        control_group="control",
        allocation_scheme="independent",
        plan=AnalysisPlan(),
    )
    defs = Definitions(fact_sources=[fs], exposures=[exposure], experiments=[experiment])
    analysis = make_analysis(
        con,
        defs,
        experiment=experiment,
        metrics=[],
        _exp_lookup_exposures={exposure.name: exposure},
    )

    result = _srm_result(analysis.srm(expected={"control": 0.5, "treatment": 0.5}))
    assert result.observed == {"control": 1, "treatment": 1}
