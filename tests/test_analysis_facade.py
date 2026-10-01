"""Test the Analysis facade against example definitions."""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from importlib import import_module
from typing import cast
from uuid import uuid4

import ibis
import pytest
from pydantic import ValidationError

from increment import Analysis, IdentificationError
from increment import readouts as readout_functions
from increment.breakout.estimates import (
    BreakoutEstimates,
    DailyLiftEstimates,
    DailyMetricValues,
    LiftEstimates,
)
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.diagnostics import SRMResult
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.query.artifact_contract import ArtifactContractError
from increment.semantics.design import AdjustmentSet, Observational
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    MeanMetric,
    RetentionMetric,
)
from tests.analysis_factory import make_analysis, make_analysis_like
from tests.sequential_cases import registration
from tests.test_sequential_public_sources import gaussian_plan
from tests.test_unit_day_artifact_facade import _extensions, _published
from tests.warning_codes import warning_codes


def _lift_rows(rows: object) -> list[LiftEstimate]:
    return cast(list[LiftEstimate], rows)


@contextmanager
def _ignore_ibis_deprecation():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
            category=DeprecationWarning,
            module=r"ibis\.backends\.duckdb",
        )
        yield


@pytest.fixture(scope="session", autouse=True)
def dataframe_dependencies():
    """Load shared dependencies during setup rather than the first test call."""
    import_module("pandas")
    import_module("ibis.backends.duckdb")


@pytest.mark.slow
class TestAnalysisReturnsCollectionTypes:
    """Uses new_onboarding_v2 (examples/definitions/): declares
    purchase_rate/avg_session_duration plus a d7_retention guardrail
    (excluded via metrics= for the day-axis methods, which reject any
    RetentionMetric) and a country breakout (for run_breakout and the
    dimensioned day-axis paths)."""

    def analysis(self, con) -> Analysis:
        return Analysis(
            experiment_name="new_onboarding_v2",
            definitions_path="examples/definitions/",
            con=con,
        )

    def test_run_returns_lift_estimates(self, seeded_defs, seeded_con):
        # The seeded warehouse, not the 4-unit micro fixture: at n=2/arm the micro warehouse's purchase_rate is refused by infer_lift's precision guards (saturated control arm); this test is about the return type.
        import pandas as pd

        results = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run()
        assert isinstance(results, LiftEstimates)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results.to_frame(), pd.DataFrame)

    @pytest.mark.slow
    def test_run_breakout_returns_breakout_estimates(self, con):
        with pytest.warns(
            UserWarning,
            match=r"run_breakout: segment .* (?:excluded|skipped|no usable|fewer than)",
        ):
            with _ignore_ibis_deprecation():
                results = self.analysis(con).run_breakout()
        assert isinstance(results, BreakoutEstimates)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_daily_returns_daily_metric_values(self, con):
        analysis = self.analysis(con)
        non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
        results = analysis.run_daily(metrics=non_retention)
        assert isinstance(results, DailyMetricValues)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_daily_dimensioned_returns_daily_metric_values(self, con):
        """The dimensioned path builds `segment_results` independently of
        the un-dimensioned path - both branches need the wrap, not just
        one."""
        analysis = self.analysis(con)
        non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
        results = analysis.run_daily(dimension="country", metrics=non_retention)
        assert isinstance(results, DailyMetricValues)
        assert len(results) > 0

    def test_run_daily_lift_returns_daily_lift_estimates(self, con):
        analysis = self.analysis(con)
        non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
        results = analysis.run_daily_lift(metrics=non_retention)
        assert isinstance(results, DailyLiftEstimates)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_daily_lift_dimensioned_returns_daily_lift_estimates(self, con):
        analysis = self.analysis(con)
        non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
        results = analysis.run_daily_lift(dimension="country", metrics=non_retention)
        assert isinstance(results, DailyLiftEstimates)
        assert len(results) > 0

    def test_run_asof_returns_daily_metric_values(self, con):
        analysis = self.analysis(con)
        non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
        results = analysis.run_asof(metrics=non_retention)
        assert isinstance(results, DailyMetricValues)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_asof_lift_returns_daily_lift_estimates(self, con):
        analysis = self.analysis(con)
        non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
        results = analysis.run_asof_lift(metrics=non_retention)
        assert isinstance(results, DailyLiftEstimates)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_empty_metrics_returns_typed_empty_collections(self, con):
        """`metrics=[]` (rather than a data-driven sparse result) is the
        cleanest way to force the early-return branch every day-axis
        method has - confirming even THAT branch returns the concrete
        type, not a bare `[]`, is the whole point: it's the easiest
        branch to accidentally leave unwrapped."""
        analysis = self.analysis(con)
        assert isinstance(analysis.run_daily(metrics=[]), DailyMetricValues)
        assert isinstance(analysis.run_daily_lift(metrics=[]), DailyLiftEstimates)
        assert isinstance(analysis.run_asof(metrics=[]), DailyMetricValues)
        assert isinstance(analysis.run_asof_lift(metrics=[]), DailyLiftEstimates)

    def test_run_asof_lift_rejects_removed_correction_before_empty_metrics_return(self, con):
        analysis = self.analysis(con)

        with pytest.raises(TypeError):
            analysis.run_asof_lift(
                metrics=[],
                correction="holm",  # ty: ignore[unknown-argument]
            )


def test_analysis_build_state_refuses_definitions_without_a_declared_design():
    from types import SimpleNamespace

    from increment.analysis import Analysis

    src = SimpleNamespace(context=SimpleNamespace(design=None))
    with pytest.raises(InvalidRequestError) as exc_info:
        Analysis._build_state(
            src=src,  # ty: ignore[invalid-argument-type]
            defs=object(),  # ty: ignore[invalid-argument-type]
            experiment=None,
            con=None,
            session=None,
            experiment_name="x",
            backend=None,
        )
    assert exc_info.value.code == "facade.analysis.definitions_state_requires_identification"


def test_analysis_from_source_refuses_a_contrast_design_that_disagrees_with_the_source():
    from types import SimpleNamespace

    from increment._study import SwitchbackStudyEnvelope
    from increment.analysis import Analysis
    from increment.decision import ContrastContext
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    design = Randomized(control_group="control")
    other_design = Randomized(control_group="other", allocation={"other": 0.5, "treatment": 0.5})
    context = ContrastContext(
        study_id="t",
        study=SwitchbackStudyEnvelope(
            identification=other_design,
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(),
                window=SwitchbackWindow(washout_steps=1, observation_steps=2),
            ),
        ),
        metrics=(),
        procedures={},
    )
    src = SimpleNamespace(context=context)
    with pytest.raises(InvalidRequestError) as exc_info:
        Analysis._from_source(src, design)  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "facade.analysis.source_context_design_disagrees_arm"


def test_analysis_from_source_refuses_an_arm_design_that_disagrees_with_the_source():
    from types import SimpleNamespace

    from increment.analysis import Analysis
    from increment.semantics.design import Randomized

    design = Randomized(control_group="control")
    other_design = Randomized(control_group="other")
    context = SimpleNamespace(design=other_design)
    src = SimpleNamespace(context=context, operations=frozenset())
    with pytest.raises(InvalidRequestError) as exc_info:
        Analysis._from_source(src, design)  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "facade.analysis.source_context_design_disagrees_arm"


def test_run_empty_metrics_on_experiment_less_double_returns_empty_without_reading(
    monkeypatch,
):
    """A bare double built without ``defs=`` (e.g. a seam source's own
    test substrate) has no ``_experiment`` at all. The empty-selection
    path must not inspect experiment state or acquire moments."""
    from tests.analysis_factory import _native_source

    analysis = make_analysis(defs=None)

    def fail_on_read(*args, **kwargs):
        raise AssertionError("run() acquired moments for an empty selection")

    monkeypatch.setattr(_native_source(analysis), "moments", fail_on_read)

    assert analysis.run(metrics=[]) == LiftEstimates([])


def test_run_empty_metrics_still_refuses_source_without_design_before_reading(monkeypatch):
    from increment.sources import MomentsSource

    metric = MeanMetric(name="purchase_rate", entity="user", fact="purchase_rate")
    source = MomentsSource(
        [],
        metrics=[metric],
        study_id="missing-design",
        design=None,
        plan=AnalysisPlan(),
    )
    read_calls = []

    def fail_on_read(*args, **kwargs):
        read_calls.append((args, kwargs))
        raise AssertionError("run() read moments before validating the request")

    monkeypatch.setattr(source, "moments", fail_on_read)
    analysis = Analysis._from_source(source)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(metrics=[])

    assert raised.value.code == "readout.design.required"
    assert read_calls == []


def test_run_empty_metrics_on_experiment_less_double_still_refuses_call_time_policy():
    """The empty-selection branch still refuses a non-default call-time
    policy kwarg before returning early - an empty result must not
    silently swallow a dropped guardrail."""
    analysis = make_analysis(defs=None)
    with pytest.raises(TypeError):
        analysis.run(metrics=[], margins={"x": 0.05})  # ty: ignore[unknown-argument]


def test_run_empty_metrics_on_experiment_less_double_still_validates_estimands():
    analysis = make_analysis(defs=None)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(metrics=[], estimands=("typo",))
    assert raised.value.code == "readout.estimands.unknown"


@pytest.mark.slow
def test_analysis_run_native_randomized_forwards_prior(seeded_con, seeded_defs):
    from increment.estimation import Normal

    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con) as analysis:
        flat = next(r for r in _lift_rows(analysis.run()) if r.metric == "purchase_rate")
        shrunk = next(
            r
            for r in _lift_rows(analysis.run(prior=Normal(mu=0.0, sigma=0.01)))
            if r.metric == "purchase_rate"
        )

    assert flat.require_lift().value > 0.0
    assert abs(shrunk.require_lift().value) < abs(flat.require_lift().value)


def _capture_readout(con, definitions_path, readout: str) -> dict[str, float | None]:
    """Re-capture one readout with the SAME keying as scripts/capture_parity_baseline.py.

    Keep this in lockstep with that script - if the two disagree on keys, the
    baseline comparison checks different things and silently passes.
    """
    a = Analysis("new_onboarding_v2", definitions_path, con)

    def value_of(row) -> float | None:
        value = row.value
        if value is None:
            assert row.unavailable is not None
            return None
        return value.value

    def lift_value(row) -> float | None:
        lift = row.lift
        if lift is None:
            if getattr(row, "reference_kind", None) == "binomial":
                assert row.binomial_set is not None and not row.binomial_set.point_available
                return None
            if hasattr(row, "excluded"):
                assert row.excluded is not None
            else:
                assert row.unavailable is not None
            return None
        return lift.value

    if readout == "run":
        return {
            f"{e.metric}|{e.group_id}": lift_value(e)
            for e in _lift_rows(a.run(decision_method=Method(name="unadjusted")))
        }
    if readout == "run_daily":
        return {f"{v.metric}|{v.group_id}|{v.ds.isoformat()}": value_of(v) for v in a.run_daily()}
    if readout == "run_asof":
        return {f"{v.metric}|{v.group_id}|{v.ds.isoformat()}": value_of(v) for v in a.run_asof()}
    if readout == "run_breakout":
        return {
            f"{e.metric}|{e.group_id}|{e.dimension}={e.dimension_value}": lift_value(e)
            for e in a.run_breakout(decision_method=Method(name="unadjusted"))
        }
    if readout == "run_daily_lift":
        return {
            f"{e.metric}|{e.group_id}|{e.ds}|{e.estimand}": lift_value(e)
            for e in a.run_daily_lift(decision_method=Method(name="unadjusted"))
        }
    if readout == "run_asof_lift":
        return {
            f"{e.metric}|{e.group_id}|{e.ds}|{e.estimand}": lift_value(e)
            for e in a.run_asof_lift(decision_method=Method(name="unadjusted"))
        }
    raise ValueError(f"unknown readout {readout!r}")


@pytest.mark.slow
@pytest.mark.parametrize(
    "readout", ["run", "run_daily", "run_asof", "run_breakout", "run_daily_lift", "run_asof_lift"]
)
def test_no_readout_drifts_from_the_pinned_baseline(seeded_con, seeded_defs, readout):
    """Every readout on the shipped example seed answers what it is pinned to.

    A run()-only gate would miss the day-axis semantics - retention observation
    bands and asof cumulation - which is exactly where the spine's right edge
    has the most leverage.

    This also locks fused-vs-materialized parity: the baseline is captured from
    ONE Analysis reused across all six readouts (so it materializes from the
    second on), while this test builds a fresh - always fused - instance per
    readout. A number that depends on ``store`` shows up here as drift.

    A failure is not automatically a bug, but it is always a semantics change
    reaching every user with comparable data: find the commit that moved the
    number, decide whether the new answer is the intended one, and only then
    regenerate (see ``scripts/capture_parity_baseline.py``).
    """
    from tests.test_parity import load_parity_baseline

    expected = load_parity_baseline(readout)
    got = _capture_readout(seeded_con, seeded_defs, readout)

    assert set(got) == set(expected), (
        f"{readout}: key set changed. missing={sorted(set(expected) - set(got))[:5]} "
        f"new={sorted(set(got) - set(expected))[:5]}"
    )
    for key, want in expected.items():
        if want is None:
            assert got[key] is None, f"{readout}/{key} moved {want} -> {got[key]}"
        else:
            assert got[key] is not None, f"{readout}/{key} moved {want} -> {got[key]}"
            assert got[key] == pytest.approx(want, rel=1e-12, nan_ok=True), (
                f"{readout}/{key} moved {want} -> {got[key]}"
            )


def test_analysis_from_unit_summary_runs():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    a = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    results = a.run()
    assert isinstance(results, LiftEstimates)
    # treatment mean 30.0, control mean 17.75
    assert results[0].require_lift().value == pytest.approx(30.0 / 17.75 - 1, rel=1e-9)


def test_analysis_from_unit_summary_srm():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    a = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    result = a.srm(expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(result, SRMResult)
    assert result.observed == {"treatment": 2, "control": 2}
    assert result.is_srm is False


def test_analysis_srm_defaults_to_anytime_valid_inference():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "user_id": range(8),
            "group": ["control"] * 4 + ["treatment"] * 4,
            "revenue": [1.0] * 8,
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="group",
        control="control",
        metrics={"revenue": "mean"},
    )

    result = analysis.srm(expected={"control": 0.5, "treatment": 0.5})

    assert isinstance(result, SRMResult)
    assert result.inference == "always_valid"
    assert result.alpha == 0.001


def test_analysis_srm_forwards_explicit_fixed_inference():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "user_id": range(8),
            "group": ["control"] * 4 + ["treatment"] * 4,
            "revenue": [1.0] * 8,
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="group",
        control="control",
        metrics={"revenue": "mean"},
    )

    result = analysis.srm(inference="fixed", alpha=0.05)

    assert isinstance(result, SRMResult)
    assert result.inference == "fixed"
    assert result.alpha == 0.05
    assert result.log_e_value is None


def test_analysis_from_unit_summary_on_unassigned_threads_through():
    """The facade forwards on_unassigned to the frame source: default
    refuses null group labels naming the knob; 'exclude' surfaces the
    excluded count in unit_counts() and beside srm() with no phantom arm,
    no extra estimate, and no chi-square degree of freedom."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(10)],
            "variant": ["treatment"] * 4 + ["control"] * 4 + [None] * 2,
            "revenue": [25.0, 35.0, 22.5, 13.0, 18.0, 21.0, 16.5, 27.0, 9.0, 11.0],
        }
    )
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
        )
    assert raised.value.code == "source.frame.unassigned"
    a = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        on_unassigned="exclude",
    )
    assert [e.group_id for e in _lift_rows(a.run())] == ["treatment"]
    result = a.srm(expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(result, SRMResult)
    assert result.observed == {"treatment": 4, "control": 4}
    assert result.df == 1
    assert result.unassigned_units == 2


@pytest.mark.slow
def test_analysis_native_srm(seeded_con, seeded_defs, monkeypatch):
    """The native (`from_definitions`) `srm()` branch: exposure counts
    read straight off `first_exposures`, never a metric's moments.

    Regression coverage for the branch itself - ``examples._seed.seed_event_log``
    splits ``new_onboarding_v2`` exactly 50/50 (treatment/control), so a balanced result here
    proves the group-by-count query is correct, not just that some number
    came back.
    """
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con) as a:
        queries: list[str] = []
        to_pyarrow = seeded_con.to_pyarrow

        def record_query(expr, *args, **kwargs):
            queries.append(str(expr))
            return to_pyarrow(expr, *args, **kwargs)

        monkeypatch.setattr(seeded_con, "to_pyarrow", record_query)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = a.srm(expected={"control": 0.5, "treatment": 0.5})
        assert "query.integrity.mixed_assignments_excluded" not in warning_codes(caught)
        assert isinstance(result, SRMResult)
        assert result.observed["treatment"] == result.observed["control"]
        assert result.is_srm is False
        assert result.inference == "always_valid"
        assert result.mixed_assignment_units == 0
        assert "(mixed assignment)" not in result.observed
        assert len(queries) == 2
        assert "first_exposure_ts" in queries[-1]
        assert not any("measure_stats" in query for query in queries)


@pytest.mark.slow
def test_analysis_native_srm_zero_fills_explicit_missing_arm_for_always_valid_prefix(
    seeded_defs,
):
    """A native cumulative prefix retains an explicit, unobserved arm."""
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=28)
    con.raw_sql("DELETE FROM analytics.event_log WHERE group_id = 'treatment'")

    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, con) as a:
        result = a.srm(expected={"control": 0.5, "treatment": 0.5})

    assert isinstance(result, SRMResult)
    assert result.inference == "always_valid"
    assert result.observed == {"control": 14, "treatment": 0}
    assert result.is_srm is True


@pytest.mark.slow
def test_analysis_native_srm_always_valid_requires_explicit_allocation(seeded_defs, monkeypatch):
    """Definition-backed analyses have no implicit known allocation."""
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=28)

    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, con) as a:
        queries: list[str] = []
        to_pyarrow = con.to_pyarrow

        def record_query(expr, *args, **kwargs):
            queries.append(str(expr))
            return to_pyarrow(expr, *args, **kwargs)

        monkeypatch.setattr(con, "to_pyarrow", record_query)
        with pytest.raises(InvalidRequestError) as raised:
            a.srm()
        assert raised.value.code == "estimation.diagnostics.always_srm_predeclared"
        assert queries == []
        fixed = a.srm(inference="fixed")

    assert isinstance(fixed, SRMResult)
    assert fixed.observed == {"control": 14, "treatment": 14}
    assert fixed.log_e_value is None


def _extend_mixed_assignment_event_horizon(con) -> None:
    """Keep the seeded purchase fact fresh through the experiment horizon."""
    con.raw_sql(
        """
        INSERT INTO analytics.event_log
        SELECT * REPLACE (
            TIMESTAMP '2025-02-15 12:00:00' AS event_at,
            'horizon_sentinel' AS user_id,
            'horizon-sentinel' AS session_id
        )
        FROM analytics.event_log
        WHERE event = 'purchase'
        LIMIT 1
        """
    )


def _insert_mixed_assignment(con) -> None:
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "SELECT * REPLACE (CASE WHEN group_id = 'treatment' THEN 'control' ELSE 'treatment' END AS group_id) "
        "FROM analytics.event_log "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id IS NOT NULL "
        "LIMIT 1"
    )


def _insert_null_assignment(con) -> None:
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "SELECT * REPLACE (NULL AS group_id) "
        "FROM analytics.event_log "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id IS NOT NULL "
        "LIMIT 1"
    )


def _replace_mixed_assignment(con) -> None:
    rows = con.to_pyarrow(
        con.sql(
            """
            SELECT user_id
            FROM analytics.event_log
            WHERE experiment_id = 'new_onboarding_v2'
            GROUP BY user_id
            HAVING COUNT(DISTINCT group_id) > 1
            LIMIT 1
            """
        )
    ).to_pylist()
    assert rows
    old_unit = str(rows[0]["user_id"]).replace("'", "''")
    con.raw_sql(
        "DELETE FROM analytics.event_log "
        f"WHERE experiment_id = 'new_onboarding_v2' AND user_id = '{old_unit}'"
    )
    _insert_mixed_assignment(con)


def _remove_second_assignment(con) -> None:
    rows = con.to_pyarrow(
        con.sql(
            """
            SELECT user_id, group_id
            FROM analytics.event_log
            WHERE experiment_id = 'new_onboarding_v2'
              AND user_id IN (
                  SELECT user_id
                  FROM analytics.event_log
                  WHERE experiment_id = 'new_onboarding_v2'
                  GROUP BY user_id
                  HAVING COUNT(DISTINCT group_id) > 1
              )
            GROUP BY user_id, group_id
            ORDER BY COUNT(*), user_id, group_id
            LIMIT 1
            """
        )
    ).to_pylist()
    assert rows
    unit = str(rows[0]["user_id"]).replace("'", "''")
    group = str(rows[0]["group_id"]).replace("'", "''")
    con.raw_sql(
        "DELETE FROM analytics.event_log "
        f"WHERE experiment_id = 'new_onboarding_v2' AND user_id = '{unit}' "
        f"AND group_id = '{group}'"
    )


def _mixed_assignment_connection():
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200)
    _extend_mixed_assignment_event_horizon(con)
    _insert_mixed_assignment(con)
    return con


@pytest.mark.slow
def test_analysis_native_mixed_assignment_errors_before_analysis(seeded_defs):
    con = _mixed_assignment_connection()

    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, con) as analysis:
        with pytest.raises(InvalidRequestError) as raised:
            analysis.run()
    assert raised.value.code == "query.integrity.mixed_assignment_units"


@pytest.mark.slow
@pytest.mark.parametrize("policy", ["warn", "exclude"])
def test_analysis_native_null_assignment_is_accounted_separately(seeded_defs, policy):
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200)
    _extend_mixed_assignment_event_horizon(con)
    _insert_null_assignment(con)

    with Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        con,
        on_mixed_assignment=policy,
    ) as analysis:
        if policy == "warn":
            with pytest.warns(IncrementWarning) as rec:
                with _ignore_ibis_deprecation():
                    analysis.run()
            assert "query.integrity.mixed_assignments_excluded" in warning_codes(rec)
        else:
            with _ignore_ibis_deprecation():
                analysis.run()
        result = analysis.srm(expected={"control": 0.5, "treatment": 0.5})

    assert isinstance(result, SRMResult)
    assert result.unassigned_units == 1
    assert result.mixed_assignment_units == 0
    assert "(unassigned)" not in result.observed


@pytest.mark.slow
def test_analysis_native_null_assignment_errors_before_analysis(seeded_defs):
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200)
    _insert_null_assignment(con)

    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, con) as analysis:
        with pytest.raises(InvalidRequestError) as raised:
            analysis.run()
    assert raised.value.code == "query.integrity.unassigned_assignment_units"


@pytest.mark.slow
def test_analysis_revalidates_mixed_assignments_before_each_readout(seeded_defs):
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200)
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, con) as analysis:
        result = analysis.srm(expected={"control": 0.5, "treatment": 0.5})
        assert isinstance(result, SRMResult)
        assert result.mixed_assignment_units == 0

        _insert_mixed_assignment(con)

        with pytest.raises(InvalidRequestError) as raised:
            analysis.run()
    assert raised.value.code == "query.integrity.mixed_assignment_units"


@pytest.mark.slow
@pytest.mark.parametrize("policy", ["warn", "exclude"])
def test_analysis_rebuilds_materialized_exposures_when_assignments_change(seeded_defs, policy):
    """A live mixed assignment cannot leave materialized stats stale."""
    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200)
    _extend_mixed_assignment_event_horizon(con)
    with Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        con,
        store="always",
        on_mixed_assignment=policy,
    ) as analysis:
        with _ignore_ibis_deprecation():
            analysis.run()

        _insert_mixed_assignment(con)

        if policy == "warn":
            with pytest.warns(UserWarning):
                with _ignore_ibis_deprecation():
                    rows = analysis.run()
        else:
            with _ignore_ibis_deprecation():
                rows = analysis.run()
        counts = analysis.srm(expected={"control": 0.5, "treatment": 0.5})
        assert isinstance(counts, SRMResult)
        assert counts.mixed_assignment_units == 1
        assert sum(counts.observed.values()) == 199
        current = {
            (row.metric, row.group_id, row.method): row.require_lift().value
            for row in _lift_rows(rows)
        }
        with Analysis.from_definitions(
            "new_onboarding_v2", seeded_defs, con, store="none", on_mixed_assignment="exclude"
        ) as fresh:
            expected = {
                (row.metric, row.group_id, row.method): row.require_lift().value
                for row in _lift_rows(fresh.run())
            }
        assert current == pytest.approx(expected)


@pytest.mark.slow
def test_analysis_rebuilds_materialized_exposures_without_fetching_mixed_identities(
    seeded_defs, monkeypatch
):
    """Same-count contamination changes stay safe without identity transport."""
    con = _mixed_assignment_connection()

    def reject_identity_batches(*args, **kwargs):
        raise AssertionError("mixed-assignment validation must not fetch identity batches")

    monkeypatch.setattr(con, "to_pyarrow_batches", reject_identity_batches)

    with Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        con,
        store="always",
        on_mixed_assignment="exclude",
    ) as analysis:
        with _ignore_ibis_deprecation():
            before = {
                (row.metric, row.group_id, row.method): row.require_lift().value
                for row in _lift_rows(analysis.run())
            }

        _replace_mixed_assignment(con)
        with _ignore_ibis_deprecation():
            current = {
                (row.metric, row.group_id, row.method): row.require_lift().value
                for row in _lift_rows(analysis.run())
            }
        repaired = analysis.srm(expected={"control": 0.5, "treatment": 0.5})
        assert isinstance(repaired, SRMResult)
        assert repaired.mixed_assignment_units == 1

        with Analysis.from_definitions(
            "new_onboarding_v2", seeded_defs, con, store="none", on_mixed_assignment="exclude"
        ) as fresh:
            expected = {
                (row.metric, row.group_id, row.method): row.require_lift().value
                for row in _lift_rows(fresh.run())
            }
        assert current == pytest.approx(expected)
        assert any(current[key] != before[key] for key in current)


@pytest.mark.slow
def test_analysis_rebuilds_materialized_exposures_when_contamination_is_repaired(
    seeded_defs,
):
    con = _mixed_assignment_connection()

    with Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        con,
        store="always",
        on_mixed_assignment="exclude",
    ) as analysis:
        with _ignore_ibis_deprecation():
            analysis.run()
        contaminated = analysis.srm(expected={"control": 0.5, "treatment": 0.5})
        assert isinstance(contaminated, SRMResult)
        assert sum(contaminated.observed.values()) == 199

        _remove_second_assignment(con)
        with _ignore_ibis_deprecation():
            analysis.run()
        repaired = analysis.srm(expected={"control": 0.5, "treatment": 0.5})
        assert isinstance(repaired, SRMResult)
        assert repaired.mixed_assignment_units == 0
        assert sum(repaired.observed.values()) == 200
        current = {
            (row.metric, row.group_id, row.method): row for row in _lift_rows(analysis.run())
        }
        with Analysis.from_definitions(
            "new_onboarding_v2",
            "examples/definitions",
            con,
            store="always",
            on_mixed_assignment="exclude",
        ) as fresh:
            expected = {
                (row.metric, row.group_id, row.method): row for row in _lift_rows(fresh.run())
            }
        assert set(current) == set(expected)
        for key, row in expected.items():
            assert current[key].require_lift().value == pytest.approx(row.require_lift().value)


@pytest.mark.slow
def test_analysis_native_mixed_assignment_warns_once_and_remains_accounted(seeded_defs):
    con = _mixed_assignment_connection()

    with Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        con,
        on_mixed_assignment="warn",
    ) as analysis:
        with pytest.warns(IncrementWarning) as rec:
            with _ignore_ibis_deprecation():
                analysis.run()
        assert "query.integrity.mixed_assignments_excluded" in warning_codes(rec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analysis.srm(expected={"control": 0.5, "treatment": 0.5})

    assert "query.integrity.mixed_assignments_excluded" not in warning_codes(caught)
    assert isinstance(result, SRMResult)
    assert result.mixed_assignment_units == 1
    assert sum(result.observed.values()) == 199
    assert "(mixed assignment)" not in result.observed


@pytest.mark.slow
def test_analysis_native_mixed_assignment_can_be_explicitly_excluded(seeded_defs):
    con = _mixed_assignment_connection()

    with Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        con,
        on_mixed_assignment="exclude",
    ) as analysis:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis.run()
            result = analysis.srm(expected={"control": 0.5, "treatment": 0.5})

    assert "query.integrity.mixed_assignments_excluded" not in warning_codes(caught)
    assert isinstance(result, SRMResult)
    assert result.mixed_assignment_units == 1


@pytest.mark.parametrize(
    ("use_examples", "available"),
    [
        (False, ()),
        (
            True,
            ("aov_decomposition", "new_onboarding_v2", "pricing_tier_test", "session_checkout"),
        ),
    ],
)
def test_analysis_unknown_experiment_is_coded_before_warehouse_access(
    tmp_path, use_examples, available
):
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("{}\n")
    path = "examples/definitions" if use_examples else definitions
    con = ibis.duckdb.connect()
    try:
        with pytest.raises(InvalidRequestError) as raised:
            Analysis.from_definitions("missing-experiment", path, con)
        assert raised.value.code == "facade.analysis.unknown_experiment"
        assert raised.value.context["requested"] == "missing-experiment"
        assert raised.value.context["available"] == available
    finally:
        con.disconnect()


def test_analysis_native_rejects_unknown_mixed_assignment_policy(seeded_con, seeded_defs):
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_definitions(
            "new_onboarding_v2",
            seeded_defs,
            seeded_con,
            on_mixed_assignment="invalid",  # ty: ignore[invalid-argument-type]
        )
    assert raised.value.code == "facade.analysis.invalid_mixed_assignment_policy"
    assert raised.value.context["on_mixed_assignment"] == "invalid"


def test_analysis_native_rejects_unknown_store_before_loading(seeded_con, monkeypatch):
    def fail_load(_path):
        pytest.fail("invalid store reached definitions loader")

    monkeypatch.setattr("increment.analysis.load", fail_load)
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_definitions(
            "new_onboarding_v2",
            "unused-definitions.yml",
            seeded_con,
            store="alway",  # ty: ignore[invalid-argument-type]
        )
    assert raised.value.code == "facade.analysis.invalid_store_policy"
    assert raised.value.context["store"] == "alway"


def test_from_unit_summary_day_axis_raises_capability_error():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "variant": ["treatment", "control"],
            "revenue": [25.0, 22.5],
        }
    )
    analysis = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    for method in (
        analysis.run_daily,
        analysis.run_daily_lift,
        analysis.run_asof,
        analysis.run_asof_lift,
    ):
        with pytest.raises(CapabilityError) as exc_info:
            method(dimension="country")
        assert exc_info.value.code == "facade.analysis.no_definitions"


@pytest.mark.parametrize(
    "method",
    ["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"],
)
def test_from_unit_summary_dimensioned_day_axis_raises_capability_error(method):
    import pandas as pd

    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "variant": ["treatment", "control"],
                "revenue": [25.0, 22.5],
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError) as exc_info:
        getattr(analysis, method)(dimension="country")
    assert exc_info.value.code == "facade.analysis.no_definitions"


@pytest.mark.parametrize(
    ("method", "expected_code", "named_method"),
    [
        ("panel_sql", "facade.analysis.operation", None),
        ("summary_sql", "source.frame.sql_unsupported", None),
        ("materialize", "facade.analysis.operation", None),
        ("run_breakout", "facade.analysis.operation", None),
        ("run_daily_lift", "facade.analysis.no_definitions", "run_daily_lift"),
        ("run_asof", "facade.analysis.no_definitions", "run_asof"),
        ("run_asof_lift", "facade.analysis.no_definitions", "run_asof_lift"),
        ("breakout_summaries", "facade.analysis.operation", None),
    ],
)
def test_from_unit_summary_refusals_are_coded_and_actionable(method, expected_code, named_method):
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "variant": ["treatment", "control"],
            "revenue": [25.0, 22.5],
        }
    )
    analysis = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    with pytest.raises(CapabilityError) as raised:
        getattr(analysis, method)()
    assert raised.value.code == expected_code
    if named_method is not None:
        assert raised.value.context["method"] == named_method
    analysis.close()


@pytest.fixture
def sitewide_con():
    """A private warehouse: the sitewide cases rewrite arm assignment, and the
    session-seeded `seeded_con` is shared with every other test."""
    import ibis

    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con)
    return con


def test_sitewide_unknown_metric_is_coded(seeded_defs, seeded_con):
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.sitewide("not_a_declared_metric")
    assert raised.value.code == "facade.analysis.unknown_metric"
    assert raised.value.context["metric"] == "not_a_declared_metric"
    analysis.close()


def test_sitewide_control_group_missing_is_coded(seeded_defs, sitewide_con):
    sitewide_con.raw_sql(
        "UPDATE analytics.event_log SET group_id = 'treatment' "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id = 'control'"
    )
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, sitewide_con) as analysis:
        with pytest.raises(InvalidRequestError) as raised:
            analysis.sitewide("purchase_rate")
    assert raised.value.code == "facade.analysis.sitewide_control_missing"
    assert raised.value.context["control_group"] == "control"
    assert raised.value.context["available"] == ("treatment",)


def test_sitewide_no_treatment_arm_is_coded(seeded_defs, sitewide_con):
    sitewide_con.raw_sql(
        "DELETE FROM analytics.event_log "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id = 'treatment'"
    )
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, sitewide_con) as analysis:
        with pytest.raises(InvalidRequestError) as raised:
            analysis.sitewide("purchase_rate")
    assert raised.value.code == "facade.analysis.sitewide_no_treatment_arm"
    assert raised.value.context["control_group"] == "control"


def test_sitewide_multiple_arms_require_explicit_arm_is_coded(seeded_defs, sitewide_con):
    sitewide_con.raw_sql(
        "UPDATE analytics.event_log SET group_id = 'treatment_b' "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id = 'treatment' "
        "AND user_id IN ("
        "SELECT DISTINCT user_id FROM analytics.event_log "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id = 'treatment' "
        "ORDER BY user_id LIMIT 50)"
    )
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, sitewide_con) as analysis:
        with pytest.raises(InvalidRequestError) as raised:
            analysis.sitewide("purchase_rate")
    assert raised.value.code == "facade.analysis.sitewide_arm_required"
    assert raised.value.context["arms"] == ("treatment", "treatment_b")


def test_sitewide_named_arm_not_enrolled_is_coded(seeded_defs, sitewide_con):
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, sitewide_con) as analysis:
        with pytest.raises(InvalidRequestError) as raised:
            analysis.sitewide("purchase_rate", arm="not_an_arm")
    assert raised.value.code == "facade.analysis.sitewide_arm_not_enrolled"
    assert raised.value.context["arm"] == "not_an_arm"


def test_srm_invalid_population_is_coded(seeded_defs, seeded_con):
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.srm(population="bogus")  # ty: ignore[invalid-argument-type]
    assert raised.value.code == "facade.analysis.invalid_population"
    assert raised.value.context["population"] == "bogus"
    analysis.close()


@pytest.fixture
def unit_summary_analysis():
    """A runnable `from_unit_summary` (seam-family) Analysis - same shape
    as ``test_analysis_from_unit_summary_runs``' inline DataFrame, factored
    out for tests that need to `run()` it more than once."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    return Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )


@pytest.fixture
def informative_prior_summary_frame():
    import pandas as pd

    rows = [
        ("u01", "control", 22.40, 9.10, "s01"),
        ("u02", "control", 10.00, 3.75, "s02"),
        ("u03", "control", 41.05, 28.60, "s03"),
        ("u04", "control", 14.20, 5.05, "s04"),
        ("u05", "control", 10.00, 0.00, "s05"),
        ("u06", "control", 28.75, 15.20, "s06"),
        ("u07", "treatment", 32.10, 8.90, "s07"),
        ("u08", "treatment", 15.60, 4.10, "s08"),
        ("u09", "treatment", 51.30, 30.15, "s09"),
        ("u10", "treatment", 10.00, 2.20, "s10"),
        ("u11", "treatment", 37.85, 16.40, "s11"),
        ("u12", "treatment", 19.95, 6.75, "s12"),
    ]
    return pd.DataFrame(
        rows,
        columns=["user_id", "variant", "revenue", "pre_revenue", "store_id"],
    )


def test_run_composes_cuped_with_prior(informative_prior_summary_frame):
    from increment.estimation import Normal

    analysis = Analysis.from_unit_summary(
        informative_prior_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", covariate="pre_revenue")],
    )
    cuped_method = [Method(name="cuped", variance_reduction="cuped")]

    unadjusted = _lift_rows(analysis.run())[0]
    cuped = _lift_rows(analysis.run(decision_method=cuped_method[0]))[0]
    shrunk = _lift_rows(
        analysis.run(decision_method=cuped_method[0], prior=Normal(mu=0.0, sigma=0.01))
    )[0]

    assert cuped.require_lift().value < unadjusted.require_lift().value
    assert abs(shrunk.require_lift().value) < abs(cuped.require_lift().value)


def test_run_per_metric_declared_priors_shrink_independently(informative_prior_summary_frame):
    """Two metrics with distinct declared `MetricSpec.prior`s each shrink
    toward THEIR OWN prior, not a shared one - proves `config.prior`
    reaches the per-metric dispatch, not just the call-wide scalar."""
    from increment.estimation import Normal

    frame = informative_prior_summary_frame.copy()
    frame["orders"] = frame["revenue"] / 2.0 + 1.0

    flat_analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue"), MetricSpec(name="orders")],
    )
    by_name = {r.metric: r for r in _lift_rows(flat_analysis.run())}

    bound_analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", prior=Normal(mu=0.0, sigma=0.01)),
            MetricSpec(name="orders", prior=Normal(mu=0.0, sigma=0.02)),
        ],
    )
    shrunk_by_name = {r.metric: r for r in _lift_rows(bound_analysis.run())}

    for name in ("revenue", "orders"):
        assert abs(shrunk_by_name[name].require_lift().value) < abs(
            by_name[name].require_lift().value
        ), f"{name}: declared prior did not shrink its own estimate"


def test_run_call_wide_prior_overrides_declared_per_metric_prior(informative_prior_summary_frame):
    from increment.estimation import Normal

    analysis = Analysis.from_unit_summary(
        informative_prior_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", prior=Normal(mu=0.0, sigma=100.0))],
    )
    declared = _lift_rows(analysis.run())[0]
    overridden = _lift_rows(analysis.run(prior=Normal(mu=0.0, sigma=0.01)))[0]

    assert abs(overridden.require_lift().value) < abs(declared.require_lift().value), (
        "call-wide prior= must win over the declared (near-flat) MetricSpec.prior"
    )


def test_run_preserves_cluster_prior_refusal_for_a_declared_per_metric_prior(
    informative_prior_summary_frame,
):
    """The existing cluster+informative-prior CapabilityError must also
    fire for a DECLARED per-metric prior, not just a call-wide one: a
    declared prior reaches the same dispatch site as a call-wide one,
    so this refusal must see it too."""
    from increment.estimation import Normal

    analysis = Analysis.from_unit_summary(
        informative_prior_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", prior=Normal(mu=0.0, sigma=0.01))],
        cluster="store_id",
    )
    with pytest.raises(CapabilityError) as exc_info:
        analysis.run()
    assert exc_info.value.code == "arm.adjustment.cluster_prior"


def test_run_preserves_cluster_prior_refusal(informative_prior_summary_frame):
    from increment.estimation import Normal

    analysis = Analysis.from_unit_summary(
        informative_prior_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        cluster="store_id",
    )

    with pytest.raises(CapabilityError) as exc_info:
        analysis.run(prior=Normal(mu=0.0, sigma=0.01))
    assert exc_info.value.code == "arm.adjustment.cluster_prior"


def _unit_summary_analysis_with_plan(plan):
    """Same shape as the ``unit_summary_analysis`` fixture, but with a
    declared plan baked into the source at construction - the only way
    a frame-backed ``Analysis`` can express ``inference=``/``alternative=``
    now: ``readouts.run()`` reads both off ``src.plan``, and
    ``Analysis.run()``'s own ``inference=``/``alternative=`` kwargs never
    reach it on this path (no ``execute()`` fallback exists for a
    frame-backed source, unlike the native/definitions path)."""
    import pandas as pd

    from increment.semantics.design import Randomized

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [1, 1, 0, 1],
            "exposure": [0, 1, 2, 3],
        }
    )
    design = Randomized(control_group="control")
    return Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        control=None,
        metrics={"revenue": "conversion"},
        design=design,
        plan=plan,
        exposure_date="exposure",
    )


def test_run_preserves_sequential_prior_refusal():
    """A declared prior conflicts with a declared AlwaysValid plan the
    same way it always has - inference is plan-declared now, not a
    call-time override, but the refusal survives the cutover."""
    from increment.estimation import Normal

    analysis = _unit_summary_analysis_with_plan(
        gaussian_plan([MetricSpec(name="revenue", type="conversion")], law="bernoulli")
    )
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError) as raised,
    ):
        warnings.simplefilter("ignore", UserWarning)
        analysis.run(prior=Normal(mu=0.0, sigma=0.01))
    assert getattr(raised.value, "code", None) == "sequential.route.unsupported"


def test_run_forwards_inference_label(unit_summary_analysis):
    """A declared plan's inference labels every estimate; an undeclared
    plan stays 'fixed'. ``Analysis.run(inference=...)`` no longer
    reaches ``readouts.run()`` on the frame path - the source's own
    declared plan is the only mechanism now."""
    fixed = unit_summary_analysis.run()
    assert {row.inference for row in fixed} == {"fixed"}
    seq_analysis = _unit_summary_analysis_with_plan(
        gaussian_plan([MetricSpec(name="revenue", type="conversion")], law="bernoulli")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # frame path is open-ended
        seq = seq_analysis.run()
    assert {r.inference for r in seq} == {"always_valid"}
    for row in seq:
        result = row.require_sequential_result()
        assert result.checkpoint.model.law == "bernoulli"
        assert result.checkpoint.control.n == result.checkpoint.treatment.n == 2
        assert row.require_lift().value == pytest.approx(1.0)
        assert type(row).model_validate_json(row.model_dump_json()) == row


def test_run_forwards_prior(unit_summary_analysis):
    from increment.estimation import Normal

    flat = unit_summary_analysis.run()[0]
    shrunk = unit_summary_analysis.run(prior=Normal(mu=0.0, sigma=0.01))[0]

    assert flat.require_lift().value > 0.0
    assert abs(shrunk.require_lift().value) < abs(flat.require_lift().value)


def test_run_forwards_alternative_label(unit_summary_analysis):
    """A declared plan's alternative labels every estimate and rejects
    unknown values. ``Analysis.run(alternative=...)`` no longer reaches
    ``readouts.run()`` on the frame path - the source's own declared
    plan is the only mechanism now. Uses a primary-role plan (not the
    frame path's secondary-by-default derivation) so the BH/FCR family
    machinery never re-estimates the interval at a level other than
    this metric's own alpha_share."""
    two_sided = unit_summary_analysis.run()
    assert {r.alternative for r in two_sided} == {"two-sided"}

    one_sided = _unit_summary_analysis_with_plan(
        AnalysisPlan(primary="revenue", alternative="greater")
    ).run()
    assert {r.alternative for r in one_sided} == {"greater"}
    assert all(r.require_lift().level == pytest.approx(0.90) for r in one_sided)

    with pytest.raises(ValidationError) as raised:
        AnalysisPlan(alternative="bigger")  # ty: ignore[invalid-argument-type]  - the bad literal IS the case under test
    [error] = raised.value.errors()
    assert error["loc"] == ("alternative",)
    assert error["type"] == "literal_error"


@pytest.mark.slow
def test_analysis_run_native_declared_margin_abs_reaches_the_estimate(seeded_con, seeded_defs):
    """Twin of test_run_declared_margin_abs_reaches_the_estimate for the
    NATIVE (from_definitions -> _build_for_metrics) path: a margin_abs-
    declared metric produces a LiftEstimate with null_abs stamped and a
    stat_sig decided on the additive interval - not a refusal, and not a
    silently substituted superiority-at-zero test."""
    # Public LiftEstimate.stat_sig() below covers the serialized decision.

    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    # avg_session_duration declares preferred_direction=increase in the
    # example definitions; layer the absolute guardrail on top.
    a = make_analysis_like(
        a,
        [
            m.model_copy(update={"margin_abs": 100.0}) if m.name == "avg_session_duration" else m
            for m in a.metrics
        ],
    )
    results = _lift_rows(a.run())
    rows = [r for r in results if r.metric == "avg_session_duration"]
    assert rows
    est = rows[0]
    assert est.null_abs == pytest.approx(-100.0)
    assert est.null_lift == 0.0
    assert est.alternative == "greater"
    assert est.abs_lb is not None and est.abs_ub is not None
    # The decision is the abs interval vs null_abs, nothing else.
    assert est.stat_sig() is (est.abs_lb > -100.0)
    # Other metrics stay on the relative/zero path untouched.
    other = [r for r in results if r.metric == "purchase_rate"]
    assert other and other[0].null_abs is None


def test_analysis_run_native_validates_before_materializing(seeded_con, seeded_defs, monkeypatch):
    """Static run refusals happen before any warehouse result is fetched."""
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    monkeypatch.setattr(
        seeded_con,
        "to_pyarrow",
        lambda *_args, **_kwargs: pytest.fail("run reached warehouse reduction before refusal"),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        analysis.run(estimands=("late",))
    assert exc_info.value.code == "readout.assignment.estimands"


@pytest.mark.slow
def test_analysis_run_native_exercises_primary_secondary_guardrail_roles(seeded_con, seeded_defs):
    """Definitions-backed (DuckDB) `Analysis.run()` end-to-end, exercising
    all three declared roles through the full facade in one call - proves
    `moments_source()`'s `design=`/`plan=` threading fix actually reaches
    `readouts.run()`'s role-based dispatch (primary split by n_arms,
    guardrail/secondary unsplit, discovery stamped only on the secondary)
    on the NATIVE path, not just the frame path `test_readouts_plan.py`
    already covers directly. `new_onboarding_v2` declares exactly the 3
    metrics needed (purchase_rate, avg_session_duration, d7_retention),
    with `d7_retention` already a declared guardrail carrying a prior
    binding and `preferred_direction`."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    a = make_analysis_like(
        a,
        experiment=a.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    primary="purchase_rate",
                    secondaries=["avg_session_duration"],
                    guardrails=a.experiment.plan.guardrails,
                )
            }
        ),
    )
    results = _lift_rows(a.run())
    by_metric = {r.metric: r for r in results}
    assert set(by_metric) == {"purchase_rate", "avg_session_duration", "d7_retention"}

    primary = by_metric["purchase_rate"]
    assert primary.role == "primary"
    assert primary.discovery is None
    # 1 declared primary, 1 enrolled treatment arm: alpha_share =
    # 0.05/1 primaries, split again by n_arms=1 -> unsplit 0.05.
    assert primary.require_lift().level == pytest.approx(0.95)

    secondary = by_metric["avg_session_duration"]
    assert secondary.role == "secondary"
    assert secondary.discovery in (True, False)  # a single-metric family always resolves one

    guardrail = by_metric["d7_retention"]
    assert guardrail.role == "guardrail"
    assert guardrail.discovery is None
    # d7_retention carries a prior binding but no margin/margin_abs, so it
    # tests one-sided against zero on its preferred_direction=increase
    # adverse side; alpha_share stays unsplit at the plan's nominal alpha,
    # and the one-sided alpha-doubling identity gives level=1-2*0.05=0.9.
    assert guardrail.alternative == "greater"
    assert guardrail.require_lift().level == pytest.approx(0.9)


@pytest.mark.slow
def test_observational_estimand_guardrail_row_survives_native_run(
    seeded_con, seeded_defs, monkeypatch
):
    """Analysis.run() on the native path returns every row the readout
    emits for a guardrail, whatever its estimand label: an observational
    adjustment row carries 'ate'/'plr_slope'/'overlap_subpopulation_ate',
    not 'itt', and must never be filtered out by estimand. IPTW/DML/AIPW
    cannot reach the native (warehouse) path today --
    DefinitionsMomentSource.unit_frame refuses any covariates, and every
    adjustment needs at least one -- so this drives the native run with a
    canned readouts.run() result standing in for that row shape, holding
    the invariant even if that capability boundary ever moves."""
    from increment.estimation.results import Estimate

    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    a = make_analysis_like(
        a,
        design=Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x",))),
    )

    lift = Estimate(value=0.01, lb=-0.02, ub=0.04, level=0.9)
    canned = [
        LiftEstimate(
            metric="purchase_rate",
            group_id="treatment",
            method="iptw",
            method_role="decision",
            lift=lift,
            estimand="ate",
        ),
        LiftEstimate(
            metric="d7_retention",
            group_id="treatment",
            method="iptw",
            method_role="decision",
            lift=lift,
            estimand="ate",
        ),
    ]
    monkeypatch.setattr(readout_functions, "run", lambda *args, **kwargs: canned)
    results = a.run(decision_method=Method(name="iptw"))
    metrics_seen = {r.metric for r in results}
    assert "d7_retention" in metrics_seen, (
        f"guardrail row silently dropped after the estimand relabel, got {metrics_seen}"
    )


def test_observational_relative_margin_refused_at_constructor(seeded_con, seeded_defs):
    """A source constructor refuses observational relative margins."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    observational = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("x",))
    )
    relative_guardrail = MeanMetric(
        name="revenue_relative_guardrail",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        preferred_direction="decrease",
        margin=0.01,
    )

    with pytest.raises(UnsupportedRequestError) as exc:
        make_analysis_like(a, metrics=[*a.metrics, relative_guardrail], design=observational)
    assert exc.value.code == "plan.observational.relative_margin"
    assert exc.value.context["metrics"] == ("revenue_relative_guardrail",)

    # The same refusal when the relative margin is already present and only
    # the design changes on the rebuilt source.
    a_with_margin = make_analysis_like(a, [*a.metrics, relative_guardrail])
    with pytest.raises(UnsupportedRequestError) as exc2:
        make_analysis_like(a_with_margin, design=observational)
    assert exc2.value.code == "plan.observational.relative_margin"
    assert exc2.value.context["metrics"] == ("revenue_relative_guardrail",)

    # Valid flows still reach a source with an absolute-axis margin.
    absolute_guardrail = MeanMetric(
        name="revenue_absolute_guardrail",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        preferred_direction="decrease",
        margin_abs=1.0,
    )
    abs_analysis = make_analysis_like(
        a, metrics=[*a.metrics, absolute_guardrail], design=observational
    )
    assert abs_analysis.experiment.control_group == "control"

    # ...and no margin at all under Observational.
    plain_obs_analysis = make_analysis_like(a, design=observational)
    assert plain_obs_analysis.experiment.control_group == "control"


@pytest.mark.slow
def test_analysis_run_native_routed_refuses_call_time_epistemic_policy_kwargs(
    seeded_con, seeded_defs
):
    """The same silent-ignore bug the frame/moments seam had (see
    `test_from_moments_run_refuses_call_time_epistemic_policy_kwargs`)
    also existed on the NATIVE routed path (a Definitions-backed
    experiment with a Randomized, non-quantile, non-clustered design --
    `new_onboarding_v2`'s shape, and the most common case): `run()`'s
    `readouts.run()` call sites here dropped alpha=/alternative=/
    inference=/margins=/null_lifts=/margins_abs= entirely (no forwarding
    kwarg at all), so e.g. `run(alpha=0.30)` silently kept `level=0.9`
    and `run(margins={...})` silently left `null_lift=0.0`. Each must
    now raise TypeError naming itself instead of no-oping."""
    from increment import AlwaysValid

    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    baseline = _lift_rows(a.run())
    # The declared plan selects purchase_rate as a lone in-family secondary: BH's
    # cutoff is q=0.10 and the cap holds the level at 0.95. This stable baseline
    # shows the raises below refuse alpha=0.30 rather than silently ignore it.
    assert baseline[0].require_lift().level == pytest.approx(0.95)

    with pytest.raises(TypeError):
        a.run(alpha=0.30)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(alternative="greater")  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(inference=AlwaysValid(registration=registration()))  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(margins={"purchase_rate": 0.02})  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(null_lifts={"purchase_rate": 0.02})  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(margins_abs={"purchase_rate": 0.02})  # ty: ignore[unknown-argument]


@pytest.mark.parametrize("param", ["margins", "null_lifts", "margins_abs"])
def test_analysis_run_native_unknown_margin_key_refused(seeded_con, seeded_defs, param):
    """A margins=/null_lifts=/margins_abs= value on new_onboarding_v2's
    plan-driven routed path (Randomized, non-quantile, non-clustered) is
    refused outright, before ever reaching a per-key unknown-metric
    check -- a typo'd key never even gets there; the refusal is broader,
    not narrower, than "never silently dropped"."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    with pytest.raises(TypeError):
        a.run(**{param: {"purchase_rte": 0.01}})  # ty: ignore[invalid-argument-type]


def test_analysis_run_native_call_time_margin_zero_refused(seeded_con, seeded_defs):
    """margins= (any value, including 0.0) is refused outright on the
    plan-driven routed path before it could silently flip a two-sided
    read one-sided at the zero null -- refused louder than the old
    per-value check, not by it."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    with pytest.raises(TypeError):
        a.run(margins={"purchase_rate": 0.0})  # ty: ignore[unknown-argument]


@pytest.mark.slow
def test_analysis_run_native_filters_by_name(seeded_con, seeded_defs):
    """run(metrics=["purchase_rate"]) returns only that metric's rows on
    the definitions-backed (native) path - sm8m completion."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    results = a.run(metrics=["purchase_rate"])
    assert results
    assert {r.metric for r in results} == {"purchase_rate"}


@pytest.mark.slow
def test_analysis_run_native_binding_drives_cuped_dispatch(seeded_pre_period_con, seeded_defs):
    """A declared ``ExperimentMetric`` method-role binding drives
    ``run()``'s per-metric dispatch on the definitions-backed path - a sibling
    metric with no binding keeps the design default (unadjusted only).
    Needs real pre-period activity for a non-degenerate CUPED covariate,
    hence the seeded (not synthetic 4-unit) population."""
    from increment.semantics.models import ExperimentMetric, MethodSpec

    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    a = make_analysis_like(
        a,
        experiment=a.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        "purchase_rate",
                        ExperimentMetric(
                            metric="avg_session_duration",
                            decision_method=MethodSpec(name="unadjusted"),
                            sensitivity_methods=(
                                MethodSpec(name="cuped", variance_reduction="cuped"),
                            ),
                        ),
                    ],
                    guardrails=a.experiment.plan.guardrails,
                )
            }
        ),
    )
    results = a.run()
    by_metric_method = {(r.metric, r.method) for r in results}
    assert ("avg_session_duration", "cuped") in by_metric_method
    assert ("avg_session_duration", "unadjusted") in by_metric_method
    assert ("purchase_rate", "unadjusted") in by_metric_method
    assert ("purchase_rate", "cuped") not in by_metric_method


@pytest.mark.slow
def test_analysis_run_native_skips_pre_events_without_a_cuped_request(
    seeded_con, seeded_defs, monkeypatch
):
    """A native run without a CUPED request does not issue pre-period queries."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    assert a.experiment.n_pre_periods == 14
    queries: list[str] = []
    to_pyarrow = seeded_con.to_pyarrow

    def record_query(expr, *args, **kwargs):
        queries.append(ibis.to_sql(expr))
        return to_pyarrow(expr, *args, **kwargs)

    monkeypatch.setattr(seeded_con, "to_pyarrow", record_query)
    results = a.run()
    assert results
    # 9 queries at 192c91c (was 6): a duplicate-group-membership
    # fingerprint/dedup pass now runs an extra 3 times across exposure
    # resolution. None of the 9 filter on ts < (verified below), so this
    # is a stale count, not a pre-period leak.
    assert len(queries) == 9
    assert not any(
        "ts <" in query.lower().replace('"ts"', "ts").replace("`ts`", "ts") for query in queries
    )


def test_analysis_run_native_unknown_name_raises(seeded_con, seeded_defs):
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    with pytest.raises(InvalidRequestError) as exc_info:
        a.run(metrics=["typo"])
    assert exc_info.value.code == "facade.analysis_config.unknown_metric_declared"
    assert exc_info.value.context["unknown_names"] == ("typo",)


def test_analysis_run_native_duplicate_names_raise(seeded_con, seeded_defs):
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    with pytest.raises(InvalidRequestError) as exc_info:
        a.run(metrics=["purchase_rate", "purchase_rate"])
    assert exc_info.value.code == "facade.analysis_config.duplicate_metric_name"
    assert exc_info.value.context["duplicate_names"] == ("purchase_rate",)


@pytest.mark.slow
def test_analysis_run_daily_string_and_object_forms_agree(seeded_con, seeded_defs):
    """run_daily(metrics=["purchase_rate"]) (string form) selects the
    same effective metric set as run_daily(metrics=[<the Metric
    object>]) - both backends now share one name-resolution seam.
    Compares structurally (metric/ds/group_id keys), not exact floating
    values: two independent DuckDB aggregate executions can differ at
    the ULP level under parallel execution, which is orthogonal to what
    this test verifies."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    metric = next(m for m in a.metrics if m.name == "purchase_rate")
    by_name = a.run_daily(metrics=["purchase_rate"])
    by_object = a.run_daily(metrics=[metric])

    def keys(values):
        return {(row.metric, row.ds, row.group_id) for row in values}

    assert by_name
    assert keys(by_name) == keys(by_object)
    assert {row.metric for row in by_name} == {"purchase_rate"}


def test_analysis_run_native_filtered_margins_key_raises_before_reduction(seeded_con, seeded_defs):
    """A margins= value on new_onboarding_v2's plan-driven routed path is
    refused at the call boundary -- the refusal fires on `margins=` being
    set at all, not on this particular key naming a declared-but-FILTERED
    metric -- so a filtered-metric key can never reach a per-key
    unknown-metric check or a query reduction."""
    a = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)

    with pytest.raises(TypeError):
        a.run(metrics=["purchase_rate"], margins={"avg_session_duration": 0.01})  # ty: ignore[unknown-argument]


def test_analysis_from_unit_summary_encouragement_uptake():
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = [
        ("u01", "control", 10.0, 0),
        ("u02", "control", 12.0, 0),
        ("u03", "control", 9.0, 0),
        ("u04", "control", 20.0, 1),
        ("u05", "treatment", 25.0, 1),
        ("u06", "treatment", 30.0, 1),
        ("u07", "treatment", 28.0, 1),
        ("u08", "treatment", 15.0, 0),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "variant", "revenue", "clicked"])

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        # Low z-floor: keep LATE reported on this 4-units-per-arm fixture
        # instead of exercising weak-instrument suppression here.
        min_first_stage_z=0.5,
    )
    a = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=design,
        uptake="clicked",
    )
    results = a.run()
    assert {r.estimand for r in results} == {"itt", "compliance", "late"}


@pytest.mark.parametrize("estimands", [("late",), ("late", "compliance")])
@pytest.mark.slow
def test_analysis_run_guardrail_experiment_weak_instrument_narrowed_estimands_keeps_diagnostic(
    estimands,
):
    """Keep outcome-specific LATE diagnostics and requested design-wide compliance."""
    con = ibis.duckdb.connect()
    control_units = [f"c{i}" for i in range(1, 21)]
    treat_units = [f"t{i}" for i in range(1, 21)]
    # Weak first stage: only 1 of 20 treated units ever clicks.
    clickers = {"t1"}

    exposure_rows = [
        {
            "user_id": u,
            "ts": datetime(2025, 1, 1, 9, 0, 0),
            "event": "exposed",
            "group_id": "control",
            "revenue": None,
            "errors": None,
            "experiment_id": "weak_guardrail_exp",
        }
        for u in control_units
    ] + [
        {
            "user_id": u,
            "ts": datetime(2025, 1, 1, 9, 0, 0),
            "event": "exposed",
            "group_id": "treatment",
            "revenue": None,
            "errors": None,
            "experiment_id": "weak_guardrail_exp",
        }
        for u in treat_units
    ]
    click_rows = [
        {
            "user_id": u,
            "ts": datetime(2025, 1, 2, 9, 0, 0),
            "event": "clicked",
            "group_id": None,
            "revenue": None,
            "errors": None,
            "experiment_id": None,
        }
        for u in clickers
    ]
    purchase_rows = [
        {
            "user_id": u,
            "ts": datetime(2025, 1, 3, 9, 0, 0),
            "event": "purchase",
            "group_id": None,
            "revenue": 8.0 + (i % 5),  # varying amounts - real ddof=1 variance
            "errors": None,
            "experiment_id": None,
        }
        for i, u in enumerate(control_units + treat_units)
    ]
    error_rows = [
        {
            "user_id": u,
            "ts": datetime(2025, 1, 3, 9, 0, 0),
            "event": "error_event",
            "group_id": None,
            "revenue": None,
            "errors": float(1 + (i % 3)),  # varying counts - real ddof=1 variance
            "experiment_id": None,
        }
        for i, u in enumerate(control_units + treat_units)
    ]
    con.create_table(
        "weak_guardrail_events",
        obj=exposure_rows + click_rows + purchase_rows + error_rows,
    )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM weak_guardrail_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "error_event", "column": "errors"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": "errors",
                    "entity": "user_id",
                    "fact": "error_event",
                    "aggregation": "sum",
                    "preferred_direction": "decrease",
                },
            ],
            "experiments": [
                {
                    "name": "weak_guardrail_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"], "guardrails": ["errors"]},
                }
            ],
        }
    )

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    analysis = make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves revenue via uptake"
            ),
            one_sided=True,
            min_first_stage_z=4.0,  # default floor - 1/20 compliance must trip it
        ),
    )

    results = analysis.run(estimands=estimands)

    by_metric: defaultdict[str, set[str]] = defaultdict(set)
    for row in _lift_rows(results):
        by_metric[row.metric].add(row.estimand)

    # A guardrail's suppressed LATE leaves the same outcome-specific
    # diagnostic as any other metric's -- never an unrequested itt row.
    expected = {"errors": {"compliance"}, "revenue": {"compliance"}}
    if "compliance" in estimands:
        expected["uptake"] = {"compliance"}
    assert dict(by_metric) == expected
    for row in _lift_rows(results):
        if row.estimand == "compliance":
            assert row.lift is not None
            assert row.lift.value == pytest.approx(1 / 20)


def test_analysis_from_unit_panel_encouragement_staggered_entry():
    """Analysis.from_unit_panel + Encouragement + a windowed uptake, with
    staggered enrollment - doubles as a regression guard at the public
    API layer for the densified-panel anchoring bug (window must anchor
    to each unit's OWN first observed day, never the panel's global
    earliest date; see increment/frame.py's
    test_from_unit_panel_uptake_window_anchors_to_each_units_own_entry
    for the frame-layer version of this same scenario).

    u1 enters day 0 (rows for days 0-5), u2 enters day 3 (rows only from
    day 3 - days 0-2 exist only after densification, zero-filled).
    window_days=3. u1 clicks day 1 (elapsed since ITS OWN entry = 1,
    inside [0,3)); u2 clicks day 4 (elapsed since ITS OWN entry = 1,
    inside [3,6)). Both must count - a global-anchor bug would evaluate
    u2's day-4 click as elapsed=4 from day 0, outside [0,3), and drop it.
    """
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = []
    for day in range(6):
        rows.append(("u1", "treatment", day, 1.0, 1.0 if day == 1 else 0.0))
    for day in range(3, 6):
        rows.append(("u2", "treatment", day, 1.0, 1.0 if day == 4 else 0.0))
    for day in range(6):
        # Control revenues must differ across units: zero-variance arms are refused by the degenerate-arm guard.
        rows.append(("u3", "control", day, 1.0, 0.0))
        rows.append(("u4", "control", day, 2.0, 0.0))
    df = pd.DataFrame(rows, columns=["user_id", "variant", "day", "revenue", "clicked"])

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        min_first_stage_z=0.001,
    )
    a = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=design,
        uptake="clicked",
    )
    results = _lift_rows(a.run())
    compliance = next(r for r in results if r.estimand == "compliance")
    assert compliance.require_lift().value == pytest.approx(
        1.0
    )  # both u1 and u2 counted (2/2 treated units)


def test_from_unit_panel_run_forwards_prior():
    import pandas as pd

    from increment.estimation import Normal

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "variant": ["treatment"] * 4 + ["control"] * 4,
            "day": ["2026-01-01", "2026-01-02"] * 4,
            "revenue": [10.0, 20.0, 20.0, 20.0, 8.0, 10.0, 10.0, 14.0],
        }
    )
    analysis = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )

    flat = _lift_rows(analysis.run())[0]
    shrunk = _lift_rows(analysis.run(prior=Normal(mu=0.0, sigma=0.01)))[0]

    assert flat.require_lift().value > 0.0
    assert abs(shrunk.require_lift().value) < abs(flat.require_lift().value)


@pytest.fixture(scope="session")
def published_artifact():
    """Share published input only across read-only adoption tests."""
    published = _published()
    yield published
    published[1].close()
    published[0].disconnect()


@pytest.mark.slow
def test_public_artifact_adoption_constructs_and_reopens(published_artifact):
    _con, native, context, store, ref = published_artifact
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    assert adopted.run(metrics=["purchase_rate"])
    adopted.close()
    assert adopted.run(metrics=["purchase_rate"])


@pytest.mark.slow
def test_public_artifact_adoption_has_one_build_and_no_upstream_work(monkeypatch):
    from ibis.backends.duckdb import Backend

    from increment.query.session import WarehouseArtifactStore

    publications: list[object] = []
    reads: list[str] = []
    original_begin = WarehouseArtifactStore.begin_publication
    original_to_pyarrow = Backend.to_pyarrow

    def record_publication(self, *args, **kwargs):
        publications.append((args, kwargs))
        return original_begin(self, *args, **kwargs)

    def record_to_pyarrow(self, expr, *args, **kwargs):
        reads.append(str(expr))
        return original_to_pyarrow(self, expr, *args, **kwargs)

    monkeypatch.setattr(WarehouseArtifactStore, "begin_publication", record_publication)
    monkeypatch.setattr(Backend, "to_pyarrow", record_to_pyarrow)
    _con, _native_analysis, context, store, ref = _published()
    assert len(publications) == 1
    reads.clear()
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    assert adopted.run(metrics=["purchase_rate"])
    assert len(publications) == 1
    assert not any("event_log" in query.lower() for query in reads)


def test_public_artifact_adoption_validates_fixed_reference(published_artifact):
    _con, _native_analysis, context, store, ref = published_artifact
    bad_ref = ref.model_copy(update={"generation_id": uuid4()})
    with pytest.raises(ArtifactContractError) as raised:
        Analysis.from_unit_day_artifact(store, bad_ref, expected_context=context)
    assert raised.value.code in {"artifact.refresh.invalid_ref", "artifact.snapshot.mixed"}


def test_public_artifact_adoption_refuses_malformed_base_locator(published_artifact):
    _con, _native_analysis, context, store, ref = published_artifact
    bad_ref = ref.model_copy(
        update={"manifest": ref.manifest.model_copy(update={"name": "bad\nlocator"})}
    )
    with pytest.raises(ArtifactContractError) as raised:
        Analysis.from_unit_day_artifact(store, bad_ref, expected_context=context)
    assert raised.value.code == "artifact.identifier.unsafe"


def test_public_artifact_adoption_refuses_zero_io_unsafe_locator(monkeypatch, published_artifact):
    _con, _native_analysis, context, store, ref = published_artifact
    bad_ref = ref.model_copy(
        update={"manifest": ref.manifest.model_copy(update={"name": "bad\nlocator"})}
    )
    calls: list[str] = []
    validate = store.validate_locator
    open_snapshot = store.open_snapshot

    def record_validate(*args, **kwargs):
        calls.append("validate")
        return validate(*args, **kwargs)

    def record_open(*args, **kwargs):
        calls.append("open")
        return open_snapshot(*args, **kwargs)

    monkeypatch.setattr(store, "validate_locator", record_validate)
    monkeypatch.setattr(store, "open_snapshot", record_open)
    with pytest.raises(ArtifactContractError):
        Analysis.from_unit_day_artifact(store, bad_ref, expected_context=context)
    assert calls == []


@pytest.mark.slow
def test_public_artifact_adoption_handles_snapshot_generation_race():
    _con, native, context, store, first = _published()
    second = native.publish_unit_day_artifact(store, refresh_of=first)
    raced = first.model_copy(update={"generation_id": second.generation_id})
    with pytest.raises(ArtifactContractError) as raised:
        Analysis.from_unit_day_artifact(store, raced, expected_context=context)
    assert raised.value.code in {"artifact.refresh.invalid_ref", "artifact.snapshot.mixed"}


@pytest.mark.slow
def test_public_artifact_adoption_matches_native_results(published_artifact):
    _con, native, context, store, ref = published_artifact
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    native_rows = _lift_rows(native.run(metrics=["purchase_rate"]))
    adopted_rows = _lift_rows(adopted.run(metrics=["purchase_rate"]))
    assert [(row.metric, row.group_id) for row in adopted_rows] == [
        (row.metric, row.group_id) for row in native_rows
    ]
    assert [row.require_lift().value for row in adopted_rows] == pytest.approx(
        [row.require_lift().value for row in native_rows]
    )


def test_public_artifact_adoption_refuses_missing_extension(published_artifact):
    _con, native, context, store, ref = published_artifact
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    with pytest.raises(ArtifactContractError) as raised:
        adopted.run_breakout(metrics=["purchase_rate"])
    assert raised.value.code == "artifact.extension.missing"


@pytest.mark.slow
def test_public_artifact_adoption_rejects_malformed_extension_relation():
    _con, native, context, store, _ref = _published()
    request = next(iter(_extensions(context, "breakout_dimension")))
    malformed = request.model_copy(update={"property_name": "missing_property"})
    with pytest.raises(ArtifactContractError) as raised:
        native.publish_unit_day_artifact(store, extensions=[malformed])
    assert raised.value.code == "artifact.extension.missing"


@pytest.mark.slow
def test_value_scale_refusal_carries_the_same_code_on_every_family() -> None:
    """value_scale= on a randomized analysis is one refusal with one stable
    code, whichever constructor built the analysis."""
    import ibis

    from examples._seed import seed_event_log
    from increment.analysis import Analysis
    from increment.errors import CodedError

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200)
    analysis = Analysis.from_definitions(
        "new_onboarding_v2", "examples/definitions", con, store="none"
    )
    with pytest.raises(CodedError) as refusal:
        analysis.run(value_scale={"purchase_rate": "absolute"})
    assert refusal.value.code == "readout.randomized.value_scale"


def _unit_summary_frame(tenure, group, revenue):
    import pyarrow as pa

    return pa.table(
        {
            "user_id": list(tenure),
            "variant": [group[u] for u in tenure],
            "revenue": [revenue[u] for u in tenure],
            "tenure": list(tenure.values()),
        }
    )


def test_estimate_cate_matches_across_definitions_and_unit_summary(tmp_path):
    """A randomized experiment needs no declared design: estimate_cate reads
    the ad hoc covariate from the warehouse exactly as from a dataframe."""
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, group, revenue = covariate_defs_and_con(tmp_path)
    native = Analysis.from_definitions("cov_test", defs_path, con)
    oracle = Analysis.from_unit_summary(
        _unit_summary_frame(tenure, group, revenue),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        control="control",
    )
    results = [
        analysis.estimate_cate("revenue", control="control", interact=["tenure"])
        for analysis in (native, oracle)
    ]
    native_result, oracle_result = results
    assert native_result.n == oracle_result.n == len(tenure)
    for field in ("ate", "se", "lb", "ub"):
        assert getattr(native_result, field) == pytest.approx(
            getattr(oracle_result, field), rel=1e-9
        )
    assert native_result.beta == pytest.approx(oracle_result.beta, rel=1e-9)


def test_estimate_cate_matches_across_unit_panel_and_unit_summary(tmp_path):
    """CATE/HTE's shared unit_frame read is live on from_unit_panel now that
    the panel source serves unwindowed metrics and constant-within-unit
    covariates -- exactly the metric types CATE already supports on every
    other source."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(6)
    n = 40
    tenure = {f"u{i}": float(rng.normal(100, 15)) for i in range(n)}
    group = {f"u{i}": ("control" if i % 2 == 0 else "treatment") for i in range(n)}
    revenue = {f"u{i}": 5.0 + 0.1 * i for i in range(n)}
    frame = pd.DataFrame(
        {
            "user_id": list(tenure),
            "variant": [group[u] for u in tenure],
            "revenue": [revenue[u] for u in tenure],
            "tenure": list(tenure.values()),
            "date": ["2025-01-01"] * n,
        }
    )
    panel = Analysis.from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        control="control",
    )
    panel_result = panel.estimate_cate("revenue", control="control", interact=["tenure"])

    summary = Analysis.from_unit_summary(
        frame.drop(columns=["date"]),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        control="control",
    )
    summary_result = summary.estimate_cate("revenue", control="control", interact=["tenure"])

    assert panel_result.n == summary_result.n == n
    for field in ("ate", "se", "lb", "ub"):
        assert getattr(panel_result, field) == pytest.approx(
            getattr(summary_result, field), rel=1e-9
        )
    assert panel_result.beta == pytest.approx(summary_result.beta, rel=1e-9)


def test_observational_ate_matches_across_definitions_artifact_and_unit_summary(tmp_path):
    """Confounded assignment adjusted on a declared pre-exposure covariate: the
    warehouse and artifact routes return the dataframe route's rows exactly."""
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics import load
    from increment.semantics.artifact import UnitCovariateRequest
    from tests.analysis_factory import lift_rows
    from tests.covariate_cases import confounded_defs_and_con

    defs_path, con, tenure, group, revenue = confounded_defs_and_con(tmp_path)
    native = Analysis.from_definitions("obs_test", defs_path, con)
    defs = load(defs_path)
    experiment = defs.experiment("obs_test")
    assert experiment is not None
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = native.publish_unit_day_artifact(
        store, extensions=[UnitCovariateRequest(property_name="tenure", source_name="events")]
    )
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    oracle = Analysis.from_unit_summary(
        _unit_summary_frame(tenure, group, revenue),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
    )
    expected = {r.group_id: r for r in lift_rows(oracle.run())}
    assert expected
    for analysis in (native, adopted):
        rows = {r.group_id: r for r in lift_rows(analysis.run())}
        assert set(rows) == set(expected)
        for group_id, row in rows.items():
            oracle_row = expected[group_id]
            assert row.method == oracle_row.method
            for field in ("value", "lb", "ub"):
                assert getattr(row.lift, field) == pytest.approx(
                    getattr(oracle_row.lift, field), rel=1e-9
                )


def test_only_a_declared_observational_design_routes_readouts_through_the_unit_frame(tmp_path):
    """A randomized from_definitions readout keeps reading the moments cube;
    an observational design cannot be declared without covariates at all."""
    from tests.analysis_factory import lift_rows
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=False)
    randomized = Analysis.from_definitions("cov_test", defs_path, con)
    randomized_methods = {row.method for row in lift_rows(randomized.run())}
    assert randomized_methods == {"unadjusted"}

    defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=True)
    observational = Analysis.from_definitions("cov_test", defs_path, con)
    # The observational readout adjusts on the covariate through the unit
    # frame; this fixture's deterministic tenure separates the arms perfectly,
    # so that route's overlap gate refuses instead of returning unadjusted rows.
    with pytest.raises(IdentificationError):
        observational.run()

    defs_path.write_text(
        defs_path.read_text().replace(
            "      covariates:\n        - {property: tenure, source: events}\n",
            "      covariates: []\n",
        )
    )
    from increment.errors import DefinitionError
    from increment.semantics import load

    with pytest.raises(DefinitionError) as raised:
        load(defs_path)
    assert raised.value.code == "definition.validation"
