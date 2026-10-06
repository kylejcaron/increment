"""End-to-end regression test over the realistic_demo normalized warehouse.

The shipped `examples/realistic_demo/definitions/*.yaml` bind to Parquet via
hardcoded relative paths (`read_parquet('examples/realistic_demo/warehouse/
<table>/**/*.parquet')`), not via views bound on a connection object - so
this module regenerates a small REAL warehouse at that exact relative path
and drives the actual shipped YAML against it through the public `Analysis`
facade, rather than a synthetic in-memory substitute (which would test
something other than what ships).

Restores the demo's normal (2 partitions x 2000 users) warehouse as the very
last thing in the module, so a full test run doesn't leave a tiny warehouse
behind for anyone running the example manually afterward.
"""

from __future__ import annotations

import math
import subprocess
import sys
import warnings
from collections import Counter
from datetime import date
from pathlib import Path
from typing import cast

import ibis
import pyarrow.parquet as pq
import pytest

from increment import Analysis, MetricSpec
from increment.errors import IncrementWarning
from increment.estimation.results import LiftEstimate
from increment.results import NotApplicable, SRMResult
from increment.semantics.models import RatioMetric, RetentionMetric
from tests.analysis_factory import make_analysis_like
from tests.warning_codes import warning_codes


def _lift_rows(rows: object) -> list[LiftEstimate]:
    return cast(list[LiftEstimate], rows)


# Slow: regenerating the small warehouse and restoring the normal one runs the
# generator twice in subprocesses. The xdist group keeps this module and the
# notebook smoke tests on one worker because both rewrite
# `examples/realistic_demo/warehouse` in place.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.xdist_group("realistic_demo_warehouse"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFINITIONS = REPO_ROOT / "examples" / "realistic_demo" / "definitions"
EXPERIMENT = "checkout_redesign"
METRIC_NAMES = {
    "conversion_rate",
    "revenue_per_user",
    "average_order_value",
    "d7_retention",
    "checkout_latency_ms",
}


def _generate(*, partitions: int, users_per_partition: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "examples/realistic_demo/generate.py",
            "--partitions",
            str(partitions),
            "--users-per-partition",
            str(users_per_partition),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


@pytest.fixture(scope="module", autouse=True)
def small_warehouse():
    """1 partition x 300 users is enough for every metric/breakout to have
    non-degenerate data (generate.py's own post-write checks already guard
    against saturation/small-N edge cases) and runs in well under a second.
    Restores the demo's normal 2-partitions x 2000-users state on teardown.
    """
    _generate(partitions=1, users_per_partition=300)
    yield
    _generate(partitions=2, users_per_partition=2000)


@pytest.fixture(scope="module")
def con():
    # No view registration needed - the shipped fact-source SQL reads
    # Parquet directly, so a bare in-memory DuckDB connection suffices as long as cwd is the repo root (pytest's default here).
    return ibis.duckdb.connect()


@pytest.fixture
def analysis(con):
    return Analysis(EXPERIMENT, DEFINITIONS, con)


def _metric_spec(metric) -> MetricSpec:
    """Translate a semantic ``Metric`` into the ``MetricSpec`` dict
    ``Analysis.from_moments`` needs (it takes ``increment.frame``
    ``MetricSpec``s, not the semantic ``Metric`` models the definitions
    loader returns).

    Carries ``window_days``/``threshold_days`` through: dropping
    ``window_days`` would still give exact parity but degrades sequential
    inference to an "open-ended" warning - carrying it is correct either
    way.
    """
    if isinstance(metric, RatioMetric):
        return MetricSpec(
            name=metric.name,
            type="ratio",
            numerator=metric.numerator.fact,
            denominator=metric.denominator.fact,
            window_days=metric.numerator.window_days,
        )
    if isinstance(metric, RetentionMetric):
        return MetricSpec(name=metric.name, type="retention", threshold_days=metric.threshold_days)
    return MetricSpec(name=metric.name, type=metric.type, window_days=metric.window_days)


def test_latency_source_contains_repeated_checkout_measurements():
    """Post-exposure checkout RUM loads include repeated user measurements."""
    events = pq.read_table(REPO_ROOT / "examples/realistic_demo/warehouse/events")
    rows = events.to_pylist()
    checkout_loads = [
        row
        for row in rows
        if row["event_name"] == "page_load"
        and row["page"] == "/checkout"
        and date(2025, 1, 15) <= row["event_ts"].date() < date(2025, 2, 1)
    ]
    counts = Counter(row["user_id"] for row in checkout_loads)
    assert counts
    assert min(counts.values()) >= 1
    assert max(counts.values()) > 1


def test_run_returns_finite_estimates_for_all_declared_metrics(analysis):
    estimates = analysis.run()
    assert {e.metric for e in estimates} == METRIC_NAMES
    for e in estimates:
        assert math.isfinite(e.require_lift().value)
        lift = e.require_lift()
        # The retention guardrail has a calibrated one-sided interval.
        # The quantile readout still reports its central interval.
        expected_open_side = "upper" if e.metric == "d7_retention" else None
        assert lift.open_side == expected_open_side
        assert lift.lb is not None and math.isfinite(lift.lb)
        if expected_open_side == "upper":
            assert e.alternative == "greater"
            assert lift.ub is None
        else:
            assert lift.ub is not None and math.isfinite(lift.ub)


def test_srm_reports_no_mismatch_on_triggered_population(analysis):
    """Guards the generator's counterfactual (arm-independent) triggering:
    an arm-dependent trigger rate would surface here as a sample-ratio
    mismatch."""
    result = analysis.srm(expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(result, SRMResult)
    assert not isinstance(result, NotApplicable)
    assert result.is_srm is False


def test_breakout_country_segments_have_no_null_bin(analysis):
    with pytest.warns(IncrementWarning) as record:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            estimates = analysis.run_breakout()
    assert all(code.startswith("breakout.estimates.") for code in warning_codes(record))
    country_rows = [
        e for e in estimates if e.metric == "conversion_rate" and e.dimension == "country"
    ]
    assert {e.dimension_value for e in country_rows} == {"US", "GB", "DE"}
    assert len(country_rows) == 3
    assert {e.group_id for e in country_rows} == {"treatment"}
    values = []
    for e in country_rows:
        lift = e.lift
        if lift is None:
            assert e.excluded is not None
            continue
        values.append(lift.value)
    assert values, "expected at least one estimable country segment"
    assert len(set(values)) == len(values)


def test_breakout_plan_segments_are_pre_exposure_free_only(analysis):
    """The single most important assertion in this module: the only check
    that distinguishes a correct ``as_of=pre_exposure`` temporal
    resolution from a naive event-time join - it proves the generator
    never leaks a post-exposure plan upgrade into a pre-exposure lookup.

    A known, already-fixed defect this exact assertion would have caught:
    an earlier generator version let pre-experiment events predate a
    user's own signup, which nulled ~4.3% of plan resolutions. If this
    assertion fails, that class of bug is back - don't loosen it, find
    the regression.
    """
    with pytest.warns(IncrementWarning) as record:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            estimates = analysis.run_breakout()
    assert all(code.startswith("breakout.estimates.") for code in warning_codes(record))
    plan_segments = {e.dimension_value for e in estimates if e.dimension == "plan"}
    assert plan_segments == {"free"}


def test_daily_and_asof_lift_cover_all_metrics(analysis):
    """The day-axis / dense-panel code paths, distinct from the
    whole-window run() path exercised above."""
    fixed_plan = analysis.experiment.plan.model_copy(update={"inference": None})
    analysis = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(update={"plan": fixed_plan}),
    )
    daily = analysis.run_daily_lift()
    assert daily
    assert {e.metric for e in daily} == METRIC_NAMES

    asof = analysis.run_asof_lift()
    assert asof
    assert {e.metric for e in asof} == METRIC_NAMES


def test_export_from_moments_round_trips_run_exactly(analysis, tmp_path):
    """A real export/rehydrate must reproduce run() exactly, including the
    declared plan: without it a guardrail comes back unassigned and
    two-sided, widening its interval past what was pre-registered.
    """
    baseline = {(e.metric, e.group_id): e.require_lift() for e in _lift_rows(analysis.run())}

    path = tmp_path / "moments.parquet"
    analysis.export(path)
    rows = pq.read_table(path).to_pylist()
    assert {row["moments_format"] for row in rows} == {10}

    specs = [_metric_spec(m) for m in analysis.metrics]
    rehydrated_src = Analysis.from_moments(rows, metrics=specs, control="control")
    rehydrated = _lift_rows(rehydrated_src.run())

    # The declared plan survives, observable through the public result rows.
    declared = next(e for e in _lift_rows(analysis.run()) if e.metric == "d7_retention")
    carried = next(e for e in rehydrated if e.metric == "d7_retention")
    assert declared.role == "guardrail"
    assert carried.role == declared.role
    assert carried.alternative == declared.alternative

    assert {(e.metric, e.group_id) for e in rehydrated} == set(baseline)
    for e in rehydrated:
        base = baseline[(e.metric, e.group_id)]
        lift = e.require_lift()
        assert lift.value is not None and base.value is not None
        assert abs(lift.value - base.value) < 1e-9
        assert lift.open_side == base.open_side
        for endpoint, expected in ((lift.lb, base.lb), (lift.ub, base.ub)):
            if expected is None:
                assert endpoint is None
            else:
                assert endpoint is not None
                assert abs(endpoint - expected) < 1e-9


def test_export_round_trips_registered_native_actual_state(analysis, con, tmp_path):
    from increment.semantics import load
    from increment.semantics.models import AnalysisPlan
    from tests.analysis_factory import make_analysis
    from tests.sequential_cases import registered_native

    metric = next(m for m in analysis.metrics if m.type == "mean")
    metric = metric.model_copy(update={"window_days": 7})
    # The artifact extension catalog follows the saved experiment's metric
    # roster, so declare only this supported bounded Gaussian target there too.
    experiment = analysis.experiment.model_copy(
        update={"plan": AnalysisPlan(secondaries=[metric.name])}
    )
    definitions = load(DEFINITIONS).model_copy(
        update={"experiments": (experiment,), "metrics": (metric,)}
    )
    selected = make_analysis(con, definitions, experiment=experiment)
    # A plain-mean metric under a fixed Randomized mechanism: the public
    # scalar_mean law admits this today; it is only
    # encouragement mechanisms that still refuse.
    registered = registered_native(selected, public_mean=True)
    end = analysis.experiment.end
    assert end is not None
    checkpoint = registered.capture_sequential(finalized=True, as_of=end.date())
    # The finite schedule must be declared independently of observed sample size.
    # This test transports an anytime checkpoint; finite-look state is tested
    # with fixed counts in test_readouts_plan's registered finite-look witness.
    path = tmp_path / "sequential.parquet"
    registered.export(path)
    rows = pq.read_table(path).to_pylist()
    replay = Analysis.from_moments(rows, metrics=[_metric_spec(metric)], control="control")
    assert replay.sequential_snapshot() == checkpoint
    assert (
        _lift_rows(replay.run())[0].require_sequential_result()
        == _lift_rows(registered.run())[0].require_sequential_result()
    )
