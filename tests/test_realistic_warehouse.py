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

from increment import Analysis, CupedCovariate, Method, MetricSpec
from increment.breakout.estimates import LiftEstimates
from increment.errors import IncrementWarning, InvalidRequestError
from increment.estimation.results import LiftEstimate
from increment.results import NotApplicable, SRMResult
from increment.semantics.models import RatioMetric, RetentionMetric
from tests.analysis_factory import make_analysis_like
from tests.warning_codes import warning_codes


def _lift_rows(rows: object) -> list[LiftEstimate]:
    return cast(list[LiftEstimate], rows)


_RATIO_SKEW_WARNING = "estimation.engine.ratio_denominator_skew"


def _assert_breakout_warnings(record):
    codes = warning_codes(record)
    assert any(code.startswith("breakout.estimates.") for code in codes)
    assert all(
        code.startswith("breakout.estimates.") or code == _RATIO_SKEW_WARNING for code in codes
    )


def _assert_ratio_skew_warnings(record):
    messages = [
        warning.message for warning in record if isinstance(warning.message, IncrementWarning)
    ]
    assert messages
    assert {message.code for message in messages} == {_RATIO_SKEW_WARNING}
    assert any(
        message.context.get("metric") == "average_order_value"
        and message.context.get("arm") == "treatment"
        for message in messages
    )


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
    result = analysis.srm(expected={"control": 0.5, "treatment": 0.5}, inference="always_valid")
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
    _assert_breakout_warnings(record)
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
    _assert_breakout_warnings(record)
    plan_segments = {e.dimension_value for e in estimates if e.dimension == "plan"}
    assert plan_segments == {"free"}


@pytest.mark.integration
def test_daily_and_asof_lift_cover_all_metrics(analysis):
    """The day-axis / dense-panel code paths, distinct from the
    whole-window run() path exercised above."""
    fixed_plan = analysis.experiment.plan.model_copy(update={"inference": None})
    analysis = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(update={"plan": fixed_plan}),
    )
    with pytest.warns(IncrementWarning) as daily_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            daily = analysis.run_daily_lift()
    _assert_ratio_skew_warnings(daily_warnings)
    assert daily
    assert {e.metric for e in daily} == METRIC_NAMES

    with pytest.warns(IncrementWarning) as asof_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            asof = analysis.run_asof_lift()
    _assert_ratio_skew_warnings(asof_warnings)
    assert asof
    assert {e.metric for e in asof} == METRIC_NAMES


@pytest.mark.integration
def test_realistic_warehouse_artifact_replay_matches_native(analysis, con):
    from increment.estimation.readout_types import CellKey
    from increment.query.session import WarehouseArtifactStore
    from tests.readout_journeys import assert_prior_free_sampling_family

    store = WarehouseArtifactStore(con, schema_name="m3_realistic")
    reference = analysis.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(
        store, reference, expected_context=analysis.artifact_context
    )
    expected_result = analysis.run()
    observed_result = adopted.run()
    assert isinstance(expected_result, LiftEstimates)
    assert isinstance(observed_result, LiftEstimates)
    expected_metadata = expected_result.metadata
    observed_metadata = observed_result.metadata
    assert expected_metadata is not None and observed_metadata is not None
    expected_rows = _lift_rows(expected_result)
    observed_rows = _lift_rows(observed_result)
    expected = {CellKey.from_row(row): row for row in expected_rows}
    observed = {CellKey.from_row(row): row for row in observed_rows}
    assert expected.keys() == observed.keys()
    assert_prior_free_sampling_family(expected_result)
    assert_prior_free_sampling_family(observed_result)
    for cell, expected_row in expected.items():
        observed_row = observed[cell]
        # The independently sourced snapshots/families have source-bound IDs;
        # compare their semantic contents and linkage, not their hashes.
        _assert_nested_approx_equal(
            observed_row.model_dump(mode="json", exclude={"source_snapshot_id", "family_id"}),
            expected_row.model_dump(mode="json", exclude={"source_snapshot_id", "family_id"}),
        )
        assert observed_row.source_snapshot_id is not None
        assert expected_row.source_snapshot_id is not None
        assert observed_row.family_id is not None
        assert expected_row.family_id is not None
        assert observed_row.family_procedure == expected_row.family_procedure
        assert observed_row.family_size == expected_row.family_size
        assert observed_metadata.record(observed_row.source_snapshot_id, cell) is not None
        assert expected_metadata.record(expected_row.source_snapshot_id, cell) is not None
    adopted.close()


def _assert_nested_approx_equal(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_approx_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_approx_equal(actual_item, expected_item)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


@pytest.mark.integration
def test_export_from_moments_round_trips_run_exactly(analysis, tmp_path):
    """A real export/rehydrate must reproduce run() exactly, including the
    declared plan: without it a guardrail comes back unassigned and
    two-sided, widening its interval past what was pre-registered.
    """
    from increment.semantics.models import ConversionMetric, MeanMetric
    from tests.analysis_factory import make_analysis_like

    unequal_allocation = analysis.experiment.model_copy(
        update={"allocation": {"control": 0.6, "treatment": 0.4}}
    )
    analysis = make_analysis_like(analysis, experiment=unequal_allocation)
    metrics = {metric.name: metric for metric in analysis.metrics}

    assert isinstance(metrics["conversion_rate"], ConversionMetric)
    assert metrics["conversion_rate"].window_days == 14
    assert isinstance(metrics["revenue_per_user"], MeanMetric)
    assert metrics["revenue_per_user"].window_days == 14
    source_rows = _lift_rows(analysis.run())
    baseline = {(e.metric, e.group_id): e.require_lift() for e in source_rows}

    path = tmp_path / "moments.parquet"
    analysis.export(path)
    rows = pq.read_table(path).to_pylist()
    assert {row["moments_format"] for row in rows} == {12}

    specs = [_metric_spec(m) for m in analysis.metrics]
    rehydrated_src = Analysis.from_moments(
        rows, metrics=specs, design=analysis.experiment.resolved_design()
    )
    rehydrated = _lift_rows(rehydrated_src.run())

    source_health = analysis.health_check()
    replay_health = rehydrated_src.health_check()
    source_assigned = next(
        check for check in source_health.checks if check.check == "assigned_allocation"
    )
    replay_assigned = next(
        check for check in replay_health.checks if check.check == "assigned_allocation"
    )
    assert source_assigned.availability == "available"
    assert source_assigned.srm is not None
    assert source_assigned.srm.expected == {"control": 0.6, "treatment": 0.4}
    assert replay_assigned.availability == "available"
    assert replay_assigned.srm == source_assigned.srm
    feed_checks = [check for check in source_health.checks if check.check == "feed_completeness"]
    assert [check.availability for check in feed_checks] == ["not_applicable"]
    assert feed_checks[0].conclusion is None

    declared = next(e for e in _lift_rows(analysis.run()) if e.metric == "d7_retention")
    carried = next(e for e in rehydrated if e.metric == "d7_retention")
    assert declared.role == "guardrail"
    assert carried.role == declared.role
    assert carried.alternative == declared.alternative

    source_by_cell = {(e.metric, e.group_id): e for e in source_rows}
    assert {(e.metric, e.group_id) for e in rehydrated} == set(baseline)
    assert len({row.source_snapshot_id for row in source_rows}) == 1
    assert len({row.source_snapshot_id for row in rehydrated}) == 1
    assert all(row.source_snapshot_id is not None for row in (*source_rows, *rehydrated))
    for e in rehydrated:
        source = source_by_cell[(e.metric, e.group_id)]
        assert e.family_id is not None and source.family_id is not None
        assert e.family_procedure == source.family_procedure
        assert e.family_size == source.family_size
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
    assert checkpoint.records
    checkpoint_size = len(checkpoint.model_dump_json().encode("utf-8"))
    assert checkpoint_size > 0
    look_allocations = [cell.alpha for cell in checkpoint.registration.roster]
    assert look_allocations
    assert all(0 < alpha <= checkpoint.registration.q for alpha in look_allocations)
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


def test_duckdb_sequential_capture_streams_bounded_batches_without_closing_connection(
    monkeypatch,
):
    from increment.query.sequential_capture import record_batches

    connection = ibis.duckdb.connect()
    relation = ibis.memtable([{"value": index} for index in range(4097)])
    converted = []
    to_duckdb_relation = connection._to_duckdb_relation

    def capture(relation, *args, **kwargs):
        converted.append(relation)
        return to_duckdb_relation(relation, *args, **kwargs)

    monkeypatch.setattr(connection, "_to_duckdb_relation", capture)
    batches = list(record_batches(connection, relation))
    assert len(converted) == 1
    assert relation.columns == ("value",)
    assert [batch.num_rows for batch in batches] == [4096, 1]
    assert sum(batch.num_rows for batch in batches) == 4097
    assert connection.to_pyarrow(relation).num_rows == 4097
    connection.disconnect()


def test_small_late_panel_and_moments_replay_preserve_semantics(tmp_path):  # noqa: PLR0915
    """Exercise unequal allocation, an immature late cohort, and nullable CUPED input."""
    import pyarrow as pa

    from increment import AnalysisPlan, Randomized
    from increment.estimation.readout_types import CellKey
    from tests.readout_journeys import assert_prior_free_sampling_family

    control_n, treatment_n = 72, 48
    rows = []
    for index in range(control_n + treatment_n):
        group = "control" if index < control_n else "treatment"
        arm_index = index if group == "control" else index - control_n
        exposure = (
            date(2025, 1, 1)
            if arm_index < int((control_n if group == "control" else treatment_n) * 0.6)
            else date(2025, 1, 4)
        )
        covariate = None if index % 11 == 0 else float(index % 7)
        for day in range(7):
            observed_day = date.fromordinal(exposure.toordinal() + day)
            if observed_day > date(2025, 1, 9):
                continue
            rows.append(
                {
                    "unit_id": f"u{index:03d}",
                    "group_id": group,
                    "ds": observed_day,
                    "exposure_date": exposure,
                    "revenue": float((index % 5) + (group == "treatment")) if day == 2 else 0.0,
                    "converted": int(day == 2 and index % (3 if group == "treatment" else 4) == 0),
                    "pre_outcome": float(index % 9),
                    "pre_x": covariate,
                }
            )
    panel = pa.Table.from_pylist(rows)
    design = Randomized(
        control_group="control",
        allocation={"control": 0.6, "treatment": 0.4},
        allocation_scheme="independent",
    )
    plan = AnalysisPlan(primary="revenue", secondaries=("converted", "pre_outcome"))
    metrics = [
        MetricSpec(
            name="revenue",
            type="mean",
            value_column="revenue",
            window_days=7,
            missing="zero",
            preferred_direction="increase",
        ),
        MetricSpec(
            name="converted",
            type="conversion",
            value_column="converted",
            window_days=7,
            missing="zero",
            preferred_direction="increase",
        ),
        MetricSpec(
            name="pre_outcome",
            type="mean",
            value_column="pre_outcome",
            decision_method=Method.cuped(
                covariates=CupedCovariate(name="pre_x", kind="continuous", missing="impute")
            ),
            preferred_direction="increase",
        ),
    ]
    with pytest.warns(IncrementWarning) as warning_record:
        source = Analysis.from_unit_panel(
            panel,
            unit="unit_id",
            group="group_id",
            date="ds",
            exposure_date="exposure_date",
            observation_end=date(2025, 1, 9),
            design=design,
            metrics=metrics,
            plan=plan,
            experiment_id="m3-late-panel",
        )
    imputation_warning = warning_record[0].message
    assert isinstance(imputation_warning, IncrementWarning)
    assert imputation_warning.code == "frame.validation.metric_covariate_missing_imputed"
    assert imputation_warning.context["n_missing"] == 73

    with pytest.warns(IncrementWarning) as censoring_warnings:
        source_result = source.run(decision_method=Method(name="unadjusted"))
    assert censoring_warnings
    for warning in censoring_warnings:
        assert isinstance(warning.message, IncrementWarning)
        assert warning.message.code == "frame.censoring.dropped_units"
        assert warning.message.context["dropped"] == 49
    assert isinstance(source_result, LiftEstimates)
    assert_prior_free_sampling_family(source_result)
    assert any(row.group_id == "treatment" and row.lift is not None for row in source_result)
    path = tmp_path / "m3-late-panel.parquet"
    source.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(),
        design=design,
        metrics=metrics,
        plan=plan,
        experiment_id="m3-late-panel",
    )
    replay_result = replay.run(decision_method=Method(name="unadjusted"))
    assert isinstance(replay_result, LiftEstimates)
    assert_prior_free_sampling_family(replay_result)
    expected = {CellKey.from_row(row): row for row in source_result}
    observed = {CellKey.from_row(row): row for row in replay_result}
    assert expected.keys() == observed.keys()
    assert source_result.metadata is not None and replay_result.metadata is not None
    for cell, expected_row in expected.items():
        observed_row = observed[cell]
        _assert_nested_approx_equal(
            observed_row.model_dump(mode="json", exclude={"source_snapshot_id", "family_id"}),
            expected_row.model_dump(mode="json", exclude={"source_snapshot_id", "family_id"}),
        )
        assert expected_row.source_snapshot_id is not None
        assert observed_row.source_snapshot_id is not None
        assert source_result.metadata.record(expected_row.source_snapshot_id, cell) is not None
        assert replay_result.metadata.record(observed_row.source_snapshot_id, cell) is not None
    source_health = source.health_check()
    replay_health = replay.health_check()
    source_srm = next(
        check for check in source_health.checks if check.check == "assigned_allocation"
    )
    replay_srm = next(
        check for check in replay_health.checks if check.check == "assigned_allocation"
    )
    assert source_srm.availability == replay_srm.availability == "available"
    assert source_srm.srm is not None and replay_srm.srm == source_srm.srm
    assert source_srm.srm.expected == {"control": 0.6, "treatment": 0.4}

    # This source capability is explicitly refused: panel data has no per-unit
    # window collapse to which CUPED can attach the pre-period covariate.
    with pytest.raises(InvalidRequestError) as refused:
        Analysis.from_unit_panel(
            panel,
            unit="unit_id",
            group="group_id",
            date="ds",
            exposure_date="exposure_date",
            design=design,
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    value_column="revenue",
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(name="pre_x", kind="continuous")
                    ),
                    window_days=7,
                    preferred_direction="increase",
                )
            ],
            experiment_id="m3-windowed-covariate-refusal",
        )
    assert refused.value.code == "frame.validation.from_unit_panel"
    source.close()
    replay.close()
