"""Design-cohort compliance, independent of outcome selection and wire transport."""

import json
import math
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta
from typing import Any, cast

import narwhals as nw
import pyarrow as pa
import pytest

from increment import Analysis, readouts
from increment.errors import CapabilityError, CodedError, IncrementRuntimeWarning, IncrementWarning
from increment.estimation.encouragement import (
    _arm_stats_from_compliance,
    estimate_compliance,
)
from increment.frame import MetricSpec, from_unit_panel, from_unit_summary
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
from increment.sources import ComplianceArm, ComplianceSummary, MomentsSource
from tests.analysis_factory import _moment_source
from tests.warning_codes import warning_codes


def design(window=None):
    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=window),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="Only uptake changes the outcome"
        ),
        one_sided=True,
        min_first_stage_z=0.001,
    )


def witness(order=("full", "drop")):
    rows = []
    for group in ("control", "treatment"):
        for i in range(100):
            d = float(group == "treatment" and i < 50)
            y = 2.0 + (i % 2) + 2 * d
            observed = i < 9 or i == 50
            rows.append(
                {
                    "unit": f"{group}{i}",
                    "arm": group,
                    "clicked": d,
                    "full": y,
                    "drop": y if observed else None,
                }
            )
    warning = pytest.warns(IncrementWarning) if "drop" in order else nullcontext()
    with warning as rec:
        source = from_unit_summary(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="arm",
            control="control",
            design=design(),
            metrics=[
                MetricSpec(name=name, type="mean", missing="drop" if name == "drop" else "error")
                for name in order
            ],
            experiment_id="witness",
        )
    if "drop" in order:
        assert rec is not None
        assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    return source


def compliance_missingness_source(*, missing_treatment: bool, clustered: bool = False):
    rows = []
    for group in ("control", "treatment"):
        for i in range(20):
            rows.append(
                {
                    "unit": f"{group}{i}",
                    "arm": group,
                    "cluster": f"{group}{i // 4}",
                    "clicked": float(group == "treatment" and i < 10),
                    "full": float(2 + (i % 2) + (group == "treatment")),
                    "drop": (
                        None
                        if missing_treatment and group == "treatment"
                        else float(2 + (i % 2) + (group == "treatment"))
                    ),
                }
            )
    return from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        control="control",
        design=design(),
        metrics=[
            MetricSpec(name=name, type="mean", missing="drop" if name == "drop" else "error")
            for name in ("full", "drop")
        ],
        experiment_id="compliance-missingness",
        cluster="cluster" if clustered else None,
    )


@pytest.mark.parametrize("metrics", [("drop",), ("full", "drop"), ("drop", "full")])
def test_compliance_only_readout_is_independent_of_outcome_missingness(metrics):
    with pytest.warns(IncrementWarning):
        missing = compliance_missingness_source(missing_treatment=True)
    complete = compliance_missingness_source(missing_treatment=False)

    expected = readouts.run(complete, metrics=metrics, estimands=["compliance"])
    actual = readouts.run(missing, metrics=metrics, estimands=["compliance"])

    assert [(row.estimand, row.metric) for row in actual] == [("compliance", "uptake")]
    observed, oracle = actual[0].require_lift(), expected[0].require_lift()
    assert (observed.value, observed.lb, observed.ub) == pytest.approx(
        (oracle.value, oracle.lb, oracle.ub)
    )
    assert observed.value == pytest.approx(0.5)


def test_mixed_outcome_request_still_requires_available_outcomes():
    with pytest.warns(IncrementWarning):
        source = compliance_missingness_source(missing_treatment=True)
    with pytest.raises(CodedError) as error:
        readouts.run(source, metrics=["drop"], estimands=["compliance", "late"])
    assert error.value.code == "readout.arms.no_treatment"


def test_compliance_only_control_source_refuses_no_treatment():
    rows = [
        {
            "unit": f"control{i}",
            "arm": "control",
            "clicked": 0.0,
            "full": float(2 + (i % 2)),
            "drop": float(2 + (i % 2)),
        }
        for i in range(20)
    ]
    source = from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        control="control",
        design=design(),
        metrics=[
            MetricSpec(name="full", type="mean"),
            MetricSpec(name="drop", type="mean"),
        ],
        experiment_id="compliance-control-only",
    )

    with pytest.raises(CodedError) as error:
        readouts.run(source, metrics=["drop"], estimands=["compliance"])

    assert error.value.code == "readout.arms.no_treatment"
    assert error.value.context["control_group"] == "control"
    assert error.value.context["observed_arms"] == ("control",)


def _warehouse_compliance_analysis(
    outcome, *, clustered=False, breakout=False, unusable_outcome=False, triggered=False
):
    from increment import AnalysisPlan
    from increment.semantics.models import Definitions
    from tests.analysis_factory import make_analysis
    from tests.parity_harness.cases import (
        _encouragement_defs_dict,
        _encouragement_rows_for_parity,
    )
    from tests.parity_harness.dataset import duckdb_connection

    rows = _encouragement_rows_for_parity()
    for row in rows:
        row["cohort"] = row["user_id"]
    definitions = _encouragement_defs_dict(
        AnalysisPlan(secondaries=[] if outcome is None else ["revenue"])
    )
    if outcome is None:
        definitions["metrics"] = []
    if unusable_outcome:
        definitions["fact_sources"][0]["sql"] = (
            "SELECT * EXCLUDE (revenue), CAST(user_id AS DOUBLE) AS revenue FROM events"
        )
    definitions["fact_sources"][0]["properties"] = [
        {"name": "cohort", "column": "cohort", "dtype": "string", "as_of": "static"}
    ]
    experiment = definitions["experiments"][0]
    if triggered:
        definitions["exposures"].append({"name": "activated", "fact": "exposure"})
        experiment["trigger"] = "activated"
    if clustered:
        experiment["cluster"] = "cohort"
    if breakout:
        experiment["breakouts"] = [{"property": "cohort"}]
    if outcome == "ratio":
        definitions["metrics"][0] = {
            "type": "ratio",
            "name": "revenue",
            "entity": "user_id",
            "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 1},
            "denominator": {"fact": "exposure", "aggregation": "count", "window_days": 1},
        }
    elif outcome == "retention":
        definitions["metrics"][0] = {
            "type": "retention",
            "name": "revenue",
            "entity": "user_id",
            "fact": "purchase",
            "threshold_days": 1,
        }
    elif outcome == "percentile":
        definitions["metrics"][0]["winsorization"] = {"upper_percentile": 0.9}
    con = duckdb_connection(rows)
    return make_analysis(con, Definitions.model_validate(definitions), experiment="exp"), con


@pytest.mark.slow
@pytest.mark.parametrize(
    ("outcome", "outcome_refusal"),
    [
        ("ratio", "readout.encouragement.cluster_ratio"),
        ("retention", "readout.encouragement.retention"),
        ("percentile", "readout.metric.percentile_winsorization"),
    ],
)
def test_compliance_only_ignores_outcome_configuration(outcome, outcome_refusal):
    from tests.analysis_factory import lift_rows

    clustered = outcome == "ratio"
    analysis, con = _warehouse_compliance_analysis(outcome, clustered=clustered)
    oracle, oracle_con = _warehouse_compliance_analysis("mean", clustered=clustered)
    try:
        observed = lift_rows(analysis.run(estimands=("compliance",)))
        expected = lift_rows(oracle.run(estimands=("compliance",)))[0].require_lift()
        assert [(row.metric, row.estimand) for row in observed] == [("uptake", "compliance")]
        actual = observed[0].require_lift()
        assert actual.value == pytest.approx(0.5)
        assert (actual.value, actual.lb, actual.ub) == pytest.approx(
            (expected.value, expected.lb, expected.ub)
        )
        with pytest.raises(CodedError) as error:
            analysis.run(estimands=("itt",))
        assert error.value.code == outcome_refusal
    finally:
        analysis.close()
        oracle.close()
        con.disconnect()
        oracle_con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("triggered", [False, True])
@pytest.mark.parametrize(("outcome", "metrics"), [("mean", None), ("mean", []), (None, None)])
def test_public_compliance_does_not_reduce_unusable_outcomes(triggered, outcome, metrics):
    from tests.analysis_factory import lift_rows

    analysis, con = _warehouse_compliance_analysis(
        outcome, unusable_outcome=True, triggered=triggered
    )
    oracle, oracle_con = _warehouse_compliance_analysis("mean", triggered=triggered)
    try:
        observed = lift_rows(analysis.run(metrics=metrics, estimands=("compliance",)))
        expected = lift_rows(oracle.run(estimands=("compliance",)))
        populations = ("assigned", "triggered") if triggered else ("assigned",)
        assert [(row.metric, row.estimand, row.analysis_population) for row in observed] == [
            ("uptake", "compliance", population) for population in populations
        ]
        for actual_row, expected_row in zip(observed, expected, strict=True):
            actual, target = actual_row.require_lift(), expected_row.require_lift()
            assert (actual.value, actual.lb, actual.ub) == pytest.approx(
                (target.value, target.lb, target.ub)
            )
            assert actual.value == pytest.approx(0.5)
    finally:
        analysis.close()
        oracle.close()
        con.disconnect()
        oracle_con.disconnect()


@pytest.mark.slow
def test_compliance_only_refuses_declared_segmentation():
    analysis, con = _warehouse_compliance_analysis("mean", breakout=True)
    try:
        with pytest.raises(CodedError) as error:
            readouts.run(_moment_source(analysis), estimands=("compliance",), by=("cohort",))
        assert error.value.code == "readout.run.segment_unsupported"
        assert error.value.context["metric"] == "uptake"
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("order", [("full", "drop"), ("drop", "full"), ("full",), ("drop",)])
def test_original_order_missingness_and_metric_count_witness(order):
    source = witness(order)
    rows = readouts.run(source, estimands=["compliance", "late"])
    compliance = [row for row in rows if row.estimand == "compliance"]
    assert len(compliance) == 1
    assert compliance[0].require_lift().value == pytest.approx(0.50)
    assert compliance[0].require_lift().lb is not None
    for metric in source.context.metrics:
        arms = {row["group_id"]: row for row in source.moments(metric)}
        t, c = arms["treatment"], arms["control"]
        first_stage = t["sum_d"] / t["n"] - c["sum_d"] / c["n"]
        assert first_stage == pytest.approx(0.9 if metric.name == "drop" else 0.5)
        itt = t["ref_y"] + t["cy1"] / t["n"] - c["ref_y"] - c["cy1"] / c["n"]
        late = next(
            row
            for row in rows
            if row.metric == metric.name
            and row.estimand == "late"
            and row.value_scale == "absolute"
        )
        assert late.require_lift().value == pytest.approx(itt / first_stage)


def clustered_source():
    rows = []
    pairs = [(2, 0), (4, 1), (7, 5), (12, 11)] * 3
    for group in ("control", "treatment"):
        for g, (size, uptake) in enumerate(pairs):
            for i in range(size):
                d = float(group == "treatment" and i < uptake)
                rows.append(
                    {
                        "unit": f"{group}{g}_{i}",
                        "arm": group,
                        "cluster": f"{group}{g}",
                        "clicked": d,
                        "y": 2 + d + i % 2,
                    }
                )
    source = from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        control="control",
        cluster="cluster",
        design=design(),
        metrics={"y": "mean"},
        experiment_id="unequal",
    )
    return source, pairs


def test_unequal_correlated_cluster_sizes_have_member_weighted_variance():
    source, pairs = clustered_source()
    summary = source.compliance_summary(design())
    t = summary.arm("treatment")
    assert t is not None
    k = len(pairs)
    n = sum(size for size, _ in pairs)
    rate = sum(uptake for _, uptake in pairs) / n
    scores = [uptake - rate * size for size, uptake in pairs]
    oracle = sum((score - sum(scores) / k) ** 2 for score in scores) / (k - 1) / k / (n / k) ** 2
    assert rate != pytest.approx(sum(uptake / size for size, uptake in pairs) / k)
    assert t.n_units == n
    with pytest.warns(IncrementRuntimeWarning) as rec:
        result = estimate_compliance(summary, design()).results[0]
    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    from scipy.stats import t as student_t

    # The control uptake is constant, so the reference dof is k - 1.
    critical = student_t.isf(0.025, k - 1)
    lift = result.require_lift()
    assert lift.lb is not None and lift.ub is not None
    assert lift.value == pytest.approx(rate)
    assert (lift.ub - lift.lb) / 2 == pytest.approx(critical * math.sqrt(oracle), rel=1e-12)
    assert lift.lb == pytest.approx(rate - critical * math.sqrt(oracle))
    assert lift.ub == pytest.approx(rate + critical * math.sqrt(oracle))


@pytest.mark.parametrize("clustered", [False, True])
def test_portable_compliance_is_independent_of_missing_outcomes(tmp_path, clustered):
    import pyarrow.parquet as pq

    from tests.analysis_factory import lift_rows

    with pytest.warns(IncrementWarning):
        source = compliance_missingness_source(missing_treatment=True, clustered=clustered)
    complete = compliance_missingness_source(missing_treatment=False, clustered=clustered)
    path = tmp_path / "moments.parquet"
    source.export_moments(path)
    public = Analysis.from_moments(
        pq.read_table(path).to_pylist(),
        metrics=[MetricSpec(name="drop", missing="drop")],
        design=design(),
    )
    try:
        with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext():
            expected = readouts.run(complete, metrics=["drop"], estimands=["compliance"])[0]
            actual = lift_rows(public.run(metrics=["drop"], estimands=["compliance"]))[0]
        observed, oracle = actual.require_lift(), expected.require_lift()
        assert (observed.value, observed.lb, observed.ub) == pytest.approx(
            (oracle.value, oracle.lb, oracle.ub)
        )
        assert observed.value == pytest.approx(0.5)
        assert actual.reference_kind == expected.reference_kind
        assert actual.reference_df == expected.reference_df
    finally:
        public.close()


def test_summary_freezes_nested_callers_and_canonicalizes_arms():
    arms = [ComplianceArm("treatment", 20, 10), ComplianceArm("control", 20, 0)]
    summary = ComplianceSummary("s", "clicked", 7, True, None, None, cast(Any, arms))
    arms.clear()
    assert tuple(arm.group_id for arm in summary.arms) == ("control", "treatment")
    payload = cast(Any, summary.to_wire())
    copy = ComplianceSummary.from_wire(payload, study_id="s")
    payload["arms"][0]["uptake_total"] = 20
    assert copy == summary
    with pytest.raises(FrozenInstanceError):
        cast(Any, summary.arms[0]).uptake_total = 20
    with pytest.raises(FrozenInstanceError):
        cast(Any, summary).arms = ()


@pytest.mark.parametrize(
    "change",
    [
        {"n_units": 1.5},
        {"n_units": True},
        {"uptake_total": float("nan")},
        {"uptake_total": 21},
        {"ref_uptake": 0.0},
        {"n_clusters": 2},
    ],
)
def test_malformed_arm_state_is_coded(change):
    payload: dict[str, Any] = {"group_id": "t", "n_units": 20, "uptake_total": 10, **change}
    with pytest.raises(CodedError):
        ComplianceArm(**payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ref_uptake", 100.0),
        ("cluster_uptake2", -1.0),
        ("cluster_cross", 1000000.0),
        ("cluster_size1", None),
        ("n_clusters", 2.5),
    ],
)
def test_inconsistent_centered_families_refuse(field, value):
    arm = clustered_source()[0].compliance_summary(design()).arm("treatment")
    assert arm is not None
    with pytest.raises(CodedError):
        replace(arm, **{field: value})


def test_nonzero_reference_residuals_survive_conversion_and_wire():
    # Cluster values U=(0,1,5), M=(2,4,7), references deliberately away from means.
    t = ComplianceArm("treatment", 13, 6, 3, 1.0, 3.0, 17.0, 2.0, 7.0, 29.0, 20.0)
    c = ComplianceArm("control", 13, 0, 3, 0.0, 0.0, 0.0, 2.0, 7.0, 29.0, 0.0)
    summary = ComplianceSummary("s", "clicked", None, True, "g", None, (t, c))
    restored = ComplianceSummary.from_wire(summary.to_wire(), study_id="s")
    restored_arm = restored.arm("treatment")
    assert restored_arm is not None
    stats = _arm_stats_from_compliance(restored_arm, study_id="s", cluster="g")
    assert (stats.cx1, stats.cden1, stats.cxden) == (3.0, 7.0, 20.0)
    assert stats.mean_x() == 2.0
    assert stats.mean_den() == pytest.approx(13 / 3)


def test_duplicate_arms_and_design_window_mismatch_refuse():
    arm = ComplianceArm("control", 20, 0)
    with pytest.raises(CodedError) as error:
        ComplianceSummary("s", "clicked", 7, True, None, None, (arm, arm))
    assert error.value.code == "source.compliance_summary.duplicate_arms"
    with pytest.raises(TypeError):
        cast(Any, error.value.context)["duplicates"] = ()
    summary = witness().compliance_summary(design())
    with pytest.raises(CodedError) as mismatch:
        estimate_compliance(summary, design(7))
    assert mismatch.value.code == "source.compliance_summary.design_mismatch"


@pytest.mark.parametrize(
    "corruption",
    [
        "partial_rows",
        "null_payload",
        "conflict",
        "duplicate_key",
        "duplicate_arm",
        "wrong_version",
        "wrong_boolean",
    ],
)
def test_malformed_current_cube_refuses_at_construction(tmp_path, corruption):
    import pyarrow.parquet as pq

    source = witness()
    path = tmp_path / "cube.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    payload = json.loads(rows[0]["compliance_summary"])
    if corruption == "partial_rows":
        del rows[0]["compliance_summary"]
    elif corruption == "null_payload":
        for row in rows:
            row["compliance_summary"] = None
    elif corruption == "conflict":
        payload["window_days"] = 7
        rows[0]["compliance_summary"] = payload
    elif corruption == "duplicate_key":
        rows[0]["compliance_summary"] = '{"version":1,"version":1}'
    else:
        if corruption == "duplicate_arm":
            payload["arms"].append(payload["arms"][0])
        elif corruption == "wrong_version":
            payload["version"] = 2
        else:
            payload["one_sided"] = "false"
        for row in rows:
            row["compliance_summary"] = payload
    with pytest.raises(CodedError):
        MomentsSource(rows, metrics=source.context.metrics, study_id="witness", design=design())


def test_legacy_cube_refusal_is_distinct_from_malformed_state(tmp_path):
    import pyarrow.parquet as pq

    source = witness()
    path = tmp_path / "cube.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        del row["compliance_summary"]
    legacy = MomentsSource(
        rows, metrics=source.context.metrics, study_id="witness", design=design()
    )
    with pytest.raises(CapabilityError) as error:
        legacy.compliance_summary(design())
    assert error.value.code == "source.compliance_summary.legacy_uptake_state"


def test_panel_asof_excludes_future_enrollment_and_out_of_window_uptake():
    start = date(2025, 1, 1)
    rows = []
    for group in ("control", "treatment"):
        for i, click in enumerate((-1, 0, 6, 7)):
            for day in range(-1, 10):
                rows.append(
                    {
                        "unit": f"{group}{i}",
                        "arm": group,
                        "ds": start + timedelta(days=day),
                        "exposed": start,
                        "clicked": float(group == "treatment" and day == click),
                        "y": 1.0,
                    }
                )
    rows.extend(
        {
            "unit": "future",
            "arm": "treatment",
            "ds": start + timedelta(days=day),
            "exposed": start + timedelta(days=9),
            "clicked": 0.0,
            "y": 1.0,
        }
        for day in range(10)
    )
    source = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        control="control",
        date="ds",
        exposure_date="exposed",
        design=design(7),
        metrics={"y": "mean"},
    )
    for day, expected in [(0, 0.25), (5, 0.25), (6, 0.5), (7, 0.5)]:
        arm = source.compliance_summary(design(7), as_of=start + timedelta(days=day)).arm(
            "treatment"
        )
        assert arm is not None
        assert arm.n_units == 4
        assert arm.uptake_total / arm.n_units == expected


def test_cube_snapshots_caller_owned_nested_compliance_payload(tmp_path):
    import pyarrow.parquet as pq

    source = witness()
    path = tmp_path / "cube.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    payload = json.loads(rows[0]["compliance_summary"])
    for row in rows:
        row["compliance_summary"] = payload
    cube = MomentsSource(rows, metrics=source.context.metrics, study_id="witness", design=design())
    expected = cube.compliance_summary(design())
    payload["arms"][1]["uptake_total"] = 90
    payload["arms"].clear()
    assert cube.compliance_summary(design()) == expected
    assert estimate_compliance(expected, design()).results[0].require_lift().value == pytest.approx(
        0.5
    )


@pytest.mark.parametrize(
    "as_of", [date(2025, 1, 7), datetime(2025, 1, 7), 7, 7.5, "day_7", "2025-01-07"]
)
def test_asof_identity_survives_standalone_wire(as_of):
    summary = replace(witness().compliance_summary(design()), as_of=as_of)
    assert ComplianceSummary.from_wire(summary.to_wire(), study_id="witness") == summary


@pytest.mark.parametrize(
    "labels",
    [
        (datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 10)),
        (-1, 2, 10),
        (-1.5, 2.5, 10.5),
        ("d-1", "d2", "d10"),
        ("2025-01-01", "2025-01-02", "2025-01-10"),
    ],
)
@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("path", ["analysis", "readouts"])
@pytest.mark.parametrize("explicit_anchor", [False, True])
def test_panel_asof_compliance_preserves_day_axis_identity_and_order(
    labels, backend, path, explicit_anchor
):
    rows = [
        {
            "unit": f"{group}{unit}",
            "arm": group,
            "exposed": labels[0],
            "ds": day,
            "clicked": float(group == "treatment" and unit < (index + 1) * 5),
            "y": 2.0 + unit % 2,
        }
        for group in ("control", "treatment")
        for unit in range(20)
        for index, day in enumerate(labels)
    ]
    frame = nw.from_native(pa.Table.from_pylist(list(reversed(rows))))
    native = frame.to_native() if backend == "pyarrow" else getattr(frame, f"to_{backend}")()
    source = from_unit_panel(
        native,
        unit="unit",
        group="arm",
        date="ds",
        control="control",
        design=design(),
        metrics={"y": "mean"},
        exposure_date="exposed" if explicit_anchor else None,
    )
    analysis = Analysis.from_unit_panel(
        native,
        unit="unit",
        group="arm",
        date="ds",
        design=design(),
        metrics={"y": "mean"},
        exposure_date="exposed" if explicit_anchor else None,
    )
    results = (
        analysis.run_asof_lift
        if path == "analysis"
        else lambda **kw: readouts.asof_lift(source, **kw)
    )(estimands=["compliance", "itt"])
    compliance = {
        row.ds: row.require_lift().value for row in results if row.estimand == "compliance"
    }
    assert compliance == pytest.approx(dict(zip(labels, (0.25, 0.5, 0.75), strict=True)))
    assert list(compliance) == list(labels)
    for result in results:
        assert type(result).model_validate_json(result.model_dump_json()) == result
    if path == "analysis":
        exported = (
            nw.from_native(cast(Any, results).to_frame(backend=backend)).get_column("ds").to_list()
        )
        assert list(dict.fromkeys(exported)) == list(labels)
    for day in labels:
        summary = source.compliance_summary(design(), as_of=day)
        assert summary.as_of == day
        assert type(summary.as_of) is type(day)
        arm = summary.arm("treatment")
        assert arm is not None and arm.n_units == 20


@pytest.mark.parametrize("corruption", ["count", "missing_arm", "extra_arm"])
def test_cube_rejects_compliance_assignment_population_disagreement(tmp_path, corruption):
    import pyarrow.parquet as pq

    source = witness(("full",))
    path = tmp_path / "cube.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    payload = json.loads(rows[0]["assignment_counts"])
    if corruption == "count":
        payload["treatment"] += 1
    elif corruption == "missing_arm":
        del payload["treatment"]
    else:
        payload["extra"] = 100
    for row in rows:
        row["assignment_counts"] = payload
    with pytest.raises(CodedError) as error:
        MomentsSource(rows, metrics=source.context.metrics, study_id="witness", design=design())
    assert error.value.code == "source.compliance_summary.invalid_state"


def test_cube_allows_special_assignment_counts_outside_compliance(tmp_path):
    import pyarrow.parquet as pq

    from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

    source = witness(("full",))
    path = tmp_path / "cube.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    payload = json.loads(rows[0]["assignment_counts"])
    payload.update({MIXED_ASSIGNMENT_LABEL: 2, UNASSIGNED_LABEL: 3})
    for row in rows:
        row["assignment_counts"] = payload
    cube = MomentsSource(rows, metrics=source.context.metrics, study_id="witness", design=design())
    assert cube.compliance_summary(design()) == source.compliance_summary(design())
    assert cube.unit_counts() == payload


@pytest.mark.parametrize("labels", [(2, 10, 12), ("day_2", "day_10", "day_12")])
def test_panel_compliance_day_axis_respects_staggered_enrollment_and_window(labels):
    rows = [
        {
            "unit": f"{group}{unit}",
            "arm": group,
            "ds": day,
            "clicked": float(
                group == "treatment" and index == (0 if unit == 0 else 2 if unit == 1 else 1)
            ),
            "y": 1.0,
        }
        for group in ("control", "treatment")
        for unit in range(4)
        for index, day in enumerate(labels)
        if unit < 2 or index >= 1
    ]
    source = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        date="ds",
        control="control",
        design=design(9),
        metrics={"y": "mean"},
    )
    for day, count, uptake in zip(labels, (2, 4, 4), (1, 3, 3), strict=True):
        arm = source.compliance_summary(design(9), as_of=day).arm("treatment")
        assert arm is not None
        assert (arm.n_units, arm.uptake_total) == (count, uptake)


def test_cube_rejects_valid_compliance_arm_with_wrong_member_count(tmp_path):
    import pyarrow.parquet as pq

    source = witness(("full",))
    path = tmp_path / "cube.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    payload = json.loads(rows[0]["compliance_summary"])
    payload["arms"][1]["n_units"] += 1
    for row in rows:
        row["compliance_summary"] = payload
    with pytest.raises(CodedError) as error:
        MomentsSource(rows, metrics=source.context.metrics, study_id="witness", design=design())
    assert error.value.code == "source.compliance_summary.invalid_state"


@pytest.mark.parametrize("axis", ["date", "datetime", "numeric", "structured"])
@pytest.mark.parametrize("path", ["analysis", "readouts"])
@pytest.mark.parametrize("outcome_window", [3, 5])
def test_completed_compliance_excludes_open_staggered_cohorts(axis, path, outcome_window):

    def label(day):
        if axis == "structured":
            return f"day_{day}"
        if axis == "numeric":
            return day
        start = date(2025, 1, 1) if axis == "date" else datetime(2025, 1, 1)
        return start + timedelta(days=day)

    rows = [
        {
            "unit": f"{group}{cohort}{i}",
            "arm": group,
            "ds": label(day),
            "exposed": label(cohort),
            "clicked": float(group == "treatment" and day == cohort + 2 and (cohort > 0 or i < 10)),
            "y": 1.0 + i % 2,
        }
        for group in ("control", "treatment")
        for cohort in (0, 2)
        for i in range(20)
        for day in range(cohort, 7)
    ]
    table = pa.Table.from_pylist(rows)
    source = from_unit_panel(
        table,
        unit="unit",
        group="arm",
        date="ds",
        control="control",
        exposure_date="exposed",
        design=design(3),
        metrics=[MetricSpec(name="y", type="mean", window_days=outcome_window)],
    )
    results = (
        Analysis.from_unit_panel(
            table,
            unit="unit",
            group="arm",
            date="ds",
            design=design(3),
            exposure_date="exposed",
            metrics=[MetricSpec(name="y", type="mean", window_days=outcome_window)],
        ).run_asof_lift
        if path == "analysis"
        else lambda **kw: readouts.asof_lift(source, **kw)
    )(estimands=["compliance"], completed_windows_only=True)
    compliance = {row.ds: row for row in results if row.estimand == "compliance"}
    assert set(compliance) == {label(day) for day in range(3, 7)}
    for day, expected in ((3, 0.5), (4, 0.5), (5, 0.75), (6, 0.75)):
        assert compliance[label(day)].require_lift().value == pytest.approx(expected)
        assert compliance[label(day)].inference == "fixed"
        summary = source.compliance_summary(
            design(3), as_of=label(day), completed_windows_only=True
        )
        arm = summary.arm("treatment")
        assert arm is not None and arm.n_units == (20 if day < 5 else 40)


@pytest.mark.parametrize("window,as_of", [(None, 3), (3, None)])
def test_completed_compliance_requires_bounded_window_and_asof(window, as_of):
    source = from_unit_panel(
        pa.Table.from_pylist(
            [
                {"unit": group, "arm": group, "ds": 0, "clicked": 0.0, "y": 1.0}
                for group in ("control", "treatment")
            ]
        ),
        unit="unit",
        group="arm",
        date="ds",
        control="control",
        design=design(window),
        metrics={"y": "mean"},
    )
    with pytest.raises(CodedError) as error:
        source.compliance_summary(design(window), as_of=as_of, completed_windows_only=True)
    assert error.value.code == "source.compliance_summary.completed_window_required"


@pytest.mark.parametrize("strong_design", [True, False])
@pytest.mark.parametrize("estimands", [["late"], ["compliance", "late"]])
@pytest.mark.parametrize("order", [("full", "y"), ("y", "full")])
def test_late_suppression_uses_outcome_cohort(strong_design, estimands, order):
    declared = design().model_copy(update={"min_first_stage_z": 2.0})
    rows = [
        {
            "unit": f"{group}{i}",
            "arm": group,
            "clicked": float(group == "treatment" and i < (50 if strong_design else 3)),
            "y": (2.0 + i % 2 if (i >= 50 if strong_design else i < 3) else None),
            "full": 2.0 + i % 2,
        }
        for group in ("control", "treatment")
        for i in range(100)
    ]
    with pytest.warns(IncrementWarning) as rec:
        source = from_unit_summary(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="arm",
            control="control",
            design=declared,
            metrics=[
                MetricSpec(name=name, type="mean", missing="drop" if name == "y" else "error")
                for name in order
            ],
        )
    assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    results = readouts.run(source, estimands=estimands)
    late = [row for row in results if row.estimand == "late"]
    diagnostics = [row for row in results if "late suppressed" in (row.note or "")]
    assert {row.metric for row in late} == {"full" if strong_design else "y"}
    assert {row.metric for row in diagnostics} == {"y" if strong_design else "full"}
    design_rows = [row for row in results if row.metric == "uptake"]
    assert len(design_rows) == int("compliance" in estimands)
    assert all("late suppressed" not in (row.note or "") for row in design_rows)
