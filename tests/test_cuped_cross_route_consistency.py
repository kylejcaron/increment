from __future__ import annotations

import pyarrow as pa
import pytest

from increment import Analysis, Method, MetricSpec
from increment.errors import CodedError
from increment.estimation.results import LiftEstimate
from tests.test_cuped_route_evidence import _records, _scalar


@pytest.mark.parametrize("scalar", [False, True])
def test_panel_repeated_conversion_refuses_before_scalar_inference(scalar):
    records = [
        {
            **row,
            "day": day,
            "converted": float(i % 3 == 0 or (day == 1 and i % 2 == 0)),
        }
        for i, row in enumerate(_records())
        for day in (0, 1)
    ]
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(records),
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="converted",
                type="conversion",
                decision_method=_scalar("adjusted", "history_a") if scalar else Method.unadjusted(),
            )
        ],
    )
    with pytest.raises(CodedError) as raised:
        analysis.run()
    assert raised.value.code == "frame.moments.metric_type_conversion"
    assert raised.value.context["metric"] == "converted"
    assert raised.value.context["value"] == pytest.approx(2.0)


def _linear_binding_case(tmp_path):
    import yaml

    from increment.semantics.models import CupedCovariateBinding
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, _, tenure, groups, revenue = _scalar_native_case(tmp_path)
    con.raw_sql(
        "INSERT INTO cov_events (user_id, ts, event, experiment_id, group_id, revenue, "
        "tenure, history_b, segment) "
        "SELECT user_id, TIMESTAMP '2025-07-25', 'purchase', NULL, NULL, "
        "(CAST(tenure AS INTEGER) * 7) % 11, tenure, history_b, segment "
        "FROM cov_events WHERE tenure IS NOT NULL"
    )
    declared = Method.cuped_multicovariate(
        name="linear",
        covariates=CupedCovariateBinding(
            name="history", kind="continuous", missing="refuse", property="tenure", source="events"
        ),
    )
    definitions = yaml.safe_load(path.read_text())
    definitions["experiments"][0]["n_pre_periods"] = 30
    definitions["experiments"][0]["plan"]["primary"] = {
        "metric": "revenue",
        "decision_method": declared.model_dump(mode="json", exclude_none=True),
    }
    path.write_text(yaml.safe_dump(definitions))
    expected_table = pa.Table.from_pylist(
        [
            {
                "unit": unit,
                "arm": groups[unit],
                "revenue": revenue[unit],
                "history": float((int(tenure[unit]) * 7) % 11),
            }
            for unit in groups
        ]
    )
    return path, con, expected_table


def _rebound_linear(binding):
    from increment.semantics.models import CupedCovariateBinding, Measure

    recipe = (
        {"property": "history_b", "source": "events"}
        if binding == "property"
        else {"measure": Measure(fact="purchase", aggregation="sum")}
    )
    return Method.cuped_multicovariate(
        name="linear",
        covariates=CupedCovariateBinding.model_validate(
            {"name": "history", "kind": "continuous", "missing": "refuse", **recipe}
        ),
    )


@pytest.mark.parametrize("binding", ["property", "measure"])
def test_native_multicovariate_override_acquires_effective_binding(tmp_path, binding):
    from increment.semantics.models import CupedCovariate

    path, con, table = _linear_binding_case(tmp_path)
    requested = _rebound_linear(binding)
    try:
        with Analysis.from_definitions("cov_test", path, con) as native:
            (actual,) = native.run(decision_method=requested, sensitivity_methods=())
        with Analysis.from_unit_summary(
            table,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped_multicovariate(
                        name="linear",
                        covariates=CupedCovariate(
                            name="history", kind="continuous", missing="refuse"
                        ),
                    ),
                )
            ],
        ) as frame:
            (expected,) = frame.run()
        assert isinstance(actual, LiftEstimate) and isinstance(expected, LiftEstimate)
        assert actual.require_lift().value == pytest.approx(
            expected.require_lift().value, rel=1e-10
        )
        assert actual.require_lift().log_se == pytest.approx(
            expected.require_lift().log_se, rel=1e-10
        )
    finally:
        con.disconnect()


@pytest.mark.parametrize("binding", ["property", "measure"])
def test_artifact_multicovariate_override_refuses_unpublished_binding_before_reads(
    tmp_path, monkeypatch, binding
):
    from increment.readouts import run
    from tests.test_cuped_route_evidence import _artifact_source

    path, con, _ = _linear_binding_case(tmp_path)
    source = _artifact_source(path, con)
    reads = []

    def unread(*args, **kwargs):
        reads.append("evidence")
        pytest.fail("unpublished multicovariate binding accessed artifact evidence")

    monkeypatch.setattr(con, "to_pyarrow", unread)
    try:
        with pytest.raises(CodedError) as raised:
            run(source, decision_method=_rebound_linear(binding), sensitivity_methods=())
        assert raised.value.code == "facade.analysis_config.covariate_binding_conflict"
        assert raised.value.context["covariate"] == "history"
        assert reads == []
    finally:
        source.close()
        con.disconnect()


def test_native_scalar_imputation_uses_population_before_window_censoring(tmp_path):
    import yaml

    from increment.readouts import run
    from increment.semantics.models import CupedCovariateBinding
    from tests.test_cuped_covariate_sources import _scalar_native_case
    from tests.test_cuped_route_evidence import _artifact_source

    path, con, _, tenure, groups, revenue = _scalar_native_case(tmp_path)
    declared = Method.cuped(
        name="adjusted",
        covariates=CupedCovariateBinding(
            name="history_a",
            kind="continuous",
            missing="impute",
            property="tenure",
            source="events",
        ),
    )
    definitions = yaml.safe_load(path.read_text())
    experiment = definitions["experiments"][0]
    experiment["n_pre_periods"] = 30
    experiment["end"] = "2025-08-20"
    experiment["plan"]["primary"] = {
        "metric": "revenue",
        "decision_method": declared.model_dump(mode="json", exclude_none=True),
    }
    path.write_text(yaml.safe_dump(definitions))
    con.raw_sql(
        "UPDATE cov_events SET ts = TIMESTAMP '2025-08-18 09:00:00' "
        "WHERE experiment_id = 'cov_test' AND user_id IN ('u9', 'u19')"
    )
    con.raw_sql(
        "INSERT INTO cov_events (user_id, ts, event, experiment_id, group_id, revenue, tenure) "
        "SELECT user_id, TIMESTAMP '2025-08-17', 'exposure', NULL, NULL, NULL, 1000.0 "
        "FROM cov_events WHERE tenure IS NOT NULL AND user_id IN ('u9', 'u19')"
    )
    con.raw_sql("UPDATE cov_events SET tenure = NULL WHERE user_id IN ('u0', 'u10')")
    # Observed enrolled covariates sum to 312 + 2*1000; the two missing
    # values are filled before the two immature outcomes are excluded.
    population_mean = 2312.0 / 18
    mature = [unit for unit in groups if unit not in {"u9", "u19"}]
    frame = Analysis.from_unit_summary(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "arm": groups[unit],
                    "revenue": revenue[unit],
                    "history_a": population_mean if unit in {"u0", "u10"} else tenure[unit],
                }
                for unit in mature
            ]
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="revenue", decision_method=_scalar("adjusted", "history_a"))],
    )
    try:
        (expected,) = frame.run()
        assert isinstance(expected, LiftEstimate)
        with Analysis.from_definitions("cov_test", path, con) as native:
            (native_row,) = native.run()
        artifact = _artifact_source(path, con)
        try:
            (artifact_row,) = run(artifact)
        finally:
            artifact.close()
        for row in (native_row, artifact_row):
            assert isinstance(row, LiftEstimate)
            assert row.require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert row.require_lift().log_se == pytest.approx(
                expected.require_lift().log_se, rel=1e-10
            )
    finally:
        frame.close()
        con.disconnect()


def _refusing_breakout_records():
    return [
        {
            **row,
            "segment": "unavailable" if index % 2 == 0 else "healthy",
            "history_a": None if index == 0 else row["history_a"],
            "uptake": float(row["arm"] == "treatment" and index % 3 != 0),
        }
        for index, row in enumerate(_records())
    ]


def test_encouragement_breakout_preserves_refused_scalar_decision_cells():
    from tests.test_cuped_route_evidence import _frame

    scalar = _scalar("adjusted", "history_a", missing="refuse")
    analysis = _frame(
        _refusing_breakout_records(),
        (scalar, Method.unadjusted()),
        panel=False,
        encouragement=True,
    )
    result = analysis.run_breakout()
    indexed = {(row.dimension_value, row.method, row.estimand): row for row in result}
    for estimand in ("itt", "late"):
        key = ("unavailable", "adjusted", estimand)
        assert key in indexed
        failed = indexed[key]
        assert failed.method_role == "decision"
        assert failed.lift is None
        assert failed.failure_code == "frame.validation.metric_covariate_column"
        assert failed.failure_context["covariate"] == "history_a"
        assert indexed[("unavailable", "naive", estimand)].require_lift() is not None
        assert indexed[("healthy", "adjusted", estimand)].require_lift() is not None
    assert result.metadata is not None
    failed_records = [
        cell
        for cell in result.metadata.cells
        if cell.cell.dimension_value == "unavailable" and cell.cell.method == "adjusted"
    ]
    assert {cell.cell.estimand for cell in failed_records} == {"itt", "late"}
    assert all(
        cell.failure is not None and cell.failure.code == "frame.validation.metric_covariate_column"
        for cell in failed_records
    )


@pytest.mark.parametrize("role", ["decision", "sensitivity"])
def test_breakout_scalar_refusal_preserves_requested_posterior_unavailability(role):
    from increment import Normal
    from tests.test_cuped_route_evidence import _frame

    scalar, naive = _scalar("adjusted", "history_a", missing="refuse"), Method.unadjusted()
    methods = (scalar, naive) if role == "decision" else (naive, scalar)
    result = _frame(_refusing_breakout_records(), methods, panel=False).run_breakout(
        prior=Normal(mu=0.0, sigma=0.1)
    )
    failed = next(
        row for row in result if row.dimension_value == "unavailable" and row.method == "adjusted"
    )
    assert failed.failure_code == "frame.validation.metric_covariate_column"
    assert failed.posterior_available is False
    assert failed.posterior_reason_code == failed.failure_code
    assert failed.posterior_reason_context == failed.failure_context
    assert result.metadata is not None
    record = next(
        cell
        for cell in result.metadata.cells
        if cell.cell.dimension_value == "unavailable" and cell.cell.method == "adjusted"
    )
    assert record.posterior is not None and record.posterior.available is False
    assert record.posterior.reason_code == failed.failure_code
    assert record.posterior.reason_context == failed.failure_context


def test_scalar_only_breakout_refusal_remains_in_bh_family():
    import warnings

    from increment import AnalysisPlan, MultiplicitySpec

    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(_refusing_breakout_records()),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=_scalar("adjusted", "history_a", missing="refuse"),
            )
        ],
        breakouts=["segment"],
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="bh")),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = analysis.run_breakout()
    assert not captured
    failed = next(row for row in result if row.dimension_value == "unavailable")
    assert failed.failure_code == "frame.validation.metric_covariate_column"
    assert failed.excluded == "estimation_failed"
    assert failed.discovery is False
    assert result.metadata is not None
    record = next(
        cell for cell in result.metadata.cells if cell.cell.dimension_value == "unavailable"
    )
    assert record.failure is not None and record.failure.code == failed.failure_code


@pytest.mark.parametrize(
    "override",
    [
        "run",
        "construction",
        "renamed_run",
        "renamed_construction",
        "matching_after_unadjusted",
        "matching_renamed",
    ],
)
def test_portable_moments_enforce_recorded_scalar_preparation(tmp_path, monkeypatch, override):
    import pyarrow.parquet as pq

    from increment.frame import from_unit_summary
    from increment.sources import MomentsSource

    original = _scalar("adjusted", "history_a", missing="zero")
    source = from_unit_summary(
        pa.Table.from_pylist(_records()),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="outcome", decision_method=original)],
    )
    destination = tmp_path / "prepared.parquet"
    source.export_moments(destination)
    payload = pq.read_table(destination).to_pylist()
    changed = _scalar(
        "adjusted",
        "history_b" if override.startswith("renamed_") else "history_a",
        missing="impute",
    )
    if override in {"matching_after_unadjusted", "matching_renamed"}:
        restored = Analysis.from_moments(
            payload,
            control="control",
            metrics=[MetricSpec(name="outcome", decision_method=Method.unadjusted())],
        )
        (expected,) = Analysis.from_moments(
            payload, metrics=[MetricSpec(name="outcome")], control="control"
        ).run()
        requested = (
            _scalar("adjusted", "history_b", missing="zero")
            if override == "matching_renamed"
            else original
        )
        (actual,) = restored.run(decision_method=requested, sensitivity_methods=())
        assert isinstance(actual, LiftEstimate) and isinstance(expected, LiftEstimate)
        assert actual.require_lift().value == pytest.approx(
            expected.require_lift().value, rel=1e-10
        )
        assert actual.require_lift().log_se == pytest.approx(
            expected.require_lift().log_se, rel=1e-10
        )
        return
    reads = []

    def unread(*args, **kwargs):
        reads.append("evidence")
        pytest.fail("changed preparation accessed fixed moments")

    restored = Analysis.from_moments(
        payload, metrics=[MetricSpec(name="outcome")], control="control"
    )
    monkeypatch.setattr(MomentsSource, "moments", unread)
    with pytest.raises(CodedError) as raised:
        if override in {"construction", "renamed_construction"}:
            Analysis.from_moments(
                payload,
                metrics=[MetricSpec(name="outcome", decision_method=changed)],
                control="control",
            )
        else:
            restored.run(decision_method=changed, sensitivity_methods=())
    assert raised.value.code == "readout.scalar_method_evidence"
    assert reads == []


def test_artifact_publication_deduplicates_shared_measure_covariate(tmp_path):
    import yaml

    from increment.readouts import run
    from increment.semantics.models import CupedCovariateBinding, Measure
    from tests.test_cuped_route_evidence import _artifact_source

    path, con, _ = _linear_binding_case(tmp_path)
    binding = CupedCovariateBinding(
        name="history",
        kind="continuous",
        missing="zero",
        measure=Measure(fact="purchase", aggregation="sum"),
    )
    first = Method.cuped(name="decision", covariates=binding)
    second = Method.cuped(
        name="sensitivity", covariates=binding.model_copy(update={"missing": "impute"})
    )
    definitions = yaml.safe_load(path.read_text())
    definitions["experiments"][0]["plan"]["primary"] = {
        "metric": "revenue",
        "decision_method": first.model_dump(mode="json", exclude_none=True),
        "sensitivity_methods": [second.model_dump(mode="json", exclude_none=True)],
    }
    path.write_text(yaml.safe_dump(definitions))
    try:
        source = _artifact_source(path, con)
        try:
            rows = run(source)
            assert {row.method for row in rows} == {"decision", "sensitivity"}
            assert all(row.require_lift() is not None for row in rows)
        finally:
            source.close()
    finally:
        con.disconnect()


@pytest.mark.parametrize("override", [False, True])
def test_sequential_subset_retains_unselected_declared_family_methods(override):
    from fractions import Fraction

    from increment import AnalysisPlan, Randomized
    from increment.semantics.models import InferenceSpec, PredeclaredAdjustment

    metrics = [
        MetricSpec(
            name=name,
            decision_method=_scalar(f"decision_{name}", "history_a"),
            sensitivity_methods=(_scalar(f"sensitivity_{name}", "history_a"),),
        )
        for name in ("outcome", "second")
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(
            [
                {**row, "second": 2 * row["outcome"] + index % 5}
                for index, row in enumerate(_records())
            ]
        ),
        unit="unit",
        group="arm",
        exposure_date="exposure",
        metrics=metrics,
        design=Randomized(
            control_group="control",
            allocation={"control": Fraction(1, 2), "treatment": Fraction(1, 2)},
        ),
        plan=AnalysisPlan(
            primary="outcome",
            secondaries=["second"],
            inference=InferenceSpec(
                kind="asymptotic_mean",
                adjustments={
                    name: PredeclaredAdjustment(coefficient=Fraction(1, 2), center=Fraction(4))
                    for name in ("outcome", "second")
                },
            ),
        ),
    )
    full = analysis.run()
    kwargs = (
        {
            "decision_method": _scalar("requested", "history_a"),
            "sensitivity_methods": (_scalar("requested_companion", "history_a"),),
        }
        if override
        else {}
    )
    subset = analysis.run(metrics=["outcome"], **kwargs)
    assert full.metadata is not None and subset.metadata is not None

    def methods_in_families(result, name):
        return {
            cell.method
            for family in result.metadata.scope.families
            for cell in family.members
            if cell.metric == name
        }

    assert methods_in_families(subset, "second") == methods_in_families(full, "second")
    assert methods_in_families(subset, "second") == {"decision_second"}
    assert {cell.cell.method for cell in full.metadata.cells if cell.cell.metric == "second"} == {
        "decision_second",
        "sensitivity_second",
    }
    assert {row.method for row in subset} == (
        {"requested", "requested_companion"}
        if override
        else {"decision_outcome", "sensitivity_outcome"}
    )


def test_native_scalar_snapshot_pins_separate_encouragement_uptake(tmp_path):
    from datetime import datetime

    import yaml

    from increment import Encouragement, UptakeSpec
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, tenure, groups, revenue = _scalar_native_case(tmp_path)
    activated = {f"u{index}" for index in range(10, 18)}
    con.create_table(
        "separate_uptake",
        obj=pa.Table.from_pylist(
            [
                {"user_id": unit, "ts": datetime(2025, 8, 2), "event": "activated"}
                for unit in sorted(activated)
            ]
        ),
    )
    definitions = yaml.safe_load(path.read_text())
    definitions["fact_sources"].append(
        {
            "name": "uptake",
            "sql": "SELECT * FROM separate_uptake",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "activated", "column": None}],
        }
    )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="activated"),
        exclusion_restriction={
            "acknowledged": True,
            "justification": "Assignment affects outcomes only through activation.",
        },
    )
    definitions["experiments"][0]["design"] = design.model_dump(
        mode="json", exclude_none=True, exclude={"min_first_stage_z", "control_group"}
    )
    definitions["experiments"][0]["plan"]["primary"]["sensitivity_methods"] = []
    path.write_text(yaml.safe_dump(definitions))
    frame = Analysis.from_unit_summary(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "arm": groups[unit],
                    "revenue": revenue[unit],
                    "history_a": tenure[unit],
                    "uptake": float(unit in activated),
                }
                for unit in groups
            ]
        ),
        unit="unit",
        group="arm",
        design=design,
        uptake="uptake",
        metrics=[MetricSpec(name="revenue", decision_method=_scalar("scalar_a", "history_a"))],
    )
    expected = {}
    for row in frame.run(estimands=("itt", "late")):
        assert isinstance(row, LiftEstimate)
        expected[row.estimand] = row.require_lift()
    try:
        with Analysis.from_definitions("cov_test", path, con) as native:
            actual = {}
            for row in native.run(estimands=("itt", "late")):
                assert isinstance(row, LiftEstimate)
                actual[row.estimand] = row.require_lift()
        assert actual.keys() == expected.keys() == {"itt", "late"}
        for estimand, lift in actual.items():
            assert lift.value == pytest.approx(expected[estimand].value, rel=1e-10)
            assert lift.log_se == pytest.approx(expected[estimand].log_se, rel=1e-10)
    finally:
        con.disconnect()


def test_native_single_metric_scalar_snapshot_retains_shared_horizon(tmp_path):
    from datetime import datetime

    import yaml

    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, tenure, groups, revenue = _scalar_native_case(tmp_path)
    con.raw_sql(
        "UPDATE cov_events SET segment = 'qualifying', "
        "tenure = 10 + CAST(substr(user_id, 2) AS INTEGER) "
        "WHERE event = 'purchase' AND ts < TIMESTAMP '2025-08-10'"
    )
    con.raw_sql(
        "UPDATE cov_events SET segment = 'freshness_only', revenue = 0.0 "
        "WHERE event = 'purchase' AND ts > TIMESTAMP '2025-08-10'"
    )
    con.raw_sql(
        "UPDATE cov_events SET ts = TIMESTAMP '2025-08-10' "
        "WHERE event = 'exposure' AND experiment_id IS NOT NULL "
        "AND user_id IN ('u9', 'u19')"
    )
    con.create_table(
        "coverage_events",
        obj=pa.Table.from_pylist(
            [
                {
                    "user_id": "outside_the_cohort",
                    "ts": datetime(2025, 8, 20),
                    "event": "observed",
                    "value": 1.0,
                }
            ]
        ),
    )
    definitions = yaml.safe_load(path.read_text())
    definitions["experiments"][0]["end"] = None
    definitions["metrics"][0]["filters"] = [
        {"property": "segment", "op": "equals", "values": ["qualifying"]}
    ]
    definitions["fact_sources"].append(
        {
            "name": "coverage",
            "sql": "SELECT * FROM coverage_events",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "observed", "column": "value"}],
        }
    )
    definitions["metrics"].append(
        {
            "name": "coverage",
            "type": "mean",
            "entity": "user_id",
            "fact": "observed",
            "aggregation": "sum",
            "window_days": 7,
        }
    )
    definitions["experiments"][0]["plan"]["primary"]["sensitivity_methods"] = []
    definitions["experiments"][0]["plan"]["secondaries"] = ["coverage"]
    path.write_text(yaml.safe_dump(definitions))
    frame = Analysis.from_unit_summary(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "arm": groups[unit],
                    "revenue": 0.0 if unit in {"u9", "u19"} else revenue[unit],
                    "history_a": tenure[unit],
                }
                for unit in groups
            ]
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="revenue", decision_method=_scalar("scalar_a", "history_a"))],
    )
    (expected,) = frame.run()
    try:
        with Analysis.from_definitions("cov_test", path, con) as native:
            (actual,) = native.run(metrics=["revenue"])
        assert isinstance(actual, LiftEstimate) and isinstance(expected, LiftEstimate)
        assert actual.require_lift().value == pytest.approx(
            expected.require_lift().value, rel=1e-10
        )
        assert actual.require_lift().log_se == pytest.approx(
            expected.require_lift().log_se, rel=1e-10
        )
    finally:
        con.disconnect()


@pytest.mark.parametrize("decision", ["scalar", "mixed", "naive"])
@pytest.mark.parametrize("companion", ["changed", "added"])
def test_concat_preserves_shared_decision_family_across_companion_snapshots(decision, companion):
    import json

    from increment.estimation.readout_types import ReadoutResults
    from tests.test_cuped_identity import _frame_readout

    first = _frame_readout(decision, "scalar" if companion == "changed" else None, panel=False)
    changed = _frame_readout(decision, "scalar", panel=False, companion_scale=2.0)
    assert first.metadata is not None and changed.metadata is not None
    assert first.metadata.scope.snapshot_id != changed.metadata.scope.snapshot_id
    assert {family.family_id for family in first.metadata.scope.families} == {
        family.family_id for family in changed.metadata.scope.families
    }
    combined = first.concat(changed)
    assert len(combined) == len(first) + len(changed)
    assert combined.metadata is not None
    assert len(combined.metadata.scope.by_source) == 2
    assert {family.source_snapshot_id for family in combined.metadata.scope.families} == set(
        combined.metadata.scope.by_source
    )
    assert {family.family_id for family in combined.metadata.scope.families} == {
        family.family_id for family in first.metadata.scope.families
    }
    restored = ReadoutResults.model_validate_json(combined.model_dump_json())
    assert restored.metadata == combined.metadata
    for field in ("members", "complete"):
        payload = json.loads(combined.model_dump_json())
        family = payload["metadata"]["scope"]["families"][1]
        removed = next(
            member for member in family["members"] if member["method_role"] == "decision"
        )
        family[field] = (
            [member for member in family["members"] if member != removed]
            if field == "members"
            else not family["complete"]
        )
        with pytest.raises(CodedError) as raised:
            ReadoutResults.model_validate_json(json.dumps(payload))
        assert raised.value.code == "readout.serialization.identity_mismatch"
