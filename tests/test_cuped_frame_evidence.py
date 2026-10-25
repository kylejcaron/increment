from __future__ import annotations

from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from increment import Analysis, Method, MetricSpec
from increment.errors import CodedError, IncrementWarning
from increment.estimation.results import LiftEstimate
from increment.evidence import (
    EvidenceProvider,
    MomentRows,
    MomentsRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)
from increment.frame import CupedCovariate, from_unit_panel, from_unit_summary
from increment.readouts import breakout, run
from tests._override_warnings import allow_declared_overrides


def _moments(source, metric, **kwargs) -> MomentRows:
    return cast(
        MomentRows,
        request_evidence(evidence_provider(source), MomentsRequest(metric, **kwargs)),
    )


def _records() -> list[dict[str, Any]]:
    return [
        {
            "unit": f"{arm}-{i}",
            "arm": arm,
            "day": 1,
            "exposure": 1,
            "segment": "all",
            "outcome": 5.0 + i % 13 + 0.2 * (i % 7) + 2.0 * (arm == "treatment"),
            "history": None if i % 9 == 0 else float(i % 7 + 1),
        }
        for arm in ("control", "treatment")
        for i in range(48)
    ]


def _scalar(
    name: str,
    *,
    column: str = "history",
    missing: Literal["zero", "impute", "refuse"] = "zero",
) -> Method:
    return Method.cuped(
        name=name,
        covariates=CupedCovariate(name=column, kind="continuous", missing=missing),
    )


def _source(table: pa.Table, specs: list[MetricSpec], *, panel: bool, breakouts=()):
    kwargs = {"date": "day"} if panel else {}
    factory = from_unit_panel if panel else from_unit_summary
    return factory(
        table,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        breakouts=breakouts,
        **kwargs,
    )


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("role", ["decision", "sensitivity"])
def test_adjusted_frame_rechecks_every_override_column(panel, role, monkeypatch):
    with pytest.warns(IncrementWarning):
        source = _source(
            pa.Table.from_pylist(_records()),
            [MetricSpec(name="outcome", decision_method=_scalar("declared"))],
            panel=panel,
        )
    override = _scalar("override", column="nope")
    monkeypatch.setattr(
        source,
        "moments_for_method",
        lambda *args, **kwargs: pytest.fail("preflight must precede scalar acquisition"),
    )
    with pytest.raises(CodedError) as raised:
        if role == "decision":
            run(source, decision_method=override)
        else:
            run(source, sensitivity_methods=(override,))
    assert raised.value.code == "source.frame.metric_missing"


@pytest.mark.parametrize("panel", [False, True])
def test_method_specific_breakout_request_refuses_instead_of_returning_totals(panel):
    source = _source(
        pa.Table.from_pylist(_records()),
        [MetricSpec(name="outcome")],
        panel=panel,
        breakouts=("segment",),
    )
    with pytest.raises(CodedError) as raised:
        source.moments_for_method(
            source.context.metrics[0],
            _scalar("custom"),
            by=("segment",),
        )
    assert raised.value.code == "evidence.breakout_unavailable"


@pytest.mark.parametrize("by", [(), ("segment",)])
def test_panel_covariate_missing_policy_is_per_metric(by, tmp_path):
    specs = [
        MetricSpec(
            name=f"outcome_{missing}",
            value_column="outcome",
            decision_method=_scalar(missing, missing=missing),
        )
        for missing in ("impute", "zero")
    ]
    table = pa.Table.from_pylist(_records())
    with pytest.warns(IncrementWarning):
        joint = _source(table, specs, panel=True, breakouts=("segment",))
    joint_results = breakout(joint, "segment") if by else run(joint)
    if not by:
        joint.export_moments(tmp_path / "joint.parquet")
        joint_export = pq.read_table(tmp_path / "joint.parquet").to_pylist()
    for spec in specs:
        with pytest.warns(IncrementWarning):
            alone = _source(table, [spec], panel=True, breakouts=("segment",))
        joint_metric = next(metric for metric in joint.context.metrics if metric.name == spec.name)
        alone_metric = alone.context.metrics[0]
        observed = {row["group_id"]: row for row in _moments(joint, joint_metric, by=by)}
        expected = {row["group_id"]: row for row in _moments(alone, alone_metric, by=by)}
        assert observed.keys() == expected.keys()
        for arm, row in expected.items():
            for key in ("n", "ref_x", "cx1", "cx2", "cxy"):
                assert observed[arm][key] == pytest.approx(row[key], rel=1e-10)
        alone_results = breakout(alone, "segment") if by else run(alone)
        actual_fit = next(row for row in joint_results if row.metric == spec.name).require_lift()
        expected_fit = alone_results[0].require_lift()
        assert actual_fit.value == pytest.approx(expected_fit.value, rel=1e-10)
        assert actual_fit.log_se == pytest.approx(expected_fit.log_se, rel=1e-10)
        if not by:
            alone.export_moments(tmp_path / f"{spec.name}.parquet")
            exported = pq.read_table(tmp_path / f"{spec.name}.parquet").to_pylist()
            actual_rows = {
                row["group_id"]: row for row in joint_export if row["metric"] == spec.name
            }
            for row in exported:
                for key in ("n", "ref_x", "cx1", "cx2", "cxy"):
                    assert actual_rows[row["group_id"]][key] == pytest.approx(row[key], rel=1e-10)


def test_summary_breakout_imputes_before_outcome_deletion():
    table = pa.table(
        {
            "unit": ["a", "b", "c", "d"],
            "arm": ["control", "control", "treatment", "treatment"],
            "outcome": [1.0, None, 3.0, 4.0],
            "history": [0.0, 100.0, None, 10.0],
            "segment": ["all"] * 4,
        }
    )
    with pytest.warns(IncrementWarning):
        source = _source(
            table,
            [
                MetricSpec(
                    name="outcome",
                    missing="drop",
                    decision_method=_scalar("scalar", missing="impute"),
                )
            ],
            panel=False,
            breakouts=("segment",),
        )
    metric = source.context.metrics[0]
    headline = {row["group_id"]: row for row in _moments(source, metric)}
    breakout = {row["group_id"]: row for row in _moments(source, metric, by=("segment",))}
    expected_mean = ((0.0 + 100.0 + 10.0) / 3.0 + 10.0) / 2.0
    for observed in (headline, breakout):
        row = observed["treatment"]
        assert row["ref_x"] + row["cx1"] / row["n"] == pytest.approx(expected_mean)


def test_panel_scalar_fixed_caps_are_applied_once():
    table = pa.Table.from_pylist(
        [{**row, "history": float(i % 7)} for i, row in enumerate(_records())]
    )
    method = _scalar("scalar")
    spec = MetricSpec(
        name="outcome",
        decision_method=method,
        winsorization={"lower_value": 6.0, "upper_value": 14.0},
    )
    panel = _source(table, [spec], panel=True)
    summary = _source(table, [spec], panel=False)
    panel_rows = {
        row["group_id"]: row for row in _moments(panel, panel.context.metrics[0], method=method)
    }
    summary_rows = {
        row["group_id"]: row for row in _moments(summary, summary.context.metrics[0], method=method)
    }
    assert sum(row["winsor_n_lower"] + row["winsor_n_upper"] for row in summary_rows.values()) > 0
    for arm, expected in summary_rows.items():
        for key in ("winsor_n_lower", "winsor_n_upper", "ref_y", "cy1", "cy2", "cxy"):
            assert panel_rows[arm][key] == pytest.approx(expected[key], rel=1e-10)
    panel_result, summary_result = run(panel)[0], run(summary)[0]
    assert isinstance(panel_result, LiftEstimate) and isinstance(summary_result, LiftEstimate)
    assert panel_result.require_lift().value == pytest.approx(
        summary_result.require_lift().value, rel=1e-10
    )
    assert panel_result.require_lift().log_se == pytest.approx(
        summary_result.require_lift().log_se, rel=1e-10
    )


@pytest.mark.parametrize("metric_type", ["mean", "retention"])
def test_panel_scalar_override_preserves_unit_reduction_refusal(metric_type):
    rows = [{**row, "history": float(i % 7)} for i, row in enumerate(_records())]
    rows += [{**row, "day": 3, "outcome": 1000.0} for row in rows]
    spec = (
        MetricSpec(name="outcome", type="retention", threshold_days=2)
        if metric_type == "retention"
        else MetricSpec(name="outcome", window_days=2)
    )
    source = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposure",
        observation_end=4,
        control="control",
        metrics=[spec],
    )
    with pytest.raises(CodedError) as ordinary:
        request_evidence(evidence_provider(source), UnitRowsRequest(source.context.metrics[0]))
    with pytest.raises(CodedError) as adjusted:
        run(source, decision_method=_scalar("scalar"))
    assert adjusted.value.code == ordinary.value.code == "source.frame.unit_frame_panel"


def test_sequential_frame_scalar_sensitivity_has_coded_policy_refusal():
    from increment import AnalysisPlan
    from increment.semantics.design import Randomized
    from increment.semantics.models import InferenceSpec

    spec = MetricSpec(name="outcome", sensitivity_methods=(_scalar("scalar"),))
    plan = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean"))
    rows = [{**row, "history": float(i % 7)} for i, row in enumerate(_records())]
    with pytest.raises(CodedError) as raised:
        Analysis.from_unit_panel(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposure",
            metrics=[spec],
            plan=plan,
            design=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
        )
    assert raised.value.code == "sequential.source.invalid"


def test_mixed_breakout_collision_mapping_matches_each_standalone():
    rows = [
        {
            **row,
            "outcome": 10.0 + i % 13 + (row["arm"] == "treatment"),
            "y": float(i % 7),
            "__covariate_y": float((i * 5) % 11),
        }
        for i, row in enumerate(_records())
    ]
    methods = tuple(
        Method.cuped_multicovariate(
            name=name, covariates=CupedCovariate(name=column, kind="continuous")
        )
        for name, column in (("first", "y"), ("second", "__covariate_y"))
    )

    def estimates(selected):
        analysis = Analysis.from_unit_panel(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="arm",
            date="day",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome", decision_method=selected[0], sensitivity_methods=selected[1:]
                )
            ],
            breakouts=["segment"],
        )
        return {(row.dimension_value, row.method): row for row in analysis.run_breakout()}

    joint = estimates(methods)
    for method in methods:
        for key, expected in estimates((method,)).items():
            assert joint[key].require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert joint[key].require_lift().log_se == pytest.approx(
                expected.require_lift().log_se, rel=1e-10
            )


@pytest.mark.integration
@pytest.mark.parametrize("ingress", ["summary", "panel", "native", "artifact"])
@pytest.mark.parametrize("role", ["decision", "sensitivity"])
@pytest.mark.parametrize("missing", ["refuse", "impute"])
@pytest.mark.slow
@allow_declared_overrides
def test_scalar_missing_failure_is_local_to_its_method(tmp_path, ingress, role, missing):
    treatment_group = "treatment"
    expected_code = (
        "frame.validation.metric_covariate_column"
        if missing == "refuse"
        else "frame.validation.metric_covariate_impute_all_null"
    )
    if ingress in ("summary", "panel"):
        rows = [
            {
                **row,
                "history": None if missing == "impute" or i == 0 else float(i % 7),
                "other_history": float((i * 5) % 11),
            }
            for i, row in enumerate(_records())
        ]
        bad = _scalar("bad", missing=missing)
        good = _scalar("good", column="other_history")
        mixed = Method.cuped_multicovariate(
            name="mixed", covariates=CupedCovariate(name="other_history", kind="continuous")
        )
        methods = (
            (bad, Method.unadjusted(), good, mixed)
            if role == "decision"
            else (Method.unadjusted(), bad, good, mixed)
        )
        specs = [
            MetricSpec(name="outcome", decision_method=methods[0], sensitivity_methods=methods[1:]),
            MetricSpec(name="other", value_column="outcome"),
        ]
        analysis = (
            Analysis.from_unit_panel(
                pa.Table.from_pylist(rows),
                unit="unit",
                group="arm",
                date="day",
                control="control",
                metrics=specs,
            )
            if ingress == "panel"
            else Analysis.from_unit_summary(
                pa.Table.from_pylist(rows),
                unit="unit",
                group="arm",
                control="control",
                metrics=specs,
            )
        )
        metric_name = "outcome"
    else:
        import json

        import yaml

        from tests.test_artifact_covariates import _publish_and_reopen
        from tests.test_cuped_covariate_sources import _scalar_native_case

        path, con, declared, _tenure, _groups, _revenue = _scalar_native_case(tmp_path)
        scalar, good = declared
        assert scalar.covariates is not None
        bad = Method.cuped(
            name="bad",
            covariates=tuple(
                covariate.model_copy(update={"missing": missing}) for covariate in scalar.covariates
            ),
        )
        mixed = Method.cuped_multicovariate(name="mixed", covariates=good.covariates)
        methods = (
            (bad, Method.unadjusted(), good, mixed)
            if role == "decision"
            else (Method.unadjusted(), bad, good, mixed)
        )
        definitions = yaml.safe_load(path.read_text())
        definitions["experiments"][0]["plan"]["primary"] = {
            "metric": "revenue",
            "decision_method": methods[0].model_dump(mode="json", exclude_none=True),
            "sensitivity_methods": [
                method.model_dump(mode="json", exclude_none=True) for method in methods[1:]
            ],
        }
        path.write_text(json.dumps(definitions))
        con.raw_sql(
            "UPDATE cov_events SET tenure = NULL"
            + (" WHERE user_id = 'u1'" if missing == "refuse" else "")
        )
        if ingress == "artifact":
            analysis, _manifest = _publish_and_reopen(
                path, con, request_covariate=False, full_catalog=True
            )
        else:
            analysis = Analysis.from_definitions("cov_test", path, con)
        metric_name = "revenue"
        treatment_group = next(arm for arm in _groups.values() if arm != "control")
    result = analysis.run()
    by_method = {
        row.method: row
        for row in result
        if isinstance(row, LiftEstimate) and row.metric == metric_name
    }
    assert set(by_method) == {method.name for method in methods}
    failed = by_method["bad"]
    assert failed.lift is None
    assert failed.failure_code == expected_code
    assert failed.method_role == role
    assert failed.metric == metric_name and failed.group_id == treatment_group
    assert result.metadata is not None
    record = next(
        item
        for item in result.metadata.cells
        if item.cell.metric == metric_name and item.cell.method == "bad"
    )
    assert record.failure is not None and record.failure.code == expected_code
    assert record.sampling is not None and record.sampling.available is False
    for method in methods:
        if method.name != "bad":
            assert by_method[method.name].lift is not None
            assert by_method[method.name].failure_code is None
            (standalone,) = analysis.run(
                metrics=[metric_name], decision_method=method, sensitivity_methods=()
            )
            assert isinstance(standalone, LiftEstimate)
            assert by_method[method.name].require_lift().value == pytest.approx(
                standalone.require_lift().value, rel=1e-10
            )
            assert by_method[method.name].require_lift().log_se == pytest.approx(
                standalone.require_lift().log_se, rel=1e-10
            )
    if ingress in ("summary", "panel"):
        assert any(
            isinstance(row, LiftEstimate) and row.metric == "other" and row.lift is not None
            for row in result
        )


@pytest.mark.parametrize("encouragement", [False, True])
@pytest.mark.parametrize("role", ["decision", "sensitivity"])
def test_scalar_missing_cell_retains_posterior_unavailability(encouragement, role):
    from increment.estimation.inference import Normal
    from increment.semantics.design import Encouragement, Randomized, UptakeSpec

    rows = [
        {**row, "uptake": int(row["arm"] == "treatment" and i % 4 != 0)}
        for i, row in enumerate(_records())
    ]
    bad, naive = _scalar("bad", missing="refuse"), Method.unadjusted()
    methods = (bad, naive) if role == "decision" else (naive, bad)
    design = (
        Encouragement(control_group="control", uptake=UptakeSpec(fact="uptake"))
        if encouragement
        else Randomized(control_group="control")
    )
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        design=design,
        uptake="uptake" if encouragement else None,
        metrics=[
            MetricSpec(
                name="outcome",
                preferred_direction="decrease",
                decision_method=methods[0],
                sensitivity_methods=methods[1:],
            )
        ],
    )
    result = analysis.run(prior=Normal(mu=0.0, sigma=0.1), estimands=("itt",))
    failed = next(row for row in result if row.method == "bad")
    assert isinstance(failed, LiftEstimate)
    assert failed.preferred_direction == "decrease"
    assert failed.lift is None and failed.method_role == role
    assert failed.posterior_available is False
    assert failed.posterior_reason_code in {
        "frame.validation.metric_covariate_column",
        "estimation.results.lift.t_posterior_sufficient_statistics",
    }


def test_override_preflight_uses_public_covariate_inventory():
    with pytest.warns(IncrementWarning):
        source = _source(
            pa.Table.from_pylist(_records()),
            [MetricSpec(name="outcome", decision_method=_scalar("declared"))],
            panel=False,
        )

    class PublicSource(EvidenceProvider):
        evidence = source.evidence
        context = source.context
        capabilities = source.capabilities
        shape = source.shape
        available_covariates = source.available_covariates
        operations = source.operations
        breakouts = source.breakouts
        unit_frame = source.unit_frame
        unit_counts = source.unit_counts
        cluster_counts = source.cluster_counts
        compliance_dates = source.compliance_dates
        compliance_summary = source.compliance_summary
        sql = source.sql
        close = source.close

        def moments(self, *args, **kwargs):
            pytest.fail("preflight must precede outcome acquisition")

        def moments_for_method(self, *args, **kwargs):
            pytest.fail("preflight must precede scalar acquisition")

    with pytest.raises(CodedError) as raised:
        run(PublicSource(), decision_method=_scalar("override", column="nope"))
    assert raised.value.code == "facade.analysis_config.cuped_linear_covariates"
