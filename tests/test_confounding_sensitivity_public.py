"""Public conversion-risk-ratio sensitivity and matched-ingress contracts."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from increment import (
    AdjustmentSet,
    Analysis,
    IdentificationGate,
    Method,
    MetricSpec,
    Observational,
    from_frame,
)
from increment.errors import CodedError, UnsupportedRequestError
from increment.estimators import confounding_sensitivity
from increment.results import ConfoundingSensitivity, LiftEstimate
from tests.analysis_factory import view_readout
from tests.capability_consumers_scalar import CONFOUNDING_SENSITIVITY, completed, refused
from tests.parity_harness.cases import (
    _close_parity_analysis,
    _observational_aipw_conversion_case,
    _observational_conversion_case,
)
from tests.parity_harness.comparison import nested_close

_RISKS = np.array([[0.08, 0.20], [0.30, 0.45], [0.06, 0.16]])


def _frame(*, trimmed=False, three_arms=False):
    rng = np.random.default_rng(20261013)
    n = 40_000
    segment = rng.integers(0, 2, size=n)
    if three_arms:
        propensities = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        cumulative = np.cumsum(propensities[segment], axis=1)
        arm = (rng.random(n)[:, None] > cumulative[:, :-1]).sum(axis=1)
        labels = np.array(["control", "t1", "t2"])
    else:
        propensity = np.where(segment == 0, 0.3, 0.995 if trimmed else 0.7)
        arm = (rng.random(n) < propensity).astype(int)
        labels = np.array(["control", "treatment"])
    converted = (rng.random(n) < _RISKS[arm, segment]).astype(int)
    return pl.DataFrame(
        {
            "unit": np.arange(n),
            "arm": labels[arm],
            "segment": segment,
            "converted": converted,
            "cluster": np.arange(n) // 20,
        }
    )


def _design(*, trimmed=False):
    return Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("segment",)),
        gate=IdentificationGate(overlap="trim" if trimmed else "refuse"),
    )


def _analysis(df, *, trimmed=False, clustered=False, metric_type="conversion"):
    return from_frame(
        df,
        shape="summary",
        unit="unit",
        group="arm",
        metrics=[MetricSpec(name="converted", type=metric_type)],
        design=_design(trimmed=trimmed),
        cluster="cluster" if clustered else None,
    )


def _assert_rr_scale(row: LiftEstimate, expected_rr):
    assert row.metric_type == "conversion"
    result = confounding_sensitivity(row, threshold_risk_ratio=1.05)
    assert isinstance(result, ConfoundingSensitivity)
    assert result.metric_type == "conversion"
    lift = row.require_lift()
    assert lift.lb is not None and lift.ub is not None
    se_rr = (lift.ub - lift.lb) / (2 * 1.96)
    assert result.observed_risk_ratio == 1 + lift.value
    assert abs(result.observed_risk_ratio - expected_rr) < 4 * se_rr
    assert result.interval_level == lift.level
    return result


@pytest.mark.parametrize("clustered", [False, True], ids=["unit", "cluster"])
def test_iptw_and_aipw_rows_report_marginal_risk_ratio(clustered):
    experiment, resolver = _analysis(_frame(), clustered=clustered)
    try:
        rows = view_readout(
            experiment,
            resolver,
            decision_method=Method(name="iptw"),
            sensitivity_methods=[Method(name="aipw")],
        )
        assert {row.method for row in rows} == {"iptw", "aipw"}
        for row in rows:
            assert isinstance(row, LiftEstimate)
            assert row.estimand == "ate"
            _assert_rr_scale(row, (0.30 + 0.45) / (0.08 + 0.20))
            if clustered:
                assert row.n_clusters == 2_000
    finally:
        resolver.close()


@pytest.mark.parametrize("method", ["iptw", "aipw"])
@pytest.mark.parametrize("scale", ["log", "absolute"])
def test_marginal_risk_ratio_check_rejects_other_scales(method, scale):
    experiment, resolver = _analysis(_frame())
    try:
        (row,) = view_readout(experiment, resolver, decision_method=Method(name=method))
        assert isinstance(row, LiftEstimate)
        lift = row.require_lift()
        assert lift.lb is not None and lift.ub is not None
        if scale == "log":
            transformed = {
                "value": math.log1p(lift.value),
                "lb": math.log1p(lift.lb),
                "ub": math.log1p(lift.ub),
            }
        else:
            assert row.abs_diff is not None
            assert row.abs_lb is not None and row.abs_ub is not None
            transformed = {"value": row.abs_diff, "lb": row.abs_lb, "ub": row.abs_ub}
        mutant = row.model_copy(update={"lift": lift.model_copy(update=transformed)})
        with pytest.raises(AssertionError):
            _assert_rr_scale(mutant, _RISKS[1].mean() / _RISKS[0].mean())
    finally:
        resolver.close()


def test_trimmed_rows_are_accepted_with_population():
    experiment, resolver = _analysis(_frame(trimmed=True), trimmed=True)
    try:
        rows = view_readout(
            experiment,
            resolver,
            decision_method=Method(name="iptw"),
            sensitivity_methods=[Method(name="aipw")],
        )
        assert {row.method for row in rows} == {"iptw", "aipw"}
        for row in rows:
            assert isinstance(row, LiftEstimate)
            assert row.estimand == "overlap_subpopulation_ate"
            result = _assert_rr_scale(row, 0.30 / 0.08)
            assert result.population == row.population is not None
    finally:
        resolver.close()


def test_three_arm_rows_are_bounded_over_all_arms():
    experiment, resolver = _analysis(_frame(three_arms=True))
    try:
        rows = view_readout(
            experiment,
            resolver,
            decision_method=Method(name="iptw"),
            sensitivity_methods=[Method(name="aipw")],
        )
        assert {(row.group_id, row.method) for row in rows} == {
            (arm, method) for arm in ("t1", "t2") for method in ("iptw", "aipw")
        }
        expected = {"t1": (0.30 + 0.45) / (0.08 + 0.20), "t2": (0.06 + 0.16) / (0.08 + 0.20)}
        for row in rows:
            assert isinstance(row, LiftEstimate)
            assert row.estimand == "ate"
            _assert_rr_scale(row, expected[row.group_id])
    finally:
        resolver.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ["iptw", "aipw"])
def test_cross_ingress_sensitivity_parity(method):
    case = (
        _observational_conversion_case()
        if method == "iptw"
        else _observational_aipw_conversion_case()
    )
    dumps = {}
    for ingress in (
        "from_unit_summary",
        "from_unit_panel",
        "from_definitions",
        "from_unit_day_artifact",
    ):
        analysis = completed(CONFOUNDING_SENSITIVITY, ingress, case.build[ingress])
        try:
            rows: list[LiftEstimate] = []
            for row in analysis.run():
                assert isinstance(row, LiftEstimate)
                rows.append(row)
            assert {row.group_id for row in rows} == {"treat_a", "treat_b"}
            dumps[ingress] = {}
            for row in rows:
                result = completed(
                    CONFOUNDING_SENSITIVITY,
                    ingress,
                    lambda row=row: confounding_sensitivity(row),
                )
                assert result.analysis_population == row.analysis_population
                assert result.source_snapshot_id == row.source_snapshot_id
                dumps[ingress][row.group_id] = result.model_dump(exclude={"source_snapshot_id"})
        finally:
            _close_parity_analysis(analysis)
    for ingress, actual in dumps.items():
        assert nested_close(dumps["from_unit_summary"], actual), ingress

    def moments_request():
        analysis = case.build["from_moments"]()
        try:
            return analysis.run()
        finally:
            _close_parity_analysis(analysis)

    error = refused(CONFOUNDING_SENSITIVITY, "from_moments", moments_request)
    assert isinstance(error, CodedError)
    assert error.code == "evidence.covariate_unavailable"

    from tests.test_frame_switchback import _assignment, _rows

    error = refused(
        CONFOUNDING_SENSITIVITY,
        "from_switchback_panel",
        lambda: Analysis.from_switchback_panel(
            pl.DataFrame(_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_design(),
            assignment=_assignment(),
        ),
    )
    assert isinstance(error, CodedError)
    assert error.code == "source.frame.switchback.identification"


def test_randomized_row_is_refused():
    experiment, resolver = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    try:
        (row,) = view_readout(experiment, resolver)
        assert isinstance(row, LiftEstimate)
        with pytest.raises(UnsupportedRequestError) as excinfo:
            confounding_sensitivity(row)
        assert excinfo.value.code == "sensitivity.confounding.estimand_unsupported"
    finally:
        resolver.close()


def test_dml_row_is_refused():
    experiment, resolver = _analysis(_frame())
    try:
        (row,) = view_readout(experiment, resolver, decision_method=Method(name="dml"))
        assert isinstance(row, LiftEstimate)
        assert row.estimand == "plr_slope"
        with pytest.raises(UnsupportedRequestError) as excinfo:
            confounding_sensitivity(row)
        assert excinfo.value.code == "sensitivity.confounding.estimand_unsupported"
    finally:
        resolver.close()


def test_mean_metric_row_is_refused_from_its_metric_type():
    experiment, resolver = _analysis(_frame(), metric_type="mean")
    try:
        (row,) = view_readout(experiment, resolver, decision_method=Method(name="iptw"))
        assert isinstance(row, LiftEstimate)
        assert row.estimand == "ate"
        assert row.metric_type == "mean"
        with pytest.raises(UnsupportedRequestError) as excinfo:
            confounding_sensitivity(row)
        assert excinfo.value.code == "sensitivity.confounding.outcome_unsupported"
        assert excinfo.value.context["metric_type"] == "mean"
    finally:
        resolver.close()


def test_sensitivity_report_row_round_trips_as_report_only_cell():
    import json

    from increment.result_envelope import ResultCell

    experiment, resolver = _analysis(_frame(), clustered=True)
    try:
        (lift,) = view_readout(experiment, resolver)
        report = confounding_sensitivity(lift, threshold_risk_ratio=1.05)
        cell = report.as_cell()
        assert cell.row_kind == "sensitivity"
        payload = json.loads(json.dumps(cell.to_payload()))
        restored = ResultCell.from_payload(payload)
        assert restored.supports["sampling"] == frozenset()
        assert restored.row_payload is not None
        assert restored.row_payload["data"]["interval_unavailable_reason"] == (
            report.interval_unavailable_reason
        )
        assert ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0].row_payload == cell.row_payload
    finally:
        resolver.close()


@pytest.mark.integration
def test_conversion_family_confounding_sensitivity_keeps_fcr_identity():
    from increment import AnalysisPlan

    rng = np.random.default_rng(52)
    n = 12_000
    segment = rng.integers(0, 2, size=n)
    treatment = rng.random(n) < np.where(segment == 0, 0.35, 0.65)
    arm = np.where(treatment, "treatment", "control")
    primary = rng.random(n) < np.where(
        treatment, np.where(segment == 0, 0.38, 0.52), np.where(segment == 0, 0.10, 0.20)
    )
    secondary = rng.random(n) < np.where(
        treatment, np.where(segment == 0, 0.45, 0.60), np.where(segment == 0, 0.08, 0.18)
    )
    experiment, resolver = from_frame(
        pl.DataFrame(
            {
                "unit": np.arange(n),
                "arm": arm,
                "segment": segment,
                "primary": primary.astype(int),
                "secondary": secondary.astype(int),
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        metrics=[
            MetricSpec(name="primary", type="conversion"),
            MetricSpec(name="secondary", type="conversion"),
        ],
        design=_design(),
        plan=AnalysisPlan(primary="primary", secondaries=["secondary"], q=0.2),
    )
    decision_method = Method(name="iptw")
    sensitivity_methods = [Method(name="aipw"), Method(name="unadjusted")]
    rows = view_readout(
        experiment,
        resolver,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
    )
    assert rows.metadata is not None
    assert rows.metadata.coherence == "captured"
    secondary_rows = [row for row in rows if row.metric == "secondary"]
    assert {(row.method, row.method_role) for row in secondary_rows} == {
        ("iptw", "decision"),
        ("aipw", "sensitivity"),
        ("unadjusted", "sensitivity"),
    }
    selected = next(row for row in secondary_rows if row.method == "iptw")
    assert isinstance(selected, LiftEstimate)
    assert selected.discovery is True
    assert selected.family_axes == ("metric", "arm")
    assert selected.family_q == pytest.approx(0.2)
    assert selected.require_lift().lb is not None
    assert selected.require_lift().ub is not None
    result = confounding_sensitivity(selected)
    lift = selected.require_lift()
    assert result.observed_risk_ratio == pytest.approx(1 + lift.value)
    assert result.interval_level == lift.level
    assert result.population == selected.population
    assert result.source_snapshot_id == selected.source_snapshot_id
    assert result.bias_factor_point == pytest.approx(math.sqrt(result.observed_risk_ratio))
    assert result.bias_factor_interval is not None
    assert result.assumptions
    assert result.robustness_not_identification is True
    resolver.close()
