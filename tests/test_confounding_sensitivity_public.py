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
    confounding_sensitivity,
)
from increment.errors import CodedError, UnsupportedRequestError
from increment.results import ConfoundingSensitivity, LiftEstimate
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


def _analysis(df, *, trimmed=False, clustered=False):
    return Analysis.from_unit_summary(
        df,
        unit="unit",
        group="arm",
        metrics=[MetricSpec(name="converted", type="conversion")],
        design=_design(trimmed=trimmed),
        cluster="cluster" if clustered else None,
    )


def _assert_rr_scale(row: LiftEstimate, expected_rr):
    result = confounding_sensitivity(row, outcome="binary", threshold_risk_ratio=1.05)
    assert isinstance(result, ConfoundingSensitivity)
    lift = row.require_lift()
    assert lift.lb is not None and lift.ub is not None
    se_rr = (lift.ub - lift.lb) / (2 * 1.96)
    assert result.observed_risk_ratio == 1 + lift.value
    assert abs(result.observed_risk_ratio - expected_rr) < 4 * se_rr
    assert result.interval_level == lift.level
    return result


@pytest.mark.parametrize("clustered", [False, True], ids=["unit", "cluster"])
def test_iptw_and_aipw_rows_report_marginal_risk_ratio(clustered):
    analysis = _analysis(_frame(), clustered=clustered)
    try:
        rows = analysis.run(
            decision_method=Method(name="iptw"), sensitivity_methods=[Method(name="aipw")]
        )
        assert {row.method for row in rows} == {"iptw", "aipw"}
        for row in rows:
            assert isinstance(row, LiftEstimate)
            assert row.estimand == "ate"
            _assert_rr_scale(row, (0.30 + 0.45) / (0.08 + 0.20))
            if clustered:
                assert row.n_clusters == 2_000
    finally:
        analysis.close()


@pytest.mark.parametrize("method", ["iptw", "aipw"])
@pytest.mark.parametrize("scale", ["log", "absolute"])
def test_marginal_risk_ratio_check_rejects_other_scales(method, scale):
    analysis = _analysis(_frame())
    try:
        (row,) = analysis.run(decision_method=Method(name=method))
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
        analysis.close()


def test_trimmed_rows_are_accepted_with_population():
    analysis = _analysis(_frame(trimmed=True), trimmed=True)
    try:
        rows = analysis.run(
            decision_method=Method(name="iptw"), sensitivity_methods=[Method(name="aipw")]
        )
        assert {row.method for row in rows} == {"iptw", "aipw"}
        for row in rows:
            assert isinstance(row, LiftEstimate)
            assert row.estimand == "overlap_subpopulation_ate"
            result = _assert_rr_scale(row, 0.30 / 0.08)
            assert result.population == row.population is not None
    finally:
        analysis.close()


def test_three_arm_rows_are_bounded_over_all_arms():
    analysis = _analysis(_frame(three_arms=True))
    try:
        rows = analysis.run(
            decision_method=Method(name="iptw"), sensitivity_methods=[Method(name="aipw")]
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
        analysis.close()


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
            dumps[ingress] = {
                row.group_id: completed(
                    CONFOUNDING_SENSITIVITY,
                    ingress,
                    lambda row=row: confounding_sensitivity(row, outcome="binary"),
                ).model_dump()
                for row in rows
            }
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
    assert error.code == "source.moments.covariate_unavailable"

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
    analysis = Analysis.from_unit_summary(
        _frame(),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    try:
        (row,) = analysis.run()
        assert isinstance(row, LiftEstimate)
        with pytest.raises(UnsupportedRequestError) as excinfo:
            confounding_sensitivity(row, outcome="binary")
        assert excinfo.value.code == "sensitivity.confounding.estimand_unsupported"
    finally:
        analysis.close()


def test_dml_row_is_refused():
    analysis = _analysis(_frame())
    try:
        (row,) = analysis.run(decision_method=Method(name="dml"))
        assert isinstance(row, LiftEstimate)
        assert row.estimand == "plr_slope"
        with pytest.raises(UnsupportedRequestError) as excinfo:
            confounding_sensitivity(row, outcome="binary")
        assert excinfo.value.code == "sensitivity.confounding.estimand_unsupported"
    finally:
        analysis.close()
