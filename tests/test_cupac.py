from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import pytest

import increment
from increment.errors import InvalidRequestError
from increment.estimation.results import LiftEstimate
from tests.mc import nominal_band

_MU0 = 20.0
_TAU = 0.6
_N_HIST = 5000
_TRUE_RELATIVE_LIFT = _TAU / _MU0
_Scenario = Literal["useful", "useless", "shifted"]
_Metric = Literal["mean", "ratio"]
_MISSING = "frame.validation.metric_covariate_column"
_SCENARIOS: tuple[_Scenario, ...] = ("useful", "useless", "shifted")
_METRICS: tuple[_Metric, ...] = ("mean", "ratio")


def _basis(features: np.ndarray) -> np.ndarray:
    f1, f2 = features[:, 0], features[:, 1]
    return np.column_stack((np.ones(len(features)), f1, f2, f1**2, f1 * f2))


def _experiment(
    rng: np.random.Generator, n_per_arm: int, scenario: _Scenario, metric: _Metric
) -> pd.DataFrame:
    historical_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
    historical_features = historical_rng.normal(size=(_N_HIST, 3))
    f1h, f2h = historical_features[:, 0], historical_features[:, 1]
    historical_signal = 3 * f1h + 2 * f2h + 1.5 * (f1h**2 - 1)
    historical_outcome = _MU0 + historical_signal + historical_rng.normal(0, 2, _N_HIST)
    experiment_features = rng.normal(size=(2 * n_per_arm, 3))
    if scenario == "useless":
        history = np.column_stack((np.ones(_N_HIST), historical_features[:, 2]))
        score = np.column_stack((np.ones(2 * n_per_arm), experiment_features[:, 2]))
    else:
        history, score = _basis(historical_features), _basis(experiment_features)
    coefficients = np.linalg.lstsq(history, historical_outcome, rcond=None)[0]
    prediction = score @ coefficients

    f1, f2 = experiment_features[:, 0], experiment_features[:, 1]
    if scenario == "shifted":
        signal = 1.5 * f1 + 2 * f2 + 3.0 * f1 * f2
    else:
        signal = 3 * f1 + 2 * f2 + 1.5 * (f1**2 - 1)
    assignment = np.repeat(["control", "treatment"], n_per_arm)
    mean = _MU0 + signal + _TAU * (assignment == "treatment")
    if metric == "mean":
        revenue = mean + rng.normal(0, 2, 2 * n_per_arm)
        sessions = np.ones(2 * n_per_arm, dtype=int)
    else:
        sessions = 1 + rng.poisson(3, 2 * n_per_arm)
        revenue = sessions * mean + rng.normal(0, 2 * np.sqrt(sessions))
    return pd.DataFrame(
        {
            "user_id": np.arange(2 * n_per_arm),
            "variant": assignment,
            "revenue": revenue,
            "sessions": sessions,
            "prediction": prediction,
        }
    )


def _metric(metric: _Metric, *, refuse: bool = False) -> increment.MetricSpec:
    covariate = (
        increment.CupedCovariate(name="prediction", kind="continuous", missing="refuse")
        if refuse
        else "prediction"
    )
    cuped = increment.Method.cuped(covariates=covariate)
    if metric == "mean":
        return increment.MetricSpec(name="revenue", sensitivity_methods=(cuped,))
    return increment.MetricSpec(
        name="revenue_per_session",
        type="ratio",
        numerator="revenue",
        denominator="sessions",
        sensitivity_methods=(cuped,),
    )


def _readout(frame: pd.DataFrame, metric: _Metric, *, refuse: bool = False):
    experiment, reader = increment.from_frame(
        frame,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_metric(metric, refuse=refuse)],
    )
    rows: dict[str, LiftEstimate] = {}
    for row in increment.run(experiment, reader).readout():
        assert isinstance(row, LiftEstimate)
        rows[row.method] = row
    return rows


def _coverage_trial(i: int, *, metric: _Metric, scenario: _Scenario, n_per_arm: int):
    frame = _experiment(np.random.default_rng(20261012 + i), n_per_arm, scenario, metric)
    lift = _readout(frame, metric)["cuped"].require_lift()
    return lift.lb <= _TRUE_RELATIVE_LIFT <= lift.ub, lift.value - _TRUE_RELATIVE_LIFT


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("scenario", _SCENARIOS)
@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_cupac_coverage_and_bias(metric, scenario):
    reps = 1500
    covered, errors = zip(
        *(
            _coverage_trial(i, metric=metric, scenario=scenario, n_per_arm=1000)
            for i in range(reps)
        ),
        strict=True,
    )
    rate = sum(covered) / reps
    coverage_mcse = math.sqrt(rate * (1 - rate) / reps)
    mean_error = float(np.mean(errors))
    error_mcse = float(np.std(errors, ddof=1) / math.sqrt(reps))
    lo, hi = nominal_band(0.95, reps, k=3.0)
    assert lo <= rate <= hi
    assert abs(mean_error) <= 3 * error_mcse
    print(
        f"{metric}-{scenario}: coverage={rate:.6f}, mean_signed_error={mean_error:.8g}, "
        f"coverage_mcse={coverage_mcse:.8g}, error_mcse={error_mcse:.8g}"
    )


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_cupac_coverage_smoke(metric, scenario):
    reps = 40
    covered = [
        _coverage_trial(i, metric=metric, scenario=scenario, n_per_arm=200)[0] for i in range(reps)
    ]
    lo, _ = nominal_band(0.95, reps, k=3.0)
    assert sum(covered) / reps >= lo


@pytest.mark.parametrize("metric", _METRICS)
def test_cupac_precision_tracks_prediction_quality(metric):
    widths = {}
    for index, scenario in enumerate(_SCENARIOS):
        frame = _experiment(np.random.default_rng(731_002 + index), 20_000, scenario, metric)
        rows = _readout(frame, metric)
        adjusted, unadjusted = (rows[name].require_lift() for name in ("cuped", "unadjusted"))
        widths[scenario] = (adjusted.ub - adjusted.lb) / (unadjusted.ub - unadjusted.lb)
        if metric == "mean":
            by_arm = frame.groupby("variant")
            centered_prediction = frame["prediction"] - by_arm["prediction"].transform("mean")
            centered_revenue = frame["revenue"] - by_arm["revenue"].transform("mean")
            rho = np.corrcoef(centered_prediction, centered_revenue)[0, 1]
            assert widths[scenario] == pytest.approx(math.sqrt(1 - rho**2), rel=0.05)
    assert 0.97 <= widths["useless"] <= 1.03
    assert widths["useful"] < widths["shifted"] < widths["useless"]
    print(f"{metric} precision: width_ratios={widths}")


def test_cupac_panel_requires_unit_constant_prediction():
    frame = _experiment(np.random.default_rng(612_005), 20, "useful", "mean")
    panel = pd.concat(
        [
            frame.assign(date=pd.Timestamp("2026-01-01")),
            frame.assign(date=pd.Timestamp("2026-01-02"), prediction=frame["prediction"] + 1),
        ],
        ignore_index=True,
    )
    with pytest.raises(InvalidRequestError) as raised:
        experiment, reader = increment.from_frame(
            panel,
            shape="panel",
            unit="user_id",
            group="variant",
            date="date",
            control="control",
            metrics=[_metric("mean")],
        )
        increment.run(experiment, reader)
    assert raised.value.code == "frame.frame_panel.unit_covariate_varies"


@pytest.mark.parametrize("metric", _METRICS)
def test_cupac_missing_predictions_leave_unadjusted_decision_available(metric):
    frame = _experiment(np.random.default_rng(612_006), 100, "useful", metric)
    complete = _readout(frame, metric, refuse=True)
    missing = (frame["variant"] == "treatment") & (
        frame["prediction"] > frame["prediction"].median()
    )
    rows = _readout(frame.assign(prediction=frame["prediction"].mask(missing)), metric, refuse=True)

    decision = rows["unadjusted"]
    assert decision.method_role == "decision"
    assert decision.require_lift() == complete["unadjusted"].require_lift()
    sensitivity = rows["cuped"]
    assert sensitivity.method_role == "sensitivity"
    assert sensitivity.lift is None
    assert sensitivity.failure_code == _MISSING
