from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from increment import Analysis
from increment.errors import InvalidRequestError
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.semantics.models import MeanMetric
from tests.estimation.test_cuped import _summary_df
from tests.mc import nominal_band

MU0 = 20.0
TAU = 0.6
N_HIST = 5000
_TRUE_RELATIVE_LIFT = TAU / MU0
_SCENARIOS: tuple[Literal["useful", "useless", "shifted"], ...] = (
    "useful",
    "useless",
    "shifted",
)
_METRICS: tuple[Literal["mean", "ratio"], ...] = ("mean", "ratio")


def _basis(f: np.ndarray) -> np.ndarray:
    f1, f2 = f[:, 0], f[:, 1]
    return np.column_stack((np.ones(len(f)), f1, f2, f1**2, f1 * f2))


def _fit_frozen(features_hist: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(features_hist, y_hist, rcond=None)[0]


def _experiment(
    rng: np.random.Generator,
    n_per_arm: int,
    scenario: Literal["useful", "useless", "shifted"],
    metric: Literal["mean", "ratio"],
) -> pd.DataFrame:
    # Historical and experiment streams are separately seeded by each caller.
    hist_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
    hist_features = hist_rng.normal(size=(N_HIST, 3))
    f1h, f2h = hist_features[:, 0], hist_features[:, 1]
    historical_signal = 3 * f1h + 2 * f2h + 1.5 * (f1h**2 - 1)
    historical_outcome = MU0 + historical_signal + hist_rng.normal(0, 2, N_HIST)
    if scenario == "useless":
        coefficients = _fit_frozen(
            np.column_stack((np.ones(N_HIST), hist_features[:, 2])), historical_outcome
        )
        experiment_features = rng.normal(size=(2 * n_per_arm, 3))
        prediction = (
            np.column_stack((np.ones(2 * n_per_arm), experiment_features[:, 2])) @ coefficients
        )
    else:
        coefficients = _fit_frozen(_basis(hist_features[:, :2]), historical_outcome)
        experiment_features = rng.normal(size=(2 * n_per_arm, 3))
        prediction = _basis(experiment_features[:, :2]) @ coefficients

    f1, f2 = experiment_features[:, 0], experiment_features[:, 1]
    if scenario == "shifted":
        signal = 1.5 * f1 + 2 * f2 + 3.0 * f1 * f2
    else:
        signal = 3 * f1 + 2 * f2 + 1.5 * (f1**2 - 1)
    assignment = np.repeat(["control", "treatment"], n_per_arm)
    treatment = (assignment == "treatment").astype(float)
    mean = MU0 + signal + TAU * treatment
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


def _analysis(frame: pd.DataFrame, metric: Literal["mean", "ratio"]) -> Analysis:
    spec = (
        MetricSpec(
            name="revenue",
            type="mean",
            value_column="revenue",
            covariate="prediction",
            covariate_missing="error",
        )
        if metric == "mean"
        else MetricSpec(
            name="revenue_per_session",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            covariate="prediction",
            covariate_missing="error",
        )
    )
    return Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[spec],
    )


def _summary_lifts(frame: pd.DataFrame):
    arms = []
    for arm, rows in frame.groupby("variant", sort=False):
        y = rows["revenue"].to_numpy()
        x = rows["prediction"].to_numpy()
        arms.append(
            ArmStats.from_raw_sums(
                study_id="cupac",
                metric="revenue",
                group_id=str(arm),
                n=len(rows),
                sum_y=float(y.sum()),
                sum_y2=float((y**2).sum()),
                sum_x=float(x.sum()),
                sum_x2=float((x**2).sum()),
                sum_xy=float((x * y).sum()),
            )
        )
    results = estimate_lift(
        metrics=[MeanMetric(name="revenue", entity="user", fact="revenue")],
        summary=_summary_df(arms),
        control_group="control",
        methods=[
            Method(name="unadjusted"),
            Method(name="cuped", variance_reduction="cuped"),
        ],
    ).results
    return {row.method: row.require_lift() for row in results if isinstance(row, LiftEstimate)}


def _lifts(frame: pd.DataFrame, metric: Literal["mean", "ratio"]):
    if metric == "mean":
        return _summary_lifts(frame)
    rows = _analysis(frame, metric).run(
        decision_method=Method(name="unadjusted"),
        sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
    )
    result = {row.method: row.require_lift() for row in rows if isinstance(row, LiftEstimate)}
    return {name: result[name] for name in ("unadjusted", "cuped")}


def _coverage_trial(
    i: int,
    *,
    metric: Literal["mean", "ratio"],
    scenario: Literal["useful", "useless", "shifted"],
    n_per_arm: int,
):
    frame = _experiment(np.random.default_rng(20261012 + i), n_per_arm, scenario, metric)
    lift = _lifts(frame, metric)["cuped"]
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
    rate = sum(covered) / reps
    lo, _ = nominal_band(0.95, reps, k=3.0)
    assert rate >= lo


@pytest.mark.parametrize("metric", _METRICS)
def test_cupac_precision_tracks_prediction_quality(metric):
    widths = {}
    slopes = {}
    for index, scenario in enumerate(_SCENARIOS):
        frame = _experiment(np.random.default_rng(731_002 + index), 20_000, scenario, metric)
        lifts = _lifts(frame, metric)
        widths[scenario] = (lifts["cuped"].ub - lifts["cuped"].lb) / (
            lifts["unadjusted"].ub - lifts["unadjusted"].lb
        )
        control = frame.loc[frame["variant"] == "control"]
        slopes[scenario] = np.polyfit(control["prediction"], control["revenue"], 1)[0]
        if metric == "mean":
            by_arm = frame.groupby("variant")
            centered_prediction = frame["prediction"] - by_arm["prediction"].transform("mean")
            centered_revenue = frame["revenue"] - by_arm["revenue"].transform("mean")
            rho = np.corrcoef(centered_prediction, centered_revenue)[0, 1]
            assert widths[scenario] == pytest.approx(math.sqrt(1 - rho**2), rel=0.05)
    assert 0.97 <= widths["useless"] <= 1.03
    assert widths["useful"] < widths["shifted"] < widths["useless"]
    print(f"{metric} precision: width_ratios={widths}, control_calibration_slopes={slopes}")


@pytest.mark.parametrize("metric", _METRICS)
def test_cupac_summary_and_panel_paths_agree(metric):
    frame = _experiment(np.random.default_rng(612_004), 500, "useful", metric)
    fractions = (0.2, 0.3, 0.5)
    panel_rows = []
    for row in frame.to_dict("records"):
        allocated_sessions = [0, 0, row["sessions"]]
        for day, fraction in enumerate(fractions):
            panel_rows.append(
                {
                    **row,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "revenue": row["revenue"] * fraction,
                    "sessions": allocated_sessions[day],
                }
            )
    panel = pd.DataFrame(panel_rows)
    totals = panel.groupby("user_id")[["revenue", "sessions"]].sum()
    expected = frame.set_index("user_id")[["revenue", "sessions"]]
    pd.testing.assert_frame_equal(
        totals.sort_index(),
        expected.sort_index(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    spec = (
        MetricSpec(
            name="revenue",
            type="mean",
            value_column="revenue",
            covariate="prediction",
            covariate_missing="error",
        )
        if metric == "mean"
        else MetricSpec(
            name="revenue_per_session",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            covariate="prediction",
            covariate_missing="error",
        )
    )
    summary = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[spec],
    )
    panel_analysis = Analysis.from_unit_panel(
        panel,
        unit="user_id",
        group="variant",
        date="date",
        control="control",
        metrics=[spec],
    )
    method = Method(name="cuped", variance_reduction="cuped")
    summary_rows = summary.run(sensitivity_methods=(method,))
    panel_rows_out = panel_analysis.run(sensitivity_methods=(method,))
    summary_row = next(row for row in summary_rows if row.method == "cuped")
    panel_row = next(row for row in panel_rows_out if row.method == "cuped")
    from tests.parity_harness.comparison import nested_close

    summary_dump = summary_row.model_dump()
    panel_dump = panel_row.model_dump()
    summary_dump.pop("source_snapshot_id")
    panel_dump.pop("source_snapshot_id")
    assert nested_close(summary_dump, panel_dump)


def test_cupac_panel_requires_unit_constant_prediction():
    frame = _experiment(np.random.default_rng(612_005), 20, "useful", "mean")
    panel = pd.concat(
        [
            frame.assign(date=pd.Timestamp("2026-01-01")),
            frame.assign(
                date=pd.Timestamp("2026-01-02"),
                prediction=frame["prediction"] + 1,
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(InvalidRequestError) as raised:
        analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    value_column="revenue",
                    covariate="prediction",
                )
            ],
        )
        analysis.run()
    assert raised.value.code == "frame.frame_panel.unit_covariate_varies"


def test_cupac_missing_predictions_refuse_under_error_policy():
    frame = _experiment(np.random.default_rng(612_006), 100, "useful", "mean")
    frame.loc[frame.index[:10], "prediction"] = np.nan
    with pytest.raises(InvalidRequestError) as raised:
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    value_column="revenue",
                    covariate="prediction",
                    covariate_missing="error",
                )
            ],
        )
        analysis.run()
    assert raised.value.code == "frame.validation.metric_covariate_column"
