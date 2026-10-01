"""CUPED theta-uncertainty calibration of the fixed-horizon lift estimator.

Pre-registered design (fixed before any interval was inspected)
---------------------------------------------------------------
Mechanism
    Equal allocation, ``n`` units per arm, one pre-period covariate ``x ~
    N(0, 1)`` and outcome ``y = mu_arm + beta * x + e`` with ``beta = rho *
    sigma_y``, ``e ~ N(0, sigma_y**2 * (1 - rho**2))``, ``sigma_y = 2``,
    ``mu_c = 10`` and ``mu_t = 10.5``. The slope is homogeneous across arms
    and the allocation is 1:1; unequal allocation and heterogeneous slopes are
    deliberately out of scope here.
Truth
    Relative lift 0.05. The known-beta oracle uses a raw-array delta-method
    gradient that propagates the shared estimated covariate anchor, and cuts
    its interval at the same Welch-Satterthwaite t reference production uses
    for an unpriored fixed-horizon row, with degrees of freedom from its own
    known-beta per-arm variances. It is a comparator, not exact finite-sample
    inference. Width ratios describe reported uncertainty; they do not
    isolate the additional sampling variance caused by estimating theta.
Counts
    ``rho`` in {0.1, 0.3, 0.6, 0.9} x ``n`` per arm in {20, 50, 200, 2000}:
    16 cells.
Seeds
    ``numpy.random.default_rng(cell_seed(rho, n))`` with ``cell_seed =
    3_000_000 + 10_000 * round(10 * rho) + n``.
Repetitions
    3000 per cell. Each replication reduces the two arms to the centered
    moments the dataframe entry emits and calls ``estimate_lift`` on them
    directly (under a millisecond); the first replication of every cell
    also runs the full ``Analysis.from_unit_summary`` path on the same
    units and must agree with the direct call, so the public entry stays
    under test without paying its frame validation 3000 times per cell.
Statistic
    95% coverage of the CUPED interval and of the oracle interval (binomial
    MCSE ``sqrt(p(1-p)/3000)`` = 0.0040 at 0.95); the geometric mean
    log-scale interval width ratio ``CUPED / oracle`` and its delta-method
    MCSE; fitted and oracle reported-SE / empirical-SD ratios (relative
    MCSE approximately ``1 / sqrt(2 * (reps - 1))`` = 0.013). The SE-ratio
    approximation assumes near-Normal log estimates and uncertainty dominated
    by their empirical SD; it neglects mean-SE noise and its covariance with SD.
Tolerance / MC uncertainty
    Every pinned number is asserted within k=3 of its MCSE (plus half the
    last recorded decimal: 5e-5, or 5e-9 for widths) of the value measured
    at REPS replications under the Welch t reference. These fixed-seed
    regression bands are not simultaneous confidence bounds or promises of
    nominal finite-sample coverage. Each fitted point, SE and reference
    degrees of freedom must also agree with independent raw-array gradient
    calculations. No exact finite-N formula or monotonicity is assumed for
    log widths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.mc import CoverageSet, mcse

MU_C = 10.0
LIFT = 0.05
SIGMA_Y = 2.0
ALPHA = 0.05
RHOS = (0.1, 0.3, 0.6, 0.9)
NS = (20, 50, 200, 2000)
REPS = 3000
PIN_ROUNDING = 5e-5
WIDTH_PIN_ROUNDING = 5e-9

Cell = tuple[float, int]
GRID: tuple[Cell, ...] = tuple((rho, n) for rho in RHOS for n in NS)

# Fitted CUPED at REPS replications: (coverage, reported-SE / empirical-SD).
PINNED: dict[Cell, tuple[float, float]] = {
    (0.1, 20): (0.9500, 0.9771),
    (0.1, 50): (0.9480, 0.9844),
    (0.1, 200): (0.9467, 0.9889),
    (0.1, 2000): (0.9510, 0.9965),
    (0.3, 20): (0.9443, 0.9699),
    (0.3, 50): (0.9343, 0.9381),
    (0.3, 200): (0.9453, 0.9876),
    (0.3, 2000): (0.9487, 1.0049),
    (0.6, 20): (0.9380, 0.9489),
    (0.6, 50): (0.9423, 0.9746),
    (0.6, 200): (0.9393, 0.9762),
    (0.6, 2000): (0.9530, 1.0076),
    (0.9, 20): (0.9443, 0.9713),
    (0.9, 50): (0.9487, 0.9821),
    (0.9, 200): (0.9497, 0.9932),
    (0.9, 2000): (0.9493, 0.9945),
}


# Known-beta oracle at REPS replications:
# (oracle coverage, geometric width ratio, SE ratio, width MCSE).
ORACLE_PINNED: dict[Cell, tuple[float, float, float, float]] = {
    (0.1, 20): (0.9560, 0.98651468, 1.0087, 0.000341762608012),
    (0.1, 50): (0.9503, 0.99475237, 0.9945, 0.000136938697057),
    (0.1, 200): (0.9480, 0.99872945, 0.9918, 3.17867925899e-05),
    (0.1, 2000): (0.9500, 0.99987011, 0.9965, 3.31771436314e-06),
    (0.3, 20): (0.9497, 0.98675648, 0.9928, 0.000343032606106),
    (0.3, 50): (0.9377, 0.99482760, 0.9481, 0.000132740574636),
    (0.3, 200): (0.9473, 0.99878143, 0.9887, 3.2139845502e-05),
    (0.3, 2000): (0.9477, 0.99987528, 1.0050, 3.17488681953e-06),
    (0.6, 20): (0.9443, 0.98675331, 0.9795, 0.000349176273817),
    (0.6, 50): (0.9437, 0.99493615, 0.9825, 0.000128012401983),
    (0.6, 200): (0.9390, 0.99874651, 0.9786, 3.28807572116e-05),
    (0.6, 2000): (0.9537, 0.99988109, 1.0079, 3.05499196203e-06),
    (0.9, 20): (0.9487, 0.98612659, 1.0004, 0.000356235284118),
    (0.9, 50): (0.9480, 0.99493811, 0.9882, 0.0001317145652),
    (0.9, 200): (0.9500, 0.99874165, 0.9957, 3.29550411556e-05),
    (0.9, 2000): (0.9487, 0.99987369, 0.9944, 3.28015468346e-06),
}


def cell_seed(rho: float, n: int) -> int:
    return 3_000_000 + 10_000 * int(round(rho * 10)) + n


def draw(rng: np.random.Generator, rho: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """One replication's outcome and pre-period covariate, control rows first."""
    beta = rho * SIGMA_Y
    sigma_e = SIGMA_Y * math.sqrt(1.0 - rho * rho)
    x = rng.normal(0.0, 1.0, size=2 * n)
    mu = np.repeat(np.array([MU_C, MU_C * (1.0 + LIFT)]), n)
    y = mu + beta * x + rng.normal(0.0, sigma_e, size=2 * n)
    return y, x


def unit_frame(y: np.ndarray, x: np.ndarray):
    """The one-row-per-unit frame the public dataframe entry reads."""
    import polars as pl

    n = len(y) // 2
    return pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": np.repeat(np.array(["control", "treatment"]), n),
            "y": y,
            "x": x,
        }
    )


def moment_row(group_id: str, y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    """The centered arm moments the dataframe entry reduces these units to."""
    ref_y = float(np.mean(y))
    ref_x = float(np.mean(x))
    dy = y - ref_y
    dx = x - ref_x
    return {
        "experiment_id": "frame",
        "metric": "y",
        "group_id": group_id,
        "x_role": "covariate",
        "n": len(y),
        "ref_y": ref_y,
        "cy1": float(np.sum(dy)),
        "cy2": float(np.sum(dy * dy)),
        "ref_x": ref_x,
        "cx1": float(np.sum(dx)),
        "cx2": float(np.sum(dx * dx)),
        "cxy": float(np.sum(dx * dy)),
    }


def _assert_public_entry_agrees(direct, y: np.ndarray, x: np.ndarray) -> None:
    """``Analysis.from_unit_summary`` on the same units yields the same row."""
    from increment import Analysis, Method, MetricSpec
    from tests.analysis_factory import lift_rows

    spec = MetricSpec(
        name="y",
        type="mean",
        covariate="x",
        decision_method=Method(name="cuped", variance_reduction="cuped"),
    )
    (public,) = lift_rows(
        Analysis.from_unit_summary(
            unit_frame(y, x), unit="user_id", group="variant", control="control", metrics=[spec]
        ).run()
    )
    assert (public.method, public.reference_kind) == (direct.method, direct.reference_kind)
    assert public.reference_df == pytest.approx(direct.reference_df, rel=1e-10)
    assert public.lift is not None and direct.lift is not None
    for field in ("log_mean", "log_se", "lb", "ub"):
        assert getattr(public.lift, field) == pytest.approx(
            getattr(direct.lift, field), rel=1e-10, abs=1e-12
        ), field


def _log_reference(
    means: np.ndarray, covariances: np.ndarray, theta: float, n: int
) -> tuple[float, float, float]:
    """Raw-array delta-method log lift, its SE and its Welch-Satterthwaite
    degrees of freedom; rows are control then treatment."""
    anchor = means[:, 1].mean()
    mc, mt = means[:, 0] - theta * (means[:, 1] - anchor)
    k = 0.5 / mt + 0.5 / mc
    gc = np.array([-1 / mc, theta * k])
    gt = np.array([1 / mt, -theta * k])
    var_c = gc @ covariances[0] @ gc
    var_t = gt @ covariances[1] @ gt
    df = (var_c + var_t) ** 2 / ((var_c**2 + var_t**2) / (n - 1))
    return math.log(mt) - math.log(mc), math.sqrt(var_c + var_t), df


def study(cell: Cell, reps: int, seed: int) -> dict[str, float]:
    from scipy.stats import t as t_dist

    from increment import Method
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric

    rho, n = cell
    rng = np.random.default_rng(seed)
    covset = CoverageSet()
    log_width_ratio = []
    log_est = []
    log_se = []
    oracle_log_est = []
    oracle_log_se = []
    metrics = [MeanMetric(name="y", entity="user", fact="y", aggregation="sum")]
    methods = [Method(name="cuped", variance_reduction="cuped")]
    for rep in range(reps):
        y, x = draw(rng, rho, n)
        rows = [moment_row("control", y[:n], x[:n]), moment_row("treatment", y[n:], x[n:])]
        (cuped,) = estimate_lift(metrics, rows, "control", methods=methods).results
        if rep == 0:
            _assert_public_entry_agrees(cuped, y, x)
        assert cuped.method == "cuped"
        lift = cuped.lift
        assert lift is not None and lift.lb is not None and lift.ub is not None
        assert lift.log_mean is not None and lift.log_se is not None
        assert cuped.reference_kind == "t" and cuped.reference_df is not None
        # Column-major so each arm's column reductions sum in the order the pins were recorded.
        raw = np.empty((2 * n, 2), order="F")
        raw[:, 0] = y
        raw[:, 1] = x
        arms = (raw[:n], raw[n:])
        means = np.array([arm.mean(axis=0) for arm in arms])
        covariances = np.array([np.cov(arm, rowvar=False, ddof=1) / n for arm in arms])
        theta = covariances[:, 0, 1].sum() / covariances[:, 1, 1].sum()
        fitted_mean, fitted_se, fitted_df = _log_reference(means, covariances, theta, n)
        assert lift.log_mean == pytest.approx(fitted_mean, rel=1e-10, abs=1e-12)
        assert lift.log_se == pytest.approx(fitted_se, rel=1e-10, abs=1e-12)
        assert cuped.reference_df == pytest.approx(fitted_df, rel=1e-10)
        oracle_mean, oracle_se, oracle_df = _log_reference(means, covariances, rho * SIGMA_Y, n)
        critical = float(t_dist.isf(ALPHA / 2, oracle_df))
        covset.record(
            cuped=lift.lb <= LIFT <= lift.ub,
            oracle=abs(oracle_mean - math.log1p(LIFT)) <= critical * oracle_se,
        )
        width_cuped = math.log1p(lift.ub) - math.log1p(lift.lb)
        log_width_ratio.append(math.log(width_cuped / (2 * critical * oracle_se)))
        oracle_log_est.append(oracle_mean)
        oracle_log_se.append(oracle_se)
        log_est.append(lift.log_mean)
        log_se.append(lift.log_se)
    ratios = np.asarray(log_width_ratio)
    oracle_se_ratio = float(np.mean(oracle_log_se) / np.std(oracle_log_est, ddof=1))
    return {
        "cuped": covset["cuped"].rate,
        "oracle": covset["oracle"].rate,
        "width_ratio": float(math.exp(ratios.mean())),
        "width_ratio_se": float(math.exp(ratios.mean()) * ratios.std(ddof=1) / math.sqrt(reps)),
        "se_ratio": float(np.mean(log_se) / np.std(log_est, ddof=1)),
        "se_ratio_se": float(np.mean(log_se) / np.std(log_est, ddof=1) / math.sqrt(2 * (reps - 1))),
        "oracle_se_ratio": oracle_se_ratio,
        "oracle_se_ratio_se": oracle_se_ratio / math.sqrt(2 * (reps - 1)),
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cell", GRID, ids=lambda c: f"rho{c[0]}-n{c[1]}")
def test_cuped_interval_calibration_is_pinned(cell: Cell):
    result = study(cell, REPS, cell_seed(*cell))
    cuped, se_ratio = PINNED[cell]
    assert abs(result["cuped"] - cuped) <= 3 * mcse(cuped, REPS), (cell, result)
    assert abs(result["se_ratio"] - se_ratio) <= 3 * result["se_ratio_se"] + PIN_ROUNDING, (
        cell,
        result,
    )
    oracle, width_ratio, oracle_se_ratio, width_ratio_se = ORACLE_PINNED[cell]
    assert abs(result["oracle"] - oracle) <= 3 * mcse(oracle, REPS) + PIN_ROUNDING, (
        cell,
        result,
    )
    assert abs(result["width_ratio"] - width_ratio) <= (3 * width_ratio_se + WIDTH_PIN_ROUNDING), (
        cell,
        result,
    )
    assert abs(result["oracle_se_ratio"] - oracle_se_ratio) <= (
        3 * oracle_se_ratio / math.sqrt(2 * (REPS - 1)) + PIN_ROUNDING
    ), (cell, result)
    if cell[1] >= 200:
        assert abs(result["oracle"] - 0.95) <= 3 * mcse(0.95, REPS), (cell, result)


def test_cuped_theta_uncertainty_smoke():
    """Small-N twin checks parity, coverage, widths and oracle SE calibration."""
    cell: Cell = (0.6, 20)
    reps = 8
    result = study(cell, reps, cell_seed(*cell))
    assert result["cuped"] >= 0.95 - 3 * mcse(0.95, reps), result
    oracle, width_ratio, oracle_se_ratio, width_ratio_se = ORACLE_PINNED[cell]
    assert abs(result["oracle"] - oracle) <= 3 * mcse(oracle, reps) + PIN_ROUNDING, result
    # Recover baseline log-width dispersion and apply the pinned geometric delta factor.
    smoke_width_se = width_ratio_se * math.sqrt(REPS / reps)
    assert abs(result["width_ratio"] - width_ratio) <= (3 * smoke_width_se + WIDTH_PIN_ROUNDING), (
        result
    )
    # Fix the smoke band's scale to the full-study pin so inflated SEs cannot widen it.
    oracle_se_ratio_se = oracle_se_ratio / math.sqrt(2 * (reps - 1))
    assert abs(result["oracle_se_ratio"] - oracle_se_ratio) <= (
        3 * oracle_se_ratio_se + PIN_ROUNDING
    ), result
