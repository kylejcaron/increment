"""DerSimonian-Laird + Hartung-Knapp-Sidik-Jonkman pooling at small K.

Pre-registered design (fixed before any interval was inspected)
---------------------------------------------------------------
Mechanism
    ``K`` segment estimates on the log relative-lift scale, ``est_k ~ N(mu +
    delta_k, v_k)`` with ``delta_k ~ N(0, tau2)``; the sampling variances are
    unequal, spread geometrically from 0.01 to 0.04 (log-scale SEs 0.10 to
    0.20). ``tau2 = r * mean(v)`` for ``r`` in {0, 0.5, 2}. Each replication
    is fed to ``segment_heterogeneity`` as directly constructed breakout rows
    carrying ``log_mean``/``log_se`` (no absolute-scale moments, so only the
    relative-scale pooled row is produced).
Truth
    The random-effects mean ``mu = 0.10``.
Counts
    ``K`` in {2, 3, 4, 5, 8, 12} x ``r`` in {0, 0.5, 2}: 18 cells, 10000
    replications each (~0.7 ms per replication).
Seeds
    ``numpy.random.default_rng(cell_seed(K, r))``, ``cell_seed = 5_000_000 +
    100 * K + round(10 * r)``.
Statistics
    Coverage of ``mu`` by the production HKSJ pooled interval (``t_(K-1)``
    reference on the HKSJ variance, DL ``tau2``, no floor); coverage of the
    plug-in random-effects interval ``mu_hat +/- z * sqrt(1 / sum w)`` rebuilt
    inline from the same DL ``tau2`` the production row reports; the fraction
    of non-degenerate replications where HKSJ is narrower than plug-in; and
    the rate at which the production degeneracy fallback fires (the HKSJ
    variance below 1e-8 of plug-in, surfaced as a warning naming the
    fallback). Binomial MCSE ``sqrt(p(1-p)/10000)`` (0.0022 at 0.95).
Tolerance / MC uncertainty
    Every pinned rate is asserted within k=3 of its MCSE (plus 5e-5 for the
    four-decimal pin) of the value measured on the commit that introduced
    this study. The degeneracy fallback is additionally exercised
    deterministically on exact ties, where it must fire on every call.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from tests.mc import CoverageSet, mcse
from tests.warning_codes import warning_codes

MU = 0.10
ALPHA = 0.05
V_MIN = 0.01
V_MAX = 0.04
KS = (2, 3, 4, 5, 8, 12)
TAU2_RATIOS = (0.0, 0.5, 2.0)
REPS = 10000
PIN_ROUNDING = 5e-5
Z = float(norm.isf(ALPHA / 2))

Cell = tuple[int, float]
GRID: tuple[Cell, ...] = tuple((k, ratio) for k in KS for ratio in TAU2_RATIOS)

# Measured on the commit that introduced this study at 10000 reps:
# (HKSJ coverage, plug-in coverage, HKSJ-narrower fraction, degeneracy rate).
PINNED: dict[Cell, tuple[float, float, float, float]] = {
    (2, 0.0): (0.9496, 0.9631, 0.1188, 0.0001),
    (2, 0.5): (0.9426, 0.8905, 0.1001, 0.0001),
    (2, 2.0): (0.9413, 0.7997, 0.0684, 0.0000),
    (3, 0.0): (0.9513, 0.9623, 0.1858, 0.0000),
    (3, 0.5): (0.9402, 0.9110, 0.1281, 0.0000),
    (3, 2.0): (0.9367, 0.8519, 0.0651, 0.0000),
    (4, 0.0): (0.9519, 0.9637, 0.2319, 0.0000),
    (4, 0.5): (0.9373, 0.9123, 0.1348, 0.0000),
    (4, 2.0): (0.9416, 0.8718, 0.0486, 0.0000),
    (5, 0.0): (0.9475, 0.9598, 0.2663, 0.0000),
    (5, 0.5): (0.9431, 0.9180, 0.1303, 0.0000),
    (5, 2.0): (0.9432, 0.8863, 0.0387, 0.0000),
    (8, 0.0): (0.9493, 0.9640, 0.3262, 0.0000),
    (8, 0.5): (0.9487, 0.9302, 0.1207, 0.0000),
    (8, 2.0): (0.9490, 0.9091, 0.0179, 0.0000),
    (12, 0.0): (0.9505, 0.9602, 0.3511, 0.0000),
    (12, 0.5): (0.9427, 0.9294, 0.0991, 0.0000),
    (12, 2.0): (0.9432, 0.9219, 0.0239, 0.0000),
}


def cell_seed(k: int, ratio: float) -> int:
    return 5_000_000 + 100 * k + int(round(ratio * 10))


def segment_variances(k: int) -> np.ndarray:
    """``K`` sampling variances spread geometrically from V_MIN to V_MAX."""
    return V_MIN * (V_MAX / V_MIN) ** (np.arange(k) / (k - 1))


def build_estimates(est: np.ndarray, var: np.ndarray):
    """Directly constructed breakout rows with known log-scale moments."""
    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.estimation.results import Estimate

    rows = []
    for i, (e, v) in enumerate(zip(est, var, strict=True)):
        se = math.sqrt(v)
        rows.append(
            BreakoutEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                dimension="segment",
                dimension_value=f"s{i}",
                source=None,
                lift=Estimate(
                    value=math.expm1(e),
                    lb=math.expm1(e - Z * se),
                    ub=math.expm1(e + Z * se),
                    level=1.0 - ALPHA,
                    log_mean=e,
                    log_se=se,
                ),
            )
        )
    return BreakoutEstimates(rows)


def pooled_row(est: np.ndarray, var: np.ndarray):
    """``(relative-scale summary row, degeneracy fallback fired)`` from
    ``segment_heterogeneity``; the fallback is surfaced as a warning."""
    from increment.breakout.heterogeneity import segment_heterogeneity

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary, _segments = segment_heterogeneity(build_estimates(est, var), alpha=ALPHA)
    codes = warning_codes(caught)
    assert all(code == "breakout.heterogeneity.hksj_degenerate_variance_fallback" for code in codes)
    (row,) = [s for s in summary if s.scale == "relative"]
    return row, bool(codes)


def study(cell: Cell, reps: int, seed: int) -> dict[str, float]:
    k, ratio = cell
    rng = np.random.default_rng(seed)
    var = segment_variances(k)
    tau2 = ratio * float(var.mean())
    covset = CoverageSet()
    degenerate = narrower = compared = 0
    for _ in range(reps):
        delta = rng.normal(0.0, math.sqrt(tau2), size=k) if tau2 > 0 else np.zeros(k)
        est = MU + delta + rng.normal(0.0, np.sqrt(var), size=k)
        row, is_degenerate = pooled_row(est, var)
        degenerate += is_degenerate
        assert row.tau2 is not None and row.pooled.lb is not None and row.pooled.ub is not None
        w = 1.0 / (var + row.tau2)
        mu_hat = float((w * est).sum() / w.sum())
        half = Z * math.sqrt(1.0 / w.sum())
        hksj_lb, hksj_ub = math.log1p(row.pooled.lb), math.log1p(row.pooled.ub)
        covset.record(
            hksj=hksj_lb <= MU <= hksj_ub,
            plug_in=(mu_hat - half) <= MU <= (mu_hat + half),
        )
        if not is_degenerate:
            compared += 1
            narrower += (hksj_ub - hksj_lb) < 2 * half
    return {
        "hksj": covset["hksj"].rate,
        "plug_in": covset["plug_in"].rate,
        "narrower": narrower / compared if compared else float("nan"),
        "compared": compared,
        "degenerate": degenerate / reps,
    }


def _within(measured: float, pinned: float, reps: int, what) -> None:
    tolerance = 3 * mcse(max(pinned, 1 / reps), reps) + PIN_ROUNDING
    assert abs(measured - pinned) <= tolerance, (what, measured, pinned, tolerance)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cell", GRID, ids=lambda c: f"K{c[0]}-r{c[1]}")
def test_hksj_small_k_calibration_is_pinned(cell: Cell):
    result = study(cell, REPS, cell_seed(*cell))
    hksj, plug_in, narrower, degenerate = PINNED[cell]
    _within(result["hksj"], hksj, REPS, (cell, "hksj"))
    _within(result["plug_in"], plug_in, REPS, (cell, "plug_in"))
    _within(result["narrower"], narrower, int(result["compared"]), (cell, "narrower"))
    _within(result["degenerate"], degenerate, REPS, (cell, "degenerate"))


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("k", KS)
def test_hksj_holds_nominal_where_plug_in_collapses(k: int):
    """Directional invariant independent of the pinned values: under real
    heterogeneity (``tau2 = 2 * mean(v)``) the plug-in interval undercovers
    by more than 3 MCSE at every K while HKSJ stays within 2.5 points of
    nominal, and HKSJ is narrower than plug-in in a strictly smaller fraction
    of replications under heterogeneity than under homogeneity."""
    heterogeneous = study((k, 2.0), REPS, cell_seed(k, 2.0))
    homogeneous = study((k, 0.0), REPS, cell_seed(k, 0.0))
    assert heterogeneous["plug_in"] < 0.95 - 3 * mcse(0.95, REPS), heterogeneous
    assert abs(heterogeneous["hksj"] - 0.95) <= 0.025, heterogeneous
    assert heterogeneous["narrower"] < homogeneous["narrower"], (heterogeneous, homogeneous)


def test_degeneracy_fallback_fires_on_exact_ties():
    """Identical segment estimates make the HKSJ variance exactly zero: the
    production row must fall back to the plug-in interval (and say so)
    rather than emit a zero-width interval."""
    var = segment_variances(4)
    est = np.full(4, MU)
    row, is_degenerate = pooled_row(est, var)
    assert is_degenerate
    assert row.tau2 is not None and row.pooled.lb is not None and row.pooled.ub is not None
    w = 1.0 / (var + row.tau2)
    half = Z * math.sqrt(1.0 / w.sum())
    assert math.log1p(row.pooled.ub) - math.log1p(row.pooled.lb) == pytest.approx(2 * half)


def test_hksj_small_k_smoke():
    """Fast twin: 40 replications at K=5 under moderate heterogeneity keep
    the pipeline honest (HKSJ coverage inside the nominal k=3 band, no
    degeneracy on continuous data)."""
    cell: Cell = (5, 0.5)
    reps = 40
    result = study(cell, reps, cell_seed(*cell))
    assert result["hksj"] >= 0.95 - 3 * mcse(0.95, reps), result
    assert result["degenerate"] == 0.0, result
