"""Welch reference calibration for the cluster-randomized encouragement LATE.

Pre-registered design (fixed before any interval was inspected)
---------------------------------------------------------------
Mechanism
    Two arms of ``K_t`` / ``K_c`` clusters. Cluster ``j`` has ``m_j`` units
    (``equal``: ``m_j = 20``; ``lognormal``: ``m_j = max(2, round(20 *
    LogNormal(-0.75**2 / 2, 0.75)))``, mean 20), a shared effect ``b_j ~
    N(0, sigma_b**2)`` and unit noise ``e_ij ~ N(0, 2**2)``. The control
    between-cluster variance is 1 and the treatment one is ``ratio`` times
    that (``ratio`` in {1, 4, 16}), so the arms genuinely differ in
    between-cluster variance. One-sided encouragement: uptake ``d_ij ~
    Bernoulli(0.6)`` in the treatment arm, 0 in control, ``y_ij = 10 + b_j +
    2 * d_ij + e_ij``.
Truth
    Additive LATE 2.0.
Counts
    ``(K_t, K_c)`` in {(5, 5), (5, 25), (10, 50)} x ``ratio`` in {1, 4, 16} x
    dispersion in {equal, lognormal}: 18 cells.
Seeds
    ``numpy.random.default_rng(cell_seed(...))``, one stream per cell;
    ``cell_seed`` is the explicit table ``1_000_000 + 10000*K_t + 100*K_c +
    10*log4(ratio) + [dispersion == lognormal]``.
Repetitions
    2000 per cell (public dataframe entry, a few ms per call).
Statistic
    95% interval coverage of the truth over emitted intervals under the
    production reference, a Welch-Satterthwaite t over the two arms' own
    cluster-ratio variance components, and, side by side, the coverage a
    pooled ``t_(K_t + K_c - 2)`` reference over the same components would
    have given. Both intervals are rebuilt inline from the raw cluster
    totals; the production interval must match the inline Welch one to
    1e-9 relative in both its SE and its degrees of freedom, which pins the
    components as production's own. Every emitted row must also report the
    Welch reference by name (``reference_kind == "t"``), so an estimator
    that adopts another reference fails here explicitly rather than only
    through the parity gap. Refused replications are counted and reported,
    never dropped silently, and each must be the named mechanism: an inline
    first-stage z below the design's ``min_first_stage_z`` with the surviving
    absolute compliance row carrying the ``late suppressed: first-stage
    z=...`` diagnostic. Emitted replications must sit on the other side of
    the same gate, so the count is exactly the gate's.
Tolerance / MC uncertainty
    Binomial MCSE ``sqrt(p(1-p)/reps)``: 0.005-0.0085 at 2000 reps. Every
    pinned coverage is asserted within k=3 MCSE of the value measured on the
    commit that introduced this study, so an estimator change moves the
    pinned number visibly.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import t as t_dist

from tests.mc import CoverageSet, mcse

if TYPE_CHECKING:
    import polars as pl

MU_C = 10.0
SIGMA_E = 2.0
SIGMA_B_C = 1.0
M_MEAN = 20
LOGNORMAL_SIGMA = 0.75
TAU_LATE = 2.0
UPTAKE_RATE = 0.6
ALPHA = 0.05
REPS = 2000

Cell = tuple[int, int, float, str]
GRID: tuple[Cell, ...] = tuple(
    (k_t, k_c, ratio, dispersion)
    for (k_t, k_c) in ((5, 5), (5, 25), (10, 50))
    for ratio in (1.0, 4.0, 16.0)
    for dispersion in ("equal", "lognormal")
)

# Measured on the commit that introduced this study at 2000 reps:
# (pooled-df coverage, Welch coverage, refused replications) per cell.
PINNED: dict[Cell, tuple[float, float, int]] = {
    (5, 5, 1.0, "equal"): (0.9530, 0.9595, 0),
    (5, 5, 1.0, "lognormal"): (0.8978, 0.9088, 4),
    (5, 5, 4.0, "equal"): (0.9375, 0.9490, 0),
    (5, 5, 4.0, "lognormal"): (0.8994, 0.9144, 2),
    (5, 5, 16.0, "equal"): (0.9360, 0.9515, 0),
    (5, 5, 16.0, "lognormal"): (0.8749, 0.8974, 2),
    (5, 25, 1.0, "equal"): (0.9290, 0.9535, 1),
    (5, 25, 1.0, "lognormal"): (0.8778, 0.9109, 3),
    (5, 25, 4.0, "equal"): (0.8985, 0.9430, 0),
    (5, 25, 4.0, "lognormal"): (0.8349, 0.8854, 1),
    (5, 25, 16.0, "equal"): (0.8915, 0.9455, 0),
    (5, 25, 16.0, "lognormal"): (0.8203, 0.8974, 2),
    (10, 50, 1.0, "equal"): (0.9400, 0.9535, 0),
    (10, 50, 1.0, "lognormal"): (0.9110, 0.9270, 0),
    (10, 50, 4.0, "equal"): (0.9295, 0.9560, 0),
    (10, 50, 4.0, "lognormal"): (0.8930, 0.9215, 0),
    (10, 50, 16.0, "equal"): (0.9260, 0.9470, 0),
    (10, 50, 16.0, "lognormal"): (0.8745, 0.9085, 0),
}


def cell_seed(k_t: int, k_c: int, ratio: float, dispersion: str) -> int:
    log4_ratio = int(round(math.log(ratio, 4.0)))
    return 1_000_000 + 10_000 * k_t + 100 * k_c + 10 * log4_ratio + (dispersion == "lognormal")


def cluster_sizes(rng: np.random.Generator, k: int, dispersion: str) -> np.ndarray:
    if dispersion == "equal":
        return np.full(k, M_MEAN, dtype=int)
    raw = M_MEAN * rng.lognormal(-0.5 * LOGNORMAL_SIGMA**2, LOGNORMAL_SIGMA, size=k)
    return np.maximum(2, np.rint(raw)).astype(int)


def draw_arm(
    rng: np.random.Generator,
    k: int,
    sigma_b: float,
    mu: float,
    dispersion: str,
    *,
    uptake_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """``(m_j, cluster index per unit, y, d)`` for one arm; ``d`` is None
    without uptake."""
    m = cluster_sizes(rng, k, dispersion)
    b = rng.normal(0.0, sigma_b, size=k)
    n = int(m.sum())
    cluster = np.repeat(np.arange(k), m)
    y = mu + b[cluster] + rng.normal(0.0, SIGMA_E, size=n)
    d = None
    if uptake_rate is not None:
        d = (rng.random(n) < uptake_rate).astype(int)
        y = y + TAU_LATE * d
    return m, cluster, y, d


def welch_df(v_t: float, v_c: float, k_t: int, k_c: int) -> float:
    return (v_t + v_c) ** 2 / (v_t**2 / (k_t - 1) + v_c**2 / (k_c - 1))


Arm = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

_LABELS: dict[str, list[str]] = {}


def labels(prefix: str, n: int) -> list[str]:
    """``[f"{prefix}0", ..., f"{prefix}{n - 1}"]``, grown once and sliced."""
    known = _LABELS.setdefault(prefix, [])
    if len(known) < n:
        known.extend(f"{prefix}{i}" for i in range(len(known), n))
    return known[:n]


def late_frame(rng: np.random.Generator, cell: Cell) -> tuple[pl.DataFrame, Arm, Arm]:
    """One replication's unit frame plus each arm's raw cluster arrays."""
    import polars as pl

    k_t, k_c, ratio, dispersion = cell
    sigma_b_t = SIGMA_B_C * math.sqrt(ratio)
    m_c, cl_c, y_c, _ = draw_arm(rng, k_c, SIGMA_B_C, MU_C, dispersion)
    m_t, cl_t, y_t, d_t = draw_arm(rng, k_t, sigma_b_t, MU_C, dispersion, uptake_rate=UPTAKE_RATE)
    assert d_t is not None
    n_c, n_t = y_c.shape[0], y_t.shape[0]
    frame = pl.DataFrame(
        {
            "user_id": labels("c", n_c) + labels("t", n_t),
            "variant": ["control"] * n_c + ["treatment"] * n_t,
            "store_id": np.repeat(labels("c", k_c), m_c).tolist()
            + np.repeat(labels("t", k_t), m_t).tolist(),
            "y": np.concatenate([y_c, y_t]),
            "took": np.concatenate([np.zeros(n_c, dtype=int), d_t]),
        }
    )
    return frame, (m_c, cl_c, y_c, np.zeros(n_c, dtype=int)), (m_t, cl_t, y_t, d_t)


def late_arm_components(
    m: np.ndarray, cl: np.ndarray, y: np.ndarray, d: np.ndarray, tau: float, k: int
) -> tuple[float, float]:
    """``(variance, R_d)``: one arm's cluster-robust Wald-ratio influence
    variance at ``tau`` (before the ``1 / b**2`` scaling) and its uptake
    ratio ``sum d_j / sum m_j``."""
    g = np.bincount(cl, weights=y, minlength=k)
    u = np.bincount(cl, weights=d.astype(float), minlength=k)
    mf = m.astype(float)
    r_y = g.sum() / mf.sum()
    r_d = u.sum() / mf.sum()
    psi = ((g - r_y * mf) - tau * (u - r_d * mf)) / mf.mean()
    return float(psi.var(ddof=1) / k), float(r_d)


# The inline z and production's accumulate in different orders; within this
# band of the gate either side of the emit decision is consistent.
GATE_TOLERANCE = 1e-9


def first_stage_z(arm_c: Arm, arm_t: Arm) -> float:
    """The clustered first-stage z production gates LATE emission on: the
    uptake-ratio difference over the sum of each arm's own ddof=1
    ratio-residual variance of ``sum d_j / sum m_j``."""

    def uptake_ratio(arm: Arm) -> tuple[float, float]:
        m, cl, _y, d = arm
        k = m.shape[0]
        u = np.bincount(cl, weights=d.astype(float), minlength=k)
        mf = m.astype(float)
        r_d = u.sum() / mf.sum()
        return float(((u - r_d * mf) / mf.mean()).var(ddof=1) / k), float(r_d)

    v_c, rd_c = uptake_ratio(arm_c)
    v_t, rd_t = uptake_ratio(arm_t)
    difference = rd_t - rd_c
    variance = v_t + v_c
    if variance == 0:
        return math.copysign(math.inf, difference) if difference else 0.0
    return difference / math.sqrt(variance)


def _encouragement_design():
    import increment as inc

    return inc.Encouragement(
        control_group="control",
        uptake=inc.UptakeSpec(fact="took"),
        exclusion_restriction=inc.ExclusionRestriction(
            acknowledged=True, justification="The nudge only acts through uptake."
        ),
        one_sided=True,
    )


def late_study(cell: Cell, reps: int, seed: int) -> dict[str, float]:
    """Welch (production) vs pooled-df coverage of the additive LATE through
    ``Analysis.from_unit_summary(..., design=Encouragement, cluster=...)``."""
    from increment import Analysis
    from tests.analysis_factory import lift_rows

    k_t, k_c, _ratio, _dispersion = cell
    design = _encouragement_design()
    rng = np.random.default_rng(seed)
    covset = CoverageSet()
    t_pooled = t_dist.isf(ALPHA / 2, k_t + k_c - 2)
    max_parity_gap = 0.0
    suppressed = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="metric .* total clusters", category=RuntimeWarning
        )
        for _ in range(reps):
            frame, arm_c, arm_t = late_frame(rng, cell)
            results = lift_rows(
                Analysis.from_unit_summary(
                    frame,
                    unit="user_id",
                    group="variant",
                    metrics={"y": "mean"},
                    design=design,
                    cluster="store_id",
                ).run(estimands=("late",))
            )
            lates = [r for r in results if r.estimand == "late" and r.value_scale == "absolute"]
            z_fs = first_stage_z(arm_c, arm_t)
            if not lates:
                # The first-stage gate withheld the LATE row: the inline z is
                # below the gate and the surviving compliance row carries the
                # suppression diagnostic. Any other missing row fails here.
                assert z_fs < design.min_first_stage_z + GATE_TOLERANCE, (cell, z_fs)
                (comp,) = [
                    r for r in results if r.estimand == "compliance" and r.value_scale == "absolute"
                ]
                assert comp.note is not None and "late suppressed: first-stage z=" in comp.note, (
                    cell,
                    comp.note,
                )
                suppressed += 1
                continue
            assert z_fs >= design.min_first_stage_z - GATE_TOLERANCE, (cell, z_fs)
            (late,) = lates
            assert late.reference_kind == "t" and late.reference_df is not None, (
                cell,
                late.reference_kind,
                late.reference_df,
            )
            lift = late.lift
            assert lift is not None and lift.lb is not None and lift.ub is not None
            tau_hat = lift.value
            v_t, rd_t = late_arm_components(*arm_t, tau_hat, k_t)
            v_c, rd_c = late_arm_components(*arm_c, tau_hat, k_c)
            b = rd_t - rd_c
            se_inline = math.sqrt(v_t + v_c) / b
            df_inline = welch_df(v_t / b**2, v_c / b**2, k_t, k_c)
            se_production = (lift.ub - tau_hat) / t_dist.isf(ALPHA / 2, late.reference_df)
            max_parity_gap = max(
                max_parity_gap,
                abs(se_inline - se_production) / se_production,
                abs(df_inline - late.reference_df) / late.reference_df,
            )
            covset.record(
                pooled=(tau_hat - t_pooled * se_inline)
                <= TAU_LATE
                <= (tau_hat + t_pooled * se_inline),
                welch=lift.lb <= TAU_LATE <= lift.ub,
            )
    return {
        "pooled": covset["pooled"].rate,
        "welch": covset["welch"].rate,
        "emitted": covset["welch"].reps,
        "refused": suppressed,
        "parity_gap": max_parity_gap,
    }


def _assert_pinned(result: dict[str, float], pinned: tuple[float, float, int], reps: int, cell):
    pooled, welch, refused = pinned
    emitted = int(result["emitted"])
    assert result["parity_gap"] <= 1e-9, (cell, result["parity_gap"])
    # The Welch df never exceeds K_t + K_c - 2, so its interval contains the
    # pooled one replication by replication: coverage can only be higher.
    assert result["welch"] >= result["pooled"], (cell, result)
    assert abs(result["refused"] / reps - refused / reps) <= 3 * mcse(
        max(refused, 1) / reps, reps
    ), (
        cell,
        result["refused"],
    )
    assert abs(result["pooled"] - pooled) <= 3 * mcse(pooled, emitted), (
        cell,
        result["pooled"],
        pooled,
    )
    assert abs(result["welch"] - welch) <= 3 * mcse(welch, emitted), (cell, result["welch"], welch)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cell", GRID, ids=lambda c: f"Kt{c[0]}-Kc{c[1]}-r{int(c[2])}-{c[3]}")
def test_encouragement_late_welch_coverage_is_pinned(cell: Cell):
    result = late_study(cell, REPS, cell_seed(*cell))
    _assert_pinned(result, PINNED[cell], REPS, cell)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_unbalanced_arms_with_unequal_variance_undercover_under_pooled_df():
    """The directional finding, independent of the exact pinned values: at
    ``(K_t, K_c) = (5, 25)`` with 16x the between-cluster variance in the
    small arm, a pooled-df interval undercovers by more than 3 MCSE while
    the production Welch interval stays within 3 MCSE of nominal."""
    cell: Cell = (5, 25, 16.0, "equal")
    result = late_study(cell, REPS, cell_seed(*cell))
    emitted = int(result["emitted"])
    assert result["pooled"] < 0.95 - 3 * mcse(0.95, emitted), result
    assert abs(result["welch"] - 0.95) <= 3 * mcse(0.95, emitted), result


def test_late_welch_df_study_smoke():
    """Fast twin: the inline components reproduce production's interval and
    reference, and the Welch interval contains the pooled one, at six
    replications of the balanced equal-variance cell."""
    cell: Cell = (5, 5, 1.0, "equal")
    result = late_study(cell, 6, cell_seed(*cell))
    assert result["parity_gap"] <= 1e-9, result
    assert result["welch"] >= result["pooled"]
