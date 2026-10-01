"""Cluster-robust variance for LATE (Encouragement designs).

An Encouragement design's LATE row rides the delta-method Wald-ratio path,
not the influence-function seam the observational estimators cluster.
Declaring a cluster switches ``mean_y``/``mean_d`` to ratios of cluster
totals via the same linearized delta method
:class:`~increment.estimation.variance.ClusterVarianceModel` already uses
for ITT, with the Wald ratio's covariance matched to the cluster-grain
covariance of the two ratio numerators.

Reference distribution is the same qualified working policy used by the other
cluster paths: each enrolled arm needs at least two clusters; below 40 total
clusters a warning marks the asymptotic reference as fragile. The
complier-RELATIVE LATE row and CUPED are both refused under a declared cluster.
"""

from __future__ import annotations

from datetime import datetime

import ibis
import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

from increment.analysis import Analysis
from increment.errors import CapabilityError, IncrementRuntimeWarning, InvalidRequestError
from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.encouragement import (
    _first_stage,
    _first_stage_cluster,
    _late_additive,
    _late_additive_cluster,
    estimate_encouragement,
)
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.estimation.sequential import AlwaysValid
from increment.estimation.variance import cluster_uptake_moments
from increment.frame import MetricSpec, from_unit_summary
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
from increment.semantics.models import Definitions, MeanMetric
from tests.analysis_factory import make_analysis
from tests.mc import CoverageSet, mcse
from tests.sequential_cases import registration
from tests.warning_codes import warning_codes

DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="took"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="test fixture, not a real design"
    ),
    one_sided=True,
)
METRIC = MeanMetric(name="revenue", entity="user_id", fact="purchase", aggregation="sum")


# Singleton-cluster identity: K == n reproduces the unit-grain LATE.


def _singleton_table(seed: int, n_per_arm: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_per_arm):
        rows.append(
            {
                "user_id": f"c{i}",
                "variant": "control",
                "store_id": f"cs{i}",
                "revenue": 5.0 + float(rng.normal(0, 1)),
                "took": 0,
            }
        )
    for i in range(n_per_arm):
        d = 1 if rng.random() < 0.6 else 0
        rows.append(
            {
                "user_id": f"t{i}",
                "variant": "treatment",
                "store_id": f"ts{i}",
                "revenue": 5.0 + 2.0 * d + float(rng.normal(0, 1)),
                "took": d,
            }
        )
    return pa.Table.from_pylist(rows)


def _se_from_interval(est, dof: float | None) -> float:
    """The additive-scale SE recovered from a two-sided 95% interval's own
    reference (t_(dof) when clustered, else Normal)."""
    crit = t_dist.ppf(0.975, dof) if dof is not None else norm.ppf(0.975)
    return (est.require_lift().ub - est.require_lift().value) / crit


def test_singleton_cluster_late_matches_unclustered_exactly():
    """Every store a single unit: the clustered additive LATE row must
    reproduce the unclustered point estimate and SE (only the t-vs-Normal
    reference widens the interval - point estimates and SE never move)."""
    tbl = _singleton_table(seed=3, n_per_arm=30)

    def _run(cluster: str | None):
        return Analysis.from_unit_summary(
            tbl,
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=DESIGN,
            uptake="took",
            cluster=cluster,
        ).run(estimands=("late",))

    flat = next(r for r in _run(None) if r.value_scale == "absolute")
    clustered = next(r for r in _run("store_id") if r.value_scale == "absolute")

    assert flat.n_clusters is None and flat.dof is None
    # Clustered dof is _late_additive_cluster's Welch-Satterthwaite reduction
    # over each arm's clamped delta-method component, not pooled K - 2.
    # Independent arm draws put it near, not at, 60 - 2 == 58.
    assert clustered.n_clusters == 60
    assert clustered.dof == pytest.approx(57.995059107552116)
    assert clustered.lift.value == pytest.approx(flat.lift.value, rel=1e-9)
    assert _se_from_interval(clustered, clustered.dof) == pytest.approx(
        _se_from_interval(flat, None), rel=1e-9
    )


def test_singleton_cluster_first_stage_and_late_helpers_match_unit_grain():
    """Direct check of the math functions themselves: a singleton-cluster
    ArmStats (g_j=y_j, m_j=1, cluster uptake total=d_j) must reproduce
    ``_first_stage``/``_late_additive``'s (b, var_b)/(tau, se) to float
    precision - the m_j==1 terms (var_size, cov(*, size)) vanish
    identically, collapsing the cluster ratio delta method to the
    unit-grain moments exactly."""
    rng = np.random.default_rng(11)
    ys_c = list(rng.normal(20.0, 5.0, size=12))
    ds_c = [1 if rng.random() < 0.3 else 0 for _ in range(12)]
    ys_t = list(rng.normal(25.0, 5.0, size=12))
    ds_t = [1 if rng.random() < 0.7 else 0 for _ in range(12)]

    def unit_arm(group_id: str, ys: list[float], ds: list[int]) -> ArmStats:
        n = len(ys)
        sum_y = sum(ys)
        sum_y2 = sum(y * y for y in ys)
        sum_d = float(sum(ds))
        sum_yd = sum(y * d for y, d in zip(ys, ds, strict=True))
        sum_y2d = sum(y * y * d for y, d in zip(ys, ds, strict=True))
        return ArmStats.from_raw_sums(
            study_id="e",
            metric="m",
            group_id=group_id,
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_d=sum_d,
            sum_yd=sum_yd,
            sum_y2d=sum_y2d,
        )

    def singleton_cluster_arm(group_id: str, ys: list[float], ds: list[int]) -> ArmStats:
        g = np.array(ys)
        m = np.ones_like(g)
        xd = np.array(ds, dtype=float)
        n = len(ys)
        ref_y = g.mean()
        ref_den = m.mean()
        ref_x = xd.mean()
        return ArmStats(
            study_id="e",
            metric="m",
            group_id=group_id,
            n=n,
            ref_y=ref_y,
            cy1=float((g - ref_y).sum()),
            cy2=float(((g - ref_y) ** 2).sum()),
            ref_x=ref_x,
            cx1=float((xd - ref_x).sum()),
            cx2=float(((xd - ref_x) ** 2).sum()),
            cxy=float(((xd - ref_x) * (g - ref_y)).sum()),
            ref_den=ref_den,
            cden1=float((m - ref_den).sum()),
            cden2=float(((m - ref_den) ** 2).sum()),
            cyden=float(((g - ref_y) * (m - ref_den)).sum()),
            cxden=float(((xd - ref_x) * (m - ref_den)).sum()),
            x_role="uptake_total",
        )

    c_unit, t_unit = unit_arm("C", ys_c, ds_c), unit_arm("T", ys_t, ds_t)
    c_cl, t_cl = singleton_cluster_arm("C", ys_c, ds_c), singleton_cluster_arm("T", ys_t, ds_t)

    b_unit, var_b_unit = _first_stage(t_unit, c_unit)
    b_cl, var_b_cl, _dof_b_cl = _first_stage_cluster(t_cl, c_cl)
    assert b_cl == pytest.approx(b_unit, rel=1e-9)
    assert var_b_cl == pytest.approx(var_b_unit, rel=1e-9)

    tau_unit, se_unit = _late_additive(t_unit, c_unit)
    tau_cl, se_cl, _dof_cl = _late_additive_cluster(t_cl, c_cl)
    assert tau_cl == pytest.approx(tau_unit, rel=1e-9)
    assert se_cl == pytest.approx(se_unit, rel=1e-9)


def test_singleton_cluster_late_survives_a_negative_arm_outcome_mean():
    """``_late_additive_cluster`` is the purely ADDITIVE Wald-ratio
    numerator - it must never require a positive outcome mean the way
    the log-relative path does. A signed metric (e.g. profit) with a
    negative control-arm mean must reproduce the unclustered LATE exactly,
    not refuse."""

    def singleton_cluster_arm(group_id: str, ys: list[float], ds: list[int]) -> ArmStats:
        g = np.array(ys)
        m = np.ones_like(g)
        xd = np.array(ds, dtype=float)
        n = len(ys)
        ref_y, ref_den, ref_x = g.mean(), m.mean(), xd.mean()
        return ArmStats(
            study_id="e",
            metric="m",
            group_id=group_id,
            n=n,
            ref_y=ref_y,
            cy1=float((g - ref_y).sum()),
            cy2=float(((g - ref_y) ** 2).sum()),
            ref_x=ref_x,
            cx1=float((xd - ref_x).sum()),
            cx2=float(((xd - ref_x) ** 2).sum()),
            cxy=float(((xd - ref_x) * (g - ref_y)).sum()),
            ref_den=ref_den,
            cden1=float((m - ref_den).sum()),
            cden2=float(((m - ref_den) ** 2).sum()),
            cyden=float(((g - ref_y) * (m - ref_den)).sum()),
            cxden=float(((xd - ref_x) * (m - ref_den)).sum()),
            x_role="uptake_total",
        )

    rng = np.random.default_rng(29)
    # Control mean is NEGATIVE (a profit-like metric can be); treatment
    # mean stays modest so the Wald ratio (a/b) is finite and sane.
    ys_c = list(rng.normal(-8.0, 2.0, size=14))
    ds_c = [1 if rng.random() < 0.3 else 0 for _ in range(14)]
    ys_t = list(rng.normal(-3.0, 2.0, size=14))
    ds_t = [1 if rng.random() < 0.7 else 0 for _ in range(14)]
    assert sum(ys_c) < 0.0 and sum(ys_t) < 0.0  # the guard this pins would trip on both

    def unit_arm(group_id: str, ys: list[float], ds: list[int]) -> ArmStats:
        sum_y = sum(ys)
        sum_d = float(sum(ds))
        return ArmStats.from_raw_sums(
            study_id="e",
            metric="m",
            group_id=group_id,
            n=len(ys),
            sum_y=sum_y,
            sum_y2=sum(y * y for y in ys),
            sum_d=sum_d,
            sum_yd=sum(y * d for y, d in zip(ys, ds, strict=True)),
            sum_y2d=sum(y * y * d for y, d in zip(ys, ds, strict=True)),
        )

    c_unit, t_unit = unit_arm("C", ys_c, ds_c), unit_arm("T", ys_t, ds_t)
    c_cl, t_cl = singleton_cluster_arm("C", ys_c, ds_c), singleton_cluster_arm("T", ys_t, ds_t)

    tau_unit, se_unit = _late_additive(t_unit, c_unit)
    tau_cl, se_cl, _dof_cl = _late_additive_cluster(t_cl, c_cl)  # must not raise
    assert tau_cl == pytest.approx(tau_unit, rel=1e-9)
    assert se_cl == pytest.approx(se_unit, rel=1e-9)


# Hand-computed clustered reduction, tiny K, pinned against an
# independent (Cochran ratio-linearization) computation.


def _cluster_arm(group_id: str, g: list[float], d: list[float], m: list[float]) -> ArmStats:
    """Build a cluster-grain ArmStats from per-cluster (y-sum, uptake-sum,
    size) triples - exactly the shape ``group_summary(cluster=...)``
    emits."""
    G, D, M = np.array(g), np.array(d), np.array(m)
    n = len(g)
    ref_y, ref_x, ref_den = G.mean(), D.mean(), M.mean()
    return ArmStats(
        study_id="e",
        metric="m",
        group_id=group_id,
        n=n,
        ref_y=ref_y,
        cy1=float((G - ref_y).sum()),
        cy2=float(((G - ref_y) ** 2).sum()),
        ref_x=ref_x,
        cx1=float((D - ref_x).sum()),
        cx2=float(((D - ref_x) ** 2).sum()),
        cxy=float(((D - ref_x) * (G - ref_y)).sum()),
        ref_den=ref_den,
        cden1=float((M - ref_den).sum()),
        cden2=float(((M - ref_den) ** 2).sum()),
        cyden=float(((G - ref_y) * (M - ref_den)).sum()),
        cxden=float(((D - ref_x) * (M - ref_den)).sum()),
        x_role="uptake_total",
    )


# control:   4 clusters, G=(30,50,20,40), D=(1,2,0,1) takers, M=(3,5,2,4) units
# treatment: 4 clusters, G=(60,90,40,100), D=(3,4,2,5) takers, M=(4,5,3,6) units
_HAND_C = _cluster_arm("C", [30.0, 50.0, 20.0, 40.0], [1.0, 2.0, 0.0, 1.0], [3.0, 5.0, 2.0, 4.0])
_HAND_T = _cluster_arm("T", [60.0, 90.0, 40.0, 100.0], [3.0, 4.0, 2.0, 5.0], [4.0, 5.0, 3.0, 6.0])
# Pinned via independent Cochran-ratio-linearization (R = sum(N)/sum(M),
# Var(R) via psi=N-R*M); see _late_additive_cluster's derivation notes.
_HAND_TAU = 12.419354838709676
_HAND_SE = 2.140525292262549
_HAND_B = 0.4920634920634921
_HAND_VAR_B = 0.005736360717624234


def test_cluster_late_matches_hand_computed_k4():
    tau, se, _dof = _late_additive_cluster(_HAND_T, _HAND_C)
    assert tau == pytest.approx(_HAND_TAU, rel=1e-9)
    assert se == pytest.approx(_HAND_SE, rel=1e-9)


def test_cluster_first_stage_matches_hand_computed_k4():
    b, var_b, _dof = _first_stage_cluster(_HAND_T, _HAND_C)
    assert b == pytest.approx(_HAND_B, rel=1e-9)
    assert var_b == pytest.approx(_HAND_VAR_B, rel=1e-9)


def test_cluster_size_row_refused_by_late_reduction():
    """A clustered summary built with NO uptake declaration carries cluster
    SIZE in the x family, marked x_role="cluster_size". Feeding it to the
    cluster-robust LATE first stage must RAISE: with x == size the takeup
    ratio mean_x/mean_den would be size/size = 1.0, silently collapsing
    LATE to ITT. The declaration is what tells the reduction the x family
    is not an uptake total."""
    M = [3.0, 5.0, 2.0, 4.0]
    Mv = np.array(M)
    ref = Mv.mean()
    # x carries cluster size (x == m), the shape a non-uptake clustered
    # collapse emits - distinguishable ONLY by the x_role declaration.
    size_arm = ArmStats(
        study_id="e",
        metric="m",
        group_id="C",
        n=len(M),
        ref_y=30.0,
        cy1=0.0,
        cy2=100.0,
        ref_x=ref,
        cx1=float((Mv - ref).sum()),
        cx2=float(((Mv - ref) ** 2).sum()),
        cxy=0.0,
        ref_den=ref,
        cden1=float((Mv - ref).sum()),
        cden2=float(((Mv - ref) ** 2).sum()),
        cyden=0.0,
        cxden=float(((Mv - ref) ** 2).sum()),
        x_role="cluster_size",
    )
    with pytest.raises(InvalidRequestError) as raised:
        cluster_uptake_moments(size_arm)
    assert raised.value.code == "estimation.variance.cluster_robust_late"
    assert raised.value.context["x_role"] == "cluster_size"
    with pytest.raises(InvalidRequestError) as raised:
        _first_stage_cluster(size_arm, size_arm)
    assert raised.value.code == "estimation.variance.cluster_robust_late"
    assert raised.value.context["x_role"] == "cluster_size"

    # A copied row may retain the non-uptake x moments after ref_x is lost;
    # it must keep the declaration refusal rather than leaking ArmStats' view
    # assertion.
    partial_size = size_arm.model_copy(update={"ref_x": None})
    with pytest.raises(InvalidRequestError) as raised:
        cluster_uptake_moments(partial_size)
    assert raised.value.code == "estimation.variance.cluster_robust_late"
    assert raised.value.context["x_role"] == "cluster_size"


@pytest.mark.parametrize(
    ("update", "code"),
    [
        ({"x_role": "bogus"}, "estimation.variance.cluster_robust_late"),
        (
            {"ref_x": None, "cx1": None, "cx2": None, "cxy": None, "cxden": None},
            "estimation.variance.cluster_robust_late_needs_cluster_uptake_family",
        ),
    ],
)
def test_late_reduction_refuses_a_declaration_its_x_family_does_not_carry(update, code):
    """A row copied past validation may declare a role its x family cannot
    name; the reduction refuses on the declaration rather than crashing."""
    M = np.array([3.0, 5.0, 2.0, 4.0])
    D = np.array([1.0, 2.0, 0.0, 3.0])
    uptake_arm = ArmStats(
        study_id="e",
        metric="m",
        group_id="T",
        n=len(M),
        ref_y=30.0,
        cy1=0.0,
        cy2=100.0,
        ref_x=D.mean(),
        cx1=0.0,
        cx2=float(((D - D.mean()) ** 2).sum()),
        cxy=0.0,
        ref_den=M.mean(),
        cden1=0.0,
        cden2=float(((M - M.mean()) ** 2).sum()),
        cyden=0.0,
        cxden=float(((D - D.mean()) * (M - M.mean())).sum()),
        x_role="uptake_total",
    )
    cluster_uptake_moments(uptake_arm)
    with pytest.raises(InvalidRequestError) as raised:
        cluster_uptake_moments(uptake_arm.model_copy(update=update))
    assert raised.value.code == code


# Coverage simulation: ICC>0 clustered-vs-iid LATE coverage.

_MU = 5.0
_TAU = 2.0
_P = 0.5
_SIGMA_B = 1.0
_SIGMA_E = 1.0


def _arm_rows(rng: np.random.Generator, group: str, k: int, m: int, treated: bool):
    """One arm's (unit-grain, cluster-grain) format-2 row pair for a
    one-sided encouragement DGP with a per-cluster outcome shock b_j
    (ICC>0): y = mu + b_j + tau*d + e, d ~ Bernoulli(p) i.i.d. in
    treatment, d==0 structurally in control. The exclusion restriction
    holds exactly (non-compliers share the control's y distribution), so
    the true additive LATE is exactly ``tau``."""
    g = rng.normal(0.0, _SIGMA_B, size=k)
    cluster_id = np.repeat(np.arange(k), m)
    n = k * m
    d = (rng.random(n) < _P).astype(float) if treated else np.zeros(n)
    e = rng.normal(0.0, _SIGMA_E, size=n)
    y = _MU + g[cluster_id] + _TAU * d + e

    sum_y, sum_y2 = float(y.sum()), float((y**2).sum())
    sum_d = float(d.sum())
    sum_yd = float((y * d).sum())
    sum_y2d = float((y * y * d).sum())
    unit_row = centered_row_from_raw_sums(
        {
            "experiment_id": "e",
            "metric": "m",
            "group_id": group,
            "n": n,
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_d": sum_d,
            "sum_yd": sum_yd,
            "sum_y2d": sum_y2d,
        }
    )

    G = np.array([y[cluster_id == j].sum() for j in range(k)])
    D = np.array([d[cluster_id == j].sum() for j in range(k)])
    M = np.full(k, float(m))
    ref_y, ref_den, ref_x = G.mean(), M.mean(), D.mean()
    cluster_row = {
        "experiment_id": "e",
        "metric": "m",
        "group_id": group,
        "n": k,
        "ref_y": ref_y,
        "cy1": float((G - ref_y).sum()),
        "cy2": float(((G - ref_y) ** 2).sum()),
        "ref_x": ref_x,
        "cx1": float((D - ref_x).sum()),
        "cx2": float(((D - ref_x) ** 2).sum()),
        "cxy": float(((D - ref_x) * (G - ref_y)).sum()),
        "ref_den": ref_den,
        "cden1": float((M - ref_den).sum()),
        "cden2": float(((M - ref_den) ** 2).sum()),
        "cyden": float(((G - ref_y) * (M - ref_den)).sum()),
        "cxden": float(((D - ref_x) * (M - ref_den)).sum()),
        "x_role": "uptake_total",
    }
    return unit_row, cluster_row


def _coverage(reps: int, k: int, m: int, seed: int) -> tuple[float, float]:
    """(clustered, iid) coverage of the true additive LATE (``_TAU``) over
    *reps* draws."""
    rng = np.random.default_rng(seed)
    covset = CoverageSet()
    for _ in range(reps):
        uc, cc = _arm_rows(rng, "control", k, m, treated=False)
        ut, ct = _arm_rows(rng, "treatment", k, m, treated=True)
        flat = estimate_encouragement([METRIC], [uc, ut], DESIGN, estimands=("late",)).results
        clustered = estimate_encouragement(
            [METRIC], [cc, ct], DESIGN, estimands=("late",), cluster="store"
        ).results
        f = next(r for r in flat if r.value_scale == "absolute")
        c = next(r for r in clustered if r.value_scale == "absolute")
        f_lift = f.require_lift()
        c_lift = c.require_lift()
        assert f_lift.lb is not None and f_lift.ub is not None
        assert c_lift.lb is not None and c_lift.ub is not None
        covset.record(
            iid=f_lift.lb <= _TAU <= f_lift.ub,
            clustered=c_lift.lb <= _TAU <= c_lift.ub,
        )
    clustered_rate, iid_rate = covset.rates("clustered", "iid")
    return clustered_rate, iid_rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_clustered_late_cis_are_nominal_while_iid_undercovers():
    clustered, iid = _coverage(reps=400, k=40, m=20, seed=7)
    # Nominal 95%: binomial noise at 400 reps is ~+/-2pp.
    assert 0.90 <= clustered <= 0.99, clustered
    assert iid < 0.60, iid


def test_clustered_late_cis_beat_iid_coverage_smoke():
    """Small-N smoke twin of the parameter_recovery check above.

    Bounds are ``nominal +/- k*mcse(nominal, reps)`` (see tests/mc.py):
    0.95 for clustered (matches the parameter_recovery sibling's nominal
    target), 0.4375 for iid (this DGP's large-N iid coverage, from the
    same sibling). k=3 gives ~3x binomial-noise headroom.
    """
    reps, k = 30, 3.0
    clustered, iid = _coverage(reps=reps, k=25, m=20, seed=3)
    assert clustered >= 0.95 - k * mcse(0.95, reps), clustered
    assert iid <= 0.4375 + k * mcse(0.4375, reps), iid
    assert clustered > iid


# Refusals: kept, and message-pinned.


def _cluster_table() -> pa.Table:
    rng = np.random.default_rng(5)
    rows = []
    for j in range(20):
        for _ in range(6):
            rows.append(
                {
                    "user_id": f"c{len(rows)}",
                    "variant": "control",
                    "store_id": f"cs{j}",
                    "revenue": 5.0 + float(rng.normal(0, 1)),
                    "revenue_pre": float(rng.normal(0, 1)),
                    "took": 0,
                }
            )
    for j in range(20):
        for _ in range(6):
            d = 1 if rng.random() < 0.6 else 0
            rows.append(
                {
                    "user_id": f"t{len(rows)}",
                    "variant": "treatment",
                    "store_id": f"ts{j}",
                    "revenue": 5.0 + 2.0 * d + float(rng.normal(0, 1)),
                    "revenue_pre": float(rng.normal(0, 1)),
                    "took": d,
                }
            )
    return pa.Table.from_pylist(rows)


def _native_cluster_analysis(*, n_pre_periods: int = 0, ratio: bool = False, design=DESIGN):
    """Small definitions-backed source for public native guard coverage."""
    rows = []
    for arm, store_prefix in (("control", "c"), ("treatment", "t")):
        for i in range(10):
            user_id = f"{store_prefix}{i}"
            store_id = f"{store_prefix}s{i}"
            rows.extend(
                [
                    {
                        "user_id": user_id,
                        "event_at": datetime(2025, 1, 1, 9),
                        "event": "exposure",
                        "group_id": arm,
                        "experiment_id": "native_late",
                        "store_id": store_id,
                        "revenue": None,
                        "took": None,
                    },
                    {
                        "user_id": user_id,
                        "event_at": datetime(2025, 1, 2, 9),
                        "event": "purchase",
                        "group_id": None,
                        "experiment_id": None,
                        "store_id": None,
                        "revenue": 5.0 + (arm == "treatment"),
                        "took": None,
                    },
                    {
                        "user_id": user_id,
                        "event_at": datetime(2025, 1, 2, 10),
                        "event": "session_end",
                        "group_id": None,
                        "experiment_id": None,
                        "store_id": None,
                        "revenue": None,
                        "took": None,
                    },
                ]
            )
            if arm == "treatment" and i % 2 == 0:
                rows.append(
                    {
                        "user_id": user_id,
                        "event_at": datetime(2025, 1, 2, 11),
                        "event": "took",
                        "group_id": None,
                        "experiment_id": None,
                        "store_id": None,
                        "revenue": None,
                        "took": 1,
                    }
                )
    con = ibis.duckdb.connect()
    con.create_table("native_late_events", obj=pa.Table.from_pylist(rows))
    metric = (
        {
            "type": "ratio",
            "name": "rpo",
            "entity": "user_id",
            "numerator": {"fact": "purchase", "aggregation": "sum"},
            "denominator": {"fact": "session_end", "aggregation": "count"},
        }
        if ratio
        else {
            "type": "mean",
            "name": "revenue",
            "entity": "user_id",
            "fact": "purchase",
            "aggregation": "sum",
        }
    )
    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM native_late_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "session_end", "column": None},
                        {"name": "took", "column": "took"},
                    ],
                }
            ],
            "exposures": [{"name": "assignment", "fact": "exposure"}],
            "metrics": [metric],
            "experiments": [
                {
                    "name": "native_late",
                    "exposure": "assignment",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "cluster": "store_id",
                    "n_pre_periods": n_pre_periods,
                    "plan": {"secondaries": [metric["name"]]},
                }
            ],
        }
    )
    return make_analysis(con, defs, experiment="native_late", _design=design)


def test_cluster_with_cuped_method_refuses():
    """CUPED stays refused under a declared cluster - named up front,
    never falling through to a wrong number."""
    tbl = _cluster_table()
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=DESIGN,
        uptake="took",
        cluster="store_id",
    )
    rows = list(src.raw_moments)
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            rows,
            DESIGN,
            estimands=("late",),
            methods=[Method(name="cuped", variance_reduction="cuped")],
            cluster="store_id",
        )
    assert raised.value.code == "arm.adjustment.cluster_cuped"


def test_cluster_with_covariate_and_encouragement_still_refuses_at_frame_entry():
    """The CUPED covariate itself is refused at the frame-entry gate too
    (not just inside the estimator), mirroring the randomized path."""
    tbl = _cluster_table()
    with pytest.raises(CapabilityError) as raised:
        from_unit_summary(
            tbl,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", type="mean", covariate="revenue_pre")],
            design=DESIGN,
            uptake="took",
            cluster="store_id",
        )
    assert raised.value.code == "source.frame.cluster_capability"


def test_cluster_with_ratio_metric_and_encouragement_refuses_at_frame_entry():
    tbl = _cluster_table()
    tbl = tbl.append_column("sessions", pa.array([1.0] * tbl.num_rows))
    with pytest.raises(CapabilityError) as raised:
        from_unit_summary(
            tbl,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="sessions")
            ],
            design=DESIGN,
            uptake="took",
            cluster="store_id",
        )
    assert raised.value.code == "source.frame.cluster_capability"


def test_cluster_with_ratio_metric_and_encouragement_refuses_in_the_estimator():
    """The frame gate above only guards the from_unit_summary seam. A caller
    holding clustered moment rows can reach estimate_encouragement directly
    (including estimands=("itt",), which would otherwise slip through
    estimate_lift now that clustered ratio metrics are servable there), so
    the refusal is pinned at the estimator too."""
    from increment.semantics.models import RatioMetric as _RatioMetric

    ratio = _RatioMetric(
        name="revenue",
        entity="user_id",
        numerator={"fact": "purchase", "aggregation": "sum"},
        denominator={"fact": "session", "aggregation": "count"},
    )
    tbl = _cluster_table()
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=DESIGN,
        uptake="took",
        cluster="store_id",
    )
    for estimands in (("itt",), ("late",), ("itt", "late")):
        with pytest.raises(CapabilityError) as raised:
            estimate_encouragement(
                [ratio],
                list(src.raw_moments),
                DESIGN,
                estimands=estimands,
                cluster="store_id",
            )
        assert raised.value.code == "estimation.encouragement.cluster.ratio"


def test_native_encouragement_cluster_refuses_cuped_by_pre_periods():
    """Public panel SQL preserves the native CUPED/cluster refusal."""
    analysis = _native_cluster_analysis(n_pre_periods=14)
    with pytest.raises(CapabilityError) as raised:
        analysis.panel_sql()
    assert raised.value.code == "source.native.operation"


def test_native_encouragement_cluster_refuses_ratio_metric():
    """Public panel SQL preserves the native ratio/cluster refusal."""
    analysis = _native_cluster_analysis(ratio=True)
    with pytest.raises(CapabilityError) as raised:
        analysis.panel_sql()
    assert raised.value.code == "source.native.operation"


def test_cluster_with_informative_prior_refuses():
    tbl = _cluster_table()
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=DESIGN,
        uptake="took",
        cluster="store_id",
    )
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            list(src.raw_moments),
            DESIGN,
            estimands=("late",),
            cluster="store_id",
            prior=Normal(mu=0.0, sigma=0.1),
        )
    assert raised.value.code == "arm.adjustment.cluster_prior"


def test_cluster_with_sequential_inference_refuses():
    tbl = _cluster_table()
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=DESIGN,
        uptake="took",
        cluster="store_id",
    )
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            list(src.raw_moments),
            DESIGN,
            estimands=("late",),
            cluster="store_id",
            inference=AlwaysValid(registration=registration("gaussian")),
        )

    assert raised.value.code == "sequential.route.unsupported"


def test_cluster_relative_late_is_withheld_not_computed():
    """The complier-relative LATE row is withheld under cluster (needs a
    moment family this reduction does not build), named on the additive
    row's note rather than silently omitted or wrong."""
    tbl = _cluster_table()
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=DESIGN,
        uptake="took",
        cluster="store_id",
    )
    out = estimate_encouragement(
        [METRIC], list(src.raw_moments), DESIGN, estimands=("late",), cluster="store_id"
    ).results
    late_rows = [r for r in out if r.estimand == "late"]
    assert len(late_rows) == 1  # relative row never separately emitted
    assert late_rows[0].value_scale == "absolute"
    note = late_rows[0].note
    assert note is not None
    assert "relative form withheld" in note
    assert "cluster-robust relative" in note


def test_below_ten_total_clusters_is_admitted_with_warning():
    rng = np.random.default_rng(9)
    uc, cc = _arm_rows(rng, "control", k=3, m=10, treated=False)
    ut, ct = _arm_rows(rng, "treatment", k=3, m=10, treated=True)
    with pytest.warns(IncrementRuntimeWarning) as rec:
        rows = estimate_encouragement(
            [METRIC], [cc, ct], DESIGN, estimands=("late",), cluster="store"
        )
    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    assert rows


def test_small_cluster_warning_attributed_to_caller():
    """The small-K advisory must name the caller of estimate_encouragement,
    not an internal helper frame."""
    rng = np.random.default_rng(11)
    _, cc = _arm_rows(rng, "control", k=8, m=10, treated=False)
    _, ct = _arm_rows(rng, "treatment", k=8, m=10, treated=True)
    with pytest.warns(IncrementRuntimeWarning) as caught:
        estimate_encouragement([METRIC], [cc, ct], DESIGN, estimands=("late",), cluster="store")
    advisories = [
        w
        for w in caught
        if getattr(w.message, "code", None) == "estimation.engine.small_total_clusters"
    ]
    assert advisories
    assert all(w.filename == __file__ for w in advisories)
