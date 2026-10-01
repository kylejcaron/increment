"""Coverage of cluster-robust CIs for a RATIO metric under intra-cluster
correlation - the ratio twin of tests/estimation/test_cluster_coverage.py.

DGP: unit i in cluster j has denominator d_ij (1 + Poisson) and numerator
y_ij = d_ij * (mu + b_j) + e_ij, with b_j ~ N(0, sigma_b^2) per cluster and
e_ij ~ N(0, sigma_e^2) per unit. Both arms share mu, so the true lift of the
estimand sum(y)/sum(d) is 0. A clustered row carries (num_j, den_j, K); the
iid row carries the same unit-grain (y, d) pairs at n = K*m. Only the
clustered reading sees the b_j variance, so only it stays near nominal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mc import CoverageSet, mcse

MU = 5.0
SIGMA_B = 0.5
SIGMA_E = 0.5


def _coverage(reps: int, k_per_arm: int, m: int, seed: int, mu: float = MU) -> tuple[float, float]:
    """(clustered, iid) coverage of the true lift 0 over *reps* draws."""
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import RatioMetric

    metric = RatioMetric(
        name="m",
        entity="u",
        numerator={"fact": "f", "aggregation": "sum"},
        denominator={"fact": "g", "aggregation": "sum"},
    )
    rng = np.random.default_rng(seed)
    covset = CoverageSet()

    def _raw(group: str, n: int, num, den) -> dict:
        return centered_row_from_raw_sums(
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": group,
                "n": n,
                "sum_y": float(num.sum()),
                "sum_y2": float((num * num).sum()),
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": float(den.sum()),
                "sum_den2": float((den * den).sum()),
                "sum_yden": float((num * den).sum()),
            }
        )

    def _rows(group: str) -> tuple[dict, dict]:
        b = rng.normal(0.0, SIGMA_B, size=k_per_arm)
        d = 1.0 + rng.poisson(2.0, size=k_per_arm * m).astype(float)
        y = d * (mu + np.repeat(b, m)) + rng.normal(0.0, SIGMA_E, size=k_per_arm * m)
        # The clustered collapse: num_j and den_j are the CLUSTER totals of
        # the metric's own numerator and denominator, n = K.
        num_j = y.reshape(k_per_arm, m).sum(axis=1)
        den_j = d.reshape(k_per_arm, m).sum(axis=1)
        return _raw(group, y.size, y, d), _raw(group, k_per_arm, num_j, den_j)

    for _ in range(reps):
        unit_c, cl_c = _rows("C")
        unit_t, cl_t = _rows("T")
        (flat,) = estimate_lift([metric], [unit_c, unit_t], control_group="C").results
        (clustered,) = estimate_lift(
            [metric], [cl_c, cl_t], control_group="C", cluster="store"
        ).results
        clustered_lift = clustered.require_lift()
        assert clustered_lift.lb is not None and clustered_lift.ub is not None
        if flat.lift is None:
            # The log-scale positivity guard refuses the unclustered route for a
            # negative arm mean, so "iid" counts a miss; the clustered Fieller
            # route under test always yields a lift.
            iid_covered = False
        else:
            flat_lift = flat.require_lift()
            assert flat_lift.lb is not None and flat_lift.ub is not None
            iid_covered = flat_lift.lb <= 0.0 <= flat_lift.ub
        covset.record(
            iid=iid_covered,
            clustered=clustered_lift.lb <= 0.0 <= clustered_lift.ub,
        )
    clustered_rate, iid_rate = covset.rates("clustered", "iid")
    return clustered_rate, iid_rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_clustered_ratio_cis_are_nominal_while_iid_undercovers():
    clustered, iid = _coverage(reps=400, k_per_arm=40, m=20, seed=7)
    # Nominal 95%: binomial noise at 400 reps is ~+/-2pp.
    assert 0.92 <= clustered <= 0.98, clustered
    # The cluster effect enters the ratio through d_ij * b_j, so the iid
    # ratio interval misses nearly all of the between-cluster variance.
    assert iid < 0.70, iid


def test_clustered_ratio_cis_beat_iid_coverage_smoke():
    """Small-N smoke twin of the parameter_recovery check above.

    Bounds are ``nominal +/- k*mcse(nominal, reps)`` (see tests/mc.py):
    0.95 for clustered, 0.36 for iid (this DGP's large-N iid coverage, see
    the parameter_recovery sibling above). k=3 gives ~3x binomial-noise
    headroom.
    """
    reps, k = 25, 3.0
    clustered, iid = _coverage(reps=reps, k_per_arm=20, m=10, seed=11)
    assert clustered >= 0.95 - k * mcse(0.95, reps), clustered
    assert iid <= 0.36 + k * mcse(0.36, reps), iid
    assert clustered > iid


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_clustered_ratio_cis_are_nominal_with_a_negative_control_mean():
    """docs/limitations.md:270-286 promises a signed control mean on the
    clustered route; this is the coverage proof, not just an admission
    check -- both arms share mu=-5.0 so the true lift is still 0, but
    every control-arm cluster total (num_j) is now negative, the exact
    case ratio_moments used to refuse before it reached the Fieller
    construction that never needed the check."""
    clustered, iid = _coverage(reps=400, k_per_arm=40, m=20, seed=17, mu=-5.0)
    assert 0.92 <= clustered <= 0.98, clustered


def test_clustered_ratio_negative_control_mean_produces_a_fieller_set_smoke():
    """Small-N smoke twin: asserts admission (a Fieller set is produced,
    not refused) rather than tight coverage."""
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import RatioMetric

    metric = RatioMetric(
        name="m",
        entity="u",
        numerator={"fact": "f", "aggregation": "sum"},
        denominator={"fact": "g", "aggregation": "sum"},
    )
    rng = np.random.default_rng(23)
    k, m = 20, 10

    def _rows(group: str, mu: float) -> dict:
        d = 1.0 + rng.poisson(2.0, size=k * m).astype(float)
        y = d * mu + rng.normal(0.0, SIGMA_E, size=k * m)
        num_j = y.reshape(k, m).sum(axis=1)
        den_j = d.reshape(k, m).sum(axis=1)
        return centered_row_from_raw_sums(
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": group,
                "n": k,
                "sum_y": float(num_j.sum()),
                "sum_y2": float((num_j * num_j).sum()),
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": float(den_j.sum()),
                "sum_den2": float((den_j * den_j).sum()),
                "sum_yden": float((num_j * den_j).sum()),
            }
        )

    control = _rows("C", mu=-5.0)
    treatment = _rows("T", mu=2.0)
    (result,) = estimate_lift(
        [metric], [control, treatment], control_group="C", cluster="store"
    ).results
    assert result.relative_confidence_set is not None
    assert result.relative_unavailable_reason is None
