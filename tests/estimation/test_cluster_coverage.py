"""Coverage of cluster-robust CIs under intra-cluster correlation.

DGP: y_ij = mu + b_j + e_ij with b_j ~ N(0, sigma_b^2) per cluster and
e_ij ~ N(0, sigma_e^2) per unit; true lift is 0 (both arms share mu). With
sigma_b = sigma_e the ICC is 0.5 and the design effect at m units/cluster
is 1 + (m-1)*0.5 - the iid interval is ~sqrt(design effect) too narrow,
while the clustered t interval stays ~nominal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mc import CoverageSet, mcse

MU = 5.0
SIGMA_B = 0.5
SIGMA_E = 0.5


def _coverage(reps: int, k_per_arm: int, m: int, seed: int) -> tuple[float, float]:
    """(clustered, iid) coverage of the true lift 0 over *reps* draws."""
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="m", entity="u", fact="f", aggregation="sum")
    rng = np.random.default_rng(seed)
    covset = CoverageSet()

    def _rows(group: str) -> tuple[dict, dict]:
        b = rng.normal(0.0, SIGMA_B, size=k_per_arm)
        y = MU + np.repeat(b, m) + rng.normal(0.0, SIGMA_E, size=k_per_arm * m)
        g = y.reshape(k_per_arm, m).sum(axis=1)
        base = {"experiment_id": "e", "metric": "m", "group_id": group}
        ref_y = float(y.mean())
        unit = {
            **base,
            "n": y.size,
            "ref_y": ref_y,
            "cy1": float(np.sum(y - ref_y)),
            "cy2": float(np.sum((y - ref_y) ** 2)),
        }
        # The clustered collapse: g_j is the y family, the constant cluster
        # size m the den family - so every den-centered moment is exactly 0.
        ref_g = float(g.mean())
        cluster = {
            **base,
            "n": k_per_arm,
            "ref_y": ref_g,
            "cy1": float(np.sum(g - ref_g)),
            "cy2": float(np.sum((g - ref_g) ** 2)),
            "ref_den": float(m),
            "cden1": 0.0,
            "cden2": 0.0,
            "cyden": 0.0,
        }
        return unit, cluster

    for _ in range(reps):
        unit_c, cl_c = _rows("C")
        unit_t, cl_t = _rows("T")
        (flat,) = estimate_lift([metric], [unit_c, unit_t], control_group="C").results
        (clustered,) = estimate_lift(
            [metric], [cl_c, cl_t], control_group="C", cluster="store"
        ).results
        flat_lift = flat.require_lift()
        clustered_lift = clustered.require_lift()
        assert flat_lift.lb is not None and flat_lift.ub is not None
        assert clustered_lift.lb is not None and clustered_lift.ub is not None
        covset.record(
            iid=flat_lift.lb <= 0.0 <= flat_lift.ub,
            clustered=clustered_lift.lb <= 0.0 <= clustered_lift.ub,
        )
    clustered_rate, iid_rate = covset.rates("clustered", "iid")
    return clustered_rate, iid_rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_clustered_cis_are_nominal_while_iid_undercovers():
    clustered, iid = _coverage(reps=400, k_per_arm=40, m=20, seed=7)
    # Nominal 95%: binomial noise at 400 reps is ~+/-2pp.
    assert 0.92 <= clustered <= 0.98, clustered
    # Design effect 1 + 19*0.5 = 10.5 -> the iid interval is ~3.2x too
    # narrow; its coverage collapses far below anything noise explains.
    assert iid < 0.70, iid


def test_clustered_cis_beat_iid_coverage_smoke():
    """Small-N smoke twin of the parameter_recovery check above.

    Bounds are ``nominal +/- k*mcse(nominal, reps)`` (see tests/mc.py),
    not hand-tuned to the seed: k=3 gives ~3x binomial-noise headroom, so
    the check stays a real trip-wire without flaking on RNG variance.
    ``0.45`` is this DGP's large-N iid coverage (see the parameter_recovery
    sibling above; the design effect collapses it well below nominal).
    """
    reps, k = 25, 3.0
    clustered, iid = _coverage(reps=reps, k_per_arm=20, m=10, seed=11)
    assert clustered >= 0.95 - k * mcse(0.95, reps), clustered
    assert iid <= 0.45 + k * mcse(0.45, reps), iid
    assert clustered > iid
