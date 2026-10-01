"""Coverage of the adjusted estimators' cluster-robust CIs under ICC>0.

DGP (phased-geo shape): cluster j carries a covariate z_j and a random
effect b_j ~ N(0, sigma_b^2); assignment is at CLUSTER grain - in the
parameter-recovery runs it is confounded by z_j via P(T|z_j) =
expit(0.8 * z_j) - and units add e_ij ~ N(0, sigma_e^2). True lift is 0.
With sigma_b = sigma_e the ICC is 0.5 and the design effect at m
units/cluster is 1 + (m-1)*0.5: the iid IF interval is ~sqrt(DE) too
narrow, the clustered t interval stays ~nominal (mirroring
test_cluster_coverage.py for the randomized path).

The smoke twin runs unconfounded with a constant stub propensity and the
closed-form ridge outcome model, exercising the clustered-vs-iid seam for
all three estimators without per-rep scipy fits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.mc import CoverageSet, mcse

MU = 5.0
BETA_Z = 0.5
SIGMA_B = 0.5
SIGMA_E = 0.5


class _ConstantPropensity:
    """Stub learner pinned at the smoke DGP's true e=0.5; the class itself
    is the zero-arg factory DML/AIPW expect."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], 0.5)


def _sources(rng: np.random.Generator, k_total: int, m: int, *, confounded: bool):
    """(clustered, flat) sources over one draw of the clustered DGP."""
    import pyarrow as pa

    from increment.frame import from_unit_summary

    z = rng.normal(0.0, 1.0, size=k_total)
    if confounded:
        while True:
            treated = rng.random(k_total) < 1.0 / (1.0 + np.exp(-0.8 * z))
            if 2 <= treated.sum() <= k_total - 2:
                break
    else:
        treated = np.arange(k_total) % 2 == 1
    b = rng.normal(0.0, SIGMA_B, size=k_total)
    y = MU + BETA_Z * np.repeat(z, m) + np.repeat(b, m) + rng.normal(0.0, SIGMA_E, size=k_total * m)
    tbl = pa.table(
        {
            "u": [f"u{i}" for i in range(k_total * m)],
            "g": np.repeat(np.where(treated, "T", "C"), m),
            "store": np.repeat([f"s{j}" for j in range(k_total)], m),
            "y": y,
            "z": np.repeat(z, m),
        }
    )
    return (
        from_unit_summary(
            tbl, unit="u", group="g", control="C", metrics={"y": "mean"}, cluster="store"
        ),
        from_unit_summary(tbl, unit="u", group="g", control="C", metrics={"y": "mean"}),
    )


def _coverage(
    name: str, reps: int, k_total: int, m: int, seed: int, *, confounded: bool
) -> tuple[float, float]:
    """(clustered, iid) coverage of the true lift 0 over *reps* draws."""
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational

    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("z",)),
        gate=IdentificationGate(overlap="trim"),
    )
    # Smoke: stub the propensity at the DGP's true 0.5.
    method = Method(name=name, propensity_learner=None if confounded else _ConstantPropensity)

    rng = np.random.default_rng(seed)
    covset = CoverageSet()
    with warnings.catch_warnings():
        # The |SMD|>0.1 balance advisory is DGP noise at these sizes.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(reps):
            clustered_src, flat_src = _sources(rng, k_total, m, confounded=confounded)
            (cl,) = estimate_ate(clustered_src, design, methods=[method]).results
            (flat,) = estimate_ate(flat_src, design, methods=[method]).results
            cl_lift = cl.require_lift()
            flat_lift = flat.require_lift()
            assert cl_lift.lb is not None and cl_lift.ub is not None
            assert flat_lift.lb is not None and flat_lift.ub is not None
            covset.record(
                clustered=cl_lift.lb <= 0.0 <= cl_lift.ub,
                iid=flat_lift.lb <= 0.0 <= flat_lift.ub,
            )
    clustered_rate, iid_rate = covset.rates("clustered", "iid")
    return clustered_rate, iid_rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("name", ["iptw", "dml", "aipw"])
def test_clustered_adjusted_cis_are_nominal_while_iid_undercovers(name):
    clustered, iid = _coverage(name, reps=400, k_total=80, m=20, seed=7, confounded=True)
    # Nominal-or-conservative 95% (binomial noise ~+/-2pp); IPTW's plug-in
    # IF treats e as known, mildly conservative - undercoverage is the bug.
    assert 0.92 <= clustered <= 0.99, clustered
    # Design effect 1 + 19*0.5 = 10.5 -> the iid IF interval is ~3.2x too
    # narrow; coverage collapses far below anything noise explains.
    assert iid < 0.70, iid


@pytest.mark.parametrize("name", ["iptw", "dml", "aipw"])
def test_clustered_adjusted_cis_beat_iid_coverage_smoke(name):
    """Small-N unconfounded smoke twin of the parameter_recovery check.

    Bounds are ``nominal +/- k*mcse(nominal, reps)`` (see tests/mc.py):
    0.95 for clustered, 0.55 for iid (comfortably above every method's
    measured smoke-N iid coverage, ~0.52-0.56). k=3 gives ~3x
    binomial-noise headroom.
    """
    reps, k = 25, 3.0
    clustered, iid = _coverage(name, reps=reps, k_total=40, m=10, seed=11, confounded=False)
    assert clustered >= 0.95 - k * mcse(0.95, reps), clustered
    assert iid <= 0.55 + k * mcse(0.55, reps), iid
    assert clustered > iid
