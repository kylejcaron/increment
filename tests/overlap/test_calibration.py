"""Monte-Carlo recovery of the joint-cell contrasts and their planning.

Pre-registered before any outcome was inspected:

* Randomization: independent Bernoulli assignment of every jointly eligible unit
  at the declared probabilities (0.5, 0.25), outcomes normal with cell means
  ``MU`` and cell standard deviations ``SDS``; one ``numpy`` stream per
  replication seeded ``seed0 + r``.
* Statistic: coverage of the two-sided 95% Welch interval for every contrast,
  and the replication mean of each estimate.
* Tolerance: coverage inside ``nominal_band(0.95, reps, k=3)``; bias within
  ``3 * sd / sqrt(reps)``. Power: simulated rejection within 3 MCSE of the
  analytic normal-approximation power at its MDE, at a size where cells are
  never sparse.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from increment._overlap.admission import Admission, EligibilityAudit
from increment._overlap.frame import cell_moments
from increment.overlap import CONTRASTS, OverlapCells, estimate_overlap_effects
from increment.power import overlap_minimum_detectable_effect, simulate_overlap_power
from tests.mc import Coverage, nominal_band
from tests.overlap.fixtures import DESIGN

MU = {(0, 0): 10.0, (1, 0): 10.4, (0, 1): 9.8, (1, 1): 11.1}
SDS = {(0, 0): 1.0, (1, 0): 2.0, (0, 1): 1.5, (1, 1): 2.5}
P1, P2 = 0.5, 0.25
TRUTH = {
    "interaction": MU[(1, 1)] - MU[(1, 0)] - MU[(0, 1)] + MU[(0, 0)],
    "joint_launch": MU[(1, 1)] - MU[(0, 0)],
    "first_only_launch": MU[(1, 0)] - MU[(0, 0)],
    "second_only_launch": MU[(0, 1)] - MU[(0, 0)],
    "first_given_second_treatment": MU[(1, 1)] - MU[(0, 1)],
    "second_given_first_treatment": MU[(1, 1)] - MU[(1, 0)],
    "marginal_first": P2 * (MU[(1, 1)] - MU[(0, 1)]) + (1 - P2) * (MU[(1, 0)] - MU[(0, 0)]),
    "marginal_second": P1 * (MU[(1, 1)] - MU[(1, 0)]) + (1 - P1) * (MU[(0, 1)] - MU[(0, 0)]),
}


def _replicate(rng: np.random.Generator, n_total: int) -> OverlapCells:
    first = (rng.random(n_total) < P1).astype(np.int8)
    second = (rng.random(n_total) < P2).astype(np.int8)
    index = first + 2 * second
    means = np.array([MU[c] for c in ((0, 0), (1, 0), (0, 1), (1, 1))])
    sds = np.array([SDS[c] for c in ((0, 0), (1, 0), (0, 1), (1, 1))])
    y = means[index] + sds[index] * rng.standard_normal(n_total)
    audit = EligibilityAudit(
        units=n_total,
        jointly_eligible=n_total,
        first_only_eligible=0,
        second_only_eligible=0,
        neither_eligible=0,
    )
    return OverlapCells._admit(
        Admission(
            DESIGN, ("y",), cell_moments(DESIGN, "y", first, second, y), audit, dt.timedelta(0)
        )
    )


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("n_total", [240, 2000])
def test_every_contrast_interval_covers_at_nominal_and_is_unbiased(n_total: int) -> None:
    reps, seed0 = 1500, 830_000 + n_total
    coverage = {c: Coverage() for c in CONTRASTS}
    estimates: dict[str, list[float]] = {c: [] for c in CONTRASTS}
    for r in range(reps):
        readout = estimate_overlap_effects(
            _replicate(np.random.default_rng(seed0 + r), n_total), contrasts=CONTRASTS
        )
        for row in readout.rows:
            assert row.available, (r, row.contrast, row.unavailable_reason)
            assert row.estimate is not None and row.lb is not None and row.ub is not None
            coverage[row.contrast].record(row.lb <= TRUTH[row.contrast] <= row.ub)
            estimates[row.contrast].append(row.estimate)
    low, high = nominal_band(0.95, reps, k=3)
    for contrast in CONTRASTS:
        assert low <= coverage[contrast].rate <= high, (contrast, coverage[contrast].rate)
        values = np.array(estimates[contrast])
        assert abs(values.mean() - TRUTH[contrast]) <= 3 * values.std(ddof=1) / math.sqrt(reps)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("contrast", ["interaction", "joint_launch"])
def test_simulated_power_matches_the_analytic_mde(contrast: str) -> None:
    n_total, reps, family_size = 1600, 2000, 2
    mde = overlap_minimum_detectable_effect(
        DESIGN, contrast, n_total=n_total, sd=2.0, power=0.8, family_size=family_size
    )
    means = {(0, 0): 0.0, (1, 0): 0.0, (0, 1): 0.0, (1, 1): mde}
    result = simulate_overlap_power(
        DESIGN,
        contrast,
        cell_means=means,
        n_total=n_total,
        sd=2.0,
        family_size=family_size,
        replications=reps,
        seed=8300,
    )
    assert result.availability_rate == 1.0
    assert abs(result.rejection_rate - 0.8) <= 3 * math.sqrt(0.8 * 0.2 / reps)
