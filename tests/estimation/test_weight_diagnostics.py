"""Weight diagnostics use the retained estimator weights, not arm counts or df."""

from __future__ import annotations

import numpy as np
import pytest

from increment.estimation._adjust.weight_diagnostics import _weight_summary


@pytest.mark.parametrize("scale", [1.0, 1e307, 1e-307])
def test_weight_summary_is_scale_invariant_without_squaring_raw_weights(scale):
    # W=6, sum(w²)=14: ESS=18/7, largest unit carries half the weight.
    n, ess, largest = _weight_summary(np.array([0.0, 1.0, 2.0, 3.0]) * scale)
    assert n == 3
    assert ess == pytest.approx(18 / 7)
    assert largest == pytest.approx(0.5)


def test_cluster_summary_uses_arm_weight_totals_not_unit_ess():
    # Cluster totals [3,3,0]: ESS=2, maximum share=1/2, two contributing clusters.
    n, ess, largest = _weight_summary(np.array([1.0, 2.0, 3.0, 0.0]), np.array([0, 0, 1, 2]), 3)
    assert (n, ess, largest) == pytest.approx((2, 2.0, 0.5))


def test_cluster_summary_scales_before_accumulating_near_overflow():
    n, ess, largest = _weight_summary(np.array([1e308, 1e308, 1e308]), np.array([0, 0, 1]), 2)
    assert n == 2
    assert ess == pytest.approx(9 / 5)
    assert largest == pytest.approx(2 / 3)
