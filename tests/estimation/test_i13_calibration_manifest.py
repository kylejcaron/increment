"""Representative cluster-randomized calibration contracts.

The original fixed-R gates run in the bounded campaign.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import pytest

from increment.errors import IncrementWarning
from tests.estimation._i13_adapters import estimate_of, truth_of
from tests.estimation._i13_calibration import (
    IntervalResult,
    prospective_plan,
    run_cell,
)
from tests.estimation._i13_manifest import (
    ACCEPTANCE_MANIFEST,
)
from tests.warning_codes import warning_codes


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ("mean", "ratio", "late", "sitewide", "iptw", "aipw", "dml"))
def test_adapter_smoke(estimator):
    cell = next(
        c
        for c in ACCEPTANCE_MANIFEST
        if c.estimator == estimator
        and c.design.dgp.k_t == 20
        and c.design.dgp.k_c == 20
        and c.nuisance in ("none", "oracle_propensity")
        and c.scale in ("absolute", "absolute_sidecar")
    )
    warning = pytest.warns(IncrementWarning) if estimator == "sitewide" else nullcontext()
    with warning as rec:
        result = run_cell(
            cell.design, 2, truth_of=partial(truth_of, cell), estimate_of=partial(estimate_of, cell)
        )
    if estimator == "sitewide":
        assert rec is not None
        assert "estimation.sitewide.cluster_baseline_assumption" in warning_codes(rec)
    assert result.failed == 0, result.failure_reasons
    assert result.point_estimable == 2, result.exclusion_reasons
    assert result.interval_estimable == 2, result.exclusion_reasons


def test_interval_accounting_distinguishes_point_and_open_set():
    assert IntervalResult(1, None, 2, "lower").contains(-100)
    assert IntervalResult(1, 0, None, "upper").contains(100)
    assert not IntervalResult(None, 0, 2).interval_available
    assert IntervalResult(1, None, None).point_available
    assert not IntervalResult(1, None, None).interval_available


@pytest.mark.slow
def test_prospective_design_uses_bound_precision_and_nominal_certification():
    from scipy.stats import binom

    from tests.mc import binomial_error_upper_bound

    plan = prospective_plan(eta=0.00005)
    assert plan.reps > 7600
    assert plan.boundary_margin <= 0.0025
    assert plan.nominal_false_failure <= 0.00005
    assert binomial_error_upper_bound(plan.passing_count, plan.reps, 0.00005) <= 0.055
    assert binomial_error_upper_bound(plan.passing_count + 1, plan.reps, 0.00005) > 0.055
    assert binom.sf(plan.passing_count, plan.reps, 0.05) <= 0.00005
    # Independent numerical planning witnesses supplied by the scientific-tolerance policy.
    assert binomial_error_upper_bound(6500, 130000, 0.00005) == pytest.approx(0.05239183593814554)
    assert binom.sf(6831, 130000, 0.05) == pytest.approx(1.4120811756581416e-5)


def test_r03_four_attempt_accounting_has_independent_populations():
    from tests.estimation._i13_manifest import ClusterDGP, ManifestCell

    cell = ManifestCell(
        "accounting", "principal", "mean_ratio", ClusterDGP(5, 5, size_mean=1), "fixture"
    )
    observations = (
        IntervalResult(0.0, -1.0, 1.0),
        IntervalResult(2.0, 1.0, 3.0),
        IntervalResult(1.0, None, None, unavailable_reason="uncertainty_missing"),
        IntervalResult(None, None, None, unavailable_reason="point_missing"),
    )
    result = run_cell(
        cell, 4, truth_of=lambda sample: 0.0, estimate_of=lambda sample, i: observations[i]
    )
    assert (
        result.attempted,
        result.point_estimable,
        result.interval_estimable,
        result.hits,
        result.excluded,
        result.failed,
    ) == (4, 3, 2, 1, 1, 0)
    assert result.coverage_conditional == 0.5
    assert result.coverage_unconditional == 0.25
    assert result.bias == 1.0
    assert result.bias_mcse == pytest.approx((1 / 3) ** 0.5)
    assert result.exclusion_reasons == {"point_missing": 1}
    assert result.interval_unavailable_reasons == {"uncertainty_missing": 1, "point_missing": 1}


@pytest.mark.parametrize("reps", (0, -1, True, False, 1.5))
def test_replication_count_refuses_before_generation(reps):
    def unexpected_estimate(sample, i) -> IntervalResult:
        pytest.fail("invalid replication count reached the estimator")

    cell = ACCEPTANCE_MANIFEST[0].design
    with pytest.raises(ValueError):
        run_cell(cell, reps, truth_of=lambda sample: 0.0, estimate_of=unexpected_estimate)
