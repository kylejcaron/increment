"""Fast regression checks for prespecified comparative-efficiency budgets."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from calibration.comparative_efficiency import Comparison, evaluate_comparison


@pytest.fixture
def ordinary_comparison():
    return Comparison(
        regime="dense_mean_equal_allocation",
        candidate_width=1.0,
        reference_width=1.0,
        candidate_power=0.8,
        reference_power=0.8,
        candidate_available=True,
        reference_available=True,
        candidate_elapsed_s=0.01,
        reference_elapsed_s=0.01,
        guarantee="asymptotic",
        reference="Welch matched mean",
    )


def test_declared_dense_reference_passes(ordinary_comparison):
    assert evaluate_comparison(ordinary_comparison).passed


def test_efficiency_budgets_are_inclusive_at_the_limit_and_fail_beyond(ordinary_comparison):
    width_limit = 1.0 + 0.10
    assert evaluate_comparison(replace(ordinary_comparison, candidate_width=width_limit)).passed
    assert evaluate_comparison(
        replace(ordinary_comparison, candidate_width=np.nextafter(width_limit, np.inf))
    ).failure_dimensions == ("width",)

    power_limit = 0.8 - 0.05
    assert evaluate_comparison(replace(ordinary_comparison, candidate_power=power_limit)).passed
    assert evaluate_comparison(
        replace(ordinary_comparison, candidate_power=np.nextafter(power_limit, -np.inf))
    ).failure_dimensions == ("power",)

    assert evaluate_comparison(
        replace(ordinary_comparison, candidate_n=110, reference_n=100)
    ).passed
    assert evaluate_comparison(
        replace(ordinary_comparison, candidate_n=111, reference_n=100)
    ).failure_dimensions == ("required_n",)


def test_real_public_inference_path_flags_inflated_and_rounded_intervals():
    from scipy.stats import t as student_t

    from increment import Analysis, MetricSpec
    from tests.analysis_factory import lift_rows

    rng = np.random.default_rng(174)
    n = 80
    control = rng.normal(100.0, 2.0, n)
    treatment = rng.normal(102.0, 2.0, n)
    frame = pl.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate((control, treatment)),
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="y", type="mean")],
    )
    try:
        (row,) = lift_rows(analysis.run())
    finally:
        analysis.close()
    assert row.abs_lb is not None and row.abs_ub is not None

    variance_control = float(control.var(ddof=1))
    variance_treatment = float(treatment.var(ddof=1))
    standard_error = np.sqrt(variance_control / n + variance_treatment / n)
    df = (variance_control / n + variance_treatment / n) ** 2 / (
        (variance_control / n) ** 2 / (n - 1) + (variance_treatment / n) ** 2 / (n - 1)
    )
    critical = float(student_t.isf(0.025, df))
    difference = float(treatment.mean() - control.mean())
    reference_lb = difference - critical * standard_error
    reference_ub = difference + critical * standard_error
    reference_width = reference_ub - reference_lb
    production_width = row.abs_ub - row.abs_lb
    baseline = Comparison(
        regime="public_mean_readout",
        candidate_width=production_width,
        reference_width=reference_width,
        candidate_power=float(row.abs_lb > 0 or row.abs_ub < 0),
        reference_power=float(reference_lb > 0 or reference_ub < 0),
        candidate_available=True,
        reference_available=True,
        candidate_elapsed_s=0.01,
        reference_elapsed_s=0.01,
        guarantee="asymptotic",
        reference="independently calculated Welch interval on the identical sample",
    )
    assert evaluate_comparison(baseline).passed

    inflated = evaluate_comparison(replace(baseline, candidate_width=1.5 * production_width))
    assert not inflated.passed
    assert inflated.failure_dimensions == ("width",)

    rounded_width = float(np.ceil(row.abs_ub) - np.floor(row.abs_lb))
    rounded = evaluate_comparison(replace(baseline, candidate_width=rounded_width))
    assert not rounded.passed
    assert rounded.failure_dimensions == ("width",)


def test_public_readout_detects_disabled_useful_cuped():
    from increment import Analysis, Method, MetricSpec
    from tests.analysis_factory import lift_rows

    rng = np.random.default_rng(175)
    n = 100
    covariate = rng.normal(size=2 * n)
    outcomes = (
        np.concatenate((np.full(n, 100.0), np.full(n, 102.0)))
        + 4.0 * covariate
        + rng.normal(0, 0.5, 2 * n)
    )
    frame = pl.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "y": outcomes,
            "x": covariate,
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="y", type="mean", covariate="x")],
    )
    try:
        rows = lift_rows(
            analysis.run(
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
            )
        )
    finally:
        analysis.close()
    widths = {
        row.method: row.lift.ub - row.lift.lb
        for row in rows
        if row.lift is not None and row.lift.ub is not None and row.lift.lb is not None
    }
    assert widths["cuped"] < widths["unadjusted"]
    comparison = Comparison(
        regime="public_cuped_adjustment",
        candidate_width=widths["unadjusted"],
        reference_width=widths["cuped"],
        candidate_power=None,
        reference_power=None,
        candidate_available=True,
        reference_available=True,
        candidate_elapsed_s=0.01,
        reference_elapsed_s=0.01,
        guarantee="asymptotic",
        reference="same public readout with declared pre-assignment CUPED",
    )
    result = evaluate_comparison(comparison)
    assert not result.passed
    assert result.failure_dimensions == ("width",)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"candidate_width": 1.5}, ("width",)),  # materially inflated standard error
        ({"candidate_power": 0.5}, ("power",)),  # disabled useful CUPED effect
        ({"candidate_width": 1.2}, ("width",)),  # coarse outward rounding
    ],
    ids=("inflated-se", "disabled-cuped", "coarse-rounding"),
)
def test_dense_negative_controls_violate_their_prespecified_efficiency_dimension(
    ordinary_comparison, changes, expected
):
    result = evaluate_comparison(replace(ordinary_comparison, **changes))
    assert not result.passed
    assert result.failure_dimensions == expected


def test_numerical_loss_and_coded_refusal_are_not_statistical_width_loss(ordinary_comparison):
    result = evaluate_comparison(
        replace(
            ordinary_comparison,
            candidate_width=None,
            candidate_available=False,
            candidate_elapsed_s=0.5,
            reference_elapsed_s=0.01,
            numerical_tolerance_loss=0.002,
            refusal_code="estimation.engine.lift_guard",
        )
    )
    assert result.elapsed_ratio == 50.0
    assert result.availability_loss
    assert result.failure_dimensions == ("availability",)
    assert result.width_loss is None
    assert result.numerical_tolerance_loss == 0.002
    assert result.refusal_code == "estimation.engine.lift_guard"


def test_valid_sparse_discrete_guarantee_is_not_failed_for_conservatism():
    discrete = Comparison(
        regime="sparse_binomial_exact",
        candidate_width=2.0,
        reference_width=1.0,
        candidate_power=0.35,
        reference_power=0.8,
        candidate_available=True,
        reference_available=True,
        candidate_elapsed_s=0.02,
        reference_elapsed_s=0.01,
        guarantee="finite-sample",
        reference="Katz asymptotic log risk ratio",
        efficiency_comparable=False,
    )
    result = evaluate_comparison(discrete)
    assert result.passed
    assert result.width_loss is None
    assert result.power_loss is None
    assert result.guarantee != "asymptotic"


def test_zero_control_binomial_set_contributes_rejection_without_finite_width():
    from calibration.comparative_binary import _production_outcome
    from tests.estimation._conversion_counts import lift_row

    row = lift_row((0, 1000, 40, 1000))
    assert row.reference_kind == "binomial"
    assert row.lift is None
    width, rejected = _production_outcome(row)
    assert width is None
    assert rejected is True
