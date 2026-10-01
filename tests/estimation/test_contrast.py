import math
from typing import Any, cast

import pytest
from scipy.stats import t as _t

from increment.decision import ContrastDecisionProcedure, FixedInference, NoFamily
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.contrast import (
    ContrastPartition,
    ContrastStats,
    estimate_contrast,
    reduce_contrast_partitions,
)
from increment.semantics.unit_cycle import UnitCycleTApproximation


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.contrast.contrast_reduction_least",
            lambda: reduce_contrast_partitions([]),
        ),
        (
            "estimation.contrast.stats_contraststats",
            lambda: estimate_contrast("not-a-stats", None),  # ty: ignore[invalid-argument-type]
        ),  # estimation/contrast.py::estimate_contrast
    ],
)
def test_contrast_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


def _stats(**overrides):
    values: dict[str, Any] = {
        "metric": "orders",
        "aggregation": "sum",
        "probability_ct": 0.5,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "carryover_order": 0,
        "observation_steps": 3,
        "retained_steps": 3,
        "control_group": "control",
        "treatment_group": "treatment",
        "n_units": 4,
        "n_cycles": 12,
        "ct_cycles": 7,
        "tc_cycles": 5,
        "reference_delta": 10.0,
        "mean_residual": 2.0,
        "m2_delta": 6.0,
    }
    values.update(overrides)
    return ContrastStats(**values)


def _procedure(**overrides):
    values: dict[str, Any] = {
        "metric": "orders",
        "reference": UnitCycleTApproximation(),
        "role": "primary",
        "alternative": "two-sided",
        "null_abs": 0.0,
        "alpha": 0.05,
    }
    values.update(overrides)
    return ContrastDecisionProcedure(**values)


def test_estimate_uses_shifted_mean_and_unit_t_reference():
    result = estimate_contrast(_stats(), _procedure()).results[0]

    expected_se = math.sqrt(6.0 / (4 * 3))
    critical = _t.ppf(0.975, 3)
    assert result.estimate.value == 12.0
    assert result.standard_error == pytest.approx(expected_se)
    assert result.dof == 3.0
    assert result.estimate.lb == pytest.approx(12.0 - critical * expected_se)
    assert result.estimate.ub == pytest.approx(12.0 + critical * expected_se)
    assert result.estimate.level == 0.95
    assert result.estimand == "retained_window_total_difference"
    assert result.method == "switchback_unit_t_approximation"
    assert result.method_role == "decision"
    assert result.assignment == "switchback"
    assert result.inference == "fixed"
    assert result.reference == "unit_t_approximation"


def test_one_sided_alpha_doubles_display_level_and_any_maps_estimand():
    result = estimate_contrast(
        _stats(aggregation="any", probability_ct=0.8),
        _procedure(alternative="greater", alpha=0.025, role="guardrail"),
    ).results[0]

    assert result.estimate.level == pytest.approx(0.95)
    assert result.estimand == "retained_window_conversion_difference"
    assert result.aggregation == "any"
    assert result.probability_ct == 0.8
    assert result.role == "guardrail"


def test_large_reference_is_recovered_without_unshifted_cancellation():
    stats = _stats(
        n_units=2,
        reference_delta=1e308,
        mean_residual=-1e292,
        m2_delta=0.0,
    )
    result = estimate_contrast(stats, _procedure()).results[0]

    assert result.estimate.value == math.fsum((stats.reference_delta, stats.mean_residual))
    assert math.isfinite(result.estimate.value)
    assert result.standard_error == 0.0


def test_minimum_positive_subnormal_m2_preserves_positive_standard_error():
    m2_delta = math.nextafter(0.0, 1.0)
    stats = _stats(n_units=2, m2_delta=m2_delta)

    result = estimate_contrast(stats, _procedure()).results[0]

    expected_se = math.sqrt(m2_delta) / math.sqrt(1.0) / math.sqrt(2.0)
    assert result.standard_error == pytest.approx(expected_se)
    assert result.standard_error is not None and result.standard_error > 0.0


def test_equal_group_labels_are_rejected_during_stats_revalidation():
    values = _stats().model_dump()
    values.update(control_group="same", treatment_group="same")
    stats = ContrastStats.model_construct(**values)

    with pytest.raises(InvalidRequestError) as revalidation_exc:
        estimate_contrast(stats, _procedure()).results[0]
    assert (
        revalidation_exc.value.code == "estimation.contrast.contrast_stats.control_group_treatment"
    )


def test_equal_group_labels_are_rejected_during_stats_construction():
    with pytest.raises(InvalidRequestError) as construction_exc:
        _stats(control_group="same", treatment_group="same")
    assert (
        construction_exc.value.code == "estimation.contrast.contrast_stats.control_group_treatment"
    )


def _partition(**overrides):
    values: dict[str, Any] = {
        "metric": "orders",
        "aggregation": "sum",
        "probability_ct": 0.5,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "carryover_order": 0,
        "observation_steps": 3,
        "retained_steps": 3,
        "control_group": "control",
        "treatment_group": "treatment",
        "unit_deltas": {"unit": 1.0},
        "cycles_by_unit": {"unit": 1},
    }
    values.update(overrides)
    return ContrastPartition(**values)


def test_partition_empty_unit_guards_share_one_code():
    with pytest.raises(InvalidRequestError) as via_unit_deltas:
        _partition(unit_deltas={})
    with pytest.raises(InvalidRequestError) as via_cycles:
        _partition(cycles_by_unit={})

    assert via_unit_deltas.value.code == "estimation.contrast.contrast_partition.contain_least_one"
    assert via_cycles.value.code == via_unit_deltas.value.code


def test_partition_equal_group_labels_share_stats_code():
    with pytest.raises(InvalidRequestError) as raised:
        _partition(control_group="same", treatment_group="same")
    assert raised.value.code == "estimation.contrast.contrast_stats.control_group_treatment"


def test_partition_reduction_nonfinite_centered_moments_share_one_code():
    common: dict[str, Any] = {
        "metric": "orders",
        "aggregation": "sum",
        "probability_ct": 0.5,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "carryover_order": 0,
        "observation_steps": 3,
        "retained_steps": 3,
        "control_group": "control",
        "treatment_group": "treatment",
    }
    parts = [
        ContrastPartition(**common, unit_deltas={"a": 7.5e307}, cycles_by_unit={"a": 1}),
        ContrastPartition(**common, unit_deltas={"b": -7.5e307}, cycles_by_unit={"b": 1}),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        reduce_contrast_partitions(parts)
    assert raised.value.code == "estimation.contrast.contrast_reduction_delta_mean_finite"


def test_partition_reduction_produces_estimable_unit_t_stats():
    common: dict[str, Any] = {
        "metric": "orders",
        "aggregation": "sum",
        "probability_ct": 0.5,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "carryover_order": 0,
        "observation_steps": 3,
        "retained_steps": 3,
        "control_group": "control",
        "treatment_group": "treatment",
    }
    stats = reduce_contrast_partitions(
        [
            ContrastPartition(
                **common,
                unit_deltas={"a": 1.0, "b": 3.0},
                cycles_by_unit={"a": 2, "b": 2},
                ct_counts_by_unit={"a": 1, "b": 2},
            ),
            ContrastPartition(
                **common,
                unit_deltas={"c": 5.0},
                cycles_by_unit={"c": 2},
                ct_counts_by_unit={"c": 0},
            ),
        ]
    )
    result = estimate_contrast(stats, _procedure()).results[0]

    assert stats.n_units == 3
    assert stats.n_cycles == 6
    assert (stats.ct_cycles, stats.tc_cycles) == (3, 3)
    assert (result.ct_cycles, result.tc_cycles) == (3, 3)
    assert result.estimate.value == pytest.approx(3.0)
    assert result.dof == 2.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_units": 1},
        {"reference_delta": float("inf")},
        {"mean_residual": float("nan")},
        {"m2_delta": -1.0},
        {"ct_cycles": 6},
    ],
)
def test_invalid_stats_are_refused(overrides):
    # model_construct bypasses the constructor, exercising estimator-side
    # revalidation rather than pydantic's constructor validation.
    stats = ContrastStats.model_construct(**{**_stats().model_dump(), **overrides})
    with pytest.raises((TypeError, ValueError)):
        estimate_contrast(stats, _procedure()).results[0]


def test_metric_identity_and_input_types_are_defensive():
    with pytest.raises(InvalidRequestError) as metric_exc:
        estimate_contrast(_stats(), _procedure(metric="revenue")).results[0]
    assert metric_exc.value.code == "estimation.contrast.contrast_stats_metric"
    with pytest.raises(InvalidRequestError) as stats_exc:
        estimate_contrast(cast(ContrastStats, object()), _procedure()).results[0]
    assert stats_exc.value.code == "estimation.contrast.stats_contraststats"
    with pytest.raises(InvalidRequestError) as procedure_exc:
        estimate_contrast(_stats(), cast(ContrastDecisionProcedure, object())).results[0]
    assert procedure_exc.value.code == "estimation.contrast.procedure_contrastdecisionprocedure"


def test_sequential_and_multiplicity_instances_are_refused():
    sequential = _procedure().model_construct(
        metric="orders",
        role="primary",
        alternative="two-sided",
        null_abs=0.0,
        alpha=0.05,
        family=NoFamily(),
        inference=FixedInference.model_construct(kind="always_valid"),
    )
    with pytest.raises(CapabilityError) as sequential_exc:
        estimate_contrast(_stats(), sequential).results[0]
    assert sequential_exc.value.code == "contrast.inference"

    multiplicity = _procedure().model_construct(
        metric="orders",
        role="primary",
        alternative="two-sided",
        null_abs=0.0,
        alpha=0.05,
        family=NoFamily.model_construct(kind="bonferroni"),
        inference=FixedInference(),
    )
    with pytest.raises(CapabilityError) as multiplicity_exc:
        estimate_contrast(_stats(), multiplicity).results[0]
    assert multiplicity_exc.value.code == "contrast.decision"


def test_unsupported_metric_shape_is_refused_before_computation():
    values = _stats().model_dump()
    values["aggregation"] = "ratio"
    stats = ContrastStats.model_construct(**values)
    with pytest.raises(CapabilityError) as ratio_exc:
        estimate_contrast(stats, _procedure()).results[0]
    assert ratio_exc.value.code == "contrast.metric"


class TestTinyAlphaContrastEndToEnd:
    """A small but valid alpha (family-corrected) must produce an interval, not
    a rounding artifact: the quantile must come from the tail, and the reported
    level rounds to 1.0 so the requested rate has to be carried alongside it."""

    def test_a_tiny_alpha_contrast_reports_its_interval(self):
        result = estimate_contrast(_stats(), _procedure(alpha=1e-20)).results[0]
        estimate = result.estimate
        lb, ub = estimate.lb, estimate.ub
        assert lb is not None
        assert ub is not None
        assert math.isfinite(lb)
        assert math.isfinite(ub)
        assert ub > lb
        # level rounds to exactly 1.0 at this alpha; alpha carries the truth.
        assert estimate.level == 1.0
        assert estimate.alpha == pytest.approx(1e-20)

    def test_an_ordinary_alpha_is_unchanged(self):
        estimate = estimate_contrast(_stats(), _procedure(alpha=0.05)).results[0].estimate
        assert estimate.level == pytest.approx(0.95)
