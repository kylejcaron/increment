"""Numerical and directional contracts for joint relative inversion."""

import math

import pytest
from scipy.stats import norm, t

from increment.errors import CodedError
from increment.estimation.results import (
    JointContrastReference,
    RelativeConfidenceSet,
    relative_confidence_set,
)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e300])
def test_deterministic_ratio_survives_common_measurement_scaling(scale):
    result = relative_confidence_set(
        JointContrastReference(a=scale, c=scale, var_a=0, var_c=0, cov_ac=0)
    )
    assert result.geometry == "bounded"
    assert result.intervals == ((1.0, 1.0),)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_directional_closure_contains_both_disconnected_components(alternative):
    reference = JointContrastReference(a=1, c=0, var_a=0.01, var_c=1, cov_ac=0)
    central = relative_confidence_set(reference)
    assert central.geometry == "disconnected"
    result = relative_confidence_set(reference, alternative=alternative)
    assert result.geometry == "all_real"
    assert result.contains(-1e100) and result.contains(1e100)
    assert result.p_value(0) == 1.0


def test_linear_degeneracy_keeps_correct_ray():
    critical = float(norm.isf(0.025))
    result = relative_confidence_set(
        JointContrastReference(a=1, c=critical, var_a=0, var_c=1, cov_ac=0)
    )
    assert result.geometry == "one_sided"
    assert not result.contains(0)
    assert result.contains(1)
    assert result.intervals[0][1] is None


def test_negative_baseline_direction_uses_ratio_not_numerator_sign():
    result = relative_confidence_set(
        JointContrastReference(a=2, c=-1, var_a=0.04, var_c=0.01, cov_ac=0), alternative="greater"
    )
    assert result.p_value(0) == 1.0
    assert result.contains(0)


def test_subnormal_tail_allocation_rounds_down_and_survives_serialization():
    smallest = math.ulp(0.0)
    alpha = 3 * smallest
    result = relative_confidence_set(
        JointContrastReference(a=0, c=1, var_a=1, var_c=0, cov_ac=0), alpha=alpha
    )
    restored = RelativeConfidenceSet.model_validate_json(result.model_dump_json())
    assert restored.alpha == alpha
    assert restored.reference == result.reference
    assert restored.intervals[0][1] >= norm.isf(smallest)


def test_covariance_validation_does_not_overflow():
    with pytest.raises(CodedError):
        JointContrastReference(a=0, c=1, var_a=1e300, var_c=1e300, cov_ac=1.1e300)


def test_zero_denominator_has_set_without_fabricated_point():
    result = relative_confidence_set(JointContrastReference(a=0, c=0, var_a=1, var_c=1, cov_ac=0))
    assert result.geometry == "all_real"
    assert result.estimate() is None
    assert result.p_value(0) == 1.0


def test_unrepresentable_finite_endpoint_is_unavailable_not_unbounded():
    result = relative_confidence_set(
        JointContrastReference(a=1, c=1e-320, var_a=0, var_c=0, cov_ac=0)
    )
    assert result.geometry == "unavailable"
    assert result.reason == "endpoint_unrepresentable"
    assert result.contains(0) is None
    assert result.estimate() is None


def test_moderate_student_statistic_retains_available_tail():
    reference = JointContrastReference(a=8, c=1, var_a=1, var_c=0, cov_ac=0, kind="t", df=1e6)
    result = relative_confidence_set(reference)
    assert result.p_value(0) == pytest.approx(2 * t.sf(8, 1e6), rel=1e-12)


def test_persisted_intervals_cannot_contradict_joint_reference():
    reference = JointContrastReference(a=0, c=1, var_a=1, var_c=0, cov_ac=0)
    with pytest.raises(CodedError):
        RelativeConfidenceSet(
            reference=reference, alpha=0.05, geometry="bounded", intervals=((10, 11),)
        )


def test_rounded_joint_covariance_encloses_neighboring_rank_one_gram():
    from fractions import Fraction

    from increment.estimation.results import _joint_reference_from_exact

    x, y = Fraction(1), Fraction(math.nextafter(1.0, math.inf))
    reference, reason = _joint_reference_from_exact(
        a=1.0, c=1.0, var_a=x * x, var_c=y * y, cov_ac=x * y
    )
    assert reason is None and reference is not None
    da = Fraction(reference.var_a) - x * x
    dc = Fraction(reference.var_c) - y * y
    db = Fraction(reference.cov_ac) - x * y
    assert da >= 0 and dc >= 0 and da * dc >= db * db
    JointContrastReference.model_validate_json(reference.model_dump_json())


def test_unavailable_joint_covariance_preserves_additive_output_without_evidence():
    from increment.estimation.results import LiftEstimate

    row = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=None,
        scale="linear",
        abs_diff=1.0,
        abs_se=0.2,
        relative_unavailable_reason="joint_covariance_indefinite",
    )
    restored = LiftEstimate.model_validate_json(row.model_dump_json())
    assert restored.abs_diff == 1.0 and restored.abs_se == 0.2
    assert restored.lift is None and not restored.stat_sig()
    with pytest.raises(CodedError) as raised:
        restored.p_value()
    assert raised.value.code == "estimation.results.joint.unavailable"
    with pytest.raises(CodedError):
        restored.require_lift()


def test_joint_covariance_unavailability_cannot_claim_an_interval():
    from increment.estimation.results import Estimate, LiftEstimate

    with pytest.raises(CodedError):
        LiftEstimate(
            metric="revenue",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.5, lb=0.1, ub=0.9, alpha=0.05, level=0.95),
            scale="linear",
            abs_diff=1.0,
            abs_se=0.2,
            relative_unavailable_reason="joint_covariance_indefinite",
        )


def test_cross_term_rounding_cannot_reduce_joint_covariance():
    from fractions import Fraction

    from increment.estimation.results import _joint_reference_from_exact

    small = Fraction(1, 2**53)
    va, vc, cov = Fraction(2), 1 + small**2, 1 + small
    reference, reason = _joint_reference_from_exact(a=1.0, c=1.0, var_a=va, var_c=vc, cov_ac=cov)
    assert reason is None and reference is not None
    da, dc = Fraction(reference.var_a) - va, Fraction(reference.var_c) - vc
    db = Fraction(reference.cov_ac) - cov
    assert da >= 0 and dc >= 0 and da * dc >= db * db


def _shared_joint_from_values(control, treatment):
    import pyarrow as pa

    from tests.estimation.test_adjust_cluster import _joint_estimate

    records = [
        {
            "u": f"{group}-{cluster}",
            "g": group,
            "store": f"s{cluster}",
            "y": float(value),
            "z": None,
            "den": 1.0,
        }
        for group, values in (("C", control), ("T", treatment))
        for cluster, value in values
    ]
    return _joint_estimate(pa.Table.from_pylist(records))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_paired_cluster_covariance_retains_exact_rank_one_geometry():
    row = _shared_joint_from_values(
        list(enumerate((12, 8, 12, 8))), list(enumerate((25, 17, 25, 17)))
    )
    assert row.abs_diff == 11 and row.abs_se == pytest.approx(math.sqrt(4 / 3))
    assert row.relative_confidence_set is not None
    reference = row.relative_confidence_set.reference
    assert (reference.var_a, reference.var_c, reference.cov_ac) == pytest.approx((4 / 3,) * 3)
    assert row.lift is not None and row.lift.value == 1.1
    assert row.lift.lb is not None and row.lift.lb < row.lift.value < row.lift.ub


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_paired_zero_relative_variance_does_not_certify_a_constant_effect():
    row = _shared_joint_from_values(
        list(enumerate((1, 2, 4, 3, 7))), list(enumerate((4, 8, 16, 12, 28)))
    )
    assert row.lift is not None and row.lift.value == 3
    assert row.relative_confidence_set is None
    assert row.relative_unavailable_reason == "zero_relative_variance"
    assert row.stat_sig() is False
    assert row.abs_lb < row.abs_diff < row.abs_ub
    with pytest.raises(CodedError) as error:
        row.p_value()
    assert error.value.code == "estimation.results.joint.unavailable"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_indefinite_response_covariance_preserves_valid_additive_interval():
    row = _shared_joint_from_values([(2, 13), (3, 7), (4, 10)], [(1, 20), (2, 26), (3, 14)])
    assert row.abs_diff == 10 and row.abs_se == pytest.approx(math.sqrt(3 / 5))
    assert row.abs_lb is not None and row.abs_ub is not None
    assert row.abs_lb < 10 < row.abs_ub
    assert row.relative_confidence_set is None
    assert row.relative_unavailable_reason == "joint_covariance_indefinite"
    assert row.lift is not None and row.lift.value == 1
    assert row.lift.lb is None and row.lift.ub is None
    with pytest.raises(CodedError):
        row.p_value()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_zero_control_mean_retains_disconnected_set_without_fake_point():
    row = _shared_joint_from_values([(1, 1), (2, -1)], [(1, 2), (2, 0)])
    assert row.abs_diff == 1
    assert row.lift is None
    assert row.relative_confidence_set is not None
    assert row.relative_confidence_set.geometry == "disconnected"
    assert row.relative_confidence_set.contains(0) is False
    assert row.relative_confidence_set.contains(10) is True


@pytest.mark.parametrize("kind,df", [("normal", None), ("t", 7.0)])
@pytest.mark.parametrize("alternative,sign", [("greater", 1), ("less", -1)])
@pytest.mark.parametrize("baseline_sign", [-1, 1])
def test_directional_endpoint_and_p_value_use_full_tail(kind, df, alternative, sign, baseline_sign):
    distribution = norm if df is None else t(df)
    reference = JointContrastReference(
        a=sign * baseline_sign * 2.0,
        c=baseline_sign,
        var_a=1,
        var_c=0,
        cov_ac=0,
        kind=kind,
        df=df,
    )
    result = relative_confidence_set(reference, alternative=alternative)
    finite = result.intervals[0][0 if sign == 1 else 1]
    # The endpoint is defined by its tail mass: verify it through the forward
    # survival function rather than a library quantile whose last digits vary.
    assert distribution.sf(2.0 - sign * finite) == pytest.approx(0.05, rel=1e-12, abs=0)
    assert result.intervals[0][1 if sign == 1 else 0] is None
    assert result.p_value(0) == pytest.approx(distribution.sf(2), rel=1e-12)
    assert result.contains(sign * (2.0 - distribution.isf(0.04))) is False
    assert result.contains(sign * (2.0 - distribution.isf(0.06))) is True
    restored = RelativeConfidenceSet.model_validate_json(result.model_dump_json())
    assert restored.intervals == result.intervals
    assert restored.p_value(0) == result.p_value(0)


@pytest.mark.parametrize("kind,df", [("normal", None), ("t", 7.0)])
@pytest.mark.parametrize("z", [0.25, 2.0, 8.0, 16.0, 40.0])
@pytest.mark.parametrize("alternative,multiplier", [("two-sided", 2), ("greater", 1)])
def test_joint_tail_paths_match_reference(kind, df, z, alternative, multiplier):
    reference = JointContrastReference(a=z, c=1, var_a=1, var_c=0, cov_ac=0, kind=kind, df=df)
    p = relative_confidence_set(reference, alternative=alternative).p_value(0)
    distribution = norm if df is None else t(df)
    expected = multiplier * distribution.sf(z)
    if expected == 0.0:
        assert p == math.ulp(0.0)
    else:
        assert p == pytest.approx(expected, rel=1e-11, abs=0)


def test_student_log_tail_retains_extreme_finite_probability():
    reference = JointContrastReference(a=1e200, c=1, var_a=1, var_c=0, cov_ac=0, kind="t", df=1)
    assert relative_confidence_set(reference).p_value(0) == pytest.approx(
        2 / (math.pi * 1e200), rel=1e-10, abs=0
    )


def _inferred_joint_row(*, alternative="two-sided", zero_variance=False):
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.inference import infer_ate

    reference = JointContrastReference(
        a=2,
        c=3,
        var_a=0 if zero_variance else 0.04,
        var_c=0.09,
        cov_ac=0,
        kind="t",
        df=3,
    )
    return infer_ate(
        metric="m",
        group_id="T",
        method="iptw",
        method_role="decision",
        point=99,
        scores=ScoreStats(metric="m", contrast="T", n=10, sum_psi=0, sum_psi2=100),
        # Independently rounded caller inputs must never override the joint reference.
        abs_diff=math.nextafter(2.0, math.inf),
        abs_se=math.nextafter(0.2, math.inf),
        abs_dof=7.5,
        joint_reference=reference,
        alternative=alternative,
        null_abs=0,
    )


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("zero_variance", [False, True])
def test_infer_ate_canonicalizes_joint_additive_projection(alternative, zero_variance):
    from increment.estimation.results import LiftEstimate

    row = _inferred_joint_row(alternative=alternative, zero_variance=zero_variance)
    ref = row.relative_confidence_set.reference
    assert row.abs_diff == ref.a
    assert row.abs_se == (math.sqrt(ref.var_a) or None)
    if zero_variance:
        assert (row.abs_lb, row.abs_ub) == (None, None)
        assert row.abs_reference_kind is None and row.abs_reference_df is None
        assert row.stat_sig() is False
    else:
        assert row.reference_df == 3 and row.abs_reference_df == 7.5
        q = t.isf(0.025 if alternative == "two-sided" else 0.05, 7.5)
        assert row.abs_lb == pytest.approx(2 - q * 0.2)
        assert row.abs_ub == pytest.approx(2 + q * 0.2)
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row


def _joint_row_model_payload(model_name, *, zero_variance=False):
    from datetime import date

    from increment.breakout.estimates import BreakoutEstimate, DailyLiftEstimate
    from increment.estimation.results import LiftEstimate

    model = {"primary": LiftEstimate, "breakout": BreakoutEstimate, "daily": DailyLiftEstimate}[
        model_name
    ]
    row = _inferred_joint_row(zero_variance=zero_variance)
    payload = {key: value for key, value in row.model_dump().items() if key in model.model_fields}
    if model_name == "breakout":
        payload.update(dimension="country", dimension_value="US")
    elif model_name == "daily":
        payload.update(ds=date(2026, 1, 1))
    return model, payload


@pytest.mark.parametrize("model_name", ["primary", "breakout", "daily"])
@pytest.mark.parametrize("field", ["abs_diff", "abs_se", "abs_lb", "abs_ub"])
@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_serialized_joint_additive_mutations_rejected_exactly(model_name, field, direction):
    model, payload = _joint_row_model_payload(model_name)
    original = model.model_validate(payload)
    assert model.model_validate_json(original.model_dump_json()) == original
    payload[field] = math.nextafter(payload[field], direction)
    with pytest.raises(CodedError) as raised:
        model.model_validate(payload)
    assert raised.value.code == "estimation.results.joint.invalid_set"


@pytest.mark.parametrize("model_name", ["primary", "breakout", "daily"])
@pytest.mark.parametrize("field,value", [("abs_se", 0.0), ("abs_lb", 2.0), ("abs_ub", 2.0)])
def test_zero_variance_joint_rows_reject_fabricated_additive_evidence(model_name, field, value):
    model, payload = _joint_row_model_payload(model_name, zero_variance=True)
    model.model_validate_json(model.model_validate(payload).model_dump_json())
    payload[field] = value
    with pytest.raises(CodedError):
        model.model_validate(payload)


@pytest.mark.parametrize("model_name", ["primary", "breakout", "daily"])
def test_joint_additive_welch_reference_mutation_requires_new_bounds(model_name):
    model, payload = _joint_row_model_payload(model_name)
    payload["abs_reference_df"] = 30.0
    with pytest.raises(CodedError) as raised:
        model.model_validate(payload)
    assert raised.value.code == "estimation.results.joint.invalid_set"


@pytest.mark.parametrize("alternative", ["greater", "less"])
@pytest.mark.parametrize("alpha", [0.5, 0.75])
def test_directional_joint_set_retains_alpha_domain(alternative, alpha):
    reference = JointContrastReference(a=0, c=1, var_a=1, var_c=0, cov_ac=0)
    with pytest.raises(CodedError) as raised:
        relative_confidence_set(reference, alternative=alternative, alpha=alpha)
    assert raised.value.code == "estimation.results.joint.invalid_set"


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_directional_subnormal_alpha_uses_undivided_tail(alternative):
    alpha = math.ulp(0.0)
    reference = JointContrastReference(a=0, c=1, var_a=1, var_c=0, cov_ac=0)
    result = relative_confidence_set(reference, alpha=alpha, alternative=alternative)
    assert result.geometry == "one_sided"
    endpoint = result.intervals[0][0 if alternative == "greater" else 1]
    assert endpoint is not None
    assert abs(endpoint) == pytest.approx(norm.isf(alpha), rel=1e-14)
