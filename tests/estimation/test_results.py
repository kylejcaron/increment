"""Tests for Estimate/LiftEstimate - log_mean/log_se (Estimate) and
abs_diff/abs_se (LiftEstimate) fields, plus the behavioral contracts on the
value objects themselves: ``excludes()`` boundary semantics, the symmetric
z-quantile interval width, and numeric fidelity through ``to_frame``.

The field tests pin the schema contract (defaults, settability, frozen
immutability, and - the load-bearing claim - invisibility of
`Estimate`-level fields to every `to_frame` path); ``infer_lift`` populates
log_mean/log_se and ``estimate_lift`` populates abs_diff/abs_se - see
tests/estimation/test_inference.py and test_engine.py.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd
import pytest
from scipy.stats import t as _t

from increment.breakout.estimates import to_frame
from increment.errors import CodedError, InvalidRequestError
from increment.estimation.results import BinomialConfidenceSet, Estimate, LiftEstimate


def test_result_deserialization_rejects_unknown_alternative():
    from increment.estimation.inference import infer_lift

    row = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.1,
        se_t=0.05,
        se_c=0.05,
        alpha=0.05,
    )
    payload = row.model_dump()
    payload["alternative"] = "typo"
    with pytest.raises(InvalidRequestError) as raised:
        LiftEstimate.model_validate(payload)
    assert raised.value.code == "estimation.binomial.unknown_alternative"
    assert raised.value.context["alternative"] == "typo"


def test_open_bound_conversion_greater_direction_opens_the_upper_side():
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import open_bound_from_two_sided_at_target

    target_alpha = 0.6
    row = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.30 - 0.10,
        se_t=0.05,
        se_c=0.04,
        alpha=target_alpha / 2.0,
        alternative="greater",
    )
    assert row.require_lift().alpha == pytest.approx(target_alpha)
    assert row.require_lift().level == pytest.approx(1.0 - target_alpha)

    converted = open_bound_from_two_sided_at_target(row)

    # The converted bound is the target_alpha quantile of the row's own
    # posterior: exactly 1 - target_alpha of its mass sits above it.
    bound = converted.require_lift().lb
    assert bound is not None
    assert row.prob_beyond(bound) == pytest.approx(1.0 - target_alpha, abs=1e-12)
    assert converted.require_lift().ub is None
    assert converted.require_lift().open_side == "upper"
    assert converted.require_lift().alpha == pytest.approx(target_alpha)


def test_open_bound_conversion_less_direction_opens_the_lower_side():
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import open_bound_from_two_sided_at_target

    row = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.10 - 0.30,
        se_t=0.05,
        se_c=0.04,
        alpha=0.025,
        alternative="less",
    )

    converted = open_bound_from_two_sided_at_target(row)

    assert converted.require_lift().lb is None
    assert converted.require_lift().open_side == "lower"
    assert converted.require_lift().ub == pytest.approx(
        math.expm1(row._posterior().isf(0.05)), rel=1e-14
    )


def test_open_bound_conversion_is_noop_for_two_sided():
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import open_bound_from_two_sided_at_target

    row = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.30 - 0.10,
        se_t=0.05,
        se_c=0.04,
        alpha=0.05,
        alternative="two-sided",
    )

    assert open_bound_from_two_sided_at_target(row) == row


def test_open_side_rejects_alternative_mismatch():
    with pytest.raises(InvalidRequestError) as exc_info:
        LiftEstimate(
            metric="m",
            group_id="t",
            estimand="itt",
            method="unadjusted",
            method_role="decision",
            inference="fixed",
            alternative="two-sided",
            scale="log",
            value_scale="relative",
            null_lift=0.0,
            lift=Estimate(
                value=0.1,
                lb=0.05,
                ub=None,
                level=0.95,
                alpha=0.05,
                open_side="upper",
            ),
        )
    assert exc_info.value.code == "estimation.results.lift.open_side_alternative_mismatch"


def test_open_side_round_trips_with_explicit_unbounded_endpoint():
    estimate = Estimate(
        value=0.1,
        lb=0.05,
        ub=None,
        level=0.95,
        alpha=0.05,
        open_side="upper",
    )

    dumped = estimate.model_dump()
    restored = Estimate.model_validate_json(estimate.model_dump_json())

    assert dumped["open_side"] == "upper"
    assert dumped["ub"] is None
    assert restored == estimate


def test_closed_interval_serialization_adds_null_open_side():
    estimate = Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95)

    assert estimate.model_dump()["open_side"] is None


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {"lb": None, "ub": None, "open_side": "upper"},
            "estimation.results.estimate.open_side_needs_calibrated_bound",
        ),
        (
            {"lb": 0.05, "ub": 0.15, "open_side": "upper"},
            "estimation.results.estimate.open_side_bound_must_be_none",
        ),
    ],
)
def test_open_side_rejects_invalid_bound_shapes(kwargs, code):
    with pytest.raises(InvalidRequestError) as exc_info:
        Estimate(value=0.1, level=0.95, **kwargs)
    assert exc_info.value.code == code


def test_open_side_excludes_uses_only_the_calibrated_bound():
    lower_open = Estimate(value=-0.1, lb=None, ub=-0.05, level=0.95, alpha=0.05, open_side="lower")
    upper_open = Estimate(value=0.1, lb=0.05, ub=None, level=0.95, alpha=0.05, open_side="upper")

    assert lower_open.excludes() is True
    assert upper_open.excludes() is True
    assert lower_open.excludes(-0.05) is False
    assert upper_open.excludes(0.05) is False


def test_open_bound_conversion_recovers_the_original_posterior():
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import open_bound_from_two_sided_at_target

    row = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.30 - 0.10,
        se_t=0.05,
        se_c=0.04,
        alpha=0.025,
        alternative="greater",
    )
    converted = open_bound_from_two_sided_at_target(row)

    # Equal posterior readouts at two distinct thresholds: the conversion
    # recovers the same posterior rather than a shifted or rescaled one.
    assert converted.prob_beyond(0.10) == pytest.approx(row.prob_beyond(0.10), rel=1e-14)
    assert converted.p_value() == pytest.approx(row.p_value(), rel=1e-14)


@pytest.mark.slow
def test_open_bound_conversion_preserves_always_valid_boundary():
    from fractions import Fraction

    from increment.estimation.results import open_bound_from_two_sided_at_target
    from increment.estimation.sequential_runtime import estimate_sequential
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli(alternative="greater", alpha=Fraction(1, 40))
    row = estimate_sequential(snapshot, policy).results[0]
    converted = open_bound_from_two_sided_at_target(row)
    assert converted is row
    assert converted.require_sequential_result().bounds.alpha == Fraction(1, 40)
    assert converted.require_sequential_result().bounds.alternative == "greater"
    assert converted.require_lift().open_side == "upper"


class TestEstimateLogFields:
    def test_default_none(self):
        est = Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95)
        assert est.log_mean is None
        assert est.log_se is None

    def test_settable(self):
        est = Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95, log_mean=0.0953, log_se=0.02)
        assert est.log_mean == 0.0953
        assert est.log_se == 0.02

    def test_frozen(self):
        est = Estimate(value=0.1, log_mean=0.0953)
        with pytest.raises((TypeError, ValueError)):
            est.log_mean = 0.5  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_no_interval_required(self):
        """log_mean/log_se are independent of the lb/ub-requires-level pairing."""
        est = Estimate(value=0.1, log_mean=0.0953, log_se=0.02)
        assert est.lb is None
        assert est.ub is None

    def test_invisible_to_to_frame(self):
        """log_mean/log_se on a nested Estimate never appear as to_frame columns;
        to_frame only ever reads value/lb/ub off the Estimate-typed field."""
        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(
                    value=0.1, lb=0.05, ub=0.15, level=0.95, log_mean=0.0953, log_se=0.02
                ),
            )
        ]
        frame = to_frame(estimates)
        assert "log_mean" not in frame.columns
        assert "log_se" not in frame.columns


class TestLiftEstimateAbsFields:
    def test_default_none(self):
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        )
        assert est.abs_diff is None
        assert est.abs_se is None
        assert est.null_abs is None
        assert est.abs_lb is None
        assert est.abs_ub is None

    def test_settable(self):
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            abs_diff=2.0,
            abs_se=0.4,
            null_abs=-0.01,
            abs_lb=1.2,
            abs_ub=2.8,
        )
        assert est.abs_diff == 2.0
        assert est.abs_se == 0.4
        assert est.null_abs == -0.01
        assert est.abs_lb == 1.2
        assert est.abs_ub == 2.8

    def test_frozen(self):
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1),
            abs_diff=2.0,
        )
        with pytest.raises((TypeError, ValueError)):
            est.abs_diff = 9.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_visible_as_to_frame_columns(self):
        """Dataframes preserve numeric additive estimates, bounds, and nulls."""
        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                abs_diff=2.0,
                abs_se=0.4,
                null_abs=-0.01,
                abs_lb=1.2,
                abs_ub=2.8,
            )
        ]
        frame = to_frame(estimates)
        assert isinstance(frame, pd.DataFrame)

        assert frame["abs_diff"].iloc[0] == 2.0
        assert frame["abs_se"].iloc[0] == 0.4
        assert frame["null_abs"].iloc[0] == -0.01
        assert frame["abs_lb"].iloc[0] == 1.2
        assert frame["abs_ub"].iloc[0] == 2.8


@pytest.mark.parametrize("field", ["abs_diff", "abs_se", "null_abs", "abs_lb", "abs_ub"])
def test_nonfinite_absolute_sidecars_are_rejected(field: str) -> None:
    kwargs: dict[str, Any] = {field: math.nan}
    with pytest.raises(InvalidRequestError) as raised:
        LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1),
            **kwargs,
        )
    assert raised.value.code == "model.field.nonfinite"


@pytest.mark.parametrize("bad_null_lift", [math.nan, math.inf, -math.inf])
def test_nonfinite_null_lift_is_rejected(bad_null_lift: float) -> None:
    with pytest.raises(InvalidRequestError) as raised:
        LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1),
            null_lift=bad_null_lift,
        )
    assert raised.value.code == "model.field.nonfinite"


def test_default_null_lift_still_zero() -> None:
    """Regression: omitting null_lift= is unaffected by the new Field
    constraint - the default still constructs and reads back as 0.0."""
    estimate = LiftEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1),
    )
    assert estimate.null_lift == 0.0


def test_unavailable_absolute_sidecars_remain_nullable_in_frame() -> None:
    estimate = LiftEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1),
    )
    frame = cast("pd.DataFrame", to_frame([estimate]))

    for field in ("abs_diff", "abs_se", "null_abs", "abs_lb", "abs_ub"):
        assert str(frame[field].dtype) == "Float64"
        assert frame[field].iloc[0] is pd.NA

    assert frame["reference_kind"].iloc[0] == "normal"
    assert str(frame["reference_df"].dtype) == "Float64"
    assert frame["reference_df"].iloc[0] is pd.NA


class TestLiftEstimateReferenceValidation:
    def _kwargs(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "metric": "rev",
            "group_id": "treatment",
            "method": "unadjusted",
            "method_role": "decision",
            "lift": Estimate(value=0.1),
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize("reference_df", [None, 0.0, -1.0, math.inf, math.nan])
    def test_constructor_rejects_t_without_finite_positive_df(self, reference_df):
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(**self._kwargs(reference_kind="t", reference_df=reference_df))
        assert exc_info.value.code == "estimation.results.lift.reference_df_required_for_t"

    @pytest.mark.parametrize("reference_kind", ["normal", "sequential"])
    def test_rejects_reference_df_for_a_non_t_reference(self, reference_kind):
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(**self._kwargs(reference_kind=reference_kind, reference_df=5.0))
        assert exc_info.value.code == "estimation.results.lift.reference_df_set_for_normal"

    def test_model_validate_rejects_dof_reference_df_mismatch(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate.model_validate(self._kwargs(dof=5.0, reference_kind="t", reference_df=8.0))
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    @pytest.mark.parametrize(
        ("inference", "reference_kind"),
        [("always_valid", "normal"), ("fixed", "sequential")],
    )
    def test_rejects_inference_reference_mismatch(self, inference, reference_kind):
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(**self._kwargs(inference=inference, reference_kind=reference_kind))
        assert exc_info.value.code == "estimation.results.lift.inference_reference_kind_mismatch"

    def test_rejects_sequential_reference_with_legacy_dof(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(
                **self._kwargs(
                    inference="always_valid",
                    dof=5.0,
                    reference_kind="sequential",
                )
            )
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_model_validate_infers_t_reference_from_legacy_dof(self):
        payload = self._kwargs(dof=12.0)
        before = payload.copy()
        est = LiftEstimate.model_validate(payload)
        assert est.reference_kind == "t"
        assert est.reference_df == 12.0
        assert payload == before

    def test_legacy_sequential_label_requires_raw_checkpoint(self):
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            LiftEstimate.model_validate(self._kwargs(inference="always_valid"))
        assert raised.value.code == "sequential.continuation.legacy"

    @pytest.mark.parametrize(
        "explicit",
        [{"reference_kind": "normal"}, {"reference_df": None}],
    )
    def test_explicit_reference_keys_suppress_legacy_dof_inference(self, explicit):
        payload = self._kwargs(dof=5.0, **explicit)
        before = payload.copy()
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate.model_validate(payload)
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"
        assert payload == before

    def test_t_reference_round_trips_through_json(self):
        est = LiftEstimate(**self._kwargs(dof=5.0, reference_kind="t", reference_df=5.0))
        again = LiftEstimate.model_validate_json(est.model_dump_json())
        assert again == est
        assert again.model_dump()["reference_kind"] == "t"
        assert again.model_dump()["reference_df"] == 5.0


class TestLiftEstimatePreferredDirection:
    """preferred_direction default/settability - the schema-contract half
    of the fix; the behavioral guards it enables (prob_favorable() etc.
    refusing when it's None) are pinned in test_decision_stats.py and
    test_inference.py, not duplicated here."""

    def test_default_none(self):
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        )
        assert est.preferred_direction is None

    def test_settable(self):
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            preferred_direction="decrease",
        )
        assert est.preferred_direction == "decrease"


class TestEstimateExcludes:
    """excludes(null) is True iff the interval lies STRICTLY on one side of
    null - a null sitting exactly on an endpoint is NOT excluded."""

    LB = 0.05
    UB = 0.15

    def _est(self):
        return Estimate(value=0.1, lb=self.LB, ub=self.UB, level=0.95)

    def test_null_exactly_on_lower_endpoint_not_excluded(self):
        assert self._est().excludes(self.LB) is False

    def test_null_exactly_on_upper_endpoint_not_excluded(self):
        assert self._est().excludes(self.UB) is False

    def test_null_just_inside_lower_endpoint_not_excluded(self):
        assert self._est().excludes(math.nextafter(self.LB, math.inf)) is False

    def test_null_just_inside_upper_endpoint_not_excluded(self):
        assert self._est().excludes(math.nextafter(self.UB, -math.inf)) is False

    def test_null_just_below_lower_endpoint_excluded(self):
        assert self._est().excludes(math.nextafter(self.LB, -math.inf)) is True

    def test_null_just_above_upper_endpoint_excluded(self):
        assert self._est().excludes(math.nextafter(self.UB, math.inf)) is True

    def test_default_null_is_zero(self):
        assert self._est().excludes() is True  # [0.05, 0.15] sits above 0

    def test_point_only_estimate_never_excludes(self):
        assert Estimate(value=0.1).excludes(0.0) is False


class TestIntervalWidth:
    """The interval infer_lift stamps onto its Estimate is the symmetric
    z-quantile of the Normal posterior on the LOG scale: log-scale width
    == 2 * z * se, with z read off the nominal level."""

    def test_infer_lift_log_scale_width_is_two_z_se(self):
        from scipy.stats import norm

        from increment.estimation.inference import infer_lift

        se_t, se_c = 0.03, 0.04
        se = math.sqrt(se_t**2 + se_c**2)  # 0.05, arms independent
        est = infer_lift(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            log_rr=0.15 - 0.05,
            se_t=se_t,
            se_c=se_c,
            alpha=0.05,
        )
        lift = est.require_lift()
        assert lift.level == pytest.approx(0.95)
        z = norm.ppf((1.0 + 0.95) / 2.0)
        assert lift.lb is not None and lift.ub is not None
        # lb/ub are expm1-transformed; the symmetric width lives on the log scale.
        width_log = math.log1p(lift.ub) - math.log1p(lift.lb)
        assert width_log == pytest.approx(2.0 * z * se, rel=1e-9)

    def test_infer_lift_interval_centered_on_log_point(self):
        from increment.estimation.inference import infer_lift

        est = infer_lift(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            log_rr=0.15 - 0.05,
            se_t=0.03,
            se_c=0.04,
        )
        lift = est.require_lift()
        assert lift.lb is not None and lift.ub is not None
        mid_log = (math.log1p(lift.lb) + math.log1p(lift.ub)) / 2.0
        assert mid_log == pytest.approx(math.log1p(lift.value), rel=1e-9)


class TestToFrameNumericRoundTrip:
    """Every numeric field survives to_frame bit-for-bit - no float
    munging, rounding, or dtype coercion on the way into the frame."""

    def test_values_unchanged_in_frame(self):
        value = 0.123456789012345
        lb = -0.0123456789012345
        ub = 0.271828182845904
        abs_diff = 2.5000000000001
        abs_se = 0.3750000000002
        null_lift = 0.0100000000003
        null_abs = -0.0100000000004
        abs_lb = 1.7650000000005
        abs_ub = 3.2350000000006
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            null_lift=null_lift,
            lift=Estimate(value=value, lb=lb, ub=ub, level=0.9),
            abs_diff=abs_diff,
            abs_se=abs_se,
            null_abs=null_abs,
            abs_lb=abs_lb,
            abs_ub=abs_ub,
        )
        frame = to_frame([est])
        assert isinstance(frame, pd.DataFrame)
        assert frame["lift"].iloc[0] == value
        assert frame["lb"].iloc[0] == lb
        assert frame["ub"].iloc[0] == ub
        assert frame["abs_diff"].iloc[0] == abs_diff
        assert frame["abs_se"].iloc[0] == abs_se
        assert frame["null_lift"].iloc[0] == null_lift
        assert frame["null_abs"].iloc[0] == null_abs
        assert frame["abs_lb"].iloc[0] == abs_lb
        assert frame["abs_ub"].iloc[0] == abs_ub


class TestEstimateAdversarialConstruction:
    """Estimate is the frozen public value object downstream consumers
    (meta-analysis, breakout, viz, serialization round-trips) trust;
    inverted or out-of-range constructions must refuse at the boundary
    instead of letting excludes() answer confidently. Estimates are
    finite-only; unavailable rows are represented by their surrounding
    result models rather than an Estimate placeholder."""

    def test_inverted_interval_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.0, lb=5.0, ub=-5.0, level=0.95)
        assert exc_info.value.code == "estimation.results.estimate.interval_inverted_lb"

    def test_level_above_one_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.0, lb=-1.0, ub=1.0, level=17.5)
        assert exc_info.value.code == "estimation.results.estimate.level_interval"

    def test_level_zero_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.0, lb=-1.0, ub=1.0, level=0.0)
        assert exc_info.value.code == "estimation.results.estimate.level_interval"

    def test_nan_level_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.0, lb=-1.0, ub=1.0, level=float("nan"))
        assert exc_info.value.code == "estimation.results.estimate.level_interval"

    def test_interval_without_level_refused_even_when_omitted(self):
        """The documented invariant is "level required iff lb/ub set".
        Pydantic field validators skip unprovided defaults, so omitting
        level entirely used to slip past the check that an explicit
        level=None tripped - two spellings of the same input must
        validate identically."""
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.0, lb=1.0, ub=2.0)
        assert exc_info.value.code == "estimation.results.estimate.level_lb_ub"
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.0, lb=1.0, ub=2.0, level=None)
        assert exc_info.value.code == "estimation.results.estimate.level_lb_ub"

    def test_nonfinite_values_refused(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(InvalidRequestError) as raised:
                Estimate(value=bad)
            assert raised.value.code == "model.field.nonfinite"

    def test_point_only_has_no_level(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.1, level=0.95)
        assert exc_info.value.code == "estimation.results.estimate.level_none_lb"

    def test_half_nan_interval_refused(self):
        """One NaN endpoint beside a finite one is corrupt, not a placeholder."""
        for _field, bounds in (("ub", (5.0, float("nan"))), ("lb", (float("nan"), 5.0))):
            with pytest.raises(InvalidRequestError) as raised:
                Estimate(value=0.0, lb=bounds[0], ub=bounds[1], level=0.95)
            assert raised.value.code == "model.field.nonfinite"

    def test_degenerate_interval_allowed(self):
        est = Estimate(value=0.1, lb=0.1, ub=0.1, level=0.95)
        assert est.lb == est.ub == 0.1

    def test_valid_estimate_unchanged(self):
        est = Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95)
        assert est.excludes(0.0) is True

    def test_conflicting_level_and_effective_alpha_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95, alpha=0.10)
        assert exc_info.value.code == "estimation.results.estimate.level_contradicts_effective"

    def test_posterior_rechecks_alpha_after_model_copy(self):
        estimate = Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95, alpha=0.05)
        result = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=estimate,
        )
        tampered = result.model_copy(update={"lift": estimate.model_copy(update={"alpha": 0.10})})
        with pytest.raises(InvalidRequestError) as exc_info:
            tampered.prob_beyond(0.0)
        assert exc_info.value.code == "estimation.results.lift.liftestimate_level_contradicts"

    def test_posterior_rechecks_alpha_on_prior_backed_estimates(self):
        from increment.estimation.priors import StudentTPrior

        estimate = Estimate(
            value=0.1,
            lb=0.05,
            ub=0.15,
            level=0.95,
            alpha=0.05,
            log_mean=0.09531,
            log_se=0.02,
        )
        result = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=estimate,
            prior_spec=StudentTPrior(nu=4.0, scale=0.05),
        )
        tampered = result.model_copy(update={"lift": estimate.model_copy(update={"alpha": 0.10})})
        with pytest.raises(InvalidRequestError) as exc_info:
            tampered.prob_beyond(0.0)
        assert exc_info.value.code == "estimation.results.lift.liftestimate_level_contradicts"


def test_lift_estimate_carries_winsorization_diagnostics():
    est = LiftEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1),
        winsor_upper_percentile=0.99,
        winsor_upper_bound=100.0,
        winsor_control_n=10,
        winsor_control_n_lower=0,
        winsor_control_n_upper=1,
        winsor_treatment_n=10,
        winsor_treatment_n_lower=0,
        winsor_treatment_n_upper=2,
    )
    assert est.winsor_control_fraction_upper == 0.1
    assert est.winsor_treatment_fraction_upper == 0.2
    assert "winsor_upper_bound" in to_frame([est]).columns


def _clustered_lift_estimate(
    *,
    log_mean: float,
    log_se: float,
    dof: float,
    alternative: str = "two-sided",
    inference: str = "fixed",
    null_lift: float = 0.0,
) -> LiftEstimate:
    """A hand-built cluster-robust row: the interval is a t_{dof} quantile
    pair, so `_posterior()` refuses it - `p_value()` must read
    lift.log_mean/lift.log_se/dof directly instead."""
    return LiftEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        inference=inference,
        alternative=alternative,
        null_lift=null_lift,
        dof=dof,
        lift=Estimate(value=math.expm1(log_mean), log_mean=log_mean, log_se=log_se),
    )


class TestPValueClusterRobust:
    """p_value() on a `dof is not None` row reads its t reference directly,
    matching scipy's own alternative= convention ("two-sided"/"greater"/
    "less") since LiftEstimate.alternative reuses those exact literals."""

    def test_two_sided_matches_scipy_t_sf(self):
        log_mean, log_se, dof = 0.15, 0.05, 8.0
        est = _clustered_lift_estimate(log_mean=log_mean, log_se=log_se, dof=dof)
        want = 2.0 * _t.sf(abs(log_mean / log_se), dof)
        assert est.p_value() == pytest.approx(want)

    def test_greater_uses_upper_tail(self):
        log_mean, log_se, dof = 0.15, 0.05, 8.0
        est = _clustered_lift_estimate(
            log_mean=log_mean, log_se=log_se, dof=dof, alternative="greater"
        )
        want = _t.sf(log_mean / log_se, dof)
        assert est.p_value() == pytest.approx(want)

    def test_less_uses_lower_tail(self):
        log_mean, log_se, dof = 0.15, 0.05, 8.0
        est = _clustered_lift_estimate(
            log_mean=log_mean, log_se=log_se, dof=dof, alternative="less"
        )
        want = _t.cdf(log_mean / log_se, dof)
        assert est.p_value() == pytest.approx(want)

    def test_greater_and_less_are_complementary_tails(self):
        # A negative z: "greater" (upper tail) is now the likely tail and
        # "less" (lower tail) the unlikely one - proves the sign is read
        # from z, not hardcoded to whichever tail happened to be small in
        # the positive-z cases above.
        log_mean, log_se, dof = -0.15, 0.05, 8.0
        est_greater = _clustered_lift_estimate(
            log_mean=log_mean, log_se=log_se, dof=dof, alternative="greater"
        )
        est_less = _clustered_lift_estimate(
            log_mean=log_mean, log_se=log_se, dof=dof, alternative="less"
        )
        assert est_greater.p_value() == pytest.approx(_t.sf(log_mean / log_se, dof))
        assert est_less.p_value() == pytest.approx(_t.cdf(log_mean / log_se, dof))
        assert est_greater.p_value() > est_less.p_value()

    def test_public_gaussian_sequential_estimate_refuses_before_row_access(self):
        from increment import estimate_sequential
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        snapshot, _, policy = raw_gaussian(n=4)
        with pytest.raises(CapabilityError) as raised:
            estimate_sequential(snapshot, policy)
        assert raised.value.code == "sequential.route.unsupported"

    def test_dof_none_path_two_sided_default_is_unchanged(self):
        """Regression: a non-clustered (dof=None) row at the default
        alternative="two-sided"/null_lift=0.0 computes the same
        always-two-sided-against-zero p-value as before."""
        z = _t.ppf(0.975, 1e9)  # ~ norm.ppf(0.975)
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(
                value=math.expm1(0.10),
                lb=math.expm1(0.10 - z * 0.05),
                ub=math.expm1(0.10 + z * 0.05),
                level=0.95,
                alpha=0.05,
            ),
        )
        from scipy.stats import norm

        assert est.p_value() == pytest.approx(2 * norm.cdf(-2.0), rel=1e-6)

    def test_dof_none_path_honors_alternative(self):
        """A non-clustered (dof=None) row now honors a declared one-sided
        `alternative` -- a "greater" test reads the one-sided tail (half
        the always-two-sided-against-zero value this used to return),
        matching the cluster-robust branch's own alternative-aware
        convention."""
        z = _t.ppf(0.975, 1e9)  # ~ norm.ppf(0.975)
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            alternative="greater",
            lift=Estimate(
                value=math.expm1(0.10),
                lb=math.expm1(0.10 - z * 0.05),
                ub=math.expm1(0.10 + z * 0.05),
                level=0.95,
                alpha=0.05,
            ),
        )
        from scipy.stats import norm

        assert est.p_value() == pytest.approx(norm.sf(2.0), rel=1e-6)

    def test_dof_none_path_honors_null_lift(self):
        """A non-clustered row with a nonzero declared `null_lift` (margin)
        computes its p-value against that shifted null, not 0 -- the
        margin sibling of `test_dof_none_path_honors_alternative`."""
        mu, sigma = math.log1p(0.10), 0.02
        z = _t.ppf(0.975, 1e9)
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            alternative="greater",
            null_lift=0.05,
            lift=Estimate(
                value=math.expm1(mu),
                lb=math.expm1(mu - z * sigma),
                ub=math.expm1(mu + z * sigma),
                level=0.95,
                alpha=0.05,
            ),
        )
        from scipy.stats import norm

        z_against_margin = (mu - math.log1p(0.05)) / sigma
        assert est.p_value() == pytest.approx(norm.sf(z_against_margin), rel=1e-6)
        # Sanity: this must differ from testing against 0 -- proves the
        # shifted null actually moved the answer, not a no-op.
        assert est.p_value() != pytest.approx(norm.sf(mu / sigma), rel=1e-3)

    def test_dof_set_path_honors_null_lift(self):
        """A cluster-robust row with a declared margin must test against the
        shifted null on the t reference, not against zero."""
        from scipy.stats import t as t_dist

        est = _clustered_lift_estimate(
            log_mean=0.05, log_se=0.02, dof=24, alternative="greater", null_lift=0.04
        )
        null = math.log1p(0.04)
        expected = float(t_dist.sf((0.05 - null) / 0.02, 24))
        assert est.p_value() == pytest.approx(expected)
        # against-zero would be sf(2.5) — assert we are NOT computing that
        assert est.p_value() != pytest.approx(float(t_dist.sf(2.5, 24)))

    def test_dof_set_path_domain_guard_raises_for_null_lift_at_or_below_negative_one(self):
        """The log-scale domain guard mirrored from the dof-None branch
        must also fire on the dof-set (cluster-robust) branch - a
        null_lift <= -1 is not representable on the log scale regardless
        of which branch computes the p-value."""
        est = _clustered_lift_estimate(
            log_mean=0.05, log_se=0.02, dof=24, alternative="greater", null_lift=-1.0
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            est.p_value()
        assert (
            exc_info.value.code == "estimation.results.lift.p_value_null_lift_not_representable_dof"
        )

    def test_dof_set_path_honors_null_lift_two_sided(self):
        """The shifted null must also compose correctly with the default
        two-sided tail, not just "greater" - the two-sided formula reuses
        the same transformed `z`, but this is unverified by the
        "greater"-only case above."""
        from scipy.stats import t as t_dist

        est = _clustered_lift_estimate(log_mean=0.05, log_se=0.02, dof=24, null_lift=0.04)
        null = math.log1p(0.04)
        expected = 2.0 * float(t_dist.sf(abs((0.05 - null) / 0.02), 24))
        assert est.p_value() == pytest.approx(expected)
        assert est.p_value() != pytest.approx(2.0 * float(t_dist.sf(2.5, 24)))

    def test_missing_log_stats_raises_with_accurate_message(self):
        """A dof-set row with no log-scale sufficient statistics is the
        shape every clustered infer_ate/estimate_encouragement row actually
        has (Estimate(value, lb, ub, level) with dof forwarded, no
        log_mean/log_se) - it is a real production row, not a hand-built
        fake, so the guard message must not claim otherwise."""
        est = LiftEstimate(
            metric="rev",
            group_id="treatment",
            method="iptw",
            method_role="decision",
            dof=8.0,
            lift=Estimate(value=0.1, lb=0.02, ub=0.18, level=0.95),
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            est.p_value()
        assert exc_info.value.code == "estimation.results.lift.liftestimate_dof_set"

    def test_nonpositive_log_se_raises_value_error_not_zero_division(self):
        est = _clustered_lift_estimate(log_mean=0.15, log_se=0.0, dof=8.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            est.p_value()
        assert exc_info.value.code == "estimation.results.lift.liftestimate_log_se"


class TestPValueQuantile:
    """p_value() on a row with `quantile_p_value` set returns that stored
    value directly, regardless of `null_lift`/`alternative`/`lift` -- it
    was computed once by inverting the quantile estimator's own interval
    construction and is not re-derived here."""

    def test_returns_the_stored_quantile_p_value(self):
        est = LiftEstimate(
            metric="lat",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, log_mean=math.log1p(0.1), log_se=0.05),
            quantile_p_value=0.0321,
        )
        assert est.p_value() == 0.0321

    def test_quantile_p_value_wins_over_a_nonzero_null_lift(self):
        """The stored value is returned as-is -- it is not re-tested
        against this row's own `null_lift`, since the inversion already
        baked `null=0.0` into the computation (quantile rows always test
        against zero; see the module's merge-order note)."""
        est = LiftEstimate(
            metric="lat",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            null_lift=0.05,
            lift=Estimate(value=0.1, log_mean=math.log1p(0.1), log_se=0.05),
            quantile_p_value=0.0321,
        )
        assert est.p_value() == 0.0321


def test_lift_estimate_repr_surfaces_published_decision_fields():
    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        estimand="itt",
        discovery=True,
        lift=Estimate(value=0.12, lb=0.03, ub=0.21, level=0.95),
    )

    text = repr(estimate)

    assert "metric='revenue'" in text
    assert "group='treatment'" in text
    assert "value=+0.12 (relative)" in text
    assert "interval=(+0.03, +0.21)" in text
    assert "stat_sig=True" in text
    assert "discovery=True" in text


def test_lift_estimate_repr_does_not_cross_scales_for_absolute_null():
    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        null_abs=5.0,
        lift=Estimate(value=0.12, lb=0.03, ub=0.21, level=0.95),
    )

    assert "stat_sig=False" in repr(estimate)


def test_lift_estimate_repr_distinguishes_open_from_unavailable_interval():
    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        alternative="greater",
        lift=Estimate(
            value=0.12,
            lb=0.03,
            ub=None,
            open_side="upper",
            level=0.95,
            alpha=0.05,
        ),
    )

    text = repr(estimate)

    assert "interval=(+0.03, +inf)" in text
    assert "stat_sig=True" in text


def test_closed_normal_reconstruction_refuses_unrepresentable_tail():
    result = LiftEstimate(
        metric="value",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=1.0, alpha=math.ulp(0.0)),
        scale="linear",
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        result.prob_beyond(0.0)
    assert exc_info.value.code == "estimation.tails.unresolvable"


def _binomial_set(**overrides: Any) -> BinomialConfidenceSet:
    base: dict[str, Any] = {
        "lower": -0.5,
        "upper": 0.6666666666666667,
        "alpha": 0.05,
        "decision_alpha": 0.05,
        "level": 0.95,
        "geometry": "central",
        "x_c": 3,
        "n_c": 10,
        "x_t": 5,
        "n_t": 10,
        "nuisance_beta": 1e-6,
    }
    base.update(overrides)
    return BinomialConfidenceSet(**base)


def _binomial_row(**overrides: Any) -> dict[str, Any]:
    bset_override = overrides.pop("binomial_set", None)
    bset: BinomialConfidenceSet = bset_override if bset_override is not None else _binomial_set()
    if "lift" in overrides:
        lift = overrides.pop("lift")
    else:
        point = (bset.n_c * bset.x_t) / (bset.n_t * bset.x_c) - 1.0
        lift = Estimate(
            value=point, lb=bset.lower, ub=bset.upper, level=bset.level, alpha=bset.alpha
        )
    base: dict[str, Any] = {
        "metric": "m",
        "group_id": "g",
        "method": "binomial",
        "method_role": "decision",
        "alternative": "two-sided",
        "reference_kind": "binomial",
        "lift": lift,
        "binomial_set": bset,
        "scale": "linear",
    }
    base.update(overrides)
    return base


class TestBinomialConfidenceSetInvariants:
    def test_zero_control_with_finite_upper_refuses(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            _binomial_set(x_c=0, upper=0.8)
        assert exc_info.value.code == "estimation.results.binomial.counts_out_of_range"

    def test_zero_control_with_none_upper_is_accepted(self):
        bset = _binomial_set(x_c=0, upper=None)
        assert bset.point_available is False


class TestLiftEstimateBinomialCrossInvariants:
    """The persisted counts, point availability, displayed bounds,
    alpha/level, and alternative/geometry must never contradict each
    other -- a `p_value()`/`stat_sig()` caller (counts-based) and a
    display consumer (bounds-based) must always agree.
    """

    def test_astra_contradictory_row_is_refused(self):
        """Zero-control counts (no finite point) paired with a finite
        lift -- every piece of this row individually looked plausible;
        only the cross-check catches the contradiction.
        """
        bset = _binomial_set(x_c=0, upper=None, geometry="central")
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(
                **_binomial_row(
                    lift=Estimate(value=0.0, lb=-0.5, ub=0.5, level=0.95, alpha=0.05),
                    binomial_set=bset,
                )
            )
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_finite_lift_with_point_unavailable_set_refuses(self):
        bset = _binomial_set(x_c=0, upper=None)
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(
                **_binomial_row(
                    lift=Estimate(value=0.0, lb=-1.0, ub=0.8, level=0.95, alpha=0.05),
                    binomial_set=bset,
                )
            )
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_none_lift_with_point_available_set_refuses(self):
        bset = _binomial_set()  # x_c=3 > 0: point available
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(**_binomial_row(lift=None, binomial_set=bset))
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_lift_value_mismatched_with_persisted_counts_refuses(self):
        bset = _binomial_set(x_c=3, n_c=10, x_t=5, n_t=10)
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(
                **_binomial_row(
                    lift=Estimate(
                        value=999.0, lb=bset.lower, ub=bset.upper, level=0.95, alpha=0.05
                    ),
                    binomial_set=bset,
                )
            )
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_lift_bounds_mismatched_with_binomial_set_bounds_refuses(self):
        bset = _binomial_set(lower=-0.5, upper=0.8)
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(
                **_binomial_row(
                    lift=Estimate(
                        value=0.6666666666666667, lb=-0.1, ub=2.0, level=0.95, alpha=0.05
                    ),
                    binomial_set=bset,
                )
            )
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_alpha_level_mismatched_with_binomial_set_refuses(self):
        bset = _binomial_set(alpha=0.05, level=0.95)
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(
                **_binomial_row(
                    lift=Estimate(
                        value=0.6666666666666667,
                        lb=bset.lower,
                        ub=bset.upper,
                        level=0.90,
                        alpha=0.10,
                    ),
                    binomial_set=bset,
                )
            )
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_geometry_alternative_mismatch_refuses(self):
        # alternative="greater" implies geometry="lower_bound"; labeling
        # a structurally-central (finite upper, x_c > 0) set "central"
        # while claiming "greater" must be refused even though every
        # OTHER field (open_side, lb/ub, alpha/level) is self-consistent.
        bset = _binomial_set(geometry="central", upper=0.8)
        lift = Estimate(
            value=0.6666666666666667,
            lb=bset.lower,
            ub=None,
            level=0.95,
            alpha=0.05,
            open_side="upper",
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate(**_binomial_row(alternative="greater", lift=lift, binomial_set=bset))
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"

    def test_consistent_row_round_trips_through_json(self):
        row = LiftEstimate(**_binomial_row())
        assert LiftEstimate.model_validate_json(row.model_dump_json()) == row

    def test_json_mutation_reintroducing_the_contradiction_refuses(self):
        """A valid row dumped to JSON, then mutated to independently
        change the binomial_set's bounds (as if a consumer patched only
        one representation), must fail re-validation.
        """
        row = LiftEstimate(**_binomial_row())
        payload = row.model_dump()
        payload["binomial_set"] = {**payload["binomial_set"], "upper": 999.0}
        with pytest.raises(InvalidRequestError) as exc_info:
            LiftEstimate.model_validate(payload)
        assert exc_info.value.code == "estimation.results.lift.binomial_lift_availability"


def _registered_sequential_row():
    from increment import estimate_sequential
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli(n=4)
    return estimate_sequential(snapshot, policy).results[0]


def test_winsor_confidence_set_cannot_carry_sequential_evidence():
    """A real winsor row must not bypass the sequential-authority invariant."""
    from increment.errors import CapabilityError
    from increment.estimation.winsor import estimate_winsor_lift
    from tests.estimation.test_winsor_bootstrap import _state

    raw = _state(stream=23)
    row = estimate_winsor_lift(raw, "C", "T")
    assert row.confidence_set is not None

    payload = row.model_dump()
    payload["sequential_result"] = _registered_sequential_row().sequential_result.model_dump()

    with pytest.raises(CapabilityError) as exc_info:
        LiftEstimate.model_validate(payload)
    assert exc_info.value.code == "sequential.source.invalid"


def test_joint_relative_set_cannot_carry_sequential_evidence():
    """A real joint-relative row must not bypass the sequential-authority invariant."""
    from scipy.stats import norm as _norm

    from increment.errors import CapabilityError
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    relative = relative_confidence_set(
        JointContrastReference(a=2.0, c=3.0, var_a=0.04, var_c=0.09, cov_ac=0.01)
    )
    # The additive sidecar's own numbers are incidental here; they only have to
    # be the two-sided 0.05 Wald interval of (2.0, 0.2).
    half_width = _norm.isf(0.025) * 0.2
    row = LiftEstimate(
        metric="m",
        group_id="g",
        method="unadjusted",
        method_role="decision",
        inference="fixed",
        reference_kind="normal",
        value_scale="relative",
        scale="linear",
        lift=relative.estimate(),
        relative_confidence_set=relative,
        abs_diff=2.0,
        abs_se=0.2,
        abs_lb=2.0 - half_width,
        abs_ub=2.0 + half_width,
        abs_reference_kind="normal",
    )
    payload = row.model_dump()
    payload["sequential_result"] = _registered_sequential_row().sequential_result.model_dump()

    with pytest.raises(CapabilityError) as exc_info:
        LiftEstimate.model_validate(payload)
    assert exc_info.value.code == "sequential.source.invalid"


class TestNonpositiveArmMeanUnavailableReason:
    """The new relative_unavailable_reason value ('nonpositive_arm_mean')
    is a deliberate, always-conservative stand-in -- unlike the two
    existing joint-covariance reasons, its p_value() is KNOWN (the
    maximal, never-rejecting value), not indeterminate."""

    def _row(self, **overrides: Any) -> LiftEstimate:
        fields: dict[str, Any] = {
            "metric": "refunds",
            "group_id": "treatment",
            "method": "unadjusted",
            "method_role": "decision",
            "alternative": "two-sided",
            "null_lift": 0.0,
            "lift": None,
            "scale": "linear",
            "relative_confidence_set": None,
            "relative_unavailable_reason": "nonpositive_arm_mean",
            "abs_diff": -1.0,
            "abs_se": 0.05,
            "abs_lb": -1.1,
            "abs_ub": -0.9,
            "abs_reference_kind": "t",
            "abs_reference_df": 398.0,
            "reference_kind": "t",
            "reference_df": 398.0,
        }
        fields.update(overrides)
        return LiftEstimate(**fields)

    def test_p_value_returns_the_maximal_conservative_value(self):
        assert self._row().p_value() == 1.0

    def test_stat_sig_is_false(self):
        assert self._row().stat_sig() is False

    def test_existing_joint_covariance_reasons_still_raise(self):
        row = self._row(relative_unavailable_reason="joint_covariance_indefinite")
        with pytest.raises(CodedError) as exc_info:
            row.p_value()
        assert exc_info.value.code == "estimation.results.joint.unavailable"

    def test_null_abs_declared_tests_the_additive_scale_normally(self):
        """A guardrail/secondary with an absolute margin must still get
        a real additive significance test on a genuine degenerate cell --
        null_abs is checked first, unaffected by the relative scale being
        unavailable. Drives a real ``estimate_lift()`` row (not a hand-built
        row with overridden reference fields) so this exercises whatever
        reference kind engine.py actually produces for this hazard."""
        from increment.estimation.armstats import ArmStats
        from increment.estimation.decision_types import PValueEvidence
        from increment.estimation.engine import estimate_lift
        from increment.semantics.models import MeanMetric

        control = ArmStats.from_raw_sums(
            study_id="e",
            metric="refunds",
            group_id="control",
            n=200,
            sum_y=200.0,
            sum_y2=207.96,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="e",
            metric="refunds",
            group_id="treatment",
            n=200,
            sum_y=0.0,
            sum_y2=0.0,
        )
        rows = [
            {
                "experiment_id": arm.study_id,
                "metric": arm.metric,
                "group_id": arm.group_id,
                "n": float(arm.n),
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
            }
            for arm in (control, treatment)
        ]
        computation = estimate_lift(
            [MeanMetric(name="refunds", entity="user", fact="refunds")],
            rows,
            control_group="control",
            null_abs=-0.5,
            alternative="less",
        )
        (row,) = computation.results
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert row.stat_sig() is True  # abs_ub near -1.0, well below null_abs=-0.5
        (evidence,) = computation.evidence.values()
        assert isinstance(evidence, PValueEvidence)
        assert evidence.p_value < 0.05


class TestJointRelativeRowsExcludesNonpositiveArmMean:
    """readouts._common._joint_relative_rows decides whether a metric's FCR
    re-estimation forces two-sided (a genuine joint/Fieller construction
    is inherently two-sided) or keeps the declared one-sided alternative.
    A nonpositive_arm_mean row is an ordinary additive Wald result, not
    a joint construction, and must not taint its metric's OTHER arms."""

    def _row(self, **overrides: Any) -> LiftEstimate:
        from increment.estimation.results import LiftEstimate

        fields: dict[str, Any] = {
            "metric": "refunds",
            "group_id": "treatment",
            "method": "unadjusted",
            "method_role": "decision",
            "estimand": "itt",
            "alternative": "two-sided",
            "null_lift": 0.0,
            "lift": None,
            "scale": "linear",
            "relative_confidence_set": None,
            "relative_unavailable_reason": "nonpositive_arm_mean",
            "abs_diff": -1.0,
            "abs_se": 0.05,
            "abs_lb": -1.1,
            "abs_ub": -0.9,
            "abs_reference_kind": "t",
            "abs_reference_df": 398.0,
            "reference_kind": "t",
            "reference_df": 398.0,
        }
        fields.update(overrides)
        return LiftEstimate(**fields)

    def test_a_nonpositive_arm_mean_row_alone_is_not_joint(self):
        from increment.readouts._common import _joint_relative_rows

        assert _joint_relative_rows([self._row()], metric="refunds") is False

    def test_a_genuine_joint_covariance_row_is_still_joint(self):
        from increment.readouts._common import _joint_relative_rows

        row = self._row(relative_unavailable_reason="joint_covariance_indefinite")
        assert _joint_relative_rows([row], metric="refunds") is True
