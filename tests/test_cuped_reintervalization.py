from __future__ import annotations

import math
from typing import Literal, cast

import pytest
from scipy.stats import t

from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
from increment.estimation.inference import infer_ate
from increment.estimation.results import Estimate, LiftEstimate


def _independent_mean_row() -> LiftEstimate:
    reference = IndependentMeanReference(
        components=(
            IndependentMeanComponent.from_values("control", [1.0, 2.0, 4.0], coefficient=-1),
            IndependentMeanComponent.from_values("treatment", [3.0, 5.0, 6.0], coefficient=1),
        )
    )
    return infer_ate(
        "m",
        "treatment",
        "independent_mean",
        reference.point,
        reference,
        method_role="decision",
        value_scale="absolute",
    )


def _t_row(
    *, scale: Literal["log", "linear"] = "log", alternative: str = "two-sided"
) -> LiftEstimate:
    lift = Estimate(
        value=math.expm1(0.2) if scale == "log" else 0.2,
        lb=math.expm1(0.1) if scale == "log" else 0.1,
        ub=math.expm1(0.3) if scale == "log" else 0.3,
        alpha=0.1,
        level=0.9,
        log_mean=0.2,
        log_se=0.05,
    )
    return LiftEstimate(
        metric="m",
        method_role="decision",
        group_id="t",
        method="linear",
        lift=lift,
        alternative=alternative,
        abs_diff=0.2,
        abs_se=0.05,
        abs_lb=0.1,
        abs_ub=0.3,
        abs_reference_kind="t",
        abs_reference_df=10,
        abs_alpha=0.1,
        reference_kind="t",
        reference_df=10,
        dof=10,
        scale=scale,
    )


def test_t_reintervalization_keeps_linear_estimates_on_the_additive_scale():
    row = _t_row(scale="linear")
    projected = row.reintervalize(0.05)
    critical = float(t.isf(0.025, 10))

    assert projected.require_lift().value == pytest.approx(0.2)
    assert projected.require_lift().lb == pytest.approx(0.2 - critical * 0.05)
    assert projected.require_lift().ub == pytest.approx(0.2 + critical * 0.05)
    assert projected.abs_lb == pytest.approx(0.2 - critical * 0.05)
    assert projected.abs_ub == pytest.approx(0.2 + critical * 0.05)


def test_directional_t_reintervalization_matches_estimator_alpha_convention():
    row = _t_row(alternative="greater")
    projected = row.reintervalize(0.025)
    critical = float(t.isf(0.025, 10))

    assert projected.require_lift().alpha == pytest.approx(0.05)
    assert projected.require_lift().level == pytest.approx(0.95)
    assert projected.require_lift().lb == pytest.approx(math.expm1(0.2 - critical * 0.05))
    assert projected.require_lift().ub == pytest.approx(math.expm1(0.2 + critical * 0.05))


def test_student_t_absolute_margin_p_value_uses_retained_degrees_of_freedom():
    row = _t_row(alternative="greater").model_copy(update={"null_abs": 0.0})

    assert row.p_value() == pytest.approx(float(t.cdf(-4.0, 10)))


@pytest.mark.parametrize("alpha", [1e-100, 1e-200])
def test_log_t_reintervalization_withholds_only_the_unrepresentable_interval(
    alpha: float,
) -> None:
    from increment.estimation.results import LiftEstimate
    from tests.test_cuped_linear_acceptance import _linear_breakout_analysis

    (row,) = cast(tuple[LiftEstimate, ...], tuple(_linear_breakout_analysis().run()))

    projected = row.reintervalize(alpha)

    assert projected.failure_code is None
    assert projected.relative_unavailable_reason is None
    assert projected.relative_error_code == "estimation.tails.unresolvable"
    lift = projected.require_lift()
    assert lift.value == pytest.approx(row.require_lift().value)
    assert lift.lb is None and lift.ub is None and lift.level is None
    assert lift.log_mean == row.require_lift().log_mean
    assert lift.log_se == row.require_lift().log_se
    assert projected.p_value() == pytest.approx(row.p_value())
    assert projected.stat_sig() is False
    assert projected.abs_diff == row.abs_diff
    assert projected.abs_lb is not None and row.abs_lb is not None
    assert projected.abs_lb < row.abs_lb
    assert projected.abs_alpha == pytest.approx(alpha)


@pytest.mark.parametrize("alpha", [1e-100, 1e-200])
def test_log_t_reintervalization_keeps_sampling_evidence_without_additive(alpha: float) -> None:
    row = _t_row().model_copy(
        update={
            "abs_diff": None,
            "abs_se": None,
            "abs_lb": None,
            "abs_ub": None,
            "abs_reference_kind": None,
            "abs_reference_df": None,
            "abs_alpha": None,
        }
    )

    projected = row.reintervalize(alpha)

    assert projected.relative_error_code == "estimation.tails.unresolvable"
    assert projected.require_lift().lb is None
    assert projected.p_value() == pytest.approx(row.p_value())


def test_independent_mean_t_reintervalization_updates_reference_for_json_replay() -> None:
    row = _independent_mean_row()

    projected = row.reintervalize(0.01, interval_alternative="less")

    assert projected.independent_mean_reference is not None
    assert projected.independent_mean_reference.alpha == 0.01
    assert projected.independent_mean_reference.alternative == "less"
    assert projected.lift is not None
    assert projected.lift.alpha == pytest.approx(0.02)
    assert LiftEstimate.model_validate_json(projected.model_dump_json()) == projected


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_opened_independent_mean_reintervalization_rebuilds_central_state(alternative):
    from increment.estimation.results import open_bound_from_two_sided_at_target

    row = _independent_mean_row().reintervalize(0.025, interval_alternative="greater")
    opened = open_bound_from_two_sided_at_target(row)
    assert opened.require_lift().open_side == "upper"
    projected = opened.reintervalize(0.01, interval_alternative=alternative)
    assert projected.require_lift().open_side is None
    assert projected.require_lift().lb is not None
    assert projected.require_lift().ub is not None
    assert projected.independent_mean_reference is not None
    assert projected.independent_mean_reference.interval == "central"
    restored = LiftEstimate.model_validate_json(projected.model_dump_json())
    assert restored == projected
    assert restored.p_value() == pytest.approx(projected.p_value())
