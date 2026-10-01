"""A conversion metric with an in-experiment CUPED covariate on the automatic asymptotic route."""

from __future__ import annotations

from fractions import Fraction as F

import pyarrow as pa
import pytest

from increment import Analysis, InferenceSpec, Method, MetricSpec
from increment.semantics.models import AnalysisPlan
from tests.binary_sequential_cases import DESIGN, frame_analysis, unit_rows

SPEC = InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=400)
TRUE_LIFT = 0.42 / 0.30 - 1


def _joint_moments(units, variant):
    """Exact ``(n, mean, centered scatter)`` of the per-unit (Y, X) rows of one arm."""
    rows = [u for u in units if u["variant"] == variant]
    n = len(rows)
    y = sum(u["purchase"] for u in rows)
    x = sum(u["pre_purchase"] for u in rows)
    yx = sum(u["purchase"] * u["pre_purchase"] for u in rows)
    cross = F(yx) - F(y * x, n)
    return n, (F(y, n), F(x, n)), ((F(y) - F(y * y, n), cross), (cross, F(x) - F(x * x, n)))


def _fixed_horizon_cuped(units):
    return Analysis.from_unit_summary(
        pa.Table.from_pylist(units),
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(
                name="purchase",
                type="conversion",
                covariate="pre_purchase",
                decision_method=Method(name="cuped", variance_reduction="cuped"),
            )
        ],
        design=DESIGN,
        plan=AnalysisPlan(primary="purchase"),
        experiment_id="exp",
    )


def test_conversion_with_a_covariate_registers_adjusted_mean_over_the_joint_bernoulli_moments():
    units = unit_rows(n=400, cuped=True)
    analysis = frame_analysis(units, SPEC, cuped=True)
    snapshot = analysis.capture_sequential(finalized=True)
    assert snapshot.registration.models[0].law == "adjusted_mean"
    for variant in ("control", "treatment"):
        arm = snapshot.arm("purchase", variant)
        n, mean, scatter = _joint_moments(units, variant)
        assert (arm.n, arm.mean, arm.scatter) == (n, mean, scatter)
    row = analysis.run()[0]
    assert row.method == "cuped" and row.inference == "asymptotic_mean"
    # The coefficient read from the retained cross moments at this look is the
    # one the fixed-horizon CUPED estimator fits, so both report the same point.
    fixed = _fixed_horizon_cuped(units).run()[0]
    assert fixed.method == "cuped"
    assert row.require_lift().value == pytest.approx(fixed.require_lift().value, rel=1e-12)


def test_conversion_cuped_interval_covers_the_true_lift_and_narrows_on_the_covariate():
    units = unit_rows(n=400, cuped=True)
    adjusted = frame_analysis(units, SPEC, cuped=True).run()[0]
    plain = frame_analysis(
        [{k: v for k, v in u.items() if k != "pre_purchase"} for u in units], SPEC
    )
    unadjusted = plain.run()[0]
    assert adjusted.inference == "asymptotic_mean"
    assert unadjusted.inference == "asymptotic_mean"
    lift = adjusted.require_lift()
    assert adjusted.require_asymptotic_sequential_result().bounds.status == "bounded"
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < lift.value < lift.ub
    assert lift.lb <= TRUE_LIFT <= lift.ub
    # pre_purchase predicts purchase (0.6 against 0.2), so the joint set is
    # visibly narrower than the scalar one on the same outcomes.
    reference = unadjusted.require_lift()
    assert reference.lb is not None and reference.ub is not None
    assert lift.ub - lift.lb < 0.9 * (reference.ub - reference.lb)
