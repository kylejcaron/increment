"""Call-time lift option overrides as they reach run_daily_lift's rows."""

import pytest

from increment import Analysis, AnalysisPlan
from increment.errors import InvalidRequestError
from increment.estimation.engine import Method
from increment.estimation.priors import StudentTPrior


def _panel_analysis(**overrides):
    import polars as pl

    rows = [
        {
            "unit": f"u{i}",
            "group": "control" if i % 2 == 0 else "treat",
            "ds": "2025-01-01",
            "y": float(i % 3) + 1.0 + (0.5 if i % 2 else 0.0),
        }
        for i in range(12)
    ]
    return Analysis.from_unit_panel(
        pl.DataFrame(rows),
        unit="unit",
        group="group",
        date="ds",
        control="control",
        metrics={"y": "mean"},
        **overrides,
    )


def test_defaults_resolve_the_single_declared_metric_config():
    rows = _panel_analysis().run_daily_lift()
    assert {row.metric for row in rows} == {"y"}
    assert {row.method for row in rows} == {"unadjusted"}
    assert {row.method_role for row in rows} == {"decision"}
    assert all(row.lift is not None and row.lift.alpha == 0.05 for row in rows)


def test_call_time_prior_is_applied_to_the_estimate():
    (default,) = _panel_analysis().run_daily_lift()
    (shrunk,) = _panel_analysis().run_daily_lift(prior=StudentTPrior(nu=4.0, scale=0.05))
    assert default.lift is not None and shrunk.lift is not None
    assert default.lift.value == shrunk.lift.value
    # Call-time priors update the separate posterior, not prior-free `lift`
    # (docs/guides/priors-and-decisions.md:7-9).
    assert default.posterior_available is None
    assert shrunk.posterior_available is True
    posterior_estimate = shrunk.posterior_estimate
    assert posterior_estimate is not None
    assert 0.0 < abs(posterior_estimate) < abs(default.lift.value)


def test_explicit_methods_stamp_decision_and_sensitivity_roles():
    rows = _panel_analysis().run_daily_lift(
        decision_method=Method(name="dec"), sensitivity_methods=[Method(name="sens")]
    )
    assert {(row.method, row.method_role) for row in rows} == {
        ("dec", "decision"),
        ("sens", "sensitivity"),
    }


def test_decision_method_override_wins_over_the_unadjusted_default():
    """decision_method= must force the named method into the decision role
    even though 'unadjusted', also present via sensitivity_methods, is what
    the no-override default would pick."""
    rows = _panel_analysis().run_daily_lift(
        decision_method=Method(name="dec"), sensitivity_methods=[Method(name="unadjusted")]
    )
    assert {(row.method, row.method_role) for row in rows} == {
        ("dec", "decision"),
        ("unadjusted", "sensitivity"),
    }


def test_decision_method_alone_is_the_only_method():
    rows = _panel_analysis().run_daily_lift(decision_method=Method(name="dec"))
    assert {(row.method, row.method_role) for row in rows} == {("dec", "decision")}


def test_plan_alpha_reaches_the_interval():
    (row,) = _panel_analysis(plan=AnalysisPlan(alpha=0.2)).run_daily_lift()
    assert row.lift is not None
    assert row.lift.alpha == 0.2
    assert row.lift.level == pytest.approx(0.8)


def test_cuped_sensitivity_method_needs_the_pre_period_covariate():
    with pytest.raises(InvalidRequestError) as raised:
        _panel_analysis().run_daily_lift(
            sensitivity_methods=[Method(name="cuped", variance_reduction="cuped")]
        )
    assert raised.value.code == "estimation.cuped.arm_no_covariate"
