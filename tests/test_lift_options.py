"""Call-time lift option overrides as they reach run_daily_lift's rows."""

import warnings
from typing import Any

import pytest

from increment import AnalysisPlan, MetricSpec, Scope, from_frame
from increment.errors import IncrementWarning, InvalidRequestError
from increment.estimation.engine import Method
from increment.estimation.priors import StudentTPrior
from increment.estimators import Normal
from tests.analysis_factory import view_readout


def _daily_lift(pair: tuple[Any, Any], **options: Any) -> Any:
    experiment, resolver = pair
    return view_readout(experiment, resolver, Scope(view="daily", grain="daily"), **options)


def _panel_analysis(**overrides):
    import polars as pl

    rows = [
        {
            "unit": f"u{i}",
            "group": "control" if i % 2 == 0 else "treat",
            "ds": "2025-01-01",
            "y": float(i % 3) + 1.0 + (0.5 if i % 2 else 0.0),
            "z": float(i % 4) + 2.0 + (0.5 if i % 2 else 0.0),
        }
        for i in range(12)
    ]
    return from_frame(
        pl.DataFrame(rows),
        shape="panel",
        unit="unit",
        group="group",
        date="ds",
        control="control",
        metrics=overrides.pop("metric_specs", {"y": "mean"}),
        **overrides,
    )


def test_defaults_resolve_the_single_declared_metric_config():
    rows = _daily_lift(_panel_analysis())
    assert {row.metric for row in rows} == {"y"}
    assert {row.method for row in rows} == {"unadjusted"}
    assert {row.method_role for row in rows} == {"decision"}
    assert all(row.lift is not None and row.lift.alpha == 0.05 for row in rows)


def test_call_time_prior_is_applied_to_the_estimate():
    (default,) = _daily_lift(_panel_analysis())
    (shrunk,) = _daily_lift(
        _panel_analysis(),
        prior=StudentTPrior(nu=4.0, scale=0.05),
    )
    assert default.lift is not None and shrunk.lift is not None
    assert default.lift.value == shrunk.lift.value
    # Call-time priors update the separate posterior, not prior-free `lift`
    # (docs/guides/priors-and-decisions.md:3).
    assert default.posterior_available is None
    assert shrunk.posterior_available is True
    posterior_estimate = shrunk.posterior_estimate
    assert posterior_estimate is not None
    assert 0.0 < abs(posterior_estimate) < abs(default.lift.value)


def test_explicit_methods_stamp_decision_and_sensitivity_roles():
    rows = _daily_lift(
        _panel_analysis(),
        decision_method=Method(name="dec"),
        sensitivity_methods=[Method(name="sens")],
    )
    assert {(row.method, row.method_role) for row in rows} == {
        ("dec", "decision"),
        ("sens", "sensitivity"),
    }


def test_run_warns_once_per_setting_when_override_replaces_declaration():
    declared_decision = Method(name="declared")
    declared_sensitivity = Method(name="declared-sensitivity")
    declared_prior = Normal(mu=0.0, sigma=0.1)
    metrics = [
        MetricSpec(
            name=name,
            type="mean",
            decision_method=declared_decision,
            sensitivity_methods=(declared_sensitivity,),
            prior=declared_prior,
        )
        for name in ("y", "z")
    ]
    experiment, resolver = _panel_analysis(metric_specs=metrics)

    with pytest.warns(IncrementWarning) as caught:
        rows = view_readout(
            experiment,
            resolver,
            decision_method=Method(name="override"),
            sensitivity_methods=[Method(name="override-sensitivity")],
            prior=StudentTPrior(nu=4.0, scale=0.05),
        )

    warnings = [record.message for record in caught if isinstance(record.message, IncrementWarning)]
    assert [warning.code for warning in warnings] == [
        "analysis.run.override_replaces_declared",
        "analysis.run.override_replaces_declared",
        "analysis.run.override_replaces_declared",
    ]
    assert {warning.context["setting"] for warning in warnings} == {
        "decision_method",
        "sensitivity_methods",
        "prior",
    }
    assert all(warning.context["metrics"] == ("y", "z") for warning in warnings)
    by_setting = {warning.context["setting"]: warning for warning in warnings}
    assert by_setting["decision_method"].context["declared"] == (
        ("y", declared_decision),
        ("z", declared_decision),
    )
    assert by_setting["decision_method"].context["override"] == Method(name="override")
    assert by_setting["sensitivity_methods"].context["override"] == (
        Method(name="override-sensitivity"),
    )
    assert by_setting["prior"].context["declared"] == (
        ("y", declared_prior),
        ("z", declared_prior),
    )
    assert all(row.prior_spec == StudentTPrior(nu=4.0, scale=0.05) for row in rows)
    assert {(row.metric, row.method, row.method_role) for row in rows} == {
        ("y", "override", "decision"),
        ("y", "override-sensitivity", "sensitivity"),
        ("z", "override", "decision"),
        ("z", "override-sensitivity", "sensitivity"),
    }


def test_run_does_not_warn_when_override_matches_or_nothing_declared():
    declared = Method(name="declared")
    experiment, resolver = _panel_analysis(
        metric_specs=[
            MetricSpec(name="y", type="mean", decision_method=declared),
            MetricSpec(name="z", type="mean"),
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IncrementWarning)
        rows = view_readout(experiment, resolver, decision_method=declared)
    overrides = [
        record.message
        for record in caught
        if getattr(record.message, "code", None) == "analysis.run.override_replaces_declared"
    ]
    assert overrides == []
    assert {row.method for row in rows} == {"declared"}


def test_explicit_prior_none_override_is_reported():
    experiment, resolver = _panel_analysis(
        metric_specs=[MetricSpec(name="y", type="mean", prior=Normal(mu=0.0, sigma=0.1))]
    )

    with pytest.warns(IncrementWarning) as caught:
        rows = view_readout(experiment, resolver, prior=None)
    overrides = [
        record.message
        for record in caught
        if isinstance(record.message, IncrementWarning)
        and record.message.code == "analysis.run.override_replaces_declared"
    ]
    assert len(overrides) == 1
    assert overrides[0].context["setting"] == "prior"
    assert overrides[0].context["metrics"] == ("y",)
    assert all(row.posterior_available is None for row in rows)


def test_decision_method_override_wins_over_the_unadjusted_default():
    """decision_method= must force the named method into the decision role
    even though 'unadjusted', also present via sensitivity_methods, is what
    the no-override default would pick."""
    rows = _daily_lift(
        _panel_analysis(),
        decision_method=Method(name="dec"),
        sensitivity_methods=[Method(name="unadjusted")],
    )
    assert {(row.method, row.method_role) for row in rows} == {
        ("dec", "decision"),
        ("unadjusted", "sensitivity"),
    }


def test_decision_method_alone_is_the_only_method():
    rows = _daily_lift(_panel_analysis(), decision_method=Method(name="dec"))
    assert {(row.method, row.method_role) for row in rows} == {("dec", "decision")}


def test_plan_alpha_reaches_the_interval():
    (row,) = _daily_lift(_panel_analysis(plan=AnalysisPlan(alpha=0.2)))
    assert row.lift is not None
    assert row.lift.alpha == 0.2
    assert row.lift.level == pytest.approx(0.8)


def test_cuped_sensitivity_method_needs_the_pre_period_covariate():
    with pytest.raises(InvalidRequestError) as raised:
        _daily_lift(
            _panel_analysis(),
            sensitivity_methods=[Method(name="cuped", variance_reduction="cuped")],
        )
    assert raised.value.code == "estimation.cuped.arm_no_covariate"
