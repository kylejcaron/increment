"""Behavioral contracts for callable experiment-health diagnostics."""

from collections.abc import MutableMapping
from typing import cast

import pyarrow as pa
import pytest
from pydantic import ValidationError

from increment import Analysis
from increment.results import ExperimentHealth, HealthCheck
from increment.semantics.design import Randomized


def test_health_models_are_frozen_and_own_nested_context() -> None:
    source = {"risk": ["counterfactual trigger invariance is unknown"]}
    check = HealthCheck(
        check="trigger_assumptions",
        analysis_population="triggered",
        availability="available",
        conclusion="descriptive",
        context=source,
    )
    source["risk"].append("changed after construction")
    report = ExperimentHealth(checks=(check,))
    assert report.checks[0].context["risk"] == ("counterfactual trigger invariance is unknown",)
    with pytest.raises((TypeError, AttributeError)):
        cast(MutableMapping[str, object], report.checks[0].context).update({"risk": ()})
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        type(report).__setattr__(report, "checks", ())


def test_health_json_replay_preserves_checks_and_risks() -> None:
    report = ExperimentHealth(
        checks=(
            HealthCheck(
                check="trigger_assumptions",
                analysis_population="triggered",
                availability="available",
                conclusion="descriptive",
                context={
                    "triggered_look_policy": "exploratory",
                    "family_guarantee": None,
                    "risk": ("balanced rates do not establish trigger invariance",),
                },
            ),
        )
    )
    restored = ExperimentHealth.model_validate_json(report.model_dump_json())
    assert restored.checks == report.checks
    assert "passed" not in restored.model_dump()
    assert "healthy" not in restored.model_dump()
    assert "causal_validity" not in restored.model_dump()


def test_untriggered_summary_has_explicit_unavailable_triggered_rows() -> None:
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Randomized(
            control_group="C",
            allocation={"C": 0.5, "T": 0.5},
            allocation_scheme="independent",
        ),
    )
    report = analysis.health_check()
    assert [check.check for check in report.checks] == [
        "assigned_allocation",
        "triggered_balance",
        "trigger_rates",
        "trigger_maturity",
        "feed_completeness",
        "trigger_assumptions",
    ]
    assert report.checks[0].availability == "available"
    assert all(check.availability == "not_applicable" for check in report.checks[1:])
    assert all(check.conclusion is None for check in report.checks[1:])


def test_balanced_trigger_rates_do_not_clear_assumption_risk(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(tmp_path, _events(n_per_arm=100, trigger_rate=0.5))
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    rates = next(check for check in report.checks if check.check == "trigger_rates")
    assumptions = next(check for check in report.checks if check.check == "trigger_assumptions")
    assert rates.availability == "available"
    assert rates.rates["C"] == pytest.approx(rates.rates["T"])
    assert assumptions.context["counterfactual_trigger_invariance"].startswith("assumed")
    assert assumptions.context["trigger_order_stationarity"].startswith("assumed")
    assert assumptions.context["family_guarantee"] is None


def test_health_missing_certificates_are_unknown(tmp_path) -> None:
    import ibis

    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    analysis = Analysis.from_definitions("exp", _write_defs(tmp_path), con)
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    triggered = next(check for check in report.checks if check.check == "triggered_balance")
    assert triggered.availability == "unavailable"
    assert triggered.conclusion is None
    assert triggered.code == "source.native.trigger_evidence_required"
    completeness = [check for check in report.checks if check.check == "feed_completeness"]
    assert completeness
    assert all(check.availability == "unavailable" for check in completeness)
    assert all(check.conclusion is None for check in completeness)
    assert all(check.context["watermark"] is None for check in completeness)


def test_assigned_failure_remains_visible_beside_triggered_balance(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    rows = _events(n_per_arm=400, trigger_rate=0.5).to_pylist()
    triggered_c = {
        row["user_id"] for row in rows if row["event"] == "saw_surface" and row["group_id"] == "C"
    }
    remove = set(sorted(f"C{i}" for i in range(400) if f"C{i}" not in triggered_c)[:170])
    uneven = pa.Table.from_pylist([row for row in rows if row["user_id"] not in remove])
    analysis = _analysis(tmp_path, uneven, allocation_scheme="independent")
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    triggered = next(check for check in report.checks if check.check == "triggered_balance")
    assert assigned.availability == "available"
    assert assigned.conclusion == "warning"
    assert assigned.srm is not None and assigned.srm.is_srm
    assert triggered.availability == "available"
    assert triggered.srm is not None and not triggered.srm.is_srm
