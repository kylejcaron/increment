"""Behavioral contracts for callable experiment-health diagnostics."""

from collections.abc import Mapping, MutableMapping
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


def test_health_preserves_assigned_failure_beside_triggered_balance(tmp_path) -> None:
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


def test_health_unsupported_assignment_law_is_not_a_unit_arm_pass(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(
        tmp_path, _events(n_per_arm=20, trigger_rate=0.5), allocation_scheme="blocked"
    )
    try:
        assigned = next(
            check
            for check in analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5}).checks
            if check.check == "assigned_allocation"
        )
    finally:
        analysis.close()
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.allocation_scheme_unsupported"
    assert assigned.conclusion is None
    assert assigned.srm is None


def _health_snapshot_scenario(tmp_path, monkeypatch) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    original = _events(n_per_arm=100, trigger_rate=0.5)
    con.create_table("events", obj=original)
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    analysis = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    source = analysis._src
    pin = source._pinned_source
    inserted = False

    def pin_then_append(**kwargs):
        nonlocal inserted
        pinned = pin(**kwargs)
        if not inserted:
            inserted = True
            con.insert(
                "events",
                obj=pa.Table.from_pylist(
                    [
                        {
                            "user_id": f"{arm}new",
                            "group_id": arm,
                            "ts": dt.datetime(2024, 1, 2),
                            "event": event,
                            "value": 1.0,
                            "experiment_id": "exp",
                        }
                        for arm in ("C", "T")
                        for event in ("enrolled", "saw_surface", "revenue")
                    ]
                ),
            )
        return pinned

    monkeypatch.setattr(source, "_pinned_source", pin_then_append)
    try:
        first = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
        second = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    first_assigned = next(check for check in first.checks if check.check == "assigned_allocation")
    second_assigned = next(check for check in second.checks if check.check == "assigned_allocation")
    first_rates = next(check for check in first.checks if check.check == "trigger_rates")
    second_rates = next(check for check in second.checks if check.check == "trigger_rates")
    first_srm, second_srm = first_assigned.srm, second_assigned.srm
    first_rate_values, second_rate_values = first_rates.rates, second_rates.rates
    assert first_srm is not None and second_srm is not None
    assert first_rate_values is not None and second_rate_values is not None
    assert first_srm.observed["C"] == 100
    assert second_srm.observed["C"] == 101
    assert first_rate_values["C"] == pytest.approx(
        sum(
            row["event"] == "saw_surface" and row["group_id"] == "C" for row in original.to_pylist()
        )
        / 100
    )
    assert second_rate_values["C"] == pytest.approx((first_rate_values["C"] * 100 + 1) / 101)
    assert first.source_snapshot_id and second.source_snapshot_id
    assert first.source_snapshot_id != second.source_snapshot_id
    assert first.source_cutoff == second.source_cutoff == cutoff.isoformat()
    restored = ExperimentHealth.model_validate_json(first.model_dump_json())
    assert restored.source_snapshot_id == first.source_snapshot_id
    assert restored.source_cutoff == first.source_cutoff


def test_health_checks_share_one_snapshot(tmp_path, monkeypatch) -> None:
    _health_snapshot_scenario(tmp_path, monkeypatch)


def test_health_second_call_sees_new_rows(tmp_path, monkeypatch) -> None:
    _health_snapshot_scenario(tmp_path, monkeypatch)


def test_health_artifact_trigger_extensions_are_optional_but_explicit(tmp_path) -> None:
    """Matched native/artifact diagnostics; portable shapes do not retain trigger evidence."""
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=30, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    native = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    native_report = native.health_check()
    artifact = _publish_and_adopt(
        con,
        native,
        kinds=("trigger_population", "assignment_counts", "trigger_measure_stats"),
    )
    try:
        report = artifact.health_check()
    finally:
        _close_parity_analysis(artifact)
    native_by_check = {check.check: check for check in native_report.checks}
    artifact_by_check = {check.check: check for check in report.checks}
    for name in (
        "assigned_allocation",
        "triggered_balance",
        "trigger_rates",
        "trigger_assumptions",
    ):
        assert artifact_by_check[name].model_dump(mode="json") == native_by_check[name].model_dump(
            mode="json"
        )
    native_maturity = {
        check.metric: check.model_dump(mode="json")
        for check in native_report.checks
        if check.check == "trigger_maturity"
    }
    artifact_maturity = {
        check.metric: check.model_dump(mode="json")
        for check in report.checks
        if check.check == "trigger_maturity"
    }
    assert artifact_maturity == native_maturity
    native_completeness = {
        (
            check.availability,
            check.conclusion,
            check.code,
            check.context.get("watermark"),
            check.context.get("reason"),
        )
        for check in native_report.checks
        if check.check == "feed_completeness"
    }
    artifact_completeness = [check for check in report.checks if check.check == "feed_completeness"]
    assert {
        (
            check.availability,
            check.conclusion,
            check.code,
            check.context.get("watermark"),
            check.context.get("reason"),
        )
        for check in artifact_completeness
    } == native_completeness
    assert any(check.feed == "trigger_population" for check in artifact_completeness)
    assert any(check.feed == "trigger_measure_stats:revenue" for check in artifact_completeness)


def test_health_models_round_trip_pickle_deepcopy_and_nested_immutability() -> None:
    import copy
    import pickle

    unavailable = HealthCheck(
        check="triggered_balance",
        analysis_population="triggered",
        availability="unavailable",
        code="source.native.trigger_evidence_required",
        context={"reason": "no trigger counts", "details": {"feeds": ["events"]}},
    )
    rates = HealthCheck(
        check="trigger_rates",
        analysis_population="triggered",
        availability="available",
        conclusion="descriptive",
        context={"description": "observed trigger shares"},
        rates={"C": 0.25, "T": 0.25},
    )
    report = ExperimentHealth(checks=(unavailable, rates))
    for restored in (
        pickle.loads(pickle.dumps(report)),
        copy.deepcopy(report),
        ExperimentHealth.model_validate_json(report.model_dump_json()),
    ):
        assert restored == report
        details = cast(Mapping[str, object], restored.checks[0].context["details"])
        assert cast(tuple[str, ...], details["feeds"]) == ("events",)
        assert restored.checks[1].rates == {"C": 0.25, "T": 0.25}
        with pytest.raises((TypeError, AttributeError)):
            cast(MutableMapping[str, object], restored.checks[0].context["details"]).update(
                {"feeds": ()}
            )

        with pytest.raises((TypeError, AttributeError)):
            cast(MutableMapping[str, float], restored.checks[1].rates).update({"C": 0.5})


def test_health_artifact_without_trigger_extensions_is_unavailable(tmp_path) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    native = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    artifact = _publish_and_adopt(con, native, kinds=("assignment_counts",))
    try:
        report = artifact.health_check()
    finally:
        _close_parity_analysis(artifact)
    for name in ("triggered_balance", "trigger_rates"):
        check = next(row for row in report.checks if row.check == name)
        assert check.availability == "unavailable"
        assert check.conclusion is None
        assert check.code is not None


def test_health_unit_panel_reports_only_supported_population() -> None:
    panel = Analysis.from_unit_panel(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "day": ["2025-01-01"] * 4,
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        date="day",
        metrics={"value": "mean"},
        control="C",
    )
    report = panel.health_check()
    assert report.checks[0].availability == "unavailable"
    assert report.checks[0].code == "integrity.allocation_scheme_missing"


def test_health_switchback_does_not_substitute_unit_arm_srm() -> None:
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    switchback = Analysis.from_switchback_panel(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "cycle": cycle,
                    "period": period,
                    "step": step,
                    "group": group,
                    "value": float(cycle + period + step),
                }
                for unit, order in (("u1", ("C", "T")), ("u2", ("T", "C")))
                for cycle in range(2)
                for period, group in enumerate(order)
                for step in range(2)
            ]
        ),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(control_group="C", allocation={"C": 0.5, "T": 0.5}),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    assigned = switchback.health_check().checks[0]
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.switchback_assignment_law"
    assert assigned.srm is None


def test_health_moments_replay_does_not_reconstruct_trigger_population() -> None:
    from tests.test_analysis_moments import _legacy_fixed10_rows

    replay = Analysis.from_moments(
        _legacy_fixed10_rows(trigger_name="saw_surface"),
        metrics={"revenue": "mean"},
        control="control",
    )
    report = replay.health_check()
    triggered_balance = next(check for check in report.checks if check.check == "triggered_balance")
    trigger_rates = next(check for check in report.checks if check.check == "trigger_rates")
    completeness = next(check for check in report.checks if check.check == "feed_completeness")
    assert triggered_balance.availability == "unavailable"
    assert triggered_balance.code is not None
    assert trigger_rates.availability == "unavailable"
    assert trigger_rates.code is not None
    assert completeness.availability == "unavailable"
    assert completeness.context["watermark"] is None


def test_health_unavailable_is_not_nonrejection() -> None:
    check = HealthCheck(
        check="triggered_balance",
        analysis_population="triggered",
        availability="unavailable",
        code="source.native.trigger_evidence_required",
        context={"reason": "trigger counts were not retained"},
    )
    assert check.conclusion is None
    with pytest.raises(ValidationError, match="cannot have a conclusion"):
        HealthCheck(
            check="triggered_balance",
            analysis_population="triggered",
            availability="unavailable",
            conclusion="not_rejected",
            code="source.native.trigger_evidence_required",
            context={"reason": "trigger counts were not retained"},
        )


def test_health_expected_allocation_completes_independent_design_admission() -> None:
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
        design=Randomized(control_group="C", allocation_scheme="independent"),
    )
    assigned = next(
        check
        for check in analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5}).checks
        if check.check == "assigned_allocation"
    )
    assert assigned.availability == "available"
    assert assigned.srm is not None
    assert assigned.srm.expected == {"C": 0.5, "T": 0.5}


def test_health_extreme_valid_srm_statistic_round_trips_through_json() -> None:
    import math

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
        design=Randomized(control_group="C", allocation_scheme="independent"),
    )
    report = analysis.health_check(expected_allocation={"C": 5e-324, "T": 1.0})
    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    assert assigned.srm is not None
    assert math.isinf(assigned.srm.chi2_stat)
    assert report.source_snapshot_id is not None

    restored = ExperimentHealth.model_validate_json(report.model_dump_json())
    assert restored.checks == report.checks
    assert restored.source_snapshot_id == report.source_snapshot_id


def test_health_empty_trigger_population_is_unavailable_not_a_report_failure(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(
        tmp_path, _events(n_per_arm=20, trigger_rate=0.0), allocation_scheme="independent"
    )
    try:
        report = analysis.health_check()
    finally:
        analysis.close()
    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    triggered = next(check for check in report.checks if check.check == "triggered_balance")
    rates = next(check for check in report.checks if check.check == "trigger_rates")
    assert assigned.availability == "available"
    assert triggered.availability == "unavailable"
    assert triggered.code == "estimation.diagnostics.total_observed_count"
    assert triggered.conclusion is None
    assert rates.availability == "available"
    assert rates.rates == {"C": 0.0, "T": 0.0}


def test_health_validates_expected_allocation_before_empty_trigger_counts(tmp_path) -> None:
    from increment.errors import InvalidRequestError
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(
        tmp_path, _events(n_per_arm=20, trigger_rate=0.0), allocation_scheme="blocked"
    )
    try:
        with pytest.raises(InvalidRequestError) as caught:
            analysis.health_check(expected_allocation={"C": 0.0, "T": 1.0})
    finally:
        analysis.close()
    assert caught.value.code == "estimation.diagnostics.every_expected_share_positive"


def test_health_srm_report_supports_pickle_and_deepcopy() -> None:
    import copy
    import pickle

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
    assigned = report.checks[0]
    assert assigned.srm is not None
    assert assigned.availability == "available"
    for restored in (pickle.loads(pickle.dumps(report)), copy.deepcopy(report)):
        restored_srm = restored.checks[0].srm
        assert restored_srm is not None
        assert restored_srm == assigned.srm
        assert restored_srm.observed == assigned.srm.observed


@pytest.mark.parametrize(
    ("look_policy", "guarantee"),
    [("exploratory", None), ("outcome_independent", "asymptotic_sequential")],
)
def test_health_assumption_discloses_triggered_family_guarantee(
    tmp_path, look_policy, guarantee
) -> None:
    from tests.test_triggered_sequential import _FAMILY_PLAN, _analysis, _events

    plan = _FAMILY_PLAN
    if look_policy == "outcome_independent":
        plan += "        triggered_look_policy: outcome_independent\n"
    secondaries = ("aux_a", "aux_b")
    analysis, _ = _analysis(
        tmp_path,
        _events(n_per_arm=20, secondaries=secondaries),
        plan=plan,
        secondaries=secondaries,
    )
    try:
        assumptions = next(
            check
            for check in analysis.health_check().checks
            if check.check == "trigger_assumptions"
        )
    finally:
        analysis.close()
    assert assumptions.context["triggered_look_policy"] == look_policy
    assert assumptions.context["family_guarantee"] == guarantee
    if guarantee is not None:
        assert assumptions.context["family_guarantee_status"] == "assumed, not verified"


def test_health_missing_assignment_scheme_suggests_supported_routes() -> None:
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
        design=Randomized(control_group="C"),
    )
    assigned = next(
        check
        for check in analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5}).checks
        if check.check == "assigned_allocation"
    )
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.allocation_scheme_missing"
    route = str(assigned.context["route"])
    assert "allocation_scheme='independent'" in route
    assert "Analysis.srm(inference='fixed')" in route
