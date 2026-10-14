"""Public readout journeys retain evidence for interpretation and replay."""

import json
import warnings
from typing import Any, cast

import pandas as pd
import pytest

from increment import Analysis, AnalysisPlan, MetricSpec
from increment._interpretation import (
    interpret_readout,
    readout_evidence_columns,
    render_readout_text,
)
from increment.errors import WireFormatError
from increment.estimation.readout_types import CellKey, ReadoutResults
from increment.semantics.design import Randomized


def test_low_reliability_interval_is_qualified_in_shared_interpretation():
    from types import SimpleNamespace

    from increment._interpretation import render_effect_interpretation
    from increment.estimation.results import Estimate

    row = SimpleNamespace(
        value_scale="relative",
        null_lift=0.0,
        null_abs=None,
        alternative="greater",
        lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
        low_reliability=True,
        stat_sig=False,
    )

    interpretation = render_effect_interpretation(row)

    assert interpretation is not None
    assert "low reliability" in interpretation.lower()
    assert "increase" not in interpretation.lower()


def test_explicit_non_significance_suppresses_excluding_interval_claim():
    from types import SimpleNamespace

    from increment._interpretation import render_effect_interpretation
    from increment.estimation.results import Estimate

    row = SimpleNamespace(
        value_scale="relative",
        null_lift=0.0,
        null_abs=None,
        alternative="greater",
        lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
        stat_sig=False,
    )

    interpretation = render_effect_interpretation(row)

    assert interpretation is not None
    assert "does not reject" in interpretation.lower()
    assert "increase" not in interpretation.lower()


def _mean_frame(*, n_control=100, n_treatment=100, constant_guard=False):
    rows = []
    for group, count, shift in (
        ("control", n_control, 0.0),
        ("treatment", n_treatment, 0.2),
    ):
        for i in range(count):
            rows.append(
                {
                    "unit": f"{group}-{i}",
                    "arm": group,
                    "value": float(i % 11 + shift),
                    "guard": 1.0 if constant_guard else float(i % 7 + 1),
                    "cluster": f"{group}-{i % 3}",
                }
            )
    return pd.DataFrame(rows)


def _run_mean(
    frame,
    *,
    allocation=True,
    guardrail=False,
    cluster=None,
    control="control",
    sensitivity_methods=None,
):
    if control != "control":
        frame = frame.copy()
        frame["arm"] = frame["arm"].replace({"control": control})
    design = Randomized(
        control_group=control,
        allocation={control: 0.5, "treatment": 0.5} if allocation else None,
        allocation_scheme="independent" if allocation else None,
    )
    metrics = [MetricSpec(name="value", type="mean")]
    plan = AnalysisPlan(primary="value", guardrails=["guard"] if guardrail else [])
    if guardrail:
        metrics.append(MetricSpec(name="guard", type="mean", preferred_direction="decrease"))
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        design=design,
        metrics=metrics,
        plan=plan,
        cluster=cluster,
    )
    return (
        analysis.run()
        if sensitivity_methods is None
        else analysis.run(sensitivity_methods=sensitivity_methods)
    )


def _interpret(results):
    return interpret_readout(results, results.metadata)


def test_failed_declared_allocation_is_integrity_first_and_not_laundered():
    results = _run_mean(_mean_frame(n_control=900, n_treatment=100))
    interpretation = _interpret(results)
    assert {group.integrity_status for group in interpretation.groups} == {"failed"}


def test_not_rejected_integrity_is_described_as_no_detected_mismatch_not_proof():
    interpretation = _interpret(_run_mean(_mean_frame()))
    assert {group.integrity_status for group in interpretation.groups} == {"not_rejected"}


def test_missing_allocation_declaration_is_not_success():
    interpretation = _interpret(_run_mean(_mean_frame(), allocation=False))
    assert {group.integrity_status for group in interpretation.groups} == {
        "not_checked_missing_declaration"
    }


def test_advisory_context_is_immutable_after_construction_and_restore():
    import pickle

    from increment.estimation.advisory import ReadoutAdvisory

    original = {"nested": {"values": [1, 2]}}
    advisory = ReadoutAdvisory(code="test.advisory", context=original, message="detail")
    original["nested"]["values"][0] = 99
    assert cast(Any, advisory.context)["nested"]["values"] == (1, 2)
    restored = (
        ReadoutAdvisory.model_validate_json(advisory.model_dump_json()),
        pickle.loads(pickle.dumps(advisory)),
    )
    for value in (advisory, *restored):
        with pytest.raises(TypeError):
            cast(Any, value.context)["nested"]["values"] = (7,)


def test_roster_completeness_is_distinct_from_view_partiality():
    from tests.test_analysis_moments import _legacy_fixed10_rows

    results = Analysis.from_moments(
        _legacy_fixed10_rows(),
        metrics=[MetricSpec(name="revenue", type="mean")],
        control="control",
    ).run()
    source = next(iter(cast(Any, results).metadata.scope.by_source.values()))
    roster = next(item for item in source.rosters if item.analysis_population == "assigned")
    assert source.decision_complete_by_population["assigned"] is False
    assert roster.complete is False
    assert cast(Any, results)[0].sampling_available is True
    interpretation = _interpret(results)
    group = interpretation.groups[0]
    assert group.partial is False
    assert group.decision_complete is False
    assert group.roster_complete is False
    assert group.roster_source == "observed_union"
    columns = readout_evidence_columns(interpretation, results[0])
    assert columns["decision_scope_complete"] is False
    assert columns["roster_complete"] is False
    text = render_readout_text(interpretation)
    assert "observed_union" in text and text.count("incomplete") >= 2


def test_required_guardrail_failure_remains_visible_when_filtered_out():
    with pytest.warns(Warning):
        results = _run_mean(_mean_frame(constant_guard=True), guardrail=True)
    filtered = results.filter(lambda row: row.metric == "value")
    interpretation = _interpret(filtered)
    assert interpretation.partial
    assert any(
        cell.key.metric == "guard" for group in interpretation.groups for cell in group.hidden_cells
    )

    columns = readout_evidence_columns(interpretation, filtered[0])
    assert columns["integrity_status"] == "not_rejected"
    assert columns["multiplicity_status"] == filtered[0].multiplicity_status
    assert isinstance(columns["hidden_cells"], str)
    hidden_cells = json.loads(columns["hidden_cells"])
    hidden = next(item for item in hidden_cells if item["cell"]["metric"] == "guard")
    assert hidden["code"]
    assert hidden["context"]["reason"] == "zero_variance"
    assert hidden["available"] is False


def test_registered_cell_failure_renders_its_spec_explanation():
    from increment.estimation.readout_types import _CELL_FAILURE_REFUSALS

    rows = []
    for group in ("control", "treatment"):
        for i in range(30):
            rows.append(
                {
                    "unit": f"{group}-{i}",
                    "arm": group,
                    "value": float(i),
                    "guard": float(i) if group == "control" else None,
                }
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Analysis.from_unit_summary(
            pd.DataFrame(rows),
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(name="value", type="mean"),
                MetricSpec(
                    name="guard", type="mean", preferred_direction="decrease", missing="drop"
                ),
            ],
            plan=AnalysisPlan(primary="value", guardrails=["guard"]),
        ).run()
    interpretation = _interpret(results)
    guard = next(
        cell
        for group in interpretation.groups
        for cell in group.source_cells
        if cell.key.metric == "guard"
    )
    assert guard.failure_code in _CELL_FAILURE_REFUSALS
    expected = guard.failure_message
    assert expected is not None and not expected.startswith(guard.failure_code)
    assert "guard" in expected and "treatment" in expected
    assert "control" in expected and "'control'" not in expected
    source_id = next(iter(cast(Any, results).metadata.scope.by_source))
    columns = cast(
        Any,
        readout_evidence_columns(interpretation, guard.key, source_snapshot_id=source_id),
    )
    assert columns["failure_code"] == guard.failure_code
    failure_context = json.loads(cast(str, columns["failure_context"]))
    assert failure_context["metric"] == "guard"
    assert failure_context["group_id"] == "treatment"
    assert failure_context["observed_arms"] == ["control"]
    assert columns["failure_message"] == expected
    assert expected in render_readout_text(interpretation)
    # The unavailable guardrail's row lines must name it, not read as the primary's.
    for line in str(results).splitlines():
        if any(part in line for part in ("test alternative", "result identity", "interpretation:")):
            assert line.split(" ", 1)[0] in {"value", "guard"}, line
    assert any(
        line.startswith("guard test alternative: less") for line in str(results).splitlines()
    )


def test_failure_rendering_fills_metric_and_falls_back_without_naming_an_unknown_arm():
    from increment.estimation.readout_types import render_readout_failure

    # Randomized scope records the requested arm and the observed arms: render it.
    complete = {"group_id": "treatment", "observed_arms": ["control"]}
    rendered = render_readout_failure(
        "readout.cell.missing_arm", complete, identity={"metric": "revenue"}
    )
    assert rendered is not None
    assert "revenue" in rendered and "treatment" in rendered and "control" in rendered
    # Day-axis and compliance contexts do not say which arm lacks observations: fall back.
    for context in ({"observed_units": 0}, {"group_id": "treatment", "control_group": "control"}):
        assert (
            render_readout_failure("readout.cell.missing_arm", context, identity={"metric": "m"})
            is None
        )
    # Stored context wins over identity.
    stored = render_readout_failure(
        "readout.cell.missing_arm",
        {"metric": "stored", **complete},
        identity={"metric": "identity"},
    )
    assert stored is not None and "stored" in stored and "identity" not in stored


def test_compliance_date_missing_control_does_not_blame_the_treatment_arm():
    import datetime as dt

    from increment._day_axis import _missing_compliance_rows
    from increment._interpretation import interpret_readout
    from increment._source_types import ComplianceArm, ComplianceSummary

    summary = ComplianceSummary(
        "s", "clicked", 7, True, None, None, (ComplianceArm("treatment", 20, 10),)
    )
    rows = _missing_compliance_rows(
        summary,
        dt.date(2024, 1, 3),
        arm_roster=("control", "treatment"),
        control_group="control",
        population="assigned",
        inference=None,
        reason="missing_arm",
    )
    (row,) = rows
    assert row.unavailable == "no_control_arm"
    (group,) = interpret_readout(rows, None).groups
    (cell,) = group.source_cells
    assert cell.failure_message is not None
    assert "requested arm treatment" not in cell.failure_message
    assert cell.failure_message.startswith("readout.cell.missing_arm")


def test_v1_readout_envelope_refuses_with_recompute_route():
    results = _run_mean(_mean_frame())
    payload = json.loads(results.model_dump_json())
    payload["schema_version"] = 1
    with pytest.raises(WireFormatError) as caught:
        ReadoutResults.model_validate_json(json.dumps(payload))
    assert caught.value.code == "readout.serialization.unsupported_version"
    assert "recompute" in str(caught.value.context["remedy"]).lower()


def test_unscoped_plain_list_reports_unknown_scope_and_integrity():
    results = _run_mean(_mean_frame(), control="holdout")
    interpretation = interpret_readout(list(results), None)
    assert interpretation.groups
    assert {group.integrity_status for group in interpretation.groups} == {"unknown"}
    assert all(group.control_group is None for group in interpretation.groups)


def test_small_cluster_advisory_survives_suppressed_warning_and_json_restore():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _run_mean(_mean_frame(), cluster="cluster")
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    interpretation = _interpret(restored)
    advisories = [
        advisory
        for record in restored.metadata.cells
        for advisory in record.advisories
        if advisory.code == "estimation.engine.small_total_clusters"
    ]
    assert len(advisories) == 1
    assert advisories[0].context["n_clusters"] == 6
    assert restored.metadata == results.metadata
    assert "estimation.engine.small_total_clusters" in {
        advisory.code for group in interpretation.groups for advisory in group.advisories
    }
    from increment.tables import readout_table

    assert "estimation.engine.small_total_clusters" in readout_table(restored).gt().as_raw_html()


def test_ratio_denominator_precision_note_is_an_interpreted_advisory():
    rows = []
    for group in ("control", "treatment"):
        for i in range(20):
            rows.append(
                {
                    "unit": f"{group}-{i}",
                    "arm": group,
                    "revenue": float((i % 13) + 1),
                    "sessions": 0.1 if i % 2 else 20.0,
                }
            )
    results = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions")],
        plan=AnalysisPlan(primary="rps"),
    ).run()
    text = render_readout_text(_interpret(results))
    assert "ratio_denominator_precision" in text
    assert results.metadata is not None
    codes = {advisory.code for record in results.metadata.cells for advisory in record.advisories}
    assert "estimation.engine.ratio_denominator_precision" in codes
    from increment.tables import readout_table

    assert (
        "estimation.engine.ratio_denominator_precision" in readout_table(results).gt().as_raw_html()
    )


def test_ratio_denominator_skew_advisory_survives_saving_with_warnings_suppressed():
    import numpy as np

    rng = np.random.default_rng(20261009)
    rows = [
        {
            "unit": f"{group}-{i}",
            "arm": group,
            "revenue": float(rng.normal(2.0, 1.0)),
            "sessions": float(rng.lognormal(0.0, 1.5)),
        }
        for group in ("control", "treatment")
        for i in range(100)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Analysis.from_unit_summary(
            pd.DataFrame(rows),
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions")
            ],
            plan=AnalysisPlan(primary="rps"),
        ).run()
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata is not None
    codes = {advisory.code for record in restored.metadata.cells for advisory in record.advisories}
    assert "estimation.engine.ratio_denominator_skew" in codes
    column_codes = readout_evidence_columns(_interpret(restored), restored[0])
    assert "estimation.engine.ratio_denominator_skew" in json.dumps(column_codes, default=str)
    from increment.tables import readout_table

    assert "estimation.engine.ratio_denominator_skew" in readout_table(restored).gt().as_raw_html()


def test_evidence_columns_use_the_row_source_when_cell_keys_repeat():
    failed = _run_mean(_mean_frame(n_control=900, n_treatment=100))
    accepted = _run_mean(_mean_frame())
    joined = failed.concat(accepted)
    interpretation = _interpret(joined)
    failed_row = next(
        row for row in joined if row.source_snapshot_id == failed[0].source_snapshot_id
    )
    accepted_row = next(
        row for row in joined if row.source_snapshot_id == accepted[0].source_snapshot_id
    )
    assert CellKey.from_row(failed_row) == CellKey.from_row(accepted_row)
    assert readout_evidence_columns(interpretation, failed_row)["integrity_status"] == "failed"
    assert (
        readout_evidence_columns(interpretation, accepted_row)["integrity_status"] == "not_rejected"
    )


def test_filtered_available_cell_stays_available_and_sensitivity_cells_keep_advisories():
    from increment.estimation.engine import Method

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _run_mean(
            _mean_frame(),
            guardrail=True,
            cluster="cluster",
            sensitivity_methods=(Method(name="sensitivity"),),
        )
    primary_only = results.filter(lambda row: row.metric == "value")
    columns = readout_evidence_columns(_interpret(primary_only), primary_only[0])
    assert columns["available_cell_count"] == 2
    assert json.loads(cast(str, columns["hidden_cells"]))[0]["available"] is True
    sensitivity_row = next(row for row in results if row.method_role == "sensitivity")
    sensitivity_columns = readout_evidence_columns(_interpret(results), sensitivity_row)
    assert sensitivity_columns["integrity_status"] == "not_rejected"
    assert any(
        item["code"] == "estimation.engine.small_total_clusters"
        for item in json.loads(cast(str, sensitivity_columns["advisories"]))
    )


def test_switchback_contrast_is_sampling_available_and_renders_its_effect():
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    rows = []
    for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control"))):
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": float(cycle + period + step + (group == "treatment")),
                        }
                    )
    results = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={"value": UnitCycleTApproximation()},
    ).run()
    interpretation = _interpret(results)
    assert interpretation.groups[0].available_cells
    result = cast(Any, results[0])
    assert f"{result.estimate.value:.6g}" in render_readout_text(interpretation)
    assert f"{result.null_abs:.6g}" in render_readout_text(interpretation)


def test_effect_carriers_and_winsor_geometry_render_without_reinversion():
    binary_rows = []
    for group in ("control", "treatment"):
        for i in range(40):
            binary_rows.append(
                {
                    "unit": f"{group}-{i}",
                    "arm": group,
                    "converted": int(group == "treatment" and i % 2 == 0),
                }
            )
    binary = Analysis.from_unit_summary(
        pd.DataFrame(binary_rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
        plan=AnalysisPlan(primary="converted"),
    ).run()
    binary_row = cast(Any, binary[0])
    assert binary_row.lift is None and binary_row.binomial_set is not None
    binary_interpretation = _interpret(binary)
    assert binary_interpretation.groups[0].available_cells
    assert "binomial" in render_readout_text(binary_interpretation)

    winsorized = Analysis.from_unit_summary(
        _mean_frame(),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="value", type="mean", winsorization={"upper_percentile": 0.75})],
        plan=AnalysisPlan(primary="value"),
    ).run()
    winsor_text = render_readout_text(_interpret(winsorized))
    winsor_row = cast(Any, winsorized[0])
    assert winsor_row.confidence_set.qualification in winsor_text
    assert f"{winsor_row.confidence_set.relative.lower.value:.6g}" in winsor_text


def test_nonpositive_mean_keeps_sampling_available_additive_interval():
    rows = []
    for i in range(40):
        rows.extend(
            (
                {"unit": f"control-{i}", "arm": "control", "value": 10.0 + (i % 2)},
                {
                    "unit": f"treatment-{i}",
                    "arm": "treatment",
                    "value": -1.0 if i % 2 == 0 else 1.0,
                },
            )
        )
    results = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="value", type="mean")],
        plan=AnalysisPlan(primary="value"),
    ).run()
    row = cast(Any, results[0])
    assert row.lift is None
    assert row.abs_diff is not None and row.abs_lb is not None and row.abs_ub is not None
    interpreted = _interpret(results)
    assert interpreted.groups[0].source_cells[0].sampling_available
    assert f"{row.abs_diff:.6g}" in render_readout_text(interpreted)
    from increment.tables import readout_table

    table_html = readout_table(results).gt().as_raw_html()
    assert "Additive effect" in table_html
    assert f"{row.abs_diff:.6g}" in table_html


def test_absolute_margin_renders_tested_additive_null():
    from increment.semantics.models import ExperimentMetric

    frame = _mean_frame()
    results = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="value", type="mean"),
            MetricSpec(name="guard", type="mean", preferred_direction="decrease"),
        ],
        plan=AnalysisPlan(
            primary="value", guardrails=[ExperimentMetric(metric="guard", margin_abs=0.5)]
        ),
    ).run()
    guard = next(row for row in results if row.metric == "guard")
    assert guard.null_abs is not None
    text = render_readout_text(_interpret(results))
    assert f"{guard.null_abs:.6g}" in text


def test_constant_sensitivity_guard_does_not_create_orphan_advisory_key():
    from increment.estimation.engine import Method

    frame = _mean_frame(constant_guard=True)
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="value", type="mean"),
            MetricSpec(name="guard", type="mean", preferred_direction="decrease"),
        ],
        plan=AnalysisPlan(primary="value", guardrails=["guard"]),
    )
    with pytest.warns(Warning):
        results = analysis.run(
            decision_method=Method(name="decision"),
            sensitivity_methods=(Method(name="sensitivity"),),
        )
    assert results


def test_finalized_registered_bernoulli_does_not_get_open_ended_approximation_advisory():
    from tests.test_sequential_public_sources import _analysis, _frame, _plan

    metric = MetricSpec(name="outcome", type="conversion")
    plan = _plan([metric], "bernoulli")
    analysis = _analysis(_frame([0] * 40, [1, 0] * 20), [metric], plan)
    analysis.capture_sequential(finalized=True)
    results = analysis.run()
    row = results[0]
    assert row.lift is None
    assert row.sequential_result is not None
    bounds = row.sequential_result.bounds
    assert bounds.lower is not None
    advisories = [advisory for record in results.metadata.cells for advisory in record.advisories]
    assert "estimation.engine.open_ended_sequential" not in {item.code for item in advisories}
    assert f"{bounds.lower:.6g}" in render_readout_text(_interpret(results))


def test_encouragement_late_advisory_survives_suppressed_warning_and_json_restore():
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = []
    for group in ("control", "treatment"):
        for cluster in range(3):
            for unit in range(10):
                rows.append(
                    {
                        "unit": f"{group}-{cluster}-{unit}",
                        "arm": group,
                        "cluster": f"{group}-{cluster}",
                        "clicked": int(group == "treatment" and unit < 7),
                        "value": float(unit + (group == "treatment")),
                    }
                )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment affects outcome through uptake"
        ),
        one_sided=True,
        min_first_stage_z=0.001,
    )
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        design=design,
        metrics=[MetricSpec(name="value", type="mean")],
        plan=AnalysisPlan(primary="value"),
        cluster="cluster",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = analysis.run(estimands=("late",))
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    late_records = [record for record in restored.metadata.cells if record.cell.estimand == "late"]
    assert late_records
    assert any(
        item.code == "estimation.engine.small_total_clusters"
        for record in late_records
        for item in record.advisories
    )


def test_posterior_unavailability_reason_is_separate_from_sampling_availability():
    from increment.estimation.inference import Normal

    rows = [
        {
            "unit": f"{group}-{i}",
            "arm": group,
            "converted": int(group == "treatment" and i % 2 == 0),
        }
        for group in ("control", "treatment")
        for i in range(40)
    ]
    results = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
        plan=AnalysisPlan(primary="converted"),
    ).run(prior=Normal(mu=0, sigma=1))
    from increment.tables import readout_table

    table_html = readout_table(results).gt().as_raw_html()
    assert "Posterior unavailable" in table_html
    reason_code = getattr(results[0], "posterior_reason_code", None)
    assert reason_code is not None and reason_code in table_html
    interpretation = _interpret(results)
    cell = interpretation.groups[0].source_cells[0]
    assert cell.sampling_available is True
    assert cell.posterior_available is False
    assert cell.posterior_reason_code is not None


def test_unscoped_rows_keep_row_level_failure_and_posterior_facts():
    rows = list(_run_mean(_mean_frame()))
    success_columns = readout_evidence_columns(interpret_readout(rows, None), rows[0])
    assert success_columns["integrity_status"] == "unknown"
    assert success_columns["sampling_available"] is True
    assert success_columns["point_available"] is True

    failed = rows[0].model_copy(
        update={
            "lift": None,
            "failure_code": "evidence.p_value.unavailable",
            "failure_context": {"reason": "missing_sampling_statistic"},
            "posterior_available": False,
            "posterior_reason_code": "estimation.results.lift.posterior_unavailable",
            "posterior_reason_context": {"reason": "no posterior"},
        }
    )
    interpretation = interpret_readout([failed], None)
    columns = readout_evidence_columns(interpretation, failed)
    assert columns["sampling_available"] is False
    assert columns["point_available"] is False
    assert columns["failure_code"] == "evidence.p_value.unavailable"
    assert json.loads(cast(str, columns["failure_context"])) == {
        "reason": "missing_sampling_statistic"
    }
    assert columns["posterior_available"] is False
    assert columns["posterior_reason_code"] == "estimation.results.lift.posterior_unavailable"
    text = render_readout_text(interpretation)
    assert "evidence.p_value.unavailable" in text
    assert "estimation.results.lift.posterior_unavailable" in text


def test_sequential_text_labels_ratio_geometry_for_finite_unbounded_and_disconnected_sets():
    from tests.estimation.test_asymptotic_mean import _evaluate

    finite = _evaluate([1, 2, 3] * 40, [8, 10, 12] * 40)[0]
    unbounded = _evaluate([1, 2, 3] * 40, [8, 10, 12] * 40, alternative="greater")[0]
    disconnected = _evaluate([-1, 1] * 40, [10, 12] * 40)[0]
    finite_bounds = finite.require_asymptotic_sequential_result().bounds
    unbounded_bounds = unbounded.require_asymptotic_sequential_result().bounds
    disconnected_bounds = disconnected.require_asymptotic_sequential_result().bounds
    assert finite_bounds.lower is not None and finite_bounds.upper is not None
    assert unbounded_bounds.upper is None
    assert disconnected_bounds.status == "disconnected"
    decreasing = _evaluate([8, 10, 12] * 40, [1, 2, 3] * 40)[0]
    decreasing_bounds = decreasing.require_asymptotic_sequential_result().bounds
    assert decreasing_bounds.upper is not None and decreasing_bounds.upper < 1.0
    text = render_readout_text(interpret_readout([finite, unbounded, disconnected], None))
    assert text.count("sequential ratio confidence set") == 3
    assert "disconnected" in text
    assert "+∞" in text
    assert "effect direction supported (decrease)" in render_readout_text(
        interpret_readout([decreasing], None)
    )


def test_missing_additive_interval_is_not_reported_as_an_unbounded_set():
    rows = [
        {
            "unit": f"{group}-{i}",
            "arm": group,
            "converted": int(group == "treatment"),
        }
        for group in ("control", "treatment")
        for i in range(40)
    ]
    result = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
        plan=AnalysisPlan(primary="converted"),
    ).run()[0]
    from increment.estimation.results import LiftEstimate

    assert isinstance(result, LiftEstimate)
    assert result.abs_diff == 1.0 and result.abs_lb is None and result.abs_ub is None
    text = render_readout_text(interpret_readout([result], None))
    assert "additive interval unavailable" in text
    assert "additive effect: 1" in text
    assert "set [−∞, +∞]" not in text


def test_signed_support_rank_set_uses_correct_unbounded_endpoint_direction():
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport

    raw = WinsorRawState(
        metric="revenue",
        study_id="signed",
        population="assigned",
        missingness="error",
        quantile=0.75,
        inference=WinsorInferenceSpec(method="joint-rank-projection-v1"),
        support=WinsorSupport(lower=-10, provenance="external signed support"),
        arms=(
            RawArm(group_id="control", values=(-2, -1, 0, 1)),
            RawArm(group_id="treatment", values=(-1, 0, 1, 2)),
        ),
    )
    result = estimate_winsor_lift(raw, "control", "treatment", alpha=0.0001)
    assert result.confidence_set is not None
    assert result.confidence_set.additive.lower.status == "unbounded"
    text = render_readout_text(interpret_readout([result], None))
    assert "additive confidence interval: [−∞, +∞]" in text
    assert "additive interval unavailable" not in text
    from increment.breakout.estimates import LiftEstimates
    from increment.tables import readout_table

    html = readout_table(LiftEstimates([result])).gt().as_raw_html()
    assert "[−∞, +∞]" in html
    assert "interval unavailable" not in html


def test_absolute_scale_null_is_not_labeled_as_relative_lift():
    row = list(_run_mean(_mean_frame()))[0].model_copy(
        update={
            "value_scale": "absolute",
            "null_abs": None,
            "null_lift": 0.0,
            "alternative": "greater",
        }
    )
    text = render_readout_text(interpret_readout([row], None))
    assert "null (absolute units): 0" in text
    assert "null (relative lift):" not in text
