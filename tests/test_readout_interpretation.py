"""Public readout journeys retain evidence for interpretation and replay."""

import json
import warnings

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


def _run_mean(frame, *, allocation=True, guardrail=False, cluster=None):
    design = Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5} if allocation else None,
        allocation_scheme="independent" if allocation else None,
    )
    metrics = [MetricSpec(name="value", type="mean")]
    plan = AnalysisPlan(primary="value", guardrails=["guard"] if guardrail else [])
    if guardrail:
        metrics.append(MetricSpec(name="guard", type="mean", preferred_direction="decrease"))
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        design=design,
        metrics=metrics,
        plan=plan,
        cluster=cluster,
    ).run()


def _interpret(results):
    return interpret_readout(results, results.metadata)


def test_failed_declared_allocation_is_integrity_first_and_not_laundered():
    results = _run_mean(_mean_frame(n_control=900, n_treatment=100))
    text = render_readout_text(_interpret(results))
    assert "failed" in text


def test_not_rejected_integrity_is_described_as_no_detected_mismatch_not_proof():
    results = _run_mean(_mean_frame())
    text = render_readout_text(_interpret(results))
    assert "not_rejected — no detected mismatch; not proof" in text


def test_missing_allocation_declaration_is_not_success():
    results = _run_mean(_mean_frame(), allocation=False)
    text = render_readout_text(_interpret(results))
    assert "not_checked_missing_declaration" in text
    assert "unknown" not in text


def test_required_guardrail_failure_remains_visible_when_filtered_out():
    with pytest.warns(Warning):
        results = _run_mean(_mean_frame(constant_guard=True), guardrail=True)
    filtered = results.filter(lambda row: row.metric == "value")
    interpretation = _interpret(filtered)
    text = render_readout_text(interpretation)
    assert "zero_variance" in text
    assert "hidden" in text
    assert "partial" in text

    columns = readout_evidence_columns(interpretation, CellKey.from_row(filtered[0]))
    assert columns["integrity_status"] == "not_rejected"
    assert columns["multiplicity_status"] == filtered[0].multiplicity_status
    assert isinstance(columns["hidden_cells"], str)
    hidden_cells = json.loads(columns["hidden_cells"])
    hidden = next(item for item in hidden_cells if item["cell"]["metric"] == "guard")
    assert hidden["code"]
    assert hidden["context"]["reason"] == "zero_variance"


def test_v1_readout_envelope_refuses_with_recompute_route():
    results = _run_mean(_mean_frame())
    payload = json.loads(results.model_dump_json())
    payload["schema_version"] = 1
    with pytest.raises(WireFormatError) as caught:
        ReadoutResults.model_validate_json(json.dumps(payload))
    assert caught.value.code == "readout.serialization.unsupported_version"
    assert "recompute" in str(caught.value.context["remedy"]).lower()


def test_unscoped_plain_list_reports_unknown_scope_and_integrity():
    results = _run_mean(_mean_frame())
    text = render_readout_text(interpret_readout(list(results), None))
    assert "unknown" in text
    assert "Integrity: unknown" in text


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
    assert "small_total_clusters" in render_readout_text(interpretation)


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
    advisories = [advisory for record in results.metadata.cells for advisory in record.advisories]
    assert len(advisories) == 1
    assert advisories[0].code == "estimation.engine.ratio_denominator_precision"
    assert advisories[0].message.startswith("ratio_denominator_precision:")
