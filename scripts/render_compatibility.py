"""Render the method compatibility reference from declared and observed support.

Evaluate ``MATRIX``, ``PAIRS`` and ``SCENARIOS`` once, then share the results
between static tables, expandable details and the browser's JSON payload.
The browser looks up these results; it never derives a status.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from tests.compatibility_catalog import (
    CAPABILITY_PROVENANCE,
    MATRIX,
    PAIR_PROVENANCE,
    PAIRS,
    SCENARIOS,
    Cell,
    Provenance,
    Scenario,
    validate_evidence,
    validate_scenario_outcome,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/guides/compatibility.md"

# Share explicit labels with the browser so displayed statuses cannot drift.
# JSON callers use plain strings; tests check that every typed status has a label.
STATUS_LABELS: dict[str, str] = {
    "supported": "SUPPORTED",
    "limited": "LIMITED",
    "refused": "REFUSED",
    "not_applicable": "NOT APPLICABLE",
    "unknown": "UNKNOWN",
    "participates": "PARTICIPATES",
    "excluded": "EXCLUDED",
}


def _display_status(value: str) -> str:
    """Look up a display label; raise ``KeyError`` for an unknown status.

    Internal keys remain snake_case. Only user-facing labels use this mapping.
    """
    return STATUS_LABELS[value]


def _detail(cell: Cell) -> str:
    """Append the complete runtime advisory to a cell's explanation.

    ``cell.warns`` is a test-matching fragment, not display text. Keep the base
    explanation and punctuate it before appending ``cell.advisory``.
    """
    base = cell.note or cell.reason or cell.fragment or "Probed end to end."
    if not cell.advisory:
        return base
    punctuated = base if base.rstrip().endswith((".", "!", "?")) else f"{base.rstrip()}."
    return f"{punctuated} {cell.advisory}"


def _refusal_payload(cell: Cell) -> dict | None:
    """Return refusal code, exception and fragment, or ``None`` if not refused.

    An undeclared code has the explicit display value ``"unavailable"``.
    """
    if cell.status != "refused":
        return None
    assert cell.raises is not None and cell.fragment is not None
    return {
        "code": cell.code,
        "code_display": cell.refusal_code_display,
        "exception": cell.raises.__name__,
        "fragment": cell.fragment,
    }


def _cell_payload(cell: Cell, provenance: Provenance) -> dict:
    """Return a cell's status, explanation, owner, probe and refusal metadata."""
    return {
        "runtime": cell.runtime_status,
        "family": "not_applicable",
        "explanation": _detail(cell),
        "evidence_source": "probe",
        "owner": provenance.owner,
        "probe": provenance.probe,
        "refusal": _refusal_payload(cell),
    }


def _provenance_detail(cell: Cell, provenance: Provenance) -> str:
    """Show the same owner, evidence and refusal facts with or without JavaScript."""
    parts = [
        "<p>Owner: " + html.escape(provenance.owner) + "</p>",
        "<p>Evidence: " + html.escape(provenance.probe) + "</p>",
    ]
    refusal = _refusal_payload(cell)
    if refusal is not None:
        parts.append(
            "<p>Refusal code: "
            + html.escape(refusal["code_display"])
            + "; exception: "
            + html.escape(refusal["exception"])
            + "; fragment: "
            + html.escape(refusal["fragment"])
            + "</p>"
        )
    return "".join(parts)


# Cell identity uses capability/metric_type; pair identity uses left/right.
# Other axes distinguish interactions. Keep aligned with the browser's IDENTITY_AXES.
IDENTITY_AXES = frozenset({"capability", "metric_type", "left", "right"})


def _interaction_label(scenario: Scenario) -> str:
    """Label the axes that distinguish this scenario from others in its cell.

    Match the browser's ``interactionLabel`` so static and interactive details
    show the same context, derived from the scenario rather than a fixed label.
    """
    context = sorted((axis, value) for axis, value in scenario.axes if axis not in IDENTITY_AXES)
    return ", ".join(
        f"{axis.replace('_', ' ').title()}: {value.replace('_', ' ')}" for axis, value in context
    )


def _floor_text(floor: dict) -> str:
    """Render a sampling floor's kind and numeric fields without assuming a variant."""
    kind = floor["kind"]
    extra = ", ".join(f"{key}={value}" for key, value in floor.items() if key != "kind")
    return f"{kind} ({extra})" if extra else kind


def _finding_summary(finding: dict) -> str:
    """One finding's runtime plus whichever of warning/refusal code/
    assumptions/sampling floor/reference are present."""
    parts = [f"{finding['capability']}: runtime={_display_status(finding['runtime'])}"]
    if finding.get("warning"):
        parts.append(f"warning: {finding['warning']}")
    if finding.get("refusal_code"):
        parts.append(f"refusal code: {finding['refusal_code']}")
    if finding.get("assumptions"):
        parts.append(f"assumptions: {', '.join(finding['assumptions'])}")
    if finding.get("floor"):
        parts.append(f"sampling floor: {_floor_text(finding['floor'])}")
    if finding.get("reference"):
        parts.append(f"reference: {finding['reference']}")
    return "; ".join(parts)


def _contract_provenance(findings: list[dict]) -> str:
    """Summarize the capability and evidence contract behind each finding."""
    return "; ".join(
        f"{finding['capability']} ({finding['evidence_source']}"
        + (f", ref={finding['reference']}" if finding.get("reference") else "")
        + ")"
        for finding in findings
    )


def _reference_payload():
    validate_evidence(SCENARIOS)
    scenarios = {}
    for scenario in sorted(SCENARIOS, key=lambda item: item.id):
        report = scenario.evaluate()
        validate_scenario_outcome(scenario, report)
        scenarios[scenario.id] = {
            "axes": dict(scenario.axes),
            "runtime": report.overall,
            "family": report.overall_family,
            "explanation": scenario.explanation,
            "alternative": scenario.alternative,
            "evidence": [
                {"kind": evidence.kind, "path": evidence.path, "test": evidence.test}
                for evidence in scenario.evidence
            ],
            "findings": [finding.model_dump(mode="json") for finding in report.findings],
        }
    cells = {
        f"{capability}|{metric}": _cell_payload(cell, CAPABILITY_PROVENANCE[capability])
        for capability, column in sorted(MATRIX.items())
        for metric, cell in column.items()
    }
    pairs = {
        f"{left}|{right}": _cell_payload(cell, PAIR_PROVENANCE[left, right])
        for (left, right), cell in sorted(PAIRS.items())
    }
    return {
        "cells": cells,
        "pairs": pairs,
        "scenarios": scenarios,
        "status_labels": STATUS_LABELS,
    }


def render_reference() -> str:
    payload = _reference_payload()
    metric_types = tuple(next(iter(MATRIX.values())))
    headings = "".join(f'<th scope="col">{html.escape(metric)}</th>' for metric in metric_types)
    rows = []
    base_details = []
    for capability, column in sorted(MATRIX.items()):
        cells = []
        for metric in metric_types:
            cell = column[metric]
            provenance = CAPABILITY_PROVENANCE[capability]
            cells.append(
                '<td><button type="button" class="compatibility-cell" '
                f'data-capability="{html.escape(capability)}" '
                f'data-metric="{html.escape(metric)}" '
                f'data-status="{html.escape(cell.runtime_status)}" '
                f'aria-label="{html.escape(f"{capability} {metric}: {_display_status(cell.runtime_status)}")}">'
                f"{html.escape(_display_status(cell.runtime_status))}</button></td>"
            )
            base_details.append(
                "<details><summary>"
                + html.escape(f"{capability} × {metric}: {_display_status(cell.runtime_status)}")
                + "</summary><p>"
                + html.escape(_detail(cell))
                + "</p>"
                + _provenance_detail(cell, provenance)
                + "<p>Evidence source: probe</p></details>"
            )
        rows.append(f'<tr><th scope="row">{html.escape(capability)}</th>{"".join(cells)}</tr>')
    pair_rows = []
    pair_details = []
    for (left, right), cell in sorted(PAIRS.items()):
        provenance = PAIR_PROVENANCE[left, right]
        pair_rows.append(
            '<tr><th scope="row">'
            + html.escape(f"{left} × {right}")
            + '</th><td><button type="button" class="compatibility-cell compatibility-pair" '
            + f'data-left="{html.escape(left)}" data-right="{html.escape(right)}" '
            + f'data-status="{html.escape(cell.runtime_status)}" '
            + f'aria-label="{html.escape(f"{left} {right}: {_display_status(cell.runtime_status)}")}">'
            + html.escape(_display_status(cell.runtime_status))
            + "</button></td><td>"
            + html.escape(_detail(cell))
            + "</td></tr>"
        )
        pair_details.append(
            "<details><summary>"
            + html.escape(f"{left} × {right}: {_display_status(cell.runtime_status)}")
            + "</summary><p>"
            + html.escape(_detail(cell))
            + "</p>"
            + _provenance_detail(cell, provenance)
            + "<p>Evidence source: probe</p></details>"
        )
    scenario_details = []
    for scenario in sorted(SCENARIOS, key=lambda item: item.id):
        observed = payload["scenarios"][scenario.id]
        findings = observed["findings"]
        evidence = "".join(
            "<li>" + html.escape(f"{item['kind']}: {item['path']}::{item['test']}") + "</li>"
            for item in observed["evidence"]
        )
        findings_html = "".join(
            "<li>" + html.escape(_finding_summary(finding)) + "</li>" for finding in findings
        )
        alternative = (
            "<p>Alternative: " + html.escape(scenario.alternative) + "</p>"
            if scenario.alternative is not None
            else ""
        )
        label = _interaction_label(scenario)
        context = "<p>Context: " + html.escape(label) + "</p>" if label else ""
        scenario_details.append(
            "<details><summary>"
            + html.escape(scenario.id)
            + "</summary>"
            + context
            + "<p>Runtime: "
            + html.escape(_display_status(observed["runtime"]))
            + "; Family: "
            + html.escape(_display_status(observed["family"]))
            + "</p><p>"
            + html.escape(scenario.explanation)
            + "</p>"
            + alternative
            + "<p>Evidence</p><ul>"
            + evidence
            + "</ul>"
            + "<p>Contract provenance: "
            + html.escape(_contract_provenance(findings))
            + "</p>"
            + "<p>Findings</p><ul>"
            + findings_html
            + "</ul></details>"
        )
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (
        "<!-- Generated by: uv run python scripts/render_compatibility.py -->\n"
        "# Method compatibility reference\n\n"
        "Use the matrix to check support for a capability and metric type. "
        "Select a cell to see tested combinations with other methods. "
        "An unlisted combination is not evidence of support.\n\n"
        '!!! warning "Supported does not mean calibrated"\n'
        "    SUPPORTED means the combination runs and passes the runtime checks "
        "behind this page. It does not establish statistical calibration. "
        "Any open calibration gaps still apply; check "
        "[Statistical limitations](../limitations.md) before using a result.\n\n"
        '<div id="compatibility-reference">'
        + '<div class="compatibility-scroll"><table class="compatibility-matrix">'
        + "<thead><tr><th>Capability</th>"
        + headings
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + "<h2>Capability interactions</h2>"
        + '<div class="compatibility-scroll"><table class="compatibility-pairs">'
        + "<thead><tr><th>Combination</th><th>Status</th><th>Reason</th>"
        + "</tr></thead><tbody>"
        + "".join(pair_rows)
        + "</tbody></table></div>"
        + '<section id="compatibility-drawer" aria-live="polite"></section>'
        + '<section id="compatibility-static-details"><h2>Detailed reference</h2>'
        + "".join(base_details + pair_details + scenario_details)
        + "</section></div>"
        + '<script id="compatibility-data" type="application/json">'
        + data.replace("</", "<\\/")
        + "</script>\n"
    )


def write_reference(path: Path = DEFAULT_OUTPUT) -> None:
    path.write_text(render_reference())


def check_reference(path: Path = DEFAULT_OUTPUT) -> bool:
    return path.exists() and path.read_text() == render_reference()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return 0 if check_reference() else 1
    write_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
