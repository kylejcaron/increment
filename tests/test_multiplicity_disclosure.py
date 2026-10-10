"""Multiplicity provenance must not become a correction inferred from visible rows."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, MetricSpec
from increment.decision import MultiplicityFamily


def _analysis(plan=None):
    n = 400
    noise = np.tile(np.linspace(-1.0, 1.0, n), 2)
    treatment = np.repeat([0.0, 1.0], n)
    return Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(2 * n)],
                "arm": ["control"] * n + ["treatment"] * n,
                **{f"m{i}": 10.0 + noise + treatment * 0.2 for i in range(6)},
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name=f"m{i}") for i in range(6)],
        plan=plan,
    )


def test_five_unplanned_metrics_are_visibly_unadjusted():
    from increment.estimation.readout_types import ReadoutResults
    from increment.tables import estimates_to_readout

    rows = _analysis().run(metrics=[f"m{i}" for i in range(5)])
    assert len(rows) == 5
    assert {row.role for row in rows} == {None}
    assert {row.multiplicity_status for row in rows} == {"undeclared_plan"}
    assert set(rows.to_frame()["multiplicity_status"]) == {"undeclared_plan"}
    assert {row["multiplicity_status"] for row in estimates_to_readout(rows)} == {"undeclared_plan"}
    restored = ReadoutResults.model_validate_json(rows.model_dump_json())
    assert {row.multiplicity_status for row in restored} == {"undeclared_plan"}
    assert restored.metadata == rows.metadata


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("undeclared_plan", "Unadjusted (no declared plan)"),
        ("unassigned_in_plan", "Unadjusted (unassigned in plan)"),
        ("exploratory_unadjusted", "Exploratory (unadjusted)"),
        ("exploratory_family", "Exploratory family"),
        ("declared_plan", "Declared plan"),
    ],
)
def test_readout_table_renders_multiplicity_disclosure(status, label):
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    html = (
        readout_table(
            [
                {
                    "metric": "revenue",
                    "group_id": "treatment",
                    "method": "unadjusted",
                    "lift": 0.05,
                    "lower": -0.01,
                    "higher": 0.11,
                    "multiplicity_status": status,
                }
            ]
        )
        .gt()
        .as_raw_html()
    )
    assert "Multiplicity" in html
    assert label in html


def test_primary_allocation_and_family_survive_filter_and_additional_declared_metric_selection():
    names = [f"m{i}" for i in range(5)]
    analysis = _analysis(AnalysisPlan(primary=names))
    rows = analysis.run(metrics=names)
    assert {row.role for row in rows} == {"primary"}
    assert {row.multiplicity_status for row in rows} == {"declared_plan"}
    assert [row.require_lift().alpha for row in rows] == pytest.approx([0.01] * 5)
    families = rows.metadata.scope.families
    primary = [family for family in families if family.name == "primary"]
    assert len(primary) == 1
    assert {cell.metric for cell in primary[0].members} == set(names)
    filtered = rows.filter(lambda row: row.metric == "m0")
    assert filtered.metadata.scope.families == families
    assert filtered[0].family_id == rows[0].family_id
    assert filtered.to_frame()["multiplicity_status"].tolist() == ["declared_plan"]
    assert filtered[0].require_lift().alpha == pytest.approx(0.01)
    extended = analysis.run(metrics=names + ["m5"])
    assert [row.require_lift().alpha for row in extended if row.metric in names] == pytest.approx(
        [0.01] * 5
    )
    extra = next(row for row in extended if row.metric == "m5")
    assert extra.role == "secondary"
    assert extra.multiplicity_status == "declared_plan"
    extended_primary = next(
        family for family in extended.metadata.scope.families if family.name == "primary"
    )
    assert extended_primary.members == primary[0].members


@pytest.mark.parametrize(
    ("role", "correction", "expected"),
    [
        (None, None, "undeclared_plan"),
        ("unassigned", None, "unassigned_in_plan"),
        ("primary", None, "declared_plan"),
        ("secondary", "bh", "declared_plan"),
        ("guardrail", None, "declared_plan"),
        ("exploratory", None, "exploratory_unadjusted"),
        ("exploratory", "none", "exploratory_unadjusted"),
        ("exploratory", "bonferroni", "exploratory_family"),
        ("exploratory", "e_bh", "exploratory_family"),
    ],
)
def test_provenance_preserves_role_distinctions(role, correction, expected):
    from increment.estimation.multiplicity import multiplicity_status

    family = (
        None
        if correction is None
        else MultiplicityFamily(
            name="breakout",
            correction=correction,
            q=0.1 if correction in ("bh", "e_bh") else None,
            guarantee="fdr"
            if correction in ("bh", "e_bh")
            else "fwer"
            if correction == "bonferroni"
            else "none",
        )
    )
    assert multiplicity_status(role, family) == expected
