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
    ("status", "note"),
    [
        ("undeclared_plan", "Unadjusted for multiplicity (no declared plan)"),
        ("unassigned_in_plan", "Unadjusted for multiplicity (unassigned in plan)"),
        ("exploratory_unadjusted", "Exploratory, unadjusted for multiplicity"),
        ("exploratory_family", "Exploratory family"),
        ("declared_plan", None),
    ],
)
def test_readout_table_summarizes_multiplicity_as_informative_notes(status, note):
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    table = readout_table(
        [
            {
                "metric": "revenue",
                "group_id": "treatment",
                "method": "unadjusted",
                "lift": 0.05,
                "lower": -0.01,
                "higher": 0.11,
                "multiplicity_status": status,
                "decision_scope_complete": True,
                "view_partial": False,
            },
            {
                "metric": "signups",
                "group_id": "treatment",
                "method": "unadjusted",
                "lift": 0.02,
                "lower": -0.01,
                "higher": 0.05,
                "multiplicity_status": status,
                "decision_scope_complete": True,
                "view_partial": False,
            },
        ]
    )
    html = table.gt().as_raw_html()

    assert ">Multiplicity<" not in html
    assert ">Decision scope<" not in html
    assert ">Readout view<" not in html
    assert "decision cell" not in html
    assert "Partial view" not in html
    assert ">Discovery<" not in html
    if note is None:
        assert "Unadjusted for multiplicity" not in html
        assert "Exploratory family" not in html
    else:
        assert note in html
        assert html.count(note) == 1


def test_readout_table_summarizes_incomplete_partial_and_failure_details_once():
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    code = "estimation.engine.lift_guard." + "nested." * 10
    table = readout_table(
        [
            {
                "metric": "revenue",
                "group_id": "treatment",
                "method": "unadjusted",
                "lift": 0.05,
                "lower": -0.01,
                "higher": 0.11,
                "decision_scope_complete": False,
                "view_partial": True,
                "failure_code": code,
                "failure_context": {"reason": "an unusually long failure reason " * 8},
                "multiplicity_status": "undeclared_plan",
            },
            {
                "metric": "signups",
                "group_id": "treatment",
                "method": "unadjusted",
                "lift": 0.02,
                "lower": -0.01,
                "higher": 0.05,
                "decision_scope_complete": False,
                "view_partial": True,
                "multiplicity_status": "undeclared_plan",
            },
        ]
    )
    html = table.gt().as_raw_html()

    assert ">Decision scope<" not in html
    assert ">Readout view<" not in html
    assert ">Multiplicity<" not in html
    assert "Decision scope is incomplete; see Failure." in html
    assert "decision cells are incomplete" not in html
    assert "see Failure" in html
    assert "Partial view" in html
    assert "Failure details are abbreviated" in html
    assert "an unusually long failure reason " * 2 not in html
    assert html.count("Partial view") == 1
    assert html.count(code) == 1
    assert "Failure" in html


def test_mixed_multiplicity_notes_keep_full_status_mapping_for_truncated_groups():
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    rows = [
        {
            "metric": f"m{i}",
            "group_id": "treatment",
            "method": "unadjusted",
            "lift": 0.05,
            "lower": -0.01,
            "higher": 0.11,
            "multiplicity_status": ("exploratory_family" if i < 4 else "exploratory_unadjusted"),
            "decision_scope_complete": True,
            "view_partial": False,
        }
        for i in range(8)
    ]

    html = readout_table(rows).gt().as_raw_html()

    assert ">Multiplicity<" not in html
    assert "¹ Exploratory family (4 of 8 rows)" in html
    assert "² Exploratory, unadjusted for multiplicity (4 of 8 rows)" in html
    assert "m0¹" in html and "m3¹" in html
    assert "m4²" in html and "m7²" in html
    assert all(f"m{i}" in html for i in range(8))


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


def test_mixed_multiplicity_notes_identify_only_rows_they_describe():
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    rows = [
        {
            "metric": metric,
            "group_id": "treatment",
            "method": "unadjusted",
            "lift": 0.05,
            "lower": -0.01,
            "higher": 0.11,
            "multiplicity_status": status,
        }
        for metric, status in (
            ("primary_metric", "declared_plan"),
            ("conversion", "exploratory_unadjusted"),
            ("revenue", "undeclared_plan"),
            ("signup", "exploratory_family"),
        )
    ]

    html = readout_table(rows).gt().as_raw_html()

    assert "1 of 4 rows" in html
    assert "primary_metric" in html
    assert "conversion¹" in html
    assert "revenue²" in html
    assert "signup³" in html
    assert "¹ Exploratory, unadjusted for multiplicity (1 of 4 rows)" in html
    assert "² Unadjusted for multiplicity (no declared plan) (1 of 4 rows)" in html
    assert "³ Exploratory family (1 of 4 rows)" in html
    subtitle = html.rsplit("gt_subtitle", maxsplit=1)[-1].split("</td>", 1)[0]
    assert all(
        name not in subtitle for name in ("primary_metric", "conversion", "revenue", "signup")
    )


def test_mixed_multiplicity_notes_use_markers_without_row_lists():
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    rows = [
        {
            "metric": f"metric_{index:02d}",
            "group_id": "treatment",
            "method": "unadjusted",
            "lift": 0.05,
            "lower": -0.01,
            "higher": 0.11,
            "multiplicity_status": "declared_plan" if index == 0 else "exploratory_unadjusted",
        }
        for index in range(40)
    ]

    html = readout_table(rows).gt().as_raw_html()
    subtitle = html.rsplit("gt_subtitle", maxsplit=1)[-1].split("</td>", 1)[0]

    assert "39 of 40 rows" in subtitle
    assert "¹ Exploratory, unadjusted for multiplicity (39 of 40 rows)" in subtitle
    assert "metric_00" not in subtitle
    assert "metric_01¹" in html and "metric_39¹" in html
    assert ">Multiplicity<" not in html


def test_mixed_multiplicity_markers_disambiguate_shared_metric_rows():
    from increment.tables import readout_table

    pytest.importorskip("coeftable")
    html = (
        readout_table(
            [
                {
                    "metric": "conversion",
                    "group_id": "treatment",
                    "method": "decision",
                    "segment": "US",
                    "lift": 0.05,
                    "lower": -0.01,
                    "higher": 0.11,
                    "multiplicity_status": "exploratory_family",
                },
                {
                    "metric": "conversion",
                    "group_id": "treatment",
                    "method": "sensitivity",
                    "segment": "MX",
                    "lift": 0.02,
                    "lower": -0.01,
                    "higher": 0.05,
                    "multiplicity_status": "exploratory_unadjusted",
                },
            ]
        )
        .gt()
        .as_raw_html()
    )

    assert "conversion¹" in html and "conversion²" in html
    assert "¹ Exploratory family (1 of 2 rows)" in html
    assert "² Exploratory, unadjusted for multiplicity (1 of 2 rows)" in html
