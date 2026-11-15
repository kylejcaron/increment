"""Multiplicity provenance must not become a correction inferred from visible rows."""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pyarrow as pa
import pytest

from increment import AnalysisPlan, MetricSpec, from_frame
from increment.decision import MultiplicityFamily
from tests.analysis_factory import view_readout


def _analysis(plan=None):
    n = 400
    noise = np.tile(np.linspace(-1.0, 1.0, n), 2)
    treatment = np.repeat([0.0, 1.0], n)
    return from_frame(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(2 * n)],
                "arm": ["control"] * n + ["treatment"] * n,
                **{f"m{i}": 10.0 + noise + treatment * 0.2 for i in range(6)},
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name=f"m{i}") for i in range(6)],
        plan=plan,
    )


def test_old_selection_payload_does_not_infer_family_procedure():
    from increment.estimation.results import LiftEstimate, _selection_qualification

    restored_old = LiftEstimate.model_construct(
        discovery=True,
        family_procedure=None,
        family_q=0.05,
        family_size=4,
    )
    qualification = _selection_qualification(restored_old)
    assert qualification is not None
    assert "procedure not recorded" in qualification
    assert "Bonferroni" not in qualification
    assert "Benjamini-Hochberg" not in qualification


def test_five_unplanned_metrics_are_visibly_unadjusted():
    from increment.estimation.readout_types import ReadoutResults
    from increment.tables import estimates_to_readout

    rows = view_readout(*_analysis(), metrics=[f"m{i}" for i in range(5)])
    assert len(rows) == 5
    assert {row.role for row in rows} == {None}
    assert {row.multiplicity_status for row in rows} == {"undeclared_plan"}
    assert set(
        nw.from_native(rows.to_frame(), eager_only=True)["multiplicity_status"].to_list()
    ) == {"undeclared_plan"}
    assert {row["multiplicity_status"] for row in estimates_to_readout(rows)} == {"undeclared_plan"}
    restored = ReadoutResults.model_validate_json(rows.model_dump_json())
    assert {row.multiplicity_status for row in restored} == {"undeclared_plan"}
    assert restored.metadata == rows.metadata


def test_readout_table_preserves_failure_code():
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
    assert code in html


def test_mixed_multiplicity_table_keeps_all_metric_rows():
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

    assert all(f"m{i}" in html for i in range(8))


def test_primary_allocation_and_family_survive_filter_and_additional_declared_metric_selection():
    names = [f"m{i}" for i in range(5)]
    experiment, resolver = _analysis(AnalysisPlan(primary=names))
    rows = view_readout(experiment, resolver, metrics=names)
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
    assert nw.from_native(filtered.to_frame(), eager_only=True)[
        "multiplicity_status"
    ].to_list() == ["declared_plan"]
    assert filtered[0].require_lift().alpha == pytest.approx(0.01)
    extended = view_readout(experiment, resolver, metrics=names + ["m5"])
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
