from __future__ import annotations

import pytest

from tests.capability_entry_point_catalog import (
    INGRESSES,
    TABLE_COLUMNS,
    CaseBinding,
    CellClause,
    ConsumerBinding,
    EntryRow,
    MatrixBinding,
    UnmeasuredBinding,
    resolve_binding,
    resolve_cell,
    validate_catalog,
)


def _row(bindings: dict[str, tuple[CellClause, ...]]) -> EntryRow:
    return EntryRow("sample", "Sample", "ingresses", bindings, {})


def test_unmeasured_binding_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        resolve_binding(UnmeasuredBinding(" "))


def test_resolve_cell_preserves_guardrail_request_variants_and_codes() -> None:
    clauses = (
        CellClause(
            "ITT included", CaseBinding("encouragement_margin_guardrail", "from_unit_summary")
        ),
        CellClause(
            "ITT excluded",
            CaseBinding("encouragement_margin_guardrail_late_only", "from_unit_summary"),
        ),
    )
    resolved = resolve_cell(clauses)
    assert [condition for condition, _ in resolved] == ["ITT included", "ITT excluded"]
    assert [display.status for _, display in resolved] == ["supported", "refused"]
    assert resolved[1][1].codes == ("readout.encouragement.margin",)


def test_stored_context_projection_keeps_format_two_and_format_one_outcomes() -> None:
    from tests.capability_entry_point_catalog import ENTRY_ROWS

    row = next(row for row in ENTRY_ROWS if row.id == "artifact-context-formats")
    clauses = resolve_cell(row.bindings["from_unit_day_artifact"])
    assert [condition for condition, _ in clauses] == ["format 2", "format 1"]
    assert [display.status for _, display in clauses] == ["supported", "refused"]
    assert clauses[1][1].codes == ("artifact.format.unsupported",)


def test_unknown_case_or_matrix_ids_fail_instead_of_inferencing() -> None:
    with pytest.raises(KeyError):
        resolve_binding(CaseBinding("not-a-real-case", INGRESSES[0]))
    with pytest.raises(KeyError):
        resolve_binding(MatrixBinding("not-a-real-cell", INGRESSES[0], "rows"))


def test_catalog_requires_every_entry_point_and_nonempty_cells() -> None:
    row = _row({ingress: (CellClause("", UnmeasuredBinding("explicit")),) for ingress in INGRESSES})
    validate_catalog((row,))
    with pytest.raises(ValueError):
        validate_catalog((EntryRow("partial", "Partial", "ingresses", {}, {}),))


def test_unknown_consumer_metadata_reference_fails_validation() -> None:
    columns = TABLE_COLUMNS["triggered-auxiliary"]
    row = EntryRow(
        "unknown-consumer-reference",
        "Unknown consumer reference",
        "triggered-auxiliary",
        {
            column: (
                CellClause(
                    "",
                    ConsumerBinding("missing-contract", "missing-outcome")
                    if column == columns[0]
                    else UnmeasuredBinding("explicit"),
                ),
            )
            for column in columns
        },
        {},
    )
    with pytest.raises((KeyError, ValueError)):
        validate_catalog((row,))


def test_linear_cuped_interactions_are_explicitly_classified():
    from tests.capability_contracts import CUPED_LINEAR_INTERACTIONS

    assert {
        interaction.name: (interaction.status, interaction.code)
        for interaction in CUPED_LINEAR_INTERACTIONS
    } == {
        "scalar_cuped": ("supported", None),
        "ratio": ("supported", None),
        "clustering": ("unfinished", "estimation.engine.cuped_linear_cluster"),
        "sequential": ("unfinished", "estimation.engine.cuped_linear_sequential"),
        "winsorization_fixed_value": ("supported", None),
        "winsorization_percentile": (
            "unfinished",
            "estimation.engine.cuped_linear_winsorization",
        ),
        "breakouts": ("supported", None),
        "multiplicity_roles": ("supported", None),
        "triggered_population": ("supported", None),
    }
    routes = {
        interaction.name: interaction.route_forward for interaction in CUPED_LINEAR_INTERACTIONS
    }
    assert routes["scalar_cuped"]
    assert routes["clustering"]
    assert "6rhs" in routes["sequential"]
    assert routes["winsorization_fixed_value"]
    assert routes["winsorization_percentile"]


def test_cuped_interaction_catalog_rows_cover_ingress_specific_outcomes():
    from tests.capability_contracts import CUPED_LINEAR_INTERACTIONS
    from tests.capability_entry_point_catalog import ENTRY_ROWS, resolve_cell

    rows = {row.id: row for row in ENTRY_ROWS}
    for interaction in CUPED_LINEAR_INTERACTIONS:
        row = rows[f"cuped-linear-interaction-{interaction.name}"]
        assert set(row.bindings) == set(INGRESSES)
        for ingress in (
            "from_unit_summary",
            "from_unit_panel",
            "from_definitions",
            "from_unit_day_artifact",
        ):
            display = resolve_cell(row.bindings[ingress])[0][1]
            disposition = interaction.for_ingress(ingress)
            assert display.status == disposition.status
            assert display.codes == ((disposition.code,) if disposition.code else ())
        for ingress, code in (
            ("from_moments", "facade.analysis_config.cuped_linear_moments"),
            ("from_switchback_panel", "source.frame.switchback.metric"),
        ):
            display = resolve_cell(row.bindings[ingress])[0][1]
            assert display.status == "source_limited"
            assert display.codes == (code,)

    summary_breakout = resolve_cell(
        rows["cuped-linear-interaction-breakouts"].bindings["from_unit_summary"]
    )[0][1]
    assert (summary_breakout.status, summary_breakout.codes) == (
        "source_limited",
        ("facade.analysis.operation",),
    )
    for ingress in ("from_unit_summary", "from_unit_panel"):
        triggered = resolve_cell(
            rows["cuped-linear-interaction-triggered_population"].bindings[ingress]
        )[0][1]
        assert (triggered.status, triggered.codes) == (
            "source_limited",
            ("facade.analysis.trigger_unsupported",),
        )
    panel_cluster = resolve_cell(
        rows["cuped-linear-interaction-clustering"].bindings["from_unit_panel"]
    )[0][1]
    assert (panel_cluster.status, panel_cluster.codes) == (
        "source_limited",
        ("source.frame_panel.cluster_grain",),
    )


def test_linear_cuped_matrix_dispositions_cover_six_ingresses():
    from tests.parity_harness.matrix import (
        LINEAR_CUPED_DISPOSITIONS,
        LINEAR_CUPED_SOURCE_REFUSALS,
    )

    assert set(LINEAR_CUPED_DISPOSITIONS) == set(INGRESSES)
    assert {
        ingress for ingress, status in LINEAR_CUPED_DISPOSITIONS.items() if status == "supported"
    } == {"from_unit_summary", "from_unit_panel", "from_definitions", "from_unit_day_artifact"}
    assert {
        ingress: LINEAR_CUPED_DISPOSITIONS[ingress] for ingress in LINEAR_CUPED_SOURCE_REFUSALS
    } == {"from_moments": "source_limited", "from_switchback_panel": "source_limited"}
