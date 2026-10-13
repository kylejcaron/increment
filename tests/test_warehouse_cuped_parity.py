"""Ratio CUPED and the adjusted sequential laws on the warehouse readers.

The dataframe path is the oracle: identical per-unit data through the
definitions reader and the unit-day artifact reader must produce the same
adjusted moments, the same interval and the same retained sequential state.
"""

from __future__ import annotations

import ibis
import pytest
import yaml

from increment.errors import DefinitionError
from increment.semantics.models import Definitions
from tests.warehouse_cuped_cases import (
    check_day_axis_parity,
    check_fixed_horizon_parity,
    check_sequential_parity,
    definitions_yaml,
    event_rows,
    unit_rows,
)


@pytest.fixture(scope="module")
def duck():
    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows(unit_rows()))
    return con


def test_ratio_cuped_fixed_horizon_matches_the_dataframe_oracle(duck, tmp_path):
    intervals = check_fixed_horizon_parity(duck, "duckdb", "events", tmp_path)
    plain, adjusted = intervals["duckdb"]
    assert adjusted.require_lift().ub - adjusted.require_lift().lb < 0.5 * (
        plain.require_lift().ub - plain.require_lift().lb
    )


@pytest.mark.integration
@pytest.mark.slow
def test_adjusted_sequential_laws_match_frame_capture(duck):
    widths = check_sequential_parity(duck, "duckdb", "events")
    for plain, adjusted in widths.values():
        assert adjusted.require_lift().ub - adjusted.require_lift().lb < 0.5 * (
            plain.require_lift().ub - plain.require_lift().lb
        )


def test_ratio_cuped_binding_without_a_pre_period_refuses_by_name():
    plan = (
        "      secondaries:\n        - metric: rpo\n"
        "          decision_method: {name: cuped, variance_reduction: cuped}\n"
    )
    with pytest.raises(DefinitionError) as raised:
        Definitions.model_validate(
            yaml.safe_load(definitions_yaml("duckdb", "events", n_pre_periods=0, plan=plan))
        )
    assert raised.value.code == "definition.experiment.metric_declares_cuped"


def test_day_axis_ratio_cuped_matches_total_and_both_readers_agree(duck):
    check_day_axis_parity(duck, "duckdb", "events")


def test_mixed_cuped_mean_ratio_rows_match_summary_and_panel_with_roles():
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close

    rows = matrix_data.linear_cuped_rows()
    summary = matrix_cases.linear_cuped_frame_case(rows, panel=False)
    panel = matrix_cases.linear_cuped_frame_case(rows, panel=True)
    try:
        left, right = summary.run(), panel.run()
        assert {(r.metric, r.group_id, r.method, r.method_role) for r in left} == {
            (r.metric, r.group_id, r.method, r.method_role) for r in right
        }
        assert {r.method_role for r in left} == {"decision"}
        assert {r.metric for r in left} == {"revenue", "rpo", "signups"}

        def public_rows(result):
            fields = []
            for row in result:
                dumped = row.model_dump(mode="json")
                dumped.pop("source_snapshot_id", None)
                dumped.pop("source_id", None)
                fields.append(dumped)
            return sorted(fields, key=lambda row: (row["metric"], row["group_id"], row["method"]))

        assert nested_close(public_rows(left), public_rows(right))
        assert left.metadata.scope.cells and right.metadata.scope.cells
        assert left.metadata.scope.snapshot_id and right.metadata.scope.snapshot_id
    finally:
        summary.close()
        panel.close()


def test_mixed_cuped_parity_detects_a_changed_ingress_covariate():
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close

    rows = matrix_data.linear_cuped_rows()
    summary = matrix_cases.linear_cuped_frame_case(rows, panel=False)
    panel = matrix_cases.linear_cuped_frame_case(rows, panel=True, corrupt=True)
    try:
        left = next(
            row for row in summary.run() if row.metric == "revenue" and row.group_id == "treatment"
        )
        right = next(
            row for row in panel.run() if row.metric == "revenue" and row.group_id == "treatment"
        )
        assert not nested_close(left.model_dump(mode="json"), right.model_dump(mode="json"))
    finally:
        summary.close()
        panel.close()


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ({"n_per_arm": 20}, "estimation.cuped_linear.support_region"),
        (
            {"n_per_arm": 60, "level_support_failure": True},
            "estimation.cuped_linear.level_support",
        ),
    ],
)
def test_mixed_cuped_support_refusals_are_exact(rows, code):
    from tests.parity_harness import matrix_cases, matrix_data

    units = matrix_data.linear_cuped_rows(**rows)
    for panel in (False, True):
        analysis = matrix_cases.linear_cuped_frame_case(units, panel=panel)
        try:
            results = analysis.run()
            assert any(row.sampling_reason_code == code for row in results), [
                (row.metric, row.sampling_reason_code, row.sampling_reason_context)
                for row in results
            ]
        finally:
            analysis.close()


def test_mixed_cuped_mean_ratio_rows_match_all_supported_ingresses(tmp_path):
    from increment import Analysis
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    units = matrix_data.linear_cuped_rows()
    connection = ibis.duckdb.connect()
    connection.create_table("mixed_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "mixed_cuped.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("mixed_events"))
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name="mixed_cuped_artifacts")
    expected_context = artifact_context(definitions, experiment, "error")
    artifact_ref = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(
        store, artifact_ref, expected_context=expected_context
    )
    summary = matrix_cases.linear_cuped_frame_case(units, panel=False)
    panel = matrix_cases.linear_cuped_frame_case(units, panel=True)
    try:
        results = [summary.run(), panel.run(), native.run(), adopted.run()]

        from tests.parity_harness.runner import (
            _normalize,
            _row_identity,
            _validate_source_snapshot_ids,
        )

        names = ("summary", "panel", "definitions", "artifact")
        normalized = []
        for name, result in zip(names, results, strict=True):
            family_for_row = _validate_source_snapshot_ids("mixed-cuped", name, result)
            normalized.append(_normalize(result, family_for_row=family_for_row))
        for name, payload in zip(names[1:], normalized[1:], strict=True):
            assert nested_close(normalized[0], payload), name

        expected_keys = {_row_identity(row) for row in results[0]}
        assert {row.metric for row in results[0]} == {"revenue", "rpo", "signups"}
        for result in results[1:]:
            assert {_row_identity(row) for row in result} == expected_keys

        def retained_signature(result):
            metadata = result.metadata
            assert metadata is not None
            scope = metadata.scope
            signature = set()
            for row in result:
                cell = next(
                    cell
                    for cell in scope.cells
                    if (
                        cell.metric,
                        cell.group_id,
                        cell.method,
                        cell.method_role,
                        cell.analysis_population,
                    )
                    == (
                        row.metric,
                        row.group_id,
                        row.method,
                        row.method_role,
                        row.analysis_population,
                    )
                )
                record = next(
                    record
                    for record in metadata.cells
                    if record.cell == cell and record.source_snapshot_id == row.source_snapshot_id
                )
                assert record.sampling is not None
                assert record.sampling.available == row.sampling_available
                signature.add(
                    (
                        cell.kind,
                        cell.metric,
                        cell.group_id,
                        cell.method,
                        cell.method_role,
                        cell.analysis_population,
                        record.sampling.available,
                        record.sampling.evidence is not None,
                    )
                )
            return frozenset(signature), tuple(scope.populations)

        expected_state = retained_signature(results[0])
        assert all(retained_signature(result) == expected_state for result in results[1:])
    finally:
        summary.close()
        panel.close()
        adopted.close()
        native.close()
