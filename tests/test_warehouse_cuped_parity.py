"""Ratio CUPED and the adjusted sequential laws on the warehouse readers.

The dataframe path is the oracle: identical per-unit data through the
definitions reader and the unit-day artifact reader must produce the same
adjusted moments, the same interval and the same retained sequential state.
"""

from __future__ import annotations

import math

import ibis
import pytest
import yaml

from increment.errors import DefinitionError
from increment.estimation.results import LiftEstimate
from increment.semantics.models import Definitions
from tests.warehouse_cuped_cases import (
    check_day_axis_parity,
    check_fixed_horizon_parity,
    check_sequential_parity,
    definitions_yaml,
    event_rows,
    unit_rows,
)


def _source_scope_signature(metadata):
    scope = metadata.scope
    sources = tuple(
        sorted(
            (
                record.model_dump(mode="json", exclude={"source_snapshot_id"})
                for record in scope.by_source.values()
            ),
            key=repr,
        )
    )
    families = tuple(
        sorted(
            (
                family.model_dump(mode="json", exclude={"source_snapshot_id", "family_id"})
                for family in scope.families
            ),
            key=repr,
        )
    )
    return sources, families


def _retained_mixed_cuped_signature(result):
    metadata = result.metadata
    assert metadata is not None
    scope = metadata.scope
    signature = {}
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
        assert (
            record.posterior.available if record.posterior is not None else None
        ) == row.posterior_available
        identity = (
            cell.kind,
            cell.metric,
            cell.group_id,
            cell.method,
            cell.method_role,
            cell.analysis_population,
        )
        signature[identity] = {
            "failure": record.failure.model_dump(mode="json")
            if record.failure is not None
            else None,
            "sampling": record.sampling.model_dump(mode="json"),
            "posterior": (
                record.posterior.model_dump(mode="json") if record.posterior is not None else None
            ),
            "advisories": [advisory.model_dump(mode="json") for advisory in record.advisories],
        }
    cells = tuple(
        sorted(
            (
                tuple(cell.model_dump(mode="json", exclude={"source"}).items())
                for cell in scope.cells
            ),
            key=repr,
        )
    )
    decisions = tuple(
        sorted(
            (
                tuple(cell.model_dump(mode="json", exclude={"source"}).items())
                for cell in scope.decision_cells
            ),
            key=repr,
        )
    )
    return signature, cells, decisions, tuple(scope.populations)


def _retained_breakout_signature(result):
    metadata = result.metadata
    assert metadata is not None
    scope = metadata.scope

    def cell_dump(cell):
        return cell.model_dump(mode="json", exclude={"source"})

    families = tuple(
        sorted(
            (
                {
                    "analysis_population": family.analysis_population,
                    "view": family.view,
                    "dimension": family.dimension,
                    "name": family.name,
                    "family": (
                        family.family.model_dump(mode="json") if family.family is not None else None
                    ),
                    "members": tuple(cell_dump(member) for member in family.members),
                    "complete": family.complete,
                }
                for family in scope.families
            ),
            key=repr,
        )
    )
    records = tuple(
        sorted(
            (
                {
                    "cell": cell_dump(record.cell),
                    "failure": (
                        record.failure.model_dump(mode="json")
                        if record.failure is not None
                        else None
                    ),
                    "sampling": (
                        record.sampling.model_dump(mode="json")
                        if record.sampling is not None
                        else None
                    ),
                    "posterior": (
                        record.posterior.model_dump(mode="json")
                        if record.posterior is not None
                        else None
                    ),
                    "advisories": [
                        advisory.model_dump(mode="json") for advisory in record.advisories
                    ],
                }
                for record in metadata.cells
            ),
            key=repr,
        )
    )
    source_scopes = tuple(
        sorted(
            (
                {
                    "cells": tuple(cell_dump(cell) for cell in source.cells),
                    "decision_cells": tuple(cell_dump(cell) for cell in source.decision_cells),
                    "rosters": tuple(roster.model_dump(mode="json") for roster in source.rosters),
                    "decision_complete_by_population": dict(source.decision_complete_by_population),
                    "integrity": tuple(
                        integrity.model_dump(mode="json") for integrity in source.integrity
                    ),
                }
                for source in scope.by_source.values()
            ),
            key=repr,
        )
    )
    return (
        tuple(cell_dump(cell) for cell in scope.cells),
        tuple(cell_dump(cell) for cell in scope.decision_cells),
        tuple(scope.populations),
        families,
        records,
        source_scopes,
    )


def _assert_usable_cuped_rows(results):
    for result in results:
        for row in result:
            assert row.sampling_available is True, (
                row.metric,
                row.group_id,
                row.method_role,
                row.sampling_reason_code,
            )
            assert row.lift is not None
            assert all(
                math.isfinite(value)
                for value in (
                    row.lift.value,
                    row.lift.log_se,
                    row.lift.lb,
                    row.lift.ub,
                    row.dof,
                )
            )


@pytest.fixture(scope="module")
def duck():
    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows(unit_rows()))
    return con


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.integration
def test_mixed_cuped_parity_detects_a_changed_ingress_covariate():
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close

    rows = matrix_data.linear_cuped_rows()
    summary = matrix_cases.linear_cuped_frame_case(rows, panel=False)
    panel = matrix_cases.linear_cuped_frame_case(rows, panel=True, corrupt=True)
    try:

        def statistical_outputs(analysis):
            return {
                (row.metric, row.group_id, row.method, row.method_role): {
                    "estimate": row.lift.value if row.lift is not None else None,
                    "se": row.lift.log_se if row.lift is not None else None,
                    "dof": row.dof,
                    "interval": ((row.lift.lb, row.lift.ub) if row.lift is not None else None),
                    "sampling_available": row.sampling_available,
                    "posterior_available": row.posterior_available,
                    "failure": (row.sampling_reason_code, row.sampling_reason_context),
                }
                for row in analysis.run()
            }

        assert not nested_close(statistical_outputs(summary), statistical_outputs(panel))
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
@pytest.mark.slow
def test_mixed_cuped_support_refusals_are_exact(rows, code, tmp_path):
    from increment import Analysis
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    units = matrix_data.linear_cuped_rows(**rows)
    analyses = [
        matrix_cases.linear_cuped_frame_case(units, panel=False),
        matrix_cases.linear_cuped_frame_case(units, panel=True),
    ]
    con = ibis.duckdb.connect()
    con.create_table("mixed_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "mixed_cuped_refusal.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("mixed_events"))
    native = Analysis.from_definitions("mixed_cuped", definitions_path, con)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(con, schema_name="mixed_cuped_refusal_artifacts")
    context = artifact_context(definitions, experiment, "error")
    ref = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    analyses.extend((native, adopted))
    try:
        expected_failures = None
        for analysis in analyses:
            results = analysis.run()
            failures = tuple(
                sorted(
                    (
                        (
                            row.metric,
                            row.group_id,
                            row.method,
                            row.method_role,
                            row.analysis_population,
                            row.sampling_reason_code,
                            row.sampling_reason_context,
                        )
                        for row in results
                        if isinstance(row, LiftEstimate) and row.sampling_reason_code is not None
                    ),
                    key=repr,
                )
            )
            assert any(failure[5] == code for failure in failures), failures
            if expected_failures is None:
                expected_failures = failures
            else:
                assert failures == expected_failures
    finally:
        for analysis in analyses:
            analysis.close()
        con.disconnect()


@pytest.mark.slow
def test_mixed_cuped_mean_ratio_conversion_rows_match_all_supported_ingresses(tmp_path):
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
        _assert_usable_cuped_rows(results)

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

        expected_source_scope = _source_scope_signature(results[0].metadata)
        for name, result in zip(names[1:], results[1:], strict=True):
            assert nested_close(expected_source_scope, _source_scope_signature(result.metadata)), (
                name
            )
        assert {row.metric for row in results[0]} == {"revenue", "rpo", "signups"}
        for result in results[1:]:
            assert {_row_identity(row) for row in result} == expected_keys

        expected_state = _retained_mixed_cuped_signature(results[0])
        for name, result in zip(names[1:], results[1:], strict=True):
            assert nested_close(expected_state, _retained_mixed_cuped_signature(result)), name

    finally:
        summary.close()
        panel.close()
        adopted.close()
        native.close()


@pytest.mark.slow
def test_scalar_cuped_composes_with_mixed_cuped_on_all_four_ingresses(tmp_path):
    from increment import Analysis
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows()
    for row in units:
        if row["pre_revenue"] is None:
            row["pre_revenue"] = 12.0
    connection = ibis.duckdb.connect()
    connection.create_table("mixed_scalar_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "mixed_cuped_scalar.yaml"
    definitions_path.write_text(
        mixed_cuped_definitions_yaml(
            "mixed_scalar_events",
            scalar_cuped=True,
            n_pre_periods=30,
        )
    )
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name="mixed_scalar_artifacts")
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    artifact_ref = native.publish_unit_day_artifact(store, extensions=extensions)
    adopted = Analysis.from_unit_day_artifact(store, artifact_ref, expected_context=context)
    summary = matrix_cases.linear_cuped_frame_case(units, panel=False, scalar_cuped=True)
    panel = matrix_cases.linear_cuped_frame_case(units, panel=True, scalar_cuped=True)
    try:
        results = [summary.run(), panel.run(), native.run(), adopted.run()]
        for result in results:
            adjusted = [
                row
                for row in result
                if isinstance(row, LiftEstimate)
                and row.metric == "revenue"
                and row.method in {"cuped", "linear"}
            ]
            assert {row.method for row in adjusted} == {"cuped", "linear"}
            for row in adjusted:
                assert row.sampling_available is True
                assert row.lift is not None
                assert row.lift.value is not None and math.isfinite(row.lift.value)
    finally:
        summary.close()
        panel.close()
        adopted.close()
        native.close()
        connection.disconnect()


@pytest.mark.slow
def test_fixed_value_winsorization_with_both_bounds_matches_all_supported_ingresses(
    tmp_path,
):
    from increment import Analysis
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close
    from tests.parity_harness.runner import _normalize, _validate_source_snapshot_ids
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows()
    assert min(row["revenue"] for row in units) < 7.0
    assert max(row["revenue"] for row in units) > 11.0
    connection = ibis.duckdb.connect()
    connection.create_table("mixed_winsor_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "mixed_cuped_winsor.yaml"
    definitions_path.write_text(
        mixed_cuped_definitions_yaml("mixed_winsor_events", fixed_winsorization=True)
    )
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name="mixed_winsor_artifacts")
    context = artifact_context(definitions, experiment, "error")
    artifact_ref = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, artifact_ref, expected_context=context)
    summary = matrix_cases.linear_cuped_frame_case(units, panel=False, fixed_winsorization=True)
    panel = matrix_cases.linear_cuped_frame_case(units, panel=True, fixed_winsorization=True)
    try:
        results = [
            summary.run(metrics=["revenue"]),
            panel.run(metrics=["revenue"]),
            native.run(metrics=["revenue"]),
            adopted.run(metrics=["revenue"]),
        ]
        names = ("summary", "panel", "definitions", "artifact")
        normalized = []
        for name, result in zip(names, results, strict=True):
            family_for_row = _validate_source_snapshot_ids("mixed-cuped-winsor", name, result)
            normalized.append(_normalize(result, family_for_row=family_for_row))
            _assert_usable_cuped_rows([result])
        for payload in normalized[1:]:
            assert nested_close(normalized[0], payload)
        retained = _retained_mixed_cuped_signature(results[0])
        source_scope = _source_scope_signature(results[0].metadata)
        for result in results[1:]:
            assert nested_close(retained, _retained_mixed_cuped_signature(result))
            assert nested_close(source_scope, _source_scope_signature(result.metadata))
    finally:
        summary.close()
        panel.close()
        adopted.close()
        native.close()
        connection.disconnect()


@pytest.mark.slow
def test_mixed_cuped_breakout_rows_match_panel_definitions_and_artifact(tmp_path, monkeypatch):
    from increment import Analysis
    from increment.errors import CodedError
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=120)
    connection = ibis.duckdb.connect()
    connection.create_table("mixed_breakout_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "mixed_cuped_breakout.yaml"
    definitions_path.write_text(
        mixed_cuped_definitions_yaml("mixed_breakout_events", breakout=True)
    )
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name="mixed_cuped_breakout_artifacts")
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    artifact_ref = native.publish_unit_day_artifact(store, extensions=extensions)
    adopted = Analysis.from_unit_day_artifact(
        store,
        artifact_ref,
        expected_context=context,
    )
    summary = matrix_cases.linear_cuped_frame_case(units, panel=False)
    panel = matrix_cases.linear_cuped_frame_case(units, panel=True, breakout=True)
    from increment.query import source as artifact_source
    from increment.query.native_source import DefinitionsMomentSource
    from tests.test_warehouse_cuped_qr import _guard_no_unit_keyed_transfers

    def forbidden_unit_transfer(*args, **kwargs):
        pytest.fail("warehouse mixed-CUPED breakout used a per-unit frame")

    monkeypatch.setattr(DefinitionsMomentSource, "unit_frame", forbidden_unit_transfer)
    monkeypatch.setattr(
        artifact_source._ArtifactFacadeSource, "unit_frame", forbidden_unit_transfer
    )
    _guard_no_unit_keyed_transfers(monkeypatch)
    try:
        with pytest.raises(CodedError) as raised:
            summary.run_breakout()
        assert raised.value.code == "facade.analysis.operation"

        results = [panel.run_breakout(), native.run_breakout(), adopted.run_breakout()]
        expected_keys = {
            (metric, region, arm, "linear", "decision")
            for metric in ("revenue", "rpo", "signups")
            for region in ("east", "west")
            for arm in ("treatment",)
        }
        for rows in results:
            assert {
                (
                    row.metric,
                    row.dimension_value,
                    row.group_id,
                    row.method,
                    row.method_role,
                )
                for row in rows
            } == expected_keys
            for row in rows:
                assert row.dimension == "region"
                assert row.sampling_available is True
                assert row.lift is not None
                assert all(
                    value is not None and math.isfinite(value)
                    for value in (
                        row.lift.value,
                        row.lift.log_se,
                        row.lift.lb,
                        row.lift.ub,
                        row.dof,
                    )
                )

        from tests.parity_harness.runner import _normalize, _validate_source_snapshot_ids

        names = ("panel", "definitions", "artifact")
        normalized = []
        for name, rows in zip(names, results, strict=True):
            family_for_row = _validate_source_snapshot_ids("mixed-cuped-breakout", name, rows)
            normalized.append(_normalize(rows, family_for_row=family_for_row))
        expected_rows = normalized[0]
        expected_scope = _retained_breakout_signature(results[0])
        for payload, rows in zip(normalized[1:], results[1:], strict=True):
            assert nested_close(expected_rows, payload)
            assert nested_close(expected_scope, _retained_breakout_signature(rows))
    finally:
        summary.close()
        panel.close()
        adopted.close()
        native.close()
        connection.disconnect()
