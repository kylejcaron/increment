"""Warehouse CUPED checks the parity matrix does not express.

Cross-source agreement of scalar and mixed CUPED is the parity matrix's job
(``tests/parity_harness/matrix.py`` cells and its dedicated QR rows). These tests keep
what one source owns: declaration refusals, the day-axis total identity, mixed-CUPED
support refusals shared with the local fit, breakouts that never transfer unit rows, and
the source content digest.
"""

from __future__ import annotations

import math
from typing import Any, cast

import ibis
import pytest
import yaml

from increment import Method, View, from_definitions, from_unit_day_artifact
from increment.artifacts import publish_unit_day_artifact
from increment.errors import DefinitionError
from increment.estimation.results import LiftEstimate
from increment.semantics.models import Definitions
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import view_readout
from tests.parity_harness.views import read
from tests.warehouse_cuped_cases import _open, definitions_yaml, event_rows, unit_rows


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


def _native_and_adopted(connection, definitions_path, schema_name, *, all_extensions=False):
    """The mixed-CUPED definitions pair and its published, re-adopted unit-day artifact."""
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load

    native = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name=schema_name)
    context = artifact_context(definitions, experiment, "error")
    extensions = (
        [entry.request for entry in unit_day_artifact_extension_catalog(context)]
        if all_extensions
        else []
    )
    ref = publish_unit_day_artifact(*native, store, extensions=extensions)
    return native, from_unit_day_artifact(store, ref, expected_context=context)


def _close_pairs(*pairs):
    for _, resolver in pairs:
        cast(Any, resolver).close()


@backend_fixture(scope="module")
def duck():
    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows(unit_rows()))
    return con


@pytest.mark.slow
def test_day_axis_ratio_cuped_final_as_of_equals_the_total(duck):
    """Cumulated through its last day, the as-of ratio CUPED lift is the whole-experiment lift."""
    cuped = Method(name="cuped", variance_reduction="cuped")
    plan = (
        "      secondaries:\n        - metric: rpo\n"
        "          decision_method: {name: unadjusted}\n"
        "          sensitivity_methods: [{name: cuped, variance_reduction: cuped}]\n"
    )
    experiment, resolver = _open(duck, "duckdb", "events", plan, artifact=False)
    try:
        (total_row,) = [r for r in view_readout(experiment, resolver) if r.method == "cuped"]
        asof = {
            (r.group_id, r.ds): r
            for r in view_readout(
                experiment,
                resolver,
                View(view="asof", grain="asof"),
                metrics=["rpo"],
                sensitivity_methods=[cuped],
            )
            if r.method == "cuped"
        }
    finally:
        resolver.close()
    assert isinstance(total_row, LiftEstimate)
    total = total_row.require_lift()
    final = asof[("treatment", max(ds for _, ds in asof))].require_lift()
    assert final.value == pytest.approx(total.value, abs=1e-12)
    assert final.lb == pytest.approx(total.lb, abs=1e-12)
    assert final.ub == pytest.approx(total.ub, abs=1e-12)


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


@pytest.mark.integration
def test_mixed_cuped_parity_detects_a_changed_ingress_covariate():
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.parity_harness.comparison import nested_close

    rows = matrix_data.linear_cuped_rows()
    summary = matrix_cases.linear_cuped_frame_case(rows, panel=False)
    panel = matrix_cases.linear_cuped_frame_case(rows, panel=True, corrupt=True)
    try:

        def statistical_outputs(bound):
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
                for row in read(bound)
            }

        assert not nested_close(statistical_outputs(summary), statistical_outputs(panel))
    finally:
        _close_pairs(summary, panel)


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
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    units = matrix_data.linear_cuped_rows(**rows)
    pairs = [
        matrix_cases.linear_cuped_frame_case(units, panel=False),
        matrix_cases.linear_cuped_frame_case(units, panel=True),
    ]
    con = ibis.duckdb.connect()
    con.create_table("mixed_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "mixed_cuped_refusal.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("mixed_events"))
    pairs.extend(_native_and_adopted(con, definitions_path, "mixed_cuped_refusal_artifacts"))
    try:
        expected_failures = None
        for bound in pairs:
            results = read(bound)
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
        _close_pairs(*pairs)
        con.disconnect()


@pytest.mark.slow
def test_mixed_cuped_breakout_rows_match_panel_definitions_and_artifact(tmp_path, monkeypatch):
    from increment.errors import CodedError
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
    native, adopted = _native_and_adopted(
        connection, definitions_path, "mixed_cuped_breakout_artifacts", all_extensions=True
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
            read(summary, "breakout")
        assert raised.value.code == "evidence.breakout_unavailable"

        results = [read(panel, "breakout"), read(native, "breakout"), read(adopted, "breakout")]
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
        _close_pairs(summary, panel, adopted, native)
        connection.disconnect()


@pytest.mark.integration
def test_warehouse_multicovariate_digest_tracks_covariate_only_changes(tmp_path):
    import copy

    from tests.analysis_factory import view_readout
    from tests.parity_harness import matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=24)
    events = mixed_cuped_event_rows(units)
    changed_events = copy.deepcopy(events)
    for row in changed_events:
        if row["pre_revenue"] is not None:
            row["pre_revenue"] = 2.5 * row["pre_revenue"] + 7.0

    connections = [ibis.duckdb.connect(), ibis.duckdb.connect()]
    resolvers = []
    try:
        for index, (connection, payload) in enumerate(
            zip(connections, (events, changed_events), strict=True)
        ):
            connection.create_table("mixed_events", obj=payload)
            definitions_path = tmp_path / f"warehouse-cuped-{index}.yaml"
            definitions_path.write_text(mixed_cuped_definitions_yaml("mixed_events"))
            resolvers.append(
                from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
            )

        digests = []
        for experiment, resolver in resolvers:
            rows = view_readout(experiment, resolver, metrics=["revenue"])
            assert rows.metadata is not None
            assert rows.metadata.source_content_digest is not None
            digests.append(rows.metadata.source_content_digest)
        assert digests[0] != digests[1]
    finally:
        for _, resolver in resolvers:
            cast(Any, resolver).close()
        for connection in connections:
            connection.disconnect()
