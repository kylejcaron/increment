import json
from pathlib import Path
from typing import Any, cast

import pytest

from increment import Scope, from_definitions, from_frame, from_switchback_frame
from tests.analysis_factory import view_readout


def test_envelope_codec_has_versioned_row_kind_and_component_state_tags():
    from increment.result_envelope import ResultCell

    schema = ResultCell.to_arrow([]).schema
    assert schema.field("schema_version").type.bit_width == 32
    assert "row_kind" in schema.names
    assert schema.field("sampling").type.names == [
        "status",
        "code",
        "reason",
        "context",
        "advisories",
        "construction",
        "kind",
        "state_reason",
        "supports",
    ]


def test_schema_one_golden_payload_decodes_as_lift_estimate():
    from increment.result_envelope import ResultCell

    fixture = (
        Path(__file__).parent / "goldens" / "decision_wire" / "result_envelope_v1.payload.json"
    )
    payload = json.loads(fixture.read_text())
    cell = ResultCell.from_payload(payload)
    assert cell.row_kind == "lift_estimate"
    assert cell.to_payload()["schema_version"] == 3
    assert cell.identity.key.metric == "orders"


def test_schema_one_arrow_golden_decodes_as_lift_estimate():

    import pyarrow as pa

    from increment.result_envelope import ResultCell

    fixture = (
        Path(__file__).parent / "goldens" / "decision_wire" / "result_envelope_v1.payload.json"
    )
    cell = ResultCell.from_payload(json.loads(fixture.read_text()))
    v2 = ResultCell.to_arrow([cell])
    old_names = (
        "schema_version",
        "identity",
        "label",
        "family",
        "source_snapshot_id",
        "provenance",
        "summary",
        "retained_row",
        "projection_fields",
        "sampling",
        "posterior",
        "sequential",
    )
    old_component = pa.struct(
        [
            v2.schema.field("sampling").type.field(name)
            for name in (
                "status",
                "code",
                "reason",
                "context",
                "advisories",
                "construction",
                "supports",
            )
        ]
    )
    schema = pa.schema(
        [
            pa.field(name, old_component)
            if name in {"sampling", "posterior", "sequential"}
            else v2.schema.field(name)
            for name in old_names
        ]
    )
    entry = v2.to_pylist()[0]
    entry = {name: entry[name] for name in old_names}
    entry["schema_version"] = 1
    for name in ("sampling", "posterior", "sequential"):
        entry[name] = {
            key: value for key, value in entry[name].items() if key not in {"kind", "state_reason"}
        }
    restored = ResultCell.from_arrow(pa.Table.from_pylist([entry], schema=schema))[0]
    assert restored.row_kind == "lift_estimate"


def test_schema_one_analysis_envelope_decodes_version_one_cells():
    from increment.result_envelope import AnalysisEnvelope, ResultCell

    fixture = (
        Path(__file__).parent / "goldens" / "decision_wire" / "result_envelope_v1.payload.json"
    )
    cell = ResultCell.from_payload(json.loads(fixture.read_text()))
    payload = {
        "schema_version": 1,
        "cells": [cell.to_payload() | {"schema_version": 1}],
        "metadata": None,
    }
    restored = AnalysisEnvelope.from_payload(payload)
    assert restored.cells[0].row_kind == "lift_estimate"


def test_row_kind_must_match_retained_payload_kind():
    import copy

    from increment.errors import CodedError
    from increment.result_envelope import ResultCell

    fixture = (
        Path(__file__).parent / "goldens" / "decision_wire" / "result_envelope_v1.payload.json"
    )
    payload = json.loads(fixture.read_text())
    payload.update(schema_version=2, row_kind="breakout")
    with pytest.raises(CodedError) as raised:
        ResultCell.from_payload(copy.deepcopy(payload))
    assert raised.value.code == "results.cell.schema_unsupported"


def test_unavailable_breakout_reason_survives_cell_and_table_round_trip():
    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.result_envelope import AnalysisEnvelope, ResultCell
    from increment.tables import readout_table

    row = BreakoutEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="US",
        source=None,
        lift=None,
        excluded="few_units",
    )
    (cell,) = BreakoutEstimates([row]).as_cells()
    restored = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
    assert restored.summary is not None
    assert restored.summary.reason == "few_units"
    html = readout_table(AnalysisEnvelope((restored,), None))._repr_html_()
    assert "few_units" in html


# All five warehouse-produced row kinds must survive detached codecs.
@pytest.mark.slow
@pytest.mark.integration
def test_breakout_and_day_axis_producers_round_trip_tagged_rows(seeded_con, seeded_defs):
    import gc

    from increment.declarations import CellKey
    from increment.result_envelope import AnalysisEnvelope, ResultCell

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        outputs = (
            ("breakout", view_readout(experiment, resolver, Scope(view="breakout", grain="total"))),
            (
                "daily_metric",
                view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True)),
            ),
            ("daily_lift", view_readout(experiment, resolver, Scope(view="daily", grain="daily"))),
            (
                "asof_metric",
                view_readout(experiment, resolver, Scope(view="asof", grain="asof", values=True)),
            ),
            ("asof_lift", view_readout(experiment, resolver, Scope(view="asof", grain="asof"))),
        )
        for row_kind, rows in outputs:
            assert rows
            cells = rows.as_cells()
            assert len(cells) == len(rows)
            for _row, cell in zip(rows, cells, strict=True):
                assert cell.row_kind == row_kind
                estimate = getattr(_row, "lift", None) or getattr(_row, "value", None)
                assert cell.summary is not None
                if estimate is None:
                    assert cell.summary.point is None
                    if cell.summary.lower is None and cell.summary.upper is None:
                        assert cell.summary.reason is not None
                else:
                    assert cell.summary.point == pytest.approx(estimate.value)
                    assert cell.summary.lower == estimate.lb
                    assert cell.summary.upper == estimate.ub
                if row_kind == "breakout":
                    assert cell.identity.scope.view == "breakout"
                    assert cell.identity.scope.segment
                elif row_kind.startswith("asof"):
                    assert cell.identity.scope.view == "asof"
                    assert cell.identity.scope.grain == "asof"
                elif row_kind.startswith("daily"):
                    assert cell.identity.scope.view == "daily"
                    assert cell.identity.scope.grain == "daily"
                if row_kind == "daily_metric":
                    assert cell.identity.key.value_scale == "absolute"
                    assert cell.summary.value_scale == "absolute"
                payload = json.loads(json.dumps(cell.to_payload()))
                if row_kind == "daily_lift":
                    assert rows.metadata is not None
                    expected_family = rows.metadata.family_of(
                        _row.source_snapshot_id, CellKey.from_row(_row)
                    )
                    assert cell.family == expected_family
                detached = ResultCell.from_payload(payload)
                assert detached.identity == cell.identity
                assert detached.row_payload == cell.row_payload
            expected_payloads = [cell.row_payload for cell in cells]
            expected_rows = len(cells)
            table = ResultCell.to_arrow(cells)
            del rows, cells
            gc.collect()
            restored = ResultCell.from_arrow(table)
            assert [cell.row_kind for cell in restored] == [row_kind] * expected_rows
            assert [cell.row_payload for cell in restored] == expected_payloads
            envelope = AnalysisEnvelope(tuple(restored), None)
            assert (
                AnalysisEnvelope.from_payload(json.loads(json.dumps(envelope.to_payload()))).cells
                == envelope.cells
            )
    finally:
        cast(Any, resolver).close()


@pytest.mark.integration
def test_reopened_daily_cells_use_dates_as_table_identity(seeded_con, seeded_defs):
    from increment.result_envelope import AnalysisEnvelope, ResultCell
    from increment.tables import readout_table

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        rows = view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True))
        cells = [
            cell
            for cell in rows.as_cells()
            if cell.identity.key.metric == "purchase_rate"
            and cell.identity.key.group_id == "control"
        ]
        assert len(cells) >= 2
        first, second = ResultCell.from_arrow(ResultCell.to_arrow(cells[:2]))
        assert first.identity.key.ds != second.identity.key.ds
        html = readout_table(AnalysisEnvelope((first, second), None))._repr_html_()
        assert str(first.identity.key.ds) in html
        assert str(second.identity.key.ds) in html
    finally:
        cast(Any, resolver).close()


# Both as-of row families retain completion policy across every collection codec.
@pytest.mark.slow
@pytest.mark.integration
def test_asof_lift_view_survives_collection_round_trips(seeded_con, seeded_defs):
    import copy
    import pickle

    from increment.estimation.readout_types import ReadoutResults

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        outputs = (
            (
                view_readout(
                    experiment,
                    resolver,
                    Scope(view="asof", grain="asof", values=True, completion_policy=True),
                ),
                "asof_metric",
            ),
            (
                view_readout(
                    experiment, resolver, Scope(view="asof", grain="asof", completion_policy=True)
                ),
                "asof_lift",
            ),
        )
        for rows, row_kind in outputs:
            constructed = type(rows)(
                rows,
                metadata=rows.metadata,
                source=rows.source,
                sequential_snapshot=rows.sequential_snapshot,
                envelope_view="asof",
                completion_policy=True,
            )
            transformed = (
                constructed,
                rows[:1],
                rows.filter(lambda row: True),
                copy.deepcopy(rows),
                pickle.loads(pickle.dumps(rows)),
                ReadoutResults.model_validate_json(rows.model_dump_json()),
            )
            for item in transformed:
                cell = item.as_cells()[0]
                assert cell.row_kind == row_kind
                assert cell.identity.scope.completion_policy is True
    finally:
        cast(Any, resolver).close()


@pytest.mark.integration
def test_daily_lift_cells_preserve_collection_family(seeded_con, seeded_defs):
    from increment.declarations import CellKey

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        rows = view_readout(experiment, resolver, Scope(view="daily", grain="daily"))
        assert rows and rows.metadata is not None
        row = next(
            row
            for row in rows
            if rows.metadata.family_of(row.source_snapshot_id, CellKey.from_row(row)) is not None
        )
        expected = rows.metadata.family_of(row.source_snapshot_id, CellKey.from_row(row))
        cell = next(
            cell for candidate, cell in zip(rows, rows.as_cells(), strict=True) if candidate is row
        )
        assert cell.family == expected
    finally:
        cast(Any, resolver).close()


def test_daily_collection_concat_refuses_incompatible_view_scope():
    from increment.breakout.estimates import DailyMetricValues
    from increment.errors import CodedError

    daily = DailyMetricValues([], envelope_view="daily")
    asof = DailyMetricValues([], envelope_view="asof")
    with pytest.raises(CodedError) as raised:
        daily.concat(asof)
    assert raised.value.code == "readout.collection.concat_metadata_mismatch"


def test_restored_set_only_breakout_uses_public_confidence_projection():
    import pandas as pd

    from increment.result_envelope import AnalysisEnvelope, ResultCell, render

    rows = pd.DataFrame(
        [
            {"unit": f"{arm}-{index}", "arm": arm, "segment": "A", "value": float(success)}
            for arm, outcomes in (("control", [0] * 200), ("treatment", [0] * 100 + [1] * 100))
            for index, success in enumerate(outcomes)
        ]
    )
    experiment, resolver = from_frame(
        rows,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "conversion"},
        breakouts=("segment",),
    )
    try:
        results = view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
        (row,) = results
        assert row.lift is None and row.binomial_set is not None
        (cell,) = results.as_cells()
        cell = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
        assert cell.summary is not None
        assert cell.summary.lower == pytest.approx(6.0016, abs=0.0001)
        assert cell.summary.lower == row.binomial_set.lower
        assert cell.summary.alpha == row.binomial_set.alpha
        assert cell.summary.geometry == row.binomial_set.geometry
        assert "6.0016" in render(AnalysisEnvelope((cell,), None), "brief")
    finally:
        resolver.close()


@pytest.mark.integration
def test_mixed_lift_and_report_cells_render_and_project_to_table(seeded_con, seeded_defs):
    import json

    from increment import Method
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from increment.tables import readout_table

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        lift_rows = view_readout(experiment, resolver, decision_method=Method(name="unadjusted"))
        breakout_rows = view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
        cells = (*lift_rows.as_cells(), *breakout_rows.as_cells())
        decoded = tuple(
            ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))) for cell in cells
        )
        envelope = AnalysisEnvelope(decoded, None)
        breakout = next(
            cell
            for cell in decoded
            if cell.row_kind == "breakout"
            and cell.row_payload is not None
            and cell.row_payload["data"]["lift"] is not None
        )
        assert breakout.row_payload is not None
        assert breakout.summary is not None
        assert breakout.summary.point == pytest.approx(
            breakout.row_payload["data"]["lift"]["value"]
        )
        assert repr(breakout.summary.point) in render(envelope, "brief")
        assert "breakout" in render(envelope, "brief")
        html = readout_table(envelope, nest_by="segment")._repr_html_()
        assert "d7_retention" in html
    finally:
        cast(Any, resolver).close()


@pytest.mark.integration
def test_filtered_breakout_cell_retains_full_family_scope(seeded_con, seeded_defs):
    from increment.declarations import CellKey
    from increment.result_envelope import ResultCell

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        rows = view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
        row = rows[0]
        assert rows.metadata is not None
        family = rows.metadata.family_of(row.source_snapshot_id, CellKey.from_row(row))
        assert family is not None and len(family.members) > 1
        filtered = rows.filter(
            lambda item: (
                item.metric == row.metric
                and item.dimension == row.dimension
                and item.dimension_value == row.dimension_value
            )
        )
        cells = filtered.as_cells()
        assert cells[0].family == family
        restored = ResultCell.from_arrow(ResultCell.to_arrow(cells))
        assert restored[0].family == family
    finally:
        cast(Any, resolver).close()


@pytest.mark.integration
def test_switchback_contrast_producer_round_trips_descriptive_state():
    import pandas as pd

    from increment.result_envelope import ResultCell
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    rows = [
        {
            "unit": unit,
            "cycle": cycle,
            "period": period,
            "step": step,
            "group": group,
            "value": float(cycle + period + step + (group == "treatment")),
        }
        for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control")))
        for cycle in range(2)
        for period, group in enumerate(order)
        for step in range(2)
    ]
    experiment, resolver = from_switchback_frame(
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
    )
    try:
        results = view_readout(experiment, resolver)
        cells = results.as_cells()
        assert cells
        for cell in cells:
            assert cell.summary is not None
            assert cell.row_payload is not None
            assert cell.summary.point == pytest.approx(
                cell.row_payload["data"]["estimate"]["value"]
            )
            assert cell.identity.key.value_scale == "absolute"
            assert cell.summary.value_scale == "absolute"
            detached = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
            assert detached.row_payload == cell.row_payload
        restored = ResultCell.from_arrow(ResultCell.to_arrow(cells))
        assert [item.row_payload for item in restored] == [item.row_payload for item in cells]
    finally:
        resolver.close()
