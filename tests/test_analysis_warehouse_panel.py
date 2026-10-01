"""Regression coverage for the artifact replacement of warehouse adoption."""

from __future__ import annotations

import pytest

from increment import Analysis
from increment.errors import CapabilityError
from tests.analysis_factory import lift_rows
from tests.test_unit_day_artifact_facade import _native


def test_randomized_and_capability_error_exported() -> None:
    import increment

    assert increment.Randomized(control_group="control").control_group == "control"
    assert increment.CapabilityError is CapabilityError


def test_warehouse_panel_adoption_path_is_removed_without_a_compatibility_shim() -> None:
    assert not hasattr(Analysis, "from_warehouse_panel")


def test_frame_sql_adapter_remains_distinct_from_warehouse_adoption() -> None:
    from increment.query.source import SqlPanelSource

    assert callable(SqlPanelSource.from_frame_via_memtable)
    assert not callable(getattr(SqlPanelSource, "from_table", None))


def test_artifact_open_is_the_adopted_panel_replacement() -> None:
    _connection, native, context, store = _native()
    reference = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    try:
        native_result = lift_rows(native.run(metrics=["purchase_rate"]))[0]
        adopted_result = lift_rows(adopted.run(metrics=["purchase_rate"]))[0]
        assert adopted_result.metric == native_result.metric == "purchase_rate"
        assert adopted_result.group_id == native_result.group_id == "treatment"
        assert adopted_result.require_lift().value == pytest.approx(
            native_result.require_lift().value
        )
        assert adopted_result.require_lift().lb == pytest.approx(native_result.require_lift().lb)
        assert adopted_result.require_lift().ub == pytest.approx(native_result.require_lift().ub)
    finally:
        adopted.close()
        native.close()
