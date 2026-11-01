"""Regression coverage for the artifact replacement of warehouse adoption."""

from __future__ import annotations

import pytest

from increment import from_unit_day_artifact
from increment.artifacts import publish_unit_day_artifact
from increment.estimators import CapabilityError
from tests.analysis_factory import lift_rows, view_readout
from tests.test_unit_day_artifact_facade import _native


def test_randomized_and_capability_error_canonical_imports() -> None:
    import increment
    import increment.errors as errors

    assert increment.Randomized(control_group="control").control_group == "control"
    assert CapabilityError is errors.CapabilityError


def test_frame_sql_adapter_remains_distinct_from_warehouse_adoption() -> None:
    from increment.query.source import SqlPanelSource

    assert callable(SqlPanelSource.from_frame_via_memtable)
    assert not callable(getattr(SqlPanelSource, "from_table", None))


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_open_is_the_adopted_panel_replacement() -> None:
    _connection, (native_experiment, native_resolver), context, store = _native()
    reference = publish_unit_day_artifact(native_experiment, native_resolver, store)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, reference, expected_context=context
    )
    from increment.estimation.engine import Method

    unadjusted = Method(name="unadjusted")
    try:
        native_result = lift_rows(
            view_readout(
                native_experiment,
                native_resolver,
                metrics=["purchase_rate"],
                decision_method=unadjusted,
            )
        )[0]
        adopted_result = lift_rows(
            view_readout(
                adopted_experiment,
                adopted_resolver,
                metrics=["purchase_rate"],
                decision_method=unadjusted,
            )
        )[0]
        assert adopted_result.metric == native_result.metric == "purchase_rate"
        assert adopted_result.group_id == native_result.group_id == "treatment"
        assert adopted_result.require_lift().value == pytest.approx(
            native_result.require_lift().value
        )
        assert adopted_result.require_lift().lb == pytest.approx(native_result.require_lift().lb)
        assert adopted_result.require_lift().ub == pytest.approx(native_result.require_lift().ub)
    finally:
        adopted_resolver.close()
        native_resolver.close()
