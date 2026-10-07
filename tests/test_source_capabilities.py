"""Source operation refusals and runtime protocol checks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from increment._source_operations import PanelSQLOperation, SummarySqlOperation
from increment.analysis import Analysis
from increment.errors import CapabilityError
from increment.query._native_triggered import TriggeredPopulationSource
from increment.sources import (
    MomentSource,
    SourceOperation,
    require_operation,
)
from tests.test_native_core_operations import _assert_operation_refusal


def test_triggered_population_source_limits_grains_and_uses_triggered_counts() -> None:
    source = SimpleNamespace(
        capabilities=frozenset({"total", "daily", "asof"}),
        breakouts=(),
        shape=None,
        _validate_trigger_capability=lambda *, operation: None,
        triggered_counts=lambda: (
            "cluster",
            {"control": 2, "treatment": 3},
            {"control": 7, "treatment": 11},
        ),
    )

    triggered = TriggeredPopulationSource(source)

    assert triggered.capabilities == frozenset({"total"})
    assert triggered.unit_counts() == {"control": 7, "treatment": 11}
    assert triggered.cluster_counts() == {"control": 2, "treatment": 3}
    with pytest.raises(CapabilityError) as raised:
        triggered.sql()
    _assert_operation_refusal(
        raised.value,
        operation="sql",
        request={"grain": "total", "population": "triggered"},
        offered=("total",),
    )


def test_triggered_population_source_uses_primary_counts_for_unclustered_units() -> None:
    source = SimpleNamespace(
        capabilities=frozenset({"total", "daily", "asof"}),
        breakouts=(),
        shape=None,
        _validate_trigger_capability=lambda *, operation: None,
        triggered_counts=lambda: ("unit", {"control": 7, "treatment": 11}, {}),
    )

    assert TriggeredPopulationSource(source).unit_counts() == {
        "control": 7,
        "treatment": 11,
    }


class _OperationsDescriptorSource:
    """Arm source whose `operations` descriptor raises when it is read."""

    context = SimpleNamespace(study_id="descriptor", design=None)
    descriptor_reads = 0

    @property
    def operations(self) -> frozenset[SourceOperation]:
        type(self).descriptor_reads += 1
        raise AttributeError("operations descriptor failed")


def test_from_source_does_not_invoke_operations_descriptor() -> None:
    source = cast(MomentSource, _OperationsDescriptorSource())

    Analysis._from_source(source)

    assert _OperationsDescriptorSource.descriptor_reads == 0
    # The descriptor is genuinely armed: touching it raises and counts.
    with pytest.raises(AttributeError):
        _ = source.operations
    assert _OperationsDescriptorSource.descriptor_reads == 1


def test_require_operation_rejects_declaration_drift() -> None:
    from tests.source_conformance import sql_totals_source

    source: MomentSource = sql_totals_source()
    assert require_operation(source, "summary_sql", SummarySqlOperation) is source
    with pytest.raises(CapabilityError) as raised:
        require_operation(source, "panel_sql", PanelSQLOperation)
    assert raised.value.code == "source.operation.unsupported"
    assert raised.value.context == {
        "operation": "panel_sql",
        "source": "SqlPanelSource",
    }
