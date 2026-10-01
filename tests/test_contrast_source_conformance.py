"""Conformance checks for the fixed switchback contrast adapter."""

from __future__ import annotations

import pytest

from tests.source_conformance import switchback_source


def test_switchback_source_serves_each_declared_metric_and_diagnostics() -> None:
    source = switchback_source()
    metric = source.metrics[0]
    stats = source.contrast_stats(metric)
    assert stats.metric == metric.name
    assert stats.n_units >= 1
    report = source.diagnostic_report()
    assert report["schedule_complete"] is True
    assert report["n_units"] == stats.n_units
    source.close()


def test_switchback_absent_metric_is_a_stable_coded_refusal() -> None:
    from increment.errors import CapabilityError
    from increment.semantics.models import MeanMetric

    source = switchback_source()
    missing = MeanMetric(name="missing", entity="user", fact="missing")
    with pytest.raises(CapabilityError) as raised:
        source.contrast_stats(missing)
    assert raised.value.code == "source.frame.switchback.metric"
    assert raised.value.context["metric"] == "missing"
    source.close()


def test_switchback_result_serialization_and_to_frame() -> None:
    from tests.source_conformance import switchback_analysis

    analysis = switchback_analysis()
    results = analysis.run()
    assert results
    for result in results:
        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored == result
    frame = results.to_frame()
    assert len(frame) == len(results)
    columns = frame.column_names if hasattr(frame, "column_names") else list(frame.columns)
    assert {"metric", "estimate", "lb", "ub"} <= set(columns)
    analysis.close()
