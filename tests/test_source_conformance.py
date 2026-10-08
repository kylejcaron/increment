"""Generated conformance checks for arm evidence adapters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.source_conformance import arm_adapters, definitions_source, public_result_factories


def _expect_code(call: Callable[[], Any], code: str) -> None:
    from increment.errors import CapabilityError
    from increment.query.artifact_contract import ArtifactContractError

    with pytest.raises((CapabilityError, ArtifactContractError)) as raised:
        call()
    assert raised.value.code == code


def _adapter_cases(tmp_path: Path):
    return arm_adapters(tmp_path)


_ADAPTER_NAMES = tuple(adapter.name for adapter in arm_adapters(Path("/tmp")))
_EXPECTED_ADAPTER_NAMES = frozenset(
    {
        "definitions",
        "frame_summary",
        "frame_panel",
        "moments",
        "moments_with_counts",
        "breakout_moments",
        "sql_totals",
        "unit_day_artifact",
    }
)


@pytest.mark.parametrize("adapter_index", range(len(_ADAPTER_NAMES)), ids=_ADAPTER_NAMES)
def test_each_adapter_advertises_a_complete_moment_source_contract(
    tmp_path: Path, adapter_index: int
) -> None:
    adapters = _adapter_cases(tmp_path)
    assert len(adapters) == len(_ADAPTER_NAMES)
    assert {adapter.name for adapter in adapters} == _EXPECTED_ADAPTER_NAMES
    adapter = adapters[adapter_index]
    source, metric = adapter.build()
    try:
        assert isinstance(source.capabilities, frozenset)
        assert isinstance(source.operations, frozenset)
        assert source.context.metrics
        for grain in source.capabilities:
            by = [source.breakouts[0]] if adapter.name == "breakout_moments" else ()
            rows = source.moments(metric, grain=grain, by=by)
            assert rows, f"{adapter.name} returned no {grain} evidence"
        if adapter.name == "unit_day_artifact":
            from increment.semantics.models import Breakout

            assert {extension.kind for extension in source.manifest.extensions} >= {
                "breakout_dimension",
                "factor_dimension",
                "cuped_preperiod",
                "assignment_counts",
                "trigger_population",
                "site_volume",
            }
            assert {
                "moments_source",
                "day_source",
                "export_moments",
                "breakout_source",
                "breakout_sources",
                "breakout_summaries",
                "factor_summaries",
                "triggered_counts",
                "triggered_source",
                "sitewide_evidence",
            } <= source.operations
            assert source.moments_source(metrics=[metric]) is source
            assert source.day_source(metrics=[metric]) is source
            source.export_moments(tmp_path / "artifact-moments.parquet")
            breakout = Breakout(property="country", source="event_log")
            assert source.breakout_source(breakout, metrics=[metric]).moments(
                metric, by=["country"]
            )
            assert source.breakout_sources([breakout], metrics=[metric])
            assert source.breakout_summaries(metrics=[metric])
            assert source.factor_summaries(metrics=[metric])
            assert source.unit_counts()
            assert source.triggered_counts()
            assert source.triggered_source().moments(metric)
            assert source.sitewide_evidence(metric).site_total > 0
    finally:
        source.close()


@pytest.mark.parametrize("adapter_index", range(len(_ADAPTER_NAMES)), ids=_ADAPTER_NAMES)
def test_absent_grains_and_extensions_are_stable_coded_refusals(
    tmp_path: Path, adapter_index: int
) -> None:
    adapter = _adapter_cases(tmp_path)[adapter_index]
    source, metric = adapter.build()
    try:
        if adapter.name == "unit_day_artifact":
            _expect_code(
                lambda: source.moments(metric, by=["missing"]),
                "artifact.extension.missing",
            )
            _expect_code(lambda: source.cluster_counts(), "artifact.extension.missing")
            _expect_code(
                lambda: source.unit_frame(metric, covariates=["missing"]),
                "artifact.extension.missing",
            )
            return
        if adapter.name == "frame_panel":
            _expect_code(
                lambda: source.moments(metric, grain="invalid"),
                "source.frame_panel.grain",
            )
            _expect_code(
                lambda: source.moments(metric, by=["missing"]), "source.frame.undeclared_breakout"
            )
        elif adapter.name == "definitions":
            _expect_code(lambda: source.moments(metric, grain="invalid"), "source.native.grain")
            _expect_code(lambda: source.moments(metric, by=["country"]), "source.native.operation")
        elif adapter.name == "breakout_moments":
            _expect_code(lambda: source.moments(metric, grain="daily"), "source.breakout.grain")
            _expect_code(lambda: source.moments(metric, by=["other"]), "source.breakout.dimension")
        elif adapter.name == "sql_totals":
            _expect_code(lambda: source.moments(metric, grain="daily"), "source.sql.grain")
            _expect_code(lambda: source.moments(metric, by=["missing"]), "source.sql.operation")
        elif adapter.name == "frame_summary":
            _expect_code(lambda: source.moments(metric, grain="daily"), "source.frame.grain")
            _expect_code(
                lambda: source.moments(metric, by=["missing"]), "source.frame.breakouts_unsupported"
            )
        elif adapter.name in {"moments", "moments_with_counts"}:
            _expect_code(lambda: source.moments(metric, grain="daily"), "source.moments.grain")
            _expect_code(lambda: source.moments(metric, by=["missing"]), "source.moments.breakout")

        if adapter.name == "frame_summary":
            _expect_code(lambda: source.sql(), "source.frame.sql_unsupported")
        elif adapter.name == "frame_panel":
            _expect_code(lambda: source.sql(), "source.frame.sql_unsupported")
        elif adapter.name == "moments":
            _expect_code(lambda: source.unit_frame(metric), "source.moments.unit_grain")
            _expect_code(lambda: source.unit_counts(), "source.moments.assignment_counts")
            _expect_code(lambda: source.sql(), "source.moments.sql")
        elif adapter.name == "moments_with_counts":
            _expect_code(lambda: source.unit_frame(metric), "source.moments.unit_grain")
            assert source.unit_counts() == {"control": 2, "treatment": 2}
            _expect_code(lambda: source.sql(), "source.moments.sql")
        elif adapter.name == "breakout_moments":
            _expect_code(lambda: source.unit_frame(metric), "source.breakout.unit_grain")
            _expect_code(lambda: source.unit_counts(), "source.breakout.assignment_counts")
            _expect_code(lambda: source.sql(), "source.breakout.sql")
        elif adapter.name == "sql_totals":
            _expect_code(lambda: source.unit_frame(metric), "source.sql.unit_grain")
            _expect_code(lambda: source.cluster_counts(), "source.sql.cluster_grain")
            _expect_code(lambda: source.sql(grain="daily"), "source.sql.sql_grain")
        elif adapter.name == "definitions":
            _expect_code(
                lambda: source.day_source(metrics=[metric]).unit_frame(metric),
                "source.native.unit_grain",
            )
            _expect_code(lambda: source.cluster_counts(), "source.native.warehouse_cluster_grain")
            _expect_code(lambda: source.sql(grain="daily"), "source.native_sql_grain")
        else:
            _expect_code(lambda: source.unit_frame(metric), "source.frame.unit_frame_panel")
            _expect_code(lambda: source.cluster_counts(), "source.frame.cluster_grain")
    finally:
        source.close()


def test_day_source_breakout_absent_grain_uses_native_refusal(tmp_path: Path) -> None:
    source, breakout = definitions_source(tmp_path)
    metric = source.context.metrics[0]
    try:
        _expect_code(
            lambda: source.day_source(metrics=[metric]).breakout_moments(
                metric,
                breakout,
                grain="weekly",
            ),
            "source.native.grain",
        )
    finally:
        source.close()


def test_definitions_operations_are_callable(tmp_path: Path) -> None:
    source, breakout = definitions_source(tmp_path, store="always")
    metric = source.context.metrics[0]
    try:
        assert {
            "allocation_history",
            "readout_snapshot",
            "materialize",
            "triggered_counts",
            "triggered_source",
            "sitewide_evidence",
            "export_moments",
            "moments_source",
            "panel_sql",
            "summary_sql",
            "breakout_summaries",
            "factor_summaries",
            "breakout_source",
            "breakout_sources",
            "day_source",
        } <= source.operations
        history = source.allocation_history().to_pylist()
        last_day = max(row["ds"] for row in history)
        assert {
            row["group_id"]: row["n_cumulative"] for row in history if row["ds"] == last_day
        } == source.unit_counts()
        source.materialize()
        assert source.sitewide_evidence(metric).site_total > 0
        assert source.moments_source(metrics=[metric]).moments(metric)
        assert source.panel_sql()
        assert source.summary_sql()
        assert source.breakout_summaries(metrics=[metric])
        source.factor_summaries(metrics=[metric])
        source.breakout_source(breakout, metrics=[metric]).close()
        assert len(source.breakout_sources([breakout], metrics=[metric])) == 1
        assert source.day_source(metrics=[metric]).moments(metric, grain="daily")
        source.export_moments(tmp_path / "moments.parquet")
    finally:
        source.close()


def test_definitions_trigger_operations_are_callable(tmp_path: Path) -> None:
    source, _breakout = definitions_source(tmp_path, trigger=True)
    metric = source.context.metrics[0]
    try:
        assert source.triggered_counts()[1]
        assert source.triggered_source().moments(metric)
        assert source.moments_source(metrics=[metric]).moments(metric)
        assert source.panel_sql()
        assert source.summary_sql()
        source.export_moments(tmp_path / "triggered-moments.parquet")
    finally:
        source.close()


def test_definitions_absent_trigger_operations_refuse(tmp_path: Path) -> None:
    source, _breakout = definitions_source(tmp_path)
    try:
        for call in (source.triggered_counts, source.triggered_source):
            _expect_code(call, "source.native.operation")
    finally:
        source.close()


@pytest.mark.parametrize(
    "name",
    [
        "definitions",
        "frame_summary",
        "frame_panel",
        "moments",
        "breakout_moments",
        "sql_totals",
        "unit_day_artifact",
    ],
)
def test_every_arm_evidence_result_round_trips_and_has_a_frame(tmp_path: Path, name: str) -> None:
    factories = dict(public_result_factories(tmp_path))
    result = factories[name]()
    assert result, f"{name} returned no evidence rows"
    for row in result:
        restored = type(row).model_validate_json(row.model_dump_json())
        assert restored == row
    if hasattr(result, "to_frame"):
        frame = result.to_frame()
    else:
        from increment.breakout.estimates import to_frame

        frame = to_frame(result, model=type(result[0]))
    assert len(frame) == len(result)
    columns = frame.column_names if hasattr(frame, "column_names") else list(frame.columns)
    assert columns


_DIGEST_ADAPTER_NAMES = tuple(name for name in _ADAPTER_NAMES if name != "breakout_moments")


@pytest.mark.parametrize("adapter_name", _DIGEST_ADAPTER_NAMES)
def test_readout_snapshot_replays_and_tracks_changed_ingress_arms(
    tmp_path: Path, adapter_name: str
) -> None:
    from increment import readouts
    from increment.errors import CapabilityError
    from increment.estimation.readout_types import ReadoutResults

    source, _metric = next(
        adapter for adapter in _adapter_cases(tmp_path) if adapter.name == adapter_name
    ).build()

    class ChangedTreatmentSource:
        def __init__(self, delegate):
            self._delegate = delegate

        def __getattr__(self, name):
            return getattr(self._delegate, name)

        @staticmethod
        def _rename_treatment(counts):
            renamed = dict(counts)
            if "treatment" in renamed:
                renamed["treatment_digest_change"] = renamed.pop("treatment")
            return renamed

        def assignment_counts(self, *, population="assigned"):
            method = getattr(self._delegate, "assignment_counts", None)
            if not callable(method):
                raise CapabilityError(
                    "source does not expose population counts",
                    code="readout.scope.source_digest_unavailable",
                    context={},
                )
            return self._rename_treatment(method(population=population))

        def unit_counts(self):
            return self._rename_treatment(self._delegate.unit_counts())

        def cluster_counts(self):
            return self._rename_treatment(self._delegate.cluster_counts())

        def moments(self, *args, **kwargs):
            rows = self._delegate.moments(*args, **kwargs)
            changed = False
            output = []
            for row in rows:
                copy = dict(row)
                if copy.get("group_id") == "treatment":
                    copy["group_id"] = "treatment_digest_change"
                    changed = True
                output.append(copy)
            assert changed, "adapter did not expose the fixture's treatment evidence"
            return output

    first = readouts.run(source)
    replay = readouts.run(source)
    mutated = readouts.run(ChangedTreatmentSource(source))
    assert first and replay and mutated
    first_id = first[0].source_snapshot_id
    assert first_id is not None
    assert {row.source_snapshot_id for row in replay} == {first_id}
    mutated_id = mutated[0].source_snapshot_id
    assert mutated_id is not None and mutated_id != first_id
    restored = ReadoutResults.model_validate_json(first.model_dump_json())
    assert restored.source == first.source
    assert restored.metadata == first.metadata


def test_sql_totals_advertised_summary_sql_succeeds() -> None:
    from tests.source_conformance import sql_totals_source

    source = sql_totals_source()
    try:
        assert source.operations == frozenset({"summary_sql"})
        sql = source.summary_sql()
        assert set(sql) == {"revenue"}
        assert sql["revenue"]
    finally:
        source.close()
