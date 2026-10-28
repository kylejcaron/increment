"""Small, isolated evidence-adapter fixtures shared by conformance suites.

The factories intentionally construct fresh in-memory sources.  They do not use
shared pytest connections or real warehouses; each call owns its DuckDB session
and all rows are tiny enough for a unit test.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from increment.semantics.unit_cycle import UnitCycleTApproximation


@dataclass(frozen=True)
class ArmAdapter:
    """A fresh source factory plus the metric and optional breakout it serves."""

    name: str
    factory: Callable[[], Any]
    metric_name: str = "revenue"
    analysis_factory: Callable[[], Any] | None = None

    def build(self) -> tuple[Any, Any]:
        source = self.factory()
        metric = next(
            metric for metric in source.context.metrics if metric.name == self.metric_name
        )
        return source, metric


def unit_summary_table() -> Any:
    """One row per unit with a mean metric and a breakout-like spare column."""
    import pyarrow as pa

    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(1, 9)],
            "variant": [
                "control",
                "control",
                "control",
                "control",
                "treatment",
                "treatment",
                "treatment",
                "treatment",
            ],
            "revenue": [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0],
            "segment": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def unit_panel_table() -> Any:
    """A three-day unit panel with a declared segment dimension."""
    import pyarrow as pa

    rows: list[dict[str, Any]] = []
    start = dt.date(2026, 1, 1)
    for unit, group, segment, base in (
        ("u1", "control", "a", 1.0),
        ("u2", "control", "b", 2.0),
        ("u3", "control", "a", 3.0),
        ("u4", "control", "b", 4.0),
        ("u5", "treatment", "a", 4.0),
        ("u6", "treatment", "b", 5.0),
        ("u7", "treatment", "a", 6.0),
        ("u8", "treatment", "b", 7.0),
    ):
        for day in range(3):
            rows.append(
                {
                    "user_id": unit,
                    "variant": group,
                    "ds": start + dt.timedelta(days=day),
                    "revenue": base + day,
                    "segment": segment,
                }
            )
    return pa.Table.from_pylist(rows)


def mean_metric() -> Any:
    from increment.semantics.models import MeanMetric

    return MeanMetric(name="revenue", entity="user_id", fact="revenue")


def _format3_row(
    *, group: str, metric: str = "revenue", segment: str | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": "conformance",
        "metric": metric,
        "group_id": group,
        "n": 2,
        "successes": None,
        "ref_y": 2.0 if group == "control" else 4.0,
        "cy1": 0.0,
        "cy2": 1.0,
        "ref_x": None,
        "cx1": None,
        "cx2": None,
        "cxy": None,
        "ref_den": None,
        "cden1": None,
        "cden2": None,
        "cyden": None,
        "sum_d": None,
        "cyd": None,
        "cy2d": None,
        "cxd": None,
        "x_role": None,
        "winsor_lower_percentile": None,
        "winsor_upper_percentile": None,
        "winsor_lower_bound": None,
        "winsor_upper_bound": None,
        "winsor_n": None,
        "winsor_n_lower": None,
        "winsor_n_upper": None,
        "moments_format": 2,
    }
    if segment is not None:
        row["segment"] = segment
    return row


def _legacy_fixed10_rows(
    rows: list[dict[str, Any]], *, assignment_counts: str | None = None
) -> list[dict[str, Any]]:
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan

    wire = compiled_plan_to_json(compile_decision_plan(None, [mean_metric()]))
    result = []
    for row in rows:
        result.append(
            {
                **row,
                "moments_format": 11,
                "decision_plan": wire,
                **(
                    {"assignment_counts": assignment_counts}
                    if assignment_counts is not None
                    else {}
                ),
            }
        )
    return result


def _artifact_analysis(tmp_path: Path) -> Any:
    import ibis

    from examples._seed import seed_event_log
    from increment import Analysis, SourceSnapshotEvidence
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.session import WarehouseArtifactStore
    from tests.test_unit_day_artifact_facade import _definitions

    definitions = _definitions(
        tmp_path,
        trigger="session_start",
        breakouts=[{"property": "country"}],
        cuped_metrics=("purchase_rate",),
        site_volume_only=True,
    )
    con = ibis.duckdb.connect()
    seed_event_log(con, with_pre_period=True)
    evidence = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)
    native = Analysis.from_definitions(
        "new_onboarding_v2",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(evidence, {"event_log": evidence}),
    )
    context = native._artifact_context(
        native._defs,  # ty: ignore[invalid-argument-type]
        native._experiment,
        native._on_mixed_assignment,
    )
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    requests = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    native.close()
    return adopted


def unit_day_artifact_analysis(tmp_path: Path) -> Any:
    """Publish the definitions arm and reopen it through the artifact store."""
    return _artifact_analysis(tmp_path)


def frame_summary_source() -> Any:
    from increment.frame import from_unit_summary

    return from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        experiment_id="summary-conformance",
    )


def frame_panel_source() -> Any:
    from increment.frame import from_unit_panel

    return from_unit_panel(
        unit_panel_table(),
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics={"revenue": "mean"},
        experiment_id="panel-conformance",
        breakouts=["segment"],
    )


def frame_panel_analysis() -> Any:
    from increment import Analysis

    return Analysis.from_unit_panel(
        unit_panel_table(),
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics={"revenue": "mean"},
        experiment_id="panel-conformance",
        breakouts=["segment"],
    )


def _current_fixed11_rows() -> list[dict[str, Any]]:
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import Analysis

    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "arm": ["control", "control", "treatment", "treatment"],
                "revenue": [10.0, 12.0, 18.0, 20.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"revenue": "mean"},
        control="control",
        experiment_id="moments-current-conformance",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "current-fixed11.parquet"
        analysis.export(path)
        return pq.read_table(path).to_pylist()


def moments_source() -> Any:
    from increment.semantics.design import Randomized
    from increment.sources import MomentsSource

    return MomentsSource(
        _current_fixed11_rows(),
        metrics=[mean_metric()],
        study_id="moments-current-conformance",
        design=Randomized(control_group="control"),
    )


def legacy_fixed10_moments_source() -> Any:
    from increment.semantics.design import Randomized
    from increment.sources import MomentsSource

    metric = mean_metric()
    return MomentsSource(
        _legacy_fixed10_rows([_format3_row(group="control"), _format3_row(group="treatment")]),
        metrics=[metric],
        study_id="moments-conformance",
        design=Randomized(control_group="control"),
    )


def legacy_fixed10_moments_with_counts_source() -> Any:
    import json

    from increment.semantics.design import Randomized
    from increment.sources import MomentsSource

    metric = mean_metric()
    counts = json.dumps({"control": 2, "treatment": 2}, separators=(",", ":"))
    rows = _legacy_fixed10_rows(
        [_format3_row(group="control"), _format3_row(group="treatment")],
        assignment_counts=counts,
    )
    return MomentsSource(
        rows,
        metrics=[metric],
        study_id="moments-counts-conformance",
        design=Randomized(control_group="control"),
    )


def breakout_moments_source() -> Any:
    from increment.semantics.design import Randomized
    from increment.sources import BreakoutMomentsSource

    metric = mean_metric()
    return BreakoutMomentsSource(
        [
            _format3_row(group="control", segment="a"),
            _format3_row(group="treatment", segment="a"),
        ],
        dimension="segment",
        metrics=[metric],
        study_id="breakout-conformance",
        design=Randomized(control_group="control"),
    )


def sql_totals_source() -> Any:
    import ibis

    from increment.query.source import SqlPanelSource
    from increment.semantics.design import Randomized

    con = ibis.duckdb.connect()
    return SqlPanelSource.from_frame_via_memtable(
        con,
        unit_summary_table(),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        study_id="sql-conformance",
        design=Randomized(control_group="control"),
    )


def definitions_source(
    tmp_path: Path,
    *,
    trigger: bool = False,
    store: Literal["auto", "always", "none"] = "none",
    partial_trigger: bool = False,
    cluster: bool = False,
    sql_exposures: bool = False,
    return_analysis: bool = False,
    window_days: int | None = 1,
) -> tuple[Any, Any]:
    """Build an in-memory definitions source with one breakout.

    The no-trigger context is intentional for refusal checks; the optional
    trigger event exercises the complete triggered operation surface.
    """
    import ibis
    import yaml

    from increment import Analysis, SourceSnapshotEvidence

    rows = []
    for index, (group, country, value) in enumerate(
        (
            ("control", "US", 1.0),
            ("control", "CA", 2.0),
            ("treatment", "US", 4.0),
            ("treatment", "CA", 5.0),
        )
    ):
        user = f"u{index + 1}"
        membership = {"store_id": f"{group}-store"} if cluster else {}
        rows.extend(
            [
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 9),
                    "event": "assignment",
                    "group_id": group,
                    **membership,
                    "country_code": country,
                    "revenue": None,
                },
                *(
                    [
                        {
                            "user_id": user,
                            "ts": dt.datetime(2026, 1, 1, 10),
                            "event": "trigger",
                            "group_id": group,
                            **membership,
                            "revenue": None,
                        }
                    ]
                    if trigger and (not partial_trigger or index < 3)
                    else []
                ),
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 12),
                    "event": "purchase",
                    "group_id": None,
                    **membership,
                    "country_code": country,
                    "revenue": value,
                },
            ]
        )
    defs = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "assignment", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
                "properties": [
                    {
                        "name": "country",
                        "column": "country_code",
                        "dtype": "string",
                        "as_of": "static",
                    }
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "assignment"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": window_days,
            }
        ],
        "experiments": [
            {
                "name": "conformance",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-03",
                "control_group": "control",
                "plan": {"primary": "revenue"},
                "breakouts": [] if cluster else [{"property": "country"}],
            }
        ],
    }
    if trigger:
        defs["fact_sources"][0]["facts"].append({"name": "trigger", "column": None})
        defs["exposures"].append({"name": "trigger", "fact": "trigger"})
        defs["experiments"][0]["trigger"] = "trigger"
    if sql_exposures:
        defs["exposures"][0] = {
            "name": "assignment",
            "sql": "SELECT user_id AS unit_id, ts, group_id FROM raw_events WHERE event = 'assignment'",
        }
        if trigger:
            defs["exposures"][1] = {
                "name": "trigger",
                "sql": "SELECT user_id AS unit_id, ts, group_id FROM raw_events WHERE event = 'trigger'",
            }
    if cluster:
        defs["experiments"][0]["cluster"] = "store_id"
    defs_path = tmp_path / "conformance.yaml"
    defs_path.write_text(yaml.safe_dump(defs, sort_keys=False))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", obj=rows)
    evidence = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    analysis = Analysis.from_definitions(
        "conformance",
        defs_path,
        con,
        store=store,
        source_snapshot_evidence=(
            SourceSnapshotEvidence(evidence, {"events": evidence}) if trigger else None
        ),
    )
    experiment = analysis.experiment
    return (analysis if return_analysis else analysis._src), (
        experiment.breakouts[0] if experiment.breakouts else None
    )


def unit_day_artifact_source(tmp_path: Path) -> Any:
    """Publish and adopt the canonical unit-day artifact fixture."""
    return _artifact_analysis(tmp_path)._src


def native_day_source(tmp_path: Path) -> Any:
    source, _ = definitions_source(tmp_path)
    return source.day_source(metrics=source.context.metrics)


def definitions_triggered_source(tmp_path: Path) -> Any:
    source, _ = definitions_source(tmp_path, trigger=True)
    return source.triggered_source()


def _encouragement_panel_inputs() -> tuple[Any, Any]:
    from datetime import date

    import pyarrow as pa

    from increment.semantics.design import Encouragement

    design = Encouragement.model_validate(
        {
            "mechanism": "encouragement",
            "control_group": "control",
            "uptake": {"fact": "clicked"},
            "exclusion_restriction": {"justification": "the prompt has no direct effect"},
        }
    )
    rows = [
        {
            "unit": f"{arm}-{index}",
            "arm": arm,
            "day": date(2026, 1, day),
            "exposure": date(2026, 1, 1),
            "revenue": float(index + day),
            "clicked": int(arm == "treatment" and index % 2 == 0),
        }
        for index in range(2)
        for day in (1, 2)
        for arm in ("control", "treatment")
    ]
    return pa.Table.from_pylist(rows), design


def encouragement_panel_source() -> Any:
    from datetime import date

    from increment.frame import from_unit_panel

    rows, design = _encouragement_panel_inputs()
    return from_unit_panel(
        rows,
        unit="unit",
        group="arm",
        date="day",
        control="control",
        exposure_date="exposure",
        uptake="clicked",
        observation_end=date(2026, 1, 4),
        design=design,
        metrics={"revenue": "mean"},
    )


def encouragement_panel_analysis() -> Any:
    from datetime import date

    from increment import Analysis

    rows, design = _encouragement_panel_inputs()
    return Analysis.from_unit_panel(
        rows,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposure",
        uptake="clicked",
        observation_end=date(2026, 1, 4),
        design=design,
        metrics={"revenue": "mean"},
    )


def arm_adapters(tmp_path: Path) -> tuple[ArmAdapter, ...]:
    """Factories for every arm adapter present at the Barrier D base."""
    return (
        ArmAdapter(
            "definitions",
            lambda: definitions_source(tmp_path)[0],
            analysis_factory=lambda: definitions_source(tmp_path, return_analysis=True)[0],
        ),
        ArmAdapter("frame_summary", frame_summary_source),
        ArmAdapter("frame_panel", frame_panel_source, analysis_factory=frame_panel_analysis),
        ArmAdapter("moments", moments_source),
        ArmAdapter("legacy_fixed10", legacy_fixed10_moments_source),
        ArmAdapter("legacy_fixed10_with_counts", legacy_fixed10_moments_with_counts_source),
        ArmAdapter("breakout_moments", breakout_moments_source),
        ArmAdapter("sql_totals", sql_totals_source),
        ArmAdapter(
            "unit_day_artifact",
            lambda: unit_day_artifact_source(tmp_path),
            metric_name="purchase_rate",
            analysis_factory=lambda: unit_day_artifact_analysis(tmp_path),
        ),
    )


def provider_adapters(tmp_path: Path) -> tuple[ArmAdapter, ...]:
    """Factories reserved for typed evidence-provider conformance."""
    return (
        *arm_adapters(tmp_path),
        ArmAdapter("native_day", lambda: native_day_source(tmp_path)),
        ArmAdapter("definitions_triggered", lambda: definitions_triggered_source(tmp_path)),
        ArmAdapter(
            "encouragement_panel",
            encouragement_panel_source,
            analysis_factory=encouragement_panel_analysis,
        ),
    )


def switchback_frame() -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control"))):
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": float(cycle + period + step + (group == "treatment")),
                        }
                    )
    return pd.DataFrame(rows)


def switchback_source() -> Any:
    from increment.frame import from_switchback_panel
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    return from_switchback_panel(
        switchback_frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        contrast_references={"value": UnitCycleTApproximation()},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )


def switchback_analysis() -> Any:
    from increment import Analysis
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    return Analysis.from_switchback_panel(
        switchback_frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        contrast_references={"value": UnitCycleTApproximation()},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )


def public_result_factories(tmp_path: Path) -> tuple[tuple[str, Callable[[], Any]], ...]:
    """Public Analysis constructors returning each arm evidence family."""

    from increment import Analysis

    def definitions_result() -> Any:
        from increment import readouts

        source, _ = definitions_source(tmp_path)
        return readouts.run(source)

    def summary_result() -> Any:
        analysis = Analysis.from_unit_summary(
            unit_summary_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
        return analysis.run()

    def panel_result() -> Any:
        analysis = Analysis.from_unit_panel(
            unit_panel_table(),
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
        )
        return analysis.run_daily_lift()

    def moments_result() -> Any:
        from increment import readouts

        source = moments_source()
        return readouts.run(source)

    def breakout_result() -> Any:
        analysis = Analysis.from_unit_panel(
            unit_panel_table(),
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=["segment"],
        )
        return analysis.run_breakout()

    def sql_result() -> Any:
        from increment import readouts

        source = sql_totals_source()
        try:
            return readouts.run(source)
        finally:
            source.close()

    def artifact_result() -> Any:
        analysis = unit_day_artifact_analysis(tmp_path)
        try:
            return analysis.run(metrics=["purchase_rate"])
        finally:
            analysis.close()

    return (
        ("definitions", definitions_result),
        ("frame_summary", summary_result),
        ("frame_panel", panel_result),
        ("moments", moments_result),
        ("breakout_moments", breakout_result),
        ("sql_totals", sql_result),
        ("unit_day_artifact", artifact_result),
    )


def _captured_source_session(source: Any) -> Any:
    return source._session


def _analysis_connection(analysis: Any) -> Any:
    return analysis._con


def _clear_captured_source_tables(provider: Any) -> None:
    provider.source._session._source_tables.clear()


__all__ = [
    "ArmAdapter",
    "arm_adapters",
    "provider_adapters",
    "breakout_moments_source",
    "definitions_source",
    "frame_panel_source",
    "frame_summary_source",
    "legacy_fixed10_moments_source",
    "legacy_fixed10_moments_with_counts_source",
    "moments_source",
    "unit_day_artifact_analysis",
    "public_result_factories",
    "sql_totals_source",
    "switchback_analysis",
    "switchback_frame",
    "switchback_source",
    "unit_day_artifact_source",
    "unit_panel_table",
    "unit_summary_table",
]
