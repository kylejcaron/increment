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
    breakout_factory: Callable[[], Any] | None = None

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


def _format10_rows(
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
                "moments_format": 10,
                "decision_plan": wire,
                **(
                    {"assignment_counts": assignment_counts}
                    if assignment_counts is not None
                    else {}
                ),
            }
        )
    return result


def _format10_row(*, group: str, assignment_counts: str | None = None) -> dict[str, Any]:
    return _format10_rows(
        [_format3_row(group=group)],
        assignment_counts=assignment_counts,
    )[0]


def _artifact_analysis(tmp_path: Path) -> Any:
    import ibis

    from examples._seed import seed_event_log
    from increment import Analysis
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
    native = Analysis.from_definitions("new_onboarding_v2", definitions, con)
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


def moments_source() -> Any:
    from increment.semantics.design import Randomized
    from increment.sources import MomentsSource

    metric = mean_metric()
    return MomentsSource(
        _format10_rows([_format3_row(group="control"), _format3_row(group="treatment")]),
        metrics=[metric],
        study_id="moments-conformance",
        design=Randomized(control_group="control"),
    )


def moments_with_assignment_counts_source() -> Any:
    import json

    from increment.semantics.design import Randomized
    from increment.sources import MomentsSource

    metric = mean_metric()
    counts = json.dumps({"control": 2, "treatment": 2}, separators=(",", ":"))
    rows = _format10_rows(
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
    tmp_path: Path, *, trigger: bool = False, store: Literal["auto", "always", "none"] = "none"
) -> tuple[Any, Any]:
    """Build an in-memory definitions source with one breakout.

    The no-trigger context is intentional for refusal checks; the optional
    trigger event exercises the complete triggered operation surface.
    """
    import ibis
    import yaml

    from increment import Analysis

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
        rows.extend(
            [
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 9),
                    "event": "assignment",
                    "group_id": group,
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
                            "revenue": None,
                        }
                    ]
                    if trigger
                    else []
                ),
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 12),
                    "event": "purchase",
                    "group_id": None,
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
                "window_days": 1,
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
                "breakouts": [{"property": "country"}],
            }
        ],
    }
    if trigger:
        defs["fact_sources"][0]["facts"].append({"name": "trigger", "column": None})
        defs["exposures"].append({"name": "trigger", "fact": "trigger"})
        defs["experiments"][0]["trigger"] = "trigger"
    defs_path = tmp_path / "conformance.yaml"
    defs_path.write_text(yaml.safe_dump(defs, sort_keys=False))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", obj=rows)
    analysis = Analysis.from_definitions("conformance", defs_path, con, store=store)
    experiment = analysis.experiment
    return analysis._src, experiment.breakouts[0]


def unit_day_artifact_source(tmp_path: Path) -> Any:
    """Publish and adopt the canonical unit-day artifact fixture."""
    return _artifact_analysis(tmp_path)._src


def arm_adapters(tmp_path: Path) -> tuple[ArmAdapter, ...]:
    """Factories for every arm adapter present at the Barrier D base."""
    return (
        ArmAdapter("definitions", lambda: definitions_source(tmp_path)[0]),
        ArmAdapter("frame_summary", frame_summary_source),
        ArmAdapter("frame_panel", frame_panel_source),
        ArmAdapter("moments", moments_source),
        ArmAdapter("moments_with_counts", moments_with_assignment_counts_source),
        ArmAdapter("breakout_moments", breakout_moments_source),
        ArmAdapter("sql_totals", sql_totals_source),
        ArmAdapter(
            "unit_day_artifact",
            lambda: unit_day_artifact_source(tmp_path),
            metric_name="purchase_rate",
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


__all__ = [
    "ArmAdapter",
    "arm_adapters",
    "breakout_moments_source",
    "definitions_source",
    "frame_panel_source",
    "frame_summary_source",
    "moments_source",
    "unit_day_artifact_analysis",
    "moments_with_assignment_counts_source",
    "public_result_factories",
    "sql_totals_source",
    "switchback_analysis",
    "switchback_frame",
    "switchback_source",
    "unit_day_artifact_source",
    "unit_panel_table",
    "unit_summary_table",
]
