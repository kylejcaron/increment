"""Static caller contracts for source-specific builder options."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from os import PathLike

from increment import (
    AnalysisPlan,
    Randomized,
    from_definitions,
    from_frame,
    from_moments,
    from_switchback_frame,
    from_unit_day_artifact,
)
from increment.artifacts import SourceSnapshotEvidence
from increment.frame import IntoDataFrame
from increment.query.artifact_contract import ArtifactStore
from increment.semantics.artifact import ArtifactContext, UnitDayArtifactRef
from increment.semantics.unit_cycle import UnitCycleReference
from increment.switchback import SwitchbackAssignment


def frame_calls(frame: IntoDataFrame, design: Randomized, plan: AnalysisPlan) -> None:
    from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        experiment_id="summary",
        design=design,
        plan=plan,
        uptake="uptake",
        on_unassigned="exclude",
        cluster="cluster",
        intervention_grain="cluster",
        exposure_date="exposure",
        breakouts=["country"],
    )
    from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
        experiment_id="panel",
        plan=plan,
        uptake="uptake",
        exposure_date="exposure",
        observation_end=dt.date(2025, 1, 2),
        breakouts=["country"],
        on_unassigned="exclude",
        cluster="cluster",
    )
    from_frame(
        frame, shape="summary", unit="unit", group="arm", control="control", metrics=["value"]
    )
    from_frame(
        frame, shape="panel", unit="unit", group="arm", date="day", design=design, metrics=["value"]
    )
    from_frame(
        frame,
        shape="summary",  # ty: ignore[invalid-argument-type]
        unit="unit",
        group="arm",
        control="control",
        metrics=["value"],
        date="day",
    )
    from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=["value"],
        observation_end=2,
    )  # ty: ignore[no-matching-overload]
    from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=["value"],
        intervention_grain="cluster",
    )  # ty: ignore[no-matching-overload]
    from_frame(frame, shape="panel", unit="unit", group="arm", control="control", metrics=["value"])  # ty: ignore[invalid-argument-type]


def retained_calls(
    frame: IntoDataFrame,
    rows: Sequence[Mapping[str, object]],
    store: ArtifactStore,
    ref: UnitDayArtifactRef,
    context: ArtifactContext,
    design: Randomized,
    assignment: SwitchbackAssignment,
    references: Mapping[str, UnitCycleReference],
    plan: AnalysisPlan,
) -> None:
    from_unit_day_artifact(store, ref, expected_context=context, verification="lazy_digest")
    from_moments(
        rows, metrics={"value": "mean"}, control="control", experiment_id="moments", plan=plan
    )
    from_moments(rows, metrics={"value": "mean"}, design=design)
    from_switchback_frame(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="arm",
        metrics={"value": "mean"},
        identification=design,
        assignment=assignment,
        experiment_id="switchback",
        plan=plan,
        contrast_references=references,
    )
    from_unit_day_artifact(store, ref, expected_context=context, verification="eager")  # ty: ignore[invalid-argument-type]
    from_moments(rows, metrics={"value": "mean"}, date="day")  # ty: ignore[unknown-argument]
    from_switchback_frame(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="arm",
        metrics=["value"],
        identification=design,
        assignment=assignment,
        control="control",  # ty: ignore[unknown-argument]
    )


def definitions_calls(path: str | PathLike[str], evidence: SourceSnapshotEvidence) -> None:
    from_definitions(path)
    from_definitions(
        path,
        experiment_name="experiment",
        backend="duckdb",
        store="none",
        on_mixed_assignment="exclude",
        source_snapshot_evidence=evidence,
    )
    from_definitions(path, store="invalid")  # ty: ignore[invalid-argument-type]
