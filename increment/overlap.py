"""Capture, analyze, and plan two independent overlapping experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from increment._frame_overlap import capture_overlap_frame
from increment._overlap_sdk import OverlapSdkArchive, sdk_overlap_decision, sdk_overlap_plan
from increment.estimation.overlap import (
    OverlapResult,
    _overlap_joint_decision,
    _overlap_rows,
    analyze_overlap,
)
from increment.power.overlap import (
    OverlapPowerResult,
    evaluate_overlap_scenario,
    prepare_overlap_scenario,
)
from increment.semantics.overlap import (
    OverlapCapture,
    OverlapClaim,
    OverlapEvidence,
    OverlapFactor,
    OverlapFeed,
    OverlapLaw,
    OverlapMetric,
    OverlapPlan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ibis.expr.types import Table
    from narwhals.typing import IntoDataFrame

    from increment.estimation.overlap import OverlapRow
    from increment.power.overlap import JointDecision, OverlapDraw
    from increment.semantics.overlap import CellId

__all__ = [
    "OverlapCapture",
    "OverlapClaim",
    "OverlapEvidence",
    "OverlapFactor",
    "OverlapFeed",
    "OverlapLaw",
    "OverlapMetric",
    "OverlapPlan",
    "OverlapPowerResult",
    "OverlapResult",
    "OverlapSdkArchive",
    "analyze_overlap",
    "capture_overlap_frame",
    "capture_overlap_warehouse",
    "sdk_overlap_decision",
    "sdk_overlap_plan",
    "simulate_overlap",
]


def capture_overlap_warehouse(
    plan: OverlapPlan,
    *,
    capture: OverlapCapture,
    roster: Table,
    decisions: Table,
    deliveries: Table,
    outcomes: Table,
) -> OverlapEvidence:
    """Capture only aggregates from four verified warehouse relations."""
    from increment.query.overlap import capture_overlap_warehouse as capture_bound

    return capture_bound(
        plan,
        capture=capture,
        roster=roster,
        decisions=decisions,
        deliveries=deliveries,
        outcomes=outcomes,
    )


def _evaluate_overlap_draw(
    plan: OverlapPlan, draw: OverlapDraw
) -> tuple[tuple[OverlapRow, ...], JointDecision]:
    rows = _overlap_rows(plan, draw.cells, target_count=sum(draw.counts))
    return rows, _overlap_joint_decision(rows, delivery_conflicts=False)


def simulate_overlap(
    plan: OverlapPlan,
    *,
    n_target: int,
    cell_outcomes: Mapping[CellId, IntoDataFrame],
    replications: int,
    seed: int,
) -> OverlapPowerResult:
    """Estimate decision rates for a specified design and four outcome distributions."""
    scenario = prepare_overlap_scenario(plan, n_target=n_target, cell_outcomes=cell_outcomes)
    return evaluate_overlap_scenario(
        scenario,
        replications=replications,
        seed=seed,
        evaluate=lambda draw: _evaluate_overlap_draw(plan, draw),
    )
