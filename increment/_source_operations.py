"""Narrow callable protocols for source-specific operations.

These protocols deliberately stay out of the package root. ``MomentSource``
remains the common statistical seam; operation protocols describe only the
complete native operations that a caller may opt into.
"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from increment.errors import CodedError, _freeze

if TYPE_CHECKING:
    import pyarrow as pa

    from increment.analysis import Analysis
    from increment.estimation.diagnostics import SRMResult
    from increment.estimation.results import LiftEstimate
    from increment.query.native_source import DayEvidenceSource
    from increment.semantics.models import Breakout, Metric
    from increment.sequential_state import SequentialCheckpoint
    from increment.sources import BreakoutMomentsSource, MomentSource


@dataclass(frozen=True, slots=True)
class DashboardGroupData:
    """Immutable observed evidence for one metric/arm dashboard cell."""

    metric: str
    group_id: str
    assigned_units: int | None
    eligible_units: int
    observed_value: float | None
    analysis_input_value: float | None
    sum_value: float | None
    event_count: int | None
    retained_units: int | None
    numerator: float | None
    denominator: float | None
    excluded_not_mature: int | None
    excluded_no_observed_day: int | None
    excluded_other: int | None
    observation_end: dt.date | None
    window_start_days: int | None
    window_end_days: int | None
    source_kind: Literal["pinned_warehouse", "retained_checkpoint"]
    prefix_id: str | None
    unavailable: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unavailable", _freeze(self.unavailable))


# A declared breakout choice as (declared source or None, property).
BreakoutChoice = tuple[str | None, str]
# (view, metric or None for every declared metric, completed windows only, breakout choice)
ExploreKey = tuple[str, str | None, bool, BreakoutChoice | None]


@dataclass(frozen=True, slots=True)
class DashboardExploreCapture:
    """One Explore request answered inside the pinned read.

    Holds the rows the source returned, or the coded refusal it raised, never both. The refusal
    is a detached copy with no traceback, so no frame keeps the pinned analysis alive, and
    every :meth:`load` yields a fresh collection or a fresh copy of that refusal.
    """

    view: str
    metric: str | None
    completed_windows_only: bool
    breakout: BreakoutChoice | None
    collection: Callable[[Iterable[Any]], Sequence[Any]] | None
    rows: tuple[Any, ...]
    refusal: CodedError | None

    @classmethod
    def answered(
        cls,
        key: ExploreKey,
        rows: Sequence[Any],
        *,
        collection: Callable[[Iterable[Any]], Sequence[Any]],
    ) -> DashboardExploreCapture:
        view, metric, completed, breakout = key
        return cls(view, metric, completed, breakout, collection, tuple(rows), None)

    @classmethod
    def refused(cls, key: ExploreKey, refusal: CodedError) -> DashboardExploreCapture:
        view, metric, completed, breakout = key
        return cls(view, metric, completed, breakout, None, (), copy.copy(refusal))

    @property
    def key(self) -> ExploreKey:
        return (self.view, self.metric, self.completed_windows_only, self.breakout)

    def load(self) -> Sequence[Any]:
        """A fresh collection of the captured rows, or a fresh copy of the captured refusal."""
        if self.refusal is not None:
            raise copy.copy(self.refusal)
        assert self.collection is not None
        return self.collection(self.rows)


@dataclass(frozen=True, slots=True)
class DashboardSnapshotPayload:
    """Complete dashboard evidence from one pinned source read."""

    allocation: SRMResult | None
    allocation_refusal: tuple[str, str] | None
    allocation_history: pa.Table | None
    allocation_history_refusal: tuple[str, str] | None
    estimates: tuple[LiftEstimate, ...]
    group_data: tuple[DashboardGroupData, ...]
    explore: tuple[DashboardExploreCapture, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimates", tuple(self.estimates))
        object.__setattr__(self, "group_data", tuple(self.group_data))
        object.__setattr__(self, "explore", tuple(self.explore))


class DashboardSnapshotHandler(Protocol):
    def __call__(
        self, analysis: Analysis, breakouts: Sequence[tuple[Breakout, Analysis]], /
    ) -> DashboardSnapshotPayload:
        """Read the payload from ``analysis``; each breakout pairs with an analysis reading
        only that declared breakout, so one breakout's refusal cannot hide another's."""
        ...


@runtime_checkable
class AllocationHistoryOperation(Protocol):
    def allocation_history(self) -> pa.Table: ...


@runtime_checkable
class MaterializeOperation(Protocol):
    def materialize(self) -> None: ...


@runtime_checkable
class TriggeredPopulationOperation(Protocol):
    def triggered_source(self) -> MomentSource: ...


@runtime_checkable
class DashboardGroupDataOperation(Protocol):
    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
    ) -> tuple[DashboardGroupData, ...]: ...


@runtime_checkable
class TriggeredCountsOperation(Protocol):
    def triggered_counts(
        self,
    ) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]: ...


@runtime_checkable
class SummarySqlOperation(Protocol):
    """Public summary SQL operation shared by SQL-backed sources."""

    def summary_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]: ...


@runtime_checkable
class BreakoutSourcesOperation(Protocol):
    def breakout_sources(
        self, breakouts: Sequence[Breakout], *, metrics: Sequence[Metric]
    ) -> Sequence[BreakoutMomentsSource]: ...


@runtime_checkable
class SitewideEvidenceOperation(Protocol):
    def sitewide_evidence(self, metric: Any, *, include_ratio: bool = False) -> Any: ...


@runtime_checkable
class ExportMomentsOperation(Protocol):
    def export_moments(self, path: str | Path) -> None: ...


@runtime_checkable
class MomentsSourceOperation(Protocol):
    def _moments_for_metrics(
        self,
        methods: list[Any] | None = None,
        selected: Sequence[Any] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        # Internal source option remains positional for protocol compatibility.
        narrow_cuped: bool = False,  # noqa: FBT001, FBT002
    ) -> Any: ...

    def assignment_counts(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> dict[str, int]: ...


@runtime_checkable
class PanelSQLOperation(Protocol):
    def _panel_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]: ...


@runtime_checkable
class SummarySQLOperation(Protocol):
    def _summary_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]: ...


@runtime_checkable
class ReadoutSnapshotOperation(Protocol):
    """Source-owned snapshot spanning one complete readout operation."""

    def readout_snapshot(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"],
        uptake_facts: Sequence[str] = (),
        include_breakouts: bool = False,
    ) -> AbstractContextManager[MomentSource]: ...


@runtime_checkable
class BreakoutSourceOperation(Protocol):
    def breakout_source(self, breakout: Breakout, *, metrics: Sequence[Metric]) -> MomentSource: ...


@runtime_checkable
class DaySourceOperation(Protocol):
    def day_source(self, *, metrics: Sequence[Metric]) -> DayEvidenceSource: ...


__all__ = [
    "BreakoutChoice",
    "ExploreKey",
    "DashboardExploreCapture",
    "DashboardGroupData",
    "DashboardGroupDataOperation",
    "AllocationHistoryOperation",
    "DashboardSnapshotHandler",
    "DashboardSnapshotPayload",
    "BreakoutSourceOperation",
    "BreakoutSourcesOperation",
    "DaySourceOperation",
    "ExportMomentsOperation",
    "MaterializeOperation",
    "MomentsSourceOperation",
    "PanelSQLOperation",
    "SitewideEvidenceOperation",
    "SummarySQLOperation",
    "ReadoutSnapshotOperation",
    "SummarySqlOperation",
    "TriggeredCountsOperation",
    "TriggeredPopulationOperation",
]
