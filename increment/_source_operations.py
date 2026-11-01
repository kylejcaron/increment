"""Typed payloads and product-operation protocols for source adapters.

Evidence acquisition is described by `EvidenceProvider` declarations.
"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from increment.errors import CodedError, _freeze

if TYPE_CHECKING:
    import pyarrow as pa

    from increment.estimation.diagnostics import SRMResult
    from increment.estimation.results import LiftEstimate
    from increment.semantics.models import Breakout, Metric
    from increment.sequential_state import SequentialCheckpoint
    from increment.sources import EvidenceSource


@dataclass(frozen=True, slots=True)
class TriggeredHealthFeed:
    """Source-owned completeness observation for one triggered feed or metric."""

    evidence_kind: str
    feed: str | None
    metric: str | None
    complete_through: dt.datetime | None


@dataclass(frozen=True, slots=True)
class TriggeredHealthEvidence:
    """Typed source-owned completeness and cutoff evidence for health reports."""

    observation_cutoff: dt.datetime | None
    feeds: tuple[TriggeredHealthFeed, ...]


@runtime_checkable
class TriggeredHealthEvidenceOperation(Protocol):
    def triggered_health_evidence(self) -> TriggeredHealthEvidence: ...


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
    analysis_population: Literal["assigned", "triggered"] = "assigned"

    def __post_init__(self) -> None:
        object.__setattr__(self, "unavailable", _freeze(self.unavailable))


# A declared breakout choice as (declared source or None, property).
BreakoutChoice = tuple[str | None, str]
# (population, view, metric or None, completed windows only, breakout choice)
ExploreKey = tuple[Literal["assigned", "triggered"], str, str | None, bool, BreakoutChoice | None]


@dataclass(frozen=True, slots=True)
class DashboardExploreCapture:
    """One Explore request answered inside the pinned read.

    Holds the rows the source returned, or the coded refusal it raised, never both. The refusal
    is a detached copy with no traceback, so no frame keeps the pinned source alive, and
    every :meth:`load` yields a fresh collection or a fresh copy of that refusal.
    """

    population: Literal["assigned", "triggered"]
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
        population, view, metric, completed, breakout = key
        return cls(population, view, metric, completed, breakout, collection, tuple(rows), None)

    @classmethod
    def refused(cls, key: ExploreKey, refusal: CodedError) -> DashboardExploreCapture:
        population, view, metric, completed, breakout = key
        return cls(population, view, metric, completed, breakout, None, (), copy.copy(refusal))

    @property
    def key(self) -> ExploreKey:
        return (
            self.population,
            self.view,
            self.metric,
            self.completed_windows_only,
            self.breakout,
        )

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
    population_estimates: Mapping[Literal["assigned", "triggered"], tuple[LiftEstimate, ...]] = (
        field(default_factory=dict)
    )
    triggered_allocation: SRMResult | None = None
    triggered_allocation_refusal: tuple[str, str] | None = None
    triggered_allocation_history: pa.Table | None = None
    triggered_allocation_history_refusal: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimates", tuple(self.estimates))
        object.__setattr__(self, "group_data", tuple(self.group_data))
        object.__setattr__(self, "explore", tuple(self.explore))


@runtime_checkable
class AllocationHistoryOperation(Protocol):
    def allocation_history(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> pa.Table: ...


@runtime_checkable
class MaterializeOperation(Protocol):
    def materialize(self) -> None: ...


@runtime_checkable
class DashboardGroupDataOperation(Protocol):
    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> tuple[DashboardGroupData, ...]: ...


@runtime_checkable
class SummarySqlOperation(Protocol):
    """Public summary SQL operation shared by SQL-backed sources."""

    def summary_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]: ...


@runtime_checkable
class SitewideEvidenceOperation(Protocol):
    def sitewide_evidence(
        self,
        metric: Any,
        *,
        include_ratio: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> Any: ...


@runtime_checkable
class ExportMomentsOperation(Protocol):
    def export_moments(self, path: str | Path) -> None: ...


@runtime_checkable
class PanelSQLOperation(Protocol):
    def _panel_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]: ...


@runtime_checkable
class NativePanelSqlOperation(Protocol):
    def panel_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]: ...

    def build_panel_for_metric(
        self,
        exposures: Any,
        metric: Metric,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> Any: ...


@runtime_checkable
class SummarySQLOperation(Protocol):
    def _summary_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]: ...


@runtime_checkable
class ExploratorySourceOperation(Protocol):
    def exploratory_source(self, *, metrics: Sequence[Metric]) -> EvidenceSource: ...


__all__ = [
    "BreakoutChoice",
    "ExploreKey",
    "DashboardExploreCapture",
    "DashboardGroupData",
    "DashboardGroupDataOperation",
    "AllocationHistoryOperation",
    "DashboardSnapshotPayload",
    "ExploratorySourceOperation",
    "ExportMomentsOperation",
    "MaterializeOperation",
    "NativePanelSqlOperation",
    "SitewideEvidenceOperation",
    "SummarySQLOperation",
    "SummarySqlOperation",
    "TriggeredHealthEvidence",
    "TriggeredHealthEvidenceOperation",
    "TriggeredHealthFeed",
]
