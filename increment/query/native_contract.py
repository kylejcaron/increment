"""Evidence shapes and capability protocols of the native warehouse path.

``DefinitionsMomentSource`` (``native_source``) implements these protocols and
builds these shapes. Consumers import the contracts here; ``Analysis`` still
imports the concrete warehouse adapter for construction.
Nothing here may import an adapter module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ibis import Table

from increment.errors import CapabilityError, RefusalSpec
from increment.estimation.armstats import ArmStats
from increment.semantics.models import Breakout, Metric
from increment.sources import (
    BreakoutMomentsSource,
    MomentSource,
    SourceContext,
    SourceOperation,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from increment.sources import Grain


@dataclass(frozen=True, slots=True)
class SitewideEvidence:
    """Typed evidence needed by the sitewide estimator."""

    metric: Metric
    site_total: float
    arm_stats: tuple[ArmStats, ...]
    control_group: str
    cluster: str | None
    site_total_denominator: float | None = None
    cluster_counts: dict[str, int] | None = None
    unit_counts: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _LegacyBreakoutSource:
    """Source-name shim for the private pre-Task8 day-axis adapter."""

    name: str


@dataclass(frozen=True, slots=True)
class DimensionedDayEvidence:
    """Named day-axis result for one breakout and metric."""

    source_name: str
    rows: list[dict[str, Any]]

    def __iter__(self):
        """Keep the private ``_day_axis_source`` tuple contract readable."""
        yield _LegacyBreakoutSource(self.source_name)
        yield self.rows


@runtime_checkable
class DayEvidenceSource(MomentSource, Protocol):
    """Typed day-axis evidence source owned by the native warehouse path."""

    capabilities: frozenset[Grain]
    operations: frozenset[SourceOperation]
    breakouts: tuple[str, ...]

    @property
    def context(self) -> SourceContext: ...

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "daily",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]: ...

    def breakout_moments(
        self,
        metric: Metric,
        breakout: Breakout,
        *,
        grain: Grain = "daily",
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> DimensionedDayEvidence: ...

    def close(self) -> None: ...


@runtime_checkable
class NativeCoreSource(MomentSource, Protocol):
    """Source-owned native core workflows."""

    operations: frozenset[SourceOperation]

    def materialize(self) -> None: ...

    def triggered_counts(
        self,
    ) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]: ...

    def triggered_source(self) -> MomentSource: ...

    def trigger_rates(self) -> dict[str, float]: ...

    def export_moments(self, path: str | Path) -> None: ...

    def moments_source(
        self,
        *,
        metrics: Sequence[Metric],
        methods: list[Any] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        narrow_cuped: bool = False,
    ) -> MomentSource: ...

    def panel_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]: ...

    def summary_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]: ...
    def build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
        horizon_metrics: Sequence[Metric] | None = None,
    ) -> Any: ...


@runtime_checkable
class NativeViewSource(Protocol):
    """Source-owned native view workflows."""

    operations: frozenset[SourceOperation]

    def sitewide_evidence(
        self, metric: Metric, *, include_ratio: bool = False
    ) -> SitewideEvidence: ...

    def breakout_summaries(
        self, *, metrics: Sequence[Metric]
    ) -> dict[str, dict[str, pa.Table]]: ...

    def factor_summaries(self, *, metrics: Sequence[Metric]) -> dict[str, pa.Table]: ...

    def breakout_source(
        self, breakout: Breakout, *, metrics: Sequence[Metric]
    ) -> BreakoutMomentsSource: ...

    def breakout_sources(
        self, breakouts: Sequence[Breakout], *, metrics: Sequence[Metric]
    ) -> Sequence[BreakoutMomentsSource]: ...

    def day_source(self, *, metrics: Sequence[Metric]) -> DayEvidenceSource: ...


_NATIVE_COVARIATE_RESERVED = RefusalSpec(
    "source.native.covariate_reserved",
    CapabilityError,
    lambda *, covariate, cluster: (
        f"unit_frame: {covariate!r} is a reserved metadata column name"
        + (f" carrying the identity of declared cluster {cluster!r}" if cluster is not None else "")
        + " -- rename the property to request it as a covariate."
    ),
)
