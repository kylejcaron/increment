from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from increment.query._native_refusals import _refuse_operation
from increment.semantics.design import Encouragement
from increment.semantics.models import Metric
from increment.sources import ComplianceSummary, MomentSource, SourceContext, SourceOperation

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

    from increment.sources import Grain


class TriggeredPopulationSource:
    """MomentSource view that narrows a native source to triggered units."""

    operations: frozenset[SourceOperation] = frozenset()
    shape: Literal["unit_summary", "unit_panel"] | None = None
    _population: Literal["triggered"] = "triggered"

    def __init__(self, source: Any) -> None:
        source._validate_trigger_capability(operation="triggered_source")
        self._source = source
        self.capabilities: frozenset[Grain] = frozenset({"total"})
        self.breakouts = source.breakouts
        self.shape = source.shape

    @property
    def context(self) -> SourceContext:
        return self._source.context

    @property
    def source_name(self) -> str | None:
        return getattr(self._source, "source_name", None)

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        return self._source.moments(
            metric,
            grain=grain,
            by=by,
            completed_windows_only=completed_windows_only,
            include_covariate=include_covariate,
            population="triggered",
        )

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> pa.Table:
        return self._source.unit_frame(
            metric,
            covariates=covariates,
            population="triggered",
            outcome_stage=outcome_stage,
        )

    def unit_counts(self) -> dict[str, int]:
        grain, counts, unit_counts = self._source.triggered_counts()
        return unit_counts if grain == "cluster" else counts

    def cluster_counts(self) -> dict[str, int]:
        grain, counts, _unit_counts = self._source.triggered_counts()
        if grain != "cluster":
            return self._source.cluster_counts()
        return counts

    def compliance_dates(self) -> Sequence[object]:
        return self._source.triggered_compliance_dates()

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        return self._source.compliance_summary(
            design,
            as_of=as_of,
            completed_windows_only=completed_windows_only,
            population="triggered",
        )

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        _refuse_operation(
            operation="sql",
            request={"grain": grain, "population": self._population},
            offered=tuple(sorted(self.capabilities)),
            route=(
                "read triggered moments through moments(); summary SQL describes the "
                "assigned population, not the triggered one"
            ),
        )

    def close(self) -> None:
        self._source.close()

    @contextmanager
    def readout_snapshot(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"],
        uptake_facts: Sequence[str] = (),
        include_breakouts: bool = False,
    ) -> Iterator[MomentSource]:
        with self._source.readout_snapshot(
            metrics=metrics,
            population="assigned",
            uptake_facts=uptake_facts,
            include_breakouts=include_breakouts,
        ) as pinned:
            yield self if pinned is self._source else pinned.triggered_source()
