from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from increment._evidence_types import AssignmentRequest, ComplianceSummary, SourceContext
from increment.query._native_refusals import _refuse_operation
from increment.semantics.design import Encouragement
from increment.semantics.models import Metric
from increment.sources import EvidenceSource, SourceOperation

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

    from increment._analysis_config import ResolvedMetricConfig
    from increment._evidence_types import EvidenceScope
    from increment._source_types import EvidenceCapabilities
    from increment.sources import Grain


class TriggeredPopulationSource:
    """MomentSource view that narrows a native source to triggered units."""

    operations: frozenset[SourceOperation] = frozenset()
    shape: Literal["unit_summary", "unit_panel"] | None = None

    @property
    def evidence(self) -> EvidenceCapabilities:
        from increment._source_types import declared_evidence

        evidence = declared_evidence(self._source)
        scalar = evidence.scalar
        return replace(
            evidence,
            kind="definitions",
            grains=self.capabilities,
            breakouts=tuple(self.breakouts),
            shape=self.shape,
            operations=frozenset(
                operation
                for operation in evidence.operations
                if operation in {"readout_snapshot", "readout_execution"}
            ),
            count_grains=frozenset(
                {"unit"} | ({"cluster"} if "cluster" in evidence.count_grains else set())
            ),
            populations=frozenset({"triggered"}),
            sequential_capture="none",
            sql_grains=frozenset(),
            scalar=replace(scalar, breakout_evidence=False) if scalar is not None else None,
        )

    _population: Literal["triggered"] = "triggered"

    def __init__(self, source: Any) -> None:
        source._validate_trigger_capability(operation="triggered_source")
        self._source = source
        self.capabilities: frozenset[Grain] = frozenset({"total"})
        self.breakouts = source.breakouts
        self.shape = source.shape

    @property
    def population(self) -> Literal["triggered"]:
        return "triggered"

    @property
    def trigger_name(self) -> str | None:
        return self.context.trigger_name

    @property
    def context(self) -> SourceContext:
        return self._source.context

    @property
    def source_name(self) -> str | None:
        return getattr(self._source, "source_name", None)

    def identity_assignment_evidence(self, request: AssignmentRequest) -> Any:
        """Delegate identity evidence to the underlying ledger-owning source."""
        operation = getattr(self._source, "identity_assignment_evidence", None)
        return operation(request) if callable(operation) else None

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        from increment._evidence_refusals import require_evidence

        require_evidence(self, grain=grain, by=by)
        return self._source.moments(
            metric,
            grain=grain,
            by=by,
            completed_windows_only=completed_windows_only,
            include_covariate=include_covariate,
            population="triggered",
        )

    def moments_for_method(
        self,
        metric: Metric,
        method: Any,
        *,
        by: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        from increment._evidence_refusals import require_evidence

        require_evidence(self, by=by)
        return self._source.moments_for_method(metric, method, by=by, population="triggered")

    def fit_cuped_multicovariate(
        self,
        metric: Metric,
        method: Any,
        *,
        control: str,
        arm_order: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        kwargs["population"] = "triggered"
        return self._source.fit_cuped_multicovariate(
            metric,
            method,
            control=control,
            arm_order=arm_order,
            **kwargs,
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
        from increment._evidence_refusals import require_evidence

        require_evidence(self, operation="sql")
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

    def readout_execution(self):
        return self._source.readout_execution()

    @contextmanager
    def readout_snapshot(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"],
        uptake_facts: Sequence[str] = (),
        include_breakouts: bool = False,
        breakouts: Sequence[Any] = (),
        configs: Sequence[ResolvedMetricConfig] | None = None,
        assignment_requests: Sequence[AssignmentRequest] = (),
    ) -> Iterator[EvidenceSource]:
        with self._source.readout_snapshot(
            metrics=metrics,
            population="assigned",
            uptake_facts=uptake_facts,
            include_breakouts=include_breakouts,
            breakouts=breakouts,
            configs=configs,
            assignment_requests=assignment_requests,
        ) as pinned:
            yield self if pinned is self._source else pinned.triggered_source()

    def evidence_scope_streams(self, scope: EvidenceScope) -> Mapping[str, str]:
        return self._source.evidence_scope_streams(scope)
