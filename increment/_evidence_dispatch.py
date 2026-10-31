"""Evidence-family dispatch for arm moments and fixed switchback contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, NoReturn

from increment import readouts
from increment._analysis_config import UNSET, _Unset
from increment._literals import ValueScale
from increment._readout_request import ReadoutRequest
from increment._unit_cycle import _refuse as _unit_refuse
from increment.decision import ContrastDecisionProcedure, ContrastSource
from increment.declarations import Method
from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals
from increment.errors import refuse as _refuse
from increment.estimation.contrast import compute_contrast
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.inference import Prior
from increment.estimation.results import LiftEstimate
from increment.semantics.models import Metric
from increment.sources import EvidenceSource

if TYPE_CHECKING:
    from increment.kernel.context import EvidenceExecutionContext
_CONTRAST_REQUEST = RefusalSpec(
    "readout.contrast.request",
    InvalidRequestError,
    lambda *, message, **_: message,
)


def _reject_contrast_request(*, message: str, **context: object) -> NoReturn:
    _refuse(_CONTRAST_REQUEST, message=message, **context)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.evidence_dispatch.contrast_handler.readout_request_no": "contrast readout request has no procedure for metric {metric_name!r}",
    },
)
_raise = raiser(_REFUSALS)


@dataclass(frozen=True, slots=True)
class ContrastReadoutRequest:
    """Fixed-horizon contrast inputs, one immutable procedure per metric."""

    metrics: tuple[Metric, ...]
    procedures: Mapping[str, ContrastDecisionProcedure]

    def __post_init__(self) -> None:
        metric_names = [metric.name for metric in self.metrics]
        duplicate_names = sorted({name for name in metric_names if metric_names.count(name) > 1})
        procedure_map = dict(self.procedures)
        expected = set(metric_names)
        actual = set(procedure_map)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        mismatches = sorted(
            f"{key!r}->{procedure.metric!r}"
            for key, procedure in procedure_map.items()
            if procedure.metric != key
        )
        if duplicate_names or missing or extra or mismatches:
            _reject_contrast_request(
                message=(
                    "contrast readout request must contain exactly one procedure per metric"
                    f" (duplicates={duplicate_names!r}, missing={missing!r}, "
                    f"extra={extra!r}, mismatches={mismatches!r})"
                ),
                duplicates=duplicate_names,
                missing=missing,
                extra=extra,
                mismatches=mismatches,
            )
        object.__setattr__(self, "procedures", MappingProxyType(procedure_map))


@dataclass(frozen=True, slots=True)
class ArmMomentHandler:
    """Run the existing parallel arm-moment readout family."""

    family: ClassVar[Literal["arm_moments"]] = "arm_moments"
    decision_method: Method | _Unset = UNSET
    sensitivity_methods: Sequence[Method] | _Unset = UNSET
    prior: Prior | None | _Unset = UNSET
    estimands: tuple[str, ...] | None = None
    value_scale: Mapping[str, ValueScale] | None = None

    def run(self, source: EvidenceSource, request: ReadoutRequest) -> list[LiftEstimate]:
        return readouts.arm_moments(
            source,
            decision_method=self.decision_method,
            sensitivity_methods=self.sensitivity_methods,
            metrics=[metric.name for metric in request.metrics],
            prior=self.prior,
            estimands=self.estimands,
            value_scale=self.value_scale,
            population=request.population,
        )


@dataclass(frozen=True, slots=True)
class ContrastHandler:
    """Estimate one fixed additive contrast per selected metric."""

    family: ClassVar[Literal["contrast"]] = "contrast"

    @staticmethod
    def preflight(source: ContrastSource, request: ContrastReadoutRequest) -> None:
        """Validate all contrast procedures before any evidence is acquired."""
        for metric in request.metrics:
            try:
                procedure = request.procedures[metric.name]
            except KeyError:
                _raise(
                    "facade.evidence_dispatch.contrast_handler.readout_request_no",
                    metric_name=metric.name,
                )
            if (
                source.context.study.assignment.sequence.scheme == "independent_bernoulli_order"
                and procedure.reference is None
            ):
                _unit_refuse(
                    "unit_cycle.envelope_required",
                    reason="unit_cycle_reference_required",
                    metric=metric.name,
                )

    def run(
        self,
        source: ContrastSource,
        request: ContrastReadoutRequest,
        *,
        execution_context: EvidenceExecutionContext,
    ) -> ContrastResults:
        from increment._evidence_loaders import load_contrast_stats
        from increment._evidence_types import ContrastStatsRequest
        from increment._source_identity import source_identity
        from increment.readouts._contrast_scope import scope_contrast_results

        self.preflight(source, request)
        results = []
        stats = {}
        for metric in request.metrics:
            procedure = request.procedures[metric.name]
            stats[metric.name] = load_contrast_stats(
                source,
                ContrastStatsRequest(metric),
                execution_context=execution_context,
            )
            results.append(compute_contrast(stats[metric.name], procedure).results[0])
        rows = scope_contrast_results(
            results,
            metrics=request.metrics,
            stats=stats,
            procedures=request.procedures,
            source_identity_record=source_identity(source),
            coherence=(
                None
                if "provider" in execution_context.effective_modes.values()
                else execution_context.coherence
            ),
            effective_modes=execution_context.effective_modes,
        )
        if execution_context.shared_capture and rows.metadata is not None:
            from increment.estimation.readout_types import ReadoutCapture

            rows._metadata = rows.metadata.model_copy(
                update={
                    "captures": (
                        ReadoutCapture(
                            snapshot_id=execution_context.run_id,
                            coherence=execution_context.coherence,
                            effective_modes=execution_context.effective_modes,
                        ),
                    ),
                }
            )
        return rows


@dataclass(frozen=True, slots=True)
class ContrastEvidenceOperation:
    """Own one switchback evidence lifecycle and dispatch its contrast readout."""

    def run(
        self,
        source: ContrastSource,
        request: ContrastReadoutRequest,
        *,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> ContrastResults:
        handler = ContrastHandler()
        handler.preflight(source, request)
        from increment._evidence_scopes import (
            switchback_scope_provider,
            validate_switchback_requests,
        )
        from increment._evidence_types import EvidenceScope

        if execution_context is not None:
            from increment._evidence_types import ContrastStatsRequest

            stats_requests = tuple(ContrastStatsRequest(metric) for metric in request.metrics)
            validate_switchback_requests(source, stats_requests, execution_context)
            return handler.run(source, request, execution_context=execution_context)

        from uuid import uuid4

        from increment.kernel.context import EvidenceExecutionContext

        owner = EvidenceExecutionContext(
            provider=switchback_scope_provider(source),
            scope=EvidenceScope(
                grain="total",
                population="assigned",
                metrics=request.metrics,
            ),
            run_id=uuid4().hex,
        )
        with owner.evidence_scope():
            return handler.run(source, request, execution_context=owner)


__all__ = [
    "ArmMomentHandler",
    "ContrastEvidenceOperation",
    "ContrastHandler",
    "ContrastReadoutRequest",
]
