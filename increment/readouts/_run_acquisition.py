from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from increment._analysis_config import UNSET, _Unset, effective_methods
from increment._evidence_types import EvidenceScope
from increment._literals import ValueScale
from increment.declarations import Method
from increment.estimation.inference import Prior
from increment.readouts._requests import (
    _design_compliance_only,
    _prepare_run_request,
    _sequential_compliance_requested,
)
from increment.sources import EvidenceSource

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment.kernel.context import EvidenceExecutionContext
    from increment.semantics.models import Metric


@dataclass(frozen=True, slots=True)
class RunAcquisitionRequest:
    """One complete typed request for declared or combined whole-window reads."""

    source: EvidenceSource
    decision_method: Method | _Unset = UNSET
    sensitivity_methods: Sequence[Method] | _Unset = UNSET
    prior: Prior | None | _Unset = UNSET
    metrics: Sequence[str] | None = None
    by: Sequence[str] = ()
    estimands: Sequence[str] | None = None
    value_scale: Mapping[str, ValueScale] | None = None
    request_population: Literal["assigned", "triggered"] = "assigned"
    design: Any | None = None
    family_population: Literal["assigned", "triggered"] | None = None
    family_populations: tuple[Literal["assigned", "triggered"], ...] = ()
    family_horizon: tuple[Metric, ...] | None = None
    family_scope_metrics: tuple[Metric, ...] | None = None
    family_uptake_facts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedRunAcquisition:
    """One validated whole-window request and its complete typed read scope."""

    selected: tuple[Metric, ...]
    configs: tuple[ResolvedMetricConfig, ...]
    scope: EvidenceScope


class RunAcquisitionOwner:
    """Prepare and own the complete evidence acquisition for whole-window runs."""

    @staticmethod
    def provider_for(source: EvidenceSource) -> object:
        """Select the evidence adapter at the acquisition boundary."""
        from increment.decision import ContrastContext
        from increment.readouts._evidence import is_custom_source

        if isinstance(source.context, ContrastContext):
            from increment._evidence_scopes import switchback_scope_provider

            return switchback_scope_provider(source)
        if is_custom_source(source):
            return source
        from increment.readouts._provider import native_scope_provider

        return native_scope_provider(source)

    @staticmethod
    def prepare(request: RunAcquisitionRequest) -> PreparedRunAcquisition:
        """Resolve configs and build a complete single-run or family scope."""
        source = request.source
        selected, configs = _prepare_run_request(
            source,
            decision_method=request.decision_method,
            sensitivity_methods=request.sensitivity_methods,
            prior=request.prior,
            metrics=request.metrics,
            by=request.by,
            estimands=request.estimands,
            value_scale=request.value_scale,
            population=request.request_population,
        )
        if request.design is not None:
            from increment.readouts._design_scope import _config_snapshot

            for config in configs:
                _config_snapshot(config, request.design)

        if request.family_population is None:
            scope = RunAcquisitionOwner._single_scope(
                source,
                selected,
                configs,
                estimands=request.estimands,
                population=request.request_population,
            )
        else:
            compliance_only = _design_compliance_only(source.context, request.estimands)
            sequential_compliance = _sequential_compliance_requested(
                source.context, request.estimands
            )
            omit_outcomes = compliance_only or sequential_compliance
            scope = EvidenceScope(
                metrics=(
                    ()
                    if omit_outcomes
                    else tuple(selected)
                    if request.family_scope_metrics is None
                    else request.family_scope_metrics
                ),
                horizon_metrics=() if omit_outcomes else request.family_horizon,
                population=request.family_population,
                populations=request.family_populations,
                grain="total",
                estimands=(None if request.estimands is None else tuple(request.estimands)),
                configs=() if omit_outcomes else tuple(configs),
                snapshot=True,
                execution=True,
                include_breakouts=False,
            )
        return PreparedRunAcquisition(tuple(selected), tuple(configs), scope)

    @staticmethod
    def _single_scope(
        source: EvidenceSource,
        selected: Sequence[Metric],
        configs: Sequence[ResolvedMetricConfig],
        *,
        estimands: Sequence[str] | None,
        population: Literal["assigned", "triggered"],
    ) -> EvidenceScope:
        from increment.estimation.sequential import SEQUENTIAL_POLICIES
        from increment.readouts._evidence import has_verified_sequential_checkpoint

        design = source.context.design
        design_summary_only = _design_compliance_only(source.context, estimands)
        sequential = isinstance(source.context.plan.inference, SEQUENTIAL_POLICIES)
        needs_live_evidence = (bool(selected) or design_summary_only) and not sequential
        if design_summary_only:
            needs_snapshot = False
        else:
            needs_snapshot = sequential or (
                needs_live_evidence
                and (
                    (design is not None and design.mechanism == "observational")
                    or (
                        design is not None
                        and any(
                            method.variance_reduction in ("cuped", "cuped_multicovariate")
                            for config in configs
                            for method in effective_methods(config, design=design)
                        )
                    )
                    or any(
                        getattr(getattr(metric, "winsorization", None), "has_percentile", False)
                        for metric in selected
                    )
                )
            )
        capture_all_family = sequential and not design_summary_only
        uptake = getattr(source.context.design, "uptake", None)
        return EvidenceScope(
            metrics=(tuple(source.context.metrics) if capture_all_family else tuple(selected)),
            horizon_metrics=() if design_summary_only else None,
            population=population,
            configs=(tuple(source.context.configs) if capture_all_family else tuple(configs)),
            uptake_facts=() if uptake is None else (uptake.fact,),
            snapshot=needs_snapshot,
            execution=bool(selected) or design_summary_only,
            checkpoint_only=(
                sequential
                and not design_summary_only
                and bool(selected)
                and has_verified_sequential_checkpoint(source)
            ),
        )

    @staticmethod
    def combined_scope(
        source: EvidenceSource,
        requests: Sequence[PreparedRunAcquisition],
        *,
        additional_scopes: Sequence[EvidenceScope] = (),
    ) -> EvidenceScope:
        """Combine prepared typed requests through the shared evidence inventory."""
        from increment._evidence_scopes import capture_requirements

        return capture_requirements(
            source,
            (*tuple(item.scope for item in requests), *additional_scopes),
        )

    @staticmethod
    def capture(
        source: EvidenceSource,
        requests: Sequence[PreparedRunAcquisition],
        *,
        run_id: str,
        coherence: Literal["captured", "live"],
        shared_capture: bool = False,
        execution_context_scope: Callable[[EvidenceExecutionContext], Any] | None = None,
    ) -> AbstractContextManager[EvidenceExecutionContext]:
        """Open one source-owned context over the combined prepared scopes."""
        from contextlib import contextmanager

        from increment.kernel.context import EvidenceExecutionContext

        @contextmanager
        def owned() -> Iterator[EvidenceExecutionContext]:
            context = EvidenceExecutionContext(
                RunAcquisitionOwner.provider_for(source),
                RunAcquisitionOwner.combined_scope(source, requests),
                run_id=run_id,
                coherence=coherence,
                shared_capture=shared_capture,
            )
            ambient = (
                nullcontext()
                if execution_context_scope is None
                else execution_context_scope(context)
            )
            with context.evidence_scope(), ambient:
                yield context

        return owned()

    @staticmethod
    def execute[ResultT](
        source: EvidenceSource,
        requests: Sequence[PreparedRunAcquisition],
        *,
        run_id: str,
        coherence: Literal["captured", "live"],
        shared_capture: bool,
        executor: Callable[[EvidenceExecutionContext], ResultT],
        execution_context_scope: Callable[[EvidenceExecutionContext], Any] | None = None,
    ) -> ResultT:
        """Own the combined acquisition lifecycle and invoke its readout handler."""
        with RunAcquisitionOwner.capture(
            source,
            requests,
            run_id=run_id,
            coherence=coherence,
            shared_capture=shared_capture,
            execution_context_scope=execution_context_scope,
        ) as context:
            return executor(context)
