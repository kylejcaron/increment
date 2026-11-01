"""Adapt the legacy Analysis execution seam to the kernel resolver protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic_core import PydanticSerializationError

from increment._binding import ResolverBinding
from increment._canonical import declaration_value
from increment._evidence_refusals import require_evidence
from increment._runnability import (
    context_payload,
    operation_options,
    operation_parameters,
    scope_operation,
    scope_refusals,
)
from increment._source_types import EvidenceCapabilities
from increment.errors import CodedError
from increment.kernel.contracts import Registration, Scope, contract_hash
from increment.kernel.experiment import Experiment, SourceReference


def _schema_for_source(source: Any) -> dict[str, str] | None:
    describe = getattr(source, "declaration_schema", None)
    return describe() if callable(describe) else None


def _source_digest(source: Any) -> str | None:
    identity = getattr(source, "declaration_identity", None)
    return (
        contract_hash("source_declaration_v1", declaration_value(identity()))
        if callable(identity)
        else None
    )


def _run_identity_hashes(
    experiment: Experiment, runtime_options: Mapping[str, object]
) -> tuple[str, str]:
    from increment.kernel.experiment import _json_value

    try:
        analysis_spec_hash = experiment.analysis_spec_hash
        execution_identity_hash = experiment.execution_identity_hash
    except CodedError as error:
        if error.code != "kernel.method.implementation_ref_missing":
            raise
        from increment._runnability import scope_operation

        analysis_spec_hash = contract_hash(
            "analysis_spec.runtime_only",
            {
                "experiment_id": experiment.experiment_id,
                "design": _json_value(experiment.design),
                "metrics": _json_value(experiment.metrics),
                "registration_hash": experiment.registration.registration_hash,
                "source_refs": [item.canonical_payload() for item in experiment.source_refs],
                "scopes": [item.canonical_payload() for item in experiment.scopes],
                "operations": {
                    scope.hash(): scope_operation(experiment, scope) for scope in experiment.scopes
                },
                "request_options_runtime": declaration_value(experiment.request_options),
            },
        )
        execution_identity_hash = contract_hash(
            "execution_identity",
            {
                "analysis_spec_hash": analysis_spec_hash,
                "implementations": [
                    {
                        "method_instance_id": key[0],
                        "learner_role": key[1],
                        "implementation": value.canonical_payload(),
                    }
                    for key, value in sorted(experiment.implementation_refs.items())
                ],
                "package_version": experiment.package_version,
            },
        )
    if runtime_options:
        try:
            request = _json_value(runtime_options)
        except CodedError as error:
            if error.code != "kernel.method.implementation_ref_missing":
                raise
            request = {"runtime_options": declaration_value(runtime_options)}
        except PydanticSerializationError:
            request = {"runtime_options": declaration_value(runtime_options)}
        analysis_spec_hash = contract_hash(
            "analysis_spec.run_options",
            {"analysis_spec_hash": analysis_spec_hash, "options": request},
        )
        execution_identity_hash = contract_hash(
            "execution_identity",
            {
                "analysis_spec_hash": analysis_spec_hash,
                "implementations": [
                    {
                        "method_instance_id": key[0],
                        "learner_role": key[1],
                        "implementation": value.canonical_payload(),
                    }
                    for key, value in sorted(experiment.implementation_refs.items())
                ],
                "package_version": experiment.package_version,
            },
        )
    return analysis_spec_hash, execution_identity_hash


def experiment_for_analysis(
    analysis: Any,
    *,
    run_options: dict[str, object] | None = None,
    record_schema: bool = False,
    return_run_result: bool = False,
) -> tuple[Experiment, AnalysisResolver]:
    if run_options is not None:
        run_options = {"population": None, **run_options}
    state = analysis._state
    source = state.source
    context = source.context
    from increment.decision import ContrastContext

    contrast = isinstance(context, ContrastContext)
    design = context.study if contrast else context.design
    registration = (
        Registration(
            registration_hash=contract_hash("registration", context.model_dump(mode="json"))
        )
        if contrast
        else Registration.from_plan(context.plan)
    )
    capabilities = getattr(source, "evidence", None)
    if not isinstance(capabilities, EvidenceCapabilities):
        capabilities = None
    kind = capabilities.kind if capabilities is not None else "switchback"
    scope = Scope(
        view="switchback" if contrast else "run",
        grain="total",
        population=cast(
            Literal["assigned", "triggered"], (run_options or {}).get("population") or "assigned"
        ),
    )
    metrics = {
        metric.name: metric.model_dump(mode="json")
        for metric in getattr(context, "metrics", ())
        if hasattr(metric, "name") and hasattr(metric, "model_dump")
    }
    experiment = Experiment(
        experiment_id=state.experiment_name,
        design=(design.model_dump(mode="json") if hasattr(design, "model_dump") else {}),
        metrics=metrics,
        registration=registration,
        source_refs=(
            SourceReference(
                name="source",
                uri=f"{kind}://{state.experiment_name}",
                expected_digest=_source_digest(source),
                schema=_schema_for_source(source) if record_schema else None,
                capabilities=capabilities,
            ),
        ),
        scopes=(scope,),
        analysis_context={
            **context_payload(source),
            "operation_parameters": operation_parameters(analysis),
        },
    )
    return experiment, AnalysisResolver(
        analysis, experiment, run_options=run_options, return_run_result=return_run_result
    )


class AnalysisResolver(ResolverBinding):
    """Execute through the established Analysis implementation."""

    def __init__(
        self,
        analysis: Any,
        experiment: Experiment,
        *,
        run_options: dict[str, object] | None = None,
        return_run_result: bool = False,
    ) -> None:
        super().__init__(
            analysis._binding,
            experiment,
            reopen_artifact=lambda _binding: analysis._ensure_artifact_source(),
        )
        self._analysis = analysis
        self._run_options = dict(run_options or {})
        self._return_run_result = return_run_result
        self._run_result_factory: Callable[..., Any] | None = None

    def set_run_result_factory(self, factory: Callable[..., Any]) -> None:
        self._run_result_factory = factory

    def resolve(self, source_refs: Sequence[SourceReference], ctx: object | None = None) -> Any:
        provider = super().resolve(source_refs, ctx)
        self._analysis._binding = self.binding
        return provider

    def run_experiment(
        self, experiment: Experiment, provider: object, *, options: dict[str, object] | None = None
    ) -> object:
        if provider is not self._source:
            self._unavailable(experiment, "declarations")
        planned = []
        for scope in experiment.scopes:
            operation = scope_operation(experiment, scope)
            refusals = scope_refusals(experiment, scope)
            if refusals:
                raise refusals[0]
            selection = tuple(identity.property for identity in scope.segment)
            if scope.view == "breakout" and selection != tuple(
                getattr(self._source, "breakouts", ())
            ):
                require_evidence(self._source, by=selection)
                self._unavailable(experiment, "segment")
            kwargs = operation_options(experiment, scope, {**self._run_options, **(options or {})})
            if self._return_run_result and kwargs.get("store") is not None:
                raise CodedError(
                    "Experiment execution does not support the Analysis.run store option",
                    code="kernel.run.invalid_input",
                    context={"field": "store"},
                )
            planned.append((scope, operation, kwargs))
        scoped_rows = None
        if len(planned) > 1:
            from increment._evidence_scopes import capture_requirements
            from increment.estimation.readout_types import ScopedReadout
            from increment.readouts._line2 import operation_scope, owned_execution_context

            requests = [
                operation_scope(self._analysis, operation, kwargs)
                for _, operation, kwargs in planned
            ]
            coherences = {kwargs.get("coherence", "captured") for _, _, kwargs in planned}
            if len(coherences) != 1:
                self._unavailable(experiment, "coherence")
            coherence = coherences.pop()
            scope_request = capture_requirements(self._source, requests)
            capture = next(
                (
                    kwargs["capture"]
                    for _, _, kwargs in planned
                    if kwargs.get("capture") is not None
                ),
                None,
            )
            needs_evidence = bool(
                capture is not None
                or scope_request.metrics
                or scope_request.configs
                or scope_request.uptake_facts
                or scope_request.assignment_requests
                or scope_request.breakouts
                or scope_request.include_breakouts
            )
            acquisition = (
                owned_execution_context(self._analysis, scope_request, coherence=coherence)
                if needs_evidence
                else nullcontext()
            )
            with acquisition:
                if capture is not None:
                    self._analysis.capture_sequential(**capture)
                scoped_rows = []
                for scope, operation, kwargs in planned:
                    kwargs.pop("capture", None)
                    execute = (
                        self._analysis._run_legacy
                        if operation == "run"
                        else getattr(self._analysis, operation)
                    )
                    single = replace(
                        experiment,
                        scopes=(scope,),
                        operations={scope.hash(): operation},
                    )
                    scope_hash, _ = _run_identity_hashes(single, kwargs)
                    scoped_rows.append(
                        ScopedReadout(
                            execute(**kwargs),
                            scope,
                            dict(kwargs),
                            scope_hash,
                        )
                    )
            result = scoped_rows[0].collection
        else:
            scope, operation, kwargs = planned[0]
            capture = kwargs.pop("capture", None)
            if capture is not None:
                self._analysis.capture_sequential(**capture)
            execute = (
                self._analysis._run_legacy
                if operation == "run"
                else getattr(self._analysis, operation)
            )
            result = execute(**kwargs)
        if self.catalog_writer is not None:
            schema = _schema_for_source(self._source)
            captured = replace(
                self._experiment,
                source_refs=tuple(
                    replace(reference, schema=schema) for reference in self._experiment.source_refs
                ),
            )
            self.catalog_writer(captured)
        if self._return_run_result:
            factory = self._run_result_factory
            if factory is None:
                self._unavailable(experiment, "RunResult adapter")
            assert factory is not None

            registration = experiment.registration
            context = self._analysis._state.source.context
            compiled_plan = (
                None if not hasattr(context, "plan") else self._analysis.compiled_decision_plan
            )
            identity_options = (
                kwargs
                if scoped_rows is None
                else {
                    "views": tuple(
                        {"scope": item.scope, "options": item.options} for item in scoped_rows
                    )
                }
            )
            analysis_spec_hash, execution_identity_hash = _run_identity_hashes(
                experiment, identity_options
            )
            return factory(
                self._analysis,
                result,
                registration,
                compiled_plan,
                experiment_id=experiment.experiment_id,
                analysis_spec_hash=analysis_spec_hash,
                execution_identity_hash=execution_identity_hash,
                run_id=f"run-{uuid4().hex}",
                declared_scope=scope,
                identity_options=kwargs,
                scoped_rows=scoped_rows,
            )
        return result

    @staticmethod
    def _unavailable(experiment: Experiment, field: str) -> None:
        raise CodedError(
            "The bound analysis cannot execute these experiment declarations",
            code="kernel.run.execution_unavailable",
            context={"experiment_id": experiment.experiment_id, "field": field},
        )


def run_analysis(analysis: Any, **options: object) -> object:
    """Execute a facade run through experiment resolution and verification."""
    from increment.kernel.resolver import run as run_experiment

    execution_context = options.get("_execution_context")
    experiment, resolver = experiment_for_analysis(analysis, run_options=options)
    return run_experiment(experiment, resolver, ctx=execution_context)
