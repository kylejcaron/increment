"""Adapt the legacy Analysis execution seam to the kernel resolver protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic_core import PydanticSerializationError

from increment._canonical import declaration_value
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


class AnalysisResolver:
    """Execute through the established Analysis implementation."""

    def __init__(
        self,
        analysis: Any,
        experiment: Experiment,
        *,
        run_options: dict[str, object] | None = None,
        return_run_result: bool = False,
    ) -> None:
        self._analysis = analysis
        self._source = analysis._state.source
        self._experiment = experiment
        self._run_options = dict(run_options or {})
        self._return_run_result = return_run_result
        self._run_result_factory: Callable[..., Any] | None = None
        self.catalog_writer: Callable[[Experiment], None] | None = None

    def resolve(self, source_refs: Sequence[SourceReference], ctx: object | None = None) -> Any:
        del source_refs, ctx
        return self._source

    def set_run_result_factory(self, factory: Callable[..., Any]) -> None:
        self._run_result_factory = factory

    def verify(self, source_refs: Sequence[SourceReference], provider: object) -> list[CodedError]:
        failures = []
        if provider is not self._source or tuple(
            (reference.name, reference.uri) for reference in source_refs
        ) != tuple((reference.name, reference.uri) for reference in self._experiment.source_refs):
            return [
                CodedError(
                    "Source references do not identify the bound provider",
                    code="kernel.resolver.digest_mismatch",
                    context={"source_refs": tuple(reference.name for reference in source_refs)},
                )
            ]
        actual_digest = _source_digest(provider)
        for reference in source_refs:
            if reference.expected_digest is not None and reference.expected_digest != actual_digest:
                failures.append(
                    CodedError(
                        "Source declarations do not match the bound provider",
                        code="kernel.resolver.digest_mismatch",
                        context={
                            "source_ref": reference.name,
                            "expected": reference.expected_digest,
                            "actual": actual_digest,
                        },
                    )
                )
        capabilities = getattr(provider, "evidence", None)
        for reference in source_refs:
            if reference.capabilities is not None and reference.capabilities != capabilities:
                failures.append(
                    CodedError(
                        "Source declarations do not match the bound provider",
                        code="kernel.resolver.capability_mismatch",
                        context={"source_ref": reference.name},
                    )
                )
        return failures

    def capture_sequential(
        self,
        *,
        finalized: bool,
        previous: Any | None = None,
        as_of: Any | None = None,
        freeze: Sequence[str] = (),
    ) -> Any:
        """Capture or adopt a sequential checkpoint on the bound analysis."""
        return self._analysis.capture_sequential(
            finalized=finalized, previous=previous, as_of=as_of, freeze=freeze
        )

    def run_experiment(
        self, experiment: Experiment, provider: object, *, options: dict[str, object] | None = None
    ) -> object:
        declared = self._experiment
        invalid = (
            provider is not self._source
            or experiment.experiment_id != declared.experiment_id
            or experiment.design != declared.design
            or experiment.registration != declared.registration
            or declaration_value(experiment.analysis_context)
            != declaration_value(declared.analysis_context)
            or any(
                declared.metrics.get(name) != metric for name, metric in experiment.metrics.items()
            )
        )
        if invalid:
            self._unavailable(experiment, "declarations")
        planned = []
        for scope in experiment.scopes:
            operation = scope_operation(experiment, scope)
            refusals = scope_refusals(experiment, scope)
            if refusals:
                raise refusals[0]
            if scope.view == "breakout" and tuple(
                identity.property for identity in scope.segment
            ) != tuple(getattr(self._source, "breakouts", ())):
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
