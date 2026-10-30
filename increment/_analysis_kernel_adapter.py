"""Adapt the legacy Analysis execution seam to the kernel resolver protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal, cast

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


def experiment_for_analysis(
    analysis: Any,
    *,
    run_options: dict[str, object] | None = None,
    record_schema: bool = False,
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
    return experiment, AnalysisResolver(analysis, experiment, run_options=run_options)


class AnalysisResolver:
    """Execute through the established Analysis implementation."""

    def __init__(
        self,
        analysis: Any,
        experiment: Experiment,
        *,
        run_options: dict[str, object] | None = None,
    ) -> None:
        self._analysis = analysis
        self._source = analysis._state.source
        self._experiment = experiment
        self._run_options = dict(run_options or {})
        self.catalog_writer: Callable[[Experiment], None] | None = None

    def resolve(self, source_refs: Sequence[SourceReference], ctx: object | None = None) -> Any:
        del source_refs, ctx
        return self._source

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
            or len(experiment.scopes) != 1
        )
        if invalid:
            self._unavailable(experiment, "declarations")
        scope = experiment.scopes[0]
        operation = scope_operation(experiment, scope)
        refusals = scope_refusals(experiment, scope)
        if refusals:
            raise refusals[0]
        if scope.view == "breakout" and tuple(
            identity.property for identity in scope.segment
        ) != tuple(getattr(self._source, "breakouts", ())):
            self._unavailable(experiment, "segment")
        kwargs = operation_options(experiment, scope, {**self._run_options, **(options or {})})
        capture = kwargs.pop("capture", None)
        if capture is not None:
            self._analysis.capture_sequential(**capture)
        execute = (
            self._analysis._run_legacy if operation == "run" else getattr(self._analysis, operation)
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
