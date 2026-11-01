from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self

from increment._canonical import declaration_value
from increment.decision import AnalysisState, ContrastContext, DefinitionsArmAnalysisState
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from increment._evidence_types import BreakoutIdentity
    from increment._literals import Backend
    from increment.kernel.experiment import Experiment, SourceReference
    from increment.kernel.resolver import Resolver
    from increment.query.artifact_contract import ArtifactContext, ArtifactStore
    from increment.semantics.artifact import UnitDayArtifactRef
    from increment.semantics.models import Breakout


@dataclass(frozen=True, slots=True)
class _ArtifactOpenSpec:
    store: ArtifactStore
    ref: UnitDayArtifactRef
    expected_context: ArtifactContext
    verification: Literal["lazy_digest"]


_UNDECLARED_BREAKOUT = RefusalSpec(
    "facade.analysis.undeclared_breakout",
    InvalidRequestError,
    template="breakout {requested!r} is not declared by this experiment; declared: {declared!r}",
)


@dataclass(frozen=True, slots=True)
class SourceBinding:
    state: AnalysisState
    artifact_open: _ArtifactOpenSpec | None = None
    input_backend: Backend | None = None
    owns_source: bool = True
    selected_breakouts: tuple[Breakout, ...] | None = None

    @property
    def source(self) -> Any:
        return self.state.source

    @property
    def context(self) -> Any:
        return self.source.context

    @property
    def definitions(self) -> Any:
        return (
            self.state.definitions if isinstance(self.state, DefinitionsArmAnalysisState) else None
        )

    @property
    def plan(self) -> Any:
        return self.context.plan

    @property
    def design(self) -> Any:
        context = self.context
        return (
            context.study.identification if isinstance(context, ContrastContext) else context.design
        )

    @property
    def metrics(self) -> list[Any]:
        return list(self.context.metrics)

    @property
    def experiment(self) -> Any:
        if isinstance(self.state, DefinitionsArmAnalysisState):
            return self.state.experiment
        artifact_experiment = getattr(self.source, "artifact_experiment", None)
        if artifact_experiment is not None:
            if self.selected_breakouts is None:
                return artifact_experiment
            selected = tuple(
                item for item in artifact_experiment.breakouts if item in self.selected_breakouts
            )
            return artifact_experiment.model_copy(update={"breakouts": selected})
        raise AttributeError("seam analysis has no Experiment definition")

    @property
    def declared_breakouts(self) -> tuple[Breakout, ...]:
        """The breakouts this binding reads: declared definitions, else the source's dimensions."""
        from increment._source_types import declared_evidence
        from increment.semantics.models import Breakout

        if isinstance(self.state, DefinitionsArmAnalysisState):
            definitions = tuple(self.state.experiment.breakouts)
        else:
            artifact_experiment = getattr(self.source, "artifact_experiment", None)
            definitions = (
                () if artifact_experiment is None else tuple(artifact_experiment.breakouts)
            )
        if not definitions:
            definitions = tuple(
                Breakout(property=name) for name in declared_evidence(self.source).breakouts
            )
        if self.selected_breakouts is None:
            return definitions
        return tuple(item for item in definitions if item in self.selected_breakouts)

    @property
    def breakout_selection(self) -> tuple[BreakoutIdentity, ...] | None:
        """Selected breakout identities, or None when every declared breakout is read."""
        if self.selected_breakouts is None:
            return None
        from increment._evidence_types import BreakoutIdentity

        return tuple(
            BreakoutIdentity(item.property, item.source) for item in self.declared_breakouts
        )

    @property
    def session(self) -> Any:
        return self.state.session if isinstance(self.state, DefinitionsArmAnalysisState) else None

    @property
    def connection(self) -> Any:
        return (
            self.state.connection if isinstance(self.state, DefinitionsArmAnalysisState) else None
        )

    @property
    def store(self) -> Literal["auto", "always", "none"]:
        return self.state.store if isinstance(self.state, DefinitionsArmAnalysisState) else "none"

    @property
    def backend(self) -> str | None:
        return (
            self.state.backend or None
            if isinstance(self.state, DefinitionsArmAnalysisState)
            else None
        )

    @property
    def on_mixed_assignment(self) -> Literal["error", "warn", "exclude"]:
        return (
            self.state.on_mixed_assignment
            if isinstance(self.state, DefinitionsArmAnalysisState)
            else "error"
        )

    @property
    def experiment_name(self) -> str:
        return self.state.experiment_name

    @property
    def _src(self) -> Any:
        return self.source

    @property
    def _context(self) -> Any:
        return self.context

    @property
    def _defs(self) -> Any:
        return self.definitions

    @property
    def _plan(self) -> Any:
        return self.plan

    @property
    def _design(self) -> Any:
        return self.design

    @property
    def _metrics(self) -> list[Any]:
        return self.metrics

    @property
    def _experiment(self) -> Any:
        return self.experiment

    @property
    def _session(self) -> Any:
        return self.session

    @property
    def _con(self) -> Any:
        return self.connection

    @property
    def _store(self) -> Literal["auto", "always", "none"]:
        return self.store

    @property
    def _backend(self) -> str | None:
        return self.backend

    @property
    def _on_mixed_assignment(self) -> Literal["error", "warn", "exclude"]:
        return self.on_mixed_assignment

    def reopened(self, source: Any) -> SourceBinding:
        return replace(self, state=replace(self.state, source=source), owns_source=True)

    def with_source(self, source: Any) -> SourceBinding:
        return replace(
            self,
            state=replace(self.state, source=source),
            artifact_open=None,
            owns_source=False,
        )

    def with_breakouts(self, breakouts: tuple[Breakout, ...]) -> SourceBinding:
        declared = self.declared_breakouts
        if any(item not in declared for item in breakouts):
            requested = tuple((item.source, item.property) for item in breakouts)
            refuse(
                _UNDECLARED_BREAKOUT,
                requested=requested,
                declared=tuple((item.source, item.property) for item in declared),
            )
        selected = tuple(item for item in declared if item in breakouts)
        if not isinstance(self.state, DefinitionsArmAnalysisState):
            return replace(self, selected_breakouts=selected, owns_source=False)
        experiment = self.state.experiment.model_copy(update={"breakouts": selected})
        return replace(
            self,
            state=replace(self.state, experiment=experiment),
            owns_source=False,
        )


def _execution_unavailable(experiment: Experiment, field: str) -> NoReturn:
    raise CodedError(
        "The bound analysis cannot execute these experiment declarations",
        code="kernel.run.execution_unavailable",
        context={"experiment_id": experiment.experiment_id, "field": field},
    )


def _validate_experiment(experiment: Experiment, declared: Experiment) -> None:
    invalid = (
        experiment.experiment_id != declared.experiment_id
        or experiment.design != declared.design
        or experiment.registration != declared.registration
        or declaration_value(experiment.analysis_context)
        != declaration_value(declared.analysis_context)
        or any(declared.metrics.get(name) != metric for name, metric in experiment.metrics.items())
    )
    if invalid:
        _execution_unavailable(experiment, "declarations")


def _schema_for_source(source: Any) -> dict[str, str] | None:
    describe = getattr(source, "declaration_schema", None)
    return describe() if callable(describe) else None


def _source_digest(source: Any) -> str | None:
    from increment.kernel.contracts import contract_hash

    identity = getattr(source, "declaration_identity", None)
    return (
        contract_hash("source_declaration_v1", declaration_value(identity()))
        if callable(identity)
        else None
    )


def _run_identity_hashes(
    experiment: Experiment, runtime_options: Mapping[str, object]
) -> tuple[str, str]:
    from pydantic_core import PydanticSerializationError

    from increment._runnability import scope_operation
    from increment.kernel.contracts import contract_hash
    from increment.kernel.experiment import _json_value

    try:
        analysis_spec_hash = experiment.analysis_spec_hash
        execution_identity_hash = experiment.execution_identity_hash
    except CodedError as error:
        if error.code != "kernel.method.implementation_ref_missing":
            raise
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


def experiment_for_binding(
    binding: SourceBinding,
    *,
    operation_parameters: Mapping[str, object],
    record_schema: bool = False,
    population: Literal["assigned", "triggered"] = "assigned",
) -> Experiment:
    """Declare the portable experiment a source binding executes."""
    from increment._runnability import context_payload
    from increment._source_types import EvidenceCapabilities
    from increment.kernel.contracts import Registration, Scope, contract_hash
    from increment.kernel.experiment import Experiment, SourceReference

    state = binding.state
    source = state.source
    context = source.context
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
        population=population,
    )
    metrics = {
        metric.name: metric.model_dump(mode="json")
        for metric in getattr(context, "metrics", ())
        if hasattr(metric, "name") and hasattr(metric, "model_dump")
    }
    return Experiment(
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
            "operation_parameters": operation_parameters,
        },
    )


class ResolverBinding:
    def __init__(
        self,
        binding: SourceBinding,
        experiment: Experiment,
        catalog_writer: Callable[[Experiment], None] | None = None,
        reopen_artifact: Callable[[SourceBinding], object] | None = None,
    ) -> None:
        self.binding = binding
        self.experiment = experiment
        self.catalog_writer = catalog_writer
        self._reopen_artifact = reopen_artifact
        self._closed = False

    @property
    def _source(self) -> Any:
        return self.binding.state.source

    @property
    def _experiment(self) -> Experiment:
        return self.experiment

    def validate_experiment(self, experiment: Experiment) -> None:
        _validate_experiment(experiment, self.experiment)

    def resolve(self, source_refs: Sequence[SourceReference], ctx: object | None = None) -> object:
        del source_refs, ctx
        source = self.binding.state.source
        if self.binding.artifact_open is not None and getattr(source, "closed", False):
            if self._reopen_artifact is None:
                return source
            source = self._reopen_artifact(self.binding)
            self.binding = self.binding.reopened(source)
        self._closed = False
        return source

    def verify(self, source_refs: Sequence[SourceReference], provider: object) -> list[CodedError]:
        failures = []
        if provider is not self.binding.state.source or tuple(
            (r.name, r.uri) for r in source_refs
        ) != tuple((r.name, r.uri) for r in self.experiment.source_refs):
            return [
                CodedError(
                    "Source references do not identify the bound provider",
                    code="kernel.resolver.digest_mismatch",
                    context={"source_refs": tuple(r.name for r in source_refs)},
                )
            ]
        identity = getattr(provider, "declaration_identity", None)
        from increment.kernel.contracts import contract_hash

        actual_digest = (
            contract_hash("source_declaration_v1", declaration_value(identity()))
            if callable(identity)
            else None
        )
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
            if reference.capabilities is not None and reference.capabilities != capabilities:
                failures.append(
                    CodedError(
                        "Source declarations do not match the bound provider",
                        code="kernel.resolver.capability_mismatch",
                        context={"source_ref": reference.name},
                    )
                )
        return failures

    def close(self) -> None:
        if self._closed:
            return
        if self.binding.owns_source:
            source = self.binding.state.source
            close = getattr(source, "close", None)
            if callable(close):
                close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def binding_for(experiment: Experiment, resolver: Resolver) -> SourceBinding:
    from increment.kernel.resolver import _resolve_verified

    _resolve_verified(experiment, resolver)
    if not isinstance(resolver, ResolverBinding):
        from increment.kernel.contracts import _refuse

        _refuse("kernel.run.provider_execution_unavailable")
    return resolver.binding
