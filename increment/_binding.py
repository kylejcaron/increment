from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Self

from increment._canonical import declaration_value
from increment.decision import AnalysisState, ContrastContext, DefinitionsArmAnalysisState
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
            return artifact_experiment
        raise AttributeError("seam analysis has no Experiment definition")

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
        if not isinstance(self.state, DefinitionsArmAnalysisState):
            raise AttributeError("breakouts can only be narrowed on a definitions binding")
        declared = self.state.experiment.breakouts
        if any(item not in declared for item in breakouts):
            requested = tuple((item.source, item.property) for item in breakouts)
            refuse(
                _UNDECLARED_BREAKOUT,
                requested=requested,
                declared=tuple((item.source, item.property) for item in declared),
            )
        experiment = self.state.experiment.model_copy(update={"breakouts": breakouts})
        return replace(
            self,
            state=replace(self.state, experiment=experiment),
            owns_source=False,
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
        raise CodedError(
            "The bound analysis cannot execute these experiment declarations",
            code="kernel.run.execution_unavailable",
            context={"experiment_id": experiment.experiment_id, "field": "declarations"},
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
