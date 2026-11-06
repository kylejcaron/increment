from __future__ import annotations

import reprlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self

from increment._canonical import declaration_value
from increment._execute import execute_views, operation_parameters, selected_binding
from increment._runnability import operation_options, scope_refusals
from increment.decision import AnalysisState, ContrastContext, DefinitionsArmAnalysisState
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from increment._evidence_types import BreakoutIdentity, SequentialCaptureRequest
    from increment._literals import Backend
    from increment.estimation.readout_types import ScopedReadout
    from increment.kernel.context import EvidenceExecutionContext
    from increment.kernel.experiment import DeclarationValue, Experiment, SourceReference
    from increment.semantics.models import Breakout


_UNDECLARED_BREAKOUT = RefusalSpec(
    "facade.analysis.undeclared_breakout",
    InvalidRequestError,
    template="breakout {requested!r} is not declared by this experiment; declared: {declared!r}",
)


@dataclass(frozen=True, slots=True)
class SourceBinding:
    state: AnalysisState
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

    def with_source(self, source: Any) -> SourceBinding:
        return replace(self, state=replace(self.state, source=source), owns_source=False)

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


def _differing_field(experiment: Experiment, declared: Experiment) -> str | None:
    """The first declaration in which an experiment departs from its reader's experiment."""
    if experiment.experiment_id != declared.experiment_id:
        return "experiment_id"
    if experiment.design != declared.design:
        return "design"
    if experiment.registration != declared.registration:
        return "registration"
    if declaration_value(experiment.analysis_context) != declaration_value(
        declared.analysis_context
    ):
        return "analysis_context"
    if any(declared.metrics.get(name) != metric for name, metric in experiment.metrics.items()):
        return "metrics"
    return None


def _validate_experiment(experiment: Experiment, declared: Experiment) -> None:
    """Refuse an experiment the reader was not built for; views and a metric subset are free."""
    field = _differing_field(experiment, declared)
    if field is not None:
        raise CodedError(
            f"Experiment {experiment.experiment_id!r} differs in {field} from the experiment "
            f"this reader was built for ({declared.experiment_id!r}); pass the experiment the "
            "builder returned with this reader, or build a new reader for this experiment",
            code="kernel.run.execution_unavailable",
            context={
                "experiment_id": experiment.experiment_id,
                "reader_experiment_id": declared.experiment_id,
                "field": field,
            },
        )


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


def experiment_for_binding(
    binding: SourceBinding,
    *,
    operation_parameters: Mapping[str, DeclarationValue],
    record_schema: bool = False,
    population: Literal["assigned", "triggered"] = "assigned",
) -> Experiment:
    """Declare the portable experiment a source binding executes."""
    from increment._runnability import context_payload
    from increment._source_types import EvidenceCapabilities
    from increment.kernel.contracts import Registration, View, contract_hash
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
    scope = View(
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
        views=(scope,),
        analysis_context={
            **context_payload(source),
            "operation_parameters": operation_parameters,
        },
    )


# How each builder's source reads, by the scheme of its recorded source reference.
_READS: Mapping[str, str] = MappingProxyType(
    {
        "unit_summary": "dataframe snapshot",
        "unit_panel": "dataframe snapshot",
        "switchback": "dataframe snapshot",
        "definitions": "warehouse, read live",
        "artifact": "artifact",
        "moments": "moments file",
    }
)


class ExperimentReader:
    """Read the data behind one experiment for ``run`` and the functions that read data.

    A builder returns it with the experiment. A dataframe reader reads the frame it
    was built from, a warehouse reader reads the warehouse live on every call, and an
    offline reader restored from a saved catalog cannot read. Use it in a ``with``
    block or call ``close()``; afterwards every read refuses, so build a new reader to
    read again. Closing never closes a connection you supplied.
    """

    def __init__(
        self,
        binding: SourceBinding | None,
        experiment: Experiment,
        *,
        catalog_writer: Callable[[Experiment], None] | None = None,
        definitions_guard: Callable[[], None] | None = None,
    ) -> None:
        self._bound = binding
        self._experiment = experiment
        self._catalog_writer = catalog_writer
        self._definitions_guard = definitions_guard
        self._closed = False

    @property
    def experiment(self) -> Experiment:
        """The experiment this reader was built with."""
        return self._experiment

    @property
    def closed(self) -> bool:
        """Whether the reader is closed; a closed reader refuses every read."""
        return self._closed

    @property
    def _binding(self) -> SourceBinding:
        from increment.kernel.contracts import _refuse

        if self._closed:
            _refuse("kernel.run.reader_closed", experiment_id=self._experiment.experiment_id)
        if self._bound is None:
            _refuse("kernel.run.reader_offline", experiment_id=self._experiment.experiment_id)
        return self._bound

    @property
    def _source(self) -> Any:
        return self._binding.state.source

    def _validate_experiment(self, experiment: Experiment) -> None:
        _validate_experiment(experiment, self._experiment)

    def _resolve(self, source_refs: Sequence[SourceReference], ctx: object | None = None) -> object:
        """The one acquisition step every read takes, refused after close, offline or on edits."""
        del source_refs, ctx
        binding = self._binding
        if self._definitions_guard is not None:
            self._definitions_guard()
        return binding.state.source

    def _verify(self, source_refs: Sequence[SourceReference], provider: object) -> list[CodedError]:
        from increment._source_types import provider_refusal

        failures = []
        if provider is not self._binding.state.source or tuple(
            (r.name, r.uri) for r in source_refs
        ) != tuple((r.name, r.uri) for r in self._experiment.source_refs):
            return [
                provider_refusal(
                    "kernel.provider.digest_mismatch",
                    source_refs=tuple(r.name for r in source_refs),
                    detail="the source references do not name this reader's source",
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
                    provider_refusal(
                        "kernel.provider.digest_mismatch",
                        source_refs=(reference.name,),
                        detail=(
                            f"recorded digest {reference.expected_digest!r}, "
                            f"source digest {actual_digest!r}"
                        ),
                    )
                )
            capabilities = getattr(provider, "evidence", None)
            if reference.capabilities is not None and reference.capabilities != capabilities:
                failures.append(
                    provider_refusal(
                        "kernel.provider.capability_mismatch", source_ref=reference.name
                    )
                )
        return failures

    def _run_experiment(
        self,
        experiment: Experiment,
        provider: object,
        *,
        options: Mapping[str, object] | None = None,
        capture: SequentialCaptureRequest | None = None,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> tuple[ScopedReadout, ...]:
        """Admit every selected view before any read, then execute them in one capture."""
        if experiment.request_options.get("store") is not None:
            raise CodedError(
                "Experiment execution does not accept a store request option; "
                "pass results_store to run()",
                code="kernel.run.invalid_input",
                context={"field": "store"},
            )
        merged = dict(options or {})
        for scope in experiment.views:
            refusals = scope_refusals(experiment, scope)
            if refusals:
                raise refusals[0]
            selected_binding(self._binding, experiment, scope)
            operation_options(experiment, scope, merged)
        scoped = execute_views(
            self._binding,
            experiment,
            experiment.views,
            options=merged,
            capture=capture,
            execution_context=execution_context,
        )
        if self._catalog_writer is not None:
            schema = _schema_for_source(self._source)
            self._catalog_writer(
                replace(
                    self._experiment,
                    source_refs=tuple(
                        replace(reference, schema=schema)
                        for reference in self._experiment.source_refs
                    ),
                )
            )
        return scoped

    def close(self) -> None:
        """Release what the builder opened; a connection you supplied stays open."""
        if self._closed:
            return
        if self._bound is not None and self._bound.owns_source:
            close = getattr(self._bound.state.source, "close", None)
            if callable(close):
                close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        """Refuse to pickle: a reader holds the frame or connection it reads."""
        from increment.kernel.contracts import _refuse

        _refuse("kernel.run.reader_unportable", experiment_id=self._experiment.experiment_id)

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Refuse to copy: a reader holds the frame or connection it reads."""
        del memo
        self.__reduce__()

    def __repr__(self) -> str:
        """Name the experiment, how the reader reads and whether it is closed; never reads."""
        if self._bound is None:
            reads = "offline: cannot read"
        else:
            kind = self._experiment.source_refs[0].uri.partition("://")[0]
            reads = _READS.get(kind, kind)
        return (
            f"{type(self).__name__}(experiment_id={reprlib.repr(self._experiment.experiment_id)}, "
            f"reads={reads!r}, closed={self._closed})"
        )


def require_reader(reader: object) -> ExperimentReader:
    """Refuse anything but the reader a builder returned, before any read."""
    if not isinstance(reader, ExperimentReader):
        from increment.kernel.contracts import _refuse

        _refuse("kernel.run.reader_required", received=type(reader).__name__)
    return reader


def binding_for(experiment: Experiment, reader: object) -> SourceBinding:
    from increment.kernel.resolver import _resolve_verified

    reader = require_reader(reader)
    _resolve_verified(experiment, reader)
    return reader._binding


def resolver_for(
    binding: SourceBinding,
    *,
    record_schema: bool = False,
    catalog_writer: Callable[[Experiment], None] | None = None,
    definitions_guard: Callable[[], None] | None = None,
) -> tuple[Experiment, ExperimentReader]:
    """Declare the experiment a binding executes and the reader that owns it."""
    experiment = experiment_for_binding(
        binding,
        operation_parameters=operation_parameters(),
        record_schema=record_schema,
    )
    reader = ExperimentReader(
        binding,
        experiment,
        catalog_writer=catalog_writer,
        definitions_guard=definitions_guard,
    )
    return experiment, reader
