from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from increment._analysis_config import UNSET, _Unset
from increment._binding import ResolverBinding, _run_identity_hashes
from increment._interpretation import EffectEstimate
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, WireFormatError, refuse
from increment.estimation.readout_types import (
    ReadoutAssembly,
    ScopedReadout,
    merge_collection_metadata,
)
from increment.kernel.contracts import (
    BreakoutIdentity,
    CellKey,
    CompiledPlanBinding,
    Registration,
    Scope,
    _refuse,
    method_config_hash,
    method_estimator_id,
    metric_hash,
)
from increment.kernel.ports import (
    DecisionLogPort,
    ResultsStorePort,
    RunHandlePort,
    RunRecord,
    program_store_operations,
)
from increment.kernel.program_refusals import refuse_store
from increment.result_envelope import AnalysisEnvelope, ResultCell, _cell_order, result_rows

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pydantic import BaseModel

    from increment._analysis_execution import Analysis
    from increment._binding import SourceBinding
    from increment._evidence_types import SequentialCaptureRequest
    from increment._literals import Backend, ValueScale
    from increment.breakout.estimates import EstimateList
    from increment.decide.evaluator import Decisions
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.contrast_results import ContrastResults
    from increment.estimation.engine import Method
    from increment.estimation.inference import Prior
    from increment.estimation.readout_types import ReadoutMetadata
    from increment.kernel.context import EvidenceExecutionContext
    from increment.kernel.experiment import Experiment
    from increment.kernel.resolver import Resolver
    from increment.semantics.models import Metric


_CONTRAST_CONTEXT_NOT_PORTABLE = RefusalSpec(
    "wire.procedure.contrast_context",
    WireFormatError,
    template="switchback contrast runs carry no portable arm decision plan; {action} is unavailable",
)
_LIVE_RAW_SET_NOT_PORTABLE = RefusalSpec(
    "wire.result.raw_outcomes_unavailable",
    WireFormatError,
    template="a live confidence-set result depends on raw outcomes; {action} is unavailable",
)
_READOUT_SCOPE_REQUIRED = RefusalSpec(
    "results.readout.scope_required",
    InvalidRequestError,
    template="run {run_id} holds several result views; pass one of its scopes to readout(scope)",
    keys=frozenset({"scope_hashes"}),
)
_READOUT_SCOPE_UNKNOWN = RefusalSpec(
    "results.readout.scope_unknown",
    InvalidRequestError,
    template="run {run_id} did not execute the requested scope; pass a scope the run selected",
    keys=frozenset({"scope_hash"}),
)
_READOUT_NOT_RETAINED = RefusalSpec(
    "results.readout.not_retained",
    InvalidRequestError,
    template=(
        "run {run_id} keeps only its saved result cells; read its cells or envelope, "
        "or run the experiment again for its result collections"
    ),
)


@dataclass(frozen=True, slots=True)
class RunResult:
    cells: tuple[ResultCell, ...]
    effects: tuple[EffectEstimate | None, ...]
    decisions: Decisions
    run: Any
    live_rows: tuple[Any | None, ...] = ()
    _scoped: tuple[ScopedReadout, ...] = field(default=(), repr=False, compare=False)
    _row_models: tuple[type[BaseModel], ...] = field(default=(), repr=False, compare=False)
    _input_backend: Backend | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_handle(cls, handle: RunHandlePort, *, registration: Registration) -> RunResult:
        """Restore a saved run's portable cells and decide them again under ``registration``.

        The saved decision plan must belong to ``registration``. Live readouts are
        not stored, so every restored cell reports none and ``readout`` refuses.
        """
        from increment.decide import decide

        decisions = decide(handle, registration)
        envelope = handle.envelope()
        return cls(
            cells=envelope.cells,
            effects=tuple(cell.summary for cell in envelope.cells),
            decisions=decisions,
            run=handle.run,
            live_rows=(None,) * len(envelope.cells),
        )

    @property
    def metadata(self) -> ReadoutMetadata | None:
        """The run's scope, integrity, family and capture record."""
        return self.run.metadata

    @property
    def envelope(self) -> AnalysisEnvelope:
        """The portable result cells with the run's metadata."""
        return AnalysisEnvelope(self.cells, self.run.metadata)

    def readout(self, scope: Scope | None = None) -> EstimateList[Any] | ContrastResults:
        """Return the result collection one executed scope produced, unchanged.

        ``scope`` may be omitted only when the run executed one scope. A result
        restored from a store or a pickle keeps its portable cells only.
        """
        if not self._scoped:
            refuse(_READOUT_NOT_RETAINED, run_id=self.run.run_id)
        if scope is None:
            if len(self._scoped) != 1:
                refuse(
                    _READOUT_SCOPE_REQUIRED,
                    run_id=self.run.run_id,
                    scope_hashes=tuple(
                        part.scope.hash() for part in self._scoped if part.scope is not None
                    ),
                )
            return self._scoped[0].collection
        scope_hash = scope.hash()
        for part in self._scoped:
            if part.scope is not None and part.scope.hash() == scope_hash:
                return part.collection
        refuse(_READOUT_SCOPE_UNKNOWN, run_id=self.run.run_id, scope_hash=scope_hash)

    def to_frame(
        self, *, backend: Backend | None = None, columns: Sequence[str] | None = None
    ) -> IntoDataFrame:
        """Project every result row with its view and qualification columns as one native frame.

        Each row leads with ``scope_view`` and ``scope_values``, the ``view`` and
        ``values`` of the scope that produced it, and ``scope_key``, which identifies
        that view, then keeps its collection's ``to_frame`` columns; qualification
        columns describe the whole run. A restored result projects its saved cells
        the same way, and a live confidence-set row appears as ``render`` shows it.
        Views with different row types share their common columns;
        ``readout(scope).to_frame()`` returns one view's frame. ``columns`` selects
        typed columns as there.
        """
        from itertools import groupby

        from increment._literals import default_frame_backend
        from increment.breakout.projection import result_frame

        groups = [
            (model, tuple(rows))
            for model, rows in groupby(result_rows(self.cells, self.live_rows), key=type)
        ]
        if not groups:
            if not self._row_models:
                refuse(_READOUT_NOT_RETAINED, run_id=self.run.run_id)
            groups = [(model, ()) for model in self._row_models]
        return result_frame(
            cast(Any, groups),
            self.metadata,
            scopes=tuple(cell.identity.scope for cell in self.cells),
            backend=default_frame_backend(backend or self._input_backend),
            columns=columns,
        )

    def __reduce__(self):
        reason = self.run.decision_plan_unavailable_reason
        if reason is not None:
            raise WireFormatError(
                "An in-memory-only decision plan cannot be serialized",
                code=reason,
                context={"run_id": self.run.run_id},
            )
        if any(row is not None for row in self.live_rows):
            refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="serialization")
        return type(self), (
            self.cells,
            self.effects,
            self.decisions,
            self.run,
            self.live_rows,
            (),
            self._row_models,
            self._input_backend,
        )


@dataclass(frozen=True, slots=True)
class _InMemoryRunHandle:
    run: RunRecord

    def get(self, identity):
        for cell in self.run.cells:
            if cell.identity.hash() == identity.hash():
                return cell
        raise CodedError(
            "Cell is absent from the in-memory run",
            code="store.run.roster_mismatch",
            context={"identity_hash": identity.hash()},
        )

    def envelope(self) -> AnalysisEnvelope:
        return AnalysisEnvelope(self.run.cells, self.run.metadata)


def _validate_registered_options(
    compiled_plan: CompiledDecisionPlan, options: Mapping[str, object]
) -> None:
    values = dict(options)
    requested_metrics = values.pop("metrics", None)
    if requested_metrics is not None:
        names = {
            item if isinstance(item, str) else getattr(item, "name", None)
            for item in cast(Sequence[object], requested_metrics)
        }
        if names != set(compiled_plan.procedures):
            raise CodedError(
                "Registered run cannot omit or add decision metrics",
                code="store.decision.reference_invalid",
                context={},
            )
    if values:
        raise CodedError(
            "Run options cannot override registered decision or population settings",
            code="store.decision.reference_invalid",
            context={"options": tuple(sorted(values))},
        )


def _require_registered_plan(registration: Registration, plan: CompiledDecisionPlan) -> None:
    if not registration.matches(plan):
        raise CodedError(
            "Analysis execution plan differs from its pre-estimation registration",
            code="store.decision.reference_invalid",
            context={},
        )


def _validate_analysis_binding(
    analysis: Analysis,
    registration: Registration,
    compiled_plan: CompiledDecisionPlan,
    options: Mapping[str, object],
) -> None:
    _require_registered_plan(registration, analysis.compiled_decision_plan)
    _validate_registered_options(compiled_plan, options)


def _contrast_context(binding: SourceBinding) -> Any | None:
    context = binding.context
    return context if not hasattr(context, "plan") and hasattr(context, "procedures") else None


def _events(run: RunRecord, decisions: Decisions, binding: CompiledPlanBinding) -> tuple[Any, ...]:
    operations = program_store_operations()
    cells_by_hash = {cell.identity.hash(): cell for cell in run.cells}
    metadata = run.metadata
    if metadata is None:
        raise CodedError(
            "Registered decision events require complete run metadata",
            code="store.run.metadata_mismatch",
            context={"run_id": run.run_id},
        )
    scope = metadata.scope
    output: list[Any] = []
    for verdict in decisions.families:
        if verdict.procedure not in {"bh", "e_bh", "bonferroni"}:
            continue
        family_scope = next(
            family
            for family in scope.families
            if family.family_id == verdict.family_id
            and family.source_snapshot_id == verdict.source_snapshot_id
            and str(family.analysis_population) == verdict.analysis_population
        )
        references = tuple(
            operations.DecisionReference(cells_by_hash[member.identity_hash].identity, run.run_id)
            for member in verdict.members
            if member.identity_hash in cells_by_hash
        )
        if len(references) != len(verdict.members):
            raise CodedError(
                "Family decision does not resolve to the complete stored roster",
                code="store.run.roster_mismatch",
                context={"family_id": verdict.family_id},
            )
        family_q = family_scope.family.q if family_scope.family is not None else None
        alpha_spent = float(family_q or 0.05)
        threshold = alpha_spent
        source = scope.by_source.get(verdict.source_snapshot_id)
        integrity = (
            "unknown"
            if source is None
            else ",".join(
                sorted(
                    {
                        result.status.value
                        for result in source.integrity
                        if str(result.analysis_population) == "assigned"
                    }
                )
            )
            or "unknown"
        )
        event_rule = cast(
            Literal["bh", "ebh", "bonferroni"],
            {"bh": "bh", "e_bh": "ebh", "bonferroni": "bonferroni"}[verdict.procedure],
        )
        output.append(
            operations.DecisionEvent(
                experiment_id=run.experiment_id,
                family_id=verdict.family_id,
                registration_hash=run.registration_hash or binding.registration_hash,
                decided_at=datetime.now(UTC),
                rule=event_rule,
                threshold=threshold,
                alpha_spent=alpha_spent,
                references=references,
                evidence_digest=operations.run_digest(run),
                integrity_status=integrity,
                outcome=verdict.status,
                reason_codes=verdict.reason_codes,
                decision_plan=binding,
            )
        )
    return tuple(output)


def _execution_config_for_cell(binding: SourceBinding, cell: Any) -> tuple[Any, tuple[Any, ...]]:
    from increment._analysis_config import effective_methods

    key = cell.identity.key
    if key.metric == "uptake" and key.estimand == "compliance":
        uptake = getattr(binding.design, "uptake", None)
        if uptake is not None:
            from increment.estimation.engine import Method

            return uptake, (Method(name="unadjusted"),)

    configs = getattr(binding.context, "configs", None)
    if configs is None:
        raise CodedError(
            "Execution method identity is unavailable",
            code="store.run.identity_incomplete",
            context={"field": "method_identity"},
        )
    config = next((item for item in configs if item.metric.name == key.metric), None)
    if config is None:
        raise CodedError(
            "Cell metric has no execution configuration",
            code="store.run.identity_incomplete",
            context={"field": "metric_identity"},
        )
    return config.metric, tuple(effective_methods(config, design=binding.design))


def _persistence_identity(
    binding: SourceBinding,
    results: Any,
    decision_method: Any,
    sensitivity_methods: Any,
    extra_metrics: Sequence[Any] = (),
    runtime_request: Mapping[str, object] | None = None,
):
    from increment._analysis_config import UNSET
    from increment._canonical import declaration_value
    from increment.decision import ContrastAnalysisState
    from increment.estimation.engine import Method
    from increment.kernel.contracts import contract_hash

    persistence_identity = program_store_operations().PersistenceIdentity

    design = binding.design
    if isinstance(binding.state, ContrastAnalysisState):
        procedures = binding.context.procedures
        registration_hash = Registration(
            registration_hash=contract_hash(
                "registration",
                {
                    "design": design.model_dump(mode="json"),
                    "procedures": {
                        name: procedure.model_dump(mode="json")
                        for name, procedure in sorted(procedures.items())
                    },
                },
            )
        ).hash()
    else:
        procedures = binding.plan.procedures
        registration_hash = Registration.from_plan(binding.plan).hash()
    cells = results.as_cells()
    cell_metrics = {cell.identity.key.metric for cell in cells}
    metrics = {metric.name: metric for metric in (*binding.metrics, *extra_metrics)}
    if "uptake" in cell_metrics and "uptake" not in metrics:
        from increment.semantics.models import MeanMetric

        uptake = getattr(design, "uptake", None)
        metrics["uptake"] = MeanMetric(
            name="uptake",
            entity="__uptake__",
            fact="__uptake__",
            aggregation="avg_event",
            window_days=getattr(uptake, "window_days", None),
        )
    methods: dict[tuple[str, str, str], Method] = {}
    source_refs: dict[tuple[str, str], str] = {}
    method_overrides: list[Method] = []
    if isinstance(decision_method, Method):
        method_overrides.append(decision_method)
    if sensitivity_methods is not UNSET:
        method_overrides.extend(cast(Sequence[Method], sensitivity_methods))

    source = binding.source
    source_context = getattr(source, "context", None)
    declaration_identity = getattr(source, "declaration_identity", None)
    if callable(declaration_identity):
        declaration = declaration_identity()
        declaration_unavailable_reason = None
    else:
        declaration = {"context": declaration_value(source_context)}
        declaration_unavailable_reason = "source_declaration_identity_unavailable"
    source_declaration_hash = contract_hash("source_declaration_v1", declaration_value(declaration))
    if runtime_request:
        registration_hash = contract_hash(
            "analysis_runtime_request_v1",
            {
                "registration_hash": registration_hash,
                "request": declaration_value(runtime_request),
            },
        )
    for cell in cells:
        key = cell.identity.key
        estimator = key.method or "unadjusted"
        role = key.method_role or "decision"
        procedure = procedures.get(key.metric)
        declared_methods: tuple[Method | None, ...] = (
            ()
            if procedure is None
            else (
                getattr(procedure, "decision_method", None),
                *getattr(procedure, "sensitivity_methods", ()),
            )
        )
        selected = next(
            (
                method
                for method in (*method_overrides, *declared_methods)
                if isinstance(method, Method) and method.name == estimator
            ),
            None,
        )
        if selected is None:
            selected = Method.cuped() if estimator == "cuped" else Method(name=estimator)
        methods[(key.metric, estimator, role)] = selected
        source_refs[(key.analysis_population, key.metric)] = source_declaration_hash
    return persistence_identity(
        design=design,
        metrics=metrics,
        registration_hash=registration_hash,
        methods=methods,
        source_refs=source_refs,
        source_declaration_identity_unavailable_reason=declaration_unavailable_reason,
    )


def _execution_identity_fields(
    binding: SourceBinding,
    cell: ResultCell,
    identity_methods: Mapping[tuple[str, str, str], Any] | None = None,
    identity_metrics: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    key = cell.identity.key
    if key.metric == "uptake" and key.estimand == "compliance":
        metric = _execution_config_for_cell(binding, cell)[0]
    else:
        metric = None if identity_metrics is None else identity_metrics.get(key.metric)
    if key.kind == "contrast":
        if metric is None:
            metric = next((item for item in binding.metrics if item.name == key.metric), None)
        if metric is None:
            raise CodedError(
                "Contrast cell has no matching metric declaration",
                code="store.run.identity_incomplete",
                context={"field": "metric_identity"},
            )
        return {
            "metric_hash": metric_hash(metric),
            "estimator_id": None,
            "method_config_hash": None,
            "method_instance_id": None,
        }
    if key.method is None and key.method_role is None:
        if metric is None:
            metric = _execution_config_for_cell(binding, cell)[0]
        return {
            "metric_hash": metric_hash(metric),
            "estimator_id": None,
            "method_config_hash": None,
            "method_instance_id": None,
        }
    method = (
        None
        if identity_methods is None
        else identity_methods.get(
            (key.metric, key.method or "unadjusted", key.method_role or "decision")
        )
    )
    methods: Sequence[Any] = ()
    if metric is None or method is None:
        configured_metric, methods = _execution_config_for_cell(binding, cell)
        if metric is None:
            metric = configured_metric
    if method is None:
        for index, candidate in enumerate(methods):
            role = "decision" if index == 0 else "sensitivity"
            if candidate.name == key.method and role == key.method_role:
                method = candidate
                break
    if method is None:
        raise CodedError(
            "Cell method has no matching execution configuration",
            code="store.run.identity_incomplete",
            context={"field": "method_identity"},
        )
    estimator = method_estimator_id(method)
    if estimator is None:
        raise CodedError(
            "Cell estimator has no registered execution identity",
            code="store.run.identity_incomplete",
            context={"field": "estimator_id"},
        )
    method_instance_id = f"{key.metric}:{key.method_role}:{method.name}"
    try:
        config_hash = method_config_hash(method, method_instance_id=method_instance_id)
    except CodedError as error:
        if error.code != "kernel.method.implementation_ref_missing":
            raise
        config_hash = None
    return {
        "metric_hash": metric_hash(metric),
        "estimator_id": estimator,
        "method_config_hash": config_hash,
        "method_instance_id": method_instance_id,
    }


def _metadata_with_cell_scales(metadata: Any, cells: Sequence[ResultCell]) -> Any:
    """Project records to emitted cells while retaining the declared family scope."""
    if metadata is None:
        return None

    cells_by_source: dict[str | None, list[CellKey]] = {}
    for cell in cells:
        cells_by_source.setdefault(cell.identity.source_snapshot_id, []).append(cell.identity.key)

    key_map: dict[Any, Any] = {}
    records = []
    for record in metadata.cells:
        candidates = cells_by_source.get(record.source_snapshot_id, [])
        if record.cell in candidates:
            key = record.cell
        else:
            matching = [
                candidate
                for candidate in candidates
                if candidate.model_copy(update={"value_scale": None})
                == record.cell.model_copy(update={"value_scale": None})
            ]
            if len(matching) != 1:
                continue
            key = matching[0]
        key_map[record.cell] = key
        records.append(record.model_copy(update={"cell": key}) if key != record.cell else record)

    filtered = len(records) != len(metadata.cells)
    if not filtered and not any(old != new for old, new in key_map.items()):
        return metadata

    def rekey(key):
        return key_map.get(key, key)

    by_source = {
        source_id: source.model_copy(
            update={
                "cells": tuple(sorted({rekey(key) for key in source.cells}, key=_cell_order)),
                "decision_cells": tuple(
                    sorted({rekey(key) for key in source.decision_cells}, key=_cell_order)
                ),
            }
        )
        for source_id, source in metadata.scope.by_source.items()
    }
    scope = metadata.scope.model_copy(
        update={
            "cells": tuple(sorted({rekey(key) for key in metadata.scope.cells}, key=_cell_order)),
            "decision_cells": tuple(
                sorted({rekey(key) for key in metadata.scope.decision_cells}, key=_cell_order)
            ),
            "by_source": by_source,
            "families": tuple(
                family.model_copy(
                    update={
                        "members": tuple(
                            sorted({rekey(key) for key in family.members}, key=_cell_order)
                        )
                    }
                )
                for family in metadata.scope.families
            ),
        }
    )
    records = tuple(
        sorted(
            records,
            key=lambda record: (
                0 if record.cell.analysis_population == "assigned" else 1,
                record.source_snapshot_id,
                _cell_order(record.cell),
            ),
        )
    )
    partial_reason = metadata.partial_reason
    if filtered:
        partial_reason = tuple(
            item
            for item in ("slice", "filter", "cross_snapshot_concat")
            if item in (*partial_reason, "filter")
        )
    return metadata.model_copy(
        update={
            "scope": scope,
            "cells": records,
            "partial": metadata.partial or filtered,
            "partial_reason": partial_reason,
        }
    )


def _execution_overrides(
    binding: SourceBinding,
    rows: Any,
    options: Mapping[str, object] | None,
    *,
    contrast: bool,
) -> tuple[Mapping[tuple[str, str, str], Any] | None, Mapping[str, Any] | None, bool]:
    if contrast:
        return None, {metric.name: metric for metric in binding.metrics}, False
    if not options or not any(
        name in options
        for name in ("decision_method", "sensitivity_methods", "metrics", "exploratory_metrics")
    ):
        return None, None, False

    selected_metrics = options.get("metrics", ())
    extra_metrics = (
        tuple(
            metric
            for metric in selected_metrics
            if not isinstance(metric, str)
            and hasattr(metric, "name")
            and hasattr(metric, "model_dump")
        )
        if isinstance(selected_metrics, Sequence)
        else ()
    )
    exploratory_names = options.get("exploratory_metrics")
    if exploratory_names:
        from increment._exploratory import select_exploratory

        extra_metrics += tuple(
            select_exploratory(binding, cast(Sequence[str], exploratory_names), caller="run")
        )
    identity = _persistence_identity(
        binding,
        rows,
        cast(Any, options.get("decision_method")),
        cast(Any, options.get("sensitivity_methods", ())),
        extra_metrics=cast(Any, extra_metrics),
    )
    runtime_method_unavailable = False
    for (metric_name, _method_name, role), method in identity.methods.items():
        if (
            method.folds is not None
            or method.propensity_learner is not None
            or method.outcome_learner is not None
        ):
            runtime_method_unavailable = True
            break
        try:
            method_config_hash(method, method_instance_id=f"{metric_name}:{role}:{method.name}")
        except CodedError as error:
            if error.code != "kernel.method.implementation_ref_missing":
                raise
            runtime_method_unavailable = True
            break
    return identity.methods, identity.metrics, runtime_method_unavailable


def _scope_for_cell(cell: ResultCell, declared_scope: Scope | None) -> Scope:
    key = cell.identity.key
    if declared_scope is None:
        if key.kind == "segment":
            return Scope(
                view="breakout",
                grain="total",
                population=key.analysis_population,
                segment=(BreakoutIdentity(property=key.dimension or "", source=key.source),),
            )
        return Scope(
            view="run",
            grain="total",
            population=key.analysis_population,
        )

    resolved_scope = cell.identity.scope
    scope_update: dict[str, Any] = {"population": key.analysis_population}
    if resolved_scope is not None:
        scope_update["time_basis"] = resolved_scope.time_basis
    raw_day = key.ds
    day: date | None = None
    if isinstance(raw_day, datetime):
        day = raw_day.date()
    elif isinstance(raw_day, date):
        day = raw_day
    elif isinstance(raw_day, str):
        try:
            day = date.fromisoformat(raw_day)
        except ValueError:
            try:
                day = datetime.fromisoformat(raw_day).date()
            except ValueError:
                day = None
    if isinstance(day, date):
        scope_update["through_day"] = day
    if key.ds_basis is not None:
        scope_update["time_basis"] = key.ds_basis

    row_scope = declared_scope.model_copy(update=scope_update)
    if (declared_scope.segment or declared_scope.view == "breakout") and key.dimension is not None:
        row_scope = row_scope.model_copy(
            update={"segment": (BreakoutIdentity(property=key.dimension, source=key.source),)}
        )
    return row_scope


def _live_winsor_effect(row: Any) -> EffectEstimate:
    confidence_set = row.confidence_set
    value_scale = getattr(row, "value_scale", "relative")
    interval = confidence_set.additive if value_scale == "absolute" else confidence_set.relative
    point = confidence_set.additive_point if value_scale == "absolute" else confidence_set.point
    reasons = tuple(
        f"{side} endpoint {endpoint.status}: {endpoint.reason}"
        for side, endpoint in (("lower", interval.lower), ("upper", interval.upper))
        if endpoint.status != "finite" and endpoint.reason is not None
    )
    reason = (
        getattr(row, "relative_unavailable_reason", None)
        if point is None and value_scale != "absolute"
        else None
    )
    if reason is None and reasons:
        reason = "; ".join(reasons)
    return EffectEstimate(
        point,
        interval.lower.value if interval.lower.status == "finite" else None,
        interval.upper.value if interval.upper.status == "finite" else None,
        confidence_set.alpha,
        value_scale,
        reason,
        qualification=confidence_set.qualification,
    )


def _selected_percentile_winsorization(
    binding: SourceBinding, options: Mapping[str, object]
) -> bool:
    metrics = tuple(binding.metrics)
    selected = options.get("metrics")
    if selected is None:
        candidates = metrics
    else:
        selected_metrics = cast(Sequence[object], selected)
        names = {
            value if isinstance(value, str) else getattr(value, "name", None)
            for value in selected_metrics
        }
        candidates = tuple(metric for metric in metrics if metric.name in names)
    return any(
        getattr(getattr(metric, "winsorization", None), "has_percentile", False)
        for metric in candidates
    )


def _merge_scoped_readouts(parts: Sequence[ScopedReadout]) -> Any:
    rows = parts[0].collection
    sequential_snapshot = rows.sequential_snapshot
    for part in parts[1:]:
        if part.collection.metadata is None:
            continue
        if rows.metadata is None:
            rows = part.collection
            sequential_snapshot = sequential_snapshot or rows.sequential_snapshot
            continue
        metadata, source, merged = merge_collection_metadata(
            rows, part.collection, scoped_assembly=True
        )
        sequential_snapshot = sequential_snapshot or part.collection.sequential_snapshot
        rows = ReadoutAssembly(tuple(merged.values()), metadata, source, sequential_snapshot)
    return rows


def _run_rows(
    source_binding: SourceBinding,
    parts: tuple[ScopedReadout, ...],
    registration: Registration,
    compiled_plan: CompiledDecisionPlan | None,
    *,
    experiment_id: str,
    analysis_spec_hash: str,
    execution_identity_hash: str,
    run_id: str,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
) -> RunResult:
    operations = program_store_operations()
    contrast_context = _contrast_context(source_binding)
    overrides = [
        _execution_overrides(
            source_binding, part.collection, part.options, contrast=contrast_context is not None
        )
        for part in parts
    ]
    runtime_method_unavailable = any(item[2] for item in overrides)
    rows = _merge_scoped_readouts(parts)
    if contrast_context is not None:
        if results_store is not None or decision_log is not None:
            refuse(
                _CONTRAST_CONTEXT_NOT_PORTABLE,
                action="saving",
            )
        binding = None
        binding_unavailable_reason = "wire.procedure.contrast_context"
    else:
        try:
            if runtime_method_unavailable:
                raise WireFormatError(
                    "A runtime method has no portable implementation reference",
                    code="wire.procedure.runtime_method",
                    context={},
                )
            if compiled_plan is None:
                raise CodedError(
                    "A run decision plan is unavailable",
                    code="store.run.identity_incomplete",
                    context={"field": "compiled_plan"},
                )
            binding = CompiledPlanBinding.from_plan(registration, compiled_plan)
            binding_unavailable_reason = None
        except WireFormatError as error:
            if (
                error.code != "wire.procedure.runtime_method"
                or results_store is not None
                or decision_log is not None
            ):
                raise
            binding = None
            binding_unavailable_reason = error.code
    created_at = datetime.now(UTC)
    cells: list[ResultCell] = []
    live_rows: list[Any | None] = []
    for part, (identity_methods, identity_metrics, _) in zip(parts, overrides, strict=True):
        collection = part.collection
        live_rows_by_key = {
            ResultCell.from_row(row, collection.metadata, created_at=created_at).identity.key: row
            for row in collection
            if getattr(row, "confidence_set", None) is not None
        }
        if live_rows_by_key and results_store is not None:
            refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="saving")
        for cell in collection.as_cells(created_at=created_at):
            live_rows.append(live_rows_by_key.get(cell.identity.key))
            row_scope = _scope_for_cell(cell, part.scope)
            identity = cell.identity.model_copy(
                update={
                    **_execution_identity_fields(
                        source_binding, cell, identity_methods, identity_metrics
                    ),
                    "experiment_id": experiment_id,
                    "analysis_spec_hash": part.analysis_spec_hash or analysis_spec_hash,
                    "scope": row_scope,
                }
            )
            provenance = replace(
                cell.provenance,
                run_id=run_id,
                execution_identity_hash=execution_identity_hash,
                artifact_manifest_sha256=(
                    cell.provenance.artifact_manifest_sha256
                    or (
                        None
                        if collection.metadata is None
                        else collection.metadata.artifact_manifest_sha256
                    )
                ),
                artifact_relation_digests=(
                    cell.provenance.artifact_relation_digests
                    or (
                        ()
                        if collection.metadata is None
                        else collection.metadata.artifact_relation_digests
                    )
                ),
            )
            cells.append(replace(cell, identity=identity, provenance=provenance))
    run_record = operations.Run(
        experiment_id=experiment_id,
        run_id=run_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        registration_hash=registration.registration_hash,
        expected_roster=tuple(cell.identity for cell in cells),
        cells=tuple(cells),
        metadata=_metadata_with_cell_scales(rows.metadata, cells),
        captured_at=created_at,
        committed_at=datetime.now(UTC),
        scope_analysis_hashes=tuple(
            dict.fromkeys(
                (
                    analysis_spec_hash,
                    *(cast(str, cell.identity.analysis_spec_hash) for cell in cells),
                )
            )
        ),
        cell_scope_hashes=tuple(
            (cell.identity.hash(), cast(str, cell.identity.analysis_spec_hash)) for cell in cells
        ),
        decision_plan=binding,
        decision_plan_unavailable_reason=binding_unavailable_reason,
        empty_selection_reason="empty_metric_selection" if not cells else None,
    )
    if results_store is None:
        handle: RunHandlePort = _InMemoryRunHandle(run_record)
    else:
        results_store.put(run_record)
        handle = results_store.open_run(experiment_id, analysis_spec_hash, run_id)
    from increment.decide import decide

    decisions = decide(handle, registration, contrast_context or compiled_plan)
    events = ()
    if decision_log is not None:
        assert binding is not None
        events = _events(handle.run, decisions, binding)
    envelope = handle.envelope()
    result = RunResult(
        cells=envelope.cells,
        effects=tuple(
            _live_winsor_effect(row) if row is not None else cell.summary
            for cell, row in zip(envelope.cells, live_rows, strict=True)
        ),
        decisions=decisions,
        run=handle.run,
        live_rows=tuple(live_rows),
        _scoped=parts,
        _row_models=tuple(dict.fromkeys(type(part.collection)._model for part in parts)),
        _input_backend=next(
            (getattr(part.collection, "_default_backend", None) for part in parts), None
        ),
    )
    if decision_log is not None:
        for event in events:
            decision_log.append(event)
    return result


def _experiment_result(
    binding: SourceBinding,
    experiment: Experiment,
    scoped: Sequence[ScopedReadout],
    *,
    run_id: str,
    results_store: ResultsStorePort | None,
    decision_log: DecisionLogPort | None,
) -> RunResult:
    """Convert the executed scoped readouts into one identified, decided run."""
    single = len(scoped) == 1
    identity_options = (
        scoped[0].options
        if single
        else {"views": tuple({"scope": item.scope, "options": item.options} for item in scoped)}
    )
    analysis_spec_hash, execution_identity_hash = _run_identity_hashes(experiment, identity_options)
    return _run_rows(
        binding,
        tuple(scoped),
        experiment.registration,
        binding.plan if hasattr(binding.context, "plan") else None,
        experiment_id=experiment.experiment_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        run_id=run_id,
        results_store=results_store,
        decision_log=decision_log,
    )


def _readout_overrides(**options: object) -> dict[str, object]:
    """Keep the readout options a caller set; an explicit ``prior=None`` differs from omitted."""
    unset_defaults = {"decision_method", "sensitivity_methods", "prior"}
    return {
        name: value
        for name, value in options.items()
        if (value is not UNSET if name in unset_defaults else value is not None)
        and not (name == "coherence" and value == "captured")
    }


def _scope_population(experiment: Experiment, overrides: dict[str, object]) -> dict[str, object]:
    """Admit a run population only where every selected scope already declares it."""
    population = overrides.pop("population", None)
    if population is not None and any(
        scope.population != population for scope in experiment.scopes
    ):
        _refuse("kernel.request.population_scope_only")
    return overrides


def _preflight_persistence(
    resolver: ResolverBinding,
    experiment: Experiment,
    overrides: Mapping[str, object],
    *,
    saving: bool,
) -> None:
    """Refuse a run that cannot be saved or logged under its registration, before any read."""
    from increment._runnability import operation_options, typed_request_options

    binding = resolver.binding
    contrast = _contrast_context(binding) is not None
    if not contrast:
        _require_registered_plan(experiment.registration, binding.plan)
    resolver.validate_experiment(experiment)
    if contrast:
        refuse(_CONTRAST_CONTEXT_NOT_PORTABLE, action="saving")
    CompiledPlanBinding.from_plan(experiment.registration, binding.plan)
    registered = {**typed_request_options(experiment.request_options), **overrides}
    if not registered.get("metrics", True):
        # An empty selection is an empty run, which the store refuses itself.
        del registered["metrics"]
    _validate_registered_options(binding.plan, registered)
    if saving and any(
        _selected_percentile_winsorization(binding, operation_options(experiment, scope, overrides))
        for scope in experiment.scopes
    ):
        refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="saving")


@overload
def run(
    experiment: Experiment,
    resolver: Resolver,
    *,
    views: Sequence[Scope] | None = None,
    run_id: str | None = None,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    capture: SequentialCaptureRequest | None = None,
    execution_context: EvidenceExecutionContext | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str | Metric] | None = None,
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    exploratory_metrics: Sequence[str] | None = None,
    population: Literal["assigned", "triggered"] | None = None,
    coherence: Literal["captured", "live"] = "captured",
) -> RunResult: ...


@overload
def run(
    experiment: Analysis,
    resolver: Registration,
    compiled_plan: CompiledDecisionPlan,
    *,
    experiment_id: str,
    analysis_spec_hash: str,
    execution_identity_hash: str,
    run_id: str,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str | Metric] | None = None,
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    exploratory_metrics: Sequence[str] | None = None,
    population: Literal["assigned", "triggered"] | None = None,
    coherence: Literal["captured", "live"] = "captured",
) -> RunResult: ...


def run(  # noqa: PLR0913
    experiment: Experiment | Analysis,
    resolver: Resolver | Registration | None = None,
    compiled_plan: CompiledDecisionPlan | None = None,
    *,
    views: Sequence[Scope] | None = None,
    run_id: str | None = None,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    capture: SequentialCaptureRequest | None = None,
    execution_context: EvidenceExecutionContext | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str | Metric] | None = None,
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    exploratory_metrics: Sequence[str] | None = None,
    population: Literal["assigned", "triggered"] | None = None,
    coherence: Literal["captured", "live"] = "captured",
    experiment_id: str | None = None,
    analysis_spec_hash: str | None = None,
    execution_identity_hash: str | None = None,
) -> Any:
    """Run an experiment through its resolver, or a legacy Analysis through its registration.

    ``views`` selects declared Experiment scopes by hash, in declaration order.
    ``None`` executes all declared scopes in one capture. The readout options
    match ``Analysis.run``; only options set by the caller apply. A population
    must match every selected scope, which declares it. ``results_store`` and
    ``decision_log`` save or log the run under the experiment's registration,
    checked before the source is read; ``run_id`` is generated when omitted.
    Legacy Analysis uses its registered scope and does not accept view selection.
    """
    from increment.kernel.experiment import Experiment

    overrides = _readout_overrides(
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        estimands=estimands,
        value_scale=value_scale,
        exploratory_metrics=exploratory_metrics,
        population=population,
        coherence=coherence,
    )
    if isinstance(experiment, Experiment):
        from increment.kernel.resolver import _select_views
        from increment.kernel.resolver import run as run_experiment

        unsupported = {
            "compiled_plan": compiled_plan,
            "experiment_id": experiment_id,
            "analysis_spec_hash": analysis_spec_hash,
            "execution_identity_hash": execution_identity_hash,
        }
        supplied = tuple(name for name, value in unsupported.items() if value is not None)
        if supplied:
            raise CodedError(
                "Experiment execution derives its own registration and run identities",
                code="kernel.run.invalid_input",
                context={"fields": supplied},
            )
        if resolver is None:
            raise CodedError(
                "Running an experiment requires a source resolver",
                code="kernel.run.resolver_missing",
                context={},
            )
        selected = _select_views(experiment, views)
        overrides = _scope_population(selected, overrides)
        persisted = {"run_id": run_id, "results_store": results_store, "decision_log": decision_log}
        identified = tuple(name for name, value in persisted.items() if value is not None)
        if identified and not isinstance(resolver, ResolverBinding):
            raise CodedError(
                "Saving or identifying a run requires a resolver returned by a module builder",
                code="kernel.run.invalid_input",
                context={"fields": identified},
            )
        if run_id is not None and not run_id:
            refuse_store(
                "store.run.identity_incomplete",
                "A complete run identity is required",
                field="run_id",
            )
        if results_store is not None or decision_log is not None:
            _preflight_persistence(
                cast(ResolverBinding, resolver),
                selected,
                overrides,
                saving=results_store is not None,
            )
        scoped = run_experiment(
            selected,
            cast("Resolver", resolver),
            options=overrides,
            capture=capture,
            execution_context=execution_context,
        )
        if not isinstance(resolver, ResolverBinding):
            return scoped
        from uuid import uuid4

        return _experiment_result(
            resolver.binding,
            selected,
            scoped,
            run_id=f"run-{uuid4().hex}" if run_id is None else run_id,
            results_store=results_store,
            decision_log=decision_log,
        )
    for option, value in (
        ("views", views),
        ("capture", capture),
        ("execution_context", execution_context),
    ):
        if value is not None:
            raise CodedError(
                "This option requires an Experiment",
                code="kernel.run.invalid_input",
                context={"field": option},
            )
    analysis = cast(Any, experiment)
    registration = resolver
    if (
        not isinstance(registration, Registration)
        or compiled_plan is None
        or experiment_id is None
        or analysis_spec_hash is None
        or execution_identity_hash is None
        or run_id is None
    ):
        raise CodedError(
            "Legacy Analysis runs require a registration, plan, and run identities",
            code="store.run.identity_incomplete",
            context={"field": "run_arguments"},
        )
    # Legacy calls retain the established Analysis execution and return contract.
    if not registration.matches(compiled_plan):
        raise CodedError(
            "Compiled plan differs from the pre-evidence registration",
            code="store.decision.reference_invalid",
            context={},
        )
    _validate_analysis_binding(analysis, registration, compiled_plan, overrides)
    if results_store is not None and _selected_percentile_winsorization(
        analysis._binding, overrides
    ):
        refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="saving")
    if results_store is not None or decision_log is not None:
        # Check wire portability before Analysis.run() can read the source.
        CompiledPlanBinding.from_plan(registration, compiled_plan)
    execute = cast(Callable[..., Any], analysis.run)
    rows = execute(**overrides)
    return _run_rows(
        analysis._binding,
        (ScopedReadout(rows, None, {}),),
        registration,
        compiled_plan,
        experiment_id=experiment_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        run_id=run_id,
        results_store=results_store,
        decision_log=decision_log,
    )
