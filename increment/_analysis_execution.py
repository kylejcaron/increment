"""Shared analysis execution for bound experiment sources."""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

    from increment._source_types import _BuiltinMomentSource
    from increment.breakout.estimates import (
        BreakoutEstimates,
        DailyLiftEstimates,
        DailyMetricValues,
        LiftEstimates,
    )
    from increment.kernel.context import EvidenceExecutionContext
    from increment.query.artifact_contract import ArtifactContext, ArtifactStore
    from increment.query.session import SourceSnapshotEvidence, WarehouseSession
    from increment.semantics.artifact import ArtifactExtensionRequest, UnitDayArtifactRef
    from increment.semantics.models import Experiment


from increment import readouts
from increment._analysis_config import (
    UNSET,
    ResolvedMetricConfig,
    _Unset,
    effective_methods,
    normalize_display_correction,
    overlay_configs,
    select_exploratory_metrics,
    select_metrics,
)
from increment._breakout_readouts import BreakoutReadouts, BreakoutRequest
from increment._day_axis import (
    _CLUSTERED_DAY_AXIS,
    _TRIGGER_UNSUPPORTED,
    DayAxisReadouts,
)
from increment._evidence_dispatch import ContrastEvidenceOperation, ContrastReadoutRequest
from increment._evidence_refusals import CONTRAST_UNAVAILABLE as _CONTRAST_UNAVAILABLE
from increment._evidence_types import (
    CountEvidence,
    CountsRequest,
    DayAxisRequest,
    EvidenceScope,
    SequentialCaptureRequest,
    SourceContext,
)
from increment._literals import Backend, Correction, ValueScale
from increment._source_operations import (
    AllocationHistoryOperation,
    DashboardBreakoutReads,
    DashboardGroupData,
    DashboardGroupDataOperation,
    DashboardSnapshotPayload,
    ExploratorySourceOperation,
    ExportMomentsOperation,
    MaterializeOperation,
    SummarySqlOperation,
    TriggeredHealthEvidenceOperation,
)
from increment._source_types import declared_evidence, is_custom_source
from increment._study import ParallelStudyEnvelope
from increment._whole_window import (
    WholeWindowReadouts,
    WholeWindowRequest,
    _require_analysis_operation,
    reject_role_overrides_under_contrast,
    validate_contrast_request,
)
from increment.decision import (
    AnalysisState,
    ArmAnalysisState,
    ContrastAnalysisState,
    ContrastContext,
    ContrastSource,
    DefinitionsArmAnalysisState,
    SeamArmAnalysisState,
)
from increment.errors import (
    CapabilityError,
    CodedError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.errors import refuse as _refuse
from increment.estimation.assignment_integrity import (
    ALLOCATION_MISSING_OR_NONCONSTANT_REFUSAL,
    ALLOCATION_SCHEME_MISSING_REFUSAL,
)
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.engine import Method
from increment.evidence import evidence_provider, request_evidence
from increment.plan import (
    bind_automatic_sequential_plan,
    compile_decision_plan,
    with_unassigned_procedures,
)
from increment.readouts._evidence import (
    NATIVE_ROUTE,
    day_axis_route,
    is_artifact_route,
    source_route,
)
from increment.semantics.models import Breakout, Definitions, Experiment
from increment.semantics.unit_cycle import UnitCycleReference
from increment.sequential_source import native_observation_mapping
from increment.sources import EvidenceSource, _health_snapshot_id

__all__ = ["Analysis", "fit_predeclared_adjustment"]


def _input_backend_of(frame: IntoDataFrame) -> Backend:
    import narwhals as nw

    return cast("Backend", nw.from_native(frame, eager_only=True).implementation.value)


_SWITCHBACK_ONLY = RefusalSpec(
    "facade.analysis.switchback_only",
    CapabilityError,
    template="{method}() is only available for switchback evidence.",
)
_INVALID_MIXED_ASSIGNMENT_POLICY = RefusalSpec(
    "facade.analysis.invalid_mixed_assignment_policy",
    InvalidRequestError,
    template="on_mixed_assignment must be 'error', 'warn', or 'exclude', got {on_mixed_assignment!r}",
)

_INVALID_STORE_POLICY = RefusalSpec(
    "facade.analysis.invalid_store_policy",
    InvalidRequestError,
    template="store must be 'auto', 'always', or 'none'; got {store!r}",
)

_INVALID_POPULATION = RefusalSpec(
    "facade.analysis.invalid_population",
    InvalidRequestError,
    template="population must be 'assigned' or 'triggered'; got {population!r}",
)

_UNDECLARED_BREAKOUT = RefusalSpec(
    "facade.analysis.undeclared_breakout",
    InvalidRequestError,
    template="breakout {requested!r} is not declared by this experiment; declared: {declared!r}",
)
_EXPLORATORY_SOURCE_LIMITED = RefusalSpec(
    "facade.analysis.exploratory_metrics_source_limited",
    CapabilityError,
    template="{method}: exploratory_metrics adds metrics from the saved definitions, which this source does not carry -- use Analysis.from_definitions.",
)
_EXPLORATORY_SEQUENTIAL = RefusalSpec(
    "facade.analysis.exploratory_metrics_sequential",
    UnsupportedRequestError,
    template="{method}: exploratory_metrics {names!r} are chosen after data are visible, so no registered sequential model covers them under {inference}; read the declared metrics alone, or use a fixed-horizon plan.",
)
_UNCORRECTED_SEQUENTIAL = RefusalSpec(
    "facade.analysis.uncorrected_segments_sequential",
    UnsupportedRequestError,
    template="uncorrected segment rows have no fixed-horizon p-values under {inference}; read the registered breakout with run_breakout() instead.",
)


def fit_predeclared_adjustment(
    frame,
    *,
    unit: str,
    group: str,
    control: str,
    metric: str,
    covariate: str,
    covariate_missing: Literal["impute", "zero", "error"] = "impute",
):
    """Fit a sequential CUPED coefficient and covariate centre from pre-period data.

    *frame* is one row per assigned unit observed BEFORE the experiment reads
    any outcome: *metric* is the metric measured in a pre-period window and
    the in-experiment covariate is measured relative to exposure. The rows
    are reduced to the same centered moments the fixed-horizon frame path
    builds, and the coefficient is ``estimation.cuped.fit_cuped``'s inverse-n
    weighted within-arm slope with the centre its count-weighted pooled
    covariate mean, so sequential and fixed-horizon CUPED agree on what the
    coefficient is.

    The result is a declaration: pass it through ``InferenceSpec.adjustments``
    or ``ScalarMeanModel.adjustment`` and capture retains
    ``Y - coefficient * (X - center)``. Costs relative to the fixed-horizon
    fit: the coefficient is not contrast-optimal for the in-experiment
    outcome, and ``coefficient * (E[X] - center)`` is a first-order bias on
    the ratio scale the scalar-mean route reports whenever the covariate
    drifts between the pre-period and the experiment.
    """
    from fractions import Fraction

    from increment.estimation.cuped import fit_cuped
    from increment.estimation.engine import Method, _df_to_arms
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import CupedCovariate
    from increment.semantics.sequential import PredeclaredAdjustment

    covariate_declaration = CupedCovariate(
        name=covariate,
        kind="continuous",
        missing="refuse" if covariate_missing == "error" else covariate_missing,
    )
    source = from_unit_summary(
        frame,
        unit=unit,
        group=group,
        control=control,
        metrics=[
            MetricSpec(
                name=metric,
                decision_method=Method.cuped(covariates=covariate_declaration),
            )
        ],
        experiment_id="pre-period",
    )
    fit = fit_cuped(_df_to_arms(source.raw_moments))
    return PredeclaredAdjustment(
        coefficient=Fraction(fit.theta), center=Fraction(fit.mean_x_pooled)
    )


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "integrity.allocation_scheme_missing": ALLOCATION_SCHEME_MISSING_REFUSAL,
        "integrity.allocation_missing_or_nonconstant": ALLOCATION_MISSING_OR_NONCONSTANT_REFUSAL,
        "facade.analysis.definitions_state_requires_identification": "definitions state requires a declared parallel identification",
        "facade.analysis.unknown_experiment": RefusalSpec(
            "facade.analysis.unknown_experiment",
            InvalidRequestError,
            lambda *, requested, available: (
                f"experiment {requested!r} is not declared in these definitions "
                f"(available: {list(available)!r})"
            ),
        ),
        "facade.analysis.source_context_design_disagrees_arm": "source context design disagrees with Analysis construction",
        "store.run.legacy_analysis_required": RefusalSpec(
            "store.run.legacy_analysis_required",
            InvalidRequestError,
            template="Switchback results do not have a registered family decision plan; use {route}",
        ),
    },
)
SOURCE_CONTEXT_DESIGN_DISAGREES_ARM = _REFUSALS[
    "facade.analysis.source_context_design_disagrees_arm"
]

_raise = raiser(_REFUSALS)


if TYPE_CHECKING:
    import pyarrow as pa
    from ibis.expr.types import Table
    from narwhals.typing import IntoDataFrame

    from increment.decision import CompiledDecisionPlan
    from increment.estimation.cate import CateResult, Covariate
    from increment.estimation.diagnostics import NotApplicable, SRMResult
    from increment.estimation.health import ExperimentHealth, HealthCheck
    from increment.estimation.inference import Prior
    from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
    from increment.estimation.targeting import (
        CateValidation,
        ClusterBootstrap,
        TargetingRule,
        TargetingSelection,
    )
    from increment.frame import MetricsArg
    from increment.power import Baseline
    from increment.power.switchback import SwitchbackBaseline
    from increment.query.source import ArtifactMomentSource
    from increment.semantics.assignment import SwitchbackAssignment
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import AnalysisPlan, Breakout, Metric
    from increment.sequential_state import SequentialCheckpoint
    from increment.switchback import SwitchbackAssignmentDiagnostic


# Internal helpers


def _declared_design(experiment: Experiment) -> Randomized | Encouragement | Observational:
    """Map a definitions-declared experiment to its `readouts.run` design."""
    return experiment.resolved_design()


@dataclass(frozen=True, slots=True)
class _ArtifactOpenSpec:
    """Immutable reopen pin for an artifact-backed analysis, fixed at construction."""

    store: ArtifactStore
    ref: UnitDayArtifactRef
    expected_context: ArtifactContext
    verification: Literal["lazy_digest"]


# Public facade


def _stamp_exploratory_rows(rows):
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.multiplicity import multiplicity_status
    from increment.estimation.readout_types import (
        CellKey,
        FamilyScope,
        ReadoutMetadata,
        ReadoutScope,
        cell_order,
        family_identity,
    )

    metadata = rows.metadata
    source_rows = list(rows)
    updated = [
        row.model_copy(
            update={
                "role": "exploratory",
                "multiplicity_status": multiplicity_status("exploratory", None),
            }
        )
        for row in source_rows
    ]
    if metadata is None:
        result = LiftEstimates(
            updated,
            source=rows.source,
            default_backend=rows._default_backend,
        )
        return result

    moved = {(row.source_snapshot_id, CellKey.from_row(row)) for row in source_rows}
    families: list[FamilyScope] = []
    for declared_family in metadata.scope.families:
        family = FamilyScope.model_validate(declared_family)
        assert isinstance(family, FamilyScope)
        retained = tuple(
            cell for cell in family.members if (family.source_snapshot_id, cell) not in moved
        )
        if retained:
            retained_family = family.model_copy(update={"members": retained})
            assert isinstance(retained_family, FamilyScope)
            families.append(retained_family)

    family_ids = {}
    by_family = {}
    for row in source_rows:
        source_id = row.source_snapshot_id
        population = row.analysis_population
        key = (source_id, population)
        by_family.setdefault(key, []).append(CellKey.from_row(row))
    for (source_id, population), members in by_family.items():
        family_id = family_identity(source_id, population, "run", None, None, "exploratory")
        families.append(
            FamilyScope(
                family_id=family_id,
                analysis_population=population,
                source_snapshot_id=source_id,
                view="run",
                dimension=None,
                source=None,
                name="exploratory",
                family=None,
                members=tuple(sorted(set(members), key=cell_order)),
                complete=True,
            )
        )
        family_ids[(source_id, population)] = family_id
    updated = [
        row.model_copy(
            update={"family_id": family_ids[(row.source_snapshot_id, row.analysis_population)]}
        )
        for row in updated
    ]
    scope = ReadoutScope(
        snapshot_id=metadata.scope.snapshot_id,
        cells=metadata.scope.cells,
        decision_cells=metadata.scope.decision_cells,
        populations=metadata.scope.populations,
        families=tuple(
            sorted(
                families,
                key=lambda family: (
                    family.family_id
                    if isinstance(family, FamilyScope)
                    else str(family["family_id"])
                ),
            )
        ),
        by_source=metadata.scope.by_source,
    )
    metadata = ReadoutMetadata(
        scope=scope,
        cells=metadata.cells,
        partial=metadata.partial,
        partial_reason=metadata.partial_reason,
        source_content_digest_unavailable_reason=metadata.source_content_digest_unavailable_reason,
        source_content_digest=metadata.source_content_digest,
        artifact_manifest_sha256=metadata.artifact_manifest_sha256,
        artifact_relation_digests=metadata.artifact_relation_digests,
        coherence=metadata.coherence,
        effective_modes=metadata.effective_modes,
        captures=metadata.captures,
    )
    return LiftEstimates(
        updated,
        metadata=metadata,
        source=rows.source,
        sequential_snapshot=rows.sequential_snapshot,
        default_backend=rows._default_backend,
    )


def _cuped_recommendation_admitted(
    source: _BuiltinMomentSource, config: ResolvedMetricConfig
) -> bool:
    """Check the proposed CUPED method against the shared arm-support contract."""
    from increment.estimation.arm_contract import (
        MetricCapabilities,
        Supported,
        arm_cuped_advisory_support,
    )
    from increment.estimation.engine import is_open_ended
    from increment.sequential_source import adjustment_kind

    context = source.context
    metric = config.metric
    inference = context.plan.inference
    registration = getattr(inference, "registration", None)
    adjustment = adjustment_kind(registration, metric.name) if registration is not None else None
    capabilities = MetricCapabilities(
        metric_type=metric.type if metric.type not in ("total", "active") else "mean",
        value_scale="relative",
        winsorization=(
            "percentile"
            if getattr(getattr(metric, "winsorization", None), "has_percentile", False)
            else "fixed"
            if getattr(metric, "winsorization", None) is not None
            else "none"
        ),
        outcome_window="unbounded" if is_open_ended(metric) else "bounded",
        uptake_window="not_applicable",
    )
    support = arm_cuped_advisory_support(
        dependence="cluster" if context.cluster is not None else "iid",
        metric=capabilities,
        inference=inference,
        adjustment=adjustment,
        prior_present=config.prior is not None,
    )
    return isinstance(support, Supported)


_T = TypeVar("_T")


_ACTIVE_EXECUTION_CONTEXT: ContextVar[Any | None] = ContextVar(
    "_ACTIVE_EXECUTION_CONTEXT", default=None
)


@contextmanager
def _execution_context_scope(context: Any) -> Iterator[None]:
    token = _ACTIVE_EXECUTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_EXECUTION_CONTEXT.reset(token)


def _current_execution_context() -> Any | None:
    return _ACTIVE_EXECUTION_CONTEXT.get()


class Analysis:
    """Analyse one A/B experiment from declarative definitions.

    Parameters
    ----------
    experiment_name : str
        The ``name`` field of an experiment in the definitions YAML.
    definitions_path : str | Path
        Path to a definitions YAML file, or a directory of YAML files.
    con : SQLBackend
        Ibis backend connection holding the definitions' fact tables,
        bound once at construction.
    backend : str | None
        Ibis dialect name for introspected SQL. Defaults to ``con``'s
        own dialect.
    store : Literal["auto", "always", "none"]
        ``"auto"`` (default) materializes from the second top-level reduction;
        ``"always"`` starts with the first. Each qualifying call rescans current
        warehouse data; nested reductions reuse that call's materialization.
        ``"none"`` keeps reductions unmaterialized.
    on_mixed_assignment : Literal["error", "warn", "exclude"]
        Policy for invalid warehouse assignments: units observed in more
        than one arm or with no assignment label. ``"error"`` (default)
        refuses; ``"warn"``/``"exclude"`` drop them, warning once or not at
        all.
    source_snapshot_evidence : SourceSnapshotEvidence | None
        Caller-supplied event-time cutoff and optional certified per-feed
        completeness watermarks; absent certification remains unknown.
    """

    _input_backend: Backend | None = None
    _readout_snapshot_active: bool = False
    _readout_srm_observations: (
        dict[tuple[Any, ...], tuple[Any, tuple[str, Mapping[str, Any], str] | None]] | None
    ) = None

    def __init__(
        self,
        experiment_name: str,
        definitions_path: str | Path,
        con: SQLBackend,
        *,
        backend: str | None = None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
        source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    ) -> None:
        from increment.query.native_source import DefinitionsMomentSource
        from increment.query.session import WarehouseSession
        from increment.semantics.loader import load, verify_sql_admission_matches_execution

        if on_mixed_assignment not in ("error", "warn", "exclude"):
            _refuse(_INVALID_MIXED_ASSIGNMENT_POLICY, on_mixed_assignment=on_mixed_assignment)
        if store not in ("auto", "always", "none"):
            _refuse(_INVALID_STORE_POLICY, store=store)
        defs = load(definitions_path)
        experiment = defs.experiment(experiment_name)
        if experiment is None:
            _raise(
                "facade.analysis.unknown_experiment",
                requested=experiment_name,
                available=tuple(sorted(e.name for e in defs.experiments)),
            )
        metrics_by_name = {m.name: m for m in defs.metrics}
        metrics = [
            metrics_by_name[name] for name in experiment.metric_names if name in metrics_by_name
        ]
        design = _declared_design(experiment)
        experiment_plan = bind_automatic_sequential_plan(
            experiment.plan,
            metrics,
            design=design,
            source_id=experiment.name,
            source_mapping=native_observation_mapping(
                defs, experiment, on_mixed_assignment=on_mixed_assignment
            ),
            pre_period_covariate=experiment.n_pre_periods > 0,
            trigger=experiment.trigger,
        )
        if experiment_plan is not experiment.plan:
            experiment = Experiment.model_validate(
                experiment.model_copy(update={"plan": experiment_plan})
            )
            defs = defs.model_copy(
                update={
                    "experiments": tuple(
                        experiment if item.name == experiment.name else item
                        for item in defs.experiments
                    )
                }
            )
        compiled_plan = compile_decision_plan(
            experiment_plan,
            metrics,
            path="warehouse",
            design=design,
        )
        session = WarehouseSession(con, defs, source_snapshot_evidence=source_snapshot_evidence)
        verify_sql_admission_matches_execution(defs, con)
        session.drop_materialized()
        src = DefinitionsMomentSource(
            session,
            experiment,
            metrics,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
            backend=backend,
            design=design,
            plan=compiled_plan,
        )
        self._state = self._build_state(
            src=src,
            defs=defs,
            experiment=experiment,
            con=con,
            session=session,
            experiment_name=experiment_name,
            backend=backend,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
        )
        self._artifact_open_spec = None

    @classmethod
    def _build_state(
        cls,
        *,
        src: EvidenceSource | ContrastSource,
        defs: Definitions | None,
        experiment: Experiment | None,
        con: SQLBackend | None,
        session: WarehouseSession | None,
        experiment_name: str,
        backend: str | None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    ) -> AnalysisState:
        context = src.context
        if isinstance(context, ContrastContext):
            return ContrastAnalysisState(
                source=cast("ContrastSource", src),
                experiment_name=experiment_name,
            )
        design = context.design
        study = None if design is None else ParallelStudyEnvelope(identification=design)
        if defs is None:
            return SeamArmAnalysisState(
                source=cast("_BuiltinMomentSource", src),
                study=study,
                experiment_name=experiment_name,
            )
        if study is None:
            _raise("facade.analysis.definitions_state_requires_identification")
        return DefinitionsArmAnalysisState(
            source=cast("_BuiltinMomentSource", src),
            study=study,
            definitions=defs,
            experiment=cast("Experiment", experiment),
            connection=cast("SQLBackend", con),
            session=cast("WarehouseSession", session),
            experiment_name=experiment_name,
            backend=backend or "",
            store=store,
            on_mixed_assignment=on_mixed_assignment,
            exposure_lookup={e.name: e for e in defs.exposures},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release this analysis's resources without closing caller connections."""
        close = getattr(self._state.source, "close", None)
        if callable(close):
            close()

    def _require_arm_state(
        self, method: str, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> ArmAnalysisState:
        if self._state.family != "arm_moments":
            if population == "triggered":
                _refuse(
                    _TRIGGER_UNSUPPORTED,
                    method=method,
                    experiment="<unknown>",
                    trigger=None,
                    route=(
                        "switchback"
                        if self._state.family == "contrast"
                        else source_route(self._src)
                    ),
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
            _refuse(_CONTRAST_UNAVAILABLE, method=method)
        return self._state

    def _ensure_artifact_source(self) -> _BuiltinMomentSource:
        source = cast("_BuiltinMomentSource", self._state.source)
        spec = self._artifact_open_spec

        if spec is not None and cast("ArtifactMomentSource", source).closed:
            from increment.query.source import open_artifact

            source = open_artifact(
                spec.store,
                spec.ref,
                expected_context=spec.expected_context,
                verification=spec.verification,
            )
            self._state = replace(self._state, source=source)
        return source

    @property
    def _src(self) -> Any:
        return self._ensure_artifact_source()

    @property
    def _context(self) -> SourceContext | ContrastContext:
        return self._src.context

    @property
    def _defs(self) -> Definitions | None:
        state = self._state
        return state.definitions if isinstance(state, DefinitionsArmAnalysisState) else None

    @property
    def _experiment_name(self) -> str:
        return self._state.experiment_name

    @property
    def _backend(self) -> str | None:
        state = self._state
        if isinstance(state, DefinitionsArmAnalysisState):
            return state.backend or None
        return None

    @property
    def _con(self) -> SQLBackend | None:
        state = self._state
        return state.connection if isinstance(state, DefinitionsArmAnalysisState) else None

    @property
    def _store(self) -> Literal["auto", "always", "none"]:
        state = self._state
        return state.store if isinstance(state, DefinitionsArmAnalysisState) else "none"

    @property
    def _on_mixed_assignment(self) -> Literal["error", "warn", "exclude"]:
        state = self._state
        return (
            state.on_mixed_assignment if isinstance(state, DefinitionsArmAnalysisState) else "error"
        )

    @property
    def _session(self) -> WarehouseSession | None:
        state = self._state
        return state.session if isinstance(state, DefinitionsArmAnalysisState) else None

    @property
    def _metrics(self) -> list[Metric]:
        return list(self._context.metrics)

    def _execution_config_for_cell(self, cell: Any) -> tuple[Any, tuple[Method, ...]]:
        key = cell.identity.key
        if key.metric == "uptake" and key.estimand == "compliance":
            uptake = getattr(self._design, "uptake", None)
            if uptake is not None:
                return uptake, (Method(name="unadjusted"),)

        configs = getattr(self._context, "configs", None)
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
        return config.metric, tuple(effective_methods(config, design=self._design))

    @property
    def _design(self) -> Randomized | Encouragement | Observational | None:
        if isinstance(self._context, ContrastContext):
            return self._context.study.identification
        return self._context.design

    @property
    def _plan(self) -> CompiledDecisionPlan:
        return cast("SourceContext", self._context).plan

    @property
    def _experiment(self) -> Experiment:
        state = self._require_arm_state("experiment")
        if isinstance(state, DefinitionsArmAnalysisState):
            return state.experiment
        artifact_experiment = getattr(self._src, "artifact_experiment", None)
        if isinstance(artifact_experiment, Experiment):
            return artifact_experiment
        raise AttributeError("seam analysis has no Experiment definition")

    @staticmethod
    def _artifact_context(
        definitions: Definitions,
        experiment: Experiment,
        policy: Literal["error", "warn", "exclude"],
        design: Any | None = None,
    ) -> ArtifactContext:
        from increment.query.artifact_publish import artifact_context

        return artifact_context(
            definitions,
            experiment,
            policy,
            encouragement_uptake=design
            if getattr(design, "mechanism", None) == "encouragement"
            else None,
        )

    @staticmethod
    def _select_artifact_extensions(
        context: ArtifactContext,
        extensions: Sequence[Any],
    ) -> tuple[Any, ...]:
        from increment.query.artifact_contract import (
            ArtifactContractError,
            unit_day_artifact_extension_catalog,
        )
        from increment.query.artifact_digest import canonical_json

        entries = unit_day_artifact_extension_catalog(context)
        selected: list[Any] = []
        seen: set[str] = set()
        for requested in extensions:
            request = getattr(requested, "request", requested)
            if isinstance(request, Mapping):
                payload = dict(request)
            elif hasattr(request, "model_dump"):
                payload = request.model_dump(mode="json")
            else:
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "extension request must be a closed typed request",
                )
            identity = canonical_json(payload)
            if identity in seen:
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "duplicate extension request",
                )
            matches = [
                entry
                for entry in entries
                if canonical_json(entry.request.model_dump(mode="json")) == identity
            ]
            if len(matches) != 1:
                code = "artifact.extension.missing" if not matches else "artifact.extension.invalid"
                raise ArtifactContractError(
                    code,
                    "extension request is not selected exactly once in trusted context",
                )
            seen.add(identity)
            selected.append(matches[0].request)
        return tuple(selected)

    def publish_unit_day_artifact(
        self,
        store: ArtifactStore,
        *,
        extensions: Sequence[ArtifactExtensionRequest] = (),
        refresh_of: UnitDayArtifactRef | None = None,
    ) -> UnitDayArtifactRef:
        """Publish this definitions-backed arm analysis as an immutable artifact."""
        from increment.query.native_contract import NativeCoreSource

        state = self._require_arm_state("publish_unit_day_artifact")
        if not isinstance(state, DefinitionsArmAnalysisState):
            from increment._evidence_refusals import refuse_operation

            refuse_operation(
                self._src,
                "materialize",
                tuple(sorted(declared_evidence(self._src).operations)),
                route="publish_unit_day_artifact() requires Analysis.from_definitions",
            )
        native = _require_analysis_operation(
            self._src,
            "materialize",
            NativeCoreSource,
            message="publish_unit_day_artifact() requires a definitions-backed source",
        )
        context = self._artifact_context(
            cast(Definitions, self._defs),
            self._experiment,
            self._on_mixed_assignment,
            design=self._src.context.design,
        )
        selected = self._select_artifact_extensions(context, extensions)
        return cast(Any, native).publish_unit_day_artifact(
            store,
            extensions=selected,
            refresh_of=refresh_of,
        )

    @classmethod
    def _from_unit_day_artifact(
        cls,
        store: ArtifactStore,
        ref: UnitDayArtifactRef,
        *,
        expected_context: ArtifactContext,
        verification: Literal["lazy_digest"] = "lazy_digest",
    ) -> Self:
        """Open one caller-trusted immutable artifact generation."""
        from increment.query.source import open_artifact

        source = open_artifact(
            store,
            ref,
            expected_context=expected_context,
            verification=verification,
        )
        return cls._from_source(
            source,
            source.context.design,
            artifact_open_spec=_ArtifactOpenSpec(store, ref, expected_context, verification),
        )

    @classmethod
    def _from_definitions(
        cls,
        experiment_name: str,
        definitions_path: str | Path,
        con: SQLBackend,
        *,
        backend: str | None = None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
        source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    ) -> Self:
        """Build an :class:`Analysis` from a definitions YAML path/directory.

        Supply ``source_snapshot_evidence`` when the upstream source provides an
        explicit event-time cutoff; optional per-feed watermarks record only
        certified completeness, and absent certification stays unknown.
        Triggered membership, outcomes, and trigger-evidence publication refuse
        without this evidence; assigned-only operations do not require it.
        Unknown experiment names raise ``InvalidRequestError`` before warehouse access.
        """
        return cls(
            experiment_name,
            definitions_path,
            con,
            backend=backend,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
            source_snapshot_evidence=source_snapshot_evidence,
        )

    @classmethod
    def _from_source(
        cls,
        src: EvidenceSource | ContrastSource,
        design: Randomized | Encouragement | Observational | None = None,
        *,
        defs: Definitions | None = None,
        experiment: Experiment | None = None,
        con: SQLBackend | None = None,
        session: WarehouseSession | None = None,
        experiment_name: str | None = None,
        backend: str | None = None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
        artifact_open_spec: _ArtifactOpenSpec | None = None,
    ) -> Self:
        """Build an instance from the source's immutable construction state."""
        context = src.context
        if isinstance(context, ContrastContext):
            if design is not None and context.study.identification != design:
                _raise("facade.analysis.source_context_design_disagrees_arm")
        elif design is not None and context.design != design:
            _raise("facade.analysis.source_context_design_disagrees_arm")
        self = super().__new__(cls)
        self._artifact_open_spec = artifact_open_spec
        self._state = cls._build_state(
            src=src,
            defs=defs,
            experiment=experiment,
            con=con,
            session=session,
            experiment_name=experiment_name or src.context.study_id,
            backend=backend,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
        )
        return self

    @classmethod
    # Public factory signature is the API for summary-backed analysis.
    def _from_unit_summary(  # noqa: PLR0913
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        group: str,
        control: object | None = None,
        metrics: MetricsArg,
        experiment_id: str = "frame",
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
        uptake: str | None = None,
        on_unassigned: Literal["error", "exclude"] = "error",
        cluster: str | None = None,
        intervention_grain: Literal["unit", "cluster"] = "unit",
        exposure_date: str | None = None,
        breakouts: Sequence[str] = (),
    ) -> Self:
        """Build an :class:`Analysis` from a one-row-per-unit dataframe.

        No warehouse or YAML - pure narwhals over any supported frame
        library. See :func:`increment.frame.from_unit_summary` for the
        full parameter contract. Day-axis methods, ``panel_sql``/
        ``summary_sql``, and ``materialize`` raise
        :class:`CapabilityError`: this shape has no day axis or SQL
        substrate.

        Exactly one of *control*/*design* must be given: *control* is
        sugar for ``Randomized(control_group=control)``; *design* also
        accepts :class:`Observational` (routes :meth:`run` through
        IPTW) or :class:`Encouragement` (ITT/compliance/LATE, reading
        the uptake column named by *uptake*).

        *plan* declares this analysis's epistemic policy up front
        (run-wide ``alpha``/``alternative``/``inference``, plus
        per-metric ``ExperimentMetric.margin``/``margin_abs``) --
        required for :meth:`run` to apply role-based dispatch
        (primary/guardrail/secondary alpha allocation, non-inferiority
        margins) on this seam source instead of refusing the
        equivalent call-time kwargs. Omitted (default): the plan is
        undeclared; alpha is 0.05 and no sequential inference applies.

        *on_unassigned* controls a null value in *group*: ``"error"``
        (default) refuses; ``"exclude"`` drops those rows, with the
        count surfaced in ``unit_counts()`` and :meth:`srm`.

        *cluster* names the randomization-grain column when assignment
        was coarser than *unit*. ``intervention_grain`` separately declares
        whether deployment acts on units or whole clusters - see
        :func:`increment.frame.from_unit_summary` for the full contract.

        *exposure_date* names each unit's exposure date or day index.
        Sequential inference requires it and reveals units in that order,
        never in row order. With an ``on_unassigned="exclude"`` policy a
        null exposure drops the unit, as on :meth:`from_unit_panel`.

        *breakouts* names unit-stable columns used only for whole-window
        ``run_breakout`` reads. This source remains total-only: it does not
        expose daily or as-of methods, or a synthetic day axis.
        """
        from increment.query.fact_resolution import _require_design

        _require_design(control, design, constructor="Analysis.from_unit_summary")

        if control is None:
            assert design is not None
            control = str(design.control_group)
        from increment.frame import from_unit_summary as _make_source

        src = _make_source(
            frame,
            unit=unit,
            group=group,
            control=cast(str, control),
            metrics=metrics,
            experiment_id=experiment_id,
            uptake=uptake,
            design=design,
            plan=plan,
            on_unassigned=on_unassigned,
            cluster=cluster,
            intervention_grain=intervention_grain,
            exposure_date=exposure_date,
            breakouts=breakouts,
        )
        analysis = cls._from_source(src, src.context.design)
        analysis._input_backend = _input_backend_of(frame)
        return analysis

    @classmethod
    # Public factory signature is the API for panel-backed analysis.
    def _from_unit_panel(  # noqa: PLR0913
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        group: str,
        date: str,
        control: object | None = None,
        metrics: MetricsArg,
        experiment_id: str = "frame",
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
        uptake: str | None = None,
        exposure_date: str | None = None,
        observation_end: dt.date | dt.datetime | str | int | float | None = None,
        breakouts: Sequence[str] = (),
        on_unassigned: Literal["error", "exclude"] = "error",
        cluster: str | None = None,
    ) -> Self:
        """Build an :class:`Analysis` from a one-row-per-unit-per-day dataframe.

        No warehouse or YAML - pure narwhals. See
        :func:`increment.frame.from_unit_panel` for the full data model.
        Supports :meth:`run`, :meth:`run_daily`, :meth:`run_daily_lift`,
        :meth:`run_asof`, and :meth:`run_asof_lift`. ``metrics=`` may narrow
        the constructor-declared metric set, but cannot add undeclared metrics.
        Breakout declarations are fixed at construction; only a *dimension*
        named in *breakouts* is accepted. Every result stamps
        ``source=None``. ``panel_sql``/``summary_sql`` and
        ``materialize`` still raise :class:`CapabilityError`.

        Exactly one of *control*/*design* must be given - see
        :meth:`from_unit_summary` for the contract.

        *plan* declares this analysis's epistemic policy up front - see
        :meth:`from_unit_summary` for the contract.

        *exposure_date* names each unit's day-0 date column, required
        whenever a metric declares ``window_days`` or
        ``type="retention"``; it also gates unwindowed outcomes to
        on-or-after-exposure rows. *observation_end* bounds late-
        enrollee censoring for windowed/retention metrics.

        *breakouts* declares stable per-unit reporting columns: values
        must not vary within a unit, nulls form a ``"__null__"``
        segment, and names cannot overlap roles or metric inputs.
        An unwindowed mean, ratio or conversion metric may declare a CUPED
        covariate that is constant within each unit; a covariate that
        varies within a unit refuses. Windowed and retention metrics
        refuse a panel covariate: for a windowed metric, compute each unit's
        windowed value upstream and declare it as an unwindowed metric on
        :meth:`from_unit_summary`; no frame source serves a per-unit retention
        value, so remove the covariate to run retention without CUPED.
        Sequential CUPED is not available from this constructor.

        *on_unassigned* controls an unusable unit (null *group*, or a
        null *exposure_date* when given): ``"error"`` (default)
        refuses; ``"exclude"`` drops those units, with the count
        surfaced in ``unit_counts()`` and :meth:`srm`.
        """
        from increment.query.fact_resolution import _require_design

        _require_design(control, design, constructor="Analysis.from_unit_panel")

        if control is None:
            assert design is not None
            control = str(design.control_group)
        from increment.frame import from_unit_panel as _make_source

        src = _make_source(
            frame,
            unit=unit,
            group=group,
            date=date,
            control=cast(str, control),
            metrics=metrics,
            experiment_id=experiment_id,
            uptake=uptake,
            design=design,
            plan=plan,
            exposure_date=exposure_date,
            observation_end=observation_end,
            breakouts=breakouts,
            on_unassigned=on_unassigned,
            cluster=cluster,
        )
        analysis = cls._from_source(src, src.context.design)
        analysis._input_backend = _input_backend_of(frame)
        return analysis

    @classmethod
    def _from_moments(  # noqa: PLR0915
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        metrics: MetricsArg,
        control: object | None = None,
        experiment_id: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
    ) -> Self:
        """Build an :class:`Analysis` that rehydrates a moments cube
        (e.g. exported via :meth:`export`, or a warehouse table of
        pre-aggregated moments).

        Per-arm ``n``/mean/variance or exact conversion-success summaries
        are not wire rows; use :func:`increment.estimators.moments_from_summaries` to
        build the supported fixed-horizon mean/conversion cube. That helper
        does not synthesize ratio, covariate, clustered, or sequential state;
        use :meth:`from_unit_summary` when those inputs are required.

        At moment grain, ``run()`` works. ``unit_counts()``, ``srm()``, and
        health allocation checks require source-level ``assignment_counts``;
        without that field they are unavailable with
        ``source.moments.assignment_counts``. The unit-frame method (needed
        for IPTW/DML) raises :class:`CapabilityError`, as do ``sql()`` and
        every method the other seam constructors already refuse (day axis,
        ``panel_sql``, ``run_breakout``, ``materialize``).

        *rows* must carry centered ``group_summary`` columns, a fixed-horizon
        ``moments_format=12`` stamp, integer ``n``, nullable integer
        ``successes``, the ratio denominator's third moment ``cden3`` on
        every denominator row, a source identity record, and a complete
        embedded ``decision_plan`` on every fixed-horizon row. Format 13 adds
        method-scoped ``record_kind="cuped_evidence"`` envelopes for several
        scalar families or categorical-only multicovariate requests; a request
        the cube carries no evidence for is refused before estimation. The
        earlier fixed formats 10 and 11 remain readable; their ratio rows
        predate ``cden3``, so their lift rows report the denominator-skew
        coverage check as unavailable. Declared binary outcomes require exact
        success counts.
        Sequential exports use ``moments_format=10`` checkpoint envelopes
        and a version-3 sequential wire plan; the preceding sequential
        format 9 remains readable.
        Encouragement exports additionally carry cluster identity, member
        counts, and full bivariate uptake/size moments on every row. Supply
        the same Encouragement design when reloading; an omitted experiment_id
        is recovered from the cube for this design. An explicit identity must
        match the exported cohort. A
        cube without that state cannot provide design-level compliance.
        An empty-metric fixed-horizon Encouragement export carries one
        complete format-12 ``design_summary`` envelope. Reload it with
        ``metrics=[]``; both its embedded and effective plans must remain
        fixed-horizon and metric-free, even when *plan* is supplied.

        Exactly one of *control*/*design* must be given - see
        :meth:`from_unit_summary` for the contract.

        *plan* explicitly overrides the embedded compiled plan when supplied.
        Without an override, the payload's complete roles, methods,
        priors, inference, and view policies are used as exported.
        """

        from increment._frame_validation import (
            _coerce_source_metrics,
            canonical_control_for_moments,
            canonical_design_for_moments,
            validate_moments_control,
        )
        from increment.frame import synthesise_metric
        from increment.semantics.design import Encouragement, Randomized
        from increment.sources import MomentsSource

        design = canonical_design_for_moments(design)
        from increment.query.fact_resolution import _require_design

        _require_design(control, design, constructor="Analysis.from_moments")

        if control is None:
            assert design is not None
            control = canonical_control_for_moments(design.control_group)
        else:
            control = canonical_control_for_moments(control)
        validate_moments_control(rows, control)
        if design is None:
            design = MomentsSource.design_summary_design(rows, control_group=control) or Randomized(
                control_group=control
            )
        specs = _coerce_source_metrics(metrics, design=design)

        from increment._analysis_config import (
            refuse_cuped_linear_moments,
            refuse_missing_scalar_evidence,
        )
        from increment._cuped_evidence import invalid_evidence, method_key
        from increment.sources import cuped_evidence_items

        evidence_items = cuped_evidence_items(rows, require_plan=plan is None)
        evidence = {
            metric: {key: item.kind for key, item in items.items()}
            for metric, items in evidence_items.items()
        }
        portable_metrics: set[str] = set()
        for spec in specs:
            for item in evidence_items.get(spec.name, {}).values():
                if item.kind == "categorical_cells":
                    expected_components = 2 if spec.type == "ratio" else 1
                    actual_components = (
                        0 if item.cells is None else len(item.cells.outcome_references)
                    )
                elif item.kind == "fitted_multicovariate":
                    expected_components = 2 if spec.type == "ratio" else 1
                    actual_components = 0 if item.fit_state is None else len(item.fit_state.slopes)
                else:
                    continue
                if actual_components != expected_components:
                    invalid_evidence(
                        f"{item.kind} evidence for metric {spec.name!r} has "
                        f"{actual_components} components; expected {expected_components}"
                    )
            methods_for_metric = (
                () if spec.decision_method is None else (spec.decision_method,)
            ) + spec.sensitivity_methods
            carried = evidence.get(spec.name, {})
            for method in methods_for_metric:
                carried_kind = carried.get(method_key(method))
                if method.variance_reduction == "cuped_multicovariate":
                    if carried_kind not in ("categorical_cells", "fitted_multicovariate"):
                        refuse_cuped_linear_moments(spec.name)
                    portable_metrics.add(spec.name)
            if "scalar_moments" in carried.values():
                # Method-scoped evidence identifies each scalar family exactly.
                for method in methods_for_metric:
                    if (
                        method.variance_reduction == "cuped"
                        and carried.get(method_key(method)) != "scalar_moments"
                    ):
                        refuse_missing_scalar_evidence("Analysis.from_moments", method.name)
                portable_metrics.add(spec.name)
                continue
            requested_covariates = {
                covariate.name
                for method in methods_for_metric
                if method.variance_reduction == "cuped"
                for covariate in method.covariates or ()
            }
            provenance_names = {
                str(row["covariate_name"])
                for row in rows
                if row.get("metric") == spec.name and row.get("covariate_name") is not None
            }
            if (
                requested_covariates
                and provenance_names
                and provenance_names != requested_covariates
            ):
                from increment._analysis_config import refuse_unidentified_scalar_moments

                refuse_unidentified_scalar_moments(spec.name)
            if requested_covariates and any(
                row.get("metric") == spec.name
                and row.get("ref_x") is not None
                and row.get("x_role") not in (None, "covariate")
                for row in rows
            ):
                from increment._analysis_config import refuse_unidentified_scalar_moments

                refuse_unidentified_scalar_moments(spec.name)
        metric_catalog = [synthesise_metric(spec) for spec in specs]
        from increment._analysis_config import resolve_configs

        configs = resolve_configs(
            metric_catalog,
            bindings=None,
            specs={spec.name: spec for spec in specs},
            methods=None,
            prior=None,
        )
        from increment._analysis_config import validate_scalar_route

        validate_scalar_route(
            tuple(config for config in configs if config.metric.name not in portable_metrics),
            route="Analysis.from_moments",
            representation="moments",
            design=design,
        )
        if experiment_id is None:
            experiment_id = (
                str(rows[0].get("experiment_id", "moments"))
                if rows
                and (
                    isinstance(design, Encouragement)
                    or rows[0].get("record_kind") == "sequential_checkpoint"
                )
                else "moments"
            )
        src = MomentsSource(
            rows,
            metrics=metric_catalog,
            study_id=experiment_id,
            design=design,
            plan=plan,
            configs=configs,
        )
        return cls._from_source(src, design)

    @classmethod
    def _from_switchback_panel(
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        cycle: str,
        period: str,
        step: str,
        group: str,
        metrics: MetricsArg,
        identification: Randomized,
        assignment: SwitchbackAssignment,
        experiment_id: str = "frame",
        plan: AnalysisPlan | None = None,
        contrast_references: Mapping[str, UnitCycleReference] | None = None,
    ) -> Self:
        """Build a fixed-horizon analysis under the declared switchback law.

        ``assignment.sequence`` accepts ``IndependentBernoulliOrder`` for
        independent unit-cycle draws (a declared envelope or explicit t approximation), or
        ``SharedScheduleOrder`` for one Bernoulli CT/TC draw per complete
        two-period block and fixed roster (block-t inference).

        Both estimate the mean-unit retained-window additive effect. Retain
        steps at or after ``washout_steps + carryover_order``; orders 0/1/2
        work when ``observation_steps > carryover_order``. Absence of further
        carryover is a declared assumption, not a diagnostic conclusion.
        Results and diagnostics preserve the law, independent grain, window
        lengths and unit/cycle/block counts.

        Incomplete schedules and implausible CT/TC splits refuse with
        ``source.frame.switchback.schedule`` before statistics are built.
        Compatibility with the schedule checks cannot prove randomization.
        """
        from increment.switchback import from_switchback_panel as _make_source

        src = _make_source(
            frame,
            unit=unit,
            cycle=cycle,
            period=period,
            step=step,
            group=group,
            metrics=metrics,
            identification=identification,
            assignment=assignment,
            experiment_id=experiment_id,
            plan=plan,
            contrast_references=contrast_references,
        )
        analysis = cls._from_source(cast("ContrastSource", src), identification)
        analysis._input_backend = _input_backend_of(frame)
        return analysis

    def materialize(self) -> Self:
        """Rebuild spine and per-metric stats from current warehouse data.

        Each call performs a fresh scan, overriding ``store="auto"``'s
        second-reduction threshold. A no-op only for ``store="none"`` or
        an experiment declaring no metrics.

        Returns
        -------
        Self
            This analysis, using its refreshed warehouse data.

        Notes
        -----
        A frame or moments source raises :class:`CapabilityError` because
        it has no warehouse relation to materialize.
        """
        self._require_arm_state("materialize")
        src = _require_analysis_operation(
            self._src,
            "materialize",
            MaterializeOperation,
            message=(
                "materialize() needs a native Analysis.from_definitions "
                "instance -- a frame/warehouse/moments-backed analysis has "
                "nothing to materialize."
            ),
        )
        if self._store != "none" and self._metrics:
            src.materialize()
        return self

    def dashboard_snapshot(
        self, operation: Callable[[Self], DashboardSnapshotPayload], *, metrics: Sequence[Metric]
    ) -> DashboardSnapshotPayload:
        """Prepare one complete dashboard payload against an isolated, pinned source.

        *metrics* names every metric the payload reads, so the pin covers exactly their
        fact sources. Pass the declared metrics together with any
        :attr:`available_metrics` the operation will add through ``exploratory_metrics``:
        an added metric outside the pin is refused rather than read from a later state.
        """
        self._require_arm_state("dashboard_snapshot")
        _require_analysis_operation(
            self._src,
            "readout_snapshot",
            message=(
                "dashboard_snapshot() needs a native Analysis.from_definitions instance "
                "with a warehouse-backed source."
            ),
        )
        scoped = evidence_provider(self._src).scope(
            EvidenceScope(
                metrics=tuple(metrics),
                snapshot=True,
                include_breakouts=True,
            )
        )
        with scoped.source as pinned:
            from increment.readouts._evidence import _builtin_source

            return self._with_pinned_source(_builtin_source(pinned), operation)

    def _with_pinned_source(
        self, pinned: _BuiltinMomentSource, operation: Callable[[Self], _T]
    ) -> _T:
        isolated = copy.copy(self)
        isolated._state = replace(self._state, source=pinned)
        # The pinned snapshot must never be reopened over: drop any reopen pin
        # the copy would otherwise inherit.
        isolated._artifact_open_spec = None
        isolated._readout_snapshot_active = True
        isolated._readout_srm_observations = {}
        try:
            return operation(isolated)
        finally:
            delattr(isolated, "_readout_snapshot_active")
            delattr(isolated, "_readout_srm_observations")

    def dashboard_breakout_reads(self, breakout: Breakout) -> DashboardBreakoutReads:
        """The dashboard's Explore reads through one declared breakout alone.

        A dimension-wide read covers every declared breakout of that property, so one
        source's refusal would refuse its siblings. These reads hand the source only
        *breakout*; families are computed per breakout, so their rows equal this breakout's
        rows from the dimension-wide reads. Only native ``Analysis.from_definitions``
        instances support it: other arm-family sources refuse with
        ``evidence.breakout_unavailable`` and switchback evidence with
        ``facade.analysis.contrast_unavailable``. An undeclared *breakout* is refused
        with ``facade.analysis.undeclared_breakout``.

        Parameters
        ----------
        breakout : Breakout
            One of this experiment's declared breakouts.

        Returns
        -------
        DashboardBreakoutReads
            The cumulative-lift, cumulative-value, daily-value and segment reads.
        """
        state = self._require_arm_state("dashboard_breakout_reads")
        if not isinstance(state, DefinitionsArmAnalysisState):
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self._src,
                (breakout.property,),
                tuple(declared_evidence(self._src).breakouts),
                route="dashboard_breakout_reads() requires a definitions-backed breakout source",
            )
        declared = state.experiment.breakouts
        if breakout not in declared:
            _refuse(
                _UNDECLARED_BREAKOUT,
                requested=(breakout.source, breakout.property),
                declared=tuple((item.source, item.property) for item in declared),
            )
        scoped = copy.copy(self)
        experiment = state.experiment.model_copy(update={"breakouts": (breakout,)})
        scoped._state = replace(state, experiment=experiment)
        return DashboardBreakoutReads(breakout, scoped)

    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> tuple[DashboardGroupData, ...]:
        """Return group data for the requested assigned or triggered population."""
        self._require_arm_state("dashboard_group_data")
        selected = select_metrics(
            self._metrics,
            metrics,
            caller="dashboard_group_data",
            require_declared_definitions=True,
        )
        src = _require_analysis_operation(
            self._src,
            "dashboard_group_data",
            DashboardGroupDataOperation,
            message=(
                "dashboard_group_data() needs a native Analysis.from_definitions "
                "instance with a warehouse-backed source."
            ),
        )
        return tuple(
            src.dashboard_group_data(
                metrics=selected, checkpoints=checkpoints, population=population
            )
        )

    def allocation_history(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> pa.Table:
        """Daily and cumulative enrollment history for one named population.

        Columns are ``experiment_id``, ``ds`` (the declared day boundary),
        ``group_id``, ``n_daily``, ``n_cumulative`` and ``analysis_population``.
        Assigned histories use first-exposure dates; triggered histories use
        first eligible trigger dates and count only trigger-eligible units.
        Counts are independent of metric maturity.

        Sources without raw enrollment events and clustered experiments refuse
        this unit-grain timeline with a coded capability error.
        """
        src = _require_analysis_operation(
            self._src,
            "allocation_history",
            AllocationHistoryOperation,
            message=(
                "allocation_history() needs a native Analysis.from_definitions instance; "
                "a frame/warehouse/moments-backed analysis retains no raw event stream "
                "to build a daily enrollment timeline from."
            ),
        )
        return src.allocation_history(population=population)

    def health_check(
        self, *, expected_allocation: Mapping[str, float] | None = None
    ) -> ExperimentHealth:
        from increment._health import check_health

        if getattr(self, "_readout_snapshot_active", False) or isinstance(
            self._state, ContrastAnalysisState
        ):
            return self._collect_health(expected_allocation)

        def collect(pinned: _BuiltinMomentSource) -> ExperimentHealth:
            if pinned is self._src:
                return self._collect_health(expected_allocation)
            return self._with_pinned_source(
                pinned,
                lambda isolated: isolated._collect_health(expected_allocation),
            )

        return check_health(self._src, metrics=self._metrics, collect=collect)

    def _collect_health(
        self,
        expected_allocation: Mapping[str, float] | None,
        *,
        allocation_observations: Mapping[
            str,
            tuple[SRMResult | NotApplicable | None, tuple[str, Mapping[str, Any], str] | None],
        ]
        | None = None,
    ) -> ExperimentHealth:
        from increment.estimation.assignment_integrity import assignment_integrity
        from increment.estimation.diagnostics import NotApplicable
        from increment.estimation.health import ExperimentHealth, HealthCheck

        experiment = getattr(self._src, "artifact_experiment", None)
        trigger_name = getattr(experiment, "trigger", None) or getattr(
            self._context, "trigger_name", None
        )
        triggered = trigger_name is not None
        checks: list[HealthCheck] = []
        # Reuse SRM results already collected on this isolated pinned analysis.
        if allocation_observations is None:
            captured = getattr(self, "_readout_srm_observations", None)
            if captured is not None:
                expected_key = (
                    None
                    if expected_allocation is None
                    else tuple(sorted(expected_allocation.items()))
                )
                allocation_observations = {
                    population: captured[key]
                    for population in ("assigned", "triggered")
                    if (key := (population, expected_key, 0.001, "always_valid")) in captured
                } or None

        def srm_check(population: Literal["assigned", "triggered"]) -> HealthCheck:
            check_name = "assigned_allocation" if population == "assigned" else "triggered_balance"
            if population == "assigned" and isinstance(self._state, ContrastAnalysisState):
                return HealthCheck(
                    check=check_name,
                    analysis_population=population,
                    availability="unavailable",
                    code="integrity.switchback_assignment_law",
                    context={
                        "reason": "Switchback order randomization is not a unit-arm allocation law."
                    },
                )
            if population == "assigned":
                admission_design = self._design
                if (
                    admission_design is not None
                    and expected_allocation is not None
                    and getattr(admission_design, "allocation_scheme", None) == "independent"
                    and getattr(admission_design, "allocation", None) is None
                ):
                    admission_design = admission_design.model_copy(
                        update={"allocation": dict(expected_allocation)}
                    )
                integrity = assignment_integrity(admission_design, None)
                if integrity.status == "not_applicable":
                    return HealthCheck(
                        check=check_name,
                        analysis_population=population,
                        availability="not_applicable",
                        context={
                            "reason": "The declared design has no assignment allocation to test."
                        },
                    )
                if integrity.status != "not_checked_missing_counts":
                    return HealthCheck(
                        check=check_name,
                        analysis_population=population,
                        availability="unavailable",
                        code=integrity.code,
                        context={
                            **dict(integrity.context),
                            "reason": f"Assignment-law admission is {integrity.status}.",
                            **(
                                {
                                    "route": (
                                        "Declare allocation_scheme='independent' on a design "
                                        "that uses an independent constant-probability draw, or "
                                        "pass the target shares as Analysis.srm(inference='fixed', "
                                        "expected={...}) for one predeclared look."
                                    )
                                }
                                if integrity.code == "integrity.allocation_scheme_missing"
                                else {}
                            ),
                        },
                    )
            if allocation_observations is not None and population in allocation_observations:
                result, refusal = allocation_observations[population]
                if result is None:
                    code, refusal_context, reason = refusal or (
                        "",
                        {},
                        "No allocation result was captured.",
                    )
                    availability = (
                        "not_applicable"
                        if code == "dashboard.allocation_not_applicable"
                        else "unavailable"
                    )
                    return HealthCheck(
                        check=check_name,
                        analysis_population=population,
                        availability=availability,
                        code=code or None,
                        context={**dict(refusal_context), "reason": reason},
                    )
            else:
                try:
                    result = self.srm(
                        expected=None if expected_allocation is None else dict(expected_allocation),
                        population=population,
                    )
                except CapabilityError as refusal:
                    return HealthCheck(
                        check=check_name,
                        analysis_population=population,
                        availability="unavailable",
                        code=refusal.code,
                        context={**dict(refusal.context), "reason": str(refusal)},
                    )
                except InvalidRequestError as refusal:
                    if refusal.code not in {
                        "estimation.diagnostics.total_observed_count",
                        "estimation.diagnostics.always_srm_predeclared",
                    } or (
                        refusal.code == "estimation.diagnostics.always_srm_predeclared"
                        and population != "triggered"
                    ):
                        raise
                    context = {**dict(refusal.context), "reason": str(refusal)}
                    if refusal.code == "estimation.diagnostics.always_srm_predeclared":
                        context["route"] = (
                            "Declare allocation on the design or pass expected_allocation "
                            "to health_check()."
                        )
                    return HealthCheck(
                        check=check_name,
                        analysis_population=population,
                        availability="unavailable",
                        code=refusal.code,
                        context=context,
                    )
            if isinstance(result, NotApplicable):
                return HealthCheck(
                    check=check_name,
                    analysis_population=population,
                    availability="not_applicable",
                    context={"reason": result.reason},
                )
            return HealthCheck(
                check=check_name,
                analysis_population=population,
                availability="available",
                conclusion=(
                    "warning"
                    if result.is_srm
                    or result.low_expected_count
                    or result.unassigned_units
                    or result.mixed_assignment_units
                    else "not_rejected"
                ),
                context={
                    "grain": result.grain,
                    "observed": dict(result.observed),
                    "expected": dict(result.expected),
                    "unassigned_units": result.unassigned_units,
                    "mixed_assignment_units": result.mixed_assignment_units,
                    "low_expected_count": result.low_expected_count,
                },
                srm=result,
            )

        checks.append(srm_check("assigned"))
        if triggered:
            checks.append(srm_check("triggered"))
            try:
                rates = self.trigger_rates()
            except CapabilityError as refusal:
                checks.append(
                    HealthCheck(
                        check="trigger_rates",
                        analysis_population="triggered",
                        availability="unavailable",
                        code=refusal.code,
                        context={**dict(refusal.context), "reason": str(refusal)},
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        check="trigger_rates",
                        analysis_population="triggered",
                        availability="available",
                        conclusion="descriptive",
                        context={"description": "Observed trigger share by assigned arm."},
                        rates=rates,
                    )
                )
            self._append_trigger_maturity(checks)
            self._append_feed_completeness(checks)
            checks.append(self._trigger_assumptions())
        else:
            for check_name in (
                "triggered_balance",
                "trigger_rates",
                "trigger_maturity",
                "feed_completeness",
                "trigger_assumptions",
            ):
                checks.append(
                    HealthCheck(
                        check=check_name,
                        analysis_population="triggered",
                        availability="not_applicable",
                        context={"reason": "This experiment does not declare a trigger."},
                    )
                )
        check_order = (
            "assigned_allocation",
            "triggered_balance",
            "trigger_rates",
            "trigger_maturity",
            "feed_completeness",
            "trigger_assumptions",
        )
        checks.sort(
            key=lambda check: (
                check_order.index(check.check),
                check.analysis_population,
                check.metric or "",
                check.feed or "",
            )
        )
        return ExperimentHealth(
            checks=tuple(checks),
            source_snapshot_id=self._health_snapshot_id(checks),
            source_cutoff=self._health_cutoff(),
        )

    def _append_trigger_maturity(self, checks: list[HealthCheck]) -> None:
        from increment.estimation.health import HealthCheck
        from increment.query.artifact_contract import ArtifactContractError

        edges = getattr(self._src, "triggered_observation_edges", None)
        for metric in sorted(self._metrics, key=lambda item: item.name):
            if not callable(edges):
                checks.append(
                    HealthCheck(
                        check="trigger_maturity",
                        analysis_population="triggered",
                        metric=metric.name,
                        availability="unavailable",
                        code="evidence.operation_unavailable",
                        context={
                            "constructor": type(self._src).__name__,
                            "requested": "triggered_observation_edges",
                            "offered": tuple(sorted(getattr(self._src, "operations", ()))),
                            "route": "This source does not retain trigger maturity evidence.",
                        },
                    )
                )
                continue
            try:
                observed, finalized = edges(metric)
            except (CapabilityError, ArtifactContractError) as refusal:
                checks.append(
                    HealthCheck(
                        check="trigger_maturity",
                        analysis_population="triggered",
                        metric=metric.name,
                        availability="unavailable",
                        code=refusal.code,
                        context={**dict(refusal.context), "reason": str(refusal)},
                    )
                )
                continue
            checks.append(
                HealthCheck(
                    check="trigger_maturity",
                    analysis_population="triggered",
                    metric=metric.name,
                    availability="available",
                    conclusion="warning" if finalized is None else "descriptive",
                    context={
                        "observed_edge": observed.isoformat(),
                        "finalized_edge": None if finalized is None else finalized.isoformat(),
                        "reason": "The metric window is not certified complete."
                        if finalized is None
                        else "The retained finalized prefix is shown; no later horizon is inferred.",
                    },
                )
            )

    def _append_feed_completeness(self, checks: list[HealthCheck]) -> None:
        from increment.estimation.health import HealthCheck

        if isinstance(self._src, TriggeredHealthEvidenceOperation):
            evidence = self._src.triggered_health_evidence()
            for item in evidence.feeds:
                feed = item.feed or item.evidence_kind
                if item.metric is not None:
                    feed = f"{feed}:{item.metric}"
                watermark = item.complete_through
                checks.append(
                    HealthCheck(
                        check="feed_completeness",
                        analysis_population="triggered",
                        metric=item.metric,
                        feed=feed,
                        availability="unavailable" if watermark is None else "available",
                        conclusion=None if watermark is None else "descriptive",
                        context={
                            "evidence_kind": item.evidence_kind,
                            "watermark": None if watermark is None else watermark.isoformat(),
                            "reason": (
                                "A SQL-backed assignment exposure has no fact feed whose "
                                "completeness can be certified."
                                if item.evidence_kind == "native_sql_assignment"
                                else "A SQL-backed trigger has no fact feed whose completeness "
                                "can be certified."
                                if item.evidence_kind == "native_sql_trigger"
                                else "No certified completeness watermark is retained for this feed."
                                if watermark is None
                                else "Certified completeness watermark at the source cutoff."
                            ),
                        },
                    )
                )
            if evidence.feeds:
                return
        checks.append(
            HealthCheck(
                check="feed_completeness",
                analysis_population="triggered",
                availability="unavailable",
                context={
                    "reason": "This source retains no per-feed completeness certificates.",
                    "watermark": None,
                },
            )
        )

    def _trigger_assumptions(self) -> HealthCheck:
        from increment.estimation.family import sequential_family_guarantee
        from increment.estimation.health import HealthCheck

        runtime_inference = self._plan.inference
        declared_plan = getattr(getattr(self._src, "artifact_experiment", None), "plan", None)
        if declared_plan is None and isinstance(self._state, DefinitionsArmAnalysisState):
            declared_plan = self._state.experiment.plan
        inference = getattr(declared_plan, "inference", runtime_inference)
        registration = getattr(inference, "triggered_registration", None)
        family_guarantee = (
            sequential_family_guarantee(registration) if registration is not None else None
        )
        return HealthCheck(
            check="trigger_assumptions",
            analysis_population="triggered",
            availability="available",
            conclusion="descriptive",
            context={
                "counterfactual_trigger_invariance": "assumed, not established by balance or rates",
                "trigger_order_stationarity": "assumed, not established by balance or rates",
                "no_effect_before_trigger": "assumed, not established by balance or rates",
                "triggered_look_predictability": (
                    "assumed, not verified: looks, stops and freezes use only already-revealed "
                    "triggered-chain information, not assigned results that reveal future "
                    "triggered outcomes, including the same metric"
                    if family_guarantee is not None
                    else None
                ),
                "family_guarantee": family_guarantee,
                "family_guarantee_status": (
                    "assumed, not verified" if family_guarantee is not None else None
                ),
                "risk": (
                    "Balanced trigger rates do not verify counterfactual trigger invariance.",
                    "Trigger-order stationarity remains an assumption.",
                    "Assigned results can reveal future triggered outcomes, including "
                    "the same metric; they must not inform triggered family stops or freezes.",
                ),
            },
        )

    def _health_snapshot_id(self, checks: Sequence[HealthCheck]) -> str | None:
        evidence = [check.model_dump(mode="json") for check in checks]
        return _health_snapshot_id(self._src, self._health_cutoff(), evidence)

    def _health_cutoff(self) -> str | None:
        if isinstance(self._src, TriggeredHealthEvidenceOperation):
            cutoff = self._src.triggered_health_evidence().observation_cutoff
            return None if cutoff is None else cutoff.isoformat()
        evidence = self._session.source_snapshot_evidence if self._session is not None else None
        return None if evidence is None else evidence.observation_cutoff_ts.isoformat()

    def unit_counts(self) -> dict[str, int]:
        """Return enrolled unit counts by canonical group label.

        Excluded unassigned units appear under ``"(unassigned)"``. Sources
        without total-grain count evidence refuse with a coded capability error.
        """
        source = self._src
        if not is_custom_source(source):
            source = _require_analysis_operation(
                source,
                "unit_counts",
                message="unit_counts() requires a source with enrolled total-grain counts.",
            )
        request = CountsRequest(kind="unit", population=getattr(source, "population", "assigned"))
        return dict(
            cast(
                CountEvidence,
                request_evidence(evidence_provider(source), request),
            ).counts
        )

    def srm(
        self,
        *,
        expected: dict[str, float] | None = None,
        alpha: float = 0.001,
        inference: Literal["always_valid", "fixed"] = "always_valid",
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> SRMResult | NotApplicable:
        """Check assigned or triggered counts against a declared allocation.

        The default is an anytime-valid check for cumulative prefixes under
        a known allocation with constant conditional arm probabilities; pass
        ``inference="fixed"`` for one predeclared Pearson look. On a clustered
        experiment, the chi-square runs over distinct cluster counts per arm.

        ``expected`` supplies the target allocation for this call; otherwise
        the design must declare ``allocation``. SRM never assumes equal shares
        or infers its null from observed counts.

        Parameters
        ----------
        expected : dict[str, float] | None
            Expected allocation proportions per group.
        alpha : float
            Lifetime significance threshold (default 0.001).
        inference : Literal["always_valid", "fixed"]
            Evidence controlling ``is_srm``. Default ``"always_valid"``.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) checks enrolled counts.
            ``"triggered"`` narrows to the declared trigger population first.

        Raises
        ------
        CapabilityError
            If the requested population is unavailable or no target allocation
            is declared. Declare ``Randomized(control_group=..., allocation={...},
            allocation_scheme=...)`` to check the design allocation.

        Returns
        -------
        SRMResult | NotApplicable
            Observational designs return :class:`NotApplicable`, since they
            have no target allocation to test against. Runs normally under
            :class:`Encouragement`.
        """
        self._require_arm_state("srm")
        if population not in ("assigned", "triggered"):
            _refuse(_INVALID_POPULATION, population=population)
        src = self._src
        observations = getattr(self, "_readout_srm_observations", None)
        key = None
        if observations is not None:
            key = (
                population,
                None if expected is None else tuple(sorted(expected.items())),
                alpha,
                inference,
            )
        try:
            if population == "triggered":
                src = _require_analysis_operation(
                    src,
                    "triggered_counts",
                    message=(
                        "srm(population='triggered') needs a native "
                        "Analysis.from_definitions instance -- a frame/warehouse/"
                        "moments-backed analysis retains no raw event stream to "
                        "narrow to a triggered population."
                    ),
                )
                if expected is None and getattr(self._design, "allocation", None) is None:
                    # Match assignment_integrity: no scheme -> allocation_scheme_missing; independent with no allocation -> allocation_missing_or_nonconstant.
                    _raise(
                        (
                            "integrity.allocation_scheme_missing"
                            if getattr(self._design, "allocation_scheme", None) is None
                            else "integrity.allocation_missing_or_nonconstant"
                        ),
                        population=population,
                    )
                from increment.estimation.diagnostics import (
                    complete_srm_support,
                    resolve_srm_expected,
                    sample_ratio_mismatch,
                )

                expected = resolve_srm_expected(
                    expected,
                    allocation=getattr(self._design, "allocation", None),
                    inference=inference,
                )
                triggered = evidence_provider(src).counts(CountsRequest(kind="triggered"))
                grain, counts, unit_counts = (
                    cast("Literal['unit', 'cluster']", triggered.grain),
                    dict(triggered.counts),
                    dict(triggered.unit_counts),
                )
                counts = complete_srm_support(counts, expected=expected)
                if unit_counts:
                    unit_counts = complete_srm_support(unit_counts, expected=expected)
                result = sample_ratio_mismatch(
                    counts,
                    expected=expected,
                    alpha=alpha,
                    inference=inference,
                    grain=grain,
                    unit_counts=unit_counts,
                )
            else:
                result = readouts.srm(
                    src,
                    expected=expected,
                    alpha=alpha,
                    inference=inference,
                )
        except CapabilityError as refusal:
            if observations is not None and key is not None:
                observations[key] = (
                    None,
                    (refusal.code, dict(refusal.context), str(refusal)),
                )
            raise
        if observations is not None and key is not None:
            observations[key] = (result, None)
        return result

    def trigger_rates(self) -> dict[str, float]:
        """Observed triggered share per arm.

        Feed a pooled estimate (e.g. the mean of these per-arm rates) into
        ``Baseline(..., trigger_rate=...)``, then ``required_sample_size(...)``,
        to plan a follow-up at the right size.
        """
        from increment.query.native_contract import NativeCoreSource

        source = self._src
        if is_artifact_route(source):
            if "triggered_counts" not in declared_evidence(source).operations:
                from increment._evidence_refusals import refuse_operation

                refuse_operation(
                    source,
                    "triggered_counts",
                    tuple(sorted(declared_evidence(source).operations)),
                    route="trigger_rates() requires complete trigger evidence",
                )
            return source.trigger_rates()
        src = _require_analysis_operation(
            source,
            "triggered_counts",
            NativeCoreSource,
            message=(
                "trigger_rates() needs a native Analysis.from_definitions "
                "instance with a warehouse backend"
            ),
        )
        return src.trigger_rates()

    def _validate_trigger_fires_in_every_arm(self, source: object | None = None) -> None:
        from increment.readouts._evidence import validate_trigger_fires_in_every_arm

        validate_trigger_fires_in_every_arm(
            cast(Any, self._src if source is None else source),
            self._experiment.trigger,
        )

    def _cate_source(self, method: str) -> _BuiltinMomentSource:
        self._require_arm_state(method)
        experiment = getattr(self, "_experiment", None)
        if experiment is None or experiment.trigger is None:
            return cast("_BuiltinMomentSource", self._src)
        triggered = _require_analysis_operation(
            self._src,
            "triggered_source",
            message=(
                f"{method}() needs a source-owned triggered unit view; this source "
                "cannot reconstruct trigger membership."
            ),
        )
        return triggered.triggered_source()

    def sitewide(
        self,
        metric: str,
        *,
        arm: str | None = None,
        alpha: float = 0.05,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> SitewideImpact | SitewideRatioImpact:
        """Whole-site impact of *metric*'s lift to every unit.

        Native (:meth:`from_definitions`) and adopted artifact sources with a
        published site-volume extension combine enrolled-arm moments with
        whole-site volume evidence. Both populations require an additive metric
        aligned with the calendar-window site total: metrics must be unwindowed
        and use sum or count aggregation. Conversion metrics are refused because
        their per-unit indicator is not additive across event records; declare a
        count metric on the conversion fact to read conversion events site-wide.
        ``population="assigned"`` scales the assignment-window lift
        to all enrolled units. For a triggered experiment,
        ``population="triggered"`` scales the trigger-stratum lift to units in
        finalized trigger membership, while outcomes remain aligned to the
        experiment's calendar window; it additionally requires a declared
        experiment end and discloses the conditional membership assumptions
        on the result.

        One result per call, for one treatment arm - required when
        more than one non-control arm is enrolled. Every other
        enrolled arm is netted out of the counterfactual baseline and
        counted in the ship-to-all population: the number answers
        "``arm`` shipped to every unit, versus no treatment shipped."

        Parameters
        ----------
        metric : str
            Name of a metric declared on this experiment.
        arm : str | None
            ``group_id`` of the treatment arm to score.
        alpha : float
            Two-sided significance level for both intervals. Default 0.05.
        population : Literal["assigned", "triggered"]
            selects assignment membership by default or finalized trigger
            membership for a triggered experiment. Triggered sitewide impact
            uses assignment-anchored calendar-window outcomes (not trigger-
            anchored ``run()`` windows); it is conditional on membership
            independence from treatment and never-triggered units being
            unaffected.

        Returns
        -------
        SitewideImpact | SitewideRatioImpact
            ``SitewideRatioImpact`` for a :class:`RatioMetric`, else ``SitewideImpact``.

        Notes
        -----
        The readout raises :class:`CapabilityError` for sources without
        site volume, windowed or non-additive metrics, triggered impact
        without a declared experiment end, retention or quantile metrics,
        and clusters that do not decompose or whose mean cluster sizes differ by
        more than 20%. Use an unwindowed sum or count metric. Unexposed
        units in a treated cluster can bias relative impact downward.

        It raises :class:`ValueError` for an undeclared metric, no enrolled
        arm, an unknown requested arm, or alpha outside the open unit interval.
        """
        if population not in ("assigned", "triggered"):
            _refuse(_INVALID_POPULATION, population=population)
        experiment = getattr(self, "_experiment", None)
        from increment._sitewide import SitewideReadouts, SitewideRequest

        return SitewideReadouts(src=self._src, experiment=experiment).run(
            SitewideRequest(
                metric_name=metric,
                arm=arm,
                alpha=alpha,
                population=population,
            )
        )

    def export(self, path: str | Path) -> None:
        """Write this experiment's moments (per-metric ``group_summary``
        rows, unioned) to a parquet file at *path*.

        Native, frame, and artifact sources export total moments and
        design-level compliance state. :meth:`from_moments` reloads them. Every row carries a ``moments_format`` version
        stamp; :meth:`from_moments` validates and strips it, and
        refuses a file written in a future wire format instead of
        misreading it.

        A fixed-horizon export refuses a quantile metric on every source with
        ``source.frame.quantile_no_moments`` (``readout.observational.quantile``
        under an observational design), from the metric catalog alone before any
        evidence is read: a moments cube holds no per-unit values. A registered
        sequential checkpoint holds unit-record proofs, not moments rows, so it
        still exports when a catalog quantile is not among its registered models.
        """
        src = _require_analysis_operation(
            self._src,
            "export_moments",
            ExportMomentsOperation,
            message=(
                "export() requires a native, frame, or artifact source with "
                "the export_moments operation."
            ),
        )
        src.export_moments(path)

    def _readout_source(
        self,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> _BuiltinMomentSource:
        """Validate a requested population; the evidence request selects it."""
        if population == "triggered":
            if is_custom_source(self._src):
                from increment._evidence_refusals import require_population

                require_population(self._src, population)
            else:
                _require_analysis_operation(
                    self._src,
                    "triggered_source",
                    message=(
                        "A triggered readout requires a source retaining the raw event stream "
                        "and its declared trigger."
                    ),
                )
        return self._src

    # Properties

    @property
    def experiment(self) -> Experiment:
        """The resolved experiment definition."""
        return self._experiment

    @property
    def artifact_context(self) -> ArtifactContext:
        """The trusted artifact context for this definitions-backed analysis.

        Pass this value as ``expected_context`` when reopening an artifact
        published from the same analysis.
        """
        state = self._require_arm_state("artifact_context")
        if not isinstance(state, DefinitionsArmAnalysisState):
            from increment._evidence_refusals import refuse_operation

            refuse_operation(
                self._src,
                "materialize",
                tuple(sorted(declared_evidence(self._src).operations)),
                route="artifact_context requires Analysis.from_definitions",
            )
        return self._artifact_context(
            state.definitions,
            self.experiment,
            self._on_mixed_assignment,
            design=self._design,
        )

    @property
    def metrics(self) -> list[Metric]:
        """All metrics and guardrails for this experiment."""
        return list(self._metrics)

    @property
    def compiled_decision_plan(self) -> CompiledDecisionPlan:
        """The immutable decision plan this analysis will execute."""
        plan = getattr(self, "_plan", None)
        if plan is None:
            _raise("store.run.legacy_analysis_required", route="Analysis.run()")
        return plan

    @property
    def available_metrics(self) -> list[Metric]:
        """Saved metrics the experiment does not declare and could estimate, in definitions order.

        These are the names ``exploratory_metrics=`` accepts: per-unit metrics on the
        experiment's own unit. Report-only ``total``/``active`` metrics (no per-unit variance)
        and metrics of another entity are not offered. Only ``Analysis.from_definitions``
        carries saved definitions to add from; every other source refuses with
        ``facade.analysis.exploratory_metrics_source_limited``.
        """
        state = self._state
        if not isinstance(state, DefinitionsArmAnalysisState):
            _refuse(_EXPLORATORY_SOURCE_LIMITED, method="available_metrics")
        declared = {*state.experiment.metric_names, *(metric.name for metric in self._metrics)}
        return [
            metric
            for metric in state.definitions.metrics
            if metric.name not in declared
            and metric.type not in ("total", "active")
            and getattr(metric, "entity", None) == state.experiment.unit
        ]

    def _exploratory_metrics(self, names: Sequence[str] | None, *, caller: str) -> list[Metric]:
        """Resolve ``exploratory_metrics=`` before any query; ``[]`` when none were named."""
        if not names:
            return []
        if not isinstance(self._state, DefinitionsArmAnalysisState):
            _refuse(_EXPLORATORY_SOURCE_LIMITED, method=caller)
        added = select_exploratory_metrics(
            self.available_metrics,
            {*self._experiment.metric_names, *(metric.name for metric in self._metrics)},
            names,
            caller=caller,
        )
        inference = self._plan.inference
        if getattr(inference, "registration", None) is not None:
            _refuse(
                _EXPLORATORY_SEQUENTIAL,
                method=caller,
                names=[metric.name for metric in added],
                inference=type(inference).__name__,
            )
        return added

    def _exploratory_analysis(self, added: Sequence[Metric], *, caller: str) -> Self:
        """This analysis reading from a sibling source that also carries *added*."""
        state = self._require_arm_state(caller)
        src = _require_analysis_operation(
            self._src,
            "exploratory_source",
            ExploratorySourceOperation,
            message=f"{caller}(exploratory_metrics=...) needs a native Analysis.from_definitions instance.",
        )
        derived = copy.copy(self)
        derived._state = replace(state, source=src.exploratory_source(metrics=added))
        return derived

    # Caching and materialization

    # SQL introspection

    def build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
        horizon_metrics: Sequence[Metric] | None = None,
    ) -> Any:
        """Return one native metric's live panel tuple, independent of TEMP tables."""
        from increment.query.native_contract import NativeCoreSource

        src = _require_analysis_operation(
            self._src,
            "panel_sql",
            NativeCoreSource,
            message=(
                "build_panel_for_metric() needs a native "
                "Analysis.from_definitions instance with a warehouse backend"
            ),
        )
        return src.build_panel_for_metric(
            exposures,
            metric,
            population=population,
            horizon_metrics=horizon_metrics,
        )

    def panel_sql(self, breakouts: list[Breakout] | None = None) -> dict[str, str]:
        """Return per-metric panel SQL without executing."""
        from increment.query.native_contract import NativeCoreSource

        src = _require_analysis_operation(
            self._src,
            "panel_sql",
            NativeCoreSource,
            message=(
                "panel_sql() needs a native Analysis.from_definitions "
                "instance -- a frame/warehouse/moments-backed analysis has "
                "no SQL substrate to introspect."
            ),
        )
        return src.panel_sql(breakouts=breakouts or ())

    def summary_sql(self, breakouts: list[Breakout] | None = None) -> dict[str, str]:
        """Return the per-metric group-summary SQL without executing.

        Parameters
        ----------
        breakouts : list[Breakout] | None
            When given, adds one dimensioned group_summary SQL entry

        Returns
        -------
        dict[str, str]
            ``{metric_name: group_summary_SQL_string}``, plus one
            breakout entry per requested breakout.
        """
        self._require_arm_state("summary_sql")
        src = self._src
        if "summary_sql" not in declared_evidence(src).operations:
            return src.sql()
        summary = _require_analysis_operation(
            src,
            "summary_sql",
            SummarySqlOperation,
            message=(
                "summary_sql() needs a native Analysis.from_definitions "
                "instance -- a frame/warehouse/moments-backed analysis has "
                "no SQL substrate to introspect."
            ),
        )
        return summary.summary_sql(breakouts=breakouts or ())

    def planning_baseline(self, metric: str) -> Baseline | SwitchbackBaseline:
        """The planning input a power solver needs, computed from THIS
        analysis's own data and declared design -- i.e. a prior or pilot
        experiment (or an in-flight experiment's pre-treatment data), not
        the future test you are about to plan. If you have no prior
        experiment or pilot data, build a Baseline directly instead
        (Baseline(mean=, var=), Baseline.from_proportion,
        Baseline.from_absorption, Baseline.from_ratio).

        An arm analysis returns a :class:`~increment.power.Baseline` for
        ``required_sample_size``/``achieved_power``/
        ``minimum_detectable_effect``. Outcome moments describe the analyzed
        population: under a declared trigger, the triggered population, with
        ``trigger_rate`` its observed share of the assigned control arm;
        with no trigger declared, the assigned population and
        ``trigger_rate=1.0``. From that population's control arm: mean/var
        (a ratio metric's linearized moments; a quantile metric returns a
        ``QuantileBaseline`` over its per-unit control values); cuped_rho,
        when the metric declares CUPED, as the control arm's variance
        reduction under the runtime's own CUPED fit; compliance from the
        arms' uptake difference when the design is Encouragement; and
        cluster_icc/cluster_size_cv from contributing clusters, with ICC on
        the analyzed score. avg_cluster_size retains assigned units per
        randomized cluster; cluster_participation is the pilot fraction of
        assigned clusters contributing analyzed units.

        A switchback analysis (``from_switchback_panel``) returns the
        pilot-fitted :class:`~increment.power.switchback.SwitchbackBaseline`
        for the ``switchback_*`` solvers.

        Refuses by SOURCE, naming which, where the constructor cannot
        supply what the metric type or declared design needs -- including
        a declared trigger whose evidence the source does not carry."""
        from increment._planning_baseline import planning_baseline

        if self._state.family == "contrast":
            context = cast("ContrastContext", self._context)
            spec = select_metrics(context.metrics, [metric], caller="planning_baseline")[0]
            return cast("ContrastSource", self._src).planning_baseline(spec)
        experiment = getattr(self, "_experiment", None)
        trigger = getattr(experiment, "trigger", None)
        if trigger is None:
            from increment._source_types import is_custom_source

            if is_custom_source(self._src):
                trigger = getattr(self._context, "trigger_name", None)
        return planning_baseline(
            cast("_BuiltinMomentSource", self._src),
            metric,
            trigger=trigger,
            trigger_rates=self.trigger_rates,
        )

    def capture_sequential(
        self, *, finalized: bool, previous=None, as_of=None, freeze: Sequence[str] = ()
    ):
        """Capture a registered finalized prefix; freshness is not finalization.

        Pass a previous snapshot to verify append-only continuation. Corrections
        to any retained unit outcome, assignment or definition refuse the path.
        Panel and relational sources require an explicit ``as_of`` horizon.

        ``freeze`` names metrics to stop monitoring at exactly this capture:
        every registered cell of each named metric (each arm and segment) that
        has data keeps this capture's evidence at every later look; see
        ``declare_sequential_freeze``. A frozen secondary still takes part in
        its family, whose e-BH selection is redone at each look over frozen and
        current evidence, so its discovery and selected interval can change.

        A triggered family guarantee requires stopping and freezing decisions
        based only on already-revealed triggered-chain information. Assigned
        results can reveal future triggered outcomes, including the same
        metric, and must not inform those decisions.
        """
        from increment.readouts._evidence import load_sequential_capture

        source = cast(Any, self._state.source)
        request = SequentialCaptureRequest(
            finalized=finalized, previous=previous, as_of=as_of, freeze=tuple(freeze)
        )
        context = _current_execution_context()
        if context is None:
            return load_sequential_capture(source, request)

        from increment._source_types import is_custom_source
        from increment.readouts._evidence import _builtin_source

        context.validate_request(
            source, tuple(source.context.metrics), tuple(source.context.configs)
        )
        with context.borrow() as scoped:
            capture_source = scoped if is_custom_source(source) else _builtin_source(scoped)
            snapshot = load_sequential_capture(capture_source, request)
        if not is_custom_source(source):
            from increment.sequential_source import adopt_source_snapshot

            snapshot = adopt_source_snapshot(source, snapshot)
        return snapshot

    def sequential_snapshot(self, *, previous=None):
        """Retrieve a current exact checkpoint, optionally proving its parent."""
        from increment.readouts._evidence import current_source_snapshot

        source = cast(Any, self._state.source)
        context = _current_execution_context()
        if context is None:
            return current_source_snapshot(source, previous=previous)

        from increment._source_types import is_custom_source
        from increment.readouts._evidence import _builtin_source

        context.validate_request(
            source, tuple(source.context.metrics), tuple(source.context.configs)
        )
        with context.borrow() as scoped:
            snapshot_source = scoped if is_custom_source(source) else _builtin_source(scoped)
            snapshot = current_source_snapshot(snapshot_source, previous=previous)
        if not is_custom_source(source):
            from increment.sequential_source import adopt_source_snapshot

            snapshot = adopt_source_snapshot(source, snapshot)
        return snapshot

    def __repr__(self) -> str:
        study = getattr(self._state, "study", None)
        design = None if study is None else study.identification
        details = [f"  experiment={self._state.experiment_name!r}"]
        if design is not None:
            details.append(f"  design={type(design).__name__}")
        return "Analysis(\n" + ",\n".join(details) + "\n)"

    __str__ = __repr__

    def __repr_args__(self) -> list[tuple[str, object]]:
        study = getattr(self._state, "study", None)
        args: list[tuple[str, object]] = [("experiment", self._state.experiment_name)]
        if study is not None:
            args.append(("design", type(study.identification).__name__))
        return args

    def _persistence_identity(
        self,
        results: Any,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        extra_metrics: Sequence[Metric] = (),
        runtime_request: Mapping[str, object] | None = None,
    ):
        from increment.kernel.contracts import Registration, contract_hash
        from increment.store.producer import PersistenceIdentity

        design = cast(Any, self._design)
        if isinstance(self._state, ContrastAnalysisState):
            procedures = cast("ContrastContext", self._context).procedures
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
            procedures = self._plan.procedures
            registration_hash = Registration.from_plan(self._plan).hash()
        cells = results.as_cells()
        cell_metrics = {cell.identity.key.metric for cell in cells}
        metrics = {metric.name: metric for metric in (*self._metrics, *extra_metrics)}
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
        from increment._canonical import declaration_value

        source_context = getattr(self._src, "context", None)
        declaration_identity = getattr(self._src, "declaration_identity", None)
        if callable(declaration_identity):
            declaration = declaration_identity()
            declaration_unavailable_reason = None
        else:
            declaration = {"context": declaration_value(source_context)}
            declaration_unavailable_reason = "source_declaration_identity_unavailable"
        source_declaration_hash = contract_hash(
            "source_declaration_v1", declaration_value(declaration)
        )
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
        return PersistenceIdentity(
            design=design,
            metrics=metrics,
            registration_hash=registration_hash,
            methods=methods,
            source_refs=source_refs,
            source_declaration_identity_unavailable_reason=declaration_unavailable_reason,
        )

    def run(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        estimands: Sequence[str] | None = None,
        value_scale: Mapping[str, ValueScale] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] | None = None,
        coherence: Literal["captured", "live"] = "captured",
        store: Any | None = None,
        _execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates | ContrastResults:
        """Run the analysis through its immutable experiment and bound resolver."""
        from increment._analysis_kernel_adapter import run_analysis

        options = {
            "decision_method": decision_method,
            "sensitivity_methods": sensitivity_methods,
            "prior": prior,
            "metrics": metrics,
            "estimands": estimands,
            "value_scale": value_scale,
            "exploratory_metrics": exploratory_metrics,
            "population": population,
            "coherence": coherence,
            "store": store,
            "_execution_context": _execution_context,
        }
        if _execution_context is None:
            return cast("LiftEstimates | ContrastResults", run_analysis(self, **options))
        with _execution_context_scope(_execution_context):
            return cast("LiftEstimates | ContrastResults", run_analysis(self, **options))

    def _run_legacy(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        estimands: Sequence[str] | None = None,
        value_scale: Mapping[str, ValueScale] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] | None = None,
        coherence: Literal["captured", "live"] = "captured",
        store: Any | None = None,
        _execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates | ContrastResults:
        """Run the full A/B test analysis pipeline.

        Epistemic policy defaults come from the source's `AnalysisPlan` and
        per-metric `ExperimentMetric` bindings; the parameters below override
        them for this call only.

        Parameters
        ----------
        decision_method : Method | _Unset
            Decision estimator override; UNSET inherits each metric's
            declaration. An explicit override replaces a declared,
            non-default value for that run.
        sensitivity_methods : Sequence[Method] | _Unset
            Additional estimator overrides; UNSET inherits each declaration.
            An explicit override replaces a declared, non-default value for
            that run.
        prior : Prior | None | _Unset
            Informative prior override; UNSET inherits each declaration, None
            resets it. An explicit override replaces a declared, non-default
            value for that run. A mixture prior is refused on
            observational/encouragement rows - pass ``Normal``.
        metrics : Sequence[str | Metric] | None
            Narrow the estimated metrics. ``None`` (default) selects
            every declared metric.
        estimands : Sequence[str] | None
            Under :class:`Encouragement` only: among ``"itt"``/
            ``"compliance"``/``"late"``, default all three.
        value_scale : Mapping[str, ValueScale] | None
            Observational-only: report a metric's rows as additive ATE
            instead of relative lift.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to estimate in addition to the declared
            ones (``from_definitions`` only). Their rows carry ``role="exploratory"``
            under the default unassigned procedure (two-sided, at the plan's full
            ``alpha``), join no plan family, and leave every declared row unchanged.
            ``metrics=[]`` with ``exploratory_metrics`` returns only the added rows.
        population : Literal["assigned", "triggered"] | None
            Source population for the arm readout. When omitted, return each captured population
            as separate result rows; an explicit value selects only that population. Triggered
            reads require a declared trigger and sufficient source evidence.
        coherence : Literal["captured", "live"]
            Arm-readout source consistency. ``"captured"`` (default) requests one
            source-owned snapshot; warehouse sources require TEMP materialization
            and refuse if it is unavailable. ``"live"`` opts out of that guarantee
            and marks results as live-qualified.
        store : Any | None
            Optional result store. When provided, persist the complete run
            and expose its ``run_id`` and ``analysis_spec_hash`` in result metadata.
        _execution_context : EvidenceExecutionContext | None
            Internal run owner used when this readout is composed with other operations.

        Returns
        -------
        LiftEstimates | ContrastResults
            Arm evidence returns one estimate per (metric x method x
            non-control arm). Switchback evidence returns one contrast per
            selected metric. Switchback calls accept only UNSET role/prior overrides;
            ``estimands`` and ``value_scale`` remain arm-only.
            Each arm estimate carries an explicit ``analysis_population`` axis
            (``assigned`` or ``triggered``); the two populations are distinct
            result identities.
        """

        _execution_context = _current_execution_context()
        from increment.breakout.estimates import LiftEstimates

        if isinstance(self._state, ContrastAnalysisState) and population == "triggered":
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run",
                experiment="<unknown>",
                trigger=None,
                route="switchback",
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        added = self._exploratory_metrics(exploratory_metrics, caller="run")
        runtime_request: dict[str, object] = {}
        if prior is not UNSET:
            runtime_request["prior"] = prior
        if estimands is not None:
            runtime_request["estimands"] = tuple(estimands)
        if value_scale is not None:
            runtime_request["value_scale"] = value_scale
        if isinstance(self._state, ContrastAnalysisState):
            rows = self._run_contrast(
                decision_method,
                sensitivity_methods,
                prior,
                metrics,
                estimands,
                value_scale,
                execution_context=_execution_context,
            )
            rows._default_backend = self._input_backend
            if store is not None:
                from uuid import uuid4

                store.persist_results(
                    rows,
                    self._state.experiment_name,
                    uuid4().hex,
                    self._persistence_identity(
                        rows, decision_method, sensitivity_methods, runtime_request=runtime_request
                    ),
                )
            return rows
        selected = select_metrics(cast("Sequence[Metric]", self._metrics), metrics, caller="run")

        def request(chosen: Sequence[Metric]) -> WholeWindowRequest:
            return WholeWindowRequest(
                metrics=tuple(chosen),
                estimands=tuple(estimands) if estimands is not None else None,
                value_scale=value_scale,
                population=population,
                coherence=coherence,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )

        rows = (
            self._whole_window().run(request(selected), execution_context=_execution_context)
            if selected or not added
            else LiftEstimates()
        )
        if not isinstance(rows, LiftEstimates):
            rows = LiftEstimates(rows)
        if added:
            extra = (
                self._exploratory_analysis(added, caller="run")
                ._whole_window()
                .run(request(added), execution_context=_execution_context)
            )
            extra = LiftEstimates(
                [row for row in extra if row.estimand != "compliance"],
                metadata=extra.metadata,
                source=extra.source,
                sequential_snapshot=extra.sequential_snapshot,
                default_backend=extra._default_backend,
            )
            extra = _stamp_exploratory_rows(extra)
            rows = extra if rows.metadata is None else rows.concat(extra)
        self._warn_run_advisories(
            selected,
            added,
            estimands=estimands,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
        if _execution_context is not None and rows.metadata is not None:
            metadata = rows.metadata
            rows._metadata = type(metadata).model_validate(
                {
                    **metadata.__dict__,
                    "coherence": (
                        None
                        if "provider" in _execution_context.effective_modes.values()
                        else _execution_context.coherence
                    ),
                    "effective_modes": dict(_execution_context.effective_modes),
                }
            )
        rows._default_backend = getattr(self, "_input_backend", None)
        if store is not None:
            from uuid import uuid4

            store.persist_results(
                rows,
                self._state.experiment_name,
                uuid4().hex,
                self._persistence_identity(
                    rows,
                    decision_method,
                    sensitivity_methods,
                    added,
                    runtime_request=runtime_request,
                ),
            )
        return rows

    def _warn_run_advisories(
        self,
        selected: Sequence[Metric],
        added: Sequence[Metric],
        *,
        estimands: Sequence[str] | None,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
    ) -> None:
        from increment._frame_validation import _warn

        state = self._state
        assert not isinstance(state, ContrastAnalysisState)
        source = state.source

        # A compliance-only encouragement read uses no outcome configuration.
        if (
            estimands is not None
            and not {"itt", "late"}.intersection(estimands)
            and getattr(source.context.design, "mechanism", None) == "encouragement"
        ):
            return
        selected_names = {metric.name for metric in selected}
        declared_configs = {
            config.metric.name: config
            for config in source.context.configs
            if config.metric.name in selected_names
        }
        default_decision = (
            Method(name="iptw")
            if getattr(source.context.design, "mechanism", None) == "observational"
            else Method(name="unadjusted")
        )
        conflicts: dict[str, list[tuple[str, object]]] = {
            "decision_method": [],
            "sensitivity_methods": [],
            "prior": [],
        }
        for name, config in declared_configs.items():
            if (
                decision_method is not UNSET
                and not config.decision_defaulted
                and config.decision_method != default_decision
                and config.decision_method != decision_method
            ):
                conflicts["decision_method"].append((name, config.decision_method))
            if (
                sensitivity_methods is not UNSET
                and config.sensitivity_methods
                and tuple(config.sensitivity_methods)
                != tuple(cast(Sequence[Method], sensitivity_methods))
            ):
                conflicts["sensitivity_methods"].append((name, tuple(config.sensitivity_methods)))
            if (
                prior is not UNSET
                and config.prior is not None
                and not config.prior_is_global
                and config.prior != prior
            ):
                conflicts["prior"].append((name, config.prior))

        overrides = {
            "decision_method": decision_method,
            "sensitivity_methods": (
                tuple(cast(Sequence[Method], sensitivity_methods))
                if sensitivity_methods is not UNSET
                else sensitivity_methods
            ),
            "prior": prior,
        }
        for setting, declarations in conflicts.items():
            if declarations:
                _warn(
                    "analysis.run.override_replaces_declared",
                    setting=setting,
                    metrics=tuple(name for name, _ in declarations),
                    declared=tuple(declarations),
                    override=overrides[setting],
                )

        config_sources: list[tuple[Any, _BuiltinMomentSource]] = []
        if selected:
            config_sources.extend(
                (config, state.source)
                for config in overlay_configs(
                    selected,
                    state.source.context.configs,
                    methods=None,
                    prior=None,
                    prior_override=prior,
                    decision_method=decision_method,
                    sensitivity_methods=sensitivity_methods,
                )
            )
        if added:
            exploratory = self._exploratory_analysis(added, caller="run")
            exploratory_source = cast("_BuiltinMomentSource", exploratory._state.source)
            config_sources.extend(
                (config, exploratory_source)
                for config in overlay_configs(
                    added,
                    exploratory_source.context.configs,
                    methods=None,
                    prior=None,
                    prior_override=prior,
                    decision_method=decision_method,
                    sensitivity_methods=sensitivity_methods,
                )
            )
        unused: list[str] = []
        for config, source in config_sources:
            covariate_available = config.covariate_available
            design = source.context.design
            if (
                not covariate_available
                or not config.decision_defaulted
                or config.methods_explicitly_empty
                or getattr(design, "mechanism", None) == "observational"
                or any(
                    method.variance_reduction == "cuped" for method in config.sensitivity_methods
                )
                or not _cuped_recommendation_admitted(source, config)
            ):
                continue
            unused.append(config.metric.name)
        if unused:
            _warn(
                "analysis.cuped.covariate_unused",
                metric_names=tuple(unused),
                route=(
                    "Request Method(name='cuped', variance_reduction='cuped') "
                    "as decision_method or a sensitivity method in run()."
                ),
            )

    def _run_contrast(
        self,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        metrics: Sequence[str | Metric] | None,
        estimands: Sequence[str] | None,
        value_scale: Mapping[str, ValueScale] | None,
        *,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> ContrastResults:
        validate_contrast_request(
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            estimands=estimands,
            value_scale=value_scale,
        )
        context = cast("ContrastContext", self._context)
        selected = select_metrics(cast("Sequence[Metric]", context.metrics), metrics, caller="run")
        request = ContrastReadoutRequest(
            metrics=tuple(selected),
            procedures={
                metric.name: context.procedures[metric.name]
                for metric in selected
                if metric.name in context.procedures
            },
        )
        source = cast("ContrastSource", self._src)
        return ContrastEvidenceOperation().run(
            source,
            request,
            execution_context=execution_context,
        )

    def _whole_window(self) -> WholeWindowReadouts:
        self._require_arm_state("run")
        return WholeWindowReadouts(
            src=cast("_BuiltinMomentSource", self._src),
            plan=self._plan,
            design=self._design,
            experiment=getattr(self, "_experiment", None),
            readout_source=self._readout_source,
            validate_trigger=self._validate_trigger_fires_in_every_arm,
            arm_moments=readouts.arm_moments,
        )

    def estimate_cate(
        self,
        metric: str,
        *,
        control: str | None = None,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        alpha: float = 0.05,
        ard: bool = False,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
    ) -> CateResult:
        """Fit CATE using this analysis's unit source and declared intervention."""
        from increment.cate import estimate_cate

        if control is None:
            assert self._design is not None
            control = self._design.control_group
        return estimate_cate(
            self._cate_source("estimate_cate"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            alpha=alpha,
            ard=ard,
            cluster_weight=cluster_weight,
        )

    def validate_cate(
        self,
        metric: str,
        *,
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        n_groups: int = 5,
        alpha: float = 0.05,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
        bootstrap: ClusterBootstrap | None = None,
        include_evaluation_population: bool = False,
    ) -> CateValidation:
        """Validate CATE on an honest, cluster-atomic holdout.

        Member counts and independent cluster counts are reported separately;
        unavailable uncertainty retains its reason and numeric nulls.
        """
        from increment.cate import validate_cate

        return validate_cate(
            self._cate_source("validate_cate"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            n_groups=n_groups,
            alpha=alpha,
            cluster_weight=cluster_weight,
            bootstrap=bootstrap,
            include_evaluation_population=include_evaluation_population,
        )

    def targeting_rule(
        self,
        metric: str,
        *,
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        fraction: float,
        alpha: float = 0.05,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
        bootstrap: ClusterBootstrap | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
        include_evaluation_population: bool = False,
    ) -> TargetingRule:
        """Evaluate a precommitted deployment fraction on an honest holdout."""
        from increment.cate import targeting_rule

        return targeting_rule(
            self._cate_source("targeting_rule"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            fraction=fraction,
            alpha=alpha,
            cluster_weight=cluster_weight,
            bootstrap=bootstrap,
            deploy_grain=deploy_grain,
            include_evaluation_population=include_evaluation_population,
        )

    def select_targeting_rule(  # noqa: PLR0913
        self,
        metric: str,
        *,
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        fractions: Sequence[float],
        cost_per_treated: float = 0.0,
        n_folds: int = 5,
        seed: int,
        alpha: float = 0.05,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
        bootstrap: ClusterBootstrap | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
        include_evaluation_population: bool = False,
    ) -> TargetingSelection:
        """Select a deployment budget on inner folds, then report the untouched holdout."""
        from increment.cate import select_targeting_rule

        return select_targeting_rule(
            self._cate_source("select_targeting_rule"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            fractions=fractions,
            cost_per_treated=cost_per_treated,
            n_folds=n_folds,
            seed=seed,
            alpha=alpha,
            cluster_weight=cluster_weight,
            bootstrap=bootstrap,
            deploy_grain=deploy_grain,
            include_evaluation_population=include_evaluation_population,
        )

    def assignment_diagnostic(self) -> SwitchbackAssignmentDiagnostic:
        """Return switchback assignment integrity evidence."""
        if self._state.family != "contrast":
            _refuse(_SWITCHBACK_ONLY, method="assignment_diagnostic")
        source = self._src
        return source.diagnostics

    def breakout_summaries(
        self, *, metrics: Sequence[str | Metric] | None = None
    ) -> dict[str, dict[str, pa.Table]]:
        """Per-breakout, per-metric moment tables.

        For every ``Experiment.breakouts`` x metric pair, computes both
        the arm-level ``group_summary`` and the day- or retention-cohort-level
        ``daily_group_summary``, executes them, and returns the
        pyarrow tables. Materializes the moment tables only - it does
        not turn them into ``LiftEstimate``s.


        Parameters
        ----------
        metrics : Sequence[str | Metric] | None
            Restrict to declared names or field-equal metric definitions.
            ``None`` (default) computes for every declared metric.

        Returns
        -------
        dict[str, dict[str, pa.Table]]
            ``{f"{metric.name}:{breakout.property}:{source_name}" :
            {"group_summary": pa.Table, "daily_group_summary": pa.Table}}`` when
            no trigger is declared. Triggered experiments add ``:assigned`` /
            ``:triggered`` key suffixes and an ``analysis_population`` column.
        """
        from increment.breakout.estimates import reject_quantile_metrics, reject_retention_metrics
        from increment.query.native_contract import NativeViewSource

        self._require_arm_state("breakout_summaries")
        if not isinstance(self._src, NativeViewSource):
            from increment._evidence_refusals import refuse_breakout
            from increment._source_types import declared_evidence

            offered = tuple(sorted(declared_evidence(self._src).breakouts))
            experiment = getattr(self._state, "experiment", None)
            requested = (
                tuple(breakout.property for breakout in experiment.breakouts)
                if experiment is not None
                else ()
            )
            refuse_breakout(
                self._src,
                requested,
                offered,
                route=(
                    "breakout_summaries() needs a native Analysis.from_definitions or "
                    "Analysis.from_unit_day_artifact instance with breakout evidence"
                ),
            )
        native = _require_analysis_operation(
            self._src,
            "breakout_summaries",
            NativeViewSource,
            message=(
                "breakout_summaries() needs a native Analysis.from_definitions or "
                "Analysis.from_unit_day_artifact instance with breakout evidence."
            ),
        )
        effective = select_metrics(
            cast("Sequence[Metric]", self._metrics),
            metrics,
            caller="breakout_summaries",
            require_declared_definitions=True,
        )
        breakouts = self._experiment.breakouts
        if not breakouts or not self._metrics:
            return {}
        reject_quantile_metrics(
            effective,
            "breakout_summaries()",
            reason="quantiles do not decompose over segment moments",
        )
        reject_retention_metrics(effective, "breakout_summaries", view="cohort")
        triggered = self._experiment.trigger is not None
        populations = ("assigned", "triggered") if triggered else ("assigned",)
        native.validate_populations(populations, operation="breakout_summaries")
        if triggered:
            import pyarrow as pa
        results: dict[str, dict[str, pa.Table]] = {}
        for population in populations:
            summaries = native.breakout_summaries(metrics=effective, population=population)
            for key, tables in summaries.items():
                if not triggered:
                    results[key] = tables
                    continue
                results[f"{key}:{population}"] = {
                    name: table.append_column(
                        "analysis_population", pa.array([population] * table.num_rows)
                    )
                    for name, table in tables.items()
                }
        return results

    def factor_summaries(self) -> dict[str, pa.Table]:
        """Per-(factor level x arm) moment tables for every declared factor.

        Same centered moment shape as :meth:`breakout_summaries`'s
        ``group_summary``, keyed off ``Experiment.factors``, without a
        day-level table - absorption consumes whole-window moments
        only. Tables carry CUPED/denominator columns populated only
        when applicable; a pre-exposure factor is resolved to its
        latest value strictly before each unit's first exposure.

        Returns
        -------
        dict[str, pa.Table]
            ``{f"{metric.name}:{factor.property}:{source_name}[:population]": pa.Table}``.
            Triggered experiments include both populations and an
            ``analysis_population`` column; empty when no factors are declared.
        """
        from increment.breakout.estimates import reject_quantile_metrics
        from increment.query.native_contract import NativeViewSource

        self._require_arm_state("factor_summaries")
        native = _require_analysis_operation(
            self._src,
            "factor_summaries",
            NativeViewSource,
            message=(
                "factor_summaries() needs a native Analysis.from_definitions or "
                "Analysis.from_unit_day_artifact instance with factor evidence."
            ),
        )
        if not self._experiment.factors or not self._metrics:
            return {}
        reject_quantile_metrics(
            self._metrics,
            "factor_summaries()",
            reason="quantiles do not decompose over factor moments",
        )
        triggered = self._experiment.trigger is not None
        populations = ("assigned", "triggered") if triggered else ("assigned",)
        native.validate_populations(populations, operation="factor_summaries")
        if triggered:
            import pyarrow as pa
        results: dict[str, pa.Table] = {}
        for population in populations:
            summaries = native.factor_summaries(metrics=self._metrics, population=population)
            for key, table in summaries.items():
                result_key = f"{key}:{population}" if triggered else key
                results[result_key] = (
                    table.append_column(
                        "analysis_population", pa.array([population] * table.num_rows)
                    )
                    if triggered
                    else table
                )
        return results

    def run_breakout(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] | None = None,
    ) -> BreakoutEstimates:
        """Per-segment lift estimates for every declared breakout.

        For each declared :class:`Breakout`, builds a moments source
        scoped to that breakout's totals-grain moments and hands it to
        :func:`increment.readouts.breakout`, which calls
        :func:`~increment.estimation.engine.estimate_lift` once per
        segment. Each breakout resolves its own :class:`FactSource`
        independently, so two breakouts sharing a ``property`` name
        but resolving to different sources are never conflated.

        Parameters
        ----------
        decision_method : Method | _Unset
            Explicit decision-method override; UNSET inherits the declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Explicit sensitivity methods; UNSET inherits the declaration.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets.
        metrics : Sequence[str | Metric] | None
            Narrow the estimated metrics. ``None`` selects every declared metric.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to break out in addition to the declared
            ones (``from_definitions`` only). Their rows equal those of the same metric
            declared in a plan of its own: the plan's breakout multiplicity applies to
            the added metrics' cells as one family of their own, so no declared row
            changes. ``metrics=[]`` with ``exploratory_metrics`` returns only the added rows.
        population : Literal["assigned", "triggered"] | None
            Select one source population; ``None`` returns captured populations
            as separate result rows. Triggered reads require declared trigger
            evidence.
        Notes
        -----
        Alpha, inference, and breakout multiplicity are read from the
        resolved plan attached to this analysis. Sensitivity analyses
        require constructing another source with another plan. Every
        declared breakout is always returned (no per-call dimension
        filter); an observational panel refuses this method by name.

        Returns
        -------
        BreakoutEstimates
            One row per (breakout x metric x method x non-control arm
            x segment); encouragement paths warn and skip control-free
            segments instead. ``source`` names the resolved
            :class:`FactSource`. Empty when no breakouts are declared.
        """
        execution_context = _current_execution_context()
        if execution_context is None:
            self._require_arm_state(
                "run_breakout", population=population if population is not None else "assigned"
            )
        rows = self._run_breakout(
            _execution_context=execution_context,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            population=population,
        )
        rows._default_backend = self._input_backend
        return rows

    def _run_breakout(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        correction: Correction | None = None,
        population: Literal["assigned", "triggered"] | None = None,
        _execution_context=None,
    ) -> BreakoutEstimates:
        from increment.breakout.estimates import BreakoutEstimates

        state = self._state
        experiment = (
            state.experiment
            if isinstance(state, DefinitionsArmAnalysisState)
            else getattr(state.source, "artifact_experiment", None)
            if state.family == "arm_moments"
            else None
        )
        if population == "triggered" and experiment is not None and experiment.trigger is None:
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run_breakout",
                experiment=experiment.name,
                trigger=None,
                route="native",
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        if experiment is not None and not experiment.breakouts:
            select_metrics(
                cast("Sequence[Metric]", self._src.context.metrics),
                metrics,
                caller="run_breakout",
            )
            if exploratory_metrics:
                self._exploratory_metrics(exploratory_metrics, caller="run_breakout")
            return BreakoutEstimates([])
        populations = (
            (population,)
            if population is not None
            else (
                ("assigned", "triggered")
                if experiment is not None and experiment.trigger is not None
                else ("assigned",)
            )
        )
        if "triggered" in populations:
            if isinstance(self._state, ContrastAnalysisState):
                _refuse(
                    _TRIGGER_UNSUPPORTED,
                    method="run_breakout",
                    experiment="<unknown>",
                    trigger=None,
                    route="switchback",
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
            reader = BreakoutReadouts(src=self._src, experiment=experiment)
            registration = getattr(self._plan.inference, "registration", None)
            if registration is None:

                def validate_candidate(candidate, selected_metrics):
                    if not selected_metrics:
                        return
                    breakout_policy = candidate._plan.view_policies.for_view(
                        "breakout",
                        mechanism=getattr(getattr(candidate, "_design", None), "mechanism", None),
                        segmented=True,
                    )
                    candidate_reader = BreakoutReadouts(
                        src=candidate._src,
                        experiment=getattr(candidate, "_experiment", None),
                    )
                    candidate_reader.validate_request(
                        BreakoutRequest(
                            metrics=tuple(selected_metrics),
                            decision_method=decision_method,
                            sensitivity_methods=sensitivity_methods,
                            prior=prior,
                            correction=correction
                            or normalize_display_correction(breakout_policy.correction),
                            q=(
                                breakout_policy.q
                                if breakout_policy.q is not None
                                else candidate._plan.q
                            ),
                            population="triggered",
                        )
                    )

                selected = select_metrics(
                    cast("Sequence[Metric]", self._src.context.metrics),
                    metrics,
                    caller="run_breakout",
                )
                validate_candidate(self, selected)
                added = self._exploratory_metrics(exploratory_metrics, caller="run_breakout")
                if added:
                    exploratory = self._exploratory_analysis(added, caller="run_breakout")
                    validate_candidate(exploratory, added)
            reader.validate_populations(populations)
        combined: BreakoutEstimates | None = None
        for population in populations:
            rows = self._run_breakout_population(
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                metrics=metrics,
                exploratory_metrics=exploratory_metrics,
                correction=correction,
                population=population,
                _execution_context=_execution_context,
            )
            combined = rows if combined is None else combined.concat(rows)
        return BreakoutEstimates([]) if combined is None else combined

    def _run_breakout_population(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        correction: Correction | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        _execution_context=None,
    ) -> BreakoutEstimates:
        """:meth:`run_breakout`, optionally under an explicit *correction* in place of the
        plan's breakout multiplicity (``"none"`` leaves every cell standing alone)."""
        from increment.breakout.estimates import BreakoutEstimates

        added = self._exploratory_metrics(exploratory_metrics, caller="run_breakout")
        if isinstance(self._state, ContrastAnalysisState):
            reject_role_overrides_under_contrast(
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )
        self._require_arm_state("run_breakout")
        inference = self._plan.inference
        if correction is not None and getattr(inference, "registration", None) is not None:
            _refuse(_UNCORRECTED_SEQUENTIAL, inference=type(inference).__name__)
        breakout_policy = self._plan.view_policies.for_view(
            "breakout",
            mechanism=getattr(getattr(self, "_design", None), "mechanism", None),
            segmented=True,
        )
        effective_correction = correction or normalize_display_correction(
            breakout_policy.correction
        )
        selected = select_metrics(
            cast("Sequence[Metric]", self._src.context.metrics), metrics, caller="run_breakout"
        )
        rows = (
            self._breakout_rows(
                selected,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                correction=correction,
                population=population,
                _execution_context=_execution_context,
            )
            if selected or not added
            else BreakoutEstimates([])
        )
        if added:
            exploratory = self._exploratory_analysis(added, caller="run_breakout")
            exploratory_rows = exploratory._breakout_rows(
                added,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                correction=correction,
                population=population,
                _execution_context=_execution_context,
                _scope_source=self._src,
                _scope_analysis=self,
            )
            rows = exploratory_rows if not rows else rows.concat(exploratory_rows)
        from increment.estimation.multiplicity import row_multiplicity_status

        return BreakoutEstimates(
            [
                row.model_copy(
                    update={
                        "multiplicity_status": row_multiplicity_status(
                            row,
                            correction=effective_correction if row.family_id is not None else None,
                        )
                    }
                )
                for row in rows
            ],
            metadata=rows.metadata,
        )

    def _breakout_rows(
        self,
        selected: Sequence[Metric],
        *,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        correction: Correction | None,
        population: Literal["assigned", "triggered"] = "assigned",
        _execution_context=None,
        _scope_source=None,
        _scope_analysis=None,
    ) -> BreakoutEstimates:
        policy = self._plan.view_policies.for_view(
            "breakout",
            mechanism=getattr(getattr(self, "_design", None), "mechanism", None),
            segmented=True,
        )
        request = BreakoutRequest(
            metrics=tuple(selected),
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            correction=correction or normalize_display_correction(policy.correction),
            q=policy.q if policy.q is not None else self._plan.q,
            population=population,
        )
        experiment = getattr(self, "_experiment", None)
        source = getattr(self._src, "summary_breakout_source", None)
        reader_source = source if source is not None else self._src
        reader = BreakoutReadouts(
            src=reader_source,
            experiment=experiment,
            execution_context=_execution_context,
            scope_source=self._src if _scope_source is None else _scope_source,
            analysis=self if _scope_analysis is None else _scope_analysis,
        )
        return reader.run(request)

    def _refuse_clustered_day_axis(self, method: str) -> None:
        """A declared cluster is total-grain only; see `_CLUSTERED_DAY_AXIS`."""
        if self._experiment.cluster is not None:
            _refuse(
                _CLUSTERED_DAY_AXIS,
                method=method,
                experiment=self._experiment.name,
                cluster=self._experiment.cluster,
            )

    def run_daily(
        self,
        *,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyMetricValues:
        """Per-day absolute metric values, optionally broken out by segment.

        The time-series counterpart to :meth:`run`: reduces the same
        unit-day panel with ``daily_group_summary`` instead of
        ``group_summary``. This value-only path does not request pre-period
        covariates. Direct CUPED callers should use
        :meth:`run_daily_lift` with role-aware method overrides; that lift path uses a
        fixed pre-exposure unit covariate ``x`` with daily (or
        retention-cohort) outcome ``y`` when native pre-period moments are
        requested.

        Passing *dimension* switches to the per-day-per-segment view,
        routed through ``breakout_moments()`` for each matching
        ``(breakout, metric)`` pair.

        Parameters
        ----------
        metrics : Sequence[str | Metric] | None
            Override the declared metrics + guardrails. A bounded-band
            :class:`RetentionMetric` is accepted; an
            unbounded one always raises (see ``Notes``).
        dimension : str | None
            Break each day out by one declared breakout's dimension.
            Must match a declared breakout ``property``.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); ``metrics=[]`` reads only these. A
            dimensioned read keeps its refusal of metrics the experiment does not declare.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.

        Returns
        -------
        DailyMetricValues
            One row per (metric x day x arm), or per (metric x day x
            arm x segment) when *dimension* is given. A slice too small
            to estimate (fewer than 2 units, or a non-positive mean)
            remains present with ``value=None`` and an ``unavailable``
            reason instead of being dropped.

        Notes
        -----
        The readout raises :class:`ValueError` for an unbounded retention
        band, an unmatched dimension, or a dimensioned request for an
        undeclared metric. Use :meth:`run_asof` or :meth:`run_asof_lift`
        for unbounded retention monitoring.
        """

        _execution_context = _current_execution_context()
        added = self._exploratory_metrics(exploratory_metrics, caller="run_daily")
        self._require_arm_state("run_daily")
        selected = self._select_day_axis_metrics(metrics, caller="run_daily", added=added)
        req = DayAxisRequest(
            caller="run_daily",
            grain="daily",
            metrics=tuple(selected),
            dimension=dimension,
            population=population,
        )
        rows = self._run_day_axis_values(req, _execution_context, coherence=coherence)
        rows._default_backend = self._input_backend
        rows._envelope_view = "daily"
        return rows

    def _daily_lift(self, **kwargs: Any) -> DailyLiftEstimates:
        """Internal day-axis boundary: always supply an effective compiled plan.

        Call-time Metric objects absent from the analysis plan receive locally
        compiled unassigned procedures at the plan's alpha, matching the method/prior
        defaults the facade already resolves for them.
        """
        from increment.breakout.estimates import run_daily_lift as _run_daily_lift_estimates

        plan = with_unassigned_procedures(
            self._plan,
            cast("Sequence[Metric]", kwargs["metrics"]),
            design=getattr(self, "_design", None),
        )
        return _run_daily_lift_estimates(plan=plan, **kwargs)

    def _select_day_axis_metrics(
        self,
        metrics: Sequence[str | Metric] | None,
        *,
        caller: str,
        added: Sequence[Metric] = (),
    ) -> list[Metric]:
        # Only the native route accepts undeclared call-time Metric objects.
        native = day_axis_route(self._src) == NATIVE_ROUTE
        selected = select_metrics(
            self._metrics, metrics, caller=caller, allow_undeclared_objects=native
        )
        if not added:
            return selected
        # Added metrics follow the declared ones; a shared name is a duplicate.
        return select_metrics((), [*selected, *added], caller=caller, allow_undeclared_objects=True)

    @staticmethod
    def _stamp_exploratory(
        rows: DailyLiftEstimates,
        added: Sequence[Metric],
        *,
        correction: Correction | None = None,
    ) -> DailyLiftEstimates:
        """Mark added metrics exploratory under the effective view correction."""
        from increment.breakout.estimates import DailyLiftEstimates

        names = {metric.name for metric in added}
        if not names:
            return rows
        from increment.estimation.multiplicity import stamp_multiplicity_status

        return DailyLiftEstimates(
            stamp_multiplicity_status(
                [
                    row.model_copy(update={"role": "exploratory"}) if row.metric in names else row
                    for row in rows
                ],
                correction=correction,
            ),
            metadata=rows.metadata,
        )

    @staticmethod
    def _scoped_execution_context(context: Any):
        return _execution_context_scope(context)

    def _validate_day_axis_before_context(self, req: DayAxisRequest, execution_context=None):
        if req.population == "triggered" or (
            execution_context is None
            and (req.dimension is not None or not is_custom_source(self._src))
        ):
            from increment._day_axis import validate_day_axis

            validate_day_axis(
                req,
                src=self._src,
                design=self._design,
                plan=self._plan,
                route=day_axis_route(self._src),
                experiment=getattr(self, "_experiment", None),
            )

    def _run_day_axis_values(
        self,
        req: DayAxisRequest,
        execution_context=None,
        *,
        coherence: Literal["captured", "live"] = "captured",
    ):
        self._validate_day_axis_before_context(req, execution_context)
        from increment.readouts._line2 import run_standalone_day_axis_values

        return run_standalone_day_axis_values(self, req, execution_context, coherence=coherence)

    def _day_axis(self, execution_context=None) -> DayAxisReadouts:
        return DayAxisReadouts(
            src=self._src,
            plan=self._plan,
            design=self._design,
            experiment=getattr(self, "_experiment", None),
            daily_lift=self._daily_lift,
            execution_context=execution_context,
        )

    def _preflight_line4_views(
        self,
        *,
        population: Literal["assigned", "triggered"],
        readouts: Sequence[Literal["run", "daily", "daily_lift", "asof", "asof_lift"]],
    ) -> None:
        """Validate line-four day-axis requests before its scope opens."""
        caller_by_name: dict[
            str, Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"]
        ] = {
            "daily": "run_daily",
            "daily_lift": "run_daily_lift",
            "asof": "run_asof",
            "asof_lift": "run_asof_lift",
        }
        day_axis = self._day_axis()
        experiment = getattr(self._state, "experiment", None)
        for name in readouts:
            if name == "run":
                continue
            grain = "daily" if name.startswith("daily") else "asof"
            request = DayAxisRequest(
                caller=caller_by_name[name],
                grain=grain,
                metrics=tuple(self._metrics),
                dimension=None,
                estimands=None,
                population=population,
                experiment=experiment,
            )
            day_axis.validate_request(request, lift=name.endswith("_lift"))

    def run_daily_lift(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        estimands: Sequence[str] | None = None,
        dimension: str | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyLiftEstimates:
        """Per-day relative lift estimates, optionally broken out by segment.

        The time-series counterpart to :meth:`run`: builds each metric's
        per-day moments like :meth:`run_daily`, concatenates them into
        one :func:`increment.breakout.estimates.run_daily_lift` call.
        *dimension* routes through ``breakout_moments()`` per breakout.

        The outcome ``y`` is daily, or retention-cohort, while CUPED's ``x``
        is a fixed pre-exposure unit covariate. Native CUPED is requested
        only when pre-period moments are materialized. Sequential inference
        is refused for disjoint daily/cohort slices; use
        :meth:`run_asof_lift` for cumulative monitoring.

        Parameters
        ----------
        decision_method : Method | _Unset
            Explicit decision-method override; UNSET inherits the declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Explicit sensitivity methods; UNSET inherits the declaration.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets.
        metrics : Sequence[str | Metric] | None
            Override the declared metrics + guardrails. A bounded-band
            :class:`RetentionMetric` is accepted; an
            unbounded one always raises (see ``Notes``).
        estimands : Sequence[str] | None
            Requested encouragement estimands. Per-day LATE is refused because
            each daily first stage is weakly identified; ``("compliance",)``
            is supported as an assignment-anchored uptake estimate.
        dimension : str | None
            Break each day's lift out by one declared breakout's
            dimension. Must match a declared breakout ``property``.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` selects a unit on its first eligible trigger day;
            outcomes anchor at trigger, while compliance uptake stays anchored
            at assignment. Both require explicit source evidence and are
            supported on definitions and unit-day artifacts.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.

        Returns
        -------
        DailyLiftEstimates
            One row per (day x metric x method x non-control arm), or
            with a segment axis when *dimension* is given. An
            unestimable slice remains present with ``lift=None`` and an
            ``unavailable`` reason instead of being dropped.

        Notes
        -----
        The readout raises :class:`ValueError` for an unbounded retention
        band, an unmatched dimension, a dimensioned request for an undeclared
        metric, or per-day LATE: each daily first stage is weakly identified.
        It raises :class:`UnsupportedRequestError` for sequential inference
        on disjoint daily or cohort slices, which cannot support monitoring.
        """

        _execution_context = _current_execution_context()
        self._require_arm_state("run_daily_lift", population=population)
        selected = self._select_day_axis_metrics(metrics, caller="run_daily_lift")
        req = DayAxisRequest(
            caller="run_daily_lift",
            grain="daily",
            metrics=tuple(selected),
            dimension=dimension,
            estimands=tuple(estimands) if estimands is not None else None,
            population=population,
        )
        self._validate_day_axis_before_context(req, _execution_context)
        lift_options = None
        if _execution_context is not None:
            lift_options = self._day_axis(_execution_context).resolve_lift_options(
                req.metrics,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )
        if _execution_context is None:
            from increment.readouts._line2 import run_standalone_day_axis_lift

            rows = run_standalone_day_axis_lift(
                self,
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                coherence=coherence,
            )
        else:
            rows = self._day_axis(_execution_context).lift(
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                options=lift_options,
            )
        rows._default_backend = self._input_backend
        rows._envelope_view = "daily"
        return rows

    def run_asof(
        self,
        *,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyMetricValues:
        """Per-day absolute metric values "as of day N".


        The running-total counterpart to :meth:`run_daily`: reduces the
        same panel with ``asof_group_summary`` instead of
        ``daily_group_summary``, and deliberately skips
        ``window_bound_stats`` - the as-of builder does its own per-unit
        window masking instead of dropping a closed-window row.

        Passing *dimension* switches to the per-day-per-segment as-of
        view, routed through ``breakout_moments(grain="asof")``.

        Parameters
        ----------
        metrics : Sequence[str | Metric] | None
            Override the declared/constructed metric set.
        dimension : str | None
            Break each day's running total out by one declared dimension.
        completed_windows_only : bool
            ``False`` (default) is the provisional monitoring series: a
            unit is admitted as soon as it is post-exposure, zero-filled
            until it has a real observation; a retention metric is
            admitted at band open. ``True`` gates on full maturity
            instead: a bounded mean/conversion/ratio metric is admitted
            only once ``ds >= first_exposure_date + window_days``; under
            :class:`Encouragement`, both the outcome and uptake windows
            must be bounded and the gate is the LATER of the two right
            edges (raises otherwise); a retention metric is admitted at
            band close rather than band open. An unbounded (no
            ``window_days``) mean, conversion, or ratio metric with no
            uptake is unaffected, since it has no right edge to wait
            out; combined with a declared uptake it raises instead
            (uptake makes both windows required). Also raises for an
            unbounded retention band, which has no closing day to gate
            on either.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); ``metrics=[]`` reads only these. A
            dimensioned read keeps its refusal of metrics the experiment does not declare.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.

        Returns
        -------
        DailyMetricValues
            One row per (metric x day x arm), or with a segment axis
            when *dimension* is given. An unestimable slice remains
            present with ``value=None`` and an ``unavailable`` reason.
            A unit's contribution freezes at its last in-window value, so
            this view's final day and :meth:`run`'s whole-window estimate can legitimately
            disagree for an experiment with late enrollees.

        Notes
        -----
        The readout raises :class:`ValueError` for an unmatched dimension,
        or completed-window retention with an unbounded band.
        """

        _execution_context = _current_execution_context()
        added = self._exploratory_metrics(exploratory_metrics, caller="run_asof")
        self._require_arm_state("run_asof")
        selected = self._select_day_axis_metrics(metrics, caller="run_asof", added=added)
        req = DayAxisRequest(
            caller="run_asof",
            grain="asof",
            metrics=tuple(selected),
            dimension=dimension,
            completed_windows_only=completed_windows_only,
            population=population,
        )
        rows = self._run_day_axis_values(req, _execution_context, coherence=coherence)
        rows._default_backend = self._input_backend
        rows._envelope_completion_policy = completed_windows_only
        rows._envelope_view = "asof"
        return rows

    def run_asof_lift(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        estimands: Sequence[str] | None = None,
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyLiftEstimates:
        """As-of relative lift history, or a registered sequential checkpoint.

        For fixed-horizon inference, the running-total counterpart to
        :meth:`run_daily_lift`: as-of
        moments (:meth:`run_asof`) concatenated into one
        :func:`increment.breakout.estimates.run_daily_lift` call.

        Registered sequential inference returns only the current labeled,
        finalized checkpoint, not a reconstructed per-day history. Dimensions
        are unsupported here; use the registered breakout checkpoint instead.

        Parameters
        ----------
        decision_method : Method | _Unset
            Explicit decision-method override; UNSET inherits the declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Explicit sensitivity methods; UNSET inherits the declaration.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets. Native day-axis
            sources support CUPED for mean, conversion and ratio metrics when their
            fixed pre-period covariate moments are materialized (a ratio metric
            adjusts its numerator and denominator against the numerator's
            pre-period total), and for bounded retention metrics on
            cohort-indexed daily paths.
            As-of retention CUPED is accepted on the calendar axis,
            including dimensioned monitoring and bounded completed-window
            reads; unbounded bands remain monitoring-only. CUPED remains
            refused for quantile metrics, clustered day-axis
            analyses, and dataframe-panel sources.
        metrics : Sequence[str | Metric] | None
            Override the declared/constructed metric set.
        dimension : str | None
            Break each day's fixed-horizon lift out by one declared dimension.
            Unsupported for registered sequential inference.
        estimands : Sequence[str] | None
            Under :class:`Encouragement` only: ``"itt"``/``"compliance"``/``"late"``, default all three.
            A registered sequential checkpoint requested with exactly
            ``("compliance",)`` reads uptake alone and consumes no outcome metric,
            so a retention metric in the catalog does not affect it (see *Notes*).
        completed_windows_only : bool
            For fixed-horizon inference, ``False`` (default) is the provisional gate:
            a unit is admitted as soon as it is post-exposure, zero-filled
            until it has a real observation. ``True`` gates on full
            maturity: a bounded mean/conversion/ratio metric with no
            declared uptake is admitted once its own outcome window has
            closed; under :class:`Encouragement`, both the outcome and
            uptake windows must be bounded and the gate is the LATER of
            the two right edges (raises otherwise). A retention metric
            gates at band close rather than band open. An unbounded
            mean, conversion, or ratio metric with no uptake is
            unaffected; combined with a declared uptake it raises
            instead. Registered inference uses finalized outcomes; ``True``
            remains required under :class:`Encouragement` with
            ``AlwaysValid(registration=...)``.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); their rows carry ``role="exploratory"``
            and ``metrics=[]`` reads only these. A registered sequential plan refuses them.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.
        Notes
        -----
        Alpha, inference, and segmented as-of multiplicity come from
        the resolved plan attached to this analysis. Segmented as-of BH is
        refused, as are observational designs and unsupported
        retention/window combinations.

        The readout raises :class:`ValueError` for completed-window unbounded
        retention, an unmatched dimension, or non-ITT estimands outside an
        encouragement design. It raises :class:`CapabilityError` for retention
        outcomes under encouragement or dimensioned registered inference.
        A registered compliance-only checkpoint consumes no outcome metric,
        so neither retention refusal applies. Segmented as-of BH and
        observational designs raise :class:`UnsupportedRequestError`.

        Returns
        -------
        DailyLiftEstimates
            Fixed-horizon: one row per (day x metric x method x non-control arm),
            optionally per segment. Unestimable slices retain ``lift=None`` and
            an ``unavailable`` reason. Registered sequential inference returns
            only the current checkpoint rows, all with its date label.
        """

        _execution_context = _current_execution_context()
        added = self._exploratory_metrics(exploratory_metrics, caller="run_asof_lift")
        self._require_arm_state("run_asof_lift", population=population)
        selected = self._select_day_axis_metrics(metrics, caller="run_asof_lift", added=added)
        req = DayAxisRequest(
            caller="run_asof_lift",
            grain="asof",
            metrics=tuple(selected),
            dimension=dimension,
            completed_windows_only=completed_windows_only,
            estimands=tuple(estimands) if estimands is not None else None,
            population=population,
        )
        self._validate_day_axis_before_context(req, _execution_context)
        lift_options = None
        if _execution_context is not None:
            lift_options = self._day_axis(_execution_context).resolve_lift_options(
                req.metrics,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )
        asof_policy = self._plan.view_policies.for_view(
            "asof",
            mechanism=getattr(self._design, "mechanism", None),
            segmented=dimension is not None,
        )
        if _execution_context is None:
            from increment.readouts._line2 import run_standalone_day_axis_lift

            lift_rows = run_standalone_day_axis_lift(
                self,
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                coherence=coherence,
            )
        else:
            lift_rows = self._day_axis(_execution_context).lift(
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                options=lift_options,
            )
        rows = self._stamp_exploratory(
            lift_rows,
            added,
            correction=normalize_display_correction(asof_policy.correction),
        )
        rows._default_backend = self._input_backend
        rows._envelope_completion_policy = completed_windows_only
        rows._envelope_view = "asof"
        return rows
