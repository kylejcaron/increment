"""Analysis: the public facade for A/B test analysis.

A thin wrapper over ``increment.query``: builds the query, reduces
moments, and dispatches every readout through the shared
``MomentSource`` protocol regardless of which constructor built it.
Pure statistics; never imports a visualisation library.

Usage
-----
    con = ibis.duckdb.connect(...)
    analysis = Analysis.from_definitions("my_experiment", "definitions/", con)
    results = analysis.run()
"""

from __future__ import annotations

import copy
import datetime as dt
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ibis.backends.sql import SQLBackend

from increment import readouts
from increment._analysis_config import (
    UNSET,
    _Unset,
    effective_methods,
    normalize_display_correction,
    select_exploratory_metrics,
    select_metrics,
)
from increment._breakout_readouts import BreakoutReadouts, BreakoutRequest
from increment._day_axis import (
    _CLUSTERED_DAY_AXIS,
    _TRIGGER_UNSUPPORTED,
    DayAxisReadouts,
    DayAxisRequest,
    _day_axis_source_route,
)
from increment._evidence_dispatch import ContrastHandler, ContrastReadoutRequest
from increment._literals import Correction, ValueScale
from increment._sitewide import SitewideReadouts, SitewideRequest
from increment._source_operations import (
    AllocationHistoryOperation,
    DashboardBreakoutReads,
    DashboardGroupData,
    DashboardGroupDataOperation,
    DashboardSnapshotHandler,
    DashboardSnapshotPayload,
    ExploratorySourceOperation,
    ExportMomentsOperation,
    MaterializeOperation,
    ReadoutSnapshotOperation,
    SummarySqlOperation,
    TriggeredCountsOperation,
    TriggeredPopulationOperation,
)
from increment._source_types import classify_source
from increment._study import ParallelStudyEnvelope
from increment._whole_window import (
    _ANALYSIS_OPERATION,
    WholeWindowReadouts,
    WholeWindowRequest,
    _require_analysis_operation,
    reject_role_overrides_under_contrast,
)
from increment.breakout.estimates import (
    BreakoutEstimates,
    DailyLiftEstimates,
    DailyMetricValues,
    LiftEstimates,
    reject_quantile_metrics,
    reject_retention_metrics,
)
from increment.breakout.estimates import run_daily_lift as _run_daily_lift_estimates
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
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.errors import refuse as _refuse
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.engine import Method
from increment.plan import (
    bind_automatic_sequential_plan,
    compile_decision_plan,
    with_unassigned_procedures,
)
from increment.query.artifact_contract import (
    ArtifactContractError,
    ArtifactStore,
    unit_day_artifact_extension_catalog,
)
from increment.query.artifact_digest import canonical_json
from increment.query.fact_resolution import _require_design
from increment.query.integrity import validate_trigger_fires_in_every_arm
from increment.query.native_contract import NativeCoreSource, NativeViewSource
from increment.query.native_source import DefinitionsMomentSource
from increment.query.session import SourceSnapshotEvidence, WarehouseSession
from increment.query.source import open_artifact
from increment.semantics.artifact import (
    ArtifactContext,
    ArtifactExtensionRequest,
    UnitDayArtifactRef,
)
from increment.semantics.loader import load, verify_sql_admission_matches_execution
from increment.semantics.models import Breakout, Definitions, Experiment
from increment.semantics.unit_cycle import UnitCycleReference
from increment.sequential_source import native_observation_mapping
from increment.sources import (
    MomentSource,
    SourceContext,
    SourceOperation,
)

__all__ = ["Analysis", "fit_predeclared_adjustment"]

_CONTRAST_UNAVAILABLE = RefusalSpec(
    "facade.analysis.contrast_unavailable",
    CapabilityError,
    template="{method}() is unavailable for switchback contrast evidence.",
)
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
_TRIGGERED_SITEWIDE_UNSUPPORTED = RefusalSpec(
    "analysis.sitewide.triggered_population_unsupported",
    CapabilityError,
    template=(
        "sitewide() is an all-units deployment estimand and cannot isolate trigger-eligible "
        "units from the whole-site evidence for {experiment!r}; use run() for the triggered "
        "cohort effect."
    ),
)


def fit_predeclared_adjustment(
    frame,
    *,
    unit: str,
    group: str,
    control: str,
    outcome: str,
    covariate: str,
    covariate_missing: Literal["impute", "zero", "error"] = "impute",
):
    """Fit a sequential CUPED coefficient and covariate centre from pre-period data.

    *frame* is one row per assigned unit observed BEFORE the experiment reads
    any outcome: *outcome* is the metric measured in a pre-period window and
    *covariate* its covariate measured relative to that window, the same way
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
    from increment.estimation.engine import _df_to_arms
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.sequential import PredeclaredAdjustment

    source = from_unit_summary(
        frame,
        unit=unit,
        group=group,
        control=control,
        metrics=[
            MetricSpec(name=outcome, covariate=covariate, covariate_missing=covariate_missing)
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
        "analysis.sitewide.triggered_population_unsupported": _TRIGGERED_SITEWIDE_UNSUPPORTED,
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


class _LegacyOperationSource:
    """Compatibility view for sources predating operation declarations."""

    operations: frozenset[SourceOperation] = frozenset()

    def __init__(self, src: object) -> None:
        self._src = src

    def __getattr__(self, name: str) -> Any:
        return getattr(self._src, name)


@dataclass(frozen=True, slots=True)
class _ArtifactOpenSpec:
    """Immutable reopen pin for an artifact-backed analysis, fixed at construction."""

    store: ArtifactStore
    ref: UnitDayArtifactRef
    expected_context: ArtifactContext
    verification: Literal["lazy_digest"]


# Public facade


def _stamp_exploratory_rows(rows):
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
        return LiftEstimates(updated, source=rows.source)

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
    )
    return LiftEstimates(
        updated,
        metadata=metadata,
        source=rows.source,
        sequential_snapshot=rows.sequential_snapshot,
    )


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
    store : {"auto", "always", "none"}
        ``"auto"`` (default) materializes from the second top-level reduction;
        ``"always"`` starts with the first. Each qualifying call rescans current
        warehouse data; nested reductions reuse that call's materialization.
        ``"none"`` keeps reductions unmaterialized.
    on_mixed_assignment : {"error", "warn", "exclude"}
        Policy for invalid warehouse assignments: units observed in more
        than one arm or with no assignment label. ``"error"`` (default)
        refuses; ``"warn"``/``"exclude"`` drop them, warning once or not at
        all.
    source_snapshot_evidence : SourceSnapshotEvidence | None
        Caller-supplied event-time cutoff and optional certified per-feed
        completeness watermarks; absent certification remains unknown.
    """

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
        src: MomentSource | ContrastSource,
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
                source=cast("MomentSource", src),
                study=study,
                experiment_name=experiment_name,
            )
        if study is None:
            _raise("facade.analysis.definitions_state_requires_identification")
        return DefinitionsArmAnalysisState(
            source=cast("MomentSource", src),
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

    def __enter__(self) -> Analysis:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release this analysis's resources without closing caller connections."""
        cast("MomentSource", self._state.source).close()

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
                        else classify_source(self._src)
                    ),
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
            _refuse(_CONTRAST_UNAVAILABLE, method=method)
        return self._state

    def _ensure_artifact_source(self) -> MomentSource:
        source = cast("MomentSource", self._state.source)
        spec = self._artifact_open_spec
        if spec is not None and cast("ArtifactMomentSource", source).closed:
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
        state = self._require_arm_state("publish_unit_day_artifact")
        if not isinstance(state, DefinitionsArmAnalysisState):
            _refuse(
                _ANALYSIS_OPERATION,
                message="publish_unit_day_artifact() requires Analysis.from_definitions",
                operation="materialize",
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
    def from_unit_day_artifact(
        cls,
        store: ArtifactStore,
        ref: UnitDayArtifactRef,
        *,
        expected_context: ArtifactContext,
        verification: Literal["lazy_digest"] = "lazy_digest",
    ) -> Analysis:
        """Open one caller-trusted immutable artifact generation."""
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
    def from_definitions(
        cls,
        experiment_name: str,
        definitions_path: str | Path,
        con: SQLBackend,
        *,
        backend: str | None = None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
        source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    ) -> Analysis:
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
        src: MomentSource | ContrastSource,
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
    ) -> Analysis:
        """Build an instance from the source's immutable construction state."""
        context = src.context
        if isinstance(context, ContrastContext):
            if design is not None and context.study.identification != design:
                _raise("facade.analysis.source_context_design_disagrees_arm")
        else:
            try:
                inspect.getattr_static(src, "operations")
            except AttributeError:
                src = cast("MomentSource", _LegacyOperationSource(src))
            if design is not None and cast("SourceContext", src.context).design != design:
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
    def from_unit_summary(  # noqa: PLR0913
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        group: str,
        control: str | None = None,
        metrics: MetricsArg,
        experiment_id: str = "frame",
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
        uptake: str | None = None,
        on_unassigned: Literal["error", "exclude"] = "error",
        cluster: str | None = None,
        intervention_grain: Literal["unit", "cluster"] = "unit",
        exposure_date: str | None = None,
    ) -> Analysis:
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
        """
        _require_design(control, design, constructor="Analysis.from_unit_summary")
        from increment.frame import from_unit_summary as _make_source
        from increment.semantics.design import Randomized

        design = design or Randomized(control_group=cast("str", control))
        src = _make_source(
            frame,
            unit=unit,
            group=group,
            control=design.control_group,
            metrics=metrics,
            experiment_id=experiment_id,
            uptake=uptake,
            design=design,
            plan=plan,
            on_unassigned=on_unassigned,
            cluster=cluster,
            intervention_grain=intervention_grain,
            exposure_date=exposure_date,
        )
        return cls._from_source(src, design)

    @classmethod
    # Public factory signature is the API for panel-backed analysis.
    def from_unit_panel(  # noqa: PLR0913
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        group: str,
        date: str,
        control: str | None = None,
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
    ) -> Analysis:
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
        _require_design(control, design, constructor="Analysis.from_unit_panel")
        from increment.frame import from_unit_panel as _make_source
        from increment.semantics.design import Randomized

        design = design or Randomized(control_group=cast("str", control))
        src = _make_source(
            frame,
            unit=unit,
            group=group,
            date=date,
            control=design.control_group,
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
        return cls._from_source(src, design)

    @classmethod
    def from_moments(
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        metrics: MetricsArg,
        control: str | None = None,
        experiment_id: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
    ) -> Analysis:
        """Build an :class:`Analysis` that rehydrates a moments cube
        (e.g. exported via :meth:`export`, or a warehouse table of
        pre-aggregated moments).

        Enters at moment grain: ``run()``/``srm()`` work; the unit-frame
        method (needed for IPTW/DML) raises :class:`CapabilityError`, as do
        ``sql()`` and every method the other seam constructors already refuse
        (day axis, ``panel_sql``, ``run_breakout``, ``materialize``).

        *rows* must carry centered ``group_summary`` columns, a fixed-horizon
        ``moments_format=11`` stamp, integer ``n``, nullable integer
        ``successes``, a source identity record, and a complete embedded
        ``decision_plan`` on every fixed-horizon row. The preceding fixed
        format 10 remains readable. Declared binary outcomes require exact
        success counts. Sequential exports use ``moments_format=10``
        checkpoint envelopes and a version-3 sequential wire plan; the
        preceding sequential format 9 remains readable.
        Encouragement exports additionally carry cluster identity, member
        counts, and full bivariate uptake/size moments on every row. Supply
        the same Encouragement design when reloading; an omitted experiment_id
        is recovered from the cube for this design. An explicit identity must
        match the exported cohort. A
        cube without that state cannot provide design-level compliance.
        An empty-metric fixed-horizon Encouragement export carries one
        complete format-11 ``design_summary`` envelope. Reload it with
        ``metrics=[]``; both its embedded and effective plans must remain
        fixed-horizon and metric-free, even when *plan* is supplied.

        Exactly one of *control*/*design* must be given - see
        :meth:`from_unit_summary` for the contract.

        *plan* explicitly overrides the embedded compiled plan when supplied.
        Without an override, the payload's complete roles, methods,
        priors, inference, and view policies are used as exported.
        """
        _require_design(control, design, constructor="Analysis.from_moments")
        from increment._frame_validation import _coerce_source_metrics
        from increment.frame import synthesise_metric
        from increment.semantics.design import Encouragement, Randomized
        from increment.sources import MomentsSource

        design = design or Randomized(control_group=cast("str", control))
        specs = _coerce_source_metrics(metrics, design=design)
        metric_catalog = [synthesise_metric(spec) for spec in specs]
        from increment._analysis_config import resolve_configs

        configs = resolve_configs(
            metric_catalog,
            bindings=None,
            specs={spec.name: spec for spec in specs},
            methods=None,
            prior=None,
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
    def from_switchback_panel(
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
    ) -> Analysis:
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
        return cls._from_source(cast("ContrastSource", src), identification)

    def materialize(self) -> Analysis:
        """Rebuild spine and per-metric stats from current warehouse data.

        Each call performs a fresh scan, overriding ``store="auto"``'s
        second-reduction threshold. A no-op only for ``store="none"`` or
        an experiment declaring no metrics.

        Raises
        ------
        CapabilityError
            On a seam-family instance: nothing to materialize.
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
        self, operation: DashboardSnapshotHandler, *, metrics: Sequence[Metric]
    ) -> DashboardSnapshotPayload:
        """Prepare one complete dashboard payload against an isolated, pinned source.

        *metrics* names every metric the payload reads, so the pin covers exactly their
        fact sources. Pass the declared metrics together with any
        :attr:`available_metrics` the operation will add through ``exploratory_metrics``:
        an added metric outside the pin is refused rather than read from a later state.
        """
        self._require_arm_state("dashboard_snapshot")
        src = _require_analysis_operation(
            self._src,
            "readout_snapshot",
            ReadoutSnapshotOperation,
            message=(
                "dashboard_snapshot() needs a native Analysis.from_definitions instance "
                "with a warehouse-backed source."
            ),
        )
        with src.readout_snapshot(
            metrics=metrics, population="assigned", include_breakouts=True
        ) as pinned:
            isolated = copy.copy(self)
            isolated._state = replace(self._state, source=pinned)
            # The pinned snapshot must never be reopened over: drop any reopen pin
            # the copy would otherwise inherit.
            isolated._artifact_open_spec = None
            return operation(isolated)

    def dashboard_breakout_reads(self, breakout: Breakout) -> DashboardBreakoutReads:
        """The dashboard's Explore reads through one declared breakout alone.

        A dimension-wide read covers every declared breakout of that property, so one
        source's refusal would refuse its siblings. These reads hand the source only
        *breakout*; families are computed per breakout, so their rows equal this breakout's
        rows from the dimension-wide reads. Only native ``Analysis.from_definitions``
        instances support it: other arm-family sources refuse with
        ``facade.analysis.operation`` and switchback evidence with
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
            _refuse(
                _ANALYSIS_OPERATION,
                message=(
                    "dashboard_breakout_reads() needs a native Analysis.from_definitions instance."
                ),
                operation="dashboard_breakout_reads",
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

    def srm(
        self,
        *,
        expected: dict[str, float] | None = None,
        alpha: float = 0.001,
        inference: Literal["always_valid", "fixed"] = "always_valid",
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> SRMResult | NotApplicable:
        """Sample-ratio-mismatch check on the enrolled per-group counts.

        The default is an anytime-valid check for cumulative prefixes
        under a known allocation with constant conditional arm
        probabilities; pass ``inference="fixed"`` for one predeclared
        Pearson look. On a clustered experiment, the chi-square runs
        over distinct cluster counts per arm.

        Parameters
        ----------
        expected : dict[str, float] | None
            Expected allocation proportions per group. Required (or a
            declared ``allocation``) for ``inference="always_valid"``.
        alpha : float
            Lifetime significance threshold (default 0.001).
        inference : {"always_valid", "fixed"}
            Evidence controlling ``is_srm``. Default ``"always_valid"``.
        population : {"assigned", "triggered"}
            ``"assigned"`` (default) checks enrolled counts.
            ``"triggered"`` narrows to the declared trigger population
            first; raises ``CapabilityError`` on a seam-family instance.

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
        if population == "triggered":
            src = _require_analysis_operation(
                src,
                "triggered_counts",
                TriggeredCountsOperation,
                message=(
                    "srm(population='triggered') needs a native "
                    "Analysis.from_definitions instance -- a frame/warehouse/"
                    "moments-backed analysis retains no raw event stream to "
                    "narrow to a triggered population."
                ),
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
            grain, counts, unit_counts = src.triggered_counts()
            counts = complete_srm_support(counts, expected=expected)
            if unit_counts:
                unit_counts = complete_srm_support(unit_counts, expected=expected)
            return sample_ratio_mismatch(
                counts,
                expected=expected,
                alpha=alpha,
                inference=inference,
                grain=grain,
                unit_counts=unit_counts,
            )
        return readouts.srm(
            src,
            expected=expected,
            alpha=alpha,
            inference=inference,
        )

    def trigger_rates(self) -> dict[str, float]:
        """Observed triggered share per arm.

        Feed a pooled estimate (e.g. the mean of these per-arm rates) into
        ``Baseline(..., trigger_rate=...)``, then ``required_sample_size(...)``,
        to plan a follow-up at the right size.
        """
        source = self._src
        if classify_source(source) == "artifact":
            if "triggered_counts" not in source.operations:
                _refuse(
                    _ANALYSIS_OPERATION,
                    message="trigger_rates() requires complete trigger evidence",
                    operation="triggered_counts",
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

    def _validate_trigger_fires_in_every_arm(self) -> None:
        validate_trigger_fires_in_every_arm(self.trigger_rates(), self._experiment.trigger)

    def _cate_source(self, method: str) -> MomentSource:
        self._require_arm_state(method)
        experiment = getattr(self, "_experiment", None)
        if experiment is None or experiment.trigger is None:
            return cast("MomentSource", self._src)
        if classify_source(self._src) == "artifact":
            return self._readout_source(population="triggered")
        triggered = _require_analysis_operation(
            self._src,
            "triggered_source",
            TriggeredPopulationOperation,
            message=(
                f"{method}() needs a source-owned triggered unit view; this source "
                "cannot reconstruct trigger membership."
            ),
        )
        return triggered.triggered_source()

    def sitewide(
        self, metric_name: str, *, arm: str | None = None, alpha: float = 0.05
    ) -> SitewideImpact | SitewideRatioImpact:
        """Whole-site impact of shipping *metric_name*'s lift to every unit.

        Native (:meth:`from_definitions`) and adopted artifact sources with a
        published site-volume evidence extension combine enrolled-arm moments
        with whole-site volume evidence. The result selects the sum-metric or
        ratio-metric delta method from the declared metric.

        One result per call, for one treatment arm - required when
        more than one non-control arm is enrolled. Every other
        enrolled arm is netted out of the counterfactual baseline and
        counted in the ship-to-all population: the number answers
        "``arm`` shipped to every unit, versus no treatment shipped."

        Parameters
        ----------
        metric_name : str
            Name of a metric declared on this experiment.
        arm : str | None
            ``group_id`` of the treatment arm to score.
        alpha : float
            Two-sided significance level for both intervals. Default 0.05.

        Returns
        -------
        SitewideImpact | SitewideRatioImpact
            ``SitewideRatioImpact`` for a :class:`RatioMetric`, else ``SitewideImpact``.

        Raises
        ------
        CapabilityError
            On a seam-family instance, a
            :class:`RetentionMetric`/:class:`QuantileMetric` (no
            site-wide reading), or a clustered metric that doesn't
            decompose or whose arms' mean cluster sizes differ by more
            than 20%. Spillover caveat: unexposed units in a treated
            cluster bias the reported relative impact low.
        ValueError
            *metric_name* is not declared, has no enrolled arm, or
            ``arm`` names an arm that isn't enrolled; ``alpha`` is not
            strictly between 0 and 1.
        """
        experiment = getattr(self, "_experiment", None)
        if experiment is not None and experiment.trigger is not None:
            _raise(
                "analysis.sitewide.triggered_population_unsupported",
                experiment=experiment.name,
            )
        return SitewideReadouts(src=self._src, experiment=getattr(self, "_experiment", None)).run(
            SitewideRequest(metric_name=metric_name, arm=arm, alpha=alpha)
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
        selected: Sequence[Metric] | None = None,
        design_summary_only: bool = False,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    ) -> MomentSource:
        """The source `readouts` should read this readout from.

        Normally the moments cube: one warehouse reduction serves every
        metric. Design-level compliance, clustered, quantile and pooled-percentile
        analyses use the warehouse source directly. A triggered population uses
        the source-owned population view on that direct path.
        """
        if classify_source(self._src) == "artifact":
            if population == "triggered":
                return (self._src).triggered_source()
            return self._src
        # Compliance and per-unit methods need the original source.
        needs_direct_source = (
            design_summary_only
            or self._experiment.cluster is not None
            or getattr(self._design, "mechanism", None) == "observational"
            or any(
                getattr(m, "type", None) == "quantile"
                or getattr(getattr(m, "winsorization", None), "has_percentile", False)
                for m in (selected or self._metrics)
            )
        )
        if needs_direct_source:
            if population == "assigned":
                return self._src
            triggered = _require_analysis_operation(
                self._src,
                "triggered_source",
                TriggeredPopulationOperation,
                message=(
                    "_readout_source(population='triggered') needs a native "
                    "Analysis.from_definitions instance -- a frame/warehouse/"
                    "moments-backed analysis retains no raw event stream to "
                    "narrow to a triggered population."
                ),
            )
            return triggered.triggered_source()
        src = _require_analysis_operation(
            self._src,
            "moments_source",
            NativeCoreSource,
            message=(
                "moments_source() needs a native Analysis.from_definitions "
                "instance -- a frame/warehouse/moments-backed analysis has "
                "nothing beyond what you already hold to re-serve."
            ),
        )
        source_methods = None
        if decision_method is not UNSET or sensitivity_methods is not UNSET:
            source_methods = []
            seen_method_names: set[str] = set()
            for metric in selected or self._metrics:
                config = next(
                    config
                    for config in self._src.context.configs
                    if config.metric.name == metric.name
                )
                base_methods = effective_methods(config, design=self._src.context.design)
                effective = (
                    (
                        cast(Method, decision_method)
                        if decision_method is not UNSET
                        else base_methods[0]
                    ),
                    *(
                        tuple(cast("Sequence[Method]", sensitivity_methods))
                        if sensitivity_methods is not UNSET
                        else base_methods[1:]
                    ),
                )
                for method in effective:
                    if method.name not in seen_method_names:
                        seen_method_names.add(method.name)
                        source_methods.append(method)
                    elif method.variance_reduction == "cuped":
                        # Native reduction accepts one global method catalog;
                        # retain the capability union when metrics reuse a
                        # label with different variance-reduction settings.
                        index = next(
                            i
                            for i, existing in enumerate(source_methods)
                            if existing.name == method.name
                        )
                        source_methods[index] = method
        return src.moments_source(
            metrics=selected or self._metrics,
            methods=source_methods,
            population=population,
            narrow_cuped=True,
        )

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
            _refuse(
                _ANALYSIS_OPERATION,
                message="artifact_context requires Analysis.from_definitions",
                operation="materialize",
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

    def _exploratory_analysis(self, added: Sequence[Metric], *, caller: str) -> Analysis:
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
        if "summary_sql" not in src.operations:
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
        return planning_baseline(
            cast("MomentSource", self._src),
            metric,
            trigger=None if experiment is None else experiment.trigger,
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
        """
        from increment.sequential_source import adopt_source_snapshot, source_snapshot
        from increment.sequential_state import declare_sequential_freeze, sequential_refuse

        source = self._state.source
        capture = getattr(source, "capture_sequential", None)
        if capture is None:
            if not finalized or as_of is not None:
                sequential_refuse(
                    "source.invalid",
                    "this source requires its construction-time finalized checkpoint",
                )
            snapshot = source_snapshot(source, previous=previous)
        else:
            if as_of is None:
                sequential_refuse("source.invalid", "capture requires an explicit as_of horizon")
            snapshot = capture(finalized=finalized, previous=previous, as_of=as_of)
        if freeze:
            snapshot = adopt_source_snapshot(
                source, declare_sequential_freeze(snapshot, tuple(freeze))
            )
        return snapshot

    def sequential_snapshot(self, *, previous=None):
        """Retrieve a current exact checkpoint, optionally proving its parent."""
        from increment.sequential_source import source_snapshot

        return source_snapshot(self._state.source, previous=previous)

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
    ) -> LiftEstimates | ContrastResults:
        """Run the full A/B test analysis pipeline.

        Epistemic policy defaults come from the source's `AnalysisPlan` and
        per-metric `ExperimentMetric` bindings; the parameters below override
        them for this call only.

        Parameters
        ----------
        metrics : Sequence[str | Metric] | None
            Narrow the estimated metrics. ``None`` (default) selects
            every declared metric.
        decision_method : Method | _Unset
            Decision estimator override; UNSET inherits each metric's declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Additional estimator overrides; UNSET inherits each declaration.
        prior : Normal | StudentTPrior | MixturePrior | None | _Unset
            Informative prior override; UNSET inherits, None resets it. A
            mixture prior is refused on observational/encouragement rows -
            pass ``Normal``.
        estimands : Sequence[str] | None
            Under :class:`Encouragement` only: among ``"itt"``/
            ``"compliance"``/``"late"``, default all three.
        value_scale : Mapping[str, Literal["relative", "absolute"]] | None
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
        if isinstance(self._state, ContrastAnalysisState):
            return self._run_contrast(
                decision_method, sensitivity_methods, prior, metrics, estimands, value_scale
            )
        selected = select_metrics(cast("Sequence[Metric]", self._metrics), metrics, caller="run")

        def request(chosen: Sequence[Metric]) -> WholeWindowRequest:
            return WholeWindowRequest(
                metrics=tuple(chosen),
                estimands=tuple(estimands) if estimands is not None else None,
                value_scale=value_scale,
                population=population,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )

        rows = (
            self._whole_window().run(request(selected))
            if selected or not added
            else LiftEstimates()
        )
        if not isinstance(rows, LiftEstimates):
            rows = LiftEstimates(rows)
        if added:
            extra = (
                self._exploratory_analysis(added, caller="run")._whole_window().run(request(added))
            )
            extra = LiftEstimates(
                [row for row in extra if row.estimand != "compliance"],
                metadata=extra.metadata,
                source=extra.source,
                sequential_snapshot=extra.sequential_snapshot,
            )
            extra = _stamp_exploratory_rows(extra)
            rows = extra if rows.metadata is None else rows.concat(extra)
        return rows

    def _run_contrast(
        self,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        metrics: Sequence[str | Metric] | None,
        estimands: Sequence[str] | None,
        value_scale: Mapping[str, ValueScale] | None,
    ) -> ContrastResults:
        reject_role_overrides_under_contrast(
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
        if estimands is not None or value_scale:
            self._require_arm_state("run")
        context = cast("ContrastContext", self._context)
        selected = select_metrics(cast("Sequence[Metric]", context.metrics), metrics, caller="run")
        request = ContrastReadoutRequest(
            metrics=tuple(selected),
            procedures={metric.name: context.procedures[metric.name] for metric in selected},
        )
        return ContrastHandler().run(cast("ContrastSource", self._src), request)

    def _whole_window(self) -> WholeWindowReadouts:
        self._require_arm_state("run")
        return WholeWindowReadouts(
            src=cast("MomentSource", self._src),
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
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        alpha: float = 0.05,
        ard: bool = False,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
    ) -> CateResult:
        """Fit CATE using this analysis's unit source and declared intervention."""
        from increment.cate import estimate_cate

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
        self._require_arm_state("breakout_summaries")
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
        decision_method, sensitivity_methods, prior
            Explicit role/prior overrides; UNSET inherits each metric's declaration.
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
        return self._run_breakout(
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            population=population,
        )

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
    ) -> BreakoutEstimates:
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
    ) -> BreakoutEstimates:
        """:meth:`run_breakout`, optionally under an explicit *correction* in place of the
        plan's breakout multiplicity (``"none"`` leaves every cell standing alone)."""
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
        return BreakoutReadouts(src=self._src, experiment=getattr(self, "_experiment", None)).run(
            request
        )

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
            unbounded one always raises (see ``Raises``).
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

        Returns
        -------
        DailyMetricValues
            One row per (metric x day x arm), or per (metric x day x
            arm x segment) when *dimension* is given. A slice too small
            to estimate (fewer than 2 units, or a non-positive mean)
            remains present with ``value=None`` and an ``unavailable``
            reason instead of being dropped.

        Raises
        ------
        ValueError
            The effective metric list contains a `RetentionMetric` with
            an unbounded band (use :meth:`run_asof`/:meth:`run_asof_lift`
            instead), or *dimension* matches no declared breakout
            property, or (dimensioned only) *metrics* names an
            undeclared metric.
        """
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
        return self._day_axis().values(req)

    def _daily_lift(self, **kwargs: Any) -> DailyLiftEstimates:
        """Internal day-axis boundary: always supply an effective compiled plan.

        Call-time Metric objects absent from the analysis plan receive locally
        compiled unassigned procedures at the plan's alpha, matching the method/prior
        defaults the facade already resolves for them.
        """
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
        native = _day_axis_source_route(self._src) == "native"
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

    def _day_axis(self) -> DayAxisReadouts:
        return DayAxisReadouts(
            src=self._src,
            plan=self._plan,
            design=self._design,
            experiment=getattr(self, "_experiment", None),
            daily_lift=self._daily_lift,
        )

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
        decision_method, sensitivity_methods : Method | Sequence[Method] | _Unset
            Explicit role overrides; UNSET inherits the source metric bindings.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets.
        metrics : Sequence[str | Metric] | None
            Override the declared metrics + guardrails. A bounded-band
            :class:`RetentionMetric` is accepted; an
            unbounded one always raises (see ``Raises``).
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

        Returns
        -------
        DailyLiftEstimates
            One row per (day x metric x method x non-control arm), or
            with a segment axis when *dimension* is given. An
            unestimable slice remains present with ``lift=None`` and an
            ``unavailable`` reason instead of being dropped.

        Raises
        ------
        ValueError
            The effective metric list contains a `RetentionMetric` with
            an unbounded band, *dimension* matches no declared breakout
            property, (dimensioned only) an undeclared *metrics* entry,
            an Encouragement request that includes per-day LATE, which is
            unsupported because each daily first stage is weakly identified.
        UnsupportedRequestError
            A daily or cohort sequential-inference plan is refused because
            those disjoint slices cannot support sequential monitoring.
        """
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
        return self._day_axis().lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )

    def run_asof(
        self,
        *,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
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

        Returns
        -------
        DailyMetricValues
            One row per (metric x day x arm), or with a segment axis
            when *dimension* is given. An unestimable slice remains
            present with ``value=None`` and an ``unavailable`` reason.
            A unit's contribution freezes at its last in-window value, so
            this view's final day and :meth:`run`'s whole-window estimate can legitimately
            disagree for an experiment with late enrollees.

        Raises
        ------
        ValueError
            *completed_windows_only* is ``True`` while the effective
            metric list contains a `RetentionMetric` with an unbounded
            band, or *dimension* matches no declared breakout property.
        """
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
        return self._day_axis().values(req)

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
        decision_method, sensitivity_methods : Method | Sequence[Method] | _Unset
            Explicit role overrides; UNSET inherits the source metric bindings.
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
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); their rows carry ``role="exploratory"``
            and ``metrics=[]`` reads only these. A registered sequential plan refuses them.
        dimension : str | None
            Break each day's fixed-horizon lift out by one declared dimension.
            Unsupported for registered sequential inference.
        estimands : Sequence[str] | None
            Under :class:`Encouragement` only: ``"itt"``/``"compliance"``/``"late"``, default all three.
            A registered sequential checkpoint requested with exactly
            ``("compliance",)`` reads uptake alone and consumes no outcome metric,
            so a retention metric in the catalog does not affect it (see *Raises*).
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
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        Notes
        -----
        Alpha, inference, and segmented as-of multiplicity come from
        the resolved plan attached to this analysis. Segmented as-of BH is
        refused, as are observational designs and unsupported
        retention/window combinations.

        Returns
        -------
        DailyLiftEstimates
            Fixed-horizon: one row per (day x metric x method x non-control arm),
            optionally per segment. Unestimable slices retain ``lift=None`` and
            an ``unavailable`` reason. Registered sequential inference returns
            only the current checkpoint rows, all with its date label.
        Raises
        ------
        ValueError
            An unbounded retention band with *completed_windows_only* in a request
            that consumes outcome metrics, an unmatched *dimension*, or
            non-``("itt",)`` *estimands* under a non-encouragement design.
        CapabilityError
            A `RetentionMetric` under an :class:`Encouragement` design in a request
            that consumes outcome metrics, or a dimension requested with registered
            sequential inference. A registered compliance-only checkpoint
            (``estimands=("compliance",)``) consumes no outcome metric, so neither
            retention refusal applies to it.
        UnsupportedRequestError
            Segmented as-of BH or observational designs.
        """
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
        asof_policy = self._plan.view_policies.for_view(
            "asof",
            mechanism=getattr(self._design, "mechanism", None),
            segmented=dimension is not None,
        )
        return self._stamp_exploratory(
            self._day_axis().lift(
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            ),
            added,
            correction=normalize_display_correction(asof_policy.correction),
        )
