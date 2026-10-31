"""DefinitionsMomentSource: the warehouse `MomentSource` over the
definitions/ibis pipeline - the raw-events sibling of `SqlPanelSource`,
`FrameTotalsSource`, `FramePanelSource`, `MomentsSource`. The only
substrate offering every grain plus SQL introspection, because it
alone retains raw fact tables.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import ibis
import ibis.expr.types as ir
from ibis import Table
from ibis import to_sql as ibis_to_sql

from increment._analysis_config import (
    effective_methods,
    ensure_covariate_binding_compatible,
    overlay_configs,
    resolve_configs,
)
from increment._cuped_evidence import (
    CUPED_EVIDENCE_FORMAT,
    method_key,
    portable_cuped_rows,
    portable_requests,
)
from increment._evidence_refusals import winsorization_request_payload
from increment._evidence_types import (
    AssignmentEvidence,
    AssignmentRequest,
    BreakoutIdentity,
    EvidenceScope,
    evidence_scope_dependencies,
)
from increment._frame_validation import (
    COVARIATE_COLUMN_MISSING as _NATIVE_COVARIATE_MISSING,
)
from increment._frame_validation import (
    COVARIATE_IMPUTE_ALL_NULL as _NATIVE_COVARIATE_IMPUTE_ALL_NULL,
)
from increment._immutable import _FrozenMapping
from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW
from increment._source_operations import DashboardGroupData
from increment._source_types import EvidenceCapabilities, ScalarEvidenceDeclaration
from increment._window import NO_DATA_SIGNAL, resolve_window_days
from increment.errors import CodedError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.estimation._readout_refusals import refuse_observational_quantile
from increment.query._native_dashboard import (
    _dashboard_quantiles,
    _observed_dashboard_group,
    _retained_dashboard_groups,
)
from increment.query._native_day_source import (
    _DaySource,
    _refuse_missing_pre_period,
    _refuse_unsupported_day_metric,
)
from increment.query._native_materialization import _NativeMaterializationMixin
from increment.query._native_refusals import (
    _NATIVE_COMPLIANCE_DESIGN_MISMATCH,
    _NATIVE_COVARIATE_AMBIGUOUS,
    _NATIVE_COVARIATE_AS_OF,
    _NATIVE_COVARIATE_DTYPE,
    _NATIVE_COVARIATE_KIND,
    _NATIVE_COVARIATE_UNRESOLVED,
    _NATIVE_TRIGGER_EVIDENCE_REQUIRED,
    _percentile_winsorization,
    _refuse_operation,
)
from increment.query._native_triggered import TriggeredPopulationSource
from increment.query.artifact_contract import ArtifactStore
from increment.query.artifact_publish import ArtifactPublisher, _effective_snapshot_edge
from increment.query.builders import (
    _apply_filter,
    _attach_uptake_flag,
    _dense_unit_days,
    _local_date,
    _observable_window_flags,
    _scope_exposure_events,
    _uptake_events_in_elapsed_window,
    _utc_timestamp_literal,
    _windowed_fact_sum,
    breakout_property_table,
    canonical_dimension_value,
    cluster_exposure_counts,
    cohort_group_summary,
    compliance_event_horizon,
    daily_exposure_counts,
    daily_group_summary,
    first_exposures,
    group_summary,
    join_breakout_dimension,
    join_pre_period_covariate,
    join_unit_covariates,
    metric_events,
    panel_spine,
    post_exposure_stats,
    pre_period_stats,
    site_volume,
    triggered_population,
    union_event_horizon,
    unit_day_panel,
    unit_day_spine_stats,
    unit_totals,
    window_bound_stats,
    winsorize_unit_totals,
)
from increment.query.fact_resolution import (
    _find_fact_source,
    _rename_to_builder_cols,
    _resolve_breakout_fact_source,
    _resolve_value_column,
    validate_fact_source_schema,
)
from increment.query.integrity import (
    arm_counts,
    enforce_mixed_assignments,
    mixed_assignment_snapshot,
    validate_cluster_labels,
    validate_cluster_uniqueness,
)
from increment.query.native_contract import (
    _NATIVE_COVARIATE_RESERVED,
    _OUTCOMES_INCOMPLETE_ROUTES,
    _SITEWIDE_OUTCOMES_INCOMPLETE,
    DayEvidenceSource,
    SitewideEvidence,
)
from increment.query.session import SourceScope, SourceSnapshotEvidence, WarehouseSession
from increment.semantics.artifact import UnitDayArtifactRef
from increment.semantics.design import Encouragement
from increment.semantics.models import (
    Breakout,
    ConversionMetric,
    CupedCovariate,
    CupedCovariateBinding,
    Definitions,
    Experiment,
    Factor,
    FactSource,
    MeanMetric,
    Measure,
    Metric,
    ObservationalDeclaration,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
)
from increment.sequential_source import SequentialSourceMixin
from increment.sources import (
    ASSIGNMENT_COUNTS_FIELD,
    COMPLIANCE_SUMMARY_FIELD,
    MIXED_ASSIGNMENT_LABEL,
    UNASSIGNED_LABEL,
    BreakoutMomentsSource,
    ComplianceArm,
    ComplianceSummary,
    SourceContext,
    SourceOperation,
    _BuiltinMomentSource,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Observational, Randomized
    from increment.sequential_state import SequentialCheckpoint
    from increment.sources import Grain, _BuiltinMomentSource


_SNAPSHOT_HORIZON_RELATION_KEY = "__increment_readout_snapshot_horizon__"


@dataclass(frozen=True, slots=True)
class _ScalarBreakoutPreparation:
    dimension: str
    properties: Table
    refusals: dict[str, list[tuple[str, str, RefusalSpec, Mapping[str, object]]]]


_RETAINED_SNAPSHOT_INTEGRITY_SOURCES: ContextVar[tuple[int, ...]] = ContextVar(
    "_RETAINED_SNAPSHOT_INTEGRITY_SOURCES", default=()
)


class DefinitionsMomentSource(_NativeMaterializationMixin, SequentialSourceMixin):
    """MomentSource over the definitions/ibis pipeline - the warehouse
    sibling of SqlPanelSource, FrameTotalsSource, FramePanelSource,
    MomentsSource. The one substrate offering every grain plus SQL
    introspection, because it alone retains raw fact tables.

    Public ``design``/``plan`` read-only properties mirror the frame
    sources' convention (``FrameTotalsSource.design``/``.plan``,
    ``FramePanelSource.design``/``.plan``): the identification mechanism
    and the fully-resolved ``AnalysisPlan`` this instance was constructed
    under. ``plan`` is a required constructor argument, so it is always
    set (never ``None``). ``design`` defaults to ``None`` for direct source
    construction, while ``Analysis.__init__`` resolves and passes the
    declared design on the native facade path. This instance has no
    ``design`` setter; callers that need a different design should build a
    new source through the appropriate constructor.

    """

    supports_retained_snapshot_integrity_reuse: bool = True

    capabilities: frozenset[Grain] = frozenset({"total", "daily", "asof"})
    operations: frozenset[SourceOperation] = frozenset(
        {
            "allocation_history",
            "readout_snapshot",
            "readout_execution",
            "dashboard_group_data",
            "materialize",
            "triggered_counts",
            "triggered_source",
            "sitewide_evidence",
            "export_moments",
            "moments_source",
            "panel_sql",
            "summary_sql",
            "breakout_summaries",
            "factor_summaries",
            "breakout_source",
            "breakout_sources",
            "day_source",
            "exploratory_source",
            "unit_counts",
        }
    )
    shape: Literal["unit_summary", "unit_panel"] | None = None

    @property
    def evidence(self) -> EvidenceCapabilities:
        return EvidenceCapabilities(
            kind="definitions",
            grains=self.capabilities,
            breakouts=tuple(self.breakouts),
            shape=self.shape,
            operations=self.operations,
            capture="snapshot",
            unit_rows=True,
            raw_outcomes=True,
            method_fit="warehouse",
            compliance_axis="asof" if isinstance(self.context.design, Encouragement) else "none",
            count_grains=frozenset(
                {"unit", "assignment"}
                | ({"cluster"} if self.context.cluster is not None else set())
                | ({"triggered"} if self.context.trigger_name is not None else set())
            ),
            populations=frozenset(
                {"assigned"} | ({"triggered"} if self.context.trigger_name is not None else set())
            ),
            sequential_capture="asof",
            scalar=ScalarEvidenceDeclaration(
                bindings=None,
                moments_only=False,
                recipe_fixed=False,
                recipe_configs=(),
                anonymous_metrics=frozenset(),
                method_evidence=True,
                breakout_evidence=True,
                covariate_grains=None,
            ),
            available_covariates=None,
            assignment_relation=(
                "ledger" if self._experiment.assignment_ledger is not None else "reduced"
            ),
            sql_grains=frozenset({"total"}),
        )

    def declaration_identity(self) -> Mapping[str, Any]:
        """Return source declarations without capture or observation values."""
        return {
            "experiment": self._experiment,
            "definitions": self._defs,
            "metrics": self._metrics,
            "observation_mapping": self._sequential_observation_mapping(),
        }

    def __init__(
        self,
        session: WarehouseSession,
        experiment: Experiment,
        metrics: Sequence[Metric],
        *,
        store: Literal["auto", "always", "none"],
        on_mixed_assignment: Literal["error", "warn", "exclude"],
        backend: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: CompiledDecisionPlan,
    ) -> None:
        self._session = session
        self._con = session.con
        self._defs: Definitions = session.defs
        self._experiment = experiment
        self._experiment_name = experiment.name
        self.breakouts: Sequence[str] = tuple(sorted({b.property for b in experiment.breakouts}))
        self._metrics: list[Metric] = list(metrics)
        self._horizon_metrics = tuple(self._metrics)
        self._store: Literal["auto", "always", "none"] = store
        self._on_mixed_assignment: Literal["error", "warn", "exclude"] = on_mixed_assignment
        self._backend = backend
        self._declaration_schema: dict[str, str] | None = None
        from increment.plan import refuse_observational_relative_margin

        refuse_observational_relative_margin(design, plan)

        self._context = SourceContext(
            study_id=self._experiment_name,
            design=design,
            plan=plan,
            metrics=self._horizon_metrics,
            configs=resolve_configs(
                self._metrics,
                bindings=self._experiment.bindings,
                specs=None,
                methods=None,
                prior=None,
                covariate_available_metrics=(
                    tuple(metric.name for metric in self._metrics)
                    if self._experiment.n_pre_periods > 0
                    else ()
                ),
            ),
            cluster=self._experiment.cluster,
            intervention_grain=self._experiment.intervention_grain,
            trigger_name=self._experiment.trigger,
        )
        from increment._analysis_config import validate_scalar_route
        from increment.estimation.sequential import SEQUENTIAL_POLICIES

        if isinstance(plan.inference, SEQUENTIAL_POLICIES):
            validate_scalar_route(
                self.context.configs,
                route="native.sequential_capture",
                representation="native_sequential",
                design=design,
            )
        self._exposures_cache: Table | None = None
        from increment.sequential_source import validate_source_mapping

        validate_source_mapping(self._context, self._sequential_observation_mapping())
        self._trigger_cache: Table | None = None
        self._mixed_assignment_units: int | None = None
        self._unassigned_assignment_units: int | None = None
        self._mixed_assignment_fingerprint: int | None = None
        self._mixed_assignment_prefix_id: str | None = None
        self._retained_snapshot_integrity = False
        self._mixed_assignment_warned = False
        self._union_horizon_cache: ir.Scalar | None = None
        self._union_horizon_metric_cache: dict[tuple[Metric, ...], ir.Scalar] = {}
        self._fixed_event_horizon: dt.date | None = None
        self._has_fixed_event_horizon_override = False
        self._panel_cache: dict[
            tuple[str, Metric] | tuple[str, Metric, tuple[Metric, ...]],
            tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar],
        ] = {}
        self._cuped_relation_cache: dict[tuple[str, str, dt.date | None], Table] = {}
        self._reduction_calls = 0
        self._reduction_batch_depth = 0
        self._assignment_validation_depth = 0
        self._materialized = False
        self._exp_lookup_exposures = {e.name: e for e in self._defs.exposures}

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def design(self):
        """The identification design captured at construction."""
        return self._context.design

    @property
    def plan(self) -> CompiledDecisionPlan:
        """The compiled decision plan captured at construction."""
        return self._context.plan

    def _get_exposures(self) -> Table:
        """Return the cached lazy exposure expression without executing it."""
        if self._exposures_cache is None:
            exposure_events_table = self._build_exposure_events_table()
            self._exposures_cache = first_exposures(exposure_events_table, self._experiment)
        return self._exposures_cache

    def _get_trigger_population(self, *, operation: str) -> Table:
        """Eligible enrolled units with their true first-trigger anchors."""
        if self._trigger_cache is None:
            self._validate_trigger_capability(operation=operation)
            evidence = self._require_source_snapshot_evidence(operation=operation)
            trigger_events = self._build_trigger_events_table()
            self._trigger_cache = triggered_population(
                self._get_exposures(),
                trigger_events,
                observation_cutoff=evidence.observation_cutoff_ts,
            )
        return self._trigger_cache

    def _require_source_snapshot_evidence(self, *, operation: str) -> SourceSnapshotEvidence:
        evidence = self._session.source_snapshot_evidence
        if evidence is None:
            _refuse(
                _NATIVE_TRIGGER_EVIDENCE_REQUIRED,
                operation=operation,
                route=(
                    "pass source_snapshot_evidence=SourceSnapshotEvidence(...) to "
                    "Analysis.from_definitions using the upstream source's explicit "
                    "event-time cutoff; provide feed watermarks only when certified"
                ),
            )
        return evidence

    def _validate_trigger_capability(self, *, operation: str) -> None:
        """Refuse a triggered population before resolving warehouse tables."""
        experiment = self._experiment
        if experiment.trigger is None:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": experiment.name,
                    "population": "triggered",
                    "trigger": experiment.trigger,
                },
                offered=("assigned",),
                route="declare `trigger: <exposure name>` on the experiment",
            )
        self._require_source_snapshot_evidence(operation=operation)

    def validate_populations(
        self,
        populations: Sequence[Literal["assigned", "triggered"]],
        *,
        operation: str,
    ) -> None:
        """Validate every requested population before starting a view reduction."""
        if "triggered" in populations:
            self._validate_trigger_capability(operation=operation)

    def _invalidate_materialization(self) -> None:
        """Drop the previous operation's TEMP tables and cached panel expressions."""
        if not self._materialized:
            return
        self._session.drop_materialized()
        self._materialized = False
        self._panel_cache = {}
        self._cuped_relation_cache = {}

    def _mixed_assignment_snapshot(self, exposure_events: Table) -> tuple[int, int, int]:
        """Count mixed and NULL assignments with one bounded aggregate."""
        return mixed_assignment_snapshot(self._con, exposure_events, self._experiment)

    def _mixed_assignment_count(self, exposure_events: Table) -> int:
        """Count mixed units with one bounded aggregate result."""
        return self._mixed_assignment_snapshot(exposure_events)[0]

    def _validate_mixed_assignments(self) -> None:
        """Enforce assignment integrity, refreshing only for live warehouse inputs."""
        if self._assignment_validation_depth:
            return
        retained_snapshot = getattr(self, "_sequential_snapshot", None)
        # A checkpoint's counts are authoritative only inside its checkpoint-backed run scope.
        retained_integrity = (
            self._retained_snapshot_integrity
            and retained_snapshot is not None
            and retained_snapshot.finalized
            and retained_snapshot.assignment_counts is not None
            and self._mixed_assignment_prefix_id == retained_snapshot.prefix_id
            and id(self) in _RETAINED_SNAPSHOT_INTEGRITY_SOURCES.get()
        )
        if retained_integrity:
            enforce_mixed_assignments(
                retained_snapshot.assignment_counts.get(MIXED_ASSIGNMENT_LABEL, 0),
                self._on_mixed_assignment,
                unassigned_count=retained_snapshot.assignment_counts.get(UNASSIGNED_LABEL, 0),
                already_warned=self._mixed_assignment_warned,
            )
            return
        if (
            self._session._source_tables is not None
            and self._mixed_assignment_fingerprint is not None
        ):
            assert self._mixed_assignment_units is not None
            assert self._unassigned_assignment_units is not None
            self._mixed_assignment_warned = enforce_mixed_assignments(
                self._mixed_assignment_units,
                self._on_mixed_assignment,
                unassigned_count=self._unassigned_assignment_units,
                already_warned=self._mixed_assignment_warned,
            )
            return
        exposure_events_table = self._build_exposure_events_table()
        mixed, unassigned, fingerprint = self._mixed_assignment_snapshot(exposure_events_table)
        previous_fingerprint = getattr(self, "_mixed_assignment_fingerprint", None)
        self._mixed_assignment_units = mixed
        self._unassigned_assignment_units = unassigned
        self._mixed_assignment_fingerprint = fingerprint
        sequential_snapshot = getattr(self, "_sequential_snapshot", None)
        self._mixed_assignment_prefix_id = (
            sequential_snapshot.prefix_id
            if sequential_snapshot is not None and sequential_snapshot.finalized
            else None
        )
        if previous_fingerprint is not None and previous_fingerprint != fingerprint:
            self._invalidate_materialization()
        self._mixed_assignment_warned = enforce_mixed_assignments(
            mixed,
            getattr(self, "_on_mixed_assignment", "error"),
            unassigned_count=unassigned,
            already_warned=getattr(self, "_mixed_assignment_warned", False),
        )

    def _adopt_assignment_validation(self, pinned: DefinitionsMomentSource) -> None:
        """Keep live integrity state and materialization caches aligned with a pinned read."""
        fingerprint = pinned._mixed_assignment_fingerprint
        if fingerprint is None:
            return
        previous = self._mixed_assignment_fingerprint
        if previous is not None and previous != fingerprint:
            self._invalidate_materialization()
        self._mixed_assignment_units = pinned._mixed_assignment_units
        self._unassigned_assignment_units = pinned._unassigned_assignment_units
        self._mixed_assignment_fingerprint = fingerprint
        self._mixed_assignment_prefix_id = pinned._mixed_assignment_prefix_id
        self._mixed_assignment_warned = pinned._mixed_assignment_warned

    @contextmanager
    def _validated_assignments(self) -> Iterator[None]:
        """Validate assignments once for a series of per-date reads."""
        self._validate_mixed_assignments()
        self._assignment_validation_depth += 1
        try:
            yield
        finally:
            self._assignment_validation_depth -= 1

    def _get_fact_table(self, fact_source: FactSource, *, require_event: bool = True) -> Table:
        """Memoized fact table with event-schema validation for event consumers."""
        return self._session.fact_table(
            fact_source, self._experiment.unit, require_event=require_event
        )

    def _site_volume_scope_names(
        self, site_volume_metrics: frozenset[str], sources: dict[str, FactSource]
    ) -> tuple[set[str], set[str]]:
        """Resolve which fact sources (and the dims they join) back a
        requested site_volume metric, so `_pinned_source` can drop their
        unit hint without touching the window hint.

        A site_volume request reads whole-population events in [start, end]
        (artifact_publish.py's own _relevant_events site branch), never
        scoped to enrolled units -- resolve which fact source(s) back each
        requested site-volume metric name the same way add_fact already
        resolves an ordinary metric's fact. A dim a site_volume-referenced
        fact source joins (e.g. a property the metric filters on) must not
        keep its own unit filter either: the fact source's rows now include
        non-enrolled units, but a dim scoped to enrolled units only would
        left-join those rows to a NULL property and silently fail any
        filter on it, re-shrinking the "whole site" count back to roughly
        the enrolled population through the join instead of through the
        fact source directly.
        """
        site_volume_source_names: set[str] = set()
        if site_volume_metrics:
            for metric in self._metrics:
                if metric.name not in site_volume_metrics:
                    continue
                fact_names = (
                    (metric.numerator.fact, metric.denominator.fact)
                    if isinstance(metric, RatioMetric)
                    else (metric.fact,)
                )
                for fact_name in fact_names:
                    fact_source, _ = _find_fact_source(self._defs, fact_name)
                    site_volume_source_names.add(fact_source.name)
        site_volume_dim_names = {
            dim_name
            for source in sources.values()
            if source.name in site_volume_source_names
            for dim_name in source.dims
        }
        return site_volume_source_names, site_volume_dim_names

    def _scope_enrolled_units(self) -> Table:
        """Distinct enrolled ``(unit_id, experiment_id)`` pairs, used only to
        narrow the pinned snapshot's non-exposure branches (`SourceScope`).

        Reads the exposure's raw rows directly (``dim_joined=False``)
        rather than through `_get_exposures`'s dim-joined, first-exposure
        machinery: population membership never depends on a joined
        dimension property or on picking one row per unit -- being
        over-inclusive here only means a scoped branch copies a few extra
        rows, never that it drops one (see the margins note below). Reusing
        `_get_exposures` would force a second dim-joined build of the same
        fact source `_assemble_publication` builds again from the pinned
        snapshot.
        """
        events = self._build_events_table_for_exposure(self._experiment.exposure, dim_joined=False)
        return events.select("unit_id", "experiment_id").distinct()

    def evidence_scope_covariate_streams(
        self, scope: EvidenceScope
    ) -> Mapping[str, tuple[str, ...]]:
        dependencies = evidence_scope_dependencies(scope)
        bindings: dict[str, list[str]] = {}
        adjustments = dependencies["adjustments"]
        if adjustments is None:
            adjustments = tuple(
                config for config in self._context.configs if config.metric in scope.metrics
            )
        for config in adjustments:
            for method in effective_methods(config, design=self.context.design):
                for covariate in method.covariates or ():
                    covariate = self._resolve_covariate_binding(covariate, config.metric)
                    if isinstance(covariate, CupedCovariateBinding) and covariate.measure:
                        continue
                    property_name = getattr(covariate, "property", None) or covariate.name
                    source = self._resolve_unit_covariate_source(
                        property_name,
                        declared_source=getattr(covariate, "source", None),
                        kind=getattr(covariate, "kind", None),
                    )
                    source_names = bindings.setdefault(property_name, [])
                    if source.name not in source_names:
                        source_names.append(source.name)
        design = self._experiment.design
        if isinstance(design, ObservationalDeclaration):
            for covariate in design.covariates:
                property_name = covariate.property
                source_name = self._resolve_unit_covariate_source(
                    property_name,
                    declared_source=covariate.source,
                    kind=getattr(covariate, "kind", None),
                ).name
                source_names = bindings.setdefault(property_name, [])
                if source_name not in source_names:
                    source_names.append(source_name)
        return {
            property_name: tuple(source_names) for property_name, source_names in bindings.items()
        }

    def evidence_scope_breakout_streams(
        self, scope: EvidenceScope
    ) -> Mapping[BreakoutIdentity, str]:
        """Resolve each selected breakout to its captured fact-source stream."""
        bindings = {}
        for breakout in scope.breakouts:
            try:
                source = _resolve_breakout_fact_source(
                    self._defs,
                    self._experiment,
                    Breakout(property=breakout.property, source=breakout.source),
                )
            except CodedError:
                continue
            bindings[breakout] = source.name
        return bindings

    def identity_assignment_evidence(self, request: AssignmentRequest) -> AssignmentEvidence | None:
        """Reduce an explicitly role-mapped assignment relation inside the query boundary."""
        binding = self._experiment.assignment_ledger
        if binding is None:
            return None
        if request.experiment != self.context.study_id:
            from increment._evidence_refusals import refuse_assignment_selection

            refuse_assignment_selection(
                self,
                requested={"experiment": request.experiment},
                offered={"experiment": self.context.study_id},
            )
        from decimal import Decimal

        from increment.query.ledger import Ledger

        roles = binding.roles
        ledger = Ledger(self._session.source_sql(binding.sql), roles)
        reduced = ledger.reduce(
            experiment=request.experiment,
            population=request.population,
            version=request.version,
            period=request.period,
            all_periods=request.all_periods,
            key="cluster" if self._experiment.cluster is not None else None,
        )
        allocation = getattr(self.context.design, "allocation", None)
        probabilities: tuple[tuple[str, float], ...] = ()
        if allocation:
            weights = tuple((str(arm), Decimal(str(value))) for arm, value in allocation.items())
            scale = max(weight for _, weight in weights)
            total = sum((weight / scale for _, weight in weights), Decimal(0))
            probabilities = tuple(
                sorted((arm, float((weight / scale) / total)) for arm, weight in weights)
            )
        return AssignmentEvidence(
            experiment=request.experiment,
            population=request.population,
            versions=reduced.versions,
            periods=reduced.periods,
            arm_counts=reduced.arm_counts,
            independence_grain=reduced.independence_grain,
            probabilities=probabilities,
            randomization_law=getattr(self.context.design, "allocation_scheme", None),
        )

    def assignment_identity_rows(self) -> tuple[Mapping[str, Any], ...] | None:
        """Materialize the selected unit assignment ledger for artifact publication."""
        binding = self._experiment.assignment_ledger
        if binding is None:
            return None
        from increment.query.ledger import Ledger

        if "unit" not in binding.roles:
            return None

        source_relation = self._session.source_sql(binding.sql)
        relation = source_relation.filter(
            source_relation[binding.roles["experiment"]] == self._experiment.name
        )
        versions = relation.select(binding.roles["version"]).distinct().execute(limit=None)
        rows = []
        for version in sorted(int(value) for value in versions[binding.roles["version"]]):
            reduced = Ledger(relation, binding.roles).rows(
                experiment=self._experiment.name,
                population="assigned",
                version=version,
                all_periods=True,
            )
            rows.extend(
                {
                    "experiment_id": row.experiment,
                    "unit_id": row.unit,
                    "version": row.version,
                    "period": json.dumps(row.period, ensure_ascii=False),
                    "arm": row.arm,
                }
                for row in reduced.rows
            )
        rows.sort(
            key=lambda row: (
                str(row["experiment_id"]).encode("utf-8"),
                str(row["unit_id"]).encode("utf-8"),
                int(row["version"]),
                str(row["period"]).encode("utf-8"),
                str(row["arm"]).encode("utf-8"),
            )
        )
        return tuple(rows)

    def declaration_schema(self) -> dict[str, str]:
        """Return recorded input dtypes, enabling capture for subsequent readouts."""
        if self._declaration_schema is None:
            self._declaration_schema = {}
        return dict(self._declaration_schema)

    def _record_declaration_schema(self, captured: DefinitionsMomentSource) -> None:
        """Retain relation dtypes from the selected inputs of an existing capture."""
        if self._declaration_schema is None:
            return
        queries = {
            **{source.name: source.sql for source in self._defs.fact_sources},
            **{f"dimension:{source.name}": source.sql for source in self._defs.dim_sources},
            **{
                f"exposure:{exposure.name}": exposure.sql
                for exposure in self._defs.exposures
                if exposure.sql is not None
            },
        }
        for name, query in queries.items():
            columns = captured._session.recorded_schema(query)
            if columns is not None:
                self._declaration_schema.update(
                    {f"{name}.{column}": dtype for column, dtype in columns.items()}
                )

    def evidence_scope_streams(self, scope: EvidenceScope) -> Mapping[str, str]:
        dependencies = evidence_scope_dependencies(scope)
        tables = self._session._source_tables
        fact_names = set(dependencies["uptake"])
        for metric in (*dependencies["outcomes"], *dependencies["horizons"]):
            numerator = getattr(metric, "numerator", None)
            denominator = getattr(metric, "denominator", None)
            if numerator is not None and denominator is not None:
                fact_names.update((numerator.fact, denominator.fact))
            else:
                fact_names.add(metric.fact)

        sources: dict[str, FactSource] = {}
        for config in dependencies["adjustments"] or ():
            for method in effective_methods(config, design=self.context.design):
                for covariate in method.covariates or ():
                    covariate = self._resolve_covariate_binding(covariate, config.metric)
                    if isinstance(covariate, CupedCovariateBinding) and covariate.measure:
                        fact_names.add(covariate.measure.fact)
        fact_sources = {source.name: source for source in self._defs.fact_sources}
        sources.update(
            {
                source_name: fact_sources[source_name]
                for source_names in self.evidence_scope_covariate_streams(scope).values()
                for source_name in source_names
            }
        )
        for breakout in dependencies["breakouts"]:
            try:
                source = _resolve_breakout_fact_source(
                    self._defs,
                    self._experiment,
                    Breakout(property=breakout.property, source=breakout.source),
                )
            except CodedError:
                continue
            sources[source.name] = source

        streams: dict[str, str] = {}
        for fact in fact_names:
            source, _ = _find_fact_source(self._defs, fact)
            sources[source.name] = source
        dimensions: set[str] = set()
        for source in sources.values():
            streams[source.name] = source.name
            dimensions.update(source.dims)
        streams.update({f"dimension:{name}": f"dimension:{name}" for name in dimensions})
        for request in dependencies["assignments"]:
            if self._experiment.assignment_ledger is not None:
                stream = f"assignment-ledger:{request.experiment}"
                streams[stream] = stream

        for name in (self._experiment.exposure, self._experiment.trigger):
            if name is None:
                continue
            exposure = self._exp_lookup_exposures[name]
            if exposure.sql is not None:
                streams[f"exposure:{name}"] = f"exposure:{name}"
            elif exposure.fact is not None:
                source, _ = _find_fact_source(self._defs, exposure.fact)
                streams[source.name] = source.name
                streams.update(
                    {
                        f"dimension:{dimension}": f"dimension:{dimension}"
                        for dimension in source.dims
                    }
                )

        if tables is None:
            return dict(sorted(streams.items()))
        return self._captured_streams(tables)

    def _captured_streams(self, tables: Any) -> Mapping[str, str]:
        labels: dict[str, set[str]] = {}
        for source in self._defs.fact_sources:
            labels.setdefault(source.sql, set()).add(source.name)
        for dimension in self._defs.dim_sources:
            labels.setdefault(dimension.sql, set()).add(f"dimension:{dimension.name}")
        for exposure in self._defs.exposures:
            if exposure.sql is not None:
                labels.setdefault(exposure.sql, set()).add(f"exposure:{exposure.name}")
        if self._experiment.assignment_ledger is not None:
            labels.setdefault(self._experiment.assignment_ledger.sql, set()).add(
                f"assignment-ledger:{self._experiment.name}"
            )
        captured_streams: dict[str, str] = {}
        for query in tables:
            identities = labels.get(query)
            if identities is None:
                captured_streams[f"source:{query}"] = query
            else:
                captured_streams.update({identity: identity for identity in identities})
        return dict(sorted(captured_streams.items()))

    def _pinned_source(  # noqa: PLR0915, C901
        self,
        *,
        metrics: Sequence[Metric] | None = None,
        extra_sources: Sequence[str] = (),
        horizon_metrics: Sequence[Metric] = (),
        uptake_facts: Sequence[str] = (),
        site_volume_metrics: frozenset[str] = frozenset(),
        configs: Sequence[ResolvedMetricConfig] | None = None,
        snapshot_horizon_relation: Table | None = None,
        assignment_requests: Sequence[AssignmentRequest] = (),
    ) -> DefinitionsMomentSource:
        """Capture the experiment's input streams before any data validation."""
        if self._session._source_tables is not None:
            return self
        capture_scope = EvidenceScope(
            metrics=tuple(self._horizon_metrics if metrics is None else metrics),
            horizon_metrics=tuple(horizon_metrics),
            configs=tuple(self._context.configs if configs is None else configs),
            uptake_facts=tuple(uptake_facts),
            assignment_requests=tuple(assignment_requests),
        )
        dependencies = evidence_scope_dependencies(capture_scope)
        requested_configs = tuple(
            config
            for config in dependencies["adjustments"]
            if metrics is None or config.metric in metrics
        )
        capture_scope = replace(capture_scope, configs=requested_configs)
        capture_streams = self.evidence_scope_streams(capture_scope)
        covariate_streams = self.evidence_scope_covariate_streams(capture_scope)
        fact_sources = {source.name: source for source in self._defs.fact_sources}
        sources = {name: fact_sources[name] for name in capture_streams if name in fact_sources}
        event_source_names: set[str] = set()
        required_facts: set[str] = set()
        queries: list[str] = []
        column_hint_candidates: dict[str, list[tuple[str | None, str | None]]] = {}
        column_projections: dict[str, tuple[str, ...]] = {}
        # Exposure/trigger SQL stays unscoped: it defines enrollment and the
        # observation window, so filtering it by those same values is circular.
        # Shared fact sources also remain unscoped.
        unscoped_queries: set[str] = set()
        exposure_sql_queries: set[str] = set()

        def add_fact(fact: str, *, unscoped: bool = False) -> str:
            source, _ = _find_fact_source(self._defs, fact)
            required_facts.add(fact)
            sources[source.name] = source
            event_source_names.add(source.name)
            if unscoped:
                unscoped_queries.add(source.sql)
            return source.name

        # Freshness sees every unit in each requested metric's fact stream,
        # not just the enrolled units retained for outcome reduction.
        freshness_source_names: set[str] = set()
        horizon_metrics = tuple(
            dict.fromkeys((*dependencies["outcomes"], *dependencies["horizons"]))
        )
        for metric in horizon_metrics:
            if isinstance(metric, RatioMetric):
                freshness_source_names.add(add_fact(metric.numerator.fact))
                freshness_source_names.add(add_fact(metric.denominator.fact))
            else:
                freshness_source_names.add(add_fact(metric.fact))

        for name in (self._experiment.exposure, self._experiment.trigger):
            if name is None:
                continue
            exposure = self._exp_lookup_exposures[name]
            if exposure.sql is not None:
                queries.append(exposure.sql)
                exposure_sql_queries.add(exposure.sql)
                unscoped_queries.add(exposure.sql)
            else:
                assert exposure.fact is not None
                add_fact(exposure.fact, unscoped=True)
        if dependencies["assignments"] and self._experiment.assignment_ledger is not None:
            ledger = self._experiment.assignment_ledger
            ledger_sql = ledger.sql
            if ledger_sql not in queries:
                queries.append(ledger_sql)
            unscoped_queries.add(ledger_sql)
            column_projections[ledger_sql] = tuple(
                sorted(set(column_projections.get(ledger_sql, ())) | set(ledger.roles.values()))
            )
        if metrics is None:
            uptake = getattr(self._context.design, "uptake", None)
            if uptake is not None:
                add_fact(uptake.fact)
        for fact in dependencies["uptake"]:
            add_fact(fact)

        # Sources read only for a breakout/factor/covariate property lookup:
        # `_build_breakout_properties_table` takes the latest value over the
        # whole source (static) or strictly before exposure (pre_exposure),
        # so these never get the window's lower time bound either.
        property_source_names = set(extra_sources) | {
            source_name
            for source_names in covariate_streams.values()
            for source_name in source_names
        }
        for source_name in extra_sources:
            source = next(fs for fs in self._defs.fact_sources if fs.name == source_name)
            sources[source.name] = source
        for config in requested_configs:
            for method in effective_methods(config, design=self.context.design):
                for covariate in method.covariates or ():
                    covariate = self._resolve_covariate_binding(covariate, config.metric)
                    if (
                        isinstance(covariate, CupedCovariateBinding)
                        and covariate.measure is not None
                    ):
                        add_fact(covariate.measure.fact)

        site_volume_source_names, site_volume_dim_names = self._site_volume_scope_names(
            site_volume_metrics, sources
        )
        freshness_dim_names = {
            dim_name
            for source in sources.values()
            if source.name in freshness_source_names
            for dim_name in source.dims
        }
        dimensions = {name for source in sources.values() for name in source.dims}
        for source in sources.values():
            required_columns = {
                *source.entities,
                self._experiment.unit,
                *((self._experiment.cluster,) if self._experiment.cluster is not None else ()),
                source.timestamp_column,
                "experiment_id",
                "group_id",
                "event",
                *(property_.column for property_ in source.properties),
                *(
                    fact.column
                    for fact in source.facts
                    if fact.name in required_facts and fact.column is not None
                ),
            }
            column_projections[source.sql] = tuple(
                sorted(set(column_projections.get(source.sql, ())) | required_columns)
            )
            queries.append(source.sql)
            column_hint_candidates.setdefault(source.sql, []).append(
                self._source_column_hint(
                    source,
                    unscoped_queries=unscoped_queries,
                    site_volume_source_names=site_volume_source_names,
                    freshness_source_names=freshness_source_names,
                    property_source_names=property_source_names,
                )
            )
        dim_by_name = {d.name: d for d in self._defs.dim_sources}
        for dim_name in dimensions:
            dim = dim_by_name[dim_name]
            required_columns = {dim.entity, *(property_.column for property_ in dim.properties)}
            if dim.validity is not None:
                required_columns.update(
                    column
                    for column in (
                        dim.validity.valid_from,
                        dim.validity.valid_to,
                        dim.validity.changed_at,
                    )
                    if column is not None
                )
            column_projections[dim.sql] = tuple(
                sorted(set(column_projections.get(dim.sql, ())) | required_columns)
            )
            queries.append(dim.sql)
            hint = (
                (None, None)
                if dim_name in site_volume_dim_names or dim_name in freshness_dim_names
                else (dim.entity if dim.entity == self._experiment.unit else None, None)
            )
            column_hint_candidates.setdefault(dim.sql, []).append(hint)
        # A SQL-backed exposure can share a query with a fact or dimension
        # projection. Keep its enrollment coordinates in that physical stream.
        for sql in exposure_sql_queries & column_projections.keys():
            required_columns = {"unit_id", "ts", "group_id", "experiment_id"}
            column_projections[sql] = tuple(sorted(set(column_projections[sql]) | required_columns))
            column_hint_candidates.setdefault(sql, []).append((None, None))
        column_hints: dict[str, tuple[str | None, str | None]] = {}
        for sql, hints in column_hint_candidates.items():
            unit_col, timestamp_col = hints[0]
            column_hints[sql] = (
                unit_col if all(hint[0] == unit_col for hint in hints) else None,
                timestamp_col if all(hint[1] == timestamp_col for hint in hints) else None,
            )
        if queries:
            self._session._require_snapshot_temp_privilege()
        enrolled = self._scope_enrolled_units()
        # The generous lower margin limits only the snapshot copy, not builder
        # windows, and absorbs day-boundary edge cases. Add no upper ts bound:
        # `_data_as_of` needs later events to prove freshness; a horizon cap would
        # stop it at the last in-window event and censor every enrolled unit.
        window_start = self._experiment.start - dt.timedelta(
            days=self._experiment.n_pre_periods + 1
        )
        scope = SourceScope(enrolled_units=enrolled, window_start=window_start)
        for source in sources.values():
            self._session.preflight_fact_source_schema(
                source, require_event=source.name in event_source_names
            )

        pinned = DefinitionsMomentSource(
            self._session.pin_sources(
                tuple(queries),
                column_hints=column_hints,
                column_projections=column_projections,
                scope=scope,
                extra_relations=(
                    {}
                    if snapshot_horizon_relation is None
                    else {_SNAPSHOT_HORIZON_RELATION_KEY: snapshot_horizon_relation}
                ),
            ),
            self._experiment,
            self._metrics,
            store="none",
            on_mixed_assignment=self._on_mixed_assignment,
            backend=self._backend,
            design=self._context.design,
            plan=self._context.plan,
        )
        for source in sources.values():
            validate_fact_source_schema(
                pinned._session.source_sql(source.sql),
                source,
                require_event=source.name in event_source_names,
            )
        # Internal narrowing keeps this enclosing readout's captured horizon.
        pinned._horizon_metrics = horizon_metrics
        if configs is not None:
            pinned._context = replace(pinned._context, configs=tuple(configs))
        sequential_snapshot = getattr(self, "_sequential_snapshot", None)
        if sequential_snapshot is not None:
            pinned.adopt_sequential_snapshot(sequential_snapshot)
        self._record_declaration_schema(pinned)
        return pinned

    def _add_observational_covariate_sources(
        self, sources: dict[str, FactSource], property_source_names: set[str]
    ) -> None:
        """Pin every source an observational design's declared covariates
        resolve to, the same way a breakout property source is pinned."""
        design = self._experiment.design
        if not isinstance(design, ObservationalDeclaration):
            return
        for covariate in design.covariates:
            cov_source = self._resolve_unit_covariate_source(covariate.property)
            sources[cov_source.name] = cov_source
            property_source_names.add(cov_source.name)

    def _source_column_hint(
        self,
        source: FactSource,
        *,
        unscoped_queries: set[str],
        site_volume_source_names: set[str],
        freshness_source_names: set[str],
        property_source_names: set[str],
    ) -> tuple[str | None, str | None]:
        """The `(unit_column, timestamp_column)` hint `pin_sources` scopes
        *source* by: never for an unscoped (exposure/trigger) or site-volume
        source; a freshness-bearing source (declared/selected metrics'
        numerator/denominator) never by unit, since `_data_as_of` must see
        every unit; a property-bearing source (breakout/factor/covariate)
        never by the window's lower time bound, since it takes the latest
        value over the whole source or strictly before exposure."""
        if source.sql in unscoped_queries:
            return None, None
        if source.name in site_volume_source_names:
            return None, source.timestamp_column
        unit_col = (
            self._experiment.unit
            if self._experiment.unit in source.entities
            and source.name not in freshness_source_names
            else None
        )
        ts_col = None if source.name in property_source_names else source.timestamp_column
        return unit_col, ts_col

    @contextmanager
    def _pinned_source_execution(
        self,
        *,
        metrics: Sequence[Metric] | None = None,
        extra_sources: Sequence[str] = (),
        horizon_metrics: Sequence[Metric] = (),
        uptake_facts: Sequence[str] = (),
        site_volume_metrics: frozenset[str] = frozenset(),
        configs: Sequence[ResolvedMetricConfig] | None = None,
        snapshot_horizon_relation: Table | None = None,
        assignment_requests: Sequence[AssignmentRequest] = (),
    ):
        """Bind assignment, outcomes, freshness and extensions to one execution."""
        pinned = self._pinned_source(
            metrics=metrics,
            horizon_metrics=horizon_metrics,
            extra_sources=extra_sources,
            uptake_facts=uptake_facts,
            site_volume_metrics=site_volume_metrics,
            configs=configs,
            snapshot_horizon_relation=snapshot_horizon_relation,
            assignment_requests=assignment_requests,
        )
        try:
            yield pinned
        finally:
            if pinned is not self:
                self._adopt_assignment_validation(pinned)
                pinned.close()

    def _declared_breakout_sources(self) -> tuple[str, ...]:
        """Fact sources backing the declared breakouts, in declaration order.

        A breakout whose property cannot be resolved is skipped here: the readout that needs it
        refuses with the resolver's own coded error, so pinning must not pre-empt that refusal.
        """
        names: dict[str, None] = {}
        for breakout in self._experiment.breakouts:
            try:
                source = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
                names[source.name] = None
            except CodedError:
                continue
        return tuple(names)

    @contextmanager
    def readout_execution(self) -> Iterator[None]:
        """Share one warehouse reduction across the readout's metric and method cells."""
        with self._reduction_batch():
            yield

    @contextmanager
    def readout_snapshot(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"],
        uptake_facts: Sequence[str] = (),
        include_breakouts: bool = False,
        breakouts: Sequence[BreakoutIdentity] = (),
        configs: Sequence[ResolvedMetricConfig] | None = None,
        reuse_retained_snapshot_integrity: bool = False,
        assignment_requests: Sequence[Any] = (),
        horizon_metrics: Sequence[Metric] = (),
    ) -> Iterator[_BuiltinMomentSource]:
        """Pin selected metrics, requested uptake streams, and optionally breakout properties.

        ``include_breakouts`` also captures each declared breakout's property stream, so a
        segment or dimensioned readout of the pinned source reads the same execution snapshot
        as the headline instead of failing on, or silently re-reading, an unpinned stream.

        """
        # A finalized retained checkpoint owns both the result rows and assignment counts.
        # Reopening a live pin here would mix that checkpoint with warehouse state.
        retained = getattr(self, "_sequential_snapshot", None)
        if (
            reuse_retained_snapshot_integrity
            and retained is not None
            and retained.finalized
            and retained.assignment_counts is not None
            and self._retained_snapshot_integrity
            and self._mixed_assignment_prefix_id == retained.prefix_id
        ):
            token = _RETAINED_SNAPSHOT_INTEGRITY_SOURCES.set(
                (*_RETAINED_SNAPSHOT_INTEGRITY_SOURCES.get(), id(self))
            )
            try:
                yield self if population == "assigned" else self.triggered_source()
            finally:
                _RETAINED_SNAPSHOT_INTEGRITY_SOURCES.reset(token)
            return
        if self._session._source_tables is None:
            self._session._require_snapshot_temp_privilege()
        selected_breakouts = tuple(
            Breakout(property=item.property, source=item.source) for item in breakouts
        )
        extra_sources = (
            tuple(
                dict.fromkeys(
                    _resolve_breakout_fact_source(self._defs, self._experiment, item).name
                    for item in selected_breakouts
                )
            )
            if breakouts
            else self._declared_breakout_sources()
            if include_breakouts
            else ()
        )
        use_shared_horizon = (
            bool(metrics)
            and self._experiment.observation_horizon is None
            and self._session.source_snapshot_evidence is None
            and self._session._source_tables is None
        )
        snapshot_horizon_relation = None
        if use_shared_horizon:
            horizon = self._union_event_horizon((*horizon_metrics, *metrics))
            snapshot_horizon_relation = (
                ibis.literal(1).name("_one").as_table().select(edge=horizon.cast("date"))
            )
        with self._pinned_source_execution(
            metrics=metrics,
            horizon_metrics=horizon_metrics,
            uptake_facts=uptake_facts,
            extra_sources=extra_sources,
            configs=configs,
            snapshot_horizon_relation=snapshot_horizon_relation,
            assignment_requests=tuple(assignment_requests),
        ) as pinned:
            previous_horizon = pinned._fixed_event_horizon
            previous_override = pinned._has_fixed_event_horizon_override
            if use_shared_horizon:
                horizon_row = pinned._session.fetch_arrow(
                    pinned._session.source_sql(_SNAPSHOT_HORIZON_RELATION_KEY)
                ).to_pylist()[0]
                horizon_edge = horizon_row["edge"]
                if isinstance(horizon_edge, dt.datetime):
                    horizon_edge = horizon_edge.date()
                pinned._fixed_event_horizon = horizon_edge
                pinned._has_fixed_event_horizon_override = True
                pinned._union_horizon_cache = None
                pinned._union_horizon_metric_cache = {}
                pinned._panel_cache = {}
            try:
                yield pinned if population == "assigned" else pinned.triggered_source()
            finally:
                if use_shared_horizon:
                    pinned._fixed_event_horizon = previous_horizon
                    pinned._has_fixed_event_horizon_override = previous_override
                    pinned._union_horizon_cache = None
                    pinned._union_horizon_metric_cache = {}
                    pinned._panel_cache = {}

    def readout_view(
        self, template: DefinitionsMomentSource, *, metrics: Sequence[Metric]
    ) -> DefinitionsMomentSource:
        """Lend a family's original horizon and declarations over this pinned session."""
        import copy

        view = copy.copy(self)
        view._metrics = template._metrics
        view._context = template._context
        view._horizon_metrics = tuple(dict.fromkeys((*template._horizon_metrics, *metrics)))
        view._has_fixed_event_horizon_override = False
        view._fixed_event_horizon = None
        view._union_horizon_cache = None
        view._union_horizon_metric_cache = {}
        view._panel_cache = {}
        view._cuped_relation_cache = {}
        return view

    def exploratory_source(self, *, metrics: Sequence[Metric]) -> DefinitionsMomentSource:
        """A sibling source over the declared metrics plus *metrics*, in this session.

        Each added metric compiles the default unassigned procedure at this plan's alpha
        (:func:`increment.plan.with_unassigned_procedures`), so no declared procedure,
        family, or configuration changes. It shares this source's session, so a pinned
        snapshot serves it from the same pin; it never materializes (``store="none"``)
        and holds nothing to close.
        """
        from increment.plan import with_unassigned_procedures

        added = tuple(metrics)
        plan = with_unassigned_procedures(self._context.plan, added, design=self._context.design)
        sibling = DefinitionsMomentSource(
            self._session,
            self._experiment,
            [*self._metrics, *added],
            store="none",
            on_mixed_assignment=self._on_mixed_assignment,
            backend=self._backend,
            design=self._context.design,
            plan=plan,
        )
        sibling._horizon_metrics = (
            *self._horizon_metrics,
            *(metric for metric in added if metric not in self._horizon_metrics),
        )
        return sibling

    def _metric_primary_events(self, metric: Metric) -> Table:
        """This metric's own primary events table (numerator events for a
        ``RatioMetric``).

        Reuses ``_get_fact_table``'s cache, so calling this per declared
        metric (as ``_union_event_horizon`` does) never re-scans a fact
        source already touched by this readout.
        """
        if isinstance(metric, RatioMetric):
            num_fs, num_fact_def = _find_fact_source(self._defs, metric.numerator.fact)
            num_fact_tbl = self._get_fact_table(num_fs)
            num_value_col = _resolve_value_column(num_fs, num_fact_def)
            return metric_events(num_fact_tbl, metric, value_column=num_value_col, part="numerator")
        fs, fact_def = _find_fact_source(self._defs, metric.fact)
        fact_tbl = self._get_fact_table(fs)
        value_col = _resolve_value_column(fs, fact_def)
        return metric_events(fact_tbl, metric, value_column=value_col)

    def _union_event_horizon(self, metrics: Sequence[Metric] | None = None) -> ir.Scalar:
        """Memoized union of declared and any caller-selected metrics.

        Day-axis callers can supply an effective metric variant (including
        an ad-hoc fact) without changing this source's declared context. The
        selected streams must still extend the shared spine, or late events
        silently disappear from daily/as-of reductions.
        """
        if self._has_fixed_event_horizon_override:
            if self._fixed_event_horizon is None:
                return cast("ir.Scalar", ibis.null().cast("date"))
            return ibis.literal(self._fixed_event_horizon)
        if metrics is None:
            horizon_metrics = self._horizon_metrics
        else:
            horizon_metrics_list = list(self._horizon_metrics)
            for metric in metrics:
                if metric not in horizon_metrics_list:
                    horizon_metrics_list.append(metric)
            horizon_metrics = tuple(horizon_metrics_list)

        if horizon_metrics == self._horizon_metrics:
            if self._union_horizon_cache is None:
                self._union_horizon_cache = self._union_event_horizon_for(horizon_metrics)
            return self._union_horizon_cache

        cached = self._union_horizon_metric_cache.get(horizon_metrics)
        if cached is None:
            cached = self._union_event_horizon_for(horizon_metrics)
            self._union_horizon_metric_cache[horizon_metrics] = cached
        return cached

    def _union_event_horizon_for(self, metrics: Sequence[Metric]) -> ir.Scalar:
        """Build one event horizon for an explicit effective metric set."""
        event_tables = [self._metric_primary_events(m) for m in metrics]
        for metric in metrics:
            if isinstance(metric, RatioMetric):
                den_fs, den_fact_def = _find_fact_source(self._defs, metric.denominator.fact)
                den_tbl = self._get_fact_table(den_fs)
                den_col = _resolve_value_column(den_fs, den_fact_def)
                event_tables.append(
                    metric_events(den_tbl, metric, value_column=den_col, part="denominator")
                )
        return union_event_horizon(event_tables, self._experiment)

    @property
    def _publisher(self) -> ArtifactPublisher:
        # Built per call: bound accessors and audit counters must stay live.
        return ArtifactPublisher(
            experiment=self._experiment,
            definitions=self._defs,
            connection=self._con,
            metrics=self._metrics,
            on_mixed_assignment=self._on_mixed_assignment,
            no_data_signal=self._NO_DATA_SIGNAL,
            get_exposures=self._get_exposures,
            get_fact_table=self._get_fact_table,
            build_breakout_properties_table=self._build_breakout_properties_table,
            counts_for_population=partial(
                self._counts_for_population, operation="publish_unit_day_artifact"
            ),
            get_trigger_population=partial(
                self._get_trigger_population, operation="publish_unit_day_artifact"
            ),
            validate_cluster_labels=self._validate_cluster_labels,
            validate_mixed_assignments=self._validate_mixed_assignments,
            data_as_of=self._data_as_of,
            source_snapshot_evidence=self._session.source_snapshot_evidence,
        )

    def publish_unit_day_artifact(
        self,
        store: ArtifactStore,
        *,
        extensions: Sequence[Any] = (),
        refresh_of: UnitDayArtifactRef | None = None,
    ) -> UnitDayArtifactRef:
        """Publish an immutable unit-day artifact; failures are not atomic.
        Inspect visible manifests and verify backing relations or drop the
        affected generation before retrying; see the data model guide.
        """
        return self._publisher.publish(
            store, extensions=extensions, source=self, refresh_of=refresh_of
        )

    def _build_breakout_properties_table(
        self, fact_source: FactSource, property_name: str, exposures: Table
    ) -> Table:
        """Build a deduplicated ``(unit_id, <property_name>)`` table for a
        breakout dimension or a unit covariate.

        One row per unit. ``Property.as_of == "static"`` takes the
        latest value over the whole source; ``"pre_exposure"`` takes the
        latest value strictly before the unit's ``first_exposure_ts``, so
        a segment never conditions on a post-exposure value.
        ``"event_time"`` never reaches here - it is rejected at
        definitions-load time. *exposures* is required so the
        ``pre_exposure`` branch is always exercisable; a unit with no
        qualifying value is absent from the result and lands in the
        ``"__null__"`` bin via the caller's left join.
        """
        raw = self._get_fact_table(fact_source, require_event=False)
        prop_def = next(p for p in self._defs.properties_of(fact_source) if p.name == property_name)
        scoped_exposures = exposures if prop_def.as_of == "pre_exposure" else None
        return breakout_property_table(raw, property_name, scoped_exposures)

    def _resolve_unit_covariate_source(
        self,
        name: str,
        *,
        declared_source: str | None = None,
        kind: Literal["continuous", "categorical"] | None = None,
    ) -> FactSource:
        """Resolve a covariate by its declared source or the existing unique-property rule."""
        unit = self._experiment.unit
        design = self._experiment.design
        declared = (
            next((c for c in design.covariates if c.property == name), None)
            if isinstance(design, ObservationalDeclaration)
            else None
        )
        source = declared_source or (declared.source if declared is not None else None)
        if source is not None:
            candidates = [
                fs
                for fs in self._defs.fact_sources
                if fs.name == source
                and unit in fs.entities
                and any(p.name == name for p in self._defs.properties_of(fs))
            ]
            if not candidates:
                _refuse(
                    _NATIVE_COVARIATE_UNRESOLVED,
                    covariate=name,
                    unit=unit,
                    source=source,
                )
        else:
            candidates = [
                fs
                for fs in self._defs.fact_sources
                if unit in fs.entities and any(p.name == name for p in self._defs.properties_of(fs))
            ]
            if not candidates:
                _refuse(_NATIVE_COVARIATE_UNRESOLVED, covariate=name, unit=unit, source=None)
            if len(candidates) > 1:
                _refuse(
                    _NATIVE_COVARIATE_AMBIGUOUS,
                    covariate=name,
                    candidates=tuple(sorted(fs.name for fs in candidates)),
                )
        fact_source = candidates[0]
        prop = next(p for p in self._defs.properties_of(fact_source) if p.name == name)
        if prop.as_of == "event_time":
            _refuse(
                _NATIVE_COVARIATE_AS_OF,
                covariate=name,
                source=fact_source.name,
                as_of=prop.as_of,
            )
        if prop.dtype not in ("int", "float", "bool", "string"):
            _refuse(
                _NATIVE_COVARIATE_DTYPE,
                covariate=name,
                source=fact_source.name,
                dtype=prop.dtype,
            )
        if kind is not None:
            compatible = (
                prop.dtype == "string"
                if kind == "categorical"
                else prop.dtype
                in (
                    "int",
                    "float",
                    "bool",
                )
            )
            if not compatible:
                _refuse(
                    _NATIVE_COVARIATE_KIND,
                    covariate=name,
                    source=fact_source.name,
                    dtype=prop.dtype,
                    kind=kind,
                )

        return fact_source

    def _pre_period_measure_covariate(self, measure: Measure, name: str, exposures: Table) -> Table:
        from increment.query.builders import _days_literal, _local_date

        if self._experiment.n_pre_periods <= 0:
            _refuse(
                _NATIVE_COVARIATE_UNRESOLVED,
                covariate=name,
                unit=self._experiment.unit,
                source="n_pre_periods=0",
            )
        fact_source, fact = _find_fact_source(self._defs, measure.fact)
        measure_metric = MeanMetric(
            name=name,
            entity=self._experiment.unit,
            fact=measure.fact,
            aggregation=measure.aggregation,
            filters=measure.filters,
            window_days=measure.window_days,
        )
        events = metric_events(
            self._get_fact_table(fact_source),
            measure_metric,
            value_column=_resolve_value_column(fact_source, fact),
        )
        anchors = exposures.select(
            "unit_id",
            "first_exposure_ts",
            first_exposure_date=_local_date(exposures.first_exposure_ts, self._experiment),
        ).distinct()
        joined = events.join(anchors, events.unit_id == anchors.unit_id, rname="{name}_cuped")
        pre_start = joined.first_exposure_date - _days_literal(
            self._experiment.n_pre_periods
        ).as_interval("D")
        scoped = joined.filter(
            (_local_date(joined.ts, self._experiment) >= pre_start)
            & (joined.ts < joined.first_exposure_ts)
        )
        aggregation = measure.aggregation
        if aggregation == "sum":
            values = scoped.group_by("unit_id").agg(**{name: scoped.value.sum()})
        elif aggregation == "count":
            values = scoped.group_by("unit_id").agg(**{name: scoped.value.count().cast("float64")})
        elif aggregation == "avg_event":
            values = scoped.group_by("unit_id").agg(**{name: scoped.value.mean()})
        elif aggregation == "avg_calendar_day":
            values = scoped.group_by("unit_id").agg(
                **{
                    name: scoped.value.sum() / self._experiment.n_pre_periods,
                }
            )
        elif aggregation == "count_distinct":
            values = scoped.group_by("unit_id").agg(
                **{name: scoped.value.nunique().cast("float64")}
            )
        elif aggregation == "min":
            values = scoped.group_by("unit_id").agg(**{name: scoped.value.min()})
        else:
            values = scoped.group_by("unit_id").agg(**{name: scoped.value.max()})
        return values

    def _resolve_covariate_binding(
        self, covariate: CupedCovariate, metric: Metric
    ) -> CupedCovariate:
        if isinstance(covariate, CupedCovariateBinding):
            return covariate
        binding = self._experiment.bindings.get(metric.name)
        if binding is not None:
            methods = (
                () if binding.decision_method is None else (binding.decision_method,)
            ) + binding.sensitivity_methods
            for method in methods:
                for declared in method.covariates or ():
                    if declared.name == covariate.name and isinstance(
                        declared, CupedCovariateBinding
                    ):
                        return declared.model_copy(
                            update={"kind": covariate.kind, "missing": covariate.missing}
                        )
        return covariate

    def _bound_scalar_covariate_pre_stats(
        self,
        covariate: Any,
        exposures: Table,
        metric: Metric,
        *,
        breakout: _ScalarBreakoutPreparation | None = None,
    ) -> Table:
        covariate = self._resolve_covariate_binding(covariate, metric)
        measure = getattr(covariate, "measure", None)
        if measure is not None:
            values = self._pre_period_measure_covariate(measure, covariate.name, exposures)
        else:
            property_name = getattr(covariate, "property", None) or covariate.name
            source = self._resolve_unit_covariate_source(
                property_name,
                declared_source=getattr(covariate, "source", None),
                kind=getattr(covariate, "kind", None),
            )
            values = self._build_breakout_properties_table(source, property_name, exposures)
            if property_name != covariate.name:
                values = values.select("unit_id", **{covariate.name: values[property_name]})
        values = (
            exposures.select("unit_id")
            .distinct()
            .left_join(values, "unit_id")
            .select("unit_id", covariate.name)
        )
        value = values[covariate.name].cast("float64")
        normalized = ibis.ifelse(value.isnan(), ibis.null().cast("float64"), value)
        values = values.select("unit_id", **{covariate.name: normalized})
        if covariate.missing in ("refuse", "impute"):
            population = values
            if breakout is not None:
                population = values.left_join(breakout.properties, "unit_id")
                label = ibis.coalesce(
                    canonical_dimension_value(breakout.properties[breakout.dimension]), "__null__"
                ).name("__cuped_dimension__")
                grouped = population.group_by(label)
            else:
                grouped = population
            counts = self._session.fetch_arrow(
                grouped.aggregate(
                    n_missing=population[covariate.name].isnull().cast("int64").sum(),
                    n_observed=population[covariate.name].count(),
                )
            ).to_pylist()
            all_missing = not any(row["n_observed"] for row in counts)
            for row in counts:
                n_missing = int(row["n_missing"] or 0)
                context: dict[str, object] = {"metric": metric.name, "covariate": covariate.name}
                if covariate.missing == "refuse" and n_missing:
                    spec = _NATIVE_COVARIATE_MISSING
                    context["n_missing"] = n_missing
                elif covariate.missing == "impute" and all_missing:
                    spec = _NATIVE_COVARIATE_IMPUTE_ALL_NULL
                else:
                    continue
                if breakout is None:
                    _refuse(spec, **context)
                breakout.refusals.setdefault(metric.name, []).append(
                    (
                        breakout.dimension,
                        str(row["__cuped_dimension__"]),
                        spec,
                        _FrozenMapping(context),
                    )
                )
        return values.select("unit_id", sum_value=values[covariate.name])

    def _build_exposure_events_table(self) -> Table:
        """Build the raw exposure-events table from the experiment's exposure.

        Scoped to this experiment's ``experiment_id``: two experiments
        can share one fact-based exposure definition (e.g. both trigger
        on "first page view"), so without this filter ``first_exposures``
        would silently mix cohorts whenever an ``experiment_id`` value
        collides across experiments sharing an exposure.
        """
        return self._build_events_table_for_exposure(self._experiment.exposure)

    def _build_trigger_events_table(self) -> Table:
        """Same as :meth:`_build_exposure_events_table`, for the declared
        trigger's own exposure definition rather than the assignment one.
        """
        assert self._experiment.trigger is not None
        return self._build_events_table_for_exposure(self._experiment.trigger)

    def _build_events_table_for_exposure(
        self, exposure_name: str, *, dim_joined: bool = True
    ) -> Table:
        """Shared body: raw events for the named exposure, scoped to this
        experiment's ``experiment_id`` (see :meth:`_build_exposure_events_table`).
        """
        exposure_def = self._exp_lookup_exposures[exposure_name]
        if exposure_def.sql is not None:
            tbl = self._session.source_sql(exposure_def.sql)
            # first_exposures() requires unit_id/ts/group_id; a documented
            # (entity_id, first_exposure_ts)-only result would otherwise
            # surface as an opaque AttributeError deep inside first_exposures.
            required = frozenset({"unit_id", "ts", "group_id"})
            missing = sorted(required - set(tbl.columns))
            if missing:
                _refuse_operation(
                    operation="exposure_events",
                    request={
                        "experiment": self._experiment.name,
                        "exposure": exposure_name,
                        "required": tuple(sorted(required)),
                        "missing": tuple(missing),
                    },
                    offered=tuple(sorted(tbl.columns)),
                    route=(
                        "select unit_id, ts, and group_id (plus an optional experiment_id) "
                        "in the exposure sql; enrollment and arm assignment read them"
                    ),
                )
            if "experiment_id" in tbl.columns:
                tbl = tbl.filter(tbl.experiment_id == self._experiment.name)
            else:
                # The column is documented as optional, and downstream reductions
                # read it unconditionally, so supply it rather than crashing on a
                # query that returned exactly the required three columns.
                tbl = tbl.mutate(experiment_id=ibis.literal(self._experiment.name))
            return tbl

        # Exposure._exactly_one_source enforces sql XOR fact, so fact is
        # guaranteed non-None once sql is None.
        assert exposure_def.fact is not None
        # Fact-based exposure: check raw columns for experiment_id first. On a sourceless
        # fact table, _get_fact_table synthesizes an all-NULL one, so a post-rename check
        # would always pass and silently drop every row.
        fs, _fact = _find_fact_source(self._defs, exposure_def.fact)
        source_table = self._session.source_sql(fs.sql)
        validate_fact_source_schema(source_table, fs)
        raw_has_experiment_id = "experiment_id" in source_table.columns
        # Scoping before pinning (`dim_joined=False`) skips the dim join when the
        # source declares no dims: scoping reads no joined property, and
        # `_assemble_publication` rebuilds the joined table from the snapshot.
        # Declared dims still join because `exposure_def.filters` may read them.
        if dim_joined or fs.dims:
            tbl = self._get_fact_table(fs)
        else:
            tbl = _rename_to_builder_cols(source_table, fs, self._experiment.unit)
        tbl = tbl.filter(tbl.event == exposure_def.fact)
        if raw_has_experiment_id:
            # NULL experiment_id can mark unrelated rows sharing this fact, such as
            # pre-period events, and those stay excluded. An all-NULL column carries
            # no signal, so coalesce it like a missing column rather than enrolling
            # nobody; one windowed aggregate detects that case.
            column_unused = ~tbl.experiment_id.notnull().any()
            tbl = tbl.mutate(
                experiment_id=ibis.ifelse(column_unused, self._experiment.name, tbl.experiment_id)
            )
            tbl = tbl.filter(tbl.experiment_id == self._experiment.name)
        # Declared Exposure.filters (fact-based exposures only; validated at load
        # time) narrow enrollment.
        for flt in exposure_def.filters:
            tbl = _apply_filter(tbl, flt)
        return tbl

    def _build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        population: Literal["assigned", "triggered"] = "assigned",
        *,
        horizon_metrics: Sequence[Metric] | None = None,
        use_cache: bool = True,
    ) -> tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar]:
        """Cached wrapper over ``_build_panel_for_metric_impl``.

        The cache is keyed by the effective metric object, not just its
        name, so a same-name caller variant cannot reuse a declared panel.
        Expanded horizons use a horizon-aware key, keeping them separate
        from the default entry with its shorter spine.
        Public lazy queries bypass this operation-owned cache so subsequent
        readouts cannot invalidate their relations by dropping TEMP tables.
        """
        effective_horizon = list(self._horizon_metrics)
        for candidate in (*tuple(horizon_metrics or ()), metric):
            if candidate not in effective_horizon:
                effective_horizon.append(candidate)
        horizon_key = tuple(effective_horizon)
        if horizon_key == self._horizon_metrics:
            key: tuple[str, Metric] | tuple[str, Metric, tuple[Metric, ...]] = (
                population,
                metric,
            )
            horizon_argument = None
        else:
            key = (population, metric, horizon_key)
            horizon_argument = horizon_key
        if not use_cache:
            return self._build_panel_for_metric_impl(
                exposures,
                metric,
                population=population,
                horizon_metrics=horizon_argument,
            )
        cached = self._panel_cache.get(key)
        if cached is None:
            cached = self._build_panel_for_metric_impl(
                exposures,
                metric,
                population=population,
                horizon_metrics=horizon_argument,
            )
            self._panel_cache[key] = cached
        return cached

    def _metric_snapshot_edges(
        self, metric: Metric, evidence: SourceSnapshotEvidence
    ) -> tuple[dt.date, dt.date | None]:
        measure_facts = (
            (metric.numerator.fact, metric.denominator.fact)
            if isinstance(metric, RatioMetric)
            else (metric.fact,)
        )
        contributing_feeds = tuple(
            _find_fact_source(self._defs, fact)[0].name for fact in measure_facts
        )
        effective_edges = tuple(
            _effective_snapshot_edge(
                evidence.observation_cutoff_ts,
                evidence.complete_through_by_feed.get(feed),
                self._experiment.day_boundary_offset,
                self._experiment.observation_horizon_day,
            )[0]
            for feed in contributing_feeds
        )
        observed_edge = min(effective_edges)
        certified_edge = (
            observed_edge
            if contributing_feeds
            and all(
                evidence.complete_through_by_feed.get(feed) is not None
                for feed in contributing_feeds
            )
            else None
        )
        return observed_edge, certified_edge

    def triggered_observation_edges(self, metric: Metric) -> tuple[dt.date, dt.date | None]:
        """Return the effective event edge and optional certification edge for one metric."""
        evidence = self._require_source_snapshot_evidence(operation="triggered_observation_edges")
        return self._metric_snapshot_edges(metric, evidence)

    def _require_sitewide_outcome_completeness(self, metric: Metric, *, certified: bool) -> None:
        """Refuse outcomes known to stop before the experiment end day.

        Triggered membership needs certified feed edges; assigned readings use
        the effective snapshot edge, which already honours declared watermarks.
        """
        end_day = self._experiment.end_day
        observed_edge, certified_edge = self.triggered_observation_edges(metric)
        edge = certified_edge if certified else observed_edge
        if end_day is None or edge is None or edge < end_day:
            _refuse(
                _SITEWIDE_OUTCOMES_INCOMPLETE,
                route=_OUTCOMES_INCOMPLETE_ROUTES[certified],
            )

    def _validate_triggered_sitewide_metric(self, metric: Metric) -> None:
        components = (
            (metric.numerator, metric.denominator) if isinstance(metric, RatioMetric) else (metric,)
        )
        if self._experiment.end is None or any(
            getattr(component, "window_days", None) is not None
            or getattr(component, "aggregation", None) not in ("sum", "count")
            for component in components
        ):
            _refuse_operation(
                operation="sitewide_evidence",
                request={
                    "metric": metric.name,
                    "window_days": getattr(metric, "window_days", None),
                    "aggregation": getattr(metric, "aggregation", None),
                },
                offered={"calendar_window": "unwindowed sum or count"},
                route="declare an unwindowed sum or count metric and an experiment end aligned to the calendar window",
            )
        self._require_sitewide_outcome_completeness(metric, certified=True)

    def triggered_health_evidence(self) -> Any:
        from increment._source_operations import TriggeredHealthEvidence, TriggeredHealthFeed

        snapshot = self._session.source_snapshot_evidence
        cutoff = None if snapshot is None else snapshot.observation_cutoff_ts
        watermarks = {} if snapshot is None else snapshot.complete_through_by_feed
        feeds: dict[tuple[str, str], None] = {}

        def add_feed(name: str, evidence_kind: str) -> None:
            feeds.setdefault((name, evidence_kind), None)

        def add_fact(fact: str) -> None:
            name = _find_fact_source(self._defs, fact)[0].name
            add_feed(name, "native_feed")

        assignment = self._exp_lookup_exposures[self._experiment.exposure]
        if assignment.fact is not None:
            add_fact(assignment.fact)
        else:
            add_feed(assignment.name, "native_sql_assignment")
        trigger_name = self._experiment.trigger
        if trigger_name is not None:
            trigger = self._exp_lookup_exposures[trigger_name]
            if trigger.fact is None:
                add_feed(trigger_name, "native_sql_trigger")
            else:
                add_fact(trigger.fact)
        for metric in self._metrics:
            facts = (
                (metric.numerator.fact, metric.denominator.fact)
                if isinstance(metric, RatioMetric)
                else (metric.fact,)
            )
            for fact in facts:
                add_fact(fact)
        uptake = getattr(self._context.design, "uptake", None)
        if uptake is not None:
            add_fact(uptake.fact)
        for covariate in getattr(self._experiment.design, "covariates", ()):
            source = self._resolve_unit_covariate_source(covariate.property)
            add_feed(source.name, "native_feed")
        return TriggeredHealthEvidence(
            observation_cutoff=cutoff,
            feeds=tuple(
                TriggeredHealthFeed(
                    evidence_kind=evidence_kind,
                    feed=name,
                    metric=None,
                    complete_through=(
                        None
                        if evidence_kind in {"native_sql_assignment", "native_sql_trigger"}
                        else watermarks.get(name)
                    ),
                )
                for name, evidence_kind in feeds
            ),
        )

    def _certify_sequential_feeds(self, registration, as_of: dt.date) -> None:
        """Refuse a trigger-declared capture unless every contributing feed is certified.

        The trigger feed and every registered outcome, denominator and uptake
        feed must carry a watermark whose certified whole day reaches ``as_of``;
        an inclusive timestamp cutoff alone certifies nothing. A trigger declared
        through SQL has no feed to certify and refuses.
        """
        from increment.query.sequential_capture import certify_capture_feed
        from increment.sequential_state import sequential_refuse

        evidence = self._require_source_snapshot_evidence(operation="capture_sequential")
        trigger_name = self._experiment.trigger
        if trigger_name is None:
            return
        trigger = self._exp_lookup_exposures[trigger_name]
        if trigger.fact is None:
            sequential_refuse(
                "source.invalid",
                f"trigger {trigger.name!r} is declared through SQL, so no feed watermark can "
                "certify its completeness; declare the trigger exposure through a fact on a "
                "fact source and certify that feed",
                feed=trigger.name,
                certified_day=None,
                as_of=as_of.isoformat(),
            )
        feeds = [_find_fact_source(self._defs, trigger.fact)[0].name]
        metrics = {m.name: m for m in self.context.metrics}
        design = self.context.design
        for model in registration.models:
            if model.observable == "uptake":
                if not isinstance(design, Encouragement):
                    sequential_refuse("source.invalid", "uptake requires encouragement assignment")
                facts = (design.uptake.fact,)
            else:
                metric = metrics[model.metric]
                facts = (
                    (metric.numerator.fact, metric.denominator.fact)
                    if isinstance(metric, RatioMetric)
                    else (metric.fact,)
                )
            feeds.extend(_find_fact_source(self._defs, fact)[0].name for fact in facts)
        for feed in dict.fromkeys(feeds):
            certify_capture_feed(
                feed=feed,
                cutoff=evidence.observation_cutoff_ts,
                complete_through=evidence.complete_through_by_feed.get(feed),
                experiment=self._experiment,
                as_of=as_of,
            )

    def _after_trigger_events(self, events: Table, exposures: Table) -> Table:
        """Events strictly after each unit's first eligible trigger, up to the pinned cutoff."""
        evidence = self._require_source_snapshot_evidence(operation="triggered_outcomes")
        anchors = exposures.select("unit_id", "first_trigger_ts")
        joined = events.join(anchors, events.unit_id == anchors.unit_id).filter(
            (events.ts > anchors.first_trigger_ts)
            & (events.ts <= _utc_timestamp_literal(events.ts, evidence.observation_cutoff_ts))
        )
        return joined.select(**{name: events[name] for name in events.columns})

    def _assignment_anchored(self, cohort: Table) -> Table:
        """The triggered cohort re-anchored at assignment for its pre-period covariate.

        Values observed between assignment and the trigger never become covariates.
        """
        exposures = self._get_exposures()
        assignment = exposures.select("unit_id", assignment_anchor=exposures.first_exposure_ts)
        joined = cohort.join(assignment, "unit_id")
        return joined.select(
            **{
                name: assignment.assignment_anchor if name == "first_exposure_ts" else cohort[name]
                for name in cohort.columns
            }
        )

    def _window_close_day(self, longest_window_days: int):
        """The boundary-local day each unit's longest registered window closes."""
        window = ibis.interval(days=max(0, longest_window_days - 1))

        def final_day(table: Table):
            return _local_date(table.first_exposure_ts, self._experiment) + window

        return final_day

    def _pinned_spine_experiment(
        self, metric: Metric, evidence: SourceSnapshotEvidence
    ) -> Experiment:
        """Build the bounded dense-panel experiment for pinned evidence."""
        spine_end, _ = self._metric_snapshot_edges(metric, evidence)
        return self._experiment.model_copy(
            update={"observation_end": dt.datetime.combine(spine_end, dt.time())}
        )

    def _build_panel_for_metric_impl(
        self,
        exposures: Table,
        metric: Metric,
        population: Literal["assigned", "triggered"] = "assigned",
        *,
        horizon_metrics: Sequence[Metric] | None = None,
    ) -> tuple[Table, Table, Table, Table, Table | None, Table, str | None, ir.Scalar]:
        """Build the dense unit-day panel and the spine + sufficient-stats
        pair for one metric.

        Shared by the un-broken-out pipelines (``_panel_sql_for_metrics``,
        ``_summary_sql_for_metrics``, ``_moments_for_metrics``) and the
        breakout pipeline (``breakout_summaries``), which all
        need the same panel/spine/stats/covariate-source tables per
        metric, built independently from the same underlying events.

        Returns ``(panel, spine, stats, events, den_events, fact_tbl,
        value_col, data_as_of)``: ``events`` is the raw events table
        ``spine``/``stats`` were derived from. ``spine``'s right edge is
        ``_union_event_horizon`` over the declared metrics plus any
        caller-selected effective metric, so a metric's numeric answer never
        depends on whether this is a fused or a later materialized readout.
        ``den_events`` is the raw denominator events for a ``RatioMetric``
        (else ``None``); ``fact_tbl``/``value_col`` carry the CUPED
        pre-period covariate through. ``data_as_of`` is a lazy scalar
        (never ``None`` - an absent signal is the coalesced sentinel
        instead), the earlier of numerator/denominator freshness for a ratio
        metric.
        """
        den_events: Table | None = None
        if isinstance(metric, RatioMetric):
            num_fs, num_fact_def = _find_fact_source(self._defs, metric.numerator.fact)
            num_fact_tbl = self._get_fact_table(num_fs)
            num_value_col = _resolve_value_column(num_fs, num_fact_def)
            events = metric_events(
                num_fact_tbl, metric, value_column=num_value_col, part="numerator"
            )
            den_fs, den_fact_def = _find_fact_source(self._defs, metric.denominator.fact)
            den_fact_tbl = self._get_fact_table(den_fs)
            den_value_col = _resolve_value_column(den_fs, den_fact_def)
            den_events = metric_events(
                den_fact_tbl, metric, value_column=den_value_col, part="denominator"
            )
            fact_tbl = num_fact_tbl
            value_col = num_value_col
            data_as_of = cast(
                ir.Scalar,
                ibis.least(
                    self._data_as_of(num_fact_tbl, metric.numerator.fact),
                    self._data_as_of(den_fact_tbl, metric.denominator.fact),
                ),
            )
        else:
            fs, fact_def = _find_fact_source(self._defs, metric.fact)
            fact_tbl = self._get_fact_table(fs)
            value_col = _resolve_value_column(fs, fact_def)
            events = metric_events(fact_tbl, metric, value_column=value_col)
            data_as_of = self._data_as_of(fact_tbl, metric.fact)

        outcome_exposures = exposures
        evidence = (
            self._require_source_snapshot_evidence(operation="triggered_outcomes")
            if population == "triggered"
            else self._session.source_snapshot_evidence
        )
        if evidence is not None:
            cutoff = _utc_timestamp_literal(events.ts, evidence.observation_cutoff_ts)
            events = events.filter(events.ts <= cutoff)
            if den_events is not None:
                den_cutoff = _utc_timestamp_literal(den_events.ts, evidence.observation_cutoff_ts)
                den_events = den_events.filter(den_events.ts <= den_cutoff)

        if population == "triggered" and evidence is not None:
            events = self._after_trigger_events(events, exposures)
            if den_events is not None:
                den_events = self._after_trigger_events(den_events, exposures)
            outcome_exposures = exposures.mutate(first_exposure_ts=exposures.first_trigger_ts)
        spine_experiment = self._experiment
        spine_end: dt.date | None = None
        if evidence is not None:
            spine_experiment = self._pinned_spine_experiment(metric, evidence)
            spine_end = spine_experiment.observation_horizon_day
            if spine_end is not None:
                data_as_of = ibis.literal(spine_end)

        horizon = self._union_event_horizon(horizon_metrics)
        if spine_end is not None:
            horizon = cast("ir.Scalar", ibis.least(horizon, ibis.literal(spine_end)))
        panel = unit_day_panel(
            outcome_exposures,
            events,
            spine_experiment,
            metric_name=metric.name,
            end_date=horizon,
        )
        spine, stats = unit_day_spine_stats(
            outcome_exposures, events, spine_experiment, metric.name, end_date=horizon
        )
        return panel, spine, stats, events, den_events, fact_tbl, value_col, data_as_of

    _NO_DATA_SIGNAL = NO_DATA_SIGNAL

    def _data_as_of(self, fact_tbl: Table, fact: str) -> ir.Scalar:
        """Lazy scalar: the latest date *fact_tbl* has loaded *fact* through.

        A max(ts) lower bound on true pipeline freshness, not an exact
        watermark. Scoped to rows where ``event == fact``, since a fact
        source commonly backs several facts with different real-world
        latency, and an unfiltered max would let a fresher sibling fact
        mask a lagging one. The day is bucketed via ``_local_date``
        under the experiment's ``day_boundary`` so freshness censors in
        the same calendar the panel's day buckets live in.

        Stays lazy - composed into the same ibis expression
        ``unit_totals`` builds rather than resolved here, so
        ``panel_sql``/``summary_sql`` never trigger a warehouse
        round-trip just to compute it. Coalesced against a far-future
        sentinel so a fact with zero rows doesn't propagate SQL NULL
        through ``ibis.least`` and wrongly cap censoring to nothing.
        """
        scoped = fact_tbl.filter(fact_tbl.event == fact)
        return cast(
            "ir.Scalar",
            ibis.coalesce(
                _local_date(scoped.ts, self._experiment).max(),
                ibis.literal(self._NO_DATA_SIGNAL),
            ),
        )

    def _resolve_breakout_props(
        self, breakouts: Sequence[Breakout | Factor], exposures: Table
    ) -> list[tuple[Breakout | Factor, FactSource, Table]]:
        """Resolve and build each breakout's/factor's deduplicated properties table.

        Shared by ``_panel_sql_for_metrics``/``_summary_sql_for_metrics``,
        ``breakout_summaries``, and
        ``factor_summaries``, which all need the same (breakout, fact
        source, properties table) triples, resolved once regardless of
        which metric is being processed. ``Factor`` shares ``Breakout``'s
        ``property``/``source`` shape, so this resolver serves both.

        The resolved ``FactSource`` travels with the properties table so
        callers can qualify per-breakout keys by source name: two
        breakouts can share a ``property`` name while resolving to
        different sources, so the source name keeps their keys from
        colliding. *exposures* is required (not defaulted to ``None``)
        since every caller already holds it.
        """
        result: list[tuple[Breakout | Factor, FactSource, Table]] = []
        for b in breakouts:
            fs = _resolve_breakout_fact_source(self._defs, self._experiment, b)
            props = self._build_breakout_properties_table(fs, b.property, exposures)
            result.append((b, fs, props))
        return result

    def _build_pre_events(
        self, fact_tbl: Table, metric: Metric, value_col: str | None
    ) -> Table | None:
        """Build the CUPED pre-period covariate events table, or None.

        Shared by ``_summary_sql_for_metrics``/``_moments_for_metrics`` and
        ``breakout_summaries``, which all build the pre-period events from the same
        fact/value_column/aggregation as the outcome, gated on
        ``Experiment.n_pre_periods``.

        A ratio metric's covariate is its NUMERATOR's pre-period total:
        ``fact_tbl``/``value_col`` are the numerator's here and
        ``metric_events`` defaults to the numerator part, so the same one
        covariate feeds both component slopes (docs/guides/cuped.md).
        """
        if self._experiment.n_pre_periods > 0:
            return metric_events(fact_tbl, metric, value_column=value_col)
        return None

    def _validate_cluster_labels(self, exposures: Table, cluster: str) -> None:
        """Loud data-quality gates on the declared cluster column.

        Checks the raw (undeduplicated) admitted exposure rows for a
        conflicting label first: `first_exposures` picks one label per
        unit (argmin on ts), which would otherwise hide two distinct
        non-null labels for the same unit behind a clean-looking result.
        """
        raw = _scope_exposure_events(self._build_exposure_events_table(), self._experiment)
        raw = raw.semi_join(
            exposures.select("unit_id", "experiment_id").distinct(),
            ["unit_id", "experiment_id"],
        )
        validate_cluster_uniqueness(raw, cluster, self._experiment.name, con=self._con)
        validate_cluster_labels(
            exposures,
            cluster,
            self._experiment.name,
            enforce_purity=getattr(self._context.design, "mechanism", None) != "observational",
            con=self._con,
        )

    def _uptake_source_events(self, fact_table: Table, fact_name: str) -> Table:
        """Resolve the uptake event stream within the pinned source snapshot."""
        events = fact_table.filter(fact_table.event == fact_name)
        evidence = self._session.source_snapshot_evidence
        if evidence is not None:
            cutoff = _utc_timestamp_literal(events.ts, evidence.observation_cutoff_ts)
            events = events.filter(events.ts <= cutoff)
            feed, _ = _find_fact_source(self._defs, fact_name)
            edge, _ = _effective_snapshot_edge(
                evidence.observation_cutoff_ts,
                evidence.complete_through_by_feed.get(feed.name),
                self._experiment.day_boundary_offset,
                self._experiment.observation_horizon_day,
            )
            events = events.filter(_local_date(events.ts, self._experiment) <= ibis.literal(edge))
        return events

    def _uptake_certified_edge(self, fact_name: str) -> dt.date | None:
        evidence = self._session.source_snapshot_evidence
        if evidence is None:
            return None
        feed, _ = _find_fact_source(self._defs, fact_name)
        _, certified_edge = _effective_snapshot_edge(
            evidence.observation_cutoff_ts,
            evidence.complete_through_by_feed.get(feed.name),
            self._experiment.day_boundary_offset,
            self._experiment.observation_horizon_day,
        )
        return certified_edge

    def _resolve_uptake(
        self, cluster: str | None, *, operation: str
    ) -> tuple[Table | None, int | None]:
        """An Encouragement design's uptake fact table and window, or
        ``(None, None)``.

        Resolved once per readout and shared across every metric's
        ``unit_totals`` call; depends only on ``self._design``, never on
        which metric is being processed. A declared ``cluster`` first runs
        :meth:`_refuse_clustered_encouragement`.
        """
        from increment.semantics.design import Encouragement as _Encouragement

        _design = self._context.design
        if isinstance(_design, _Encouragement) and cluster is not None:
            self._refuse_clustered_encouragement(_design, cluster, operation=operation)
        if isinstance(_design, _Encouragement):
            uptake_fs, _uptake_fact = _find_fact_source(self._defs, _design.uptake.fact)
            uptake_fact_tbl = self._get_fact_table(uptake_fs)
            uptake_events = self._uptake_source_events(uptake_fact_tbl, _design.uptake.fact)
            return uptake_events, _design.uptake.window_days
        return None, None

    def _refuse_clustered_encouragement(
        self, design: Encouragement, cluster: str, *, operation: str
    ) -> None:
        """Refuse what a clustered LATE reduction cannot carry.

        A declared ``cluster`` leaves no per-unit CUPED covariate moments to
        adjust with, and the clustered denominator family is already spoken
        for by the LATE first stage's cluster sizes, so a ratio/quantile
        metric's own denominator has nowhere to live under this design.
        Both checks read only declarations, never warehouse state.
        """
        n_pre_periods = self._experiment.n_pre_periods
        if n_pre_periods and n_pre_periods > 0:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "design": design.mechanism,
                    "cluster": cluster,
                    "n_pre_periods": n_pre_periods,
                },
                offered={"n_pre_periods": 0},
                route=(
                    "drop the cluster declaration or n_pre_periods; the clustered LATE "
                    "reduction has no per-unit covariate moments to adjust with"
                ),
            )
        unsupported = tuple(m for m in self._metrics if m.type in ("ratio", "quantile"))
        if unsupported:
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "design": design.mechanism,
                    "cluster": cluster,
                    "metrics": tuple(m.name for m in unsupported),
                    "metric_types": tuple(m.type for m in unsupported),
                },
                offered=tuple(m.name for m in self._metrics if m.type not in ("ratio", "quantile")),
                route=(
                    "remove the ratio and quantile metrics from this clustered "
                    "encouragement experiment; the LATE first stage reads cluster sizes "
                    "from the denominator family, and LATE's Wald numerator is undefined "
                    "for a ratio metric"
                ),
            )

    def compliance_dates(self) -> Sequence[object]:
        """Enrollment spine before outcome missingness or completion filters."""
        self._validate_mixed_assignments()
        design = cast(Encouragement, self._context.design)
        uptake_fs, _ = _find_fact_source(self._defs, design.uptake.fact)
        table = self._get_fact_table(uptake_fs)
        uptake = self._uptake_source_events(table, design.uptake.fact)
        exposures = self._get_exposures()
        spine = panel_spine(
            exposures,
            self._experiment,
            end_date=compliance_event_horizon(exposures, uptake, self._experiment),
        )
        days = spine.filter(spine.ds.notnull()).select("ds").distinct().order_by("ds")
        return [row["ds"] for row in self._session.fetch_arrow(days).to_pylist()]

    def triggered_compliance_dates(self) -> Sequence[object]:
        """Dense trigger-entry history through the observed uptake horizon."""
        self._validate_trigger_capability(operation="compliance_dates")
        design = cast(Encouragement, self._context.design)
        trigger_exposures = self._get_trigger_population(operation="compliance_dates")
        trigger_days = (
            trigger_exposures.mutate(
                ds=_local_date(trigger_exposures.first_trigger_ts, self._experiment)
            )
            .filter(trigger_exposures.first_trigger_ts.notnull())
            .select("ds")
            .distinct()
        )
        trigger_bounds = (
            ibis.literal(1)
            .name("_one")
            .as_table()
            .select(
                first=trigger_days.ds.min(),
                last=trigger_days.ds.max(),
            )
        )
        bounds = self._session.fetch_arrow(trigger_bounds).to_pylist()[0]
        if bounds["first"] is None:
            return []
        assignment_exposures = self._get_exposures()
        uptake_fs, _ = _find_fact_source(self._defs, design.uptake.fact)
        uptake_table = self._get_fact_table(uptake_fs)
        uptake_events = self._uptake_source_events(uptake_table, design.uptake.fact)
        horizon = compliance_event_horizon(assignment_exposures, uptake_events, self._experiment)
        horizon_table = ibis.literal(1).name("_one").as_table().select(edge=horizon)
        horizon_day = self._session.fetch_arrow(horizon_table).to_pylist()[0]["edge"]
        snapshot_edge = None
        evidence = self._session.source_snapshot_evidence
        if evidence is not None:
            snapshot_edge, _ = _effective_snapshot_edge(
                evidence.observation_cutoff_ts,
                evidence.complete_through_by_feed.get(uptake_fs.name),
                self._experiment.day_boundary_offset,
                self._experiment.observation_horizon_day,
            )
            horizon_day = snapshot_edge if horizon_day is None else min(horizon_day, snapshot_edge)
        last = max(bounds["last"], horizon_day) if horizon_day is not None else bounds["last"]
        if snapshot_edge is not None:
            last = min(last, snapshot_edge)
        return [
            bounds["first"] + dt.timedelta(days=offset)
            for offset in range((last - bounds["first"]).days + 1)
        ]

    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> tuple[DashboardGroupData, ...]:
        """Read group aggregates, or decode the displayed retained checkpoints."""
        if checkpoints is not None:
            return _retained_dashboard_groups(
                metrics, checkpoints, self.sequential_snapshot().chain(population)
            )
        rows: list[DashboardGroupData] = []

        self._validate_mixed_assignments()
        with self._reduction_batch():
            exposures = (
                self._get_trigger_population(operation="dashboard_group_data")
                if population == "triggered"
                else self._get_exposures()
            )
            for metric in metrics:
                _panel, spine, stats, _events, den_events, _facts, _value, data_as_of = (
                    self._build_panel_for_metric(exposures, metric, population=population)
                )
                den_stats = (
                    post_exposure_stats(
                        den_events,
                        exposures,
                        source_key=f"{metric.name}:den",
                        experiment=self._experiment,
                    )
                    if den_events is not None
                    else None
                )
                raw_totals = unit_totals(
                    spine,
                    stats,
                    metric,
                    self._experiment,
                    den_stats=den_stats,
                    data_as_of=data_as_of,
                    warn_on_censoring=False,
                )
                dense = _dense_unit_days(spine, stats)
                has_observed_day, window_complete, observation_end = _observable_window_flags(
                    dense, metric, self._experiment, data_as_of
                )
                if self._experiment.observation_horizon is None:
                    observation_end = (
                        dense.ds.max()
                        .as_scalar()
                        .notnull()
                        .ifelse(observation_end, ibis.null().cast("date"))
                    )
                flagged = dense.mutate(
                    _has_observed_day=has_observed_day,
                    _window_complete=window_complete,
                    _observation_end=observation_end,
                )
                accounting = flagged.group_by("group_id").aggregate(
                    assigned=flagged.unit_id.nunique(),
                    no_observed=flagged.unit_id.nunique(where=~flagged._has_observed_day),
                    not_mature=flagged.unit_id.nunique(
                        where=flagged._has_observed_day & ~flagged._window_complete
                    ),
                    observation_end=flagged._observation_end.max(),
                )
                eligible_days = flagged.semi_join(raw_totals.select("unit_id"), "unit_id")
                event_dense = window_bound_stats(eligible_days, metric)
                if isinstance(metric, RetentionMetric):
                    event_dense = event_dense.filter(
                        event_dense.ds
                        >= event_dense.first_exposure_date + ibis.interval(days=metric.band[0])
                    )
                event_counts = event_dense.group_by("group_id").aggregate(
                    event_count=event_dense.n_events.sum()
                )
                # Normalize descriptive moments so finite means survive overflowing totals.
                # Raw unit values still supply the quantile and eligibility.
                group_window = ibis.window(group_by=["group_id"])
                value_columns = ("y", "y_den") if isinstance(metric, RatioMetric) else ("y",)
                scales = {
                    f"scale_{name}": raw_totals[name].abs().max().over(group_window)
                    for name in value_columns
                }
                scaled = raw_totals.mutate(**scales)
                normalized = scaled.mutate(
                    **{
                        name: scaled[name]
                        / (scaled[f"scale_{name}"] > 0).ifelse(scaled[f"scale_{name}"], 1.0)
                        for name in value_columns
                    }
                )
                scale_rows = scaled.group_by("group_id").aggregate(
                    **{name: scaled[name].max() for name in scales}
                )
                aggregates = group_summary(normalized).left_join(
                    scale_rows, "group_id", rname="{name}_scale"
                )
                if isinstance(metric, QuantileMetric):
                    quantiles = _dashboard_quantiles(raw_totals, metric.quantile)
                    aggregates = aggregates.left_join(quantiles, "group_id")
                result = accounting.left_join(
                    aggregates, "group_id", rname="{name}_summary"
                ).left_join(event_counts, "group_id", rname="{name}_events")
                window = (
                    metric.band
                    if isinstance(metric, RetentionMetric)
                    else (0, resolve_window_days(metric))
                )
                for record in self._session.fetch_arrow(result).to_pylist():
                    rows.append(
                        _observed_dashboard_group(metric, record, window, population=population)
                    )
        return tuple(rows)

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> ComplianceSummary:
        """Design-level uptake sufficient state off the warehouse's own
        declared uptake fact -- independent of any outcome metric's
        maturity gate, missingness, or declaration order.

        Resolves the uptake fact directly (not through :meth:`_resolve_uptake`,
        whose CUPED/ratio guards are scoped to this source's whole declared
        metric set and would spuriously fire on a metric unrelated to
        compliance). Reuses
        :func:`~increment.query.builders._attach_uptake_flag`'s
        metric-independent uptake-window attachment and
        :func:`~increment.query.builders.group_summary`'s cluster bivariate
        collapse -- the same primitives a clustered metric's own
        compliance-relevant ``x`` family already goes through.
        """
        if completed_windows_only:
            from increment._evidence_types import validate_compliance_completion

            validate_compliance_completion(design, as_of=as_of)
        if design != self._context.design:
            expected = self._context.design
            _refuse(
                _NATIVE_COMPLIANCE_DESIGN_MISMATCH,
                expected=None if expected is None else expected.model_dump(mode="json"),
                received=design.model_dump(mode="json"),
            )
        if population == "triggered":
            self._validate_trigger_capability(operation="compliance_summary")
        self._validate_mixed_assignments()
        cluster = self._context.cluster
        uptake_fs, _uptake_fact = _find_fact_source(self._defs, design.uptake.fact)
        uptake_fact_tbl = self._get_fact_table(uptake_fs)
        uptake_events = self._uptake_source_events(uptake_fact_tbl, design.uptake.fact)
        window_days = design.uptake.window_days
        triggered = population == "triggered"
        exposures = (
            self._get_trigger_population(operation="compliance_summary")
            if triggered
            else self._get_exposures()
        )
        if cluster is not None:
            self._validate_cluster_labels(exposures, cluster)
        if as_of is not None:
            membership_anchor = (
                exposures.first_trigger_ts if triggered else exposures.first_exposure_ts
            )
            exposures = exposures.filter(
                _local_date(membership_anchor, self._experiment) <= ibis.literal(as_of)
            )
            uptake_events = uptake_events.filter(
                _local_date(uptake_events.ts, self._experiment) <= ibis.literal(as_of)
            )
        uptake_exposures = exposures
        if triggered:
            uptake_exposures = self._get_exposures().semi_join(
                exposures.select("unit_id").distinct(), "unit_id"
            )
        if completed_windows_only:
            uptake_exposures = uptake_exposures.filter(
                _local_date(uptake_exposures.first_exposure_ts, self._experiment)
                + ibis.interval(days=window_days)
                <= ibis.literal(as_of)
            )
            certified_edge = self._uptake_certified_edge(design.uptake.fact)
            if certified_edge is not None:
                certified_edge_exclusive = (
                    dt.datetime.combine(certified_edge + dt.timedelta(days=1), dt.time())
                    - self._experiment.day_boundary_offset
                ).replace(tzinfo=dt.UTC)
                uptake_exposures = uptake_exposures.filter(
                    uptake_exposures.first_exposure_ts
                    <= _utc_timestamp_literal(
                        uptake_exposures.first_exposure_ts,
                        certified_edge_exclusive - dt.timedelta(days=cast(int, window_days)),
                    )
                )
            exposures = exposures.semi_join(
                uptake_exposures.select("unit_id").distinct(), "unit_id"
            )
        totals = _attach_uptake_flag(exposures, uptake_exposures, uptake_events, window_days)
        totals = totals.mutate(y=totals.d, metric=ibis.literal("uptake"))
        base_rows = self._session.fetch_arrow(
            totals.group_by("group_id").agg(
                n_units=totals.count(),
                uptake_total=totals.d.sum(),
            )
        ).to_pylist()
        rows: dict[str, dict[str, Any]] = {
            str(r["group_id"]): {
                "group_id": str(r["group_id"]),
                "n_units": int(r["n_units"]),
                "uptake_total": float(r["uptake_total"]),
            }
            for r in base_rows
        }
        if cluster is not None:
            cluster_rows = self._session.fetch_arrow(
                group_summary(totals, cluster=cluster, uptake=True)
            ).to_pylist()
            for r in cluster_rows:
                group_id = str(r["group_id"])
                rows[group_id].update(
                    {
                        name: int(r[field]) if name == "n_clusters" else float(r[field])
                        for name, field in COMPLIANCE_ARM_FROM_CLUSTER_ROW.items()
                    }
                )
        return ComplianceSummary(
            study_id=self._experiment_name,
            control_group=str(design.control_group),
            cohort=design.uptake.fact,
            window_days=design.uptake.window_days,
            one_sided=design.one_sided,
            cluster=cluster,
            as_of=as_of,
            arms=tuple(ComplianceArm(**row) for row in rows.values()),
        )

    def _preflight_breakout(self, breakout: Breakout, *, operation: str) -> FactSource:
        """Resolve and validate one breakout without touching warehouse state.

        A request is declared only when both its property and resolved
        ``FactSource`` identity match an experiment breakout. Inferred and
        explicit source forms therefore share the same declaration check.
        """
        property_name = getattr(breakout, "property", None)
        requested_source = getattr(breakout, "source", None)
        declared = tuple(b for b in self._experiment.breakouts if b.property == property_name)
        if not declared:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self,
                () if property_name is None else (property_name,),
                tuple(sorted({b.property for b in self._experiment.breakouts})),
                route="declare the breakout on the experiment or request a declared property",
            )

        declared_sources = {
            _resolve_breakout_fact_source(self._defs, self._experiment, item).name
            for item in declared
        }
        try:
            resolved = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
        except ValueError:
            if requested_source is not None:
                from increment._evidence_refusals import refuse_breakout

                refuse_breakout(
                    self,
                    () if property_name is None else (property_name,),
                    tuple(sorted(declared_sources)),
                    route="request the breakout with one of its declared sources",
                )
            raise
        if resolved.name not in declared_sources:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self,
                () if property_name is None else (property_name,),
                tuple(sorted(declared_sources)),
                route="request the breakout with one of its declared sources",
            )
        return resolved

    def _preflight_day_request(
        self,
        metrics: Sequence[Metric],
        *,
        dimension: str | None,
        include_covariate: bool,
        operation: str,
        require_uptake: bool = False,
    ) -> None:
        """Reject unsupported day evidence before any panel is built."""
        if dimension is not None:
            declared = sorted({b.property for b in self._experiment.breakouts})
            if dimension not in declared:
                from increment._evidence_refusals import refuse_breakout

                refuse_breakout(
                    self,
                    (dimension,),
                    tuple(declared),
                    route="declare the breakout on the experiment or request a declared property",
                )
        if include_covariate:
            _refuse_missing_pre_period(self._experiment, metrics, operation=operation)
        cluster = self._experiment.cluster
        if cluster is not None:
            # Clustered day-axis evidence is unsupported.  Keep the
            # encouragement-specific compatibility refusals local and pure so
            # an unsupported request never resolves the uptake fact first.
            from increment.semantics.design import Encouragement as _Encouragement

            design = self._context.design
            if isinstance(design, _Encouragement):
                self._refuse_clustered_encouragement(design, cluster, operation=operation)
            _refuse_operation(
                operation=operation,
                request={
                    "experiment": self._experiment.name,
                    "cluster": cluster,
                    "metrics": tuple(metric.name for metric in metrics),
                    "dimension": dimension,
                },
                offered=("total",),
                route="request total-grain readouts; cluster-robust inference is total-grain only",
            )
        if require_uptake:
            self._resolve_uptake(None, operation=operation)

    def _refuse_observational_quantile(self, metric: Metric) -> None:
        """An observational design has no quantile estimator on any ingress path."""
        if metric.type == "quantile" and getattr(self._context.design, "mechanism", None) == (
            "observational"
        ):
            refuse_observational_quantile(metric)

    def _refuse_quantile_metrics(self, effective: Sequence[Metric], *, operation: str) -> None:
        """A quantile has no moments representation: there is no summary
        SQL to introspect and nothing an exported moments cube could carry.
        """
        non_additive = [m.name for m in effective if m.type == "quantile"]
        if operation == "moments":
            for metric in effective:
                self._refuse_observational_quantile(metric)
        if non_additive:
            _refuse_operation(
                operation=operation,
                request={"metrics": tuple(non_additive), "metric_type": "quantile"},
                offered=tuple(m.name for m in effective if m.type != "quantile"),
                route=(
                    "estimate quantiles with run(), which reads retained per-unit rows; "
                    "a quantile has no moments representation"
                ),
            )

    def _refuse_percentile_winsorized(self, metrics: Sequence[Metric], operation: str) -> None:
        """Percentile winsorization needs an executed quantile, which the
        totals-grain breakout/factor summary paths never compute."""
        for metric in metrics:
            config = _percentile_winsorization(metric)
            if config is not None:
                _refuse_operation(
                    operation=operation,
                    request=winsorization_request_payload((metric,)),
                    offered=tuple(m.name for m in metrics if _percentile_winsorization(m) is None),
                    route=(
                        "use fixed-value winsorization (lower_value/upper_value); breakout "
                        "and factor summaries never compute an executed quantile"
                    ),
                )

    def _refuse_declared_trigger(self, operation: str) -> None:
        """Reject views that cannot preserve the declared population."""
        trigger = self._experiment.trigger
        if trigger is not None:
            _refuse_operation(
                operation=operation,
                request={"experiment": self._experiment.name, "trigger": trigger},
                offered=("assigned",),
                route=(
                    "use run() for the triggered readout, or analyze an experiment with no "
                    "declared trigger"
                ),
            )

    def _panel_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]:
        """Per-metric panel SQL, keyed by metric name, plus one dimensioned
        entry per requested breakout, keyed
        ``f"{metric.name}:{breakout.property}:{source_name}"`` (qualified by
        source name so two breakouts sharing a property but resolving to
        different sources don't collide).

        *selected*, when given, narrows the metric set this call builds
        for; ``None`` (default) uses ``self._metrics`` (every declared
        metric).
        """
        effective = self._metrics if selected is None else list(selected)
        self._refuse_quantile_metrics(effective, operation="panel_sql")
        if not effective:
            return {}
        exposures = (
            self._get_trigger_population(operation="panel_sql")
            if population == "triggered"
            else self._get_exposures()
        )
        breakout_props = self._resolve_breakout_props(breakouts or [], exposures)
        # Uptake fact (Encouragement only): unused here, but resolving it also
        # runs the cluster+CUPED/ratio-quantile refusal gate every stage shares.
        self._resolve_uptake(self._experiment.cluster, operation="panel_sql")
        sql_results: dict[str, str] = {}
        for metric in effective:
            panel, *_rest = self._build_panel_for_metric(exposures, metric, population)
            sql_results[metric.name] = ibis_to_sql(panel, dialect=self._backend or self._con.name)
            for breakout, fs, properties_table in breakout_props:
                dim_panel = join_breakout_dimension(panel, properties_table, [breakout.property])
                key = f"{metric.name}:{breakout.property}:{fs.name}"
                sql_results[key] = ibis_to_sql(dim_panel, dialect=self._backend or self._con.name)
        return sql_results

    def _summary_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]:
        """Per-metric ``group_summary`` SQL, keyed by metric name, plus one
        dimensioned entry per requested breakout (same keying as
        `_panel_sql_for_metrics`).

        *selected* narrows the metric set the same way as
        `_panel_sql_for_metrics`.
        """
        effective = self._metrics if selected is None else list(selected)
        self._refuse_quantile_metrics(effective, operation="summary_sql")
        if not effective:
            return {}
        exposures = (
            self._get_trigger_population(operation="summary_sql")
            if population == "triggered"
            else self._get_exposures()
        )
        cluster = self._experiment.cluster
        breakout_props = self._resolve_breakout_props(breakouts or [], exposures)
        uptake_events, uptake_window_days = self._resolve_uptake(cluster, operation="summary_sql")
        sql_results: dict[str, str] = {}
        for metric in effective:
            panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population, use_cache=False)
            )
            summary, _pre_stats, _den_stats, _totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=False,
            )
            sql_results[metric.name] = ibis_to_sql(summary, dialect=self._backend or self._con.name)
            for breakout, fs, properties_table in breakout_props:
                summary_by, _pre_stats_by, _den_stats_by, _totals_by = self._build_metric_summary(
                    metric,
                    exposures,
                    spine,
                    stats,
                    den_events,
                    fact_tbl,
                    value_col,
                    data_as_of,
                    cluster=None,
                    uptake_events=uptake_events,
                    uptake_window_days=uptake_window_days,
                    warn_on_censoring=False,
                    by=[breakout.property],
                    properties_table=properties_table,
                )
                key = f"{metric.name}:{breakout.property}:{fs.name}"
                sql_results[key] = ibis_to_sql(summary_by, dialect=self._backend or self._con.name)
        return sql_results

    def _moments_for_metrics(
        self,
        methods: list[Any] | None = None,
        selected: Sequence[Metric] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        # Internal flag controlling whether CUPED narrowing is applied.
        narrow_cuped: bool = False,  # noqa: FBT001, FBT002
    ) -> pa.Table:
        """Every selected metric's ``group_summary`` moments, unioned into
        one arrow table -- the shared substrate ``export()``/``moments()``/
        ``DefinitionsMomentSource.moments_source()`` all serve from.

        *methods*/*prior* are opaque here (typed loosely to keep this
        module's own imports SQL/query-only, never reaching into the
        estimator layer): forwarded verbatim to
        `_analysis_config.resolve_configs`, which types and interprets
        them, and never inspected in this module.

        *narrow_cuped* (default ``False``) narrows "want cuped" to only
        the metrics whose resolved *methods* request it, instead of the
        unconditional default (an exported moments cube must carry the
        covariate whenever ``n_pre_periods>0`` allows it). Set only by
        ``DefinitionsMomentSource.moments_source``, whose cube is consumed immediately by
        the same call that resolved *methods*.
        """
        # Import lazily because pyarrow is provided by live ibis connections.
        import pyarrow as pa

        effective = self._metrics if selected is None else list(selected)
        self._refuse_quantile_metrics(effective, operation="moments")
        if not effective:
            return pa.table({})
        configs = overlay_configs(
            effective,
            self._context.configs,
            methods=methods,
            prior=prior,
        )
        if population == "triggered":
            self._validate_trigger_capability(operation="moments")
        if not self._reduction_batch_depth:
            self._note_reduction()
        # Build exposure events once (shared across all metrics)
        exposures = (
            self._get_trigger_population(operation="moments")
            if population == "triggered"
            else self._get_exposures()
        )

        cluster = self._experiment.cluster
        if cluster is not None:
            self._validate_cluster_labels(exposures, cluster)

        # Uptake fact (Encouragement only): resolved once and shared like
        # `exposures`; the cluster+CUPED/ratio-quantile refusals live in _resolve_uptake.
        uptake_events, uptake_window_days = self._resolve_uptake(cluster, operation="moments")

        tables: list[pa.Table] = []
        for metric, config in zip(effective, configs, strict=True):
            panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population)
            )
            families = {
                method_key(method): method
                for method in effective_methods(config, design=self.context.design)
                if method.variance_reduction == "cuped"
            }
            family = next(iter(families.values())) if len(families) == 1 else None
            covariate = next(iter(family.covariates or ()), None) if family is not None else None
            pre_stats_override = (
                self._bound_scalar_covariate_pre_stats(covariate, exposures, metric)
                if covariate is not None
                else None
            )
            # Several scalar families cannot share one cube's x columns: the cube then
            # carries outcome moments only, and each family is acquired by its own request.
            want_cuped = (
                (family is not None if narrow_cuped else True) if len(families) <= 1 else False
            )
            summary, _pre_stats, _den_stats, _totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
                want_cuped=want_cuped,
                pre_stats_override=pre_stats_override,
                covariate_missing=covariate.missing if covariate is not None else None,
            )
            tables.append(self._session.fetch_arrow(summary))

        return pa.concat_tables(tables)

    def _windowed_metric_sum(
        self,
        fact_tbl: Table,
        metric: Metric,
        value_col: str | None,
        part: Literal["numerator", "denominator"],
    ) -> Table:
        """One ratio part's whole-site windowed sum over *fact_tbl*.

        Delegates to ``builders._windowed_fact_sum`` -- shared with
        ``site_volume`` -- because ``_site_volume_row`` needs two
        independently-resolved fact tables rather than one shared table.
        A missing-rows sum is a typed 0.0, never SQL NULL, matching
        ``site_volume``'s convention, so an empty window cannot reach
        ``sitewide_evidence``'s ``float(row["y"])`` as NULL.
        """
        return _windowed_fact_sum(
            fact_tbl,
            metric,
            self._experiment,
            value_column=value_col,
            part=part,
            observation_cutoff=self._site_observation_cutoff(),
        )

    def _site_observation_cutoff(self) -> dt.datetime | None:
        """The pinned snapshot's cutoff, which arm moments also respect."""
        evidence = self._session.source_snapshot_evidence
        return None if evidence is None else evidence.observation_cutoff_ts

    def _site_volume_row(self, metric: Metric) -> dict[str, Any]:
        """Whole-site ``{"metric", "y", "y_den"}`` for *metric*.

        ``site_volume`` takes a single fact table and reuses it for both
        a ratio metric's numerator and denominator, correct only when
        the two share one physical fact source. A ratio metric's
        numerator and denominator may resolve to different fact
        sources, so this always resolves them independently via two
        ``_windowed_metric_sum`` calls; a non-ratio metric has only one
        fact table and calls ``site_volume`` directly.
        """
        if isinstance(metric, RatioMetric):
            num_fs, num_fact_def = _find_fact_source(self._defs, metric.numerator.fact)
            num_fact_tbl = self._get_fact_table(num_fs)
            num_value_col = _resolve_value_column(num_fs, num_fact_def)
            den_fs, den_fact_def = _find_fact_source(self._defs, metric.denominator.fact)
            den_fact_tbl = self._get_fact_table(den_fs)
            den_value_col = _resolve_value_column(den_fs, den_fact_def)
            num_row = self._session.fetch_arrow(
                self._windowed_metric_sum(num_fact_tbl, metric, num_value_col, "numerator")
            ).to_pylist()[0]
            den_row = self._session.fetch_arrow(
                self._windowed_metric_sum(den_fact_tbl, metric, den_value_col, "denominator")
            ).to_pylist()[0]
            return {"metric": metric.name, "y": num_row["y"], "y_den": den_row["y"]}
        fs, fact_def = _find_fact_source(self._defs, metric.fact)
        fact_tbl = self._get_fact_table(fs)
        value_col = _resolve_value_column(fs, fact_def)
        volume = site_volume(
            fact_tbl,
            metric,
            self._experiment,
            value_column=value_col,
            observation_cutoff=self._site_observation_cutoff(),
        )
        return self._session.fetch_arrow(volume).to_pylist()[0]

    def sitewide_evidence(
        self,
        metric: Metric,
        *,
        include_ratio: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> SitewideEvidence:
        """Prepare calendar-window evidence for assigned or triggered units."""
        if isinstance(metric, RatioMetric) and not include_ratio:
            _refuse_operation(
                operation="sitewide_evidence",
                request={
                    "metric": metric.name,
                    "metric_type": metric.type,
                    "include_ratio": include_ratio,
                },
                offered={"include_ratio": True},
                route=(
                    "request the ratio reducer with include_ratio=True; the named "
                    "evidence carries one site total"
                ),
            )
        winsorization = getattr(metric, "winsorization", None)
        if winsorization is not None:
            _refuse_operation(
                operation="sitewide_evidence",
                request=winsorization_request_payload((metric,)),
                offered={"winsorization": None},
                route=(
                    "use the unwinsorized metric; whole-site outcomes cannot reuse the "
                    "per-unit transformed experiment metric"
                ),
            )
        if population == "triggered":
            self._validate_triggered_sitewide_metric(metric)
            self._validate_trigger_capability(operation="sitewide_evidence")
            self._validate_mixed_assignments()
        else:
            if (
                self._session.source_snapshot_evidence is not None
                and self._experiment.end_day is not None
            ):
                self._require_sitewide_outcome_completeness(metric, certified=False)
            self._note_reduction()
        exposures = self._get_exposures()
        if population == "triggered":
            members = self._get_trigger_population(operation="sitewide_evidence")
            exposures = exposures.join(members.select("unit_id"), "unit_id")
        cluster = self._experiment.cluster
        cluster_counts: dict[str, int] | None = None
        unit_counts: dict[str, int] | None = None
        if cluster is not None:
            self._validate_cluster_labels(exposures, cluster)
            count_rows = self._session.fetch_arrow(
                cluster_exposure_counts(exposures, self._experiment)
            ).to_pylist()
            cluster_counts = {str(r["group_id"]): int(r["n_clusters"]) for r in count_rows}
            unit_counts = {str(r["group_id"]): int(r["n_units"]) for r in count_rows}
        uptake_events, uptake_window_days = self._resolve_uptake(
            cluster, operation="sitewide_evidence"
        )
        original_experiment = self._experiment
        if self._experiment.end is not None:
            self._experiment = self._experiment.model_copy(
                update={"observation_end": self._experiment.end}
            )
        try:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, use_cache=False)
            )
            summary, _pre_stats, _den_stats, _totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
            )
        finally:
            self._experiment = original_experiment
        from increment.estimation.engine import _df_to_arms

        row = self._site_volume_row(metric)
        arm_stats = tuple(_df_to_arms(self._session.fetch_arrow(summary)))
        assigned_counts: dict[str, int] = {}
        all_exposures = self._get_exposures()
        count_rows = self._session.fetch_arrow(
            all_exposures.group_by("group_id").aggregate(n_units=all_exposures.unit_id.nunique())
        ).to_pylist()
        for count_row in count_rows:
            group_id = count_row["group_id"]
            if group_id is not None:
                assigned_counts[str(group_id)] = int(count_row["n_units"])
        population_counts: dict[str, int] = {}
        population_count_rows = self._session.fetch_arrow(
            exposures.group_by("group_id").aggregate(n_units=exposures.unit_id.nunique())
        ).to_pylist()
        for count_row in population_count_rows:
            group_id = count_row["group_id"]
            if group_id is not None:
                population_counts[str(group_id)] = int(count_row["n_units"])
        for group_id in assigned_counts:
            population_counts.setdefault(group_id, 0)
        snapshot = self._session.source_snapshot_evidence
        snapshot_id = None
        if snapshot is not None:
            from hashlib import sha256

            from increment._canonical import canonical_json_bytes

            snapshot_identity = {
                "observation_cutoff_ts": snapshot.observation_cutoff_ts.isoformat(),
                "complete_through_by_feed": {
                    feed: None if instant is None else instant.isoformat()
                    for feed, instant in sorted(snapshot.complete_through_by_feed.items())
                },
            }
            snapshot_id = "sha256:" + sha256(canonical_json_bytes(snapshot_identity)).hexdigest()
        return SitewideEvidence(
            metric=metric,
            site_total=float(row["y"]),
            arm_stats=arm_stats,
            control_group=self._experiment.control_group,
            cluster=cluster,
            site_total_denominator=(
                float(row["y_den"]) if isinstance(metric, RatioMetric) else None
            ),
            cluster_counts=cluster_counts,
            unit_counts=unit_counts,
            analysis_population=population,
            window=(
                None
                if self._experiment.end is None
                else (
                    self._experiment.start_day,
                    self._experiment.end_day or self._experiment.start_day,
                )
            ),
            membership_cutoff=(
                None
                if population == "assigned" or snapshot is None
                else snapshot.observation_cutoff_ts
            ),
            snapshot_id=snapshot_id,
            population_counts=population_counts,
            assigned_counts=assigned_counts,
        )

    def _day_axis_source(
        self,
        *,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        _legacy_facade: bool = True,
    ) -> _DaySource:
        """This experiment's day-axis panels, served as a `MomentSource`."""
        if population == "triggered":
            self._validate_trigger_capability(operation="day_axis_source")
        if selected is None:
            context = self._context
        else:
            selected_metrics = tuple(selected)
            configs_by_name = {config.metric.name: config for config in self._context.configs}
            missing = [metric for metric in selected_metrics if metric.name not in configs_by_name]
            if missing:
                configs_by_name.update(
                    {
                        config.metric.name: config
                        for config in resolve_configs(
                            missing,
                            bindings=None,
                            specs=None,
                            methods=None,
                            prior=None,
                        )
                    }
                )
            procedures = {
                name: procedure
                for name, procedure in self._context.plan.procedures.items()
                if name in {metric.name for metric in selected_metrics}
            }
            if missing:
                from increment.plan import with_unassigned_procedures

                defaults = with_unassigned_procedures(self._context.plan, missing, design=None)
                procedures.update(
                    {metric.name: defaults.procedures[metric.name] for metric in missing}
                )
            context_plan = type(self._context.plan)(
                declared=self._context.plan.declared,
                alpha=self._context.plan.alpha,
                q=self._context.plan.q,
                path=self._context.plan.path,
                inference=self._context.plan.inference,
                procedures=procedures,
                view_policies=self._context.plan.view_policies,
            )
            context = replace(
                self._context,
                metrics=selected_metrics,
                configs=tuple(configs_by_name[metric.name] for metric in selected_metrics),
                plan=context_plan,
            )
        selected_metrics = tuple(self._metrics if selected is None else selected)
        horizon_metrics = list(self._horizon_metrics)
        for metric in selected_metrics:
            if metric not in horizon_metrics:
                horizon_metrics.append(metric)
        effective_horizon = tuple(horizon_metrics)

        def build_panel(exposures: Table, metric: Metric, horizon_metrics: tuple[Metric, ...]):
            return self._build_panel_for_metric(
                exposures,
                metric,
                population=population,
                horizon_metrics=horizon_metrics,
            )

        def build_asof_panel(
            exposures: Table,
            metric: Metric,
            horizon_metrics: tuple[Metric, ...],
            include_covariate: bool = False,  # noqa: FBT001, FBT002
            include_uptake_horizon: bool = False,  # noqa: FBT001, FBT002
        ):
            return self._build_asof_panels(
                exposures,
                metric,
                horizon_metrics=horizon_metrics,
                include_covariate=include_covariate,
                include_uptake_horizon=include_uptake_horizon,
                population=population,
            )

        exposures = (
            (lambda: self._get_trigger_population(operation="day_axis_source"))
            if population == "triggered"
            else self._get_exposures
        )
        return _DaySource(
            self._experiment,
            context,
            exposures,
            build_panel,
            build_asof_panel,
            breakouts=tuple(sorted({b.property for b in self._experiment.breakouts})),
            triggered_observation_edges=self.triggered_observation_edges,
            build_breakout_properties=self._build_breakout_properties_table,
            fetch_arrow=self._session.fetch_arrow,
            defs=self._defs,
            preflight_breakout=self._preflight_breakout,
            legacy_facade=_legacy_facade,
            preflight_day_request=self._preflight_day_request,
            note_reduction=self._note_reduction,
            horizon_metrics=effective_horizon,
            population=population,
            compliance_provider=self.compliance_summary,
            compliance_dates_provider=self.compliance_dates,
            validated_assignments=self._validated_assignments,
        )

    def _breakout_moments_source(
        self,
        breakout: Breakout,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
        configs: Sequence[ResolvedMetricConfig] | None = None,
    ) -> BreakoutMomentsSource:
        """One declared `Breakout`'s totals-grain moments, served as a
        `MomentSource` scoped to exactly this breakout.

        Reuses the same ``pre_stats``/``den_stats`` construction
        ``breakout_summaries()`` performs per metric, so CUPED stays
        correct by construction. Builds only the totals-grain half
        ``breakout_summaries()`` also builds, skipping its day-axis half
        entirely since ``run_breakout`` never reads it.

        Passes ``cluster=None`` to ``_build_metric_summary``, matching
        ``breakout_summaries()``'s own totals-grain call; ``ratio_metrics=``
        still flows through underneath but stays inert at this grain, since
        ``group_summary`` only reads it under a declared ``cluster``. Under
        the declared uptake fact is joined at unit grain before this
        breakout-scoped reduction so its additive moments stay available.
        """
        from increment._analysis_config import validate_scalar_route

        configs = self._context.configs if configs is None else configs
        names = {metric.name for metric in metrics}
        validate_scalar_route(
            tuple(config for config in configs if config.metric.name in names),
            route="native.breakout_source",
            representation="breakout",
            design=self.context.design,
        )
        exposures = (
            self._get_trigger_population(operation="breakout_moments")
            if population == "triggered"
            else self._get_exposures()
        )
        breakout_source = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
        properties_table = self._build_breakout_properties_table(
            breakout_source,
            breakout.property,
            exposures,
        )
        by = [breakout.property]
        effective_horizon = list(self._horizon_metrics)
        for requested in metrics:
            if requested not in effective_horizon:
                effective_horizon.append(requested)
        effective_horizon = tuple(effective_horizon)
        uptake_events, uptake_window_days = self._resolve_uptake(
            self._experiment.cluster, operation="breakout_moments"
        )

        rows: list[dict[str, object]] = []
        scalar_preparation = _ScalarBreakoutPreparation(breakout.property, properties_table, {})
        for metric in metrics:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(
                    exposures,
                    metric,
                    horizon_metrics=effective_horizon,
                    population=population,
                )
            )
            config = next(item for item in configs if item.metric.name == metric.name)
            scalar_covariate = next(
                (
                    covariate
                    for method in effective_methods(config, design=self.context.design)
                    if method.variance_reduction == "cuped"
                    for covariate in method.covariates or ()
                ),
                None,
            )
            pre_stats_override = (
                self._bound_scalar_covariate_pre_stats(
                    scalar_covariate, exposures, metric, breakout=scalar_preparation
                )
                if scalar_covariate is not None
                else None
            )
            summary_by, _pre_stats, _den_stats, _totals_by = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=None,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
                by=by,
                properties_table=properties_table,
                want_cuped=any(
                    method.variance_reduction == "cuped"
                    for method in effective_methods(config, design=self.context.design)
                ),
                pre_stats_override=pre_stats_override,
                covariate_missing=(
                    getattr(scalar_covariate, "missing", None)
                    if scalar_covariate is not None
                    else None
                ),
            )
            # Row dicts, never pa.concat_tables: differently winsorized metrics get
            # different winsor-column dtypes (null vs double), which it rejects.
            rows.extend(self._session.fetch_arrow(summary_by).to_pylist())

        return BreakoutMomentsSource(
            rows,
            dimension=breakout.property,
            metrics=list(metrics),
            study_id=self._experiment_name,
            source_name=breakout_source.name,
            design=self._context.design,
            plan=self._context.plan,
            configs=configs,
            _source=self,
            population=population,
            scalar_refusals=scalar_preparation.refusals,
        )

    def breakout_summaries(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, dict[str, pa.Table]]:
        """Build totals and activity-day or retention-cohort breakout summaries."""

        from increment.breakout.estimates import reject_retention_metrics

        breakouts = self._experiment.breakouts

        if not breakouts or not metrics:
            return {}
        self._refuse_quantile_metrics(metrics, operation="breakout_summaries")
        self._refuse_percentile_winsorized(metrics, "breakout_summaries")
        reject_retention_metrics(metrics, "breakout_summaries", view="cohort")
        self.validate_populations((population,), operation="breakout_summaries")
        self._note_reduction()
        exposures = (
            self._get_trigger_population(operation="breakout_summaries")
            if population == "triggered"
            else self._get_exposures()
        )
        breakout_props = self._resolve_breakout_props(breakouts, exposures)
        effective_horizon = list(self._horizon_metrics)
        for requested in metrics:
            if requested not in effective_horizon:
                effective_horizon.append(requested)
        effective_horizon = tuple(effective_horizon)
        uptake_events, uptake_window_days = self._resolve_uptake(
            self._experiment.cluster, operation="breakout_summaries"
        )
        results: dict[str, dict[str, pa.Table]] = {}
        for metric in metrics:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(
                    exposures,
                    metric,
                    horizon_metrics=effective_horizon,
                    population=population,
                )
            )
            den_panel: Table | None = None
            daily_panel: Table | None = None
            den_end_date = self._union_event_horizon(effective_horizon)
            den_spine_experiment = self._experiment
            spine_evidence = self._session.source_snapshot_evidence
            if population == "triggered":
                spine_evidence = self._require_source_snapshot_evidence(
                    operation="breakout_summaries"
                )
            if spine_evidence is not None:
                den_spine_experiment = self._pinned_spine_experiment(metric, spine_evidence)
                pinned_end = den_spine_experiment.observation_end
                assert pinned_end is not None
                den_end_date = cast(
                    "ir.Scalar", ibis.least(den_end_date, ibis.literal(pinned_end.date()))
                )
            if not isinstance(metric, RetentionMetric):
                if den_events is not None:
                    den_exposures = exposures
                    if population == "triggered":
                        den_exposures = exposures.mutate(
                            first_exposure_ts=exposures.first_trigger_ts
                        )
                    den_panel = unit_day_panel(
                        den_exposures,
                        den_events,
                        den_spine_experiment,
                        metric_name=metric.name,
                        end_date=den_end_date,
                    )
                    den_panel = window_bound_stats(den_panel, metric)
                daily_panel = window_bound_stats(_panel, metric)
            for breakout, fs, properties_table in breakout_props:
                by = [breakout.property]
                summary_by, pre_stats_by, _den_stats_by, _totals_by = self._build_metric_summary(
                    metric,
                    exposures,
                    spine,
                    stats,
                    den_events,
                    fact_tbl,
                    value_col,
                    data_as_of,
                    cluster=None,
                    uptake_events=uptake_events,
                    uptake_window_days=uptake_window_days,
                    warn_on_censoring=True,
                    by=by,
                    properties_table=properties_table,
                )
                if isinstance(metric, RetentionMetric):
                    daily_by = cohort_group_summary(
                        spine,
                        stats,
                        metric,
                        self._experiment,
                        pre_stats=pre_stats_by,
                        by=by,
                        properties_table=properties_table,
                        data_as_of=data_as_of,
                        warn_on_censoring=False,
                    )
                else:
                    assert daily_panel is not None
                    dim_panel = join_breakout_dimension(daily_panel, properties_table, by)
                    daily_by = daily_group_summary(
                        dim_panel, metric=metric, by=by, den_panel=den_panel
                    )
                key = f"{metric.name}:{breakout.property}:{fs.name}"
                results[key] = {
                    "group_summary": self._session.fetch_arrow(summary_by),
                    "daily_group_summary": self._session.fetch_arrow(daily_by),
                }
        return results

    def factor_summaries(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, pa.Table]:
        """Build whole-window summaries for every declared absorption factor."""
        factors = self._experiment.factors
        if not factors or not metrics:
            return {}
        self._refuse_quantile_metrics(metrics, operation="factor_summaries")
        self._refuse_percentile_winsorized(metrics, "factor_summaries")
        self.validate_populations((population,), operation="factor_summaries")
        self._note_reduction()
        exposures = (
            self._get_trigger_population(operation="factor_summaries")
            if population == "triggered"
            else self._get_exposures()
        )
        factor_props = self._resolve_breakout_props(factors, exposures)
        uptake_events, uptake_window_days = self._resolve_uptake(
            self._experiment.cluster, operation="factor_summaries"
        )
        results: dict[str, pa.Table] = {}
        for metric in metrics:
            _panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population=population)
            )
            for factor, fs, properties_table in factor_props:
                by = [factor.property]
                summary_by, _pre_stats_by, _den_stats_by, _totals_by = self._build_metric_summary(
                    metric,
                    exposures,
                    spine,
                    stats,
                    den_events,
                    fact_tbl,
                    value_col,
                    data_as_of,
                    cluster=None,
                    uptake_events=uptake_events,
                    uptake_window_days=uptake_window_days,
                    warn_on_censoring=True,
                    by=by,
                    properties_table=properties_table,
                )
                key = f"{metric.name}:{factor.property}:{fs.name}"
                results[key] = self._session.fetch_arrow(summary_by)
        return results

    def assigned_breakout_values(
        self, breakout: Breakout, *, source_name: str | None = None
    ) -> Sequence[str]:
        """Return dimension values among assigned units without reducing outcomes."""
        fact_source = self._preflight_breakout(breakout, operation="assigned_breakout_values")
        if source_name is not None and fact_source.name != source_name:
            return []
        exposures = self._get_exposures()
        properties = self._build_breakout_properties_table(
            fact_source, breakout.property, exposures
        )
        joined = join_breakout_dimension(exposures, properties, [breakout.property])
        values = self._session.fetch_arrow(joined.select(breakout.property).distinct()).to_pylist()
        return sorted(
            {str(row[breakout.property]) for row in values if row[breakout.property] is not None}
        )

    def breakout_source(
        self,
        breakout: Breakout,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
        configs: Sequence[ResolvedMetricConfig] | None = None,
    ) -> BreakoutMomentsSource:
        """Return one breakout-scoped moments source."""
        self._refuse_quantile_metrics(metrics, operation="breakout_source")
        self._preflight_breakout(breakout, operation="breakout_source")
        with self._reduction_batch():
            return self._breakout_moments_source(
                breakout, metrics=metrics, population=population, configs=configs
            )

    def breakout_sources(
        self,
        breakouts: Sequence[Breakout],
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
        configs: Sequence[ResolvedMetricConfig] | None = None,
    ) -> tuple[BreakoutMomentsSource, ...]:
        """Return every breakout view in one reduction batch."""
        self._refuse_quantile_metrics(metrics, operation="breakout_sources")
        ordered = tuple(breakouts)
        for breakout in ordered:
            self._preflight_breakout(breakout, operation="breakout_sources")
        with self._reduction_batch():
            return tuple(
                self._breakout_moments_source(
                    breakout, metrics=metrics, population=population, configs=configs
                )
                for breakout in ordered
            )

    def day_source(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> DayEvidenceSource:
        """Return the source-owned daily/as-of evidence view."""
        if population == "triggered":
            self._validate_trigger_capability(operation="day_source")
        self._refuse_quantile_metrics(metrics, operation="day_source")
        for metric in metrics:
            _refuse_unsupported_day_metric(metric, operation="day_source")
        return self._day_axis_source(selected=metrics, population=population, _legacy_facade=False)

    def _build_asof_panels(
        self,
        exposures: Table,
        metric: Metric,
        *,
        horizon_metrics: Sequence[Metric] | None = None,
        include_covariate: bool = False,
        include_uptake_horizon: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> tuple[
        Table,
        Table | None,
        Table | None,
        int | None,
        ir.Scalar | None,
        dt.date | None,
    ]:
        """Build outcome/uptake panels, the uptake window and outcome coverage edge.

        Shared by ``run_asof`` and ``run_asof_lift``, both un-dimensioned
        and per-segment: all four paths need the identical
        ``(panel, den_panel)`` pair and deliberately omit
        ``window_bound_stats`` on either side, since
        ``asof_group_summary`` does its own per-unit window masking -
        binding first would filter post-window rows away before the
        builder ever saw them.

        ``include_covariate`` attaches one fixed, zero-filled pre-period
        total per unit before the as-of reduction. It is requested only by
        CUPED lift readouts; value-only paths leave the panel unchanged.

        ``den_panel`` is ``None`` for every non-ratio metric. On the
        per-segment paths it is never dimension-joined: the builder
        joins it to the already-dimensioned numerator panel at
        ``(unit_id, ds)`` and reads every grouping column off that
        side, so joining the dimension onto the denominator too would
        produce a column nothing reads.

        ``uptake_panel``/``uptake_window_days`` mirror
        ``_summary_sql_for_metrics``/``_moments_for_metrics``'s own uptake resolution and are built
        whenever ``self._design`` is an ``Encouragement`` design,
        regardless of caller. ``uptake_panel`` is dense, like
        ``den_panel``; both are ``None`` for every non-``Encouragement``
        design, which ``asof_group_summary`` reads as "no uptake data,
        leave sum_d/cyd/cy2d null". ``run_asof`` ignores both return
        values; ``run_asof_lift`` passes them straight through.
        """
        (
            panel,
            _spine,
            stats,
            _events,
            den_events,
            fact_tbl,
            value_col,
            _data_as_of,
        ) = self._build_panel_for_metric(
            exposures,
            metric,
            population=population,
            horizon_metrics=horizon_metrics,
        )
        outcome_exposures = exposures
        if population == "triggered":
            outcome_exposures = exposures.mutate(
                __uptake_first_exposure_date=_local_date(
                    exposures.first_exposure_ts, self._experiment
                ),
                __uptake_first_exposure_ts=exposures.first_exposure_ts,
                first_exposure_ts=exposures.first_trigger_ts,
            )

        horizon = self._union_event_horizon(horizon_metrics)
        evidence = (
            self._require_source_snapshot_evidence(operation="triggered_outcomes")
            if population == "triggered"
            else self._session.source_snapshot_evidence
        )
        spine_experiment = self._experiment
        pinned_edge: dt.date | None = None
        outcome_edge: ir.Scalar | None = None
        if evidence is not None:
            spine_experiment = self._pinned_spine_experiment(metric, evidence)
            pinned_edge = spine_experiment.observation_horizon_day
            if pinned_edge is not None:
                horizon = ibis.literal(pinned_edge)
                outcome_edge = horizon
                spine = panel_spine(outcome_exposures, spine_experiment, end_date=horizon)
                panel = _dense_unit_days(spine.filter(spine.ds.notnull()), stats).mutate(
                    metric=ibis.literal(metric.name)
                )
        uptake_panel: Table | None = None
        uptake_window_days: int | None = None
        uptake_certified_edge: dt.date | None = None
        _design = self._context.design
        if isinstance(_design, Encouragement):
            uptake_certified_edge = self._uptake_certified_edge(_design.uptake.fact)
        if isinstance(_design, Encouragement):
            uptake_fs, _uptake_fact = _find_fact_source(self._defs, _design.uptake.fact)
            uptake_fact_tbl = self._get_fact_table(uptake_fs)
            # unit_day_panel needs metric_events's (unit_id, ts, metric, value) shape;
            # a synthetic ConversionMetric lets metric_events filter and stamp rows.
            uptake_events = metric_events(
                self._uptake_source_events(uptake_fact_tbl, _design.uptake.fact),
                ConversionMetric(
                    name="__uptake", entity=self._experiment.unit, fact=_design.uptake.fact
                ),
            )
            if (
                include_uptake_horizon
                and evidence is None
                and self._experiment.observation_horizon is None
            ):
                outcome_edge = horizon
                uptake_edge = compliance_event_horizon(exposures, uptake_events, self._experiment)
                horizon = cast(
                    "ir.Scalar",
                    ibis.coalesce(ibis.greatest(horizon, uptake_edge), horizon, uptake_edge),
                )
                spine = panel_spine(outcome_exposures, spine_experiment, end_date=horizon)
                panel = _dense_unit_days(spine.filter(spine.ds.notnull()), stats).mutate(
                    metric=ibis.literal(metric.name)
                )
            uptake_events = _uptake_events_in_elapsed_window(
                exposures, uptake_events, _design.uptake.window_days
            )
            uptake_panel = unit_day_panel(
                exposures,
                uptake_events,
                spine_experiment,
                metric_name="__uptake",
                end_date=horizon,
                include_exposure=True,
            )
            uptake_window_days = _design.uptake.window_days
        if include_covariate and self._experiment.n_pre_periods > 0:
            pre_events = self._build_pre_events(fact_tbl, metric, value_col)
            if pre_events is not None:
                pre_stats = pre_period_stats(
                    pre_events,
                    exposures,
                    self._experiment,
                    source_key=f"{metric.name}:pre",
                )
                panel = join_pre_period_covariate(panel, pre_stats)
        den_panel: Table | None = None
        if den_events is not None:
            den_panel = unit_day_panel(
                outcome_exposures,
                den_events,
                self._experiment,
                metric_name=metric.name,
                end_date=horizon,
            )

        return (
            panel,
            den_panel,
            uptake_panel,
            uptake_window_days,
            outcome_edge,
            uptake_certified_edge,
        )

    def triggered_source(self) -> _BuiltinMomentSource:
        """Return a source-owned view of the triggered population."""
        return TriggeredPopulationSource(self)

    def _compliance_wire_payload(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> str | None:
        """JSON ``compliance_summary()`` payload for the moments-cube wire
        format (mirrors ``ASSIGNMENT_COUNTS_FIELD``/``DECISION_PLAN_FIELD``),
        or ``None`` when the design has no declared uptake cohort (only an
        Encouragement design does). Clustered payloads retain the cluster
        identity, member counts, and full bivariate uptake/size moments.
        """
        from increment.semantics.design import Encouragement as _Encouragement

        design = self._context.design
        if not isinstance(design, _Encouragement):
            return None
        summary = self.compliance_summary(design, population=population)
        return json.dumps(summary.to_wire(), sort_keys=True, separators=(",", ":"))

    def moments_source(
        self,
        *,
        metrics: Sequence[Metric],
        methods: list[Any] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        narrow_cuped: bool = False,
    ) -> _BuiltinMomentSource:
        """Build an in-process moments source for *metrics*.

        This is the source-owned counterpart to the export/from-moments wire
        boundary. The cube is reduced once, stamped with the same format and
        source-level assignment counts as an export, then rehydrated through
        :class:`MomentsSource` without exposing warehouse builders or caches.
        Several scalar families of one metric travel as format-13 envelopes;
        mixed requests stay with this source, which serves their unit evidence.
        """
        cluster = self._context.cluster
        if cluster is not None and not isinstance(self._context.design, Encouragement):
            from increment.sources import refuse_cluster_grain_transport

            refuse_cluster_grain_transport(
                operation="moments_source",
                source="native",
                cluster=cluster,
                design=self._context.design,
                route_forward=(
                    "analyse the clustered source directly; direct clustered inference "
                    "preserves cluster-grain degrees of freedom and counts"
                ),
            )
        import pyarrow as pa

        from increment._source_identity import source_identity
        from increment.decision_wire import compiled_plan_to_json
        from increment.sources import ASSIGNMENT_COUNTS_FIELD, MOMENTS_FORMAT

        identity_record = source_identity(self)
        identity_payload = json.dumps(identity_record, sort_keys=True, separators=(",", ":"))
        selected = list(metrics)
        requests = portable_requests(
            overlay_configs(selected, self._context.configs, methods=methods, prior=prior),
            design=self._context.design,
            categorical=False,
        )
        if requests is not None:
            from increment.sources import stamp_fixed_rows

            view = self if population == "assigned" else self.triggered_source()
            portable = portable_cuped_rows(
                cast(_BuiltinMomentSource, view),
                requests,
                categorical_cells=None,
                identity=identity_record,
            )
            _grain, assignment_counts, unit_counts = self._counts_for_population(
                population, operation="moments_source"
            )
            assignment_counts.update(unit_counts)
            rows = stamp_fixed_rows(
                portable,
                version=CUPED_EVIDENCE_FORMAT,
                identity=identity_payload,
                counts=json.dumps(assignment_counts, sort_keys=True, separators=(",", ":")),
                compliance=self._compliance_wire_payload(population=population),
            )
            return self._rehydrated_moments(rows, identity_record, view)
        cube = self._moments_for_metrics(
            methods=methods,
            selected=selected,
            prior=prior,
            population=population,
            narrow_cuped=narrow_cuped,
        )
        cube = cube.append_column(
            "moments_format",
            pa.array([MOMENTS_FORMAT] * cube.num_rows, type=pa.int64()),
        )
        _grain, assignment_counts, unit_counts = self._counts_for_population(
            population, operation="moments_source"
        )
        assignment_counts.update(unit_counts)
        cube = cube.append_column(
            ASSIGNMENT_COUNTS_FIELD,
            pa.array(
                [json.dumps(assignment_counts, sort_keys=True, separators=(",", ":"))]
                * cube.num_rows,
                type=pa.string(),
            ),
        )
        compliance_payload = self._compliance_wire_payload(population=population)
        if compliance_payload is not None:
            cube = cube.append_column(
                COMPLIANCE_SUMMARY_FIELD,
                pa.array([compliance_payload] * cube.num_rows, type=pa.string()),
            )
        if cube.num_rows == 0 and not self._metrics:
            from increment.sources import MomentsSource, _design_summary_row

            row = _design_summary_row(
                self._experiment_name,
                compiled_plan_to_json(self._context.plan),
                json.dumps(assignment_counts, sort_keys=True, separators=(",", ":")),
                compliance_payload,
                self.context.trigger_name,
                identity_payload,
            )
            return MomentsSource(
                [row],
                metrics=self._metrics,
                study_id=self._experiment_name,
                bindings_by_name=self._experiment.bindings,
                design=self._context.design,
                plan=self._context.plan,
                path="warehouse",
                _source=self if population == "assigned" else self.triggered_source(),
                configs=self.context.configs,
                _source_identity_record=identity_record,
            )
        return self._rehydrated_moments(
            cube.to_pylist(),
            identity_record,
            self if population == "assigned" else self.triggered_source(),
        )

    def _rehydrated_moments(
        self, rows: list[dict[str, Any]], identity_record: dict[str, Any], source: Any
    ) -> _BuiltinMomentSource:
        from increment.sources import MomentsSource

        return MomentsSource(
            rows,
            metrics=self._metrics,
            study_id=self._experiment_name,
            bindings_by_name=self._experiment.bindings,
            design=self._context.design,
            plan=self._context.plan,
            path="warehouse",
            _trusted_trigger_name=self.context.trigger_name,
            _source_snapshot_evidence=self._session.source_snapshot_evidence,
            _source_identity_record=identity_record,
            _source=source,
            configs=self.context.configs,
        )

    def build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
        horizon_metrics: Sequence[Metric] | None = None,
    ) -> Any:
        """Return live panel relations independent of operation-owned TEMP tables."""
        return self._build_panel_for_metric(
            exposures,
            metric,
            population,
            horizon_metrics=horizon_metrics,
            use_cache=False,
        )

    def panel_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]:
        """Return panel SQL for every declared metric and requested breakout."""
        return self._panel_sql_for_metrics(breakouts=list(breakouts))

    def summary_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]:
        """Return group-summary SQL for every declared metric and breakout."""
        return self._summary_sql_for_metrics(breakouts=list(breakouts))

    def export_moments(self, path: str | Path) -> None:
        """Write this experiment's moments (per-metric ``group_summary``
        rows, unioned) to a parquet file at *path*.

        The native family's transport format for
        ``Analysis.from_moments`` - see that facade method's docstring
        for the wire-format contract.

        Fixed-horizon exports carry ``group_summary`` rows, or one metric-free
        ``design_summary`` envelope, stamped ``moments_format=12`` with the
        originating source identity. A metric with several scalar families or
        a categorical-only ``cuped_multicovariate`` request is exported in
        format 13 with method-scoped evidence envelopes; categorical evidence
        is reduced to compact cells in the warehouse, or the export refuses
        with ``query.cuped.general_qr_required``. Registered sequential
        exports carry one typed ``sequential_checkpoint`` envelope stamped
        ``moments_format=10`` with the same identity. The earlier
        fixed-horizon formats 10 and 11
        and sequential format 9 remain readable. ``from_moments`` validates
        and strips these transport columns before constructing its immutable
        context. Formats 1-8 and missing or partial
        decision-plan payloads are refused; supported inference variants and
        view policies round-trip without fallback.
        """
        if getattr(self.context.plan.inference, "registration", None) is not None:
            from increment.sources import export_source_moments

            return export_source_moments(
                self, path, observational_refusal=self._refuse_observational_quantile
            )

        if self._experiment.cluster is not None and not isinstance(
            self._context.design, Encouragement
        ):
            from increment.sources import refuse_cluster_grain_transport

            refuse_cluster_grain_transport(
                operation="export_moments",
                source="native",
                cluster=self._experiment.cluster,
                design=self._context.design,
                route_forward=(
                    "analyse the clustered source directly; direct clustered inference "
                    "preserves cluster-grain degrees of freedom and counts"
                ),
            )
        from increment.sources import refuse_quantile_moments_export

        refuse_quantile_moments_export(
            self._metrics,
            design=self._context.design,
            observational_refusal=self._refuse_observational_quantile,
        )
        if not self._metrics:
            from increment.sources import export_source_moments

            return export_source_moments(
                self, path, observational_refusal=self._refuse_observational_quantile
            )
        from increment._source_identity import source_identity
        from increment.decision_wire import compiled_plan_to_json
        from increment.query.cuped import (
            categorical_request_cells,
            portable_qr_requests,
        )
        from increment.sources import stamp_fixed_rows, write_moments_table

        identity_record = source_identity(self)
        identity_payload = json.dumps(identity_record, sort_keys=True, separators=(",", ":"))
        requests = portable_requests(
            self.context.configs, design=self.context.design, categorical=True
        )
        if requests is not None:
            with self.readout_snapshot(
                metrics=tuple(metric for metric, _, _, _ in requests),
                population="assigned",
                configs=self.context.configs,
            ) as pinned:
                requests = portable_qr_requests(pinned, requests, population="assigned")
                pinned_identity = source_identity(pinned)
                pinned_identity_payload = json.dumps(
                    pinned_identity, sort_keys=True, separators=(",", ":")
                )
                portable = portable_cuped_rows(
                    pinned,
                    requests,
                    categorical_cells=lambda metric, method: categorical_request_cells(
                        pinned, metric, method, population="assigned"
                    ),
                    identity=pinned_identity,
                )
                rows = stamp_fixed_rows(
                    portable,
                    version=CUPED_EVIDENCE_FORMAT,
                    plan=compiled_plan_to_json(self._context.plan),
                    identity=pinned_identity_payload,
                    counts=json.dumps(pinned.unit_counts(), sort_keys=True, separators=(",", ":")),
                    compliance=cast("DefinitionsMomentSource", pinned)._compliance_wire_payload(),
                )
                write_moments_table(path, rows, CUPED_EVIDENCE_FORMAT)
            return
        # Import lazily because parquet export requires the live ibis session.
        import pyarrow as pa
        import pyarrow.parquet as pq

        from increment.sources import (
            DECISION_PLAN_FIELD,
            MOMENTS_FORMAT,
            SOURCE_IDENTITY_FIELD,
            _validate_moment_counts,
        )

        moments = self._moments_for_metrics()
        for row in moments.to_pylist():
            _validate_moment_counts(row)
        moments = moments.append_column(
            "moments_format", pa.array([MOMENTS_FORMAT] * moments.num_rows, type=pa.int64())
        )
        assignment_counts = self.unit_counts()
        moments = moments.append_column(
            ASSIGNMENT_COUNTS_FIELD,
            pa.array(
                [json.dumps(assignment_counts, sort_keys=True, separators=(",", ":"))]
                * moments.num_rows,
                type=pa.string(),
            ),
        )
        moments = moments.append_column(
            DECISION_PLAN_FIELD,
            pa.array(
                [compiled_plan_to_json(self._context.plan)] * moments.num_rows,
                type=pa.string(),
            ),
        )
        moments = moments.append_column(
            SOURCE_IDENTITY_FIELD,
            pa.array([identity_payload] * moments.num_rows, type=pa.string()),
        )
        compliance_payload = self._compliance_wire_payload()
        if compliance_payload is not None:
            moments = moments.append_column(
                COMPLIANCE_SUMMARY_FIELD,
                pa.array([compliance_payload] * moments.num_rows, type=pa.string()),
            )
        moments = moments.replace_schema_metadata(
            {
                **(moments.schema.metadata or {}),
                b"increment.moments_format": str(MOMENTS_FORMAT).encode(),
            }
        )
        pq.write_table(moments, str(path))

    # - integrity/counting helpers ------------------------------------

    def _arm_counts(self, population: Table) -> dict[str, int]:
        """Unit counts per arm off an already-narrowed population table."""
        return arm_counts(self._con, population)

    def trigger_rates(self) -> dict[str, float]:
        """Observed triggered share per arm."""
        self._validate_trigger_capability(operation="trigger_rates")
        self._validate_mixed_assignments()
        assigned = self._arm_counts(self._get_exposures())
        triggered = self._arm_counts(self._get_trigger_population(operation="trigger_rates"))
        return {arm: (triggered.get(arm, 0) / n if n else 0.0) for arm, n in assigned.items()}

    def _counts_for_population(
        self, population: Literal["assigned", "triggered"], *, operation: str
    ) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]:
        """Return assignment counts for one source-owned population."""
        if population == "triggered":
            self._validate_trigger_capability(operation=operation)
        self._validate_mixed_assignments()
        exposures = (
            self._get_trigger_population(operation=operation)
            if population == "triggered"
            else self._get_exposures()
        )
        cluster = self._experiment.cluster
        if cluster is None:
            counts = self._arm_counts(exposures)
            unit_counts: dict[str, int] = {}
            grain: Literal["unit", "cluster"] = "unit"
        else:
            self._validate_cluster_labels(exposures, cluster)
            rows = self._session.fetch_arrow(
                cluster_exposure_counts(exposures, self._experiment)
            ).to_pylist()
            counts = {str(r["group_id"]): int(r["n_clusters"]) for r in rows}
            unit_counts = {str(r["group_id"]): int(r["n_units"]) for r in rows}
            grain = "cluster"
        if population == "assigned":
            mixed = self._mixed_assignment_units or 0
            unassigned = self._unassigned_assignment_units or 0
            if mixed:
                counts[MIXED_ASSIGNMENT_LABEL] = mixed
            if unassigned:
                counts[UNASSIGNED_LABEL] = unassigned
        return grain, counts, unit_counts

    def triggered_counts(
        self,
    ) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]:
        """Return assignment counts for the source's triggered population."""
        return self._counts_for_population("triggered", operation="triggered_counts")

    def assignment_counts(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> dict[str, int]:
        """Source-level enrolled counts independent of metric maturity."""
        return self._counts_for_population(population, operation="assignment_counts")[1]

    def allocation_history(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> pa.Table:
        """Return enrolled-unit history at the assigned or trigger date."""
        cluster = self._experiment.cluster
        if cluster is not None:
            _refuse_operation(
                operation="allocation_history",
                request={"experiment": self._experiment.name, "cluster": cluster},
                offered=("unit",),
                route="use srm() or triggered_counts() for the cluster-grain randomization check",
            )
        self._validate_mixed_assignments()
        exposures = (
            self._get_trigger_population(operation="allocation_history")
            if population == "triggered"
            else self._get_exposures()
        )
        if population == "triggered":
            exposures = exposures.mutate(first_exposure_ts=exposures.first_trigger_ts)
        history = daily_exposure_counts(exposures, self._experiment).order_by(["ds", "group_id"])
        import pyarrow as pa

        result = self._session.fetch_arrow(history)
        return cast(
            "pa.Table",
            result.append_column("analysis_population", pa.array([population] * result.num_rows)),
        )

    # - MomentSource protocol -------------------------------------------

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> list[dict[str, Any]]:
        if grain != "total" and population != "assigned":
            from increment._evidence_refusals import refuse_grain

            refuse_grain(
                self,
                grain,
                ("total",),
                route="request grain='total' for the triggered population",
            )
        if grain != "total":
            # Preserve direct MomentSource calls as source-owned lazy reads.
            return self._day_axis_source(selected=[metric], _legacy_facade=False).moments(
                metric,
                grain=grain,
                by=by,
                completed_windows_only=completed_windows_only,
                include_covariate=include_covariate,
            )
        if by:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self,
                tuple(by),
                self.breakouts,
                route="use breakout_moments() for a dimensioned totals view",
            )
        combined = self._moments_for_metrics(selected=[metric], population=population)
        return cast("list[dict[str, Any]]", combined.to_pylist())

    def moments_for_method(
        self,
        metric: Metric,
        method: Any,
        *,
        by: Sequence[str] = (),
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> list[dict[str, Any]]:
        """Materialize the scalar family for exactly one method request."""
        if by:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self,
                tuple(by),
                self.breakouts,
                route="use breakout_moments() for dimensioned totals",
            )
        combined = self._moments_for_metrics(
            methods=[method],
            selected=[metric],
            narrow_cuped=True,
            population=population,
        )
        return cast("list[dict[str, Any]]", combined.to_pylist())

    def _sequential_observation_mapping(self):
        from increment.sequential_source import native_observation_mapping

        return native_observation_mapping(
            self._defs, self._experiment, on_mixed_assignment=self._on_mixed_assignment
        )

    def capture_sequential(self, *, finalized: bool, as_of: dt.date, previous=None):
        """Capture only units finalized through the explicit common horizon."""
        from increment.query.artifact_publish import artifact_context
        from increment.query.sequential_capture import (
            capture_relations,
            record_batches,
            validate_relational_capture,
        )
        from increment.sequential_state import sequential_refuse

        recipe_id = artifact_context(
            self._defs,
            self._experiment,
            self._on_mixed_assignment,
            metrics=self.context.metrics,
            encouragement_uptake=self.context.design
            if isinstance(self.context.design, Encouragement)
            else None,
        ).sha256
        previous = previous if previous is not None else getattr(self, "_sequential_snapshot", None)
        covariate = self._experiment.n_pre_periods > 0
        registration, _ = validate_relational_capture(
            self, finalized=finalized, as_of=as_of, previous=previous, covariate=covariate
        )
        monitor_triggered = self.context.trigger_name is not None and (
            previous is None or previous.triggered is not None
        )
        if monitor_triggered:
            self._certify_sequential_feeds(registration, as_of)
        uptake_facts = (
            (self.context.design.uptake.fact,)
            if isinstance(self.context.design, Encouragement)
            and any(model.observable == "uptake" for model in registration.models)
            else ()
        )
        with self._pinned_source_execution(
            metrics=[
                metric
                for metric in self.context.metrics
                if any(
                    model.metric == metric.name and model.observable == "outcome"
                    for model in registration.models
                )
            ],
            uptake_facts=uptake_facts,
        ) as pinned:
            pinned._validate_mixed_assignments()
            assignment_counts = pinned.assignment_counts(population="assigned")

            def capture_population(population, parent, triggered=None):
                selected_registration = (
                    registration
                    if population == "assigned"
                    else pinned.context.plan.inference.triggered_registration
                )
                anchors = (
                    pinned._get_exposures()
                    if population == "assigned"
                    else pinned._get_trigger_population(operation="capture_sequential")
                )
                if population == "triggered":
                    anchors = anchors.mutate(first_exposure_ts=anchors.first_trigger_ts)
                final_day = pinned._window_close_day(
                    selected_registration.reveal.longest_window_days
                )
                cohort = anchors.filter(final_day(anchors) <= ibis.literal(as_of))
                metrics = {metric.name: metric for metric in pinned.context.metrics}

                def relation_for(model):
                    if model.observable == "uptake":
                        return uptake(cohort)
                    return outcome(metrics[model.metric], cohort, population)

                return capture_relations(
                    pinned,
                    relation_for,
                    cohort,
                    lambda relation: record_batches(pinned._con, relation),
                    recipe_id=recipe_id,
                    finalized=finalized,
                    as_of=as_of,
                    previous=parent,
                    covariate=covariate,
                    assignment_counts=(
                        assignment_counts
                        if population == "assigned"
                        else pinned.assignment_counts(population="triggered")
                    ),
                    population=population,
                    triggered=triggered,
                    final_day=final_day if monitor_triggered else None,
                )

            def outcome(metric, cohort, population):
                if isinstance(metric, RatioMetric):
                    fs, fact = _find_fact_source(pinned._defs, metric.numerator.fact)
                    table = pinned._get_fact_table(fs)
                    events = metric_events(
                        table,
                        metric,
                        value_column=_resolve_value_column(fs, fact),
                        part="numerator",
                    )
                    dfs, dfact = _find_fact_source(pinned._defs, metric.denominator.fact)
                    den_events = metric_events(
                        pinned._get_fact_table(dfs),
                        metric,
                        value_column=_resolve_value_column(dfs, dfact),
                        part="denominator",
                    )
                else:
                    fs, fact = _find_fact_source(pinned._defs, metric.fact)
                    events = metric_events(
                        pinned._get_fact_table(fs),
                        metric,
                        value_column=_resolve_value_column(fs, fact),
                    )
                    den_events = None
                pre_events = events
                pre_cohort = cohort
                if population == "triggered":
                    events = pinned._after_trigger_events(events, cohort)
                    if den_events is not None:
                        den_events = pinned._after_trigger_events(den_events, cohort)
                    pre_cohort = pinned._assignment_anchored(cohort)
                den_stats = (
                    post_exposure_stats(
                        den_events,
                        cohort,
                        source_key=f"{metric.name}:den",
                        experiment=pinned._experiment,
                    )
                    if den_events is not None
                    else None
                )
                spine, stats = unit_day_spine_stats(
                    cohort, events, pinned._experiment, metric.name, end_date=ibis.literal(as_of)
                )
                # The same zero-filled pre-period total fixed-horizon CUPED reads.
                pre_stats = (
                    pre_period_stats(
                        pre_events, pre_cohort, pinned._experiment, source_key=f"{metric.name}:pre"
                    )
                    if covariate
                    else None
                )
                totals = unit_totals(
                    spine,
                    stats,
                    metric,
                    pinned._experiment,
                    pre_stats=pre_stats,
                    den_stats=den_stats,
                    data_as_of=as_of,
                    warn_on_censoring=False,
                )
                # A fixed threshold clips each unit's finalized total as it is revealed.
                return winsorize_unit_totals(totals, metric)

            def uptake(cohort):
                design = pinned.context.design
                if not isinstance(design, Encouragement):
                    sequential_refuse("source.invalid", "uptake requires encouragement assignment")
                fs, _ = _find_fact_source(pinned._defs, design.uptake.fact)
                table = pinned._get_fact_table(fs)
                events = pinned._uptake_source_events(table, design.uptake.fact)
                return _attach_uptake_flag(cohort, cohort, events, design.uptake.window_days)

            triggered = None
            if (
                monitor_triggered
                and pinned.context.plan.inference.triggered_registration is not None
            ):
                triggered = capture_population(
                    "triggered", None if previous is None else previous.triggered
                )
            snapshot = capture_population("assigned", previous, triggered)
        return self._retain_sequential_snapshot_integrity(snapshot)

    def _retain_sequential_snapshot_integrity(self, snapshot):
        self._sequential_snapshot = snapshot
        if snapshot.finalized and snapshot.assignment_counts is not None:
            self._retained_snapshot_integrity = True
            self._mixed_assignment_prefix_id = snapshot.prefix_id
            self._mixed_assignment_units = snapshot.assignment_counts.get(MIXED_ASSIGNMENT_LABEL, 0)
            self._unassigned_assignment_units = snapshot.assignment_counts.get(UNASSIGNED_LABEL, 0)
        else:
            self._retained_snapshot_integrity = False
            self._mixed_assignment_prefix_id = None
        return snapshot

    def breakout_unit_frame(
        self,
        metric: Metric,
        *,
        breakout: Breakout,
        covariates: Sequence[str] = (),
        population: Literal["assigned", "triggered"] = "assigned",
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> pa.Table:
        from increment._source_types import breakout_dimension_column

        units = self.unit_frame(
            metric, covariates=covariates, population=population, outcome_stage=outcome_stage
        )
        exposures = (
            self._get_trigger_population(operation="breakout_unit_frame")
            if population == "triggered"
            else self._get_exposures()
        )
        fact_source = _resolve_breakout_fact_source(self._defs, self._experiment, breakout)
        properties = self._build_breakout_properties_table(
            fact_source, breakout.property, exposures
        )
        dimensions = self._session.fetch_arrow(
            properties.select(
                "unit_id",
                **{breakout_dimension_column(covariates): properties[breakout.property]},
            )
        )
        return units.join(dimensions, keys="unit_id", join_type="left outer")

    def fit_cuped_multicovariate(
        self,
        metric: Metric,
        method,
        *,
        control: str,
        arm_order: tuple[str, ...],
        population: Literal["assigned", "triggered"] = "assigned",
        breakout_dimension: str | None = None,
        breakout_source: str | None = None,
        breakout_value: str | None = None,
        breakout_is_null: bool = False,
        return_components: bool = False,
    ):
        from increment.query.cuped import fit_multicovariate_request

        return fit_multicovariate_request(
            self,
            metric,
            method,
            control=control,
            arm_order=arm_order,
            population=population,
            breakout_dimension=breakout_dimension,
            breakout_source=breakout_source,
            breakout_value=breakout_value,
            breakout_is_null=breakout_is_null,
            return_components=return_components,
        )

    def cuped_outcome_domains(self, metric: Metric) -> tuple[str, ...]:
        """Return declared SQL domains for outcome components without reading rows."""
        from increment.query._cuped_cells import component_domain

        components = (
            (metric.numerator, metric.denominator) if isinstance(metric, RatioMetric) else (metric,)
        )
        domains = []
        for measure in components:
            fact_source, fact = _find_fact_source(self._defs, measure.fact)
            column = _resolve_value_column(fact_source, fact)
            datatype = self._get_fact_table(fact_source)[column].type() if column else None
            domains.append(component_domain(measure, datatype))
        return tuple(domains)

    def method_fit_input_digest(self, method_request) -> str:
        from increment.query.cuped import method_fit_input_digest

        return method_fit_input_digest(self, method_request)

    @contextmanager
    def cuped_relation(self, metric: Metric, request):
        """Keep the complete CUPED relation inside one pinned source execution."""
        from increment.query._cuped_cells import component_domain, general_qr_required

        configs = overlay_configs(
            (metric,), self.context.configs, methods=[request.method], prior=None
        )
        with self._pinned_source_execution(metrics=(metric,), configs=configs) as pinned:
            components = (
                (metric.numerator, metric.denominator)
                if isinstance(metric, RatioMetric)
                else (metric,)
            )
            for measure in components:
                source, fact = _find_fact_source(pinned._defs, measure.fact)
                column = _resolve_value_column(source, fact)
                datatype = (
                    pinned._get_fact_table(source)[column].type() if column is not None else None
                )
                if component_domain(measure, datatype) == "unknown":
                    general_qr_required(metric, "outcome_domain")
            cache_key = (
                metric.name,
                repr(request.identity),
                pinned._fixed_event_horizon,
            )
            cached = pinned._cuped_relation_cache.get(cache_key)
            if cached is not None:
                yield cached, pinned._session.fetch_arrow, pinned._session.qr_scratch
                return
            with pinned._unit_relation(
                metric,
                covariates=tuple(item.name for item in request.method.covariates or ()),
                population=request.population,
            ) as (relation, execute):
                if request.breakout_dimension is not None:
                    from increment.semantics.models import Breakout

                    breakout = Breakout(
                        property=request.breakout_dimension,
                        source=request.breakout_source,
                    )
                    exposures = (
                        pinned._get_trigger_population(operation="breakout_cuped")
                        if request.population == "triggered"
                        else pinned._get_exposures()
                    )
                    source = _resolve_breakout_fact_source(
                        pinned._defs, pinned._experiment, breakout
                    )
                    properties = pinned._build_breakout_properties_table(
                        source, breakout.property, exposures
                    )
                    relation = relation.left_join(
                        properties.select(
                            "unit_id",
                            **{"__qr_breakout_value": properties[breakout.property]},
                        ),
                        "unit_id",
                    )
                    if request.breakout_is_null:
                        relation = relation.filter(relation["__qr_breakout_value"].isnull())
                    else:
                        relation = relation.filter(
                            canonical_dimension_value(relation["__qr_breakout_value"])
                            == request.breakout_value
                        )
                table_name = pinned._session.temp_name(
                    "cuped_qr_relation", str(len(pinned._cuped_relation_cache))
                )
                relation = pinned._session.materialize_table(table_name, relation, required=True)
                pinned._cuped_relation_cache[cache_key] = relation
                yield relation, execute, pinned._session.qr_scratch

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        population: Literal["assigned", "triggered"] = "assigned",
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> pa.Table:
        with self._unit_relation(
            metric, covariates=covariates, population=population, outcome_stage=outcome_stage
        ) as (relation, execute):
            return cast("pa.Table", execute(relation))

    @contextmanager
    def _unit_relation(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        population: Literal["assigned", "triggered"] = "assigned",
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ):
        if outcome_stage not in ("raw", "transformed"):
            from increment.winsor import winsor_refuse

            winsor_refuse("invalid_state", "Unknown unit outcome stage.")
        if population == "triggered":
            self._validate_trigger_capability(operation="unit_frame")
        cluster = self._experiment.cluster
        if "cluster_id" in covariates and cluster != "cluster_id":
            _refuse(_NATIVE_COVARIATE_RESERVED, covariate="cluster_id", cluster=cluster)
        metric_methods = tuple(
            method
            for config in self.context.configs
            if config.metric.name == metric.name
            for method in effective_methods(config, design=self.context.design)
        )
        declared_covariates = {
            covariate.name: self._resolve_covariate_binding(covariate, metric)
            for method in metric_methods
            for covariate in method.covariates or ()
        }
        design = self._experiment.design
        adjustment_covariates = {
            covariate.property: covariate for covariate in getattr(design, "covariates", ())
        }
        covariate_sources: dict[str, tuple[Any, str, bool]] = {}
        covariate_measures: dict[str, Measure] = {}
        for name in dict.fromkeys(covariates):
            if name == "cluster_id":
                continue
            adjustment_declared = adjustment_covariates.get(name)
            metric_declared = declared_covariates.get(name)
            if (
                adjustment_declared is not None
                and isinstance(metric_declared, CupedCovariateBinding)
                and metric_declared.measure is None
            ):
                ensure_covariate_binding_compatible(
                    adjustment_declared,
                    metric_declared,
                    current_alias=name,
                )
            if (
                adjustment_declared is None
                and isinstance(metric_declared, CupedCovariateBinding)
                and metric_declared.measure is not None
            ):
                covariate_measures[name] = metric_declared.measure
                continue
            declared = adjustment_declared or metric_declared
            declared_property = getattr(declared, "property", None)
            property_name = declared_property if declared_property is not None else name
            fact_source = self._resolve_unit_covariate_source(
                property_name,
                declared_source=getattr(declared, "source", None),
                kind=getattr(declared, "kind", None) if declared is not None else None,
            )
            prop = next(p for p in self._defs.properties_of(fact_source) if p.name == property_name)
            covariate_sources[name] = (
                fact_source,
                property_name,
                prop.dtype == "string",
            )
        if outcome_stage == "raw" and self._session._source_tables is None:
            measure_sources = tuple(
                _find_fact_source(self._defs, covariate.measure.fact)[0].name
                for covariate in declared_covariates.values()
                if isinstance(covariate, CupedCovariateBinding)
                and covariate.measure is not None
                and covariate.name not in adjustment_covariates
                and covariate.name in covariates
            )
            property_sources = tuple(
                source.name for source, _property, _categorical in covariate_sources.values()
            )
            extra_sources = tuple(dict.fromkeys((*measure_sources, *property_sources)))
            with self._pinned_source_execution(
                metrics=(metric,), extra_sources=extra_sources
            ) as pinned:
                with pinned._unit_relation(
                    metric,
                    covariates=covariates,
                    population=population,
                    outcome_stage=outcome_stage,
                ) as evidence:
                    yield evidence
                return
        with self._reduction_batch():
            exposures = (
                self._get_trigger_population(operation="unit_frame")
                if population == "triggered"
                else self._get_exposures()
            )
            # Covariates are scoped to before assignment, whichever population is read.
            assigned = exposures if population == "assigned" else self._get_exposures()
            if cluster is not None:
                self._validate_cluster_labels(exposures, cluster)
            uptake_events, uptake_window_days = self._resolve_uptake(
                cluster, operation="unit_frame"
            )
            panel, spine, stats, _events, den_events, fact_tbl, value_col, data_as_of = (
                self._build_panel_for_metric(exposures, metric, population)
            )
            _summary, _pre_stats, _den_stats, totals = self._build_metric_summary(
                metric,
                exposures,
                spine,
                stats,
                den_events,
                fact_tbl,
                value_col,
                data_as_of,
                cluster=cluster,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                warn_on_censoring=True,
                want_cuped=False,
            )
            if (
                outcome_stage == "raw"
                and getattr(metric, "winsorization", None) is not None
                and "y_raw" not in totals.columns
            ):
                from increment._evidence_refusals import refuse_raw_outcomes

                refuse_raw_outcomes(
                    self,
                    offered=False,
                    route="use a source that retains pre-winsor outcomes",
                )
            for name, measure in covariate_measures.items():
                values = self._pre_period_measure_covariate(measure, name, assigned)
                join_key = f"_cuped_unit_id_{name}"
                values = values.rename({join_key: "unit_id"})
                totals = totals.left_join(values, totals["unit_id"] == values[join_key]).drop(
                    join_key
                )
            for name, (fact_source, property_name, categorical) in covariate_sources.items():
                properties = self._build_breakout_properties_table(
                    fact_source, property_name, assigned
                )
                if property_name != name:
                    properties = properties.select(
                        unit_id=properties["unit_id"], **{name: properties[property_name]}
                    )
                totals = join_unit_covariates(totals, properties, name, categorical=categorical)
            relation = self._project_unit_relation(
                totals, metric, (*covariate_sources, *covariate_measures), cluster, outcome_stage
            )
            yield relation, self._session.fetch_arrow

    @staticmethod
    def _project_unit_relation(totals, metric, covariates, cluster, outcome_stage):
        select_cols = {
            "unit_id": totals["unit_id"],
            "group_id": totals["group_id"],
            "y": totals["y_raw"]
            if outcome_stage == "raw" and "y_raw" in totals.columns
            else totals["y"],
        }
        if isinstance(metric, RatioMetric):
            select_cols["y_den"] = totals["y_den"]
        for name in covariates:
            select_cols[name] = totals[name]
        if cluster is not None:
            select_cols["cluster_id"] = totals[cluster].cast("string")
        return totals.select(**select_cols)

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        if grain != "total":
            from increment._evidence_refusals import require_evidence

            require_evidence(self, grain=grain, operation="sql")
        return self._summary_sql_for_metrics()

    def unit_counts(self) -> dict[str, int]:
        """Per-arm enrolled unit counts, with the mixed-assignment
        accounting entry folded in - matches every other native
        readout's "refresh the integrity policy, then count" contract.
        """
        self._validate_mixed_assignments()
        counts = self._arm_counts(self._get_exposures())
        mixed = self._mixed_assignment_units or 0
        unassigned = self._unassigned_assignment_units or 0
        if mixed:
            counts[MIXED_ASSIGNMENT_LABEL] = mixed
        if unassigned:
            counts[UNASSIGNED_LABEL] = unassigned
        return counts

    def cluster_counts(self) -> dict[str, int]:
        from increment._evidence_refusals import refuse_cluster, require_evidence

        cluster = self._experiment.cluster
        if cluster is None:
            require_evidence(self, cluster=True)
            refuse_cluster(
                self,
                offered=(),
                route="use a source that retains cluster identities or cluster counts",
            )
        exposures = self._get_exposures()
        self._validate_cluster_labels(exposures, cluster)
        rows = self._session.fetch_arrow(
            cluster_exposure_counts(exposures, self._experiment)
        ).to_pylist()
        return {str(r["group_id"]): int(r["n_clusters"]) for r in rows}

    def close(self) -> None:
        self._session.drop_materialized()
        self._exposures_cache = None
        self._trigger_cache = None
        self._mixed_assignment_units = None
        self._unassigned_assignment_units = None
        self._mixed_assignment_warned = False
        self._mixed_assignment_fingerprint = None
        self._union_horizon_cache = None
        self._union_horizon_metric_cache = {}
        self._panel_cache = {}
        self._cuped_relation_cache = {}
        self._reduction_calls = 0
        self._materialized = False
        self._reduction_batch_depth = 0
        self._assignment_validation_depth = 0

    # - public aliases for the native-only capability names -------------

    sitewide_volume = _site_volume_row

    def day_axis_source(self, *, selected: Sequence[Metric] | None = None) -> _DaySource:
        """Public direct adapter with source-owned reduction accounting."""
        return self._day_axis_source(selected=selected, _legacy_facade=False)

    breakout_moments = _breakout_moments_source
