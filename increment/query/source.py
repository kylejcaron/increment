"""SQL-backed MomentSource: centered moments computed by ibis, not narwhals.

Exists to prove `increment.readouts` is substrate-independent: the same
readout functions, run against a source built here versus one built by
`increment.frame` from an equivalent in-memory dataframe, must produce
identical numbers to 1e-12 relative tolerance (not byte-identical -
float summation order differs across substrates). Asserted in
`tests/test_readouts.py::test_same_readout_serves_both_substrates`.

Reuses `MetricSpec` and its coercion helpers from `increment._metric_specs`
rather than re-deriving the same metrics parsing: that logic is already
backend-agnostic pydantic, not narwhals-specific.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import ibis
import ibis.expr.types as ir
import narwhals as nw

from increment._analysis_config import effective_methods
from increment._frame_validation import _reject_windowed_specs
from increment._metric_specs import MetricsArg, MetricSpec, coerce_metrics, synthesise_metric
from increment.errors import CapabilityError, CodedError, refuse
from increment.estimation._readout_refusals import refuse_observational_quantile
from increment.query.artifact_contract import (
    REFUSALS as _ARTIFACT_REFUSALS,
)
from increment.query.artifact_contract import (
    ArtifactContractError,
    ArtifactStore,
    open_trusted_manifest_snapshot,
    unit_day_artifact_extension_catalog,
)
from increment.query.artifact_ports import artifact_query_ports
from increment.query.builders import (
    _local_date,
    declared_binary_metrics,
    group_summary,
    join_breakout_dimension,
    winsorize_unit_totals,
)
from increment.query.native_contract import (
    _NATIVE_COVARIATE_RESERVED,
    _OUTCOMES_INCOMPLETE_ROUTES,
    _SITEWIDE_OUTCOMES_INCOMPLETE,
    DimensionedDayEvidence,
    SitewideEvidence,
)
from increment.semantics.artifact import (
    ArtifactContext,
    RatioMetricMeasure,
    SimpleMetricMeasure,
    UnitCovariateExtension,
    UnitCovariateLevelExtension,
    UnitDayArtifactManifest,
    UnitDayArtifactRef,
)
from increment.semantics.models import (
    AnalysisPlan,
    Breakout,
    Experiment,
    Metric,
    RatioMetric,
    RetentionMetric,
)
from increment.sources import (
    BreakoutMomentsSource,
    EvidenceSource,
    Grain,
    SourceContext,
    SourceOperation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa
    from ibis.backends.duckdb import Backend
    from ibis.backends.sql import SQLBackend

    from increment._analysis_config import ResolvedMetricConfig
    from increment._evidence_types import ComplianceSummary
    from increment._source_types import EvidenceCapabilities
    from increment.decision import CompiledDecisionPlan
    from increment.query.artifact_reader import ArtifactMomentSource as _ArtifactBase
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Breakout, Metric

else:
    _ArtifactBase = object


def read_extension(*args: Any, **kwargs: Any) -> Any:
    return artifact_query_ports().extensions.read_extension(*args, **kwargs)


def verify_extension_relation(*args: Any, **kwargs: Any) -> Any:
    return artifact_query_ports().extensions.verify_extension_relation(*args, **kwargs)


def _restore_raw_dimension_values(*args: Any, **kwargs: Any) -> Any:
    return artifact_query_ports().reader._restore_raw_dimension_values(*args, **kwargs)


def _new_triggered_reduction_inputs(*args: Any, **kwargs: Any) -> Any:
    return artifact_query_ports().reader._TriggeredReductionInputs(*args, **kwargs)


def restrict_to_units(*args: Any, **kwargs: Any) -> Any:
    return artifact_query_ports().reader.restrict_to_units(*args, **kwargs)


def _artifact_rows(*args: Any, **kwargs: Any) -> Any:
    return artifact_query_ports().reader._rows(*args, **kwargs)


def _scalar_covariate(spec: MetricSpec):
    return next(
        (
            covariate
            for method in spec._methods
            if method.variance_reduction == "cuped"
            for covariate in method.covariates or ()
        ),
        None,
    )


def _melt_to_unit_totals(
    tbl: ir.Table, *, unit: str, group: str, study_id: str, specs: Sequence[MetricSpec]
) -> ir.Table:
    """Project one policy-adjusted ``(unit, metric)`` row per specification."""
    per_metric = []
    for spec in specs:
        covariate = _scalar_covariate(spec)
        base = tbl.select(
            unit_id=tbl[unit].cast("string"),
            experiment_id=ibis.literal(study_id),
            group_id=tbl[group].cast("string"),
            y=tbl[spec.y_column].cast("float64"),
            x=(
                tbl[covariate.name].cast("float64")
                if covariate is not None
                else ibis.literal(None, type="float64")
            ),
            y_den=(
                tbl[spec.denominator].cast("float64")
                if spec.denominator
                else ibis.literal(None, type="float64")
            ),
        )
        base = base.mutate(
            y=ibis.ifelse(base.y.isnan(), ibis.null().cast("float64"), base.y),
            x=ibis.ifelse(base.x.isnan(), ibis.null().cast("float64"), base.x),
            y_den=ibis.ifelse(base.y_den.isnan(), ibis.null().cast("float64"), base.y_den),
        )
        if covariate is not None and covariate.missing != "refuse":
            fill = (
                ibis.literal(0.0)
                if covariate.missing == "zero"
                else base.x.mean().over(ibis.window())
            )
            base = base.mutate(x=ibis.coalesce(base.x, fill))
        if spec.missing == "zero":
            base = base.mutate(y=ibis.coalesce(base.y, ibis.literal(0.0)))
            if spec.denominator:
                base = base.mutate(y_den=ibis.coalesce(base.y_den, ibis.literal(0.0)))
        elif spec.missing == "drop":
            missing = base.y.isnull()
            if spec.denominator:
                missing = missing | base.y_den.isnull()
            base = base.filter(~missing)
        per_metric.append(
            base.mutate(
                metric=ibis.literal(spec.name),
                d=ibis.literal(None, type="float64"),
            ).select(
                "unit_id",
                "experiment_id",
                "group_id",
                "metric",
                "y",
                "x",
                "y_den",
                "d",
            )
        )
    out = per_metric[0]
    for table in per_metric[1:]:
        out = out.union(table)
    return out


class SqlPanelSource:
    """`MomentSource` over centered moments computed by `group_summary`.

    Totals-only: `capabilities` is `{"total"}`. A day-axis SQL source is
    a natural extension, out of scope until this module's own tests
    need it.
    """

    capabilities: frozenset[Grain] = frozenset({"total"})
    breakouts: tuple[str, ...] = ()
    operations: frozenset[SourceOperation] = frozenset({"summary_sql"})
    shape: Literal["unit_summary", "unit_panel"] | None = None

    @property
    def evidence(self) -> EvidenceCapabilities:
        from increment._source_types import EvidenceCapabilities

        return EvidenceCapabilities(
            kind="moments",
            grains=self.capabilities,
            breakouts=self.breakouts,
            shape=self.shape,
            operations=self.operations,
            unit_rows=False,
            raw_outcomes=False,
            method_fit="none",
            compliance_axis="none",
            count_grains=frozenset({"unit"}),
            populations=frozenset({"assigned"}),
            sequential_capture="none",
            scalar=None,
            available_covariates=None,
            sql_grains=frozenset({"total"}),
        )

    def __init__(
        self,
        con: SQLBackend,
        summary: ir.Table,
        *,
        metrics: Sequence[Metric],
        study_id: str,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | CompiledDecisionPlan | None = None,
        unassigned: int = 0,
        specs: Sequence[MetricSpec] | None = None,
    ) -> None:
        self._con = con
        self._summary = summary
        self._unassigned = unassigned
        self._enrollment_counts: dict[str, int] | None = None
        metric_catalog = tuple(metrics)
        from increment.decision import CompiledDecisionPlan
        from increment.plan import (
            compile_decision_plan,
            refuse_observational_relative_margin,
            validate_compiled_encouragement_plan,
        )

        compiled = (
            plan
            if isinstance(plan, CompiledDecisionPlan)
            else compile_decision_plan(
                cast("AnalysisPlan | None", plan),
                metric_catalog,
                path="frame",
                design=design,
            )
        )
        validate_compiled_encouragement_plan(compiled, design)
        refuse_observational_relative_margin(design, compiled)
        from increment._analysis_config import resolve_configs

        self._context = SourceContext(
            study_id=study_id,
            design=design,
            plan=compiled,
            metrics=metric_catalog,
            configs=resolve_configs(
                metric_catalog,
                bindings=None,
                specs=({spec.name: spec for spec in specs} if specs is not None else None),
                methods=None,
                prior=None,
            ),
            cluster=None,
        )

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def population(self) -> Literal["assigned"]:
        return "assigned"

    @property
    def trigger_name(self) -> str | None:
        return self.context.trigger_name

    @classmethod
    def from_frame_via_memtable(
        cls,
        con: Backend,
        frame: Any,
        *,
        unit: str,
        group: str,
        metrics: MetricsArg,
        study_id: str = "frame",
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
        on_unassigned: Literal["error", "exclude"] = "error",
    ) -> SqlPanelSource:
        """Load a one-row-per-unit summary frame into *con* and compute its moments via SQL.

        Mirrors `increment.frame.from_unit_summary`'s parameters (`unit`,
        `group`, `metrics`, `on_unassigned`); the only difference is which
        substrate computes the centered moments. Runs the same validation
        prologue `FrameTotalsSource.from_frame` runs, narrowed to the checks
        this constructor's own parameters can support: it has no `cluster`/
        `uptake`, so cluster-label and uptake-binary checks do not apply here.
        Control-arm membership IS checked whenever `design` is given, using
        `design.control_group` -- every design (`Randomized`/`Encouragement`/
        `Observational`) already carries it.
        """
        from increment._frame_validation import (
            _canonicalize_group_identity,
            _coerce_metric_columns,
            _enforce_missing_policy,
            _group_counts,
            _reject_duplicate_units,
            _resolve_unassigned,
            _validate_columns,
            _validate_control,
            _validate_conversion_binary,
            _validate_frame_boundary,
        )

        specs = coerce_metrics(metrics)
        _reject_windowed_specs(specs)
        nwf = nw.from_native(frame, eager_only=True)
        _validate_columns(nwf, roles=[("unit", unit), ("group", group)], metrics=specs)
        nwf = _canonicalize_group_identity(nwf, group)
        metric_columns = sorted(
            {
                column
                for spec in specs
                for column in (spec.y_column, spec.denominator)
                if column is not None
            }
            | {
                covariate.name
                for spec in specs
                for method in spec._methods
                for covariate in (method.covariates or ())
                if covariate.kind == "continuous"
            }
        )
        nwf = _coerce_metric_columns(nwf, metric_columns)
        _validate_frame_boundary(nwf, unit=unit, date=None, specs=specs)
        nwf, unassigned = _resolve_unassigned(
            nwf,
            unit=unit,
            group=group,
            on_unassigned=on_unassigned,
            constructor="from_frame_via_memtable",
        )
        _reject_duplicate_units(nwf, unit)
        if design is not None:
            _validate_control(nwf, group, design.control_group)
        nwf = _enforce_missing_policy(nwf, specs, shape="summary")
        _validate_conversion_binary(nwf, specs)

        tbl = ibis.memtable(nwf.to_native())
        metric_objects = [synthesise_metric(s) for s in specs]
        totals = _melt_to_unit_totals(tbl, unit=unit, group=group, study_id=study_id, specs=specs)
        transformed = [
            winsorize_unit_totals(totals.filter(totals.metric == metric.name), metric)
            for metric in metric_objects
        ]
        totals = transformed[0]
        for other in transformed[1:]:
            totals = totals.union(other)
        summary = group_summary(totals, binary_metrics=declared_binary_metrics(metric_objects))
        source = cls(
            con,
            summary,
            metrics=metric_objects,
            study_id=study_id,
            design=design,
            plan=plan,
            unassigned=unassigned,
            specs=specs,
        )
        source._enrollment_counts = _group_counts(nwf, group)
        return source

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        if grain != "total":
            from increment._evidence_refusals import require_evidence

            require_evidence(self, grain=grain)
        if by:
            from increment._evidence_refusals import require_evidence

            require_evidence(self, by=tuple(by))
        rows = self._summary.filter(self._summary.metric == metric.name)
        return self._con.to_pyarrow(rows).to_pylist()

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> NoReturn:
        from increment._evidence_refusals import refuse_unit_rows, require_evidence

        if outcome_stage == "raw":
            require_evidence(self, raw_outcomes=True)

        require_evidence(self, unit_rows=True, covariates=covariates)
        refuse_unit_rows(
            self,
            offered=False,
            route="use a source that retains unit-level rows",
        )

    def unit_counts(self) -> dict[str, int]:
        from increment._labels import UNASSIGNED_LABEL

        if self._enrollment_counts is not None:
            counts = dict(self._enrollment_counts)
        else:
            first = self._context.metrics[0].name
            rows = self._con.to_pyarrow(
                self._summary.filter(self._summary.metric == first)
            ).to_pylist()
            counts = {str(r["group_id"]): int(r["n"]) for r in rows}
        if self._unassigned:
            counts[UNASSIGNED_LABEL] = self._unassigned
        return counts

    def cluster_counts(self) -> dict[str, int]:
        from increment._evidence_refusals import refuse_cluster, require_evidence

        require_evidence(self, cluster=True)
        refuse_cluster(
            self,
            offered=(),
            route="use a source that retains cluster identities or cluster counts",
        )

    def compliance_dates(self) -> Sequence[object]:
        from increment._evidence_refusals import refuse_operation

        refuse_operation(
            self,
            "compliance_dates",
            tuple(sorted(self.operations)),
            route="use a panel source for as-of compliance",
        )

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        from increment._evidence_types import raise_legacy_compliance_state

        raise_legacy_compliance_state(
            study_id=self.context.study_id,
            reason="SQL summary contains no enrollment-level uptake state",
        )

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        if grain != "total":
            from increment._evidence_refusals import require_evidence

            require_evidence(self, grain=grain, operation="sql")
        return {
            m.name: ibis.to_sql(self._summary.filter(self._summary.metric == m.name))
            for m in self._context.metrics
        }

    def _summary_sql_for_metrics(
        self,
        breakouts: list[Breakout] | None = None,
        selected: Sequence[Metric] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, str]:
        return self.sql()

    def summary_sql(self, *, breakouts: Sequence[Breakout] = ()) -> dict[str, str]:
        """Return the adopted panel's group-summary SQL."""
        if breakouts:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self,
                tuple(item.property for item in breakouts),
                tuple(self.breakouts),
                route="request summary SQL without breakouts",
            )
        return self._summary_sql_for_metrics(breakouts=list(breakouts))

    def close(self) -> None:
        pass


def _artifact_source_context(
    expected_context: ArtifactContext,
) -> tuple[Experiment, SourceContext]:
    """Rehydrate the trusted typed arm context without touching a warehouse."""
    from pydantic import TypeAdapter

    from increment.query.artifact_contract import validate_artifact_context
    from increment.semantics.design import (
        SUPPORTED_DESIGN_MECHANISMS,
        Encouragement,
        parse_design,
    )
    from increment.semantics.models import EncouragementDeclaration, _with_inherited_day_boundary

    validate_artifact_context(expected_context)
    try:
        payload = json.loads(expected_context.canonical_json)
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            "artifact.context.mismatch",
            "artifact context canonical JSON is invalid",
            context={"field": "canonical_json"},
        ) from exc

    for field in ("experiment_name", "experiment"):
        if field not in payload:
            refuse(
                _ARTIFACT_REFUSALS["artifact.context.mismatch"],
                message="artifact context omits a required field",
                field=field,
            )
    definitions_payload = payload.get("definitions")
    if not isinstance(definitions_payload, Mapping):
        refuse(
            _ARTIFACT_REFUSALS["artifact.context.mismatch"],
            message="artifact context omits a required field",
            field="definitions",
        )
    for field in ("metrics", "day_boundary"):
        if field not in definitions_payload:
            refuse(
                _ARTIFACT_REFUSALS["artifact.context.mismatch"],
                message="artifact context omits a required field",
                field=f"definitions.{field}",
            )

    try:
        metric_adapter: TypeAdapter[Metric] = TypeAdapter(Metric)
        metric_pool = [
            metric_adapter.validate_python(item) for item in definitions_payload["metrics"]
        ]
        experiment_payload = dict(payload["experiment"])
        legacy_plan = experiment_payload.get("plan")
        if (
            "plan_q_explicit" not in payload
            and isinstance(legacy_plan, dict)
            and legacy_plan.get("q") == AnalysisPlan().q
        ):
            # Older contexts serialized plan defaults without preserving explicitness.
            legacy_plan = dict(legacy_plan)
            legacy_plan.pop("q")
            experiment_payload["plan"] = legacy_plan
        effective_design = None
        design_payload = experiment_payload.get("design")
        if isinstance(design_payload, Mapping):
            mechanism = design_payload.get("mechanism")
            if mechanism == "encouragement" or (
                mechanism is not None and mechanism not in SUPPORTED_DESIGN_MECHANISMS
            ):
                effective_design = parse_design(design_payload)
                if isinstance(effective_design, Encouragement):
                    # The context carries the full effective design; Experiment.design keeps only
                    # the declaration-shaped part of it.
                    experiment_payload["design"] = EncouragementDeclaration(
                        uptake=effective_design.uptake,
                        one_sided=effective_design.one_sided,
                        exclusion_restriction=effective_design.exclusion_restriction,
                    )
        experiment = Experiment.model_validate(experiment_payload)
        experiment = _with_inherited_day_boundary(experiment, definitions_payload["day_boundary"])
        experiment_name = str(payload["experiment_name"])
    except (TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, CodedError) and exc.code == "design.mechanism_unsupported":
            raise
        raise ArtifactContractError(
            "artifact.context.mismatch",
            "artifact context does not contain a valid typed experiment",
        ) from exc
    if experiment.name != experiment_name:
        raise ArtifactContractError(
            "artifact.context.mismatch",
            "artifact context does not contain its declared experiment",
        )
    by_name = {metric.name: metric for metric in metric_pool}
    if len(by_name) != len(metric_pool) or set(by_name) != set(experiment.metric_names):
        raise ArtifactContractError(
            "artifact.context.mismatch",
            "artifact context metrics must be unique and match the experiment's metric roster",
        )
    from increment._analysis_config import resolve_configs
    from increment.plan import compile_decision_plan

    metrics = tuple(by_name[name] for name in experiment.metric_names)
    design: Randomized | Encouragement | Observational = (
        effective_design if effective_design is not None else experiment.resolved_design()
    )
    if not isinstance(design, Encouragement) and any(
        entry.request.kind == "encouragement_uptake"
        for entry in unit_day_artifact_extension_catalog(expected_context)
    ):
        raise ArtifactContractError(
            "artifact.context.mismatch",
            "artifact context lists an encouragement uptake extension without a typed design",
        )
    if experiment.plan.inference is not None and experiment.plan.inference.registration is not None:
        # The stored context binds the assigned registration; the triggered one is
        # its derivation and is rebuilt here exactly as the native constructor does.
        from increment.plan import bind_automatic_sequential_plan
        from increment.query.artifact_contract import artifact_source_mapping

        bound = bind_automatic_sequential_plan(
            experiment.plan,
            metrics,
            design=design,
            source_id=experiment.name,
            source_mapping=artifact_source_mapping(expected_context),
            pre_period_covariate=experiment.n_pre_periods > 0,
            trigger=experiment.trigger,
        )
        experiment = Experiment.model_validate(experiment.model_copy(update={"plan": bound}))
    plan = compile_decision_plan(
        experiment.plan,
        metrics,
        path="warehouse",
        design=design,
    )
    context = SourceContext(
        study_id=experiment.name,
        design=design,
        plan=plan,
        metrics=metrics,
        configs=resolve_configs(
            metrics,
            bindings=experiment.bindings,
            specs=None,
            methods=None,
            prior=None,
        ),
        cluster=experiment.cluster,
        intervention_grain=experiment.intervention_grain,
        trigger_name=experiment.trigger,
    )
    return experiment, context


def _artifact_request(extension: Any) -> dict[str, Any]:
    kind = extension.kind
    if kind == "breakout_dimension":
        return {
            "kind": kind,
            "property_name": extension.dimension_name,
            "source_name": extension.source_name,
        }
    if kind == "factor_dimension":
        return {
            "kind": kind,
            "property_name": extension.factor_name,
            "source_name": extension.source_name,
        }
    if kind == "cluster_identity":
        return {"kind": kind, "cluster_name": extension.cluster_name}
    if kind == "cuped_preperiod":
        return {"kind": kind, "metric_name": extension.metric_name}
    if kind == "cuped_measure_covariate":
        return {
            "kind": kind,
            "metric_name": extension.metric_name,
            "covariate_name": extension.covariate_name,
        }
    if kind == "assignment_counts":
        return {"kind": kind, "populations": extension.populations}
    if kind == "trigger_population":
        return {"kind": kind, "trigger_name": extension.trigger_name}
    if kind == "trigger_measure_stats":
        return {
            "kind": kind,
            "trigger_name": extension.trigger_name,
            "metric_names": extension.metric_names,
        }
    if kind == "encouragement_uptake":
        return {"kind": kind, "uptake_name": extension.uptake_name}
    if kind == "site_volume":
        return {"kind": kind, "metric_names": extension.measure_keys}
    if kind in {"unit_covariate", "unit_covariate_level"}:
        return {
            "kind": kind,
            "property_name": extension.covariate_name,
            "source_name": extension.source_name,
        }
    raise ArtifactContractError("artifact.extension.invalid", f"unknown extension kind {kind!r}")


class _ArtifactFacadeSource(_ArtifactBase):
    """Artifact adapter loaded lazily through the registered reader port."""

    def __new__(cls, *args: Any, **kwargs: Any):
        if cls is _ArtifactFacadeSource:
            return object.__new__(_artifact_facade_type())
        return object.__new__(cls)

    def __init__(
        self,
        store: ArtifactStore,
        snapshot_context: AbstractContextManager[object],
        snapshot: Any,
        manifest: UnitDayArtifactManifest,
        *,
        expected_context: ArtifactContext,
        metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
        population_units: frozenset[str] | ir.Table | None = None,
        trigger_anchors: ir.Table | None = None,
    ) -> None:
        super().__init__(
            store,
            snapshot_context,
            snapshot,
            manifest,
            metrics=metrics,
        )
        experiment, source_context = _artifact_source_context(expected_context)
        covariate_metrics = (
            {
                extension.metric_name
                for extension in manifest.extensions
                if extension.kind == "cuped_preperiod"
            }
            if experiment.n_pre_periods > 0
            else set()
        )
        if covariate_metrics:
            source_context = replace(
                source_context,
                configs=tuple(
                    replace(
                        config,
                        covariate_available=(
                            config.covariate_available or config.metric.name in covariate_metrics
                        ),
                    )
                    for config in source_context.configs
                ),
            )
        declared = {metric.name for metric in source_context.metrics}
        if any(binding.metric_name not in declared for binding in manifest.metric_measures):
            raise ArtifactContractError(
                "artifact.context.mismatch",
                "artifact context omits a metric bound by the manifest",
            )
        self._expected_context = expected_context
        self._artifact_experiment = experiment
        self._source_context = source_context
        from increment.sequential_source import validate_source_mapping

        validate_source_mapping(source_context, self._sequential_observation_mapping())
        self._population_units = population_units
        self._trigger_anchors = trigger_anchors
        self._population: Literal["assigned", "triggered"] = (
            "assigned" if population_units is None else "triggered"
        )
        kinds = {extension.kind for extension in manifest.extensions}
        operations: set[SourceOperation] = {
            "moments_source",
            "day_source",
            "export_moments",
            "artifact_experiment",
        }
        if "breakout_dimension" in kinds:
            operations.update({"breakout_source", "breakout_sources", "breakout_summaries"})
        if "factor_dimension" in kinds:
            operations.add("factor_summaries")
        if "site_volume" in kinds:
            operations.add("sitewide_evidence")
        assignment_extensions = [
            extension for extension in manifest.extensions if extension.kind == "assignment_counts"
        ]
        if any(extension.identity_relation is not None for extension in assignment_extensions):
            operations.add("allocation_history")
        has_trigger_population = any(
            extension.kind == "trigger_population" for extension in manifest.extensions
        )
        if assignment_extensions:
            operations.add("unit_counts")
        if has_trigger_population:
            operations.add("triggered_source")
        if has_trigger_population and assignment_extensions:
            operations.add("triggered_counts")
        catalog = unit_day_artifact_extension_catalog(expected_context)
        for extension in manifest.extensions:
            if extension.kind == "site_volume":
                matches = [
                    entry
                    for entry in catalog
                    if entry.request.kind == "site_volume"
                    and entry.definition_sha256 == extension.definition_sha256
                    and entry.source_provenance_sha256 == extension.source_provenance_sha256
                    and tuple(json.loads(entry.canonical_definition_json)["measure_keys"])
                    == tuple(extension.measure_keys)
                ]
            else:
                request = _artifact_request(extension)
                matches = [
                    entry
                    for entry in catalog
                    if json.dumps(entry.request.model_dump(mode="json"), sort_keys=True)
                    == json.dumps(request, sort_keys=True)
                ]
            if len(matches) != 1:
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "manifest extension descriptor does not match trusted catalog",
                )
            if (
                matches[0].definition_sha256 != extension.definition_sha256
                or matches[0].source_provenance_sha256 != extension.source_provenance_sha256
            ):
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "manifest extension descriptor hashes do not match trusted catalog",
                )
        self.operations = frozenset(operations)
        self.breakouts = tuple(
            entry.request.property_name
            for entry in catalog
            if entry.request.kind == "breakout_dimension"
        )

    @property
    def evidence(self) -> EvidenceCapabilities:
        from increment._source_types import ScalarEvidenceDeclaration
        from increment.semantics.design import Encouragement

        has_assignment_counts = any(
            extension.kind == "assignment_counts" for extension in self._manifest.extensions
        )
        has_cluster_identity = self.context.cluster is not None and any(
            extension.kind == "cluster_identity" for extension in self._manifest.extensions
        )
        triggered_view = self._population == "triggered"
        if triggered_view:
            count_grains: set[Literal["unit", "cluster", "assignment", "triggered"]] = set()
            if "triggered_counts" in self.operations:
                count_grains.update({"unit", "assignment", "triggered"})
        else:
            # The assigned exposure relation supports unit counts directly and
            # assignment counts as a fallback when the assignment grain is unit.
            count_grains = {"unit"}
            if has_assignment_counts or self.context.cluster is None or has_cluster_identity:
                count_grains.add("assignment")
            if "triggered_counts" in self.operations:
                count_grains.add("triggered")
        if has_cluster_identity:
            count_grains.add("cluster")
        return replace(
            super().evidence,
            kind="artifact",
            grains=self.capabilities,
            breakouts=self.breakouts,
            shape=self.shape,
            operations=self.operations,
            unit_rows=True,
            compliance_axis="asof" if isinstance(self.context.design, Encouragement) else "none",
            method_fit="warehouse",
            count_grains=frozenset(count_grains),
            populations=(
                frozenset({"triggered"})
                if triggered_view
                else frozenset(
                    {"assigned"}
                    | ({"triggered"} if "triggered_source" in self.operations else set())
                )
            ),
            scalar=ScalarEvidenceDeclaration(
                bindings=self.scalar_bindings,
                moments_only=False,
                recipe_fixed=False,
                recipe_configs=(),
                anonymous_metrics=frozenset(),
                method_evidence=True,
                breakout_evidence=True,
                covariate_grains=None,
            ),
            assignment_relation=(
                "ledger"
                if any(
                    extension.kind == "assignment_counts"
                    and extension.identity_relation is not None
                    for extension in self._manifest.extensions
                )
                else "reduced"
            ),
        )

    def allocation_history(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> pa.Table:
        if self.context.cluster is not None:
            from increment._evidence_refusals import refuse_operation

            refuse_operation(
                self,
                "allocation_history",
                ("unit",),
                route="use srm() or triggered_counts() for the cluster-grain randomization check",
            )
        try:
            return super().allocation_history(population=population)
        except ArtifactContractError as exc:
            if (
                exc.code == "artifact.snapshot.mixed"
                and exc.context.get("role") == "assignment_rows"
            ):
                raise CapabilityError(
                    "assignment identity relation changed after publication",
                    code="artifact.relation.digest_mismatch",
                    context={"role": "assignment_rows"},
                ) from exc
            raise

    @classmethod
    def open(
        cls,
        store: ArtifactStore,
        ref: UnitDayArtifactRef,
        *,
        expected_context: ArtifactContext,
        metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
        verification: Literal["lazy_digest"] = "lazy_digest",
    ) -> _ArtifactFacadeSource:
        if verification != "lazy_digest":
            refuse(artifact_query_ports().reader.LAZY_DIGEST_VERIFICATION)
        snapshot_context = open_trusted_manifest_snapshot(
            store, ref, expected_context=expected_context
        )
        snapshot, manifest = snapshot_context.__enter__()
        try:
            return cls(
                store,
                snapshot_context,
                snapshot,
                manifest,
                expected_context=expected_context,
                metrics=metrics,
            )
        except BaseException:
            snapshot_context.__exit__(*cast(Any, (None, None, None)))
            raise

    @property
    def artifact_experiment(self) -> Experiment:
        return self._artifact_experiment

    def _extension(self, request: Mapping[str, Any]) -> Any:
        if (
            request.get("kind") in {"breakout_dimension", "factor_dimension"}
            and request.get("source_name") is None
        ):
            candidates = [
                extension
                for extension in self._manifest.extensions
                if extension.kind == request["kind"]
                and _artifact_request(extension).get("property_name") == request["property_name"]
            ]
            if len(candidates) != 1:
                code = (
                    "artifact.extension.missing" if not candidates else "artifact.extension.invalid"
                )
                raise ArtifactContractError(
                    code,
                    "dimension does not select exactly one artifact extension",
                )
            return candidates[0]
        wanted = json.dumps(dict(request), sort_keys=True, separators=(",", ":"), default=str)
        matches = [
            extension
            for extension in self._manifest.extensions
            if json.dumps(
                _artifact_request(extension), sort_keys=True, separators=(",", ":"), default=str
            )
            == wanted
        ]
        if len(matches) != 1:
            if request.get("kind") == "trigger_measure_stats" and not matches:
                raise ArtifactContractError(
                    "artifact.evidence.unavailable",
                    "triggered outcomes require raw-event-filtered trigger-measure statistics",
                    context={
                        "extension_kind": "trigger_measure_stats",
                        "operation": "triggered_source",
                        "metric_name": (request.get("metric_names") or (None,))[0],
                        "trigger_name": request.get("trigger_name"),
                        "route": "use definitions input or republish with this metric in trigger_measure_stats",
                    },
                )
            code = "artifact.extension.missing" if not matches else "artifact.extension.invalid"
            raise ArtifactContractError(
                code,
                "requested artifact extension is not selected exactly once",
            )
        return matches[0]

    def _trigger_measure_table(self, metric: Metric) -> tuple[Any, dt.date, dt.date | None]:
        trigger_name = self._artifact_experiment.trigger
        if trigger_name is None:
            raise ArtifactContractError(
                "artifact.evidence.unavailable", "experiment has no trigger"
            )
        extension = self._extension(
            {
                "kind": "trigger_measure_stats",
                "trigger_name": trigger_name,
                "metric_names": (metric.name,),
            }
        )
        trigger = self._extension({"kind": "trigger_population", "trigger_name": trigger_name})
        if (
            extension.trigger_extension_version != 3
            or extension.trigger_population_content_sha256 != trigger.relation.content_sha256
            or extension.observation_cutoff_ts != trigger.observation_cutoff_ts
        ):
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "trigger-measure evidence is not bound to the pinned trigger population",
                context={
                    "extension_kind": "trigger_measure_stats",
                    "operation": "triggered_source",
                },
            )
        relation = self._verified_extension_relation(
            extension, request=_artifact_request(extension)
        )
        binding = next(
            item for item in self._manifest.metric_measures if item.metric_name == metric.name
        )
        if isinstance(metric, RatioMetric):
            if not isinstance(binding, RatioMetricMeasure):
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "trigger-measure metric binding does not match its declared metric",
                    context={
                        "extension_kind": "trigger_measure_stats",
                        "operation": "triggered_source",
                        "metric_name": metric.name,
                    },
                )
            window_days_by_key = {
                binding.numerator_measure_key: metric.numerator.window_days,
                binding.denominator_measure_key: metric.denominator.window_days,
            }
        else:
            if not isinstance(binding, SimpleMetricMeasure):
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "trigger-measure metric binding does not match its declared metric",
                    context={
                        "extension_kind": "trigger_measure_stats",
                        "operation": "triggered_source",
                        "metric_name": metric.name,
                    },
                )
            window_days_by_key = {binding.measure_key: metric.window_days}
        anchors = self._verify_trigger_population(operation="triggered_source")
        outside = relation.anti_join(
            anchors.select("experiment_id", "unit_id").distinct(),
            ["experiment_id", "unit_id"],
        )
        outside_count = _artifact_rows(
            self._snapshot.execute(outside.aggregate(n_invalid=outside.count()))
        )[0]["n_invalid"]
        if outside_count:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "trigger-measure row is outside its membership or measure domain",
                context={
                    "extension_kind": "trigger_measure_stats",
                    "operation": "triggered_source",
                },
            )
        joined = relation.join(
            anchors.select("experiment_id", "unit_id", "first_trigger_ts"),
            ["experiment_id", "unit_id"],
        )
        trigger_day = _local_date(joined.first_trigger_ts, self._artifact_experiment)
        cutoff_day = (
            extension.observation_cutoff_ts + self._artifact_experiment.day_boundary_offset
        ).date()
        valid = ibis.literal(False)
        for measure_key, window_days in window_days_by_key.items():
            in_window = joined.ds >= trigger_day
            if window_days is not None:
                in_window = in_window & (joined.ds < trigger_day + ibis.interval(days=window_days))
            valid = valid | ((joined.measure_key == measure_key) & in_window)
        observation_horizon = self._artifact_experiment.observation_horizon_day
        if observation_horizon is not None:
            valid = valid & (joined.ds <= observation_horizon)
        valid = valid & (joined.ds <= cutoff_day)
        invalid_count = _artifact_rows(
            self._snapshot.execute(joined.aggregate(n_invalid=(~valid).cast("int64").sum()))
        )[0]["n_invalid"]
        if invalid_count:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "trigger-measure date is outside its trigger-relative window or pinned cutoff",
                context={
                    "extension_kind": "trigger_measure_stats",
                    "operation": "triggered_source",
                },
            )
        observed_edge, certified_edge = self.triggered_observation_edges(metric)
        return relation, observed_edge, certified_edge

    def _triggered_reduction_inputs(self, metric: Metric) -> Any:
        relation, observed_edge, certified_edge = self._trigger_measure_table(metric)
        anchors = self._verify_trigger_population(operation="triggered_source")
        assignments = self._ensure("exposures")
        joined = assignments.join(
            anchors,
            [
                assignments.experiment_id == anchors.experiment_id,
                assignments.unit_id == anchors.unit_id,
            ],
        )
        triggered_exposures = joined.select(
            **{
                name: (
                    anchors.first_trigger_ts if name == "first_exposure_ts" else assignments[name]
                )
                for name in assignments.columns
            },
            __uptake_first_exposure_date=_local_date(
                assignments.first_exposure_ts, self._artifact_experiment
            ),
            __uptake_first_exposure_ts=assignments.first_exposure_ts,
        )
        spine_edge = ibis.literal(observed_edge, type="date")
        return _new_triggered_reduction_inputs(
            exposures=triggered_exposures.mutate(
                first_exposure_date=_local_date(
                    triggered_exposures.first_exposure_ts, self._artifact_experiment
                )
            ),
            measure_stats=relation,
            observed_edge=observed_edge,
            finalized_as_of=certified_edge,
            spine_edge=spine_edge,
        )

    def _reduce(
        self,
        metric: Metric,
        grain: Grain,
        *,
        by: str | None = None,
        properties_table: ir.Table | None = None,
        cluster: str | None = None,
        cluster_table: ir.Table | None = None,
        pre_stats: ir.Table | None = None,
        population_units: frozenset[str] | ir.Table | None = None,
        completed_windows_only: bool = False,
        calendar_end: dt.date | None = None,
    ) -> list[dict[str, Any]]:
        if self._population_units is None:
            return super()._reduce(
                metric,
                grain,
                by=by,
                properties_table=properties_table,
                cluster=cluster,
                cluster_table=cluster_table,
                pre_stats=pre_stats,
                population_units=population_units,
                completed_windows_only=completed_windows_only,
                calendar_end=calendar_end,
            )
        return self.reduce_triggered(
            metric,
            grain,
            inputs=self._triggered_reduction_inputs(metric),
            by=by,
            properties_table=properties_table,
            cluster=cluster,
            cluster_table=cluster_table,
            pre_stats=pre_stats,
            population_units=population_units,
            completed_windows_only=completed_windows_only,
        )

    def _exposure_rows(self) -> list[dict[str, Any]]:
        exposures = self._ensure("exposures")
        if self._population_units is not None:
            exposures = restrict_to_units(exposures, self._population_units)
        return _artifact_rows(self._snapshot.execute(exposures))

    def _read_extension(
        self, extension: Any, *, request: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]:
        if isinstance(self._population_units, ir.Table):
            if extension.kind in {"assignment_counts", "site_volume"}:
                return list(
                    read_extension(
                        self._snapshot,
                        extension,
                        request=request,
                        context=self._manifest.context,
                        experiment_id=self._manifest.experiment_id,
                        exposure_relation=self._ensure("exposures"),
                    )
                )
            relation = self._verified_extension_relation(
                extension, request=request or _artifact_request(extension)
            )
            membership = self._population_units.select("experiment_id", "unit_id").distinct()
            if extension.kind == "trigger_measure_stats":
                outside = relation.anti_join(membership, ["experiment_id", "unit_id"])
                outside_count = _artifact_rows(
                    self._snapshot.execute(outside.aggregate(n_invalid=outside.count()))
                )[0]["n_invalid"]
                if outside_count:
                    raise ArtifactContractError(
                        "artifact.extension.invalid",
                        "trigger-measure rows are not a subset of trigger membership",
                        context={
                            "extension_kind": extension.kind,
                            "operation": "triggered_source",
                        },
                    )
            if extension.kind not in {"site_volume", "assignment_counts"}:
                relation = relation.semi_join(membership, ["experiment_id", "unit_id"])
            return cast(list[Mapping[str, Any]], _artifact_rows(self._snapshot.execute(relation)))
        exposures = self._ensure("exposures")
        result = list(
            read_extension(
                self._snapshot,
                extension,
                request=request,
                context=self._manifest.context,
                experiment_id=self._manifest.experiment_id,
                exposure_relation=exposures,
            )
        )
        if extension.kind == "trigger_measure_stats" and self._population_units is not None:
            if any(str(row["unit_id"]) not in self._population_units for row in result):
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    "trigger-measure rows are not a subset of trigger membership",
                    context={"extension_kind": extension.kind, "operation": "triggered_source"},
                )
        if self._population_units is not None and extension.kind not in {
            "site_volume",
            "assignment_counts",
        }:
            result = [row for row in result if str(row.get("unit_id")) in self._population_units]
        return result

    def _verified_extension_relation(
        self, extension: Any, *, request: Mapping[str, Any] | None = None
    ) -> ir.Table:
        return cast(
            ir.Table,
            verify_extension_relation(
                self._snapshot,
                extension,
                request=request,
                context=self._manifest.context,
                experiment_id=self._manifest.experiment_id,
                exposure_relation=self._ensure("exposures"),
            ),
        )

    def _trigger_population_rows(
        self, *, operation: str = "triggered_counts"
    ) -> list[Mapping[str, Any]]:
        extension = self._extension(
            {"kind": "trigger_population", "trigger_name": self._artifact_experiment.trigger}
        )
        rows = self._read_extension(extension, request=_artifact_request(extension))
        if any(row["first_trigger_ts"] > extension.observation_cutoff_ts for row in rows):
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "trigger membership contains an event after the pinned observation cutoff",
                context={
                    "extension_kind": "trigger_population",
                    "operation": operation,
                    "route": "republish from trusted definitions with matching trigger evidence",
                },
            )
        return rows

    def _verify_trigger_population(self, *, operation: str) -> ir.Table:
        extension = self._extension(
            {"kind": "trigger_population", "trigger_name": self._artifact_experiment.trigger}
        )
        relation = self._verified_extension_relation(
            extension, request=_artifact_request(extension)
        )
        after_cutoff = relation.first_trigger_ts > extension.observation_cutoff_ts
        invalid_count = _artifact_rows(
            self._snapshot.execute(relation.aggregate(n_invalid=after_cutoff.cast("int64").sum()))
        )[0]["n_invalid"]
        if invalid_count:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "trigger membership contains an event after the pinned observation cutoff",
                context={
                    "extension_kind": "trigger_population",
                    "operation": operation,
                    "route": "republish from trusted definitions with matching trigger evidence",
                },
            )
        return relation

    def _trigger_group_counts(self, *, operation: str) -> dict[str, int]:
        membership = self._verify_trigger_population(operation=operation)
        exposures = self._ensure("exposures")
        joined = membership.join(
            exposures.select("experiment_id", "unit_id", "group_id"),
            ["experiment_id", "unit_id"],
        )
        summary = joined.group_by("group_id").aggregate(n_units=joined.unit_id.count())
        return {
            str(row["group_id"]): int(row["n_units"])
            for row in _artifact_rows(self._snapshot.execute(summary))
        }

    def _validate_trigger_population_counts(
        self, count_rows: Sequence[Mapping[str, Any]], observed: Mapping[str, int]
    ) -> None:
        declared = {
            str(row["group_id"]): int(row["n_units"])
            for row in count_rows
            if row["population"] == "triggered"
        }
        groups = declared.keys() | observed.keys()
        if any(declared.get(group, 0) != observed.get(group, 0) for group in groups):
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "trigger membership and triggered assignment counts disagree",
                context={
                    "extension_kind": "assignment_counts",
                    "operation": "triggered_counts",
                    "route": "republish with consistent trigger_population and assignment_counts evidence",
                },
            )

    def _dimension_table(self, extension: Any, request: Mapping[str, Any]) -> Any:
        relation = self._verified_extension_relation(extension, request=request)
        kind = extension.kind
        value_field = "factor_value" if kind == "factor_dimension" else "dimension_value"
        name = request["property_name"]
        value = ibis.ifelse(
            relation.value_is_missing, ibis.null().cast("string"), relation[value_field]
        ).name(name)
        return relation.select("unit_id", value)

    def _cluster_table(self) -> ir.Table | None:
        """Resolve declared cluster identities for total-grain reductions."""
        if self.context.cluster is None:
            return None
        cluster_extension = self._extension(
            {"kind": "cluster_identity", "cluster_name": self.context.cluster}
        )
        relation = self._verified_extension_relation(
            cluster_extension, request=_artifact_request(cluster_extension)
        )
        return relation.select("unit_id", relation.cluster_id.name(self.context.cluster))

    def _cuped_pre_stats(self, metric: Metric) -> Any:
        """Resolve *metric*'s `cuped_preperiod` extension into a covariate table."""
        cuped = self._extension({"kind": "cuped_preperiod", "metric_name": metric.name})
        relation = self._verified_extension_relation(cuped, request=_artifact_request(cuped))
        return relation.select("experiment_id", "unit_id", sum_value=relation.x)

    def _preflight_cluster_grain(self, grain: Grain) -> None:
        if self.context.cluster is not None and grain != "total":
            from increment._evidence_refusals import refuse_grain

            refuse_grain(
                self,
                grain,
                ("total",),
                route="request grain='total' for cluster-robust inference",
            )

    def _dimensioned_moments(
        self,
        metric: Metric,
        dimension: str,
        *,
        source_name: str | None = None,
        grain: Grain = "total",
        completed_windows_only: bool = False,
        include_covariate: bool = False,
        kind: str = "breakout_dimension",
    ) -> list[dict[str, Any]]:
        self._preflight_cluster_grain(grain)
        from increment._evidence_refusals import require_evidence

        require_evidence(
            self,
            grain=grain,
            by=(dimension,) if kind == "breakout_dimension" else (),
        )
        # Dimensioned callers can be the first read on a fresh artifact source.
        metric = self.validated_metric(metric)
        extension = self._extension(
            {
                "kind": kind,
                "property_name": dimension,
                "source_name": source_name,
            }
        )
        request = _artifact_request(extension)
        properties = self._dimension_table(extension, request)
        pre_stats = self._cuped_pre_stats(metric) if include_covariate else None
        cluster_table = self._cluster_table()
        return self._reduce(
            metric,
            grain,
            by=dimension,
            properties_table=properties,
            cluster=self.context.cluster,
            cluster_table=cluster_table,
            pre_stats=pre_stats,
            population_units=self._population_units,
            completed_windows_only=completed_windows_only,
        )

    def validated_metric(self, metric: Metric) -> Metric:
        """The manifest-bound metric for *metric*'s name, refusing a caller copy whose
        semantics differ. Reads no evidence, so estimators can authenticate before refusing."""
        specs = self._metric_specs()
        spec = next((candidate for candidate in specs if candidate.name == metric.name), None)
        if spec is None:
            raise ArtifactContractError(
                "artifact.metric.binding_mismatch",
                f"metric {metric.name!r} has no trusted manifest binding",
            )
        trusted = self._trusted_metric(spec)
        mismatch = type(metric) is not type(trusted) or any(
            getattr(metric, field, None) != getattr(trusted, field, None)
            for field in (
                "aggregation",
                "window_days",
                "threshold_days",
                "quantile",
                "winsorization",
            )
        )
        if isinstance(metric, RatioMetric) and isinstance(trusted, RatioMetric):
            mismatch = mismatch or any(
                getattr(actual, field, None) != getattr(expected, field, None)
                for actual, expected in (
                    (metric.numerator, trusted.numerator),
                    (metric.denominator, trusted.denominator),
                )
                for field in ("aggregation", "window_days")
            )
        if mismatch:
            raise ArtifactContractError(
                "artifact.metric.binding_mismatch",
                f"metric {metric.name!r} differs from trusted manifest semantics",
            )
        return trusted

    def _refuse_observational_quantile(self, metric: Metric) -> None:
        """An observational design has no quantile estimator on any ingress path."""
        if getattr(metric, "type", None) == "quantile" and (
            getattr(self.context.design, "mechanism", None) == "observational"
        ):
            refuse_observational_quantile(metric)

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

    def method_fit_input_digest(self, method_request) -> str:
        from increment.query.cuped import method_fit_input_digest

        return method_fit_input_digest(self, method_request)

    @contextmanager
    def cuped_relation(self, metric: Metric, request):
        """Expose only a verified warehouse relation to compact CUPED reduction."""
        from increment.query._cuped_cells import general_qr_required

        binding = next(
            (item for item in self._manifest.metric_measures if item.metric_name == metric.name),
            None,
        )
        if binding is None:
            general_qr_required(metric, "outcome_domain")
        keys = (
            (binding.numerator_measure_key, binding.denominator_measure_key)
            if binding.kind == "ratio"
            else (binding.measure_key,)
        )
        domains = {item.measure_key: item.outcome_domain for item in self._manifest.measures}
        if any(domains.get(key, "unknown") not in ("continuous", "discrete") for key in keys):
            general_qr_required(metric, "outcome_domain")
        if request.population != self._population:
            if request.population == "triggered" and self._population == "assigned":
                triggered = cast("_ArtifactFacadeSource", self.triggered_source())
                with triggered.cuped_relation(metric, request) as evidence:
                    yield evidence
                return
            general_qr_required(metric, "source_population")
        relation = self._unit_relation_with_covariates(
            metric, covariates=tuple(item.name for item in request.method.covariates or ())
        )
        if request.breakout_dimension is not None:
            extension = self._extension(
                {
                    "kind": "breakout_dimension",
                    "property_name": request.breakout_dimension,
                    "source_name": request.breakout_source,
                }
            )
            extension_relation = self._verified_extension_relation(
                extension, request=_artifact_request(extension)
            )
            dimension = extension_relation.select(
                "unit_id",
                **{
                    "__qr_breakout_missing": extension_relation.value_is_missing,
                    "__qr_breakout_value": extension_relation.dimension_value,
                },
            )
            relation = relation.left_join(dimension, "unit_id")
            if request.breakout_is_null:
                relation = relation.filter(relation["__qr_breakout_missing"])
            else:
                relation = relation.filter(
                    ~relation["__qr_breakout_missing"]
                    & (relation["__qr_breakout_value"] == request.breakout_value)
                )
        yield relation, self._snapshot.execute, getattr(self._snapshot, "qr_scratch", None)

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> Any:
        return self._snapshot.execute(
            self._unit_relation_with_covariates(
                metric, covariates=covariates, outcome_stage=outcome_stage
            )
        )

    def _unit_relation_with_covariates(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> Any:
        cluster = self.context.cluster
        if "cluster_id" in covariates and cluster != "cluster_id":
            refuse(_NATIVE_COVARIATE_RESERVED, covariate="cluster_id", cluster=cluster)
        # A requested declared cluster column reuses the attached metadata column.
        requested = [name for name in dict.fromkeys(covariates) if name != "cluster_id"]
        metric = self.validated_metric(metric)
        extensions = {name: self._covariate_extension(metric.name, name) for name in requested}
        covariate_tables: dict[str, tuple[ir.Table, str, str | None]] = {}
        for name, extension in extensions.items():
            relation = self._verified_extension_relation(
                extension, request=_artifact_request(extension)
            )
            field = "level" if extension.kind == "unit_covariate_level" else "value"
            available = (
                "has_preperiod_events" if extension.kind == "cuped_measure_covariate" else None
            )
            covariate_tables[name] = (relation, field, available)
        if self._population_units is None:
            return self._unit_relation(
                metric,
                cluster=cluster,
                resolve_cluster=self._cluster_table,
                population_units=self._population_units,
                outcome_stage=outcome_stage,
                covariate_tables=covariate_tables,
            )
        return self._unit_relation(
            metric,
            cluster=cluster,
            resolve_cluster=self._cluster_table,
            population_units=self._population_units,
            outcome_stage=outcome_stage,
            trigger_inputs=self._triggered_reduction_inputs(metric),
            covariate_tables=covariate_tables,
        )

    @property
    def scalar_bindings(self):
        """Published raw covariate recipes, independent of runtime method labels."""
        return {
            config.metric.name: tuple(
                covariate
                for method in effective_methods(config, design=self.context.design)
                for covariate in method.covariates or ()
            )
            for config in self.context.configs
        }

    def moments_for_method(
        self,
        metric: Metric,
        method: Any,
        *,
        by: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Build one scalar family from the method's declared artifact covariate."""
        if by:
            raise ArtifactContractError(
                "artifact.extension.missing",
                "artifact base has no breakout evidence extension",
            )
        from increment._analysis_config import overlay_configs, validate_scalar_route

        validate_scalar_route(
            overlay_configs((metric,), self.context.configs, methods=[method], prior=None),
            route="artifact.moments_for_method",
            design=self.context.design,
            bindings=self.scalar_bindings,
        )
        trusted = self.validated_metric(metric)
        if method.variance_reduction != "cuped":
            return self._reduce(
                trusted,
                "total",
                cluster=self.context.cluster,
                cluster_table=self._cluster_table(),
                population_units=self._population_units,
            )
        covariates = method.covariates or ()
        if not covariates:
            return self.moments(metric, grain="total", include_covariate=True)
        covariate = covariates[0]
        extension = self._covariate_extension(metric.name, covariate.name)
        covariate_relation = self._verified_extension_relation(
            extension, request=_artifact_request(extension)
        )
        if self._population_units is not None:
            covariate_relation = restrict_to_units(covariate_relation, self._population_units)
        is_missing = covariate_relation.value.isnull()
        if "value_is_missing" in covariate_relation.columns:
            is_missing = is_missing | covariate_relation.value_is_missing
        summary = covariate_relation.aggregate(
            missing_count=is_missing.cast("int64").sum(),
            observed_count=(~is_missing).cast("int64").sum(),
        )
        stats = _artifact_rows(self._snapshot.execute(summary))[0]
        missing_count = int(stats["missing_count"] or 0)
        missing_policy = getattr(covariate, "missing", "zero")
        if missing_policy == "refuse" and missing_count:
            from increment._frame_validation import _raise as frame_raise

            frame_raise(
                "frame.validation.metric_covariate_column",
                metric=metric.name,
                covariate=covariate.name,
                n_missing=missing_count,
            )
        if missing_policy in ("impute", "impute-indicator"):
            if not stats["observed_count"]:
                from increment._frame_validation import _raise as frame_raise

                frame_raise(
                    "frame.validation.metric_covariate_impute_all_null",
                    metric=metric.name,
                    covariate=covariate.name,
                )
            observed_relation = covariate_relation.filter(~is_missing)
            mean_value = _artifact_rows(
                self._snapshot.execute(
                    observed_relation.aggregate(mean_value=observed_relation.value.mean())
                )
            )[0]["mean_value"]
            pre_stats = covariate_relation.select(
                "experiment_id",
                "unit_id",
                ibis.ifelse(is_missing, mean_value, covariate_relation.value).name("sum_value"),
            )
        else:
            pre_stats = covariate_relation.select(
                "experiment_id",
                "unit_id",
                ibis.ifelse(is_missing, ibis.null(), covariate_relation.value).name("sum_value"),
            )
        return self._reduce(
            trusted,
            "total",
            cluster=self.context.cluster,
            cluster_table=self._cluster_table(),
            pre_stats=pre_stats,
            population_units=self._population_units,
        )

    def _covariate_extension(self, metric_name: str, name: str) -> Any:
        property_name = name
        source_name = None
        measure_bound = False
        context = json.loads(self._manifest.context.canonical_json)
        experiment_context = context.get("experiment", {})
        design = experiment_context.get("design") or {}
        design_covariate = next(
            (
                covariate
                for covariate in design.get("covariates", ())
                if covariate.get("property") == name
            ),
            None,
        )
        design_bound = design_covariate is not None
        if design_covariate is not None:
            source_name = design_covariate.get("source")
        else:
            plan = experiment_context.get("plan", {})
            primary = plan.get("primary")
            entries = list(primary) if isinstance(primary, list) else [primary]
            entries.extend(plan.get("secondaries") or ())
            entries.extend(plan.get("guardrails") or ())
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("metric") != metric_name:
                    continue
                methods = [
                    method
                    for method in (
                        entry.get("decision_method"),
                        *cast(Sequence[Any], entry.get("sensitivity_methods") or ()),
                    )
                    if isinstance(method, dict)
                ]
                for method in methods:
                    for covariate in cast(Sequence[Any], method.get("covariates") or ()):
                        if covariate.get("name") != name:
                            continue
                        if covariate.get("property") is not None:
                            property_name = covariate["property"]
                            source_name = covariate.get("source")
                        elif covariate.get("measure") is not None:
                            measure_bound = True
                        break
        if not measure_bound and source_name is None:
            design = experiment_context.get("design") or {}
            for covariate in design.get("covariates", ()):
                if covariate.get("property") == property_name:
                    source_name = covariate.get("source")
                    break

        matches: list[Any] = []
        if not measure_bound:
            matches.extend(
                extension
                for extension in self._manifest.extensions
                if isinstance(extension, (UnitCovariateExtension, UnitCovariateLevelExtension))
                and extension.covariate_name == property_name
                and (source_name is None or extension.source_name == source_name)
            )
        if not design_bound:
            matches.extend(
                extension
                for extension in self._manifest.extensions
                if extension.kind == "cuped_measure_covariate"
                and extension.metric_name == metric_name
                and extension.covariate_name == name
            )
        if len(matches) == 1:
            return matches[0]
        code = "artifact.extension.missing" if not matches else "artifact.extension.invalid"
        refuse(
            _ARTIFACT_REFUSALS[code],
            message=(
                f"artifact carries {'no' if not matches else 'more than one'} unit "
                f"covariate {name!r} for metric {metric_name!r}. Republish the artifact "
                "from the matching definitions so it includes this declared covariate."
            ),
            covariate=name,
            metric=metric_name,
            route="republish the artifact from the definitions that declare this CUPED covariate",
        )

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        self._preflight_cluster_grain(grain)
        from increment._evidence_refusals import require_evidence

        require_evidence(self, grain=grain, by=tuple(by))

        metric = self.validated_metric(metric)
        self._refuse_observational_quantile(metric)
        if not include_covariate:
            config = next(
                (
                    candidate
                    for candidate in self.context.configs
                    if candidate.metric.name == metric.name
                ),
                None,
            )
            methods = effective_methods(config, design=self.context.design) if config else ()
            if by and any(
                method.variance_reduction == "cuped"
                and any(
                    getattr(covariate, field, None) is not None
                    for covariate in method.covariates or ()
                    for field in ("property", "measure", "source")
                )
                for method in methods
            ):
                raise ArtifactContractError(
                    "artifact.extension.missing",
                    "artifact base has no binding-aware scalar breakout moments",
                )
            include_covariate = any(
                method.variance_reduction == "cuped"
                and (
                    not method.covariates
                    or any(
                        not any(
                            getattr(covariate, field, None) is not None
                            for field in ("property", "measure", "source")
                        )
                        for covariate in method.covariates
                    )
                )
                for method in methods
            )
        if (
            grain == "asof"
            and completed_windows_only
            and isinstance(metric, RetentionMetric)
            and metric.band[1] is None
        ):
            raise ArtifactContractError(
                "artifact.evidence.unavailable",
                "completed-window filtering is unavailable for unbounded retention",
            )
        if grain == "daily" and isinstance(metric, RetentionMetric) and metric.band[1] is None:
            raise ArtifactContractError(
                "artifact.evidence.unavailable",
                "daily evidence is unavailable for unbounded retention",
            )
        if len(by) > 1:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "artifact moments support one dimension at a time",
            )
        if include_covariate and not self._manifest.extensions:
            raise ArtifactContractError(
                "artifact.extension.missing",
                "requested CUPED evidence extension is absent",
            )
        if by:
            extension = self._extension(
                {"kind": "breakout_dimension", "property_name": by[0], "source_name": None}
            )
            return self._dimensioned_moments(
                metric,
                by[0],
                source_name=extension.source_name,
                grain=grain,
                completed_windows_only=completed_windows_only,
                include_covariate=include_covariate,
            )
        pre_stats = self._cuped_pre_stats(metric) if include_covariate else None
        cluster_table = self._cluster_table()
        return self._reduce(
            metric,
            grain,
            cluster=self.context.cluster,
            cluster_table=cluster_table,
            pre_stats=pre_stats,
            population_units=self._population_units,
            completed_windows_only=completed_windows_only,
        )

    def breakout_moments(
        self,
        metric: Metric,
        breakout: Breakout,
        *,
        grain: Grain = "daily",
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> Any:
        self._preflight_cluster_grain(grain)
        from increment._evidence_refusals import require_evidence

        require_evidence(self, grain=grain, by=(breakout.property,))
        extension = self._extension(
            {
                "kind": "breakout_dimension",
                "property_name": breakout.property,
                "source_name": breakout.source,
            }
        )
        rows = self._dimensioned_moments(
            metric,
            breakout.property,
            source_name=extension.source_name,
            grain=grain,
            completed_windows_only=completed_windows_only,
            include_covariate=include_covariate,
        )
        return DimensionedDayEvidence(
            source_name=extension.source_name,
            rows=rows,
        )

    def _require_extension_coverage(self, kind: str) -> None:
        entries = [
            entry
            for entry in unit_day_artifact_extension_catalog(self._expected_context)
            if entry.request.kind == kind
        ]
        for entry in entries:
            if not any(
                extension.kind == kind
                and extension.definition_sha256 == entry.definition_sha256
                and extension.source_provenance_sha256 == entry.source_provenance_sha256
                for extension in self._manifest.extensions
            ):
                raise ArtifactContractError(
                    "artifact.extension.missing",
                    f"artifact is missing trusted {kind} evidence",
                )

    def assigned_breakout_values(
        self, breakout: Breakout, *, source_name: str | None = None
    ) -> Sequence[str]:
        """Return dimension values among assigned units without reducing outcomes."""
        extension = self._extension(
            {
                "kind": "breakout_dimension",
                "property_name": breakout.property,
                "source_name": breakout.source,
            }
        )
        if source_name is not None and extension.source_name != source_name:
            return []
        request = _artifact_request(extension)
        properties = self._dimension_table(extension, request)
        properties = _restore_raw_dimension_values(properties, breakout.property)
        if properties is None:
            return []
        exposures = self._ensure("exposures")
        joined = join_breakout_dimension(exposures, properties, [breakout.property])
        rows = _artifact_rows(self._snapshot.execute(joined.select(breakout.property).distinct()))
        return sorted(
            {str(row[breakout.property]) for row in rows if row.get(breakout.property) is not None}
        )

    def breakout_unit_frame(
        self,
        metric: Metric,
        *,
        breakout: Breakout,
        covariates: Sequence[str] = (),
        population: Literal["assigned", "triggered"] = "assigned",
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> Any:
        import narwhals as nw
        import pyarrow as pa

        from increment._source_types import breakout_dimension_column

        if population == "triggered" and self._population_units is None:
            return cast("_ArtifactFacadeSource", self.triggered_source()).breakout_unit_frame(
                metric,
                breakout=breakout,
                covariates=covariates,
                population=population,
                outcome_stage=outcome_stage,
            )
        units = nw.from_native(
            self.unit_frame(metric, covariates=covariates, outcome_stage=outcome_stage),
            eager_only=True,
        )
        extension = self._extension(
            {
                "kind": "breakout_dimension",
                "property_name": breakout.property,
                "source_name": breakout.source,
            }
        )
        rows = self._read_extension(extension, request=_artifact_request(extension))
        values = {
            str(row["unit_id"]): None if row.get("value_is_missing") else row["dimension_value"]
            for row in rows
        }
        dimension = nw.from_native(
            pa.table(
                {
                    "unit_id": [str(value) for value in units["unit_id"].to_list()],
                    breakout_dimension_column(covariates): [
                        values.get(str(value)) for value in units["unit_id"].to_list()
                    ],
                }
            ),
            eager_only=True,
        )
        return units.join(dimension, on="unit_id", how="left").to_native()

    def breakout_source(
        self,
        breakout: Breakout,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
        configs: Sequence[ResolvedMetricConfig] | None = None,
    ) -> BreakoutMomentsSource:
        configs = self.context.configs if configs is None else configs
        for config in configs:
            if config.metric not in metrics:
                continue
            for method in effective_methods(config, design=self.context.design):
                if method.variance_reduction == "cuped" and method.covariates:
                    raise ArtifactContractError(
                        "artifact.cuped.scalar_breakout",
                        "Named scalar CUPED breakout evidence is unavailable from this artifact "
                        "route. Run the scalar headline, or use definitions or frame input "
                        "for the breakout.",
                        context={
                            "metric": config.metric.name,
                            "method": method.name,
                            "route": "from_unit_day_artifact.run_breakout",
                        },
                    )
        if population == "triggered" and self._population_units is None:
            return cast(
                "_ArtifactFacadeSource",
                self.triggered_source(),
            ).breakout_source(breakout, metrics=metrics, population=population, configs=configs)
        extension = self._extension(
            {
                "kind": "breakout_dimension",
                "property_name": breakout.property,
                "source_name": breakout.source,
            }
        )
        source_name = extension.source_name
        cuped_names = {
            config.metric.name
            for config in configs
            if any(
                method.variance_reduction == "cuped"
                for method in effective_methods(config, design=self.context.design)
            )
        }
        rows = [
            row
            for metric in metrics
            for row in self._dimensioned_moments(
                metric,
                breakout.property,
                source_name=source_name,
                grain="total",
                include_covariate=metric.name in cuped_names,
            )
        ]
        return BreakoutMomentsSource(
            rows,
            dimension=breakout.property,
            metrics=metrics,
            study_id=self.context.study_id,
            source_name=source_name,
            design=self.context.design,
            plan=self.context.plan,
            configs=configs,
            _source=self,
            population=population,
        )

    def breakout_sources(
        self,
        breakouts: Sequence[Breakout],
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
        configs: Sequence[ResolvedMetricConfig] | None = None,
    ) -> tuple[BreakoutMomentsSource, ...]:
        self._require_extension_coverage("breakout_dimension")
        return tuple(
            self.breakout_source(breakout, metrics=metrics, population=population, configs=configs)
            for breakout in breakouts
        )

    def moments_source(
        self,
        *,
        metrics: Sequence[Metric],
        methods: list[Any] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        narrow_cuped: bool = False,
    ) -> _ArtifactFacadeSource:
        if population == "triggered":
            return cast("_ArtifactFacadeSource", self.triggered_source())
        return self

    def validate_populations(
        self,
        populations: Sequence[Literal["assigned", "triggered"]],
        *,
        operation: str,
    ) -> None:
        """Validate requested populations against the pinned artifact evidence."""
        if "triggered" in populations and self._population_units is None:
            self._verify_trigger_population(operation=operation)

    def breakout_summaries(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, dict[str, Any]]:
        if population == "triggered" and self._population_units is None:
            return cast(
                "_ArtifactFacadeSource",
                self.triggered_source(),
            ).breakout_summaries(metrics=metrics)
        self._require_extension_coverage("breakout_dimension")
        import pyarrow as pa

        cuped_names = {
            config.metric.name
            for config in self.context.configs
            if any(
                method.variance_reduction == "cuped"
                for method in effective_methods(config, design=self.context.design)
            )
        }

        result: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            for extension in self._manifest.extensions:
                if extension.kind != "breakout_dimension":
                    continue
                total = self._dimensioned_moments(
                    metric,
                    extension.dimension_name,
                    source_name=extension.source_name,
                    grain="total",
                    include_covariate=metric.name in cuped_names,
                )
                daily = self._dimensioned_moments(
                    metric,
                    extension.dimension_name,
                    source_name=extension.source_name,
                    grain="daily",
                    include_covariate=metric.name in cuped_names,
                )
                result[f"{metric.name}:{extension.dimension_name}:{extension.source_name}"] = {
                    "group_summary": pa.Table.from_pylist(total),
                    "daily_group_summary": pa.Table.from_pylist(daily),
                }
        return result

    def factor_summaries(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> dict[str, Any]:
        if population == "triggered" and self._population_units is None:
            return cast(
                "_ArtifactFacadeSource",
                self.triggered_source(),
            ).factor_summaries(metrics=metrics)
        self._require_extension_coverage("factor_dimension")
        import pyarrow as pa

        cuped_names = {
            config.metric.name
            for config in self.context.configs
            if any(
                method.variance_reduction == "cuped"
                for method in effective_methods(config, design=self.context.design)
            )
        }

        result: dict[str, Any] = {}
        for metric in metrics:
            for extension in self._manifest.extensions:
                if extension.kind != "factor_dimension":
                    continue
                rows = self._dimensioned_moments(
                    metric,
                    extension.factor_name,
                    source_name=extension.source_name,
                    grain="total",
                    kind="factor_dimension",
                    include_covariate=metric.name in cuped_names,
                )
                result[f"{metric.name}:{extension.factor_name}:{extension.source_name}"] = (
                    pa.Table.from_pylist(rows)
                )
        return result

    def _assignment_counts_extension(
        self,
    ) -> tuple[Any, list[Mapping[str, Any]]] | None:
        populations = (
            ("assigned", "triggered")
            if self._artifact_experiment.trigger is not None
            else ("assigned",)
        )
        try:
            extension = self._extension({"kind": "assignment_counts", "populations": populations})
        except ArtifactContractError as exc:
            if exc.code != "artifact.extension.missing":
                raise
            return None
        return extension, self._read_extension(extension, request=_artifact_request(extension))

    def assignment_counts(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> dict[str, int]:
        if population == "triggered":
            return self.triggered_counts()[1]
        selected = self._assignment_counts_extension()
        if selected is None:
            if self.context.cluster is not None:
                try:
                    return self.cluster_counts()
                except ArtifactContractError as exc:
                    if exc.code != "artifact.extension.missing":
                        raise
                refuse(
                    _ARTIFACT_REFUSALS["artifact.evidence.unavailable"],
                    message="artifact lacks randomization-grain assignment counts",
                    population="assigned",
                    operation="assignment_counts",
                )
            return super().unit_counts()
        extension, rows = selected
        counts = {
            str(row["group_id"]): int(row["n_randomization_units"])
            for row in rows
            if row["population"] == "assigned" and row["group_id"] is not None
        }
        from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

        if extension.mixed_unit_count:
            counts[MIXED_ASSIGNMENT_LABEL] = extension.mixed_unit_count
        if extension.unassigned_unit_count:
            counts[UNASSIGNED_LABEL] = extension.unassigned_unit_count
        return counts

    def unit_counts(self) -> dict[str, int]:
        if self._population_units is not None:
            grain, counts, unit_counts = self.triggered_counts()
            return unit_counts if grain == "cluster" else counts
        selected = self._assignment_counts_extension()
        if selected is None:
            # The pinned exposure relation is sufficient for assigned unit counts.
            return super().unit_counts()
        extension, rows = selected
        counts = {
            str(row["group_id"]): int(row["n_units"])
            for row in rows
            if row["population"] == "assigned" and row["group_id"] is not None
        }
        from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

        if extension.mixed_unit_count:
            counts[MIXED_ASSIGNMENT_LABEL] = extension.mixed_unit_count
        if extension.unassigned_unit_count:
            counts[UNASSIGNED_LABEL] = extension.unassigned_unit_count
        return counts

    def cluster_counts(self) -> dict[str, int]:
        extension = self._extension(
            {"kind": "cluster_identity", "cluster_name": self.context.cluster}
        )
        rows = self._read_extension(extension, request=_artifact_request(extension))
        groups: dict[str, set[str]] = {}
        exposure_by_unit = {
            str(row["unit_id"]): str(row["group_id"])
            for row in self._exposure_rows()
            if row["group_id"] is not None
        }
        for row in rows:
            unit_id = str(row["unit_id"])
            if unit_id in exposure_by_unit:
                groups.setdefault(exposure_by_unit[unit_id], set()).add(str(row["cluster_id"]))
        return {group: len(clusters) for group, clusters in groups.items()}

    def triggered_counts(self) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]:
        extension = self._extension(
            {"kind": "assignment_counts", "populations": ("assigned", "triggered")}
        )
        rows = self._read_extension(extension, request=_artifact_request(extension))
        observed = self._trigger_group_counts(operation="triggered_counts")
        self._validate_trigger_population_counts(rows, observed)
        counts = {
            str(row["group_id"]): int(row["n_randomization_units"])
            for row in rows
            if row["population"] == "triggered" and row["group_id"] is not None
        }
        unit_counts = {
            str(row["group_id"]): int(row["n_units"])
            for row in rows
            if row["population"] == "triggered" and row["group_id"] is not None
        }
        grain: Literal["unit", "cluster"] = (
            "cluster" if self.context.cluster is not None else "unit"
        )
        return grain, counts, unit_counts if grain == "cluster" else {}

    def trigger_rates(self) -> dict[str, float]:
        grain, counts, unit_counts = self.triggered_counts()
        triggered = counts if grain == "unit" else unit_counts
        assigned = self.unit_counts()
        from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

        audit_labels = {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL}
        return {
            group: triggered.get(group, 0) / count
            for group, count in assigned.items()
            if count and group not in audit_labels
        }

    def triggered_source(self) -> EvidenceSource:
        membership = self._verify_trigger_population(operation="triggered_source")
        return _ArtifactFacadeSource(
            self._store,
            self._lifecycle,
            self._snapshot,
            self._manifest,
            expected_context=self._expected_context,
            metrics=self._metrics_arg,
            population_units=membership,
            trigger_anchors=membership,
        )

    def _triggered_sitewide_preflight(
        self, metric: Metric, extension: Any
    ) -> tuple[Any, Any, Any, frozenset[str]]:
        components = (
            (metric.numerator, metric.denominator) if isinstance(metric, RatioMetric) else (metric,)
        )
        if any(
            getattr(component, "window_days", None) is not None
            or getattr(component, "aggregation", None) not in ("sum", "count")
            for component in components
        ):
            raise ArtifactContractError(
                "artifact.evidence.unavailable",
                "triggered sitewide requires an unwindowed sum or count metric",
                context={"route": "use a calendar-aligned unwindowed sum or count metric"},
            )
        binding = next(
            item for item in self._manifest.metric_measures if item.metric_name == metric.name
        )
        measure_keys = (
            frozenset({binding.numerator_measure_key, binding.denominator_measure_key})
            if isinstance(binding, RatioMetricMeasure)
            else frozenset({binding.measure_key})
        )
        try:
            trigger_extension = self._extension(
                {
                    "kind": "trigger_population",
                    "trigger_name": self._artifact_experiment.trigger,
                }
            )
            count_extension = self._extension(
                {"kind": "assignment_counts", "populations": ("assigned", "triggered")}
            )
        except ArtifactContractError as exc:
            raise ArtifactContractError(
                "artifact.extension.missing",
                "triggered sitewide requires trigger membership and assignment counts",
                context={
                    "route": "republish with trigger_population and assignment_counts extensions"
                },
            ) from exc
        if (
            self._artifact_experiment.end is None
            or extension.first_ds != self._artifact_experiment.start_day
            or extension.last_ds != self._artifact_experiment.end_day
        ):
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "site-volume coverage does not match the experiment calendar window",
                context={"route": "republish matching site_volume and trigger_population evidence"},
            )
        self._require_sitewide_outcome_completeness(measure_keys, certified=True)
        return trigger_extension, count_extension, binding, measure_keys

    def _require_sitewide_outcome_completeness(
        self, measure_keys: frozenset[str], *, certified: bool
    ) -> None:
        """Refuse outcomes known to stop before the experiment end day.

        Triggered membership needs every edge certified. Assigned readings use
        each measure's best published edge and, like a native analysis without
        snapshot evidence, proceed when a legacy publication recorded none.
        """
        measures = [item for item in self._manifest.measures if item.measure_key in measure_keys]
        end_day = self._artifact_experiment.end_day
        if certified:
            edges = [item.certified_edge for item in measures]
            incomplete = (
                end_day is None
                or len(edges) != len(measure_keys)
                or any(edge is None or edge < end_day for edge in edges)
            )
        else:
            known = [
                item.certified_edge if item.certified_edge is not None else item.observed_edge
                for item in measures
            ]
            incomplete = end_day is not None and any(
                edge is not None and edge < end_day for edge in known
            )
        if incomplete:
            refuse(
                _SITEWIDE_OUTCOMES_INCOMPLETE,
                route=_OUTCOMES_INCOMPLETE_ROUTES[certified],
            )

    def sitewide_evidence(
        self,
        metric: Metric,
        *,
        include_ratio: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> Any:
        from increment.estimation.engine import _df_to_arms

        metric = self.validated_metric(metric)
        entry = next(
            (
                candidate
                for candidate in unit_day_artifact_extension_catalog(self._expected_context)
                if candidate.request.kind == "site_volume"
                and metric.name in candidate.request.metric_names
            ),
            None,
        )
        extension = next(
            (
                candidate
                for candidate in self._manifest.extensions
                if entry is not None
                and candidate.kind == "site_volume"
                and candidate.definition_sha256 == entry.definition_sha256
                and candidate.source_provenance_sha256 == entry.source_provenance_sha256
            ),
            None,
        )
        if extension is None:
            raise ArtifactContractError(
                "artifact.extension.missing",
                f"site-volume evidence is absent for metric {metric.name!r}",
                context={"route": "republish with this metric's site_volume extension"},
            )
        if population == "triggered":
            trigger_extension, count_extension, binding, measure_keys = (
                self._triggered_sitewide_preflight(metric, extension)
            )
        else:
            binding = next(
                item for item in self._manifest.metric_measures if item.metric_name == metric.name
            )
            measure_keys = (
                frozenset({binding.numerator_measure_key, binding.denominator_measure_key})
                if isinstance(binding, RatioMetricMeasure)
                else frozenset({binding.measure_key})
            )
            self._require_sitewide_outcome_completeness(measure_keys, certified=False)
        rows = self._read_extension(
            extension,
            request=entry.request.model_dump(mode="json") if entry is not None else None,
        )
        by_measure: dict[str, float] = {}
        for row in rows:
            key = str(row["measure_key"])
            by_measure[key] = by_measure.get(key, 0.0) + float(row["sum_value"])
        if isinstance(binding, RatioMetricMeasure):
            site_total = by_measure.get(binding.numerator_measure_key, 0.0)
            site_den = by_measure.get(binding.denominator_measure_key, 0.0)
        else:
            site_total = by_measure.get(binding.measure_key, 0.0)
            site_den = None
        exposures = self._ensure("exposures")
        assigned = (
            exposures.filter(exposures.group_id.notnull())
            .group_by("group_id")
            .aggregate(n_units=exposures.unit_id.count())
        )
        assigned_counts = {
            str(row["group_id"]): int(row["n_units"])
            for row in _artifact_rows(self._snapshot.execute(assigned))
        }
        population_counts = assigned_counts.copy()
        membership_cutoff = None
        member_ids: ir.Table | None = None
        if population == "triggered":
            count_rows = self._read_extension(
                count_extension, request=_artifact_request(count_extension)
            )
            membership_cutoff = trigger_extension.observation_cutoff_ts
            member_ids = self._verify_trigger_population(operation="sitewide_evidence")
            joined_members = member_ids.join(
                exposures.select("experiment_id", "unit_id", "group_id"),
                ["experiment_id", "unit_id"],
            )
            trigger_summary = joined_members.group_by("group_id").aggregate(
                n_units=joined_members.unit_id.count()
            )
            trigger_counts = {
                str(row["group_id"]): int(row["n_units"])
                for row in _artifact_rows(self._snapshot.execute(trigger_summary))
                if row["group_id"] is not None
            }
            self._validate_trigger_population_counts(count_rows, trigger_counts)
            population_counts = trigger_counts
            for group_id in assigned_counts:
                population_counts.setdefault(group_id, 0)
            # Artifact relation/generation checks bind base and extension relations to
            # one generation; a pinned publish writes that generation at one cutoff,
            # so membership, calendar measures, and site volume share that cutoff.
            # Adoption trusts the caller-pinned generation.
        cluster = self.context.cluster
        cluster_table = self._cluster_table()
        moment_rows = self._reduce(
            metric,
            "total",
            cluster=cluster,
            cluster_table=cluster_table,
            population_units=member_ids if population == "triggered" else None,
            calendar_end=self._artifact_experiment.end_day,
        )
        cluster_counts = None
        unit_counts = None
        if self.context.cluster is not None:
            if cluster_table is not None:
                clustered = exposures.join(cluster_table, "unit_id")
                if member_ids is not None:
                    clustered = clustered.semi_join(
                        member_ids.select("experiment_id", "unit_id"),
                        ["experiment_id", "unit_id"],
                    )
                summary = clustered.group_by("group_id").aggregate(
                    n_clusters=clustered[self.context.cluster].nunique(),
                    n_units=clustered.unit_id.nunique(),
                )
                summary_rows = _artifact_rows(self._snapshot.execute(summary))
                cluster_counts = {
                    str(row["group_id"]): int(row["n_clusters"]) for row in summary_rows
                }
                unit_counts = {str(row["group_id"]): int(row["n_units"]) for row in summary_rows}
        return SitewideEvidence(
            metric=metric,
            site_total=site_total,
            site_total_denominator=site_den,
            arm_stats=tuple(_df_to_arms(moment_rows)),
            analysis_population=population,
            control_group=self._artifact_experiment.control_group,
            cluster=self.context.cluster,
            cluster_counts=cluster_counts,
            unit_counts=unit_counts,
            window=(
                None
                if self._artifact_experiment.end is None
                else (
                    self._artifact_experiment.start_day,
                    self._artifact_experiment.end_day or self._artifact_experiment.start_day,
                )
            ),
            membership_cutoff=membership_cutoff,
            snapshot_id=f"{self._manifest.artifact_id}:{self._manifest.generation_id}",
            population_counts=population_counts,
            assigned_counts=assigned_counts,
        )


@cache
def _artifact_facade_type() -> type:
    reader_type = artifact_query_ports().reader.ArtifactMomentSource
    return type("_ArtifactFacadeSource", (_ArtifactFacadeSource, reader_type), {})


def open_artifact(
    store: ArtifactStore,
    ref: UnitDayArtifactRef,
    *,
    expected_context: ArtifactContext,
    metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
    verification: Literal["lazy_digest"] = "lazy_digest",
) -> _ArtifactFacadeSource:
    return _ArtifactFacadeSource.open(
        store,
        ref,
        expected_context=expected_context,
        metrics=metrics,
        verification=verification,
    )


ArtifactMomentSource = _ArtifactFacadeSource


__all__ = ["ArtifactMomentSource", "SqlPanelSource", "open_artifact"]
