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

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import ibis
import ibis.expr.types as ir
import narwhals as nw

from increment._analysis_config import effective_methods
from increment._frame_validation import _reject_windowed_specs
from increment._metric_specs import MetricsArg, MetricSpec, coerce_metrics, synthesise_metric
from increment.errors import (
    CapabilityError,
    RefusalSpec,
    _safe_error_value,
    refuse,
)
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
from increment.query.artifact_extensions import read_extension, read_unit_covariate
from increment.query.artifact_reader import (
    _ARTIFACT_GRAIN,
    LAZY_DIGEST_VERIFICATION,
    restrict_to_units,
)
from increment.query.artifact_reader import (
    ArtifactMomentSource as _ArtifactMomentSource,
)
from increment.query.artifact_reader import (
    _rows as _artifact_rows,
)
from increment.query.builders import (
    group_summary,
    winsorize_unit_totals,
)
from increment.query.native_contract import (
    _NATIVE_COVARIATE_RESERVED,
    DimensionedDayEvidence,
    SitewideEvidence,
)
from increment.semantics.artifact import (
    ArtifactContext,
    RatioMetricMeasure,
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
    Grain,
    MomentSource,
    SourceContext,
    SourceOperation,
)

_SQL_GRAIN = RefusalSpec(
    "source.sql.grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_SQL_UNIT_GRAIN = RefusalSpec(
    "source.sql.unit_grain",
    CapabilityError,
    template="'unit_frame' needs unit-grain rows, which this source does not retain -- only aggregated moments are available here. unit_frame() unavailable; use a frame-backed source (from_unit_summary / from_unit_panel) for unit-grain estimators.",
    keys=frozenset({"method"}),
)
_SQL_SQL_GRAIN = RefusalSpec(
    "source.sql.sql_grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_SQL_OPERATION = RefusalSpec(
    "source.sql.operation",
    CapabilityError,
    template="{operation} cannot serve {requested!r}: {reason}. {route}",
)
_SQL_CLUSTER = RefusalSpec(
    "source.sql.cluster_grain",
    CapabilityError,
    template="cluster_counts() is unavailable on {source}: {because}. The randomization-grain count needs a declared cluster column; build the source with from_unit_summary(..., cluster=...) or analyse from definitions declaring Experiment.cluster.",
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ibis.backends.duckdb import Backend
    from ibis.backends.sql import SQLBackend

    from increment._source_types import ComplianceSummary
    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Breakout, Metric


def _melt_to_unit_totals(
    tbl: ir.Table, *, unit: str, group: str, study_id: str, specs: Sequence[MetricSpec]
) -> ir.Table:
    """Project one policy-adjusted ``(unit, metric)`` row per specification."""
    per_metric = []
    for spec in specs:
        base = tbl.select(
            unit_id=tbl[unit].cast("string"),
            experiment_id=ibis.literal(study_id),
            group_id=tbl[group].cast("string"),
            y=tbl[spec.y_column].cast("float64"),
            x=(
                tbl[spec.covariate].cast("float64")
                if spec.covariate
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
        if spec.covariate and spec.covariate_missing != "error":
            fill = (
                ibis.literal(0.0)
                if spec.covariate_missing == "zero"
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
            {c for s in specs for c in (s.y_column, s.denominator, s.covariate) if c}
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
        summary = group_summary(totals)
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
            refuse(_SQL_GRAIN, grain=grain, offered=self.capabilities)
        if by:
            refuse(
                _SQL_OPERATION,
                operation="moments",
                requested={"metric": metric.name, "by": tuple(by)},
                reason="breakout dimensions are not retained by SqlPanelSource",
                route="request whole-window totals or use a frame-backed source with breakouts",
            )
        rows = self._summary.filter(self._summary.metric == metric.name)
        return self._con.to_pyarrow(rows).to_pylist()

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> NoReturn:
        if outcome_stage == "raw":
            from increment.winsor import winsor_refuse

            winsor_refuse(
                "raw_state_required", "This source does not retain exact pre-winsor unit outcomes."
            )
        refuse(_SQL_UNIT_GRAIN, method="unit_frame")

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
        refuse(
            _SQL_CLUSTER,
            source="a SQL-summary source",
            because=(
                "this source is built from a pre-aggregated unit-day stats "
                "table and takes no cluster= argument"
            ),
        )

    def compliance_dates(self) -> Sequence[object]:
        refuse(
            _SQL_OPERATION,
            operation="compliance_dates",
            requested=(),
            reason="enrollment day state is absent from this SQL summary",
            route="use a panel source for as-of compliance",
        )

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        from increment._source_types import raise_legacy_compliance_state

        raise_legacy_compliance_state(
            study_id=self.context.study_id,
            reason="SQL summary contains no enrollment-level uptake state",
        )

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        if grain != "total":
            refuse(_SQL_SQL_GRAIN, grain=grain, offered=self.capabilities)
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
            refuse(
                _SQL_OPERATION,
                operation="summary_sql",
                requested={"breakouts": tuple(_safe_error_value(item) for item in breakouts)},
                reason="adopted panels retain only whole-population group summaries",
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
    from increment.semantics.design import Encouragement
    from increment.semantics.models import EncouragementDeclaration, _with_inherited_day_boundary

    validate_artifact_context(expected_context)
    try:
        payload = json.loads(expected_context.canonical_json)
        definitions_payload = payload["definitions"]
        metric_adapter: TypeAdapter[Metric] = TypeAdapter(Metric)
        metric_pool = [
            metric_adapter.validate_python(item) for item in definitions_payload["metrics"]
        ]
        experiment_payload = dict(payload["experiment"])
        effective_design = None
        if (experiment_payload.get("design") or {}).get("mechanism") == "encouragement":
            # The context carries the full effective design; Experiment.design keeps only
            # the declaration-shaped part of it.
            effective_design = Encouragement.model_validate(experiment_payload["design"])
            experiment_payload["design"] = EncouragementDeclaration(
                uptake=effective_design.uptake,
                one_sided=effective_design.one_sided,
                exclusion_restriction=effective_design.exclusion_restriction,
            )
        experiment = Experiment.model_validate(experiment_payload)
        experiment = _with_inherited_day_boundary(experiment, definitions_payload["day_boundary"])
        experiment_name = str(payload["experiment_name"])
    except (KeyError, TypeError, ValueError, AttributeError, json.JSONDecodeError) as exc:
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
    if kind == "assignment_counts":
        return {"kind": kind, "populations": extension.populations}
    if kind == "trigger_population":
        return {"kind": kind, "trigger_name": extension.trigger_name}
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


class _ArtifactFacadeSource(_ArtifactMomentSource):
    """Artifact adapter that exposes the public operation family to Analysis."""

    def __init__(
        self,
        store: ArtifactStore,
        snapshot_context: AbstractContextManager[object],
        snapshot: Any,
        manifest: UnitDayArtifactManifest,
        *,
        expected_context: ArtifactContext,
        metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
        population_units: frozenset[str] | None = None,
    ) -> None:
        super().__init__(
            store,
            snapshot_context,
            snapshot,
            manifest,
            metrics=metrics,
        )
        experiment, source_context = _artifact_source_context(expected_context)
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
        has_trigger_population = any(
            extension.kind == "trigger_population" for extension in manifest.extensions
        )
        if has_trigger_population and assignment_extensions:
            operations.update({"triggered_counts", "triggered_source"})
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
            refuse(LAZY_DIGEST_VERIFICATION)
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
            code = "artifact.extension.missing" if not matches else "artifact.extension.invalid"
            raise ArtifactContractError(
                code,
                "requested artifact extension is not selected exactly once",
            )
        return matches[0]

    def _exposure_rows(self) -> list[dict[str, Any]]:
        exposures = self._ensure("exposures")
        if self._population_units is not None:
            exposures = restrict_to_units(exposures, self._population_units)
        return _artifact_rows(self._snapshot.execute(exposures))

    def _read_extension(
        self, extension: Any, *, request: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]:
        exposures = _artifact_rows(self._snapshot.execute(self._ensure("exposures")))
        result = list(
            read_extension(
                self._snapshot,
                extension,
                request=request,
                context=self._manifest.context,
                experiment_id=self._manifest.experiment_id,
                exposure_keys={(row["experiment_id"], row["unit_id"]) for row in exposures},
            )
        )
        if self._population_units is not None and extension.kind not in {
            "site_volume",
            "assignment_counts",
        }:
            result = [row for row in result if str(row.get("unit_id")) in self._population_units]
        return result

    def _dimension_table(self, extension: Any, request: Mapping[str, Any]) -> Any:
        rows = self._read_extension(extension, request=request)
        kind = extension.kind
        value_field = "factor_value" if kind == "factor_dimension" else "dimension_value"
        name = request["property_name"]
        return ibis.memtable(
            [
                {
                    "unit_id": row["unit_id"],
                    name: None if row.get("value_is_missing") else row[value_field],
                }
                for row in rows
            ],
            schema={"unit_id": "string", name: "string"},
        )

    def _cluster_table(self) -> ir.Table | None:
        """Resolve declared cluster identities for total-grain reductions."""
        if self.context.cluster is None:
            return None
        cluster_extension = self._extension(
            {"kind": "cluster_identity", "cluster_name": self.context.cluster}
        )
        cluster_rows = self._read_extension(
            cluster_extension, request=_artifact_request(cluster_extension)
        )
        return ibis.memtable(
            [
                {"unit_id": row["unit_id"], self.context.cluster: row["cluster_id"]}
                for row in cluster_rows
            ],
            schema={"unit_id": "string", self.context.cluster: "string"},
        )

    def _cuped_pre_stats(self, metric: Metric) -> Any:
        """Resolve *metric*'s `cuped_preperiod` extension into a covariate table."""
        cuped = self._extension({"kind": "cuped_preperiod", "metric_name": metric.name})
        pre_rows = self._read_extension(cuped, request=_artifact_request(cuped))
        return self._covariate_table(
            [
                {
                    "experiment_id": row["experiment_id"],
                    "unit_id": row["unit_id"],
                    "sum_value": row["x"],
                }
                for row in pre_rows
            ]
        )

    def _preflight_cluster_grain(self, grain: Grain, *, operation: str) -> None:
        if self.context.cluster is not None and grain != "total":
            refuse(
                _ARTIFACT_GRAIN,
                operation=operation,
                request={"grain": grain, "cluster": self.context.cluster},
                offered=("total",),
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
        self._preflight_cluster_grain(grain, operation="moments")
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

    def unit_frame(
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
        extensions = {name: self._covariate_extension(name) for name in requested}
        metric = self.validated_metric(metric)
        base = self._unit_frame(
            metric,
            cluster=cluster,
            resolve_cluster=self._cluster_table,
            population_units=self._population_units,
            outcome_stage=outcome_stage,
        )
        if not extensions:
            return base
        import narwhals as nw

        frame = nw.from_native(cast(Any, base), eager_only=True)
        unit_ids = [str(unit) for unit in frame["unit_id"].to_list()]
        for name, extension in extensions.items():
            rows = self._read_extension(extension, request=_artifact_request(extension))
            frame = frame.with_columns(
                read_unit_covariate(
                    extension,
                    rows,
                    name=name,
                    unit_ids=unit_ids,
                    implementation=frame.implementation,
                )
            )
        return frame.to_native()

    def _covariate_extension(self, name: str) -> Any:
        matches = [
            extension
            for extension in self._manifest.extensions
            if (extension.kind == "unit_covariate" or extension.kind == "unit_covariate_level")
            and extension.covariate_name == name
        ]
        if len(matches) == 1:
            return matches[0]
        code = "artifact.extension.missing" if not matches else "artifact.extension.invalid"
        refuse(
            _ARTIFACT_REFUSALS[code],
            message=(
                f"artifact carries {'no' if not matches else 'more than one'} unit "
                f"covariate {name!r}. An artifact publishes only the covariates its "
                "experiment declares under design.covariates: declare it there and "
                "republish (a declared covariate is published automatically), or "
                "read Analysis.from_definitions."
            ),
            covariate=name,
            route=(
                "declare the covariate under the experiment's observational design "
                "and republish the artifact from those definitions"
            ),
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
        self._preflight_cluster_grain(grain, operation="moments")
        if grain not in self.capabilities:
            refuse(
                _ARTIFACT_GRAIN,
                operation="moments",
                request={"grain": grain},
                offered=tuple(sorted(self.capabilities)),
                route="request one of the source's supported grains",
            )
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
            include_covariate = bool(
                config
                and any(
                    method.variance_reduction == "cuped"
                    for method in effective_methods(config, design=self.context.design)
                )
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
        self._preflight_cluster_grain(grain, operation="breakout_moments")
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

    def breakout_source(
        self, breakout: Breakout, *, metrics: Sequence[Metric]
    ) -> BreakoutMomentsSource:
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
            for config in self.context.configs
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
            configs=self.context.configs,
        )

    def breakout_sources(
        self, breakouts: Sequence[Breakout], *, metrics: Sequence[Metric]
    ) -> tuple[BreakoutMomentsSource, ...]:
        self._require_extension_coverage("breakout_dimension")
        return tuple(self.breakout_source(breakout, metrics=metrics) for breakout in breakouts)

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

    def breakout_summaries(self, *, metrics: Sequence[Metric]) -> dict[str, dict[str, Any]]:
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

    def factor_summaries(self, *, metrics: Sequence[Metric]) -> dict[str, Any]:
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

    def unit_counts(self) -> dict[str, int]:
        if self._population_units is not None:
            grain, counts, unit_counts = self.triggered_counts()
            return unit_counts if grain == "cluster" else counts
        populations = (
            ("assigned", "triggered")
            if self._artifact_experiment.trigger is not None
            else ("assigned",)
        )
        extension = self._extension({"kind": "assignment_counts", "populations": populations})
        rows = self._read_extension(extension, request=_artifact_request(extension))
        return {
            str(row["group_id"]): int(row["n_units"])
            for row in rows
            if row["population"] == "assigned"
        }

    def cluster_counts(self) -> dict[str, int]:
        extension = self._extension(
            {"kind": "cluster_identity", "cluster_name": self.context.cluster}
        )
        rows = self._read_extension(extension, request=_artifact_request(extension))
        groups: dict[str, set[str]] = {}
        exposure_by_unit = {
            str(row["unit_id"]): str(row["group_id"]) for row in self._exposure_rows()
        }
        for row in rows:
            groups.setdefault(exposure_by_unit[str(row["unit_id"])], set()).add(
                str(row["cluster_id"])
            )
        return {group: len(clusters) for group, clusters in groups.items()}

    def triggered_counts(self) -> tuple[Literal["unit", "cluster"], dict[str, int], dict[str, int]]:
        extension = self._extension(
            {"kind": "assignment_counts", "populations": ("assigned", "triggered")}
        )
        rows = self._read_extension(extension, request=_artifact_request(extension))
        counts = {
            str(row["group_id"]): int(row["n_randomization_units"])
            for row in rows
            if row["population"] == "triggered"
        }
        unit_counts = {
            str(row["group_id"]): int(row["n_units"])
            for row in rows
            if row["population"] == "triggered"
        }
        grain: Literal["unit", "cluster"] = (
            "cluster" if self.context.cluster is not None else "unit"
        )
        return grain, counts, unit_counts if grain == "cluster" else {}

    def trigger_rates(self) -> dict[str, float]:
        grain, counts, unit_counts = self.triggered_counts()
        triggered = counts if grain == "unit" else unit_counts
        assigned = self.unit_counts()
        return {
            group: triggered.get(group, 0) / count for group, count in assigned.items() if count
        }

    def triggered_source(self) -> MomentSource:
        trigger = self._extension(
            {"kind": "trigger_population", "trigger_name": self._artifact_experiment.trigger}
        )
        rows = self._read_extension(trigger, request=_artifact_request(trigger))
        return _ArtifactFacadeSource(
            self._store,
            self._lifecycle,
            self._snapshot,
            self._manifest,
            expected_context=self._expected_context,
            metrics=self._metrics_arg,
            population_units=frozenset(str(row["unit_id"]) for row in rows),
        )

    def sitewide_evidence(self, metric: Metric, *, include_ratio: bool = False) -> Any:
        from increment.estimation.engine import _df_to_arms

        metric = self.validated_metric(metric)
        entries = unit_day_artifact_extension_catalog(self._expected_context)
        entry = next(
            (
                candidate
                for candidate in entries
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
            )
        request = entry.request.model_dump(mode="json") if entry is not None else None
        rows = self._read_extension(extension, request=request)
        by_measure: dict[str, float] = {}
        for row in rows:
            key = str(row["measure_key"])
            by_measure[key] = by_measure.get(key, 0.0) + float(row["sum_value"])
        binding = next(
            item for item in self._manifest.metric_measures if item.metric_name == metric.name
        )
        if isinstance(binding, RatioMetricMeasure):
            site_total = by_measure.get(binding.numerator_measure_key, 0.0)
            site_den = by_measure.get(binding.denominator_measure_key, 0.0)
        else:
            site_total = by_measure.get(binding.measure_key, 0.0)
            site_den = None
        import pyarrow as pa

        cluster_counts = None
        unit_counts = None
        if self.context.cluster is not None:
            cluster_extension = self._extension(
                {"kind": "cluster_identity", "cluster_name": self.context.cluster}
            )
            cluster_rows = self._read_extension(
                cluster_extension, request=_artifact_request(cluster_extension)
            )
            group_by_unit = {
                str(row["unit_id"]): str(row["group_id"]) for row in self._exposure_rows()
            }
            clusters_by_group: dict[str, set[str]] = {}
            units_by_group: dict[str, set[str]] = {}
            for row in cluster_rows:
                unit_id = str(row["unit_id"])
                group = group_by_unit[unit_id]
                clusters_by_group.setdefault(group, set()).add(str(row["cluster_id"]))
                units_by_group.setdefault(group, set()).add(unit_id)
            cluster_counts = {group: len(values) for group, values in clusters_by_group.items()}
            unit_counts = {group: len(values) for group, values in units_by_group.items()}
        summary_rows = self.moments(metric)
        return SitewideEvidence(
            metric=metric,
            site_total=site_total,
            site_total_denominator=site_den,
            arm_stats=tuple(_df_to_arms(pa.Table.from_pylist(summary_rows))),
            control_group=self._artifact_experiment.control_group,
            cluster=self.context.cluster,
            cluster_counts=cluster_counts,
            unit_counts=unit_counts,
        )


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
