"""Module-level builders for portable experiment execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

    from increment._metric_specs import MetricsArg
    from increment.artifacts import SourceSnapshotEvidence
    from increment.decision import ContrastSource
    from increment.frame import IntoDataFrame
    from increment.query.artifact_contract import ArtifactStore
    from increment.query.session import WarehouseSession
    from increment.semantics.artifact import ArtifactContext, UnitDayArtifactRef
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import AnalysisPlan, Definitions
    from increment.semantics.models import Experiment as AnalysisExperiment
    from increment.semantics.unit_cycle import UnitCycleReference
    from increment.sources import EvidenceSource
    from increment.switchback import SwitchbackAssignment


from increment._binding import SourceBinding, _ArtifactOpenSpec
from increment._source_resolver import SourceResolver, resolver_for
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, raiser, refusals
from increment.kernel.contracts import _refuse
from increment.kernel.experiment import Experiment

_BINDING_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.analysis.definitions_state_requires_identification": "definitions state requires a declared parallel identification",
        "facade.analysis.unknown_experiment": RefusalSpec(
            "facade.analysis.unknown_experiment",
            InvalidRequestError,
            lambda *, requested, available: (
                f"experiment {requested!r} is not declared in these definitions "
                f"(available: {list(available)!r})"
            ),
        ),
        "facade.analysis.source_context_design_disagrees_arm": "source context design disagrees with the design passed to the builder",
        "facade.analysis.invalid_mixed_assignment_policy": RefusalSpec(
            "facade.analysis.invalid_mixed_assignment_policy",
            InvalidRequestError,
            template="on_mixed_assignment must be 'error', 'warn', or 'exclude', got {on_mixed_assignment!r}",
        ),
        "facade.analysis.invalid_store_policy": RefusalSpec(
            "facade.analysis.invalid_store_policy",
            InvalidRequestError,
            template="store must be 'auto', 'always', or 'none'; got {store!r}",
        ),
    },
)
_bind_raise = raiser(_BINDING_REFUSALS)


def _build_state(
    *,
    src: Any,
    defs: Any,
    experiment: AnalysisExperiment | None,
    con: SQLBackend | None,
    session: Any,
    experiment_name: str,
    backend: str | None,
    store: Literal["auto", "always", "none"] = "auto",
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
) -> Any:
    from increment._study import ParallelStudyEnvelope
    from increment.decision import (
        ContrastAnalysisState,
        ContrastContext,
        DefinitionsArmAnalysisState,
        SeamArmAnalysisState,
    )

    context = src.context
    if isinstance(context, ContrastContext):
        return ContrastAnalysisState(source=src, experiment_name=experiment_name)
    design = context.design
    study = None if design is None else ParallelStudyEnvelope(identification=design)
    if defs is None:
        return SeamArmAnalysisState(source=src, study=study, experiment_name=experiment_name)
    if study is None:
        _bind_raise("facade.analysis.definitions_state_requires_identification")
    return DefinitionsArmAnalysisState(
        source=src,
        study=study,
        definitions=defs,
        experiment=cast(Any, experiment),
        connection=cast(Any, con),
        session=session,
        experiment_name=experiment_name,
        backend=backend or "",
        store=store,
        on_mixed_assignment=on_mixed_assignment,
        exposure_lookup={e.name: e for e in defs.exposures},
    )


def bind_source(
    src: EvidenceSource | ContrastSource,
    design: Randomized | Encouragement | Observational | None = None,
    *,
    defs: Definitions | None = None,
    experiment: AnalysisExperiment | None = None,
    con: SQLBackend | None = None,
    session: WarehouseSession | None = None,
    experiment_name: str | None = None,
    backend: str | None = None,
    store: Literal["auto", "always", "none"] = "auto",
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    artifact_open_spec: _ArtifactOpenSpec | None = None,
) -> SourceBinding:
    from increment.decision import ContrastContext

    context = src.context
    if isinstance(context, ContrastContext):
        if design is not None and context.study.identification != design:
            _bind_raise("facade.analysis.source_context_design_disagrees_arm")
    elif design is not None and context.design != design:
        _bind_raise("facade.analysis.source_context_design_disagrees_arm")

    state = _build_state(
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
    return SourceBinding(state, artifact_open=artifact_open_spec)


def _bind_definitions(
    experiment_name: str,
    definitions_path: str | Path,
    con: SQLBackend,
    *,
    backend: str | None = None,
    store: Literal["auto", "always", "none"] = "auto",
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    source_snapshot_evidence: SourceSnapshotEvidence | None = None,
) -> SourceBinding:
    from increment.plan import bind_automatic_sequential_plan, compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics.loader import load, verify_sql_admission_matches_execution
    from increment.semantics.models import Experiment as AnalysisExperiment
    from increment.sequential_source import native_observation_mapping

    if on_mixed_assignment not in ("error", "warn", "exclude"):
        _bind_raise(
            "facade.analysis.invalid_mixed_assignment_policy",
            on_mixed_assignment=on_mixed_assignment,
        )
    if store not in ("auto", "always", "none"):
        _bind_raise("facade.analysis.invalid_store_policy", store=store)
    defs = load(definitions_path)
    experiment = defs.experiment(experiment_name)
    if experiment is None:
        _bind_raise(
            "facade.analysis.unknown_experiment",
            requested=experiment_name,
            available=tuple(sorted(e.name for e in defs.experiments)),
        )
    metrics_by_name = {m.name: m for m in defs.metrics}
    metrics = [metrics_by_name[name] for name in experiment.metric_names if name in metrics_by_name]
    design = experiment.resolved_design()
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
        experiment = AnalysisExperiment.model_validate(
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
    return bind_source(
        src,
        design,
        defs=defs,
        experiment=experiment,
        con=con,
        session=session,
        experiment_name=experiment_name,
        backend=backend,
        store=store,
        on_mixed_assignment=on_mixed_assignment,
    )


def _input_backend_of(frame: IntoDataFrame) -> Any:
    import narwhals as nw

    return nw.from_native(frame, eager_only=True).implementation.value


def _bind_summary(  # noqa: PLR0913
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
) -> SourceBinding:
    from increment.frame import from_unit_summary
    from increment.query.fact_resolution import _require_design

    _require_design(control, design, constructor="from_frame")
    if control is None:
        assert design is not None
        control = str(design.control_group)
    source = from_unit_summary(
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
    binding = bind_source(source, source.context.design)
    return SourceBinding(
        binding.state, artifact_open=binding.artifact_open, input_backend=_input_backend_of(frame)
    )


def _bind_panel(  # noqa: PLR0913
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
) -> SourceBinding:
    from increment.frame import from_unit_panel
    from increment.query.fact_resolution import _require_design

    _require_design(control, design, constructor="from_frame")
    if control is None:
        assert design is not None
        control = str(design.control_group)
    source = from_unit_panel(
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
    binding = bind_source(source, source.context.design)
    return SourceBinding(
        binding.state, artifact_open=binding.artifact_open, input_backend=_input_backend_of(frame)
    )


def _bind_artifact(
    store: ArtifactStore,
    ref: UnitDayArtifactRef,
    *,
    expected_context: ArtifactContext,
    verification: Literal["lazy_digest"] = "lazy_digest",
) -> SourceBinding:
    from increment.query.source import open_artifact

    source = open_artifact(
        store,
        ref,
        expected_context=expected_context,
        verification=verification,
    )
    return bind_source(
        source,
        source.context.design,
        artifact_open_spec=_ArtifactOpenSpec(store, ref, expected_context, verification),
    )


def _bind_switchback(
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
) -> SourceBinding:
    from increment.switchback import from_switchback_panel

    source = from_switchback_panel(
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
    binding = bind_source(cast(Any, source), identification)
    return SourceBinding(
        binding.state, artifact_open=binding.artifact_open, input_backend=_input_backend_of(frame)
    )


def _bind_moments(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: MetricsArg,
    control: object | None = None,
    experiment_id: str | None = None,
    design: Randomized | Encouragement | Observational | None = None,
    plan: AnalysisPlan | None = None,
) -> SourceBinding:
    from increment._analysis_config import (
        refuse_cuped_linear_moments,
        refuse_missing_scalar_evidence,
        resolve_configs,
        validate_scalar_route,
    )
    from increment._cuped_evidence import invalid_evidence, method_key
    from increment._frame_validation import (
        _coerce_source_metrics,
        canonical_control_for_moments,
        canonical_design_for_moments,
        validate_moments_control,
    )
    from increment.frame import synthesise_metric
    from increment.query.fact_resolution import _require_design
    from increment.semantics.design import Encouragement, Randomized
    from increment.sources import MomentsSource, cuped_evidence_items

    design = canonical_design_for_moments(design)
    _require_design(control, design, constructor="from_moments")
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
                actual_components = 0 if item.cells is None else len(item.cells.outcome_references)
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
            for method in methods_for_metric:
                if (
                    method.variance_reduction == "cuped"
                    and carried.get(method_key(method)) != "scalar_moments"
                ):
                    refuse_missing_scalar_evidence("from_moments", method.name)
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
        if requested_covariates and provenance_names and provenance_names != requested_covariates:
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
    configs = resolve_configs(
        metric_catalog,
        bindings=None,
        specs={spec.name: spec for spec in specs},
        methods=None,
        prior=None,
    )
    validate_scalar_route(
        tuple(config for config in configs if config.metric.name not in portable_metrics),
        route="from_moments",
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
    source = MomentsSource(
        rows,
        metrics=metric_catalog,
        study_id=experiment_id,
        design=design,
        plan=plan,
        configs=configs,
    )
    return bind_source(source, design)


@contextmanager
def _catalog_lock(lock_path: Path) -> Iterator[None]:
    with lock_path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            if lock.read(1) == b"":
                lock.seek(0)
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _definitions_digest(path: Path) -> str:
    files = (
        (path,)
        if path.is_file()
        else tuple(
            sorted(
                file
                for file in path.rglob("*")
                if file.is_file() and file.suffix in {".yaml", ".yml"}
            )
        )
    )
    digest = hashlib.sha256()
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        encoded = relative.encode("utf-8")
        payload = file.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _catalog_path(path: Path) -> Path:
    return (
        path.with_suffix(path.suffix + ".kernel.json")
        if path.is_file()
        else path / ".increment-kernel-catalog.json"
    )


def _write_catalog(path: Path, experiment: Experiment, expected_digest: str) -> None:
    catalog_path = _catalog_path(path)
    payload = json.loads(experiment.to_json())
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with _catalog_lock(catalog_path.with_suffix(catalog_path.suffix + ".lock")):
        digest = _definitions_digest(path)
        if digest != expected_digest:
            _refuse("kernel.definitions.catalog_stale", path=str(path))
        catalog: dict[str, Any] = {
            "schema_version": 1,
            "definitions_digest": digest,
            "experiments": {},
        }
        if catalog_path.exists():
            previous = json.loads(catalog_path.read_text(encoding="utf-8"))
            if previous.get("schema_version") == 1 and previous.get("definitions_digest") == digest:
                catalog = previous
        catalog["experiments"][experiment.experiment_id] = payload
        data = json.dumps(catalog, sort_keys=True, separators=(",", ":")) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{catalog_path.name}.", dir=catalog_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, catalog_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class _OfflineResolver:
    def __init__(
        self, definitions_path: Path, expected_digest: str, experiment: Experiment
    ) -> None:
        self.definitions_path = definitions_path
        self.expected_digest = expected_digest
        self.experiment = experiment

    def validate_experiment(self, experiment: Experiment) -> None:
        from increment._binding import _validate_experiment

        _validate_experiment(experiment, self.experiment)

    def resolve(self, source_refs: object, ctx: object | None = None) -> object:
        del source_refs, ctx
        raise CodedError(
            "Executing offline definitions requires a bound warehouse connection",
            code="kernel.run.resolver_missing",
            context={},
        )

    def verify(self, source_refs: object, provider: object) -> list[CodedError]:
        del source_refs, provider
        if _definitions_digest(self.definitions_path) != self.expected_digest:
            return [
                CodedError(
                    "Definitions changed since the offline catalog was loaded",
                    code="kernel.definitions.catalog_stale",
                    context={"path": str(self.definitions_path)},
                )
            ]
        return []


@overload
def from_frame(
    frame: IntoDataFrame,
    *,
    shape: Literal["summary"],
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
) -> tuple[Experiment, SourceResolver]: ...


@overload
def from_frame(
    frame: IntoDataFrame,
    *,
    shape: Literal["panel"],
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
) -> tuple[Experiment, SourceResolver]: ...


def from_frame(
    frame: IntoDataFrame,
    *,
    shape: Literal["summary", "panel"],
    **kwargs: Any,
) -> tuple[Experiment, SourceResolver]:
    """Build an experiment from a unit-summary or unit-panel frame."""
    if shape == "summary":
        binding = _bind_summary(frame, **kwargs)
    elif shape == "panel":
        binding = _bind_panel(frame, **kwargs)
    else:
        _refuse("kernel.request.invalid", field="shape must be summary or panel")
    return resolver_for(binding)


def from_definitions(
    definitions_path: str | os.PathLike[str],
    con: SQLBackend | None = None,
    *,
    experiment_name: str | None = None,
    backend: str | None = None,
    store: Literal["auto", "always", "none"] = "auto",
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    source_snapshot_evidence: SourceSnapshotEvidence | None = None,
) -> tuple[Experiment, SourceResolver | _OfflineResolver]:
    """Build from definitions, using a bound versioned catalog when offline."""
    path = Path(definitions_path)
    if con is None:
        catalog_path = _catalog_path(path)
        if not catalog_path.is_file():
            _refuse("kernel.definitions.catalog_missing", path=str(path))
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _refuse("kernel.definitions.catalog_stale", path=str(path))
        if (
            not isinstance(catalog, dict)
            or catalog.get("schema_version") != 1
            or catalog.get("definitions_digest") != _definitions_digest(path)
        ):
            _refuse("kernel.definitions.catalog_stale", path=str(path))
        experiments = catalog.get("experiments")
        if not isinstance(experiments, dict):
            _refuse("kernel.definitions.catalog_stale", path=str(path))
        name = experiment_name
        if name is None:
            if len(experiments) != 1:
                _refuse("kernel.request.invalid", field="experiment_name is required")
            name = next(iter(experiments))
        payload = experiments.get(name)
        if not isinstance(payload, dict):
            _refuse("kernel.definitions.catalog_missing", path=str(path))
        experiment = Experiment.from_json(json.dumps(payload))
        return experiment, _OfflineResolver(path, str(catalog["definitions_digest"]), experiment)

    from increment.semantics.loader import load

    definitions_digest = _definitions_digest(path)
    if experiment_name is None:
        definitions = load(path)
        if len(definitions.experiments) != 1:
            _refuse("kernel.request.invalid", field="experiment_name is required")
        experiment_name = definitions.experiments[0].name
    assert con is not None
    binding = _bind_definitions(
        experiment_name,
        path,
        con,
        backend=backend,
        store=store,
        on_mixed_assignment=on_mixed_assignment,
        source_snapshot_evidence=source_snapshot_evidence,
    )
    experiment, resolver = resolver_for(binding, record_schema=True)
    _write_catalog(path, experiment, definitions_digest)
    resolver.catalog_writer = lambda captured: _write_catalog(path, captured, definitions_digest)
    return experiment, resolver


def from_unit_day_artifact(
    store: ArtifactStore,
    ref: UnitDayArtifactRef,
    *,
    expected_context: ArtifactContext,
    verification: Literal["lazy_digest"] = "lazy_digest",
) -> tuple[Experiment, SourceResolver]:
    return resolver_for(
        _bind_artifact(store, ref, expected_context=expected_context, verification=verification)
    )


def from_switchback_frame(
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
) -> tuple[Experiment, SourceResolver]:
    binding = _bind_switchback(
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
    return resolver_for(binding)


def from_moments(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: MetricsArg,
    control: object | None = None,
    experiment_id: str | None = None,
    design: Randomized | Encouragement | Observational | None = None,
    plan: AnalysisPlan | None = None,
) -> tuple[Experiment, SourceResolver]:
    binding = _bind_moments(
        rows,
        metrics=metrics,
        control=control,
        experiment_id=experiment_id,
        design=design,
        plan=plan,
    )
    return resolver_for(binding)
