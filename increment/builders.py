"""Module-level builders for portable experiment execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

    from increment._metric_specs import MetricsArg
    from increment.artifacts import SourceSnapshotEvidence
    from increment.frame import IntoDataFrame
    from increment.query.artifact_contract import ArtifactStore
    from increment.semantics.artifact import ArtifactContext, UnitDayArtifactRef
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import AnalysisPlan
    from increment.semantics.unit_cycle import UnitCycleReference
    from increment.switchback import SwitchbackAssignment


from increment._analysis_kernel_adapter import (
    AnalysisResolver as _AnalysisResolver,
)
from increment._analysis_kernel_adapter import (
    experiment_for_analysis as _experiment_for_analysis,
)
from increment.errors import CodedError
from increment.kernel.contracts import _refuse
from increment.kernel.experiment import Experiment


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
) -> tuple[Experiment, _AnalysisResolver]: ...


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
) -> tuple[Experiment, _AnalysisResolver]: ...


def from_frame(
    frame: IntoDataFrame,
    *,
    shape: Literal["summary", "panel"],
    **kwargs: Any,
) -> tuple[Experiment, _AnalysisResolver]:
    """Build an experiment from a unit-summary or unit-panel frame."""
    from increment._analysis_execution import Analysis

    if shape == "summary":
        analysis = Analysis._from_unit_summary(frame, **kwargs)
    elif shape == "panel":
        analysis = Analysis._from_unit_panel(frame, **kwargs)
    else:
        _refuse("kernel.request.invalid", field="shape must be summary or panel")
    return _experiment_for_analysis(analysis, return_run_result=True)


def _accept_definitions_arguments[**P, R](builder: Callable[P, R]) -> Callable[P, R]:
    @wraps(builder)
    def parsed(*args: Any, **kwargs: Any) -> R:
        legacy = (
            len(args) == 3
            or (len(args) == 2 and isinstance(args[1], (str, os.PathLike)))
            or (len(args) == 1 and kwargs.get("definitions_path") is not None)
        )
        names = (
            ("experiment_name", "definitions_path", "con")
            if legacy
            else ("definitions_path", "con")
        )
        if len(args) > len(names):
            _refuse("kernel.request.invalid", field="too many positional definitions arguments")
        for name, value in zip(names, args, strict=False):
            if kwargs.get(name) is not None:
                _refuse("kernel.request.invalid", field=f"{name} was supplied twice")
            kwargs[name] = value
        if kwargs.get("definitions_path") is None:
            _refuse("kernel.request.invalid", field="definitions_path is required")
        return cast(Callable[..., R], builder)(**kwargs)

    return parsed


@_accept_definitions_arguments
def from_definitions(
    definitions_path: str | os.PathLike[str],
    con: SQLBackend | None = None,
    *,
    experiment_name: str | None = None,
    backend: str | None = None,
    store: Literal["auto", "always", "none"] = "auto",
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    source_snapshot_evidence: SourceSnapshotEvidence | None = None,
) -> tuple[Experiment, _AnalysisResolver | _OfflineResolver]:
    """Build from definitions, using a bound versioned catalog when offline."""
    return _from_definitions(
        definitions_path,
        con,
        experiment_name=experiment_name,
        backend=backend,
        store=store,
        on_mixed_assignment=on_mixed_assignment,
        source_snapshot_evidence=source_snapshot_evidence,
    )


@_accept_definitions_arguments
def _from_definitions(
    definitions_path: str | os.PathLike[str],
    con: SQLBackend | None = None,
    *,
    experiment_name: str | None = None,
    backend: str | None = None,
    store: Literal["auto", "always", "none"] = "auto",
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    _record_schema: bool = True,
) -> tuple[Experiment, _AnalysisResolver | _OfflineResolver]:
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

    from increment._analysis_execution import Analysis
    from increment.semantics.loader import load

    definitions_digest = _definitions_digest(path) if _record_schema else ""
    if experiment_name is None:
        definitions = load(path)
        if len(definitions.experiments) != 1:
            _refuse("kernel.request.invalid", field="experiment_name is required")
        experiment_name = definitions.experiments[0].name
    analysis = Analysis._from_definitions(
        experiment_name,
        path,
        con,
        backend=backend,
        store=store,
        on_mixed_assignment=on_mixed_assignment,
        source_snapshot_evidence=source_snapshot_evidence,
    )
    experiment, resolver = _experiment_for_analysis(
        analysis, record_schema=_record_schema, return_run_result=True
    )
    if _record_schema:
        _write_catalog(path, experiment, definitions_digest)
        resolver.catalog_writer = lambda captured: _write_catalog(
            path, captured, definitions_digest
        )
    return experiment, resolver


def from_unit_day_artifact(
    store: ArtifactStore,
    ref: UnitDayArtifactRef,
    *,
    expected_context: ArtifactContext,
    verification: Literal["lazy_digest"] = "lazy_digest",
) -> tuple[Experiment, _AnalysisResolver]:
    from increment._analysis_execution import Analysis

    return _experiment_for_analysis(
        Analysis._from_unit_day_artifact(
            store, ref, expected_context=expected_context, verification=verification
        ),
        return_run_result=True,
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
) -> tuple[Experiment, _AnalysisResolver]:
    from increment._analysis_execution import Analysis

    return _experiment_for_analysis(
        Analysis._from_switchback_panel(
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
        ),
        return_run_result=True,
    )


def from_moments(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: MetricsArg,
    control: object | None = None,
    experiment_id: str | None = None,
    design: Randomized | Encouragement | Observational | None = None,
    plan: AnalysisPlan | None = None,
) -> tuple[Experiment, _AnalysisResolver]:
    from increment._analysis_execution import Analysis

    return _experiment_for_analysis(
        Analysis._from_moments(
            rows,
            metrics=metrics,
            control=control,
            experiment_id=experiment_id,
            design=design,
            plan=plan,
        ),
        return_run_result=True,
    )


def from_unit_summary(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    return from_frame(*args, shape="summary", **kwargs)


def from_unit_panel(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    return from_frame(*args, shape="panel", **kwargs)
