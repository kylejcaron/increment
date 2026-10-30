"""Module-level builders for portable experiment execution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

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
    def __init__(self, definitions_path: Path, expected_digest: str) -> None:
        self.definitions_path = definitions_path
        self.expected_digest = expected_digest

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


def from_frame(
    frame: Any,
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


def from_definitions(
    *args: Any,
    experiment_name: str | None = None,
    definitions_path: str | os.PathLike[str] | None = None,
    con: Any | None = None,
    **kwargs: Any,
) -> tuple[Experiment, _AnalysisResolver | _OfflineResolver]:
    """Build from definitions, using a bound versioned catalog when offline."""
    record_schema = kwargs.pop("_record_schema", True)
    values = {
        "experiment_name": experiment_name,
        "definitions_path": definitions_path,
        "con": con,
    }
    legacy = (
        len(args) == 3
        or (len(args) == 2 and isinstance(args[1], (str, os.PathLike)))
        or (len(args) == 1 and definitions_path is not None)
    )
    names = (
        ("experiment_name", "definitions_path", "con") if legacy else ("definitions_path", "con")
    )
    if len(args) > len(names):
        _refuse("kernel.request.invalid", field="too many positional definitions arguments")
    for name, value in zip(names, args, strict=False):
        if values[name] is not None:
            _refuse("kernel.request.invalid", field=f"{name} was supplied twice")
        values[name] = value
    experiment_name = values["experiment_name"]
    definitions_path = values["definitions_path"]
    con = values["con"]
    if definitions_path is None:
        _refuse("kernel.request.invalid", field="definitions_path is required")

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
        return Experiment.from_json(json.dumps(payload)), _OfflineResolver(
            path, str(catalog["definitions_digest"])
        )

    from increment._analysis_execution import Analysis
    from increment.semantics.loader import load

    definitions_digest = _definitions_digest(path) if record_schema else ""
    if experiment_name is None:
        definitions = load(path)
        if len(definitions.experiments) != 1:
            _refuse("kernel.request.invalid", field="experiment_name is required")
        experiment_name = definitions.experiments[0].name
    analysis = Analysis._from_definitions(experiment_name, path, con, **kwargs)
    experiment, resolver = _experiment_for_analysis(
        analysis, record_schema=record_schema, return_run_result=True
    )
    if record_schema:
        _write_catalog(path, experiment, definitions_digest)
        resolver.catalog_writer = lambda captured: _write_catalog(
            path, captured, definitions_digest
        )
    return experiment, resolver


def from_unit_day_artifact(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    from increment._analysis_execution import Analysis

    return _experiment_for_analysis(
        Analysis._from_unit_day_artifact(*args, **kwargs), return_run_result=True
    )


def from_switchback_frame(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    from increment._analysis_execution import Analysis

    return _experiment_for_analysis(
        Analysis._from_switchback_panel(*args, **kwargs), return_run_result=True
    )


def from_moments(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    from increment._analysis_execution import Analysis

    return _experiment_for_analysis(Analysis._from_moments(*args, **kwargs), return_run_result=True)


def from_unit_summary(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    return from_frame(*args, shape="summary", **kwargs)


def from_unit_panel(*args: Any, **kwargs: Any) -> tuple[Experiment, _AnalysisResolver]:
    return from_frame(*args, shape="panel", **kwargs)
