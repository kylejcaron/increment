"""Measure per-operation evidence scopes against one run-owned capture.

Each repetition is an isolated Python child process so ``ru_maxrss`` is a
route-local peak. The benchmark deliberately uses the existing public readout
entry points in both arms. In the captured arm the breakout scope boundary is
borrowed from the already-open context; ``run`` receives that context through
its public argument, and ``daily`` reads the pinned loader.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

Route = Literal["per_operation", "captured"]
Backend = Literal["duckdb", "artifact"]


@dataclass(frozen=True, slots=True)
class Sample:
    wall_seconds: float
    peak_rss_bytes: int
    scope_opens: int
    pins: int
    materializations: int
    temp_bytes: int
    temp_bytes_after_scope: int
    session_lifetime_seconds: float
    cleanup: bool
    outputs: tuple[str, ...]
    coherence_live_seconds: float | None = None
    coherence_captured_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Measurement:
    route: Route
    backend: Backend
    rows: int
    seed: int
    repetitions: tuple[Sample, ...]
    wall_seconds: float
    wall_min_max: tuple[float, float]
    peak_rss_bytes: int
    rss_min_max: tuple[int, int]
    scope_opens: int
    pins: int
    materializations: int
    temp_bytes: int
    temp_bytes_after_scope: int
    session_lifetime_seconds: float
    cleanup: bool
    outputs: tuple[str, ...]
    outputs_equal: bool | None = None
    wall_threshold_pass: bool | None = None
    rss_threshold_pass: bool | None = None
    coherence_cost_seconds: float | None = None


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def _machine_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-Ao", "pid=,ppid=,%cpu=,%mem=,comm="],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=4)
        if len(fields) == 5:
            pid, ppid, cpu, memory, command = fields
            processes.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "cpu_percent": float(cpu),
                    "memory_percent": float(memory),
                    "command": command,
                }
            )
    processes.sort(
        key=lambda process: (process["cpu_percent"], process["memory_percent"]),
        reverse=True,
    )
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "load_average": list(os.getloadavg()),
        "top_processes": processes[:10],
    }


def _duckdb_connection(source: Any) -> Any:
    session = getattr(source, "_session", None)
    connection = getattr(session, "_con", None)
    if connection is None:
        connection = getattr(getattr(source, "_store", None), "_con", None)
    if connection is None:
        raise RuntimeError(f"{type(source).__name__} does not expose its DuckDB connection")
    return connection


def _temp_bytes(source: Any) -> int:
    connection = _duckdb_connection(source)
    value = connection.raw_sql(
        "SELECT coalesce(sum(size), 0) FROM duckdb_temporary_files()"
    ).fetchone()[0]
    return int(value or 0)


class _TempPeak:
    def __init__(self, source: Any, initial: int) -> None:
        self.source = source
        self.peak = initial
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cursor: Any | None = None
        self._error: Exception | None = None

    def __enter__(self) -> _TempPeak:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            if exc_type is None:
                self.finish()
            else:
                self._stop_sampling()
        except Exception:
            if exc_type is None:
                raise
        return False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TEMP sampling has already started")
        connection = _duckdb_connection(self.source)
        self._cursor = getattr(connection, "con", connection).cursor()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        assert self._cursor is not None
        while not self._stop.is_set():
            try:
                value = self._cursor.execute(
                    "SELECT coalesce(sum(size), 0) FROM duckdb_temporary_files()"
                ).fetchone()[0]
            except Exception as error:
                self._error = error
                return
            with self._lock:
                self.peak = max(self.peak, int(value or 0))
            self._stop.wait(0.01)

    def sample(self) -> None:
        value = _temp_bytes(self.source)
        with self._lock:
            self.peak = max(self.peak, value)

    def _stop_sampling(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            if self._error is not None:
                raise RuntimeError("independent TEMP sampling failed") from self._error

    def finish(self) -> int:
        self._stop_sampling()
        self.sample()
        return self.peak


def _temp_table_count(source: Any) -> int:
    connection = _duckdb_connection(source)
    return int(
        connection.raw_sql("SELECT count(*) FROM duckdb_tables() WHERE temporary").fetchone()[0]
    )


def _source(backend: Backend, root: Path) -> tuple[Any, Any]:
    if backend == "duckdb":
        from tests.source_conformance import definitions_source

        analysis, _ = definitions_source(root, store="always", return_analysis=True)
        source = analysis._src
        _duckdb_connection(source).raw_sql("SET enable_progress_bar = false")
        return analysis, source
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(root)
    return analysis, analysis._src


def _stable_value(value: Any) -> Any:
    if callable(getattr(value, "model_dump", None)):
        return {
            "__type__": type(value).__qualname__,
            **_stable_value(value.model_dump(mode="python")),
        }
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {
            "__type__": type(value).__qualname__,
            **{name: _stable_value(getattr(value, name)) for name in value.__dataclass_fields__},
        }
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_stable_value(item) for item in value]
    return value


def _identity(value: Any) -> tuple[str, ...]:
    fields = value if isinstance(value, dict) else {}
    identity_fields = (
        "analysis_population",
        "population",
        "ds",
        "metric",
        "group_id",
        "dimension",
        "dimension_value",
        "source",
        "method",
        "role",
    )
    return (
        str(fields.get("__type__", "")),
        *(f"{key}={fields[key]!r}" for key in identity_fields if key in fields),
        repr(value) if not any(key in fields for key in identity_fields) else "",
    )


def _canonical(rows: Any) -> str:
    values = sorted((_stable_value(row) for row in rows), key=_identity)
    metadata = getattr(rows, "metadata", None)
    qualifications = None
    if metadata is not None:
        coherence = getattr(metadata, "coherence", None)
        effective_modes = getattr(metadata, "effective_modes", None)
        if coherence is not None and effective_modes is not None:
            qualifications = {
                "coherence": coherence,
                "effective_modes": _normalize_effective_modes(effective_modes),
            }
    return json.dumps(
        {"metadata": qualifications, "rows": values},
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_effective_modes(modes: Any) -> Any:
    if not isinstance(modes, Mapping):
        return _stable_value(modes)
    normalized: dict[str, Any] = {}
    artifact_modes: list[Any] = []
    for identity, mode in modes.items():
        stable_mode = _stable_value(mode)
        if str(identity).startswith("artifact:"):
            artifact_modes.append(stable_mode)
        else:
            normalized[str(identity)] = stable_mode
    for index, mode in enumerate(sorted(artifact_modes, key=_identity)):
        normalized[f"artifact:stream:{index}"] = mode
    return dict(sorted(normalized.items()))


# Cross-mode artifact runs are independently published; R12 checks identity portability.
def _without_cross_mode_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                _normalize_effective_modes(item)
                if key == "effective_modes"
                else _without_cross_mode_identity(item)
            )
            for key, item in value.items()
            if key not in {"source_snapshot_id", "family_id"}
        }
    if isinstance(value, list):
        return [_without_cross_mode_identity(item) for item in value]
    return value


def _comparison_payload(value: Any) -> tuple[Any, Any]:
    if isinstance(value, dict) and "metadata" in value and "rows" in value:
        return value["metadata"], value["rows"]
    return None, value


def _outputs_equal(baseline: tuple[str, ...], candidate: tuple[str, ...]) -> bool:
    from tests.parity_harness.comparison import nested_close

    if len(baseline) != len(candidate):
        return False
    for expected, actual in zip(baseline, candidate, strict=True):
        expected_metadata, expected_rows = _comparison_payload(json.loads(expected))
        actual_metadata, actual_rows = _comparison_payload(json.loads(actual))
        if expected_metadata is None or actual_metadata is None:
            expected_value = expected_rows
            actual_value = actual_rows
        else:
            expected_value = {"metadata": expected_metadata, "rows": expected_rows}
            actual_value = {"metadata": actual_metadata, "rows": actual_rows}
        if not nested_close(
            _without_cross_mode_identity(expected_value),
            _without_cross_mode_identity(actual_value),
        ):
            return False
    return True


def _append_units(source: Any, count: int) -> None:
    if count <= 0:
        return
    stop = 9 + count
    source._session._con.raw_sql(
        f"""
        INSERT INTO raw_events
        SELECT
            'u' || CAST(unit_id AS VARCHAR),
            TIMESTAMP '2026-01-01 09:00:00',
            'assignment',
            CASE WHEN unit_id % 2 = 0 THEN 'control' ELSE 'treatment' END,
            CASE WHEN unit_id % 2 = 0 THEN 'US' ELSE 'CA' END,
            NULL
        FROM range(9, {stop}) AS units(unit_id)
        UNION ALL
        SELECT
            'u' || CAST(unit_id AS VARCHAR),
            TIMESTAMP '2026-01-01 12:00:00',
            'purchase',
            NULL,
            CASE WHEN unit_id % 2 = 0 THEN 'US' ELSE 'CA' END,
            CAST(unit_id % 4 AS DOUBLE) / 10000.0
        FROM range(9, {stop}) AS units(unit_id)
        """
    )


def _append_fixed_units(source: Any) -> None:
    source._session._con.raw_sql(
        "INSERT INTO raw_events VALUES "
        "('u5', TIMESTAMP '2026-01-01 09:00:00', 'assignment', 'control', 'US', NULL), "
        "('u5', TIMESTAMP '2026-01-01 12:00:00', 'purchase', NULL, 'US', 1.0), "
        "('u6', TIMESTAMP '2026-01-01 09:00:00', 'assignment', 'control', 'CA', NULL), "
        "('u6', TIMESTAMP '2026-01-01 12:00:00', 'purchase', NULL, 'CA', 2.0), "
        "('u7', TIMESTAMP '2026-01-01 09:00:00', 'assignment', 'treatment', 'US', NULL), "
        "('u7', TIMESTAMP '2026-01-01 12:00:00', 'purchase', NULL, 'US', 4.0), "
        "('u8', TIMESTAMP '2026-01-01 09:00:00', 'assignment', 'treatment', 'CA', NULL), "
        "('u8', TIMESTAMP '2026-01-01 12:00:00', 'purchase', NULL, 'CA', 5.0)"
    )


def _sample(route: Route, backend: Backend, rows: int, seed: int) -> Sample:  # noqa: C901, PLR0915
    if rows not in (8, 10_000_000):
        raise ValueError("supported workloads are eight rows and the 10M-row DuckDB stress case")
    if rows == 10_000_000 and backend != "duckdb":
        raise ValueError("the 10M-row stress case requires DuckDB")
    from increment import readouts
    from increment._evidence_types import BreakoutIdentity, EvidenceScope, EvidenceScopeResult
    from increment.evidence import evidence_provider
    from increment.readouts._evidence import _builtin_source

    with tempfile.TemporaryDirectory(prefix="kernel-context-measure-") as directory:
        analysis, source = _source(backend, Path(directory))
        if backend == "duckdb":
            _append_fixed_units(source)
            if rows == 10_000_000:
                _append_units(source, (rows - 16) // 2)
        from increment.readouts import _breakout

        breakout = analysis.experiment.breakouts[0]
        breakout_name = breakout.property
        metrics = tuple(source.context.metrics)
        configs = tuple(source.context.configs)
        scope = EvidenceScope(
            metrics=metrics,
            population="assigned",
            breakouts=(BreakoutIdentity(breakout.property, breakout.source),),
            configs=configs,
            execution=True,
            include_breakouts=True,
        )
        provider = evidence_provider(source)
        execution_context_type: Any = None
        if route == "captured":
            from increment.kernel.context import EvidenceExecutionContext

            execution_context_type = EvidenceExecutionContext

        def execute_workload(temp_sampler: _TempPeak | None = None) -> tuple[str, ...]:
            if route == "per_operation":
                run_rows = readouts.run(source)
                if temp_sampler is not None:
                    temp_sampler.sample()
                breakout_rows = readouts.breakout(source, breakout_name)
                if temp_sampler is not None:
                    temp_sampler.sample()
                daily_rows = readouts.daily(source)
                if temp_sampler is not None:
                    temp_sampler.sample()
                return (_canonical(run_rows), _canonical(breakout_rows), _canonical(daily_rows))

            assert execution_context_type is not None
            context = execution_context_type(
                provider,
                scope,
                run_id=f"measurement-{backend}-{rows}",
                coherence="captured",
            )
            with context.evidence_scope() as scoped_provider:
                run_rows = readouts.run(source, execution_context=context)
                if temp_sampler is not None:
                    temp_sampler.sample()
                pinned = _builtin_source(scoped_provider)
                breakout_rows = readouts.breakout(
                    pinned,
                    breakout_name,
                    metrics=[metric.name for metric in metrics],
                )
                if temp_sampler is not None:
                    temp_sampler.sample()
                daily_rows = readouts.daily(pinned)
                if temp_sampler is not None:
                    temp_sampler.sample()
                return (
                    _canonical(run_rows),
                    _canonical(breakout_rows),
                    _canonical(daily_rows),
                )

        execute_workload()
        scope_opens = 0
        pins = 0
        materializations = 0
        scope_lifetime = 0.0
        raw_provider = evidence_provider(source)
        provider_type = type(raw_provider)
        original_provider_scope = provider_type.scope

        def counted_provider_scope(self: Any, request: Any) -> Any:
            nonlocal scope_opens, scope_lifetime
            scope_opens += 1
            result = original_provider_scope(self, request)

            @contextmanager
            def timed_source() -> Iterator[Any]:
                nonlocal scope_lifetime
                started_scope = time.perf_counter()
                try:
                    with result.source as pinned:
                        yield pinned
                finally:
                    scope_lifetime += time.perf_counter() - started_scope

            return replace(result, source=timed_source())

        setattr(provider_type, "scope", counted_provider_scope)  # noqa: B010
        provider = evidence_provider(source)
        original_breakout_scope = _breakout.evidence_scope

        @contextmanager
        def counted_breakout_scope(*args: Any, **kwargs: Any) -> Iterator[Any]:
            nonlocal scope_opens
            scope_opens += 1
            with original_breakout_scope(*args, **kwargs) as pinned:
                yield pinned

        setattr(_breakout, "evidence_scope", counted_breakout_scope)  # noqa: B010
        original_snapshot = getattr(source, "readout_snapshot", None)
        session = getattr(source, "_session", None)
        session_type = type(session) if session is not None else None
        original_materialize = (
            getattr(session_type, "materialize_table", None) if session_type is not None else None
        )
        if callable(original_materialize):
            assert session_type is not None

            def counted_materialize(self: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal materializations
                materializations += 1
                return original_materialize(self, *args, **kwargs)

            setattr(session_type, "materialize_table", counted_materialize)  # noqa: B010
        original_pinned_source = getattr(source, "_pinned_source", None)
        if callable(original_pinned_source):
            assert session is not None

            def counted_pinned_source(*args: Any, **kwargs: Any) -> Any:
                nonlocal pins
                pinned = original_pinned_source(*args, **kwargs)
                if pinned is not source:
                    pins += 1
                return pinned

            source._pinned_source = counted_pinned_source
        elif callable(original_snapshot):

            @contextmanager
            def counted_snapshot(*args: Any, **kwargs: Any) -> Iterator[Any]:
                nonlocal pins
                pins += 1
                with original_snapshot(*args, **kwargs) as pinned:
                    yield pinned

            source.readout_snapshot = counted_snapshot

        before_temp_tables = _temp_table_count(source)
        before_temp_bytes = _temp_bytes(source)
        outputs: tuple[str, ...]
        temp_sampler = _TempPeak(source, before_temp_bytes)
        started = time.perf_counter()
        with temp_sampler:
            outputs = execute_workload(temp_sampler)
        temp_bytes = temp_sampler.peak
        setattr(provider_type, "scope", original_provider_scope)  # noqa: B010
        setattr(_breakout, "evidence_scope", original_breakout_scope)  # noqa: B010
        if callable(original_materialize):
            assert session_type is not None
            setattr(session_type, "materialize_table", original_materialize)  # noqa: B010
        if callable(original_pinned_source):
            source._pinned_source = original_pinned_source
        elif callable(original_snapshot):
            source.readout_snapshot = original_snapshot
        wall = time.perf_counter() - started
        temp_bytes_after_scope = _temp_bytes(source)
        analysis.close()
        cleanup = _temp_table_count(source) <= before_temp_tables
        return Sample(
            wall_seconds=wall,
            peak_rss_bytes=_rss_bytes(),
            scope_opens=scope_opens,
            pins=pins,
            materializations=materializations,
            temp_bytes=temp_bytes,
            temp_bytes_after_scope=temp_bytes_after_scope,
            session_lifetime_seconds=scope_lifetime,
            cleanup=cleanup,
            outputs=outputs,
        )


def _coherence_sample(rows: int, seed: int) -> tuple[float, float]:
    """Measure plain-mean snapshot cost on its own definitions-backed source."""
    if rows not in (8, 10_000_000):
        raise ValueError("supported workloads are eight rows and the 10M-row DuckDB stress case")
    from increment import readouts

    with tempfile.TemporaryDirectory(prefix="kernel-context-coherence-") as directory:
        analysis, source = _source("duckdb", Path(directory))
        try:
            _append_fixed_units(source)
            if rows == 10_000_000:
                _append_units(source, (rows - 16) // 2)
            mean_metrics = tuple(
                metric.name
                for metric in source.context.metrics
                if getattr(metric, "type", None) == "mean"
            )
            if not mean_metrics:
                raise RuntimeError("coherence-cost fixture needs a plain mean metric")
            for coherence in ("live", "captured"):
                readouts.run(source, metrics=mean_metrics, coherence=coherence)
            order = ("live", "captured") if seed % 2 else ("captured", "live")
            times: dict[str, float] = {}
            for coherence in order:
                started = time.perf_counter()
                readouts.run(source, metrics=mean_metrics, coherence=coherence)
                times[coherence] = time.perf_counter() - started
            return times["live"], times["captured"]
        finally:
            analysis.close()


def _coherence_sample_child(rows: int, seed: int) -> tuple[float, float]:
    command = [
        sys.executable,
        "-m",
        "calibration.kernel_context_measurement",
        "--coherence-sample",
        str(rows),
        str(seed),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return payload["coherence_live_seconds"], payload["coherence_captured_seconds"]


def _sample_child(
    route: Route, backend: Backend, rows: int, seed: int, *, baseline_root: Path | None
) -> Sample:
    args = ["--sample", route, backend, str(rows), str(seed)]
    if route == "per_operation" and baseline_root is not None:
        wrapper = (
            "import importlib.util,sys; "
            "root=sys.argv.pop(1); path=sys.argv.pop(1); sys.argv[0]=path; sys.path.insert(0,root); "
            "spec=importlib.util.spec_from_file_location('kernel_measurement',path); "
            "module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; "
            "spec.loader.exec_module(module); "
            "module.main()"
        )
        command = [
            sys.executable,
            "-c",
            wrapper,
            str(baseline_root),
            str(Path(__file__).resolve()),
            *args,
        ]
    else:
        command = [sys.executable, "-m", "calibration.kernel_context_measurement", *args]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=baseline_root if route == "per_operation" and baseline_root is not None else None,
    )
    return Sample(**json.loads(result.stdout))


def _summarize(
    route: Route,
    backend: Backend,
    rows: int,
    seed: int,
    samples: tuple[Sample, ...],
) -> Measurement:
    wall_values = [sample.wall_seconds for sample in samples]
    rss_values = [sample.peak_rss_bytes for sample in samples]
    coherence_costs = [
        sample.coherence_captured_seconds - sample.coherence_live_seconds
        for sample in samples
        if sample.coherence_live_seconds is not None
        and sample.coherence_captured_seconds is not None
    ]
    return Measurement(
        route=route,
        backend=backend,
        rows=rows,
        seed=seed,
        repetitions=samples,
        wall_seconds=statistics.median(wall_values),
        wall_min_max=(min(wall_values), max(wall_values)),
        peak_rss_bytes=int(statistics.median(rss_values)),
        rss_min_max=(min(rss_values), max(rss_values)),
        scope_opens=sum(sample.scope_opens for sample in samples) // len(samples),
        pins=sum(sample.pins for sample in samples) // len(samples),
        materializations=sum(sample.materializations for sample in samples) // len(samples),
        temp_bytes=max(sample.temp_bytes for sample in samples),
        temp_bytes_after_scope=max(sample.temp_bytes_after_scope for sample in samples),
        session_lifetime_seconds=statistics.median(
            sample.session_lifetime_seconds for sample in samples
        ),
        cleanup=all(sample.cleanup for sample in samples),
        outputs=samples[0].outputs,
        coherence_cost_seconds=(
            statistics.median(coherence_costs) if len(coherence_costs) == len(samples) else None
        ),
    )


def measure(
    *, route: Route, backend: Backend, rows: int, seed: int, baseline_root: Path | None = None
) -> Measurement:
    samples = tuple(
        _sample_child(
            route,
            backend,
            rows,
            seed + repetition,
            baseline_root=baseline_root,
        )
        for repetition in range(7)
    )
    if route == "captured":
        samples = _with_coherence_costs(samples, rows=rows, seed=seed)
    return _summarize(route, backend, rows, seed, samples)


def _with_coherence_costs(
    samples: tuple[Sample, ...], *, rows: int, seed: int
) -> tuple[Sample, ...]:
    measured: list[Sample] = []
    for repetition, sample in enumerate(samples):
        live, captured = _coherence_sample_child(rows, seed + repetition)
        measured.append(
            replace(
                sample,
                coherence_live_seconds=live,
                coherence_captured_seconds=captured,
            )
        )
    return tuple(measured)


def _measure_pair(
    *, backend: Backend, rows: int, seed: int, baseline_root: Path
) -> tuple[Measurement, Measurement]:
    baseline_samples: list[Sample] = []
    candidate_samples: list[Sample] = []
    for repetition in range(7):
        arms: tuple[Route, Route] = (
            ("per_operation", "captured") if repetition % 2 == 0 else ("captured", "per_operation")
        )
        for route in arms:
            sample = _sample_child(
                route,
                backend,
                rows,
                seed + repetition,
                baseline_root=baseline_root if route == "per_operation" else None,
            )
            (baseline_samples if route == "per_operation" else candidate_samples).append(sample)
    captured_samples = _with_coherence_costs(tuple(candidate_samples), rows=rows, seed=seed)
    return (
        _summarize("per_operation", backend, rows, seed, tuple(baseline_samples)),
        _summarize("captured", backend, rows, seed, captured_samples),
    )


def _has_captured_qualification(outputs: tuple[str, ...]) -> bool:
    # Workload outputs are ordered run, breakout, daily; daily rows have no metadata.
    if not outputs:
        return False
    payload = json.loads(outputs[0])
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    modes = metadata.get("effective_modes") if isinstance(metadata, dict) else None
    return (
        isinstance(metadata, dict)
        and metadata.get("coherence") == "captured"
        and isinstance(modes, dict)
        and bool(modes)
        and all(mode == "captured" for mode in modes.values())
    )


def _set_gate(baseline: Measurement, candidate: Measurement) -> tuple[Measurement, Measurement]:
    baseline_consistent = all(
        _outputs_equal(baseline.outputs, sample.outputs) for sample in baseline.repetitions
    )
    candidates_match = all(
        _outputs_equal(baseline.outputs, sample.outputs) for sample in candidate.repetitions
    )
    candidate_qualified = candidate.route != "captured" or all(
        _has_captured_qualification(sample.outputs) for sample in candidate.repetitions
    )
    outputs_equal = baseline_consistent and candidates_match and candidate_qualified
    wall_pass = outputs_equal and candidate.wall_seconds <= baseline.wall_seconds
    rss_pass = outputs_equal and candidate.peak_rss_bytes <= 1.5 * baseline.peak_rss_bytes
    return (
        replace(
            baseline,
            outputs_equal=None,
            wall_threshold_pass=None,
            rss_threshold_pass=None,
        ),
        replace(
            candidate,
            outputs_equal=outputs_equal,
            wall_threshold_pass=wall_pass,
            rss_threshold_pass=rss_pass,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", nargs=4, metavar=("ROUTE", "BACKEND", "ROWS", "SEED"))
    parser.add_argument("--coherence-sample", nargs=2, metavar=("ROWS", "SEED"))
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    if args.coherence_sample:
        with redirect_stdout(sys.stderr):
            live, captured = _coherence_sample(
                int(args.coherence_sample[0]),
                int(args.coherence_sample[1]),
            )
        print(
            json.dumps(
                {
                    "coherence_live_seconds": live,
                    "coherence_captured_seconds": captured,
                }
            )
        )
        return
    if args.sample:
        with redirect_stdout(sys.stderr):
            sample = _sample(
                args.sample[0], args.sample[1], int(args.sample[2]), int(args.sample[3])
            )
        print(json.dumps(asdict(sample)))
        return
    if args.baseline_root is None:
        parser.error("--baseline-root is required for a full measurement")
    host_start = _machine_snapshot()
    records: list[dict[str, Any]] = []
    cases = (("duckdb", 8), ("artifact", 8), ("duckdb", 10_000_000))
    for backend, rows in cases:
        baseline, candidate = _measure_pair(
            backend=backend,
            rows=rows,
            seed=1,
            baseline_root=args.baseline_root,
        )
        baseline, candidate = _set_gate(baseline, candidate)
        records.extend((asdict(baseline), asdict(candidate)))
    payload = json.dumps(
        {
            "host": {"start": host_start, "end": _machine_snapshot()},
            "measurements": records,
        },
        indent=2,
    )
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
