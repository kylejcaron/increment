"""Opt-in phase journals with resource sampling outside the measured process."""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import psutil


_THREAD_ENVIRONMENT = (
    "POLARS_MAX_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _resource_sample(process: "psutil.Process") -> dict[str, object]:
    import psutil

    with process.oneshot():
        cpu = process.cpu_times()
        memory = process.memory_info()
        try:
            threads = [thread._asdict() for thread in process.threads()]
        except psutil.AccessDenied:
            threads = None
        measured = {
            "pid": process.pid,
            "cpu_user_seconds": cpu.user,
            "cpu_system_seconds": cpu.system,
            "rss_bytes": memory.rss,
            "memory": memory._asdict(),
            "num_threads": process.num_threads(),
            "threads": threads,
            "threads_unavailable": "AccessDenied" if threads is None else None,
            "context_switches": process.num_ctx_switches()._asdict(),
            "status": process.status(),
        }
    memory = psutil.virtual_memory()
    return {
        "event": "sample",
        "monotonic_seconds": time.monotonic(),
        "process": measured,
        "system": {
            "load_average": psutil.getloadavg(),
            "cpu_times": psutil.cpu_times()._asdict(),
            "available_memory_bytes": memory.available,
            "memory_percent": memory.percent,
            "swap": psutil.swap_memory()._asdict(),
        },
    }


def _sample_until_closed(pid: int, created_at: float, destination: Path) -> None:
    import psutil

    stopped = threading.Event()

    def stop_on_eof() -> None:
        os.read(sys.stdin.fileno(), 1)
        stopped.set()

    threading.Thread(target=stop_on_eof, daemon=True).start()
    try:
        process = psutil.Process(pid)
        if process.create_time() != created_at:
            return
        with destination.open("a", encoding="utf-8", buffering=1) as stream:
            while not stopped.is_set() and process.is_running():
                stream.write(json.dumps(_resource_sample(process)) + "\n")
                stopped.wait(1)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return


class RuntimeDiagnostics:
    """Flush phase boundaries before native work can stall or be interrupted."""

    def __init__(self, directory: Path | None = None, *, nodeid: str = "") -> None:
        self._stream: TextIO | None = None
        self._sampler: subprocess.Popen[bytes] | None = None
        self._directory = directory
        self._manifest: dict[str, object] = {}
        if directory is None:
            return

        import psutil

        process = psutil.Process()
        resources = directory / "runtime-resources.jsonl"
        resources.write_text(json.dumps(_resource_sample(process)) + "\n", encoding="utf-8")
        self._stream = (directory / "runtime-events.jsonl").open("w", encoding="utf-8", buffering=1)
        self._sampler = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(process.pid),
                str(process.create_time()),
                str(resources),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        polars = sys.modules.get("polars")
        self._manifest = {
            "schema_version": 1,
            "nodeid": nodeid,
            "pid": process.pid,
            "sampler_pid": self._sampler.pid,
            "sampler_returncode": None,
            "events": "runtime-events.jsonl",
            "resources": "runtime-resources.jsonl",
            "native_pools": {
                "polars_threads": polars.thread_pool_size() if polars is not None else None,
                "environment": {
                    name: os.environ[name] for name in _THREAD_ENVIRONMENT if name in os.environ
                },
            },
        }
        self._save_manifest()

    def _save_manifest(self) -> None:
        assert self._directory is not None
        temporary = self._directory / "runtime-diagnostics.tmp"
        temporary.write_text(json.dumps(self._manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self._directory / "runtime-diagnostics.json")

    def _event(self, event: dict[str, object]) -> None:
        if self._stream is not None:
            event["monotonic_seconds"] = time.monotonic()
            self._stream.write(json.dumps(event) + "\n")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._stream is None:
            yield
            return
        started = time.monotonic()
        cpu_started = time.process_time()
        self._event({"event": "stage_start", "stage": name})
        outcome = "ok"
        exception_type = None
        try:
            yield
        except BaseException as error:
            outcome = "error"
            exception_type = type(error).__name__
            raise
        finally:
            self._event(
                {
                    "event": "stage_end",
                    "stage": name,
                    "wall_seconds": time.monotonic() - started,
                    "cpu_seconds": time.process_time() - cpu_started,
                    "outcome": outcome,
                    "exception_type": exception_type,
                }
            )

    def progress(
        self,
        *,
        completed_batches: int,
        completed_draws: int,
        planned_draws: int,
        batch_size: int,
    ) -> None:
        if self._stream is not None:
            self._event(
                {
                    "event": "progress",
                    "completed_batches": completed_batches,
                    "completed_draws": completed_draws,
                    "planned_draws": planned_draws,
                    "batch_size": batch_size,
                }
            )

    def close(self) -> None:
        if self._sampler is None:
            return
        try:
            _, errors = self._sampler.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._sampler.kill()
            _, errors = self._sampler.communicate(timeout=5)
        finally:
            if self._stream is not None:
                self._stream.close()
        self._manifest["sampler_returncode"] = self._sampler.returncode
        if errors:
            self._manifest["sampler_error"] = errors.decode("utf-8", errors="replace")
        self._save_manifest()
        if self._sampler.returncode:
            raise RuntimeError(f"Resource sampler failed: {self._manifest}")


if __name__ == "__main__":
    _sample_until_closed(int(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]))
