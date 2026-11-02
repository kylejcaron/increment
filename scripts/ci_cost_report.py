"""Print runner-minutes and CPU-minutes per job and per run for CI workflow runs.

Runner-minutes are each executed job's wall time. CPU-minutes come from the
``cpu-*`` artifacts that test jobs upload from ``pytest --cpu-report``: the
controller's own CPU plus that of its reaped children, which include the xdist
workers and the subprocesses they waited for. Each artifact name ends with the
uploading job's ID, so a CPU record belongs to exactly one job execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

REPOSITORY = "kylejcaron/increment"
NO_CPU_ARTIFACT = "no cpu artifact"


@dataclass(frozen=True)
class JobTiming:
    """One execution of a workflow job."""

    id: int
    name: str
    attempt: int
    started_at: datetime
    completed_at: datetime

    @property
    def runner_minutes(self) -> float:
        return (self.completed_at - self.started_at).total_seconds() / 60


@dataclass(frozen=True)
class CpuUsage:
    """Seconds recorded by ``pytest --cpu-report`` on the controller."""

    user_s: float
    system_s: float
    children_user_s: float
    children_system_s: float
    wall_s: float
    workers: int

    @classmethod
    def from_json(cls, data: Mapping[str, float]) -> CpuUsage:
        return cls(
            user_s=float(data["user_s"]),
            system_s=float(data["system_s"]),
            children_user_s=float(data["children_user_s"]),
            children_system_s=float(data["children_system_s"]),
            wall_s=float(data["wall_s"]),
            workers=int(data["workers"]),
        )

    @property
    def cpu_minutes(self) -> float:
        seconds = self.user_s + self.system_s + self.children_user_s + self.children_system_s
        return seconds / 60


@dataclass(frozen=True)
class JobCost:
    job: JobTiming
    cpu: CpuUsage | None

    @property
    def cpu_minutes(self) -> float | None:
        return None if self.cpu is None else self.cpu.cpu_minutes

    @property
    def unavailable(self) -> str | None:
        """Why the job has no CPU figure; a missing record is never zero."""
        return NO_CPU_ARTIFACT if self.cpu is None else None


@dataclass(frozen=True)
class CostReport:
    jobs: tuple[JobCost, ...]

    @property
    def runner_minutes(self) -> float:
        return sum(cost.job.runner_minutes for cost in self.jobs)

    @property
    def measured_jobs(self) -> int:
        return sum(cost.cpu is not None for cost in self.jobs)

    @property
    def cpu_minutes(self) -> float | None:
        """CPU-minutes of the measured jobs, or None when no job was measured."""
        measured = [cost.cpu_minutes for cost in self.jobs if cost.cpu_minutes is not None]
        return sum(measured) if measured else None


def build_report(jobs: list[JobTiming], cpu: dict[str, CpuUsage]) -> CostReport:
    """Pair each job execution with the CPU record keyed by its job ID."""
    return CostReport(tuple(JobCost(job, cpu.get(str(job.id))) for job in jobs))


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def load_jobs(run_id: int) -> list[JobTiming]:
    """Every executed job of every attempt of a run.

    A re-run copies the jobs it does not repeat into the new attempt with their
    original timestamps; each execution counts once, under its first attempt.
    """
    endpoint = f"repos/{REPOSITORY}/actions/runs/{run_id}/jobs?per_page=100&filter=all"
    executions: dict[tuple[str, str, str], JobTiming] = {}
    for line in _gh("api", "--paginate", endpoint, "--jq", ".jobs[]").splitlines():
        job = json.loads(line)
        if job["status"] != "completed":
            raise SystemExit(f"run {run_id}: job {job['name']!r} is {job['status']}")
        if job["conclusion"] == "skipped":
            continue
        timing = JobTiming(
            id=job["id"],
            name=job["name"],
            attempt=job["run_attempt"],
            started_at=datetime.fromisoformat(job["started_at"]),
            completed_at=datetime.fromisoformat(job["completed_at"]),
        )
        key = (job["name"], job["started_at"], job["completed_at"])
        if key not in executions or timing.attempt < executions[key].attempt:
            executions[key] = timing
    return sorted(executions.values(), key=lambda timing: (timing.name, timing.attempt))


def load_cpu(run_id: int) -> dict[str, CpuUsage]:
    """CPU records of a run keyed by the job ID that ends each artifact name."""
    listing = _gh(
        "api",
        "--paginate",
        f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100",
        "--jq",
        '.artifacts[] | select((.name | startswith("cpu-")) and (.expired | not)) | .name',
    )
    if not listing.strip():
        return {}
    with tempfile.TemporaryDirectory() as directory:
        _gh(
            "run",
            "download",
            str(run_id),
            "--repo",
            REPOSITORY,
            "--pattern",
            "cpu-*",
            "--dir",
            directory,
        )
        cpu: dict[str, CpuUsage] = {}
        for artifact in Path(directory).glob("cpu-*"):
            samples = [
                CpuUsage.from_json(json.loads(path.read_text(encoding="utf-8")))
                for path in sorted(artifact.glob("cpu*.json"))
            ]
            if samples:
                cpu[artifact.name.rpartition("-")[2]] = CpuUsage(
                    user_s=sum(sample.user_s for sample in samples),
                    system_s=sum(sample.system_s for sample in samples),
                    children_user_s=sum(sample.children_user_s for sample in samples),
                    children_system_s=sum(sample.children_system_s for sample in samples),
                    wall_s=sum(sample.wall_s for sample in samples),
                    workers=max(sample.workers for sample in samples),
                )
        return cpu


def _row(
    width: int, name: str, attempt: str, runner: str, cpu: str, workers: str, note: str
) -> str:
    return f"{name:<{width}}  {attempt:>7}  {runner:>10}  {cpu:>7}  {workers:>7}  {note}".rstrip()


def _total(width: int, label: str, report: CostReport) -> str:
    cpu = report.cpu_minutes
    return _row(
        width,
        label,
        "",
        f"{report.runner_minutes:.1f}",
        "-" if cpu is None else f"{cpu:.1f}",
        "",
        f"cpu for {report.measured_jobs} of {len(report.jobs)} jobs",
    )


def render(reports: Sequence[tuple[int, CostReport]]) -> str:
    """Tabulate each run's jobs and totals, then the total over all runs."""
    names = [cost.job.name for _, report in reports for cost in report.jobs]
    width = max([len("all runs"), *map(len, names)])
    lines: list[str] = []
    for run_id, report in reports:
        lines += [
            f"run {run_id}",
            _row(width, "job", "attempt", "runner-min", "cpu-min", "workers", ""),
        ]
        for cost in report.jobs:
            cpu = cost.cpu
            lines.append(
                _row(
                    width,
                    cost.job.name,
                    str(cost.job.attempt),
                    f"{cost.job.runner_minutes:.1f}",
                    "-" if cpu is None else f"{cpu.cpu_minutes:.1f}",
                    "-" if cpu is None else str(cpu.workers),
                    cost.unavailable or "",
                )
            )
        lines += [_total(width, "run total", report), ""]
    if len(reports) > 1:
        combined = CostReport(tuple(chain.from_iterable(report.jobs for _, report in reports)))
        lines.append(_total(width, "all runs", combined))
    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=int,
        required=True,
        metavar="ID",
        help="workflow run ID; repeat for several runs",
    )
    args = parser.parse_args(argv)
    reports = [(run_id, build_report(load_jobs(run_id), load_cpu(run_id))) for run_id in args.runs]
    print(render(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
