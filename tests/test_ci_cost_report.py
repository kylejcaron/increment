"""Runner and CPU minutes of a CI run, per job and in total."""

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import ci_cost_report
from scripts.ci_cost_report import CpuUsage, JobTiming, build_report

_START = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _job(job_id: int, minutes: float) -> JobTiming:
    return JobTiming(
        id=job_id,
        name=f"job {job_id}",
        attempt=1,
        started_at=_START,
        completed_at=_START + timedelta(minutes=minutes),
    )


def test_runner_minutes_sum_job_durations() -> None:
    report = build_report([_job(1, 10), _job(2, 5)], {})
    assert report.runner_minutes == 15.0


def test_cpu_minutes_include_worker_children() -> None:
    cpu = CpuUsage(
        user_s=60.0,
        system_s=0.0,
        children_user_s=540.0,
        children_system_s=0.0,
        wall_s=300.0,
        workers=2,
    )
    (job,) = build_report([_job(1, 5)], {"1": cpu}).jobs
    assert job.cpu_minutes == 10.0


def test_missing_cpu_artifact_is_unavailable_not_zero() -> None:
    report = build_report([_job(1, 10)], {})
    (job,) = report.jobs
    assert job.cpu_minutes is None
    assert job.unavailable == "no cpu artifact"
    assert report.cpu_minutes is None


def test_job_cpu_totals_every_pytest_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    def gh(*args: str) -> str:
        if args[0] == "api":
            return "cpu-full-suite-3.12-1\n"
        directory = Path(args[args.index("--dir") + 1]) / "cpu-full-suite-3.12-1"
        directory.mkdir()
        samples = (
            ("cpu.json", CpuUsage(30, 0, 30, 0, 40, 2)),
            ("cpu-numerical.json", CpuUsage(0, 20, 0, 40, 30, 2)),
        )
        for name, sample in samples:
            (directory / name).write_text(json.dumps(asdict(sample)), encoding="utf-8")
        return ""

    monkeypatch.setattr(ci_cost_report, "_gh", gh)
    cpu = ci_cost_report.load_cpu(1)
    assert cpu == {"1": CpuUsage(30, 20, 30, 40, 70, 2)}
    assert build_report([_job(1, 5)], cpu).cpu_minutes == 2.0
