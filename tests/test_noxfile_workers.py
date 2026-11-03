"""Nox pytest worker selection contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from noxfile import TIER_MARKERS, _xdist_args
from scripts.test_slots import WORKERS_PLACEHOLDER

pytestmark = pytest.mark.tooling


def test_ci_keeps_auto_workers() -> None:
    """CI delegates worker sizing to xdist."""
    assert _xdist_args({"CI": "true", "WORKERS": "2"}) == (
        "-n",
        "auto",
        "--dist",
        "loadgroup",
    )


def test_local_workers_follow_lane_allowance() -> None:
    """Local Nox resolves worker count after acquiring its lane."""
    assert _xdist_args({}) == ("-n", WORKERS_PLACEHOLDER, "--dist", "loadgroup")


def test_workers_variable_overrides_allowance() -> None:
    """WORKERS overrides the local CPU-derived allowance."""
    assert _xdist_args({"WORKERS": "2"}) == (
        "-n",
        "2",
        "--dist",
        "loadgroup",
    )


@pytest.mark.integration
def test_warehouse_sessions_expand_only_to_interpreters_with_floor_pins() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "nox", "--list", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    sessions = json.loads(result.stdout)
    warehouse_sessions = {
        "tests",
        "tests_full",
        "tests_floor",
        "tests_slow_floor",
        "tests_floor_full",
    }
    actual = {
        name: {session["python"] for session in sessions if session["name"] == name}
        for name in warehouse_sessions | {"tests_core"}
    }
    warehouse_versions = {"3.12", "3.13", "3.14"}
    assert actual == {
        **dict.fromkeys(warehouse_sessions, warehouse_versions),
        "tests_core": warehouse_versions | {"3.15"},
    }


@pytest.mark.integration
def test_compile_registry_runs_only_in_its_own_tier() -> None:
    """The slow tier leaves the registry to its own tier, which selects exactly that module."""

    def collected(tier: str) -> set[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                TIER_MARKERS[tier],
                "tests/test_warehouse_compile_registry.py",
                "tests/test_noxfile_workers.py",
            ],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        return {line.split("::")[0] for line in result.stdout.splitlines() if "::" in line}

    assert "tests/test_warehouse_compile_registry.py" not in collected("slow")
    assert collected("compile-registry") == {"tests/test_warehouse_compile_registry.py"}


@pytest.mark.integration
def test_tooling_job_is_conditional_and_required_by_ci_ok() -> None:
    workflow = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    tooling = jobs["tooling"]
    steps = tooling["steps"]

    assert tooling["needs"] == ["lint", "changes"]
    assert tooling["if"] == "${{ needs.changes.outputs.tooling_required == 'true' }}"
    test_step = next(step for step in steps if "make test-tier TIER=tooling" in step.get("run", ""))
    assert "--cpu-report cpu.json" in test_step["run"]
    cpu_upload = next(step for step in steps if step.get("name") == "Retain tooling CPU usage")
    assert cpu_upload["if"] == "always()"
    assert cpu_upload["uses"] == (
        "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
    )
    assert cpu_upload["with"] == {
        "name": "cpu-tooling-${{ job.check_run_id }}",
        "path": "cpu.json",
        "if-no-files-found": "warn",
        "retention-days": 90,
    }

    assert "tooling_required" in jobs["changes"]["outputs"]
    assert "tooling" in jobs["ci-ok"]["needs"]
    gate_step = jobs["ci-ok"]["steps"][-1]
    assert "TOOLING_REQUIRED" in gate_step["env"]
    assert "TOOLING_RESULT" in gate_step["env"]
    assert 'check_conditional_job "$TOOLING_REQUIRED" "$TOOLING_RESULT" tooling' in gate_step["run"]
