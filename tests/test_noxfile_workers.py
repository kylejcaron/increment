"""Nox pytest worker selection contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from noxfile import TIER_MARKERS, _xdist_args


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
    assert _xdist_args({}) == ("-n", "__test_slots_workers__", "--dist", "loadgroup")


def test_workers_variable_overrides_allowance() -> None:
    """WORKERS overrides the local CPU-derived allowance."""
    assert _xdist_args({"WORKERS": "2"}) == (
        "-n",
        "2",
        "--dist",
        "loadgroup",
    )


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
