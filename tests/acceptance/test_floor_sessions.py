"""Verify dependency floors in the interpreter that executes the probe."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_floor_session_executes_probe_with_declared_versions(tmp_path):
    import noxfile

    pins = [
        noxfile.DUCKDB_FLOORS["3.12"],
        next(pin for pin in noxfile.FLOOR_PINS["3.12"] if pin.startswith("numpy==")),
    ]
    probe = tmp_path / "test_probe_floor_versions.py"
    probe.write_text(
        "import duckdb, json, numpy, sys\n"
        "def test_report_versions():\n"
        "    versions = {'python': list(sys.version_info[:2]), "
        "'duckdb': duckdb.__version__, 'numpy': numpy.__version__}\n"
        "    print('FLOOR_PROBE=' + json.dumps(versions))\n"
    )
    # Nested environments are disposable, not test output; keep them outside tmp_path
    # so full-suite evidence does not archive duplicate package installations.
    with tempfile.TemporaryDirectory(prefix="increment-floor-envs-") as envdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nox",
                "--envdir",
                envdir,
                "-s",
                "tests_floor-3.12",
                "--",
                str(probe),
                "-s",
                "-q",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    assert not (tmp_path / "nox-envs").exists()

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    reports = [
        json.loads(line.removeprefix("FLOOR_PROBE="))
        for line in result.stdout.splitlines()
        if line.startswith("FLOOR_PROBE=")
    ]
    assert len(reports) == 1, output
    assert reports[0]["python"] == [3, 12]
    for pin in pins:
        requirement = Requirement(pin)
        assert reports[0][requirement.name] in requirement.specifier, output
