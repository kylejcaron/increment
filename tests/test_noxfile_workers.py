"""Nox pytest worker selection contracts."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
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


@pytest.mark.integration
def test_nightly_parts_run_numerical_cases_once_and_preserve_manual_selection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "tests" / "estimation").mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    numerical_certification\n    examples\n"
    )
    ordinary = tmp_path / "tests" / "test_ordinary.py"
    ordinary.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(4))\n"
        "def test_fast(value): assert value >= 0\n"
        "@pytest.mark.examples\n"
        "def test_example(): assert 2 + 2 == 4\n"
    )
    (tmp_path / "tests" / "estimation" / "test_conversion_route_calibration.py").write_text(
        "import pytest\n"
        "@pytest.mark.numerical_certification\n"
        "class TestEnumerationEnclosesTheExactMass:\n"
        "    @pytest.mark.parametrize('case', ['central', 'rare'])\n"
        "    def test_the_interval_holds_the_exact_rational_mass(self, case):\n"
        "        assert case in ('central', 'rare')\n"
    )
    (tmp_path / "tests" / "test_parity_harness.py").write_text(
        "import pytest\n"
        "@pytest.mark.numerical_certification\n"
        "@pytest.mark.parametrize('case', ['exact_binomial_beyond_the_former_arm_ceiling'])\n"
        "def test_parity_case(case): assert case\n"
    )
    (tmp_path / "noxfile.py").write_text(
        "import importlib.util\nimport nox\n"
        f"spec = importlib.util.spec_from_file_location('project_nox', {str(root / 'noxfile.py')!r})\n"
        "project = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(project)\n"
        "project._sync = lambda *args, **kwargs: None\n"
        "project.PYTEST_ARGS = ('-n', '0')\n"
        "@nox.session(venv_backend='none')\n"
        "def full(session): project.tests_full(session)\n"
    )
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    workflow = yaml.safe_load((root / ".github/workflows/nightly.yml").read_text())
    part_env = workflow["jobs"]["test-all"].get("env", {})

    def run(label: str, group: int | None = None, *, focused: bool = False) -> set[str]:
        report = tmp_path / f"{label}.xml"
        numerical = tmp_path / "junit-numerical.xml"
        numerical.unlink(missing_ok=True)
        child_env = {key: value for key, value in env.items() if key != "INCREMENT_NIGHTLY_PART"}
        if group is not None:
            child_env.update(
                {
                    key: str(group)
                    for key, value in part_env.items()
                    if value == "${{ matrix.group }}"
                }
            )
        args = ["tests/test_ordinary.py"] if focused else []
        args += ["-q", "-n", "0", f"--junitxml={report}"]
        if group is not None:
            args += [
                "--splits",
                "2",
                "--group",
                str(group),
                "--splitting-algorithm",
                "least_duration",
            ]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nox",
                "-f",
                str(tmp_path / "noxfile.py"),
                "-s",
                "full",
                "--",
                *args,
            ],
            cwd=tmp_path,
            env=child_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        expected_tail = not focused and group != 2
        assert numerical.exists() == expected_tail, result.stdout + result.stderr
        if expected_tail:
            cases = list(ET.parse(numerical).iter("testcase"))
            assert len(cases) == 3
            assert all(case.find("skipped") is None for case in cases)
        return {
            f"{case.get('classname')}::{case.get('name')}"
            for case in ET.parse(report).iter("testcase")
        }

    baseline = run("manual")
    assert run("first", 1)
    assert run("second", 2)
    assert len(baseline) == 5
    assert run("focused", focused=True) == baseline


@pytest.mark.xdist_group("examples_definitions_catalog")
@pytest.mark.integration
def test_nightly_full_suite_shards_keep_distinct_reports(tmp_path: Path) -> None:
    job = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())["jobs"]["test-all"]
    assert job["strategy"]["matrix"]["group"] == [1, 2]
    assert job["strategy"]["fail-fast"] is False
    assert job["timeout-minutes"] <= 90
    assert "${{ matrix.group }}" in job["name"]
    runs = [step.get("run", "") for step in job["steps"]]
    merge = next(run for run in runs if "> floor-test-durations.json" in run)
    root = Path(__file__).resolve().parents[1]
    for tier in ("fast", "slow", "exhaustive", "examples", "tooling"):
        source = root / f".github/{tier}-test-durations.json"
        if source.exists():
            target = tmp_path / source.relative_to(root)
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(source.read_bytes())
    result = subprocess.run(
        ["bash", "-eu", "-c", merge], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    weights = json.loads((tmp_path / "floor-test-durations.json").read_text())
    for tier in ("examples", "exhaustive", "tooling"):
        recorded = json.loads((root / f".github/{tier}-test-durations.json").read_text())
        assert recorded
        assert recorded.items() <= weights.items(), tier
    weekly = yaml.safe_load((root / ".github/workflows/weekly.yml").read_text())
    weekly_merge = next(
        step["run"]
        for step in weekly["jobs"]["test-all-floor"]["steps"]
        if "> floor-test-durations.json" in step.get("run", "")
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", weekly_merge], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "floor-test-durations.json").read_text()) == weights
    run = next(run for run in runs if "nox -s" in run)

    def collected(group: int | None) -> set[str]:
        args = shlex.split(run.replace("${{ matrix.group }}", str(group or 1)).split(" -- ", 1)[1])
        args[args.index("-n") + 1] = "0"
        args[args.index("--durations-path") + 1] = str(tmp_path / "floor-test-durations.json")
        args[args.index("--cpu-report") + 1] = str(tmp_path / "cpu.json")
        args = [
            f"--junitxml={tmp_path / 'collection.xml'}" if arg.startswith("--junitxml=") else arg
            for arg in args
        ]
        if group is None:
            for option in ("--splits", "--group"):
                index = args.index(option)
                del args[index : index + 2]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                TIER_MARKERS["all-except-scientific"],
                "tests/test_noxfile_workers.py",
                "tests/test_duration_report.py",
                "tests/acceptance/test_examples.py",
                "tests/acceptance/test_example_definitions.py",
                *args,
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return {line for line in result.stdout.splitlines() if line.startswith("tests/")}

    baseline, first, second = collected(None), collected(1), collected(2)
    assert first and second
    assert first | second == baseline
    assert not first & second
    examples = {node for node in baseline if node.startswith("tests/acceptance/")}
    assert examples
    assert examples <= weights.keys()
    report = next(
        step for step in job["steps"] if step.get("name") == "Retain full-suite JUnit report"
    )
    assert report["if"] == "always()"
    assert report["with"]["name"] == (
        "full-suite-junit-${{ matrix.python-version }}-shard-${{ matrix.group }}"
        "-${{ github.run_attempt }}"
    )
    assert set(report["with"]["path"].splitlines()) == {"junit.xml", "junit-numerical.xml"}


@pytest.mark.integration
def test_nightly_publishes_duration_weights_only_from_every_successful_part(tmp_path: Path) -> None:
    jobs = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())["jobs"]
    record = jobs["record-durations"]
    publish = jobs["publish-durations"]

    assert publish["needs"] == "record-durations"
    assert "needs.record-durations.result == 'success'" in publish["if"]
    parts: dict[str, set[tuple[int, int]]] = {}
    for entry in record["strategy"]["matrix"]["include"]:
        parts.setdefault(entry["tier"], set()).add((entry["group"], entry["splits"]))
    for tier, recorded in parts.items():
        (splits,) = {count for _, count in recorded}
        assert {group for group, _ in recorded} == set(range(1, splits + 1)), tier
    uploads = {
        step["with"]["path"]
        for step in publish["steps"]
        if "upload-artifact" in step.get("uses", "")
    }
    assert uploads == {f"weights/{tier}/" for tier in parts}
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".venv").symlink_to(root / ".venv", target_is_directory=True)
    (tmp_path / "scripts").symlink_to(root / "scripts", target_is_directory=True)
    for name in ("Makefile", "noxfile.py"):
        (tmp_path / name).write_bytes((root / name).read_bytes())
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "duration-fixture"\nversion = "0.0.0"\n'
        '[tool.pytest.ini_options]\nmarkers = ["examples", "tooling"]\n'
    )
    (tmp_path / "test_examples.py").write_text(
        "import pytest\npytestmark = pytest.mark.examples\n"
        "@pytest.mark.xdist_group('notebooks')\n"
        "@pytest.mark.parametrize('case', ['first', 'second'])\n"
        "def test_example(case): assert case\n"
    )
    (tmp_path / "test_tooling.py").write_text(
        "import pytest\npytestmark = pytest.mark.tooling\ndef test_tooling(): assert 1 + 1 == 2\n"
    )
    (tmp_path / ".github").mkdir()
    producer = next(
        step["run"] for step in record["steps"] if step.get("name", "").startswith("Record ")
    )
    describe = next(
        step["run"] for step in record["steps"] if step.get("name") == "Describe the recorded part"
    )
    env = {
        **os.environ,
        "UV_NO_SYNC": "1",
        "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
        "GITHUB_SHA": "a" * 40,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "WORKERS": "1",
    }
    fixture_parts = [("examples", 1, 2), ("examples", 2, 2), ("tooling", 1, 1)]
    recorded_weights: dict[str, dict[str, float]] = {}
    for tier, group, splits in fixture_parts:
        if tier not in parts:
            continue
        env.update(TIER=tier, GROUP=str(group), SPLITS=str(splits))
        result = subprocess.run(
            ["bash", "-eu", "-c", producer + "\n" + describe],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        target = tmp_path / "parts" / f"{tier}-{group}"
        target.parent.mkdir(exist_ok=True)
        (tmp_path / "durations-part").rename(target)
        measured = json.loads((target / f"{tier}-test-durations.json").read_text())
        recorded_weights.setdefault(tier, {}).update(measured)
    merge = next(
        step["run"]
        for step in publish["steps"]
        if step.get("name") == "Merge every part of each tier"
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", merge], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    examples = json.loads((tmp_path / "weights/examples/examples-test-durations.json").read_text())
    bare_nodes = {f"test_examples.py::test_example[{case}]" for case in ("first", "second")}
    recorded_examples = recorded_weights["examples"]
    assert {node.removesuffix("@notebooks") for node in recorded_examples} == bare_nodes
    assert set(examples) == recorded_examples.keys() | bare_nodes
    for node, seconds in recorded_examples.items():
        assert examples[node] == round(seconds, 3)
        assert examples[node.removesuffix("@notebooks")] == round(seconds, 3)
    tooling = json.loads((tmp_path / "weights/tooling/tooling-test-durations.json").read_text())
    assert set(tooling) == {"test_tooling.py::test_tooling"}
    assert tooling == {
        node: round(seconds, 3) for node, seconds in recorded_weights["tooling"].items()
    }
    metadata = json.loads((tmp_path / "weights/examples/metadata.json").read_text())
    assert metadata["source_sha"] == "a" * 40
    assert metadata["splits"] == 2
    assert set(parts) == {"fast", "slow", "exhaustive", "examples", "tooling"}
