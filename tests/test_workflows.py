import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

pytestmark = pytest.mark.tooling

_PINNED_REF = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _is_local_workflow(ref: str) -> bool:
    """Same-repo reusable workflows resolve to the calling commit, so they cannot be pinned."""
    return ref.startswith("./")


def _uses_refs(node: Node) -> Iterator[tuple[int, str]]:
    if isinstance(node, MappingNode):
        for key, value in node.value:
            if isinstance(key, ScalarNode) and key.value == "uses":
                ref = value.value if isinstance(value, ScalarNode) else ""
                yield value.start_mark.line + 1, ref
            yield from _uses_refs(value)
    elif isinstance(node, SequenceNode):
        for value in node.value:
            yield from _uses_refs(value)


def _unpinned_uses(path: Path) -> list[tuple[int, str]]:
    root = yaml.compose(path.read_text())
    if root is None:
        return []
    return [
        (line_no, ref)
        for line_no, ref in _uses_refs(root)
        if not _is_local_workflow(ref) and _PINNED_REF.fullmatch(ref) is None
    ]


def test_workflow_actions_are_commit_pinned() -> None:
    bad = []
    workflow_dir = Path(".github/workflows")
    paths = [*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]
    for path in paths:
        for line_no, ref in _unpinned_uses(path):
            bad.append(f"{path}:{line_no}: uses: {ref}")
    assert not bad, "\n".join(bad)


def test_workflow_guard_catches_flow_mapping(tmp_path: Path) -> None:
    workflow = tmp_path / "flow.yaml"
    workflow.write_text("jobs:\n  test:\n    steps:\n      - {uses: actions/checkout@v4}\n")
    assert _unpinned_uses(workflow) == [(4, "actions/checkout@v4")]


def test_local_reusable_workflow_is_exempt_but_third_party_still_guarded(tmp_path: Path) -> None:
    """The local-ref exemption must not become a hole for unpinned third-party actions."""
    workflow = tmp_path / "call.yaml"
    workflow.write_text(
        "jobs:\n"
        "  gate:\n"
        "    uses: ./.github/workflows/ci.yml\n"
        "  smuggled:\n"
        "    steps:\n"
        "      - uses: evil/action@main\n"
    )
    assert [ref for _line_no, ref in _unpinned_uses(workflow)] == ["evil/action@main"]


def test_remote_reusable_workflow_still_requires_a_pin(tmp_path: Path) -> None:
    workflow = tmp_path / "remote.yaml"
    workflow.write_text("jobs:\n  gate:\n    uses: other/repo/.github/workflows/ci.yml@v1\n")
    assert [ref for _line_no, ref in _unpinned_uses(workflow)] == [
        "other/repo/.github/workflows/ci.yml@v1"
    ]


def test_selection_miss_shell_closes_python_heredoc_before_shell_commands() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["selection-misses"]
    run = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Merge predictions and report misses"
    )

    python_heredoc = re.search(r"(?m)^python\s+-\s+<<['\"]PY['\"]\s*$", run)
    if python_heredoc is not None:
        assert "\nPY\n" in run[python_heredoc.end() :]
        assert run.index("\nPY\n") < run.index("shopt -s globstar")


def test_miss_workflow_merges_shard_metadata_before_reporting() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["selection-misses"]
    run = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Merge predictions and report misses"
    )

    assert "scripts/selection_map.py merge" in run
    assert "--expected-shards" in run
    assert "junit-metadata.json" in run


def test_miss_report_uses_one_writer_and_preserves_metadata_in_summary() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["selection-misses"]
    run = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Merge predictions and report misses"
    )

    assert "tee selection-misses.txt" not in run
    assert 'cat selection-misses.txt >> "$GITHUB_STEP_SUMMARY"' in run


def test_nightly_selection_map_updates_bot_branch_with_lease() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())
    job = workflow["jobs"]["refresh-selection-map"]
    run = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Open or update selection-map PR"
    )

    assert "git fetch --prune origin" in run
    assert "--force-with-lease=refs/heads/bot/selection-map:" in run
    assert "gh pr edit bot/selection-map" in run
    # A map built on another ref names a base that a squash merge drops, so it arrives stale.
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "needs.record-durations.result == 'success'" in job["if"]


def test_nightly_refresh_jobs_do_not_wait_for_the_full_suite() -> None:
    jobs = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())["jobs"]
    assert "needs" not in jobs["record-durations"]
    for name in ("refresh-selection-map", "record-durations"):
        job = jobs[name]
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "test-all" not in needs, name
        assert "needs.test-all" not in job.get("if", ""), name


def test_nightly_full_suite_shards_keep_distinct_reports() -> None:
    job = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())["jobs"]["test-all"]
    assert job["strategy"]["matrix"]["group"] == [1, 2]
    assert job["strategy"]["fail-fast"] is False
    assert job["timeout-minutes"] <= 90
    assert "${{ matrix.group }}" in job["name"]
    runs = [step.get("run", "") for step in job["steps"]]
    assert any(
        "jq -s add .github/fast-test-durations.json .github/slow-test-durations.json" in run
        and "> floor-test-durations.json" in run
        for run in runs
    )
    run = next(run for run in runs if "nox -s" in run)
    assert "--splits 2 --group ${{ matrix.group }}" in run
    assert "--splitting-algorithm least_duration" in run
    assert "--durations-path floor-test-durations.json" in run
    report = next(
        step for step in job["steps"] if step.get("name") == "Retain full-suite JUnit report"
    )
    assert report["if"] == "always()"
    assert report["with"]["name"] == (
        "full-suite-junit-${{ matrix.python-version }}-shard-${{ matrix.group }}"
        "-${{ github.run_attempt }}"
    )
    assert set(report["with"]["path"].splitlines()) == {"junit.xml", "junit-numerical.xml"}


def test_nightly_publishes_duration_weights_only_from_every_successful_part() -> None:
    jobs = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())["jobs"]
    record = jobs["record-durations"]
    publish = jobs["publish-durations"]

    assert publish["needs"] == "record-durations"
    assert "needs.record-durations.result == 'success'" in publish["if"]
    parts: dict[str, set[tuple[int, int]]] = {}
    for entry in record["strategy"]["matrix"]["include"]:
        parts.setdefault(entry["tier"], set()).add((entry["group"], entry["splits"]))
    assert set(parts) == {"fast", "slow"}
    for tier, recorded in parts.items():
        (splits,) = {count for _, count in recorded}
        assert {group for group, _ in recorded} == set(range(1, splits + 1)), tier


def test_dispatch_without_base_reports_incomplete_evidence_advisory() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    test_job = workflow["jobs"]["test"]
    prediction = next(
        step for step in test_job["steps"] if step.get("name") == "Predict selected files"
    )

    assert prediction["continue-on-error"] is True
    assert "incomplete evidence: no base" in prediction["run"]
    assert 'if [ -z "$BASE" ]' in prediction["run"]

    report_job = workflow["jobs"]["selection-misses"]
    assert report_job["continue-on-error"] is True
    report = next(
        step
        for step in report_job["steps"]
        if step.get("name") == "Merge predictions and report misses"
    )
    assert "incomplete evidence: no base" in report["run"]


@pytest.mark.integration
def test_backend_absence_keeps_process_and_socket_guards_active() -> None:
    script = """
import multiprocessing
import socket
import subprocess
import sys
from _pytest.outcomes import Failed
import pytest
sys.modules["duckdb"] = None
sys.modules["ibis"] = None
from tests import _unit_boundary as boundary
boundary.install_guards()
boundary._current_unit_nodeid = "test_network_boundary"
operations = (
    ("socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 9))),
    ("subprocess", lambda: subprocess.Popen([sys.executable, "-c", "pass"])),
    ("process", lambda: multiprocessing.Process(target=lambda: None).start()),
)
for name, operation in operations:
    try:
        operation()
    except Failed:
        continue
    raise AssertionError(f"{name} guard allowed the operation")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
def test_duckdb_resource_binding_is_skipped_when_backends_are_absent() -> None:
    script = """
import sys
sys.modules["ibis"] = None
from tests._duckdb_resources import bound_duckdb_resources
bound_duckdb_resources()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_install_job_runs_listed_modules_without_warehouse_packages() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["core-install"]
    assert job["name"] == "core-install (${{ matrix.python-version }})"
    assert job["env"]["UV_NO_SYNC"] == "1"
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert (
        "uv sync --python ${{ matrix.python-version }} --no-default-groups --group core-test"
        in commands
    )
    assert any(step.get("name") == "Run README dataframe quickstart" for step in job["steps"])
    assert "README.md" in commands
    assert "exec(compile" in commands
    assert any(step.get("name") == "Verify warehouse packages are absent" for step in job["steps"])
    assert "--collect-only -q $(cat tests/core_modules.txt)" in commands
    assert '-m "not slow" $(cat tests/core_modules.txt)' in commands
    ci_gate = workflow["jobs"]["ci-ok"]
    assert "core-install" in ci_gate["needs"]
    gate_env = next(
        step["env"]
        for step in ci_gate["steps"]
        if step.get("name") == "Check required jobs and gate decisions"
    )
    assert gate_env["CORE_INSTALL_RESULT"] == "${{ needs.core-install.result }}"


def test_core_test_dependencies_exclude_warehouse_backends() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    requirements = project["dependency-groups"]["core-test"]
    assert {
        "pytest",
        "pytest-xdist",
        "pytest-timeout",
        "hypothesis",
        "sybil",
        "tach",
        "polars",
        "pandas",
    } <= {requirement.split(">")[0].split("=")[0] for requirement in requirements}
    assert not any(
        name in requirement.lower()
        for requirement in requirements
        for name in ("ibis", "sqlglot", "duckdb")
    )
