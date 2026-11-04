import ast
import json
import re
import shlex
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
    assert set(parts) == {"fast", "slow", "exhaustive"}
    for tier, recorded in parts.items():
        (splits,) = {count for _, count in recorded}
        assert {group for group, _ in recorded} == set(range(1, splits + 1)), tier
    uploads = {
        step["with"]["path"]
        for step in publish["steps"]
        if "upload-artifact" in step.get("uses", "")
    }
    assert uploads == {f"weights/{tier}/" for tier in parts}


def test_dispatch_without_base_reports_incomplete_evidence_advisory() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    report_job = workflow["jobs"]["selection-misses"]
    prediction = next(
        step for step in report_job["steps"] if step.get("name") == "Predict selected files"
    )

    assert prediction["continue-on-error"] is True
    assert "incomplete evidence: no base" in prediction["run"]
    assert 'if [ -z "$BASE" ]' in prediction["run"]

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
from tests._duckdb_resources import bound_duckdb_resources
bound_duckdb_resources()
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
    core_test_run = next(step for step in job["steps"] if step.get("name") == "Run core tests")
    assert "--cpu-report cpu.json" in core_test_run["run"]
    cpu_upload = next(step for step in job["steps"] if step.get("name") == "Retain core CPU usage")
    assert cpu_upload["if"] == "always()"
    assert cpu_upload["continue-on-error"] is True
    assert (
        cpu_upload["with"]["name"]
        == "cpu-core-${{ matrix.python-version }}-${{ job.check_run_id }}"
    )
    assert cpu_upload["with"]["path"] == "cpu.json"
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


def test_weekly_floor_parts_cover_each_interpreter_with_distinct_reports() -> None:
    job = yaml.safe_load(Path(".github/workflows/weekly.yml").read_text())["jobs"]["test-all-floor"]
    matrix = job["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.12", "3.13", "3.14"]
    assert matrix["group"] == [1, 2, 3]
    assert job["timeout-minutes"] == 90
    assert "part ${{ matrix.group }}/3)" in job["name"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "jq -s add .github/fast-test-durations.json .github/slow-test-durations.json" in commands
    assert "--splits 3 --group ${{ matrix.group }}" in commands
    assert (
        "--splitting-algorithm least_duration --durations-path floor-test-durations.json"
        in commands
    )
    assert job["env"]["INCREMENT_FLOOR_PART"] == "${{ matrix.group }}"
    upload = next(
        step for step in job["steps"] if step.get("name") == "Retain floor-suite JUnit report"
    )
    assert "-part-${{ matrix.group }}-" in upload["with"]["name"]
    assert upload["if"] == "always()"


def test_pull_requests_run_the_full_fast_tier_on_one_python() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    fast_job = jobs["test"]
    matrix_expression = fast_job["strategy"]["matrix"]["include"]
    matrix_literals = re.findall(r"'(\[\{.*?\}\])'", matrix_expression)
    pull_request_matrix, push_matrix = map(json.loads, matrix_literals)

    assert [
        (entry["python-version"], entry["group"], entry["splits"]) for entry in pull_request_matrix
    ] == [("3.12", group, 3) for group in (1, 2, 3)]
    assert {
        (entry["python-version"], entry["group"], entry["splits"]) for entry in push_matrix
    } == {(python, group, 3) for python in ("3.12", "3.13", "3.14") for group in (1, 2, 3)}
    fast_tests = next(step for step in fast_job["steps"] if step.get("id") == "fast-tests")
    assert "matrix.python-version" in fast_tests["run"]
    assert "test-reduction-approved" in fast_tests["run"]
    report_step = next(
        step
        for step in jobs["selection-misses"]["steps"]
        if step.get("name") == "Merge predictions and report misses"
    )
    assert report_step["env"]["EXPECTED_SHARDS"] == (
        "${{ github.event_name == 'pull_request' && '3.12-1,3.12-2,3.12-3' || "
        "'3.12-1,3.12-2,3.12-3,3.13-1,3.13-2,3.13-3,3.14-1,3.14-2,3.14-3' }}"
    )

    assert not any(step.get("name") == "Predict selected files" for step in fast_job["steps"])
    selection_job = jobs["selection-misses"]
    prediction = next(
        step for step in selection_job["steps"] if step.get("name") == "Predict selected files"
    )
    assert prediction["env"]["AFFECTED_SHARD_ID"] == "all"
    assert "--co" in prediction["run"]
    assert "--splits" not in prediction["run"]
    assert "--group" not in prediction["run"]
    prediction_artifact = next(
        step for step in selection_job["steps"] if step.get("name") == "Retain selection prediction"
    )
    assert prediction_artifact["with"]["name"].startswith("affected-report-all-")

    newest = jobs["unit-newest"]
    assert newest["name"] == "unit (3.14)"
    setup_uv = next(
        step for step in newest["steps"] if step.get("uses", "").startswith("astral-sh/setup-uv@")
    )
    assert setup_uv["with"]["python-version"] == "3.14"
    assert any("make test-unit" in step.get("run", "") for step in newest["steps"])

    gate = jobs["ci-ok"]
    assert "unit-newest" in gate["needs"]
    duration_job = jobs["duration-ratchet"]
    assert duration_job["needs"] == ["test", "slow"]
    duration_report = next(
        step
        for step in duration_job["steps"]
        if step.get("name") == "Report aggregate per-layer duration ratchet"
    )
    assert 'tee -a "$GITHUB_STEP_SUMMARY"' in duration_report["run"]
    assert "continue-on-error" not in duration_report
    assert "duration-ratchet" in gate["needs"]
    assert duration_report.get("shell") == "bash"
    assert "3.12-1,3.12-2,3.12-3,slow-1" in duration_report["env"]["EXPECTED_DURATION_SHARDS"]
    assert "3.13-1,3.13-2,3.13-3,3.14-1" in duration_report["env"]["EXPECTED_DURATION_SHARDS"]
    from scripts.duration_report import DURATION_ARTIFACT_PREFIXES

    downloads = [
        step["with"]
        for step in duration_job["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert {download["pattern"] for download in downloads} == {
        f"{prefix}*" for prefix in DURATION_ARTIFACT_PREFIXES
    }
    assert {download["path"] for download in downloads} == {"duration-artifacts"}
    slow_job = jobs["slow"]
    slow_tests = next(step for step in slow_job["steps"] if step.get("id") == "slow-tests")
    assert slow_tests["env"]["INCREMENT_COLLECT_SHARD_ID"] == "slow-${{ matrix.group }}"
    slow_artifact = next(
        step for step in slow_job["steps"] if step.get("name") == "Retain slow JUnit report"
    )
    assert {"junit.xml", "junit-collection.json", "junit-metadata.json"} == set(
        slow_artifact["with"]["path"].splitlines()
    )
    gate_step = next(
        step
        for step in gate["steps"]
        if step.get("name") == "Check required jobs and gate decisions"
    )
    assert "${{ needs.duration-ratchet.result }}" in gate_step["run"]
    assert gate_step["env"]["UNIT_NEWEST_REQUIRED"] == "${{ github.event_name == 'pull_request' }}"
    assert gate_step["env"]["UNIT_NEWEST_RESULT"] == "${{ needs.unit-newest.result }}"
    run = gate_step["run"]
    assert 'check_conditional_job "$UNIT_NEWEST_REQUIRED" "$UNIT_NEWEST_RESULT" unit-newest' in run


def _selects(expression: str, *marks: str) -> bool:
    """Whether a pytest ``-m`` expression selects an item carrying exactly *marks*."""
    from _pytest.mark import MarkMatcher
    from _pytest.mark.expression import Expression

    matcher = MarkMatcher.from_markers(getattr(pytest.mark, mark).mark for mark in marks)
    return Expression.compile(expression).evaluate(matcher)


def _noxfile() -> tuple[dict[str, str], dict[str, str]]:
    """Tier markers and session sources, read without importing ``nox``.

    The core-install job runs this module without development tools.
    """
    source = Path("noxfile.py").read_text()
    tree = ast.parse(source)
    markers = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TIER_MARKERS" for t in node.targets)
    )
    functions = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    return markers, functions


def test_nightly_and_main_select_exhaustive() -> None:
    markers, functions = _noxfile()
    full_tier = markers["all-except-scientific"]
    assert _selects(full_tier, "slow", "exhaustive")
    assert _selects(full_tier, "integration", "slow", "exhaustive")
    assert '"all-except-scientific"' in functions["tests_full"]
    nightly = yaml.safe_load(Path(".github/workflows/nightly.yml").read_text())
    full_suite = nightly["jobs"]["test-all"]
    assert full_suite["if"] == "github.event_name != 'pull_request'"
    assert any('nox -s "tests_full-' in step.get("run", "") for step in full_suite["steps"])

    ci = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    # PyYAML reads the `on:` key as a boolean.
    assert ci[True]["push"] == {"branches": ["main"]}
    changes = ci["jobs"]["changes"]
    assert changes["outputs"]["exhaustive_required"] == "${{ steps.exhaustive.outputs.required }}"
    selector = next(step for step in changes["steps"] if step.get("id") == "exhaustive")
    assert "scripts/certification_ci_changes.py --gate exhaustive" in selector["run"]
    assert selector["env"]["EVENT"] == "${{ github.event_name }}"
    job = ci["jobs"]["exhaustive"]
    assert job["if"] == "${{ needs.changes.outputs.exhaustive_required == 'true' }}"
    groups = job["strategy"]["matrix"]["group"]
    assert groups == list(range(1, len(groups) + 1))
    run = "\n".join(step.get("run", "") for step in job["steps"])
    assert run.count("make test-exhaustive ") == 1
    assert f"--splits {len(groups)} --group ${{{{ matrix.group }}}}" in run
    exhaustive = markers["exhaustive"]
    assert _selects(exhaustive, "slow", "exhaustive")
    assert _selects(exhaustive, "integration", "slow", "exhaustive")
    assert not _selects(exhaustive, "slow")
    assert not _selects(exhaustive, "slow", "exhaustive", "numerical_certification")
    # Every route that runs the exhaustive tier generates the full matrix scope.
    assert nightly["env"]["INCREMENT_PARITY_MATRIX"] == "full"
    weekly = yaml.safe_load(Path(".github/workflows/weekly.yml").read_text())["jobs"]
    assert weekly["parity-matrix-10x"]["env"]["INCREMENT_PARITY_MATRIX"] == "full"
    assert any(
        step.get("run", "").startswith("make test-exhaustive ")
        for step in weekly["newest-dependencies"]["steps"]
    )
    makefile = Path("Makefile").read_text()
    for target in ("test-exhaustive", "test-all"):
        recipe = re.search(rf"(?m)^{target}:\n\t(.*)$", makefile)
        assert recipe is not None, target
        assert recipe.group(1).startswith("INCREMENT_PARITY_MATRIX=full "), target
    for target in ("check", "ci-local"):
        prerequisites = re.search(rf"(?m)^{target}:(.*)$", makefile)
        assert prerequisites is not None, target
        assert "test-exhaustive" in prerequisites.group(1).split(), target


def test_pr_slow_job_excludes_exhaustive_unless_required() -> None:
    markers, _ = _noxfile()

    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    slow_job = jobs["slow"]
    assert "if" not in slow_job
    step = next(step for step in slow_job["steps"] if step.get("name") == "Run slow tests")
    assert step["run"].startswith("make test-slow ")
    slow = markers["slow"]
    assert json.loads(step["env"]["INCREMENT_COLLECT_TIERS"]) == {"slow": slow}
    assert _selects(slow, "slow")
    assert _selects(slow, "integration", "slow")
    assert not _selects(slow, "slow", "exhaustive")
    assert not _selects(slow, "integration", "slow", "exhaustive")

    job = jobs["exhaustive"]
    assert job["needs"] == ["changes"]
    assert job["if"] == "${{ needs.changes.outputs.exhaustive_required == 'true' }}"
    gate = jobs["ci-ok"]
    assert "exhaustive" in gate["needs"]
    gate_step = next(
        step
        for step in gate["steps"]
        if step.get("name") == "Check required jobs and gate decisions"
    )
    assert gate_step["env"]["EXHAUSTIVE_REQUIRED"] == (
        "${{ needs.changes.outputs.exhaustive_required }}"
    )
    assert gate_step["env"]["EXHAUSTIVE_RESULT"] == "${{ needs.exhaustive.result }}"
    assert (
        'check_conditional_job "$EXHAUSTIVE_REQUIRED" "$EXHAUSTIVE_RESULT" exhaustive'
        in gate_step["run"]
    )


def _makefile_recipes() -> dict[str, str]:
    """Recipe text of each Makefile target."""
    recipes: dict[str, str] = {}
    target = None
    for line in Path("Makefile").read_text().splitlines():
        rule = re.match(r"([\w.-]+):(?!=)", line)
        if rule is not None:
            target = rule.group(1)
            recipes[target] = ""
        elif line.startswith("\t") and target is not None:
            recipes[target] += line + "\n"
    return recipes


def _make_tiers(arguments: list[str], recipes: dict[str, str]) -> set[str]:
    """Test tiers a ``make`` invocation runs, following nested ``$(MAKE)`` calls."""
    variables = dict(argument.split("=", 1) for argument in arguments if "=" in argument)
    tiers: set[str] = set()
    for target in arguments:
        if "=" in target or target.startswith("-"):
            continue
        if target == "test-tier":
            tiers.add(variables["TIER"])
        for nested in re.findall(r"\$\(MAKE\) (.*)", recipes.get(target, "")):
            tiers |= _make_tiers(shlex.split(nested), recipes)
    return tiers


def _nox_tiers(sessions: list[str], markers: dict[str, str], functions: dict[str, str]) -> set[str]:
    """Test tiers the named nox sessions run without narrowing them to named paths."""
    tiers: set[str] = set()
    for session in sessions:
        function = ast.parse(functions[session.split("-", 1)[0]])
        assigned: dict[str, set[str]] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, set()).add(str(node.value.value))
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id.startswith("_run_pytest")
            ):
                continue
            tier, *arguments = call.args[1:]
            if any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and Path(argument.value.split("::", 1)[0]).exists()
                for argument in arguments
            ):
                continue
            if isinstance(tier, ast.Constant):
                tiers |= {str(tier.value)} & markers.keys()
            elif isinstance(tier, ast.Name):
                tiers |= assigned.get(tier.id, set()) & markers.keys()
    return tiers


def _expand(line: str, environment: dict[str, str], job: dict) -> set[str]:
    """The command lines a run-step line becomes across the job's matrix."""
    for variable in re.finditer(r"\$\{?(\w+)\}?", line):
        value = environment.get(variable.group(1))
        if value is None:
            continue
        matrix_key = re.fullmatch(r"\$\{\{ *matrix\.([\w-]+) *\}\}", str(value))
        values = [str(value)]
        if matrix_key is not None:
            matrix = job["strategy"]["matrix"]
            values = [
                str(entry[matrix_key.group(1)])
                for entry in matrix.get("include", [])
                if matrix_key.group(1) in entry
            ] + [str(entry) for entry in matrix.get(matrix_key.group(1), [])]
        return {
            expanded
            for value in values
            for expanded in _expand(
                line[: variable.start()] + value + line[variable.end() :], environment, job
            )
        }
    return {line}


def _job_test_tiers(job: dict) -> set[str]:
    """Test tiers a workflow job runs through ``make`` targets or nox sessions."""
    markers, functions = _noxfile()
    recipes = _makefile_recipes()
    tiers: set[str] = set()
    for step in job.get("steps", []):
        environment = {**job.get("env", {}), **step.get("env", {})}
        for line in step.get("run", "").splitlines():
            for command in _expand(line, environment, job):
                if make := re.match(r"\s*make (.*)", command):
                    tiers |= _make_tiers(shlex.split(make.group(1)), recipes)
                if nox := re.search(r"\bnox -s (.*)", command):
                    sessions = shlex.split(nox.group(1))
                    sessions = sessions[: sessions.index("--")] if "--" in sessions else sessions
                    tiers |= _nox_tiers(sessions, markers, functions)
    return tiers


def test_jobs_running_the_fast_or_full_tier_fetch_full_history() -> None:
    # Integration tests in the fast and full tiers check out recorded commits
    # (tests/test_evidence_identity_parity.py), which a shallow clone lacks.
    markers, _ = _noxfile()
    shallow: list[tuple[str, str]] = []
    for workflow in ("ci", "nightly", "weekly"):
        jobs = yaml.safe_load(Path(f".github/workflows/{workflow}.yml").read_text())["jobs"]
        running = [
            name
            for name, job in jobs.items()
            if any(_selects(markers[tier], "integration") for tier in _job_test_tiers(job))
        ]
        assert running, workflow
        shallow.extend(
            (workflow, name)
            for name in running
            if not any(
                step.get("uses", "").startswith("actions/checkout@")
                and step.get("with", {}).get("fetch-depth") == 0
                for step in jobs[name]["steps"]
            )
        )
    assert shallow == []
