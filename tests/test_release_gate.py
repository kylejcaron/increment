from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
SCRIPT = Path(__file__).parents[1] / "scripts" / "release_gate.sh"
SHA = "a" * 40
OTHER_SHA = "b" * 40
RUN_URL = "https://github.com/example/repo/actions/runs/123/"
STABLE = "1.0.0"
PRERELEASE = "1.0.0rc1"
CI_SUITE, WEEKLY_SUITE, OTHER_SUITE = 1, 2, 3
CORE_INSTALL = "core-install (3.15)"
FLOOR_JOB = "full suite at dependency floors"


def _jobs(workflow: str) -> dict:
    return yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"]


def _floor_jobs() -> list[str]:
    """Display names of the weekly floor jobs, expanded over the job's matrix."""
    job = _jobs("weekly.yml")["test-all-floor"]
    axes = {
        key: value for key, value in job["strategy"]["matrix"].items() if isinstance(value, list)
    }
    names = []
    for combination in itertools.product(*axes.values()):
        name = job["name"]
        for key, value in zip(axes, combination, strict=True):
            name = name.replace(f"${{{{ matrix.{key} }}}}", str(value))
        names.append(name)
    return names


FLOOR_JOBS = _floor_jobs()
FLOOR_PYTHONS = _jobs("weekly.yml")["test-all-floor"]["strategy"]["matrix"]["python-version"]


def _check(name="ci-ok", *, status="completed", conclusion="success", **fields):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "head_sha": SHA,
        "app": {"slug": "github-actions"},
        "details_url": "https://github.com/example/repo/actions/runs/456/job/1",
        "check_suite": {"id": CI_SUITE},
        **fields,
    }


def _floors(**last):
    """Weekly floor jobs for every floor interpreter; ``last`` overrides the final one."""
    *first, final = FLOOR_JOBS
    weekly = {"check_suite": {"id": WEEKLY_SUITE}}
    return [*(_check(job, **weekly) for job in first), _check(final, **weekly | last)]


def _split_floors(parts=3):
    """Floor evidence from a job split into ``parts`` parts on every floor interpreter."""
    return [
        _check(f"{FLOOR_JOB} ({python}, part {group}/{parts})", check_suite={"id": WEEKLY_SUITE})
        for python in FLOOR_PYTHONS
        for group in range(1, parts + 1)
    ]


def _nightly(*, head_sha=SHA, status="completed", conclusion="success", event="workflow_dispatch"):
    return {"head_sha": head_sha, "status": status, "conclusion": conclusion, "event": event}


def _gate(
    tmp_path: Path,
    checks: list[list[dict]],
    statuses=None,
    nightlies=None,
    *,
    version=PRERELEASE,
    force=False,
    run_url: str | None = RUN_URL,
):
    checks_file = tmp_path / "checks.json"
    checks_file.write_text(json.dumps([{"check_runs": page} for page in checks]))
    statuses_file = tmp_path / "statuses.json"
    statuses_file.write_text(json.dumps(statuses or [{"state": "pending", "statuses": []}]))
    nightlies_file = tmp_path / "nightlies.json"
    nightlies_file.write_text(json.dumps(nightlies or [{"workflow_runs": [_nightly()]}]))
    weeklies_file = tmp_path / "weeklies.json"
    weeklies_file.write_text(
        json.dumps([{"workflow_runs": [{"head_sha": SHA, "check_suite_id": WEEKLY_SUITE}]}])
    )
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "path = next(a for a in args if a.startswith('repos/'))\n"
        "key = ('CHECKS_FILE' if '/check-runs' in path else\n"
        "       'NIGHTLIES_FILE' if '/workflows/nightly.yml/runs' in path else\n"
        "       'WEEKLIES_FILE' if '/workflows/weekly.yml/runs' in path else 'STATUSES_FILE')\n"
        "pages = json.loads(Path(os.environ[key]).read_text())\n"
        "data = pages if '--slurp' in args else pages[0]\n"
        "if '--jq' in args:\n"
        "    result = subprocess.run(['jq', args[args.index('--jq') + 1]],\n"
        "                            input=json.dumps(data), text=True, check=True, capture_output=True)\n"
        "    print(result.stdout, end='')\n"
        "else:\n"
        "    print(json.dumps(data))\n"
    )
    gh.chmod(0o755)
    env = os.environ | {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "example/repo",
        "SHA": SHA,
        "VERSION": version,
        "FORCE": str(force).lower(),
        "CHECKS_FILE": str(checks_file),
        "STATUSES_FILE": str(statuses_file),
        "NIGHTLIES_FILE": str(nightlies_file),
        "WEEKLIES_FILE": str(weeklies_file),
    }
    env.pop("RUN_URL", None)
    if run_url is not None:
        env["RUN_URL"] = run_url
    return subprocess.run(
        [str(SCRIPT)], env=env, capture_output=True, text=True, check=False, timeout=10
    )


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_pending_ci_refuses_release_even_when_another_check_passed(tmp_path, status):
    result = _gate(
        tmp_path,
        [[_check("lint"), _check("ci-ok", status=status, conclusion=None)]],
    )
    assert result.returncode != 0


def test_a_green_unrelated_check_cannot_replace_required_ci(tmp_path):
    assert _gate(tmp_path, [[_check("lint")]]).returncode != 0


@pytest.mark.parametrize(
    "fields",
    [
        {"head_sha": "b" * 40},
        {"app": {"slug": "other-app"}},
        {"conclusion": "skipped"},
        {"conclusion": "neutral"},
    ],
    ids=["different-commit", "different-app", "skipped-ci", "neutral-ci"],
)
def test_required_ci_must_succeed_on_this_commit(tmp_path, fields):
    assert _gate(tmp_path, [[_check(**fields)]]).returncode != 0


def test_successful_ci_and_optional_skips_allow_release(tmp_path):
    result = _gate(tmp_path, [[_check(), _check("optional", conclusion="skipped")]])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "nightlies,version",
    [
        ([{"workflow_runs": []}], PRERELEASE),
        ([{"workflow_runs": [_nightly(head_sha=OTHER_SHA)]}], PRERELEASE),
        ([{"workflow_runs": [_nightly(conclusion="failure")]}], PRERELEASE),
        pytest.param(
            [{"workflow_runs": [_nightly(event="pull_request")]}],
            STABLE,
            id="stable-needs-scientific-tiers-not-benchmarks-only",
        ),
    ],
)
def test_release_requires_a_successful_nightly_on_this_commit(tmp_path, nightlies, version):
    checks = [[_check(), *_floors(), _check(CORE_INSTALL)]]
    result = _gate(tmp_path, checks, nightlies=nightlies, version=version)
    assert result.returncode != 0
    assert f"run `gh workflow run nightly.yml --ref main` while main is at {SHA}" in result.stdout


def test_bump_does_not_deadlock_on_its_own_running_check(tmp_path):
    own = _check("tag", status="in_progress", conclusion=None, details_url=RUN_URL + "job/1")
    result = _gate(tmp_path, [[own, _check()]])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "fields",
    [
        {"details_url": RUN_URL.replace("123/", "1234/") + "job/1"},
        {"app": {"slug": "other-app"}, "details_url": RUN_URL + "job/1"},
    ],
)
def test_self_run_exemption_does_not_hide_other_pending_checks(tmp_path, fields):
    pending = _check("tag", status="in_progress", conclusion=None, **fields)
    assert _gate(tmp_path, [[_check(), pending]]).returncode != 0


@pytest.mark.parametrize("status,conclusion", [("queued", None), ("completed", "failure")])
def test_later_check_pages_cannot_hide_pending_or_failed_checks(tmp_path, status, conclusion):
    later = _check("slow", status=status, conclusion=conclusion)
    assert _gate(tmp_path, [[_check()], [later]]).returncode != 0


def test_required_ci_can_be_on_a_later_page(tmp_path):
    result = _gate(tmp_path, [[_check("lint")], [_check()]])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("state", ["pending", "failure", "error"])
def test_legacy_statuses_on_later_pages_also_gate_release(tmp_path, state):
    reports = [
        {"state": "success", "statuses": [{"context": "first", "state": "success"}]},
        {"state": state, "statuses": [{"context": "external", "state": state}]},
    ]
    assert _gate(tmp_path, [[_check()]], reports).returncode != 0


def test_successful_legacy_statuses_and_ci_allow_release(tmp_path):
    statuses = [{"state": "success", "statuses": [{"context": "external", "state": "success"}]}]
    result = _gate(tmp_path, [[_check()]], statuses)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_status_without_required_ci_cannot_release(tmp_path):
    statuses = [{"state": "success", "statuses": [{"context": "external", "state": "success"}]}]
    assert _gate(tmp_path, [[]], statuses).returncode != 0


def _gate_step(workflow: str, job: str) -> dict:
    return next(
        step
        for step in _jobs(workflow)[job]["steps"]
        if step.get("run", "").strip() == "scripts/release_gate.sh"
    )


def test_force_is_refused_for_a_stable_version(tmp_path):
    bump = _gate_step("bump.yml", "tag")
    assert bump["name"] == "Refuse force for a stable version"
    assert bump["if"] == "${{ inputs.force }}"
    assert bump["env"]["FORCE"] == "${{ inputs.force }}"
    assert bump["env"]["VERSION"] == "${{ steps.version.outputs.version }}"
    checks = [[_check(), *_floors(), _check(CORE_INSTALL)]]
    assert _gate(tmp_path, checks, version=STABLE, force=True).returncode != 0


def test_force_is_allowed_for_a_prerelease(tmp_path):
    result = _gate(
        tmp_path, [[]], nightlies=[{"workflow_runs": []}], version=PRERELEASE, force=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_gates_stable_tags_on_the_candidate():
    gate = _gate_step("release.yml", "build")
    assert gate["if"] == "${{ steps.check.outputs.prerelease == 'false' }}"
    assert gate["env"]["SHA"] == "${{ steps.check.outputs.candidate_sha }}"
    assert "build" in _jobs("release.yml")["publish"]["needs"]


@pytest.mark.parametrize(
    "floors",
    [
        pytest.param([], id="absent"),
        pytest.param(_floors(conclusion="skipped"), id="skipped"),
        pytest.param(_floors(status="in_progress", conclusion=None), id="pending"),
        pytest.param(_floors(conclusion="failure"), id="failed"),
        pytest.param(_floors(head_sha=OTHER_SHA), id="wrong-sha"),
        pytest.param(
            [check for check in _floors() if f"({FLOOR_PYTHONS[-1]}" not in check["name"]],
            id="interpreter-missing",
        ),
        pytest.param(_split_floors()[:-1], id="part-missing"),
        pytest.param(_floors(check_suite={"id": OTHER_SUITE}), id="not-from-weekly"),
    ],
)
def test_stable_release_requires_floor_evidence_for_every_floor_interpreter(tmp_path, floors):
    result = _gate(tmp_path, [[_check(), _check(CORE_INSTALL), *floors]], version=STABLE)
    assert result.returncode != 0
    assert f"run `gh workflow run weekly.yml --ref main` while main is at {SHA}" in result.stdout


@pytest.mark.parametrize(
    "core_install",
    [
        pytest.param([], id="absent"),
        pytest.param([_check(CORE_INSTALL, conclusion="skipped")], id="skipped"),
        pytest.param(
            [_check(CORE_INSTALL, check_suite={"id": OTHER_SUITE})], id="outside-the-ci-ok-run"
        ),
    ],
)
def test_stable_release_requires_core_install_evidence_for_3_15(tmp_path, core_install):
    result = _gate(tmp_path, [[_check(), *_floors(), *core_install]], version=STABLE)
    assert result.returncode != 0


@pytest.mark.parametrize(
    "floors",
    [pytest.param(_floors(), id="weekly-jobs"), pytest.param(_split_floors(), id="split-jobs")],
)
def test_supported_combination_permits_release(tmp_path, floors):
    result = _gate(tmp_path, [[_check(), *floors, _check(CORE_INSTALL)]], version=STABLE)
    assert result.returncode == 0, result.stdout + result.stderr


def test_prerelease_does_not_require_floor_evidence(tmp_path):
    result = _gate(tmp_path, [[_check()]], version=PRERELEASE)
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_runs_outside_a_workflow_run(tmp_path):
    result = _gate(tmp_path, [[_check()]], run_url=None)
    assert result.returncode == 0, result.stdout + result.stderr


pytestmark = pytest.mark.integration
