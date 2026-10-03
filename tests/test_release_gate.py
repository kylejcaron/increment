from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
SHA = "a" * 40
RUN_URL = "https://github.com/example/repo/actions/runs/123/"


def _step(workflow: str, job: str, name: str) -> str:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return next(step["run"] for step in document["jobs"][job]["steps"] if step.get("name") == name)


def _check(name="ci-ok", *, status="completed", conclusion="success", **fields):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "head_sha": SHA,
        "app": {"slug": "github-actions"},
        "details_url": "https://github.com/example/repo/actions/runs/456/job/1",
        **fields,
    }


def _gate(tmp_path: Path, checks: list[list[dict]], statuses=None):
    checks_file = tmp_path / "checks.json"
    checks_file.write_text(json.dumps([{"check_runs": page} for page in checks]))
    statuses_file = tmp_path / "statuses.json"
    statuses_file.write_text(json.dumps(statuses or [{"state": "pending", "statuses": []}]))
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "path = next(a for a in args if a.startswith('repos/'))\n"
        "key = 'CHECKS_FILE' if '/check-runs' in path else 'STATUSES_FILE'\n"
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
    return subprocess.run(
        ["bash", "-c", _step("bump.yml", "tag", "Require green checks on this commit")],
        env=os.environ
        | {
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_REPOSITORY": "example/repo",
            "SHA": SHA,
            "RUN_URL": RUN_URL,
            "CHECKS_FILE": str(checks_file),
            "STATUSES_FILE": str(statuses_file),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
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
