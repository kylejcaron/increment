from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "next_version.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "--initial-branch=main")
    for key, value in {
        "user.name": "Version test",
        "user.email": "version@example.invalid",
        "commit.gpgsign": "false",
        "tag.gpgsign": "false",
        "core.hooksPath": str(path / "disabled-hooks"),
    }.items():
        _git(path, "config", key, value)
    _git(path, "commit", "--allow-empty", "-qm", "Initial")
    return path


def _resolve(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize(
    "baseline,bump,stage,expected",
    [
        ("0.1.0a2", "auto", "keep-current", "0.1.0a3"),
        ("0.1.0a2", "auto", "alpha", "0.1.0a3"),
        ("0.1.0a2", "auto", "beta", "0.1.0b1"),
        ("0.1.0a2", "auto", "rc", "0.1.0rc1"),
        ("0.1.0a2", "auto", "stable", "0.1.0"),
        ("0.1.0a2", "patch", "keep-current", "0.1.1a1"),
        ("0.1.0a2", "minor", "keep-current", "0.2.0a1"),
        ("0.1.0a2", "major", "keep-current", "1.0.0a1"),
        ("0.1.0a2", "minor", "beta", "0.2.0b1"),
        ("0.1.0b9", "auto", "beta", "0.1.0b10"),
        ("0.1.0b9", "auto", "rc", "0.1.0rc1"),
        ("0.1.0rc9", "auto", "keep-current", "0.1.0rc10"),
        ("0.1.0rc9", "auto", "stable", "0.1.0"),
        ("0.1.0b9", "patch", "alpha", "0.1.1a1"),
        ("0.1.0", "auto", "keep-current", "0.1.1"),
        ("0.1.0", "auto", "stable", "0.1.1"),
        ("0.1.0", "auto", "alpha", "0.1.1a1"),
        ("0.1.0", "patch", "keep-current", "0.1.1"),
        ("0.1.0", "minor", "beta", "0.2.0b1"),
        ("0.1.0", "major", "rc", "1.0.0rc1"),
        ("3.8.7b4", "minor", "keep-current", "3.9.0b1"),
        ("3.8.7rc4", "major", "stable", "4.0.0"),
    ],
)
def test_selected_bump_and_stage_produce_the_next_version(repo, baseline, bump, stage, expected):
    _git(repo, "tag", f"v{baseline}")
    result = _resolve(repo, "--bump", bump, "--stage", stage)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"baseline": baseline, "version": expected}


@pytest.mark.parametrize(
    "baseline,stage", [("0.1.0b1", "alpha"), ("0.1.0rc1", "alpha"), ("0.1.0rc1", "beta")]
)
def test_same_base_stage_regressions_are_refused(repo, baseline, stage):
    _git(repo, "tag", f"v{baseline}")
    assert _resolve(repo, "--stage", stage).returncode != 0


def test_baseline_uses_version_order_not_tag_name_or_creation_order(repo):
    for tag in ["v0.1.0a10", "v0.1.0a2", "unrelated", "vgarbage", "v9.0.0+local"]:
        _git(repo, "tag", tag)
    result = _resolve(repo)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"baseline": "0.1.0a10", "version": "0.1.0a11"}


def test_backport_baseline_excludes_unmerged_newer_releases(repo):
    _git(repo, "tag", "v0.1.0")
    older = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--allow-empty", "-qm", "Future main")
    _git(repo, "tag", "v0.2.0")
    result = _resolve(repo, "--ref", older)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"baseline": "0.1.0", "version": "0.1.1"}


def test_an_unmerged_tag_still_reserves_the_calculated_version(repo):
    _git(repo, "tag", "v0.1.0")
    older = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--allow-empty", "-qm", "Another branch")
    _git(repo, "tag", "v0.1.1")
    assert _resolve(repo, "--ref", older).returncode != 0


def test_override_can_publish_a_backport_below_the_latest_version(repo):
    _git(repo, "tag", "v2.0.0")
    result = _resolve(repo, "--version", "1.4.2")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["version"] == "1.4.2"


@pytest.mark.parametrize("selection", [("--bump", "minor"), ("--stage", "beta")])
def test_override_cannot_silently_discard_an_explicit_selection(repo, selection):
    assert _resolve(repo, "--version", "0.1.0a3", *selection).returncode != 0


@pytest.mark.parametrize("version", ["v0.1.0", "0.1.0+local", "garbage", "0.1.0alpha3", "0.1.0\n"])
def test_override_refuses_noncanonical_or_nonpublic_versions(repo, version):
    assert _resolve(repo, "--version", version).returncode != 0


@pytest.mark.parametrize("tag,override", [("v0.1.0a3", "0.1.0a3"), ("v0.1", "0.1.0")])
def test_existing_versions_cannot_be_retagged_or_reencoded(repo, tag, override):
    _git(repo, "tag", tag)
    assert _resolve(repo, "--version", override).returncode != 0


@pytest.mark.parametrize("baseline", ["1!0.1.0", "0.1.0.post1", "0.1.0.dev1", "0.1", "0.1.0.1"])
def test_unsupported_auto_baselines_require_the_explicit_override(repo, baseline):
    _git(repo, "tag", f"v{baseline}")
    assert _resolve(repo).returncode != 0
    result = _resolve(repo, "--version", "2.0.0")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["version"] == "2.0.0"


def test_first_release_requires_an_explicit_version(repo):
    assert _resolve(repo).returncode != 0
    result = _resolve(repo, "--version", "0.1.0a1")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"baseline": None, "version": "0.1.0a1"}


def test_invalid_commit_ref_is_refused_before_any_version_is_emitted(repo):
    _git(repo, "tag", "v0.1.0a2")
    result = _resolve(repo, "--ref", "missing-ref")
    assert result.returncode != 0
    assert result.stdout == ""
