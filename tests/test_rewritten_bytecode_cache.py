"""Saved rewritten test bytecode is reused in a fresh checkout only for unchanged sources."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.rewritten_bytecode_cache import restore, save

pytestmark = [
    pytest.mark.integration,
    pytest.mark.tooling,
    pytest.mark.xdist_group("rewritten_bytecode_cache"),
]

_KEPT = "def test_kept():\n    assert 1 == 1\n"
_EDITED = "def test_edited():\n    assert 2 == 2\n"
# Same size as _EDITED: only the content tells the two versions apart.
_EDITED_LATER = "def test_edited():\n    assert 2 == 3\n"


@dataclass(frozen=True)
class FreshCheckout:
    root: Path
    cache: Path
    kept_pyc: Path | None
    kept_inode: int | None
    rerun: subprocess.CompletedProcess[str]


def _pytest(root: Path) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_", "PYTEST_XDIST_"))
        and key not in {"PYTEST_PLUGINS", "PYTEST_ADDOPTS", "PYTHONDONTWRITEBYTECODE"}
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:tach", "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.fixture(scope="module")
def fresh_checkout(tmp_path_factory: pytest.TempPathFactory) -> FreshCheckout:
    root = tmp_path_factory.mktemp("checkout")
    (root / "pytest.ini").write_text("[pytest]\n")
    kept = root / "test_kept.py"
    kept.write_text(_KEPT)
    edited = root / "test_edited.py"
    edited.write_text(_EDITED)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    first = _pytest(root)
    assert first.returncode == pytest.ExitCode.OK, first.stdout + first.stderr
    saved_mtime = edited.stat().st_mtime
    cache = tmp_path_factory.mktemp("cache") / "rewritten-bytecode"
    save(root, cache)

    # A later checkout: no bytecode, a new mtime on the unchanged test, and the
    # edited test at the saved mtime and size, so only its content differs.
    shutil.rmtree(root / "__pycache__")
    later = saved_mtime + 3600
    os.utime(kept, (later, later))
    edited.write_text(_EDITED_LATER)
    os.utime(edited, (saved_mtime, saved_mtime))
    restore(root, cache)

    kept_pyc = next((root / "__pycache__").glob("test_kept.*.pyc"), None)
    kept_inode = kept_pyc.stat().st_ino if kept_pyc is not None else None
    return FreshCheckout(root, cache, kept_pyc, kept_inode, _pytest(root))


def test_edited_test_runs_its_current_source_after_restore(fresh_checkout: FreshCheckout):
    rerun = fresh_checkout.rerun
    assert rerun.returncode == pytest.ExitCode.TESTS_FAILED, rerun.stdout + rerun.stderr


def test_unchanged_test_runs_restored_bytecode_in_a_fresh_checkout(fresh_checkout: FreshCheckout):
    assert fresh_checkout.kept_pyc is not None
    # pytest replaces a pyc it rejects, which gives the path a new inode.
    assert fresh_checkout.kept_pyc.stat().st_ino == fresh_checkout.kept_inode


def test_restore_installs_nothing_into_another_checkout_location(
    fresh_checkout: FreshCheckout, tmp_path: Path
):
    moved = tmp_path / "moved"
    shutil.copytree(fresh_checkout.root, moved, ignore=shutil.ignore_patterns("__pycache__"))

    restore(moved, fresh_checkout.cache)

    assert list(moved.rglob("*.pyc")) == []
