from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import install_dev_hooks


def _run(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


@pytest.fixture
def stable_tool(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, str]]:
    installed = shutil.which("prek")
    if installed is None:
        pytest.skip("requires the already-installed prek executable; network is not used")
    source = Path(installed).resolve()
    version_output = _run(str(source), "--version").stdout.strip()
    match = re.search(r"(?:^|\s)(\d+\.\d+\.\d+)(?:\s|$)", version_output)
    if match is None:
        pytest.fail(f"could not read installed prek version: {version_output!r}")

    # Evidence-backed runs place tmp_path inside the project; user tools cannot live there.
    with tempfile.TemporaryDirectory(prefix="increment-hook-tool-") as directory:
        tool_bin = Path(directory) / "bin"
        tool_bin.mkdir()
        executable = tool_bin / "prek"
        shutil.copy2(source, executable)
        monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tool_bin))
        tool_alias = Path(directory) / "tool-bin-alias"
        tool_alias.symlink_to(tool_bin, target_is_directory=True)
        yield tool_alias / executable.name, match.group(1)


def _repository(path: Path, *, prek_version: str) -> Path:
    path.mkdir()
    _run("git", "init", "--quiet", "--initial-branch=main", str(path))
    (path / "uv.lock").write_text(
        f'[[package]]\nname = "prek"\nversion = "{prek_version}"\n', encoding="utf-8"
    )
    (path / ".pre-commit-config.yaml").write_text(
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: local-boundary-check\n"
        "        name: local boundary check\n"
        "        entry: python .hook_check.py\n"
        "        language: system\n"
        "        types: [python]\n",
        encoding="utf-8",
    )
    (path / ".hook_check.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path('.hook-ran').write_text('yes', encoding='utf-8')\n"
        "raise SystemExit(1 if any(Path(name).read_text(encoding='utf-8') == 'reject fixture\\n' "
        "for name in sys.argv[1:]) else 0)\n",
        encoding="utf-8",
    )
    _run("git", "-C", str(path), "config", "user.name", "Hook Smoke")
    _run("git", "-C", str(path), "config", "user.email", "hook@example.invalid")
    _run("git", "-C", str(path), "add", "uv.lock", ".pre-commit-config.yaml", ".hook_check.py")
    _run("git", "-C", str(path), "commit", "--quiet", "-m", "initial")
    return path


def _configure_local_remote(repo: Path, tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    _run("git", "init", "--quiet", "--bare", str(remote))
    _run("git", "-C", str(repo), "remote", "add", "origin", str(remote))
    _run("git", "-C", str(repo), "push", "--quiet", "-u", "origin", "main")
    return remote


def test_locked_prek_version_requires_one_package(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "prek"\nversion = "0.4.11"\n', encoding="utf-8")
    assert install_dev_hooks.locked_prek_version(lock) == "0.4.11"

    lock.write_text(
        '[[package]]\nname = "prek"\nversion = "0.4.11"\n'
        '[[package]]\nname = "prek"\nversion = "0.4.12"\n',
        encoding="utf-8",
    )
    with pytest.raises(install_dev_hooks.HookInstallError):
        install_dev_hooks.locked_prek_version(lock)


def test_installed_hooks_run_in_linked_worktrees_without_project_environment(
    tmp_path: Path, stable_tool: tuple[Path, str]
) -> None:
    executable, version = stable_tool
    repo = _repository(tmp_path / "repo", prek_version=version)
    remote = _configure_local_remote(repo, tmp_path)
    second = tmp_path / "second-worktree"
    _run("git", "-C", str(repo), "worktree", "add", "--quiet", "-b", "second", str(second))

    hooks = Path(_run("git", "-C", str(repo), "rev-parse", "--git-path", "hooks").stdout.strip())
    if not hooks.is_absolute():
        hooks = repo / hooks
    hooks = hooks.resolve()
    legacy_log = tmp_path / "legacy-pre-push.log"
    old_pre_push = hooks / "pre-push"
    old_pre_push.write_text(
        "#!/bin/sh\n"
        f"printf 'argc=%s\\n' \"$#\" >> '{legacy_log}'\n"
        f"for arg in \"$@\"; do printf 'arg=%s\\n' \"$arg\" >> '{legacy_log}'; done\n"
        f"cat >> '{legacy_log}'\n",
        encoding="utf-8",
    )
    old_pre_push.chmod(0o755)
    prior_legacy = old_pre_push.read_bytes()

    install_dev_hooks.install_hooks(repo, prek=executable)
    pre_commit = hooks / "pre-commit"
    assert pre_commit.is_file()
    assert (hooks / "pre-push.legacy").read_bytes() == prior_legacy

    (repo / "bad.py").write_text("reject fixture\n", encoding="utf-8")
    _run("git", "-C", str(repo), "add", "bad.py")
    rejected = _run("git", "-C", str(repo), "commit", "-m", "bad", check=False)
    assert rejected.returncode != 0
    assert (repo / ".hook-ran").is_file()

    (second / "good.py").write_text("GOOD = True\n", encoding="utf-8")
    _run("git", "-C", str(second), "add", "good.py")
    _run("git", "-C", str(second), "commit", "--quiet", "-m", "good")
    assert (second / ".hook-ran").is_file()
    _run("git", "-C", str(second), "push", "--quiet", str(remote), "HEAD:refs/heads/second")
    legacy_output = legacy_log.read_text(encoding="utf-8")
    assert "argc=2" in legacy_output
    assert legacy_output.splitlines().count(f"arg={remote}") == 2
    pushed_commit = _run("git", "-C", str(second), "rev-parse", "HEAD").stdout.strip()
    assert (f"HEAD {pushed_commit} refs/heads/second {'0' * 40}") in legacy_output

    (repo / ".venv").mkdir()
    (repo / ".venv" / "unavailable").write_text("", encoding="utf-8")
    (repo / "bad.py").write_text("reject fixture\n", encoding="utf-8")
    _run("git", "-C", str(repo), "add", "bad.py")
    unavailable = repo / ".venv"
    parked = repo / ".venv-unavailable"
    unavailable.rename(parked)
    try:
        rerun = _run("git", "-C", str(repo), "commit", "-m", "still bad", check=False)
        assert rerun.returncode != 0
    finally:
        parked.rename(unavailable)


def test_install_refuses_conflicting_existing_legacy_ownership(
    tmp_path: Path, stable_tool: tuple[Path, str]
) -> None:
    executable, version = stable_tool
    repo = _repository(tmp_path / "repo", prek_version=version)
    hooks = Path(_run("git", "-C", str(repo), "rev-parse", "--git-path", "hooks").stdout.strip())
    if not hooks.is_absolute():
        hooks = repo / hooks
    hooks = hooks.resolve()
    custom_hook = hooks / "pre-commit"
    existing_legacy = hooks / "pre-commit.legacy"
    custom_hook.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")
    existing_legacy.write_text("#!/bin/sh\necho prior legacy\n", encoding="utf-8")
    original_custom = custom_hook.read_bytes()
    original_legacy = existing_legacy.read_bytes()

    with pytest.raises(install_dev_hooks.HookInstallError):
        install_dev_hooks.install_hooks(repo, prek=executable)

    assert custom_hook.read_bytes() == original_custom
    assert existing_legacy.read_bytes() == original_legacy


@pytest.mark.parametrize("dangling_entry", ["pre-commit", "pre-commit.legacy"])
def test_install_preserves_dangling_hook_symlinks(
    tmp_path: Path, stable_tool: tuple[Path, str], dangling_entry: str
) -> None:
    executable, version = stable_tool
    repo = _repository(tmp_path / "repo", prek_version=version)
    hooks = Path(_run("git", "-C", str(repo), "rev-parse", "--git-path", "hooks").stdout.strip())
    if not hooks.is_absolute():
        hooks = repo / hooks
    hooks = hooks.resolve()
    hook = hooks / "pre-commit"
    legacy = hooks / "pre-commit.legacy"
    custom_content = b"#!/bin/sh\necho custom\n"
    if dangling_entry == "pre-commit":
        hook.symlink_to("missing-hook-target")
        original_link = hook.readlink()
    else:
        hook.write_bytes(custom_content)
        legacy.symlink_to("missing-legacy-target")
        original_link = legacy.readlink()

    with pytest.raises(install_dev_hooks.HookInstallError):
        install_dev_hooks.install_hooks(repo, prek=executable)

    protected = hook if dangling_entry == "pre-commit" else legacy
    assert protected.is_symlink()
    assert protected.readlink() == original_link
    if dangling_entry == "pre-commit.legacy":
        assert hook.read_bytes() == custom_content


def test_install_rejects_project_environment_executable(
    tmp_path: Path, stable_tool: tuple[Path, str]
) -> None:
    executable, version = stable_tool
    repo = _repository(tmp_path / "repo", prek_version=version)
    local_tool = repo / ".venv" / "bin" / "prek"
    local_tool.parent.mkdir(parents=True)
    shutil.copy2(executable, local_tool)

    with pytest.raises(install_dev_hooks.HookInstallError):
        install_dev_hooks.install_hooks(repo, prek=local_tool)
