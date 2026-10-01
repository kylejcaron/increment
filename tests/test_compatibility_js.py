"""Runs the Node-native JS behavioral tests for
``docs/javascripts/compatibility.js`` (``tests/js/compatibility_interactions.test.mjs``)
as part of the Python test suite, so ``pytest``/``make check`` fail on a
regression instead of it surfacing only through manual browser checks.
This project has no npm/jsdom dependency; the JS tests use only Node's
built-in ``node:test`` runner and a hand-rolled DOM stub
(``tests/js/dom_stub.mjs``) covering compatibility.js's small, closed
browser-API surface.

Requires Node >= ``MIN_NODE_MAJOR`` (``node:test`` as used here needs a
current runtime). Not every contributor has Node installed locally, and
this project has no other use for it, so a missing or too-old Node
degrades to a ``pytest.skip`` outside CI. Inside CI (``CI`` env var, the
convention GitHub Actions and most other CI systems set), the same
condition is instead a hard ``pytest.fail`` -- this test is the only
thing that exercises compatibility.js at all, and a gate that silently
skips itself in the one environment nobody re-runs locally is not a gate.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_TEST = ROOT / "tests/js/compatibility_interactions.test.mjs"
MIN_NODE_MAJOR = 20


def _parse_node_major(version_output: str) -> int | None:
    """Parse ``node --version``'s ``vX.Y.Z`` output into its major version.
    ``None`` if the output does not look like a Node version string."""
    match = re.match(r"v(\d+)\.", version_output.strip())
    return int(match.group(1)) if match else None


def _node_version_output(node_path: str) -> str | None:
    result = subprocess.run([node_path, "--version"], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def _resolve_node() -> tuple[str | None, str | None]:
    """Resolve the Node executable and, only if one was found, its
    ``--version`` output. ``shutil.which`` runs first and unconditionally
    gates the version query -- ``subprocess.run(["node", ...])`` on a PATH
    with no ``node`` executable raises ``FileNotFoundError`` rather than
    returning a nonzero exit code, so querying the version without first
    confirming the executable exists is one missing-Node environment away
    from crashing this test instead of gating it."""
    node_path = shutil.which("node")
    if node_path is None:
        return None, None
    return node_path, _node_version_output(node_path)


def _running_in_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


def _node_gate(
    node_path: str | None, version_output: str | None, running_in_ci: bool
) -> tuple[str, str]:
    """Decide whether to run, skip, or fail the JS suite given Node's
    availability/version. Returns ``(action, reason)``, ``action`` one of
    ``"run"``, ``"skip"``, ``"fail"``. Locally, a missing or too-old Node
    (or unparseable ``--version`` output) degrades to a skip; in CI the
    identical condition fails instead, so the gate stays load-bearing
    where it matters."""
    if node_path is None:
        reason = "node executable not found on PATH"
        return ("fail", reason) if running_in_ci else ("skip", reason)
    if version_output is None:
        reason = "could not read `node --version` output"
        return ("fail", reason) if running_in_ci else ("skip", reason)
    major = _parse_node_major(version_output)
    if major is None:
        reason = f"could not parse a Node major version from {version_output!r}"
        return ("fail", reason) if running_in_ci else ("skip", reason)
    if major < MIN_NODE_MAJOR:
        reason = (
            f"node {version_output.strip()} is older than the required "
            f"v{MIN_NODE_MAJOR} (node:test needs it)"
        )
        return ("fail", reason) if running_in_ci else ("skip", reason)
    return ("run", "")


def test_compatibility_js_filter_behavior():
    node_path, version_output = _resolve_node()
    action, reason = _node_gate(node_path, version_output, _running_in_ci())
    if action == "skip":
        pytest.skip(reason)
    if action == "fail":
        pytest.fail(reason)
    assert node_path is not None, "gate returned 'run' without a resolved node path"

    result = subprocess.run(
        [node_path, "--test", str(JS_TEST)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"node --test {JS_TEST} failed:\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_parse_node_major_extracts_the_major_version():
    assert _parse_node_major("v25.6.1\n") == 25
    assert _parse_node_major("v20.0.0") == 20


def test_parse_node_major_returns_none_for_unparseable_output():
    assert _parse_node_major("") is None
    assert _parse_node_major("not a version string") is None


def test_node_gate_runs_when_node_is_recent_enough():
    assert _node_gate("/usr/bin/node", "v20.11.0\n", running_in_ci=False) == ("run", "")
    assert _node_gate("/usr/bin/node", "v25.6.1\n", running_in_ci=True) == ("run", "")


def test_node_gate_runs_at_exactly_the_minimum_major():
    action, _ = _node_gate("/usr/bin/node", f"v{MIN_NODE_MAJOR}.0.0\n", running_in_ci=False)
    assert action == "run"


def test_node_gate_skips_locally_when_node_is_missing():
    action, reason = _node_gate(None, None, running_in_ci=False)
    assert action == "skip"
    assert "not found" in reason


def test_node_gate_fails_in_ci_when_node_is_missing():
    action, reason = _node_gate(None, None, running_in_ci=True)
    assert action == "fail"
    assert "not found" in reason


def test_node_gate_skips_locally_when_node_is_too_old():
    action, reason = _node_gate("/usr/bin/node", "v18.19.0\n", running_in_ci=False)
    assert action == "skip"
    assert str(MIN_NODE_MAJOR) in reason


def test_node_gate_fails_in_ci_when_node_is_too_old():
    action, reason = _node_gate("/usr/bin/node", "v18.19.0\n", running_in_ci=True)
    assert action == "fail"
    assert str(MIN_NODE_MAJOR) in reason


def test_node_gate_skips_locally_when_the_version_output_is_unparseable():
    action, reason = _node_gate("/usr/bin/node", "garbage", running_in_ci=False)
    assert action == "skip"
    assert "garbage" in reason


def test_node_gate_fails_in_ci_when_the_version_output_is_unparseable():
    action, reason = _node_gate("/usr/bin/node", "garbage", running_in_ci=True)
    assert action == "fail"


def test_running_in_ci_reads_the_standard_ci_env_var(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert _running_in_ci() is True
    monkeypatch.setenv("CI", "")
    assert _running_in_ci() is False
    monkeypatch.delenv("CI", raising=False)
    assert _running_in_ci() is False


def test_resolve_node_never_invokes_subprocess_when_the_executable_is_missing(monkeypatch):
    """Resolving Node must check ``shutil.which`` first and
    never shell out for ``--version`` when no executable was found --
    ``subprocess.run(["node", ...])`` on a PATH without a ``node``
    executable raises ``FileNotFoundError`` rather than returning a
    nonzero exit code, so the old unconditional call was one path away
    from crashing this test instead of gating it."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def _sentinel(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when node is not on PATH")

    monkeypatch.setattr(subprocess, "run", _sentinel)
    assert _resolve_node() == (None, None)


def test_resolve_node_queries_version_with_the_resolved_path_when_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/fake/node")
    calls = []

    def _fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="v25.6.1\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    node_path, version_output = _resolve_node()
    assert node_path == "/opt/fake/node"
    assert version_output == "v25.6.1\n"
    # One version probe, made with the path shutil.which resolved rather than
    # whatever "node" happens to mean on PATH.
    (argv,) = calls
    assert argv[0] == "/opt/fake/node"
    assert "--version" in argv


def test_resolve_node_reports_no_version_output_when_the_version_call_fails(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/fake/node")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, stdout="", stderr="boom"),
    )
    assert _resolve_node() == ("/opt/fake/node", None)


def test_gate_result_from_a_missing_executable_is_skip_locally_and_fail_in_ci(monkeypatch):
    """End-to-end: ``_resolve_node``'s output, fed straight into
    ``_node_gate``, must skip locally and fail in CI when Node cannot be
    resolved at all."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    node_path, version_output = _resolve_node()
    assert _node_gate(node_path, version_output, running_in_ci=False)[0] == "skip"
    assert _node_gate(node_path, version_output, running_in_ci=True)[0] == "fail"
