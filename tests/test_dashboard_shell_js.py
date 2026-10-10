"""Every inline script in the shipped dashboard shell must parse.

The dashboard's behavior lives in one inline script; a syntax error there
leaves every rendered dashboard without initialization, tabs or population
switching, yet Python-side payload tests still pass. Uses the same Node gate
as the compatibility JS suite: skipped locally without Node, failed in CI.
"""

import re
import subprocess
from importlib.resources import files

import pytest

from tests.test_compatibility_js import _node_gate, _resolve_node, _running_in_ci

_SCRIPT = re.compile(r"<script(?:\s[^>]*)?>(.*?)</script>", re.DOTALL)


def _inline_scripts(html: str) -> list[str]:
    return [body for body in _SCRIPT.findall(html) if body.strip()]


def _syntax_errors(node_path: str, scripts: list[str], tmp_path) -> list[str]:
    errors = []
    for index, body in enumerate(scripts):
        path = tmp_path / f"script_{index}.js"
        path.write_text(body)
        result = subprocess.run([node_path, "--check", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append(f"script {index}: {result.stderr.strip()}")
    return errors


def _node_or_gate() -> str:
    node_path, version_output = _resolve_node()
    action, reason = _node_gate(node_path, version_output, _running_in_ci())
    if action == "skip":
        pytest.skip(reason)
    if action == "fail":
        pytest.fail(reason)
    assert node_path is not None
    return node_path


def test_dashboard_shell_scripts_parse(tmp_path):
    node_path = _node_or_gate()
    html = files("increment.dashboard").joinpath("_shell.html").read_text()
    scripts = _inline_scripts(html)
    assert scripts
    assert _syntax_errors(node_path, scripts, tmp_path) == []


def test_syntax_gate_rejects_an_unclosed_function(tmp_path):
    node_path = _node_or_gate()
    broken = "<script>(function () {\n  function render() {\n    if (x) {}\n})();</script>"
    assert _syntax_errors(node_path, _inline_scripts(broken), tmp_path)
