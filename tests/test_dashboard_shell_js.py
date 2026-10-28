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


@pytest.mark.integration
def test_dashboard_shell_scripts_parse(tmp_path):
    node_path = _node_or_gate()
    html = files("increment.dashboard").joinpath("_shell.html").read_text()
    scripts = _inline_scripts(html)
    assert scripts
    assert _syntax_errors(node_path, scripts, tmp_path) == []


def test_report_print_rules_raise_table_text_and_wrap_long_labels():
    html = files("increment.dashboard").joinpath("_shell.html").read_text()
    assert ".rpt .rpt-results .gt_row[data-inc-floor]" in html
    assert ".rpt .rpt-results .gt_col_heading[data-inc-floor]" in html
    assert (
        ".rpt .rpt-results .gt_col_heading { white-space: normal !important; overflow-wrap: anywhere !important;"
        in html
    )
    assert ".rpt .rpt-results td { overflow-wrap: anywhere; min-width: 0; }" in html


@pytest.mark.integration
def test_theme_choice_survives_os_changes_and_storage_failures():
    node_path = _node_or_gate()
    html = files("increment.dashboard").joinpath("_shell.html").read_text()
    result = subprocess.run(
        [node_path, "-e", _THEME_BEHAVIOR],
        input=html,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_THEME_BEHAVIOR = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
let html = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => html += chunk);
process.stdin.on('end', () => {
  const source = [...html.matchAll(/<script(?:\s[^>]*)?>(.*?)<\/script>/gs)]
    .map(match => match[1]).find(text => text.includes('Theme: resolved'));
  assert.ok(source, 'theme controller script is present');

  function load(saved = null, systemDark = false, storageThrows = false) {
    const attributes = new Map();
    const handlers = {};
    const buttonHandlers = {};
    const button = { setAttribute: (key, value) => { buttonHandlers[key] = value; },
      addEventListener: (key, callback) => { buttonHandlers[key] = callback; } };
    const mark = { textContent: '' };
    const label = { textContent: '' };
    const queryHandlers = {};
    const store = {
      getItem: () => { if (storageThrows) throw new Error('blocked'); return saved; },
      setItem: (_key, value) => { if (storageThrows) throw new Error('blocked'); saved = value; },
    };
    const query = {
      matches: systemDark,
      addEventListener: (_name, callback) => { queryHandlers.change = callback; },
      fire: matches => { systemDark = matches; queryHandlers.change({ matches }); },
    };
    const document = {
      documentElement: {
        getAttribute: key => attributes.get(key) ?? null,
        setAttribute: (key, value) => attributes.set(key, value),
      },
      getElementById: id => ({ 'theme-toggle': button, 'theme-toggle-mark': mark,
        'theme-toggle-label': label })[id] ?? null,
      addEventListener: (name, callback) => { handlers[name] = callback; },
    };
    const window = { localStorage: store, matchMedia: () => query, frameElement: null };
    vm.runInNewContext(source, { window, document });
    handlers.DOMContentLoaded();
    return { theme: () => attributes.get('data-theme'), buttonHandlers, query };
  }

  const chosen = load(null, false);
  assert.equal(chosen.theme(), 'light');
  chosen.buttonHandlers.click();
  assert.equal(chosen.theme(), 'dark');
  chosen.query.fire(false);
  assert.equal(chosen.theme(), 'dark', 'OS changes do not override an explicit choice');
  assert.equal(load('dark', false).theme(), 'dark', 'saved choice wins after reload');

  const privateMode = load(null, false, true);
  privateMode.buttonHandlers.click();
  assert.equal(privateMode.theme(), 'dark', 'choice still applies when storage is unavailable');
  privateMode.query.fire(false);
  assert.equal(privateMode.theme(), 'dark');
});
"""


@pytest.mark.integration
def test_syntax_gate_rejects_an_unclosed_function(tmp_path):
    node_path = _node_or_gate()
    broken = "<script>(function () {\n  function render() {\n    if (x) {}\n})();</script>"
    assert _syntax_errors(node_path, _inline_scripts(broken), tmp_path)
