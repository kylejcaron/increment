"""Behavioral coverage for the comment/docstring prose audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_verbosity import audit

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_verbosity.py"


def write_source(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_four_standalone_comment_lines_are_allowed(tmp_path: Path) -> None:
    result = audit(
        write_source(
            tmp_path,
            "# one\n# two\n# three\n# four\nvalue = 1\n",
        )
    )
    assert result["violations"] == []


def test_five_standalone_comment_lines_are_reported(tmp_path: Path) -> None:
    result = audit(write_source(tmp_path, "# one\n# two\n# three\n# four\n# five\n"))
    assert any(item["kind"] == "comment-block" for item in result["violations"])


def test_allow_long_requires_reason_and_is_block_scoped(tmp_path: Path) -> None:
    allowed = audit(
        write_source(
            tmp_path,
            "# prose: allow-long explains the compatibility boundary\n"
            "# one\n# two\n# three\n# four\n# five\n"
            "value = 1\n# six\n# seven\n# eight\n# nine\n# ten\n",
        )
    )
    assert [(item["kind"], item["line"]) for item in allowed["violations"]] == [
        ("comment-block", 8)
    ]

    blank_reason = audit(
        write_source(
            tmp_path,
            "# prose: allow-long   \n# one\n# two\n# three\n# four\n# five\n",
        )
    )
    assert any(item["kind"] == "invalid-exception" for item in blank_reason["violations"])


def test_terminology_checks_comments_and_docstrings_not_strings(tmp_path: Path) -> None:
    result = audit(
        write_source(
            tmp_path,
            'literal = "roborev and Statsig are test inputs"\n'
            "# roborev approved this\n"
            'async def f():\n    """Follow superpowers."""\n'
            "    return literal\n",
        )
    )
    assert [(item["kind"], item["line"]) for item in result["violations"]] == [
        ("terminology", 2),
        ("terminology", 4),
    ]


def test_scientific_amplitude_is_not_competitor_terminology(tmp_path: Path) -> None:
    result = audit(write_source(tmp_path, "# amplitude is measured in volts\n"))
    assert result["violations"] == []


def test_inline_tool_reference_is_caught(tmp_path: Path) -> None:
    result = audit(write_source(tmp_path, "value = 1  # per roborev review\n"))
    assert any(item["kind"] == "terminology" and item["line"] == 1 for item in result["violations"])


def test_parse_and_encoding_errors_are_reported(tmp_path: Path) -> None:
    parse_result = audit(write_source(tmp_path, "def broken(:\n"))
    assert parse_result["errors"] and parse_result["violations"]

    encoded = tmp_path / "encoded.py"
    encoded.write_bytes(b"# coding: ascii\n# \xff\n")
    encoding_result = audit(encoded)
    assert encoding_result["errors"] and encoding_result["violations"]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_exit_and_diagnostics(tmp_path: Path) -> None:
    bad = write_source(tmp_path, "# one\n# two\n# three\n# four\n# five\n")
    result = run_cli(str(bad))
    assert result.returncode != 0
    assert f"{bad}:1" in result.stdout


def test_cli_no_target_scans_shipped_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "increment").mkdir()
    (tmp_path / "increment" / "sample.py").write_text("# one\n# two\n# three\n# four\n# five\n")
    result = run_cli()
    assert result.returncode != 0
    assert "increment/sample.py:1" in result.stdout


def test_short_exception_still_requires_a_reason(tmp_path: Path) -> None:
    result = audit(write_source(tmp_path, "# prose: allow-long\nvalue = 1\n"))
    assert [(item["kind"], item["line"]) for item in result["violations"]] == [
        ("invalid-exception", 1)
    ]


def test_module_docstring_diagnostic_uses_actual_source_line(tmp_path: Path) -> None:
    path = write_source(tmp_path, '#!/usr/bin/env python\n# coding: utf-8\n\n"""superpowers"""\n')
    result = run_cli(str(path))
    assert result.returncode == 1
    assert f"{path}:4:" in result.stdout


@pytest.mark.parametrize(
    "directive",
    ["#!/usr/bin/env python", "# -*- coding: utf-8 -*-", "# fmt: off", "# noqa: E501"],
)
def test_standard_directives_do_not_count_as_prose(tmp_path: Path, directive: str) -> None:
    path = write_source(tmp_path, f"{directive}\n# one\n# two\n# three\n# four\n")
    assert audit(path)["violations"] == []


def test_empty_comment_counts_toward_the_block_limit(tmp_path: Path) -> None:
    path = write_source(
        tmp_path, "if True:\n    # one\n    # two\n    #\n    # three\n    # four\n    pass\n"
    )
    assert [(item["kind"], item["line"]) for item in audit(path)["violations"]] == [
        ("comment-block", 2)
    ]


def test_invalid_python_is_reported_after_tokenization(tmp_path: Path) -> None:
    path = write_source(tmp_path, "value =\n")
    assert audit(path)["errors"]
    assert run_cli(str(path)).returncode == 1


def test_explicit_missing_target_cannot_be_hidden_by_clean_file(tmp_path: Path) -> None:
    path = write_source(tmp_path, "value = 1\n")
    result = run_cli(str(path), str(tmp_path / "missing-directory"))
    assert result.returncode == 1
    assert "missing-directory" in result.stdout


def test_no_target_scan_includes_hook_configuration_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "noxfile.py").write_text("# one\n# two\n# three\n# four\n# five\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "example.py").write_text("# roborev review\n")
    result = run_cli()
    assert result.returncode == 1
    assert "noxfile.py:1:" in result.stdout
    assert "docs/example.py:1:" in result.stdout


def test_empty_discovery_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_cli().returncode == 1


def test_ordinary_planning_and_long_docstrings_are_allowed(tmp_path: Path) -> None:
    path = write_source(
        tmp_path,
        '"""This plan controls global constraints and signal amplitude.\n'
        + "Detailed parameter documentation.\n" * 45
        + '"""\n',
    )
    assert audit(path)["violations"] == []
