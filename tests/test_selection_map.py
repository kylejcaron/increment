from __future__ import annotations

import json
from pathlib import Path

import pytest
from coverage import CoverageData

from scripts.selection_map import build, is_stale

pytestmark = pytest.mark.integration


def test_build_maps_each_module_to_executing_test_file(tmp_path: Path) -> None:
    source = tmp_path / "increment/a.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")
    cov = CoverageData(basename=str(tmp_path / ".coverage"))
    cov.set_context("tests/test_a.py::test_x|run")
    cov.add_lines({str(source): {1}})
    cov.write()

    assert build(tmp_path / ".coverage", tmp_path, tier="fast")["modules"] == {
        "increment.a": ["tests/test_a.py"]
    }


def test_empty_import_context_does_not_create_map_entry(tmp_path: Path) -> None:
    source = tmp_path / "increment/a.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")
    cov = CoverageData(basename=str(tmp_path / ".coverage"))
    cov.set_context("")
    cov.add_lines({str(source): {1}})
    cov.write()

    assert build(tmp_path / ".coverage", tmp_path, tier="fast")["modules"] == {}


def test_stale_map_detects_unrelated_base(tmp_path: Path) -> None:
    assert is_stale(
        {"generated_at": "2026-10-25T00:00:00+00:00", "base_sha": "not-a-commit", "tier": "fast", "modules": {}},
        root=tmp_path,
    )


def test_collection_dynamic_import_maps_to_its_test_file(tmp_path: Path) -> None:
    source = tmp_path / "increment/dynamic.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    cov = CoverageData(basename=str(tmp_path / ".coverage"))
    cov.set_context("collect:tests/test_dynamic.py")
    cov.add_lines({str(source): {1}})
    cov.write()

    assert build(tmp_path / ".coverage", tmp_path, tier="fast")["modules"] == {
        "increment.dynamic": ["tests/test_dynamic.py"]
    }


def test_fixture_dependencies_survive_xdist_worker_merge(tmp_path: Path) -> None:
    source = tmp_path / "increment/fixture_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    cov = CoverageData(basename=str(tmp_path / ".coverage"))
    cov.set_context("tests/test_first.py::test_one|setup")
    cov.add_lines({str(source): {1}})
    cov.write()
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "fixtures.gw0.json").write_text(json.dumps({
        "closures": {"tests/test_first.py::test_one": ["db"]},
        "setup": {"db": ["tests/test_first.py::test_one"]},
    }))
    (fixtures / "fixtures.gw1.json").write_text(json.dumps({
        "closures": {"tests/test_second.py::test_two": ["db"]},
        "setup": {},
    }))

    result = build(tmp_path / ".coverage", tmp_path, tier="fast", fixtures=fixtures)

    assert result["modules"]["increment.fixture_source"] == [
        "tests/test_first.py", "tests/test_second.py"
    ]


def test_failed_test_excluded_from_prediction_is_reported_as_miss(tmp_path: Path) -> None:
    from scripts.selection_map import misses

    prediction = tmp_path / "affected-report.json"
    prediction.write_text(json.dumps({
        "tests/test_skipped.py": {"kept": False, "reasons": [], "tach_kept": False}
    }))
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase file="tests/test_skipped.py"><failure/></testcase></testsuite>'
    )

    result = misses(
        {"generated_at": "now", "base_sha": "base"},
        base="base", head="head", junit_reports=[junit], report=prediction,
    )

    assert result == ["miss: tests/test_skipped.py"]
    assert "base: base" in (tmp_path / "selection-misses.txt").read_text()
