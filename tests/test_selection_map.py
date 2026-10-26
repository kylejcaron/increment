from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from coverage import CoverageData

from scripts.selection_map import SelectionMap, build, is_stale

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
        {
            "generated_at": "2026-10-25T00:00:00+00:00",
            "base_sha": "not-a-commit",
            "tier": "fast",
            "modules": {},
        },
        root=tmp_path,
    )


@pytest.mark.parametrize(
    ("path", "before", "after"),
    [
        ("increment/declarations.py", "REGISTRY = {}\n", "REGISTRY = {'new': 1}\n"),
        ("increment/estimators/registry.py", "REGISTRY = {}\n", "REGISTRY = {'new': 1}\n"),
        (
            "increment/results.py",
            "_LAZY_IMPORTS = {'Name': ('increment.impl', 'Name')}\n",
            "VALUE = 1\n",
        ),
    ],
)
def test_dispatch_table_changes_invalidate_map(
    tmp_path: Path, path: str, before: str, after: str
) -> None:
    root = tmp_path / "repo"
    source = root / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(before)
    source = root / "increment/__init__.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    mapping: SelectionMap = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_sha": base,
        "tier": "fast",
        "modules": {},
    }

    (root / path).write_text(after)

    assert is_stale(mapping, root=root)


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
    project = tmp_path / "project"
    tests = project / "tests"
    tests.mkdir(parents=True)
    (project / "increment").mkdir()
    (project / "increment/__init__.py").write_text("")
    (project / "increment/fixture_source.py").write_text(
        "def touch():\n    return 'fixture dependency'\n"
    )
    (project / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (project / "conftest.py").write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def shared(request):\n"
        "    if 'test_second.py' in request.node.nodeid:\n"
        "        from increment.fixture_source import touch\n"
        "        touch()\n"
    )
    for name in ("first", "second"):
        (tests / f"test_{name}.py").write_text(
            "import pytest\n"
            "@pytest.mark.xdist_group('shared-fixture')\n"
            f"def test_{name}(shared):\n"
            "    pass\n"
        )
    repo_root = Path(__file__).resolve().parents[1]
    fixtures = project / "fixtures"
    env = dict(os.environ)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-n",
            "2",
            "--dist=loadgroup",
            "--cov=increment",
            "--cov-context=test",
            "--cov-report=",
            "--selection-fixtures",
            str(fixtures),
            "tests",
        ],
        cwd=project,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    result = build(project / ".coverage", project, tier="fast", fixtures=fixtures)
    mapping = project / "selection_map.json"
    from scripts.selection_map import write

    write(result, mapping)
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; from scripts.selection_map import load; "
            "print(json.dumps((load(Path(sys.argv[1])) or {}).get('modules', {}).get("
            "'increment.fixture_source', [])))",
            str(mapping),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert json.loads(loaded.stdout) == ["tests/test_first.py", "tests/test_second.py"]


def test_failed_test_excluded_from_prediction_is_reported_as_miss(tmp_path: Path) -> None:
    from scripts.selection_map import misses

    prediction = tmp_path / "affected-report.json"

    shard_id = "3.12-1"
    prediction.write_text(
        json.dumps(
            {
                "tests/test_skipped.py": {
                    "kept": False,
                    "reasons": [],
                    "tach_kept": False,
                },
                "__metadata__": {
                    "shards": {
                        shard_id: {
                            "collection_complete": True,
                            "collected_tests": 1,
                            "decision_files": ["tests/test_skipped.py"],
                        }
                    }
                },
            }
        )
    )
    junit_dir = tmp_path / "fast-junit-3.12-shard-1-1"
    junit_dir.mkdir()
    junit = junit_dir / "junit.xml"
    junit.write_text(
        '<testsuite><testcase file="tests/test_skipped.py"><failure/></testcase></testsuite>'
    )
    junit_metadata = junit_dir / "junit-metadata.json"
    junit_metadata.write_text(json.dumps({"shard_id": shard_id, "junit_present": True}))

    result = misses(
        {"generated_at": "now", "base_sha": "base", "tier": "fast", "modules": {}},
        base="base",
        head="head",
        junit_reports=[junit],
        junit_metadata=[junit_metadata],
        expected_shards={shard_id},
        report=prediction,
    )

    assert result == ["miss: tests/test_skipped.py"]
    assert "base: base" in (tmp_path / "selection-misses.txt").read_text()


def test_partial_prediction_is_incomplete_evidence(tmp_path: Path) -> None:
    from scripts.selection_map import misses

    shard_id = "3.12-1"
    prediction = tmp_path / "affected-report.json"
    prediction.write_text(
        json.dumps(
            {
                "__metadata__": {
                    "shards": {
                        shard_id: {
                            "collection_complete": True,
                            "collected_tests": 1,
                            "decision_files": ["tests/test_skipped.py"],
                        }
                    }
                }
            }
        )
    )
    junit = tmp_path / f"fast-junit-{shard_id.replace('-', '-shard-')}-1"
    junit.mkdir()
    (junit / "junit.xml").write_text(
        '<testsuite><testcase file="tests/test_skipped.py"><failure/></testcase></testsuite>'
    )
    junit_metadata = junit / "junit-metadata.json"
    junit_metadata.write_text(json.dumps({"shard_id": shard_id, "junit_present": True}))

    result = misses(
        {"generated_at": "now", "base_sha": "base", "tier": "fast", "modules": {}},
        base="base",
        head="head",
        junit_reports=[junit / "junit.xml"],
        junit_metadata=[junit_metadata],
        expected_shards={shard_id},
        report=prediction,
    )

    assert "incomplete evidence" in result


def test_missing_expected_junit_shard_is_incomplete_evidence(tmp_path: Path) -> None:
    from scripts.selection_map import misses

    shard_ids = {"3.12-1", "3.12-2"}
    prediction = tmp_path / "affected-report.json"
    prediction.write_text(
        json.dumps(
            {
                "__metadata__": {
                    "shards": {
                        shard_id: {
                            "collection_complete": True,
                            "collected_tests": 1,
                            "decision_files": [],
                        }
                        for shard_id in shard_ids
                    }
                }
            }
        )
    )
    junit_dir = tmp_path / "fast-junit-3.12-shard-1-1"
    junit_dir.mkdir()
    junit = junit_dir / "junit.xml"
    junit.write_text("<testsuite/>")
    junit_metadata = junit_dir / "junit-metadata.json"
    junit_metadata.write_text(json.dumps({"shard_id": "3.12-1", "junit_present": True}))

    result = misses(
        {"generated_at": "now", "base_sha": "base", "tier": "fast", "modules": {}},
        base="base",
        head="head",
        junit_reports=[junit],
        junit_metadata=[junit_metadata],
        expected_shards=shard_ids,
        report=prediction,
    )

    assert "incomplete evidence" in result


def test_merge_reports_marks_missing_decision_incomplete(tmp_path: Path) -> None:
    from scripts.selection_map import merge_reports

    shard = tmp_path / "prediction.json"
    shard.write_text(
        json.dumps(
            {
                "tests/test_present.py": {"kept": True, "reasons": [], "tach_kept": True},
                "__metadata__": {
                    "shard_id": "3.12-1",
                    "collection_complete": True,
                    "collected_tests": 2,
                    "decision_files": ["tests/test_present.py", "tests/test_missing.py"],
                },
            }
        )
    )

    merged = tmp_path / "merged.json"
    merge_reports([shard], merged, expected_shards={"3.12-1", "3.12-2"})

    payload = json.loads(merged.read_text())
    assert payload["__incomplete__"] is True
    assert set(payload["__metadata__"]["shards"]) == {"3.12-1"}
