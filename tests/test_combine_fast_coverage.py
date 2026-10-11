from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import coverage

from scripts.combine_fast_coverage import combine_shards


def _shard(
    root: Path,
    *,
    shard: int,
    sha: str = "abc123",
    attempt: int = 2,
    outcome: str = "success",
    coverage_outcome: str = "success",
    has_increment_data: bool = True,
) -> Path:
    artifact = root / f"shard-{shard}"
    artifact.mkdir(parents=True)
    project = artifact / "project"
    module_dir = project / ("increment" if has_increment_data else "other")
    module_dir.mkdir(parents=True)
    source = module_dir / "shared.py"
    source.write_text(
        "def called(first):\n"
        "    if first:\n"
        "        return 1\n"
        "    return 2\n"
        "\n"
        "def untouched():\n"
        "    return 0\n"
    )
    sys.path.insert(0, str(project))
    try:
        collector = coverage.Coverage(data_file=str(artifact / ".coverage"))
        collector.start()
        namespace: dict[str, object] = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        called = cast(Callable[[bool], object], namespace["called"])
        called(shard == 1)
        collector.stop()
        collector.json_report(outfile=str(artifact / "coverage.json"))
    finally:
        sys.path.remove(str(project))
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "sha": sha,
                "attempt": attempt,
                "python": "3.12",
                "shard": shard,
                "outcome": outcome,
                "coverage_outcome": coverage_outcome,
            }
        )
    )
    return artifact


def test_combine_fast_coverage_unions_overlapping_file_lines(tmp_path: Path):
    _shard(tmp_path, shard=1)
    _shard(tmp_path, shard=2)

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert result.complete
    assert result.coverage is not None
    expected: set[int] = set()
    for shard in (1, 2):
        report = json.loads((tmp_path / f"shard-{shard}" / "coverage.json").read_text())
        source_path = next(path for path in report["files"] if "/increment/" in path)
        expected.update(report["files"][source_path]["executed_lines"])
    combined = result.coverage["files"]["increment/shared.py"]
    assert combined["executed_lines"] == sorted(expected)
    assert combined["summary"]["covered_lines"] == len(expected)
    assert combined["summary"]["missing_lines"] > 0
    assert "not whole-package" in result.summary


def test_combine_fast_coverage_rejects_empty_increment_shard(tmp_path: Path):
    _shard(tmp_path, shard=1)
    _shard(tmp_path, shard=2, has_increment_data=False)

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert not result.complete
    assert result.coverage is None
    assert "shard 2 has no executed increment/ source statements" in result.summary


def test_combine_fast_coverage_reports_missing_shard_incomplete(tmp_path: Path):
    _shard(tmp_path, shard=1)

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert not result.complete
    assert result.coverage is None
    assert "missing shard 2" in result.summary


def test_combine_fast_coverage_rejects_wrong_sha(tmp_path: Path):
    _shard(tmp_path, shard=1)
    _shard(tmp_path, shard=2, sha="different")

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert not result.complete
    assert result.coverage is None
    assert "SHA" in result.summary


def test_combine_fast_coverage_rejects_failed_shard(tmp_path: Path):
    _shard(tmp_path, shard=1)
    _shard(tmp_path, shard=2, outcome="failure")

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert not result.complete
    assert result.coverage is None
    assert "shard 2 did not succeed" in result.summary


def test_combine_fast_coverage_rejects_collection_failure(tmp_path: Path):
    _shard(tmp_path, shard=1, coverage_outcome="failure")
    _shard(tmp_path, shard=2)

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert not result.complete
    assert result.coverage is None
    assert "shard 1 coverage collection did not succeed" in result.summary
