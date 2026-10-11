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
) -> Path:
    artifact = root / f"shard-{shard}"
    artifact.mkdir(parents=True)
    project = artifact / "project"
    package = project / "increment"
    package.mkdir(parents=True)
    source = package / f"module_{shard}.py"
    source.write_text(
        f"def called_{shard}():\n    return {shard}\n\ndef untouched_{shard}():\n    return 0\n"
    )
    sys.path.insert(0, str(project))
    try:
        collector = coverage.Coverage(data_file=str(artifact / ".coverage"))
        collector.start()
        namespace: dict[str, object] = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        called = cast(Callable[[], object], namespace[f"called_{shard}"])
        called()
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
            }
        )
    )
    return artifact


def test_combine_fast_coverage_requires_and_unions_both_successful_shards(tmp_path: Path):
    _shard(tmp_path, shard=1)
    _shard(tmp_path, shard=2)

    result = combine_shards(tmp_path, expected_sha="abc123", expected_attempt=2)

    assert result.complete
    assert result.coverage is not None
    for shard in (1, 2):
        report = json.loads((tmp_path / f"shard-{shard}" / "coverage.json").read_text())
        source_path = next(path for path in report["files"] if "/increment/" in path)
        source_report = report["files"][source_path]
        combined = result.coverage["files"][f"increment/module_{shard}.py"]
        assert combined["executed_lines"] == source_report["executed_lines"]
        assert combined["summary"]["covered_lines"] >= 2
        assert combined["summary"]["missing_lines"] > 0
    assert "not whole-package" in result.summary


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
