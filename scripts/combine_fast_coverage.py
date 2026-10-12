"""Combine complete, matching Python 3.12 fast-shard coverage artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CombineResult:
    complete: bool
    coverage: dict[str, Any] | None
    summary: str


def _package_path(path: str) -> str | None:
    parts = Path(path).parts
    try:
        index = parts.index("increment")
    except ValueError:
        return None
    return Path(*parts[index:]).as_posix()


def _incomplete(reasons: list[str]) -> CombineResult:
    return CombineResult(
        False, None, "Fast-tier executed-code coverage incomplete: " + "; ".join(reasons) + "."
    )


def combine_shards(shards_root: Path, *, expected_sha: str, expected_attempt: int) -> CombineResult:
    """Union executed statement lines only when both exact shard receipts qualify."""
    shards: dict[int, tuple[dict[str, Any], Path]] = {}
    reasons: list[str] = []
    for metadata_path in shards_root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            shard = metadata["shard"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            reasons.append(f"invalid metadata at {metadata_path}")
            continue
        if shard not in (1, 2) or shard in shards:
            reasons.append(f"unknown or duplicate shard {shard!r}")
            continue
        shards[shard] = (metadata, metadata_path.parent)

    for shard in (1, 2):
        if shard not in shards:
            reasons.append(f"missing shard {shard}")
            continue
        metadata, artifact = shards[shard]
        if metadata.get("sha") != expected_sha:
            reasons.append(f"shard {shard} SHA does not match candidate")
        if metadata.get("attempt") != expected_attempt:
            reasons.append(f"shard {shard} run attempt does not match")
        if metadata.get("python") != "3.12":
            reasons.append(f"shard {shard} is not Python 3.12")
        if metadata.get("outcome") != "success":
            reasons.append(f"shard {shard} did not succeed")
        if not (artifact / "coverage.json").is_file():
            reasons.append(f"shard {shard} has no coverage data")

    if reasons:
        return _incomplete(reasons)

    merged: dict[str, tuple[set[int], set[int]]] = {}
    try:
        for shard in (1, 2):
            _, artifact = shards[shard]
            report = json.loads((artifact / "coverage.json").read_text())
            package_files = [
                (_package_path(filename), details) for filename, details in report["files"].items()
            ]
            package_files = [
                (filename, details) for filename, details in package_files if filename is not None
            ]
            if not package_files or not any(
                details["executed_lines"] for _, details in package_files
            ):
                reasons.append(f"shard {shard} has no executed increment/ source statements")
                continue
            for filename, details in package_files:
                statements, executed = merged.setdefault(filename, (set(), set()))
                statements.update(details["executed_lines"])
                statements.update(details["missing_lines"])
                executed.update(details["executed_lines"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return _incomplete([f"invalid coverage data ({error})"])

    if reasons:
        return _incomplete(reasons)

    files: dict[str, dict[str, Any]] = {}
    for filename, (statements, executed) in sorted(merged.items()):
        covered = statements & executed
        missing = statements - covered
        files[filename] = {
            "executed_lines": sorted(executed),
            "missing_lines": sorted(missing),
            "summary": {
                "covered_lines": len(covered),
                "num_statements": len(statements),
                "missing_lines": len(missing),
                "percent_covered": 100.0 * len(covered) / len(statements) if statements else 100.0,
                "percent_covered_display": f"{100.0 * len(covered) / len(statements):.0f}"
                if statements
                else "100",
            },
            "statements": sorted(statements),
        }
    statement_count = sum(file["summary"]["num_statements"] for file in files.values())
    covered_count = sum(file["summary"]["covered_lines"] for file in files.values())
    if not files or not statement_count or not covered_count:
        return _incomplete(["coverage has no executed increment/ source statements"])

    total = {
        "covered_lines": covered_count,
        "num_statements": statement_count,
        "missing_lines": statement_count - covered_count,
        "percent_covered": 100.0 * covered_count / statement_count,
        "percent_covered_display": f"{100.0 * covered_count / statement_count:.0f}",
    }
    label = f"fast-tier executed-code coverage (in-process), Python 3.12, {expected_sha}"
    coverage_data: dict[str, Any] = {
        "meta": {"version": "1", "scope": label, "path_normalization": "increment/ relative paths"},
        "files": files,
        "totals": total,
    }
    summary = (
        f"{label}: {covered_count}/{statement_count} increment/ statements executed across "
        f"{len(files)} source files from fast shards 1 and 2. This measures executed code in the "
        "fast tier only; it is not whole-package, all-tier, or line-quality coverage."
    )
    return CombineResult(True, coverage_data, summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args(argv)

    result = combine_shards(
        arguments.shards, expected_sha=arguments.sha, expected_attempt=arguments.attempt
    )
    arguments.summary_file.write_text(result.summary + "\n")
    if result.complete:
        assert result.coverage is not None
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result.coverage, indent=2) + "\n")
    else:
        arguments.output.unlink(missing_ok=True)
    if arguments.github_output is not None:
        with arguments.github_output.open("a") as output:
            output.write(f"complete={str(result.complete).lower()}\n")
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
