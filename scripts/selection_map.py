from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict, cast


class SelectionMap(TypedDict):
    generated_at: str
    base_sha: str
    tier: str
    modules: dict[str, list[str]]


class SelectionDecision(TypedDict, total=False):
    kept: bool
    reasons: list[str]
    tach_kept: bool


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _module(path: str, root: Path) -> str | None:
    try:
        relative = Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] != "increment" or relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build(
    coverage_path: Path, root: Path, *, tier: str, fixtures: Path | None = None
) -> SelectionMap:
    from coverage import CoverageData

    data = CoverageData(basename=str(coverage_path))
    data.read()
    modules: dict[str, set[str]] = defaultdict(set)
    fixture_consumers: dict[str, set[str]] = defaultdict(set)
    fixture_setup: dict[str, set[str]] = defaultdict(set)
    if fixtures is not None and fixtures.is_dir():
        for file in sorted(fixtures.glob("fixtures.*.json")):
            try:
                worker = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for nodeid, names in worker.get("closures", {}).items():
                test = nodeid.split("::", 1)[0]
                for name in names:
                    fixture_consumers[name].add(test)
            for test_file, names in worker.get("collected_imports", {}).items():
                for name in names:
                    if isinstance(test_file, str) and isinstance(name, str):
                        modules[name].add(test_file)
            for name, nodeids in worker.get("setup", {}).items():
                fixture_setup[name].update(nodeids)

    for filename in data.measured_files():
        module = _module(filename, root)
        if module is None:
            continue
        contexts_by_line = data.contexts_by_lineno(filename)
        for contexts in contexts_by_line.values():
            for context in contexts:
                if not context:
                    continue
                nodeid, _, phase = context.partition("|")
                test_file = nodeid.split("::", 1)[0]
                if phase == "setup":
                    if test_file.startswith("tests/"):
                        modules[module].add(test_file)
                    for fixture, producers in fixture_setup.items():
                        if nodeid in producers:
                            modules[module].update(fixture_consumers.get(fixture, ()))
                elif test_file.startswith("tests/"):
                    modules[module].add(test_file)
    try:
        base_sha = _git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        base_sha = "unknown"
    selection_map: SelectionMap = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_sha": base_sha,
        "tier": tier,
        "modules": {name: sorted(files) for name, files in sorted(modules.items())},
    }
    return selection_map


def write(selection_map: SelectionMap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(path: Path) -> SelectionMap | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not all(isinstance(payload.get(key), str) for key in ("generated_at", "base_sha", "tier")):
        return None
    raw_modules = payload.get("modules")
    if not isinstance(raw_modules, dict):
        return None
    modules: dict[str, list[str]] = {}
    for name, files in raw_modules.items():
        if not isinstance(name, str) or not isinstance(files, list):
            return None
        if not all(isinstance(file, str) for file in files):
            return None
        modules[name] = files
    return {
        "generated_at": payload["generated_at"],
        "base_sha": payload["base_sha"],
        "tier": payload["tier"],
        "modules": modules,
    }


def tests_changed_since(selection_map: SelectionMap, root: Path) -> set[str]:
    base = selection_map["base_sha"]
    if not base or base == "unknown":
        return set()
    try:
        paths = set(_git(root, "diff", "--name-only", f"{base}...HEAD").splitlines())
        paths.update(_git(root, "diff", "--name-only").splitlines())
        paths.update(_git(root, "diff", "--name-only", "--cached").splitlines())
        paths.update(_git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {
        path
        for path in paths
        if path.startswith("tests/")
        and Path(path).name.endswith(".py")
        and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
    }


def _has_lazy_table(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_LAZY_IMPORTS"
        )
        for node in tree.body
    )


def is_stale(selection_map: SelectionMap, *, max_age_days: int = 14, root: Path) -> bool:
    try:
        generated = datetime.fromisoformat(selection_map["generated_at"])
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=UTC)
        if datetime.now(UTC) - generated > timedelta(days=max_age_days):
            return True
        base = selection_map["base_sha"]
        _git(root, "merge-base", "--is-ancestor", base, "HEAD")
        changed = set(_git(root, "diff", "--name-only", f"{base}...HEAD").splitlines())
        changed.update(_git(root, "diff", "--name-only").splitlines())
        changed.update(_git(root, "diff", "--name-only", "--cached").splitlines())
        changed.update(_git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError):
        return True
    for path in changed:
        if path == "increment/declarations.py" or path.startswith("increment/estimators/"):
            return True
        if not path.startswith("increment/") or not path.endswith(".py"):
            continue
        try:
            current = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            current = ""
        try:
            previous = _git(root, "show", f"{base}:{path}")
        except (OSError, subprocess.CalledProcessError):
            previous = ""
        if _has_lazy_table(current) or _has_lazy_table(previous):
            return True
    return False


def _load_decisions(
    path: Path,
) -> tuple[dict[str, SelectionDecision], dict[str, dict[str, object]], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, {}, False
    if not isinstance(payload, dict):
        return {}, {}, False
    incomplete = payload.get("__incomplete__", False)
    metadata = payload.get("__metadata__")
    if not isinstance(incomplete, bool) or not isinstance(metadata, dict):
        return {}, {}, False
    raw_shards = metadata.get("shards")
    if not isinstance(raw_shards, dict):
        return {}, {}, False
    shards: dict[str, dict[str, object]] = {}
    expected_files: set[str] = set()
    for shard_id, raw in raw_shards.items():
        if not isinstance(shard_id, str) or not isinstance(raw, dict):
            return {}, {}, False
        complete = raw.get("collection_complete")
        collected_tests = raw.get("collected_tests")
        decision_files = raw.get("decision_files")
        if (
            not isinstance(complete, bool)
            or not isinstance(collected_tests, int)
            or collected_tests < 1
            or not isinstance(decision_files, list)
            or not all(isinstance(name, str) for name in decision_files)
        ):
            return {}, {}, False
        if collected_tests > 0 and not decision_files:
            incomplete = True
        expected_files.update(decision_files)
        shards[shard_id] = raw

    decisions: dict[str, SelectionDecision] = {}
    for name, raw in payload.items():
        if name in {"__incomplete__", "__metadata__"}:
            continue
        if not isinstance(name, str) or not isinstance(raw, dict):
            return {}, {}, False
        kept = raw.get("kept")
        tach_kept = raw.get("tach_kept")
        reasons = raw.get("reasons")
        if not isinstance(kept, bool) or not isinstance(tach_kept, bool):
            return {}, {}, False
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            return {}, {}, False
        decisions[name] = {"kept": kept, "tach_kept": tach_kept, "reasons": reasons}
    complete = not incomplete and expected_files <= decisions.keys()
    return decisions, shards, complete


def misses(
    selection_map: SelectionMap | None,
    *,
    base: str,
    head: str,
    junit_reports: list[Path],
    junit_metadata: list[Path],
    expected_shards: set[str],
    report: Path,
) -> list[str]:
    import xml.etree.ElementTree as ET

    decisions, prediction_shards, report_loaded = _load_decisions(report)
    skipped = {name for name, decision in decisions.items() if not decision["kept"]}
    tach_skips = sorted(
        name
        for name, decision in decisions.items()
        if decision["tach_kept"] and not decision["kept"]
    )
    failures: set[str] = set()
    junit_files: set[str] = set()
    incomplete = (
        not report_loaded
        or set(prediction_shards) != expected_shards
        or any(not shard["collection_complete"] for shard in prediction_shards.values())
    )

    junit_shards: dict[str, Path] = {}
    for metadata_path in junit_metadata:
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            incomplete = True
            continue
        shard_id = raw_metadata.get("shard_id") if isinstance(raw_metadata, dict) else None
        junit_present = (
            raw_metadata.get("junit_present") if isinstance(raw_metadata, dict) else None
        )
        junit_path = metadata_path.parent / "junit.xml"
        if (
            not isinstance(shard_id, str)
            or shard_id in junit_shards
            or junit_present is not True
            or not junit_path.is_file()
        ):
            incomplete = True
            continue
        junit_shards[shard_id] = junit_path
    if set(junit_shards) != expected_shards or {path.resolve() for path in junit_reports} != {
        path.resolve() for path in junit_shards.values()
    }:
        incomplete = True
    for junit in junit_reports:
        try:
            root = ET.parse(junit).getroot()
        except (OSError, ET.ParseError):
            incomplete = True
            continue
        for case in root.iter("testcase"):
            failed = any(child.tag in {"failure", "error"} for child in case)
            file = case.get("file")
            if not file:
                if failed:
                    incomplete = True
                continue
            normalized = Path(file).as_posix()
            marker = normalized.find("tests/")
            test_file = normalized[marker:] if marker >= 0 else normalized
            junit_files.add(test_file)
            if failed:
                failures.add(test_file)
    if not junit_files <= decisions.keys():
        incomplete = True
    result = [f"miss: {path}" for path in sorted(failures & skipped)]
    result.extend(f"tach-alone-kept-but-skipped: {path}" for path in tach_skips)
    if incomplete:
        result.insert(0, "incomplete evidence")
    report.with_name("selection-misses.txt").write_text(
        f"base: {base}\nhead: {head}\nmap generated_at: {(selection_map or {}).get('generated_at', 'missing')}\n"
        f"map base_sha: {(selection_map or {}).get('base_sha', 'missing')}\n"
        + "\n".join(result)
        + "\n",
        encoding="utf-8",
    )
    return result


def merge_reports(reports: list[Path], output: Path, *, expected_shards: set[str]) -> None:
    merged: dict[str, object] = {}
    shards: dict[str, dict[str, object]] = {}
    incomplete = False
    for path in reports:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            incomplete = True
            continue
        if not isinstance(payload, dict):
            incomplete = True
            continue
        metadata = payload.get("__metadata__")
        if not isinstance(metadata, dict):
            incomplete = True
            continue
        shard_id = metadata.get("shard_id")
        collection_complete = metadata.get("collection_complete")
        collected_tests = metadata.get("collected_tests")
        decision_files = metadata.get("decision_files")
        decisions = {name: value for name, value in payload.items() if not name.startswith("__")}
        if (
            not isinstance(shard_id, str)
            or shard_id in shards
            or not isinstance(collection_complete, bool)
            or not isinstance(collected_tests, int)
            or collected_tests < 1
            or not isinstance(decision_files, list)
            or not all(isinstance(name, str) for name in decision_files)
        ):
            incomplete = True
            continue
        shards[shard_id] = {
            "collection_complete": collection_complete,
            "collected_tests": collected_tests,
            "decision_files": decision_files,
        }
        if sorted(decisions) != sorted(decision_files) or not collection_complete:
            incomplete = True
            continue
        merged.update(decisions)
    if set(shards) != expected_shards:
        incomplete = True
    merged["__metadata__"] = {"shards": shards}
    merged["__incomplete__"] = incomplete
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--coverage", type=Path, required=True)
    builder.add_argument("--fixtures", type=Path)
    builder.add_argument("--tier", choices=("fast", "unit", "slow"), required=True)
    builder.add_argument("--output", type=Path, required=True)
    merger = sub.add_parser("merge")
    merger.add_argument("--reports", type=Path, nargs="*", required=True)
    merger.add_argument(
        "--expected-shards", type=lambda value: set(value.split(",")), required=True
    )
    merger.add_argument("--output", type=Path, required=True)
    miss = sub.add_parser("misses")
    miss.add_argument("--base", required=True)
    miss.add_argument("--head", required=True)
    miss.add_argument("--junit", type=Path, nargs="*", required=True)
    miss.add_argument("--junit-metadata", type=Path, nargs="*", required=True)
    miss.add_argument("--expected-shards", type=lambda value: set(value.split(",")), required=True)
    miss.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        write(build(args.coverage, Path.cwd(), tier=args.tier, fixtures=args.fixtures), args.output)
    elif args.command == "merge":
        merge_reports(
            cast(list[Path], args.reports),
            args.output,
            expected_shards=cast(set[str], args.expected_shards),
        )
    else:
        selection_map = load(Path("tests/selection_map.json"))
        for item in misses(
            selection_map,
            base=args.base,
            head=args.head,
            junit_reports=cast(list[Path], args.junit),
            junit_metadata=cast(list[Path], args.junit_metadata),
            expected_shards=cast(set[str], args.expected_shards),
            report=args.report,
        ):
            print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
