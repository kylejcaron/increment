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
            for name, nodeids in worker.get("setup", {}).items():
                fixture_setup[name].update(nodeids)

    modules: dict[str, set[str]] = defaultdict(set)
    for filename in data.measured_files():
        module = _module(filename, root)
        if module is None:
            continue
        contexts_by_line = data.contexts_by_lineno(filename)
        for contexts in contexts_by_line.values():
            for context in contexts:
                if not context:
                    continue
                if context.startswith("collect:"):
                    modules[module].add(context.removeprefix("collect:").split("::", 1)[0])
                    continue
                nodeid, _, phase = context.partition("|")
                test_file = nodeid.split("::", 1)[0]
                if phase == "setup":
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
    return {path for path in paths if path.startswith("tests/test_") and path.endswith(".py")}


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


def _load_decisions(path: Path) -> tuple[dict[str, SelectionDecision], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    incomplete = payload.get("__incomplete__", False)
    if not isinstance(incomplete, bool):
        return {}, False
    decisions: dict[str, SelectionDecision] = {}
    for name, raw in payload.items():
        if name == "__incomplete__":
            continue
        if not isinstance(name, str) or not isinstance(raw, dict):
            return {}, False
        kept = raw.get("kept")
        tach_kept = raw.get("tach_kept")
        reasons = raw.get("reasons")
        if not isinstance(kept, bool) or not isinstance(tach_kept, bool):
            return {}, False
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            return {}, False
        decisions[name] = {"kept": kept, "tach_kept": tach_kept, "reasons": reasons}
    return decisions, not incomplete


def misses(
    selection_map: SelectionMap | None,
    *,
    base: str,
    head: str,
    junit_reports: list[Path],
    report: Path,
) -> list[str]:
    import xml.etree.ElementTree as ET

    decisions, report_loaded = _load_decisions(report)
    skipped = {name for name, decision in decisions.items() if not decision["kept"]}
    tach_skips = sorted(
        name
        for name, decision in decisions.items()
        if decision["tach_kept"] and not decision["kept"]
    )
    failures: set[str] = set()
    incomplete = not junit_reports or not report_loaded
    for junit in junit_reports:
        try:
            root = ET.parse(junit).getroot()
        except (OSError, ET.ParseError):
            incomplete = True
            continue
        for case in root.iter("testcase"):
            if not any(child.tag in {"failure", "error"} for child in case):
                continue
            file = case.get("file")
            if file:
                normalized = Path(file).as_posix()
                marker = normalized.find("tests/")
                failures.add(normalized[marker:] if marker >= 0 else normalized)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--coverage", type=Path, required=True)
    builder.add_argument("--fixtures", type=Path)
    builder.add_argument("--tier", choices=("fast", "unit", "slow"), required=True)
    builder.add_argument("--output", type=Path, required=True)
    miss = sub.add_parser("misses")
    miss.add_argument("--base", required=True)
    miss.add_argument("--head", required=True)
    miss.add_argument("--junit", type=Path, nargs="*", required=True)
    miss.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        write(build(args.coverage, Path.cwd(), tier=args.tier, fixtures=args.fixtures), args.output)
    else:
        selection_map = load(Path("tests/selection_map.json"))
        junit_reports = cast(list[Path], args.junit)
        for item in misses(
            selection_map,
            base=args.base,
            head=args.head,
            junit_reports=junit_reports,
            report=args.report,
        ):
            print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
