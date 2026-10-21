from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import pytest

FANOUT_LIMIT = 12


def _baseline(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2 or not fields[1].isdecimal() or fields[0] in values:
            raise ValueError(f"{path}:{line_number}: expected '<tier> <items>'")
        values[fields[0]] = int(fields[1])
    return values


def _collected_items(output: str) -> int:
    selected = re.search(r"(\d+)\s*/\s*\d+\s+tests?\s+collected", output)
    if selected:
        return int(selected.group(1))
    match = re.search(r"(\d+)\s+(?:tests?\s+)?collected", output)
    if match:
        return int(match.group(1))
    file_counts = re.findall(r"^.*\.py:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    if file_counts:
        return sum(map(int, file_counts))
    raise ValueError("stdin does not contain pytest --collect-only per-file counts")


def _has_grid_reason(decorator: ast.expr) -> bool:
    if not (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "grid"
        and decorator.args
    ):
        return False
    try:
        reason = ast.literal_eval(decorator.args[0])
    except (ValueError, TypeError):
        return False
    return isinstance(reason, str) and bool(reason.strip())


def _grid_markers(root: Path) -> set[str]:
    marked: set[str] = set()
    for path in root.rglob("test*.py"):
        resolved = path.resolve()
        try:
            relpath = resolved.relative_to(Path.cwd()).as_posix()
        except ValueError:
            relpath = os.path.relpath(resolved, Path.cwd())
        root_relpath = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), filename=relpath)

        def visit(node: ast.AST, relpath: str, root_relpath: str, stack: list[str]) -> None:
            if isinstance(node, ast.ClassDef):
                stack.append(node.name)
                for child in node.body:
                    visit(child, relpath, root_relpath, stack)
                stack.pop()
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_") and any(
                    _has_grid_reason(decorator) for decorator in node.decorator_list
                ):
                    qualname = "::".join((*stack, node.name))
                    for file_key in {relpath, root_relpath, f"tests/{root_relpath}"}:
                        marked.add(f"{file_key}::{qualname}")
                stack.append(node.name)
                for child in node.body:
                    visit(child, relpath, root_relpath, stack)
                stack.pop()
            elif isinstance(node, ast.Module):
                for child in node.body:
                    visit(child, relpath, root_relpath, stack)

        visit(tree, relpath, root_relpath, [])
    return marked


def pytest_collection_finish(session: pytest.Session) -> None:
    for item in session.items:
        print(item.nodeid)


def _check_fanout(nodeids: str, root: Path) -> list[str]:
    counts: dict[str, int] = {}
    for line in nodeids.splitlines():
        if "::test_" not in line:
            continue
        key = line.split("[", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    marked = _grid_markers(root)
    return [
        f"{key}: {count} collected items; use boundary cases plus a Hypothesis property, or @pytest.mark.grid('<reason>')"
        for key, count in sorted(counts.items())
        if count > FANOUT_LIMIT and key not in marked
    ]


def check_counts(
    tier: str, baseline_path: Path, items: int, nodeids: str = "", tests_root: Path = Path("tests")
) -> tuple[int, str]:
    try:
        baselines = _baseline(baseline_path)
        if tier not in baselines:
            raise ValueError(f"{baseline_path}: no baseline for tier {tier}")
    except (OSError, ValueError) as error:
        return 2, f"{error}\n"
    baseline = baselines[tier]
    findings = _check_fanout(nodeids, tests_root) if nodeids else []
    if items > baseline:
        delta = items - baseline
        message = (
            f"{tier}: {items} items (baseline {baseline}; +{delta})\n"
            f"tests: +{delta} items — <reason>\n"
            "Literal or computed parameter grids over 12 collected items require a reasoned grid marker.\n"
        )
        if findings:
            message += "Collected parameter fan-out findings:\n" + "\n".join(findings) + "\n"
        return 1, message
    message = (
        f"{tier} {items} (baseline {baseline}; lower count — lower the baseline)\n"
        if items < baseline
        else f"{tier} {items} (baseline {baseline})\n"
    )
    if findings:
        message += "Collected parameter fan-out findings (add a reasoned grid marker unless deliberately handling the baseline):\n"
        message += "\n".join(findings) + "\n"
    return 0, message


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--items", type=int)
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    args = parser.parse_args()
    output = sys.stdin.read() if args.items is None else ""
    try:
        items = args.items if args.items is not None else _collected_items(output)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    code, message = check_counts(args.tier, args.baseline, items, output, args.tests_root)
    print(message, end="", file=sys.stdout if code != 2 else sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
