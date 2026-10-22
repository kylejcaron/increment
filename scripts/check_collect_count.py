"""Compare collected test items per tier between a base commit and the working tree.

Both sides are collected with one ``--collect-only`` pass each; tier counts come
from each item's markers evaluated against ``TIER_MARKERS`` in ``noxfile.py``.
No count is committed, so parallel changes never conflict on a shared file.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("unit", "fast", "slow", "all")
FANOUT_LIMIT = 12
_PLUGIN = Path(__file__).with_name("collect_items_plugin.py")


@dataclass(frozen=True)
class Collection:
    tiers: Mapping[str, int]
    nodeids: tuple[str, ...]


def tier_markers(noxfile: Path = ROOT / "noxfile.py") -> dict[str, str]:
    """The marker expression of each counted tier, read without importing nox."""
    for node in ast.parse(noxfile.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TIER_MARKERS" for t in node.targets
        ):
            markers = ast.literal_eval(node.value)
            return {tier: markers[tier] for tier in TIERS}
    raise ValueError(f"{noxfile}: TIER_MARKERS not found")


def collect(
    checkout: Path, tests: str = "tests", markers: Mapping[str, str] | None = None
) -> Collection:
    """Collect *tests* inside *checkout* with that checkout's own source on the path."""
    markers = dict(markers or tier_markers())
    with tempfile.TemporaryDirectory(prefix="collect-items-") as scratch:
        plugin_dir = Path(scratch) / "plugin"
        plugin_dir.mkdir()
        shutil.copy(_PLUGIN, plugin_dir / _PLUGIN.name)
        output = Path(scratch) / "items.json"
        path = os.pathsep.join(
            p for p in (str(plugin_dir), str(checkout), os.environ.get("PYTHONPATH", "")) if p
        )
        env = {
            **os.environ,
            "PYTHONPATH": path,
            "INCREMENT_COLLECT_OUTPUT": str(output),
            "INCREMENT_COLLECT_TIERS": json.dumps(markers),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "slow or not slow",
                "-p",
                "collect_items_plugin",
                "-p",
                "no:cacheprovider",
                tests,
            ],
            cwd=checkout,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 5) or not output.exists():
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-20:])
            raise RuntimeError(
                f"collection failed in {checkout} (exit {result.returncode}):\n{tail}"
            )
        data = json.loads(output.read_text(encoding="utf-8"))
    return Collection(tiers=data["tiers"], nodeids=tuple(data["nodeids"]))


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def collect_base(repo: Path, base: str, tests: str = "tests") -> Collection:
    """Collect the merge base of *base* and ``HEAD`` in a throwaway worktree."""
    commit = _git(repo, "merge-base", base, "HEAD")
    with tempfile.TemporaryDirectory(prefix="collect-base-") as scratch:
        checkout = Path(scratch) / "base"
        _git(repo, "worktree", "add", "--detach", "--quiet", str(checkout), commit)
        try:
            return collect(checkout, tests)
        finally:
            _git(repo, "worktree", "remove", "--force", str(checkout))


def deltas(base: Mapping[str, int], head: Mapping[str, int]) -> dict[str, int]:
    return {tier: head[tier] - base[tier] for tier in TIERS}


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


def _fanout_message(nodeids: Sequence[str], tests_root: Path) -> str:
    findings = _check_fanout("\n".join(nodeids), tests_root)
    if not findings:
        return ""
    return (
        "Collected parameter fan-out findings "
        "(informational; they do not fail this check):\n"
        "Literal or computed parameter grids over 12 collected items require a reasoned grid marker.\n"
        + "\n".join(findings)
        + "\n"
    )


def report(base: Mapping[str, int], head: Mapping[str, int]) -> str:
    return "".join(
        f"{tier} {base[tier]} -> {head[tier]} ({delta:+d})\n"
        for tier, delta in deltas(base, head).items()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default="origin/main", help="compare against the merge base with this ref"
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--tests", default="tests")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        base = collect_base(repo, args.base, args.tests)
        head = collect(repo, args.tests)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"could not collect test items: {error}", file=sys.stderr)
        return 2
    print(_fanout_message(head.nodeids, repo / args.tests) + report(base.tiers, head.tiers), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
