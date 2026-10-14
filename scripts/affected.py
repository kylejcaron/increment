"""Select and run tests affected by changed repository paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FAST_MARKER = "not slow and not parameter_recovery and not numerical_certification"
SLOW_MARKER = "slow and not parameter_recovery and not numerical_certification and not examples"
_FULL_TIER_PATHS = {"pyproject.toml", "uv.lock", "conftest.py", "tach.toml", "Makefile"}


@dataclass(frozen=True, slots=True)
class Selection:
    test_files: tuple[str, ...]
    doc_pages: tuple[str, ...]
    use_tach: bool
    full_tier: bool
    uncovered: tuple[str, ...]


def _git(*arguments: str, root: Path) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    )
    return result.stdout


def _changed_paths(base: str, root: Path) -> tuple[str, ...]:
    try:
        _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", root=root)
        paths = set(_git("diff", "--name-only", f"{base}...HEAD", root=root).splitlines())
        paths.update(_git("diff", "--name-only", "--cached", root=root).splitlines())
        paths.update(_git("diff", "--name-only", root=root).splitlines())
        paths.update(_git("ls-files", "--others", "--exclude-standard", root=root).splitlines())
    except subprocess.CalledProcessError as error:
        raise ValueError(f"base is not a resolvable commit: {base}") from error
    return tuple(sorted(path for path in paths if path))


def _test_file(path: str) -> bool:
    return (
        path.startswith("tests/") and Path(path).name.startswith("test_") and path.endswith(".py")
    )


def _test_files_under(root: Path, directory: str) -> set[str]:
    target = root / directory
    if not target.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix() for path in target.rglob("test_*.py") if path.is_file()
    }


def select(base: str, root: Path) -> Selection:
    """Classify changed paths into locally runnable checks."""
    changed = _changed_paths(base, root)
    full_tier = any(path in _FULL_TIER_PATHS for path in changed)
    test_files: set[str] = set()
    doc_pages: set[str] = set()
    use_tach = False
    uncovered: set[str] = set()

    for path in changed:
        if path in _FULL_TIER_PATHS:
            continue
        if path == "README.md" or (path.startswith("docs/") and path.endswith(".md")):
            if (root / path).is_file():
                doc_pages.add(path)
        elif _test_file(path):
            if (root / path).is_file():
                test_files.add(path)
        elif path.startswith("tests/") and path.endswith("/conftest.py"):
            test_files.update(_test_files_under(root, str(Path(path).parent)))
        elif path.startswith("tests/"):
            use_tach = True
        elif path.startswith("increment/") and path.endswith(".py"):
            use_tach = True
        else:
            uncovered.add(path)

    return Selection(
        () if full_tier else tuple(sorted(test_files)),
        tuple(sorted(doc_pages)),
        use_tach and not full_tier,
        full_tier,
        tuple(sorted(uncovered)),
    )


def _pytest_invocations(
    selection: Selection, base: str, tier: str, extra_args: Sequence[str]
) -> list[list[str]]:
    markers = ["-m", {"fast": FAST_MARKER, "slow": SLOW_MARKER}[tier]]

    if selection.full_tier:
        roots = ["tests"] if selection.doc_pages else []
        return [markers + roots + list(selection.doc_pages) + list(extra_args)]

    if selection.use_tach:
        tach_args = markers + ["--tach-base", base]
        if selection.test_files or selection.doc_pages:
            tach_args.append("tests")
        tach_args.extend(extra_args)
        if not (selection.test_files or selection.doc_pages):
            return [tach_args]
        direct_args = markers + list(selection.test_files) + list(selection.doc_pages)
        direct_args.extend(extra_args)
        return [tach_args, direct_args]

    direct_args = markers + list(selection.test_files) + list(selection.doc_pages)
    direct_args.extend(extra_args)
    return [direct_args]


def _pytest_arguments(
    selection: Selection, base: str, tier: str, extra_args: Sequence[str]
) -> list[str]:
    return _pytest_invocations(selection, base, tier, extra_args)[0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tier", choices=("fast", "slow"), default="fast")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    extra_args = arguments.pytest_args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    root = Path.cwd()
    try:
        selection = select(arguments.base, root)
    except ValueError as error:
        print(f"affected: {error}", file=sys.stderr)
        return 2

    for path in selection.uncovered:
        print(f"{path}: not covered locally; CI covers it")

    if not selection.full_tier and not (
        selection.test_files or selection.doc_pages or selection.use_tach
    ):
        print("affected: no affected tests selected")
        return 0
    if selection.full_tier:
        print(f"affected: running full {arguments.tier} tier")

    status = 0
    for pytest_args in _pytest_invocations(selection, arguments.base, arguments.tier, extra_args):
        command = [sys.executable, "-m", "pytest", *pytest_args]
        result = subprocess.run(command, cwd=root, check=False).returncode
        if status == 0 and result != 0:
            status = result
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
