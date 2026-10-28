"""Select and run tests affected by changed repository paths."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_FULL_TIER_PATHS = {"uv.lock", "conftest.py", "Makefile"}


@dataclass(frozen=True, slots=True)
class Selection:
    test_files: tuple[str, ...]
    doc_pages: tuple[str, ...]
    use_tach: bool
    full_tier: bool
    uncovered: tuple[str, ...]
    affected_report: str = ".affected-report-fast.json"
    tier: str = "fast"


def _git(*arguments: str, root: Path) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    )
    return result.stdout


def _changed_paths(base: str, root: Path, *, head: str | None = None) -> tuple[str, ...]:
    try:
        _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", root=root)
        paths = set(
            _git("diff", "--name-only", f"{base}...{head or 'HEAD'}", root=root).splitlines()
        )
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


def _type_checking_import_ids(tree: ast.Module) -> set[int]:
    excluded: set[int] = set()

    def visit(node: ast.AST, *, in_type_checking: bool = False) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and in_type_checking:
            excluded.add(id(node))
        is_type_checking = isinstance(node, ast.If) and (
            (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
            or (
                isinstance(node.test, ast.Attribute)
                and node.test.attr == "TYPE_CHECKING"
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == "typing"
            )
        )
        if is_type_checking:
            assert isinstance(node, ast.If)
            visit(node.test, in_type_checking=in_type_checking)
            for statement in node.body:
                visit(statement, in_type_checking=True)
            for statement in node.orelse:
                visit(statement, in_type_checking=in_type_checking)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, in_type_checking=in_type_checking)

    visit(tree)
    return excluded


def _module_imports(path: Path, module: str, known_modules: set[str]) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return set()

    imports: set[str] = set()
    type_only_imports = _type_checking_import_ids(tree)
    for node in ast.walk(tree):
        if id(node) in type_only_imports:
            continue
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name in known_modules)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        base = node.module or ""
        if node.level:
            package = module if path.name == "__init__.py" else module.rpartition(".")[0]
            parts = package.split(".")
            if node.level > 1:
                parts = parts[: -(node.level - 1)]
            base = ".".join((*parts, *base.split("."))) if base else ".".join(parts)
        if base in known_modules:
            imports.add(base)
        imports.update(
            candidate
            for alias in node.names
            if alias.name != "*"
            if (candidate := f"{base}.{alias.name}") in known_modules
        )
    return imports


def _test_module_importers(root: Path, module_names: set[str]) -> set[str]:
    test_root = root / "tests"
    if not test_root.is_dir():
        return set()

    module_paths: dict[str, Path] = {}
    for path in test_root.rglob("*.py"):
        parts = path.relative_to(test_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_paths[".".join(("tests", *parts))] = path

    reverse_imports: dict[str, set[str]] = {}
    known_modules = set(module_paths)
    for module, path in module_paths.items():
        for imported in _module_imports(path, module, known_modules):
            reverse_imports.setdefault(imported, set()).add(module)

    visited = set(module_names)
    pending = list(module_names)
    test_importers: set[str] = set()
    while pending:
        imported = pending.pop()
        for importer in reverse_imports.get(imported, ()):
            if importer in visited:
                continue
            visited.add(importer)
            pending.append(importer)
            path = module_paths.get(importer)
            if path is not None and path.name.startswith("test_"):
                test_importers.add(path.relative_to(root).as_posix())
    return test_importers


def _pyproject_full_tier_change(base: str, root: Path) -> bool:
    try:
        previous = tomllib.loads(_git("show", f"{base}:pyproject.toml", root=root))
        current = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (subprocess.CalledProcessError, OSError, tomllib.TOMLDecodeError):
        return True

    def relevant(data: dict) -> dict:
        project = data.get("project", {})
        tool = data.get("tool", {})
        return {
            "project.dependencies": project.get("dependencies"),
            "project.optional-dependencies": project.get("optional-dependencies"),
            "dependency-groups": data.get("dependency-groups"),
            "tool.pytest": {key: value for key, value in tool.items() if key.startswith("pytest")},
            "tool.uv": tool.get("uv"),
        }

    return relevant(previous) != relevant(current)


def _changed_markers(tier: str, root: Path) -> str:
    for node in ast.parse((root / "noxfile.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TIER_MARKERS" for target in node.targets
        ):
            return ast.literal_eval(node.value)[tier]
    raise RuntimeError("noxfile.py does not define TIER_MARKERS")


def select(base: str, root: Path, *, tier: str = "fast") -> Selection:
    """Classify changed paths into locally runnable checks."""
    changed = _changed_paths(base, root)
    full_tier = any(path in _FULL_TIER_PATHS for path in changed) or (
        "pyproject.toml" in changed and _pyproject_full_tier_change(base, root)
    )
    test_files: set[str] = set()
    doc_pages: set[str] = set()
    use_tach = False
    changed_test_modules: set[str] = set()
    uncovered: set[str] = set()

    for path in changed:
        if path in _FULL_TIER_PATHS or path == "pyproject.toml":
            continue
        if path == "tach.toml":
            use_tach = True
            continue
        if path == "README.md" or (path.startswith("docs/") and path.endswith(".md")):
            if (root / path).is_file():
                doc_pages.add(path)
        elif _test_file(path):
            if (root / path).is_file():
                test_files.add(path)
                changed_test_modules.add(Path(path).with_suffix("").as_posix().replace("/", "."))
        elif path.startswith("tests/") and path.endswith("/conftest.py"):
            test_files.update(_test_files_under(root, str(Path(path).parent)))
        elif path.startswith("tests/"):
            use_tach = True
        elif path.startswith("increment/") and path.endswith(".py"):
            use_tach = True
        else:
            uncovered.add(path)
    if changed_test_modules and not full_tier:
        test_files.update(_test_module_importers(root, changed_test_modules))

    return Selection(
        () if full_tier else tuple(sorted(test_files)),
        tuple(sorted(doc_pages)),
        use_tach and not full_tier,
        full_tier,
        tuple(sorted(uncovered)),
        f".affected-report-{tier}.json",
        tier,
    )


def _without_xdist_worker_args(arguments: Sequence[str]) -> list[str]:
    """Drop xdist process-count and distribution options in all CLI forms."""
    result = []
    args = iter(arguments)
    for arg in args:
        if arg in {"-n", "--numprocesses", "--dist"}:
            next(args, None)
        elif arg.startswith("-n") and len(arg) > 2:
            continue
        elif arg.startswith("--numprocesses=") or arg.startswith("--dist="):
            continue
        else:
            result.append(arg)
    return result


def _pytest_invocations(
    selection: Selection,
    base: str,
    tier: str,
    extra_args: Sequence[str],
    *,
    kept: Sequence[str] | None = None,
) -> list[list[str]]:
    markers = ["-m", _changed_markers(tier, Path(__file__).resolve().parent.parent)]
    filter_plugin = ["-p", "scripts.affected_filter"]

    if selection.full_tier:
        roots = ["tests"] if selection.doc_pages else []
        return [markers + filter_plugin + roots + list(selection.doc_pages) + list(extra_args)]

    if selection.use_tach:
        if kept == () and not selection.test_files and not selection.doc_pages:
            return []
        return [
            markers
            + [
                "--tach-base",
                base,
                "--affected-tier",
                tier,
                "--affected-report",
                selection.affected_report,
                "-p",
                "scripts.affected_filter",
                *extra_args,
                "tests",
                *selection.doc_pages,
            ]
        ]
    if selection.test_files:
        return [
            markers
            + [
                "-p",
                "scripts.affected_filter",
                *(
                    _without_xdist_worker_args(extra_args)
                    if len(selection.test_files) <= 6
                    else extra_args
                ),
                "tests",
                *selection.doc_pages,
            ]
        ]

    direct_args = markers + filter_plugin + list(selection.doc_pages)
    direct_args.extend(extra_args)
    return [direct_args]


def _tach_head(arguments: Sequence[str]) -> str | None:
    head = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--tach-head":
            head = arguments[index + 1] if index + 1 < len(arguments) else None
            index += 2
            continue
        if argument.startswith("--tach-head="):
            head = argument.partition("=")[2]
        index += 1
    return head


def _pytest_arguments(
    selection: Selection, base: str, tier: str, extra_args: Sequence[str]
) -> list[str]:
    return _pytest_invocations(selection, base, tier, extra_args)[0]


def _combined_pytest_status(statuses: Sequence[int]) -> int | None:
    if not statuses:
        return None
    if any(status == 6 for status in statuses):
        return 5
    meaningful = [status for status in statuses if status != 5]
    return next((status for status in meaningful if status != 0), 0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tier", choices=("fast", "unit", "slow"), default="fast")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    extra_args = arguments.pytest_args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    root = Path.cwd()
    try:
        selection = select(arguments.base, root, tier=arguments.tier)
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

    kept = None
    if selection.use_tach:
        from tests._selection import kept_test_files

        kept = kept_test_files(
            root, arguments.base, tier=arguments.tier, head=_tach_head(extra_args)
        )

    statuses = []
    for pytest_args in _pytest_invocations(
        selection, arguments.base, arguments.tier, extra_args, kept=kept
    ):
        command = [sys.executable, "-m", "pytest", *pytest_args]
        environment = os.environ.copy()
        environment["AFFECTED_TEST_TIER"] = arguments.tier
        selected_files = {*(kept or ()), *selection.test_files}
        if selection.full_tier or (selection.use_tach and kept is None):
            environment.pop("AFFECTED_TEST_FILES", None)
        elif selected_files:
            environment["AFFECTED_TEST_FILES"] = json.dumps(sorted(selected_files))
        else:
            environment["AFFECTED_TEST_FILES"] = "[]"
        statuses.append(subprocess.run(command, cwd=root, env=environment, check=False).returncode)
    status = _combined_pytest_status(statuses)
    if status is None:
        print("affected: no affected tests selected")
        return 0
    if status == 0 and statuses and all(item == 5 for item in statuses):
        print("affected: no affected tests selected")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
