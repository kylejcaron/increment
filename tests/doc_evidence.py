"""Static and collection-only validation for curated documentation references."""

from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class EvidenceReference:
    kind: Literal["selector", "helper", "campaign", "evidence_id"]
    raw: str
    source: str


@dataclass(frozen=True)
class ReferenceReport:
    errors: tuple[str, ...]
    pending_nodeids: tuple[str, ...]
    class_selectors: tuple[str, ...] = ()
    file_selectors: tuple[str, ...] = ()


_CODE_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_SELECTOR = re.compile(r"tests/(?:[^/\s`]+/)*test_[^/\s:`]+\.py(?:::[^\s`]+)*")
_KNOWN_REFERENCE_PREFIXES = ("tests/", "docs/", "calibration/", "scripts/")
_FILE_SUFFIXES = (
    ".py",
    ".R",
    ".r",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".ipynb",
    ".sql",
    ".rst",
)
_TYPED = {"selector", "helper", "campaign-module", "campaign-script", "evidence-id"}


def resolve_evidence_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Resolve a module-level function or a direct class method, matching compatibility evidence."""
    class_name, sep, method_name = name.partition("::")
    if sep:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in ast.iter_child_nodes(node):
                    if (
                        isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                        and child.name == method_name
                    ):
                        return child
                raise ValueError(f"class {class_name!r} has no method named {method_name!r}")
        raise ValueError(f"no class named {class_name!r}")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise ValueError(f"no top-level function named {name!r}")


def _table_cells(line: str) -> list[str] | None:
    if "|" not in line:
        return None
    cells: list[str] = []
    current: list[str] = []
    ticks = 0
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            current.extend((char, line[index + 1]))
            index += 2
            continue
        if char == "`":
            end = index + 1
            while end < len(line) and line[end] == "`":
                end += 1
            run = end - index
            if ticks == 0:
                ticks = run
            elif run == ticks:
                ticks = 0
            current.extend(line[index:end])
            index = end
            continue
        if char == "|" and ticks == 0:
            cells.append("".join(current).strip())
            current.clear()
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    if line.lstrip().startswith("|"):
        cells = cells[1:]
    if line.rstrip().endswith("|"):
        cells = cells[:-1]
    return cells


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _cell_index(line: str, offset: int) -> int:
    prefix = line[:offset]
    delimiters = 0
    ticks = 0
    index = 0
    while index < len(prefix):
        char = prefix[index]
        if char == "\\" and index + 1 < len(prefix):
            index += 2
            continue
        if char == "`":
            end = index + 1
            while end < len(prefix) and prefix[end] == "`":
                end += 1
            run = end - index
            if ticks == 0:
                ticks = run
            elif run == ticks:
                ticks = 0
            index = end
            continue
        if char == "|" and ticks == 0:
            delimiters += 1
        index += 1
    return max(0, delimiters - int(line.lstrip().startswith("|")))


def extract_references(markdown: str, *, source: str) -> tuple[EvidenceReference, ...]:
    refs: list[EvidenceReference] = []
    lines = markdown.splitlines()
    active_header: list[str] | None = None
    for line_index, line in enumerate(lines):
        line_number = line_index + 1
        cells = _table_cells(line)
        if (
            cells
            and line_index + 1 < len(lines)
            and _is_table_separator(lines[line_index + 1])
            and any("test selector" in cell.lower() or "evidence" in cell.lower() for cell in cells)
        ):
            active_header = cells
            continue
        if _is_table_separator(line):
            continue
        selector_cells: set[int] = set()
        if cells and active_header and len(cells) == len(active_header):
            selector_cells = {
                index
                for index, header in enumerate(active_header)
                if "test selector" in header.lower() or "evidence" in header.lower()
            }
        prior_by_cell: dict[int, str] = {}
        for match in _CODE_SPAN.finditer(line):
            text = match.group(1).strip()
            location = f"{source}:{line_number}"
            kind: str | None = None
            value = text
            prefix, sep, rest = text.partition(":")
            if sep and prefix in _TYPED:
                value = rest.strip()
                if not value:
                    raise ValueError(f"{location}: empty {prefix} reference")
                if prefix == "selector":
                    kind = "selector"
                elif prefix == "helper":
                    kind = "helper"
                elif prefix.startswith("campaign-"):
                    kind, value = "campaign", text
                else:
                    kind = "evidence_id"
            elif text.startswith("::"):
                cell_index = _cell_index(line, match.start())
                if cell_index not in selector_cells or cell_index not in prior_by_cell:
                    raise ValueError(
                        f"{location}: selector shorthand is only valid after a selector in the same evidence cell"
                    )
                value = prior_by_cell[cell_index].split("::", 1)[0] + text
                kind = "selector"
            elif _SELECTOR.fullmatch(text):
                kind = "selector"
            elif not any(char.isspace() for char in text) and (
                text.startswith(_KNOWN_REFERENCE_PREFIXES) or text.endswith(_FILE_SUFFIXES)
            ):
                raise ValueError(
                    f"{location}: ambiguous untyped reference {text!r}; use helper:, campaign-module:, or campaign-script:"
                )
            if kind:
                refs.append(EvidenceReference(kind, value, location))
                if kind == "selector" and cells:
                    cell_index = _cell_index(line, match.start())
                    if cell_index in selector_cells:
                        prior_by_cell[cell_index] = value
        if not cells:
            active_header = None
    return tuple(refs)


def _has_inherited_method(tree: ast.AST, class_name: str, method_name: str) -> bool:
    classes = {
        node.name: node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef)
    }
    child = classes.get(class_name)
    if child is None:
        return False

    def has_method(name: str, visited: set[str]) -> bool:
        if name in visited:
            return False
        visited.add(name)
        base = classes.get(name)
        if base is None:
            # Imported bases cannot be inspected here; exact collection decides membership.
            return name != "object"
        if any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
            for node in ast.iter_child_nodes(base)
        ):
            return True
        return any(
            isinstance(parent, ast.Attribute)
            or (isinstance(parent, ast.Name) and has_method(parent.id, visited))
            for parent in base.bases
        )

    return any(
        isinstance(parent, ast.Attribute)
        or (isinstance(parent, ast.Name) and has_method(parent.id, set()))
        for parent in child.bases
    )


def _selector_parts(raw: str) -> tuple[str, str]:
    path, _, test = raw.partition("::")
    return path, test


def _declared_evidence_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    docs = root / "docs"
    if docs.is_dir():
        for path in docs.rglob("*.md"):
            for match in re.finditer(r"^\|\s*`(EV-[A-Z0-9-]+)`\s*\|", path.read_text(), re.M):
                ids.add(match.group(1))
    return ids


def _campaign_target(root: Path, raw: str) -> Path:
    prefix, _, payload = raw.partition(":")
    try:
        parts = shlex.split(payload)
    except ValueError as exc:
        raise ValueError(f"malformed campaign entrypoint: {exc}") from exc
    if not parts:
        raise ValueError("campaign reference has no entrypoint")
    entry = parts[0]
    if prefix == "campaign-module":
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", entry):
            raise ValueError(f"invalid campaign module {entry!r}")
        relative = Path(*entry.split("."))
        candidates = (root / relative.with_suffix(".py"), root / relative / "__init__.py")
    else:
        target = Path(entry)
        if target.is_absolute() or ".." in target.parts:
            raise ValueError("campaign script must be repository-relative")
        candidates = (root / target,)
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def check_static(
    root: Path,
    refs: Sequence[EvidenceReference],
    *,
    evidence_ids: AbstractSet[str] | None = None,
) -> ReferenceReport:
    root = root.resolve()
    declared_ids = set(evidence_ids) if evidence_ids is not None else _declared_evidence_ids(root)
    errors: list[str] = []
    pending: set[str] = set()
    class_selectors: set[str] = set()
    file_selectors: set[str] = set()
    for ref in refs:
        try:
            if ref.kind == "evidence_id":
                if ref.raw not in declared_ids:
                    errors.append(f"{ref.source}: unknown evidence id {ref.raw!r}")
            elif ref.kind == "helper":
                helper = Path(ref.raw)
                if helper.is_absolute() or ".." in helper.parts or not (root / helper).is_file():
                    errors.append(f"{ref.source}: helper path does not exist: {ref.raw}")
            elif ref.kind == "campaign":
                target = _campaign_target(root, ref.raw)
                if not target.is_file():
                    errors.append(f"{ref.source}: campaign entrypoint does not exist: {ref.raw}")
            else:
                path_text, test = _selector_parts(ref.raw)
                path = root / path_text
                if not path.is_file():
                    errors.append(f"{ref.source}: selector file does not exist: {path_text}")
                    continue
                if not test:
                    pending.add(ref.raw)
                    file_selectors.add(ref.raw)
                    continue
                try:
                    tree = ast.parse(path.read_text(), filename=str(path))
                    base_test = test.split("[", 1)[0]
                    resolve_evidence_function(tree, base_test)
                except ValueError as exc:
                    class_name, sep, method_name = base_test.partition("::")
                    if not sep and any(
                        isinstance(node, ast.ClassDef) and node.name == class_name
                        for node in ast.iter_child_nodes(tree)
                    ):
                        pending.add(ref.raw)
                        class_selectors.add(ref.raw)
                    elif sep and _has_inherited_method(tree, class_name, method_name):
                        pending.add(ref.raw)
                    else:
                        errors.append(f"{ref.source}: {path_text}::{exc}")
                else:
                    pending.add(ref.raw)
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"{ref.source}: {exc}")
    return ReferenceReport(
        tuple(errors),
        tuple(sorted(pending)),
        tuple(sorted(class_selectors)),
        tuple(sorted(file_selectors)),
    )


def selector_files(refs: Sequence[EvidenceReference]) -> tuple[Path, ...]:
    """Return only deduplicated repository-relative selector files for collection."""
    return tuple(
        sorted({Path(ref.raw.partition("::")[0]) for ref in refs if ref.kind == "selector"})
    )


def collect_nodeids(root: Path, files: Sequence[Path | str]) -> frozenset[str]:
    unique = sorted({Path(file).as_posix() for file in files})
    if not unique:
        return frozenset()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:testmon",
        "-p",
        "no:tach",
        "-m",
        "",
        "--collect-only",
        "-q",
        *unique,
    ]
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"pytest collection failed ({result.returncode}):\n{result.stdout}{result.stderr}"
        )
    nodeids = frozenset(
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    )
    return nodeids


def check_collected(report: ReferenceReport, nodeids: AbstractSet[str]) -> tuple[str, ...]:
    errors: list[str] = []
    for selector in report.pending_nodeids:
        if selector in report.class_selectors:
            prefix = f"{selector}::"
            if not any(nodeid.startswith(prefix) for nodeid in nodeids):
                errors.append(f"class selector collected no descendant tests: {selector}")
            continue
        if selector in report.file_selectors:
            prefix = f"{selector}::"
            if not any(nodeid.startswith(prefix) for nodeid in nodeids):
                errors.append(f"file selector collected no descendant tests: {selector}")
            continue
        if selector in nodeids:
            continue
        if "[" not in selector and any(nodeid.startswith(f"{selector}[") for nodeid in nodeids):
            continue
        errors.append(f"selector did not collect exact node id: {selector}")
    return tuple(errors)
