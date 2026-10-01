"""Audit Python comments and docstrings for excessive or stale prose.

The gate permits short explanatory comment blocks, checks historical and
competitor terminology only where prose can be identified structurally, and
reports source errors instead of silently skipping files.
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

# These terms describe repository process or named competitors. ``amplitude``
# is intentionally absent: scientific amplitude is valid domain terminology.
TERMINOLOGY = re.compile(
    r"(?i)\b(?:kata|roborev|superpowers|TodoWrite|subagent)\b"
    r"|\b(?:eppo|statsig|optimizely|growthbook|launchdarkly|vwo)\b|\bsplit\.io\b"
)
ALLOW_LONG = re.compile(r"^#\s*prose:\s*allow-long\s*(.*?)\s*$", re.IGNORECASE)
PRAGMA = re.compile(
    r"^#!|^#.*?coding[:=]\s*[-\w.]+"
    r"|^#\s*(?:noqa\b|type:\s*ignore\b|pyright:|mypy:|pylint:|ruff:|fmt:)",
    re.IGNORECASE,
)

DEFAULT_ROOTS = (
    "increment",
    "tests",
    "scripts",
    "calibration",
    "integration",
    "docs",
    "conftest.py",
    "noxfile.py",
    "vulture_whitelist.py",
)


def _is_standalone_comment(token: tokenize.TokenInfo, code_lines: set[int]) -> bool:
    return token.start[0] not in code_lines and not PRAGMA.match(token.string.strip())


def _comment_blocks(
    comments: list[tokenize.TokenInfo], code_lines: set[int]
) -> list[tuple[int, int]]:
    lines = sorted({t.start[0] for t in comments if _is_standalone_comment(t, code_lines)})
    blocks: list[tuple[int, int]] = []
    start = previous = 0
    for line in lines:
        if not start:
            start = previous = line
        elif line == previous + 1:
            previous = line
        else:
            blocks.append((start, previous))
            start = previous = line
    if start:
        blocks.append((start, previous))
    return blocks


def _exception_for_block(
    comments_by_line: dict[int, tokenize.TokenInfo], start: int
) -> tuple[bool, str | None]:
    token = comments_by_line.get(start)
    if token is None:
        return False, None
    match = ALLOW_LONG.match(token.string.strip())
    if match is None:
        return False, None
    reason = match.group(1).strip()
    return bool(reason), reason or "blank justification"


def _violation(kind: str, line: int, message: str) -> dict[str, Any]:
    return {"kind": kind, "line": line, "message": message}


def audit(path: Path) -> dict[str, Any]:
    """Return structured diagnostics for one Python source file."""
    key = os.path.relpath(path).replace("\\", "/")
    result: dict[str, Any] = {
        "path": key,
        "loc": 0,
        "comments": 0,
        "doc_lines": 0,
        "comment_blocks": [],
        "terminology": [],
        "errors": [],
        "violations": [],
    }
    try:
        with tokenize.open(path) as source_file:
            src = source_file.read()
    except (OSError, UnicodeError, SyntaxError) as exc:
        message = f"cannot read source: {exc}"
        result["errors"].append(message)
        result["violations"].append(_violation("error", 1, message))
        return result

    result["loc"] = len(src.splitlines())
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        message = f"cannot tokenize source: {exc}"
        result["errors"].append(message)
        result["violations"].append(_violation("error", 1, message))
        return result

    comments = [token for token in tokens if token.type == tokenize.COMMENT]
    result["comments"] = len(comments)
    code_lines: set[int] = set()
    ignored = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
    for token in tokens:
        if token.type not in ignored:
            code_lines.update(range(token.start[0], token.end[0] + 1))
    comments_by_line = {token.start[0]: token for token in comments}
    blocks = _comment_blocks(comments, code_lines)
    result["comment_blocks"] = [block for block in blocks if block[1] - block[0] + 1 > 4]
    for start, end in blocks:
        valid, reason = _exception_for_block(comments_by_line, start)
        if valid:
            continue
        if reason is not None:
            result["violations"].append(_violation("invalid-exception", start, reason))
        elif end - start + 1 > 4:
            result["violations"].append(
                _violation(
                    "comment-block", start, f"standalone comment block spans lines {start}-{end}"
                )
            )

    for token in comments:
        if PRAGMA.match(token.string.strip()):
            continue
        if TERMINOLOGY.search(token.string):
            item = _violation("terminology", token.start[0], token.string.strip()[:120])
            result["terminology"].append(item)
            result["violations"].append(item)

    try:
        tree = ast.parse(src, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        message = f"cannot parse source: {exc}"
        result["errors"].append(message)
        result["violations"].append(_violation("error", getattr(exc, "lineno", 1) or 1, message))
        return result

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        result["doc_lines"] += doc.count("\n") + 1
        for offset, line in enumerate(doc.splitlines(), start=0):
            if TERMINOLOGY.search(line):
                location = node.body[0].lineno + offset
                item = _violation("terminology", location, f"docstring: {line.strip()[:120]}")
                result["terminology"].append(item)
                result["violations"].append(item)
    return result


def _targets(arguments: list[str]) -> list[Path]:
    raw = [arg for arg in arguments if arg != "--gate"]
    if not raw:
        raw = [root for root in DEFAULT_ROOTS if Path(root).exists()]
    files: set[Path] = set()
    for target in raw:
        path = Path(target)
        if path.is_dir():
            files.update(path.rglob("*.py"))
        else:
            files.add(path)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    files = _targets(sys.argv[1:] if argv is None else argv)
    failed = False
    for path in files:
        result = audit(path)
        display_path = str(path)
        for item in result["violations"]:
            print(f"{display_path}:{item['line']}: {item['message']}")
        failed = failed or bool(result["violations"])
    if not files:
        print("audit_verbosity: no Python targets found", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
