#!/usr/bin/env python3
"""Report source files importing declarations from the estimation engine."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    importers: list[str] = []
    for path in sorted((ROOT / "increment").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.ImportFrom) and node.module == "increment.estimation.engine"
            for node in ast.walk(tree)
        ):
            importers.append(path.relative_to(ROOT).as_posix())
    print(f"{len(importers)} importers of increment.estimation.engine")
    for importer in importers:
        print(importer)


if __name__ == "__main__":
    main()
