#!/usr/bin/env python3
"""Report imports from the estimation engine and the names each module uses."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ENGINE_MODULE = "increment.estimation.engine"


def importers(root: Path | str) -> dict[str, tuple[str, ...]]:
    """Map source modules importing the engine to their imported symbol names."""
    root = Path(root)
    inventory: dict[str, tuple[str, ...]] = {}
    for path in sorted((root / "increment").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = tuple(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == _ENGINE_MODULE
            for alias in node.names
        )
        if names:
            parts = path.relative_to(root).with_suffix("").parts
            module_parts = parts[:-1] if parts[-1] == "__init__" else parts
            inventory[".".join(module_parts)] = names
    return dict(sorted(inventory.items()))


def main() -> None:
    inventory = importers(ROOT)
    print(f"{len(inventory)} importers of {_ENGINE_MODULE}")
    print("| Module | Imported names |")
    print("| --- | --- |")
    for module, names in inventory.items():
        print(f"| `{module}` | `{', '.join(names)}` |")


if __name__ == "__main__":
    main()
