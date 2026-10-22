"""Run whole test files selected by pytest-testmon for local iteration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _pytest.mark import Expression, MarkMatcher


def _marker_expression(tier: str = "fast") -> str:
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "noxfile.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TIER_MARKERS" for target in node.targets
        ):
            markers = ast.literal_eval(node.value)
            return markers[tier]
    raise RuntimeError("noxfile.py does not define TIER_MARKERS")


class _WholeSelectedFiles:
    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        collected = tuple(items)
        result = yield
        selected_files = {item.path for item in items}
        if selected_files:
            mark_expression = Expression.compile(config.option.markexpr)
            items[:] = [
                item
                for item in collected
                if item.path in selected_files
                and mark_expression.evaluate(MarkMatcher.from_markers(item.iter_markers()))
            ]
        return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("fast", "unit"), default="fast")
    arguments = parser.parse_args()
    return int(
        pytest.main(
            ["--testmon", "--testmon-forceselect", "-m", _marker_expression(arguments.tier)],
            plugins=[_WholeSelectedFiles()],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
