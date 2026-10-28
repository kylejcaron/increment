"""Filter pytest collection to files selected by the affected-test runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_SELECTED_FILES = os.environ.get("AFFECTED_TEST_FILES")
_KEPT = (
    None
    if _SELECTED_FILES is None
    else {Path(path).as_posix() for path in json.loads(_SELECTED_FILES)}
)
_DESELECTED = 0
_USER_DESELECT_PATTERNS: tuple[str, ...] = ()
_USER_KEYWORD_FILTER = False
_TIER = os.environ.get("AFFECTED_TEST_TIER", "fast")
_TIER_MARKER_EXPR = os.environ.get("AFFECTED_TEST_MARKER_EXPR", "")
_TIER_FILTER_ACTIVE = False


def pytest_configure(config: Any) -> None:
    global _USER_DESELECT_PATTERNS, _USER_KEYWORD_FILTER, _TIER_FILTER_ACTIVE
    _USER_DESELECT_PATTERNS = tuple(config.option.deselect or ())
    _USER_KEYWORD_FILTER = bool(config.option.keyword)
    _TIER_FILTER_ACTIVE = bool(_TIER_MARKER_EXPR) and config.option.markexpr == _TIER_MARKER_EXPR


def _tier_eligible(item: Any) -> bool:
    if not _TIER_FILTER_ACTIVE:
        return True
    markers = {marker.name for marker in item.iter_markers()}
    if _TIER == "slow":
        return "slow" in markers and not markers.intersection(
            {
                "parameter_recovery",
                "numerical_certification",
                "examples",
                "compile_registry",
                "benchmark",
            }
        )
    excluded = {
        "slow",
        "parameter_recovery",
        "numerical_certification",
        "examples",
        "stress",
        "benchmark",
    }
    if _TIER == "unit":
        excluded.add("integration")
    return not markers.intersection(excluded)


def pytest_deselected(items: list[Any]) -> None:
    global _DESELECTED
    _DESELECTED += sum(
        _tier_eligible(item)
        and (
            _USER_KEYWORD_FILTER
            or any(item.nodeid.startswith(pattern) for pattern in _USER_DESELECT_PATTERNS)
        )
        for item in items
    )


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    if _KEPT is None or not collection_path.is_file():
        return None

    try:
        relative = collection_path.resolve().relative_to((Path.cwd() / "tests").resolve())
    except ValueError:
        return None
    if not (relative.name.startswith("test_") or relative.name.endswith("_test.py")):
        return None

    return True if f"tests/{relative.as_posix()}" not in _KEPT else None


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    workeroutput = getattr(session.config, "workeroutput", None)
    if isinstance(workeroutput, dict):
        workeroutput["affected_user_deselected"] = _DESELECTED
    if exitstatus == 5 and _DESELECTED:
        session.exitstatus = 6
    elif exitstatus == 5 and _KEPT is not None:
        session.exitstatus = 0


def pytest_testnodedown(node: Any, error: Any) -> None:
    global _DESELECTED
    _DESELECTED += int(getattr(node, "workeroutput", {}).get("affected_user_deselected", 0))
