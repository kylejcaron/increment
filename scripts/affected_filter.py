"""Filter pytest collection to files selected by the affected-test runner."""

from __future__ import annotations

import json
import os
from typing import Any
from pathlib import Path

_SELECTED_FILES = os.environ.get("AFFECTED_TEST_FILES")
_KEPT = (
    None
    if _SELECTED_FILES is None
    else {Path(path).as_posix() for path in json.loads(_SELECTED_FILES)}
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
    if (
        exitstatus == 0
        and not getattr(session, "items", ())
        and getattr(getattr(session, "config", None), "option", None)
        and getattr(session.config.option, "deselect", ())
    ):
        session.exitstatus = 5
