"""Coordinate heavy test commands with advisory locks shared by worktrees."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TextIO

fcntl: ModuleType | None
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
else:
    fcntl = _fcntl

_HELD_FILE: TextIO | None = None


_COMMAND: list[str] | None = None


def _worker_allowance(cpu_count: int, held_slots: int, slot_count: int) -> int:
    """Size workers against both the configured lane count and current load."""
    return max(1, cpu_count // slot_count, cpu_count // held_slots)


def _held_slots(directory: Path) -> int:
    """Count held advisory locks without changing their ownership."""
    if fcntl is None or not directory.exists():
        return 0
    held = 0
    for index in range(_slot_count()):
        stream = (directory / f"slot-{index}.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held += 1
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
    return held


def _workers() -> int:
    """Return lane allowance; caller must hold a slot or is counted as one."""
    held_slots = _held_slots(_slot_directory())
    if "TEST_SLOT_HELD" not in os.environ:
        held_slots += 1
    held_slots = max(1, held_slots)
    return _worker_allowance(os.cpu_count() or 1, held_slots, _slot_count())


def _slot_directory() -> Path:
    configured = os.environ.get("TEST_SLOT_DIR")
    if configured:
        return Path(configured).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache).expanduser() if cache else Path.home() / ".cache") / "test-slots"


def _slot_count() -> int:
    return int(os.environ.get("TEST_SLOTS", "3"))


def _metadata() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "command": _COMMAND if _COMMAND is not None else sys.argv,
        "started_at": datetime.now(UTC).isoformat(),
    }


def _try_acquire(directory: Path) -> TextIO | None:
    if fcntl is None:
        return None
    for index in range(_slot_count()):
        stream = (directory / f"slot-{index}.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            stream.close()
            continue
        stream.seek(0)
        stream.truncate()
        json.dump(_metadata(), stream)
        stream.write("\n")
        stream.flush()
        os.set_inheritable(stream.fileno(), True)
        return stream
    return None


def _holders(directory: Path) -> list[str]:
    holders = []
    for index in range(_slot_count()):
        path = directory / f"slot-{index}.lock"
        for attempt in range(10):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                holders.append(
                    f"pid={data['pid']} cwd={data['cwd']} command={data['command']} "
                    f"started_at={data['started_at']}"
                )
                break
            except (OSError, ValueError, KeyError):
                if attempt < 9:
                    time.sleep(0.01)
    return holders


def hold_slot() -> None:
    """Acquire one slot for this process, unless coordination is bypassed."""
    global _HELD_FILE
    if _HELD_FILE is not None or "CI" in os.environ or "TEST_SLOT_HELD" in os.environ:
        return
    if fcntl is None:
        return
    directory = _slot_directory()
    directory.mkdir(parents=True, exist_ok=True)
    waited = False
    while True:
        stream = _try_acquire(directory)
        if stream is not None:
            os.environ["TEST_SLOT_HELD"] = "1"
            _HELD_FILE = stream
            os.nice(10)
            return
        if not waited:
            print("waiting for a heavy test slot; held by:", file=sys.stderr, flush=True)
            for holder in _holders(directory):
                print(holder, file=sys.stderr, flush=True)
            waited = True
        time.sleep(1)


def main(argv: list[str]) -> int:
    """Acquire a slot and replace this process with the requested command."""
    if argv == ["--workers"]:
        print(_workers())
        return 0
    if not argv or argv[0] != "--" or len(argv) == 1:
        print(
            "usage: python -m scripts.test_slots --workers | -- CMD [ARGS...]",
            file=sys.stderr,
        )
        return 2
    global _COMMAND
    _COMMAND = argv[1:]
    hold_slot()
    os.environ["TEST_SLOT_HELD"] = "1"
    os.execvp(argv[1], argv[1:])
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
