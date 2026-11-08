"""Carry pytest's rewritten test bytecode between CI jobs.

pytest reuses a rewritten module while the pyc header's source mtime and size
match the file, and a fresh checkout gives every file a new mtime. ``save``
stores each rewritten pyc pytest would load for a tracked source, with the
SHA-256 of that source. ``restore`` installs a stored pyc only when its source
still has that digest, at the same absolute path, under the same interpreter
and pytest, and stamps the header with the file's current mtime and size; every
other module is rewritten from source as usual.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

from _pytest.assertion.rewrite import PYC_TAIL, get_cache_dir

_MANIFEST = "manifest.json"
# PEP 552 timestamp pyc: magic, zero flags, then source mtime and size.
_PREFIX = importlib.util.MAGIC_NUMBER + bytes(4)
_HEADER_SIZE = 16


def _identity(root: Path) -> dict[str, str]:
    """Everything besides source bytes that the rewritten code depends on."""
    return {"root": str(root), "pyc": PYC_TAIL, "magic": importlib.util.MAGIC_NUMBER.hex()}


def _pyc(source: Path) -> Path:
    return get_cache_dir(source) / (source.name.removesuffix(".py") + PYC_TAIL)


def _source_stamp(source: Path) -> bytes:
    stat = source.stat()
    return struct.pack("<LL", int(stat.st_mtime) & 0xFFFFFFFF, stat.st_size & 0xFFFFFFFF)


def _digest(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _tracked_sources(root: Path) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"], cwd=root, check=True, capture_output=True
    ).stdout
    return [name for name in os.fsdecode(listed).split("\0") if name]


def save(root: Path, cache: Path) -> int:
    """Replace ``cache`` with every pyc pytest would load for the current tracked sources."""
    root = root.resolve()
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    digests: dict[str, str] = {}
    for name in _tracked_sources(root):
        source = root / name
        try:
            data = _pyc(source).read_bytes()
            current = data[:8] == _PREFIX and data[8:_HEADER_SIZE] == _source_stamp(source)
        except OSError:
            continue
        if current:
            staged = cache / Path(name).with_suffix(".pyc")
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            digests[name] = _digest(source)
    manifest = {"identity": _identity(root), "sources": digests}
    (cache / _MANIFEST).write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return len(digests)


def restore(root: Path, cache: Path) -> tuple[int, int]:
    """Install stored pycs whose sources are unchanged; return (installed, stored)."""
    root = root.resolve()
    try:
        manifest = json.loads((cache / _MANIFEST).read_text())
    except (OSError, ValueError):
        return 0, 0
    sources: dict[str, str] = manifest.get("sources", {})
    if manifest.get("identity") != _identity(root):
        return 0, len(sources)
    installed = 0
    for name, digest in sources.items():
        source = root / name
        try:
            if _digest(source) != digest:
                continue
            data = (cache / Path(name).with_suffix(".pyc")).read_bytes()
            pyc = _pyc(source)
            pyc.parent.mkdir(parents=True, exist_ok=True)
            partial = pyc.with_name(f"{pyc.name}.{os.getpid()}")
            partial.write_bytes(data[:8] + _source_stamp(source) + data[_HEADER_SIZE:])
            partial.replace(pyc)
        except OSError:
            continue
        installed += 1
    return installed, len(sources)


def main() -> None:
    """Save or restore the cache for the checkout in the working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("save", "restore"))
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    if args.action == "save":
        print(f"saved {save(Path.cwd(), args.cache)} rewritten test modules")
    else:
        installed, stored = restore(Path.cwd(), args.cache)
        print(f"restored {installed} of {stored} cached rewritten test modules")


if __name__ == "__main__":
    main()
