"""Persistent, source-versioned cache for expensive deterministic test fixtures."""

from __future__ import annotations

import copyreg
import hashlib
import importlib.metadata
import os
import pickle
import platform
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

from filelock import FileLock


def _installed_environment() -> tuple[str, ...]:
    """Return a deterministic identity for the complete installed environment."""
    distributions = sorted(
        (
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=lambda item: (item[0].casefold(), item[1]),
    )
    return tuple(f"{name}=={version}" for name, version in distributions)


def _restore_mapping_proxy(values: dict):
    return MappingProxyType(values)


def _register_mapping_proxy() -> None:
    copyreg.pickle(
        type(MappingProxyType({})),
        lambda value: (_restore_mapping_proxy, (dict(value),)),
    )


def fixture_cache_key(root: Path, *fixture_sources: Path) -> str:
    """Hash package code, fixture code, lockfile and the installed environment."""
    digest = hashlib.sha256()
    paths = {path.resolve() for path in fixture_sources}
    paths.update(path.resolve() for path in (root / "increment").rglob("*.py"))
    paths.add((root / "uv.lock").resolve())
    paths.add(Path(__file__).resolve())
    for path in sorted(paths):
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = f"external:{path.name}"
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    digest.update(platform.python_version().encode())
    digest.update(platform.platform().encode())
    for distribution in _installed_environment():
        digest.update(distribution.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def get_or_build[T](cache_root: Path, key: str, builder: Callable[[], T]) -> T:
    """Load a cached fixture or atomically build it once across processes."""
    _register_mapping_proxy()
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{key}.pickle"
    with FileLock(str(path) + ".lock"):
        if path.exists():
            with path.open("rb") as stream:
                return pickle.load(stream)
        value = builder()
        with tempfile.NamedTemporaryFile(dir=cache_root, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)
        return value
