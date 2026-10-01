"""Local fixture readiness for the bundled dashboard example.

Example-only infrastructure. The packaged `increment.dashboard` helpers never
import this module: they take a caller-owned `Analysis` and never look at the
working directory, a manifest, or a warehouse path.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

PARTITIONS = 2
USERS_PER_PARTITION = 2000
SEED = 42

DEFINITIONS = Path("examples/realistic_demo/definitions")
WAREHOUSE = Path("examples/realistic_demo/warehouse")
GENERATOR = Path("examples/realistic_demo/generate.py")


def ensure_demo_warehouse(repo_root: Path) -> dict[str, Any]:
    """Generate the deterministic demo warehouse when missing or stale.

    The warehouse is generated, never committed. Its definitions reference
    repository-relative parquet paths, so only this example needs the
    repository root.
    """
    root = Path(repo_root)
    generator_path = root / GENERATOR
    if not generator_path.is_file():
        raise FileNotFoundError(
            f"{generator_path} not found. Launch this notebook from the repository root, "
            "for example: uv run --extra dashboard --extra demo marimo edit "
            "examples/ab_testing_dashboard.py"
        )
    generate = _load_generator(generator_path)
    warehouse = root / WAREHOUSE
    manifest_path = warehouse / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None and manifest.get("generator_version") == generate.GENERATOR_VERSION:
        return manifest

    warehouse.mkdir(parents=True, exist_ok=True)
    row_counts = generate.generate(
        warehouse,
        partitions=PARTITIONS,
        users_per_partition=USERS_PER_PARTITION,
        seed=SEED,
    )
    manifest = generate.warehouse_manifest(
        partitions=PARTITIONS,
        users_per_partition=USERS_PER_PARTITION,
        seed=SEED,
        row_counts=row_counts,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def demo_provenance(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Provenance labels for the generated fixture, passed as configuration."""
    return {
        "Definitions": DEFINITIONS.as_posix(),
        "Generator version": str(manifest.get("generator_version", "unknown")),
        "Seed": str(manifest.get("seed", "unknown")),
        "Generated units": str(
            manifest.get("partitions", 0) * manifest.get("users_per_partition", 0)
        ),
    }


def _load_generator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_dashboard_demo_generate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the demo generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
