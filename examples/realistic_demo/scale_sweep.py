"""Scaling sweep over the REAL realistic_demo normalized warehouse.

Regenerates ``examples/realistic_demo/warehouse`` at a series of partition
counts and times the full 5-metric experiment ``Analysis`` (materialized)
against the real generator (``examples/realistic_demo/generate.py``) and the
real Hive-partitioned Parquet layout it writes -- not a simplified fixture.

An earlier investigation on a smaller synthetic single-file-per-table
fixture found that querying a normalized warehouse (dimension joins pushed
into SQL) costs only ~1.06-1.24x versus a pre-joined "mart" table, with no
upward trend across 16x of data, and that calling ``Analysis.materialize()``
before ``.run()`` is the fastest option at every scale measured. This module
exists to confirm (or refute) that finding against the real generator and
real partitioned warehouse this repo ships.

Standalone CLI:

    uv run python examples/realistic_demo/scale_sweep.py --scales 2 4 8

Importable:

    from examples.realistic_demo.scale_sweep import sweep
    df = sweep([2, 4, 8], users_per_partition=2000)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import ibis
import pyarrow.parquet as pq

from increment import Analysis

if TYPE_CHECKING:
    import pandas as pd

_REALISTIC_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _REALISTIC_DEMO_DIR.parent.parent
_DEFINITIONS = _REALISTIC_DEMO_DIR / "definitions"
_WAREHOUSE = _REALISTIC_DEMO_DIR / "warehouse"
_MANIFEST = _WAREHOUSE / "manifest.json"
_EXPERIMENT = "checkout_redesign"
_DEFAULT_SEED = 42


def _load_generate_module():
    """Import ``examples/realistic_demo/generate.py`` by path.

    ``examples/`` and ``examples/realistic_demo/`` ship without
    ``__init__.py`` (plain namespace packages), so a plain
    ``import examples.realistic_demo.generate`` only works when the repo
    root is on ``sys.path`` -- true for the acceptance check and the CLI
    entry point below, but not guaranteed for every caller. Loading by
    explicit file path sidesteps that entirely.
    """
    spec = importlib.util.spec_from_file_location(
        "_scale_sweep_generate", _REALISTIC_DEMO_DIR / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generate = _load_generate_module()


def _read_manifest() -> dict | None:
    if not _MANIFEST.is_file():
        return None
    return json.loads(_MANIFEST.read_text())


def _warehouse_bytes() -> int:
    return sum(p.stat().st_size for p in _WAREHOUSE.rglob("*.parquet"))


def _regenerate(*, partitions: int, users_per_partition: int, seed: int) -> dict[str, int]:
    """Regenerate the warehouse in-process (no subprocess) and write the
    manifest, mirroring what ``generate.main`` does on the CLI path."""
    _WAREHOUSE.mkdir(parents=True, exist_ok=True)
    row_counts = _generate.generate(_WAREHOUSE, partitions, users_per_partition, seed)
    manifest = _generate.warehouse_manifest(
        partitions=partitions,
        users_per_partition=users_per_partition,
        seed=seed,
        row_counts=row_counts,
    )
    _MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return row_counts


def sweep(
    partition_counts: list[int],
    users_per_partition: int = 2000,
) -> pd.DataFrame:
    """Regenerate the warehouse at each partition count, run the full
    5-metric experiment analysis (materialized) timed, and return a
    DataFrame with columns:
      partitions        -- int, the partition count for this row
      events_scanned    -- int, total events + orders rows in the warehouse
      warehouse_bytes    -- int, total size on disk of all Parquet files
      wall_clock_secs    -- float, Analysis.materialize() + .run() time
      moments_rows       -- int, row count of the exported moments cube
      moments_bytes       -- int, size on disk of the exported moments cube
    One row per requested partition count, in the order given. Restores the
    warehouse to its original size (re-running generate.py at whatever
    partition/user count it had before this function was called) when done,
    so calling this doesn't leave the demo warehouse at some arbitrary sweep
    size for whoever runs the notebook next.
    """
    import pandas as pd

    original_manifest = _read_manifest()
    original_cwd = Path.cwd()

    rows: list[dict] = []
    try:
        # The shipped fact_sources.yaml reads Parquet via CWD-RELATIVE SQL
        # (read_parquet('examples/realistic_demo/warehouse/...')), while
        # regen/measurement here use absolute paths anchored to __file__.
        # Those only coincide if cwd is the repo root, so enforce it for the
        # duration -- otherwise a caller with a different cwd (e.g. a
        # notebook opened from elsewhere) would regenerate the real
        # warehouse but silently query stale or missing Parquet.
        os.chdir(_REPO_ROOT)
        for partitions in partition_counts:
            row_counts = _regenerate(
                partitions=partitions,
                users_per_partition=users_per_partition,
                seed=_DEFAULT_SEED,
            )
            events_scanned = row_counts["events"] + row_counts["fact_orders"]
            warehouse_bytes = _warehouse_bytes()

            con = ibis.duckdb.connect()
            try:
                analysis = Analysis(_EXPERIMENT, _DEFINITIONS, con)

                start = time.perf_counter()
                analysis.materialize()
                analysis.run()
                wall_clock_secs = time.perf_counter() - start

                export_dir = _WAREHOUSE / ".scale_sweep_tmp"
                export_dir.mkdir(parents=True, exist_ok=True)
                export_path = export_dir / f"moments_{partitions:05d}.parquet"
                analysis.export(export_path)

                moments_rows = pq.read_metadata(export_path).num_rows
                moments_bytes = export_path.stat().st_size
                export_path.unlink()
            finally:
                con.disconnect()

            rows.append(
                {
                    "partitions": partitions,
                    "events_scanned": events_scanned,
                    "warehouse_bytes": warehouse_bytes,
                    "wall_clock_secs": wall_clock_secs,
                    "moments_rows": moments_rows,
                    "moments_bytes": moments_bytes,
                }
            )
    finally:
        # cwd restore must not be skippable by _restore() raising (it calls
        # generate() again, a second fallible call) -- reorder so the cwd
        # guarantee is unconditional. generate() only ever writes via the
        # absolute _WAREHOUSE path, so restoring cwd first doesn't affect it.
        os.chdir(original_cwd)
        _restore(original_manifest)

    return pd.DataFrame(
        rows,
        columns=[
            "partitions",
            "events_scanned",
            "warehouse_bytes",
            "wall_clock_secs",
            "moments_rows",
            "moments_bytes",
        ],
    )


def _restore(original_manifest: dict | None) -> None:
    """Restore the warehouse to whatever it was before ``sweep()`` ran.

    If a manifest existed beforehand, regenerate at exactly that
    partition/user/seed count. If no warehouse existed beforehand, remove
    whatever the sweep left behind so the pre-sweep (absent) state is
    restored exactly.
    """
    export_dir = _WAREHOUSE / ".scale_sweep_tmp"
    if export_dir.is_dir():
        shutil.rmtree(export_dir)

    if original_manifest is None:
        if _WAREHOUSE.exists():
            shutil.rmtree(_WAREHOUSE)
        return

    _regenerate(
        partitions=original_manifest["partitions"],
        users_per_partition=original_manifest["users_per_partition"],
        seed=original_manifest["seed"],
    )


def _format_finding(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "Fewer than two scales were swept -- not enough points to assess a trend."

    per_event = df["wall_clock_secs"] / df["events_scanned"]
    first_per_event, last_per_event = per_event.iloc[0], per_event.iloc[-1]
    ratio = last_per_event / first_per_event if first_per_event > 0 else float("nan")
    scale_ratio = df["events_scanned"].iloc[-1] / df["events_scanned"].iloc[0]

    if ratio <= 1.5:
        trend = (
            f"per-row wall-clock cost stayed roughly flat ({ratio:.2f}x over a "
            f"{scale_ratio:.1f}x growth in scanned rows), consistent with the "
            "sub-linear scaling measured on the smaller fixture"
        )
    else:
        trend = (
            f"per-row wall-clock cost grew {ratio:.2f}x over a {scale_ratio:.1f}x "
            "growth in scanned rows -- a super-linear trend that CONTRADICTS the "
            "earlier smaller-fixture finding"
        )

    return (
        f"Across partition counts {list(df['partitions'])}, {trend}. Wall-clock "
        f"went from {df['wall_clock_secs'].iloc[0]:.3f}s to "
        f"{df['wall_clock_secs'].iloc[-1]:.3f}s while events_scanned went from "
        f"{df['events_scanned'].iloc[0]} to {df['events_scanned'].iloc[-1]} and "
        f"warehouse_bytes went from {df['warehouse_bytes'].iloc[0]} to "
        f"{df['warehouse_bytes'].iloc[-1]}. materialize()+run() was timed as one "
        "measurement per the earlier finding that materializing before reducing "
        "is fastest at every scale."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="partition counts to sweep, in order (default: 2 4 8 16)",
    )
    parser.add_argument(
        "--users-per-partition",
        type=int,
        default=2000,
        help="users per partition at every scale (default: 2000)",
    )
    args = parser.parse_args(argv)

    df = sweep(args.scales, users_per_partition=args.users_per_partition)
    print(df.to_string(index=False))
    print()
    print(_format_finding(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
