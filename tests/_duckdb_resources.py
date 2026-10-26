"""Apply the resource limits used by test DuckDB connections."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_DUCKDB_SPILL_ROOT: str | None = None


def bound_duckdb_resources() -> None:
    """Bound test DuckDB connections and give each one an isolated spill path."""
    from ibis.backends.duckdb import Backend

    original_connect = Backend.do_connect

    def bounded_connect(self, *args, **kwargs):
        global _DUCKDB_SPILL_ROOT
        if kwargs.setdefault("threads", 1) is None:
            del kwargs["threads"]
        if "memory_limit" in kwargs:
            if kwargs["memory_limit"] is None:
                del kwargs["memory_limit"]
        else:
            kwargs["memory_limit"] = os.environ.get("INCREMENT_TEST_DUCKDB_MEMORY", "1GB")
        if "temp_directory" in kwargs:
            if kwargs["temp_directory"] is None:
                del kwargs["temp_directory"]
        else:
            if _DUCKDB_SPILL_ROOT is None:
                _DUCKDB_SPILL_ROOT = tempfile.mkdtemp(prefix="increment-duckdb-spill-")
                atexit.register(shutil.rmtree, _DUCKDB_SPILL_ROOT, ignore_errors=True)
            kwargs["temp_directory"] = tempfile.mkdtemp(
                prefix="connection-", dir=_DUCKDB_SPILL_ROOT
            )
        return original_connect(self, *args, **kwargs)

    Backend.do_connect = bounded_connect
