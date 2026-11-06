from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

KINDS = ("select", "create_temp", "drop", "schema", "other")


@dataclass
class StatementCensus:
    total: int = 0
    by_kind: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))

    def record(self, sql: Any) -> None:
        text = str(sql)
        kind = _kind(text)
        self.total += 1
        self.by_kind[kind] += 1


def _kind(sql: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S).lstrip()
    while cleaned.startswith("("):
        cleaned = cleaned[1:].lstrip()
    if re.search(
        r"\binformation_schema\b|\bduckdb_columns\s*\(|^PRAGMA\s+table_info\b", cleaned, re.I
    ):
        return "schema"
    if re.match(r"^(DESCRIBE|SHOW)\b", cleaned, re.I):
        return "schema"
    if re.match(r"^(SELECT|WITH)\b", cleaned, re.I):
        return "select"
    if re.match(r"^CREATE\s+(?:(?:TEMP|TEMPORARY)\s+)?(?:TABLE|VIEW)\b", cleaned, re.I):
        return "create_temp"
    if re.match(r"^DROP\b", cleaned, re.I):
        return "drop"
    return "other"


class _ConnectionProxy:
    def __init__(self, connection: Any, census: StatementCensus):
        self._connection = connection
        self._census = census

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._connection, name)
        if name not in {"execute", "sql"} or not callable(attribute):
            return attribute

        def counted(*args: Any, **kwargs: Any) -> Any:
            if args:
                self._census.record(args[0])
            result = attribute(*args, **kwargs)
            return _ConnectionProxy(result, self._census) if result is self._connection else result

        return counted


@contextmanager
def count_statements(backend: Any) -> Iterator[StatementCensus]:
    original = backend.con
    census = StatementCensus()
    backend.con = _ConnectionProxy(original, census)
    try:
        yield census
    finally:
        backend.con = original


def _binding(backend: Any):
    from increment.semantics import load
    from tests.analysis_factory import make_binding

    return make_binding(
        backend, load("examples/definitions/"), experiment="new_onboarding_v2", store="none"
    )


def _count_journey(backend: Any, operation: Any) -> StatementCensus:
    import warnings

    from increment.errors import IncrementWarning

    with count_statements(backend) as census, warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        operation()
    return census


def _read_with_resolver(resolver: Any, operation: Any) -> Any:
    with resolver:
        return operation()


def _native_scalar(backend: Any) -> StatementCensus:
    from tests.analysis_factory import view_readout

    experiment, resolver = _binding(backend)
    return _count_journey(
        backend,
        lambda: _read_with_resolver(
            resolver, lambda: view_readout(experiment, resolver, metrics=["purchase_rate"])
        ),
    )


def _native_breakout(backend: Any) -> StatementCensus:
    from increment import View
    from tests.analysis_factory import view_readout

    experiment, resolver = _binding(backend)
    return _count_journey(
        backend,
        lambda: _read_with_resolver(
            resolver,
            lambda: view_readout(
                experiment,
                resolver,
                View(view="breakout", grain="total"),
                metrics=["purchase_rate"],
            ),
        ),
    )


def _artifact_cuped_breakout(_: Any) -> StatementCensus:
    import tempfile
    from pathlib import Path

    import ibis

    from tests.parity_harness import matrix_data
    from tests.parity_harness.views import read
    from tests.test_warehouse_cuped_parity import _close_pairs, _native_and_adopted
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=120)
    backend = ibis.duckdb.connect()

    def operation() -> None:
        backend.create_table("mixed_breakout_events", obj=mixed_cuped_event_rows(units))
        with tempfile.TemporaryDirectory() as scratch:
            definitions_path = Path(scratch) / "mixed_cuped_breakout.yaml"
            definitions_path.write_text(
                mixed_cuped_definitions_yaml("mixed_breakout_events", breakout=True)
            )
            native, adopted = _native_and_adopted(
                backend, definitions_path, "mixed_cuped_breakout_artifacts", all_extensions=True
            )
            try:
                read(adopted, "breakout")
            finally:
                _close_pairs(adopted, native)

    try:
        return _count_journey(backend, operation)
    finally:
        backend.disconnect()


def _dashboard(_: Any) -> StatementCensus:
    import tempfile
    from pathlib import Path

    from tests.dashboard_snapshots import ADDED
    from tests.test_dashboard_capture import _SAVED, _config, _workspace

    with tempfile.TemporaryDirectory() as scratch:
        with _workspace(Path(scratch), units=20, metrics=_SAVED) as (backend, experiment, resolver):
            from increment.dashboard import prepare_dashboard

            return _count_journey(
                backend,
                lambda: prepare_dashboard(experiment, resolver, config=_config(ADDED)),
            )


JOURNEYS = {
    "native_scalar": _native_scalar,
    "native_breakout": _native_breakout,
    "artifact_cuped_breakout": _artifact_cuped_breakout,
    "dashboard": _dashboard,
}
