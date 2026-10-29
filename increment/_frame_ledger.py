"""Assignment ledger adapter for native dataframe inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from increment._assignment_ledger_contract import ReducedAssignment, reduce_rows, validate_roles
from increment._assignment_refusals import refuse_assignment_invalid


class FrameLedger:
    """Validate and reduce identity-bearing assignment rows from a dataframe."""

    __slots__ = ("_frame", "roles")

    def __init__(self, frame: Any, roles: Mapping[str, str]) -> None:
        self.roles = validate_roles(roles)
        if isinstance(frame, (pa.Table, pa.RecordBatch)):
            names = frame.column_names
        else:
            names = getattr(frame, "columns", ())
        columns = set(names)
        missing = set(self.roles.values()) - columns
        if missing:
            refuse_assignment_invalid("frame_columns_missing", details=sorted(missing))
        self._frame = frame

    def reduce(
        self,
        *,
        experiment: str,
        population: str,
        version: int | None = None,
        period: str | None = None,
        all_periods: bool = False,
        key: str | None = None,
    ) -> ReducedAssignment:
        """Return selected assignment rows and deterministic arm counts."""
        to_pylist = getattr(self._frame, "to_pylist", None)
        if callable(to_pylist):
            records = to_pylist()
        else:
            import narwhals as nw

            records = nw.from_native(self._frame).iter_rows(named=True)
        return reduce_rows(
            records,
            self.roles,
            experiment=experiment,
            population=population,
            version=version,
            period=period,
            all_periods=all_periods,
            key=key,
        )

    def exposure(
        self,
        *,
        experiment: str,
        version: int,
        period_order: tuple[str, ...],
        key: str | None = None,
    ) -> tuple[tuple[str, str, str, str | None], ...]:
        """Return a whole-ledger period join and next-period arm lookup."""
        from increment._assignment_ledger_contract import exposure_rows

        reduced = self.reduce(
            experiment=experiment,
            population="assigned",
            version=version,
            all_periods=True,
            key=key,
        )
        return exposure_rows(reduced, period_order=period_order, key=key)
