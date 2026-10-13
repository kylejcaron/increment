"""Validated joint-cell moments for two overlapping experiments.

``OverlapCells`` is built only through :meth:`OverlapCells.from_frame` or
:meth:`OverlapCells.from_table`. Both ingresses validate the join, the
eligibility and assignment records, and exposure timing before any moment is
computed, then reduce the jointly eligible units to per-cell counts, means and
sample variances. The estimator consumes nothing else, so both ingresses yield
the same readout from the same per-unit data.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from increment._overlap.admission import Admission, CellMoments, EligibilityAudit
from increment._overlap.contrasts import CELLS, Cell
from increment._overlap.design import ExperimentColumns, OverlapDesign

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

_ADMIT = object()


class OverlapCells:
    """Admitted per-cell moments for the jointly eligible target population."""

    __slots__ = ("audit", "design", "metrics", "moments", "outcome_window")

    def __init__(
        self,
        *,
        design: OverlapDesign,
        metrics: tuple[str, ...],
        moments: tuple[CellMoments, ...],
        audit: EligibilityAudit,
        outcome_window: dt.timedelta,
        _token: object = None,
    ) -> None:
        if _token is not _ADMIT:
            raise TypeError(
                "OverlapCells is built with OverlapCells.from_frame or OverlapCells.from_table"
            )
        self.design = design
        self.metrics = metrics
        self.moments = moments
        self.audit = audit
        self.outcome_window = outcome_window

    def __repr__(self) -> str:
        counts = {cell: self.cell_count(cell) for cell in CELLS}
        return (
            f"OverlapCells(first={self.design.first.name!r}, second={self.design.second.name!r}, "
            f"metrics={self.metrics!r}, cell_counts={counts!r})"
        )

    @classmethod
    def _admit(cls, admission: Admission) -> OverlapCells:
        return cls(
            design=admission.design,
            metrics=admission.metrics,
            moments=admission.moments,
            audit=admission.audit,
            outcome_window=admission.outcome_window,
            _token=_ADMIT,
        )

    def cell_count(self, cell: Cell) -> int:
        return next(m.n for m in self.moments if m.cell == cell)

    def moment(self, metric: str, cell: Cell) -> CellMoments:
        return next(m for m in self.moments if m.metric == metric and m.cell == cell)

    @classmethod
    def from_frame(
        cls,
        frame: IntoDataFrame,
        design: OverlapDesign,
        *,
        unit: str,
        columns: Mapping[str, ExperimentColumns],
        outcome_start: str,
        outcome_end: str,
        metrics: Mapping[str, str],
    ) -> OverlapCells:
        """Admit one row per unit from a pandas, polars, or pyarrow frame.

        Parameters
        ----------
        frame : IntoDataFrame
            One row per unit with eligibility, assignment, timing, and outcomes.
        design : OverlapDesign
            The two experiments and their joint assignment law.
        unit : str
            Unit identifier column.
        columns : Mapping[str, ExperimentColumns]
            Columns for each declared experiment, keyed by experiment name.
        outcome_start : str
            Column holding when each unit's outcome window opens.
        outcome_end : str
            Column holding when each unit's outcome window closes.
        metrics : Mapping[str, str]
            Metric name to numeric per-unit outcome column; each is a mean metric.

        Returns
        -------
        OverlapCells
            Validated per-cell moments of the jointly eligible units.
        """
        from increment._overlap.frame import admit_frame

        return cls._admit(
            admit_frame(
                frame,
                design,
                unit=unit,
                columns=columns,
                outcome_start=outcome_start,
                outcome_end=outcome_end,
                metrics=metrics,
            )
        )

    @classmethod
    def from_table(
        cls,
        table: Any,
        design: OverlapDesign,
        *,
        unit: str,
        columns: Mapping[str, ExperimentColumns],
        outcome_start: str,
        outcome_end: str,
        metrics: Mapping[str, str],
    ) -> OverlapCells:
        """Admit one row per unit from an Ibis table, validating and aggregating in the warehouse.

        Parameters
        ----------
        table : Any
            An Ibis table with one row per unit with eligibility, assignment, timing, and outcomes.
        design : OverlapDesign
            The two experiments and their joint assignment law.
        unit : str
            Unit identifier column.
        columns : Mapping[str, ExperimentColumns]
            Columns for each declared experiment, keyed by experiment name.
        outcome_start : str
            Column holding when each unit's outcome window opens.
        outcome_end : str
            Column holding when each unit's outcome window closes.
        metrics : Mapping[str, str]
            Metric name to numeric per-unit outcome column; each is a mean metric.

        Returns
        -------
        OverlapCells
            Validated per-cell moments of the jointly eligible units.
        """
        from increment.query.overlap import admit_table

        return cls._admit(
            admit_table(
                table,
                design,
                unit=unit,
                columns=columns,
                outcome_start=outcome_start,
                outcome_end=outcome_end,
                metrics=metrics,
            )
        )
