"""Dashboard reads from one pinned copy of a bound source.

A capture borrows the caller's binding with the pinned copy as its source, so
every read inside it describes the same data and none reopens the live source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from increment._analysis_config import select_metrics
from increment._binding import _UNDECLARED_BREAKOUT, SourceBinding
from increment._breakout_readouts import run_breakout, run_breakout_rows
from increment._day_axis import require_arm_state, run_asof, run_asof_lift, run_daily
from increment._evidence_types import EvidenceScope
from increment._source_admission import _require_analysis_operation
from increment._source_operations import DashboardGroupData, DashboardGroupDataOperation
from increment._source_types import declared_evidence
from increment.decision import DefinitionsArmAnalysisState
from increment.errors import refuse
from increment.evidence import evidence_provider

if TYPE_CHECKING:
    from increment._health import SrmObservations
    from increment.breakout.estimates import (
        BreakoutEstimates,
        DailyLiftEstimates,
        DailyMetricValues,
    )
    from increment.semantics.models import Breakout, Metric
    from increment.sequential_state import SequentialCheckpoint

__all__ = [
    "DashboardBreakoutReads",
    "DashboardCapture",
    "capture_dashboard",
    "dashboard_breakout_reads",
    "dashboard_group_data",
]


@dataclass(frozen=True, slots=True)
class DashboardCapture:
    """One pinned read: the borrowed binding and the allocation checks answered within it.

    Health checks reuse ``observations`` instead of asking the source again, so the
    allocation verdict a dashboard shows and the one its health report shows agree.
    """

    binding: SourceBinding
    observations: SrmObservations = field(default_factory=dict)


def capture_dashboard[T](
    binding: SourceBinding,
    operation: Callable[[DashboardCapture], T],
    *,
    metrics: Sequence[Metric],
) -> T:
    """Run *operation* against one isolated, pinned copy of *binding*'s source.

    *metrics* names every metric the operation reads, so the pin covers exactly their
    fact sources; include any saved metric the operation adds through
    ``exploratory_metrics``. An added metric outside the pin is refused rather than read
    from a later state. The caller's binding and its connection are left untouched.
    """
    require_arm_state(binding, "dashboard_snapshot")
    _require_analysis_operation(
        binding.source,
        "readout_snapshot",
        message=(
            "prepare_dashboard() needs an experiment built by from_definitions "
            "with a warehouse connection."
        ),
    )
    scoped = evidence_provider(binding.source).scope(
        EvidenceScope(metrics=tuple(metrics), snapshot=True, include_breakouts=True)
    )
    with scoped.source as pinned:
        from increment.readouts._evidence import _builtin_source

        return operation(DashboardCapture(binding.with_source(_builtin_source(pinned))))


@dataclass(frozen=True, slots=True)
class DashboardBreakoutReads:
    """The Explore reads of one declared breakout, and nothing else.

    Every read hands the source this breakout alone, so another source of the same property,
    or its refusal, never enters it. Families are computed per breakout, so the rows equal
    this breakout's rows from the dimension-wide reads.
    """

    breakout: Breakout
    binding: SourceBinding = field(repr=False)

    def run_asof_lift(
        self,
        *,
        metrics: Sequence[str],
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> DailyLiftEstimates:
        return run_asof_lift(
            self.binding,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            completed_windows_only=completed_windows_only,
            dimension=self.breakout.property,
            population=population,
        )

    def run_asof(
        self,
        *,
        metrics: Sequence[str],
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> DailyMetricValues:
        return run_asof(
            self.binding,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            completed_windows_only=completed_windows_only,
            dimension=self.breakout.property,
            population=population,
        )

    def run_daily(
        self,
        *,
        metrics: Sequence[str],
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> DailyMetricValues:
        return run_daily(
            self.binding,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            dimension=self.breakout.property,
            population=population,
        )

    def run_breakout(
        self,
        *,
        metrics: Sequence[str],
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> BreakoutEstimates:
        return run_breakout(
            self.binding,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            population=population,
        )

    def uncorrected_segments(
        self,
        *,
        metrics: Sequence[str],
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> BreakoutEstimates:
        """This breakout's segment rows without the declared correction."""
        return run_breakout_rows(
            self.binding,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            correction="none",
            population=population,
        )


def dashboard_breakout_reads(binding: SourceBinding, breakout: Breakout) -> DashboardBreakoutReads:
    """The dashboard's Explore reads through one declared breakout alone.

    A dimension-wide read covers every declared breakout of that property, so one source's
    refusal would refuse its siblings. These reads hand the source only *breakout*. Only
    definitions-backed sources support it: other arm-family sources refuse with
    ``evidence.breakout_unavailable`` and switchback evidence with
    ``facade.analysis.contrast_unavailable``. An undeclared *breakout* is refused with
    ``facade.analysis.undeclared_breakout``.
    """
    state = require_arm_state(binding, "dashboard_breakout_reads").state
    if not isinstance(state, DefinitionsArmAnalysisState):
        from increment._evidence_refusals import refuse_breakout

        refuse_breakout(
            binding.source,
            (breakout.property,),
            tuple(declared_evidence(binding.source).breakouts),
            route="dashboard_breakout_reads() requires a definitions-backed breakout source",
        )
    declared = state.experiment.breakouts
    if breakout not in declared:
        refuse(
            _UNDECLARED_BREAKOUT,
            requested=(breakout.source, breakout.property),
            declared=tuple((item.source, item.property) for item in declared),
        )
    return DashboardBreakoutReads(breakout, binding.with_breakouts((breakout,)))


def dashboard_group_data(
    binding: SourceBinding,
    *,
    metrics: Sequence[Metric],
    checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
    population: Literal["assigned", "triggered"] = "assigned",
) -> tuple[DashboardGroupData, ...]:
    """Observed group evidence for the requested assigned or triggered population."""
    require_arm_state(binding, "dashboard_group_data")
    selected = select_metrics(
        binding.metrics,
        metrics,
        caller="dashboard_group_data",
        require_declared_definitions=True,
    )
    source = _require_analysis_operation(
        binding.source,
        "dashboard_group_data",
        DashboardGroupDataOperation,
        message=(
            "prepare_dashboard() needs an experiment built by from_definitions "
            "with a warehouse connection."
        ),
    )
    return tuple(
        source.dashboard_group_data(
            metrics=selected, checkpoints=checkpoints, population=population
        )
    )
