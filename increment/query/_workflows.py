"""Experiment-bound SQL preview operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from increment._optional import require_warehouse

if TYPE_CHECKING:
    from ibis.expr.types import Scalar, Table

    from increment._binding import ExperimentReader
    from increment.kernel.experiment import Experiment
    from increment.semantics.models import Breakout, Metric


def _require_arm_binding(binding: Any, method: str) -> Any:
    if binding.state.family != "arm_moments":
        from increment._evidence_refusals import CONTRAST_UNAVAILABLE
        from increment.errors import refuse

        refuse(CONTRAST_UNAVAILABLE, method=method)
    return binding


def _build_panel_for_metric_binding(
    binding: Any,
    metric: Metric,
    *,
    population: Literal["assigned", "triggered"] = "assigned",
) -> tuple[Table, Table, Table, Table, Table | None, Table, str | None, Scalar]:
    from increment._source_admission import _require_analysis_operation
    from increment._source_operations import NativePanelSqlOperation

    src = _require_analysis_operation(
        binding.source,
        "panel_sql",
        NativePanelSqlOperation,
        message=(
            "build_panel_for_metric() needs an experiment built by from_definitions "
            "with a warehouse connection"
        ),
    )
    return src.build_panel_for_metric(metric, population=population)


def _panel_sql_binding(binding: Any, breakouts: list[Breakout] | None = None) -> dict[str, str]:
    from increment._source_admission import _require_analysis_operation
    from increment._source_operations import NativePanelSqlOperation

    src = _require_analysis_operation(
        binding.source,
        "panel_sql",
        NativePanelSqlOperation,
        message=(
            "panel_sql() needs an experiment built by from_definitions -- a frame, "
            "warehouse or moments experiment has no SQL substrate to introspect."
        ),
    )
    return src.panel_sql(breakouts=breakouts or ())


def _summary_sql_binding(binding: Any, breakouts: list[Breakout] | None = None) -> dict[str, str]:
    from increment._source_admission import _require_analysis_operation
    from increment._source_operations import SummarySqlOperation
    from increment._source_types import declared_evidence

    binding = _require_arm_binding(binding, "summary_sql")
    src = binding.source
    if "summary_sql" not in declared_evidence(src).operations:
        return src.sql()
    summary = _require_analysis_operation(
        src,
        "summary_sql",
        SummarySqlOperation,
        message=(
            "summary_sql() needs an experiment built by from_definitions -- a frame, "
            "warehouse or moments experiment has no SQL substrate to introspect."
        ),
    )
    return summary.summary_sql(breakouts=breakouts or ())


def build_panel_for_metric(
    experiment: Experiment,
    reader: ExperimentReader,
    metric: Metric,
    *,
    population: Literal["assigned", "triggered"] = "assigned",
) -> tuple[Table, Table, Table, Table, Table | None, Table, str | None, Scalar]:
    """Return one native metric's live panel tuple, independent of TEMP tables.

    The source selects the population's cohort: every enrolled unit, or the
    triggered units anchored at their first trigger.
    """
    from increment._binding import binding_for

    require_warehouse("build_panel_for_metric")
    return _build_panel_for_metric_binding(
        binding_for(experiment, reader), metric, population=population
    )


def panel_sql(
    experiment: Experiment,
    reader: ExperimentReader,
    breakouts: list[Breakout] | None = None,
) -> dict[str, str]:
    """Return per-metric panel SQL without executing."""
    from increment._binding import binding_for

    require_warehouse("panel_sql")
    return _panel_sql_binding(binding_for(experiment, reader), breakouts)


def summary_sql(
    experiment: Experiment,
    reader: ExperimentReader,
    breakouts: list[Breakout] | None = None,
) -> dict[str, str]:
    """Return per-metric group-summary SQL without executing."""
    from increment._binding import binding_for

    require_warehouse("summary_sql")
    return _summary_sql_binding(binding_for(experiment, reader), breakouts)
