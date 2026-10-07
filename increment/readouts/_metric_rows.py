from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from increment.readouts._common import _refuse_unsupported_by
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._source_types import RawOutcomeSource
    from increment.semantics.models import Metric


@dataclass(frozen=True, slots=True)
class MetricRows:
    """One metric's rows, loaded exactly once, quantile or moments-shaped."""

    rows: tuple[Mapping[str, Any], ...]
    unit_frame: Any | None
    observed_arms: frozenset[str]
    n_treatment_arms: int


def _load_metric_rows(
    src: MomentSource, metric: Metric, *, by: Sequence[str], control_group: str
) -> MetricRows:
    """Load one metric's rows once.

    Callers needing the quantile-specific alternative/cluster/breakout guard
    (`_refuse_unsupported_quantile`, which needs `test`/`cluster` -- outside
    this function's contract) call it before `_load_metric_rows`.
    """
    if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
        import narwhals as nw

        from increment.estimation.winsor import _raw_state_from_source

        # Check the arm inventory before constructing a two-arm inference state.
        # Reuse this capture so the aggregate arm gate does not trigger another read.
        raw_source = cast("RawOutcomeSource", src)
        native = raw_source.unit_frame(metric, outcome_stage="raw")

        frame = nw.from_native(native, eager_only=True)
        observed = frozenset(str(g) for g in frame["group_id"].unique().to_list())
        if not observed - {str(control_group)}:
            return MetricRows(
                rows=(),
                unit_frame=None,
                observed_arms=observed,
                n_treatment_arms=0,
            )
        raw = _raw_state_from_source(src, metric, native=native)
        references = {}
        if raw.inference.method == "positive-log-kernel-bootstrap-t-v1":
            from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference

            references = {
                (metric.name, arm.group_id): full_procedure_bootstrap_reference(
                    raw, control_group, arm.group_id
                )
                for arm in raw.arms
                if arm.group_id != control_group
            }
        observed = frozenset(a.group_id for a in raw.arms)
        # Keep the immutable raw state with the loaded evidence so repeated
        # family passes reinvert without fetching or changing the cutoff pool.
        return MetricRows(
            rows=tuple(
                {"group_id": a.group_id, "_winsor_raw_state": raw, "_winsor_references": references}
                for a in raw.arms
            ),
            unit_frame=None,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
        )
    if getattr(metric, "type", None) == "quantile":
        import narwhals as nw

        native = src.unit_frame(metric)
        frame = nw.from_native(native, eager_only=True)
        observed = frozenset(str(g) for g in frame["group_id"].to_list())
        return MetricRows(
            rows=(),
            unit_frame=native,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
        )
    _refuse_unsupported_by(metric, by)
    rows = cast("list[Mapping[str, Any]]", src.moments(metric, grain="total", by=by))
    observed = frozenset(str(r["group_id"]) for r in rows)
    return MetricRows(
        rows=tuple(rows),
        unit_frame=None,
        observed_arms=observed,
        n_treatment_arms=len(observed - {control_group}),
    )
