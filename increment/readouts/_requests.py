from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from increment._analysis_config import UNSET, _Unset, overlay_configs, select_metrics
from increment._literals import Correction, ValueScale
from increment._readout_request import ReadoutRequest, validate_request
from increment.estimation.engine import Method
from increment.estimation.sequential import SEQUENTIAL_POLICIES
from increment.semantics.design import Encouragement
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.estimation.inference import Prior
    from increment.semantics.models import Metric


def _design_compliance_only(src: MomentSource, estimands: Sequence[str] | None) -> bool:
    """Fixed-horizon compliance needs design state, not outcome configuration."""
    return (
        estimands is not None
        and set(estimands) == {"compliance"}
        and isinstance(src.context.design, Encouragement)
        and not isinstance(src.context.plan.inference, SEQUENTIAL_POLICIES)
    )


def _prepare_run_request(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    population: Literal["assigned", "triggered"] = "assigned",
) -> tuple[list[Metric], Sequence[ResolvedMetricConfig]]:
    """Resolve and validate a whole-window request before mechanism dispatch."""
    selected = select_metrics(cast("Sequence[Metric]", src.context.metrics), metrics, caller="run")
    # Fixed-horizon compliance consumes no outcome configuration.
    # Sequential requests still need their registered metric context.
    if _design_compliance_only(src, estimands):
        selected = []
    configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=configs,
        view="run",
        grain="total",
        by=by,
        estimands=estimands,
        value_scale=value_scale,
        population=population,
    )
    validate_request(request)
    return selected, configs


def _validate_run_request(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
) -> list[Metric]:
    """Validate a run request without asking the source for moments."""
    selected, _configs = _prepare_run_request(
        src,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
    )
    return selected


def _validate_breakout_request(
    src: MomentSource,
    *,
    metrics: Sequence[str | Metric] | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    correction: Correction | None = None,
    q: float | None = None,
    dimension: str,
) -> list[Metric]:
    """Validate a native breakout request before its scoped source queries."""
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="breakout"
    )
    configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=configs,
        view="breakout",
        grain="total",
        by=(dimension,),
        dimension=dimension,
        correction=correction,
        q=q,
    )
    validate_request(request)
    return selected
