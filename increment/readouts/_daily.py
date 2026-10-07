from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from increment._analysis_config import select_metrics
from increment._readout_request import ReadoutRequest, validate_request
from increment.breakout.estimates import _refuse
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from increment.semantics.models import Metric


def _validate_daily_inference(
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    *,
    method: str = "run_daily_lift",
) -> None:
    """Refuse sequential plans for disjoint daily and cohort slices.

    Routed through the same coded refusal the breakout path uses, so a caller
    reaching this facade gets the registered code and immutable context rather
    than an unstructured exception for the identical condition.
    """
    if inference is not None:
        _refuse(
            "readout.inference.disjoint_slices",
            view="daily" if method == "run_daily_lift" else "cohort",
            inference=type(inference).__name__,
        )


def daily(
    src: MomentSource,
    *,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    include_covariate: bool = False,
) -> list[dict[str, Any]]:
    """Per-day absolute metric values. Requires the source to offer daily grain.

    Absolute values only - carries no causal claim under an Observational
    design. metrics narrows the reported names; an unknown name raises
    before any moments run. ``include_covariate`` is an internal source
    requirement derived from requested methods, not a caller-facing
    statistical option.
    """
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="daily"
    )
    configs_by_name = {config.metric.name: config for config in src.context.configs}
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=[configs_by_name[m.name] for m in selected],
        view="daily",
        grain="daily",
        by=by,
    )
    validate_request(request)
    moment_kwargs = {"include_covariate": True} if include_covariate else {}
    return cast(
        "list[dict[str, Any]]",
        [
            r
            for m in selected
            for r in src.moments(
                m,
                grain="daily",
                by=by,
                **moment_kwargs,
            )
        ],
    )
