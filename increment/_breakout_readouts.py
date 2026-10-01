"""Analysis.run_breakout behind one BreakoutReadouts object."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from increment import readouts
from increment._analysis_config import _Unset
from increment._literals import Correction
from increment._source_operations import BreakoutSourcesOperation
from increment._source_types import classify_source
from increment._whole_window import _ANALYSIS_OPERATION, _require_analysis_operation
from increment.breakout.estimates import BreakoutEstimates
from increment.errors import refuse as _refuse

if TYPE_CHECKING:
    from increment.estimation.engine import Method
    from increment.estimation.inference import Prior
    from increment.semantics.models import Experiment, Metric
    from increment.sources import MomentSource

BreakoutRoute = Literal["artifact", "panel", "native", "unsupported"]


def _breakout_route(src: MomentSource) -> BreakoutRoute:
    route = classify_source(src)
    if route in ("moments",):
        return "unsupported"
    return route


@dataclass(frozen=True, slots=True)
class BreakoutRequest:
    metrics: tuple[Metric, ...]
    decision_method: Method | _Unset
    sensitivity_methods: Sequence[Method] | _Unset
    prior: Prior | None | _Unset
    correction: Correction | None
    q: float | None


@dataclass(frozen=True, slots=True)
class _BreakoutSlice:
    dimension: str
    source: MomentSource
    source_name: str | None
    stamp_source: str | None


class BreakoutReadouts:
    def __init__(self, *, src: MomentSource, experiment: Experiment | None) -> None:
        self._src = src
        self._experiment = experiment

    def _slices(
        self,
        req: BreakoutRequest,
        route: BreakoutRoute,
    ) -> Iterator[_BreakoutSlice]:
        if route == "artifact":
            assert self._experiment is not None
            breakouts = self._experiment.breakouts
            artifact = cast("BreakoutSourcesOperation", self._src)
            scoped_sources = artifact.breakout_sources(breakouts, metrics=list(req.metrics))
            for breakout, scoped in zip(breakouts, scoped_sources, strict=True):
                yield _BreakoutSlice(
                    dimension=breakout.property,
                    source=scoped,
                    source_name=None,
                    stamp_source=scoped.source_name,
                )
            return
        if route == "panel":
            for dimension in self._src.breakouts:
                yield _BreakoutSlice(
                    dimension=dimension,
                    source=self._src,
                    source_name=None,
                    stamp_source=None,
                )
            return
        assert self._experiment is not None
        breakouts = self._experiment.breakouts
        native_src = _require_analysis_operation(
            self._src,
            "breakout_sources",
            BreakoutSourcesOperation,
            message=(
                "run_breakout() needs a native Analysis.from_definitions or "
                "Analysis.from_unit_panel instance -- this source has no panel "
                "breakouts to summarise."
            ),
        )
        scoped_sources = native_src.breakout_sources(breakouts, metrics=list(req.metrics))
        for breakout, scoped in zip(breakouts, scoped_sources, strict=True):
            multi = sum(item.property == breakout.property for item in breakouts) > 1
            yield _BreakoutSlice(
                dimension=breakout.property,
                source=scoped,
                source_name=scoped.source_name if multi else None,
                stamp_source=scoped.source_name,
            )

    def run(self, req: BreakoutRequest) -> BreakoutEstimates:
        registration = getattr(self._src.context.plan.inference, "registration", None)
        if registration is not None:
            dimensions = sorted({key for cell in registration.roster for key, _ in cell.segment})
            if len(dimensions) != 1:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported",
                    "a breakout readout requires one registered segment dimension",
                )
            return readouts.breakout(
                self._src,
                dimensions[0],
                metrics=[m.name for m in req.metrics],
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                correction=req.correction,
                q=req.q,
            )
        route = _breakout_route(self._src)
        if route == "unsupported":
            _refuse(
                _ANALYSIS_OPERATION,
                message=(
                    "run_breakout() needs a native Analysis.from_definitions or "
                    "Analysis.from_unit_panel instance -- this source has no panel "
                    "breakouts to summarise."
                ),
                operation="breakout_sources",
            )
        if route in ("artifact", "native"):
            assert self._experiment is not None
            if not self._experiment.breakouts or not req.metrics:
                return BreakoutEstimates([])
        if route == "native":
            assert self._experiment is not None
            validated = readouts._validate_breakout_request(
                self._src,
                metrics=[metric.name for metric in req.metrics],
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                correction=req.correction,
                q=req.q,
                dimension=self._experiment.breakouts[0].property,
            )
            req = BreakoutRequest(
                metrics=tuple(validated),
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                correction=req.correction,
                q=req.q,
            )
        results = []
        for breakout_slice in self._slices(req, route):
            segment_results = readouts.breakout(
                breakout_slice.source,
                breakout_slice.dimension,
                source_name=breakout_slice.source_name,
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                metrics=([metric.name for metric in req.metrics] if route == "panel" else None),
                correction=req.correction,
                q=req.q,
            )
            if breakout_slice.stamp_source is not None:
                segment_results = BreakoutEstimates(
                    [
                        row.model_copy(update={"source": breakout_slice.stamp_source})
                        for row in segment_results
                    ]
                )
            results.extend(segment_results)
        return BreakoutEstimates(results)


__all__ = ["BreakoutReadouts", "BreakoutRequest"]
