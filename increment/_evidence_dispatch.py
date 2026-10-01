"""Evidence-family dispatch for arm moments and fixed switchback contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, NoReturn

from increment import readouts
from increment._analysis_config import UNSET, _Unset
from increment._literals import ValueScale
from increment._readout_request import ReadoutRequest
from increment._unit_cycle import _refuse as _unit_refuse
from increment.decision import ContrastDecisionProcedure, ContrastSource
from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals
from increment.errors import refuse as _refuse
from increment.estimation.contrast import compute_contrast
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.engine import Method
from increment.estimation.inference import Prior
from increment.estimation.results import LiftEstimate
from increment.semantics.models import Metric
from increment.sources import MomentSource

if TYPE_CHECKING:
    pass

_CONTRAST_REQUEST = RefusalSpec(
    "readout.contrast.request",
    InvalidRequestError,
    lambda *, message, **_: message,
)


def _reject_contrast_request(*, message: str, **context: object) -> NoReturn:
    _refuse(_CONTRAST_REQUEST, message=message, **context)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.evidence_dispatch.contrast_handler.readout_request_no": "contrast readout request has no procedure for metric {metric_name!r}",
    },
)
_raise = raiser(_REFUSALS)


@dataclass(frozen=True, slots=True)
class ContrastReadoutRequest:
    """Fixed-horizon contrast inputs, one immutable procedure per metric."""

    metrics: tuple[Metric, ...]
    procedures: Mapping[str, ContrastDecisionProcedure]

    def __post_init__(self) -> None:
        metric_names = [metric.name for metric in self.metrics]
        duplicate_names = sorted({name for name in metric_names if metric_names.count(name) > 1})
        procedure_map = dict(self.procedures)
        expected = set(metric_names)
        actual = set(procedure_map)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        mismatches = sorted(
            f"{key!r}->{procedure.metric!r}"
            for key, procedure in procedure_map.items()
            if procedure.metric != key
        )
        if duplicate_names or missing or extra or mismatches:
            _reject_contrast_request(
                message=(
                    "contrast readout request must contain exactly one procedure per metric"
                    f" (duplicates={duplicate_names!r}, missing={missing!r}, "
                    f"extra={extra!r}, mismatches={mismatches!r})"
                ),
                duplicates=duplicate_names,
                missing=missing,
                extra=extra,
                mismatches=mismatches,
            )
        object.__setattr__(self, "procedures", MappingProxyType(procedure_map))


@dataclass(frozen=True, slots=True)
class ArmMomentHandler:
    """Run the existing parallel arm-moment readout family."""

    family: ClassVar[Literal["arm_moments"]] = "arm_moments"
    decision_method: Method | _Unset = UNSET
    sensitivity_methods: Sequence[Method] | _Unset = UNSET
    prior: Prior | None | _Unset = UNSET
    estimands: tuple[str, ...] | None = None
    value_scale: Mapping[str, ValueScale] | None = None

    def run(self, source: MomentSource, request: ReadoutRequest) -> list[LiftEstimate]:
        return readouts.arm_moments(
            source,
            decision_method=self.decision_method,
            sensitivity_methods=self.sensitivity_methods,
            metrics=[metric.name for metric in request.metrics],
            prior=self.prior,
            estimands=self.estimands,
            value_scale=self.value_scale,
            population=request.population,
        )


@dataclass(frozen=True, slots=True)
class ContrastHandler:
    """Estimate one fixed additive contrast per selected metric."""

    family: ClassVar[Literal["contrast"]] = "contrast"

    def run(self, source: ContrastSource, request: ContrastReadoutRequest) -> ContrastResults:
        results = []
        for metric in request.metrics:
            try:
                procedure = request.procedures[metric.name]
            except KeyError:
                _raise(
                    "facade.evidence_dispatch.contrast_handler.readout_request_no",
                    metric_name=metric.name,
                )
            if (
                source.context.study.assignment.sequence.scheme == "independent_bernoulli_order"
                and procedure.reference is None
            ):
                _unit_refuse(
                    "unit_cycle.envelope_required",
                    reason="unit_cycle_reference_required",
                    metric=metric.name,
                )
            results.append(compute_contrast(source.contrast_stats(metric), procedure).results[0])
        return ContrastResults(results)


__all__ = ["ArmMomentHandler", "ContrastHandler", "ContrastReadoutRequest"]
