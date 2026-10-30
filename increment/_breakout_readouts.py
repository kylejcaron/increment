"""Analysis.run_breakout behind one BreakoutReadouts object."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from increment import readouts
from increment._analysis_config import _Unset, overlay_configs
from increment._day_axis import _TRIGGER_UNSUPPORTED
from increment._evidence_types import (
    BreakoutIdentity,
    BreakoutRoute,
    BreakoutSlice,
    BreakoutSlicesRequest,
    EvidenceScope,
)
from increment._literals import Correction
from increment.errors import refuse as _refuse
from increment.readouts._breakout import _breakout_impl
from increment.readouts._evidence import (
    ARTIFACT_ROUTE,
    NATIVE_ROUTE,
    PANEL_ROUTE,
    UNSUPPORTED_ROUTE,
    _builtin_source,
    breakout_route,
    breakout_snapshot_population,
    evidence_scope,
    load_breakout_evidence,
    serves_custom_breakouts,
    validate_populations,
)
from increment.readouts._requests import _validate_breakout_request

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment.breakout.estimates import BreakoutEstimates
    from increment.declarations import Method
    from increment.estimation.inference import Prior
    from increment.semantics.models import Experiment, Metric
    from increment.sources import _BuiltinMomentSource


@dataclass(frozen=True, slots=True)
class BreakoutRequest:
    metrics: tuple[Metric, ...]
    decision_method: Method | _Unset
    sensitivity_methods: Sequence[Method] | _Unset
    prior: Prior | None | _Unset
    correction: Correction | None
    q: float | None
    population: Literal["assigned", "triggered"]


class BreakoutReadouts:
    def __init__(
        self,
        *,
        src: _BuiltinMomentSource,
        experiment: Experiment | None,
        execution_context=None,
        scope_source: _BuiltinMomentSource | None = None,
        analysis: object | None = None,
    ) -> None:
        self._src = src
        self._scope_source = src if scope_source is None else scope_source
        self._experiment = experiment
        self._execution_context = execution_context
        self._analysis = analysis

    def _configs(self, req: BreakoutRequest) -> tuple[ResolvedMetricConfig, ...]:
        return tuple(
            overlay_configs(
                req.metrics,
                self._src.context.configs,
                methods=None,
                prior=None,
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior_override=req.prior,
            )
        )

    def _slices(self, req: BreakoutRequest) -> Iterator[BreakoutSlice]:
        definitions = () if self._experiment is None else tuple(self._experiment.breakouts)
        identities = tuple(BreakoutIdentity(item.property, item.source) for item in definitions)
        if not identities:
            from increment._source_types import declared_evidence

            identities = tuple(
                BreakoutIdentity(name, None) for name in declared_evidence(self._src).breakouts
            )
        request = BreakoutSlicesRequest(
            breakouts=identities,
            metrics=req.metrics,
            population=req.population,
            configs=self._configs(req),
            breakout_definitions=definitions,
        )
        evidence = load_breakout_evidence(
            _builtin_source(self._src),
            request,
            execution_context=self._execution_context,
            fallback_source=self._src if self._execution_context is not None else None,
        )
        return iter(evidence.slices)

    def _stamp_execution_context(self, rows):
        if rows.metadata is None or self._execution_context is None:
            return rows
        from increment._immutable import _FrozenMapping

        modes = self._execution_context.effective_modes
        coherence = None if "provider" in modes.values() else self._execution_context.coherence
        rows._metadata = rows.metadata.model_copy(
            update={
                "coherence": coherence,
                "effective_modes": _FrozenMapping(modes),
            }
        )
        return rows

    def validate_populations(
        self,
        populations: Sequence[Literal["assigned", "triggered"]],
    ) -> None:
        if "triggered" not in populations:
            return
        registration = getattr(self._src.context.plan.inference, "registration", None)
        if registration is not None:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported",
                "registered sequential inference has no triggered breakout construction",
            )
        route = breakout_route(self._src)
        custom = serves_custom_breakouts(self._src)
        if not custom and (
            route not in (ARTIFACT_ROUTE, NATIVE_ROUTE)
            or self._experiment is None
            or self._experiment.trigger is None
        ):
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run_breakout",
                experiment="<unknown>" if self._experiment is None else self._experiment.name,
                trigger=None if self._experiment is None else self._experiment.trigger,
                route=route,
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        validate_populations(self._src, populations, operation="run_breakout")

    def validate_request(self, req: BreakoutRequest) -> BreakoutRequest:
        route = breakout_route(self._src)
        if route == ARTIFACT_ROUTE and any(
            metric.type == "quantile" and metric.window_days is not None for metric in req.metrics
        ):
            from increment._metric_specs import MetricSpec

            for metric in req.metrics:
                if metric.type == "quantile" and metric.window_days is not None:
                    MetricSpec(
                        name=metric.name,
                        type="quantile",
                        quantile=metric.quantile,
                        window_days=metric.window_days,
                    )
            return req
        if route not in (ARTIFACT_ROUTE, NATIVE_ROUTE, PANEL_ROUTE) or not req.metrics:
            return req
        if self._experiment is not None and self._experiment.breakouts:
            dimension = self._experiment.breakouts[0].property
        else:
            from increment._source_types import declared_evidence

            dimensions = declared_evidence(self._src).breakouts
            if not dimensions:
                return req
            dimension = dimensions[0]
        validated = _validate_breakout_request(
            self._src,
            metrics=[metric.name for metric in req.metrics],
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
            prior=req.prior,
            correction=req.correction,
            q=req.q,
            dimension=dimension,
        )
        return BreakoutRequest(
            metrics=tuple(validated),
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
            prior=req.prior,
            correction=req.correction,
            q=req.q,
            population=req.population,
        )

    def _run_owned_context(self, req: BreakoutRequest) -> BreakoutEstimates:
        from increment.readouts._line2 import owned_execution_context

        if self._experiment is not None:
            identities = tuple(
                BreakoutIdentity(item.property, item.source) for item in self._experiment.breakouts
            )
        else:
            from increment._source_types import declared_evidence

            identities = tuple(
                BreakoutIdentity(name, None)
                for name in declared_evidence(self._scope_source).breakouts
            )
        scope = EvidenceScope(
            metrics=tuple(req.metrics),
            population=breakout_snapshot_population(self._scope_source, req.population),
            configs=self._configs(req),
            grain="total",
            breakouts=identities,
            snapshot=True,
            execution=True,
            include_breakouts=True,
        )
        with owned_execution_context(self._analysis, scope) as context:
            return BreakoutReadouts(
                src=self._src,
                experiment=self._experiment,
                execution_context=context,
                scope_source=self._scope_source,
                analysis=self._analysis,
            ).run(req)

    def run(self, req: BreakoutRequest) -> BreakoutEstimates:
        from increment.breakout.estimates import BreakoutEstimates

        registration = getattr(self._src.context.plan.inference, "registration", None)
        if registration is not None:
            from increment.sequential_source import registered_breakout_dimension

            dimension = registered_breakout_dimension(registration, population=req.population)
            return readouts.breakout(
                self._src,
                dimension,
                metrics=[m.name for m in req.metrics],
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                correction=req.correction,
                q=req.q,
                analysis_population=req.population,
            )
        if self._execution_context is not None:
            self._execution_context.validate_request(
                self._scope_source,
                req.metrics,
                self._configs(req),
            )
        route = breakout_route(self._src)
        if route == UNSUPPORTED_ROUTE:
            from increment._evidence_refusals import refuse_breakout
            from increment._source_types import declared_evidence

            refusal = declared_evidence(self._src)
            requested = (
                tuple(breakout.property for breakout in self._experiment.breakouts)
                if self._experiment is not None
                else tuple(refusal.breakouts)
            )
            refuse_breakout(
                self._src,
                requested,
                refusal.breakouts,
                route="use a source with the requested breakout dimensions",
            )
        custom = serves_custom_breakouts(self._src)
        if route in (ARTIFACT_ROUTE, NATIVE_ROUTE, PANEL_ROUTE) and not custom:
            if not req.metrics:
                return BreakoutEstimates([])
            req = self.validate_request(req)
        elif custom:
            # Validate before the provider scope opens so refusals precede any read.
            if not req.metrics:
                return BreakoutEstimates([])
            req = self.validate_request(req)
        if (
            self._execution_context is None
            and route in (ARTIFACT_ROUTE, NATIVE_ROUTE, PANEL_ROUTE)
            and not custom
            and req.population != "triggered"
        ):
            return self._run_owned_context(req)
        if route == NATIVE_ROUTE or custom:
            if self._execution_context is not None:
                return self._stamp_execution_context(self._run_slices(req, route))
            with evidence_scope(
                self._src,
                EvidenceScope(
                    metrics=tuple(req.metrics),
                    # The snapshot pins one data version; each slice selects its population.
                    population=breakout_snapshot_population(self._src, req.population),
                    configs=self._configs(req),
                    grain="total",
                    snapshot=True,
                    include_breakouts=True,
                ),
            ) as pinned:
                rows = BreakoutReadouts(src=pinned, experiment=self._experiment)._run_slices(
                    req, route
                )
                return self._stamp_execution_context(rows)

        return self._stamp_execution_context(self._run_slices(req, route))

    def _run_slices(self, req: BreakoutRequest, route: BreakoutRoute) -> BreakoutEstimates:
        from increment.breakout.estimates import BreakoutEstimates

        results: BreakoutEstimates | None = None
        breakout_fn = _breakout_impl if route == PANEL_ROUTE else readouts.breakout
        for breakout_slice in self._slices(req):
            segment_results = breakout_fn(
                breakout_slice.source,
                breakout_slice.identity.property,
                source_name=(
                    breakout_slice.stamp_source
                    if breakout_slice.stamp_source is not None
                    else breakout_slice.source_name
                ),
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                metrics=([metric.name for metric in req.metrics] if route == PANEL_ROUTE else None),
                correction=req.correction,
                q=req.q,
                analysis_population=req.population,
            )
            results = segment_results if results is None else results.concat(segment_results)
        return BreakoutEstimates([]) if results is None else results


__all__ = ["BreakoutReadouts", "BreakoutRequest"]
