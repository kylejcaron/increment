"""Breakout views over one source binding, read through BreakoutReadouts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from increment._analysis_config import (
    UNSET,
    _Unset,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._day_axis import _TRIGGER_UNSUPPORTED, require_arm_state
from increment._evidence_types import (
    BreakoutIdentity,
    BreakoutRoute,
    BreakoutSlice,
    BreakoutSlicesRequest,
    EvidenceScope,
)
from increment._exploratory import exploratory_sibling, select_exploratory
from increment._literals import Correction
from increment._source_admission import _arm_experiment, _require_arm_binding
from increment.decision import ContrastAnalysisState, DefinitionsArmAnalysisState
from increment.errors import RefusalSpec, UnsupportedRequestError
from increment.errors import refuse as _refuse
from increment.readouts._breakout import _breakout_impl
from increment.readouts._breakout import breakout as readout_breakout
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
from increment.readouts._execution_context import current_execution_context
from increment.readouts._requests import _validate_breakout_request
from increment.readouts._whole_window import reject_role_overrides_under_contrast

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._binding import SourceBinding
    from increment.breakout.estimates import BreakoutEstimates
    from increment.declarations import Method
    from increment.estimation.inference import Prior
    from increment.semantics.models import Experiment, Metric
    from increment.sources import _BuiltinMomentSource

_UNCORRECTED_SEQUENTIAL = RefusalSpec(
    "facade.analysis.uncorrected_segments_sequential",
    UnsupportedRequestError,
    template="uncorrected segment rows have no fixed-horizon p-values under {inference}; read the registered breakout with run_breakout() instead.",
)


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
        breakouts: tuple[BreakoutIdentity, ...] | None = None,
    ) -> None:
        self._src = src
        self._scope_source = src if scope_source is None else scope_source
        self._experiment = experiment
        self._execution_context = execution_context
        self._breakouts = breakouts

    def _identities(self, source: _BuiltinMomentSource) -> tuple[BreakoutIdentity, ...]:
        """The selected identities, else every declared breakout in definition order."""
        if self._breakouts is not None:
            return self._breakouts
        definitions = () if self._experiment is None else tuple(self._experiment.breakouts)
        identities = tuple(BreakoutIdentity(item.property, item.source) for item in definitions)
        if identities:
            return identities
        from increment._source_types import declared_evidence

        return tuple(BreakoutIdentity(name, None) for name in declared_evidence(source).breakouts)

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
        identities = self._identities(self._src)
        request = BreakoutSlicesRequest(
            breakouts=identities,
            metrics=req.metrics,
            population=req.population,
            configs=self._configs(req),
            breakout_definitions=definitions,
        )
        if self._execution_context is not None and self._src is not self._scope_source:
            self._execution_context.declare_source_alias(self._src, parent=self._scope_source)
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
        modes = self._execution_context.effective_modes
        coherence = None if "provider" in modes.values() else self._execution_context.coherence
        rows._metadata = rows.metadata.with_execution_context(
            snapshot_id=self._execution_context.run_id,
            coherence=coherence,
            effective_modes=modes,
            shared_capture=self._execution_context.shared_capture,
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

    def _require_registered_selection(self, population: Literal["assigned", "triggered"]) -> None:
        registration = getattr(self._src.context.plan.inference, "registration", None)
        identities = () if registration is None else self._identities(self._src)
        if not identities:
            return
        from increment.sequential_source import registered_breakout_dimension

        dimension = registered_breakout_dimension(registration, population=population)
        if {identity.property for identity in identities} != {dimension}:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported",
                "registered sequential inference reads only its registered breakout dimension",
            )

    def validate_request(self, req: BreakoutRequest) -> BreakoutRequest:
        self._require_registered_selection(req.population)
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
        identities = self._identities(self._src)
        if not identities:
            return req
        dimension = identities[0].property
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
        from uuid import uuid4

        from increment._source_types import is_custom_source
        from increment.decision import ContrastContext
        from increment.kernel.context import EvidenceExecutionContext
        from increment.readouts._execution_context import (
            execution_context_scope,
            owned_execution_context,
        )
        from increment.readouts._provider import native_scope_provider

        if self._breakouts is not None:
            identities = self._breakouts
        elif self._experiment is not None:
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
        if isinstance(self._scope_source.context, ContrastContext):
            from increment._evidence_scopes import switchback_scope_provider

            provider = switchback_scope_provider(self._scope_source)
        else:
            provider = (
                self._scope_source
                if is_custom_source(self._scope_source)
                else native_scope_provider(self._scope_source)
            )
        with owned_execution_context(
            lambda: EvidenceExecutionContext(
                provider,
                scope,
                run_id=uuid4().hex,
                coherence="captured",
                shared_capture=True,
            ),
            execution_context_scope=execution_context_scope,
        ) as context:
            return BreakoutReadouts(
                src=self._src,
                experiment=self._experiment,
                execution_context=context,
                scope_source=self._scope_source,
                breakouts=self._breakouts,
            ).run(req)

    def run(self, req: BreakoutRequest) -> BreakoutEstimates:
        from increment.breakout.estimates import BreakoutEstimates

        registration = getattr(self._src.context.plan.inference, "registration", None)
        if registration is not None:
            from increment.sequential_source import registered_breakout_dimension

            self._require_registered_selection(req.population)
            dimension = registered_breakout_dimension(registration, population=req.population)
            return readout_breakout(
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
        route = breakout_route(self._src)
        if route == UNSUPPORTED_ROUTE:
            from increment._evidence_refusals import refuse_breakout
            from increment._source_types import declared_evidence

            refusal = declared_evidence(self._src)
            requested = (
                tuple(identity.property for identity in self._breakouts)
                if self._breakouts is not None
                else tuple(breakout.property for breakout in self._experiment.breakouts)
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
        if self._execution_context is not None:
            self._execution_context.validate_request(
                self._scope_source,
                req.metrics,
                self._configs(req),
            )
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
                rows = BreakoutReadouts(
                    src=pinned, experiment=self._experiment, breakouts=self._breakouts
                )._run_slices(req, route)
                return self._stamp_execution_context(rows)

        return self._stamp_execution_context(self._run_slices(req, route))

    def _run_slices(self, req: BreakoutRequest, route: BreakoutRoute) -> BreakoutEstimates:
        from increment.breakout.estimates import BreakoutEstimates

        results: BreakoutEstimates | None = None
        breakout_fn = _breakout_impl if route == PANEL_ROUTE else readout_breakout
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


def plan_breakout_request(
    binding: SourceBinding,
    metrics: Sequence[Metric],
    *,
    population: Literal["assigned", "triggered"],
    populations: Sequence[Literal["assigned", "triggered"]],
    overrides: Mapping[str, Any],
) -> tuple[tuple[Metric, ...], tuple[ResolvedMetricConfig, ...]]:
    """Validate a breakout request before capture and return its metrics and configs."""
    plan = binding.plan
    policy = plan.view_policies.for_view(
        "breakout", mechanism=binding.design.mechanism, segmented=True
    )
    reader = BreakoutReadouts(
        src=binding.source,
        experiment=_arm_experiment(binding),
        breakouts=binding.breakout_selection,
    )
    reader.validate_populations(populations)
    normalized = reader.validate_request(
        BreakoutRequest(
            metrics=tuple(metrics),
            population=population,
            correction=normalize_display_correction(policy.correction),
            q=policy.q if policy.q is not None else plan.q,
            **overrides,
        )
    )
    return normalized.metrics, reader._configs(normalized)


def run_breakout(
    binding: SourceBinding,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str | Metric] | None = None,
    exploratory_metrics: Sequence[str] | None = None,
    population: Literal["assigned", "triggered"] | None = None,
) -> BreakoutEstimates:
    """Per-segment lift estimates for every declared breakout."""
    execution_context = current_execution_context()
    if execution_context is None:
        require_arm_state(
            binding, "run_breakout", population=population if population is not None else "assigned"
        )
    rows = run_breakout_rows(
        binding,
        execution_context=execution_context,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        exploratory_metrics=exploratory_metrics,
        population=population,
    )
    rows._default_backend = binding.input_backend
    return rows


def run_breakout_rows(
    binding: SourceBinding,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str | Metric] | None = None,
    exploratory_metrics: Sequence[str] | None = None,
    correction: Correction | None = None,
    population: Literal["assigned", "triggered"] | None = None,
    execution_context=None,
) -> BreakoutEstimates:
    """Breakout rows for each selected population, optionally under an explicit correction."""
    from increment.breakout.estimates import BreakoutEstimates

    state = binding.state
    experiment = (
        state.experiment
        if isinstance(state, DefinitionsArmAnalysisState)
        else _arm_experiment(binding)
        if state.family == "arm_moments"
        else None
    )
    if population == "triggered" and experiment is not None and experiment.trigger is None:
        _refuse(
            _TRIGGER_UNSUPPORTED,
            method="run_breakout",
            experiment=experiment.name,
            trigger=None,
            route="native",
            supported_sources=("from_definitions", "from_unit_day_artifact"),
        )
    if experiment is not None and not experiment.breakouts:
        select_metrics(binding.source.context.metrics, metrics, caller="run_breakout")
        if exploratory_metrics:
            select_exploratory(binding, exploratory_metrics, caller="run_breakout")
        return BreakoutEstimates([])
    populations = (
        (population,)
        if population is not None
        else (
            ("assigned", "triggered")
            if experiment is not None and experiment.trigger is not None
            else ("assigned",)
        )
    )
    if "triggered" in populations:
        if isinstance(state, ContrastAnalysisState):
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run_breakout",
                experiment="<unknown>",
                trigger=None,
                route="switchback",
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        reader = BreakoutReadouts(
            src=binding.source, experiment=experiment, breakouts=binding.breakout_selection
        )
        if getattr(binding.plan.inference, "registration", None) is None:

            def validate_candidate(candidate: SourceBinding, selected_metrics) -> None:
                if not selected_metrics:
                    return
                breakout_policy = candidate.plan.view_policies.for_view(
                    "breakout",
                    mechanism=getattr(candidate.design, "mechanism", None),
                    segmented=True,
                )
                BreakoutReadouts(
                    src=candidate.source,
                    experiment=_arm_experiment(candidate),
                    breakouts=candidate.breakout_selection,
                ).validate_request(
                    BreakoutRequest(
                        metrics=tuple(selected_metrics),
                        decision_method=decision_method,
                        sensitivity_methods=sensitivity_methods,
                        prior=prior,
                        correction=correction
                        or normalize_display_correction(breakout_policy.correction),
                        q=(
                            breakout_policy.q if breakout_policy.q is not None else candidate.plan.q
                        ),
                        population="triggered",
                    )
                )

            selected = select_metrics(
                binding.source.context.metrics, metrics, caller="run_breakout"
            )
            validate_candidate(binding, selected)
            added = select_exploratory(binding, exploratory_metrics, caller="run_breakout")
            if added:
                validate_candidate(
                    exploratory_sibling(binding, added, caller="run_breakout"), added
                )
        reader.validate_populations(populations)
    combined: BreakoutEstimates | None = None
    for selected_population in populations:
        rows = _run_breakout_population(
            binding,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            correction=correction,
            population=selected_population,
            execution_context=execution_context,
        )
        combined = rows if combined is None else combined.concat(rows)
    return BreakoutEstimates([]) if combined is None else combined


def _run_breakout_population(
    binding: SourceBinding,
    *,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
    metrics: Sequence[str | Metric] | None,
    exploratory_metrics: Sequence[str] | None,
    correction: Correction | None,
    population: Literal["assigned", "triggered"],
    execution_context,
) -> BreakoutEstimates:
    """One population's breakout, under *correction* in place of the plan's when given."""
    from increment.breakout.estimates import BreakoutEstimates
    from increment.estimation.multiplicity import row_multiplicity_status

    added = select_exploratory(binding, exploratory_metrics, caller="run_breakout")
    if isinstance(binding.state, ContrastAnalysisState):
        reject_role_overrides_under_contrast(
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
    _require_arm_binding(binding, "run_breakout")
    inference = binding.plan.inference
    if correction is not None and getattr(inference, "registration", None) is not None:
        _refuse(_UNCORRECTED_SEQUENTIAL, inference=type(inference).__name__)
    breakout_policy = binding.plan.view_policies.for_view(
        "breakout", mechanism=getattr(binding.design, "mechanism", None), segmented=True
    )
    effective_correction = correction or normalize_display_correction(breakout_policy.correction)
    selected = select_metrics(binding.source.context.metrics, metrics, caller="run_breakout")
    rows_for = partial(
        _breakout_rows,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        correction=correction,
        population=population,
        execution_context=execution_context,
    )
    rows = rows_for(binding, selected) if selected or not added else BreakoutEstimates([])
    if added:
        exploratory = exploratory_sibling(binding, added, caller="run_breakout")
        exploratory_rows = rows_for(exploratory, added, scope_source=binding.source)
        rows = exploratory_rows if not rows else rows.concat(exploratory_rows)
    return BreakoutEstimates(
        [
            row.model_copy(
                update={
                    "multiplicity_status": row_multiplicity_status(
                        row,
                        correction=effective_correction if row.family_id is not None else None,
                    )
                }
            )
            for row in rows
        ],
        metadata=rows.metadata,
    )


def _breakout_rows(
    binding: SourceBinding,
    selected: Sequence[Metric],
    *,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
    correction: Correction | None,
    population: Literal["assigned", "triggered"],
    execution_context,
    scope_source: _BuiltinMomentSource | None = None,
) -> BreakoutEstimates:
    plan = binding.plan
    policy = plan.view_policies.for_view(
        "breakout", mechanism=getattr(binding.design, "mechanism", None), segmented=True
    )
    request = BreakoutRequest(
        metrics=tuple(selected),
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        correction=correction or normalize_display_correction(policy.correction),
        q=policy.q if policy.q is not None else plan.q,
        population=population,
    )
    source = binding.source
    summary_source = getattr(source, "summary_breakout_source", None)
    reader = BreakoutReadouts(
        src=summary_source if summary_source is not None else source,
        experiment=_arm_experiment(binding),
        execution_context=execution_context,
        scope_source=source if scope_source is None else scope_source,
        breakouts=binding.breakout_selection,
    )
    return reader.run(request)


__all__ = [
    "BreakoutReadouts",
    "BreakoutRequest",
    "plan_breakout_request",
    "run_breakout",
    "run_breakout_rows",
]
