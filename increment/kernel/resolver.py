from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, runtime_checkable

from increment._source_types import EvidenceCapabilities, EvidenceProvider
from increment.errors import CodedError
from increment.kernel.contracts import Scope, _refuse
from increment.kernel.experiment import Experiment, SourceReference


@dataclass(frozen=True, slots=True, eq=False)
class Runnable:
    static_refusals: tuple[CodedError, ...] = ()
    deferred_checks: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Runnable):
            return NotImplemented
        return (
            self.deferred_checks == other.deferred_checks
            and self.unknown == other.unknown
            and tuple((error.code, error.context) for error in self.static_refusals)
            == tuple((error.code, error.context) for error in other.static_refusals)
        )


@runtime_checkable
class Resolver(Protocol):
    """Bind named source declarations to an evidence provider for one run."""

    def resolve(
        self, source_refs: Sequence[SourceReference], ctx: object | None = None
    ) -> object: ...

    def verify(self, source_refs: Sequence[SourceReference], provider: Any) -> list[CodedError]: ...


@runtime_checkable
class _DeclarationValidator(Protocol):
    def validate_experiment(self, experiment: Experiment) -> None: ...


def _resolve_verified(exp: Experiment, resolver: Resolver, *, ctx: object | None = None) -> object:
    """Validate declarations and verify the resolved provider before use."""
    if isinstance(resolver, _DeclarationValidator):
        resolver.validate_experiment(exp)
    provider = resolver.resolve(exp.source_refs, ctx)
    failures = resolver.verify(exp.source_refs, provider)
    if failures:
        raise failures[0]
    return provider


class BoundResolver:
    """Verify a caller-owned EvidenceProvider without executing experiments."""

    def __init__(self, provider: EvidenceProvider, *, expected_digest: str | None = None) -> None:
        self.provider = provider
        self.expected_digest = expected_digest

    def resolve(
        self,
        source_refs: Sequence[SourceReference],
        ctx: object | None = None,
    ) -> EvidenceProvider:
        return self.provider

    def verify(
        self,
        source_refs: Sequence[SourceReference],
        provider: EvidenceProvider | None = None,
    ) -> list[CodedError]:
        candidate = self.provider if provider is None else provider
        if candidate is not self.provider:
            return [_refusal("kernel.resolver.provider_mismatch", {})]
        expected = {ref.expected_digest for ref in source_refs if ref.expected_digest is not None}
        if self.expected_digest is not None:
            expected.add(self.expected_digest)
        actual = getattr(candidate.context, "source_digest", None)
        if expected and (actual is None or expected != {actual}):
            return [
                _refusal(
                    "kernel.resolver.digest_mismatch",
                    {"expected": tuple(sorted(expected)), "actual": actual},
                )
            ]
        declared = getattr(candidate, "evidence", None)
        if not isinstance(declared, EvidenceCapabilities):
            return [_refusal("kernel.resolver.capabilities_missing", {})]
        for ref in source_refs:
            if ref.capabilities is not None and ref.capabilities != declared:
                return [_refusal("kernel.resolver.capability_mismatch", {"source_ref": ref.name})]
        return []


def _refusal(code: str, context: Mapping[str, object]) -> CodedError:
    return CodedError(
        "Source declarations do not match the bound provider", code=code, context=context
    )


def resolve(estimator_id: str, exp: Experiment, metric: object) -> list[CodedError]:
    """Return source-declaration incompatibilities without acquiring evidence."""
    from increment._evidence_refusals import (
        require_any_evidence_route,
        require_evidence_declaration,
    )
    from increment._evidence_types import (
        EvidenceScope,
        SequentialCaptureRequest,
        UnitRowsRequest,
    )
    from increment.declarations import Method, estimator_refs, thaw
    from increment.estimators.registry import routes as evidence_routes

    metric_name = getattr(metric, "name", metric)
    metric_spec: Any = metric if not isinstance(metric, str) else exp.metrics.get(metric, metric)
    refusals: list[CodedError] = []

    def append_unique(refusal: CodedError) -> None:
        if not any(
            prior.code == refusal.code and prior.context == refusal.context for prior in refusals
        ):
            refusals.append(refusal)

    from increment._runnability import operation_options, scope_operation, scope_refusals

    for scope in exp.scopes:
        for refusal in scope_refusals(exp, scope):
            append_unique(refusal)
    if refusals:
        return refusals
    estimator = estimator_refs().get(estimator_id)
    if exp.analysis_context.get("family") == "contrast":
        from pydantic import TypeAdapter

        from increment._analysis_config import select_metrics
        from increment._runnability import contrast_request_refusals, typed_request_options
        from increment.semantics.models import Metric

        options = typed_request_options(exp.request_options)
        declared_metrics = tuple(
            TypeAdapter(Metric).validate_python(metric) for metric in exp.metrics.values()
        )
        try:
            select_metrics(declared_metrics, options.get("metrics"), caller="run")
        except CodedError as refusal:
            append_unique(refusal)
        for refusal in contrast_request_refusals(exp.request_options):
            append_unique(refusal)
        if estimator_id != "switchback.contrast":
            append_unique(
                CodedError(
                    "Switchback contrast readouts do not accept arm role overrides",
                    code="readout.contrast.override",
                    context={"argument": "decision_method"},
                )
            )
        return refusals
    for source in exp.source_refs:
        capabilities = source.capabilities
        if capabilities is None:
            continue
        constructor = exp.analysis_context.get("source_name") or source.name
        for scope in exp.scopes:
            inference = exp.analysis_context.get("plan", {}).get("inference") or {}
            checkpoint_projection = (
                scope_operation(exp, scope) == "run_asof_lift"
                and inference.get("registration") is not None
            )
            if exp.analysis_context:
                from increment._runnability import readout_refusals

                for refusal in readout_refusals(
                    estimator_id=estimator_id,
                    design=exp.design,
                    declaration=exp.analysis_context,
                    capabilities=capabilities,
                    scope=scope,
                    operation=scope_operation(exp, scope),
                    metric_name=str(metric_name),
                    options=operation_options(exp, scope),
                ):
                    append_unique(refusal)
            request = EvidenceScope(
                metrics=(metric_spec,),
                population=scope.population,
                grain=None if checkpoint_projection else scope.grain,
                breakouts=scope.segment,
            )
            try:
                require_evidence_declaration(capabilities, constructor=constructor, request=request)
            except CodedError as refusal:
                append_unique(refusal)
            if checkpoint_projection:
                for refusal in require_any_evidence_route(
                    capabilities,
                    ((SequentialCaptureRequest(finalized=True),),),
                    constructor=constructor,
                ):
                    append_unique(refusal)
                continue

            if estimator is None:
                continue
            methods = (
                method
                for config in exp.analysis_context.get("configs", ())
                if config["metric"]["name"] == metric_name
                for method in (config["decision_method"], *config["sensitivity_methods"])
            )
            method_payload = next(
                (
                    method
                    for method in methods
                    if (
                        method["variance_reduction"] == (estimator.selector or "none")
                        if estimator_id.startswith("randomized.")
                        else method["name"] == estimator.selector
                    )
                ),
                None,
            )
            method = None if method_payload is None else Method.model_validate(thaw(method_payload))
            routes = list(
                evidence_routes(
                    estimator,
                    {
                        "metric": metric_spec,
                        "method": method,
                        "control": exp.design.get("control_group", ""),
                        "arm_order": (),
                        "population": scope.population,
                        "design_covariates": cast(
                            Mapping[str, Any], exp.design.get("adjustment", {})
                        ).get("covariates", ()),
                    },
                )
            )

            winsorization = (
                metric_spec.get("winsorization")
                if isinstance(metric_spec, Mapping)
                else getattr(metric_spec, "winsorization", None)
            )
            percentile_bounds = (
                (
                    winsorization.get("lower_percentile"),
                    winsorization.get("upper_percentile"),
                )
                if isinstance(winsorization, Mapping)
                else (
                    getattr(winsorization, "lower_percentile", None),
                    getattr(winsorization, "upper_percentile", None),
                )
            )
            if any(value is not None for value in percentile_bounds):
                raw_outcomes = UnitRowsRequest(
                    metric=metric_spec, outcome_stage="raw", population=scope.population
                )
                routes = [route + (raw_outcomes,) for route in routes] or [(raw_outcomes,)]
            for refusal in require_any_evidence_route(
                capabilities, tuple(routes), constructor=constructor
            ):
                append_unique(refusal)
    return refusals


def runnable(exp: Experiment, metric: object) -> dict[str, Runnable]:
    """Conservative offline feasibility: absent declarations remain unknown."""
    from increment.declarations import estimator_refs

    references = estimator_refs()
    capabilities_known = exp.analysis_context.get("family") == "contrast" or all(
        source.capabilities is not None for source in exp.source_refs
    )
    missing = tuple(
        f"{source.name}.capabilities" for source in exp.source_refs if source.capabilities is None
    )
    return {
        estimator_id: Runnable(
            static_refusals=tuple(resolve(estimator_id, exp, metric)),
            deferred_checks=(
                ("sample floors", "usable denominators", "data-dependent estimator checks")
                if capabilities_known
                else ()
            ),
            unknown=() if capabilities_known else missing,
        )
        for estimator_id in (
            (*references, "switchback.contrast")
            if exp.analysis_context.get("family") == "contrast"
            else references
        )
    }


def _select_views(exp: Experiment, views: Sequence[Scope] | None) -> Experiment:
    """Narrow to the selected declared views, keeping declaration order."""
    if views is None:
        return exp
    hashes = tuple(scope.hash() for scope in views)
    declared = {scope.hash() for scope in exp.scopes}
    if not hashes or len(set(hashes)) != len(hashes) or not set(hashes) <= declared:
        raise CodedError(
            "The bound analysis cannot execute these experiment declarations",
            code="kernel.run.execution_unavailable",
            context={"experiment_id": exp.experiment_id, "field": "views"},
        )
    return replace(
        exp,
        scopes=tuple(scope for scope in exp.scopes if scope.hash() in hashes),
        operations={key: value for key, value in exp.operations.items() if key in hashes},
    )


def run(
    exp: Experiment,
    resolver: Resolver,
    *,
    ctx: object | None = None,
    options: dict[str, object] | None = None,
    views: Sequence[Scope] | None = None,
) -> Any:
    """Resolve and verify the experiment's source before handing off execution."""
    exp = _select_views(exp, views)
    provider = _resolve_verified(exp, resolver, ctx=ctx)
    execute = getattr(resolver, "run_experiment", None)
    if not callable(execute):
        _refuse("kernel.run.provider_execution_unavailable")
    return execute(exp, provider, options=options) if options else execute(exp, provider)
