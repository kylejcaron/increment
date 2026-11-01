"""Source-bound integrity, health and planning-baseline reads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

from increment._analysis_config import select_metrics
from increment._binding import SourceBinding, binding_for
from increment._evidence_refusals import INVALID_POPULATION
from increment._evidence_types import CountEvidence, CountsRequest
from increment._health import SrmObservations, check_health, srm_observation_key
from increment._planning_baseline import planning_baseline as _arm_planning_baseline
from increment._source_admission import _require_analysis_operation, _require_arm_binding
from increment._source_operations import AllocationHistoryOperation
from increment._source_types import declared_evidence, is_custom_source
from increment.errors import CapabilityError, RefusalSpec, refuse
from increment.evidence import evidence_provider, request_evidence
from increment.readouts._evidence import is_artifact_route
from increment.readouts._srm import srm as _assigned_srm
from increment.readouts._srm import triggered_srm

if TYPE_CHECKING:
    import pyarrow as pa

    from increment.decision import ContrastContext, ContrastSource
    from increment.estimation.diagnostics import NotApplicable, SRMResult
    from increment.estimation.health import ExperimentHealth
    from increment.kernel.experiment import Experiment
    from increment.kernel.resolver import Resolver
    from increment.power import Baseline
    from increment.power.switchback import SwitchbackBaseline
    from increment.sources import _BuiltinMomentSource
    from increment.switchback import SwitchbackAssignmentDiagnostic

_SWITCHBACK_ONLY = RefusalSpec(
    "facade.analysis.switchback_only",
    CapabilityError,
    template="{method}() is only available for switchback evidence.",
)


def unit_counts(experiment: Experiment, resolver: Resolver) -> dict[str, int]:
    """Return enrolled unit counts by canonical group label.

    Excluded unassigned units appear under ``"(unassigned)"``. Sources
    without total-grain count evidence refuse with a coded capability error.
    """
    return _unit_counts_binding(binding_for(experiment, resolver))


def _unit_counts_binding(binding: SourceBinding) -> dict[str, int]:
    source = binding.source
    if not is_custom_source(source):
        source = _require_analysis_operation(
            source,
            "unit_counts",
            message="unit_counts() requires a source with enrolled total-grain counts.",
        )
    request = CountsRequest(kind="unit", population=getattr(source, "population", "assigned"))
    return dict(cast(CountEvidence, request_evidence(evidence_provider(source), request)).counts)


def srm(
    experiment: Experiment,
    resolver: Resolver,
    *,
    expected: dict[str, float] | None = None,
    alpha: float = 0.001,
    inference: Literal["always_valid", "fixed"] = "always_valid",
    population: Literal["assigned", "triggered"] = "assigned",
) -> SRMResult | NotApplicable:
    """Check assigned or triggered counts against a declared allocation.

    The default is an anytime-valid check for cumulative prefixes under a
    known allocation with constant conditional arm probabilities; pass
    ``inference="fixed"`` for one predeclared Pearson look. On a clustered
    experiment, the chi-square runs over distinct cluster counts per arm.
    SRM never assumes equal shares or infers its null from observed counts.
    An unavailable population or a missing target allocation refuses with a
    coded error; declare ``Randomized(control_group=..., allocation={...},
    allocation_scheme=...)`` to check the design allocation.

    Parameters
    ----------
    experiment : Experiment
        Experiment declaration the resolver was built for.
    resolver : Resolver
        Resolver bound to the experiment's source.
    expected : dict[str, float] | None
        Target allocation for this call; otherwise the design's ``allocation``.
    alpha : float
        Lifetime significance threshold (default 0.001).
    inference : Literal["always_valid", "fixed"]
        Evidence controlling ``is_srm``. Default ``"always_valid"``.
    population : Literal["assigned", "triggered"]
        ``"assigned"`` (default) checks enrolled counts.
        ``"triggered"`` narrows to the declared trigger population first.

    Returns
    -------
    SRMResult | NotApplicable
        Observational designs return :class:`NotApplicable`, since they
        have no target allocation to test against.
    """
    return _srm_binding(
        binding_for(experiment, resolver),
        expected=expected,
        alpha=alpha,
        inference=inference,
        population=population,
    )


def _srm_binding(
    binding: SourceBinding,
    *,
    expected: dict[str, float] | None = None,
    alpha: float = 0.001,
    inference: Literal["always_valid", "fixed"] = "always_valid",
    population: Literal["assigned", "triggered"] = "assigned",
    observations: SrmObservations | None = None,
) -> SRMResult | NotApplicable:
    _require_arm_binding(binding, "srm")
    if population not in ("assigned", "triggered"):
        refuse(INVALID_POPULATION, population=population)
    try:
        if population == "triggered":
            source = _require_analysis_operation(
                binding.source,
                "triggered_counts",
                message=(
                    "srm(population='triggered') needs a native "
                    "Analysis.from_definitions instance -- a frame/warehouse/"
                    "moments-backed analysis retains no raw event stream to "
                    "narrow to a triggered population."
                ),
            )
            result = triggered_srm(
                evidence_provider(source),
                design=binding.design,
                expected=expected,
                alpha=alpha,
                inference=inference,
            )
        else:
            result = _assigned_srm(
                binding.source, expected=expected, alpha=alpha, inference=inference
            )
    except CapabilityError as refusal:
        if observations is not None:
            observations[srm_observation_key(population, expected, alpha, inference)] = (
                None,
                (refusal.code, dict(refusal.context), str(refusal)),
            )
        raise
    if observations is not None:
        observations[srm_observation_key(population, expected, alpha, inference)] = (
            result,
            None,
        )
    return result


def trigger_rates(experiment: Experiment, resolver: Resolver) -> dict[str, float]:
    """Observed triggered share per arm.

    Feed a pooled estimate (e.g. the mean of these per-arm rates) into
    ``Baseline(..., trigger_rate=...)``, then ``required_sample_size(...)``,
    to plan a follow-up at the right size.
    """
    return _trigger_rates_binding(binding_for(experiment, resolver))


def _trigger_rates_binding(binding: SourceBinding) -> dict[str, float]:
    from increment.query.native_contract import NativeCoreSource

    source = binding.source
    if is_artifact_route(source):
        operations = declared_evidence(source).operations
        if "triggered_counts" not in operations:
            from increment._evidence_refusals import refuse_operation

            refuse_operation(
                source,
                "triggered_counts",
                tuple(sorted(operations)),
                route="trigger_rates() requires complete trigger evidence",
            )
        return source.trigger_rates()
    native = _require_analysis_operation(
        source,
        "triggered_counts",
        NativeCoreSource,
        message=(
            "trigger_rates() needs a native Analysis.from_definitions "
            "instance with a warehouse backend"
        ),
    )
    return native.trigger_rates()


def allocation_history(
    experiment: Experiment,
    resolver: Resolver,
    *,
    population: Literal["assigned", "triggered"] = "assigned",
) -> pa.Table:
    """Daily and cumulative enrollment history for one named population.

    Columns are ``experiment_id``, ``ds`` (the declared day boundary),
    ``group_id``, ``n_daily``, ``n_cumulative`` and ``analysis_population``.
    Assigned histories use first-exposure dates; triggered histories use
    first eligible trigger dates and count only trigger-eligible units.
    Counts are independent of metric maturity.

    Sources without raw enrollment events and clustered experiments refuse
    this unit-grain timeline with a coded capability error.
    """
    return _allocation_history_binding(binding_for(experiment, resolver), population=population)


def _allocation_history_binding(
    binding: SourceBinding, *, population: Literal["assigned", "triggered"] = "assigned"
) -> pa.Table:
    source = _require_analysis_operation(
        binding.source,
        "allocation_history",
        AllocationHistoryOperation,
        message=(
            "allocation_history() needs a native Analysis.from_definitions instance; "
            "a frame/warehouse/moments-backed analysis retains no raw event stream "
            "to build a daily enrollment timeline from."
        ),
    )
    return source.allocation_history(population=population)


def health_check(
    experiment: Experiment,
    resolver: Resolver,
    *,
    expected_allocation: Mapping[str, float] | None = None,
) -> ExperimentHealth:
    """Report assignment balance, trigger evidence and declared trigger assumptions.

    Every check reads one source snapshot; the report carries that snapshot's
    fingerprint and cutoff. A check the source cannot answer is unavailable
    with its refusal code, and a check the design does not call for is not
    applicable; neither is a pass. A non-rejected allocation check does not
    prove valid assignment, and balanced trigger rates do not establish the
    declared trigger assumptions.

    Parameters
    ----------
    experiment : Experiment
        Experiment declaration the resolver was built for.
    resolver : Resolver
        Resolver bound to the experiment's source.
    expected_allocation : Mapping[str, float] | None
        Target arm shares for the allocation checks when the design declares none.

    Returns
    -------
    ExperimentHealth
        Every check in stable order, with the source fingerprint and cutoff.
    """
    return _health_check_binding(
        binding_for(experiment, resolver), expected_allocation=expected_allocation
    )


def _health_check_binding(
    binding: SourceBinding,
    *,
    expected_allocation: Mapping[str, float] | None = None,
    observations: SrmObservations | None = None,
) -> ExperimentHealth:
    expected = None if expected_allocation is None else dict(expected_allocation)
    return check_health(
        binding,
        expected_allocation=expected_allocation,
        srm=lambda target, population: _srm_binding(
            target, expected=expected, population=population, observations=observations
        ),
        trigger_rates=_trigger_rates_binding,
        observations=observations,
    )


def assignment_diagnostic(
    experiment: Experiment, resolver: Resolver
) -> SwitchbackAssignmentDiagnostic:
    """Return switchback assignment integrity evidence."""
    return _assignment_diagnostic_binding(binding_for(experiment, resolver))


def _assignment_diagnostic_binding(binding: SourceBinding) -> SwitchbackAssignmentDiagnostic:
    if binding.state.family != "contrast":
        refuse(_SWITCHBACK_ONLY, method="assignment_diagnostic")
    return binding.source.diagnostics


def planning_baseline(
    experiment: Experiment, resolver: Resolver, metric: str
) -> Baseline | SwitchbackBaseline:
    """The planning input a power solver needs, computed from this experiment's data.

    The experiment is a prior or pilot experiment (or an in-flight experiment's
    pre-treatment data), not the future test being planned. Without one, build
    a ``Baseline`` directly (``Baseline(mean=, var=)``,
    ``Baseline.from_proportion``, ``Baseline.from_absorption``,
    ``Baseline.from_ratio``).

    An arm experiment returns a :class:`~increment.power.Baseline` for
    ``required_sample_size``/``achieved_power``/``minimum_detectable_effect``.
    Outcome moments describe the analyzed population: under a declared trigger,
    the triggered population, with ``trigger_rate`` its observed share of the
    assigned control arm; with no trigger declared, the assigned population and
    ``trigger_rate=1.0``. From that population's control arm: mean/var (a ratio
    metric's linearized moments; a quantile metric returns a
    ``QuantileBaseline`` over its per-unit control values); cuped_rho, when the
    metric declares CUPED, as the control arm's variance reduction under the
    runtime's own CUPED fit; compliance from the arms' uptake difference when
    the design is Encouragement; and cluster_icc/cluster_size_cv from
    contributing clusters, with ICC on the analyzed score. avg_cluster_size
    retains assigned units per randomized cluster; cluster_participation is the
    pilot fraction of assigned clusters contributing analyzed units.

    A switchback experiment (``from_switchback_frame``) returns the pilot-fitted
    :class:`~increment.power.switchback.SwitchbackBaseline` for the
    ``switchback_*`` solvers.

    Refuses by source, naming which, where the source cannot supply what the
    metric type or declared design needs -- including a declared trigger whose
    evidence the source does not carry.
    """
    return _planning_baseline_binding(binding_for(experiment, resolver), metric)


def _planning_baseline_binding(
    binding: SourceBinding, metric: str
) -> Baseline | SwitchbackBaseline:
    source = binding.source
    if binding.state.family == "contrast":
        context = cast("ContrastContext", binding.context)
        spec = select_metrics(context.metrics, [metric], caller="planning_baseline")[0]
        return cast("ContrastSource", source).planning_baseline(spec)
    trigger = getattr(getattr(binding, "experiment", None), "trigger", None)
    if trigger is None and is_custom_source(source):
        trigger = getattr(binding.context, "trigger_name", None)
    return _arm_planning_baseline(
        cast("_BuiltinMomentSource", source),
        metric,
        trigger=trigger,
        trigger_rates=lambda: _trigger_rates_binding(binding),
    )
