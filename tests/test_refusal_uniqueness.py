from __future__ import annotations

import importlib
import pkgutil

import pytest

import increment


def _walk_refusal_registries() -> dict[str, list[tuple[str, object]]]:
    """Collect (module_name.attr, RefusalSpec | WarningSpec) for every code
    across every module-level spec, whether it lives in a dict registry
    (any dict attribute, not just the three conventional names --
    a prior review found modules using other dict variable names too) or
    as a standalone module-level constant (a prior finding:
    `query/artifact_publish.py`'s `_ARTIFACT_EXTENSION` and
    `query/artifact_reader.py`'s `_ARTIFACT_METRIC` are both standalone
    `RefusalSpec` constants duplicating a registry code -- a dict-only
    walk misses them entirely). Grouped by `spec.code`, not the dict
    key: a module's own local `_REFUSALS` dict can use a short internal
    key distinct from the public code string it registers (e.g.
    `increment._winsor_errors._REFUSALS` keys its entries by short name
    but registers `spec.code="estimation.winsor.<name>"`), so keying by
    the dict key would both miss real duplicates registered under
    different local key spellings and falsely group unrelated codes that
    happen to share a local key in two different modules.

    `WarningSpec` (the coded-warning counterpart used by every library
    `warnings.warn` site) shares this same code namespace: a warning code
    colliding with a refusal code -- or with another warning's code -- is
    exactly as much a drift hazard as two refusals sharing one code, and
    both spec kinds are walked the same way.
    """
    from increment.errors import RefusalSpec, WarningSpec

    spec_types = (RefusalSpec, WarningSpec)
    registrations: dict[str, list[tuple[str, object]]] = {}
    for _finder, name, _ispkg in pkgutil.walk_packages(increment.__path__, prefix="increment."):
        module = importlib.import_module(name)
        for attr, value in vars(module).items():
            if isinstance(value, spec_types):
                registrations.setdefault(value.code, []).append((f"{name}.{attr}", value))
            elif isinstance(value, dict):
                for key, spec in value.items():
                    if isinstance(spec, spec_types):
                        registrations.setdefault(spec.code, []).append(
                            (f"{name}.{attr}[{key!r}]", spec)
                        )
    return registrations


def test_every_refusal_code_resolves_to_exactly_one_spec() -> None:
    """A code may appear in more than one module-level attribute (a dict
    registry entry, or a standalone constant) -- an importer reusing a
    canonical owner's spec so its own `_raise(code)` helper resolves it
    -- but every appearance MUST be the *same* RefusalSpec/WarningSpec
    object. Two independently constructed spec instances under one code
    string is exactly how a message or exception/warning type silently
    drifts apart between entry points; a `RefusalSpec` and a `WarningSpec`
    sharing one code string is caught the same way, since they can never
    be the same object."""
    registrations = _walk_refusal_registries()
    violations = {
        code: sorted({loc for loc, _ in entries})
        for code, entries in registrations.items()
        if len({id(spec) for _, spec in entries}) > 1
    }
    assert violations == {}, (
        f"{len(violations)} refusal/warning code(s) registered as independently "
        f"constructed (drift-prone) duplicates: {violations}"
    )


def test_arm_evidence_hazard_raises_the_same_code_from_every_entry_point() -> None:
    """The cluster+prior hazard (the same dispatch shape as other
    cluster+sequential/sequential-prior examples) must raise the SAME
    code whether reached through Analysis.run() or the public
    estimate_lift()."""
    import pandas as pd

    from increment.analysis import Analysis
    from increment.errors import CapabilityError
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.estimation.engine import estimate_lift
    from increment.estimation.inference import Normal
    from increment.semantics.models import MeanMetric

    rows = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(40)],
            "store_id": [f"s{i % 20}" for i in range(40)],
            "variant": ["treatment" if i % 2 else "control" for i in range(40)],
            "revenue": [10.0 + i for i in range(40)],
        }
    )
    analysis = Analysis.from_unit_summary(
        rows,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        cluster="store_id",
    )
    with pytest.raises(CapabilityError) as via_analysis:
        analysis.run(prior=Normal(mu=0.0, sigma=0.01))

    metric = MeanMetric(name="m", entity="u", fact="f", aggregation="sum")

    def _cluster_row(group: str, ys: list[float]) -> dict:
        raw = {
            "experiment_id": "e",
            "metric": "m",
            "group_id": group,
            "n": len(ys),
            "sum_y": sum(ys),
            "sum_y2": sum(y * y for y in ys),
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": sum(1.0 for _ in ys),
            "sum_den2": sum(1.0 for _ in ys),
            "sum_yden": sum(ys),
        }
        return centered_row_from_raw_sums(raw)

    cluster_rows = [
        _cluster_row("C", [5.0 + 0.1 * ((i % 5) - 2) for i in range(50)]),
        _cluster_row("T", [5.5 + 0.1 * ((i % 5) - 2) for i in range(50)]),
    ]
    with pytest.raises(CapabilityError) as via_direct:
        estimate_lift(
            [metric],
            cluster_rows,
            control_group="C",
            cluster="store",
            prior=Normal(mu=0.0, sigma=0.01),
        )

    assert via_analysis.value.code == via_direct.value.code == "arm.adjustment.cluster_prior"


def test_sequential_cuped_hazard_raises_the_same_code_from_every_entry_point() -> None:
    """The sequential-inference + non-predeclared-CUPED hazard used to crash
    with a KeyError instead of a coded refusal: ``sequential_support_refusal``
    still returned the deleted string 'readout.adjustment.sequential_cuped',
    which none of its three consumers' own refusal registries (or, for two of
    them, ``compatibility._REFUSALS``) had registered under that name anymore.

    Each consumer is exercised at the same isolation the codebase already
    uses for the typed arm-contract dispatch itself (`test_arm_contract.py`,
    `test_sequential_predictable_transforms.py` construct typed requests
    directly rather than driving a full source/query stack).  The public
    `Analysis.run()`, `estimate_lift()`, and `estimate_encouragement()` paths
    also reach these checks for requests that bypass their earlier plan gates.
    Calling each consumer directly isolates the shared code contract and keeps
    the three entry points aligned.
    """
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest, _validate_sequential_adjustment
    from increment.errors import CapabilityError
    from increment.estimation.encouragement import _resolve_encouragement_methods
    from increment.estimation.engine import Method, _prepare_lift_estimation
    from increment.estimation.sequential import AlwaysValid
    from increment.frame import MetricSpec, synthesise_metric
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.semantics.models import MeanMetric
    from increment.sources import SourceContext
    from tests.sequential_cases import registration as reg_fixture

    # A Bernoulli registration whose models cover "outcome" only -- neither
    # "revenue" (readout path) nor "m" (engine/encouragement paths) is
    # predeclared or retained, so cuped_capability/adjustment_kind resolve
    # to the unpredictable "cuped" flavour for either metric.
    registration = reg_fixture(law="bernoulli")
    inference = AlwaysValid(registration=registration)
    cuped_method = Method(name="cuped", variance_reduction="cuped")

    # Analysis.run() path: _readout_request._validate_sequential_adjustment.
    metric = synthesise_metric(MetricSpec(name="revenue"))
    design = Randomized(control_group="control")
    plan = compile_decision_plan(None, (metric,), path="frame", design=design)
    # Post-hoc-declared inference bypasses compile_decision_plan's own
    # cuped/coefficient gate (validate_compiled_encouragement_plan), which
    # refuses this exact shape before a ReadoutRequest can even exist --
    # isolating the runtime validator this test targets.
    plan = plan.model_copy(update={"inference": inference})
    config = ResolvedMetricConfig(metric, cuped_method, (), None, False)
    request = ReadoutRequest(
        context=SourceContext(
            study_id="exp",
            design=design,
            plan=plan,
            metrics=(metric,),
            configs=(config,),
            cluster=None,
        ),
        metrics=(metric,),
        configs=(config,),
        view="run",
        grain="total",
    )
    with pytest.raises(CapabilityError) as via_readout_request:
        _validate_sequential_adjustment(request)

    # Direct estimate_lift() path: engine._prepare_lift_estimation.
    with pytest.raises(CapabilityError) as via_estimate_lift:
        _prepare_lift_estimation(
            [MeanMetric(name="m", entity="u", fact="f")],
            [],
            "control",
            [cuped_method],
            None,
            0.05,
            "two-sided",
            inference,
            0.0,
            None,
            None,
            None,
        )

    # Encouragement path: encouragement._resolve_encouragement_methods.
    with pytest.raises(CapabilityError) as via_encouragement:
        _resolve_encouragement_methods(
            [cuped_method],
            ("itt",),
            None,
            inference,
            None,
            [MeanMetric(name="m", entity="u", fact="f")],
            None,
        )

    assert (
        via_readout_request.value.code
        == via_estimate_lift.value.code
        == via_encouragement.value.code
        == "arm.adjustment.sequential_cuped"
    )


def test_every_arm_contract_unsupported_code_is_registered_in_compatibility() -> None:
    """Every ``arm.*``/``contrast.*`` code the arm/contrast evidence contract
    can return as ``Unsupported(code)`` must resolve through
    ``compatibility.ARM_COMPATIBILITY_REFUSALS`` (``refuse_unsupported`` raises
    by dict lookup, so a missing registration is a KeyError instead of the
    intended coded refusal -- the exact failure mode
    ``sequential_support_refusal`` hit). Enumerated by driving the contract's
    public dispatchers -- ``arm_runtime_support`` over its request axes,
    ``arm_planning_support`` over the plans ``ArmPlanningProcedure.standard``
    builds and over real ``Baseline`` knobs, and ``ContrastEvidenceContract`` --
    not by grepping the module's source text for string literals, so a new
    branch added later is covered automatically."""
    from increment.compatibility import ARM_COMPATIBILITY_REFUSALS
    from increment.estimation.arm_contract import (
        AbsoluteDecisionPolicy,
        ArmCompatibilityRequest,
        ArmPlanningProcedure,
        ContrastCompatibilityRequest,
        ContrastEvidenceContract,
        FamilyPolicy,
        MethodCapability,
        MetricCapabilities,
        Support,
        Unsupported,
        arm_planning_support,
        arm_runtime_support,
    )
    from increment.estimation.decision_types import FixedInference
    from increment.estimation.sequential import AlwaysValid, AsymptoticMean
    from increment.power.core import Baseline
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        ParallelAssignment,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.models import InferenceSpec
    from tests.asymptotic_cases import mean_registration
    from tests.estimation.test_arm_contract import _metric_capabilities
    from tests.sequential_cases import registration
    from tests.test_compatibility_contract import _axes, _decision

    codes: set[str] = set()

    def _record(support: Support) -> None:
        if isinstance(support, Unsupported):
            codes.add(support.refusal_code)

    # Sweep arm_runtime_support's axes: dependence, inference flavour, CUPED
    # flavour, prior presence and metric type. Inference decides which CUPED
    # coefficients are admissible, so coefficient admissibility sweeps within it.
    for dependence in ("iid", "cluster"):
        for inference in (
            FixedInference(),
            AsymptoticMean(registration=mean_registration()),
            AlwaysValid(registration=registration()),
        ):
            for variance_reduction in ("none", "predeclared_cuped", "retained_cuped"):
                for metric_type in ("mean", "conversion", "retention", "ratio", "quantile"):
                    for prior_present in (False, True):
                        _record(
                            arm_runtime_support(
                                ArmCompatibilityRequest(
                                    assignment=ParallelAssignment(),
                                    analysis=_axes(),
                                    dependence=dependence,
                                    inference=inference,
                                    estimand="itt",
                                    metric=_metric_capabilities(metric_type=metric_type),
                                    decision=_decision(),
                                    methods=(
                                        MethodCapability(
                                            role="decision",
                                            estimator=(
                                                "cuped"
                                                if variance_reduction != "none"
                                                else "unadjusted"
                                            ),
                                            variance_reduction=variance_reduction,
                                        ),
                                    ),
                                    prior_present=prior_present,
                                )
                            )
                        )

    # arm_planning_support: the plans the public constructor builds, then the
    # planning-only baseline knobs -- a real Baseline, so its own validation and
    # not a duck-typed stand-in decides which knob combinations exist.
    for metric_type in ("mean", "quantile"):
        for clustered in (False, True):
            for inference in (None, InferenceSpec(kind="asymptotic_mean")):
                _record(
                    arm_planning_support(
                        ArmPlanningProcedure.standard(
                            metric_type, clustered=clustered, inference=inference
                        )
                    )
                )
    for procedure, baseline in (
        (
            ArmPlanningProcedure.standard(),
            Baseline(mean=1.0, var=1.0, cluster_icc=0.1, avg_cluster_size=25.0),
        ),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, cluster_size_cv=0.5)),
        (ArmPlanningProcedure.standard(clustered=True), Baseline(mean=1.0, var=1.0)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, icc=0.1)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, compliance=0.5)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, trigger_rate=0.5)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, cuped_rho=0.1)),
    ):
        _record(arm_planning_support(procedure, baseline=baseline))

    # ContrastEvidenceContract's assignment guard is unreachable at its current
    # field type: SwitchbackAssignment.sequence is a two-member discriminated
    # union covering exactly the accepted schemes. Contrast inference has a
    # separate direct-estimator guard, so only that code remains registered.
    contract = ContrastEvidenceContract()
    window = SwitchbackWindow(washout_steps=0, observation_steps=4)
    assignment = SwitchbackAssignment(sequence=IndependentBernoulliOrder(), window=window)
    metric = MetricCapabilities(
        metric_type="mean",
        value_scale="absolute",
        winsorization="none",
        outcome_window="bounded",
        uptake_window="not_applicable",
    )

    def _contrast_request(**overrides: object) -> ContrastCompatibilityRequest:
        fields: dict[str, object] = {
            "assignment": assignment,
            "inference": FixedInference(),
            "estimand": "ate",
            "metric": metric,
            "decision": AbsoluteDecisionPolicy(
                alternative="two-sided",
                null_abs=0.0,
                family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
            ),
        }
        fields.update(overrides)
        return ContrastCompatibilityRequest(**fields)

    for overrides in (
        {
            "decision": AbsoluteDecisionPolicy(
                alternative="two-sided",
                null_abs=0.0,
                family=FamilyPolicy(kind="bonferroni", axes=("metric",), nominal_alpha=0.05),
            )
        },
        {
            "metric": MetricCapabilities(
                metric_type="quantile",
                value_scale="absolute",
                winsorization="none",
                outcome_window="bounded",
                uptake_window="not_applicable",
            )
        },
    ):
        _record(contract.runtime_support(_contrast_request(**overrides)))

    assert codes, "the axis sweep above must produce at least one Unsupported code"
    missing = codes - set(ARM_COMPATIBILITY_REFUSALS)
    assert missing == set(), (
        f"{len(missing)} arm-contract code(s) are unregistered in "
        f"compatibility.ARM_COMPATIBILITY_REFUSALS and would KeyError on refuse: {missing}"
    )
    # arm.metric.unsupported is the bh/e_bh branch of arm_planning_support,
    # which no valid ArmPlanningProcedure reaches: its own validator refuses
    # that family before the dispatcher runs. It must still be registered.
    assert {"arm.metric.unsupported", "contrast.inference"} <= set(ARM_COMPATIBILITY_REFUSALS)
