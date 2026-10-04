"""Typed compatibility requests for parallel-arm and contrast evidence."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from increment._literals import Alternative, ConversionInference, PreferredDirection, ValueScale
from increment.compatibility import (
    ARM_COMPATIBILITY_REFUSALS,
    ClusterFloor,
    IndependentBlockFloor,
    IndependentUnitFloor,
    PowerDesign,
    SamplingFloor,
    Support,
    Supported,
    UnitFloor,
    Unsupported,
    _conservative_divide,
    _conservative_ratio,
)
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)
from increment.estimation.conversion_route import (
    finite_sample_blocker,
    refuse_finite_sample_unavailable,
)
from increment.estimation.decision_types import FixedInference
from increment.estimation.sequential import (
    ASYMPTOTIC_PROCEDURE_POLICIES,
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    GaussianScoreMixture,
    MixedFamily,
    SequentialSupportRequest,
)
from increment.semantics.assignment import ParallelAssignment, SwitchbackAssignment
from increment.semantics.models import InferenceSpec, MethodSpec
from increment.semantics.sequential import SequentialCompliancePolicy, sequential_family_size

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "decision.family.axes_contain_unknown": RefusalSpec(
            "decision.family.axes_contain_unknown",
            InvalidRequestError,
            lambda *, unknown: f"family axes contain unknown values: {list(unknown)!r}",
        ),
        "decision.family.axes_contain_duplicates": "family axes must not contain duplicates: {axes!r}",
        "decision.family.axes_use_canonical": "family axes must use canonical order {allowed!r}, got {axes!r}",
        "decision.family.uncorrected_policy_empty": "uncorrected family policy requires empty axes",
        "decision.family.policy_least_one": "{kind} family policy requires at least one axis",
        "decision.relative_decision.one_sided_nominal": "one-sided nominal alpha must be < 0.5",
        "decision.arm_planning.scalar_power_unsupported_family": "scalar power does not support BH/e-BH family planning",
        "decision.arm_planning.uncorrected_family_size": "uncorrected planning requires family_size=1",
        "arm_planning.inference_spec_registration_unplannable": "planning derives its own registration-free boundary; pass InferenceSpec(kind='asymptotic_mean') without registration=",
        "arm_planning.inference_spec_expected_n_unplannable": "expected_decision_sample_size is solved for, not declared -- read it from PowerResult.inference_to_declare after sizing",
        "arm_planning.inference_spec_adjustments_unplannable": "predeclared CUPED adjustments are a runtime-only declaration; planning admits CUPED by passing a Baseline with cuped_rho set to the solver call",
        "arm_planning.inference_spec_segments_unplannable": (
            "standard() fixes view='total' and segmented=False, so it cannot represent "
            "declared segments {rejected_segments!r} without dropping them; use the full "
            "ArmPlanningProcedure constructor with view='breakout' and segmented=True "
            "for an explicitly segmented fixed-horizon plan -- standard() provides no "
            "segmented asymptotic planning construction (route={route!r})",
            ("view", "segmented"),
        ),
        "arm_planning.secondary_requires_count": "role='secondary' requires secondaries=<count>",
        "arm_planning.secondaries_without_secondary_role": "secondaries= is only meaningful with role='secondary'",
        "arm_planning.preferred_direction_without_guardrail_role": "preferred_direction= is only meaningful with role='guardrail'",
        "arm_planning.sequential_secondary_one_arm": "a sequential secondary's runtime registration supports exactly one treatment arm; pass comparisons=1",
        "arm_planning.guardrail_preferred_direction_conflict": "alternative disagrees with the declared preferred_direction",
        "arm_planning.guardrail_requires_one_sided": "a guardrail is evaluated at full alpha, one-sided; pass preferred_direction= or an explicit one-sided alternative=",
        "arm_planning.guardrail_neutral_requires_direction": (
            "preferred_direction='neutral' declares no direction, so a guardrail still can't be one-sided; pass preferred_direction='increase'/'decrease' or an explicit one-sided alternative='greater'/'less'",
            (
                "alternative",
                "preferred_direction",
            ),
        ),
    },
)
_raise = raiser(_REFUSALS)


class MethodCapability(CodedModel, BaseModel):
    """One estimation method's axes.

    ``predeclared_cuped`` retains a scalar transformed by a coefficient and
    centre fixed from pre-period data. ``retained_cuped`` fits the coefficient
    from joint (Y, X) moments at each look (Lindon, Ham, Tingley and Bojinov
    2022). Both sequential adjustments require ``AsymptoticMean``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    role: Literal["decision", "sensitivity"]
    estimator: str = Field(min_length=1)
    variance_reduction: Literal["none", "cuped", "predeclared_cuped", "retained_cuped"]


def cuped_capability(
    method: object, *, adjustment: Literal["predeclared", "retained"] | None
) -> Literal["none", "cuped", "predeclared_cuped", "retained_cuped"]:
    """Map a runtime method onto the variance-reduction axis for one metric."""
    if getattr(method, "variance_reduction", "none") != "cuped":
        return "none"
    if adjustment == "predeclared":
        return "predeclared_cuped"
    if adjustment == "retained":
        return "retained_cuped"
    return "cuped"


class AnalysisAxes(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    identification: Literal["randomized", "encouragement", "observational"]
    view: Literal["total", "breakout", "daily", "asof"]
    segmented: bool
    completed_windows_only: bool
    population: Literal["assigned", "triggered"]
    variance_adjustment: Literal["none", "factor_absorption"]


class MetricCapabilities(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    metric_type: Literal["mean", "conversion", "retention", "ratio", "quantile"]
    value_scale: ValueScale
    winsorization: Literal["none", "fixed", "percentile"]
    outcome_window: Literal["bounded", "unbounded"]
    uptake_window: Literal["not_applicable", "bounded", "unbounded"]


class FamilyPolicy(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["none", "bonferroni", "bh", "e_bh"]
    axes: tuple[str, ...]
    nominal_alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_axes(self) -> FamilyPolicy:
        allowed = ("metric", "arm", "segment")
        unknown = set(self.axes) - set(allowed)
        if unknown:
            _raise("decision.family.axes_contain_unknown", unknown=tuple(sorted(unknown)))
        if len(self.axes) != len(set(self.axes)):
            _raise("decision.family.axes_contain_duplicates", axes=self.axes)
        if tuple(axis for axis in allowed if axis in self.axes) != self.axes:
            _raise("decision.family.axes_use_canonical", allowed=allowed, axes=self.axes)
        if self.kind == "none" and self.axes:
            _raise("decision.family.uncorrected_policy_empty")
        if self.kind != "none" and not self.axes:
            _raise("decision.family.policy_least_one", kind=self.kind)
        return self


class RelativeDecisionPolicy(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    alternative: Alternative
    null_lift: float = Field(allow_inf_nan=False)
    signed_ratio: bool = Field(default=False, exclude_if=lambda value: not value)
    family: FamilyPolicy
    scale: Literal["relative"] = "relative"

    @model_validator(mode="after")
    def _one_sided_alpha(self) -> RelativeDecisionPolicy:
        if self.null_lift <= -1 and not self.signed_ratio:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported", "signed null ratios require scalar mean inference"
            )
        if self.alternative != "two-sided" and self.family.nominal_alpha >= 0.5:
            _raise("decision.relative_decision.one_sided_nominal")
        return self


class AbsoluteDecisionPolicy(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    alternative: Alternative
    null_abs: float = Field(allow_inf_nan=False)
    family: FamilyPolicy
    scale: Literal["absolute"] = "absolute"

    @model_validator(mode="after")
    def _one_sided_alpha(self) -> AbsoluteDecisionPolicy:
        if self.alternative != "two-sided" and self.family.nominal_alpha >= 0.5:
            _raise("decision.relative_decision.one_sided_nominal")
        return self


DecisionPolicy = Annotated[
    RelativeDecisionPolicy | AbsoluteDecisionPolicy,
    Field(discriminator="scale"),
]


RuntimeInference = (
    FixedInference | AsymptoticMean | AlwaysValid | MixedFamily | GaussianScoreMixture
)


class PlanningFamilyExpansion(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    family_size: int = Field(ge=1)


class ArmCompatibilityRequest(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    assignment: ParallelAssignment
    analysis: AnalysisAxes
    dependence: Literal["iid", "cluster"]
    inference: RuntimeInference
    estimand: str
    metric: MetricCapabilities
    decision: DecisionPolicy
    methods: tuple[MethodCapability, ...]
    prior_present: bool

    @model_validator(mode="after")
    def _signed_ratio_model(self):
        if isinstance(self.decision, RelativeDecisionPolicy) and self.decision.signed_ratio:
            if not isinstance(self.inference, ASYMPTOTIC_PROCEDURE_POLICIES):
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported", "signed null ratios require scalar mean inference"
                )
        return self

    def sequential_request(self) -> SequentialSupportRequest:
        return SequentialSupportRequest(
            inference=self.inference if isinstance(self.inference, SEQUENTIAL_POLICIES) else None,
            uses_cuped=any(method.variance_reduction == "cuped" for method in self.methods),
        )


class ArmPlanningProcedure(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    assignment: ParallelAssignment
    analysis: AnalysisAxes
    dependence: Literal["iid", "cluster"]
    inference: RuntimeInference
    estimand: str
    metric: MetricCapabilities
    decision: DecisionPolicy
    family_expansion: PlanningFamilyExpansion
    decision_method: MethodSpec
    sensitivity_methods: tuple[MethodSpec, ...]
    prior_present: bool

    @model_validator(mode="after")
    def _family_expansion_matches_policy(self) -> ArmPlanningProcedure:
        if isinstance(self.decision, RelativeDecisionPolicy) and self.decision.signed_ratio:
            from increment.sequential_state import sequential_refuse

            sequential_refuse("route.unsupported", "signed scalar mean planning is unsupported")
        if isinstance(self.inference, SEQUENTIAL_POLICIES):
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported",
                "registered likelihood runtime is not a Gaussian planning approximation; "
                "declare inference=InferenceSpec(kind='asymptotic_mean') on "
                "ArmPlanningProcedure.standard()",
            )
        policy = self.decision.family
        if policy.kind in ("bh", "e_bh"):
            _raise("decision.arm_planning.scalar_power_unsupported_family")
        if policy.kind == "none" and self.family_expansion.family_size != 1:
            _raise("decision.arm_planning.uncorrected_family_size")
        return self

    @classmethod
    def standard(  # noqa: PLR0913
        cls,
        metric_type: Literal["mean", "conversion", "retention", "ratio", "quantile"] = "mean",
        *,
        alpha: float = 0.05,
        alternative: Alternative = "two-sided",
        comparisons: int = 1,
        clustered: bool = False,
        value_scale: ValueScale = "relative",
        null_lift: float = 0.0,
        inference: InferenceSpec | None = None,
        role: Literal["primary", "secondary", "guardrail"] = "primary",
        secondaries: int | None = None,
        q: float = 0.10,
        preferred_direction: PreferredDirection | None = None,
        compliance: SequentialCompliancePolicy | None = None,
        conversion_inference: ConversionInference = "auto",
    ) -> ArmPlanningProcedure:
        """A fixed-horizon plan for an ordinary parallel A/B test.

        Every field this fills is a choice a plan must make; the defaults are
        the ordinary answers, written out here rather than hidden:
        a randomized parallel experiment, analyzed at total grain over the
        assigned population in completed windows, unadjusted, with no prior,
        no winsorization and a bounded outcome window. ``comparisons`` above
        one splits ``alpha`` across arms by Bonferroni. ``role="primary"`` is
        the default.

        ``inference=InferenceSpec(kind="asymptotic_mean")`` plans a sequential
        design against the exact boundary the runtime executes -- the same
        object you declare at runtime registration, registration-free here
        since planning solves for the sample size rather than reading one.
        ``kind="always_valid"`` has no planning construction and is refused.

        ``role="secondary"`` requires ``secondaries`` (the metric-count of the
        secondary family this procedure belongs to) and sizes at
        ``alpha=q/m`` (fixed-horizon, ``m=secondaries*comparisons``) or
        ``alpha=min(alpha, q/m)`` (sequential, ``m=secondaries`` --
        ``comparisons`` above 1 is refused, since a sequential secondary's
        runtime registration supports exactly one treatment arm), ``m``
        composed exactly as the runtime keys its own secondary family.
        ``compliance`` -- the same ``SequentialCompliancePolicy`` declared on
        ``AnalysisPlan.compliance`` -- folds the design's uptake cell into
        that family exactly as the runtime does; the caller never counts it.

        ``role="guardrail"`` forces full alpha (``comparisons`` never splits
        it) and a one-sided ``alternative``, derived from
        ``preferred_direction`` when given.

        ``conversion_inference`` is the decision method's ``Method.conversion_inference``:
        the default ``"auto"`` plans a conversion or retention metric the way the runtime
        routes it by counts (closed form where the counts are dense, the replayed
        finite-sample decision where they are sparse), and ``"finite_sample"`` plans the
        replayed finite-sample decision at every size. It is refused for a metric that is
        not a conversion or retention rate, a clustered plan, or a sequential one.

        The power solvers derive CUPED, encouragement compliance, triggered
        populations and factor absorption from the ``Baseline`` they receive,
        and a quantile metric from a ``QuantileBaseline``, so ``standard()``
        plans those designs too. Segmented views need the full constructor,
        because planning must describe the analysis actually intended.
        """
        if conversion_inference == "finite_sample":
            reason = finite_sample_blocker(
                metric_type,
                cluster="the plan's cluster" if clustered else None,
                prior_present=False,
                sequential=inference is not None,
            )
            if reason is not None:
                refuse_finite_sample_unavailable(metric_type, reason)
        if secondaries is not None and role != "secondary":
            _raise("arm_planning.secondaries_without_secondary_role")
        if role == "secondary" and secondaries is None:
            _raise("arm_planning.secondary_requires_count")
        if preferred_direction is not None and role != "guardrail":
            _raise("arm_planning.preferred_direction_without_guardrail_role")

        resolved_inference: RuntimeInference = FixedInference()
        is_sequential = False
        if inference is not None:
            if inference.kind == "always_valid":
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported",
                    "the exact e-process boundary has no planning construction; "
                    "plan a fixed-horizon procedure and monitor with always_valid "
                    "at runtime, or use asymptotic_mean for a planned sequential design",
                )
            if inference.segments:
                _raise(
                    "arm_planning.inference_spec_segments_unplannable",
                    rejected_segments=inference.segments,
                    route=(
                        "ArmPlanningProcedure(..., "
                        "analysis=AnalysisAxes(view='breakout', segmented=True), "
                        "inference=FixedInference())"
                    ),
                    view="total",
                    segmented=False,
                )
            if inference.registration is not None:
                _raise("arm_planning.inference_spec_registration_unplannable")
            if inference.expected_decision_sample_size is not None:
                _raise("arm_planning.inference_spec_expected_n_unplannable")
            if inference.adjustments:
                _raise("arm_planning.inference_spec_adjustments_unplannable")
            resolved_inference = GaussianScoreMixture()
            is_sequential = True

        if compliance is not None and not is_sequential:
            from increment.semantics.sequential import invalid_registration

            invalid_registration("compliance requires a registered sequential inference")

        resolved_alternative = alternative
        if role == "guardrail":
            direction_by_preference: dict[str, Literal["greater", "less"]] = {
                "increase": "greater",
                "decrease": "less",
            }
            direction_alternative = (
                direction_by_preference.get(preferred_direction)
                if preferred_direction is not None
                else None
            )
            if direction_alternative is not None:
                if alternative != "two-sided" and alternative != direction_alternative:
                    _raise("arm_planning.guardrail_preferred_direction_conflict")
                resolved_alternative = direction_alternative
            if resolved_alternative == "two-sided":
                if preferred_direction == "neutral":
                    _raise(
                        "arm_planning.guardrail_neutral_requires_direction",
                        preferred_direction=preferred_direction,
                        alternative=alternative,
                    )
                _raise("arm_planning.guardrail_requires_one_sided")

        if role == "secondary":
            assert secondaries is not None  # validated above
            if is_sequential:
                if comparisons > 1:
                    _raise("arm_planning.sequential_secondary_one_arm")
                family_size = sequential_family_size(secondaries, compliance)
                nominal_alpha = min(alpha * family_size, q)
            else:
                family_size = secondaries * comparisons
                nominal_alpha = q
            family = FamilyPolicy(
                kind="bonferroni", axes=("metric", "arm"), nominal_alpha=nominal_alpha
            )
        elif role == "guardrail":
            family_size = 1
            family = FamilyPolicy(kind="none", axes=(), nominal_alpha=alpha)
        else:
            family_size = comparisons
            family = FamilyPolicy(
                kind="bonferroni" if comparisons > 1 else "none",
                axes=("arm",) if comparisons > 1 else (),
                nominal_alpha=alpha,
            )
        decision: DecisionPolicy = (
            RelativeDecisionPolicy(
                alternative=resolved_alternative, null_lift=null_lift, family=family
            )
            if value_scale == "relative"
            else AbsoluteDecisionPolicy(
                alternative=resolved_alternative, null_abs=null_lift, family=family
            )
        )
        identification: Literal["randomized", "encouragement", "observational"] = (
            "encouragement" if compliance is not None else "randomized"
        )
        return cls(
            assignment=ParallelAssignment(),
            analysis=AnalysisAxes(
                identification=identification,
                view="total",
                segmented=False,
                completed_windows_only=True,
                population="assigned",
                variance_adjustment="none",
            ),
            dependence="cluster" if clustered else "iid",
            inference=resolved_inference,
            estimand="mean",
            metric=MetricCapabilities(
                metric_type=metric_type,
                value_scale=value_scale,
                winsorization="none",
                outcome_window="bounded",
                uptake_window="not_applicable",
            ),
            decision=decision,
            family_expansion=PlanningFamilyExpansion(family_size=family_size),
            decision_method=MethodSpec(name="unadjusted", conversion_inference=conversion_inference),
            sensitivity_methods=(),
            prior_present=False,
        )

    @property
    def compiled_alpha(self) -> float:
        policy = self.decision.family
        return (
            policy.nominal_alpha
            if policy.kind == "none"
            else _conservative_divide(policy.nominal_alpha, self.family_expansion.family_size)
        )

    @property
    def compiled_tail_alpha(self) -> float:
        if self.decision.alternative != "two-sided":
            return self.compiled_alpha
        return _conservative_ratio(
            self.decision.family.nominal_alpha,
            1,
            self.family_expansion.family_size * 2,
        )


class ContrastCompatibilityRequest(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    assignment: SwitchbackAssignment
    inference: FixedInference
    estimand: str
    metric: MetricCapabilities
    decision: AbsoluteDecisionPolicy


RequestT = TypeVar("RequestT")


@runtime_checkable
class EvidenceContract(Protocol[RequestT]):
    family: str

    def runtime_support(self, request: RequestT) -> Support: ...

    def planning_support(self, request: RequestT) -> Support: ...


def _arm_floor(dependence: str, metric: MetricCapabilities) -> SamplingFloor:
    if dependence == "cluster":
        return ClusterFloor(minimum_per_arm=2)
    if metric.metric_type == "quantile":
        return UnitFloor(minimum_per_arm=20)
    return UnitFloor(minimum_per_arm=2)


def _cuped_axes(methods, *, asymptotic: bool) -> tuple[bool, bool]:
    """``(cuped, coefficient_admissible)`` over one request's methods.

    Predeclared scalar transforms and retained-moment regression adjustments
    are supported by the asymptotic route, not the exact Bernoulli law.
    """
    flavours = {m.variance_reduction for m in methods} - {"none"}
    admissible = {"predeclared_cuped", "retained_cuped"} if asymptotic else set()
    return bool(flavours), flavours <= admissible


def _cuped_flavour_for_planning(method: MethodSpec, *, asymptotic: bool) -> str:
    """Planning can register retained-moment CUPED on the asymptotic route."""
    if method.variance_reduction == "cuped" and asymptotic:
        return "retained_cuped"
    return method.variance_reduction


class _PlanningMethodFlavour:
    """Duck-typed stand-in exposing only what _cuped_axes reads."""

    __slots__ = ("variance_reduction",)

    def __init__(self, variance_reduction: str) -> None:
        self.variance_reduction = variance_reduction


def _arm_support(
    *,
    dependence: str,
    metric: MetricCapabilities,
    sequential: bool,
    cuped: bool,
    coefficient_admissible: bool,
    prior_present: bool,
) -> Support:
    """Classify an arm request from the axes support actually depends on.

    Planning and runtime share these axes but not their request types: a
    prospective look schedule has no runtime form to be restated as.
    """
    if dependence == "cluster" and sequential:
        return Unsupported("arm.inference.cluster")
    if dependence == "cluster" and cuped:
        return Unsupported("arm.adjustment.cluster_cuped")
    if dependence == "cluster" and prior_present:
        return Unsupported("arm.adjustment.cluster_prior")
    if dependence == "cluster" and metric.metric_type == "quantile":
        return Unsupported("arm.metric.quantile_cluster")
    # A coefficient fitted from accumulated outcomes is admissible only where the
    # registration retains the joint moments and the route is asymptotic.
    if sequential and cuped and not coefficient_admissible:
        return Unsupported("arm.adjustment.sequential_cuped")
    if sequential and prior_present:
        return Unsupported("arm.adjustment.sequential_prior")
    if metric.metric_type == "quantile" and cuped:
        return Unsupported("arm.metric.quantile_cuped")
    if metric.metric_type == "quantile" and sequential:
        return Unsupported("arm.metric.quantile_sequential")
    return Supported(
        assumptions=("independent_units",) if dependence == "iid" else ("cluster_robust",),
        floor=_arm_floor(dependence, metric),
        reference="fixed_horizon" if not sequential else "sequential_boundary",
    )


def arm_runtime_support(request: ArmCompatibilityRequest) -> Support:
    """Return support without touching source rows or numerical moments."""
    sequential = not isinstance(request.inference, FixedInference)
    cuped, admissible = _cuped_axes(
        request.methods, asymptotic=isinstance(request.inference, ASYMPTOTIC_PROCEDURE_POLICIES)
    )
    return _arm_support(
        dependence=request.dependence,
        metric=request.metric,
        sequential=sequential,
        cuped=cuped,
        coefficient_admissible=admissible,
        prior_present=request.prior_present,
    )


def _baseline_support(request: ArmPlanningProcedure, baseline: object) -> Support | None:
    def value(name: str, default: float) -> float:
        return getattr(baseline, name, default)

    if request.dependence == "iid":
        knobs = ("cluster_icc", "cluster_size_cv", "avg_cluster_size")
        offending = [
            name
            for name in knobs
            if (value(name, 0.0) != 0.0 if name != "avg_cluster_size" else value(name, 1.0) != 1.0)
        ]
        if getattr(baseline, "cluster_participation", None) is not None:
            offending.append("cluster_participation")
        if offending:
            return Unsupported(
                "arm.baseline.iid_cluster_knobs",
                context={
                    "knob": offending[0],
                    "offending": {name: getattr(baseline, name, None) for name in offending},
                    "route": {
                        "dependence": request.dependence,
                        "population": request.analysis.population,
                    },
                    "requires": "a procedure built with ArmPlanningProcedure.standard(clustered=True) "
                    "(dependence='cluster')",
                },
            )
    if request.dependence == "cluster" and value("avg_cluster_size", 1.0) <= 1.0:
        return Unsupported(
            "arm.baseline.cluster_size_required",
            context={
                "knob": "avg_cluster_size",
                "requires": "Baseline.avg_cluster_size > 1.0 alongside dependence='cluster'",
            },
        )
    if value("icc", 0.0) != 0.0 and request.analysis.variance_adjustment != "factor_absorption":
        return Unsupported(
            "arm.baseline.absorption_undeclared",
            context={
                "knob": "icc",
                "requires": "a procedure whose analysis.variance_adjustment is "
                "'factor_absorption' -- ArmPlanningProcedure.standard() derives this "
                "automatically unless the procedure explicitly sets a different value",
            },
        )
    if value("compliance", 1.0) != 1.0 and request.analysis.identification != "encouragement":
        return Unsupported(
            "arm.baseline.compliance_undeclared",
            context={
                "knob": "compliance",
                "requires": "a procedure whose analysis.identification is 'encouragement' -- "
                "ArmPlanningProcedure.standard() derives this automatically unless the "
                "procedure explicitly sets a different value",
            },
        )
    if value("trigger_rate", 1.0) != 1.0 and request.analysis.population != "triggered":
        return Unsupported(
            "arm.baseline.triggering_undeclared",
            context={
                "knob": "trigger_rate",
                "requires": "a procedure whose analysis.population is 'triggered' -- "
                "ArmPlanningProcedure.standard() derives this automatically unless the "
                "procedure explicitly sets a different value",
            },
        )
    if value("cuped_rho", 0.0) != 0.0 and not any(
        method.variance_reduction != "none"
        for method in (request.decision_method, *request.sensitivity_methods)
    ):
        return Unsupported(
            "arm.baseline.cuped_undeclared",
            context={
                "knob": "cuped_rho",
                "requires": "a procedure whose decision_method declares CUPED -- "
                "ArmPlanningProcedure.standard() derives this automatically unless the "
                "procedure explicitly sets a different decision_method",
            },
        )
    return None


def arm_planning_support(
    request: ArmPlanningProcedure,
    *,
    baseline: object | None = None,
) -> Support:
    if request.analysis.population == "triggered" and isinstance(
        request.inference, GaussianScoreMixture
    ):
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "trigger-selected units require a separate conditional sampling "
            "law; plan a fixed-horizon procedure and monitor with raw "
            "assigned ITT, or use asymptotic_mean without a trigger_rate "
            "Baseline",
        )
    if baseline is not None:
        refusal = _baseline_support(request, baseline)
        if refusal is not None:
            return refusal
    if request.decision.family.kind in ("bh", "e_bh"):
        return Unsupported("arm.metric.unsupported")
    asymptotic = isinstance(request.inference, GaussianScoreMixture)
    translated = tuple(
        _PlanningMethodFlavour(_cuped_flavour_for_planning(m, asymptotic=asymptotic))
        for m in (request.decision_method, *request.sensitivity_methods)
    )
    cuped, admissible = _cuped_axes(translated, asymptotic=asymptotic)
    return _arm_support(
        dependence=request.dependence,
        metric=request.metric,
        sequential=not isinstance(request.inference, FixedInference),
        cuped=cuped,
        coefficient_admissible=admissible,
        prior_present=request.prior_present,
    )


class ArmEvidenceContract:
    family = "arm_moments"

    def runtime_support(self, request: ArmCompatibilityRequest) -> Support:
        return arm_runtime_support(request)

    def planning_support(
        self,
        request: ArmPlanningProcedure,
        *,
        baseline: object | None = None,
    ) -> Support:
        return arm_planning_support(request, baseline=baseline)


class ContrastEvidenceContract:
    family = "contrast"

    def runtime_support(self, request: ContrastCompatibilityRequest) -> Support:
        if request.decision.family.kind != "none":
            return Unsupported("contrast.decision")
        # ``SwitchbackAssignment.sequence`` is a validated two-member
        # discriminated union; every accepted request has a supported scheme.
        scheme = request.assignment.sequence.scheme
        # ContrastCompatibilityRequest fixes inference to FixedInference.
        if request.metric.metric_type not in ("mean", "conversion"):
            return Unsupported("contrast.metric")
        if request.metric.value_scale != "absolute":
            return Unsupported("contrast.decision")
        if scheme == "shared_schedule":
            # The independent replicate is the block, not the roster unit:
            # a shared schedule needs at least two blocks, never two units.
            return Supported(
                assumptions=("no_residual_carryover_after_discarded_steps", "independent_blocks"),
                floor=IndependentBlockFloor(minimum_total_blocks=2),
                reference="block_t",
            )
        return Supported(
            assumptions=("no_residual_carryover_after_discarded_steps", "independent_units"),
            floor=IndependentUnitFloor(minimum_total_units=2),
            reference="unit_t",
        )

    def planning_support(self, request: ContrastCompatibilityRequest) -> Support:
        return self.runtime_support(request)


ARM_EVIDENCE_CONTRACT = ArmEvidenceContract()
CONTRAST_EVIDENCE_CONTRACT = ContrastEvidenceContract()


ArmCompatibilityRequest.model_rebuild()
ArmPlanningProcedure.model_rebuild()
ContrastCompatibilityRequest.model_rebuild()

__all__ = [
    "AbsoluteDecisionPolicy",
    "AnalysisAxes",
    "ARM_COMPATIBILITY_REFUSALS",
    "ARM_EVIDENCE_CONTRACT",
    "ArmCompatibilityRequest",
    "ArmEvidenceContract",
    "ArmPlanningProcedure",
    "ClusterFloor",
    "ContrastCompatibilityRequest",
    "ContrastEvidenceContract",
    "CONTRAST_EVIDENCE_CONTRACT",
    "DecisionPolicy",
    "EvidenceContract",
    "FamilyPolicy",
    "IndependentBlockFloor",
    "IndependentUnitFloor",
    "MethodCapability",
    "MetricCapabilities",
    "PlanningFamilyExpansion",
    "PowerDesign",
    "RelativeDecisionPolicy",
    "SamplingFloor",
    "Support",
    "Supported",
    "UnitFloor",
    "Unsupported",
    "arm_planning_support",
    "arm_runtime_support",
    "cuped_capability",
]
