"""Decision procedures and analysis-state contracts for source execution.

Error control per plan role
---------------------------
``_resolve_declaration`` allocates one level per declared metric; a
selected secondary (fixed-horizon or sequential) is re-leveled once
more at FCR selection time -- see the secondary bullet below.

- primary: ``alpha / n_primaries``, split again across the metric's own
  non-control arms at estimation. Bonferroni familywise control at ``alpha``.
- guardrail: the full ``alpha``, never divided across guardrails, on the
  one-sided adverse tail fixed by the metric's own margin and
  ``preferred_direction`` (see ``_resolve_one``). This is a non-inferiority
  test: rejecting the null (harm at least as large as the margin)
  certifies the metric SAFE. Individually, each guardrail's type-I error
  (falsely certifying SAFE) is bounded by ``alpha``. The compound decision
  "ship only if every declared guardrail is SAFE" is then an
  intersection-union test: its probability of shipping while at least one
  guardrail is truly harmful is bounded by the same ``alpha``, regardless
  of how many guardrails are declared -- so no per-guardrail correction is
  needed for that compound decision's error rate. Dividing ``alpha`` by
  the guardrail count would tighten the compound bound further (to
  ``alpha / k``), but at the cost of making each individual guardrail's
  own test harder to reject, i.e. less power to certify a genuinely safe
  metric -- a real cost with no offsetting benefit the IUT bound needs.
  This package does not compute or emit the compound ship/no-ship
  verdict; it reports each guardrail's own SAFE/HARM label. A caller who
  reads several guardrail labels without applying the
  ship-only-if-all-SAFE rule has only the weaker union bound (``k *
  alpha`` across ``k`` guardrails) on the joint probability that at least
  one label is a false SAFE, not the tighter single-``alpha`` IUT bound.
- secondary: the discovery family at ``q``. A fixed-horizon cell
  estimates at the nominal level; a sequential cell
  (``auto_register_scalar_mean``/``auto_register_bernoulli``) instead
  registers at ``min(alpha, q / m)`` up front. Either way, BH/e-BH
  selects the family at ``q``, and a selected cell's interval is
  re-estimated once more at ``fcr_alpha = min(q * R / m, alpha)`` --
  the plan's nominal ``alpha``, never the (possibly tighter) registered
  per-cell allocation. ``family_guarantee`` reflects the family's
  weakest regime: any asymptotic cell in the family makes every row's
  guarantee asymptotic, never finite-sample. Bonferroni-only splitting
  (``q / n_secondaries`` per metric, no family-selection pass) survives
  only on the breakout/as-of view axis, which this bullet does not
  describe.
- unassigned: ``alpha``, no multiplicity claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    Protocol,
    cast,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from increment._literals import Alternative, Role
from increment._multiplicity import MultiplicityFamily
from increment._plan_compatibility import (
    PlanFamilyCompatibilityRequest,
    plan_family_compatibility,
)
from increment._source_types import MomentSource
from increment._study import ParallelStudyEnvelope, SwitchbackStudyEnvelope
from increment.compatibility import _conservative_divide
from increment.errors import (
    CapabilityError,
    CodedModel,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    raiser,
    refusals,
)
from increment.errors import refuse as _refuse
from increment.errors import warn as _warn_spec
from increment.estimation.decision_types import (
    DECISION_ARM_DECISION_ONE_SIDED_ALPHA,
    ArmHypothesisKey,
    AsymptoticSequentialEvidence,
    ContrastDecisionProcedure,
    ContrastHypothesisKey,
    DecisionComputation,
    DecisionFailure,
    EValueEvidence,
    FixedInference,
    NoFamily,
    PValueEvidence,
    SegmentHypothesisKey,
    TestEvidence,
)
from increment.estimation.engine import Method
from increment.estimation.inference import Prior
from increment.estimation.sequential import (
    ASYMPTOTIC_PROCEDURE_POLICIES,
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
    compose_asymptotic_or_mixed,
)
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    Experiment,
    ExperimentMetric,
    Metric,
    MetricBase,
    MultiplicitySpec,
    PlanEntry,
    resolve_null_and_alternative,
)
from increment.semantics.sequential import SequentialCompliancePolicy
from increment.semantics.unit_cycle import UnitCycleReference, _copy_references

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

    from increment.estimation.contrast import ContrastStats
    from increment.power.switchback import SwitchbackBaseline
    from increment.query.session import WarehouseSession


_SWITCHBACK_DECISION = RefusalSpec(
    "source.frame.switchback.plan",
    CapabilityError,
    template="switchback decision rejected for {reason!r}: {message}",
)


def reject_switchback_decision(*, message: str, **context: object) -> NoReturn:
    """Raise the stable refusal for unsupported switchback decision rules."""
    _refuse(_SWITCHBACK_DECISION, message=message, **context)


Renderer = Callable[..., str]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "decision.family.nofamily_membership_mark": "NoFamily membership cannot mark member=True",
        "decision.relative_arm.null_lift_greater": "relative null_lift must be greater than -1",
        "decision.absolute_arm.procedures_fixed_inference": "absolute procedures require fixed inference",
        "decision.compiled_decision.procedure_mapping_key": "procedure mapping key {key!r} does not match metric {metric!r}",
        "decision.contrast_context.duplicate_metrics": "contrast metric names must be unique: duplicates {metrics}",
        "decision.contrast_context.procedure_metrics": "contrast procedures must match declared metrics: missing {missing}, undeclared {undeclared}",
        "decision.contrast_context.reference_mismatch": "contrast references must match procedure references for metrics {metrics}",
        "plan.metrics.unknown": RefusalSpec(
            "plan.metrics.unknown",
            InvalidRequestError,
            lambda *, path, unknown, declared: (
                f"plan names metrics this {path} does not declare: {sorted(unknown)} "
                f"(declared: {sorted(declared)}) -- refusing rather than silently dropping a role"
            ),
        ),
        "plan.guardrail.direction": "metric {metric!r}: a guardrail role requires an explicitly-set, non-neutral preferred_direction (increase|decrease) on the metric -- a guardrail's non-inferiority margin has no adverse side to test against otherwise, and 'neutral' names no side",
        "plan.frame.override": RefusalSpec(
            "plan.frame.override",
            InvalidRequestError,
            lambda *, metric, field=None: (
                f"plan entry {metric!r}: "
                + (
                    f"{field}= overrides are not supported on the frame path -- declare "
                    "them on the frame's own MetricSpec for this metric instead"
                    if field is not None
                    else "decision_method=/sensitivity_methods=/prior= overrides are not "
                    "supported on the frame path -- declare them on the frame's own "
                    "MetricSpec for this metric instead"
                )
            ),
        ),
        "plan.frame.secondaries": RefusalSpec(
            "plan.frame.secondaries",
            InvalidRequestError,
            lambda *, explicit, derived: (
                f"frame plan secondaries={sorted(explicit)} does not match the default derivation "
                "(declared minus primaries minus guardrails) "
                f"={sorted(derived)} -- a frame plan's secondaries must be the metrics left over, "
                "or omitted entirely"
            ),
        ),
        "plan.metric.report_only": "metric {metric!r} is report-only (no margin/direction concept) and cannot be declared into the {role!r} testing role -- it has no null it can be tested against. Report it without naming it in a plan role.",
    },
)

_REFUSALS["decision.arm_decision.one_sided_alpha"] = DECISION_ARM_DECISION_ONE_SIDED_ALPHA
_raise = raiser(_REFUSALS)


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    _warn_spec(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "decision.family.quantile_always_valid_excluded",
    IncrementWarning,
    lambda *, metric, message: f"metric {metric!r}: {message}",
)


class FamilyMembership(CodedModel, BaseModel):
    """Eligibility is explicit: a prior does not remove sampling evidence from a declared
    family member, and an explicit non-member remains outside it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: NoFamily | MultiplicityFamily = NoFamily()
    member: bool = False

    @property
    def name(self) -> str | None:
        return None if isinstance(self.family, NoFamily) else self.family.name

    @property
    def axes(self) -> tuple[str, ...]:
        return () if isinstance(self.family, NoFamily) else self.family.axes

    @model_validator(mode="after")
    def _member_requires_family(self) -> FamilyMembership:
        if isinstance(self.family, NoFamily) and self.member:
            _raise("decision.family.nofamily_membership_mark")
        return self


class _ArmDecisionProcedure(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    metric: str = Field(min_length=1)
    role: Role
    decision_method: Method
    sensitivity_methods: tuple[Method, ...] = ()
    methods_explicitly_empty: bool = False
    alternative: Alternative
    prior: Prior | None = None
    prior_is_global: bool = True
    alpha: float = Field(gt=0.0, lt=1.0)
    family: FamilyMembership = FamilyMembership()
    inference: AsymptoticMean | AlwaysValid | MixedFamily | FixedInference = FixedInference()

    @model_validator(mode="after")
    def _one_sided_alpha(self) -> _ArmDecisionProcedure:
        if isinstance(self.inference, ASYMPTOTIC_PROCEDURE_POLICIES) and self.family.member:
            family = self.family.family
            if not isinstance(family, MultiplicityFamily) or (
                family.correction != "e_bh"
                or family.guarantee != "fdr"
                or family.validity_regime != "asymptotic_sequential"
                or family.q != float(self.inference.registration.q)
            ):
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "source.invalid", "continuous families require asymptotic e-BH FDR control"
                )
        if self.alternative != "two-sided" and self.alpha >= 0.5:
            _raise("decision.arm_decision.one_sided_alpha")
        return self


class RelativeArmDecisionProcedure(_ArmDecisionProcedure):
    """Decision rule for a relative/log-relative arm contrast."""

    axis: Literal["relative"] = "relative"
    null_lift: float = Field(allow_inf_nan=False)
    scale: Literal["relative"] = "relative"

    @model_validator(mode="after")
    def _valid_null(self) -> RelativeArmDecisionProcedure:
        if self.null_lift <= -1.0 and not isinstance(self.inference, ASYMPTOTIC_PROCEDURE_POLICIES):
            _raise("decision.relative_arm.null_lift_greater")
        return self


class AbsoluteArmDecisionProcedure(_ArmDecisionProcedure):
    """Decision rule for an additive, metric-unit arm contrast."""

    axis: Literal["absolute"] = "absolute"
    null_abs: float = Field(allow_inf_nan=False)
    scale: Literal["absolute"] = "absolute"

    @model_validator(mode="after")
    def _absolute_sequential_refusal(self) -> AbsoluteArmDecisionProcedure:
        if not isinstance(self.inference, FixedInference):
            _raise("decision.absolute_arm.procedures_fixed_inference")
        return self


ArmDecisionProcedure = RelativeArmDecisionProcedure | AbsoluteArmDecisionProcedure


class CompiledViewPolicies(CodedModel, BaseModel):
    """Multiplicity policies fixed before segmented views are read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asof: MultiplicityFamily
    randomized_breakout: MultiplicityFamily
    encouragement_breakout: MultiplicityFamily

    def for_view(
        self,
        view: Literal["asof", "breakout"],
        *,
        mechanism: str | None = None,
        segmented: bool = True,
    ) -> MultiplicityFamily:
        if view == "asof":
            return self.asof if segmented else MultiplicityFamily(name="none")
        if mechanism == "encouragement":
            return self.encouragement_breakout
        return self.randomized_breakout


DecisionProcedure = ArmDecisionProcedure | ContrastDecisionProcedure


class CompiledDecisionPlan(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    declared: bool
    alpha: float = Field(gt=0.0, lt=1.0)
    q: float = Field(gt=0.0, lt=1.0)
    q_explicit: bool = False
    path: Literal["warehouse", "frame", "frame/contrast"]
    inference: AsymptoticMean | AlwaysValid | MixedFamily | FixedInference = FixedInference()
    compliance: SequentialCompliancePolicy | None = None
    procedures: Mapping[str, DecisionProcedure]
    view_policies: CompiledViewPolicies

    @model_validator(mode="after")
    def _compliance_allocation(self):
        from increment.sequential_source import validate_compliance_allocation

        if isinstance(self.inference, SEQUENTIAL_POLICIES):
            validate_compliance_allocation(
                self.inference.registration, self.compliance, self.alpha, self.q
            )
        return self

    @field_validator("procedures")
    @classmethod
    def _freeze_procedures(
        cls, value: Mapping[str, DecisionProcedure]
    ) -> Mapping[str, DecisionProcedure]:
        for key, procedure in value.items():
            if key != procedure.metric:
                _raise(
                    "decision.compiled_decision.procedure_mapping_key",
                    key=key,
                    metric=procedure.metric,
                )
        return MappingProxyType(dict(value))


class ContrastContext(CodedModel, BaseModel):
    """Consistent metric/procedure references, preserving metric order and sorting JSON keys."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, validate_default=True
    )

    study_id: str = Field(min_length=1)
    study: SwitchbackStudyEnvelope
    metrics: tuple[Metric, ...]
    procedures: Mapping[str, ContrastDecisionProcedure]
    contrast_references: Mapping[str, UnitCycleReference] = Field(default_factory=dict)

    @field_serializer("procedures", "contrast_references")
    def _serialize_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(sorted(value.items()))

    @field_validator("contrast_references", mode="before")
    @classmethod
    def _references(cls, value: object) -> Mapping[str, UnitCycleReference]:
        return _copy_references(value)

    @field_validator("contrast_references")
    @classmethod
    def _freeze_references(
        cls, value: Mapping[str, UnitCycleReference]
    ) -> Mapping[str, UnitCycleReference]:
        return MappingProxyType(dict(value))

    @field_validator("procedures")
    @classmethod
    def _freeze_procedures(
        cls, value: Mapping[str, ContrastDecisionProcedure]
    ) -> Mapping[str, ContrastDecisionProcedure]:
        for key, procedure in value.items():
            if key != procedure.metric:
                _raise(
                    "decision.compiled_decision.procedure_mapping_key",
                    key=key,
                    metric=procedure.metric,
                )
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def _consistent_declarations(self) -> ContrastContext:
        counts = Counter(metric.name for metric in self.metrics)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            _raise("decision.contrast_context.duplicate_metrics", metrics=duplicates)
        declared = set(counts)
        if self.procedures.keys() != declared:
            _raise(
                "decision.contrast_context.procedure_metrics",
                missing=sorted(declared - self.procedures.keys()),
                undeclared=sorted(self.procedures.keys() - declared),
            )
        expected = {
            key: procedure.reference
            for key, procedure in self.procedures.items()
            if procedure.reference is not None
        }
        mismatches = sorted(
            key
            for key in expected.keys() | self.contrast_references.keys()
            if expected.get(key) != self.contrast_references.get(key)
        )
        if mismatches:
            _raise("decision.contrast_context.reference_mismatch", metrics=mismatches)
        return self


@runtime_checkable
class ContrastSource(Protocol):
    context: ContrastContext

    def contrast_stats(self, metric: Metric) -> ContrastStats: ...

    def planning_baseline(self, metric: Metric) -> SwitchbackBaseline: ...


@dataclass(frozen=True, slots=True)
class SeamArmAnalysisState:
    source: MomentSource
    study: ParallelStudyEnvelope | None
    experiment_name: str
    family: ClassVar[Literal["arm_moments"]] = "arm_moments"

    def __post_init__(self) -> None:
        design = self.source.context.design
        if design is None:
            if self.study is not None:
                raise AssertionError("an unidentified source cannot have a study")
            return
        if not isinstance(self.study, ParallelStudyEnvelope):
            raise AssertionError("an arm source requires a parallel study")
        if design != self.study.identification:
            raise AssertionError("source and study identification must match")


@dataclass(frozen=True, slots=True)
class DefinitionsArmAnalysisState:
    source: MomentSource
    study: ParallelStudyEnvelope
    definitions: Definitions
    experiment: Experiment
    connection: SQLBackend
    session: WarehouseSession
    experiment_name: str
    backend: str
    store: Literal["auto", "always", "none"]
    on_mixed_assignment: Literal["error", "warn", "exclude"]
    exposure_lookup: Mapping[str, object]
    family: ClassVar[Literal["arm_moments"]] = "arm_moments"

    def __post_init__(self) -> None:
        if not isinstance(self.study, ParallelStudyEnvelope):
            raise AssertionError("a definitions-backed arm source requires a parallel study")
        design = self.source.context.design
        if design is None or design != self.study.identification:
            raise AssertionError("source and study identification must match")


@dataclass(frozen=True, slots=True)
class ContrastAnalysisState:
    source: ContrastSource
    experiment_name: str
    family: ClassVar[Literal["contrast"]] = "contrast"

    def __post_init__(self) -> None:
        context = getattr(self.source, "context", None)
        if not isinstance(context, ContrastContext):
            raise AssertionError("a contrast source requires a contrast context")
        if not isinstance(context.study, SwitchbackStudyEnvelope):
            raise AssertionError("a contrast source requires a switchback study")
        if not callable(getattr(self.source, "contrast_stats", None)):
            raise AssertionError("a contrast source must provide contrast_stats")


ArmAnalysisState = SeamArmAnalysisState | DefinitionsArmAnalysisState
AnalysisState = ArmAnalysisState | ContrastAnalysisState


HypothesisKey = ArmHypothesisKey | SegmentHypothesisKey | ContrastHypothesisKey


def compile_contrast_procedures(
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    *,
    path: Literal["frame", "frame/contrast"] = "frame/contrast",
    contrast_references: Mapping[str, UnitCycleReference] | None = None,
) -> Mapping[str, ContrastDecisionProcedure]:
    """Compile one fixed, family-free decision rule per metric."""
    references = _copy_references(contrast_references)
    unknown = references.keys() - {metric.name for metric in metrics}
    if unknown:
        reject_switchback_decision(
            message="reference for undeclared metric",
            reason="undeclared_reference",
            metrics=sorted(unknown),
        )
    if plan is not None:
        if plan.q != 0.10:
            reject_switchback_decision(
                message="switchback contrasts do not support an explicitly supplied plan q",
                reason="multiplicity_q",
            )
        if plan.view_multiplicity is not None:
            reject_switchback_decision(
                message="switchback contrasts do not support view multiplicity",
                reason="view_multiplicity",
            )
        for entry in plan.entries():
            if isinstance(entry, str):
                continue
            unsupported: list[str] = []
            if entry.decision_method is not None:
                unsupported.append("decision_method")
            if entry.sensitivity_methods:
                unsupported.append("sensitivity_methods")
            if entry.prior is not None:
                unsupported.append("prior")
            if unsupported:
                reject_switchback_decision(
                    message=(
                        f"metric {entry.metric!r}: switchback contrasts do not support "
                        f"plan-bound {', '.join(unsupported)} overrides"
                    ),
                    reason="unsupported_plan_override",
                    metric=entry.metric,
                    fields=unsupported,
                )
    relative_metrics = {
        metric.name for metric in metrics if getattr(metric, "margin", None) is not None
    }
    if plan is not None:
        relative_metrics.update(
            entry.metric
            for entry in plan.entries()
            if not isinstance(entry, str) and entry.margin is not None
        )
    if relative_metrics:
        reject_switchback_decision(
            message=(
                "switchback contrasts do not support relative margins; "
                f"metrics: {sorted(relative_metrics)}"
            ),
            reason="relative_margin",
            metrics=sorted(relative_metrics),
        )
    if plan is not None and plan.inference is not None:
        # Automatic kinds bind no registration on this path, so refuse on the declared kind.
        reject_switchback_decision(
            message="switchback contrasts support fixed inference only",
            reason="unsupported_inference",
            inference=plan.inference.kind,
        )

    declaration = _resolve_declaration(plan, metrics, path=path)
    family_metrics = [
        procedure.metric for procedure in declaration.procedures.values() if procedure.in_family
    ]
    if family_metrics:
        reject_switchback_decision(
            message=(
                "switchback contrasts do not support secondary-family inference; "
                f"family metrics: {sorted(family_metrics)}"
            ),
            reason="family_inference",
            metrics=sorted(family_metrics),
        )
    procedures: dict[str, ContrastDecisionProcedure] = {}
    for metric in metrics:
        name = metric.name
        if name in procedures:
            reject_switchback_decision(
                message=f"duplicate metric {name!r} in contrast context",
                reason="duplicate_metric",
                metric=name,
            )
        test = declaration.procedures[name]
        if test.null_lift != 0.0:
            reject_switchback_decision(
                message=f"metric {name!r}: switchback contrasts require an additive null",
                reason="non_additive_null",
                metric=name,
                null_lift=test.null_lift,
            )
        procedures[name] = ContrastDecisionProcedure(
            metric=name,
            role=test.role,
            alternative=test.alternative,
            null_abs=test.null_abs or 0.0,
            alpha=test.alpha,
            preferred_direction=metric.declared_preferred_direction,
            reference=references.get(name),
        )
    return MappingProxyType(dict(procedures))


@dataclass(frozen=True, slots=True)
class _ProcedureResolution:
    """Intermediate declaration resolution consumed by the wire-free compiler."""

    metric: str
    role: Role
    alpha: float
    null_lift: float
    null_abs: float | None
    alternative: Alternative
    in_family: bool


@dataclass(frozen=True, slots=True)
class _PlanResolution:
    declared: bool
    alpha: float
    q: float
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None
    procedures: Mapping[str, Any]
    view_multiplicity: MultiplicitySpec | None = None


def _entry_name(entry: PlanEntry) -> str:
    return entry if isinstance(entry, str) else entry.metric


def _entry_binding(entry: PlanEntry) -> ExperimentMetric | None:
    return entry if isinstance(entry, ExperimentMetric) else None


def _build_inference(plan: AnalysisPlan) -> AsymptoticMean | AlwaysValid | MixedFamily | None:
    spec = plan.inference
    if spec is None:
        return None
    assert spec.registration is not None
    if spec.kind == "asymptotic_mean":
        return compose_asymptotic_or_mixed(spec.registration)
    return AlwaysValid(registration=spec.registration)


def _declared_alpha(plan: AnalysisPlan, role: Role, *, n_primaries: int) -> float:
    """The level one declared metric is tested at, before any arm splitting."""
    if role == "primary":
        return _conservative_divide(plan.alpha, n_primaries)
    return plan.alpha


def _resolve_declaration(
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    *,
    path: Literal["warehouse", "frame", "frame/contrast"],
) -> _PlanResolution:
    """Resolve declaration roles and nulls before building runtime procedures.

    Warehouse paths leave undeclared catalog metrics unassigned; frame paths
    derive secondary membership from the metrics left after primaries and
    guardrails.
    """
    metrics_by_name = {m.name: m for m in metrics}

    if plan is None:
        tests = {
            name: _resolve_one(
                name,
                metric,
                role="unassigned",
                alpha=0.05,
                alternative="two-sided",
                binding=None,
                inference=None,
            )
            for name, metric in metrics_by_name.items()
        }
        return _PlanResolution(
            declared=False,
            alpha=0.05,
            q=0.10,
            inference=None,
            procedures=tests,
        )

    primary_entries = plan.primaries
    guardrail_entries = plan.guardrails
    primary_names = [_entry_name(e) for e in primary_entries]
    guardrail_names = [_entry_name(e) for e in guardrail_entries]

    explicit_secondary_entries = plan.secondaries
    plan_names = set(primary_names) | set(guardrail_names)
    if explicit_secondary_entries is not None:
        plan_names |= {_entry_name(e) for e in explicit_secondary_entries}

    unknown = plan_names - set(metrics_by_name)
    if unknown:
        _raise(
            "plan.metrics.unknown",
            path=path,
            unknown=unknown,
            declared=metrics_by_name,
        )

    for name in guardrail_names:
        guardrail_metric = metrics_by_name[name]
        if (
            "preferred_direction" not in guardrail_metric.model_fields_set
            or guardrail_metric.preferred_direction in (None, "neutral")
        ):
            _raise("plan.guardrail.direction", metric=name)

    if path in ("frame", "frame/contrast"):
        for entry in (*primary_entries, *(explicit_secondary_entries or []), *guardrail_entries):
            binding = _entry_binding(entry)
            if binding is not None:
                unsupported = []
                if binding.decision_method is not None:
                    unsupported.append("decision_method")
                if binding.sensitivity_methods:
                    unsupported.append("sensitivity_methods")
                if binding.prior is not None:
                    unsupported.append("prior")
                if unsupported:
                    _raise(
                        "plan.frame.override",
                        metric=binding.metric,
                        field=unsupported[0] if len(unsupported) == 1 else None,
                    )

        declared = set(metrics_by_name)
        default_secondary_names = declared - set(primary_names) - set(guardrail_names)
        if explicit_secondary_entries is not None:
            explicit_names = {_entry_name(e) for e in explicit_secondary_entries}
            if explicit_names != default_secondary_names:
                _raise(
                    "plan.frame.secondaries",
                    explicit=explicit_names,
                    derived=default_secondary_names,
                )
        secondary_names = default_secondary_names
        secondary_entries: Sequence[PlanEntry] = (
            explicit_secondary_entries
            if explicit_secondary_entries is not None
            else [cast(PlanEntry, name) for name in sorted(secondary_names)]
        )
        role_of: dict[str, Role] = dict.fromkeys(primary_names, cast(Role, "primary"))
        role_of.update(dict.fromkeys(guardrail_names, cast(Role, "guardrail")))
        role_of.update(dict.fromkeys(secondary_names, cast(Role, "secondary")))
    else:
        secondary_entries = explicit_secondary_entries or []
        secondary_names = {_entry_name(e) for e in secondary_entries}
        role_of = dict.fromkeys(primary_names, cast(Role, "primary"))
        role_of.update(dict.fromkeys(guardrail_names, cast(Role, "guardrail")))
        role_of.update(dict.fromkeys(secondary_names, cast(Role, "secondary")))
        for name in metrics_by_name:
            role_of.setdefault(name, "unassigned")

    binding_by_name: dict[str, ExperimentMetric] = {}
    for entry in (*primary_entries, *secondary_entries, *guardrail_entries):
        binding = _entry_binding(entry)
        if binding is not None:
            binding_by_name[binding.metric] = binding
    inference = _build_inference(plan)
    n_primaries = len(primary_names)

    procedures = {
        name: _resolve_one(
            name,
            metric,
            role=role_of[name],
            alpha=_declared_alpha(plan, role_of[name], n_primaries=n_primaries),
            alternative=plan.alternative,
            binding=binding_by_name.get(name),
            inference=inference,
        )
        for name, metric in metrics_by_name.items()
    }
    return _PlanResolution(
        declared=True,
        alpha=plan.alpha,
        q=plan.q,
        inference=inference,
        procedures=procedures,
        view_multiplicity=plan.view_multiplicity,
    )


def _resolve_one(
    name: str,
    metric: Metric,
    *,
    role: Role,
    alpha: float,
    alternative: Alternative,
    binding: ExperimentMetric | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
) -> _ProcedureResolution:
    if isinstance(metric, MetricBase):
        margins = (
            {name: binding.margin} if binding is not None and binding.margin is not None else None
        )
        margins_abs = (
            {name: binding.margin_abs}
            if binding is not None and binding.margin_abs is not None
            else None
        )
        # A guardrail's tail follows its own margin and preferred_direction; the
        # plan `alternative` applies only to primaries and secondaries. Passing
        # "two-sided" makes `resolve_null_and_alternative` use the margin tail;
        # the guardrail block below handles the margin-less case.
        null_lift, null_abs, effective_alt = resolve_null_and_alternative(
            metric,
            margins,
            None,
            "two-sided" if role == "guardrail" else alternative,
            margins_abs=margins_abs,
        )
    else:
        # MetricIdentity-only metrics (currently just TotalMetric) carry no
        # margin fields (no margin/margin_abs) -- not all report-only
        # metrics fall here (ActiveMetric is report-only too but IS a
        # MetricBase and takes the branch above).
        if role != "unassigned":
            _raise("plan.metric.report_only", metric=name, role=role)
        null_lift, null_abs, effective_alt = 0.0, None, alternative
    if role == "guardrail" and null_lift == 0.0 and null_abs is None:
        # A margin-less guardrail still uses the adverse side declared by
        # its metric rather than the plan-wide alternative.
        effective_alt = "greater" if metric.preferred_direction == "increase" else "less"

    inference_kind = (
        "asymptotic_mean"
        if isinstance(inference, ASYMPTOTIC_PROCEDURE_POLICIES)
        else "always_valid"
        if isinstance(inference, AlwaysValid)
        else "fixed"
    )
    family_decision = plan_family_compatibility(
        PlanFamilyCompatibilityRequest(
            metric=name,
            metric_type=metric.type,
            role=role,
            inference=inference_kind,
        )
    )
    if family_decision.warning is not None:
        _warn(
            "decision.family.quantile_always_valid_excluded",
            metric=name,
            message=family_decision.warning,
            stacklevel=3,
        )
    in_family = family_decision.participation == "participates"

    return _ProcedureResolution(
        metric=name,
        role=role,
        alpha=alpha,
        null_lift=null_lift,
        null_abs=null_abs,
        alternative=cast(Alternative, effective_alt),
        in_family=in_family,
    )


__all__ = [
    "AbsoluteArmDecisionProcedure",
    "AnalysisState",
    "ArmAnalysisState",
    "ArmDecisionProcedure",
    "CompiledDecisionPlan",
    "CompiledViewPolicies",
    "ContrastAnalysisState",
    "ContrastContext",
    "ContrastDecisionProcedure",
    "ContrastSource",
    "DecisionProcedure",
    "DefinitionsArmAnalysisState",
    "FamilyMembership",
    "FixedInference",
    "MultiplicityFamily",
    "NoFamily",
    "RelativeArmDecisionProcedure",
    "SeamArmAnalysisState",
    "compile_contrast_procedures",
    "ArmHypothesisKey",
    "ContrastHypothesisKey",
    "DecisionComputation",
    "DecisionFailure",
    "AsymptoticSequentialEvidence",
    "EValueEvidence",
    "HypothesisKey",
    "PValueEvidence",
    "SegmentHypothesisKey",
    "TestEvidence",
]
