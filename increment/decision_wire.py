"""Pydantic wire DTOs for compiled decision plans.

The wire model is deliberately narrower than the runtime model.  A moments
cube can carry declaration-safe method names and Normal priors, but never
callable learners, fold state, or runtime-only prior implementations. Fixed
horizon plans remain on wire version 2; registered sequential plans use wire
version 3 because their registration includes the source mapping and explicit
compliance policy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from increment._immutable import _FrozenMapping
from increment._literals import (
    Alternative,
    ConversionInference,
    MultiplicityCorrection,
    PreferredDirection,
    Role,
    ValueScale,
)
from increment.decision import (
    AbsoluteArmDecisionProcedure,
    CompiledDecisionPlan,
    CompiledViewPolicies,
    ContrastDecisionProcedure,
    DecisionProcedure,
    FamilyMembership,
    FixedInference,
    MultiplicityFamily,
    NoFamily,
    RelativeArmDecisionProcedure,
    _guarantee_for_correction,
)
from increment.errors import CodedError, CodedModel, WireFormatError
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.estimation.sequential import (
    ASYMPTOTIC_PROCEDURE_POLICIES,
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
    compose_asymptotic_or_mixed,
)
from increment.semantics.models import NormalPriorSpec
from increment.semantics.rational import _JSON_RATIONAL_CONTEXT, RATIONAL_INVALID
from increment.semantics.sequential import (
    SequentialCompliancePolicy,
    SequentialRegistration,
    refuse_legacy_asymptotic_family,
)
from increment.semantics.unit_cycle import UnitCycleReference


class _WireBase(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WireMethod(_WireBase):
    name: str = Field(min_length=1)
    variance_reduction: str = "none"
    conversion_inference: ConversionInference = "auto"


class WireFixedInference(_WireBase):
    kind: Literal["fixed"] = "fixed"


class WireAlwaysValid(_WireBase):
    kind: Literal["always_valid"] = "always_valid"
    registration: SequentialRegistration


class WireAsymptoticMean(_WireBase):
    kind: Literal["asymptotic_mean"] = "asymptotic_mean"
    registration: SequentialRegistration


WireInference = Annotated[
    WireFixedInference | WireAsymptoticMean | WireAlwaysValid,
    Field(discriminator="kind"),
]


class WireNoFamily(_WireBase):
    kind: Literal["none"] = "none"


class WireMultiplicityFamily(_WireBase):
    kind: Literal["multiplicity"] = "multiplicity"
    name: str = Field(min_length=1)
    correction: MultiplicityCorrection = "none"
    q: float | None = Field(default=None, gt=0, lt=1)
    axes: tuple[str, ...] = ()
    guarantee: Literal["none", "fwer", "fdr"] = "none"
    validity_regime: Literal["finite_sample", "asymptotic_sequential"] = Field(
        default="finite_sample", exclude_if=lambda value: value == "finite_sample"
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> WireMultiplicityFamily:
        if self.validity_regime == "asymptotic_sequential" and self.correction not in (
            "none",
            "bonferroni",
            "e_bh",
        ):
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported",
                "asymptotic families support fixed-roster Bonferroni or e-BH selection only",
            )
        uses_q = self.correction in ("bh", "e_bh") or (
            self.correction == "bonferroni" and self.validity_regime == "asymptotic_sequential"
        )
        if uses_q and self.q is None:
            _raise(
                "wire.multiplicity.validate_policy",
                f"{self.correction} multiplicity requires q",
                correction=self.correction,
            )
        if not uses_q and self.q is not None:
            _raise(
                "wire.multiplicity.bh",
                f"q is only valid for BH multiplicity, got {self.correction!r}",
                correction=self.correction,
            )
        expected_guarantee = _guarantee_for_correction(self.correction)
        if self.guarantee != expected_guarantee:
            _raise(
                "wire.multiplicity.guarantee_mismatch",
                f"multiplicity guarantee {self.guarantee!r} does not match "
                f"correction {self.correction!r}",
                correction=self.correction,
                guarantee=self.guarantee,
            )
        return self

    @classmethod
    def from_runtime(cls, family: MultiplicityFamily) -> WireMultiplicityFamily:
        return cls(
            name=family.name,
            correction=family.correction,
            q=family.q,
            axes=family.axes,
            guarantee=family.guarantee,
            validity_regime=family.validity_regime,
        )


WireFamily = Annotated[WireNoFamily | WireMultiplicityFamily, Field(discriminator="kind")]


class WireFamilyMembership(_WireBase):
    family: WireFamily = WireNoFamily()
    member: bool = False


class _WireArmProcedure(_WireBase):
    kind: ValueScale
    metric: str = Field(min_length=1)
    role: Role
    decision_method: WireMethod
    sensitivity_methods: tuple[WireMethod, ...] = ()
    methods_explicitly_empty: bool = False
    alternative: Alternative
    prior: NormalPriorSpec | None = None
    prior_is_global: bool = True
    alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    family: WireFamilyMembership = WireFamilyMembership()
    inference: WireInference


class WireRelativeArmProcedure(_WireArmProcedure):
    kind: Literal["relative"] = "relative"
    axis: Literal["relative"] = "relative"
    scale: Literal["relative"] = "relative"
    null_lift: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _null_domain(self):
        if self.null_lift <= -1 and not isinstance(self.inference, WireAsymptoticMean):
            _raise(
                "wire.procedure.invalid_inference", "this inference requires a positive null ratio"
            )
        return self


class WireAbsoluteArmProcedure(_WireArmProcedure):
    kind: Literal["absolute"] = "absolute"
    axis: Literal["absolute"] = "absolute"
    scale: Literal["absolute"] = "absolute"
    inference: WireFixedInference
    null_abs: float = Field(allow_inf_nan=False)


class WireContrastProcedure(_WireBase):
    kind: Literal["contrast"] = "contrast"
    metric: str = Field(min_length=1)
    role: Role
    alternative: Alternative
    null_abs: float = Field(allow_inf_nan=False)
    alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    preferred_direction: PreferredDirection | None = None
    reference: UnitCycleReference | None = None
    family: WireNoFamily = WireNoFamily()
    inference: WireFixedInference = WireFixedInference()


WireProcedure = Annotated[
    WireRelativeArmProcedure | WireAbsoluteArmProcedure | WireContrastProcedure,
    Field(discriminator="kind"),
]


class WireViewPolicies(_WireBase):
    asof: WireMultiplicityFamily
    randomized_breakout: WireMultiplicityFamily
    encouragement_breakout: WireMultiplicityFamily


class WireCompiledDecisionPlan(_WireBase):
    wire_version: Literal[2, 3] = 2
    declared: bool
    alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    q: float = Field(gt=0, lt=1, allow_inf_nan=False)
    path: Literal["warehouse", "frame", "frame/contrast"]
    inference: WireInference
    compliance: SequentialCompliancePolicy | None = None
    procedures: Mapping[str, WireProcedure]
    view_policies: WireViewPolicies

    @field_validator("procedures")
    @classmethod
    def _freeze_procedures(cls, value: Mapping[str, WireProcedure]) -> Mapping[str, WireProcedure]:
        return _FrozenMapping(value)

    @field_serializer("procedures")
    def _serialize_procedures(self, value: Mapping[str, WireProcedure]) -> dict[str, WireProcedure]:
        return dict(value)

    @model_validator(mode="before")
    @classmethod
    def _wire_version_matches_inference(cls, value):
        if isinstance(value, Mapping):
            inference = value.get("inference")
            kind = (
                inference.get("kind")
                if isinstance(inference, Mapping)
                else getattr(inference, "kind", None)
            )
            wire_version = value.get("wire_version", 2)
            if kind == "fixed" and wire_version != 2:
                _raise(
                    "wire.payload.invalid",
                    "fixed-horizon plans require wire version 2",
                )
            if kind in ("always_valid", "asymptotic_mean") and wire_version != 3:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "continuation.legacy",
                    "legacy sequential plans cannot replay without the current observation mapping and compliance policy",
                )
            if kind == "asymptotic_mean" and isinstance(inference, Mapping):
                refuse_legacy_asymptotic_family(inference.get("registration"))
        return value

    @model_validator(mode="after")
    def _sequential_wire_version(self) -> WireCompiledDecisionPlan:
        if isinstance(self.inference, (WireAsymptoticMean, WireAlwaysValid)):
            if self.wire_version != 3:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "continuation.legacy",
                    "legacy sequential plans cannot replay without the current observation mapping and compliance policy",
                )
        elif self.wire_version != 2:
            _raise("wire.payload.invalid", "fixed-horizon plans require wire version 2")
        return self

    @model_validator(mode="after")
    def _validate_compiled_invariants(self) -> WireCompiledDecisionPlan:
        if isinstance(self.inference, (WireAsymptoticMean, WireAlwaysValid)):
            from increment.errors import CapabilityError
            from increment.sequential_source import validate_compliance_allocation

            try:
                validate_compliance_allocation(
                    self.inference.registration, self.compliance, self.alpha, self.q
                )
            except CapabilityError as exc:
                _raise("wire.compiled_plan.compliance_mismatch", str(exc))
        primary_count = sum(procedure.role == "primary" for procedure in self.procedures.values())
        if primary_count:
            from increment.compatibility import _conservative_divide

            primary_expected = _conservative_divide(self.alpha, primary_count)
        else:
            primary_expected = None
        for procedure in self.procedures.values():
            expected = primary_expected if procedure.role == "primary" else self.alpha
            if expected is not None and procedure.alpha != expected:
                _raise(
                    "wire.compiled_plan.procedure_alpha_mismatch",
                    "procedure alpha does not match compiled plan allocation",
                )
        top_inference = self.inference.model_dump(mode="json")
        for procedure in self.procedures.values():
            if procedure.inference.model_dump(mode="json") != top_inference:
                _raise(
                    "wire.compiled_plan.procedure_inference_mismatch",
                    "procedure inference does not match compiled plan inference",
                )
        expected_kind = "contrast" if self.path == "frame/contrast" else None
        if expected_kind is not None:
            if any(procedure.kind != expected_kind for procedure in self.procedures.values()):
                _raise(
                    "wire.compiled_plan.frame_contrast_requires_contrast",
                    "frame/contrast plans require contrast procedures",
                )
        elif any(procedure.kind == "contrast" for procedure in self.procedures.values()):
            _raise(
                "wire.compiled_plan.arm_plan_requires_arm",
                "arm plans require arm procedures",
            )
        return self


def _raise(code: str, message: str, **context: object) -> NoReturn:
    raise WireFormatError(message, code=code, context=context)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _method_to_wire(method: Method, *, metric: str, role: str) -> WireMethod:
    if (
        method.propensity_learner is not None
        or method.outcome_learner is not None
        or method.folds is not None
    ):
        _raise(
            "wire.procedure.runtime_method",
            "runtime method configuration cannot be carried through the moments wire format",
            metric=metric,
            role=role,
            method=method.name,
        )
    return WireMethod(
        name=method.name,
        variance_reduction=method.variance_reduction,
        conversion_inference=method.conversion_inference,
    )


def _method_from_wire(method: WireMethod, *, metric: str, role: str) -> Method:
    try:
        return Method(
            name=method.name,
            variance_reduction=method.variance_reduction,
            conversion_inference=method.conversion_inference,
        )
    except (TypeError, ValueError) as exc:
        _raise(
            "wire.procedure.invalid_method",
            f"invalid {role} method for metric {metric!r}: {exc}",
            metric=metric,
            role=role,
            method=method.name,
        )


def _prior_to_wire(prior: object, *, metric: str, role: str) -> NormalPriorSpec:
    if not isinstance(prior, Normal):
        _raise(
            "wire.procedure.runtime_prior",
            "runtime-only priors cannot be carried through the moments wire format",
            metric=metric,
            role=role,
            prior=type(prior).__name__,
        )
    if not _finite(prior.mu) or not _finite(prior.sigma) or prior.sigma <= 0:
        _raise(
            "wire.procedure.invalid_prior",
            "Normal prior parameters must be finite and sigma must be positive",
            metric=metric,
            role=role,
        )
    return NormalPriorSpec(mu=prior.mu, sigma=prior.sigma)


def _prior_from_wire(prior: NormalPriorSpec) -> Normal:
    if not _finite(prior.mu) or not _finite(prior.sigma) or prior.sigma <= 0:
        _raise(
            "wire.procedure.invalid_prior",
            "Normal prior parameters must be finite and sigma must be positive",
        )
    return Normal(mu=prior.mu, sigma=prior.sigma)


def _inference_to_wire(inference: object) -> WireInference:
    if isinstance(inference, FixedInference):
        return WireFixedInference()
    if isinstance(inference, ASYMPTOTIC_PROCEDURE_POLICIES):
        return WireAsymptoticMean(registration=inference.registration)
    if isinstance(inference, AlwaysValid):
        return WireAlwaysValid(registration=inference.registration)
    _raise(
        "wire.procedure.invalid_inference",
        f"unsupported runtime inference {type(inference).__name__}",
        inference=type(inference).__name__,
    )


def _inference_from_wire(
    inference: WireInference,
) -> FixedInference | AsymptoticMean | AlwaysValid | MixedFamily:
    try:
        if isinstance(inference, WireFixedInference):
            return FixedInference()
        if isinstance(inference, WireAsymptoticMean):
            return compose_asymptotic_or_mixed(inference.registration)
        if isinstance(inference, WireAlwaysValid):
            return AlwaysValid(registration=inference.registration)
    except (TypeError, ValueError) as exc:
        _raise(
            "wire.procedure.invalid_inference",
            f"invalid inference payload: {exc}",
            inference=getattr(inference, "kind", None),
        )
    _raise(
        "wire.procedure.invalid_inference",
        f"unsupported wire inference {type(inference).__name__}",
        inference=getattr(inference, "kind", None),
    )


def _family_to_wire(family: FamilyMembership) -> WireFamilyMembership:
    if isinstance(family.family, NoFamily):
        return WireFamilyMembership(family=WireNoFamily(), member=family.member)
    return WireFamilyMembership(
        family=WireMultiplicityFamily.from_runtime(family.family), member=family.member
    )


def _family_from_wire(
    family: WireFamilyMembership, *, prior_bound_secondary: bool
) -> FamilyMembership:
    if isinstance(family.family, WireNoFamily):
        return FamilyMembership(family=NoFamily(), member=family.member)
    return FamilyMembership(
        family=MultiplicityFamily(
            name=family.family.name,
            correction=family.family.correction,
            q=family.family.q,
            axes=family.family.axes,
            guarantee=family.family.guarantee,
            validity_regime=family.family.validity_regime,
        ),
        # A bound prior excludes the readout, not its declared named family.
        member=family.member or prior_bound_secondary,
    )


def _procedure_to_wire(procedure: object) -> WireProcedure:
    if isinstance(procedure, (RelativeArmDecisionProcedure, AbsoluteArmDecisionProcedure)):
        common: dict[str, Any] = {
            "metric": procedure.metric,
            "role": procedure.role,
            "decision_method": _method_to_wire(
                procedure.decision_method, metric=procedure.metric, role="decision"
            ),
            "sensitivity_methods": tuple(
                _method_to_wire(method, metric=procedure.metric, role="sensitivity")
                for method in procedure.sensitivity_methods
            ),
            "methods_explicitly_empty": procedure.methods_explicitly_empty,
            "alternative": procedure.alternative,
            "prior": (
                _prior_to_wire(procedure.prior, metric=procedure.metric, role="prior")
                if procedure.prior is not None
                else None
            ),
            "prior_is_global": procedure.prior_is_global,
            "alpha": procedure.alpha,
            "family": _family_to_wire(procedure.family),
            "inference": _inference_to_wire(procedure.inference),
        }
        if isinstance(procedure, RelativeArmDecisionProcedure):
            return WireRelativeArmProcedure(
                **common,
                null_lift=procedure.null_lift,
            )
        return WireAbsoluteArmProcedure(**common, null_abs=procedure.null_abs)
    if isinstance(procedure, ContrastDecisionProcedure):
        return WireContrastProcedure(
            metric=procedure.metric,
            role=procedure.role,
            alternative=procedure.alternative,
            null_abs=procedure.null_abs,
            alpha=procedure.alpha,
            preferred_direction=procedure.preferred_direction,
            reference=procedure.reference,
        )
    _raise(
        "wire.procedure.invalid",
        f"unsupported runtime decision procedure {type(procedure).__name__}",
        procedure=type(procedure).__name__,
    )


def _procedure_from_wire(procedure: WireProcedure) -> DecisionProcedure:
    if isinstance(procedure, WireContrastProcedure):
        return ContrastDecisionProcedure(
            metric=procedure.metric,
            role=procedure.role,
            alternative=procedure.alternative,
            null_abs=procedure.null_abs,
            alpha=procedure.alpha,
            preferred_direction=procedure.preferred_direction,
            reference=procedure.reference,
        )
    method_names = [
        procedure.decision_method.name,
        *(m.name for m in procedure.sensitivity_methods),
    ]
    if len(method_names) != len(set(method_names)):
        _raise(
            "wire.procedure.invalid_method",
            f"duplicate method names for metric {procedure.metric!r}",
            metric=procedure.metric,
            role="decision/sensitivity",
        )
    common: dict[str, Any] = {
        "metric": procedure.metric,
        "role": procedure.role,
        "decision_method": _method_from_wire(
            procedure.decision_method, metric=procedure.metric, role="decision"
        ),
        "sensitivity_methods": tuple(
            _method_from_wire(method, metric=procedure.metric, role="sensitivity")
            for method in procedure.sensitivity_methods
        ),
        "methods_explicitly_empty": procedure.methods_explicitly_empty,
        "alternative": procedure.alternative,
        "prior": _prior_from_wire(procedure.prior) if procedure.prior is not None else None,
        "prior_is_global": procedure.prior_is_global,
        "alpha": procedure.alpha,
        "family": _family_from_wire(
            procedure.family,
            prior_bound_secondary=(
                procedure.role == "secondary"
                and procedure.prior is not None
                and isinstance(procedure.inference, WireFixedInference)
            ),
        ),
        "inference": _inference_from_wire(procedure.inference),
    }
    if isinstance(procedure, WireRelativeArmProcedure):
        return RelativeArmDecisionProcedure(**common, null_lift=procedure.null_lift)
    return AbsoluteArmDecisionProcedure(**common, null_abs=procedure.null_abs)


def compiled_plan_to_dto(plan: CompiledDecisionPlan) -> WireCompiledDecisionPlan:
    return WireCompiledDecisionPlan(
        wire_version=3 if isinstance(plan.inference, SEQUENTIAL_POLICIES) else 2,
        declared=plan.declared,
        alpha=plan.alpha,
        q=plan.q,
        path=plan.path,
        inference=_inference_to_wire(plan.inference),
        compliance=plan.compliance,
        procedures={
            name: _procedure_to_wire(procedure) for name, procedure in plan.procedures.items()
        },
        view_policies=WireViewPolicies(
            asof=WireMultiplicityFamily.from_runtime(plan.view_policies.asof),
            randomized_breakout=WireMultiplicityFamily.from_runtime(
                plan.view_policies.randomized_breakout
            ),
            encouragement_breakout=WireMultiplicityFamily.from_runtime(
                plan.view_policies.encouragement_breakout
            ),
        ),
    )


def compiled_plan_from_dto(dto: WireCompiledDecisionPlan) -> CompiledDecisionPlan:
    try:
        dto = type(dto).model_validate(dto.model_dump(warnings=False))
        procedures = {
            name: _procedure_from_wire(procedure) for name, procedure in dto.procedures.items()
        }
        for name, procedure in procedures.items():
            if name != procedure.metric:
                _raise(
                    "wire.procedure.metric_mismatch",
                    f"procedure mapping key {name!r} does not match metric {procedure.metric!r}",
                    name=name,
                    metric=procedure.metric,
                )
        return CompiledDecisionPlan(
            declared=dto.declared,
            alpha=dto.alpha,
            q=dto.q,
            path=dto.path,
            inference=_inference_from_wire(dto.inference),
            compliance=dto.compliance,
            procedures=procedures,
            view_policies=CompiledViewPolicies(
                asof=MultiplicityFamily(**dto.view_policies.asof.model_dump(exclude={"kind"})),
                randomized_breakout=MultiplicityFamily(
                    **dto.view_policies.randomized_breakout.model_dump(exclude={"kind"})
                ),
                encouragement_breakout=MultiplicityFamily(
                    **dto.view_policies.encouragement_breakout.model_dump(exclude={"kind"})
                ),
            ),
        )
    except WireFormatError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, CodedError) and exc.code == RATIONAL_INVALID.code:
            raise
        _raise("wire.payload.invalid", f"invalid compiled decision plan: {exc}", error=str(exc))


def compiled_plan_to_dict(plan: CompiledDecisionPlan) -> dict[str, object]:
    return compiled_plan_to_dto(plan).model_dump(mode="json")


def _refuse_legacy_plan(payload: Mapping[str, object]) -> None:
    inference = payload.get("inference")
    kind = inference.get("kind") if isinstance(inference, Mapping) else None
    if kind == "group_sequential":
        _raise(
            "wire.procedure.invalid_inference",
            "group-sequential inference was removed; an always-valid plan monitors every look",
            inference="group_sequential",
        )
    if kind in ("always_valid", "asymptotic_mean") and payload.get("wire_version") != 3:
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "continuation.legacy",
            "legacy sequential plans have no committed raw-likelihood model or checkpoint",
        )
    if kind == "asymptotic_mean" and isinstance(inference, Mapping):
        refuse_legacy_asymptotic_family(inference.get("registration"))


def _decode_plan(
    payload: Mapping[str, object], *, json_input: bool = False
) -> CompiledDecisionPlan:
    _refuse_legacy_plan(payload)
    try:
        dto = WireCompiledDecisionPlan.model_validate(
            payload, context=_JSON_RATIONAL_CONTEXT if json_input else None
        )
    except WireFormatError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, CodedError) and exc.code == RATIONAL_INVALID.code:
            raise
        _raise("wire.payload.invalid", f"invalid compiled decision plan payload: {exc}")
    return compiled_plan_from_dto(dto)


def compiled_plan_from_dict(payload: Mapping[str, object]) -> CompiledDecisionPlan:
    return _decode_plan(payload)


def compiled_plan_to_json(plan: CompiledDecisionPlan) -> str:
    return json.dumps(compiled_plan_to_dict(plan), sort_keys=True, separators=(",", ":"))


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _raise("wire.payload.duplicate_key", f"duplicate JSON key {key!r}", key=key)
        result[key] = value
    return result


def compiled_plan_from_json(payload: str) -> CompiledDecisionPlan:
    """Decode one compiled plan from its JSON text.

    Structural guards reject duplicate keys, group-sequential inference,
    sequential plans whose ``wire_version`` is not 3, and e-BH-selected
    asymptotic registrations without ``asymptotic_family`` before validation.
    The immutable context retains JSON numeric provenance without parsing twice.
    """
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except WireFormatError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _raise("wire.payload.invalid", f"invalid compiled decision plan JSON: {exc}")
    if not isinstance(raw, Mapping):
        _raise("wire.payload.invalid", "compiled decision plan JSON must be an object")
    return _decode_plan(raw, json_input=True)


__all__ = [
    "WireAlwaysValid",
    "WireAsymptoticMean",
    "WireCompiledDecisionPlan",
    "WireContrastProcedure",
    "WireFamilyMembership",
    "WireFixedInference",
    "WireMethod",
    "WireMultiplicityFamily",
    "WireNoFamily",
    "WireProcedure",
    "WireRelativeArmProcedure",
    "WireAbsoluteArmProcedure",
    "WireViewPolicies",
    "compiled_plan_from_dict",
    "compiled_plan_from_dto",
    "compiled_plan_from_json",
    "compiled_plan_to_dict",
    "compiled_plan_to_dto",
    "compiled_plan_to_json",
]
