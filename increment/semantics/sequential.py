"""Pre-data declarations for raw-observation sequential likelihoods."""

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Annotated, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from increment._literals import Alternative
from increment.errors import CapabilityError, CodedModel, InvalidRequestError, RefusalSpec, refuse
from increment.semantics.rational import DeclaredRational, PortableRational

_INVALID = RefusalSpec(
    "sequential.registration.invalid", InvalidRequestError, lambda *, reason: reason
)
_LEGACY_CONTINUATION = RefusalSpec(
    "sequential.continuation.legacy", CapabilityError, lambda *, reason: reason
)
MIXED_REQUIRES_ASYMPTOTIC_MEAN = RefusalSpec(
    "sequential.registration.mixed_requires_asymptotic_mean",
    CapabilityError,
    lambda *, metrics: (
        f"this registration mixes an asymptotic law ({sorted(metrics)!r}) with Bernoulli "
        "observations; register it under InferenceSpec(kind='asymptotic_mean') with "
        "AnalysisPlan.compliance declared, which composes both automatically"
    ),
)


def _render_segmented_family_unsupported(*, segments, family, route, alternatives):
    return (
        f"{route} does not support predeclared segments {segments!r} with compliance "
        f"family={family!r}: {'; '.join(alternatives)}. Existing breakout readout "
        "constraints still apply"
    )


_SEGMENTED_FAMILY_UNSUPPORTED = RefusalSpec(
    "sequential.compliance.segmented_family_unsupported",
    CapabilityError,
    _render_segmented_family_unsupported,
)


def refuse_segmented_family_compliance(
    segments: Mapping[str, Sequence[str]], *, route: str
) -> NoReturn:
    """Reject the one automatic policy/segment combination the compiler cannot compose."""
    admitted = tuple((dimension, tuple(levels)) for dimension, levels in sorted(segments.items()))
    refuse(
        _SEGMENTED_FAMILY_UNSUPPORTED,
        segments=admitted,
        family=True,
        route=route,
        alternatives=(
            "set compliance.family=False for separately allocated uptake checks",
            "omit inference.segments for the existing composed family",
        ),
    )


def invalid_registration(reason: str) -> NoReturn:
    refuse(_INVALID, reason=reason)


class _Declaration(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class PredictivePrior(_Declaration):
    """Proper Beta, NIG or NIW parameters; unrelated to an effect posterior.

    Scalar NIG uses shape ``nu/2`` and scale ``scale[0][0]/2``. NIW uses
    inverse-Wishart degrees nu and scale matrix scale.
    """

    kind: Literal["beta", "nig", "niw"]
    a: PortableRational | None = None
    b: PortableRational | None = None
    kappa: PortableRational | None = None
    nu: PortableRational | None = None
    mean: tuple[PortableRational, ...] = ()
    scale: tuple[tuple[PortableRational, ...], ...] = ()

    @model_validator(mode="after")
    def _proper(self):
        if self.kind == "beta":
            if self.a is None or self.b is None or self.a <= 0 or self.b <= 0:
                invalid_registration("Beta requires positive a and b")
            if self.kappa is not None or self.nu is not None or self.mean or self.scale:
                invalid_registration("Beta cannot carry Gaussian hyperparameters")
            return self
        d = 1 if self.kind == "nig" else 2
        if self.a is not None or self.b is not None:
            invalid_registration("Gaussian priors cannot carry Beta hyperparameters")
        if self.kappa is None or self.nu is None or self.kappa <= 0 or self.nu <= d - 1:
            invalid_registration("Gaussian prior requires kappa>0 and nu>dimension-1")
        if len(self.mean) != d or len(self.scale) != d or any(len(r) != d for r in self.scale):
            invalid_registration("Gaussian prior dimensions disagree")
        if any(self.scale[i][i] <= 0 for i in range(d)):
            invalid_registration("Gaussian prior scale must be positive definite")
        if d == 2 and (
            self.scale[0][1] != self.scale[1][0]
            or self.scale[0][0] * self.scale[1][1] <= self.scale[0][1] ** 2
        ):
            invalid_registration("NIW scale must be symmetric positive definite")
        return self


class SequentialModel(_Declaration):
    metric: str = Field(min_length=1)
    observable: Literal["outcome", "uptake"] = "outcome"
    law: Literal["bernoulli", "gaussian", "gaussian_ratio"]
    control_prior: PredictivePrior
    treatment_prior: PredictivePrior
    positive_population_control: Literal[True]
    positive_population_denominators: bool = False

    @model_validator(mode="after")
    def _law(self):
        expected = {"bernoulli": "beta", "gaussian": "nig", "gaussian_ratio": "niw"}[self.law]
        if self.control_prior.kind != expected or self.treatment_prior.kind != expected:
            invalid_registration("sampling law and predictive priors disagree")
        if self.positive_population_denominators != (self.law == "gaussian_ratio"):
            invalid_registration("ratio models require positive population denominator means")
        if self.observable == "uptake" and self.law != "bernoulli":
            invalid_registration("uptake is a Bernoulli observable")
        return self


class PredeclaredAdjustment(_Declaration):
    """Covariate coefficient and centre fixed from pre-period data before any outcome is read.

    The asymptotic scalar-mean route captures
    ``Y - coefficient * (X - center)`` as each unit's scalar observation.
    The exact Bernoulli route does not admit this transformed outcome law;
    predictability alone does not supply its likelihood.

    Two costs relative to the fixed-horizon fit in ``estimation.cuped``:
    the coefficient is not the contrast-optimal one (that fit reads the
    in-experiment outcome), so the variance reduction is smaller on the
    same data; and each arm's adjusted mean is shifted by the same constant
    ``coefficient * (E[X] - center)``, which cancels in a difference but
    not in a ratio. The scalar-mean route reports a ratio, so pre-period to
    experiment covariate drift is a first-order bias on its lift scale.
    """

    coefficient: PortableRational
    center: PortableRational


AsymptoticLaw = Literal["scalar_mean", "adjusted_mean", "ratio_mean", "adjusted_ratio_mean"]
ASYMPTOTIC_LAWS: tuple[str, ...] = (
    "scalar_mean",
    "adjusted_mean",
    "ratio_mean",
    "adjusted_ratio_mean",
)
ADJUSTED_LAWS: tuple[str, ...] = ("adjusted_mean", "adjusted_ratio_mean")
RATIO_LAWS: tuple[str, ...] = ("ratio_mean", "adjusted_ratio_mean")
# Metric types whose per-unit value is one scalar: a conversion or retention
# outcome is a 0/1 scalar whose arm mean is the rate and whose retained
# centered scatter is the Bernoulli variance, so it takes the scalar laws.
SCALAR_METRIC_TYPES: tuple[str, ...] = ("mean", "conversion", "retention")
BINARY_METRIC_TYPES: tuple[str, ...] = ("conversion", "retention")
# Every law a public sequential entry point admits; the exact NIG/NIW laws
# stay private because their validity needs literally Gaussian data.
PUBLIC_LAWS: tuple[str, ...] = ("bernoulli", *ASYMPTOTIC_LAWS)
_RETAINED_COORDINATES = {
    "bernoulli": 1,
    "gaussian": 1,
    "gaussian_ratio": 2,
    "scalar_mean": 1,
    "adjusted_mean": 2,
    "ratio_mean": 2,
    "adjusted_ratio_mean": 3,
}


def retained_dimension(law: str) -> int:
    """Per-unit coordinates a law retains: (Y), (Y, X), (N, D) or (N, D, X)."""
    return _RETAINED_COORDINATES[law]


class ScalarMeanModel(_Declaration):
    """Count-clock AsympCS assumptions, declared before observing outcomes.

    Fixed tuning gives an asymptotic CS approximation, not finite-start
    calibration or a uniform guarantee over heavy-tailed distributions. One
    contract covers four laws that differ only in the per-unit vector each
    arm retains (``retained_dimension``) and in the functional contrasted:

    * ``scalar_mean``: (Y); the ratio of arm means, direct shifted contrast.
    * ``adjusted_mean``: (Y, X); the ratio of CUPED-adjusted arm means with
      the coefficient fitted from the retained within-arm cross moments.
    * ``ratio_mean``: (N, D); the ratio of arm ratios ``E[N]/E[D]``.
    * ``adjusted_ratio_mean``: (N, D, X); the ratio of arm ratios after each
      component is adjusted against X with its own coefficient.

    ``moments`` asserts finite ``2 + delta`` moments for the whole retained
    vector and ``positive_limiting_variance`` a positive limiting variance of
    the linearised contrast; the adjusted laws additionally need a positive
    within-arm covariate variance (so the coefficient is identified) and the
    ratio laws a population denominator mean bounded away from zero
    (``positive_population_denominators``). X is a pre-assignment covariate.
    The three non-scalar laws are delta-method linearisations, so their sets
    are asymptotic in the same sense as the scalar law plus a nuisance
    plug-in that is negligible at the boundary's rate; see
    ``estimation.asymptotic_joint`` for the argument and references.
    """

    metric: str = Field(min_length=1)
    law: AsymptoticLaw = "scalar_mean"
    observable: Literal["outcome"] = "outcome"
    construction: Literal["direct_shifted_contrast_v1", "linearised_shifted_contrast_v1"] = (
        "direct_shifted_contrast_v1"
    )
    validity_regime: Literal["asymptotic_sequential"] = "asymptotic_sequential"
    rho: PortableRational
    start_count: StrictInt = Field(ge=2)
    assignment: Literal["iid_fixed_bernoulli_randomization"]
    treatment_probability: PortableRational
    consistency_and_no_interference: Literal[True]
    segment_membership: Literal["pre_assignment"]
    unit_model: Literal["iid_stationary_potential_outcomes"]
    moments: Literal["finite_2_plus_delta"]
    positive_limiting_variance: Literal[True]
    positive_population_control: Literal[True]
    # Absent from the dump when unset so existing registration digests are unchanged.
    positive_population_denominators: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    adjustment: PredeclaredAdjustment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="before")
    @classmethod
    def _construction_for_law(cls, value):
        if isinstance(value, Mapping) and "construction" not in value:
            law = value.get("law", "scalar_mean")
            construction = (
                "direct_shifted_contrast_v1"
                if law == "scalar_mean"
                else "linearised_shifted_contrast_v1"
            )
            return {**value, "construction": construction}
        return value

    @model_validator(mode="after")
    def _valid(self):
        if self.rho <= 0 or not 0 < self.treatment_probability < 1:
            invalid_registration(
                "scalar mean requires rho>0 and fixed assignment probability in (0,1)"
            )
        if (self.law == "scalar_mean") != (self.construction == "direct_shifted_contrast_v1"):
            invalid_registration(
                "scalar_mean is the direct shifted contrast; the joint laws are linearised"
            )
        if self.positive_population_denominators != (self.law in RATIO_LAWS):
            invalid_registration(
                "ratio laws require positive_population_denominators=True; mean laws forbid it"
            )
        if self.adjustment is not None and self.law != "scalar_mean":
            invalid_registration(
                "a predeclared adjustment retains the adjusted scalar under scalar_mean; "
                "the adjusted laws fit their coefficient from retained joint moments"
            )
        return self


class SequentialCompliancePolicy(_Declaration):
    """Explicit design-level testing policy for Bernoulli uptake."""

    # A YAML plan carries these as floats; a trusted float binds the decimal as written.
    alpha: DeclaredRational
    alternative: Alternative = "two-sided"
    null_lift: DeclaredRational = Fraction(0)
    family: bool = False

    @model_validator(mode="after")
    def _valid(self):
        if not 0 < self.alpha < 1 or self.null_lift < -1:
            invalid_registration(
                "compliance needs alpha in (0,1) and a nonnegative null risk ratio"
            )
        return self


def sequential_family_size(in_family: int, compliance: "SequentialCompliancePolicy | None") -> int:
    """The runtime's own family size: in-family metrics, plus the uptake cell
    when a declared ``compliance`` policy shares its family (``family=True``).
    Shared by the runtime registration (``sequential_source.py``) and
    planning (``ArmPlanningProcedure.standard()``) so the two cannot drift."""
    return in_family + (1 if compliance is not None and compliance.family else 0)


def validate_predeclared_segments(
    segments: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Admit an automatic registration's predeclared segment family.

    The family is fixed before any outcome is read: exactly one dimension (the
    public breakout hypothesis key admits one) naming the source's segment
    column, with the distinct labels its cells will be retained under. A label
    is compared to the source's canonical string value, so it is admitted only
    as a string: nothing is coerced from a native value, and two declarations
    that would collapse to one cell are refused rather than merged.
    """
    if not isinstance(segments, Mapping):
        invalid_registration("predeclared segments map one dimension to its levels")
    if len(segments) != 1:
        invalid_registration("predeclared segments name exactly one segment dimension")
    admitted: dict[str, tuple[str, ...]] = {}
    for dimension, levels in segments.items():
        if not isinstance(dimension, str) or not dimension:
            invalid_registration("a segment dimension is a non-empty column name")
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            invalid_registration(f"segment dimension {dimension!r} needs a sequence of levels")
        labels = tuple(levels)
        if not labels:
            invalid_registration(f"segment dimension {dimension!r} declares no levels")
        if any(not isinstance(level, str) for level in labels):
            invalid_registration(
                f"segment dimension {dimension!r}: levels are the canonical string labels "
                "of the segment column, declared as strings"
            )
        if any(not level for level in labels):
            invalid_registration(f"segment dimension {dimension!r} declares an empty level")
        if len(set(labels)) != len(labels):
            invalid_registration(f"segment dimension {dimension!r} declares a level twice")
        admitted[dimension] = labels
    return admitted


SequentialSamplingModel = Annotated[SequentialModel | ScalarMeanModel, Field(discriminator="law")]


class JointReveal(_Declaration):
    """Sampling assumptions are declarations, never inferred from timestamps."""

    filtration_id: str = Field(min_length=1)
    independent_unit_vectors: Literal[True]
    simultaneous_metrics: Literal[True]
    outcome_independent_order: Literal[True]
    immutable_finalized_outcomes: Literal[True]
    longest_window_days: int = Field(ge=0, strict=True)


class SequentialCell(_Declaration):
    metric: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    estimand: Literal["itt", "compliance"] = "itt"
    segment: tuple[tuple[str, str], ...] = ()
    alternative: Alternative = "two-sided"
    null_lift: PortableRational = Fraction(0)
    alpha: PortableRational = Fraction(1, 20)
    family: bool = False

    @model_validator(mode="after")
    def _valid(self):
        if not 0 < self.alpha < 1:
            invalid_registration("cell alpha must be in (0,1)")
        if len({k for k, _ in self.segment}) != len(self.segment):
            invalid_registration("duplicate segment dimension")
        if len(self.segment) > 1:
            invalid_registration("public breakout hypotheses require one segment dimension")
        object.__setattr__(self, "segment", tuple(sorted(self.segment)))
        return self


def sequential_family_rule(models, roster) -> Literal["e_bh", "bonferroni"] | None:
    """How discoveries are selected among a registration's family cells.

    A continuous breakout -- every family cell segmented, under the
    count-clock asymptotic construction (any ``ScalarMeanModel`` law: the
    direct scalar contrast or a linearised joint contrast, all inverted at
    the same boundary) -- keeps predeclared per-cell Bonferroni (familywise
    error): each cell is judged at its own registered allocation and the
    allocations sum to at most q. Every other family is selected by e-BH at
    the registration's q.
    """
    family = [cell for cell in roster if cell.family]
    if not family:
        return None
    if all(cell.segment for cell in family) and any(isinstance(m, ScalarMeanModel) for m in models):
        return "bonferroni"
    return "e_bh"


class SequentialRegistration(_Declaration):
    """Immutable version-2 specification supplied before a producer reads observations.

    definitions_id binds metric windows/transformations, assignment, population and
    source mapping; the explicit compliance policy is carried by the compiled plan.
    Its equality is checked at capture and continuation; it is not a watermark.

    ``asymptotic_family`` is derived, never chosen: it records that the family of
    a registration with asymptotic models is selected by e-BH, so a registration
    stored when those families were Bonferroni-corrected has a different identity
    and cannot continue under the weaker false-discovery guarantee.
    """

    version: Literal[2] = 2
    source_id: str = Field(min_length=1)
    definitions_id: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    committed_before_data: Literal[True]
    reveal: JointReveal
    models: tuple[SequentialSamplingModel, ...]
    roster: tuple[SequentialCell, ...]
    q: PortableRational = Fraction(1, 10)
    asymptotic_family: Literal["e_bh"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="before")
    @classmethod
    def _current_version(cls, value):
        if isinstance(value, Mapping):
            if value.get("version", 2) != 2:
                refuse(
                    _LEGACY_CONTINUATION,
                    reason="legacy sequential registration cannot continue without the current observation mapping and compliance policy",
                )
            if "finite_looks" in value or "cumulative_spending" in value:
                refuse(
                    _LEGACY_CONTINUATION,
                    reason="finite-look registrations were removed: an always-valid registration monitors every look without a schedule, and a registration or checkpoint stored with finite_looks cannot continue; re-register",
                )
        return value

    @model_validator(mode="after")
    def _complete(self):
        if not self.models or not self.roster or not 0 < self.q < 1:
            invalid_registration("registration needs models, a retained roster and q in (0,1)")
        continuous = any(isinstance(m, ScalarMeanModel) for m in self.models)
        if continuous:
            if not all(
                isinstance(m, ScalarMeanModel)
                or (m.law == "bernoulli" and m.observable == "uptake")
                for m in self.models
            ):
                invalid_registration(
                    "exact and asymptotic hypotheses require separately registered families"
                )
            if len({c.group_id for c in self.roster}) != 1:
                invalid_registration(
                    "direct scalar mean supports one randomized treatment and control"
                )
            if (
                len(
                    {m.treatment_probability for m in self.models if isinstance(m, ScalarMeanModel)}
                )
                != 1
            ):
                invalid_registration("scalar models must declare the same assignment probability")
            if sum((c.alpha for c in self.roster if c.family), Fraction(0)) > self.q:
                invalid_registration("fixed-roster asymptotic Bonferroni allocations exceed q")
        names = [m.metric for m in self.models]
        if len(set(names)) != len(names):
            invalid_registration("duplicate sampling model")
        keys = [(c.metric, c.group_id, c.estimand, c.segment) for c in self.roster]
        if len(set(keys)) != len(keys):
            invalid_registration("duplicate retained hypothesis")
        if set(names) != {c.metric for c in self.roster}:
            invalid_registration("every declared model must have a retained hypothesis")
        for cell in self.roster:
            if cell.metric not in names or cell.group_id == self.control_group:
                invalid_registration("roster has an undeclared model or a control contrast")
            model = next(m for m in self.models if m.metric == cell.metric)
            if model.law == "bernoulli" and cell.null_lift < -1:
                invalid_registration("Bernoulli null risk ratios must be nonnegative")
            if cell.estimand == "compliance" and model.law != "bernoulli":
                invalid_registration("uptake compliance requires a Bernoulli declaration")
            if (cell.estimand == "compliance") != (model.observable == "uptake"):
                invalid_registration(
                    "raw ITT and design uptake must have distinct model identities"
                )
        object.__setattr__(self, "models", tuple(sorted(self.models, key=lambda m: m.metric)))
        object.__setattr__(
            self,
            "roster",
            tuple(sorted(self.roster, key=lambda c: (c.metric, c.group_id, c.estimand, c.segment))),
        )
        selected_by_e_bh = continuous and sequential_family_rule(self.models, self.roster) == "e_bh"
        object.__setattr__(self, "asymptotic_family", "e_bh" if selected_by_e_bh else None)
        return self


def refuse_legacy_asymptotic_family(payload: object) -> None:
    """Refuse a stored registration made when asymptotic families were Bonferroni-corrected.

    Such a registration has the same content as today's but no recorded
    ``asymptotic_family``; continuing it would silently trade its familywise
    guarantee for e-BH's false discovery rate.
    """
    if not isinstance(payload, Mapping) or "asymptotic_family" in payload:
        return
    try:
        registration = SequentialRegistration.model_validate(payload)
    except ValueError:
        return  # the caller's own validation reports a malformed registration
    if registration.asymptotic_family is not None:
        refuse(
            _LEGACY_CONTINUATION,
            reason="this registration was stored when its asymptotic family was "
            "Bonferroni-corrected (familywise error); asymptotic families are now selected by "
            "e-BH (false discovery rate), so it cannot continue under that weaker guarantee. "
            "Start a new registration by capturing without previous=, or finish the process "
            "on the release that stored it",
        )
