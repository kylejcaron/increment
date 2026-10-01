"""Canonical sequential-campaign manifest, acceptance gates, and bounded deterministic fixtures.

The canonical records are immutable coverage metadata; importing this module never
executes simulation or production inference. The acceptance gates are declarative
definitions and deciding them never requires executing the campaign.
"""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from typing import Literal, cast

F = Fraction
Observation = int | tuple[int, int]


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    law: Literal["bernoulli", "gaussian", "gaussian_ratio", "scalar_mean"] = "bernoulli"
    rate: float = 0.15
    allocation: tuple[int, int] = (1, 1)
    variance_ratio: int = 1
    null: F = F(0)
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    wrong_direction: F = F(0)
    looks: int = 14
    batch: int = 50
    schedule: str = "equal"
    family: int = 1
    null_fraction: float = 1.0
    dependence: str = "independent"
    shape: str = "gaussian"
    stopping: str = "horizon"
    missing: bool = False
    freeze: bool = False
    seed: int = 107
    scalar_control_mean: F = F(5)
    scalar_treatment_shift: F = F(0)
    scalar_rho: F = F(1, 10)
    scalar_start_count: int = 2


@dataclass(frozen=True)
class CertificationDesign:
    """Frozen certification parameters; every field enters the design digest."""

    alpha: Fraction = Fraction(1, 20)
    q: Fraction = Fraction(1, 20)
    scientific_excess: Fraction = Fraction(1, 200)
    max_mc_margin: Fraction = Fraction(1, 400)
    historical_stages: tuple[int, ...] = tuple(4096 * 2**i for i in range(8))
    # One union bound covers every budgeted gate at every interim look. Hoeffding's
    # divergence form is exact for bounded means and inverts to the repetition rule.
    mc_family_error: Fraction = Fraction(1, 100)
    mc_allocation_status: str = "resolved-union-bound-over-gates-and-interim-looks"
    mc_interim_looks: int = 8
    mc_allocation_rule: str = (
        "per_decision_error = mc_family_error / (budgeted_gates * mc_interim_looks)"
    )
    mc_tail_bound: str = (
        "Chernoff-Hoeffding divergence bound for the mean of independent [0,1] "
        "replication statistics: P(mean past threshold) <= exp(-n * KL(threshold||worst))"
    )
    mc_bound_family: str = (
        "distribution-free by requirement: several gated statistics are proportions over "
        "random denominators, so an exact binomial interval does not cover every gate"
    )
    mc_repetition_rule: str = (
        "gate replications = ceil(ln(1/per_decision_error) / KL(threshold||worst)) rounded "
        "up to replication_quantum; case replications = max over its budgeted gates"
    )
    replication_quantum: int = 1000
    mc_stage_rule: str = (
        "stage_k = ceil(case_replications / 2**(mc_interim_looks - 1 - k)); a case is "
        "refused at the first stage that carries a refusal"
    )
    mc_decision_rule: str = (
        "upper gate accepts when estimate <= target + tolerance - margin; "
        "lower gate accepts when estimate >= target - tolerance + margin"
    )
    recorded_granularity: Fraction = Fraction(1, 1_000_000)
    control_mean: Fraction = Fraction(5)
    control_variance: Fraction = Fraction(1)
    finite_moment_variance: Fraction = Fraction(5, 3)
    student_t_degrees_of_freedom: int = 5
    bernoulli_value_rule: str = "value=1 iff U < rate*declared_ratio, U uniform on [0,1)"
    gaussian_value_rule: str = (
        "value = control_mean*declared_ratio + sqrt(variance_ratio)*Z on treated arms; "
        "control arms carry unit standard deviation"
    )
    gaussian_ratio_value_rule: str = (
        "numerator = ratio_numerator_mean_scale*control_mean*declared_ratio "
        "+ ratio_numerator_noise_scale*sqrt(variance_ratio)*Z0; denominator = "
        "ratio_denominator_mean + loadings[0]*Z0 + loadings[1]*Z1, Z0 and Z1 independent"
    )
    scalar_value_rule: str = (
        "value = control_mean*(declared_ratio on treated arms else 1) "
        "+ sqrt(variance_ratio on treated arms else 1)*noise"
    )
    segment_split_rule: str = (
        "fixed-arm batches label a unit A when its within-arm index is even else B; "
        "iid scalar units label A unless the rare pre-assignment segment fires"
    )
    rare_segment_probability: Fraction = Fraction(1, 1_000_000)
    greater_effect_multiplier: Fraction = Fraction(3, 2)
    less_effect_multiplier: Fraction = Fraction(1, 2)
    ratio_numerator_mean_scale: Fraction = Fraction(2)
    ratio_denominator_mean: Fraction = Fraction(2)
    ratio_numerator_noise_scale: Fraction = Fraction(2)
    ratio_denominator_loadings: tuple[Fraction, Fraction] = (
        Fraction(1, 5),
        Fraction(1, 5),
    )
    shared_segment: str = "A"
    schedule_endpoint_rule: str = "e_i=i+1+(L*batch-L)*sum(weights[:i+1])//sum(weights)"
    schedule_increment_rule: str = "size_i=e_i-e_(i-1) with e_(-1)=0, so sum(size)=L*batch"
    equal_weight_rule: str = "weights_i=1"
    front_loaded_weight_rule: str = "weights_i=L-i"
    roster_order_rule: str = (
        "roster_index enumerates product(models, arms); every parity rule reads that index"
    )
    first_rejection_rule: str = "stop when any retained row is significant"
    first_discovery_rule: str = "stop when retained-family e-BH selects any cell"
    horizon_rule: str = "stop only after the final predeclared look"
    adaptive_rule: str = (
        "from the second look, stop when any even roster_index row has deployed log_e>0"
    )
    freeze_rule: str = (
        "before final look, retain first checkpoint when significant or when "
        "look_index=L//2 and roster_index is even"
    )
    freeze_reuse_rule: str = (
        "a frozen cell keeps its retained checkpoint at every later look and stays in the "
        "family denominator; freezing precedes the stopping predicate within a look"
    )
    availability_floor: Fraction = Fraction(99, 100)
    availability_slack: Fraction = Fraction(1, 100)
    point_availability_bound_rule: str = (
        "unavailable point fraction at the earliest evaluated look: structurally absent "
        "roster members, empty arms, and the chance the control mean reaches zero"
    )
    interval_availability_bound_rule: str = (
        "uncertified geometry comes only from structurally absent roster members and "
        "empty arms; a zero-count arm still inverts to a ray over the declared domain"
    )
    availability_conditioning: str = "per retained cell at the stopped look"
    unavailable_statuses: tuple[str, ...] = ("unavailable", "unresolved", "abstained")
    declared_point_reasons: tuple[str, ...] = (
        "missing arm in retained cell",
        "observed control mean is zero",
        "observed denominator mean is zero",
        "relative point exceeds binary64 range",
    )
    unconditional_conditioning: str = "unconditional over the whole retained path"
    selected_conditioning: str = "selection-conditioned at the stopped look via fcr_alpha"
    reference_power_rule: str = (
        "fixed-sample two-arm log-ratio z test over the full horizon at alpha/family, "
        "normal-approximation variance from the declared per-unit moments"
    )
    reference_power_qualifier: Fraction = Fraction(9, 10)
    deployed_power_floor: Fraction = Fraction(4, 5)
    power_floor_rule: str = (
        "floor = deployed_power_floor when reference >= reference_power_qualifier, else "
        "reference*deployed_power_floor/reference_power_qualifier; a floor at or below "
        "scientific_excess is recorded instead of decided"
    )


CERTIFICATION_DESIGN = CertificationDesign()


def principal_manifest() -> tuple[RuntimeCase, ...]:
    """Cross the principal nuisance/look axes; rotate supplemental axes explicitly."""
    cases = []
    allocations = ((1, 1), (1, 4), (4, 1))
    looks = (2, 14, 25, 100)
    nulls = (F(-1, 5), F(0), F(1, 5))
    alternatives = ("two-sided", "greater", "less")
    for i, (rate, allocation, count) in enumerate(
        product((0.001, 0.01, 0.15, 0.5), allocations, looks)
    ):
        for j, null in enumerate(nulls):
            base = RuntimeCase(
                name=f"bernoulli-{rate}-{allocation}-{count}-{null}",
                rate=rate,
                allocation=allocation,
                looks=count,
                null=null,
                alternative=alternatives[(i + j) % 3],
                batch=(10, 50, 200)[i % 3],
                schedule=("equal", "front_loaded")[i % 2],
                stopping="first_rejection",
            )
            cases.append(base)
    for law, seed in (("gaussian", 101), ("gaussian_ratio", 109)):
        for i, (variance, allocation, count) in enumerate(product((1, 4, 16), allocations, looks)):
            base = RuntimeCase(
                name=f"{law}-{variance}-{allocation}-{count}",
                law=law,
                variance_ratio=variance,
                allocation=allocation,
                looks=count,
                null=nulls[i % 3],
                alternative=alternatives[i % 3],
                batch=(10, 50, 200)[i % 3],
                schedule=("equal", "front_loaded")[i % 2],
                stopping="first_rejection",
                seed=seed,
            )
            cases.append(base)
    for i, (shape, allocation, count) in enumerate(
        product(("gaussian", "finite_moment"), ((1, 1), (1, 4)), (6, 14))
    ):
        cases.append(
            RuntimeCase(
                name=f"scalar_mean-{shape}-{allocation}-{count}",
                law="scalar_mean",
                shape=shape,
                allocation=allocation,
                looks=count,
                null=nulls[i % 3],
                alternative=alternatives[i % 3],
                variance_ratio=(1, 4)[i % 2],
                stopping="first_rejection",
                missing=False,
                seed=127 + i,
            )
        )
    cases.append(
        RuntimeCase(
            name="scalar_mean-family-missing-unequal-peek",
            law="scalar_mean",
            shape="finite_moment",
            allocation=(1, 4),
            family=3,
            null_fraction=1 / 3,
            dependence="shared",
            looks=6,
            batch=10,
            stopping="first_discovery",
            missing=True,
            seed=139,
        )
    )
    cases.append(
        RuntimeCase(
            name="scalar_mean-near-zero-control",
            law="scalar_mean",
            scalar_control_mean=F(1, 1000),
            scalar_treatment_shift=F(5),
            null_fraction=0,
            looks=6,
            batch=10,
            stopping="horizon",
            seed=149,
        )
    )
    for i, (size, fraction, dependence, stop, count) in enumerate(
        product(
            (1, 10, 100),
            (1.0, 0.5),
            ("independent", "shared"),
            ("first_rejection", "first_discovery", "horizon", "adaptive"),
            looks,
        )
    ):
        base = RuntimeCase(
            name=f"family-{size}-{fraction}-{dependence}-{stop}-{count}",
            family=size,
            null_fraction=fraction,
            dependence=dependence,
            stopping=stop,
            looks=count,
            allocation=allocations[i % 3],
            rate=(0.001, 0.01, 0.15, 0.5)[i % 4],
            batch=(10, 50, 200)[i % 3],
            schedule=("equal", "front_loaded")[i % 2],
            missing=size > 1 and stop == "adaptive",
            freeze=stop == "adaptive",
            seed=113,
        )
        cases.append(base)
        for law, seed in (("gaussian", 101), ("gaussian_ratio", 109)):
            cases.append(
                replace(
                    base,
                    name=f"{law}-{base.name}",
                    law=law,
                    variance_ratio=(1, 4, 16)[i % 3],
                    seed=seed,
                )
            )
    for law in ("bernoulli", "gaussian", "gaussian_ratio"):
        for alternative, wrong in product(("greater", "less"), (F(-1, 2), F(-1, 5))):
            cases.append(
                RuntimeCase(
                    name=f"wrong-{law}-{alternative}-{wrong}",
                    law=law,
                    alternative=alternative,
                    wrong_direction=wrong,
                    null=F(1, 5),
                    stopping="first_rejection",
                    seed=109,
                )
            )
        cases.append(
            RuntimeCase(
                name=f"power-{law}",
                law=law,
                batch=200,
                null_fraction=0,
                stopping="first_discovery",
                seed=101,
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class CoverageBinding:
    proofs: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
    upper_claims: tuple[str, ...] = ()
    power_reference_required: bool = False
    power_nonvacuity_required: bool = False
    status: str = "prospective-unproved"
    law_bridge: str = "not_applicable"
    availability_width: str = "freeze-applicability-and-certify"


def _coverage_binding(case: RuntimeCase) -> CoverageBinding:
    if case.law == "scalar_mean":
        claims = ("asymptotic_sequential_set_exclusion",)
    elif case.name.startswith("wrong-"):
        claims = ("ever_null_rejection", "stopped_FDR", "selected_FCR")
    elif case.name.startswith("power-"):
        claims = ("ever_null_rejection", "stopped_FDR", "selected_FCR")
    elif case.family > 1:
        claims = ("stopped_FDR", "selected_FCR")
    else:
        claims = ("ever_null_rejection", "stopped_FDR", "selected_FCR")
    mixed = "family-" in case.name and case.family > 1 and case.null_fraction == 0.5
    return CoverageBinding(
        proofs=() if case.law == "scalar_mean" else ("P1", "P2", "P3", "P4", "P5", "P6"),
        upper_claims=claims,
        power_reference_required=mixed or case.name.startswith("power-"),
        power_nonvacuity_required=case.name.startswith("power-"),
        law_bridge={
            "bernoulli": "Bernoulli-threshold-coupling",
            "gaussian": "Gaussian-rounding-and-declared-sampler-target",
            "gaussian_ratio": "Gaussian-rounding-and-declared-sampler-target",
            "scalar_mean": "registered-iid-scalar-mean-asymptotic-target",
        }[case.law],
        availability_width=(
            "preserve-unavailable-and-unbounded-geometry"
            if case.law == "scalar_mean"
            else "freeze-applicability-and-certify"
        ),
    )


CASE_COVERAGE = tuple(_coverage_binding(case) for case in principal_manifest())
COVERAGE_MANIFEST = tuple(zip(principal_manifest(), CASE_COVERAGE, strict=True))


@dataclass(frozen=True)
class IntegrationCase:
    law: Literal["bernoulli", "gaussian", "gaussian_ratio", "scalar_mean"]
    control: tuple[Observation, ...]
    treatment: tuple[Observation, ...]
    ratio: Fraction
    repeats: tuple[int, ...] = (6, 24)


INTEGRATION_CASES = (
    IntegrationCase("bernoulli", (0, 0, 0, 1), (0, 1, 1, 1), Fraction(3)),
    IntegrationCase("gaussian", (1, 2, 3, 4), (6, 10, 14, 18), Fraction(24, 5)),
    IntegrationCase(
        "gaussian_ratio",
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((8, 1), (9, 1), (8, 2), (9, 2)),
        Fraction(17, 3),
    ),
    IntegrationCase(
        "scalar_mean",
        (1, 2, 3, 4),
        (8, 10, 12, 14),
        Fraction(22, 5),
        repeats=(2, 3),
    ),
)


@dataclass(frozen=True)
class IntegrationWork:
    raw_records: int
    scalar_observations: int
    checkpoint_evaluations: int


# Operation ceilings for these deterministic witnesses, not seconds per kernel.
INTEGRATION_CEILING = IntegrationWork(600, 800, 12)


def integration_work(cases: Sequence[IntegrationCase]) -> IntegrationWork:
    records = scalars = checkpoints = 0
    for case in cases:
        if len(case.control) != len(case.treatment) or not case.control:
            raise ValueError("integration arms must have equal nonempty blocks")
        if not case.repeats or any(
            b <= a for a, b in zip((0, *case.repeats[:-1]), case.repeats, strict=True)
        ):
            raise ValueError("integration prefixes must increase strictly")
        count = 2 * len(case.control) * case.repeats[-1]
        records += count
        scalars += count * (2 if case.law == "gaussian_ratio" else 1)
        # Scalar sets retain their alpha; exact selected intervals are reinverted.
        checkpoints += len(case.repeats) + (case.law != "scalar_mean")
    return IntegrationWork(records, scalars, checkpoints)


def require_integration_budget(cases: Sequence[IntegrationCase]) -> IntegrationWork:
    """Reject excessive fixture work before expanding any raw observations."""
    work = integration_work(cases)
    if (
        work.raw_records > INTEGRATION_CEILING.raw_records
        or work.scalar_observations > INTEGRATION_CEILING.scalar_observations
        or work.checkpoint_evaluations > INTEGRATION_CEILING.checkpoint_evaluations
    ):
        raise ValueError(f"deterministic integration budget exceeded: {work}")
    return work


def campaign_schedule(case):
    weights = [case.looks - i if case.schedule == "front_loaded" else 1 for i in range(case.looks)]
    remaining = case.looks * case.batch - case.looks
    endpoints = [
        i + 1 + remaining * sum(weights[: i + 1]) // sum(weights) for i in range(case.looks)
    ]
    return tuple(b - a for a, b in zip((0, *endpoints[:-1]), endpoints, strict=True))


_SQRT_TWO = math.sqrt(2.0)
_UPPER_QUANTILES: dict[Fraction, float] = {}


def _normal_upper_quantile(tail: Fraction) -> float:
    """Invert the upper tail directly; no quantile is ever built from 1 - alpha."""
    cached = _UPPER_QUANTILES.get(tail)
    if cached is not None:
        return cached
    target, low, high = float(tail), 0.0, 40.0
    for _ in range(80):
        middle = (low + high) / 2
        if math.erfc(middle / _SQRT_TWO) / 2 > target:
            low = middle
        else:
            high = middle
    _UPPER_QUANTILES[tail] = (low + high) / 2
    return _UPPER_QUANTILES[tail]


def _normal_cdf(value: float) -> float:
    return math.erfc(-value / _SQRT_TWO) / 2


def _record_probability(value: float) -> Fraction:
    unit = CERTIFICATION_DESIGN.recorded_granularity.denominator
    return min(F(1), max(F(0), F(round(value * unit), unit)))


def _record_bound(value: float) -> Fraction:
    unit = CERTIFICATION_DESIGN.recorded_granularity.denominator
    return min(F(1), max(F(0), F(math.ceil(value * unit), unit)))


def _roster_size(case) -> int:
    if case.law == "scalar_mean":
        return max(case.family, 1)
    arms = 1 if case.family == 1 else 2
    return (case.family // arms) * arms


def _unit_moments(case, ratio: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Per-unit mean and variance of the lift metric on the control and treated arms."""
    design = CERTIFICATION_DESIGN
    if case.law == "bernoulli":
        control, treated = case.rate, case.rate * ratio
        return (control, control * (1 - control)), (treated, treated * (1 - treated))
    if case.law == "gaussian":
        mean, variance = float(design.control_mean), float(design.control_variance)
        return (mean, variance), (mean * ratio, variance * case.variance_ratio)
    if case.law == "gaussian_ratio":
        mean_scale = float(design.ratio_numerator_mean_scale)
        noise_scale = float(design.ratio_numerator_noise_scale)
        denominator = float(design.ratio_denominator_mean)
        first, second = (float(load) for load in design.ratio_denominator_loadings)
        spread = first * first + second * second

        def moment(location: float, deviation: float) -> tuple[float, float]:
            mean = mean_scale * location / denominator
            term = noise_scale * noise_scale * deviation * deviation
            term -= 2 * mean * noise_scale * deviation * first
            return mean, (term + mean * mean * spread) / (denominator * denominator)

        baseline = float(design.control_mean)
        return moment(baseline, 1.0), moment(baseline * ratio, math.sqrt(case.variance_ratio))
    base = float(
        design.control_variance if case.shape == "gaussian" else design.finite_moment_variance
    )
    mean = float(case.scalar_control_mean)
    return (mean, base), (mean * ratio, base * case.variance_ratio)


def prespecified_alternative(case) -> Fraction | None:
    """Recover the single non-null lift ratio the manifest declares for this case."""
    design = CERTIFICATION_DESIGN
    if math.ceil(case.family * case.null_fraction) >= _roster_size(case):
        return None
    ratio = (F(1) + case.null) * (
        design.less_effect_multiplier
        if case.alternative == "less"
        else design.greater_effect_multiplier
    )
    if case.law == "scalar_mean":
        ratio += case.scalar_treatment_shift / case.scalar_control_mean
    return ratio


def reference_power(case) -> Fraction | None:
    """Power of an independent fixed-sample log-ratio test at the declared horizon."""
    ratio = prespecified_alternative(case)
    if ratio is None:
        return None
    design = CERTIFICATION_DESIGN
    null_ratio = F(1) + case.null
    control_units = case.looks * case.batch * case.allocation[0]
    treated_units = case.looks * case.batch * case.allocation[1]
    (control_mean, control_var), (treated_mean, treated_var) = _unit_moments(case, float(ratio))
    error = math.sqrt(
        treated_var / (treated_units * treated_mean**2)
        + control_var / (control_units * control_mean**2)
    )
    distance = abs(math.log(float(ratio)) - math.log(float(null_ratio))) / error
    tail = design.alpha / max(case.family, 1)
    if case.alternative == "two-sided":
        tail /= 2
    return _record_probability(_normal_cdf(distance - _normal_upper_quantile(tail)))


def _empty_arm_bound(case) -> float:
    """Chance a retained cell loses an arm at the earliest evaluated look."""
    if case.law != "scalar_mean":
        return 0.0
    treated = case.allocation[1] / sum(case.allocation)
    units = campaign_schedule(case)[0] * sum(case.allocation)
    return treated**units + (1 - treated) ** units


def _structural_bound(case, tail: float) -> Fraction:
    roster = _roster_size(case)
    absent = 1 if case.missing else 0
    return _record_bound(absent / roster + (roster - absent) / roster * tail)


def interval_unavailability_bound(case) -> Fraction:
    """Bound the fraction of retained cells whose confidence geometry is uncertified."""
    return _structural_bound(case, _empty_arm_bound(case))


def point_unavailability_bound(case) -> Fraction:
    """Bound the unavailable-point fraction at the earliest evaluated look."""
    design = CERTIFICATION_DESIGN
    # Every term below falls as units accumulate, so the first look is the worst case
    # and the bound holds at whichever look the stopping rule actually reaches.
    control_units = campaign_schedule(case)[0] * case.allocation[0]
    if case.law == "bernoulli":
        tail = (1 - case.rate) ** control_units
    else:
        (mean, variance), _ = _unit_moments(case, 1.0)
        tail = _normal_cdf(-mean * math.sqrt(control_units / variance))
    if case.law == "gaussian_ratio":
        first, second = (float(load) for load in design.ratio_denominator_loadings)
        denominator = float(design.ratio_denominator_mean)
        tail += _normal_cdf(
            -denominator * math.sqrt(control_units / (first * first + second * second))
        )
    return _structural_bound(case, tail + _empty_arm_bound(case))


@dataclass(frozen=True)
class AcceptanceGate:
    name: str
    statistic: str
    direction: Literal["upper", "lower", "record"]
    target: Fraction
    tolerance: Fraction
    margin: Fraction
    conditioning: str
    reference: Fraction | None = None


@dataclass(frozen=True)
class GateVerdict:
    name: str
    status: Literal["accepted", "refused", "recorded", "unmeasured"]
    value: Fraction | None
    threshold: Fraction | None


def gate_threshold(gate: AcceptanceGate) -> Fraction | None:
    """Move the declared target against the gate by its Monte-Carlo margin."""
    if gate.direction == "record":
        return None
    if gate.direction == "upper":
        return gate.target + gate.tolerance - gate.margin
    return gate.target - gate.tolerance + gate.margin


def _tolerated_worst_case(direction: str, target: Fraction, tolerance: Fraction) -> Fraction:
    return target + tolerance if direction == "upper" else target - tolerance


def _decidable(direction: str, target: Fraction, tolerance: Fraction, margin: Fraction) -> bool:
    """A gate is decidable when its tolerated worst case stays inside (0, 1)."""
    worst = _tolerated_worst_case(direction, target, tolerance)
    return margin < worst < F(1) - margin


def _gate(
    name: str,
    statistic: str,
    direction: str,
    target: Fraction,
    tolerance: Fraction,
    margin: Fraction,
    conditioning: str,
    reference: Fraction | None = None,
) -> AcceptanceGate:
    decided = _decidable(direction, target, tolerance, margin)
    return AcceptanceGate(
        name,
        statistic,
        cast(Literal["upper", "lower", "record"], direction if decided else "record"),
        target,
        tolerance if decided else F(0),
        margin if decided else F(0),
        conditioning,
        reference,
    )


def gate_replications(gate: AcceptanceGate, error: Fraction) -> int:
    """Smallest quantised count whose divergence tail clears the per-decision error."""
    if not gate.margin:
        return 0
    worst = _tolerated_worst_case(gate.direction, gate.target, gate.tolerance)
    threshold = gate_threshold(gate)
    assert threshold is not None
    divergence = _binary_divergence(float(threshold), float(worst))
    quantum = CERTIFICATION_DESIGN.replication_quantum
    raw = math.ceil(_natural_log(F(1) / error) / divergence)
    return quantum * -(-raw // quantum)


def _binary_divergence(threshold: float, worst: float) -> float:
    return threshold * math.log(threshold / worst) + (1 - threshold) * math.log(
        (1 - threshold) / (1 - worst)
    )


def _natural_log(value: Fraction) -> Fraction:
    with localcontext() as context:
        context.prec = 50
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        return F(decimal.ln())


def _claim_gate(claim: str) -> tuple[str, Fraction, str]:
    design = CERTIFICATION_DESIGN
    if claim == "stopped_FDR":
        return "stopped_false_discovery_proportion", design.q, design.selected_conditioning
    if claim == "selected_FCR":
        return "selected_false_coverage_proportion", design.q, design.selected_conditioning
    return "ever_null_rejection", design.alpha, design.unconditional_conditioning


def case_acceptance_gates(case, binding) -> tuple[AcceptanceGate, ...]:
    """Bind every prescribed acceptance bound for one manifest case."""
    design = CERTIFICATION_DESIGN
    excess, margin = design.scientific_excess, design.max_mc_margin
    gates = []
    for claim in binding.upper_claims:
        statistic, target, conditioning = _claim_gate(claim)
        gates.append(_gate(claim, statistic, "upper", target, excess, margin, conditioning))
    for name, statistic, bound in (
        ("point_availability", "point_available_fraction", point_unavailability_bound(case)),
        (
            "interval_availability",
            "certified_interval_fraction",
            interval_unavailability_bound(case),
        ),
    ):
        floor = min(design.availability_floor, F(1) - bound)
        regime = "supported" if floor == design.availability_floor else "rare_event"
        gates.append(
            _gate(
                f"{name}_{regime}",
                statistic,
                "lower",
                floor,
                design.availability_slack,
                margin,
                design.availability_conditioning,
            )
        )
    gates.append(
        AcceptanceGate(
            "declared_unavailability_reason",
            "undeclared_point_reason_fraction",
            "upper",
            F(0),
            F(0),
            F(0),
            design.availability_conditioning,
        )
    )
    reference = reference_power(case)
    if reference is not None:
        gates.append(
            _gate(
                "useful_alternative_power",
                "nonnull_discovery",
                "lower",
                design.deployed_power_floor
                if reference >= design.reference_power_qualifier
                else reference * design.deployed_power_floor / design.reference_power_qualifier,
                excess,
                margin,
                design.power_floor_rule,
                reference,
            )
        )
    return tuple(gates)


@dataclass(frozen=True)
class CertificationLedger:
    case_count: int
    gate_count: int
    budgeted_gate_count: int
    recorded_gate_count: int
    interim_looks: int
    family_error: Fraction
    per_decision_error: Fraction
    max_case_replications: int
    binding_gate: str
    max_stage_ladder: tuple[int, ...]
    total_replications: int
    retained_historical_stages: tuple[int, ...]
    historical_stages_sufficient: bool


def case_replications(gates, error: Fraction) -> int:
    """A case runs until its hardest budgeted gate clears the per-decision error."""
    return max((gate_replications(gate, error) for gate in gates), default=0)


def certification_ledger(gated_cases) -> CertificationLedger:
    """Spend the family Monte-Carlo error over every budgeted gate and interim look."""
    design = CERTIFICATION_DESIGN
    every = [gate for _, gates in gated_cases for gate in gates]
    budgeted = [gate for gate in every if gate.margin]
    per_decision = design.mc_family_error / (len(budgeted) * design.mc_interim_looks)
    requirements = [(gate_replications(gate, per_decision), gate.name) for gate in budgeted]
    replications, binding = max(requirements)
    ladder = tuple(
        -(-replications // 2 ** (design.mc_interim_looks - 1 - stage))
        for stage in range(design.mc_interim_looks)
    )
    return CertificationLedger(
        case_count=len(gated_cases),
        gate_count=len(every),
        budgeted_gate_count=len(budgeted),
        recorded_gate_count=sum(1 for gate in every if gate.direction == "record"),
        interim_looks=design.mc_interim_looks,
        family_error=design.mc_family_error,
        per_decision_error=per_decision,
        max_case_replications=replications,
        binding_gate=binding,
        max_stage_ladder=ladder,
        total_replications=sum(case_replications(gates, per_decision) for _, gates in gated_cases),
        retained_historical_stages=design.historical_stages,
        historical_stages_sufficient=replications <= design.historical_stages[-1],
    )


ACCEPTANCE_GATES = tuple(
    (case, case_acceptance_gates(case, binding)) for case, binding in COVERAGE_MANIFEST
)
CERTIFICATION_LEDGER = certification_ledger(ACCEPTANCE_GATES)


def replication_statistics(record) -> dict[str, Fraction]:
    """Project one replication record onto the [0,1] statistics the gates decide."""
    retained = record["retained_cells"]
    statistics = {
        "ever_null_rejection": F(record["nominal_ever_null_rejection"]),
        "stopped_false_discovery_proportion": F(record["nominal_fdp"]),
        "selected_false_coverage_proportion": F(record["nominal_fcp_upper"]),
        "nonnull_discovery": F(record["nonnull_discovery"]),
    }
    if retained:
        statistics["point_available_fraction"] = F(record["available_points"], retained)
        statistics["certified_interval_fraction"] = F(record["certified_intervals"], retained)
        statistics["undeclared_point_reason_fraction"] = F(
            record["undeclared_point_reasons"], retained
        )
    return statistics


def acceptance_verdicts(gates, statistics) -> tuple[GateVerdict, ...]:
    """Decide every gate against replication means; an absent statistic never accepts."""
    verdicts: list[GateVerdict] = []
    for gate in gates:
        threshold = gate_threshold(gate)
        value = statistics.get(gate.statistic)
        status: Literal["accepted", "refused", "recorded", "unmeasured"]
        if gate.direction == "record":
            status = "recorded"
        elif value is None:
            status = "unmeasured"
        elif gate.direction == "upper":
            status = "accepted" if value <= threshold else "refused"
        else:
            status = "accepted" if value >= threshold else "refused"
        verdicts.append(GateVerdict(gate.name, status, value, threshold))
    return tuple(verdicts)


def certification_accepted(verdicts: Sequence[GateVerdict]) -> bool:
    return all(verdict.status in ("accepted", "recorded") for verdict in verdicts)


def campaign_science(case):
    """Bind the sampler and statistic declarations, independently of runtime metadata."""
    if case.law != "scalar_mean":
        return {"sampler_revision": "5f3b92fe", "assignment": "fixed-arm-batches"}
    return {
        "sampler_revision": "iid-scalar-joint-v2",
        "assignment": "iid_fixed_bernoulli_randomization",
        "treatment_probability": F(case.allocation[1], sum(case.allocation)),
        "draw_order": (
            "per-unit segment if needed, integer assignment, joint noise; reveal in draw order"
        ),
        "noise": "standard_normal" if case.shape == "gaussian" else "student_t_5_unscaled",
        "control_variance": F(1) if case.shape == "gaussian" else F(5, 3),
        "treatment_variance_ratio": case.variance_ratio,
        "control_mean": case.scalar_control_mean,
        "treatment_mean": "control_mean * declared_ratio (including treatment_shift/control_mean)",
        "metric_dependence": case.dependence,
        "missing_member": "rare pre-assignment segment may be empty; fixed roster alpha retained",
        "rare_segment_probability": F(1, 1_000_000) if case.missing else None,
        "reveal": "iid joint finalized unit vectors; all metrics simultaneous",
        "look_counts": tuple(sum(case.allocation) * n for n in campaign_schedule(case)),
        "construction": "direct_shifted_contrast_v1",
        "variance": "M2_t/n_t^2 + r^2*M2_c/n_c^2",
        "clock": "actual n_control + n_treatment for each retained cell",
        "rho": case.scalar_rho,
        "start_count_per_arm": case.scalar_start_count,
        "alpha": CERTIFICATION_DESIGN.alpha / case.family,
        "family_selection": "fixed-roster Bonferroni; no e-BH or reinversion",
        "first_discovery_rule": "stop at first asymptotic set-exclusion discovery",
        "validity_regime": "asymptotic_sequential",
        "finite_start_anytime_claim": False,
        "source_lane": "synthetic finalized records through public capture and inference",
    }


def campaign_identity(cases):
    payload = {
        "cases": [asdict(case) for case in cases],
        "science": [campaign_science(case) for case in cases],
        "coverage": [asdict(_coverage_binding(case)) for case in cases],
        "gates": [
            [asdict(gate) for gate in case_acceptance_gates(case, _coverage_binding(case))]
            for case in cases
        ],
        "certification_design": asdict(CERTIFICATION_DESIGN),
        "certification_ledger": asdict(CERTIFICATION_LEDGER),
    }
    encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def campaign_declaration(case):
    """Recover the original experiment with explicit scalar-mean diagnostics."""
    from increment import (
        AlwaysValid,
        AsymptoticMean,
        ScalarMeanModel,
        SequentialCell,
        SequentialRegistration,
    )
    from tests.asymptotic_cases import mean_registration
    from tests.sequential_cases import registration

    if case.law == "scalar_mean":
        if (
            case.shape not in ("gaussian", "finite_moment")
            or case.stopping not in ("horizon", "first_rejection", "first_discovery")
            or case.scalar_control_mean <= 0
        ):
            raise ValueError("unsupported scalar campaign declaration")
        base = mean_registration()
        arms = ("treatment",)
        models = tuple(
            ScalarMeanModel.model_validate(
                {
                    **base.models[0].model_dump(),
                    "metric": f"m{i}",
                    "treatment_probability": F(case.allocation[1], sum(case.allocation)),
                    "rho": case.scalar_rho,
                    "start_count": case.scalar_start_count,
                }
            )
            for i in range(max(case.family, 1))
        )
    else:
        arms = ("t0",) if case.family == 1 else ("t0", "t1")
        base = registration(case.law)
        models = tuple(
            base.models[0].model_copy(update={"metric": f"m{i}"})
            for i in range(case.family // len(arms))
        )
    cells, truths = [], {}
    null_count = math.ceil(case.family * case.null_fraction)
    cell_alpha = (
        CERTIFICATION_DESIGN.alpha / case.family
        if case.law == "scalar_mean" and case.family > 1
        else CERTIFICATION_DESIGN.alpha
    )
    for i, (model, arm) in enumerate(product(models, arms)):
        segment = (("segment", "A"),) if case.dependence == "shared" and arm == "t1" else ()
        missing = case.missing and i == case.family - 1
        if case.law == "scalar_mean" and missing:
            segment = (("segment", "absent"),)
        cell = SequentialCell(
            metric=model.metric,
            group_id="absent" if missing and case.law != "scalar_mean" else arm,
            segment=segment,
            alpha=cell_alpha,
            family=True,
            null_lift=case.null,
            alternative=case.alternative,
        )
        cells.append(cell)
        relative = F(1) + case.null
        if i >= null_count:
            relative *= (
                CERTIFICATION_DESIGN.less_effect_multiplier
                if case.alternative == "less"
                else CERTIFICATION_DESIGN.greater_effect_multiplier
            )
        elif case.wrong_direction:
            relative *= 1 + (
                -case.wrong_direction if case.alternative == "less" else case.wrong_direction
            )
        if case.law == "scalar_mean":
            relative += case.scalar_treatment_shift / case.scalar_control_mean
        declared_null = i < null_count
        if case.law == "scalar_mean":
            null_ratio = 1 + case.null
            declared_null = (
                relative == null_ratio
                if case.alternative == "two-sided"
                else relative <= null_ratio
                if case.alternative == "greater"
                else relative >= null_ratio
            )
        truths[cell] = (relative, declared_null)
    schedule = campaign_schedule(case)
    reg = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "models": models,
            "definitions_id": (
                "scalar-campaign-" + campaign_identity((case,))
                if case.law == "scalar_mean"
                else base.definitions_id
            ),
            "roster": tuple(cells),
            "q": CERTIFICATION_DESIGN.q,
        }
    )
    policy = (
        AsymptoticMean(registration=reg)
        if case.law == "scalar_mean"
        else AlwaysValid(registration=reg)
    )
    return reg, policy, truths, arms, schedule


def campaign_batch(case, rng, reg, truths, arms, size, offset):
    """Preserve the original sampler, draw order, and floating transformations."""
    if case.law == "scalar_mean":
        return _scalar_campaign_batch(case, rng, reg, truths, size, offset)
    rows = []
    targets = {(cell.metric, cell.group_id): float(truths[cell][0]) for cell in reg.roster}
    for arm in ("control", *arms):
        count = size * case.allocation[0 if arm == "control" else 1]
        width = 1 if case.dependence == "shared" else len(reg.models)
        noise = (
            rng.random((count, width))
            if case.law == "bernoulli"
            else rng.normal(size=(count, width, 2))
        )
        for j in range(count):
            values = {}
            for k, model in enumerate(reg.models):
                ratio = targets.get((model.metric, arm), 1.0)
                baseline = case.rate if case.law == "bernoulli" else 5.0
                location = baseline * ratio
                z = noise[j, 0 if width == 1 else k]
                if case.law == "bernoulli":
                    assert 0 <= location <= 1
                    value = int(z < location)
                else:
                    sd = math.sqrt(case.variance_ratio) if arm != "control" else 1.0
                    numerator = location + sd * z[0]
                    value = (
                        float(numerator)
                        if case.law == "gaussian"
                        else (float(2 * numerator), float(2 + 0.2 * (z[0] + z[1])))
                    )
                values[model.metric] = value
            rows.append(
                {
                    "unit_id": f"{offset + len(rows):09d}",
                    "group_id": arm,
                    "values": values,
                    "segments": {"segment": "A" if j % 2 == 0 else "B"},
                }
            )
    return rows


def _scalar_campaign_batch(case, rng, reg, truths, size, offset):
    """Generate iid (assignment, joint outcome) units without balancing or sorting."""
    targets = {cell.metric: float(truths[cell][0]) for cell in reg.roster}
    width = 1 if case.dependence == "shared" else len(reg.models)
    rows = []
    for _ in range(size * sum(case.allocation)):
        rare = case.missing and rng.integers(1_000_000) == 0
        treated = rng.integers(sum(case.allocation)) < case.allocation[1]
        noise = (
            rng.standard_t(5, size=width)
            if case.shape == "finite_moment"
            else rng.normal(size=width)
        )
        sd = math.sqrt(case.variance_ratio) if treated else 1.0
        rows.append(
            {
                "unit_id": f"{offset + len(rows):09d}",
                "group_id": "treatment" if treated else "control",
                "values": {
                    model.metric: float(
                        float(case.scalar_control_mean)
                        * (targets[model.metric] if treated else 1.0)
                        + sd * noise[0 if width == 1 else k]
                    )
                    for k, model in enumerate(reg.models)
                },
                "segments": {"segment": "absent" if rare else "A"},
            }
        )
    return rows


def _diagnostic_selection(results):
    """Numerical e-BH analogue; no Gaussian values enter certified public evidence."""
    from increment.estimation._certified import log_interval

    ordered = sorted(range(len(results)), key=lambda i: results[i].log_e, reverse=True)
    selected = ()
    for rank, index in enumerate(ordered, start=1):
        threshold = (-log_interval(CERTIFICATION_DESIGN.q * rank / len(results))).hi
        if results[index].log_e >= threshold:
            selected = tuple(ordered[:rank])
    alpha = (
        min(CERTIFICATION_DESIGN.alpha, CERTIFICATION_DESIGN.q * len(selected) / len(results))
        if selected
        else None
    )
    return selected, alpha


def campaign_replication(case, rng, declaration, checkpoint, *, interim_geometry=True):
    """Persist actual executed state; nominal-DGP diagnostics are not certification.

    ``interim_geometry`` keeps the per-look event stream and the every-look
    interval census. A caller that will not read them turns it off; nothing a
    gate decides depends on either.
    """
    from increment import capture_sequential_snapshot, estimate_sequential
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family
    from increment.estimation.sequential_result import SequentialInferenceResult, _point
    from increment.estimation.sequential_runtime import _evaluate_sequential_diagnostic
    from increment.sequential_state import (
        _capture_sequential_diagnostic_snapshot,
        declare_sequential_freeze_cells,
    )

    reg, policy, truths, arms, schedule = declaration
    parent, offset, ever = None, 0, False
    results, selected, fcr_alpha = (), (), None
    public = case.law in ("bernoulli", "scalar_mean")
    validity = (
        "asymptotic_sequential"
        if case.law == "scalar_mean"
        else "exact_bernoulli"
        if public
        else "private_diagnostic"
    )
    # Interim look geometry is diagnostic, not gated: replication_statistics
    # never reads it. Large campaigns opt out of serialising it, and the census
    # is then omitted from the record rather than reported empty.
    observed_geometry = Counter() if interim_geometry else None
    for look, size in enumerate(schedule, start=1):
        if interim_geometry:
            checkpoint(
                {"phase": "look_started", "look": look, "rng_state": rng.bit_generator.state}
            )
        rows = campaign_batch(case, rng, reg, truths, arms, size, offset)
        capture = capture_sequential_snapshot if public else _capture_sequential_diagnostic_snapshot
        snapshot = capture(
            reg,
            rows,
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
            previous=parent,
            append=parent is not None,
        )
        parent, offset = snapshot, offset + len(rows)
        if interim_geometry:
            checkpoint(
                {
                    "phase": "captured",
                    "look": look,
                    "prefix_id": snapshot.prefix_id,
                    "revealed_units": offset,
                    "states": [json.loads(state.model_dump_json()) for state in snapshot.states],
                    "frozen": [json.loads(cp.model_dump_json()) for cp in snapshot.frozen],
                    **(
                        {
                            "snapshot": json.loads(snapshot.model_dump_json()),
                            "rng_state": rng.bit_generator.state,
                        }
                        if case.law == "scalar_mean"
                        else {}
                    ),
                }
            )
        if public:
            bundle = estimate_sequential(snapshot, policy)
            results = tuple(
                row.require_asymptotic_sequential_result()
                if case.law == "scalar_mean"
                else row.require_exact_sequential_result()
                for row in bundle.results
            )
        else:
            results = _evaluate_sequential_diagnostic(snapshot, policy)
        if interim_geometry:
            checkpoint(
                {
                    "phase": "evaluated",
                    "look": look,
                    "public_guaranteed_route": case.law == "bernoulli",
                    "public_inference_route": public,
                    "validity_regime": validity,
                    "results": [json.loads(result.model_dump_json()) for result in results],
                }
            )
        ever |= any(result.rejects() and truths[result.checkpoint.cell][1] for result in results)
        if observed_geometry is not None:
            observed_geometry.update(result.bounds.status for result in results)
        if public:
            cells = [
                (
                    sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell),
                    row,
                )
                for row in bundle.results
            ]
            outcome = select_sequential_family(
                cells,
                CERTIFICATION_DESIGN.q,
                policy,
                CERTIFICATION_DESIGN.alpha,
                computation=bundle,
            )
            assert outcome.n_family == case.family
            selected = tuple(i for i, (key, _) in enumerate(cells) if key in outcome.selected)
            fcr_alpha = outcome.fcr_alpha
        else:
            selected, fcr_alpha = _diagnostic_selection(results)
        if case.freeze and look < len(schedule):
            already = {c.cell for c in parent.frozen}
            to_freeze = [
                result.checkpoint.cell
                for i, result in enumerate(results)
                if (result.rejects() or (look - 1 == len(schedule) // 2 and i % 2 == 0))
                and result.checkpoint.cell not in already
                and result.checkpoint.control.n
                and result.checkpoint.treatment.n
            ]
            if to_freeze:
                parent = declare_sequential_freeze_cells(parent, to_freeze)
        stop = (
            (case.stopping == "first_rejection" and any(result.rejects() for result in results))
            or (case.stopping == "first_discovery" and bool(selected))
            or (
                case.stopping == "adaptive"
                and look >= 2
                and any(
                    cast(SequentialInferenceResult, result).log_e > 0 for result in results[::2]
                )
            )
        )
        if interim_geometry:
            checkpoint(
                {
                    "phase": "look_complete",
                    "look": look,
                    "selected": selected,
                    "family_size": len(results),
                    "fcr_alpha": None if fcr_alpha is None else str(fcr_alpha),
                    "stop": stop,
                    "frozen_cells": [json.loads(cp.cell.model_dump_json()) for cp in parent.frozen],
                }
            )
        if stop:
            break
    selected_summary = _selected_campaign_intervals(
        case, results, selected, fcr_alpha, bundle if public else None, truths, checkpoint
    )
    reasons = Counter(result.point_reason for result in results if result.point_reason is not None)
    declared = CERTIFICATION_DESIGN.declared_point_reasons
    unavailable = CERTIFICATION_DESIGN.unavailable_statuses
    return {
        "nominal_ever_null_rejection": int(ever),
        **selected_summary,
        "selected_cells": len(selected),
        "retained_cells": len(results),
        "available_points": sum(_point(result.checkpoint)[0] is not None for result in results),
        "certified_intervals": sum(result.bounds.status not in unavailable for result in results),
        "undeclared_point_reasons": sum(
            count for reason, count in reasons.items() if reason not in declared
        ),
        "interval_statuses": dict(Counter(result.bounds.status for result in results)),
        **(
            {"all_look_interval_statuses": dict(observed_geometry)}
            if observed_geometry is not None
            else {}
        ),
        "interim_geometry_observed": observed_geometry is not None,
        "missing_point_reasons": dict(reasons),
        "completed_looks": look,
        "public_guaranteed_route": case.law == "bernoulli",
        "public_inference_route": public,
        "validity_regime": validity,
        "source_lane": "synthetic_finalized_records",
        "claim_scope": "nominal-model diagnostic; the resolved ledger is not executed",
    }


def _selected_campaign_intervals(case, results, selected, fcr_alpha, bundle, truths, checkpoint):
    from increment.estimation.asymptotic_mean import AsymptoticMeanSet
    from increment.estimation.sequential_runtime import evaluate_checkpoint, reinvert_selected

    public = case.law in ("bernoulli", "scalar_mean")
    false = missed = unknown_coverage = nonnull = 0
    selected_widths = []
    for index in selected:
        original = results[index]
        if case.law == "scalar_mean":
            result = original
        elif public:
            assert fcr_alpha is not None
            result = reinvert_selected(
                bundle.results[index], fcr_alpha, ceiling=fcr_alpha
            ).require_sequential_result()
        else:
            assert fcr_alpha is not None
            result = evaluate_checkpoint(original.checkpoint, alpha=F(fcr_alpha))
        assert result.checkpoint == original.checkpoint
        truth, declared_null = truths[result.checkpoint.cell]
        bounds = result.bounds
        false += declared_null
        nonnull += not declared_null
        unknown = bounds.status in CERTIFICATION_DESIGN.unavailable_statuses
        unknown_coverage += unknown
        if isinstance(bounds, AsymptoticMeanSet):
            # A disconnected set's outer endpoints do not describe membership.
            missed += not any(
                (part.lower is None or part.lower <= truth)
                and (part.upper is None or truth <= part.upper)
                for part in bounds.components
            )
        elif not unknown:
            missed += (
                bounds.empty
                or (bounds.lower is not None and truth < bounds.lower)
                or (bounds.upper is not None and truth > bounds.upper)
            )
        width = (
            str(bounds.upper - bounds.lower)
            if not bounds.empty and bounds.lower is not None and bounds.upper is not None
            else None
        )
        selected_widths.append(
            {
                "width": width,
                "geometry": bounds.status,
                **(
                    {"components": [json.loads(p.model_dump_json()) for p in bounds.components]}
                    if isinstance(bounds, AsymptoticMeanSet)
                    else {}
                ),
            }
        )
        checkpoint(
            {
                "phase": "selected_interval",
                "cell_index": index,
                "nominal_truth": str(truth),
                "result": json.loads(result.model_dump_json()),
            }
        )
    denominator = max(len(selected), 1)
    return {
        "nominal_fdp": str(F(false, denominator)),
        "nominal_fcp_lower": str(F(missed, denominator)),
        "nominal_fcp_upper": str(F(missed + unknown_coverage, denominator)),
        "nonnull_discovery": int(nonnull > 0),
        "selected_widths": selected_widths,
    }
