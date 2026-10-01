"""Prospective unit-cycle calibration design and independent potential-outcome oracle.

No production estimate, reducer, or planning function enters this oracle.
For each unit draw a type S=+/-1 equiprobably, independently of orders/noise.
The complete schedule has one retained aggregate per period. Counts describe
micro-observations inside that aggregate, never missing schedule cells. Total
and pre-normalized mean are distinct response meanings with different potential outcomes.

Write D_cz for the treated-minus-control potential contrast for order z,
w_1=1/(2p), w_0=1/(2(1-p)), A=C^-1 sum w_Z D_cZ, G=C^-1 sum w_Z.
E[A|S]=delta+.4S. Conditional independent-order variance is the sum of
cycle variances divided by C^2; persistent treatment noise additionally gives
rho*sigma^2*sum_{c!=d} l_c*l_d/C^2, l_c=(load_c0+load_c1)/2.
Var(A) adds Var(.4S)=.16. Var(G)=(1-2p)^2/(4p(1-p)C).
Cov(A,G)=C^-2 sum_c E_S[sum_z m_cz/(4p_z)-(m_c0+m_c1)/2].
Thus Var(A+dG)=Var(A)+2d Cov(A,G)+d^2 Var(G), without another cycle divisor.

Every cell has period, cycle and type heterogeneity. The weak population null
is delta=0; its unit/period effects are not a sharp null. Exact enumeration
below is a moment oracle conditional on fixed potentials, not a coverage proof.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np

MANIFEST_PATH = Path(__file__).with_name("unit_cycle_prospective.json")
MANIFEST = json.loads(MANIFEST_PATH.read_text())
NOISE_SD = 1.0


@dataclass(frozen=True)
class Cell:
    units: int
    cycles: int
    probability_ct: float
    noise: str
    correlation: float
    counts: str
    effect: float
    response_meaning: str

    @property
    def id(self) -> str:
        return (
            f"n{self.units}-c{self.cycles}-p{self.probability_ct}-{self.noise}"
            f"-rho{self.correlation}-{self.counts}-d{self.effect}-{self.response_meaning}"
        )


CELLS = tuple(
    Cell(*values)
    for values in product(*(MANIFEST["axes"][axis] for axis in MANIFEST["axis_order"]))
)


def schedule(cell: Cell, effect: float):
    """Return fixed [type,cycle,period] counts, baseline, treatment effect, load."""
    typ = np.array([-1.0, 1.0])[:, None, None]
    cycle = np.arange(cell.cycles)[None, :, None]
    period = np.arange(2)[None, None, :]
    counts = np.full((2, cell.cycles, 2), 2)
    if cell.counts == "unequal":
        counts = 1 + 2 * ((cycle + period + (typ > 0)) % 2)
    load = (
        counts.astype(float) if cell.response_meaning == "retained_total" else np.ones_like(counts)
    )
    baseline = load * (2 + 0.3 * cycle + 0.6 * period + 0.4 * typ * period)
    centered_cycle = (2 * cycle - (cell.cycles - 1)) / max(1, cell.cycles - 1)
    treatment = effect + 0.4 * typ + 0.2 * centered_cycle + 0.35 * (2 * period - 1)
    return counts, baseline, treatment, load


def population_moments(cell: Cell, effect: float) -> tuple[float, float, float]:
    """Exact-rational integration, rounded only at the output boundary."""
    moments = law_moments(joint_law(cell, effect))
    return float(moments["variance_a"]), float(moments["variance_g"]), float(moments["covariance"])


def enumerate_orders(control, treated, probability_ct: float):
    """Exhaust every Bernoulli order on fixed [unit,cycle,period] potentials.

    Return assignment masses, unit contributions, slopes and CT counts. This
    deliberately uses scalar potential-outcome arithmetic, not runtime helpers.
    """
    n, cycles, _ = control.shape
    masses, values, slopes, counts = [], [], [], []
    for bits in product((0, 1), repeat=n * cycles):
        mass = math.prod(probability_ct if z else 1 - probability_ct for z in bits)
        unit_values, unit_slopes = [], []
        for u in range(n):
            contributions, weights = [], []
            for c in range(cycles):
                z = bits[u * cycles + c]
                t, other = (1, 0) if z else (0, 1)
                weight = 1 / (2 * (probability_ct if z else 1 - probability_ct))
                contributions.append(weight * (treated[u, c, t] - control[u, c, other]))
                weights.append(weight)
            unit_values.append(math.fsum(contributions) / cycles)
            unit_slopes.append(math.fsum(weights) / cycles)
        masses.append(mass)
        values.append(unit_values)
        slopes.append(unit_slopes)
        counts.append(sum(bits))
    return np.array(masses), np.array(values), np.array(slopes), np.array(counts)


def streams(cell_index: int):
    """DGP/order streams are frozen independently of any inference resampling."""
    return tuple(
        np.random.default_rng(np.random.SeedSequence([MANIFEST["dgp_seed"], cell_index, stream]))
        for stream in range(4)
    )


def innovations(rng, law: str, shape):
    if law == "normal":
        return rng.normal(size=shape)
    if law == "centered_lognormal":
        return (rng.lognormal(size=shape) - math.exp(0.5)) / math.sqrt(math.e * math.expm1(1))
    return (rng.gamma(2, size=shape) - 2) / math.sqrt(2)


def draw_periods(cell: Cell, count: int, rngs):
    """Generate zero-effect potential period aggregates, orders and unit types.

    Only outcomes are generated here. The scientific harness must call the
    production contribution/reference kernels, with public-source parity probes.
    """
    order_rng, type_rng, noise_rng, reuse_rng = rngs
    shape = (count, cell.units, cell.cycles)
    ct = order_rng.random(shape) < cell.probability_ct
    types = type_rng.integers(0, 2, (count, cell.units))
    counts, baseline, treatment, load = schedule(cell, 0.0)
    counts, baseline, treatment, load = (
        item[types] for item in (counts, baseline, treatment, load)
    )
    common = innovations(noise_rng, cell.noise, (count, cell.units, 1, 1))
    independent = innovations(noise_rng, cell.noise, (*shape, 2, 3))
    mask = np.arange(3) < counts[..., None]
    averages = np.sum(independent * mask, axis=-1) / counts
    reuse = reuse_rng.random((count, cell.units, 1, 1)) < cell.correlation
    noise = NOISE_SD * load * np.where(reuse, common, averages)
    treated = np.stack((~ct, ct), axis=-1)
    periods = baseline + treated * (treatment + noise)
    return periods, ct


# Keep original grid indices: sharing MDE studies must not renumber seed streams.
DESIGNS = tuple((i, cell) for i, cell in enumerate(CELLS) if cell.effect == 0)


def assignment(cell):
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )

    return SwitchbackAssignment(
        sequence=IndependentBernoulliOrder(probability_ct=cell.probability_ct),
        window=SwitchbackWindow(washout_steps=0, observation_steps=1),
    )


def joint_law(cell, reference_effect=None):
    """Describe potentials without consulting production moments or reducers."""
    from increment.semantics.unit_cycle import UnitCycleJointLaw

    effect = cell.effect if reference_effect is None else reference_effect
    counts, baseline, treatment, load = schedule(cell, effect)
    types = []
    for typ in range(2):
        cycles = []
        for c in range(cell.cycles):
            difference = float(baseline[typ, c, 1] - baseline[typ, c, 0])
            cycles.append(
                {
                    "ct_mean": float(treatment[typ, c, 1] + difference),
                    "tc_mean": float(treatment[typ, c, 0] - difference),
                    "ct_noise_load": float(load[typ, c, 1]),
                    "tc_noise_load": float(load[typ, c, 0]),
                    "ct_innovation_count": int(counts[typ, c, 1]),
                    "tc_innovation_count": int(counts[typ, c, 0]),
                }
            )
        types.append({"weight": 1, "cycles": cycles})
    innovation = {"kind": cell.noise}
    if cell.noise != "normal":
        innovation["shape"] = 1.0 if cell.noise == "centered_lognormal" else 2.0
    return UnitCycleJointLaw(
        assignment=assignment(cell),
        metric="outcome",
        control_group="control",
        treatment_group="treatment",
        response_meaning=cell.response_meaning,
        types=types,
        innovation=innovation,
        reuse_probability=cell.correlation,
        provenance={
            "assumption_id": "I14-independent-potential-law",
            "assumption_version": "2",
            "justification": "tests._unit_cycle_design.schedule and rational integration",
            "declaration_id": replace(cell, effect=0).id,
        },
    )


def upward(value: Fraction) -> float:
    """Smallest binary64 at least the exact nonnegative rational value."""
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction(result) < value else result


def downward(value: Fraction) -> float:
    result = float(value)
    return math.nextafter(result, -math.inf) if Fraction(result) > value else result


def sqrt_upper(value: Fraction) -> float:
    result = math.sqrt(float(value))
    while Fraction(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    return result


def law_moments(law):
    """Centered rational integration of type, order, reuse and innovation variance.

    Common noise has variance E[B^2], B=mean(w*l). Independent noise has
    variance sum E[(w*l)^2/m]/C^2. Signed loads and unequal weights are allowed.
    """
    f = Fraction
    p = f(law.assignment.sequence.probability_ct)
    q, c, rho = 1 - p, len(law.types[0].cycles), f(law.reuse_probability)
    weights = [f(t.weight) for t in law.types]
    weights = [w / sum(weights) for w in weights]
    means = [sum((f(x.ct_mean) + f(x.tc_mean)) / 2 for x in t.cycles) / c for t in law.types]
    reference = sum(w * m for w, m in zip(weights, means, strict=True))
    vg = (1 - 2 * p) ** 2 / (4 * p * q * c)

    def integrate(center):
        variance, covariance = f(0), f(0)
        for weight, typ, mean in zip(weights, law.types, means, strict=True):
            order_var, cov, independent, load_var, mean_load = (f(0) for _ in range(5))
            for x in typ.cycles:
                a, b = (f(x.ct_mean) - center) / (2 * p), (f(x.tc_mean) - center) / (2 * q)
                order_var += p * q * (a - b) ** 2
                cov += p * q * (a - b) * (1 / (2 * p) - 1 / (2 * q))
                la, lb = f(x.ct_noise_load) / (2 * p), f(x.tc_noise_load) / (2 * q)
                independent += (
                    p * la * la / x.ct_innovation_count + q * lb * lb / x.tc_innovation_count
                )
                load_var += p * q * (la - lb) ** 2
                mean_load += p * la + q * lb
            noise = rho * (mean_load**2 + load_var) + (1 - rho) * independent
            variance += weight * ((order_var + noise) / c**2 + (mean - reference) ** 2)
            covariance += weight * cov / c**2
        return variance, covariance

    va, cov = integrate(f(0))
    v0, cov0 = integrate(reference)
    return {
        "reference_effect": reference,
        "mean_g": f(1),
        "variance_a": va,
        "variance_g": vg,
        "covariance": cov,
        "residual_mean": f(0),
        "residual_variance": v0,
        "residual_covariance": cov0,
    }


def envelope_for(law):
    from increment.semantics.unit_cycle import UnitCycleVarianceEnvelope

    return UnitCycleVarianceEnvelope(
        assignment=law.assignment,
        metric=law.metric,
        control_group=law.control_group,
        treatment_group=law.treatment_group,
        response_meaning=law.response_meaning,
        cycles_per_unit=len(law.types[0].cycles),
        provenance=law.provenance,
        residual_variance_upper=upward(law_moments(law)["residual_variance"]),
    )


@cache
def admission_oracle(draws, probability_ct):
    """Exhaust scipy's actual test mask; integrate its mass with exact integers."""
    from scipy.stats import binomtest

    p = Fraction(probability_ct)
    a, denominator = p.numerator, p.denominator
    b = denominator - a
    refused = tuple(
        k for k in range(draws + 1) if binomtest(k, draws, probability_ct).pvalue < 1e-6
    )
    masses = [(k, math.comb(draws, k) * a**k * b ** (draws - k)) for k in refused]
    divisor = denominator**draws
    mass = Fraction(sum(v for _, v in masses), divisor)
    removed_ct = Fraction(sum(k * v for k, v in masses), draws * divisor)
    return refused, mass, removed_ct


def analytic_evidence(cell, law=None):
    """Proof evidence, never simulated rates or invented replication counts."""
    law = joint_law(cell) if law is None else law
    moments = law_moments(law)
    _, r, removed_ct = admission_oracle(cell.units * cell.cycles, cell.probability_ct)
    alpha = Fraction(MANIFEST["alpha"])
    p = Fraction(cell.probability_ct)
    refusal_upper = Fraction(upward(r))
    effective = alpha - refusal_upper
    assert effective > 0
    g_min = 1 / (2 * max(p, 1 - p))
    v0 = moments["residual_variance"]
    weights = [Fraction(t.weight) for t in law.types]
    total = sum(weights)
    mean_ct = (
        sum(
            w * sum(Fraction(x.ct_mean) for x in t.cycles) / cell.cycles
            for w, t in zip(weights, law.types, strict=True)
        )
        / total
    )
    mean_tc = (
        sum(
            w * sum(Fraction(x.tc_mean) for x in t.cycles) / cell.cycles
            for w, t in zip(weights, law.types, strict=True)
        )
        / total
    )
    removed_point = removed_ct * mean_ct / (2 * p) + (r - removed_ct) * mean_tc / (2 * (1 - p))
    admitted_mean = (moments["reference_effect"] - removed_point) / (1 - r)
    bias = admitted_mean - moments["reference_effect"]
    point_variance = moments["variance_a"] / cell.units
    assert bias * bias * (1 - r) <= r * point_variance
    return {
        "evidence_kind": "analytic_proof",
        "cell": cell.id,
        "moments": {k: float(v) for k, v in moments.items()},
        "exact_moments": {k: str(v) for k, v in moments.items()},
        "residual_centering": "A0=A_ref-E[A_ref]*G; E[A0]=0",
        "positive_slope_lower": downward(g_min),
        "slope_upper": upward(1 / (2 * min(p, 1 - p))),
        "refusal_mass_upper": float(refusal_upper),
        "effective_alpha": float(effective),
        "unconditional_miss_upper": float(refusal_upper + effective),
        "coverage_lower": float(1 - alpha),
        "width_upper": sqrt_upper(4 * v0 / (cell.units * effective * g_min**2)),
        "admitted_point_mean": float(admitted_mean),
        "admitted_point_bias": float(bias),
        "standardized_admitted_bias": (
            float(abs(bias)) / math.sqrt(float(point_variance)) if point_variance else 0.0
        ),
        "standardized_admitted_bias_upper": sqrt_upper(r / (1 - r)),
        "assumptions": "independent units; declared law; no additional numerical failures",
    }


def _dyadic_sum(values):
    """Exact sum of binary64 values: align every term on the largest power-of-two scale."""
    ratios = [value.as_integer_ratio() for value in values]
    scale = max((denominator for _, denominator in ratios), default=1)
    return Fraction(
        sum(numerator * (scale // denominator) for numerator, denominator in ratios), scale
    )


def scalar_observation(periods, ct, p):
    """Independent scalar HT and slope reduction of supplied period responses.

    Differences are grouped by realised order, so each unit needs two exact
    dyadic sums and one weighted combination instead of a rational per cycle.
    """
    values, slopes = [], []
    probability = Fraction(p)
    weights = (1 / (2 * (1 - probability)), 1 / (2 * probability))
    for rows, orders in zip(periods.tolist(), ct.tolist(), strict=True):
        cycles = len(rows)
        differences = ([], [])
        for (first, second), is_ct in zip(rows, orders, strict=True):
            differences[is_ct].extend((second, -first))
        ct_count = sum(orders)
        value = weights[1] * _dyadic_sum(differences[1]) - weights[0] * _dyadic_sum(differences[0])
        values.append(value / cycles)
        slopes.append((ct_count * weights[1] + (cycles - ct_count) * weights[0]) / cycles)
    return (
        sum(values, Fraction()) / len(values),
        sum(slopes, Fraction()) / len(slopes),
        [float(value) for value in values],
        [float(slope) for slope in slopes],
    )


def scalar_reference(point, slope, envelope, n, refusal_upper, *, alternative="two-sided", null=0):
    """Independent Markov/Cantelli inversion; admission bound checked separately."""
    alpha = Fraction(downward(Fraction(MANIFEST["alpha"]) - Fraction(refusal_upper)))
    variance = Fraction(envelope.residual_variance_upper) / n
    squared = variance / alpha if alternative == "two-sided" else variance * (1 - alpha) / alpha
    cutoff = sqrt_upper(squared)
    x, g, c = Fraction(point), Fraction(slope), Fraction(cutoff)
    residual = x - Fraction(null) * g
    if alternative == "two-sided":
        tail = min(Fraction(1), variance / residual**2) if residual else Fraction(1)
        lower, upper = downward((x - c) / g), upward((x + c) / g)
    else:
        distance = residual if alternative == "greater" else -residual
        tail = variance / (variance + distance**2) if distance > 0 else Fraction(1)
        lower = downward((x - c) / g) if alternative == "greater" else None
        upper = upward((x + c) / g) if alternative == "less" else None
    p_value = upward(min(Fraction(1), Fraction(refusal_upper) + tail))
    rejected = p_value < MANIFEST["alpha"]
    return {
        "lower": lower,
        "upper": upper,
        "cutoff": cutoff,
        "p_value": p_value,
        "rejected": rejected,
    }
