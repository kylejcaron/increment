"""Exact false-call mass and a sampled-inversion counterexample on 240 groups of 59.

python tests/oracles/generate/gen_spillover_null.py
See calibration/protocols/spillover_published_intervals.md for the synthetic schedule.
"""

from fractions import Fraction
from math import comb, log10, sqrt
from statistics import NormalDist

import numpy as np

GROUPS = 240
ARM = GROUPS // 2
SIZE = 59
UNOFF = SIZE - 1
EXCEPTIONAL = 4
EVENTS = 14
OBSERVED_HITS = 28
ALPHA = Fraction(1, 20)
# Denominator shared by the sharp target and HT contrast on this balanced design.
SCALE = GROUPS * SIZE * UNOFF


def arm_summary(counts, denominator):
    """Mean and sample variance of rates, from multiplicities of their numerators."""
    mean = Fraction(sum(n * count for n, count in counts), ARM * denominator)
    ss = sum(count * (Fraction(n, denominator) - mean) ** 2 for n, count in counts)
    return mean, ss / (ARM - 1)


def null_summary(exceptional, hits):
    ordinary = ARM - exceptional
    active, var1 = arm_summary(
        ((UNOFF, exceptional), (EVENTS - 1, hits), (EVENTS, ordinary - hits)), UNOFF
    )
    exceptions_left = EXCEPTIONAL - exceptional
    base, var0 = arm_summary(((SIZE, exceptions_left), (EVENTS, ARM - exceptions_left)), SIZE)
    return active - base, (var0 + var1) / ARM


def exact_null():
    observed, obs_variance = null_summary(EXCEPTIONAL, OBSERVED_HITS)
    # Bound the normal critical value on both sides; no assignment changes its decision
    # anywhere in this bracket. All probability masses and comparisons below are exact.
    z_low, z_high = Fraction("1.9599639"), Fraction("1.9599640")
    normal = NormalDist()
    assert normal.cdf(-float(z_high)) < ALPHA / 2 < normal.cdf(-float(z_low))
    total = comb(GROUPS, ARM) * SIZE**ARM
    mass_sum = false_calls = sharp_tail = 0
    for x in range(EXCEPTIONAL + 1):
        ordinary = ARM - x
        first = comb(EXCEPTIONAL, x) * comb(GROUPS - EXCEPTIONAL, ordinary)
        for y in range(ordinary + 1):
            second = comb(ordinary, y) * EVENTS**y * (SIZE - EVENTS) ** (ordinary - y) * SIZE**x
            mass = first * second
            point, variance = null_summary(x, y)
            reject = point**2 > z_low**2 * variance
            assert reject == (point**2 > z_high**2 * variance)
            false_calls += mass * reject
            sharp_tail += mass * (abs(point) >= abs(observed))
            mass_sum += mass
    assert mass_sum == total
    p = Fraction(sharp_tail, total)
    assert p >= ALPHA
    radius = -normal.inv_cdf(float(ALPHA / 2)) * sqrt(obs_variance)
    point = float(observed)
    print(f"point {point:.9f}; old Wald [{point - radius:.9f}, {point + radius:.9f}]")
    print(f"old Wald false-call probability {float(Fraction(false_calls, total)):.12f}")
    print(f"exact sharp p on observed draw {float(p):.12f} (zero retained)")
    return observed


def truth(base, active):
    # tau = sum_i (T_i1 - T_i0) / (GROUPS * SIZE), scaled to the HT grid.
    return int((active.sum() - base.sum()) * UNOFF)


def sampled_tail(rng, base, active, observed, draws):
    # The first ARM of a random ranking select a uniform fixed-count subset. Do
    # not silently break a floating-point tie at its boundary on the fixed streams.
    ranks = rng.random((draws, GROUPS))
    chosen = np.argpartition(ranks, ARM - 1, axis=1)[:, :ARM]
    cutoff = np.max(np.take_along_axis(ranks, chosen, axis=1), axis=1, keepdims=True)
    assert np.all((ranks == cutoff).sum(axis=1) == 1)
    totals = active[chosen]
    hits = rng.integers(SIZE, size=totals.shape) < totals
    unoffered_events = (totals - hits).sum(axis=1)
    base_events = base.sum() - base[chosen].sum(axis=1)
    scores = 2 * SIZE * unoffered_events - 2 * UNOFF * base_events
    tau = truth(base, active)
    return int(np.count_nonzero(abs(scores - tau) >= abs(observed - tau)))


def interpolate(tails, threshold, *, upper):
    # Walk inwards from the sharp bound to the outermost accepted effect. If
    # not the bound itself, interpolate against its nearest rejected neighbor.
    ordered = sorted(tails, reverse=upper)
    i = next(i for i, value in enumerate(ordered) if tails[value] >= threshold)
    accepted = ordered[i]
    if i == 0:
        return accepted / SCALE
    rejected = ordered[i - 1]
    fraction = (threshold - tails[accepted]) / (tails[rejected] - tails[accepted])
    return float((accepted + fraction * (rejected - accepted)) / SCALE)


def sampled_interval(seed, observed):
    """RH §4.2 search on group margins; not exhaustive and not the R random stream."""
    rng = np.random.Generator(np.random.PCG64(seed))
    schedules = draws = 100
    threshold = ALPHA * draws
    ordinary = ARM - EXCEPTIONAL
    events = np.repeat(
        [UNOFF, EVENTS - 1, EVENTS], [EXCEPTIONAL, OBSERVED_HITS, ordinary - OBSERVED_HITS]
    )
    base_min = np.repeat([EVENTS, 0], ARM)
    base_max = np.repeat([EVENTS, SIZE], ARM)
    active_min = np.r_[np.zeros(ARM, dtype=int), events]
    active_max = np.r_[np.full(ARM, SIZE), events + 1]
    # For a given effect, only its largest sampled tail probability enters the hull.
    tails = {observed: draws}
    for side in (-1, 1):
        base, active = (base_max, active_min) if side < 0 else (base_min, active_max)
        extreme = truth(base, active)
        tails[extreme] = sampled_tail(rng, base, active, observed, draws)
        if tails[extreme] < threshold:
            current = observed
            step = 1 / schedules
            center = (base.sum() + active.sum()) / (GROUPS * SIZE)
            for _ in range(schedules // 2):
                target = (current + step * (extreme - current)) / SCALE
                p0, p1 = np.clip([(center - target) / 2, (center + target) / 2], 0, 1)
                b = base_min + rng.binomial(base_max - base_min, p0)
                a = active_min + rng.binomial(active_max - active_min, p1)
                candidate = truth(b, a)
                if side * (candidate - current) < 0:
                    step += 1 / schedules
                    continue
                hits = sampled_tail(rng, b, a, observed, draws)
                tails[candidate] = max(tails.get(candidate, 0), hits)
                if hits >= threshold:
                    current = candidate

    return interpolate(tails, threshold, upper=False), interpolate(tails, threshold, upper=True)


def main():
    observed = exact_null()
    observed_units = observed * SCALE
    assert observed_units.denominator == 1
    log_first = log10(comb(GROUPS, ARM))
    unknown = 2 * GROUPS * SIZE - ARM * (SIZE + UNOFF)
    print(
        f"log10 labeled schedules {unknown * log10(2):.2f}; "
        f"assignments {log_first + ARM * log10(SIZE):.2f}; "
        f"double first-stage matrix bytes {log_first + log10(8 * GROUPS):.2f}"
    )
    print(f"NumPy {np.__version__}; PCG64, B2=C2=100")
    for seed in (519, 520):
        lo, hi = sampled_interval(seed, int(observed_units))
        print(f"sampled RH {seed}: [{lo:.9f}, {hi:.9f}]")
        assert lo > 0  # excludes a compatible sharp schedule with exact p >= alpha


if __name__ == "__main__":
    main()
