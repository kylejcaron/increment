"""Sharp raw-distribution exchangeability; separate from effect inference."""

import math
from fractions import Fraction
from itertools import combinations

import numpy as np

from increment.winsor import WinsorPermutationTest, WinsorRawState, winsor_refuse


def conditional_permutation_test(
    raw: WinsorRawState, control: str, treatment: str, *, seed: int = 2729, stream: int = 0
) -> WinsorPermutationTest:
    from increment.estimation.winsor import _linear_cutoff

    if control == treatment or seed < 0 or stream < 0:
        winsor_refuse(
            "invalid_state",
            "Permutation inference requires distinct arms and nonnegative stream identifiers.",
        )
    c, t = raw.arm(control).values, raw.arm(treatment).values
    cutoff = _linear_cutoff(raw)
    values = tuple(Fraction(min(y, cutoff)) for y in (*c, *t))
    nc, nt = len(c), len(t)
    total = sum(values)

    def statistic(indices):
        sc = sum(values[i] for i in indices)
        return abs((total - sc) / nt - sc / nc)

    observed = statistic(range(nc))
    count = math.comb(nc + nt, nc)
    if count <= 100000:
        extreme = sum(
            statistic(indices) >= observed for indices in combinations(range(nc + nt), nc)
        )
        return WinsorPermutationTest(
            raw=raw,
            control=control,
            treatment=treatment,
            exact=True,
            assignments=count,
            at_least_as_extreme=extreme,
        )
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence(seed, spawn_key=(stream,)))
    )
    extreme = sum(statistic(rng.permutation(nc + nt)[:nc]) >= observed for _ in range(1999))
    return WinsorPermutationTest(
        raw=raw,
        control=control,
        treatment=treatment,
        exact=False,
        assignments=1999,
        at_least_as_extreme=extreme,
        seed=seed,
        stream=stream,
    )
