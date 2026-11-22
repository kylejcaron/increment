"""Exact hull of Rigdon–Hudgens sharp tests for tiny synthetic two-stage designs.

python tests/oracles/generate/gen_spillover_inversion.py
See calibration/protocols/spillover_published_intervals.md for targets and assumptions.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import comb, lcm, prod


@dataclass(frozen=True)
class Group:
    size: int
    quotas: tuple[int, ...]
    stratum: int
    policy: int
    events: int  # observed among the unoffered

    def compatible(self, policy: int) -> range:
        quota = self.quotas[policy]
        if quota == self.quotas[self.policy]:
            return range(self.events, self.events + quota + 1)
        return range(self.size + 1)


def first_stages(groups):
    """Uniform assignments within strata, including all nuisance policy labels."""
    strata = sorted({g.stratum for g in groups})
    blocks = [[i for i, g in enumerate(groups) if g.stratum == h] for h in strata]
    splits = [sorted(set(permutations(groups[i].policy for i in block))) for block in blocks]
    for labels in product(*splits):
        assignment = [0] * len(groups)
        for block, row in zip(blocks, labels, strict=True):
            for i, policy in zip(block, row, strict=True):
                assignment[i] = policy
        yield tuple(assignment)


def count_law(n, unoffered, total, unit_score):
    """(HT score, subset multiplicity) for every possible unoffered event count."""
    return tuple(
        (e * unit_score, comb(total, e) * comb(n - total, unoffered - e))
        for e in range(max(0, unoffered - (n - total)), min(unoffered, total) + 1)
    )


def sharp_p(schedule, truth, observed, assignments, laws):
    tail = Fraction(0)
    cutoff = abs(observed - truth)
    for assignment in assignments:
        # Nuisance labels have zero score; their conditional subset mass sums to one.
        distributions = [
            law(policy, margins)
            for law, policy, margins in zip(laws, assignment, schedule, strict=True)
        ]
        hits = 0
        for outcomes in product(*distributions):
            if abs(sum(score for score, _ in outcomes) - truth) >= cutoff:
                hits += prod(count for _, count in outcomes)
        support = prod(sum(count for _, count in distribution) for distribution in distributions)
        tail += Fraction(hits, support)
    return tail / len(assignments)


def group_laws(group, pair, unit_scores):
    base, active = pair
    counts = {
        (policy, total): count_law(
            group.size, group.size - group.quotas[policy], total, unit_scores[policy]
        )
        for policy in pair
        for total in group.compatible(policy)
    }

    def lookup(policy, margins):
        _, base_events, active_events = margins
        if policy == base:
            return counts[policy, base_events]
        if policy == active:
            return counts[policy, active_events]
        return ((0, 1),)

    return lookup


def invert(groups, pair, alpha, weights):
    assignments = tuple(first_stages(groups))
    base, active = pair
    coefficients = []
    for i, (w, g) in enumerate(zip(weights, groups, strict=True)):
        frequencies = Counter(row[i] for row in assignments)
        coefficients.append(
            {
                s: sign * w / (Fraction(frequencies[s], len(assignments)) * (g.size - g.quotas[s]))
                for s, sign in ((base, -1), (active, 1))
            }
        )
    # Put the observed HT, every randomization and every sharp truth on one integer grid.
    truth_units = [w / g.size for w, g in zip(weights, groups, strict=True)]
    scale = lcm(
        *(x.denominator for x in truth_units),
        *(x.denominator for c in coefficients for x in c.values()),
    )
    units = [{s: int(x * scale) for s, x in c.items()} for c in coefficients]
    observed = sum(c.get(g.policy, 0) * g.events for g, c in zip(groups, units, strict=True))
    laws = [group_laws(g, pair, c) for g, c in zip(groups, units, strict=True)]
    choices = [
        [
            (int(weight * scale) * (y - x), x, y)
            for x, y in product(g.compatible(base), g.compatible(active))
            if g.quotas[base] != g.quotas[active] or x == y
        ]
        for g, weight in zip(groups, truth_units, strict=True)
    ]

    def effect(schedule):
        return sum(delta for delta, _, _ in schedule)

    ordered = sorted(product(*choices), key=effect)

    def endpoint(candidates):
        for schedule in candidates:
            value = effect(schedule)
            p = sharp_p(schedule, value, observed, assignments, laws)
            if p >= alpha:
                return Fraction(value, scale), p
        raise ValueError("No compatible schedule passed the sharp test")

    return Fraction(observed, scale), endpoint(ordered), endpoint(reversed(ordered))


def report(label, groups, active, alpha, *, person):
    if person:
        roster = sum(g.size for g in groups)
        weights = [Fraction(g.size, roster) for g in groups]
    else:
        weights = [Fraction(1, len(groups))] * len(groups)
    point, (lo, p_lo), (hi, p_hi) = invert(groups, (0, active), alpha, weights)
    target = "person" if person else "group"
    print(
        f"{label} {target} {active}-0: point {point}, hull [{lo}, {hi}], "
        f"endpoint sharp p ({p_lo}, {p_hi})"
    )


def main():
    alpha = Fraction(1, 20)
    for label, sizes in (("equal", (3, 3, 3, 3)), ("unequal", (2, 3, 2, 4))):
        groups = [
            Group(n, (0, 1), 0, p, e)
            for n, p, e in zip(sizes, (0, 0, 1, 1), (0, 1, 1, 2), strict=True)
        ]
        report(label, groups, 1, alpha, person=False)
        if label == "unequal":
            report(label, groups, 1, alpha, person=True)
    groups = [
        Group(2, (0, 1, 1), 0, 0, 0),
        Group(3, (0, 1, 2), 0, 1, 1),
        Group(3, (0, 1, 2), 0, 2, 1),
        Group(2, (0, 1, 1), 1, 2, 1),
        Group(3, (0, 1, 2), 1, 0, 2),
        Group(3, (0, 1, 2), 1, 1, 0),
    ]
    for active in (1, 2):
        report("blocked", groups, active, alpha / 2, person=True)


if __name__ == "__main__":
    main()
