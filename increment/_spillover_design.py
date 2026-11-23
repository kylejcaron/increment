"""Frozen declarations and group summaries for a two-stage offer experiment."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from operator import le
from types import MappingProxyType
from typing import Never, TypeGuard

import numpy as np

from increment.errors import InvalidRequestError, raiser, refusals

type Identifier = str | int
SPILLOVER_ESTIMATOR = "spillover.two_stage"

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "spillover.design.required": "Provide the two-stage design frozen before assignment.",
        "spillover.design.invalid": "Invalid {field}: {requirement}.",
        "spillover.design.duplicate": "Duplicate keys in {field}: {requirement}.",
        "spillover.design.quota": "Invalid {field}: {requirement}.",
        "spillover.log.keys": "Invalid {field} records: {requirement}.",
        "spillover.log.binary": "Invalid {field} value: {requirement}.",
        "spillover.log.quota": "The {field} log disagrees with the design: {requirement}.",
        "spillover.roster.changed": "The frozen {field} no longer matches: {requirement}.",
        "spillover.planning.baseline_rate": (
            "Use a pre-assignment event rate between 0 and 1, or omit it to skip the power check.",
            ("baseline_rate",),
        ),
        "spillover.planning.meaningful_effect": (
            "Give a nonzero change between -1 and 1 in the probability of the outcome.",
            ("meaningful_effect",),
        ),
    },
)
_raise = raiser(_REFUSALS)

_REQUIREMENTS = {
    ("spillover.design.invalid", "offered_group"): "use different labels for offer and no offer",
    ("spillover.design.invalid", "window_days"): "give the conversion a fixed positive day count",
    ("spillover.design.invalid", "metric"): (
        "choose one fixed-window conversion for the experiment unit, without a margin or rollout cost"
    ),
    ("spillover.design.invalid", "plan"): (
        "declare one primary conversion and its alpha, without other methods or decision rules"
    ),
    ("spillover.design.invalid", "design"): (
        "use the two-stage policy comparison, not an individual-arm analysis"
    ),
    ("spillover.design.invalid", "experiment"): (
        "use the eligible roster and group offer counts without one-stage allocation, "
        "an assignment ledger, a trigger, cluster adjustment, pre-periods, breakouts or factors"
    ),
    ("spillover.design.invalid", "alpha"): "choose a false-alarm rate strictly between 0 and 1",
    (
        "spillover.design.invalid",
        "roster",
    ): "supply person and group IDs of one type each and named strata",
    (
        "spillover.design.invalid",
        "offer_counts",
    ): "supply matching group IDs, named policies and an offers column",
    (
        "spillover.design.duplicate",
        "roster",
    ): "assign each person to one group and each group to one stratum",
    ("spillover.design.duplicate", "offer_counts"): "give one count for each group and policy",
    ("spillover.design.duplicate", "comparisons"): "declare each pair once, in one direction",
    (
        "spillover.design.quota",
        "offer_counts",
    ): "give every group an integer count from zero through its size for every policy, leaving someone unoffered under each compared policy",
    (
        "spillover.design.quota",
        "groups_per_policy",
    ): "assign a positive integer number of groups to each of at least two policies in every stratum, accounting for every group",
    (
        "spillover.log.keys",
        "group_policy",
    ): "record one declared policy per roster group and no other groups",
    (
        "spillover.log.keys",
        "offered",
    ): "record one assignment per roster member and no other members",
    (
        "spillover.log.keys",
        "outcomes",
    ): "use only unique roster member IDs and record an outcome for everyone not offered",
    (
        "spillover.log.quota",
        "group_policy",
    ): "match the declared group count for each policy within each stratum",
    ("spillover.log.quota", "offered"): "match each group's offer count under its assigned policy",
    ("spillover.roster.changed", "roster"): "use the original design, roster and offer counts",
}


def refuse_input(code: str, field: str) -> Never:
    """Report the same bounded, identity-free context on both input paths."""
    if code == "spillover.design.required":
        _raise(code)
    requirement = _REQUIREMENTS.get((code, field))
    if requirement is None:
        requirement = (
            "use booleans or numeric zeros and ones"
            if code == "spillover.log.binary"
            else "supply the declared fields and types"
        )
    _raise(code, field=field, requirement=requirement)


def integer_scalar(value: object) -> TypeGuard[int | np.integer]:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, np.integer) and value.dtype.kind in "iu"
    )


def is_label(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def canonical_id(value: object) -> Identifier | None:
    if is_label(value):
        return str(value)
    if integer_scalar(value):
        return int(value)
    return None


def _label(value: object, field: str) -> str:
    if not is_label(value):
        _raise("spillover.design.invalid", field=field, requirement="supply a nonempty string")
    return str(value)


def _sequence[T](values: Sequence[T], field: str) -> tuple[T, ...]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or not values
    ):
        _raise("spillover.design.invalid", field=field, requirement="supply a nonempty sequence")
    return tuple(values)


def _comparison_values(
    policy: object, reference: object, effect: object, rate: object
) -> tuple[str, str, float, float | None]:
    policy_name = _label(policy, "policy")
    reference_name = _label(reference, "reference")
    if policy_name == reference_name:
        _raise(
            "spillover.design.invalid",
            field="comparisons",
            requirement="choose two distinct policies",
        )
    if (
        not isinstance(effect, Real)
        or isinstance(effect, (bool, np.bool_, np.timedelta64))
        or not (le(-1, effect) and le(effect, 1))
        or float(effect) == 0
    ):
        _raise("spillover.planning.meaningful_effect", meaningful_effect=effect)
    baseline = None
    if rate is not None:
        if (
            not isinstance(rate, Real)
            or isinstance(rate, (bool, np.bool_, np.timedelta64))
            or not (le(0, rate) and le(rate, 1))
        ):
            _raise("spillover.planning.baseline_rate", baseline_rate=rate)
        baseline = float(rate)
    return policy_name, reference_name, float(effect), baseline


@dataclass(frozen=True, slots=True)
class SpilloverComparison:
    """A signed change in untreated outcome probability: policy minus reference."""

    policy: str
    reference: str
    meaningful_effect: float
    baseline_rate: float | None = None

    def __post_init__(self) -> None:
        policy, reference, effect, rate = _comparison_values(
            self.policy, self.reference, self.meaningful_effect, self.baseline_rate
        )
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "meaningful_effect", effect)
        object.__setattr__(self, "baseline_rate", rate)


def _freeze_allocation(
    allocation: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(allocation, Mapping) or not allocation:
        refuse_input("spillover.design.quota", "groups_per_policy")
    frozen: dict[str, Mapping[str, int]] = {}
    policies: set[str] | None = None
    for stratum, counts in allocation.items():
        name = _label(stratum, "stratum")
        if not isinstance(counts, Mapping) or len(counts) < 2:
            refuse_input("spillover.design.quota", "groups_per_policy")
        row: dict[str, int] = {}
        for policy, count in counts.items():
            label = _label(policy, "policy")
            if not integer_scalar(count) or count < 1:
                refuse_input("spillover.design.quota", "groups_per_policy")
            row[label] = int(count)
        if policies is None:
            policies = set(row)
        elif policies != row.keys():
            refuse_input("spillover.design.quota", "groups_per_policy")
        frozen[name] = MappingProxyType(dict(sorted(row.items())))
    return MappingProxyType(dict(sorted(frozen.items())))


def _comparison_family(
    comparisons: Sequence[SpilloverComparison], policies: Mapping[str, int]
) -> tuple[SpilloverComparison, ...]:
    family = _sequence(comparisons, "comparisons")
    pairs: set[frozenset[str]] = set()
    for comparison in family:
        if not isinstance(comparison, SpilloverComparison):
            _raise(
                "spillover.design.invalid",
                field="comparisons",
                requirement="supply SpilloverComparison records",
            )
        if comparison.policy not in policies or comparison.reference not in policies:
            _raise(
                "spillover.design.invalid",
                field="comparisons",
                requirement="compare only declared policies",
            )
        pair = frozenset((comparison.policy, comparison.reference))
        if pair in pairs:
            refuse_input("spillover.design.duplicate", "comparisons")
        pairs.add(pair)
    return family


@dataclass(frozen=True, slots=True, init=False)
class Spillover:
    """Two individual assignments and the policies randomized across groups."""

    control_group: str
    offered_group: str
    groups_per_policy: Mapping[str, Mapping[str, int]]
    comparisons: tuple[SpilloverComparison, ...]

    def __init__(
        self,
        control_group: str,
        offered_group: str,
        groups_per_policy: Mapping[str, Mapping[str, int]],
        comparisons: Sequence[SpilloverComparison],
    ) -> None:
        control = _label(control_group, "control_group")
        offered = _label(offered_group, "offered_group")
        if control == offered:
            refuse_input("spillover.design.invalid", "offered_group")
        quotas = _freeze_allocation(groups_per_policy)
        family = _comparison_family(comparisons, next(iter(quotas.values())))
        object.__setattr__(self, "control_group", control)
        object.__setattr__(self, "offered_group", offered)
        object.__setattr__(self, "groups_per_policy", quotas)
        object.__setattr__(self, "comparisons", family)

    def law(self, metric_name: str, window_days: int, alpha: float) -> SpilloverDesign:
        """Bind the assignment law to the conversion window and family error rate."""
        if not integer_scalar(window_days) or window_days < 1:
            refuse_input("spillover.design.invalid", "window_days")
        horizon = f"{window_days} day" if window_days == 1 else f"{window_days} days"
        return SpilloverDesign(
            self.groups_per_policy, self.comparisons, metric_name, horizon, alpha=alpha
        )


@dataclass(frozen=True, slots=True, init=False)
class SpilloverDesign:
    """Predeclare policy counts, directed comparisons and one fixed-horizon outcome."""

    groups_per_policy: Mapping[str, Mapping[str, int]]
    comparisons: tuple[SpilloverComparison, ...]
    outcome: str
    horizon: str
    alpha: float
    policies: tuple[str, ...] = field(init=False)

    def __init__(
        self,
        groups_per_policy: Mapping[str, Mapping[str, int]],
        comparisons: Sequence[SpilloverComparison],
        outcome: str,
        horizon: str,
        *,
        alpha: float = 0.05,
    ) -> None:
        if (
            not isinstance(alpha, Real)
            or isinstance(alpha, (bool, np.bool_, np.timedelta64))
            or not 0 < alpha < 1
            or not 0 < float(alpha) < 1
        ):
            refuse_input("spillover.design.invalid", "alpha")
        quotas = _freeze_allocation(groups_per_policy)
        policies = next(iter(quotas.values()))
        family = _comparison_family(comparisons, policies)
        object.__setattr__(self, "groups_per_policy", quotas)
        object.__setattr__(self, "comparisons", family)
        object.__setattr__(self, "outcome", _label(outcome, "outcome"))
        object.__setattr__(self, "horizon", _label(horizon, "horizon"))
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "policies", tuple(policies))

    @property
    def compared_policies(self) -> frozenset[str]:
        return frozenset(p for c in self.comparisons for p in (c.policy, c.reference))

    def declaration(self) -> dict[str, object]:
        """Return an independent, JSON-ready declaration in canonical order."""
        return {
            "groups_per_policy": {s: dict(q) for s, q in self.groups_per_policy.items()},
            "comparisons": [
                {
                    "policy": c.policy,
                    "reference": c.reference,
                    "meaningful_effect": c.meaningful_effect,
                    "baseline_rate": c.baseline_rate,
                }
                for c in self.comparisons
            ],
            "outcome": self.outcome,
            "horizon": self.horizon,
            "alpha": self.alpha,
        }


@dataclass(frozen=True, slots=True)
class GroupProfile:
    """Eligible group size and prespecified offer counts in policy order."""

    group_id: Identifier
    stratum: str
    size: int
    offers: tuple[int, ...]

    def __post_init__(self) -> None:
        key = canonical_id(self.group_id)
        if key is None or not is_label(self.stratum):
            refuse_input("spillover.design.invalid", "roster")
        if not integer_scalar(self.size) or self.size <= 0:
            refuse_input("spillover.design.quota", "roster")
        if not isinstance(self.offers, Sequence) or isinstance(
            self.offers, (str, bytes, bytearray)
        ):
            refuse_input("spillover.design.quota", "offer_counts")
        if not self.offers or any(
            not integer_scalar(k) or not 0 <= k <= self.size for k in self.offers
        ):
            refuse_input("spillover.design.quota", "offer_counts")
        object.__setattr__(self, "group_id", key)
        object.__setattr__(self, "stratum", str(self.stratum))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "offers", tuple(int(k) for k in self.offers))


@dataclass(frozen=True, slots=True)
class EffectCapacity:
    """The neighboring floats enclosing the exact fraction of people whose quota changes."""

    inward: float
    outward: float


def _effect_capacity(
    groups: Sequence[GroupProfile], policy: int, reference: int, people: int
) -> EffectCapacity:
    # Only people exposed to a different offer count can change their untreated outcome.
    movable = sum(g.size for g in groups if g.offers[policy] != g.offers[reference])
    cap = movable / people
    numerator, denominator = cap.as_integer_ratio()
    rounded, exact = numerator * people, movable * denominator
    return EffectCapacity(
        math.nextafter(cap, 0.0) if rounded > exact else cap,
        math.nextafter(cap, 1.0) if rounded < exact else cap,
    )


@dataclass(frozen=True, slots=True)
class SpilloverPopulation:
    """Frozen group profiles and identity of the eligible people and assignment law."""

    design: SpilloverDesign
    groups: tuple[GroupProfile, ...]
    roster_digest: str
    effect_capacity: Mapping[tuple[str, str], EffectCapacity] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.design, SpilloverDesign):
            refuse_input("spillover.design.required", "design")
        groups = _sequence(self.groups, "groups")
        if not all(isinstance(g, GroupProfile) for g in groups):
            refuse_input("spillover.design.invalid", "groups")
        if len({type(g.group_id) for g in groups}) != 1:
            refuse_input("spillover.design.invalid", "roster")
        if len({g.group_id for g in groups}) != len(groups):
            refuse_input("spillover.design.duplicate", "roster")
        compared = self.design.compared_policies
        indices = tuple(i for i, p in enumerate(self.design.policies) if p in compared)
        if any(
            len(g.offers) != len(self.design.policies)
            or any(g.offers[i] == g.size for i in indices)
            for g in groups
        ):
            refuse_input("spillover.design.quota", "offer_counts")
        if Counter(g.stratum for g in groups) != {
            s: sum(q.values()) for s, q in self.design.groups_per_policy.items()
        }:
            refuse_input("spillover.design.quota", "groups_per_policy")
        ordered = tuple(sorted(groups, key=lambda g: (g.stratum, g.group_id)))
        object.__setattr__(self, "groups", ordered)
        object.__setattr__(self, "roster_digest", _label(self.roster_digest, "roster_digest"))
        people = sum(g.size for g in groups)
        policies = self.design.policies
        capacities = {
            (c.policy, c.reference): _effect_capacity(
                groups, policies.index(c.policy), policies.index(c.reference), people
            )
            for c in self.design.comparisons
        }
        object.__setattr__(self, "effect_capacity", MappingProxyType(capacities))


@dataclass(frozen=True, slots=True)
class PolicyCell:
    """Untreated counts and centered moments of group totals in a stratum and policy."""

    stratum: str
    policy: str
    groups: int
    mean_total: float
    ss: float
    unoffered: int
    events: int
