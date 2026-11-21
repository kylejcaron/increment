"""Frozen rosters and assignment laws for a two-stage spillover trial."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeGuard

import numpy as np
from numpy.typing import NDArray

from increment.errors import InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "spillover.design.invalid": "Invalid {field}: {requirement}.",
        "spillover.design.duplicate": "The roster repeats {field} {identifier!r}; each must be unique.",
        "spillover.design.quota": "Invalid {field}: {requirement}.",
        "spillover.log.keys": "Invalid {field} log: {requirement}.",
        "spillover.log.binary": "The {field} log entry for {identifier!r} must be a boolean or an integer 0 or 1.",
        "spillover.log.quota": (
            "The {field} log records {observed} for {identifier!r}, but the design requires {expected}."
        ),
    },
)
_raise = raiser(_REFUSALS)


def _label(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
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


def _unique(values: Iterable[str], field: str) -> None:
    seen: set[str] = set()
    for identifier in values:
        if identifier in seen:
            _raise("spillover.design.duplicate", field=field, identifier=identifier)
        seen.add(identifier)


def _integer_scalar(value: object) -> TypeGuard[int | np.integer]:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, np.integer) and value.dtype.kind in "iu"
    )


def _integer(value: object, field: str) -> int:
    if not _integer_scalar(value):
        _raise("spillover.design.quota", field=field, requirement="supply an integer count")
    return int(value)


@dataclass(frozen=True, slots=True, init=False)
class SpilloverGroup:
    """One group's eligible members and fixed treatment counts under the two policies."""

    group_id: str
    members: tuple[str, ...]
    low_treated: int
    high_treated: int
    stratum: str = "all"

    def __init__(
        self,
        group_id: str,
        members: Sequence[str],
        low_treated: int,
        high_treated: int,
        stratum: str = "all",
    ) -> None:
        group_id = _label(group_id, "group_id")
        stratum = _label(stratum, "stratum")
        member_ids = tuple(_label(person, "member ID") for person in _sequence(members, "members"))
        _unique(member_ids, "member ID")
        low = _integer(low_treated, "low_treated")
        high = _integer(high_treated, "high_treated")
        if not 0 < low < high < len(member_ids):
            _raise(
                "spillover.design.quota",
                field=f"treatment counts for group {group_id!r}",
                requirement="choose 0 < low_treated < high_treated < number of members",
            )
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "members", member_ids)
        object.__setattr__(self, "low_treated", low)
        object.__setattr__(self, "high_treated", high)


def _check_log_keys(record: Mapping[str, object], expected: Set[str], field: str) -> None:
    if not isinstance(record, Mapping):
        _raise("spillover.log.keys", field=field, requirement="supply a mapping keyed by ID")
    if record.keys() != expected:
        _raise(
            "spillover.log.keys", field=field, requirement="include every declared ID and no others"
        )


def _binary(value: object, field: str, identifier: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not _integer_scalar(value) or value not in (0, 1):
        _raise("spillover.log.binary", field=field, identifier=identifier)
    return bool(value)


def _check_quota(field: str, identifier: str, observed: int, expected: int) -> None:
    if observed != expected:
        _raise(
            "spillover.log.quota",
            field=field,
            identifier=identifier,
            observed=observed,
            expected=expected,
        )


def _untreated_successes(
    group: SpilloverGroup,
    quota: int,
    treated: Mapping[str, object],
    outcomes: Mapping[str, object],
) -> int:
    treated_count = successes = 0
    for person in group.members:
        assigned = _binary(treated[person], "treated", person)
        treated_count += assigned
        if person in outcomes:
            event = _binary(outcomes[person], "outcomes", person)
            successes += event and not assigned
        elif not assigned:
            _raise(
                "spillover.log.keys", field="outcomes", requirement="measure every untreated member"
            )
    _check_quota("treated", group.group_id, treated_count, quota)
    return successes


@dataclass(frozen=True, slots=True, init=False)
class SpilloverDesign:
    """Freeze independent policy draws across strata and offer draws across groups given policy."""

    groups: tuple[SpilloverGroup, ...]
    high_groups: Mapping[str, int]
    outcome: str
    horizon: str

    def __init__(
        self,
        groups: Sequence[SpilloverGroup],
        high_groups: Mapping[str, int],
        outcome: str,
        horizon: str,
    ) -> None:
        roster = _sequence(groups, "groups")
        if not all(isinstance(g, SpilloverGroup) for g in roster):
            _raise(
                "spillover.design.invalid",
                field="groups",
                requirement="supply SpilloverGroup records",
            )
        _unique((g.group_id for g in roster), "group ID")
        _unique((p for g in roster for p in g.members), "member ID")
        ordered = tuple(sorted(roster, key=lambda g: (g.stratum, g.group_id)))
        sizes = Counter(g.stratum for g in ordered)
        if not isinstance(high_groups, Mapping) or high_groups.keys() != sizes.keys():
            _raise(
                "spillover.design.quota",
                field="high_groups",
                requirement="specify every stratum and no others",
            )
        quotas = {s: _integer(high_groups[s], "high_groups") for s in sizes}
        for s, n in sizes.items():
            if not 0 < quotas[s] < n:
                _raise(
                    "spillover.design.quota",
                    field=f"high_groups for stratum {s!r}",
                    requirement="assign at least one group to each policy",
                )
        object.__setattr__(self, "groups", ordered)
        object.__setattr__(self, "high_groups", MappingProxyType(quotas))
        object.__setattr__(self, "outcome", _label(outcome, "outcome"))
        object.__setattr__(self, "horizon", _label(horizon, "horizon"))

    def logged_counts(
        self,
        group_high: Mapping[str, object],
        treated: Mapping[str, object],
        outcomes: Mapping[str, object],
    ) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
        """Reconcile both logs; return single-row policy and untreated-success arrays in design order."""
        _check_log_keys(group_high, {g.group_id for g in self.groups}, "group_high")
        high = np.empty((1, len(self.groups)), dtype=np.bool_)
        observed_high: Counter[str] = Counter()
        for j, g in enumerate(self.groups):
            policy = _binary(group_high[g.group_id], "group_high", g.group_id)
            high[0, j] = policy
            observed_high[g.stratum] += policy
        for stratum, quota in self.high_groups.items():
            _check_quota("group_high", stratum, observed_high[stratum], quota)

        _check_log_keys(treated, {p for g in self.groups for p in g.members}, "treated")
        if not isinstance(outcomes, Mapping) or not outcomes.keys() <= treated.keys():
            _raise(
                "spillover.log.keys", field="outcomes", requirement="use only declared member IDs"
            )
        successes = np.empty(high.shape, dtype=np.int64)
        for j, g in enumerate(self.groups):
            quota = g.high_treated if high[0, j] else g.low_treated
            successes[0, j] = _untreated_successes(g, quota, treated, outcomes)
        return high, successes
