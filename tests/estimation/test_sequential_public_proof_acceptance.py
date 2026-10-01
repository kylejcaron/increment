"""Bounded Bernoulli public mapping, not full sequential-campaign certification.

The independent finite Bernoulli oracle supplies expected likelihood decisions,
stopping times and discovery proportions; this file only checks that raw
records, appended snapshots, frozen checkpoints, public family selection and
selected reinversion preserve that bounded witness.  It makes no Gaussian,
manifest, or generalized-proof claim.
"""

from fractions import Fraction as F
from functools import cache
from itertools import product
from math import exp, prod
from typing import NamedTuple

import pytest

from increment import (
    AlwaysValid,
    JointReveal,
    PredictivePrior,
    SequentialCell,
    SequentialModel,
    SequentialRegistration,
    capture_sequential_snapshot,
    estimate_sequential,
)
from increment.estimation.decision_types import EValueEvidence, sequential_hypothesis_key
from increment.estimation.family import select_sequential_family
from increment.estimation.sequential_runtime import reinvert_selected
from increment.sequential_state import declare_sequential_freeze_cells
from tests.estimation._sequential_proof_oracle import (
    Arm,
    Path,
    adaptive_frozen_stop,
    e_bh,
    evidence,
    family_stop,
    reveal_support,
)

pytestmark = pytest.mark.slow

ALTERNATIVES = ("two-sided", "greater", "less")
Q = F(1, 3)
MIXED_Q = F(3, 5)
HORIZON = 4
# Fair independent reveals: the complete three-arm ordered path space. Every
# path-local claim below is checked over all of it, one shard per opening pair
# of reveals, which keeps each shard well inside the slow-test deadline.
PROBABILITIES = (F(1, 2),) * 3
OPENINGS = tuple(
    product([reveal for reveal, _ in reveal_support(PROBABILITIES)], repeat=HORIZON // 2)
)
# A mixed-null design whose non-null arm always succeeds, so the probability
# weighted FDR/FCR enumeration covers every positive-probability path inside a
# single process; the excluded reveals carry exactly zero mass.
MIXED_PROBABILITIES = (F(1, 2), F(1, 2), F(1))
NULL_GROUPS = frozenset({"first"})
# Exact ratio p_treatment / p_control of the mixed-null witness, per cell.
TRUTH = {"first": F(1), "second": F(2)}


def _registration(*, alternative, groups, family=False, q=Q, alpha=Q):
    prior = PredictivePrior(kind="beta", a=1, b=1)
    model = SequentialModel(
        metric="outcome",
        law="bernoulli",
        control_prior=prior,
        treatment_prior=prior,
        positive_population_control=True,
    )
    return SequentialRegistration(
        source_id="bounded-public-bernoulli",
        definitions_id="bounded-public-definition-v1",
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id="bounded-joint-units-v1",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=0,
        ),
        models=(model,),
        roster=tuple(
            SequentialCell(
                metric="outcome",
                group_id=group,
                alternative=alternative,
                alpha=alpha,
                family=family,
            )
            for group in groups
        ),
        q=q,
    )


def _records(groups, arms, *, offset=0):
    rows = []
    for i, values in enumerate(zip(*arms, strict=True)):
        for group, value in zip(groups, values, strict=True):
            rows.append(
                {
                    "unit_id": f"{offset + i:04d}-{group}",
                    "group_id": group,
                    "values": {"outcome": value},
                    "segments": {},
                }
            )
    return rows


def _capture(registration, rows, previous=None, *, append=False):
    return capture_sequential_snapshot(
        registration,
        rows,
        source_id=registration.source_id,
        definitions_id=registration.definitions_id,
        finalized=True,
        previous=previous,
        append=append,
    )


def _bits(n, successes):
    return [1] * successes + [0] * (n - successes)


class _Stopped(NamedTuple):
    """One stopped public node and the whole probability mass it represents."""

    look: int
    reveals: tuple[tuple[int, ...], ...]
    probability: F
    rows: tuple
    bundle: object
    frozen: tuple


def _family_rows(bundle):
    return tuple(
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in bundle.results
    )


def _replay_support(registration, policy, probabilities, *, schedule, freeze=None, opening=()):
    """Replay every positive-probability path one appended look at a time.

    Each look captures its single joint reveal against the previous verified
    snapshot with ``append=True``, so every enumerated path crosses the public
    continuation seam instead of being collapsed into terminal counts.
    ``schedule(look, rows, pending)`` reads only public rows and returns whether
    this look stops together with the state carried into the next look, which is
    what keeps a predictable rule predictable.  A stopped prefix is shared by all
    of its continuations, so each stopped node carries its whole subtree mass.
    ``opening`` fixes the first reveals, which shards the enumeration without
    changing any replay: every path-local claim holds shard by shard.
    """
    support = reveal_support(probabilities)
    weights = dict(support)
    observed = ("control", "first", "second")

    def walk(look, reveals, arms, mass, previous, pending):
        fixed = look < len(opening)
        for reveal, weight in ((opening[look], weights[opening[look]]),) if fixed else support:
            arrived = tuple(arm.append(bit) for arm, bit in zip(arms, reveal, strict=True))
            snapshot = _capture(
                registration,
                _records(observed, tuple((bit,) for bit in reveal), offset=look),
                previous,
                append=previous is not None,
            )
            if freeze is not None and look + 1 == HORIZON // 2:
                cell = next(c for c in registration.roster if c.group_id == freeze)
                snapshot = declare_sequential_freeze_cells(snapshot, [cell])
            bundle = estimate_sequential(snapshot, policy)
            rows = _family_rows(bundle)
            stop, carried = schedule(look + 1, rows, pending)
            if stop or look + 1 == HORIZON:
                yield _Stopped(
                    look + 1, (*reveals, reveal), mass * weight, rows, bundle, snapshot.frozen
                )
                continue
            yield from walk(look + 1, (*reveals, reveal), arrived, mass * weight, snapshot, carried)

    yield from walk(0, (), (Arm(),) * len(probabilities), F(1), None, None)


def _continuation(stopped, probabilities):
    """Any enumerated continuation of a stopped prefix witnesses the same stop.

    Both witnessed stopping rules are measurable at each look, so extending a
    stopped prefix with further support reveals can move the stop neither earlier
    nor later; padding only lets the oracle consume a full-horizon path.
    """
    filler = reveal_support(probabilities)[0][0]
    return Path(stopped.reveals + (filler,) * (HORIZON - stopped.look), F(1))


@cache
def _public_singleton(n, control_successes, treatment_successes, alternative):
    registration = _registration(alternative=alternative, groups=("treatment",))
    snapshot = _capture(
        registration,
        _records(
            ("control", "treatment"),
            (_bits(n, control_successes), _bits(n, treatment_successes)),
        ),
    )
    policy = AlwaysValid(registration=registration)
    return estimate_sequential(snapshot, policy).results[0]


@cache
def _public_family(n, control_successes, first_successes, second_successes, alternative):
    groups = ("first", "second", "never-enrolled")
    registration = _registration(alternative=alternative, groups=groups, family=True)
    snapshot = _capture(
        registration,
        _records(
            ("control", "first", "second"),
            (
                _bits(n, control_successes),
                _bits(n, first_successes),
                _bits(n, second_successes),
            ),
        ),
    )
    policy = AlwaysValid(registration=registration)
    bundle = estimate_sequential(snapshot, policy)
    cells = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in bundle.results
    ]
    return registration, policy, bundle, cells


def test_public_singleton_replay_is_conservative_against_exact_witness():
    """Every bounded pair sufficient state crosses the public raw-record seam."""
    for alternative in ALTERNATIVES:
        for n in range(HORIZON + 1):
            for control_successes in range(n + 1):
                for treatment_successes in range(n + 1):
                    exact = evidence(
                        Arm(n, control_successes),
                        Arm(n, treatment_successes),
                        alternative,
                    )
                    row = _public_singleton(n, control_successes, treatment_successes, alternative)
                    public_decision = row.stat_sig()
                    exact_decision = Q * exact >= 1
                    # A lower certificate may abstain, but must never reject
                    # where the exact witness does not; away from a boundary it
                    # must make the same decision.
                    assert not public_decision or exact_decision
                    if Q * exact != 1:
                        assert public_decision == exact_decision


def test_public_shared_control_family_retains_missing_cell_and_m3_selection():
    """Every bounded shared-control state uses the three-cell retained roster."""
    for alternative in ALTERNATIVES:
        for n in range(HORIZON + 1):
            for c in range(n + 1):
                for first in range(n + 1):
                    for second in range(n + 1):
                        registration, policy, bundle, cells = _public_family(
                            n, c, first, second, alternative
                        )
                        outcome = select_sequential_family(
                            cells, registration.q, policy, F(1, 20), computation=bundle
                        )
                        assert outcome.n_family == 3
                        missing = bundle.evidence[
                            sequential_hypothesis_key(
                                next(
                                    row.require_sequential_result().checkpoint.cell
                                    for row in bundle.results
                                    if row.group_id == "never-enrolled"
                                )
                            )
                        ]
                        assert isinstance(missing, EValueEvidence)
                        assert missing.log_e == float("-inf")

                        exact_values = (
                            evidence(Arm(n, c), Arm(n, first), alternative),
                            evidence(Arm(n, c), Arm(n, second), alternative),
                            F(0),
                        )
                        exact_indices = e_bh(exact_values, Q, m=3)
                        exact_groups = (
                            "first" if 0 in exact_indices else None,
                            "second" if 1 in exact_indices else None,
                        )
                        public_groups = frozenset(key.group_id for key in outcome.selected)
                        assert public_groups.issubset(
                            {group for group in exact_groups if group is not None}
                        )
                        ties = any(
                            Q * rank * value == 3
                            for rank, value in enumerate(sorted(exact_values, reverse=True), 1)
                        )
                        if not ties:
                            assert public_groups == {
                                group for group in exact_groups if group is not None
                            }


@pytest.mark.parametrize("opening", OPENINGS)
@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_public_adaptive_frozen_stop_and_family_match_the_oracle_on_every_path(
    alternative, opening
):
    """Production picks every stop; the oracle only says what it should have been.

    One shard per opening pair of reveals, so the shards together cover the
    complete four-look three-arm path space for each alternative, each path
    replayed one appended look at a time.
    """
    groups = ("first", "second", "never-enrolled")
    registration = _registration(alternative=alternative, groups=groups, family=True)
    policy = AlwaysValid(registration=registration)

    def schedule(look, rows, pending):
        # Only the even retained indices can schedule, exactly as the oracle does,
        # and "first" reports the frozen midpoint certificate once it is frozen.
        scheduling = [row for _, row in rows if row.group_id in ("first", "never-enrolled")]
        return any(row.require_sequential_result().log_e > 0 for row in scheduling), pending

    weights = dict(reveal_support(PROBABILITIES))
    shard_mass = prod((weights[reveal] for reveal in opening), start=F(1))
    frozen_arms = (
        Arm(HORIZON // 2, sum(reveal[0] for reveal in opening)),
        Arm(HORIZON // 2, sum(reveal[1] for reveal in opening)),
    )
    total = F(0)
    stop_mass = {}
    for stopped in _replay_support(
        registration, policy, PROBABILITIES, schedule=schedule, freeze="first", opening=opening
    ):
        expected = adaptive_frozen_stop(_continuation(stopped, PROBABILITIES), Q, alternative)
        assert stopped.look == expected.look
        total += stopped.probability
        stop_mass[stopped.look] = stop_mass.get(stopped.look, F(0)) + stopped.probability

        outcome = select_sequential_family(
            stopped.rows, registration.q, policy, F(1, 2), computation=stopped.bundle
        )
        assert outcome.n_family == 3
        public_groups = frozenset(
            row.group_id for key, row in stopped.rows if key in outcome.selected
        )
        # No reachable evidence sits on an m=3 e-BH boundary at this level, so
        # the conservative lower certificate must reproduce the exact selection.
        assert public_groups == frozenset(groups[index] for index in expected.selected)

        missing = next(row for _, row in stopped.rows if row.group_id == "never-enrolled")
        missing_evidence = stopped.bundle.evidence[
            sequential_hypothesis_key(missing.require_sequential_result().checkpoint.cell)
        ]
        assert isinstance(missing_evidence, EValueEvidence)
        assert missing_evidence.log_e == float("-inf")
        assert missing.require_sequential_result().checkpoint.status == "missing"

        retained = next(row for _, row in stopped.rows if row.group_id == "first")
        result = retained.require_exact_sequential_result()
        if stopped.look >= HORIZON // 2:
            (frozen,) = stopped.frozen
            assert result.checkpoint.status == "frozen"
            assert result.checkpoint.control.n == HORIZON // 2
            assert result.checkpoint == frozen
        else:
            assert result.checkpoint.status == "current"
            assert result.checkpoint.control.n == stopped.look

        if outcome.selected:
            alpha = outcome.fcr_alpha
            assert alpha is not None
            assert alpha == min(F(1, 2), Q * len(outcome.selected) / 3)
            for key in outcome.selected:
                row = next(row for row_key, row in stopped.rows if row_key == key)
                replayed = reinvert_selected(row, alpha, ceiling=alpha).require_sequential_result()
                assert replayed.checkpoint == row.require_sequential_result().checkpoint
                assert replayed.bounds.alpha == alpha
        else:
            assert outcome.fcr_alpha is None

    assert total == shard_mass
    # The frozen midpoint certificate is the only early stop available, and this
    # opening pair fixes it, so the whole shard stops at one look.
    early = evidence(*frozen_arms, alternative) > 1
    assert set(stop_mass) == ({HORIZON // 2} if early else {HORIZON})


class _Intervals(NamedTuple):
    """Selected-set outcome at one stopped node, with bound status kept apart."""

    misses: int
    status: dict
    informative: frozenset
    covered: frozenset


def _selected_intervals(rows, outcome, alpha, look):
    """Reinvert every selected cell and classify its confidence set explicitly.

    A full-domain, unresolved, abstained or empty set is not a coverage claim:
    only a closed ``interval`` with a finite directional endpoint strictly
    inside the nonnegative ratio domain reports an informative bound.
    """
    misses = 0
    status = {}
    informative = set()
    covered = set()
    for key in outcome.selected:
        row = next(row for row_key, row in rows if row_key == key)
        sequential = row.require_sequential_result()
        assert sequential.checkpoint.control.n == look
        replayed = reinvert_selected(row, alpha, ceiling=alpha).require_sequential_result()
        assert replayed.checkpoint == sequential.checkpoint
        assert replayed.bounds.alpha == alpha
        bounds = replayed.bounds
        status[row.group_id] = bounds.status
        if bounds.status == "interval" and bounds.lower is not None and bounds.lower > 0:
            informative.add(row.group_id)
        truth = TRUTH[row.group_id]
        missed = (
            bounds.empty
            or (bounds.lower is not None and truth < bounds.lower)
            or (bounds.upper is not None and truth > bounds.upper)
        )
        misses += missed
        if not missed:
            covered.add(row.group_id)
    return _Intervals(misses, status, frozenset(informative), frozenset(covered))


@pytest.mark.parametrize(
    ("path", "expected_look"),
    (
        (Path(((0, 0, 1),) * HORIZON, F(1)), 3),
        (Path(((0, 0, 0),) * HORIZON, F(1)), HORIZON),
    ),
)
def test_mixed_family_first_discovery_uses_public_continuation(path, expected_look):
    registration = _registration(
        alternative="greater",
        groups=("first", "second"),
        family=True,
        q=MIXED_Q,
        alpha=F(1, 2),
    )
    policy = AlwaysValid(registration=registration)
    previous = None
    for look, reveal in enumerate(path.rows, 1):
        snapshot = _capture(
            registration,
            _records(
                ("control", "first", "second"),
                tuple((value,) for value in reveal),
                offset=look - 1,
            ),
            previous,
            append=previous is not None,
        )
        bundle = estimate_sequential(snapshot, policy)
        rows = [
            (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
            for row in bundle.results
        ]
        outcome = select_sequential_family(
            rows, registration.q, policy, F(1, 2), computation=bundle
        )
        if outcome.selected or look == HORIZON:
            break
        previous = snapshot

    stopped_arms = tuple(path.prefixes())[look]
    exact_values = tuple(
        evidence(stopped_arms[0], treatment, "greater") for treatment in stopped_arms[1:]
    )
    exact_groups = frozenset(
        ("first", "second")[index] for index in e_bh(exact_values, MIXED_Q, m=2)
    )
    public_groups = frozenset(key.group_id for key in outcome.selected)
    assert look == expected_look
    assert public_groups == exact_groups
    assert all(row.require_sequential_result().checkpoint.control.n == look for _, row in rows)


def test_public_mixed_family_predictable_stop_bounds_stopped_fdr_and_fcr():
    """Stopped mixed-null FDP and selected-set noncoverage cross the public route."""
    groups = ("first", "second")
    registration = _registration(
        alternative="greater", groups=groups, family=True, q=Q, alpha=F(1, 2)
    )
    policy = AlwaysValid(registration=registration)
    # The empty prefix has evidence one in both cells, so this level cannot
    # schedule a stop before any unit is revealed.
    assert Q * 2 < 1

    def schedule(look, rows, pending):
        summed = sum(exp(float(row.require_sequential_result().log_e)) for _, row in rows)
        # Predictable: this look stops on the decision taken one reveal earlier.
        return bool(pending), float(Q) * summed >= 1

    total = fdr = fcr = discovery_mass = early_stop_mass = F(0)
    informative_mass = dict.fromkeys(groups, F(0))
    covered_mass = dict.fromkeys(groups, F(0))
    status_mass: dict[str, F] = {}
    fdp_mass: dict[F, F] = {}
    stop_mass: dict[int, F] = {}

    for stopped in _replay_support(registration, policy, MIXED_PROBABILITIES, schedule=schedule):
        expected = family_stop(_continuation(stopped, MIXED_PROBABILITIES), Q, "greater")
        assert stopped.look == expected.look
        total += stopped.probability
        stop_mass[stopped.look] = stop_mass.get(stopped.look, F(0)) + stopped.probability

        outcome = select_sequential_family(
            stopped.rows, registration.q, policy, F(1, 2), computation=stopped.bundle
        )
        assert outcome.n_family == 2
        selected_groups = frozenset(
            row.group_id for key, row in stopped.rows if key in outcome.selected
        )
        assert selected_groups == frozenset(groups[index] for index in expected.selected)
        if not outcome.selected:
            assert outcome.fcr_alpha is None
            continue

        assert outcome.fcr_alpha == min(F(1, 2), Q * len(outcome.selected) / 2)
        discovery_mass += stopped.probability
        if stopped.look < HORIZON:
            early_stop_mass += stopped.probability
        fdp = F(len(selected_groups & NULL_GROUPS), len(outcome.selected))
        assert fdp == expected.false_discovery_proportion((0,))
        fdp_mass[fdp] = fdp_mass.get(fdp, F(0)) + stopped.probability
        fdr += stopped.probability * fdp

        intervals = _selected_intervals(stopped.rows, outcome, outcome.fcr_alpha, stopped.look)
        share = F(1, len(outcome.selected))
        for status in intervals.status.values():
            status_mass[status] = status_mass.get(status, F(0)) + stopped.probability * share
        for group in intervals.informative:
            informative_mass[group] += stopped.probability
        for group in intervals.covered:
            covered_mass[group] += stopped.probability
        fcr += stopped.probability * F(intervals.misses, len(outcome.selected))

    assert total == 1
    assert set(stop_mass) == {HORIZON - 1, HORIZON}
    assert all(mass > 0 for mass in stop_mass.values())
    assert discovery_mass > 0
    assert early_stop_mass > 0
    # The non-null cell alone and both cells together are reachable stopped
    # discoveries, so FDP is finer than an all-null 1{any rejection}.
    assert set(fdp_mass) == {F(0), F(1, 2)}
    assert all(mass > 0 for mass in fdp_mass.values())
    # Availability is a positive probability mass of informative sets, not a flag,
    # and no selected set here was vacuous, unresolved, abstained or empty.
    assert all(mass > 0 for mass in informative_mass.values())
    assert status_mass == {"interval": discovery_mass}
    # The null cell's own interval excludes its true unit ratio once selected.
    assert covered_mass["second"] > 0
    assert covered_mass["first"] == 0
    assert 0 < fdr <= Q
    assert 0 < fcr <= Q
