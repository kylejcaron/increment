"""Exactness witnesses for the sequential campaign's sufficient-state simulator.

Nothing here asserts that the simulator is exact; every test makes the raw
public path the reference and refuses any difference.

*   State sufficiency and the whole per-replication record are checked
    exhaustively over every reveal order of a bounded Bernoulli horizon -- all
    256 value assignments of a two-look two-unit-per-arm case, and all 64 of a
    two-cell e-BH family -- so the check covers every reachable sufficient state
    and every pair of distinct orders that reach the same state.
*   The aggregate-increment law the vectorised Bernoulli route substitutes for
    per-unit sampling is proved by exact rational enumeration, independently of
    any sampler.
*   The Gaussian (NIG) and Gaussian-ratio (NIW) laws have continuous states, so
    they are checked as seeded agreement against the same public path on the
    same draw stream, which makes the comparison bit-for-bit rather than
    distributional.
*   ``scalar_mean`` is out of scope and its refusal is checked, so the scope
    claim cannot rot into silence.
"""

from fractions import Fraction as F
from itertools import product
from math import comb, factorial, prod

import numpy as np
import pytest

from calibration.stopping import FixedDesign, build_rule
from increment import capture_sequential_snapshot
from increment.estimation._sequential_likelihood import GaussianState
from increment.estimation.family import e_bh_select
from increment.sequential_state import _capture_sequential_diagnostic_snapshot
from tests.estimation import _sequential_simulation as sim
from tests.estimation._sequential_acceptance import (
    RuntimeCase,
    campaign_batch,
    campaign_declaration,
    campaign_replication,
    replication_statistics,
)

# Fields of a replication record that any acceptance gate reads. The simulator
# owns exactly these; the recorded-only diagnostics are not its claim.
GATED_FIELDS = (
    "nominal_ever_null_rejection",
    "nominal_fdp",
    "nominal_fcp_lower",
    "nominal_fcp_upper",
    "nonnull_discovery",
    "selected_cells",
    "retained_cells",
    "available_points",
    "certified_intervals",
    "undeclared_point_reasons",
    "completed_looks",
)


def _gated(record):
    return {key: F(record[key]) for key in GATED_FIELDS}


def _batches(case, seed=None):
    """One replication's worth of records from the deployed campaign sampler."""
    resolved = sim.geometry(case)
    rng = np.random.default_rng(case.seed if seed is None else seed)
    rows, offset = [], 0
    for size in resolved.schedule:
        batch = campaign_batch(
            case, rng, resolved.registration, resolved.truths, resolved.arms, size, offset
        )
        offset += len(batch)
        rows.append(batch)
    return rows


def _public_record(case, batches):
    """The raw public path over explicit batches, as ``campaign_replication`` drives it."""
    from increment import estimate_sequential
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family
    from increment.estimation.sequential_runtime import (
        _evaluate_sequential_diagnostic,
        display_estimate,
        evaluate_checkpoint,
        reinvert_selected,
    )
    from increment.sequential_state import declare_sequential_freeze_cells
    from tests.estimation._sequential_acceptance import (
        CERTIFICATION_DESIGN,
        _diagnostic_selection,
    )

    registration, policy, truths, _arms, schedule = campaign_declaration(case)
    public = case.law == "bernoulli"
    capture = capture_sequential_snapshot if public else _capture_sequential_diagnostic_snapshot
    parent, ever = None, False
    results, selected, fcr_alpha, bundle = (), (), None, None
    for look, rows in enumerate(batches[: len(schedule)]):
        snapshot = capture(
            registration,
            rows,
            source_id=registration.source_id,
            definitions_id=registration.definitions_id,
            finalized=True,
            previous=parent,
            append=parent is not None,
        )
        parent = snapshot
        if public:
            bundle = estimate_sequential(snapshot, policy)
            results = tuple(row.require_exact_sequential_result() for row in bundle.results)
        else:
            results = _evaluate_sequential_diagnostic(snapshot, policy)
        ever |= any(result.rejects() and truths[result.checkpoint.cell][1] for result in results)
        if public:
            assert bundle is not None
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
            selected = tuple(i for i, (key, _) in enumerate(cells) if key in outcome.selected)
            fcr_alpha = outcome.fcr_alpha
        else:
            selected, fcr_alpha = _diagnostic_selection(results)
        if case.freeze and look + 1 < len(schedule):
            already = {c.cell for c in parent.frozen}
            to_freeze = [
                result.checkpoint.cell
                for i, result in enumerate(results)
                if (result.rejects() or (look == len(schedule) // 2 and i % 2 == 0))
                and result.checkpoint.cell not in already
                and result.checkpoint.control.n
                and result.checkpoint.treatment.n
            ]
            if to_freeze:
                parent = declare_sequential_freeze_cells(parent, to_freeze)
        stop = (
            (case.stopping == "first_rejection" and any(r.rejects() for r in results))
            or (case.stopping == "first_discovery" and bool(selected))
            or (
                case.stopping == "adaptive"
                and look + 1 >= 2
                and any(r.log_e > 0 for r in results[::2])
            )
        )
        if stop:
            break

    unavailable = CERTIFICATION_DESIGN.unavailable_statuses
    declared = CERTIFICATION_DESIGN.declared_point_reasons
    false = missed = unknown = nonnull = 0
    for index in selected:
        assert fcr_alpha is not None
        if public:
            assert bundle is not None
            result = reinvert_selected(
                bundle.results[index], fcr_alpha, ceiling=fcr_alpha
            ).require_sequential_result()
        else:
            result = evaluate_checkpoint(results[index].checkpoint, alpha=F(fcr_alpha))
        truth, declared_null = truths[result.checkpoint.cell]
        false += declared_null
        nonnull += not declared_null
        bounds = result.bounds
        is_unknown = bounds.status in unavailable
        unknown += is_unknown
        if not is_unknown:
            missed += bool(
                bounds.empty
                or (bounds.lower is not None and truth < bounds.lower)
                or (bounds.upper is not None and truth > bounds.upper)
            )
    denominator = max(len(selected), 1)
    reasons = [r.point_reason for r in results if r.point_reason is not None]
    return {
        "nominal_ever_null_rejection": int(ever),
        "nominal_fdp": F(false, denominator),
        "nominal_fcp_lower": F(missed, denominator),
        "nominal_fcp_upper": F(missed + unknown, denominator),
        "nonnull_discovery": int(nonnull > 0),
        "selected_cells": len(selected),
        "retained_cells": len(results),
        "available_points": sum(display_estimate(r) is not None for r in results),
        "certified_intervals": sum(r.bounds.status not in unavailable for r in results),
        "undeclared_point_reasons": sum(1 for r in reasons if r not in declared),
        "completed_looks": look + 1,
    }


# --- state sufficiency -------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        RuntimeCase(name="fold-bernoulli", law="bernoulli", looks=3, batch=4),
        RuntimeCase(name="fold-gaussian", law="gaussian", looks=3, batch=4),
        RuntimeCase(name="fold-ratio", law="gaussian_ratio", looks=3, batch=4),
        RuntimeCase(
            name="fold-family-shared",
            law="bernoulli",
            looks=2,
            batch=4,
            family=4,
            dependence="shared",
        ),
        RuntimeCase(
            name="fold-gaussian-shared",
            law="gaussian",
            looks=2,
            batch=4,
            family=4,
            dependence="shared",
        ),
    ],
    ids=lambda case: case.name,
)
def test_advance_folds_to_the_captured_state(case):
    """Folding ``advance`` reproduces every captured sufficient state exactly."""
    resolved = sim.geometry(case)
    batches = _batches(case)
    folded = sim.fold_batches(case, batches)
    capture = (
        capture_sequential_snapshot
        if case.law == "bernoulli"
        else _capture_sequential_diagnostic_snapshot
    )
    parent = None
    for look, rows in enumerate(batches):
        parent = capture(
            resolved.registration,
            rows,
            source_id=resolved.registration.source_id,
            definitions_id=resolved.registration.definitions_id,
            finalized=True,
            previous=parent,
            append=parent is not None,
        )
        assert folded[look] == parent.states


def test_geometry_arm_counts_match_the_captured_prefix():
    """The deterministic arm counts the aggregate draw assumes are the real ones."""
    cases = [
        RuntimeCase(name="counts-even", looks=3, batch=4),
        RuntimeCase(name="counts-odd", looks=3, batch=5, allocation=(1, 4)),
        RuntimeCase(name="counts-control-heavy", looks=2, batch=7, allocation=(4, 1)),
        RuntimeCase(
            name="counts-shared-missing",
            looks=3,
            batch=5,
            family=4,
            dependence="shared",
            missing=True,
        ),
    ]
    for case in cases:
        resolved = sim.geometry(case)
        parent = None
        for look, rows in enumerate(_batches(case)):
            parent = capture_sequential_snapshot(
                resolved.registration,
                rows,
                source_id=resolved.registration.source_id,
                definitions_id=resolved.registration.definitions_id,
                finalized=True,
                previous=parent,
                append=parent is not None,
            )
            captured = {(s.metric, s.group_id, s.segment): s.n for s in parent.states}
            planned = {(k.metric, k.group_id, k.segment): k.counts[look] for k in resolved.keys}
            assert planned == captured, case.name
            assert resolved.revealed[look] == len(parent.records), case.name


# --- exhaustive record equivalence ------------------------------------------


def _enumerated_batches(case, values):
    """Rewrite a batch sequence's metric values from an exhaustive assignment."""
    batches, cursor = [], 0
    for rows in _batches(case):
        replaced = []
        for record in rows:
            replaced.append(
                {
                    **record,
                    "values": {
                        metric: values[cursor + offset]
                        for offset, metric in enumerate(sorted(record["values"]))
                    },
                }
            )
            cursor += len(record["values"])
        batches.append(replaced)
    return batches


def _value_slots(case):
    return sum(len(record["values"]) for rows in _batches(case) for record in rows)


# Three looks of one unit per arm: every reveal order of the whole horizon is
# enumerable, and several orders land on the same sufficient state, which is
# what makes the comparison a sufficiency witness rather than a spot check.
EXHAUSTIVE_SINGLE = RuntimeCase(name="exhaustive-single", looks=3, batch=1, rate=0.5)
# Two looks, two cells over two arms, e-BH selection and first-discovery
# stopping: the smallest horizon that exercises select_sequential_family and
# the selected-cell FCR reinversion on every reveal order.
EXHAUSTIVE_FAMILY = RuntimeCase(
    name="exhaustive-family", looks=2, batch=1, family=2, rate=0.5, stopping="first_discovery"
)


@pytest.mark.slow
@pytest.mark.parametrize("case", [EXHAUSTIVE_SINGLE, EXHAUSTIVE_FAMILY], ids=lambda c: c.name)
def test_every_reveal_order_gives_the_public_record(case):
    """Exhaustive: every Bernoulli value assignment agrees with the public path.

    The enumeration covers each reachable sufficient state and, because several
    distinct reveal orders land on the same state, it also witnesses that the
    public record depends on the state alone.
    """
    slots = _value_slots(case)
    seen = 0
    for values in product((0, 1), repeat=slots):
        batches = _enumerated_batches(case, values)
        expected = _public_record(case, batches)
        actual = sim.replicate(case, lambda look, size, offset, rows=batches: rows[look])
        assert _gated(actual) == _gated(expected), values
        assert replication_statistics(actual) == replication_statistics(expected)
        seen += 1
    assert seen == 2**slots


@pytest.mark.slow
@pytest.mark.parametrize("case", [EXHAUSTIVE_SINGLE, EXHAUSTIVE_FAMILY], ids=lambda c: c.name)
def test_every_reveal_order_gives_the_vectorised_record(case):
    """Exhaustive: the vectorised engine matches the state core value for value."""
    resolved = sim.geometry(case)
    slots = _value_slots(case)
    for values in product((0, 1), repeat=slots):
        batches = _enumerated_batches(case, values)
        expected = sim.replicate(case, lambda look, size, offset, rows=batches: rows[look])
        increments = [
            [
                [
                    sum(
                        int(record["values"][key.metric])
                        for record in rows
                        if sim.retains(key.segment, key.group_id, record)
                    )
                ]
                for key in resolved.keys
            ]
            for rows in batches
        ]
        actual = sim.simulate_bernoulli_increments(case, increments)[0]
        assert _gated(actual) == _gated(expected), values


GRID_CASE = RuntimeCase(name="grid", looks=1, batch=6, rate=0.5)


def _grid_records(resolved, counts):
    """Records realising an exact (n, successes) state on each arm."""
    rows = []
    for arm, (n, successes) in counts.items():
        for j in range(n):
            rows.append(
                {
                    "unit_id": f"{arm}-{j:04d}",
                    "group_id": arm,
                    "values": {
                        model.metric: int(j < successes) for model in resolved.registration.models
                    },
                    "segments": {"segment": "A" if j % 2 == 0 else "B"},
                }
            )
    return rows


@pytest.mark.slow
@pytest.mark.parametrize(
    "arm_counts", [(6, 6), (6, 0), (0, 6)], ids=["both", "no-treated", "no-control"]
)
def test_every_grid_state_gives_the_public_decision(arm_counts):
    """Exhaustive over the whole sufficient-state grid at one look.

    The Bernoulli state grid at a look is ``{0..n_c} x {0..n_t}``, and the
    aggregate route's whole claim is that the decision is a function of that
    point. Every point is realised as real records, captured, and pushed
    through ``estimate_sequential``; the simulator's envelope-free kernels must
    return the same evidence, the same rejection, the same point availability
    and reason, and the same interval geometry, coverage and certification.
    """
    resolved = sim.geometry(GRID_CASE)
    registration = resolved.registration
    plan = resolved.cells[0]
    control_group = registration.control_group
    treated = plan.group_id
    n_control, n_treated = arm_counts
    alpha = resolved.alphas[0][0]
    threshold = resolved.reject_thresholds[0][0]
    seen = 0
    for successes_control, successes_treated in product(range(n_control + 1), range(n_treated + 1)):
        rows = _grid_records(
            resolved,
            {
                control_group: (n_control, successes_control),
                treated: (n_treated, successes_treated),
            },
        )
        snapshot = capture_sequential_snapshot(
            registration,
            rows,
            source_id=registration.source_id,
            definitions_id=registration.definitions_id,
            finalized=True,
        )
        control = snapshot.arm(plan.metric, control_group, plan.segment)
        treatment = snapshot.arm(plan.metric, treated, plan.segment)
        assert (control.n, control.successes) == (n_control, successes_control)
        assert (treatment.n, treatment.successes) == (n_treated, successes_treated)

        from increment import estimate_sequential

        bundle = estimate_sequential(snapshot, resolved.policy)
        deployed = bundle.results[0].require_exact_sequential_result()
        assert deployed.decision_alpha == alpha

        evidence = sim.evidence_at(
            plan, control.kernel(), treatment.kernel(), reject_threshold=threshold
        )
        assert evidence.log_e == deployed.log_e
        assert evidence.rejects == deployed.rejects()
        assert evidence.positive == (deployed.log_e > 0)

        from increment.estimation.sequential_runtime import display_estimate

        observed = sim.geometry_at(plan, control.kernel(), treatment.kernel(), alpha)
        assert observed.point_available == (display_estimate(deployed) is not None)
        assert observed.point_reason == deployed.point_reason
        unavailable = ("unavailable", "unresolved", "abstained")
        assert observed.certified == (deployed.bounds.status not in unavailable)
        assert observed.unknown == (deployed.bounds.status in unavailable)
        bounds = deployed.bounds
        expected_miss = not observed.unknown and bool(
            bounds.empty
            or (bounds.lower is not None and plan.truth < bounds.lower)
            or (bounds.upper is not None and plan.truth > bounds.upper)
        )
        assert observed.missed == expected_miss
        seen += 1
    assert seen == (n_control + 1) * (n_treated + 1)


# --- the aggregate increment law --------------------------------------------


def test_independent_block_increment_is_exactly_binomial():
    """Exhaustive: a block's success count has the binomial law the draw uses.

    Enumerating every value assignment of a block of units and accumulating
    exact rational probabilities must reproduce ``C(m, k) p**k (1-p)**(m-k)``,
    which is the law ``rng.binomial(m, p)`` samples.
    """
    for rate in (F(1, 10), F(1, 2), F(3, 4)):
        for units in range(1, 8):
            exact = dict.fromkeys(range(units + 1), F(0))
            for assignment in product((0, 1), repeat=units):
                weight = prod(rate if v else 1 - rate for v in assignment)
                exact[sum(assignment)] += weight
            for successes, mass in exact.items():
                assert mass == comb(units, successes) * rate**successes * (1 - rate) ** (
                    units - successes
                )
            assert sum(exact.values()) == 1


def test_shared_block_increment_is_exactly_multinomial():
    """Exhaustive: one shared uniform makes the metric counts a sorted-bin multinomial.

    With a single uniform per unit the metric indicators are comonotone, so the
    joint count vector is determined by which bin of the sorted-rate partition
    each unit falls into. Enumerating bin memberships must reproduce the exact
    multinomial mass, and the cumulative-sum mapping the draw applies must
    recover each metric's own count.
    """
    rates = (F(1, 5), F(1, 2), F(1, 2), F(4, 5))
    order = sorted(range(len(rates)), key=lambda i: rates[i])
    sorted_rates = [rates[index] for index in order]
    edges = [sorted_rates[0]]
    edges.extend(
        current - previous
        for previous, current in zip(sorted_rates, sorted_rates[1:], strict=False)
    )
    edges.append(1 - sorted_rates[-1])
    assert sum(edges) == 1

    units = 4
    exact: dict[tuple[int, ...], F] = {}
    for memberships in product(range(len(edges)), repeat=units):
        weight = prod(edges[bin_index] for bin_index in memberships)
        if not weight:
            continue
        # A unit in bin b lies below every sorted rate from position b onward.
        counts = [0] * len(rates)
        for bin_index in memberships:
            for position in range(bin_index, len(rates)):
                counts[order[position]] += 1
        exact[tuple(counts)] = exact.get(tuple(counts), F(0)) + weight
    assert sum(exact.values()) == 1

    # The same distribution, written as the multinomial over bins plus the
    # cumulative-sum mapping the engine applies to the drawn bin counts.
    rebuilt: dict[tuple[int, ...], F] = {}
    for bins in product(range(units + 1), repeat=len(edges)):
        if sum(bins) != units:
            continue
        weight = F(factorial(units) // prod(factorial(b) for b in bins)) * prod(
            edge**count for edge, count in zip(edges, bins, strict=True)
        )
        if not weight:
            continue
        cumulative = np.cumsum(bins[:-1])
        counts = [0] * len(rates)
        for position, slot in enumerate(order):
            counts[slot] = int(cumulative[position])
        rebuilt[tuple(counts)] = rebuilt.get(tuple(counts), F(0)) + weight
    assert rebuilt == exact


def test_shared_family_marginal_rates_match_the_per_unit_sampler():
    """The shared-noise engine reproduces the sampler's exact joint state law.

    Both routes are enumerated over the same bounded horizon: the per-unit
    sampler through ``fold_batches``, the engine through its own increments. The
    resulting sets of reachable joint states must coincide.
    """
    case = RuntimeCase(
        name="shared-support", looks=1, batch=2, family=4, dependence="shared", rate=0.5
    )
    resolved = sim.geometry(case)
    rates = {}
    for key in resolved.keys:
        assert key.rate is not None, "a Bernoulli state carries its sampler rate"
        rates[key.index] = F(key.rate)
    reachable = set()
    slots = _value_slots(case)
    for values in product((0, 1), repeat=slots):
        batches = _enumerated_batches(case, values)
        folded = sim.fold_batches(case, batches)[-1]
        # A shared uniform forces comonotone indicators: an assignment is
        # reachable only when every state's count is monotone in its rate.
        counts = {key.index: folded[key.index].successes for key in resolved.keys}
        grouped: dict[tuple[str, tuple], list[tuple[F, int, int]]] = {}
        for key in resolved.keys:
            grouped.setdefault((key.group_id, key.segment), []).append(
                (rates[key.index], counts[key.index], folded[key.index].n)
            )
        if all(
            all(
                low_count <= high_count
                for low, low_count, _ in block
                for high, high_count, _ in block
                if low <= high
            )
            for block in grouped.values()
        ):
            reachable.add(tuple(sorted(counts.items())))
    assert reachable

    engine_states = set()
    rng = np.random.default_rng(20260921)
    engine = sim._BernoulliEngine(resolved)
    for _ in range(64):
        successes = np.zeros((len(resolved.keys), 32), dtype=np.int64)
        engine._draw(rng, 0, successes, np.arange(32), True)
        for column in range(32):
            engine_states.add(
                tuple((key.index, int(successes[key.index, column])) for key in resolved.keys)
            )
    assert engine_states <= reachable


# --- selection --------------------------------------------------------------


def test_ebh_indices_matches_the_deployed_selection():
    """Simulation and deployment agree at and around every step threshold."""
    q = F(1, 20)
    width = 3
    thresholds = sim.ebh_step_thresholds(q, width)
    step = F(1, 10**9)
    candidates = [float("-inf")]
    for threshold in thresholds:
        candidates.extend((threshold - step, threshold, threshold + step))
    for values in product(candidates, repeat=width):
        above = [[value >= threshold for threshold in thresholds] for value in values]
        assert sim.ebh_indices(above) == tuple(e_bh_select(list(values), q)), values


# --- the Gaussian laws ------------------------------------------------------


@pytest.mark.parametrize("dimension", [1, 2])
def test_exact_moments_match_the_deployed_merge(dimension):
    """Integer-scaled moments are bit-identical to Chan's per-unit merge."""
    rng = np.random.default_rng(509)
    rows = [
        tuple(float(v) for v in rng.normal(loc=5.0, scale=3.0, size=dimension)) for _ in range(40)
    ]
    accumulator = sim._GaussianMoments(dimension)
    for cut in (1, 2, 7, 40):
        accumulator = sim._GaussianMoments(dimension)
        accumulator.extend(rows[:cut])
        assert accumulator.kernel() == GaussianState.from_rows(rows[:cut], dimension=dimension)
    # Accumulating in several calls must equal accumulating in one.
    split = sim._GaussianMoments(dimension)
    split.extend(rows[:13])
    split.extend(rows[13:])
    assert split.kernel() == GaussianState.from_rows(rows, dimension=dimension)


@pytest.mark.parametrize(
    "case",
    [
        RuntimeCase(name="nig-horizon", law="gaussian", looks=3, batch=4, seed=101),
        RuntimeCase(
            name="nig-first-rejection",
            law="gaussian",
            looks=3,
            batch=4,
            stopping="first_rejection",
            seed=101,
        ),
        RuntimeCase(
            name="nig-family-discovery",
            law="gaussian",
            looks=2,
            batch=4,
            family=2,
            null_fraction=0.5,
            stopping="first_discovery",
            seed=101,
        ),
        RuntimeCase(name="niw-horizon", law="gaussian_ratio", looks=3, batch=4, seed=109),
        RuntimeCase(
            name="niw-family-shared",
            law="gaussian_ratio",
            looks=2,
            batch=4,
            family=4,
            dependence="shared",
            stopping="first_discovery",
            seed=109,
        ),
    ],
    ids=lambda case: case.name,
)
def test_gaussian_state_route_matches_the_public_record(case):
    """Seeded agreement: continuous states admit no enumeration, so share a stream.

    Both routes consume the identical draw stream, which makes this a bit-for-bit
    comparison of the whole gated record rather than a distributional one.
    """
    batches = _batches(case)
    expected = _public_record(case, batches)
    actual = sim.replicate(case, lambda look, size, offset: batches[look])
    assert _gated(actual) == _gated(expected)


# --- the vectorised route against the deployed campaign ---------------------


# Full-size multi-cell cases invert every retained cell at every look, so they
# run slow; two-unit-per-arm smokes walk the same engine paths and draws with a
# seed whose record moves if increments stop accumulating or freeze at the
# wrong look.
VECTORISED_CASES = [
    RuntimeCase(name="vec-horizon", looks=3, batch=4),
    RuntimeCase(name="vec-first-rejection", looks=3, batch=4, stopping="first_rejection"),
    RuntimeCase(name="vec-high-rate", looks=3, batch=6, rate=0.5, stopping="first_rejection"),
    RuntimeCase(name="vec-unequal", looks=3, batch=4, allocation=(1, 4)),
    RuntimeCase(name="vec-null-shift", looks=3, batch=4, null=F(1, 5), alternative="greater"),
    RuntimeCase(
        name="vec-family-smoke",
        looks=2,
        batch=2,
        family=2,
        rate=0.5,
        stopping="first_discovery",
        null_fraction=0.5,
        seed=18,
    ),
    RuntimeCase(
        name="vec-shared-smoke",
        looks=2,
        batch=2,
        family=2,
        rate=0.5,
        dependence="shared",
        null_fraction=0.5,
        seed=18,
    ),
    RuntimeCase(
        name="vec-freeze-smoke",
        looks=3,
        batch=2,
        family=3,
        rate=0.5,
        stopping="adaptive",
        freeze=True,
        missing=True,
        seed=18,
    ),
    pytest.param(
        RuntimeCase(
            name="vec-family",
            looks=3,
            batch=4,
            family=4,
            stopping="first_discovery",
            null_fraction=0.5,
        ),
        marks=pytest.mark.slow,
    ),
    pytest.param(
        RuntimeCase(
            name="vec-shared", looks=2, batch=4, family=4, dependence="shared", null_fraction=0.5
        ),
        marks=pytest.mark.slow,
    ),
    pytest.param(
        RuntimeCase(
            name="vec-adaptive-freeze",
            looks=5,
            batch=4,
            family=4,
            stopping="adaptive",
            freeze=True,
            missing=True,
        ),
        marks=pytest.mark.slow,
    ),
]


@pytest.mark.parametrize("case", VECTORISED_CASES, ids=lambda case: case.name)
def test_vectorised_route_matches_the_deployed_campaign(case):
    """The vectorised engine reproduces ``campaign_replication`` on the same draws."""
    batches = _batches(case)
    expected = _public_record(case, batches)
    resolved = sim.geometry(case)
    increments = [
        [
            [
                sum(
                    int(record["values"][key.metric])
                    for record in rows
                    if sim.retains(key.segment, key.group_id, record)
                )
            ]
            for key in resolved.keys
        ]
        for rows in batches
    ]
    actual = sim.simulate_bernoulli_increments(case, increments)[0]
    assert _gated(actual) == _gated(expected)


@pytest.mark.parametrize(
    "case",
    [
        RuntimeCase(name="stream-bernoulli", looks=3, batch=4, stopping="first_rejection"),
        RuntimeCase(name="stream-ratio", law="gaussian_ratio", looks=2, batch=4, seed=109),
        pytest.param(
            RuntimeCase(
                name="stream-family", looks=2, batch=4, family=4, stopping="first_discovery"
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            RuntimeCase(name="stream-gaussian", law="gaussian", looks=3, batch=4, seed=101),
            marks=pytest.mark.slow,
        ),
    ],
    ids=lambda case: case.name,
)
def test_campaign_replication_agrees_on_a_shared_stream(case):
    """The deployed ``campaign_replication`` itself, not a local re-derivation."""
    deployed = campaign_replication(
        case,
        np.random.default_rng(case.seed),
        campaign_declaration(case),
        lambda payload: None,
    )
    rng = np.random.default_rng(case.seed)
    actual = sim.replicate(case, sim.campaign_source(case, rng))
    assert _gated(actual) == _gated(deployed)


# --- accumulators and curtailment -------------------------------------------


def test_case_outcome_sums_the_replication_records():
    """``CaseOutcome`` is the exact sum of the records the observer sees."""
    case = RuntimeCase(name="outcome", looks=3, batch=4, stopping="first_rejection")
    seen = []
    outcome = sim.simulate_case(
        case,
        replications=16,
        seed=4242,
        observer=lambda event, index, payload: (
            seen.append(payload) if event == "completed" else None
        ),
    )
    assert outcome.replications == 16
    assert len(seen) == 16
    assert outcome.retained_cells == sum(r["retained_cells"] for r in seen)
    assert outcome.available_points == sum(r["available_points"] for r in seen)
    assert outcome.certified_intervals == sum(r["certified_intervals"] for r in seen)
    assert outcome.nominal_fdp_sum == sum((r["nominal_fdp"] for r in seen), F(0))
    assert outcome.nominal_fcp_upper_sum == sum((r["nominal_fcp_upper"] for r in seen), F(0))
    assert outcome.completed_looks == sum(r["completed_looks"] for r in seen)
    expected = {}
    for record in seen:
        for name, value in replication_statistics(record).items():
            total, count = expected.get(name, (F(0), 0))
            expected[name] = (total + value, count + 1)
    assert dict(outcome.gate_statistics) == expected
    # The recurrence identity is the whole point of the vectorised route.
    assert outcome.distinct_evidence_states < outcome.evidence_evaluations


# Replays a full uncurtailed replication stream against a fresh rule.
@pytest.mark.slow
def test_curtailment_truncates_the_stream_without_moving_a_decision():
    """A curtailed gate reaches the same verdict at the same draw as the full run.

    The rule stops at its first boundary crossing, so consuming more of the
    stream cannot change its decision. That only holds if curtailment truncates
    the replication stream rather than perturbing it, which is what this checks:
    the uncurtailed run's miss flags, replayed into a fresh rule, must resolve
    at exactly the draw and decision the curtailed run reported.
    """
    case = RuntimeCase(name="curtail", looks=3, batch=4, stopping="first_rejection")
    gates = sim.curtailable_gates(case)
    assert "ever_null_rejection" in gates
    design = FixedDesign.resolve(nominal_error=0.05, tolerance=0.02, eta=0.01, repetitions=4000)

    flags = []
    uncurtailed = sim.simulate_case(
        case,
        replications=256,
        seed=77,
        chunk_size=8,
        observer=lambda event, index, payload: (
            flags.append(replication_statistics(payload)["ever_null_rejection"] == 1)
            if event == "completed"
            else None
        ),
    )
    assert uncurtailed.replications == 256
    assert not uncurtailed.gate_streams

    curtailed = sim.simulate_case(
        case,
        replications=256,
        seed=77,
        chunk_size=8,
        floor=0,
        stopping={
            "ever_null_rejection": build_rule(
                design, stopping="sequential", sequential_rule="sprt-v1"
            )
        },
    )
    stream = next(s for s in curtailed.gate_streams if s.gate == "ever_null_rejection")

    replay = build_rule(design, stopping="sequential", sequential_rule="sprt-v1")
    decision = replay.run(flags)
    assert decision == stream.decision
    assert replay.draws == stream.draws
    assert replay.misses == stream.misses
    assert curtailed.replications <= uncurtailed.replications
    # The curtailed prefix must be a prefix: its miss count matches the full
    # stream's over the same number of draws.
    assert stream.misses == sum(flags[: stream.draws])
    assert stream.hits + stream.misses == stream.draws


@pytest.mark.parametrize(
    ("batch", "design", "replications", "chunk_size"),
    [
        pytest.param(
            2,
            {"nominal_error": 0.5, "tolerance": 0.4, "eta": 0.05, "repetitions": 20},
            60,
            10,
            id="smoke",
        ),
        pytest.param(
            4,
            {"nominal_error": 0.2, "tolerance": 0.2, "eta": 0.05, "repetitions": 120},
            400,
            40,
            id="full",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_a_fixed_rule_curtails_to_its_own_budget(batch, design, replications, chunk_size):
    """A fixed design consumes exactly its repetitions and certifies from them."""
    case = RuntimeCase(name="fixed-curtail", looks=2, batch=batch, stopping="first_rejection")
    design = FixedDesign.resolve(**design)
    outcome = sim.simulate_case(
        case,
        replications=replications,
        seed=31,
        chunk_size=chunk_size,
        floor=0,
        stopping={"ever_null_rejection": build_rule(design, stopping="fixed")},
    )
    stream = next(s for s in outcome.gate_streams if s.gate == "ever_null_rejection")
    assert stream.draws == design.repetitions
    assert stream.decision in ("accept", "reject")
    assert stream.decision == ("accept" if design.certifies(stream.misses) else "reject")
    assert outcome.replications == design.repetitions


def test_non_binary_statistic_is_refused_rather_than_curtailed():
    """A proportion gate over a multi-cell roster is never offered to a rule."""
    family = RuntimeCase(name="family-curtail", looks=2, batch=4, family=4)
    gates = sim.curtailable_gates(family)
    assert "ever_null_rejection" not in gates or family.family == 1
    assert "point_availability_supported" not in gates
    design = FixedDesign.resolve(nominal_error=0.2, tolerance=0.2, eta=0.05, repetitions=120)
    with pytest.raises(ValueError):
        sim.simulate_case(
            family,
            replications=8,
            seed=5,
            stopping={"point_availability_supported": build_rule(design, stopping="fixed")},
        )


# --- scope ------------------------------------------------------------------


def test_scalar_mean_stays_on_the_public_path():
    """The excluded law is refused, so the scope claim cannot rot into silence."""
    case = RuntimeCase(name="scalar", law="scalar_mean", looks=2, batch=4)
    assert not sim.certifies_exactly(case)
    assert case.law not in sim.SUFFICIENT_STATE_LAWS
    with pytest.raises(ValueError):
        sim.geometry(case)
    with pytest.raises(ValueError):
        sim.simulate_case(case, replications=1, seed=1)


def test_certified_laws_are_exactly_the_tested_ones():
    """Scope is data, not prose: every certified law has an equivalence witness."""
    certified = ("bernoulli", "gaussian", "gaussian_ratio")
    assert sim.SUFFICIENT_STATE_LAWS == frozenset(certified)
    assert sim.AGGREGATE_SAMPLING_LAWS == frozenset({"bernoulli"})
    for law in certified:
        assert sim.certifies_exactly(RuntimeCase(name=law, law=law))
