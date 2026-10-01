"""Bounded exact witnesses, not certification of the complete sequential-inference contract.

The independent oracle owns all expected probabilities and decisions. Production
is imported only on the comparison side. Exhaustive families use four looks;
these tests make no Gaussian, inversion, source/wire, or general-horizon claim.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from functools import cache
from itertools import product
from math import comb
from typing import cast

import pytest

from increment.estimation._sequential_likelihood import (
    BernoulliState,
    BetaPrior,
    LikelihoodCertificate,
    bernoulli_evidence,
)
from increment.estimation.family import e_bh_select
from tests.estimation._sequential_proof_oracle import (
    Alternative,
    Arm,
    Path,
    Stopped,
    adaptive_frozen_stop,
    e_bh,
    evidence,
    family_stop,
    first_rejection,
    likelihood,
    log_bounds,
    null_supremum,
    paths,
    paths_heterogeneous,
    predictive,
    unrestricted,
)

F = Fraction
ONE = F(1)
ALTERNATIVES: tuple[Alternative, ...] = ("two-sided", "greater", "less")
HORIZON = 4
LEVEL = F(1, 3)
PAIRED_STATES = tuple(
    (Arm(n, c), Arm(n, t)) for n in range(HORIZON + 1) for c, t in product(range(n + 1), repeat=2)
)


@cache
def _certificate(
    control: Arm, treatment: Arm, alternative: Alternative, ratio: Fraction = ONE
) -> LikelihoodCertificate:
    prior = BetaPrior(1, 1)
    return bernoulli_evidence(
        BernoulliState(control.n, control.successes),
        BernoulliState(treatment.n, treatment.successes),
        prior,
        prior,
        ratio=ratio,
        alternative=alternative,
    )


def _assert_enclosed(certificate: LikelihoodCertificate, q: Fraction, denominator: Fraction):
    assert certificate.status == "finite"
    assert certificate.reason is None
    for interval, exact in (
        (certificate.log_predictive, q),
        (certificate.log_null_sup, denominator),
        (certificate.log_e, q / denominator),
    ):
        assert interval is not None
        lower, upper = log_bounds(exact)
        # Containment of the independent enclosure proves lower evidence <= E.
        assert interval.lo <= lower <= upper <= interval.hi
    assert certificate.log_e is not None
    assert certificate.log_e.hi - certificate.log_e.lo <= F(1, 10**12)


@cache
def _kernel_selection(
    arms: tuple[Arm, ...], alternative: Alternative, q: Fraction
) -> tuple[int, ...]:
    """e-BH over the private certificate lower bounds, with no public route.

    This compares the deployed selector against the exact one at a sufficient
    state; the registered snapshot, checkpoint, runtime and family-selection
    route is exercised by the public witness, not here.
    """
    lower_logs = []
    for treatment in arms[1:]:
        certificate = _certificate(arms[0], treatment, alternative)
        assert certificate.log_e is not None
        lower_logs.append(certificate.log_e.lo)
    return tuple(e_bh_select(lower_logs, q))


@pytest.mark.parametrize("n", range(9))
@pytest.mark.parametrize("a,b", [(F(1), F(1)), (F(1, 2), F(3, 2)), (F(2), F(3))])
def test_predictive_normalizes_and_satisfies_both_exact_recurrences(n, a, b):
    total = F(0)
    for successes in range(n + 1):
        arm = Arm(n, successes)
        q = predictive(arm, a, b)
        success = predictive(arm.append(1), a, b)
        failure = predictive(arm.append(0), a, b)
        assert success == q * (a + successes) / (a + b + n)
        assert failure == q * (b + n - successes) / (a + b + n)
        assert success + failure == q
        if a == b == 1:
            assert q == F(1, (n + 1) * comb(n, successes))
        total += comb(n, successes) * q
    assert total == 1
    assert predictive(Arm(), a, b) == 1


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("control,treatment", PAIRED_STATES)
def test_every_reachable_pair_state_encloses_exact_evidence_and_decisions(
    control, treatment, alternative
):
    certificate = _certificate(control, treatment, alternative)
    q = predictive(control) * predictive(treatment)
    denominator = null_supremum(control, treatment, alternative)
    _assert_enclosed(certificate, q, denominator)
    exact = q / denominator
    for alpha in (LEVEL, F(1, 10)):
        assert alpha * exact != 1  # These probes avoid conservative equality ambiguity.
        assert _kernel_selection((control, treatment), alternative, alpha) == (
            (0,) if alpha * exact >= 1 else ()
        )


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("n,successes", [(n, s) for n in range(5) for s in range(n + 1)])
def test_empty_arms_and_endpoint_likelihoods_have_exact_certificates(n, successes, alternative):
    observed, empty = Arm(n, successes), Arm()
    assert likelihood(empty, F(0)) == likelihood(empty, F(1)) == 1
    assert likelihood(Arm(n, 0), F(0)) == likelihood(Arm(n, n), F(1)) == 1
    for control, treatment in ((observed, empty), (empty, observed)):
        denominator = null_supremum(control, treatment, alternative)
        assert denominator == unrestricted(observed)
        _assert_enclosed(
            _certificate(control, treatment, alternative), predictive(observed), denominator
        )


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("control,treatment", [(Arm(4, 1), Arm(2, 2)), (Arm(2, 2), Arm(4, 1))])
def test_unequal_arm_sizes_pool_counts_instead_of_averaging_rates(control, treatment, alternative):
    pooled = F(1, 2)
    equality = likelihood(control, pooled) * likelihood(treatment, pooled)
    assert null_supremum(control, treatment, "two-sided") == equality
    _assert_enclosed(
        _certificate(control, treatment, alternative),
        predictive(control) * predictive(treatment),
        null_supremum(control, treatment, alternative),
    )


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("p", [F(1, 2), F(1, 1000)])
def test_exhaustive_singleton_ever_rejection_is_bounded_exactly(p, alternative):
    total = rejected = recovered = F(0)
    count = 0
    for path in paths(HORIZON, 2, p):
        count += 1
        total += path.probability
        stopped = first_rejection(path, LEVEL, alternative)
        rejected += path.probability * stopped.all_null_fdp
        if stopped.selected:
            assert stopped.look == 3
            for arms in tuple(path.prefixes())[: stopped.look]:
                assert LEVEL * evidence(arms[0], arms[1], alternative) < 1
            final = tuple(path.prefixes())[-1]
            if LEVEL * evidence(final[0], final[1], alternative) < 1:
                recovered += path.probability
    assert count == 256
    assert total == 1
    directions = 2 if alternative == "two-sided" else 1
    assert rejected == directions * (p * (1 - p)) ** 3
    assert 0 < rejected <= LEVEL
    assert recovered > 0  # A final-count-only calculation would miss these rejections.


def test_prefix_order_changes_first_rejection_for_identical_final_counts():
    early = Path(((0, 1), (0, 1), (0, 1), (0, 0)), F(1, 256))
    late = Path(((0, 0), (0, 1), (0, 1), (0, 1)), F(1, 256))
    assert tuple(early.prefixes())[-1] == tuple(late.prefixes())[-1]
    assert first_rejection(early, LEVEL, "greater").look == 3
    assert first_rejection(early, LEVEL, "greater").selected == (0,)
    assert first_rejection(late, LEVEL, "greater").selected == ()


def test_enumeration_reaches_every_declared_pair_state_and_retains_all_paths():
    paired = tuple(paths(HORIZON, 2, F(1, 2)))
    reached = {arms for path in paired for arms in path.prefixes()}
    assert len(paired) == len({path.rows for path in paired}) == 256
    assert reached == set(PAIRED_STATES)
    assert len(reached) == 55


@pytest.mark.slow
@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("p", [F(1, 2), F(1, 1000)])
def test_exhaustive_shared_control_predictable_stop_controls_exact_fdr(p, alternative):
    """4,096 paths; all-null FDP is 1{R>0}, with the same control in both cells."""
    total = exact_fdr = F(0)
    expected_e = [F(0), F(0)]
    stop_mass: dict[int, Fraction] = {}
    reached: set[tuple[Arm, ...]] = set()
    count = 0
    for path in paths(HORIZON, 3, p):
        count += 1
        total += path.probability
        stopped = family_stop(path, LEVEL, alternative)
        prefixes = tuple(path.prefixes())
        reached.update(prefixes)
        stop_mass[stopped.look] = stop_mass.get(stopped.look, F(0)) + path.probability
        exact_fdr += path.probability * stopped.all_null_fdp
        assert _kernel_selection(prefixes[stopped.look], alternative, LEVEL) == stopped.selected
        for i, value in enumerate(stopped.evidence):
            expected_e[i] += path.probability * value
        assert stopped.all_null_fdp <= LEVEL * sum(stopped.evidence, F(0)) / 2
        if stopped.selected:
            assert all(
                LEVEL * len(stopped.selected) * stopped.evidence[i] >= 2 for i in stopped.selected
            )
    assert count == 4096
    assert len(reached) == 225
    assert total == 1
    assert set(stop_mass) == {3, 4}
    assert all(mass > 0 for mass in stop_mass.values())
    assert all(value <= 1 for value in expected_e)
    assert 0 < exact_fdr <= LEVEL * sum(expected_e, F(0)) / 2 <= LEVEL


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_exhaustive_mixed_null_predictable_stop_bounds_exact_fdr(alternative):
    """4,096 mixed-null paths: with one genuine non-null cell FDP is not 1{R>0}."""
    probabilities = (F(1, 2), F(1, 2), F(1, 10) if alternative == "less" else F(9, 10))
    total = fdr = discovery_mass = nonnull_only_mass = F(0)
    fdp_mass: dict[Fraction, Fraction] = {}
    count = 0
    for path in paths_heterogeneous(HORIZON, probabilities):
        count += 1
        total += path.probability
        stopped = family_stop(path, LEVEL, alternative)
        fdp = stopped.false_discovery_proportion((0,))
        fdp_mass[fdp] = fdp_mass.get(fdp, F(0)) + path.probability
        fdr += path.probability * fdp
        if stopped.selected:
            discovery_mass += path.probability
            if not fdp:
                nonnull_only_mass += path.probability
            assert stopped.all_null_fdp == 1
    assert count == 4096
    assert total == 1
    # Only the second cell is non-null, so both cells, the null cell alone and
    # the non-null cell alone are all reachable stopped discoveries.
    assert set(fdp_mass) == {F(0), F(1, 2), F(1)}
    assert all(mass > 0 for mass in fdp_mass.values())
    assert 0 < nonnull_only_mass < discovery_mass
    assert 0 < fdr <= LEVEL


@pytest.mark.slow
@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("p", [F(1, 2), F(1, 1000)])
def test_exhaustive_adaptive_frozen_missing_witness_is_exact_and_bounded(p, alternative):
    """The bounded four-look witness keeps 4,096 exact paths and a three-cell roster."""
    total = fdr = F(0)
    stop_mass: dict[int, Fraction] = {}
    count = 0
    for path in paths(HORIZON, 3, p):
        count += 1
        total += path.probability
        stopped = adaptive_frozen_stop(path, LEVEL, alternative)
        stop_mass[stopped.look] = stop_mass.get(stopped.look, F(0)) + path.probability
        fdr += path.probability * stopped.all_null_fdp
        assert len(stopped.evidence) == 3
        assert stopped.evidence[2] == 0
        assert stopped.all_null_fdp <= LEVEL * sum(stopped.evidence, F(0)) / 3
        if stopped.selected:
            assert all(
                LEVEL * len(stopped.selected) * stopped.evidence[i] >= 3 for i in stopped.selected
            )
        prefixes = tuple(path.prefixes())
        frozen_look = min(stopped.look, HORIZON // 2)
        frozen_control, frozen_treatment = prefixes[frozen_look][:2]
        current_control, _, current_second = prefixes[stopped.look]
        frozen_certificate = _certificate(frozen_control, frozen_treatment, alternative)
        current_certificate = _certificate(current_control, current_second, alternative)
        assert frozen_certificate.log_e is not None
        assert current_certificate.log_e is not None
        deployed = tuple(
            e_bh_select(
                (frozen_certificate.log_e.lo, current_certificate.log_e.lo, float("-inf")),
                LEVEL,
            )
        )
        assert set(deployed) <= set(stopped.selected)
        boundaries = {F(3, 1) / (LEVEL * k) for k in range(1, 4)}
        if not any(value in boundaries for value in stopped.evidence):
            assert deployed == stopped.selected
    assert count == 4096
    assert total == 1
    assert stop_mass[HORIZON] > 0
    assert any(look < HORIZON and mass > 0 for look, mass in stop_mass.items())
    assert 0 <= fdr <= LEVEL


def test_adaptive_freezes_before_the_next_look_and_does_not_substitute_retroactively():
    path = Path(
        ((0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 0)),
        F(1, 4096),
    )
    stopped = adaptive_frozen_stop(path, LEVEL, "greater")
    frozen = evidence(Arm(2, 0), Arm(2, 0), "greater")
    assert stopped.look == HORIZON
    assert stopped.evidence[0] == frozen < 1
    assert evidence(Arm(3, 0), Arm(3, 1), "greater") > frozen
    assert stopped.evidence[0] != evidence(Arm(4, 0), Arm(4, 2), "greater")


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_predictable_stop_uses_both_cells_and_selects_at_the_next_reveal(alternative):
    row = (1, 0, 0) if alternative == "less" else (0, 1, 1)
    prefix = (row, row)
    results = []
    for next_row in product((0, 1), repeat=3):
        path = Path((*prefix, next_row, (0, 0, 0)), F(1, 4096))
        stopped = family_stop(path, LEVEL, alternative)
        assert stopped.look == 3  # Already scheduled before seeing next_row.
        results.append(stopped.selected)
    assert () in results and (0, 1) in results
    weak_second = (row[0], row[1], row[0])
    path = Path((weak_second, weak_second, row, row), F(1, 4096))
    assert family_stop(path, LEVEL, alternative).look == 4


def test_exact_e_bh_keeps_roster_and_boundary_ties():
    assert e_bh((F(3), F(3)), LEVEL, m=2) == (0, 1)
    assert e_bh((F(3), F(0)), LEVEL, m=2) == ()
    assert e_bh((F(6), F(0)), LEVEL, m=2) == (0,)
    assert e_bh((F(0), F(6)), LEVEL, m=2) == (1,)
    with pytest.raises(ValueError):
        e_bh((F(3),), LEVEL, m=2)


@pytest.mark.parametrize(
    "values", [(F(4), F(1, 100)), (F(7), F(1, 100)), (F(4), F(4)), (F(3), F(3))]
)
def test_deployed_family_selection_preserves_the_full_roster_and_conservative_ties(values):
    lower_logs = tuple(log_bounds(value)[0] for value in values)
    selected = tuple(e_bh_select(lower_logs, LEVEL))
    exact = e_bh(values, LEVEL, m=2)
    assert set(selected) <= set(exact)
    if values != (F(3), F(3)):
        assert selected == exact


@pytest.mark.parametrize("alternative", ("greater", "less"))
@pytest.mark.parametrize("ratio", (F(4, 5), F(6, 5)))
@pytest.mark.parametrize("high,low", [(3, 1), (4, 0)])
def test_wrong_direction_shifted_null_uses_feasible_unrestricted_mles(
    alternative, ratio, high, low
):
    control, treatment = Arm(4, high), Arm(4, low)
    if alternative == "less":
        control, treatment = treatment, control
    c_rate, t_rate = F(control.successes, control.n), F(treatment.successes, treatment.n)
    assert t_rate < ratio * c_rate if alternative == "greater" else t_rate > ratio * c_rate
    denominator = unrestricted(control) * unrestricted(treatment)
    q = predictive(control) * predictive(treatment)
    assert q / denominator < 1
    certificate = _certificate(control, treatment, alternative, ratio)
    _assert_enclosed(certificate, q, denominator)
    assert certificate.log_e is not None
    assert e_bh_select((certificate.log_e.lo,), LEVEL) == []
    if high == 4:
        assert LEVEL * evidence(control, treatment, "two-sided") > 1


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_zero_event_prefixes_are_finite_and_do_not_reject(alternative):
    path = Path(((0, 0),) * HORIZON, F(1))
    for look, (control, treatment) in enumerate(path.prefixes()):
        exact = evidence(control, treatment, alternative)
        assert exact == F(1, (look + 1) ** 2)
        _assert_enclosed(_certificate(control, treatment, alternative), exact, F(1))
        assert _kernel_selection((control, treatment), alternative, LEVEL) == ()
    assert first_rejection(path, LEVEL, alternative).selected == ()


def test_rational_log_enclosure_has_exact_origin_reciprocity_and_refinement():
    assert log_bounds(F(1)) == (F(0), F(0))
    for value in (F(1, 7), F(2), F(3), F(17, 4)):
        lo, hi = log_bounds(value, 24)
        tighter_lo, tighter_hi = log_bounds(value, 48)
        assert lo < tighter_lo < tighter_hi < hi
        assert log_bounds(1 / value, 24) == (-hi, -lo)
    lo, hi = log_bounds(F(2), 24)
    assert log_bounds(F(4), 24) == (2 * lo, 2 * hi)


def test_oracle_records_copy_nested_inputs_and_are_immutable():
    rows = [[0, 1], [1, 0]]
    path = Path(cast(tuple[tuple[int, ...], ...], rows), F(1, 16))
    values, selected = [F(4)], [0]
    stopped = Stopped(
        2,
        cast(tuple[Fraction, ...], values),
        cast(tuple[int, ...], selected),
    )
    rows[0][0] = 1
    values[0], selected[0] = F(0), 1
    assert path.rows == ((0, 1), (1, 0))
    assert stopped.evidence == (F(4),) and stopped.selected == (0,)
    for record, field, value in ((Arm(), "n", 1), (path, "rows", ()), (stopped, "look", 3)):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, value)


@pytest.mark.parametrize("n,successes", [(-1, 0), (0, 1), (2, -1), (True, 0), (F(2), 1)])
def test_oracle_refuses_invalid_counts(n, successes):
    with pytest.raises(ValueError):
        Arm(n, successes)


def test_oracle_refuses_invalid_probability_prior_direction_path_and_level():
    with pytest.raises(ValueError):
        likelihood(Arm(), F(2))
    with pytest.raises(ValueError):
        predictive(Arm(), F(0), F(1))
    with pytest.raises(ValueError):
        null_supremum(Arm(), Arm(), cast(Alternative, "invalid"))
    with pytest.raises(ValueError):
        Path(((0, 2),), F(1))
    with pytest.raises(ValueError):
        tuple(paths(0, 2, F(1, 2)))
    with pytest.raises(ValueError):
        tuple(paths(1, 2, F(-1)))
    with pytest.raises(ValueError):
        e_bh((F(1),), F(0), m=1)
