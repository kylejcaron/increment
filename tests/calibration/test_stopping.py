"""Both stopping rules decide the same claim; the sequential one inherits its
error probabilities from the fixed design instead of choosing them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from calibration.stopping import (
    FixedDesign,
    FixedStopping,
    SequentialStopping,
    StoppingError,
    build_rule,
)
from tests.mc import binomial_error_upper_bound, family_eta, scientific_delta

# The campaign's frozen design, restated so these checks survive manifest
# re-freezes: eta budgets each directional bound of 1692 gated statistics under
# a .01 family-wise allocation, and the design needs 189,997 repetitions at .05.
NOMINAL_ERROR = 0.05
TOLERANCE = scientific_delta(NOMINAL_ERROR)
ETA = family_eta(0.01, 1692)
REPETITIONS = 189_997

# Published operating point and average sample numbers for that design.
PUBLISHED_ACCEPTANCE_COUNT = 9975
PUBLISHED_ALPHA = 3.35e-07
PUBLISHED_BETA = 7.68e-07
PUBLISHED_ASN_AT_NOMINAL = 56_695
PUBLISHED_ASN_AT_TOLERANCE = 58_210


@pytest.fixture(scope="module")
def design() -> FixedDesign:
    return FixedDesign.resolve(
        nominal_error=NOMINAL_ERROR,
        tolerance=TOLERANCE,
        eta=ETA,
        repetitions=REPETITIONS,
    )


def _paced_stream(rate: float, draws: int):
    """Miss flags spaced as evenly as a given rate allows.

    A miss is emitted exactly when ``floor(i * rate)`` increments, so the
    running miss count never differs from ``i * rate`` by as much as one. Fed
    at the indifference rate this is the worst case the draw cap exists for:
    the log-likelihood walk stays within a single increment of zero forever.
    """
    previous = 0
    for draw in range(1, draws + 1):
        current = math.floor(draw * rate)
        yield current > previous
        previous = current


def _simulate_draws(rule: SequentialStopping, rate: float, reps: int, seed: int) -> np.ndarray:
    """Draw counts at which independent SPRT runs at *rate* stop.

    Vectorised over replications: the statistic after ``m`` draws is
    ``m * hit + misses * (miss - hit)``, so one cumulative sum per block finds
    every crossing without a Python loop over draws.
    """
    rng = np.random.default_rng(seed)
    slope = rule.miss_increment - rule.hit_increment
    stopped = np.zeros(reps, dtype=np.int64)
    carried = np.zeros(reps, dtype=np.int64)
    active = np.arange(reps)
    seen = 0
    while active.size and seen < rule.max_draws:
        # Cap the working array at a few million entries, whatever the
        # replication count, so more replications cost time and not memory.
        width = min(8_000_000 // active.size, rule.max_draws - seen)
        misses = np.cumsum(rng.random((active.size, width)) < rate, axis=1)
        misses += carried[active][:, None]
        draws = seen + np.arange(1, width + 1)
        statistic = draws * rule.hit_increment + misses * slope
        crossed = (statistic <= rule.accept_boundary) | (statistic >= rule.reject_boundary)
        resolved = crossed.any(axis=1)
        first = np.where(resolved, crossed.argmax(axis=1), width - 1)
        stopped[active[resolved]] = seen + first[resolved] + 1
        carried[active] = misses[:, -1]
        active = active[~resolved]
        seen += width
    stopped[active] = rule.max_draws
    return stopped


class TestFixedDesign:
    def test_reference_design_reproduces_the_published_operating_point(self, design):
        assert design.acceptance_count == PUBLISHED_ACCEPTANCE_COUNT
        assert design.type_i_error == pytest.approx(PUBLISHED_ALPHA, rel=1e-3)
        assert design.type_ii_error == pytest.approx(PUBLISHED_BETA, rel=1e-3)

    def test_declared_error_rates_are_the_exact_binomial_tails_at_k_star(self, design):
        k = design.acceptance_count
        assert design.type_i_error == pytest.approx(
            float(binom.sf(k, REPETITIONS, NOMINAL_ERROR)), rel=1e-12
        )
        assert design.type_ii_error == pytest.approx(
            float(binom.cdf(k, REPETITIONS, NOMINAL_ERROR + TOLERANCE)), rel=1e-12
        )

    def test_certification_follows_the_exact_clopper_pearson_bound(self, design):
        edge = design.certifiable_count
        assert design.certifies(edge)
        assert not design.certifies(edge + 1)
        assert binomial_error_upper_bound(edge, REPETITIONS, ETA) <= design.error_upper_limit
        assert binomial_error_upper_bound(edge + 1, REPETITIONS, ETA) > design.error_upper_limit

    def test_certifying_a_count_is_exactly_bounding_its_tolerance_edge_acceptance(self, design):
        # Clopper-Pearson duality: CP_U(k, n, eta) <= q + delta iff the
        # probability of seeing at most k misses when the truth sits on the
        # tolerance edge is itself within eta.
        limit = design.error_upper_limit
        for count in (0, design.acceptance_count, design.certifiable_count, REPETITIONS // 10):
            edge_acceptance = float(binom.cdf(count, REPETITIONS, limit))
            assert design.certifies(count) == (edge_acceptance <= ETA)

    def test_a_design_that_cannot_certify_at_its_operating_point_is_refused(self):
        # A tenth of the repetitions leaves the Clopper-Pearson bound wider
        # than the whole tolerance, so the operating point no longer certifies
        # and its tolerance-edge acceptance blows past eta.
        with pytest.raises(StoppingError):
            FixedDesign.resolve(
                nominal_error=NOMINAL_ERROR,
                tolerance=TOLERANCE,
                eta=ETA,
                repetitions=REPETITIONS // 10,
            )

    def test_a_tolerance_too_tight_for_the_repetition_count_is_refused(self):
        # Halving the tolerance without re-sizing n drags the operating point
        # under the tolerance edge; both error probabilities leave the budget.
        with pytest.raises(StoppingError):
            FixedDesign.resolve(
                nominal_error=NOMINAL_ERROR,
                tolerance=TOLERANCE / 2,
                eta=ETA,
                repetitions=REPETITIONS,
            )

    def test_certifiable_boundary_rises_with_eta_and_with_the_tolerance(self):
        looser_budget = [
            FixedDesign.resolve(
                nominal_error=NOMINAL_ERROR, tolerance=TOLERANCE, eta=eta, repetitions=REPETITIONS
            ).certifiable_count
            for eta in (ETA, 1e-5, 1e-4, 1e-2)
        ]
        assert looser_budget == sorted(looser_budget)
        assert looser_budget[0] < looser_budget[-1]

        wider_tolerance = [
            FixedDesign.resolve(
                nominal_error=NOMINAL_ERROR,
                tolerance=tolerance,
                eta=ETA,
                repetitions=REPETITIONS,
            ).certifiable_count
            for tolerance in (0.005, 0.006, 0.007, 0.008)
        ]
        assert wider_tolerance == sorted(wider_tolerance)
        assert wider_tolerance[0] < wider_tolerance[-1]


class TestFixedStopping:
    def test_it_draws_the_whole_budget_then_certifies(self, design):
        rule = FixedStopping(design)
        assert rule.max_draws == REPETITIONS
        assert rule.observe(miss=True) == "continue"
        assert rule.run(_paced_stream(NOMINAL_ERROR, REPETITIONS - 1)) == "accept"
        assert rule.draws == REPETITIONS

    def test_a_cell_past_tolerance_fails_the_bound(self, design):
        rule = FixedStopping(design)
        assert rule.run(_paced_stream(NOMINAL_ERROR + 2 * TOLERANCE, REPETITIONS)) == "reject"
        assert rule.misses > design.certifiable_count


class TestSequentialStopping:
    def test_it_inherits_the_fixed_designs_exact_error_probabilities(self, design):
        rule = SequentialStopping(design)
        assert rule.alpha == design.type_i_error
        assert rule.beta == design.type_ii_error
        assert rule.accept_boundary == pytest.approx(
            math.log(design.type_ii_error / (1.0 - design.type_i_error)), rel=1e-12
        )
        assert rule.reject_boundary == pytest.approx(
            math.log((1.0 - design.type_ii_error) / design.type_i_error), rel=1e-12
        )

    def test_its_power_curve_returns_exactly_those_error_probabilities(self, design):
        # The inheritance claim is only worth anything if the rule really has
        # those errors: Wald's operating characteristic at the two hypotheses
        # must come back as 1 - alpha and beta.
        rule = SequentialStopping(design)
        assert rule.acceptance_probability(NOMINAL_ERROR) == pytest.approx(
            1.0 - design.type_i_error, rel=1e-9
        )
        assert rule.acceptance_probability(design.error_upper_limit) == pytest.approx(
            design.type_ii_error, rel=1e-6
        )
        # At the indifference rate the walk has no drift and the two
        # boundaries split the outcome by their distances.
        span = rule.reject_boundary - rule.accept_boundary
        assert rule.acceptance_probability(rule.indifference_rate) == pytest.approx(
            rule.reject_boundary / span
        )
        assert rule.acceptance_probability(0.0) == 1.0
        assert rule.acceptance_probability(1.0) == 0.0

    def test_its_realised_error_guarantee_stays_inside_the_designs_budget(self, design):
        rule = SequentialStopping(design)
        assert rule.type_i_guarantee >= rule.alpha
        assert rule.type_ii_guarantee >= rule.beta
        assert rule.type_i_guarantee <= design.eta

    def test_a_clean_stream_accepts_long_before_the_fixed_budget(self, design):
        rule = SequentialStopping(design)
        assert rule.run(iter(lambda: False, None)) == "accept"
        assert rule.misses == 0
        assert rule.draws < REPETITIONS / 50

    def test_a_badly_failing_stream_rejects_almost_immediately(self, design):
        rule = SequentialStopping(design)
        assert rule.run(iter(lambda: True, None)) == "reject"
        assert rule.draws < 500

    def test_expected_draws_match_the_measured_average_sample_numbers(self, design):
        rule = SequentialStopping(design)
        assert rule.expected_draws(NOMINAL_ERROR) == pytest.approx(
            PUBLISHED_ASN_AT_NOMINAL, rel=0.01
        )
        assert rule.expected_draws(design.error_upper_limit) == pytest.approx(
            PUBLISHED_ASN_AT_TOLERANCE, rel=0.01
        )
        saving = REPETITIONS / rule.expected_draws(NOMINAL_ERROR)
        assert saving == pytest.approx(3.35, rel=0.01)

    def test_the_draw_cap_reports_unresolved_instead_of_accepting(self, design):
        rule = SequentialStopping(design)
        # At the indifference rate the untruncated test needs more draws than
        # the fixed budget, so the cap is a real outcome, not a formality.
        assert rule.expected_draws(rule.indifference_rate) > rule.max_draws
        assert rule.run(_paced_stream(rule.indifference_rate, rule.max_draws)) == "unresolved"
        assert rule.draws == rule.max_draws
        with pytest.raises(StoppingError):
            rule.observe(miss=False)

    def test_boundaries_widen_as_the_tolerance_widens(self, design):
        boundaries = [
            SequentialStopping(
                FixedDesign.resolve(
                    nominal_error=NOMINAL_ERROR,
                    tolerance=tolerance,
                    eta=ETA,
                    repetitions=REPETITIONS,
                )
            )
            for tolerance in (0.005, 0.006, 0.007, 0.008)
        ]
        accept = [rule.accept_boundary for rule in boundaries]
        reject = [rule.reject_boundary for rule in boundaries]
        assert accept == sorted(accept, reverse=True)
        assert reject == sorted(reject)
        assert accept[0] > accept[-1]
        assert reject[0] < reject[-1]

    @pytest.mark.slow
    def test_simulated_average_sample_numbers_match_the_published_ones(self, design):
        rule = SequentialStopping(design)
        reps = 1600
        for rate, published in (
            (NOMINAL_ERROR, PUBLISHED_ASN_AT_NOMINAL),
            (design.error_upper_limit, PUBLISHED_ASN_AT_TOLERANCE),
        ):
            stopped = _simulate_draws(rule, rate, reps, seed=20_260_921)
            mean = float(stopped.mean())
            # The SPRT stopping time is heavy tailed: 1600 replications give just
            # under 1% standard error on the mean, so a four-SE (~3.6%) band tests
            # "within a few percent of the published number" in a couple of seconds.
            standard_error = float(stopped.std(ddof=1)) / math.sqrt(reps)
            assert standard_error / mean < 0.012, (rate, mean, standard_error)
            assert abs(mean - published) <= 4.0 * standard_error, (
                rate,
                mean,
                published,
                standard_error,
            )


class TestRuleConstruction:
    def test_it_builds_the_declared_rule(self, design):
        assert build_rule(design, stopping="fixed").rule == "fixed"
        sequential = build_rule(design, stopping="sequential", sequential_rule="sprt-v1")
        assert sequential.rule == "sprt-v1"

    @pytest.mark.parametrize(
        "stopping,sequential_rule",
        [
            ("fixed", "sprt-v1"),
            ("sequential", None),
            ("sequential", "sprt-v2"),
            ("peek", None),
        ],
    )
    def test_it_refuses_a_declaration_it_cannot_execute(self, design, stopping, sequential_rule):
        with pytest.raises(StoppingError):
            build_rule(design, stopping=stopping, sequential_rule=sequential_rule)

    def test_a_resolved_rule_refuses_further_observations(self, design):
        rule = SequentialStopping(design)
        assert rule.run(iter(lambda: True, None)) == "reject"
        with pytest.raises(StoppingError):
            rule.observe(miss=True)
