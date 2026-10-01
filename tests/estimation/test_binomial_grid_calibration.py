"""Check probability integration against exhaustive production decisions."""

from __future__ import annotations

import math

import pytest
from scipy.stats import binom as _binom

from calibration import binomial_grid as cbg
from increment.estimation import binomial_rr

pytestmark = pytest.mark.slow


# --- Outer/treatment-arm exact window: quantile + safety-inflated omitted --


class TestExactWindow:
    def test_omitted_mass_never_exceeds_the_requested_budget(self):
        for n, p in [(5000, 1e-4), (300, 0.1), (1_000_000, 1e-4), (1000, 0.5)]:
            lo, hi, omitted = cbg.exact_outer_window(n, p, 1e-10)
            assert 0 <= lo <= hi <= n
            assert omitted <= 1e-10

    def test_window_mass_covers_essentially_all_probability(self):
        n, p = 100_000, 1e-4  # expected_events = 10
        lo, hi, omitted = cbg.exact_outer_window(n, p, 1e-12)
        captured = float(_binom.cdf(hi, n, p) - _binom.cdf(lo - 1, n, p))
        assert captured >= 1.0 - 1e-9
        assert abs(1.0 - captured - omitted) < 1e-9

    def test_degenerate_p_zero_and_one(self):
        assert cbg.exact_outer_window(10, 0.0, 1e-9) == (0, 0, 0.0)
        assert cbg.exact_outer_window(10, 1.0, 1e-9) == (10, 10, 0.0)


# --- Witness lower bound: correctness of the margin subtraction ------------


class TestWitnessMarginCorrection:
    """Regression for a review finding: `_tail_plus`/`_tail_minus` already
    INFLATE their raw return by their own support-window's omitted mass
    PLUS a floating-point safety margin, so recovering a genuine lower
    bound on the exact value must subtract BOTH back out -- subtracting
    only the margin (an earlier version of this module's bug) leaves the
    "lower bound" too high by the omitted-window-mass whenever that window
    actually omits something (`n_c` large enough to trigger `binomial_rr`'s
    own internal windowing, i.e. `n_c > 4096`)."""

    def test_witness_lower_bound_never_exceeds_real_production_value(self):
        n_c, n_t, r0 = 20_000, 20_000, 1.0
        beta = binomial_rr.nuisance_beta(0.05)
        x_c_obs = 40
        true_p_c = x_c_obs / n_c
        a, b = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        window = binomial_rr._support_window(n_c, a, b)
        assert window[2] > 0.0
        lower = cbg._witness_lower_plus(true_p_c, n_c, n_t, r0, x_c_obs, a, b, window)
        assert lower is not None
        for xt in range(0, 200, 10):
            prod = binomial_rr.p_plus(r0, x_c_obs, n_c, xt, n_t, beta)
            lo = beta + lower(xt)
            assert lo <= prod + 1e-12, (xt, lo, prod)

    @pytest.mark.parametrize("tail", ["plus", "minus"])
    def test_truncated_witness_stays_below_full_probability(self, tail):
        n_c = n_t = 40
        p_c = 0.25
        x_c_obs, x_t_obs = 10, 12
        beta = binomial_rr.nuisance_beta(0.05)
        a, b = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        omitted = float(_binom.cdf(7, n_c, p_c) + _binom.sf(12, n_c, p_c))
        window = (8, 12, omitted)
        witness = cbg._witness_lower_plus if tail == "plus" else cbg._witness_lower_minus
        lower = witness(p_c, n_c, n_t, 1.0, x_c_obs, a, b, window)
        assert lower is not None
        exact = math.fsum(
            float(_binom.pmf(xc, n_c, p_c))
            * float(
                _binom.sf(xc + x_t_obs - x_c_obs - 1, n_t, p_c)
                if tail == "plus"
                else _binom.cdf(xc + x_t_obs - x_c_obs, n_t, p_c)
            )
            for xc in range(n_c + 1)
        )
        assert lower(x_t_obs) <= exact


# --- Monotonicity of the certified witness surrogate ------------------------


class TestMonotoneSurrogate:
    """The witness lower bound is a single FIXED-point evaluation of
    `binomial_rr._tail_plus`/`_tail_minus` (no adaptive branch-and-bound),
    so it must be EXACTLY monotone in x_t -- unlike production's own
    adaptive `p_plus`/`p_minus`, whose non-monotonicity motivated this
    design (see the calibration script's module docstring)."""

    def test_witness_lower_plus_is_exactly_nonincreasing(self):
        n_c, n_t, r = 300, 300, 1.0
        beta = binomial_rr.nuisance_beta(0.05)
        x_c_obs = 20
        true_p_c = 20.0 / 300
        a, b = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        window = binomial_rr._support_window(n_c, a, b)
        lower = cbg._witness_lower_plus(true_p_c, n_c, n_t, r, x_c_obs, a, b, window)
        assert lower is not None
        vals = [lower(xt) for xt in range(0, n_t + 1, 3)]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_witness_lower_minus_is_exactly_nondecreasing(self):
        n_c, n_t, r = 300, 300, 1.0
        beta = binomial_rr.nuisance_beta(0.05)
        x_c_obs = 20
        true_p_c = 20.0 / 300
        a, b_cp = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        upper_q = min(b_cp, 1.0 / r) if r > 0 else b_cp
        window = binomial_rr._support_window(n_c, a, upper_q)
        lower = cbg._witness_lower_minus(true_p_c, n_c, n_t, r, x_c_obs, a, upper_q, window)
        assert lower is not None
        vals = [lower(xt) for xt in range(0, n_t + 1, 3)]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_witness_returns_none_off_cp_hit(self):
        n_c, n_t, r = 50, 50, 1.0
        beta = binomial_rr.nuisance_beta(0.05)
        x_c_obs = 40  # far from a tiny true_p_c -> guaranteed CP-miss
        true_p_c = 1e-4
        a, b = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        assert not (a <= true_p_c <= b)
        window = binomial_rr._support_window(n_c, a, b)
        assert cbg._witness_lower_plus(true_p_c, n_c, n_t, r, x_c_obs, a, b, window) is None

    def test_infeasible_minus_domain_is_handled_without_a_search(self):
        """`r` large enough that `1/r < a` (the CP lower bound) empties the
        restricted nuisance domain -- `binomial_rr.p_minus` returns exactly
        `beta` here (see its own early-return branch); the tail resolver
        must match that without attempting a search."""
        n_c, n_t = 50, 50
        beta = binomial_rr.nuisance_beta(0.05)
        x_c_obs = 40
        r = 1e6  # 1/r ~ 1e-6, far below a
        a, _ = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        assert 1.0 / r < a
        res = cbg.resolve_minus_tail(x_c_obs, n_c, n_t, beta, r, 0.5, 0.5, 0.025)
        prod = binomial_rr.p_minus(r, x_c_obs, n_c, 25, n_t, beta)
        assert prod == pytest.approx(beta)
        expect_accept = beta >= 0.025
        accept_mass, unresolved = cbg.single_tail_accept_mass("minus", res, n_t, 0.5)
        assert unresolved <= binomial_rr._eps_margin(2)
        assert (accept_mass > 0.5) == expect_accept


# --- Duality vs. exhaustive brute force --------------------------------------


def _brute_two_sided_covered_mass(x_c_obs, n_c, n_t, alpha, r0, p_t):
    beta = binomial_rr.nuisance_beta(alpha)
    alloc = alpha / 2.0
    mass = 0.0
    for xt in range(n_t + 1):
        pp = binomial_rr.p_plus(r0, x_c_obs, n_c, xt, n_t, beta)
        pm = binomial_rr.p_minus(r0, x_c_obs, n_c, xt, n_t, beta)
        if pp >= alloc and pm >= alloc:
            mass += float(_binom.pmf(xt, n_t, p_t))
    return mass


def _brute_one_sided_accept_mass(direction, x_c_obs, n_c, n_t, target, r0, p_t):
    beta = binomial_rr.nuisance_beta(0.05)
    mass = 0.0
    for xt in range(n_t + 1):
        v = (
            binomial_rr.p_plus(r0, x_c_obs, n_c, xt, n_t, beta)
            if direction == "plus"
            else binomial_rr.p_minus(r0, x_c_obs, n_c, xt, n_t, beta)
        )
        if v >= target:
            mass += float(_binom.pmf(xt, n_t, p_t))
    return mass


class TestDualityAgreesWithBruteForce:
    """Cross-checks the duality-based threshold resolution against a FULL
    exhaustive scan calling production's real (adaptive) `p_plus`/`p_minus`
    at every integer x_t -- the actual deployed decision, not a surrogate --
    on tractable small counts. Exact agreement (to float64 rounding) is the
    claim; a threshold or band-combination bug would show up as a real
    (not MC-noise-sized) mismatch here, not a rounding artifact."""

    @pytest.mark.parametrize("x_c_obs", [0, 3, 9, 15, 21, 27, 33, 39])
    def test_two_sided_null(self, x_c_obs):
        n_c, n_t, alpha, r0, p_t = 40, 40, 0.05, 1.0, 0.2
        beta = binomial_rr.nuisance_beta(alpha)
        rp = cbg.resolve_plus_tail(x_c_obs, n_c, n_t, beta, r0, 0.2, p_t, alpha / 2.0)
        rm = cbg.resolve_minus_tail(x_c_obs, n_c, n_t, beta, r0, 0.2, p_t, alpha / 2.0)
        mine, unresolved = cbg.two_sided_covered_mass(rp, rm, n_t, p_t)
        brute = _brute_two_sided_covered_mass(x_c_obs, n_c, n_t, alpha, r0, p_t)
        assert abs(mine - brute) <= unresolved + 1e-9

    @pytest.mark.parametrize("x_c_obs", [0, 2, 5, 8, 12, 20, 35, 50])
    def test_two_sided_nonnull_rare_event(self, x_c_obs):
        n_c, n_t, true_p_c, r0 = 50, 200, 0.02, 1.5
        p_t = min(1.0, r0 * true_p_c)
        alpha = 0.05
        beta = binomial_rr.nuisance_beta(alpha)
        rp = cbg.resolve_plus_tail(x_c_obs, n_c, n_t, beta, r0, true_p_c, p_t, alpha / 2.0)
        rm = cbg.resolve_minus_tail(x_c_obs, n_c, n_t, beta, r0, true_p_c, p_t, alpha / 2.0)
        mine, unresolved = cbg.two_sided_covered_mass(rp, rm, n_t, p_t)
        brute = _brute_two_sided_covered_mass(x_c_obs, n_c, n_t, alpha, r0, p_t)
        assert abs(mine - brute) <= unresolved + 1e-9

    @pytest.mark.parametrize("direction", ["plus", "minus"])
    @pytest.mark.parametrize("x_c_obs", [0, 1, 3, 5, 8, 12, 20])
    def test_one_sided(self, direction, x_c_obs):
        n_c, n_t, true_p_c, r0 = 50, 200, 0.02, 1.5
        p_t = min(1.0, r0 * true_p_c)
        alpha = 0.05
        beta = binomial_rr.nuisance_beta(alpha)
        resolver = cbg.resolve_plus_tail if direction == "plus" else cbg.resolve_minus_tail
        res = resolver(x_c_obs, n_c, n_t, beta, r0, true_p_c, p_t, alpha)
        mine, unresolved = cbg.single_tail_accept_mass(direction, res, n_t, p_t)
        brute = _brute_one_sided_accept_mass(direction, x_c_obs, n_c, n_t, alpha, r0, p_t)
        assert abs(mine - brute) <= unresolved + 1e-9

    def test_cp_miss_x_c_still_resolves_exactly(self):
        """A x_c whose Clopper-Pearson interval misses the true p_c (no
        witness lower bound available) must still resolve to the EXACT
        brute-force value via exhaustive direct resolution within the true
        x_t window -- correctness does not depend on the witness being
        available, only speed does."""
        n_c, n_t, true_p_c, r0 = 50, 50, 0.02, 1.0
        p_t = r0 * true_p_c
        alpha = 0.05
        beta = binomial_rr.nuisance_beta(alpha)
        x_c_obs = 20  # far in the tail for true_p_c=0.02, n_c=50 (mean=1)
        a, b = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        assert not (a <= true_p_c <= b)  # confirms this IS a CP-miss case
        rp = cbg.resolve_plus_tail(x_c_obs, n_c, n_t, beta, r0, true_p_c, p_t, alpha / 2.0)
        rm = cbg.resolve_minus_tail(x_c_obs, n_c, n_t, beta, r0, true_p_c, p_t, alpha / 2.0)
        assert not rp.cp_hit
        mine, unresolved = cbg.two_sided_covered_mass(rp, rm, n_t, p_t)
        brute = _brute_two_sided_covered_mass(x_c_obs, n_c, n_t, alpha, r0, p_t)
        assert abs(mine - brute) <= unresolved + 1e-9

    def test_reversal_stress_manifold_of_xc_r_combinations(self):
        """A denser sweep (multiple x_c, alpha, r0 combinations at once) --
        a threshold off-by-one that only shows up for a specific band
        shape would plausibly be missed by a single (x_c, r0) case above
        but not by a genuinely varied manifold."""
        n_c, n_t = 25, 25
        alpha = 0.05
        for r0 in (0.5, 1.0, 2.0):
            for x_c_obs in range(0, n_c + 1, 5):
                p_c = 0.3
                p_t = min(1.0, r0 * p_c)
                beta = binomial_rr.nuisance_beta(alpha)
                rp = cbg.resolve_plus_tail(x_c_obs, n_c, n_t, beta, r0, p_c, p_t, alpha / 2.0)
                rm = cbg.resolve_minus_tail(x_c_obs, n_c, n_t, beta, r0, p_c, p_t, alpha / 2.0)
                mine, unresolved = cbg.two_sided_covered_mass(rp, rm, n_t, p_t)
                brute = _brute_two_sided_covered_mass(x_c_obs, n_c, n_t, alpha, r0, p_t)
                assert abs(mine - brute) <= unresolved + 1e-9, (r0, x_c_obs)

    def test_xc_zero_does_not_vacuously_cover_every_lift(self):
        """Regression for a docstring/reasoning error caught in review: the
        x_c=0 confidence set's UPPER endpoint is unbounded, but its LOWER
        endpoint is not automatically 0 -- a large enough x_t excludes
        small/negative candidate true lifts. Checked directly against
        `binomial_rr.confidence_interval`, not merely asserted."""
        n_c, n_t = 50, 50
        alpha = 0.05
        ci = binomial_rr.confidence_interval(0, n_c, 20, n_t, alpha=alpha, alternative="two-sided")
        lo, hi = binomial_rr.to_lift_bounds(ci)
        assert hi is None  # upper endpoint is indeed unbounded
        assert lo > -0.5  # but the lower endpoint excludes true_lift=-0.5 (RR=0.5)


# --- Point-lift conditional bias closed form --------------------------------


class TestConditionalBiasClosedForm:
    def test_matches_full_double_brute_force_enumeration(self):
        n_c, n_t = 15, 12
        p_c, p_t = 0.3, 0.45
        true_lift = p_t / p_c - 1.0
        total = 0.0
        wsum = 0.0
        for xc in range(1, n_c + 1):
            pc_ = float(_binom.pmf(xc, n_c, p_c))
            for xt in range(0, n_t + 1):
                pt_ = float(_binom.pmf(xt, n_t, p_t))
                lift = binomial_rr.point_lift(xc, n_c, xt, n_t)
                assert lift is not None
                total += pc_ * pt_ * lift
                wsum += pc_ * pt_
        brute_bias = total / wsum - true_lift
        closed_bias, p_pos = cbg.conditional_bias_exact(n_c, p_c, p_t, true_lift)
        assert closed_bias == pytest.approx(brute_bias, abs=1e-9)
        assert p_pos == pytest.approx(1.0 - float(_binom.pmf(0, n_c, p_c)))

    def test_rare_event_cell_reports_a_finite_available_probability(self):
        n_c, p_c, p_t = 5000, 1e-4, 1.5e-4  # expected_events=0.5, RR=1.5
        bias, p_pos = cbg.conditional_bias_exact(n_c, p_c, p_t, 0.5)
        assert bias is not None
        assert 0.0 < p_pos < 1.0
        assert math.isfinite(bias)


# --- End-to-end pilot smoke ---------------------------------------------


class TestPilotSubsetRunsEndToEnd:
    """A handful of genuinely tiny manifest cells (not the full `--pilot`
    CLI subset, which is sized for a documented multi-minute standalone
    run) exercised through `calibrate_cell` -- proves the full per-cell
    orchestration (outer window, all metrics, conditional/unconditional
    coverage, bias, truncation budget) produces a structurally valid,
    non-vacuous result, fast enough for the ordinary suite."""

    @pytest.fixture(scope="class")
    @classmethod
    def tiny_cells(cls):
        return [
            c
            for c in cbg.MANIFEST
            if c.p_c == 0.1 and c.expected_events in (0.5, 1) and c.ratio == (1, 1)
        ]

    def test_type_I_error_cell_is_near_or_below_nominal_alpha(self, tiny_cells):
        """A genuine type-I cell (risk_ratio == 1.0): exact rejection
        probability at the true null must sit at/under alpha, since
        `binomial_rr`'s Berger-Boos construction is conservative by
        design (`p_plus`/`p_minus` are certified UPPER bounds on the true
        p-value, so the actual rejection rate is <= nominal, never wildly
        above it) -- not an MC-noise-tolerant band, an exact one-sided
        check against the value this script itself computed exactly."""
        null_cells = [c for c in tiny_cells if c.risk_ratio == 1.0]
        for cell in null_cells:
            result = cbg.calibrate_cell(cell)
            two_sided = result["metrics"]["type_I_two"]["bound"][1]
            assert two_sided <= 0.05 + 1e-6


@pytest.mark.parametrize("risk_ratio", [0.5, 1.0, 2.0])
def test_reported_error_bounds_enclose_full_joint_enumeration(risk_ratio):
    cell = next(
        item
        for item in cbg.MANIFEST
        if item.p_c == 0.1
        and item.expected_events == 0.5
        and item.ratio == (1, 1)
        and item.risk_ratio == risk_ratio
    )
    alpha = 0.05
    beta = binomial_rr.nuisance_beta(alpha)
    totals = dict.fromkeys(
        (
            "two",
            "greater",
            "less",
            "noncoverage",
            "point_noncoverage",
            "interval",
            "point_interval",
        ),
        0.0,
    )
    for xc in range(cell.n_c + 1):
        for xt in range(cell.n_t + 1):
            mass = float(_binom.pmf(xc, cell.n_c, cell.p_c) * _binom.pmf(xt, cell.n_t, cell.p_t))
            plus = binomial_rr.p_plus(1.0, xc, cell.n_c, xt, cell.n_t, beta)
            minus = binomial_rr.p_minus(1.0, xc, cell.n_c, xt, cell.n_t, beta)
            totals["two"] += mass * (min(plus, minus) < alpha / 2.0)
            totals["greater"] += mass * (plus < alpha)
            totals["less"] += mass * (minus < alpha)
            true_plus = binomial_rr.p_plus(risk_ratio, xc, cell.n_c, xt, cell.n_t, beta)
            true_minus = binomial_rr.p_minus(risk_ratio, xc, cell.n_c, xt, cell.n_t, beta)
            miss = min(true_plus, true_minus) < alpha / 2.0
            totals["noncoverage"] += mass * miss
            totals["point_noncoverage"] += mass * miss * (xc > 0)
            interval = binomial_rr.confidence_interval(
                xc, cell.n_c, xt, cell.n_t, alpha=alpha, alternative="two-sided"
            )
            interval_miss = risk_ratio < interval.lower or (
                interval.upper is not None and risk_ratio > interval.upper
            )
            totals["interval"] += mass * interval_miss
            totals["point_interval"] += mass * interval_miss * (xc > 0)

    result = cbg.calibrate_cell(cell)
    metrics = result["metrics"]
    names = {
        "two": "type_I_two" if risk_ratio == 1 else "power_two",
        "greater": "power_greater" if risk_ratio > 1 else "type_I_greater",
        "less": "power_less" if risk_ratio < 1 else "type_I_less",
    }
    for alternative, key in names.items():
        lower, upper = metrics[key]["bound"]
        assert lower - 1e-12 <= totals[alternative] <= upper + 1e-12
    coverage = metrics["true_rr_test_noncoverage"]
    lower, upper = coverage["unconditional"]["bound"]
    assert lower - 1e-12 <= totals["noncoverage"] <= upper + 1e-12
    lower, upper = coverage["conditional_on_point_available"]["bound"]
    point_mass = float(_binom.sf(0, cell.n_c, cell.p_c))
    assert lower - 1e-12 <= totals["point_noncoverage"] / point_mass <= upper + 1e-12
    interval_coverage = metrics["true_rr_noncoverage"]
    lower, upper = interval_coverage["unconditional"]["bound"]
    assert lower <= totals["interval"] <= upper
    lower, upper = interval_coverage["conditional_on_point_available"]["bound"]
    assert lower <= totals["point_interval"] / point_mass <= upper
    assert result["acceptance"]["passed"]


class TestOutwardIntegrationAllowances:
    def test_tail_allowance_includes_production_absolute_error(self):
        for p in (0.0, 1e-300, 1e-20, 0.25):
            assert cbg._inflate_tail(p) >= p + binomial_rr._eps_margin(1)

    def test_below_precision_budget_falls_back_to_full_support(self):
        assert cbg.exact_outer_window(1000, 0.5, 1e-13) == (0, 1000, 0.0)

    @pytest.mark.parametrize("direction", ["plus", "minus"])
    def test_unresolved_points_charge_the_inflated_window(self, monkeypatch, direction):
        monkeypatch.setattr(cbg, f"_witness_lower_{direction}", lambda *args: None)
        n, p = 1000, 0.5
        lo, hi, omitted = cbg.exact_outer_window(n, p, 1e-12)
        assert 0 < lo < hi < n
        # The charged mass must carry the certified outward margin for a
        # two-term tail sum: (2 * 2048 SciPy ULP allowance + 2 terms) * eps.
        assert omitted >= 4098 * math.ulp(1.0)
        resolver = cbg.resolve_plus_tail if direction == "plus" else cbg.resolve_minus_tail
        res = resolver(500, n, n, 1e-6, 1.0, p, p, 0.025)
        assert res.unresolved_mass == omitted

    def test_upper_interval_uses_survival_difference(self):
        from decimal import Decimal, localcontext

        with localcontext() as context:
            context.prec = 80
            p = Decimal.from_float(0.1)
            exact = sum(
                Decimal(math.comb(40, k)) * p**k * (1 - p) ** (40 - k) for k in range(35, 39)
            )
        assert _binom.cdf(38, 40, 0.1) - _binom.cdf(34, 40, 0.1) == 0.0
        assert cbg._interval_mass(35, 38, 40, 0.1) == pytest.approx(float(exact), rel=1e-12, abs=0)

    def test_window_encloses_tails_with_downward_special_function_error(self, monkeypatch):
        cdf, sf = _binom.cdf, _binom.sf
        error = binomial_rr._eps_margin(1) / 4
        monkeypatch.setattr(_binom, "cdf", lambda *a: max(0.0, float(cdf(*a)) - error))
        monkeypatch.setattr(_binom, "sf", lambda *a: max(0.0, float(sf(*a)) - error))
        lo, hi, omitted = cbg.exact_outer_window(1000, 0.5, 1e-12)
        assert omitted >= float(cdf(lo - 1, 1000, 0.5) + sf(hi, 1000, 0.5))


def _cell_at(events, rr=2.0):
    return next(
        c
        for c in cbg.MANIFEST
        if c.p_c == 0.1 and c.ratio == (1, 1) and c.expected_events == events and c.risk_ratio == rr
    )


@pytest.fixture
def passing_grid():
    """Explicit gate inputs, not claimed production calibration results."""
    from dataclasses import asdict

    results = []
    for cell in cbg.MANIFEST:
        power = [0.0, 0.02] if cell.expected_events < 100 else [0.4, 0.5]
        results.append(
            {
                "cell": asdict(cell),
                "acceptance": {"passed": True},
                "production_evidence": {"mean_compact_diameter": 1 / (1 + cell.expected_events)},
                "metrics": {
                    name: {"bound": power.copy()}
                    for name in ("power_two", "power_greater", "power_less")
                },
            }
        )
    return results


class TestUsefulnessAcceptance:
    def test_full_grid_has_all_prespecified_comparisons(self, passing_grid):
        gate = cbg.grid_acceptance(passing_grid)
        assert gate["passed"]
        assert len(gate["comparisons"]) == 48
        assert sum(len(row["power"]) for row in gate["comparisons"]) == 72
        assert gate["power_null_benchmark"] == pytest.approx(0.055)

    @pytest.mark.parametrize(
        "fault", ["no_power", "no_contraction", "missing_power", "unreportable"]
    )
    def test_each_usefulness_requirement_is_enforced(self, passing_grid, fault):
        for row in passing_grid:
            if fault == "no_power":
                for metric in row["metrics"].values():
                    metric["bound"] = [0.0, 0.0]
            elif fault == "no_contraction":
                row["production_evidence"]["mean_compact_diameter"] = 1.0
            elif fault == "missing_power":
                row["metrics"] = {}
            else:
                row["acceptance"]["passed"] = False
        assert not cbg.grid_acceptance(passing_grid)["passed"]

    def test_overlapping_power_bounds_do_not_prove_improvement(self, passing_grid):
        for row in passing_grid:
            for metric in row["metrics"].values():
                metric["bound"] = [0.1, 0.5]
        assert not cbg.grid_acceptance(passing_grid)["passed"]

    @pytest.mark.parametrize("fault", ["empty", "subset", "duplicate"])
    def test_incomplete_manifest_cannot_pass(self, passing_grid, fault):
        rows = [] if fault == "empty" else passing_grid[:-1]
        if fault == "duplicate":
            rows.append(rows[0])
        assert not cbg.grid_acceptance(rows)["passed"]

    def test_actual_intervals_contract_and_report_availability(self):
        # 0.5 -> 30 -> 100 expected events: finer grids must contract further.
        evidence = [cbg.production_evidence(_cell_at(level), cbg.ALPHA) for level in (0.5, 30, 100)]
        assert all(row["reportability_passed"] for row in evidence)
        widths = [row["mean_compact_diameter"] for row in evidence]
        assert widths[0] > widths[1] > widths[2]
        for record in evidence:
            rows = record["rows"]
            assert any(not row["point_available"] and row["set_available"] for row in rows)
            assert any(row["point_available"] and row["set_available"] for row in rows)

    @pytest.mark.parametrize("fault", ["full_space", "null_set", "null_point"])
    def test_actual_output_mutations_fail_reportability(self, monkeypatch, fault):
        if fault == "full_space":
            monkeypatch.setattr(
                binomial_rr,
                "confidence_interval",
                lambda *a, **kw: binomial_rr.BinomialInterval(0.0, None, "central", 1.0),
            )
        elif fault == "null_set":
            monkeypatch.setattr(binomial_rr, "confidence_interval", lambda *a, **kw: None)
        else:
            monkeypatch.setattr(binomial_rr, "point_lift", lambda *a, **kw: None)
        result = cbg.calibrate_cell(_cell_at(0.5), interval_only=True)
        assert result["acceptance"]["error_passed"]
        assert not result["acceptance"]["reportability_passed"]
        assert not result["acceptance"]["passed"]

    @pytest.mark.parametrize("fault", [None, "missing_power", "no_contraction", "unreportable"])
    def test_cli_exit_status_includes_grid_acceptance(
        self, passing_grid, monkeypatch, tmp_path, fault
    ):
        import json
        import sys

        for row in passing_grid:
            if fault == "missing_power":
                row["metrics"] = {}
            elif fault == "no_contraction":
                row["production_evidence"]["mean_compact_diameter"] = 1.0
            elif fault == "unreportable":
                row["acceptance"]["passed"] = False
        output = tmp_path / "calibration.json"
        argv = ["binomial_grid", "--output", str(output)]
        if fault == "missing_power":
            argv.append("--interval-only")
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(cbg, "run_grid", lambda *a, **kw: passing_grid)
        assert cbg.main() == (0 if fault is None else 1)
        payload = json.loads(output.read_text())
        assert payload["acceptance"]["passed"] == (fault is None)
