"""Correctness and calibration of the exact binomial rare-event method.

Covers ``binomial_rr.py`` and its public ``estimate_lift`` dispatch for
unadjusted, unit-grain conversion/retention arm pairs: zero-cell geometry,
point-unavailable confidence sets, directional and shifted-null tests,
request refusals, and all 336 cells of the explicit truth/support manifest.
Selected raw binary fixtures also exercise native, frame and artifact paths.

``test_inference_calibration.py`` separately checks ``infer_lift`` and
``MeanVarianceModel`` through its moderate-case ``_binomial_lift_interval``
helper, bypassing the exact-binomial ``estimate_lift`` dispatch tested here.

Two independent oracles are used, neither of which calls into
``binomial_rr.py``'s own machinery:

* ``_oracle_p_plus``/``_oracle_p_minus``: full joint-binomial enumeration
  with a coarse grid-maximized nuisance -- a valid LOWER bound on the true
  supremum, hence a lower bound on the certified p-value, never an equality
  oracle (see ``binomial_rr.py``'s own docstring on why a grid maximum
  cannot itself serve as a coverage-valid p-value).
* ``bonferroni_lift_interval``: a genuinely independent joint confidence
  RECTANGLE for ``(p_c, p_t)`` built from per-arm exact Clopper-Pearson
  quantiles (``scipy.stats.beta`` directly, never ``binomial_rr.
  clopper_pearson``) combined via a Bonferroni union bound and mapped
  through the monotone ratio ``R = p_t / p_c``. This rectangle is independent
  of the production formula, not that formula reused as its own oracle.

Manifest single-draw production calls cost roughly 1-2 seconds each on
ordinary cells, dominated by the certified branch-and-bound sup's own
search rather than population size (measured on the machine this suite
was authored on, 2026-09-16: a full single-draw pass over ALL 336 grid
cells -- including the largest, n_c=1,000,000/n_t=4,000,000 -- took
321s total, mean 0.96s/cell, max 5.11s/cell). Every one of the 336
cells sits at or below ``binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE`` (the one enforced
production arm-size ceiling; see ``test_manifest_is_the_full_336_cell_
grid_with_declared_truth_and_support_status``), so every cell is
admitted and receives production and independent-reference geometry checks
from a genuine draw. These single draws check support, not coverage;
coverage is evaluated separately by the prespecified calibration tests.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta as _beta
from scipy.stats import binom as _binom

from increment.analysis import Analysis
from increment.errors import InvalidRequestError
from increment.estimation import binomial_rr
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import estimate_lift
from increment.estimation.inference import Normal
from increment.estimation.sequential import AlwaysValid
from increment.query.session import WarehouseArtifactStore
from increment.semantics.models import ConversionMetric
from increment.simulate.dgp import binomial_fixture_units, simulate_binomial_conversion_logs
from tests.mc import binomial_error_upper_bound, coverage_lower_bound, family_eta, scientific_delta

# --- Shared fixtures -------------------------------------------------------


def _conversion_metric(name: str = "conv") -> ConversionMetric:
    return ConversionMetric(name=name, entity="user", fact=name)


def _summary_df(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    """(group_id, n, successes) triples -> a group_summary DataFrame."""
    out = []
    for group_id, n, successes in rows:
        arm = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id=group_id,
            n=n,
            successes=successes,
            sum_y=float(successes),
            sum_y2=float(successes),
        )
        out.append(
            {
                "experiment_id": arm.study_id,
                "metric": arm.metric,
                "group_id": arm.group_id,
                "n": arm.n,
                "successes": arm.successes,
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
            }
        )
    return pd.DataFrame(out)


def _estimate(x_c: int, n_c: int, x_t: int, n_t: int, **kwargs):
    df = _summary_df([("control", n_c, x_c), ("treatment", n_t, x_t)])
    computation = estimate_lift(
        metrics=[_conversion_metric()], summary=df, control_group="control", **kwargs
    )
    assert computation.failures == {}, computation.failures
    assert len(computation.results) == 1
    return computation.results[0]


def _set_bounds(result) -> tuple[float, float]:
    """This row's confidence-SET bounds (always present for a binomial row,
    point-backed or not), upper coerced to +inf for a hit test."""
    bset = result.binomial_set
    assert bset is not None, "every reference_kind='binomial' row carries a set"
    return bset.lower, (math.inf if bset.upper is None else bset.upper)


# --- Independent oracle #1: full joint-binomial enumeration with a coarse --
# --- grid-maximized nuisance (a LOWER bound on the true sup, hence only a --
# --- rough independent sanity check, never treated as an equality oracle) -


def _oracle_p_plus(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    """Independent re-derivation of p_+(r): full enumeration over X_c, X_t,
    nuisance-maximized over a fine grid of q in the Clopper-Pearson
    interval. A grid maximum understates the true supremum, so this is a
    valid LOWER bound on the production p-value -- an oracle for "not too
    small", not for exact equality (see binomial_rr.py's own docstring on
    why a grid maximum cannot serve as a coverage-valid p-value itself).
    """
    a, b = binomial_rr.clopper_pearson(x_c, n_c, beta)
    k = n_c * x_t - n_t * x_c
    qs = np.linspace(a, b, 4001) if b > a else np.array([a])
    best = 0.0
    for q in qs:
        p = min(r * q, 1.0) if r > 0 else 0.0
        i = np.arange(n_c + 1)
        pmf_i = _binom.pmf(i, n_c, q)
        thresh = np.ceil((k + n_t * i) / n_c).astype(np.int64) - 1
        sf = _binom.sf(thresh, n_t, p)
        best = max(best, float(np.dot(pmf_i, sf)))
    return min(1.0, beta + best)


def _oracle_p_minus(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    """Grid lower bound over the feasible null ``p >= r*q``."""
    a, b = binomial_rr.clopper_pearson(x_c, n_c, beta)
    upper = b if r == 0.0 else min(b, 1.0 / r)
    if upper < a:
        return beta
    k = n_c * x_t - n_t * x_c
    qs = np.linspace(a, upper, 4001) if upper > a else np.array([a])
    best = 0.0
    for q in qs:
        p = r * q
        i = np.arange(n_c + 1)
        pmf_i = _binom.pmf(i, n_c, q)
        thresh = np.floor((k + n_t * i) / n_c).astype(np.int64)
        cdf = _binom.cdf(thresh, n_t, p)
        best = max(best, float(np.dot(pmf_i, cdf)))
    return min(1.0, beta + best)


@pytest.mark.parametrize(
    ("x_c", "n_c", "x_t", "n_t", "r"),
    [(3, 40, 8, 40, 1.0), (3, 40, 8, 40, 2.0), (1, 20, 1, 20, 1.0), (5, 30, 1, 30, 0.5)],
)
def test_p_plus_matches_independent_grid_oracle_as_a_lower_bound(x_c, n_c, x_t, n_t, r):
    """The certified branch-and-bound p_+ must be >= a coarse grid-maximized
    oracle (the grid is only a lower bound on the true supremum) and close
    to it (the branch-and-bound is far more refined, so it should not
    overshoot by more than a small numerical margin at this table size).
    """
    beta = binomial_rr.nuisance_beta(0.05)
    oracle = _oracle_p_plus(r, x_c, n_c, x_t, n_t, beta)
    production = binomial_rr.p_plus(r, x_c, n_c, x_t, n_t, beta)
    assert production >= oracle - 1e-9
    assert production <= oracle + 5e-3


@pytest.mark.parametrize(
    ("x_c", "n_c", "x_t", "n_t", "r"),
    [
        (3, 40, 8, 40, 1.0),
        (3, 40, 8, 40, 0.5),
        (1, 20, 1, 20, 1.0),
        (5, 30, 1, 30, 2.0),
        (5, 30, 30, 30, 2.0),
        (30, 30, 30, 30, 2.0),
    ],
)
def test_p_minus_matches_independent_grid_oracle_as_a_lower_bound(x_c, n_c, x_t, n_t, r):
    """The mirror-image check of the plus-tail test above, for p_-."""
    beta = binomial_rr.nuisance_beta(0.05)
    oracle = _oracle_p_minus(r, x_c, n_c, x_t, n_t, beta)
    production = binomial_rr.p_minus(r, x_c, n_c, x_t, n_t, beta)
    assert production >= oracle - 1e-9
    assert production <= oracle + 5e-3


@pytest.mark.slow
def test_rare_event_large_arm_matches_full_support_grid_oracle_in_both_tails():
    """The same lower-bound gate at ``x_c = 10`` in 100,000 control units, where production
    scans only a Chernoff window of the control support: a 2001-point grid over the
    Clopper-Pearson interval that enumerates EVERY control count (counts with zero float
    mass add nothing) is at most the certified p-value and within the small-table tolerance."""
    x_c, n_c, n_t = 10, 100_000, 100_000
    beta = binomial_rr.nuisance_beta(0.05)
    a, b = binomial_rr.clopper_pearson(x_c, n_c, beta)
    cells = [(x_t, r) for x_t in (10, 20) for r in (1.0, 2.0)]
    assert b <= 1.0 / max(r for _, r in cells)  # the feasible nuisance domain is all of [a, b]
    best_plus = dict.fromkeys(cells, 0.0)
    best_minus = dict.fromkeys(cells, 0.0)
    counts = np.arange(n_c + 1)
    for q in np.linspace(a, b, 2001):
        full_pmf = _binom.pmf(counts, n_c, q)
        reachable = np.flatnonzero(full_pmf > 0.0)
        pmf = full_pmf[reachable]
        for x_t, r in cells:
            k = n_c * x_t - n_t * x_c
            plus = np.ceil((k + n_t * reachable) / n_c).astype(np.int64) - 1
            minus = np.floor((k + n_t * reachable) / n_c).astype(np.int64)
            sf = _binom.sf(plus, n_t, min(r * q, 1.0))
            cdf = _binom.cdf(minus, n_t, r * q)
            best_plus[x_t, r] = max(best_plus[x_t, r], float(np.dot(pmf, sf)))
            best_minus[x_t, r] = max(best_minus[x_t, r], float(np.dot(pmf, cdf)))
    for x_t, r in cells:
        for production, best in (
            (binomial_rr.p_plus(r, x_c, n_c, x_t, n_t, beta), best_plus[x_t, r]),
            (binomial_rr.p_minus(r, x_c, n_c, x_t, n_t, beta), best_minus[x_t, r]),
        ):
            oracle = min(1.0, beta + best)
            assert production >= oracle - 1e-9, (x_t, r)
            assert production <= oracle + 5e-3, (x_t, r)


# --- Independent oracle #2: Bonferroni joint Clopper-Pearson rectangle -----


def _cp_arm_interval(x: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact two-sided ``1 - alpha`` Clopper-Pearson interval for a single
    binomial rate, computed directly from ``scipy.stats.beta`` quantiles.
    Deliberately NOT ``binomial_rr.clopper_pearson``: this oracle must stay
    independent of the module under test's own code path, even though the
    underlying algebra (beta-quantile CP endpoints) is the textbook
    formula either implementation would use.
    """
    lo = 0.0 if x == 0 else float(_beta.ppf(alpha / 2.0, x, n - x + 1))
    hi = 1.0 if x == n else float(_beta.isf(alpha / 2.0, x + 1, n - x))
    return lo, hi


def bonferroni_lift_interval(
    x_c: int, n_c: int, x_t: int, n_t: int, alpha: float
) -> tuple[float, float | None]:
    """A conservative, fully independent ``1 - alpha`` confidence interval
    for the relative lift ``R - 1``, built from per-arm exact
    Clopper-Pearson rates combined by a Bonferroni union bound.

    Each arm's CP interval is built at level ``1 - alpha/2``, so
    ``P(p_c not in I_c) <= alpha/2`` and ``P(p_t not in I_t) <= alpha/2``;
    by the union bound, ``P(p_c not in I_c OR p_t not in I_t) <= alpha``,
    so with probability ``>= 1 - alpha`` BOTH true rates lie in their
    respective intervals. The ratio ``R = p_t / p_c`` is monotone
    increasing in ``p_t`` and decreasing in ``p_c``, so whenever both true
    rates lie in their intervals, ``R`` lies in
    ``[lo(p_t)/hi(p_c), hi(p_t)/lo(p_c)]`` -- a valid, if typically much
    wider, independent reference interval. Never calls into
    ``binomial_rr.py``.
    """
    a_c, b_c = _cp_arm_interval(x_c, n_c, alpha / 2.0)
    a_t, b_t = _cp_arm_interval(x_t, n_t, alpha / 2.0)
    r_lo = 0.0 if b_c <= 0.0 else a_t / b_c
    r_hi = None if a_c <= 0.0 else b_t / a_c
    return r_lo - 1.0, (None if r_hi is None else r_hi - 1.0)


# --- Zero-cell geometry: the table binomial_rr.py's module docstring names -


class TestZeroCellGeometry:
    def test_both_arms_positive_yields_finite_point_and_finite_interval(self):
        result = _estimate(3, 100, 8, 100)
        assert result.reference_kind == "binomial"
        assert result.lift is not None
        assert result.require_lift().value == pytest.approx((100 * 8) / (100 * 3) - 1.0)
        assert result.require_lift().lb is not None and result.require_lift().ub is not None
        assert result.require_lift().lb < result.require_lift().value < result.require_lift().ub

    def test_zero_treatment_yields_exact_negative_one_point_finite_upper(self):
        result = _estimate(5, 100, 0, 100)
        assert result.lift is not None
        assert result.require_lift().value == -1.0
        # The certified lower bound is outward-rounded for numerical safety,
        # so it may sit a hair below the natural floor -1.0, never above it.
        assert result.require_lift().lb is not None
        assert result.require_lift().lb <= -1.0
        assert result.require_lift().lb == pytest.approx(-1.0, abs=1e-6)
        assert result.require_lift().ub is not None and math.isfinite(result.require_lift().ub)

    def test_zero_control_yields_no_point_but_a_finite_lower_unbounded_upper_set(self):
        result = _estimate(0, 100, 5, 100)
        assert result.lift is None
        assert result.binomial_set is not None
        assert not result.binomial_set.point_available
        assert result.binomial_set.upper is None
        assert result.binomial_set.lower > -1.0
        # Set-only rows still answer stat_sig/p_value exactly, never a failure.
        assert 0.0 <= result.p_value() <= 1.0

    def test_both_zero_yields_full_support_set_not_a_failure(self):
        result = _estimate(0, 100, 0, 100)
        assert result.lift is None
        assert result.binomial_set is not None
        # Outward-rounded for numerical safety: at or a hair below -1.0, never above.
        assert result.binomial_set.lower <= -1.0
        assert result.binomial_set.lower == pytest.approx(-1.0, abs=1e-6)
        assert result.binomial_set.upper is None
        assert result.p_value() == 1.0
        assert result.stat_sig() is False

    def test_both_arms_all_success_yields_informative_finite_bounds(self):
        result = _estimate(100, 100, 100, 100)
        assert result.lift is not None
        assert result.require_lift().value == 0.0
        assert result.require_lift().lb is not None and result.require_lift().ub is not None
        # Informative: nowhere near the full [-1, inf) support.
        assert result.require_lift().lb > -0.5
        assert result.require_lift().ub < 1.0

    def test_ordinary_case_is_not_the_full_parameter_space(self):
        """A returned set must never be the trivial full-space-only answer."""
        result = _estimate(20, 200, 25, 200)
        assert result.lift is not None
        assert not (result.require_lift().lb == -1.0 and result.require_lift().ub is None)
        assert result.require_lift().ub is not None and result.require_lift().ub < 100.0


# --- Directional and shifted-null behavior ----------------------------------


class TestDirectionalAndShiftedNull:
    def test_greater_alternative_is_open_upper_with_finite_lower(self):
        result = _estimate(3, 100, 8, 100, alternative="greater")
        assert result.lift is not None
        assert result.require_lift().open_side == "upper"
        assert result.require_lift().ub is None
        assert result.require_lift().lb is not None

    def test_less_alternative_is_closed_with_lower_at_natural_floor(self):
        result = _estimate(3, 100, 8, 100, alternative="less")
        assert result.lift is not None
        assert result.require_lift().open_side is None
        assert result.require_lift().lb == -1.0
        assert result.require_lift().ub is not None

    def test_self_null_shifted_p_value_is_large(self):
        """Testing the true generating ratio itself must not look
        significant: R = (n_c*x_t)/(n_t*x_c) tested as null_lift gives a
        p-value at (or very near) 1."""
        x_c, n_c, x_t, n_t = 3, 100, 8, 100
        true_r = (n_c * x_t) / (n_t * x_c)
        result = _estimate(x_c, n_c, x_t, n_t, null_lift=true_r - 1.0)
        assert result.p_value() > 0.9

    def test_extreme_shifted_null_is_rejected(self):
        result = _estimate(3, 100, 8, 100, null_lift=999.0)
        assert result.p_value() < 0.01
        assert result.stat_sig() is True

    def test_p_value_and_stat_sig_agree_at_the_declared_alpha(self):
        for x_c, n_c, x_t, n_t in [(3, 100, 8, 100), (0, 100, 5, 100), (0, 100, 0, 100)]:
            result = _estimate(x_c, n_c, x_t, n_t)
            assert result.stat_sig() == (result.p_value() < 0.05)


# --- Numerical stability / serialization ------------------------------------


class TestNumericalStability:
    def test_json_serialization_has_no_nan_or_infinity(self):
        for x_c, n_c, x_t, n_t in [(0, 100, 5, 100), (0, 100, 0, 100), (100, 100, 100, 100)]:
            result = _estimate(x_c, n_c, x_t, n_t)
            payload = result.model_dump_json()
            assert "NaN" not in payload
            assert "Infinity" not in payload

    def test_n_equals_one_per_arm_is_admissible(self):
        result = _estimate(1, 1, 1, 1)
        assert result.lift is not None
        assert math.isfinite(result.require_lift().value)
        assert result.require_lift().ub is not None and math.isfinite(result.require_lift().ub)

    def test_moderate_case_unchanged_p_c_15_n_1000_r_110(self):
        """The existing moderate p_c=.15, n=1000, R=1.10 case (~150/165
        events/arm) as a single deterministic point, through the new
        method: a tight, sensible interval around the true 0.10 lift."""
        result = _estimate(150, 1000, 165, 1000)
        assert result.lift is not None
        assert result.require_lift().value == pytest.approx(0.10, abs=1e-9)
        assert result.require_lift().lb is not None and result.require_lift().ub is not None
        assert -0.3 < result.require_lift().lb < 0.10 < result.require_lift().ub < 0.6

    def test_endpoint_resolution_not_reached_is_disclosed_on_the_row(self, monkeypatch):
        """A search that cannot reach its stop still reports conservative bounds and says so
        on the row; a resolved row carries no such note. A stop finer than float64 can
        resolve forces the case through the public path."""
        resolved = _estimate(3, 100, 8, 100)
        assert resolved.note is None
        binomial_rr._confidence_interval_cached.cache_clear()
        monkeypatch.setattr(binomial_rr, "_ENDPOINT_RESOLUTION", 0.0)
        monkeypatch.setattr(binomial_rr, "_ENDPOINT_FLOOR", 1e-30)
        try:
            row = _estimate(3, 100, 8, 100)
        finally:
            binomial_rr._confidence_interval_cached.cache_clear()
        assert row.note is not None and row.note.startswith(binomial_rr.PRECISION_NOTE_PREFIX)
        assert (row.stat_sig(), row.p_value()) == (resolved.stat_sig(), resolved.p_value())
        bset, expected = row.binomial_set, resolved.binomial_set
        assert bset is not None and expected is not None and expected.upper is not None
        assert bset.upper is not None
        assert 1.0 + bset.lower == pytest.approx(1.0 + expected.lower, rel=1e-3)
        assert 1.0 + bset.upper == pytest.approx(1.0 + expected.upper, rel=1e-3)

    def test_nuisance_cap_exhaustion_is_disclosed_on_the_row(self, monkeypatch):
        """A nuisance search that runs out of iterations near the decision still reports valid
        bounds, says how loose they may be, and only widens the set; the default rule is silent."""
        resolved = _estimate(3, 100, 8, 100)
        assert resolved.note is None
        monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", binomial_rr._StopRule(2.0**-14, 3))
        row = _estimate(3, 100, 8, 100)
        assert row.note is not None and row.note.startswith(binomial_rr.NUISANCE_NOTE_PREFIX)
        assert resolved.binomial_set is not None and row.binomial_set is not None
        assert row.binomial_set.lower <= resolved.binomial_set.lower
        assert row.binomial_set.upper is not None and resolved.binomial_set.upper is not None
        assert row.binomial_set.upper >= resolved.binomial_set.upper


# --- Request-level refusals --------------------------------------------------


class TestRefusals:
    @pytest.mark.slow
    def test_sequential_binary_counts_use_the_actual_beta_likelihood(self):
        from increment import estimate_sequential
        from tests.sequential_cases import (
            capture,
            records,
            registration,
        )

        reg = registration()
        snapshot = capture(reg, records([1] * 150 + [0] * 850, [1] * 165 + [0] * 835))
        result = estimate_sequential(snapshot, AlwaysValid(registration=reg)).results[0]
        assert result.binomial_set is None
        assert result.reference_kind == "sequential"
        assert result.require_lift().value == pytest.approx(0.10)
        assert result.require_sequential_result().checkpoint.control.successes == 150
        assert result.require_sequential_result().checkpoint.treatment.successes == 165

    def test_informative_prior_uses_the_established_fallback(self):
        result = _estimate(150, 1000, 165, 1000, prior=Normal(mu=0.1, sigma=0.5))
        assert result.binomial_set is None
        assert result.reference_kind == "normal"
        assert result.prior_shrunk is True

    def test_binary_provenance_required_for_non_conversion_metric_type(self):
        arm = ArmStats.from_raw_sums(
            study_id="e", metric="rev", group_id="control", n=10, sum_y=5.0, sum_y2=5.0
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            from increment.estimation.armstats import binary_counts

            binary_counts(arm, "mean")
        assert exc_info.value.code == "estimation.binomial.binary_provenance_required"

    def test_independent_units_required_for_cuped_family_arm(self):
        from increment.estimation.armstats import binary_counts

        arm = ArmStats(
            study_id="e",
            metric="conv",
            group_id="control",
            n=10,
            ref_y=0.5,
            cy1=0.0,
            cy2=2.5,
            ref_x=1.0,
            cx1=0.0,
            cx2=1.0,
            cxy=0.0,
            x_role="covariate",
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            binary_counts(arm, "conversion")
        assert exc_info.value.code == "estimation.binomial.independent_units_required"


# --- Grid manifest: all 336 truth/support cells, no subsetting --------------
# p_c {1e-4, 1e-3, .01, .1} x expected control events {.5, 1, 2, 5, 10, 30, 100}
# x allocation c:t {1:1, 1:4, 4:1} x risk ratio {.5, 1, 1.5, 2} = 4*7*3*4 = 336.


@dataclass(frozen=True)
class ManifestCell:
    p_c: float
    expected_events: float
    ratio: tuple[int, int]
    risk_ratio: float
    n_c: int
    n_t: int
    p_t: float
    true_lift: float


_P_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
_EXPECTED_EVENTS_GRID = (0.5, 1, 2, 5, 10, 30, 100)
_RATIO_GRID = ((1, 1), (1, 4), (4, 1))
_RISK_RATIO_GRID = (0.5, 1.0, 1.5, 2.0)

# `binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE` is the one enforced production applicability
# boundary for this method (checked below against the manifest's largest
# cell); no separate test-runtime boundary exists or is needed here -- see
# the module docstring for the measured real full-grid single-draw cost.


def _round_counts(p_c: float, expected_events: float, ratio: tuple[int, int]) -> tuple[int, int]:
    """Explicit-rounding formula of the prespecified grid: n_c from the
    expected event target, n_t from the declared allocation ratio.
    """
    c, t = ratio
    n_c = max(1, math.floor(expected_events / p_c + 0.5))
    n_t = max(1, math.floor(n_c * t / c + 0.5))
    return n_c, n_t


def _build_manifest() -> tuple[ManifestCell, ...]:
    cells: list[ManifestCell] = []
    for p_c in _P_C_GRID:
        for expected_events in _EXPECTED_EVENTS_GRID:
            for ratio in _RATIO_GRID:
                for risk_ratio in _RISK_RATIO_GRID:
                    n_c, n_t = _round_counts(p_c, expected_events, ratio)
                    p_t = min(1.0, risk_ratio * p_c)
                    cells.append(
                        ManifestCell(
                            p_c=p_c,
                            expected_events=expected_events,
                            ratio=ratio,
                            risk_ratio=risk_ratio,
                            n_c=n_c,
                            n_t=n_t,
                            p_t=p_t,
                            true_lift=risk_ratio - 1.0,
                        )
                    )
    return tuple(cells)


MANIFEST: tuple[ManifestCell, ...] = _build_manifest()


def test_manifest_is_the_full_336_cell_grid_with_declared_truth_and_support_status():
    """The complete, unsubsetted prespecified grid -- every cell
    carries its declared truth (n_c, n_t, p_t, true_lift) and is admitted
    under production's actual, enforced `binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE` ceiling;
    nothing is silently dropped and no admitted cell is mislabeled
    infeasible.
    """
    assert (
        len(_P_C_GRID) * len(_EXPECTED_EVENTS_GRID) * len(_RATIO_GRID) * len(_RISK_RATIO_GRID)
        == 336
    )
    assert len(MANIFEST) == 336
    assert len({(c.p_c, c.expected_events, c.ratio, c.risk_ratio) for c in MANIFEST}) == 336
    for cell in MANIFEST:
        assert cell.n_c >= 1 and cell.n_t >= 1
        assert 0.0 <= cell.p_t <= 1.0
        assert cell.true_lift == pytest.approx(cell.risk_ratio - 1.0)
    # The enforced arm-size ceiling admits all 336 cells, including the largest
    # (p_c=1e-4, expected_events=100, ratio=(1, 4): n_c=1,000,000,
    # n_t=4,000,000), so no cell here is infeasible.
    largest = max(max(c.n_c, c.n_t) for c in MANIFEST)
    assert largest == 4_000_000
    assert binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE >= largest


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestManifestExecution:
    """One production draw per cell checks support and geometry, not coverage."""

    @pytest.fixture(scope="class")
    @classmethod
    def draws(cls):
        rng = np.random.default_rng(20260916)
        return tuple(
            (
                cell,
                int(rng.binomial(cell.n_c, cell.p_c)),
                int(rng.binomial(cell.n_t, cell.p_t)),
            )
            for cell in MANIFEST
        )

    def test_point_availability_is_a_genuine_nonempty_nontotal_slice(self, draws):
        # Per-cell checks establish that these strata match returned availability.
        assert 0 < sum(x_c > 0 for _, x_c, _ in draws) < len(draws) == 336

    @pytest.mark.parametrize("cell_index", range(len(MANIFEST)))
    def test_each_manifest_draw_has_valid_production_and_oracle_geometry(self, cell_index, draws):
        cell, x_c, x_t = draws[cell_index]
        result = _estimate(x_c, cell.n_c, x_t, cell.n_t)
        bset = result.binomial_set
        assert bset is not None
        lo, hi = _set_bounds(result)
        assert math.isfinite(lo) and not math.isnan(hi) and lo <= hi
        assert (result.lift is not None) == (x_c > 0)
        if x_c > 0:
            assert result.require_lift().value == pytest.approx(
                (cell.n_c * x_t) / (cell.n_t * x_c) - 1.0
            )
        bon_lo, bon_hi = bonferroni_lift_interval(x_c, cell.n_c, x_t, cell.n_t, 0.05)
        assert (bset.upper is None) == (x_c == 0) == (bon_hi is None)
        if cell_index == 6:
            # A fixed finite-control draw witnesses overlap, not width dominance.
            assert bon_hi is not None and math.isfinite(hi)
            assert lo <= bon_hi and bon_lo <= hi


# --- Small replicated-MC calibration: type-I, power, bias, narrowing -------
# Unlike TestManifestExecution's one draw per cell, replication repeats
# production calls per cell, so this gross-miscalibration smoke gate uses three
# manifest cells and modest counts, not a full certification budget.


def _manifest_cell(
    p_c: float, expected_events: float, ratio: tuple[int, int], risk_ratio: float
) -> ManifestCell:
    return next(
        c
        for c in MANIFEST
        if c.p_c == p_c
        and c.expected_events == expected_events
        and c.ratio == ratio
        and c.risk_ratio == risk_ratio
    )


_COVERAGE_CELLS = (
    _manifest_cell(1e-4, 0.5, (1, 4), 0.5),
    _manifest_cell(1e-4, 0.5, (4, 1), 2.0),
    _manifest_cell(1e-3, 1, (1, 1), 1.0),
    _manifest_cell(1e-2, 2, (1, 4), 2.0),
    _manifest_cell(0.1, 0.5, (4, 1), 0.5),
    _manifest_cell(0.1, 2, (1, 1), 1.0),
)
_COVERAGE_REPS = 82_000
_COVERAGE_K_CRIT = 4_305
_COVERAGE_FAMILY_ALPHA = 0.01
_COVERAGE_NOMINAL_ERROR = 0.05


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPrespecifiedCellCoverage:
    """Pointwise coverage at prespecified rare/allocation corner cells.

    The replication count was chosen prospectively from the
    scientific-tolerance policy's exact one-sided binomial rule over this
    frozen six-cell family. Count-pair caching changes runtime only; every
    simulated replication remains in its cell's unconditional denominator,
    including ``x_c=0``.
    """

    def test_prospective_replication_design_meets_the_mc_budget(self):
        eta = family_eta(_COVERAGE_FAMILY_ALPHA, len(_COVERAGE_CELLS))
        delta = scientific_delta(_COVERAGE_NOMINAL_ERROR)
        upper = binomial_error_upper_bound(_COVERAGE_K_CRIT, _COVERAGE_REPS, eta)
        assert upper <= _COVERAGE_NOMINAL_ERROR + delta
        assert upper - _COVERAGE_K_CRIT / _COVERAGE_REPS <= delta / 2.0
        assert float(_binom.sf(_COVERAGE_K_CRIT, _COVERAGE_REPS, _COVERAGE_NOMINAL_ERROR)) <= eta

    @pytest.mark.parametrize(("cell_index", "cell"), tuple(enumerate(_COVERAGE_CELLS)))
    def test_pointwise_unconditional_coverage(self, cell_index, cell):
        rng = np.random.default_rng(2026091600 + cell_index)
        hit_by_counts: dict[tuple[int, int], bool] = {}
        misses = 0
        for _ in range(_COVERAGE_REPS):
            x_c = int(rng.binomial(cell.n_c, cell.p_c))
            x_t = int(rng.binomial(cell.n_t, cell.p_t))
            key = (x_c, x_t)
            hit = hit_by_counts.get(key)
            if hit is None:
                result = _estimate(x_c, cell.n_c, x_t, cell.n_t)
                lo, hi = _set_bounds(result)
                hit = lo <= cell.true_lift <= hi
                hit_by_counts[key] = hit
            misses += not hit

        eta = family_eta(_COVERAGE_FAMILY_ALPHA, len(_COVERAGE_CELLS))
        delta = scientific_delta(_COVERAGE_NOMINAL_ERROR)
        upper = binomial_error_upper_bound(misses, _COVERAGE_REPS, eta)
        observed = misses / _COVERAGE_REPS
        assert upper <= _COVERAGE_NOMINAL_ERROR + delta, (
            f"cell={cell}: undercoverage {misses}/{_COVERAGE_REPS}, {upper=}"
        )
        assert upper - observed <= delta / 2.0, (
            f"cell={cell}: Monte Carlo margin {upper - observed} exceeds {delta / 2.0}"
        )


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestReplicatedCalibration:
    REPS = 50

    def test_type_I_error_at_the_true_null(self):
        cell = _manifest_cell(0.1, 5, (1, 1), 1.0)
        rng = np.random.default_rng(1)
        rejections = 0
        for _ in range(self.REPS):
            x_c = int(rng.binomial(cell.n_c, cell.p_c))
            x_t = int(rng.binomial(cell.n_t, cell.p_t))
            if _estimate(x_c, cell.n_c, x_t, cell.n_t).stat_sig():
                rejections += 1
        from tests.mc import binomial_error_upper_bound

        upper = binomial_error_upper_bound(rejections, self.REPS, eta=0.05)
        assert upper <= 0.30, f"type-I rate {rejections}/{self.REPS} too high ({upper=})"

    def test_power_for_a_clearly_separated_alternative(self):
        cell = _manifest_cell(0.1, 30, (1, 1), 2.0)
        rng = np.random.default_rng(2)
        rejections = 0
        for _ in range(self.REPS):
            x_c = int(rng.binomial(cell.n_c, cell.p_c))
            x_t = int(rng.binomial(cell.n_t, cell.p_t))
            if _estimate(x_c, cell.n_c, x_t, cell.n_t).stat_sig():
                rejections += 1
        lower = coverage_lower_bound(rejections, self.REPS, eta=0.05)
        assert lower >= 0.25, (
            f"power {rejections}/{self.REPS} too low for a 2x true lift ({lower=})"
        )

    def test_finite_point_bias_is_bounded_by_its_own_dispersion(self):
        cell = _manifest_cell(0.1, 5, (1, 1), 1.5)
        rng = np.random.default_rng(3)
        errors = []
        for _ in range(self.REPS):
            x_c = int(rng.binomial(cell.n_c, cell.p_c))
            x_t = int(rng.binomial(cell.n_t, cell.p_t))
            result = _estimate(x_c, cell.n_c, x_t, cell.n_t)
            if result.lift is not None:
                errors.append(result.require_lift().value - cell.true_lift)
        assert len(errors) >= self.REPS // 2, "too few point-backed replications to assess bias"
        arr = np.asarray(errors)
        bias = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
        # A generous, data-derived (not magic-constant) bound: mean error
        # within several of its own Monte-Carlo standard errors of zero.
        assert abs(bias) < 6.0 * se + 0.05, f"finite-point bias {bias} (se={se}) looks systematic"

    def test_interval_narrows_with_more_expected_events(self):
        """Increasing expected control events must recover contracting
        uncertainty -- a full-space-only or always-refuse implementation
        cannot pass this. Averaged over a few reps per tier, not a single
        draw, so it is not sensitive to one unlucky small-count outcome."""
        rng = np.random.default_rng(4)
        mean_widths = []
        for expected_events in (1, 10, 100):
            cell = _manifest_cell(0.1, expected_events, (1, 1), 1.0)
            widths = []
            for _ in range(10):
                x_c = int(rng.binomial(cell.n_c, cell.p_c))
                x_t = int(rng.binomial(cell.n_t, cell.p_t))
                result = _estimate(x_c, cell.n_c, x_t, cell.n_t)
                _lo, hi = _set_bounds(result)
                if math.isfinite(hi):
                    widths.append(hi - _lo)
            mean_widths.append(sum(widths) / len(widths) if widths else math.inf)
        assert mean_widths[-1] < mean_widths[0], mean_widths


# --- Producer-path fixtures: native, frame, artifact ------------------------
# Raw binary fixtures run through native DuckDB `Analysis`, `from_unit_summary`
# and published `from_unit_day_artifact` paths, not hand-built group_summary
# rows. increment/simulate/dgp.py's binomial fixture helpers fix exact counts.

_NATIVE_DEFS_TEMPLATE = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM re_events
    timestamp_column: ts
    entities: [unit_id]
    facts:
      - name: exposure
        column: null
      - name: conversion
        column: value
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: conv
    type: conversion
    entity: unit_id
    fact: conversion
    window_days: 30
experiments:
  - name: rare_event_test
    exposure: enrolled
    unit: unit_id
    start: 2025-01-01
    end: 2025-02-15
    plan: {secondaries: [conv]}
    control_group: control
"""


def _native_analysis(x_c: int, n_c: int, x_t: int, n_t: int, *, seed: int = 0):
    raw = simulate_binomial_conversion_logs(
        x_c, n_c, x_t, n_t, seed=seed, experiment_id="rare_event_test"
    )
    import pyarrow.compute as pc

    is_exposure = pc.field("event") == "exposure"
    exposure_data = raw.filter(is_exposure).select(["unit_id", "ts", "experiment_id", "group_id"])
    fact_data = raw.filter(~is_exposure).select(["unit_id", "ts", "event", "value"])

    con = ibis.duckdb.connect()
    exp_df = exposure_data.to_pandas()
    exp_df["event"] = "exposure"
    exp_df["value"] = float("nan")
    fact_df = fact_data.to_pandas()
    fact_df["experiment_id"] = ""
    fact_df["group_id"] = None
    con.create_table("re_events", pd.concat([exp_df, fact_df], ignore_index=True))

    tmp = Path(tempfile.mkdtemp())
    defs_path = tmp / "defs.yaml"
    defs_path.write_text(_NATIVE_DEFS_TEMPLATE)
    return con, Analysis("rare_event_test", defs_path, con), defs_path


def _frame_analysis(x_c: int, n_c: int, x_t: int, n_t: int, *, seed: int = 0) -> Analysis:
    unit_ids, group_ids, converted = binomial_fixture_units(x_c, n_c, x_t, n_t, seed=seed)
    df = pd.DataFrame(
        {"unit_id": unit_ids, "group_id": group_ids, "converted": converted.astype(float)}
    )
    return Analysis.from_unit_summary(
        df,
        unit="unit_id",
        group="group_id",
        control="control",
        metrics=[{"name": "conv", "type": "conversion", "value_column": "converted"}],
    )


def _artifact_analysis(x_c: int, n_c: int, x_t: int, n_t: int, *, seed: int = 0) -> Analysis:
    from increment.query.artifact_publish import artifact_context
    from increment.semantics import load

    con, native, defs_path = _native_analysis(x_c, n_c, x_t, n_t, seed=seed)
    definitions = load(defs_path)
    experiment = next(item for item in definitions.experiments if item.name == "rare_event_test")
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = native.publish_unit_day_artifact(store)
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context)


def _conv_row(results):
    (row,) = [r for r in results if r.metric == "conv"]
    return row


def _assert_binomial_sets_agree(reference, *others) -> None:
    ref_lo, ref_hi = _set_bounds(reference)
    for other in others:
        assert other.reference_kind == "binomial"
        other_lo, other_hi = _set_bounds(other)
        assert other.binomial_set.x_c == reference.binomial_set.x_c
        assert other.binomial_set.n_c == reference.binomial_set.n_c
        assert other.binomial_set.x_t == reference.binomial_set.x_t
        assert other.binomial_set.n_t == reference.binomial_set.n_t
        assert other_lo == pytest.approx(ref_lo, rel=1e-9)
        if math.isfinite(ref_hi):
            assert math.isfinite(other_hi)
            assert other_hi == pytest.approx(ref_hi, rel=1e-9)
        else:
            assert not math.isfinite(other_hi)
        assert (other.lift is None) == (reference.lift is None)


_PRODUCER_FIXTURES = {
    "zero_control": (0, 30, 5, 30),
    "zero_treatment": (5, 30, 0, 30),
    "one_success_each": (1, 50, 1, 50),
    "all_success": (10, 10, 10, 10),
    "se_half_boundary": (3, 40, 8, 40),
    "shifted_positive_direction": (10, 100, 25, 100),
    "shifted_negative_direction": (25, 100, 10, 100),
    "moderate_unchanged": (150, 1000, 165, 1000),
}


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestProducerPathFixtures:
    """Selected raw binary fixtures through native, frame, and artifact
    producers, each checked for parity against the direct ArmStats
    reference this module's other tests use throughout."""

    @pytest.mark.parametrize(("name", "counts"), sorted(_PRODUCER_FIXTURES.items()))
    def test_native_frame_and_artifact_agree_with_the_direct_reference(self, name, counts):
        x_c, n_c, x_t, n_t = counts
        reference = _estimate(x_c, n_c, x_t, n_t)

        con, native, _defs_path = _native_analysis(x_c, n_c, x_t, n_t, seed=7)
        native_row = _conv_row(native.run())

        frame_row = _conv_row(_frame_analysis(x_c, n_c, x_t, n_t, seed=7).run())

        artifact_row = _conv_row(_artifact_analysis(x_c, n_c, x_t, n_t, seed=7).run())

        _assert_binomial_sets_agree(reference, native_row, frame_row, artifact_row)
        con.disconnect()
