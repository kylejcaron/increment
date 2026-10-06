"""Conditioning of the two moment wire formats across a coefficient-of-variation sweep.

Format 1 stores raw additive sums, so ``sum_y2`` held ``n*mean**2 + (n-1)*var``
in one float64, and the variance signal shrank against that ulp as the mean
grew: usable at CV >= 1e-6, degraded near CV ~ 1e-7, worthless at CV <= 3e-8.
These bands reflect the stored sums, so they are PINNED here unchanged -
format-1 input cannot be made better than the producer left it.

Format 2 centers at the producer while raw values are still alive, so every
reduction is an analytic expansion in centered fields with no
``O(n*mean**2)`` term - its SEs stay exact-grade to CV = 1e-12.

The subgroup (uptake) family is exact-grade only because it uses that
expansion; ``cov(y, d*y)`` is pinned directly so a refactor to recovering
and subtracting ``sum_y2d``/``sum_yd`` fails loudly, not quietly."""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from increment.errors import IncrementRuntimeWarning
from increment.estimation.armstats import ArmStats
from increment.estimation.cuped import cuped_adjust
from increment.estimation.variance import ratio_moments
from tests.warning_codes import warning_codes

# mean = 1/CV at sigma = 1. Every value is an exact power of ten, hence
# exactly representable, which the Sterbenz-shift reference below needs.
SWEEP_CV = (1e-4, 1e-6, 1e-7, 3e-8, 1e-9, 1e-12)
SMOKE_CV = (1e-6, 1e-12)


def _exact_var(v: np.ndarray) -> Fraction:
    f = [Fraction(t) for t in v]
    m = sum(f, Fraction(0)) / len(f)
    return sum(((t - m) ** 2 for t in f), Fraction(0)) / (len(f) - 1)


def _exact_cov(a: np.ndarray, b: np.ndarray) -> Fraction:
    fa = [Fraction(t) for t in a]
    fb = [Fraction(t) for t in b]
    ma = sum(fa, Fraction(0)) / len(fa)
    mb = sum(fb, Fraction(0)) / len(fb)
    pairs = ((p - ma) * (q - mb) for p, q in zip(fa, fb, strict=True))
    return sum(pairs, Fraction(0)) / (len(fa) - 1)


def _exact_mean(v: np.ndarray) -> Fraction:
    return sum((Fraction(t) for t in v), Fraction(0)) / len(v)


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want) if want else abs(got)


def _v2(
    y: np.ndarray,
    *,
    x: np.ndarray | None = None,
    den: np.ndarray | None = None,
    d: np.ndarray | None = None,
    group_id: str = "A",
) -> ArmStats:
    """The record a format-2 producer emits for these unit values."""
    r = float(np.mean(y))
    fields: dict[str, object] = {
        "n": len(y),
        "ref_y": r,
        "cy1": float(np.sum(y - r)),
        "cy2": float(np.sum((y - r) ** 2)),
    }
    if x is not None:
        rx = float(np.mean(x))
        fields |= {
            "ref_x": rx,
            "cx1": float(np.sum(x - rx)),
            "cx2": float(np.sum((x - rx) ** 2)),
            "cxy": float(np.sum((x - rx) * (y - r))),
            "x_role": "covariate",
        }
    if den is not None:
        rd = float(np.mean(den))
        fields |= {
            "ref_den": rd,
            "cden1": float(np.sum(den - rd)),
            "cden2": float(np.sum((den - rd) ** 2)),
            "cyden": float(np.sum((y - r) * (den - rd))),
        }
    if d is not None:
        fields |= {
            "sum_d": float(np.sum(d)),
            "cyd": float(np.sum(d * (y - r))),
            "cy2d": float(np.sum(d * (y - r) ** 2)),
        }
        if x is not None:
            fields["cxd"] = float(np.sum(d * (x - float(np.mean(x)))))
    return ArmStats(study_id="s", metric="m", group_id=group_id, **fields)  # ty: ignore[invalid-argument-type]


def _v1(
    y: np.ndarray,
    *,
    x: np.ndarray | None = None,
    den: np.ndarray | None = None,
    group_id: str = "A",
) -> ArmStats:
    """The same units through the format-1 adapter, warnings suppressed;
    the warning bands themselves are asserted separately below."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ArmStats.from_raw_sums(
            study_id="s",
            metric="m",
            group_id=group_id,
            n=len(y),
            sum_y=float(y.sum()),
            sum_y2=float((y * y).sum()),
            sum_x=None if x is None else float(x.sum()),
            sum_x2=None if x is None else float((x * x).sum()),
            sum_xy=None if x is None else float((x * y).sum()),
            sum_den=None if den is None else float(den.sum()),
            sum_den2=None if den is None else float((den * den).sum()),
            sum_yden=None if den is None else float((y * den).sum()),
        )


def _draw(cv: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, x, den) at sigma = 1 and mean = 1/cv, with x correlated to y."""
    rng = np.random.default_rng(seed)
    mu = 1.0 / cv
    z = rng.normal(0.0, 1.0, n)
    y = mu + z
    x = mu + 0.6 * z + rng.normal(0.0, 0.8, n)
    den = mu + rng.normal(0.0, 1.0, n)
    return y, x, den


def _se_mean(arm: ArmStats) -> float:
    return math.sqrt(arm.var_y() / arm.n)


def _se_log_ratio(arm: ArmStats) -> float:
    """Delta-method SE of log(sum(y)/sum(den)) from one arm's moments."""
    n_bar, d_bar, var_n, var_d, cov_nd = ratio_moments(arm)
    var_log_r = (var_n / n_bar**2 - 2.0 * cov_nd / (n_bar * d_bar) + var_d / d_bar**2) / arm.n
    return math.sqrt(max(var_log_r, 0.0))


def _exact_se_mean(y: np.ndarray) -> float:
    return math.sqrt(float(_exact_var(y)) / len(y))


def _exact_se_log_ratio(y: np.ndarray, den: np.ndarray) -> float:
    n = len(y)
    nb, db = _exact_mean(y), _exact_mean(den)
    v = (_exact_var(y) / nb**2 - 2 * _exact_cov(y, den) / (nb * db) + _exact_var(den) / db**2) / n
    return math.sqrt(float(v))


def _cuped_ses(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray, fmt: str
) -> list[float]:
    build = _v2 if fmt == "v2" else _v1
    arms = [build(y_c, x=x_c, group_id="c"), build(y_t, x=x_t, group_id="t")]
    return [math.sqrt(s.var / s.n) for s in cuped_adjust(arms)]


def _exact_cuped_ses(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray
) -> list[float]:
    """Exact-rational WITHIN-arm, inverse-n-weighted theta and per-arm
    adjusted variance.

    Theta is the contrast-optimal form the estimator uses: each arm keeps
    its own centered moments, weighted by ``1/n_arm``, so no between-arm
    delta enters theta. Deriving it from the concatenated arms would make
    this oracle measure a different estimator instead of the moments path's
    numerical precision.
    """
    arms = ((y_c, x_c), (y_t, x_t))
    weighted_cov = Fraction(0)
    weighted_var_x = Fraction(0)
    for y, x in arms:
        weighted_cov += _exact_cov(y, x) / len(y)
        weighted_var_x += _exact_var(x) / len(x)
    theta = weighted_cov / weighted_var_x
    out = []
    for y, x in arms:
        v = _exact_var(y) - 2 * theta * _exact_cov(y, x) + theta**2 * _exact_var(x)
        out.append(math.sqrt(float(v) / len(y)))
    return out


# --- Format-1 bands: unchanged, and unchangeable ---


@pytest.mark.parametrize(
    ("cv", "warns"),
    [(1e-6, False), (1e-7, False), (3e-8, True), (1e-9, True), (1e-12, True)],
)
def test_format_one_warn_band_is_unchanged(cv: float, warns: bool) -> None:
    """The adapter warns exactly where the raw-sum reductions warned: silent
    down to CV ~ 1e-7, warning from CV ~ 3e-8 - the sqrt(eps) crossover."""
    y, _, _ = _draw(cv, 1000, seed=0)
    kwargs = {
        "study_id": "s",
        "metric": "m",
        "group_id": "A",
        "n": len(y),
        "sum_y": float(y.sum()),
        "sum_y2": float((y * y).sum()),
    }
    if warns:
        with pytest.warns(IncrementRuntimeWarning) as rec:
            ArmStats.from_raw_sums(**kwargs)  # ty: ignore[invalid-argument-type]
        assert set(warning_codes(rec)) & {
            "estimation.armstats.centered_sum_squares_clamped",
            "estimation.armstats.centered_sum_squares_noise_floor",
        }
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ArmStats.from_raw_sums(**kwargs)  # ty: ignore[invalid-argument-type]


def test_format_one_accuracy_band_is_unchanged() -> None:
    """Usable at CV >= 1e-6, percent-scale wrong at CV ~ 1e-7, total loss
    past the sqrt(eps) crossover. Pinning the DEGRADATION matters as much as
    pinning the accuracy: it is what format 2 exists to remove."""
    n = 1000
    errs = {}
    for cv in (1e-6, 1e-7, 1e-9):
        y, _, _ = _draw(cv, n, seed=0)
        errs[cv] = _rel(_se_mean(_v1(y)), _exact_se_mean(y))
    assert errs[1e-6] < 1e-3
    assert 1e-3 < errs[1e-7] < 1.0
    assert errs[1e-9] > 0.5


# --- Format-2: exact-grade to CV = 1e-12 ---


def _assert_v2_exact(cv: float, n: int, seed: int) -> dict[str, float]:
    y, x, den = _draw(cv, n, seed)
    conv = (np.random.default_rng(seed + 91).random(n) < 0.03).astype(float)
    errs = {
        "mean": _rel(_se_mean(_v2(y)), _exact_se_mean(y)),
        # A conversion metric's y is 0/1, so its sums are exact integers in
        # float64 and it carries no cancellation class at any rate.
        "conversion": _rel(_se_mean(_v2(conv)), _exact_se_mean(conv)),
        "ratio": _rel(_se_log_ratio(_v2(y, den=den)), _exact_se_log_ratio(y, den)),
    }
    # Unequal split, so the inverse-n theta weights are load-bearing here too.
    cut = n // 4
    got = _cuped_ses(y[:cut], x[:cut], y[cut:], x[cut:], "v2")
    want = _exact_cuped_ses(y[:cut], x[:cut], y[cut:], x[cut:])
    errs["cuped_control"] = _rel(got[0], want[0])
    errs["cuped_treatment"] = _rel(got[1], want[1])
    for name, err in errs.items():
        assert err < 1e-9, f"CV={cv:g} {name}: SE relative error {err:.3g}"
    return errs


@pytest.mark.parametrize("cv", SMOKE_CV)
def test_format_two_se_is_exact_grade_smoke(cv: float) -> None:
    """Fast gate: the two ends of the sweep, small n."""
    _assert_v2_exact(cv, n=200, seed=11)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cv", SWEEP_CV)
def test_format_two_se_is_exact_grade_across_the_sweep(cv: float) -> None:
    """The full band measurement: mean, conversion, ratio and CUPED standard
    errors all stay under 1e-9 relative error down to CV = 1e-12, where the
    format-1 path has been returning pure noise for four orders of
    magnitude."""
    _assert_v2_exact(cv, n=1000, seed=0)


# --- The subgroup (uptake) family, stated separately ---


def _uptake_arm(cv: float, n: int, seed: int, fmt: str) -> tuple[ArmStats, np.ndarray, np.ndarray]:
    y, _, _ = _draw(cv, n, seed)
    rng = np.random.default_rng(seed + 500)
    d = (rng.random(n) < 0.4).astype(float)
    if fmt == "v2":
        return _v2(y, d=d), y, d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arm = ArmStats.from_raw_sums(
            study_id="s",
            metric="m",
            group_id="A",
            n=n,
            sum_y=float(y.sum()),
            sum_y2=float((y * y).sum()),
            sum_d=float(d.sum()),
            sum_yd=float((y * d).sum()),
            sum_y2d=float((y * y * d).sum()),
        )
    return arm, y, d


def _assert_uptake_exact(cv: float, n: int, seed: int) -> dict[str, float]:
    arm, y, d = _uptake_arm(cv, n, seed, "v2")
    errs = {
        "cov_yd": _rel(arm.cov_yd(), float(_exact_cov(y, d))),
        "var_yd": _rel(arm.var_yd(), float(_exact_var(y * d))),
        "cov_y_yd": _rel(arm.cov_y_yd(), float(_exact_cov(y, y * d))),
    }
    for name, err in errs.items():
        assert err < 1e-9, f"CV={cv:g} {name}: relative error {err:.3g}"
    return errs


@pytest.mark.parametrize("cv", SMOKE_CV)
def test_subgroup_expansions_are_exact_grade_smoke(cv: float) -> None:
    """``cov(y, d*y)`` is pinned directly, not just through a downstream SE.

    Recovering ``sum_y2d``/``sum_yd`` from the centered fields and
    subtracting measures no better than format 1 (1.5e0 relative error at
    mean=1e15 against 2.8e-17 for the analytic expansion), and it would
    still look perfect at mu ~ 1 - so the guard has to live here, at large
    mu, on the intermediate itself."""
    _assert_uptake_exact(cv, n=200, seed=13)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cv", SWEEP_CV)
def test_subgroup_expansions_are_exact_grade_across_the_sweep(cv: float) -> None:
    _assert_uptake_exact(cv, n=1000, seed=0)


def test_raw_recovery_of_cov_y_yd_is_decisively_worse_than_the_expansion() -> None:
    """The refactor guard the design asks for, on the hairiest consumer.

    Recovering ``sum_yd``/``sum_y2d`` from the SAME centered record and then
    applying the raw-sum formula re-forms the ``O(n*mean**2)`` term and
    throws the accuracy away again - measured here at 1e-1 relative error
    at CV = 1e-12, against exactly 0 for the analytic expansion. At mu ~ 1
    the two agree to the last bit, which is precisely why no parity suite
    can catch this and why the pin lives at extreme mu.
    """
    n = 1000
    for cv, recovery_floor in ((1e-4, None), (1e-12, 1e-3)):
        arm, y, d = _uptake_arm(cv, n, 0, "v2")
        assert arm.sum_d is not None and arm.cyd is not None and arm.cy2d is not None
        want = float(_exact_cov(y, y * d))
        assert _rel(arm.cov_y_yd(), want) < 1e-12

        r, k = arm.ref_y, arm.sum_d
        sum_y = n * r + arm.cy1
        sum_yd = arm.cyd + r * k
        sum_y2d = arm.cy2d + 2.0 * r * sum_yd - r * r * k
        recovered = (sum_y2d - sum_y * sum_yd / n) / (n - 1)
        if recovery_floor is not None:
            assert _rel(recovered, want) > recovery_floor


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cv", SWEEP_CV)
def test_format_one_subgroup_degrades_where_format_two_does_not(cv: float) -> None:
    """The contrast the format bump buys.

    The adapter salvages what it can from format-1 sums with exact-rational
    centering, but the producer already destroyed the signal: by CV = 1e-12
    it is percent-scale wrong while the centered path is exact.
    """
    arm_v1, y, d = _uptake_arm(cv, 1000, 0, "v1")
    arm_v2, _, _ = _uptake_arm(cv, 1000, 0, "v2")
    want = float(_exact_cov(y, y * d))
    assert _rel(arm_v2.cov_y_yd(), want) < 1e-9
    if cv <= 1e-12:
        assert _rel(arm_v1.cov_y_yd(), want) > 1e-3


def _additive_late(t: ArmStats, c: ArmStats) -> tuple[float, float]:
    """The additive LATE row's (point, SE) through the public estimator."""
    from increment.estimation.encouragement import estimate_encouragement
    from increment.semantics.design import Encouragement
    from increment.semantics.models import MeanMetric

    design = Encouragement.model_validate(
        {
            "mechanism": "encouragement",
            "control_group": "c",
            "uptake": {"fact": "help_click"},
            "exclusion_restriction": {
                "acknowledged": True,
                "justification": "unclicked button assumed inert",
            },
        }
    )
    metric = MeanMetric(name="m", entity="user_id", fact="purchase", aggregation="sum")
    summary = []
    for arm in (t, c):
        row = arm.model_dump()
        row["experiment_id"] = row.pop("study_id")
        summary.append(row)
    computation = estimate_encouragement([metric], summary, design, estimands=("late",))
    (late,) = [
        row
        for row in computation.results
        if row.estimand == "late" and row.value_scale == "absolute"
    ]
    assert late.lift is not None and late.lift.log_se is not None
    return late.lift.value, late.lift.log_se


def test_late_se_components_survive_a_large_offset() -> None:
    """LATE's delta-method SE at CV = 1e-6, six orders past where the
    format-1 path dies.

    The Wald numerator ``mean_t - mean_c`` is a difference of two large
    means: at CV = 1e-12 the effect falls below the grid spacing of the
    data itself, which is a property of the measurement and not of any wire
    format. What format 2 owns is every VARIANCE component feeding the SE,
    and those are asserted exact-grade across the whole sweep above.
    """
    n, cv = 4000, 1e-6
    y, _, _ = _draw(cv, n, seed=3)
    rng = np.random.default_rng(77)
    d_c = (rng.random(n) < 0.05).astype(float)
    d_t = (rng.random(n) < 0.55).astype(float)
    y_t = y + 40.0 * d_t
    c = _v2(y, d=d_c, group_id="c")
    t = _v2(y_t, d=d_t, group_id="t")
    tau, se = _additive_late(t, c)
    # Same computation on the exactly-shifted residuals (Sterbenz: y - mu is
    # exact at this magnitude), which is the well-conditioned reference.
    mu = 1.0 / cv
    c0 = _v2(y - mu, d=d_c, group_id="c")
    t0 = _v2(y_t - mu, d=d_t, group_id="t")
    tau0, se0 = _additive_late(t0, c0)
    assert _rel(tau, tau0) < 1e-9
    assert _rel(se, se0) < 1e-9
