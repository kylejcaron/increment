"""Calibration for the CATE estimator, its honest-split gate, the targeting rule and ARD.

The honest split exists to stop a fabricated subgroup number from being
reported, so the gate itself must be trustworthy first. Fifteen properties
are measured:

1. AUTOC null size: no-heterogeneity one-sided rejection at alpha=5% must
   sit in [3, 7]%, validating the plug-in SE that treats holdout rank
   weights as fixed given the out-of-sample score.
2. Monotone group effects under real heterogeneity - the gate is not merely
   conservative.
3. Flat group effects under a constant effect - the group table must not
   manufacture a gradient from the ranking.
4. Fabrication demo, pinned: in-sample top-quintile ranking on zero
   heterogeneity reads ~3x the true effect; the honest split reads truth.
5. Blindness repair: an effect with zero linear projection onto a covariate
   is invisible to a linear interaction at any n; a hinge basis recovers it.
6. Wider basis does not inflate the null: extra interaction columns widen
   the joint Wald test's df too, so null rejection stays nominal.
7. ARD earns its flag: ``ard=True`` must beat OLS on held-out prediction of
   the true effect when most of the interaction block is null.
8. Pre-committed policy value is unconditionally unbiased: the targeted set
   depends on holdout covariates and a model fit on OTHER units, so on
   zero-heterogeneity data its mean across ALL replications must equal the
   true average effect.
9. Gate-conditioned bias, pinned rather than asserted away: conditional on
   the gate opening by chance, the same policy value runs sharply high,
   because the rank test and the top-fraction contrast read the SAME
   holdout outcomes - the split removes the fitting sample's fingerprints
   but not the gate's own.
10. Nominal coverage (ATE and interaction slope): the HC2 sandwich's own
    acceptance test - both 95% intervals must cover [93, 97]% at n=4000.
11. The false discovery the contrast repairs: "one segment significant, one
    not" fires ~48% under no true difference; the interval on the
    difference itself fires at nominal.
12. Sandwich under skewed allocation: at 10/90 with the small arm carrying
    the large variance, a homoskedastic SE covers 69.9% against nominal
    95%; HC2 must stay >=93%.
13. Asymmetric design pays: richly adjusting a NON-interacted covariate
    costs the heterogeneity test no df and buys real ATE width (<=0.6x
    plain SE) on a nonlinear prognostic surface.
14. Binary outcomes: a Bernoulli outcome breaks the linear specification
    and ties variance to mean; ATE coverage must still be [92, 98]%.
15. Binary outcomes at skewed allocation: cells 12+14 combined - the thin
    10/90 arm's events are exactly the high-leverage rows; coverage must
    still be [92, 98]%.
16. Doubly robust AUTOC under confounding: null rejection remains nominal
    while the randomized-assignment IPW score over-rejects on the same draws.
17. Observational summaries: the score-based holdout ATE and each GATES group
    recover the constant treatment effect under confounding.

Cells 1-4 and 8-9 exercise ``increment.estimation.targeting`` directly,
since the calibration is a property of the statistic, not the frame
pass-through (covered in ``tests/test_cate.py``). Cells 5-7 and 10-15
exercise ``fit_cate``, since the basis, sandwich and joint Wald test sit
upstream of the split.

Cells 2-17 assert on means or rates across replications: a per-replication
check is itself a ~5% event and would flake on a correct implementation -
behind a selecting gate it is 17.6%, cell 9's whole subject.
"""

from __future__ import annotations

import functools
import math
import warnings
from functools import cache

import numpy as np
import pyarrow as pa
import pytest
from scipy.special import expit
from scipy.stats import t

from increment import IdentificationError
from increment.cate import estimate_cate, validate_cate
from increment.errors import InvalidRequestError
from increment.estimation._adjust.learners import LogisticPropensity, RidgeOutcome
from increment.estimation.cate import Covariate, fit_cate
from increment.estimation.targeting import (
    _dr_psi,
    _group_bins,
    _welch,
    targeting_rule_arrays,
    validate_cate_arrays,
)
from increment.frame import MetricSpec, from_unit_summary, synthesise_metric
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    IdentificationGate,
    Observational,
    UptakeSpec,
)
from increment.sources import MomentsSource
from tests.mc import Coverage, CoverageSet, mcse, nominal_band

_ALPHA = 0.05
_EFFECT = 0.20
# Prognostic signal: a decaying slope per covariate, so the outcome has a real
# baseline for the score to be tempted by and psi's centering to remove.
_PROGNOSTIC = (1.0, 0.2)


def _extreme_overlap_draw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    x[0], x[1] = 20.0, -20.0
    d = rng.binomial(1, expit(1.5 * x)).astype(float)
    y = 3.0 * x + d + rng.normal(size=n)
    return y, d, x[:, None], _units(n)


def test_dr_psi_trims_units_with_extreme_out_of_fold_propensity():
    y, d, X, unit_ids = _extreme_overlap_draw()
    psi, kept = _dr_psi(
        y,
        d,
        X,
        unit_ids,
        None,
        propensity_learner=LogisticPropensity,
        outcome_learner=RidgeOutcome,
        folds=5,
        seed=0,
        gate=IdentificationGate(overlap="trim"),
    )

    assert psi.shape == kept.shape == y.shape
    assert np.flatnonzero(~kept).tolist() == [0, 1]


def test_dr_psi_refuses_by_default_when_overlap_is_extreme():
    y, d, X, unit_ids = _extreme_overlap_draw()
    with pytest.raises(IdentificationError) as exc_info:
        _dr_psi(
            y,
            d,
            X,
            unit_ids,
            None,
            propensity_learner=LogisticPropensity,
            outcome_learner=RidgeOutcome,
            folds=5,
            seed=0,
            gate=IdentificationGate(),
        )
    assert exc_info.value.code == "estimation.targeting.overlap.refuse"


def test_dr_psi_never_divides_trimmed_propensity_endpoints():
    class EndpointPropensity:
        def fit(self, X, d) -> None:
            del X, d

        def predict(self, X):
            return np.where(X[:, 0] < -1.0, 0.0, np.where(X[:, 0] > 1.0, 1.0, 0.5))

    class MeanOutcome:
        def fit(self, X, d) -> None:
            del X
            self.mean = float(np.mean(d))

        def predict(self, X):
            return np.full(X.shape[0], self.mean)

    n = 40
    X = np.zeros((n, 1))
    X[:2, 0] = (-2.0, 2.0)
    d = np.tile((0.0, 1.0), n // 2)
    y = np.linspace(-1.0, 1.0, n) + d
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        psi, kept = _dr_psi(
            y,
            d,
            X,
            _units(n),
            None,
            propensity_learner=EndpointPropensity,
            outcome_learner=MeanOutcome,
            folds=5,
            seed=0,
            gate=IdentificationGate(overlap="trim"),
        )

    assert np.flatnonzero(~kept).tolist() == [0, 1]
    assert np.isfinite(psi).all()
    assert psi[~kept].tolist() == [0.0, 0.0]


def _units(n: int) -> np.ndarray:
    """Stable string ids - what the honest split hashes, per its contract."""
    return np.array([f"u{i}" for i in range(n)])


def _draw(
    rng: np.random.Generator,
    n: int,
    p: int,
    *,
    effect: float = _EFFECT,
    slope: float = 0.0,
    quad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """One randomized experiment: ``tau(x) = effect + slope x0 + quad x0^2``.

    ``slope = quad = 0`` is the zero-heterogeneity null every gate cell in
    this file is measured against.
    """
    x = rng.normal(size=(n, p))
    beta = np.linspace(*_PROGNOSTIC, p)
    d = (rng.random(n) < 0.5).astype(float)
    tau = effect + slope * x[:, 0] + quad * x[:, 0] ** 2
    y = x @ beta + tau * d + rng.normal(0.0, 1.0, n)
    return y, d, {f"x{j}": x[:, j] for j in range(p)}


def _sparse_draw(
    rng: np.random.Generator, n: int, p: int, slopes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """One randomized experiment with ``tau(x) = _EFFECT + x @ slopes``.

    The multi-covariate sibling of ``_draw``, and it also returns the TRUE
    per-unit ``tau``: cell 7 scores against the truth, not against realized
    outcomes, so there is no irreducible noise floor in its RMSE.
    """
    x = rng.normal(size=(n, p))
    d = (rng.random(n) < 0.5).astype(float)
    tau = _EFFECT + x @ slopes
    y = x @ np.linspace(*_PROGNOSTIC, p) + tau * d + rng.normal(0.0, 1.0, n)
    return y, d, {f"x{j}": x[:, j] for j in range(p)}, tau


def _interact(p: int) -> list[Covariate]:
    return [Covariate(name=f"x{j}") for j in range(p)]


def _null_rejection_rate(reps: int, n: int, p: int, seed: int) -> float:
    """Share of zero-heterogeneity replications the AUTOC test rejects."""
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    rejects = 0
    for _ in range(reps):
        y, d, cols = _draw(rng, n, p)
        validation = validate_cate_arrays(y, d, cols, ids, interact=interact, alpha=_ALPHA)
        assert validation.autoc.p_value is not None, validation.autoc.unavailable_reason
        rejects += validation.autoc.p_value < _ALPHA
    return rejects / reps


def _confounded_source(*, design=None):
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    d = rng.binomial(1, expit(1.5 * x))
    y = 3.0 * x + 1.0 * d + rng.normal(size=n)
    z = rng.normal(size=n)

    frame = pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "treatment", "control"),
            "revenue": y,
            "x": x,
            "z": z,
            "uptake": d,
        }
    )
    identification = design or (
        Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("x",)),
            gate=IdentificationGate(overlap="trim"),
        )
    )
    return from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        design=identification,
    )


def test_validate_cate_estimates_confounded_source_via_dr_score():
    result = validate_cate(_confounded_source(), "revenue", control="control", interact=["x"])
    assert isinstance(result.autoc.p_value, float)
    assert result.population in (None, "overlap_subpopulation")


def test_validate_cate_refuses_source_without_design():
    src = MomentsSource(
        [],
        metrics=[synthesise_metric(MetricSpec(name="revenue"))],
        study_id="designless",
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_cate(
            src,
            "revenue",
            control="control",
            interact=["x"],
        )
    assert exc_info.value.code == "cate.identification.unsupported_mechanism"
    assert exc_info.value.context["mechanism"] is None


def test_validate_cate_refuses_encouragement_design():
    encouragement = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment acts only through uptake"
        ),
    )
    src = _confounded_source(design=encouragement)
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_cate(src, "revenue", control="control", interact=["x"])
    assert exc_info.value.code == "cate.identification.unsupported_mechanism"
    assert exc_info.value.context["mechanism"] == "encouragement"


def test_estimate_cate_still_refuses_confounded_source():
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_cate(_confounded_source(), "revenue", control="control", interact=["x"])
    assert exc_info.value.code == "cate.identification.randomized_only"


def _confounded_draw(
    rng: np.random.Generator, n: int, p: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = rng.normal(size=(n, p))
    d = rng.binomial(1, expit(1.5 * x[:, 0])).astype(float)
    y = x @ np.linspace(*_PROGNOSTIC, p) + _EFFECT * d + rng.normal(size=n)
    return y, d, {f"x{j}": x[:, j] for j in range(p)}


_dr_psi_fn = functools.partial(
    _dr_psi,
    propensity_learner=LogisticPropensity,
    outcome_learner=RidgeOutcome,
    folds=5,
    seed=0,
    gate=IdentificationGate(overlap="trim"),
)


def _dr_validation(
    y: np.ndarray,
    d: np.ndarray,
    cols: dict[str, np.ndarray],
    unit_ids: np.ndarray,
    interact: list[Covariate],
):
    """The DR validation the public array entry produces for the same design."""
    return validate_cate_arrays(
        y,
        d,
        cols,
        unit_ids,
        interact=interact,
        adjust=(),
        adjustment=[cov.name for cov in interact],
        n_groups=5,
        alpha=_ALPHA,
        arm_summary="score",
        psi_fn=_dr_psi_fn,
    )


def _dr_null_rejection_rate(reps: int, n: int, p: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    rejects = 0
    for _ in range(reps):
        y, d, cols = _confounded_draw(rng, n, p)
        rejects += _dr_validation(y, d, cols, ids, interact).autoc.p_value < _ALPHA
    return rejects / reps


def _dr_holdout_ate_and_gates_effects(
    reps: int, n: int, p: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    ate = np.empty(reps)
    gates = np.empty((reps, 5))
    for r in range(reps):
        y, d, cols = _confounded_draw(rng, n, p)
        validation = _dr_validation(y, d, cols, ids, interact)
        ate[r] = validation.holdout_ate.value
        gates[r] = [group.effect for group in validation.groups]
    return ate, gates


def _ipw_null_rejection_rate_under_confounding(reps: int, n: int, p: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    rejects = 0
    for _ in range(reps):
        y, d, cols = _confounded_draw(rng, n, p)
        validation = validate_cate_arrays(
            y, d, cols, ids, interact=interact, adjust=(), n_groups=5, alpha=_ALPHA
        )
        assert validation.autoc.p_value is not None, validation.autoc.unavailable_reason
        rejects += validation.autoc.p_value < _ALPHA
    return rejects / reps


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestDrPsiCalibrationUnderConfounding:
    """At 500 reps, DR rejects 2.8% versus plain IPW's 26.4%.

    The nominal acceptance band is the three-MC-SE interval around 5%; the
    paired IPW result verifies that the data-generating process is meaningfully
    confounded rather than an easy randomized approximation.
    """

    @staticmethod
    @functools.cache
    def _rates() -> tuple[float, float]:
        reps, n, p, seed = 500, 2_000, 5, 20260914
        return (
            _dr_null_rejection_rate(reps, n, p, seed),
            _ipw_null_rejection_rate_under_confounding(reps, n, p, seed),
        )

    def test_dr_autoc_is_nominal(self):
        reps = 500
        dr_rate, _ = self._rates()
        lo, hi = nominal_band(_ALPHA, reps, k=3)
        assert lo <= dr_rate <= hi

    def test_plain_ipw_is_miscalibrated(self):
        _, ipw_rate = self._rates()
        assert ipw_rate > 0.15


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestDrScoreHoldoutAteAndGatesRecoverConstantEffect:
    """Score summaries recover tau=0.2 in every reported population slice.

    Across 500 reps, the holdout ATE mean is 0.2014 and the five GATES means
    are 0.2172, 0.1902, 0.2049, 0.1996, and 0.1952. Each comparison uses its
    own measured three-MC-SE bound rather than pooling away a group-specific
    bias.
    """

    def test_holdout_ate_and_every_gate_recover_the_constant_effect(self):
        reps, n, p, seed = 500, 2_000, 5, 20260915
        ate, gates = _dr_holdout_ate_and_gates_effects(reps, n, p, seed)
        ate_mcse = ate.std(ddof=1) / math.sqrt(reps)
        assert abs(ate.mean() - _EFFECT) <= 3.0 * ate_mcse
        for group in range(5):
            group_mcse = gates[:, group].std(ddof=1) / math.sqrt(reps)
            assert abs(gates[:, group].mean() - _EFFECT) <= 3.0 * group_mcse


def test_dr_score_under_confounding_smoke():
    """Fast, unmarked 8-rep n=500 variant of the two DR cells above.

    Eight draws cannot resolve a rejection rate, and with both nuisance
    models correctly specified the score is robust to a single-model slip.
    What it does catch is the confounder being ignored: the naive contrast
    sits ~1.0 above the truth here and ~2.7 above it on the public source,
    so an observational source routed to the randomized score fails.
    """
    from increment.semantics.design import Randomized

    rng = np.random.default_rng(1617)
    n, p, reps = 500, 3, 8
    ids, interact = _units(n), _interact(p)
    dr, naive = np.empty(reps), np.empty(reps)
    gates = np.empty((reps, 5))
    for r in range(reps):
        y, d, cols = _confounded_draw(rng, n, p)
        validation = _dr_validation(y, d, cols, ids, interact)
        assert validation.holdout_ate is not None
        dr[r] = validation.holdout_ate.value
        gates[r] = [group.effect for group in validation.groups]
        unadjusted = validate_cate_arrays(
            y, d, cols, ids, interact=interact, adjust=(), n_groups=5, alpha=_ALPHA
        ).holdout_ate
        assert unadjusted is not None
        naive[r] = unadjusted.value
    assert abs(dr.mean() - _EFFECT) <= 0.4, f"DR holdout ATE {dr.mean():.3f} is far from {_EFFECT}"
    assert np.abs(gates.mean(axis=0) - _EFFECT).max() <= 0.75, gates.mean(axis=0)
    assert naive.mean() - _EFFECT >= 0.5, f"naive contrast {naive.mean():.3f} is not confounded"

    adjusted = validate_cate(_confounded_source(), "revenue", control="control", interact=["x"])
    as_randomized = validate_cate(
        _confounded_source(design=Randomized(control_group="control")),
        "revenue",
        control="control",
        interact=["x"],
    )
    assert adjusted.holdout_ate is not None and as_randomized.holdout_ate is not None
    assert abs(adjusted.holdout_ate.value - 1.0) <= 0.75, adjusted.holdout_ate
    assert as_randomized.holdout_ate.value - 1.0 >= 1.5, as_randomized.holdout_ate


# 1. THE GATE: null size of the AUTOC rank test.


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestAutocNullSize:
    """Null rejection rate at alpha=5%, measured at 4.8% (n=2000, p=10, 2000
    reps; three more seeds land 4.25-4.70%). The [3, 7]% band is the
    acceptance criterion for treating holdout rank weights as fixed in the
    SE; outside it the plug-in is wrong and needs a half-sample bootstrap.
    """

    def test_one_sided_rejection_rate_is_nominal(self):
        reps, n, p = 2000, 2000, 10
        rate = _null_rejection_rate(reps, n, p, seed=20240517)
        mc_se = np.sqrt(rate * (1.0 - rate) / reps)
        assert 0.03 <= rate <= 0.07, (
            f"AUTOC null size {rate:.2%} (MC-SE {mc_se:.2%}, {reps} reps at n={n}, p={p}) "
            f"is outside [3, 7]% at nominal 5%. The plug-in standard error treats the rank "
            f"weights as fixed; this says that is not tenable. Do NOT widen the band -- "
            f"replace the standard error with a half-sample bootstrap"
        )


def test_autoc_null_size_smoke():
    """Fast, unmarked 40-rep variant of TestAutocNullSize.

    MC-SE ~3.4pp here only catches gross errors (e.g. a rank-weight SE
    scaled by ``m`` instead of ``sqrt(m)`` reads 33%). Does NOT catch a
    fitting-half leak into the holdout, which reads 15% - still inside the
    band at any rep count; the marked cell above is the only guard against it.
    """
    rate = _null_rejection_rate(40, 400, 5, seed=11)
    assert 0.0 <= rate <= 0.20, f"AUTOC null size smoke {rate:.1%} wildly off nominal 5%"


# 2. Monotone group effects when the heterogeneity is real.


def _quadratic_validation(n: int, seed: int):
    """A genuinely heterogeneous experiment, with the squared term supplied.

    ``tau`` is quadratic in ``x0``, so a design interacted on ``x0`` alone
    would see almost nothing: the symmetric part of the effect is orthogonal
    to a linear term. Handing the fit ``x0_sq`` as its own covariate is how a
    caller expresses a curved effect through this interface.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    d = (rng.random(n) < 0.5).astype(float)
    tau = _EFFECT + 0.3 * x[:, 0] + 0.3 * x[:, 0] ** 2
    y = x @ np.array([1.0, 0.5, 0.2]) + tau * d + rng.normal(0.0, 1.0, n)
    cols = {"x0": x[:, 0], "x0_sq": x[:, 0] ** 2, "x1": x[:, 1], "x2": x[:, 2]}
    interact = [Covariate(name=c) for c in cols]
    return validate_cate_arrays(y, d, cols, _units(n), interact=interact, alpha=_ALPHA)


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestMonotoneGroupsUnderRealHeterogeneity:
    """Measured this session at n=40000: group effects
    [+0.159, +0.261, +0.264, +0.582, +1.092], Q5 - Q1 = +0.933 at 12.9 SE.
    """

    def test_top_group_beats_bottom_group(self):
        validation = _quadratic_validation(40_000, seed=20240517)
        low, high = validation.groups[0], validation.groups[-1]
        diff = high.effect - low.effect
        # Disjoint sets of units, so the two Welch errors simply add in quadrature.
        se = float(np.hypot(high.se, low.se))
        assert diff > 0.0, f"Q5 - Q1 = {diff:+.4f} is not positive under a real gradient"
        assert diff > 3.0 * se, (
            f"Q5 - Q1 = {diff:+.4f} is only {diff / se:.1f} SE from zero; the group table "
            f"is not resolving heterogeneity this obvious"
        )
        assert validation.passed, "the gate refused a genuinely heterogeneous effect"


def test_monotone_groups_smoke():
    """Fast, unmarked n=4000 variant of TestMonotoneGroupsUnderRealHeterogeneity.

    A tenth of the units is a third of the separation in SE units, so the
    bound drops from 3 SE to 2.
    """
    validation = _quadratic_validation(4_000, seed=20240517)
    low, high = validation.groups[0], validation.groups[-1]
    diff = high.effect - low.effect
    se = float(np.hypot(high.se, low.se))
    assert diff > 2.0 * se, f"Q5 - Q1 = {diff:+.4f} only {diff / se:.1f} SE from zero at n=4000"


# 3. Flat group effects under a constant effect.


def _flat_group_means(reps: int, n: int, p: int, seed: int, n_groups: int = 5):
    """Per-group mean effect across replications, and the per-rep worst |z|.

    The worst ``|z|`` is measured against the holdout's own average effect,
    which is the number a reader would compare a group against.
    """
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    effects = np.empty((reps, n_groups))
    worst_z = np.empty(reps)
    for r in range(reps):
        y, d, cols = _draw(rng, n, p)
        validation = validate_cate_arrays(
            y, d, cols, ids, interact=interact, n_groups=n_groups, alpha=_ALPHA
        )
        assert validation.holdout_ate is not None, validation.unavailable_reason
        effects[r] = [g.effect for g in validation.groups]
        se = np.array([g.se for g in validation.groups])
        worst_z[r] = np.max(np.abs(effects[r] - validation.holdout_ate.value) / se)
    return effects, worst_z


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestFlatGroupsUnderConstantEffect:
    """Measured (n=2000, p=10, 500 reps, truth 0.20): group means all
    within 1.7 MC-SE of truth; per replication the worst group sits ~1.4 SE
    from the holdout average (p95 2.28), with 0.4-0.6% exceeding 3 SE.
    """

    def test_no_group_strays_from_the_average_effect(self):
        reps = 500
        effects, worst_z = _flat_group_means(reps, 2_000, 10, seed=303)
        means = effects.mean(axis=0)
        mc_se = effects.std(axis=0, ddof=1) / np.sqrt(reps)
        z = (means - _EFFECT) / mc_se
        assert np.abs(z).max() < 3.0, (
            f"group means {np.round(means, 4).tolist()} against a constant {_EFFECT}: "
            f"worst is {np.abs(z).max():.1f} MC-SE out. The ranking is inventing a gradient"
        )
        # Per replication the groups are correlated with the average they are
        # compared to, so this is a sanity bound on the tail, not a nominal rate.
        excursions = float((worst_z > 3.0).mean())
        assert excursions < 0.03, (
            f"{excursions:.1%} of replications put some group past 3 SE of the holdout "
            f"average effect under no heterogeneity at all"
        )


def test_flat_groups_smoke():
    """Fast, unmarked 25-rep variant of TestFlatGroupsUnderConstantEffect.

    Bound loosens from 3 to 4 MC-SE; the tail-excursion check is dropped
    since 25 reps make a single 3-SE excursion already ~4% likely. Held at
    25 rather than fewer: the bound is estimated from these same draws, so
    trimming further fattens the statistic's own tail faster than it saves
    time.
    """
    reps = 25
    effects, _ = _flat_group_means(reps, 1_000, 5, seed=303)
    means = effects.mean(axis=0)
    mc_se = effects.std(axis=0, ddof=1) / np.sqrt(reps)
    z = np.abs((means - _EFFECT) / mc_se).max()
    assert z < 4.0, f"flat-groups smoke: worst group mean {z:.1f} MC-SE from {_EFFECT}"


# 4. The fabrication demo, pinned forever.


def _top_quintile_in_sample(y, d, cols, interact, n_groups: int) -> float:
    """The number the split exists to prevent: fit on everything, then read
    off the top quintile of the SAME units' predicted effects.

    Same quantile cut as the honest path (``_group_bins``), so only whose
    outcomes picked the group differs between the two arms.
    """
    fit = fit_cate(y, d, cols, interact=interact, alpha=_ALPHA)
    top = _group_bins(fit.score(cols, deploy_grain="unit"), n_groups) == n_groups - 1
    effect, _ = _welch(y[top & (d == 1.0)], y[top & (d == 0.0)])
    return effect


def _fabrication(reps: int, n: int, p: int, seed: int, n_groups: int = 5):
    """In-sample and honest-split top-quintile effects, replication by replication."""
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    in_sample = np.empty(reps)
    honest = np.empty(reps)
    for r in range(reps):
        y, d, cols = _draw(rng, n, p)
        in_sample[r] = _top_quintile_in_sample(y, d, cols, interact, n_groups)
        validation = validate_cate_arrays(
            y, d, cols, ids, interact=interact, n_groups=n_groups, alpha=_ALPHA
        )
        honest[r] = validation.groups[-1].effect
    return in_sample, honest


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestFabricationIsRemovedByTheSplit:
    """The regression test that keeps the gate honest.

    Measured (n=1000, p=20, 400 reps, truth 0.20): in-sample top quintile
    averages +0.577 (2.9x truth) with NO heterogeneity; the honest-split
    top quintile averages +0.168 (1.2 MC-SE from truth). ``p=20`` rather
    than 30 because the honest split hands ``fit_cate`` only half the
    units, and 30 interacted covariates would trip its ``p/n <= 1/10`` guard.
    """

    def test_in_sample_top_quintile_fabricates_and_the_split_does_not(self):
        reps = 400
        in_sample, honest = _fabrication(reps, 1_000, 20, seed=404)
        in_mean = float(in_sample.mean())
        hon_mean = float(honest.mean())
        hon_se = float(honest.std(ddof=1)) / np.sqrt(reps)

        assert in_mean >= 2.0 * _EFFECT, (
            f"in-sample top quintile averages {in_mean:+.4f} against a true {_EFFECT} -- "
            f"only {in_mean / _EFFECT:.1f}x. The demo has stopped demonstrating; check the "
            f"design is still wide enough relative to n to overfit"
        )
        assert abs(hon_mean - _EFFECT) <= 2.0 * hon_se, (
            f"honest-split top quintile averages {hon_mean:+.4f}, "
            f"{abs(hon_mean - _EFFECT) / hon_se:.1f} MC-SE from the true {_EFFECT}. "
            f"The split is leaking the fitting sample into the reported number"
        )
        assert in_mean > hon_mean + 2.0 * _EFFECT


def test_fabrication_smoke():
    """Fast, unmarked 20-rep variant of TestFabricationIsRemovedByTheSplit.

    20 draws can't resolve the honest arm against truth, so this only
    asserts the gap that survives at any sample size: in-sample reads above
    honest.
    """
    in_sample, honest = _fabrication(20, 600, 10, seed=404)
    in_mean, hon_mean = float(in_sample.mean()), float(honest.mean())
    assert in_mean >= 1.5 * _EFFECT, f"in-sample smoke {in_mean:+.4f} vs truth {_EFFECT}"
    assert in_mean > hon_mean, (
        f"in-sample top quintile {in_mean:+.4f} did not exceed the honest {hon_mean:+.4f}"
    )


# 5. A hinge basis repairs an effect the linear interaction cannot see.

_QUAD = 0.30


def _blindness(reps: int, n: int, seed: int) -> np.ndarray:
    """Linear versus ``knots=4`` interaction on a linearly invisible effect.

    ``tau(x) = 0.20 + 0.30(x^2-1)`` with standard normal x has
    ``E[x(x^2-1)] = 0``: the linear design is blind to this heterogeneity BY
    CONSTRUCTION, at any n. Returns per replication: linear fit's largest
    |z| over interactions, whether its joint test fired, |corr| of its
    score with truth, the hinge fit's joint p-value, and corr of its score
    with truth.
    """
    rng = np.random.default_rng(seed)
    linear = [Covariate(name="x0")]
    hinged = [Covariate(name="x0", knots=4)]
    rows = []
    for _ in range(reps):
        y, d, cols = _draw(rng, n, 1, effect=_EFFECT - _QUAD, quad=_QUAD)
        truth = _EFFECT + _QUAD * (cols["x0"] ** 2 - 1.0)
        flat = fit_cate(y, d, cols, interact=linear, alpha=_ALPHA)
        bent = fit_cate(y, d, cols, interact=hinged, alpha=_ALPHA)
        rows.append(
            (
                max(abs(i.coef / i.se) for i in flat.interactions),
                float(flat.heterogeneity.p_value < _ALPHA),
                abs(float(np.corrcoef(flat.score(cols, deploy_grain="unit"), truth)[0, 1])),
                bent.heterogeneity.p_value,
                float(np.corrcoef(bent.score(cols, deploy_grain="unit"), truth)[0, 1]),
            )
        )
    return np.asarray(rows)


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestHingeBasisRepairsALinearlyInvisibleEffect:
    """Measured (n=8000, 20 reps, truth 0.20+0.30(x^2-1)): linear
    interaction fires on 0% of reps (median |z| 0.97), score correlates
    0.018 with truth. knots=4 basis correlates 0.982 (worst 0.972), joint
    Wald p never exceeds 1.2e-56. Bands, not point measurements, are
    asserted since |z| and the correlations are random variables.
    """

    def test_hinges_recover_what_a_linear_interaction_cannot(self):
        m = _blindness(reps=20, n=8000, seed=20260812)
        assert np.median(m[:, 0]) < 1.96, (
            f"linear interaction median |z| {np.median(m[:, 0]):.2f} is significant on an "
            "effect it has no projection onto; the draw is not the intended one"
        )
        assert m[:, 1].mean() <= 0.15, (
            f"the linear fit called heterogeneity on {m[:, 1].mean():.0%} of replications "
            "against a nominal 5%"
        )
        assert m[:, 2].mean() < 0.10, (
            f"linear score correlates {m[:, 2].mean():.3f} with the truth; it is supposed "
            "to be blind"
        )
        assert m[:, 4].mean() >= 0.95, (
            f"knots=4 score correlates only {m[:, 4].mean():.3f} with the truth (doc 0.982); "
            "the hinge basis has stopped recovering the curve"
        )
        assert m[:, 3].max() < 1e-6, (
            f"knots=4 joint Wald p reached {m[:, 3].max():.1e}; the recovered curve is not "
            "being called significant"
        )


def test_hinge_blindness_repair_smoke():
    """Fast, unmarked single-draw n=1000 variant of the cell above.

    One draw at a fifth of the sample can't resolve either correlation
    precisely, so only the ordering is asserted: hinge score tracks truth,
    linear score does not.
    """
    m = _blindness(reps=1, n=1000, seed=20260812)
    assert m[0, 2] < 0.30, f"linear score correlates {m[0, 2]:.3f} with a truth it cannot see"
    assert m[0, 4] > 0.80, f"hinge score correlates only {m[0, 4]:.3f} with the truth"
    assert m[0, 3] < _ALPHA, f"hinge joint Wald p {m[0, 3]:.2e} missed real heterogeneity"


# 6. The wider basis does not manufacture heterogeneity.


def _hinge_null_rejection_rate(reps: int, n: int, seed: int) -> float:
    """Share of CONSTANT-effect replications the knots=4 joint Wald rejects."""
    rng = np.random.default_rng(seed)
    interact = [Covariate(name="x0", knots=4)]
    fire = 0
    for _ in range(reps):
        y, d, cols = _draw(rng, n, 1)
        fire += fit_cate(y, d, cols, interact=interact, alpha=_ALPHA).heterogeneity.p_value < _ALPHA
    return fire / reps


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestWiderBasisDoesNotInflateTheNull:
    """Measured: 4.7% at n=2000 over 1000 reps, against 4.5% for a linear
    interaction on the same draws - the five-column hinge block buys its
    extra flexibility with degrees of freedom, not size. Band [3.5, 7.5]%
    at alpha=5% (~+/-3.5 MC-SE); a rate outside it means a broken sandwich
    or Wald block, not noise.
    """

    def test_constant_effect_rejects_at_nominal_size(self):
        rate = _hinge_null_rejection_rate(reps=1000, n=2000, seed=20260812)
        assert 0.035 <= rate <= 0.075, (
            f"knots=4 joint Wald rejects a CONSTANT effect {rate:.1%} of the time at "
            "alpha=5%. Outside [3.5, 7.5]% the wider basis is manufacturing (or hiding) "
            "heterogeneity and the inference is wrong"
        )


def test_hinge_null_size_smoke():
    """Fast, unmarked 25-rep n=1000 variant of TestWiderBasisDoesNotInflateTheNull.

    MC-SE ~4.4pp only catches order-of-magnitude errors - exactly what a
    mis-sized Wald block looks like: pricing the five-column hinge block at
    one degree of freedom reads 60% here against 4% intact.
    """
    rate = _hinge_null_rejection_rate(reps=25, n=1000, seed=909)
    assert rate <= 0.25, f"hinge null size smoke {rate:.0%} wildly off nominal 5%"


# 7. ARD earns its flag: held-out score RMSE against the true tau(x).

_ARD_SLOPE = 0.30


def _ard_holdout_rmse(reps: int, n: int, p: int, k: int, seed: int) -> np.ndarray:
    """Held-out score RMSE against the true ``tau(x)``: OLS then ARD.

    Sparse heterogeneity - the first *k* of *p* interacted covariates carry
    a real slope, the rest are exactly null, ARD's target regime. Each rep
    fits on one sample and scores an INDEPENDENT sample, so nothing can be
    won by fitting the fit sample's own noise. Both fits share the same
    design and rows; only ``ard`` differs.
    """
    rng = np.random.default_rng(seed)
    interact = _interact(p)
    slopes = np.zeros(p)
    slopes[:k] = _ARD_SLOPE
    rows = []
    for _ in range(reps):
        y, d, cols, _ = _sparse_draw(rng, n, p, slopes)
        _, _, held, tau = _sparse_draw(rng, n, p, slopes)
        fits = [
            fit_cate(y, d, cols, interact=interact, alpha=_ALPHA, ard=flag)
            for flag in (False, True)
        ]
        rows.append(
            tuple(
                float(np.sqrt(np.mean((f.score(held, deploy_grain="unit") - tau) ** 2)))
                for f in fits
            )
        )
    return np.asarray(rows)


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestArdImprovesHeldOutPrediction:
    """The gate the ``ard=True`` flag had to pass to exist at all.

    Measured (n=2000, p=20 with 3 real slopes of 0.30, 30 reps, scored out
    of sample against truth): mean held-out RMSE 0.2141 unpenalized vs
    0.1471 with ARD (ratio 0.687), ARD ahead on 30/30 reps; three other
    sparse configurations agreed (ratio 0.66-0.71). Only "at least as good"
    is asserted; if this ever inverts, withdraw the flag rather than widen
    the band - a shrinkage knob that predicts worse than not shrinking has
    nothing to offer.
    """

    def test_ard_beats_ols_out_of_sample_under_sparse_heterogeneity(self):
        m = _ard_holdout_rmse(reps=30, n=2000, p=20, k=3, seed=20260812)
        ols, ard = m[:, 0].mean(), m[:, 1].mean()
        assert ard <= ols, (
            f"ARD held-out RMSE {ard:.4f} is WORSE than the unpenalized {ols:.4f} on the "
            "design ARD exists for; the flag is not earning its place"
        )
        wins = float((m[:, 1] < m[:, 0]).mean())
        assert wins >= 0.80, (
            f"ARD only predicted better on {wins:.0%} of replications (doc 100%); the mean "
            "may still be ahead, but the improvement has stopped being reliable"
        )


def test_ard_holdout_rmse_smoke():
    """Fast, unmarked 3-rep n=400 p=10 variant of the cell above.

    Can't resolve the ratio at 3 draws, but the ORDERING held on every rep
    of every configuration measured, so an inverted mean here means the
    shrinkage is pointing the wrong way.
    """
    m = _ard_holdout_rmse(reps=3, n=400, p=10, k=2, seed=4242)
    assert m[:, 1].mean() <= m[:, 0].mean(), (
        f"ARD held-out RMSE {m[:, 1].mean():.4f} worse than unpenalized {m[:, 0].mean():.4f}"
    )


# 8-9. The pre-committed policy value: unbiased, until the gate selects on it.

_FRACTION = 0.40
# Alpha this loose reports the policy value every replication would get
# behind a 5% gate, so an UNCONDITIONAL mean can be measured directly.
_OPEN = 1.0 - 1e-9


@cache
def _policy_values(reps: int, n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-fraction policy value and AUTOC p-value, replication by replication.

    Cached: cells 8 and 9 are two readings of ONE simulation, so cell 9's
    conditional mean must come off the same draws as cell 8's unconditional one.
    """
    rng = np.random.default_rng(seed)
    ids, interact = _units(n), _interact(p)
    values = np.empty(reps)
    p_values = np.empty(reps)
    for r in range(reps):
        y, d, cols = _draw(rng, n, p)
        rule = targeting_rule_arrays(
            y, d, cols, ids, interact=interact, fraction=_FRACTION, alpha=_OPEN
        )
        assert rule.policy_value is not None, "an alpha of 1 - 1e-9 cannot close the gate"
        values[r] = rule.policy_value.value
        p_values[r] = rule.validation.autoc.p_value
    return values, p_values


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPolicyValueIsUnconditionallyUnbiased:
    """Measured (n=2000 so 1000 held out, p=10, 8000 reps, truth 0.20,
    fraction 0.40): mean policy value +0.1971 (MC-SE 0.0025), 1.2 MC-SE
    below truth.

    Why it must hold: the targeted set is a function of the holdout's
    covariates and a model fit on OTHER units, so treatment stays
    randomized inside it and a difference in means there estimates that
    subgroup's average effect - the ATE under no heterogeneity. Drift means
    the fitting half is reaching the reported number, exactly what the
    split exists to prevent.
    """

    def test_the_mean_policy_value_is_the_true_average_effect(self):
        reps = 8_000
        values, _ = _policy_values(reps, 2_000, 10, 808)
        mean = float(values.mean())
        mc_se = float(values.std(ddof=1)) / np.sqrt(reps)
        assert abs(mean - _EFFECT) <= 2.0 * mc_se, (
            f"the pre-committed top-{_FRACTION:.0%} policy value averages {mean:+.4f} over "
            f"{reps} zero-heterogeneity replications, {abs(mean - _EFFECT) / mc_se:.1f} MC-SE "
            f"from the true {_EFFECT}. Unconditionally this statistic is an ordinary "
            f"randomized comparison inside a covariate-defined set; a bias here is a leak "
            f"of the fitting half into the cut"
        )


def test_policy_value_unconditional_smoke():
    """Fast, unmarked 60-rep variant of TestPolicyValueIsUnconditionallyUnbiased.

    Bound loosens from 2 to 4 MC-SE. Held at 60 (a detection floor, not a
    round number): a fitting-half leak into the holdout puts this statistic
    4.4 MC-SE out at 60 draws but only 2.9 at 45 - inside the bound, and
    silent.
    """
    values, _ = _policy_values(60, 600, 5, 808)
    mean = float(values.mean())
    mc_se = float(values.std(ddof=1)) / np.sqrt(values.size)
    assert abs(mean - _EFFECT) < 4.0 * mc_se, (
        f"policy-value smoke {mean:+.4f} is {abs(mean - _EFFECT) / mc_se:.1f} MC-SE "
        f"from the true {_EFFECT}"
    )


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPolicyValueIsBiasedConditionalOnAPassingGate:
    """The selection the honest split does NOT remove, pinned as a property.

    Measured on the same 8000 draws as cell 8: the gate opens on 4.32% of
    zero-heterogeneity reps, and across those the mean policy value is
    +0.4724 (MC-SE 0.0099) - +136% against a truth of 0.20.

    Structural, not a defect: the rank test and the top-fraction difference
    in means share the SAME held-out outcomes, so reps where noise made the
    ranking look real are disproportionately reps where the targeted
    units' outcomes ran high. The split removes the FITTING half's
    fingerprints; nothing removes the gate's own. So this cell measures the
    bias rather than asserting it away - "conditional on passing, within 2
    SE of truth" is WRONG and would fail a correct implementation (17.6% of
    passing reps put truth outside their own band, vs 4.5% unconditionally).
    What's pinned is the DIRECTION and rough SIZE of the bias.
    """

    def test_passing_by_chance_selects_replications_that_read_high(self):
        reps = 8_000
        values, p_values = _policy_values(reps, 2_000, 10, 808)
        passed = p_values < _ALPHA
        rate = float(passed.mean())
        conditional = values[passed]
        mean = float(conditional.mean())
        mc_se = float(conditional.std(ddof=1)) / np.sqrt(conditional.size)

        assert 0.02 <= rate <= 0.08, (
            f"the gate opened on {rate:.2%} of null replications (doc 4.32%); whatever this "
            f"cell then measures is no longer the conditioning it claims to measure"
        )
        assert 1.5 * _EFFECT <= mean <= 3.5 * _EFFECT, (
            f"conditional on a gate that passed by chance the mean policy value is "
            f"{mean:+.4f} (MC-SE {mc_se:.4f}, {passed.sum()} passes), i.e. "
            f"{(mean - _EFFECT) / _EFFECT:+.0%} against the true {_EFFECT}; the doc pins "
            f"+136%. This bias is a PROPERTY of sharing holdout outcomes between the rank "
            f"test and the policy value -- if it has vanished or inverted, either the two "
            f"stopped sharing outcomes or the gate stopped gating"
        )
        assert mean > float(values.mean()) + 0.5 * _EFFECT, (
            f"the same draws average {values.mean():+.4f} unconditionally and {mean:+.4f} "
            f"behind the gate; that gap IS the selection, and it has collapsed"
        )


def test_policy_value_gate_selection_smoke():
    """Fast, unmarked 60-rep variant of TestPolicyValueIsBiasedConditionalOnAPassingGate.

    60 draws hold ~3 gate passes at 5% - too few to average - so this
    loosens the SAME conditioning to ``p < 0.25`` (selects 9-15 reps); the
    gap survives the coarser cut. Reuses cell 8's cached draws.
    """
    values, p_values = _policy_values(60, 600, 5, 808)
    selected = values[p_values < 0.25]
    gap = float(selected.mean()) - float(values.mean())
    assert gap > 0.5 * _EFFECT, (
        f"selecting the {selected.size} best-ranked of {values.size} null replications "
        f"moved the mean policy value by only {gap:+.4f}; the gate's own selection "
        f"pressure has gone"
    )


# 10. The estimator's own inference: nominal coverage, ATE and interaction.

_SLOPE = 0.30


def _coverage(reps: int, n: int, p: int, seed: int) -> tuple[float, float]:
    """Share of 95% intervals covering the truth: the ATE, then the ``d:x0`` slope."""
    rng = np.random.default_rng(seed)
    interact = _interact(p)
    covset = CoverageSet()
    for _ in range(reps):
        y, d, cols = _draw(rng, n, p, slope=_SLOPE)
        result = fit_cate(y, d, cols, interact=interact, alpha=_ALPHA)
        # The basis standardizes by the FITTING sample's sd (ddof=1), so the
        # interaction coefficient is the per-sd slope on these same rows.
        truth = _SLOPE * float(cols["x0"].std(ddof=1))
        effect = next(e for e in result.interactions if e.name == "d:x0")
        covset.record(
            ate=result.lb <= _EFFECT <= result.ub,
            slope=effect.lb <= truth <= effect.ub,
        )
    ate_rate, slope_rate = covset.rates("ate", "slope")
    return ate_rate, slope_rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestIntervalsCoverAtNominal:
    """The acceptance test for the HC2 sandwich on a well-behaved design.

    Measured (n=4000, p=2, 1000 reps, truth 0.20 with a 0.30-per-sd slope on
    x0): ATE coverage 94.7%, ``d:x0`` coverage 94.6%, both against nominal 95%.

    A THOUSAND replications, not four hundred: the [93, 97]% band is 2.9
    MC-SE wide at 1000 reps (vs 1.84 at 400), keeping the false-failure
    rate low enough to trust as a regression test rather than a coin flip.
    """

    def test_ate_and_slope_intervals_cover_at_nominal(self):
        reps = 1_000
        ate, slope = _coverage(reps, 4_000, 2, 1210)
        mc_se = mcse(0.95, reps)

        assert 0.93 <= ate <= 0.97, (
            f"ATE coverage {ate:.1%} over {reps} replications, {abs(ate - 0.95) / mc_se:.1f} "
            f"MC-SE from the nominal 95% (doc 94.7%); the HC2 sandwich is not delivering the "
            f"interval it advertises"
        )
        assert 0.93 <= slope <= 0.97, (
            f"d:x0 coverage {slope:.1%} over {reps} replications, "
            f"{abs(slope - 0.95) / mc_se:.1f} MC-SE from the nominal 95% (doc 94.6%); the "
            f"interaction block's inference is off even where the ATE's is fine"
        )


def test_interval_coverage_smoke():
    """Fast, unmarked 15-rep n=800 variant of TestIntervalsCoverAtNominal.

    Can't resolve 95% from 90% at 15 draws; catches an interval that's
    broken (inverted, zero-width, mis-centered) rather than merely mistuned.
    """
    ate, slope = _coverage(15, 800, 2, 1210)
    assert ate >= 0.6, f"coverage smoke: ATE interval covered only {ate:.0%} of 15 draws"
    assert slope >= 0.6, f"coverage smoke: d:x0 interval covered only {slope:.0%} of 15 draws"


# 11. The false discovery this estimator exists to repair.

# Sized so each slice has ~65% power alone - where "exactly one of the two
# is significant" is most likely, the regime every real segment readout lives in.
_SEGMENT_EFFECT = 0.14


def _segment_disagreement(reps: int, n: int, seed: int) -> tuple[float, float]:
    """Firing rates of the naive two-slice rule and of the estimator's contrast.

    The two segments have IDENTICAL true effects, so every firing on either
    rule is a false discovery.
    """
    rng = np.random.default_rng(seed)
    segment = Covariate(name="segment", kind="categorical")
    x_only = [Covariate(name="x0")]
    naive = correct = 0
    for _ in range(reps):
        x = rng.normal(size=n)
        in_b = rng.random(n) < 0.5
        d = (rng.random(n) < 0.5).astype(float)
        y = _PROGNOSTIC[0] * x + _SEGMENT_EFFECT * d + rng.normal(0.0, 1.0, n)
        # The naive rule: two independent fits, then compare their verdicts.
        slices = [
            fit_cate(y[mask], d[mask], {"x0": x[mask]}, interact=x_only, alpha=_ALPHA)
            for mask in (~in_b, in_b)
        ]
        fired = [s.lb > 0.0 or s.ub < 0.0 for s in slices]
        naive += fired[0] != fired[1]
        # The estimator's rule: one fit, one interval on the DIFFERENCE.
        whole = fit_cate(
            y,
            d,
            {"segment": np.where(in_b, "b", "a"), "x0": x},
            interact=[segment, *x_only],
            alpha=_ALPHA,
        )
        gap = whole.contrast({"segment": "b", "x0": 0.0}, {"segment": "a", "x0": 0.0})
        correct += gap.excludes(0.0)
    return naive / reps, correct / reps


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestSegmentContrastRepairsTheFalseDiscovery:
    """The defect the whole feature exists to fix, pinned as a rate.

    "Segment A significant, B not, so they differ" is two marginal tests
    read as one comparison; on data with NO segment difference it fires
    whenever the two slices land on opposite sides of their own thresholds
    - near 65% per-slice power that's close to the most likely outcome.

    Measured (n=2000 split evenly, identical true effect 0.14, 1000 reps):
    naive rule fires 47.8%; ``CateResult.contrast``, one interval on the
    difference itself, fires 4.9% (nominal 5%). Bands are deliberately
    loose (>=30%, <=8%) - the order of magnitude of the repair, not the
    tuning of this power point.
    """

    def test_naive_slices_fire_constantly_and_the_contrast_does_not(self):
        reps = 1_000
        naive, correct = _segment_disagreement(reps, 2_000, 1701)

        assert naive >= 0.30, (
            f"the naive two-slice rule fired on only {naive:.1%} of {reps} no-difference "
            f"replications (doc 47.8%); this cell is supposed to be measuring a defect that "
            f"is happening, so either the power point drifted or the slices stopped being "
            f"read as a comparison"
        )
        assert correct <= 0.08, (
            f"the segment contrast fired on {correct:.1%} of {reps} replications with NO "
            f"true difference (doc 4.9%, nominal 5%); the one interval that is allowed to "
            f"answer 'do these segments differ' has stopped being honest"
        )
        assert naive > 4.0 * correct, (
            f"naive {naive:.1%} against contrast {correct:.1%}: the gap that justifies "
            f"reporting a contrast at all has collapsed"
        )


def test_segment_contrast_repair_smoke():
    """Fast, unmarked 20-rep n=600 variant of the cell above.

    Pins that the naive rule disagrees with itself at all while the
    contrast stays quiet, not either rate precisely.
    """
    naive, correct = _segment_disagreement(20, 600, 1701)
    assert naive > correct, (
        f"FDR smoke: naive rule fired {naive:.0%} and the contrast {correct:.0%} on 20 "
        f"no-difference draws; the naive rule is supposed to be the noisy one"
    )


# 12. Why the sandwich is not optional: skewed allocation.

_SKEW_SHARE = 0.10
# Treated sd, control sd. The SMALL arm carries the LARGE variance, which is
# the configuration a pooled variance gets most wrong.
_SKEW_SD = (2.5, 1.0)


def _pooled_ate_se(y: np.ndarray, d: np.ndarray, x: np.ndarray) -> float:
    """SE(ATE) from the textbook OLS variance ``s^2 (Z'Z)^-1`` on Lin's design.

    Same columns ``fit_cate`` fits (intercept, treatment, centered
    covariate, interaction), so the treatment coefficient matches its ATE
    to floating point and only the variance differs - one pooled residual
    variance charged to both arms, exactly the assumption a 10/90 split
    with unequal arm noise breaks.
    """
    n = y.size
    z = np.column_stack([np.ones(n), d, x - x.mean(), d * (x - x.mean())])
    ztz_inv = np.linalg.inv(z.T @ z)
    resid = y - z @ (ztz_inv @ (z.T @ y))
    sigma_sq = float(resid @ resid) / (n - z.shape[1])
    return float(np.sqrt(sigma_sq * ztz_inv[1, 1]))


def _skewed_allocation_coverage(reps: int, n: int, seed: int) -> tuple[float, float]:
    """ATE coverage under 10/90 allocation with heteroskedastic arms.

    Two coverages on the SAME draws: the estimator's HC2 interval, then a
    homoskedastic interval built with the estimator's own t-distribution
    critical value (n - p dof) around its own point estimate - only the
    width differs, so the gap is attributable to the variance alone.
    """
    rng = np.random.default_rng(seed)
    interact = [Covariate(name="x0")]
    covset = CoverageSet()
    for _ in range(reps):
        x = rng.normal(size=n)
        d = (rng.random(n) < _SKEW_SHARE).astype(float)
        sd = np.where(d == 1.0, _SKEW_SD[0], _SKEW_SD[1])
        y = _PROGNOSTIC[0] * x + _EFFECT * d + rng.normal(0.0, 1.0, n) * sd
        result = fit_cate(y, d, {"x0": x}, interact=interact, alpha=_ALPHA)
        t_crit = float(t.ppf(1.0 - _ALPHA / 2.0, max(n - len(result.columns), 1)))
        covset.record(
            coverage=result.lb <= _EFFECT <= result.ub,
            pooled=abs(result.ate - _EFFECT) <= t_crit * _pooled_ate_se(y, d, x),
        )
    coverage_rate, pooled_rate = covset.rates("coverage", "pooled")
    return coverage_rate, pooled_rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestSandwichSurvivesSkewedAllocation:
    """The regression test for why the HC2 sandwich is not a nicety.

    A homoskedastic variance on this exact configuration covers 69.9%
    against nominal 95% - recomputed on the same draws and point
    estimates, only the interval width differs. This is why the estimator
    has no "assume constant variance" fast path.

    Configuration: 10% of units treated, treated arm 2.5x as noisy as
    control - what every holdback and cautious rollout produces. Pooling
    the two residual variances charges the 10% arm a variance it doesn't have.

    Measured (n=4000, 2000 reps): HC2 coverage 95.1%, pooled-variance
    coverage 69.9%. HC2's floor is 93% (not two-sided - overcoverage from a
    finite-sample leverage correction is conservative); the pooled
    reference is held below 80%, far from both 69.9% and the 93% floor.
    """

    def test_ten_ninety_split_with_unequal_arm_variances_still_covers(self):
        reps = 2_000
        coverage, pooled = _skewed_allocation_coverage(reps, 4_000, 31)

        assert coverage >= 0.93, (
            f"ATE coverage {coverage:.1%} over {reps} replications at "
            f"{_SKEW_SHARE:.0%}/{1 - _SKEW_SHARE:.0%} allocation with arm sds {_SKEW_SD} "
            f"(doc 95.1%); the pooled-variance interval on the same draws read "
            f"{pooled:.1%}, so a slide toward that number means the sandwich has stopped "
            f"being a sandwich"
        )
        assert pooled < 0.80, (
            f"the pooled-variance reference covered {pooled:.1%} over {reps} replications "
            f"(doc 69.9%) where HC2 read {coverage:.1%}; this cell exists to show a "
            f"homoskedastic variance failing badly on this design, and it is no longer "
            f"failing -- the reference interval, not the sandwich, is what to check"
        )


def test_skewed_allocation_coverage_smoke():
    """Fast, unmarked 20-rep n=1000 variant of TestSandwichSurvivesSkewedAllocation.

    Can't separate 95% from 70% at 20 draws, but a variance that's stopped
    being heteroskedasticity-consistent misses repeatedly even here; the
    pooled reference runs too so the contrast can't silently stop computing.
    """
    coverage, pooled = _skewed_allocation_coverage(20, 1_000, 31)
    assert coverage > pooled, (
        f"skew-allocation smoke: HC2 covered {coverage:.0%} of 20 draws and the pooled "
        f"reference {pooled:.0%}; the sandwich is supposed to be the wider one"
    )
    assert coverage >= 0.6, (
        f"skew-allocation smoke: {coverage:.0%} of 20 intervals covered the truth; the "
        f"sandwich is not surviving a 10/90 split at all"
    )


# 13. The asymmetric design: adjust wide, interact narrow.

# Prognostic curvature on x1 with ZERO linear projection: a plain fit and a
# linear adjustment for x1 buy the same width, so the gap here is the hinge basis.
_CURVE = 1.30
_CURVE_KNOTS = 6


def _adjustment_se_ratios(reps: int, n: int, seed: int) -> np.ndarray:
    """SE(ATE) with a hinge adjustment block over SE(ATE) plain, per replication."""
    rng = np.random.default_rng(seed)
    interact = [Covariate(name="x0")]
    adjust = [Covariate(name="x1", knots=_CURVE_KNOTS)]
    ratios = np.empty(reps)
    for rep in range(reps):
        x0, x1 = rng.normal(size=n), rng.normal(size=n)
        d = (rng.random(n) < 0.5).astype(float)
        tau = _EFFECT + 0.10 * x0
        y = _CURVE * (x1**2 - 1.0) + tau * d + rng.normal(0.0, 1.0, n)
        cols = {"x0": x0, "x1": x1}
        plain = fit_cate(y, d, cols, interact=interact, alpha=_ALPHA)
        rich = fit_cate(y, d, cols, interact=interact, adjust=adjust, alpha=_ALPHA)
        ratios[rep] = rich.se / plain.se
    return ratios


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestRichAdjustmentBlockBuysWidth:
    """What the ``adjust=`` block is FOR, measured rather than asserted.

    The interaction block answers "who responds differently" and each
    column costs a degree of freedom in the joint Wald test; the
    adjustment block answers "what else moves the outcome" and costs the
    heterogeneity test nothing. Design: the conditional effect is linear in
    x0 only, while the prognostic surface bends hard in x1 with zero linear
    projection - adjusting for x1 is pure gain, interacting it would be waste.

    Measured (n=4000, 200 reps, ``knots=6`` on x1): mean SE ratio 0.490,
    worst 0.520. Bar is 0.6 - the adjustment must remove at least 40% of
    the ATE's standard error.
    """

    def test_hinge_adjustment_cuts_the_ate_standard_error(self):
        ratios = _adjustment_se_ratios(200, 4_000, 2043)
        mean, worst = float(ratios.mean()), float(ratios.max())

        assert mean <= 0.6, (
            f"a knots={_CURVE_KNOTS} adjustment block on the prognostic covariate left the "
            f"ATE standard error at {mean:.3f}x the plain interacted fit's (doc 0.490x); "
            f"the adjustment block has stopped absorbing the outcome's own structure"
        )
        assert worst <= 0.7, (
            f"worst-replication SE ratio {worst:.3f} (doc 0.520x): the mean gain is being "
            f"carried by some replications while others get nothing"
        )


def test_rich_adjustment_se_smoke():
    """Fast, unmarked 3-rep n=1000 variant of TestRichAdjustmentBlockBuysWidth."""
    ratios = _adjustment_se_ratios(3, 1_000, 2043)
    assert ratios.max() <= 0.7, (
        f"adjustment smoke: worst SE ratio {ratios.max():.3f} over 3 draws; the hinge "
        f"adjustment block is not buying width at all"
    )


# 14. THE ONE OUTCOME FAMILY NOBODY HAD MEASURED: Bernoulli.

# p(y=1 | x, d) = expit(_LOGIT_BASE + _LOGIT_SLOPE x + _LOGIT_EFFECT d).
_LOGIT_BASE = -0.40
_LOGIT_SLOPE = 0.80
_LOGIT_EFFECT = 0.50


@cache
def _binary_ate() -> float:
    """The true ATE on the probability scale, by Gauss-Hermite quadrature.

    Quadrature rather than a Monte-Carlo truth: the coverage this cell
    measures is a few tenths of a percent wide, and a simulated estimand
    would put its own noise inside that.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(128)
    weights = weights / weights.sum()
    linear = _LOGIT_BASE + _LOGIT_SLOPE * nodes
    return float(weights @ (expit(linear + _LOGIT_EFFECT) - expit(linear)))


def _binary_coverage(reps: int, n: int, seed: int) -> float:
    """ATE coverage on a Bernoulli outcome, against the quadrature truth."""
    truth = _binary_ate()
    rng = np.random.default_rng(seed)
    interact = [Covariate(name="x0")]
    cov = Coverage()
    for _ in range(reps):
        x = rng.normal(size=n)
        d = (rng.random(n) < 0.5).astype(float)
        p = expit(_LOGIT_BASE + _LOGIT_SLOPE * x + _LOGIT_EFFECT * d)
        y = (rng.random(n) < p).astype(float)
        result = fit_cate(y, d, {"x0": x}, interact=interact, alpha=_ALPHA)
        cov.record(result.lb <= truth <= result.ub)
    return cov.rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestBernoulliOutcomeCoverage:
    """Every other calibration figure in this file used a Gaussian outcome;
    this is the binary case, measured for the first time.

    Randomized treatment makes Lin's interacted regression consistent for
    the ATE regardless of outcome shape, and HC2 absorbs the mean-variance
    link a Bernoulli outcome forces on the two arms - this cell checks that
    argument survives contact with data.

    Measured (n=4000, 1000 reps, true ATE 0.1087): coverage 95.0% against
    nominal 95%. Band [92, 98]% is NOT to be widened; documented responses
    to a violation are HC3 or a documented refusal for binary outcomes.
    """

    def test_binary_outcome_ate_interval_covers(self):
        reps = 1_000
        coverage = _binary_coverage(reps, 4_000, 3141)
        mc_se = mcse(0.95, reps)

        assert 0.92 <= coverage <= 0.98, (
            f"Bernoulli ATE coverage {coverage:.1%} over {reps} replications against the "
            f"true {_binary_ate():.4f} (doc 95.0%), {abs(coverage - 0.95) / mc_se:.1f} MC-SE "
            f"from nominal. DO NOT widen this band: the documented responses are HC3 or a "
            f"refusal for binary outcomes, both design changes"
        )


def test_binary_outcome_coverage_smoke():
    """Fast, unmarked 15-rep n=1000 variant of TestBernoulliOutcomeCoverage."""
    coverage = _binary_coverage(15, 1_000, 3141)
    assert coverage >= 0.6, (
        f"binary-outcome smoke: {coverage:.0%} of 15 intervals covered the true "
        f"{_binary_ate():.4f}; the sandwich is broken on a 0/1 outcome"
    )


# 15. THE INTERSECTION: a Bernoulli outcome at 10/90 allocation.


def _binary_skewed_coverage(reps: int, n: int, seed: int) -> float:
    """ATE coverage on a Bernoulli outcome under 10/90 allocation.

    Same estimand as cell 14 (expectation over x alone), so allocation
    moves precision, not the target; reuses cell 14's quadrature truth.
    """
    truth = _binary_ate()
    rng = np.random.default_rng(seed)
    interact = [Covariate(name="x0")]
    cov = Coverage()
    for _ in range(reps):
        x = rng.normal(size=n)
        d = (rng.random(n) < _SKEW_SHARE).astype(float)
        p = expit(_LOGIT_BASE + _LOGIT_SLOPE * x + _LOGIT_EFFECT * d)
        y = (rng.random(n) < p).astype(float)
        result = fit_cate(y, d, {"x0": x}, interact=interact, alpha=_ALPHA)
        cov.record(result.lb <= truth <= result.ub)
    return cov.rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestBernoulliOutcomeCoverageAtSkewedAllocation:
    """Cells 12 and 14 crossed: the hardest configuration this estimator ships.

    Cell 12 skews allocation with arm variances chosen to differ; cell 14
    keeps allocation balanced and lets the Bernoulli outcome supply its own
    heteroskedasticity. Neither measures both at once - a 10% arm at
    n=4000 is 400 rows carrying the design's highest leverage, exactly
    where HC2's ``1/(1-h)`` weighting does its finite-sample correction,
    and where HC2 and HC3 visibly disagree.

    Measured (n=4000, 1000 reps, 10/90 allocation, true ATE 0.1087):
    coverage 94.8%. Band is cell 14's [92, 98]% and not to be widened here
    either.
    """

    def test_binary_outcome_at_ten_ninety_still_covers(self):
        reps = 1_000
        coverage = _binary_skewed_coverage(reps, 4_000, 2718)
        mc_se = mcse(0.95, reps)

        assert 0.92 <= coverage <= 0.98, (
            f"Bernoulli ATE coverage {coverage:.1%} over {reps} replications at "
            f"{_SKEW_SHARE:.0%}/{1 - _SKEW_SHARE:.0%} allocation against the true "
            f"{_binary_ate():.4f} (doc 94.8%), {abs(coverage - 0.95) / mc_se:.1f} MC-SE from "
            f"nominal. DO NOT widen this band: HC3 or a documented minimum arm size are the "
            f"responses, both design changes"
        )


def test_binary_outcome_skewed_allocation_smoke():
    """Fast, unmarked 15-rep variant of the cell above.

    100 treated rows of 1000 can't resolve 95% from 90%; catches an
    interval that's stopped being an interval on a thin binary arm.
    """
    coverage = _binary_skewed_coverage(15, 1_000, 2718)
    assert coverage >= 0.6, (
        f"binary skew-allocation smoke: {coverage:.0%} of 15 intervals covered the true "
        f"{_binary_ate():.4f} at a 10/90 split; the sandwich is not surviving a thin "
        f"binary arm at all"
    )


def _cluster_fit_diagnostic_cell(
    *, icc, members, clusters, high_leverage, treated_share, interactions, weighting, reps
):
    """Fixed-design Gaussian oracle; it records coverage but does not gate it.

    Coverage acceptance needs a separate, prospectively sized calibration.
    """
    rng = np.random.default_rng(2026091606)
    sizes = np.full(clusters, members)
    if high_leverage:
        sizes[::4] *= 3
    groups = np.repeat(np.arange(clusters), sizes)
    n = groups.size
    treated = max(2, int(clusters * treated_share))
    d = (groups >= clusters - treated).astype(float)
    x = rng.normal(size=(n, interactions))
    if high_leverage:
        x[groups == clusters - 1] += 6
    cols = {f"x{j}": x[:, j] for j in range(interactions)}
    covariates = [Covariate(name=name) for name in cols]
    weights = np.ones(n) if weighting == "member_count" else 1 / sizes[groups]
    basis = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    basis -= np.average(basis, weights=weights, axis=0)
    z = np.column_stack([np.ones(n), d, basis, d[:, None] * basis])
    h = z.T @ (weights[:, None] * z)
    bread = np.linalg.inv(h)
    weighted = weights[:, None] * z
    cluster_sums = np.stack([weighted[groups == g].sum(axis=0) for g in range(clusters)])
    independent_meat = weighted.T @ weighted
    shared_meat = cluster_sums.T @ cluster_sums
    true_covariance = bread @ ((1 - icc) * independent_meat + icc * shared_meat) @ bread

    # E[S_g S_g'] accounts for regression residuals, independently of the fit.
    expected_meat = np.zeros_like(h)
    for g in range(clusters):
        rows = groups == g
        h_g = z[rows].T @ weighted[rows]
        projection = h_g @ bread
        d_g = weighted[rows].T @ weighted[rows]
        c_g = np.outer(cluster_sums[g], cluster_sums[g])
        for fraction, local, total in [
            (1 - icc, d_g, independent_meat),
            (icc, c_g, shared_meat),
        ]:
            expected_meat += fraction * (
                local
                - local @ projection.T
                - projection @ local
                + projection @ total @ projection.T
            )
    expected_covariance = clusters / (clusters - 1) * bread @ expected_meat @ bread
    truth = 0.4
    mean_y = 2 + truth * d + basis.sum(axis=1)
    records = []
    failures: dict[str, int] = {}
    for _ in range(reps):
        error = np.sqrt(icc) * rng.normal(size=clusters)[groups] + np.sqrt(1 - icc) * rng.normal(
            size=n
        )
        try:
            result = fit_cate(
                mean_y + error,
                d,
                cols,
                interact=covariates,
                cluster_ids=groups,
                cluster_weight=weighting,
            )
        except InvalidRequestError as exc:
            failures[exc.code] = failures.get(exc.code, 0) + 1
            continue
        assert (result.dimension, result.n_clusters, result.reference_df) == (
            2 + 2 * interactions,
            clusters,
            clusters - 1,
        )
        records.append(
            (
                result.ate,
                result.se**2,
                float(result.lb <= truth <= result.ub),
                float(result.heterogeneity.p_value < 0.05),
            )
        )
    return np.asarray(records), failures, true_covariance[1, 1], expected_covariance[1, 1]


def test_cluster_calibration_oracle_smoke():
    records, failures, true_variance, expected_variance = _cluster_fit_diagnostic_cell(
        icc=0.2,
        members=5,
        clusters=10,
        high_leverage=False,
        treated_share=0.5,
        interactions=1,
        weighting="equal",
        reps=2,
    )
    assert failures == {}
    assert records.shape == (2, 4)
    assert np.isfinite(records).all()
    assert np.all(records[:, 1] > 0)
    assert true_variance > 0 and expected_variance > 0


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("icc", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("members", [5, 20, 100])
@pytest.mark.parametrize("clusters", [10, 40])
@pytest.mark.parametrize("high_leverage", [False, True])
@pytest.mark.parametrize("treated_share", [0.25, 0.5])
@pytest.mark.parametrize("interactions", [1, 2])
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_fit_covariance_moments_and_coverage_diagnostics(
    icc, members, clusters, high_leverage, treated_share, interactions, weighting, record_property
):
    """Moment regression and diagnostic MC bounds, not a coverage acceptance gate.

    The 288-cell manifest, seed and 128 draws are fixed before execution.
    Coverage/size acceptance needs a larger prespecified replication budget and gate.
    """
    from scipy.stats import binomtest

    reps = 128
    records, failures, true_variance, expected_variance = _cluster_fit_diagnostic_cell(
        icc=icc,
        members=members,
        clusters=clusters,
        high_leverage=high_leverage,
        treated_share=treated_share,
        interactions=interactions,
        weighting=weighting,
        reps=reps,
    )
    record_property("attempted", reps)
    record_property("estimable", len(records))
    record_property("failed_reasons", failures)
    assert failures == {}, failures
    assert records.shape == (reps, 4)
    assert np.isfinite(records).all()
    assert np.all(records[:, 1] > 0)
    mean = records[:, :2].mean(axis=0)
    mc_error = records[:, :2].std(axis=0, ddof=1) / np.sqrt(reps)
    # Gaussian fixed-design coefficients have known variance; variance estimates
    # are quadratic forms. These bounds check moments, not nominal t coverage.
    assert abs(mean[0] - 0.4) <= 6 * np.sqrt(true_variance / reps)
    assert abs(mean[1] - expected_variance) <= 6 * mc_error[1]
    record_property("true_ate_variance", float(true_variance))
    record_property("expected_reported_variance", float(expected_variance))
    record_property("mean_reported_variance", float(mean[1]))
    for column, name in [(2, "ate_coverage"), (3, "wald_null_rejection")]:
        count = int(records[:, column].sum())
        bound = binomtest(count, reps).proportion_ci(confidence_level=1 - 0.01 / (288 * 2))
        record_property(name, count / reps)
        record_property(f"{name}_mc_interval", (float(bound.low), float(bound.high)))


# Regression screens for cluster-honest held-out validation (GATES, CLAN, AUTOC, Qini);
# prospectively allocated release gates remain a separate calibration.
_C07_CELLS = 3 * 3 * 2 * 2 * 2
_C07_REPLICATIONS = 256
_C07_BOOTSTRAPS = 199
_C07_MC_TAIL = 0.01 / (_C07_CELLS * 14)


def _c07_null_validation(k, icc, weighting, imbalanced, stress, seed, *, repetitions, alpha):
    from increment.estimation.targeting import _Holdout, _validation

    # Separate training-ranking, heldout-data and resampling streams.
    train_seed, data_seed, bootstrap_seed = np.random.SeedSequence([707, seed]).spawn(3)
    train_rng, rng = np.random.default_rng(train_seed), np.random.default_rng(data_seed)
    sizes = np.resize([5, 20, 100], k) if imbalanced else np.full(k, 20)
    ids = np.repeat(np.arange(k), sizes)
    n = ids.size
    x = rng.normal(size=n)
    if stress:
        x[np.cumsum(sizes) - 1] *= 12
    score = train_rng.normal() * x

    def error(size):
        z = rng.normal(size=size)
        return (z**2 - 1) / math.sqrt(2) if stress else z

    # Oracle frozen scores have constant effect one; CLAN's independent profile has difference zero.
    psi = 1 + math.sqrt(icc) * error(k)[ids] + math.sqrt(1 - icc) * error(n)
    profile = math.sqrt(icc) * error(k)[ids] + math.sqrt(1 - icc) * error(n)
    holdout = _Holdout(
        n_train=n,
        score=score,
        y=psi,
        d=np.arange(n) % 2,
        cols={"spend": profile},
        covariates=(Covariate(name="spend"),),
        unit_ids=np.arange(n).astype(str),
        cluster_ids=ids,
        psi_fn=lambda y, d, X, unit_ids, cluster_ids: (y, np.ones(y.size, dtype=bool)),
        cluster_weight=weighting,
    )
    return _validation(
        holdout,
        psi,
        n_groups=2,
        alpha=alpha,
        arm_summary="score",
        bootstrap_seed=int(bootstrap_seed.generate_state(1)[0]),
        bootstrap_repetitions=repetitions,
    )


def _c07_record_calibration(ledger, result, *, failure=None):
    if failure is not None:
        for counts in ledger.values():
            counts["attempted"] += 1
            counts["failed"] += 1
            counts["reasons"][failure] = counts["reasons"].get(failure, 0) + 1
        return
    entries = [
        ("autoc", (result.autoc,), (0.0,)),
        ("qini", (result.qini,), (0.0,)),
        ("gates_1", (result.groups[0],), (1.0,)),
        ("gates_2", (result.groups[1],), (1.0,)),
        ("gates_family", result.groups, (1.0, 1.0)),
        ("clan", result.clan, (0.0,)),
    ]
    for name, rows, truths in entries:
        counts = ledger.setdefault(
            name,
            {
                "attempted": 0,
                "estimable": 0,
                "excluded": 0,
                "failed": 0,
                "covered": 0,
                "rejected": 0,
                "test_estimable": 0,
                "reasons": {},
            },
        )
        counts["attempted"] += 1
        if name in ("autoc", "qini") and rows[0].p_value is not None:
            counts["test_estimable"] += 1
            counts["rejected"] += int(rows[0].p_value < result.alpha)
        available = all(row.lb is not None and row.ub is not None for row in rows)
        if not available:
            counts["excluded"] += 1
            for row in rows:
                if row.lb is None or row.ub is None:
                    assert row.unavailable_reason is not None
                    reason = row.unavailable_reason
                    counts["reasons"][reason] = counts["reasons"].get(reason, 0) + 1
            continue
        counts["estimable"] += 1
        counts["covered"] += int(
            all(row.lb <= truth <= row.ub for row, truth in zip(rows, truths, strict=True))
        )


def _c07_calibration_report(ledger):
    return {
        name: {
            **counts,
            "availability": counts["estimable"] / counts["attempted"],
            "conditional_coverage": counts["covered"] / counts["estimable"]
            if counts["estimable"]
            else None,
            "unconditional_coverage": counts["covered"] / counts["attempted"],
            "conditional_rejection": counts["rejected"] / counts["test_estimable"]
            if counts["test_estimable"]
            else None,
            "unconditional_rejection": counts["rejected"] / counts["attempted"]
            if name in ("autoc", "qini")
            else None,
        }
        for name, counts in ledger.items()
    }


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_c07_cluster_calibration_smoke(weighting):
    """Exercise the DGP and availability accounting without claiming rate calibration."""
    result = _c07_null_validation(
        10,
        0.2,
        weighting,
        True,
        True,
        0,
        repetitions=39,
        alpha=0.2,
    )
    ledger = {}
    _c07_record_calibration(ledger, result)
    report = _c07_calibration_report(ledger)
    assert len(report) == 6
    for counts in report.values():
        assert counts["attempted"] == counts["estimable"] == 1
        assert counts["failed"] == counts["excluded"] == 0
        assert counts["availability"] == 1
        assert counts["conditional_coverage"] == counts["unconditional_coverage"]
    for row in (result.autoc, result.qini, *result.groups, *result.clan):
        assert row.bootstrap_valid_repetitions == 39
        assert row.se is not None and row.se > 0
    unavailable = result.model_copy(
        update={
            "autoc": result.autoc.model_copy(
                update={
                    "lb": None,
                    "ub": None,
                    "unavailable_reason": "estimation.targeting.bootstrap_zero_variance",
                }
            ),
        }
    )
    _c07_record_calibration(ledger, unavailable)
    _c07_record_calibration(ledger, None, failure="exception:ArithmeticError")
    counts = _c07_calibration_report(ledger)["autoc"]
    assert counts["attempted"] == 3
    assert counts["estimable"] == counts["excluded"] == counts["failed"] == 1
    assert counts["availability"] == 1 / 3
    assert counts["test_estimable"] == 2
    assert counts["conditional_coverage"] == report["autoc"]["conditional_coverage"]
    assert counts["unconditional_coverage"] == report["autoc"]["unconditional_coverage"] / 3
    assert counts["reasons"] == {
        "estimation.targeting.bootstrap_zero_variance": 1,
        "exception:ArithmeticError": 1,
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize(
    ("k", "icc", "weighting", "imbalanced", "stress"),
    [
        (40, 0.2, "equal", False, False),
        (40, 0.2, "equal", False, True),
        (40, 0.2, "equal", True, False),
        (40, 0.2, "equal", True, True),
    ],
    ids=[
        "balanced-gaussian",
        "balanced-skew_high_leverage",
        "unequal_sizes-gaussian",
        "unequal_sizes-skew_high_leverage",
    ],
)
def test_c07_cluster_null_rejection_and_coverage(
    k,
    icc,
    weighting,
    imbalanced,
    stress,
    record_property,
):
    """Family-aware binomial regression screen, not the release two-point precision gate.

    All cells/seeds/counts and the .01 family MC error allocation are fixed above.
    The 14 counts per cell cover six availability/coverage pairs and two rank tests.
    Small K retains its own availability denominator and cannot borrow large-K evidence.
    """
    import json

    from scipy.stats import binom

    ledger = {
        name: {
            "attempted": 0,
            "estimable": 0,
            "excluded": 0,
            "failed": 0,
            "covered": 0,
            "rejected": 0,
            "test_estimable": 0,
            "reasons": {},
        }
        for name in ("autoc", "qini", "gates_1", "gates_2", "gates_family", "clan")
    }
    for seed in range(_C07_REPLICATIONS):
        try:
            result = _c07_null_validation(
                k,
                icc,
                weighting,
                imbalanced,
                stress,
                seed,
                repetitions=_C07_BOOTSTRAPS,
                alpha=0.05,
            )
        except Exception as exc:
            _c07_record_calibration(
                ledger, None, failure=getattr(exc, "code", f"exception:{type(exc).__name__}")
            )
        else:
            _c07_record_calibration(ledger, result)
    report = _c07_calibration_report(ledger)
    record_property("c07_calibration", json.dumps(report, sort_keys=True))
    for name, counts in report.items():
        assert counts["attempted"] == _C07_REPLICATIONS
        assert counts["estimable"] + counts["excluded"] + counts["failed"] == counts["attempted"]
        assert counts["estimable"] == counts["attempted"], report
        # GATES intervals allocate alpha/2; joint coverage allocates alpha.
        error = 0.025 if name in ("gates_1", "gates_2") else 0.05
        max_misses = int(binom.isf(_C07_MC_TAIL, _C07_REPLICATIONS, error))
        assert counts["attempted"] - counts["covered"] <= max_misses, report
        if name in ("autoc", "qini"):
            assert counts["test_estimable"] == counts["attempted"], report
            assert counts["rejected"] <= max_misses, report


def _e6v0_original_cluster_source(members_per_cluster, *, intervention_grain="unit", n_clusters=80):
    """Archived 80-cluster DGP, also checked on its first 16 clusters."""
    from increment import Randomized
    from increment.frame import from_unit_summary

    rows = []
    for g in range(n_clusters):
        treated = g % 2
        x = (g - 39.5) / 20.0
        cluster_baseline = 0.12 * treated * x + 0.4 * np.sin(g)
        for j in range(members_per_cluster):
            rows.append(
                {
                    "unit": f"g{g}-u{j}",
                    "cluster": f"g{g}",
                    "arm": "treatment" if treated else "control",
                    "x": x,
                    "y": cluster_baseline + 0.01 * ((j % 11) - 5),
                }
            )
    table = pa.Table.from_pylist(rows)
    source = from_unit_summary(
        table,
        unit="unit",
        group="arm",
        control="control",
        metrics={"y": "mean"},
        cluster="cluster",
        intervention_grain=intervention_grain,
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    )
    return table, source


@pytest.mark.slow
@pytest.mark.parametrize("members_per_cluster", [5, 100])
@pytest.mark.parametrize("intervention_grain", ["unit", "cluster"])
@pytest.mark.parametrize("n_clusters", [16, 80])
def test_e6v0_original_varying_noise_cluster_public_replay(
    members_per_cluster, intervention_grain, n_clusters, record_property
):
    """Record actual public behavior, not a Monte Carlo error-control certificate."""
    import json

    from increment import ClusterBootstrap, select_targeting_rule, targeting_rule

    table, source = _e6v0_original_cluster_source(
        members_per_cluster, intervention_grain=intervention_grain, n_clusters=n_clusters
    )
    ids = np.asarray(table["cluster"])
    units = np.asarray(table["unit"])
    options: dict = {"control": "control", "interact": ["x"]}
    fit = estimate_cate(source, "y", **options)
    assert fit.n_clusters == n_clusters and fit.reference_df == n_clusters - 1
    assert fit.se > 0 and np.isfinite(fit.vcov).all()
    bootstrap = ClusterBootstrap(seed=0, repetitions=999)
    validation = validate_cate(source, "y", bootstrap=bootstrap, **options)
    rule = targeting_rule(source, "y", fraction=0.5, bootstrap=bootstrap, **options)
    # Four inner folds retain six training clusters for four fitted directions.
    selection_folds = 4 if n_clusters == 16 else 2
    selection = select_targeting_rule(
        source,
        "y",
        fractions=(0.0, 0.5, 1.0),
        seed=0,
        n_folds=selection_folds,
        bootstrap=bootstrap,
        **options,
    )
    assert validation.n_holdout + validation.n_train == units.size
    assert 0 < validation.n_holdout < units.size
    assert validation.n_clusters is not None and 0 < validation.n_clusters <= n_clusters
    assert validation.holdout_ate_se is not None and validation.holdout_ate_se > 0
    assert validation.autoc.se is not None and validation.autoc.se > 0
    assert validation.qini.se is not None and validation.qini.se > 0
    for row in (validation.autoc, validation.qini, *validation.groups, *validation.clan):
        assert row.n_clusters is not None
        assert row.uncertainty_method == "bootstrap-t+cluster-jackknife-t"
        if row.se is None or row.lb is None or row.ub is None:
            assert row.unavailable_reason is not None
        else:
            assert row.se > 0
    assert selection.n_clusters is not None
    assert selection.rule.n_clusters == n_clusters - selection.n_clusters
    for policy in (rule, selection.rule):
        assert policy.intervention_grain == policy.deploy_grain == intervention_grain
        actions = policy.predict({"x": np.asarray(table["x"])}, cluster_ids=ids)
        assert actions.shape == (n_clusters * members_per_cluster,)
        if intervention_grain == "cluster":
            for label in np.unique(ids):
                assert np.unique(actions[ids == label]).size == 1
    record_property(
        "e6v0_original_public_replay",
        json.dumps(
            {
                "witness": "archived_original_80" if n_clusters == 80 else "additional_first_16",
                "n_clusters": n_clusters,
                "members_per_cluster": members_per_cluster,
                "selection_folds": selection_folds,
                "fit": fit.model_dump(mode="json"),
                "fit_covariance": fit.vcov,
                "fit_coefficients": fit.beta,
                "fit_columns": fit.columns,
                "validation": validation.model_dump(mode="json"),
                "rule": rule.model_dump(mode="json"),
                "selection": selection.model_dump(mode="json"),
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )


@pytest.mark.parametrize("members_per_cluster", [5, 100])
@pytest.mark.parametrize("intervention_grain", ["unit", "cluster"])
def test_nested_selection_cannot_recover_rank_by_adding_cluster_members(
    members_per_cluster, intervention_grain
):
    from increment import select_targeting_rule
    from increment.errors import InvalidRequestError

    _, source = _e6v0_original_cluster_source(
        members_per_cluster, intervention_grain=intervention_grain, n_clusters=16
    )
    with pytest.raises(InvalidRequestError) as error:
        select_targeting_rule(
            source,
            "y",
            control="control",
            interact=["x"],
            fractions=(0.0, 0.5, 1.0),
            seed=0,
            n_folds=2,
        )
    assert error.value.code == "estimation.cate.single_cluster_direction"
