"""Full-tail fixed FCR calibration and conservative sequential projection."""

import math
from typing import Literal

import pytest
from scipy.stats import norm, t

from increment.errors import InvalidRequestError
from increment.estimation.armstats import ScoreStats
from increment.estimation.inference import infer_ate, infer_lift, normal_posterior
from increment.estimation.results import LiftEstimate, open_bound_from_two_sided_at_target


def _parent(
    alpha: float,
    alternative: Literal["greater", "less"],
    reference: Literal["normal", "cluster", "welch"] = "normal",
    scale: Literal["log", "relative", "absolute"] = "log",
) -> LiftEstimate:
    if scale != "log":
        return infer_ate(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            alpha=alpha / 2,
            alternative=alternative,
            point=2e6,
            scores=ScoreStats(metric="m", contrast="t", n=1, sum_psi=0, sum_psi2=1e12),
            value_scale=scale,
            dof=7 if reference == "cluster" else None,
        )
    return infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        alpha=alpha / 2,
        alternative=alternative,
        log_rr=0.3 - 0.1,
        se_t=0.05,
        se_c=0.04,
        dof=7 if reference == "cluster" else None,
        arm_ns=(8, 13) if reference == "welch" else None,
        abs_diff=2,
        abs_se=0.5,
    )


def _parameters(row: LiftEstimate) -> tuple[float, float]:
    e = row.require_lift()
    assert e.log_mean is not None
    assert e.log_se is not None
    if row.dof is not None or row.value_scale == "absolute":
        return e.log_mean, e.log_se
    posterior = normal_posterior(e.log_mean, e.log_se)
    return posterior.mu, posterior.sigma


def _working_point(row: LiftEstimate, point: float) -> float:
    """A working-scale point as the row's posterior-consuming methods read it."""
    return math.expm1(point) if row.scale == "log" else point


@pytest.mark.parametrize("alternative", ["greater", "less"])
@pytest.mark.parametrize(
    "reference,scale",
    [
        ("normal", "log"),
        ("cluster", "log"),
        ("welch", "log"),
        ("normal", "relative"),
        ("normal", "absolute"),
        ("cluster", "absolute"),
    ],
)
# 100-digit beta inversion: df=7, or Welch df from SEs .05/.04 and counts 8/13.
@pytest.mark.parametrize(
    ("alpha", "t_critical"),
    [
        (0.05, {"cluster": 1.8945786050900073517, "welch": 1.7515645272684550276}),
        (0.5, {"cluster": 0.0, "welch": 0.0}),
        (0.6, {"cluster": -0.26316686135202275215, "welch": -0.25782614339074287954}),
        (
            math.nextafter(0.5, 0),
            {"cluster": 1.4418801017855034724e-16, "welch": 1.4145222388423991969e-16},
        ),
        (
            math.nextafter(0.5, 1),
            {"cluster": -2.8837602035710069448e-16, "welch": -2.8290444776847983937e-16},
        ),
    ],
)
def test_fixed_fcr_exact_tail_and_recovery(alpha, t_critical, alternative, reference, scale):
    parent = _parent(alpha, alternative, reference, scale)
    row = open_bound_from_two_sided_at_target(parent)
    mu, sigma = _parameters(parent)
    crit = norm.isf(alpha) if reference == "normal" else t_critical[reference]
    expected = mu + (-1 if alternative == "greater" else 1) * crit * sigma
    if scale == "log":
        expected = math.expm1(expected)
    finite = row.require_lift().lb if alternative == "greater" else row.require_lift().ub
    assert finite == pytest.approx(expected, rel=2e-14, abs=1e-15)
    assert row.require_lift().open_side == ("upper" if alternative == "greater" else "lower")
    assert row.require_lift().alpha == alpha
    assert row.require_lift().level == math.fsum((1, -alpha))
    assert (row.abs_lb, row.abs_ub) == (parent.abs_lb, parent.abs_ub)
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    assert open_bound_from_two_sided_at_target(row) is row
    assert row.p_value() == pytest.approx(parent.p_value(), rel=2e-13)
    # Each row's posterior tails must reproduce its working-scale (mu, sigma). A
    # t row reads persisted statistics; inverting its endpoints with a Normal z
    # would inflate sigma by the t/z ratio.
    assert row.prob_beyond(_working_point(row, mu)) == pytest.approx(0.5, rel=2e-13)
    assert row.prob_beyond(_working_point(row, mu + sigma)) == pytest.approx(
        norm.sf(1.0), rel=2e-13
    )


@pytest.mark.parametrize("alternative", ["greater", "less"])
@pytest.mark.parametrize("reference", ["normal", "cluster", "welch"])
@pytest.mark.parametrize("tail_fraction", [0.75, 1.25])
def test_fixed_fcr_significance_matches_allocated_tail(alternative, reference, tail_fraction):
    alpha = 0.05
    parent = _parent(alpha, alternative, reference)
    mu, sigma = _parameters(parent)
    critical = (
        norm.isf(alpha * tail_fraction)
        if reference == "normal"
        else t.isf(alpha * tail_fraction, parent.reference_df)
    )
    null = math.expm1(mu + (-1 if alternative == "greater" else 1) * critical * sigma)
    parent = parent.model_copy(update={"null_lift": null})
    row = open_bound_from_two_sided_at_target(parent)

    assert row.p_value() == pytest.approx(alpha * tail_fraction, rel=1e-10)
    assert row.stat_sig() == (tail_fraction < 1)
    assert not parent.stat_sig()
    assert row.require_lift().alpha == alpha
    assert row.require_lift().level == 1 - alpha


@pytest.mark.parametrize("alpha", [1e-20, 1e-300, 2 * math.ulp(0.0)])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_fixed_fcr_tiny_normal_tail(alpha, alternative):
    parent = _parent(alpha, alternative)
    row = open_bound_from_two_sided_at_target(parent)
    mu, sigma = _parameters(parent)
    bound = row.require_lift().lb if alternative == "greater" else row.require_lift().ub
    assert bound is not None
    signed = (mu - math.log1p(bound)) if alternative == "greater" else (math.log1p(bound) - mu)
    assert signed / sigma == pytest.approx(norm.isf(alpha), rel=2e-12)
    assert row.require_lift().alpha == alpha
    assert row.require_lift().level == 1


@pytest.mark.parametrize("alpha", [0.05, 0.6])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_open_fixed_recovery_without_raw_statistics(alpha, alternative):
    parent = _parent(alpha, alternative)
    mu, sigma = _parameters(parent)
    bound = math.expm1(mu + (-1 if alternative == "greater" else 1) * norm.isf(alpha) * sigma)
    data = parent.model_dump()
    data["lift"].update(
        log_mean=None,
        log_se=None,
        open_side="upper" if alternative == "greater" else "lower",
        lb=bound if alternative == "greater" else None,
        ub=bound if alternative == "less" else None,
    )
    row = LiftEstimate.model_validate(data)
    # Without raw statistics the row still recovers (mu, sigma) from its own
    # interval: its posterior tails must reproduce the persisted moments.
    assert row.prob_beyond(_working_point(row, mu)) == pytest.approx(0.5, rel=2e-13)
    assert row.prob_beyond(_working_point(row, mu + sigma)) == pytest.approx(
        norm.sf(1.0), rel=2e-13
    )


@pytest.mark.parametrize("alpha", [0.5, math.nextafter(0.5, 0), math.nextafter(0.5, 1)])
def test_open_fixed_singular_recovery_requires_raw_statistics(alpha):
    row = open_bound_from_two_sided_at_target(_parent(alpha, "greater"))
    row = row.model_copy(
        update={"lift": row.require_lift().model_copy(update={"log_mean": None, "log_se": None})}
    )
    with pytest.raises(InvalidRequestError) as exc:
        row.prob_beyond(0.0)
    assert exc.value.code == "estimation.results.lift.open_interval_unrecoverable"


@pytest.mark.parametrize("alternative", ["greater", "less"])
@pytest.mark.slow
def test_sequential_fcr_preserves_exact_directional_inversion_and_refuses_posterior(alternative):
    from fractions import Fraction

    from increment.errors import CapabilityError
    from increment.estimation.sequential_runtime import estimate_sequential, reinvert_selected
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli(alternative=alternative)
    parent = estimate_sequential(snapshot, policy).results[0]
    row = reinvert_selected(parent, Fraction(1, 40), ceiling=Fraction(1, 40))
    assert open_bound_from_two_sided_at_target(row) is row
    assert row.inference == "always_valid"
    assert row.require_sequential_result().bounds.alpha == Fraction(1, 40)
    assert (
        row.require_sequential_result().checkpoint == parent.require_sequential_result().checkpoint
    )
    assert row.require_sequential_result().bounds.alternative == alternative
    with pytest.raises(CapabilityError) as raised:
        row.prob_beyond(0.0)
    assert raised.value.code == "sequential.route.unsupported"


def test_fcr_conversion_refuses_prior_shrunk_rows():
    parent = _parent(0.05, "greater").model_copy(update={"prior_shrunk": True})
    with pytest.raises(InvalidRequestError) as exc:
        open_bound_from_two_sided_at_target(parent)
    assert exc.value.code == "estimation.results.lift.fcr_prior_unsupported"


@pytest.mark.parametrize("reference", ["normal", "cluster", "welch"])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_fixed_fcr_can_recalibrate_parent_without_raw_statistics(reference, alternative):
    parent = _parent(0.6, alternative, reference)
    expected = open_bound_from_two_sided_at_target(parent)
    parent = parent.model_copy(
        update={"lift": parent.require_lift().model_copy(update={"log_mean": None, "log_se": None})}
    )
    actual = open_bound_from_two_sided_at_target(parent)
    assert actual.require_lift().lb == pytest.approx(expected.require_lift().lb)
    assert actual.require_lift().ub == pytest.approx(expected.require_lift().ub)


def test_fcr_conversion_without_interval_is_noop():
    parent = _parent(0.05, "greater")
    data = parent.model_dump()
    data["lift"].update(lb=None, ub=None, alpha=None, level=None)
    parent = LiftEstimate.model_validate(data)
    assert open_bound_from_two_sided_at_target(parent) is parent


@pytest.mark.parametrize("open_interval", [False, True])
def test_fixed_fcr_does_not_recover_allocation_from_level(open_interval):
    parent = _parent(0.05, "greater")
    if open_interval:
        parent = open_bound_from_two_sided_at_target(parent)
    parent = parent.model_copy(
        update={"lift": parent.require_lift().model_copy(update={"alpha": None})}
    )
    with pytest.raises(InvalidRequestError) as exc:
        if open_interval:
            parent.prob_beyond(0.0)
        else:
            open_bound_from_two_sided_at_target(parent)
    assert exc.value.code == "estimation.results.lift.open_interval_unrecoverable"


@pytest.mark.parametrize("alpha", [0.5, 0.6])
def test_additive_encouragement_recovery_keeps_near_flat_update(alpha):
    from increment.estimation.encouragement import _nn_estimate

    lift = _nn_estimate(2e6, 1e6, None, alpha / 2, "greater")
    parent = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        estimand="late",
        scale="linear",
        value_scale="absolute",
        alternative="greater",
        lift=lift,
    )
    row = open_bound_from_two_sided_at_target(parent)
    posterior = normal_posterior(2e6, 1e6)
    assert row.require_lift().lb == pytest.approx(posterior.mu - norm.isf(alpha) * posterior.sigma)
    assert row.prob_beyond(posterior.mu) == pytest.approx(0.5, rel=1e-12)
    assert row.prob_beyond(posterior.mu + posterior.sigma) == pytest.approx(norm.sf(1.0), rel=1e-12)


@pytest.mark.parametrize("with_stats", [False, True])
def test_fixed_fcr_rejects_inconsistent_parent(with_stats):
    parent = _parent(0.05, "greater")
    updates = {"log_se": -0.1} if with_stats else {"log_mean": None, "log_se": None, "value": -0.1}
    parent = parent.model_copy(update={"lift": parent.require_lift().model_copy(update=updates)})
    with pytest.raises(InvalidRequestError) as exc:
        open_bound_from_two_sided_at_target(parent)
    assert exc.value.code == "estimation.results.lift.liftestimate_interval_symmetric"


# 100-digit beta inversion at alpha=1e-20, df=7 or Welch df=336/19.
@pytest.mark.parametrize(
    ("reference", "critical"),
    [("cluster", 1445.7919808114368588), ("welch", 49.553868719652360663)],
)
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_fixed_fcr_tiny_t_tail(reference, critical, alternative):
    alpha = 1e-20
    parent = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.3 - 0.1,
        se_t=1e-5,
        se_c=1e-5,
        alpha=alpha / 2,
        alternative=alternative,
        dof=7 if reference == "cluster" else None,
        arm_ns=(8, 13) if reference == "welch" else None,
    )
    row = open_bound_from_two_sided_at_target(parent)
    mu, sigma = _parameters(parent)
    expected = math.expm1(mu + (-1 if alternative == "greater" else 1) * critical * sigma)
    assert (
        row.require_lift().lb if alternative == "greater" else row.require_lift().ub
    ) == pytest.approx(expected, rel=2e-14)
    assert row.require_lift().alpha == alpha
    assert row.require_lift().level == 1
