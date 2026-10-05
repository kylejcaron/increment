"""The absolute (additive) lift channel for the observational path.

Two things live here: the ADDITIVE PAIR that rides on every relative
observational row (``abs_diff``/``abs_se``/``abs_lb``/``abs_ub``), computed
from influence functions each contrast already held; and the per-metric
``value_scale={"metric": "absolute"}`` opt-in, which reports the additive
ATE in the metric's own units - the additive estimate remains available when
the control mean is indistinguishable from 0 and ``tau / mu0`` has no bounded
scalar interpretation.

``Y = 0.5X + 0.3D + N(0,1)``, so ``E[Y(0)] = 0`` is the near-zero-mu0
regime where relative inference has honest Fieller geometry while the additive
effect stays calibrated. Two misspecification flanks widen the Monte Carlo
suite past this well-specified blind spot.
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pyarrow as pa
import pytest
from scipy.special import expit

from increment.errors import IncrementWarning, InvalidRequestError
from increment.estimation._adjust.aipw import aipw_estimate
from increment.estimation._adjust.iptw import iptw_estimate
from increment.estimation.adjust import estimate_ate
from increment.estimation.armstats import ScoreStats
from increment.estimation.engine import Method
from increment.estimation.inference import Normal, infer_ate, infer_lift
from increment.estimation.results import LiftEstimate
from increment.frame import MetricsArg, MetricSpec, from_unit_summary
from increment.semantics.design import AdjustmentSet, Observational
from tests.warning_codes import warning_codes

_SEED_BASE = 20260811
_TRUE_TAU = 0.3
_N = 2000
_REPS = 400
_Z = 1.959963984540054  # z(0.975)
# MCSE of the SE/empirical-SD ratio is ~0.035 at R=400, so the band is ~3 MCSE
# wide; coverage's own MCSE at 0.95 is ~0.011.
_SE_SD_BAND = (0.90, 1.10)
_COVERAGE_BAND = (0.92, 0.98)
_METHODS = ("iptw", "dml", "aipw")

_DESIGN = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))


def _oracle_table(
    n: int = _N,
    seed: int = _SEED_BASE,
    *,
    flank: str = "oracle",
    shift: float = 0.0,
    scale: float = 1.0,
) -> pa.Table:
    """The pinned confounded DGP with ``E[Y(0)] = 0`` and true tau=0.3.

    ``flank="wrong-m"`` adds a quadratic to the outcome the fitted linear
    model cannot represent (propensity still right); ``flank="wrong-e"``
    adds one to the propensity index instead (outcome model still right).
    ``shift``/``scale`` are the location-equivariance and large-unit
    regression knobs - neither changes tau's identification, only its
    units.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    index = 0.8 * x + (0.6 * (x**2 - 1.0) if flank == "wrong-e" else 0.0)
    d = (rng.random(n) < expit(index)).astype(int)
    y = 0.5 * x + _TRUE_TAU * d + rng.normal(size=n)
    if flank == "wrong-m":
        y = y + 0.4 * (x**2 - 1.0)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y * scale + shift,
            "x": x,
        }
    )


def _src(table: pa.Table, metrics: MetricsArg | None = None):
    return from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=metrics or {"revenue": "mean"},
    )


def _oracle_src(**kwargs):
    return _src(_oracle_table(**kwargs))


def _posterior_sd(est: LiftEstimate) -> float:
    """The posterior sd the row's interval was cut from."""
    lift = est.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return (lift.ub - lift.lb) / (2 * _Z)


def _one(src, method: str, **kwargs) -> LiftEstimate:
    (est,) = estimate_ate(src, _DESIGN, methods=[Method(name=method)], **kwargs).results
    return est


def _absolute(src, method: str, **kwargs) -> LiftEstimate:
    return _one(src, method, value_scale={"revenue": "absolute"}, **kwargs)


# Row shape and the additive pair on relative rows


@pytest.mark.parametrize("method", _METHODS)
def test_relative_row_carries_the_additive_pair(method):
    """Every relative observational row now ships the absolute reading of
    the same contrast: the pair plus its Wald endpoints at the row's own
    level. Before this channel existed all four fields were None."""
    est = _one(_oracle_src(n=800, shift=10.0), method)
    assert est.value_scale == "relative"
    assert est.abs_diff is not None and est.abs_se is not None
    assert est.abs_se > 0
    assert est.abs_lb == pytest.approx(est.abs_diff - _Z * est.abs_se, rel=1e-15)
    assert est.abs_ub == pytest.approx(est.abs_diff + _Z * est.abs_se, rel=1e-15)
    # The relative lift is still tau/mu0 off the SAME tau.
    assert est.require_lift().value == pytest.approx(est.abs_diff / 10.0, rel=0.05)


@pytest.mark.parametrize("method", _METHODS)
def test_absolute_row_is_additive_native_and_carries_no_sidecar(method):
    """An absolute-native row replaces the reported axis rather than
    doubling it: `lift` IS the additive ATE, and abs_diff/abs_se stay None
    because they would only re-represent it (the encouragement-LATE
    precedent)."""
    est = _absolute(_oracle_src(n=800), method)
    assert est.value_scale == "absolute"
    assert est.scale == "linear"
    assert (est.abs_diff, est.abs_se, est.abs_lb, est.abs_ub, est.null_abs) == (
        None,
        None,
        None,
        None,
        None,
    )
    lift = est.require_lift()
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < lift.value < lift.ub
    # The additive-native contract is behavioral: the reported axis and
    # sidecar fields have the expected shape, while note wording remains an
    # implementation detail.


# Identities (design validation items 5-9)


@pytest.mark.parametrize("method", _METHODS)
def test_cross_scale_identity_relative_pair_equals_absolute_row(method):
    """Both reporting modes read the SAME (tau, se) out of the same
    influence functions, so the relative row's additive pair and the
    absolute row's posterior must agree. Toleranced, not `==`: these are
    two separate estimation calls, and the flat-prior contract is what
    makes them agree to float noise at all (measured <= 2e-16)."""
    src = _oracle_src(n=1000, shift=10.0)
    rel = _one(src, method)
    absolute = _absolute(src, method)
    assert rel.abs_diff == pytest.approx(absolute.require_lift().value, rel=1e-12)
    assert rel.abs_se == pytest.approx(_posterior_sd(absolute), rel=1e-12)


@pytest.mark.parametrize("method", _METHODS)
def test_flat_prior_is_exact_at_large_additive_units(method):
    """The flat-prior tripwire. Routing an absolute row through normal_posterior's
    'approximately flat' Normal(0, 1e6) shrinks it by 1/(1 + se^2/1e12) -
    unnoticeable on unitless lifts, 0.3% here at se(tau) ~ 5e4, and 50% at
    se 1e6. `prior=None` on an absolute row therefore skips the conjugate
    update entirely, which this pins: scaling Y by 1e6 must scale the
    reported effect by exactly 1e6."""
    base = _absolute(_oracle_src(n=1000), method)
    big = _absolute(_oracle_src(n=1000, scale=1e6), method)
    assert _posterior_sd(big) > 1e4  # the regime where the default prior bites
    assert big.require_lift().value == pytest.approx(base.require_lift().value * 1e6, rel=1e-12)
    assert _posterior_sd(big) == pytest.approx(_posterior_sd(base) * 1e6, rel=1e-12)


@pytest.mark.parametrize("method", _METHODS)
def test_location_equivariance_of_the_additive_effect(method):
    """Y -> Y+C moves both arm means by C and leaves (tau, se) alone. Not
    bitwise - Hajek means do not shift by exactly C in floats (measured
    drift <= 2e-14) - so this is a tolerance at rel 1e-12, checked on the
    absolute row and on the relative row's own additive pair at two nonzero
    shifts."""
    at_zero = _absolute(_oracle_src(n=1000), method)
    at_ten = _absolute(_oracle_src(n=1000, shift=10.0), method)
    assert at_ten.require_lift().value == pytest.approx(at_zero.require_lift().value, rel=1e-12)
    assert _posterior_sd(at_ten) == pytest.approx(_posterior_sd(at_zero), rel=1e-12)

    rel_ten = _one(_oracle_src(n=1000, shift=10.0), method)
    rel_twenty = _one(_oracle_src(n=1000, shift=20.0), method)
    assert rel_twenty.abs_diff == pytest.approx(rel_ten.abs_diff, rel=1e-12)
    assert rel_twenty.abs_se == pytest.approx(rel_ten.abs_se, rel=1e-12)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_aipw_additive_pair_reduces_to_iptw_at_a_null_outcome_model():
    """Extends the existing m==0 identity to the absolute channel. AIPW's
    IF(tau) and IPTW's differ by the constant tau there, which the CENTERED
    second moment removes - so both the point and the SE must coincide."""
    from tests.estimation.test_aipw import (
        _AIPW_DESIGN,
        _aipw_metric,
        _aipw_src,
        _OracleConstant,
        _ZeroOutcome,
    )

    src = _aipw_src(n=100)

    def propensity_factory():
        return _OracleConstant(0.5)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
            category=UserWarning,
        )
        (aipw_est,) = aipw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            _AIPW_DESIGN,
            propensity_learner=propensity_factory,
            outcome_learner=lambda: _ZeroOutcome(),
            folds=4,
        )
        (iptw_est,) = iptw_estimate(
            src, _aipw_metric(src, "revenue"), _AIPW_DESIGN, learner=propensity_factory
        )
    assert aipw_est.abs_diff is not None and aipw_est.abs_se is not None
    assert aipw_est.abs_diff == pytest.approx(iptw_est.abs_diff, rel=0, abs=1e-9)
    assert aipw_est.abs_se == pytest.approx(iptw_est.abs_se, rel=0, abs=1e-9)


@pytest.mark.parametrize("method", _METHODS)
def test_one_sided_request_computes_abs_endpoints_at_doubled_alpha(method):
    """Both additive and signed relative bounds use the full directional tail."""
    src = _oracle_src(n=800, shift=10.0)
    one = _one(src, method, alternative="greater", alpha=0.05)
    two = _one(src, method, alpha=0.10)
    assert one.abs_lb is not None and one.abs_ub is not None
    assert one.abs_lb == pytest.approx(two.abs_lb, rel=1e-12)
    assert one.abs_ub == pytest.approx(two.abs_ub, rel=1e-12)
    one_set = one.relative_confidence_set
    two_set = two.relative_confidence_set
    assert one_set is not None and two_set is not None
    assert one_set.alpha == pytest.approx(0.05)
    assert two_set.alpha == pytest.approx(0.10)
    # The doubled level makes this interval NARROWER than the default 95%
    # one; un-doubled alpha would silently widen a one-sided guardrail read.
    default = _one(src, method)
    assert default.relative_confidence_set is not None
    assert one_set.intervals == ((two_set.intervals[0][0], None),)
    assert default.abs_lb is not None and default.abs_ub is not None
    assert one.abs_ub - one.abs_lb < default.abs_ub - default.abs_lb


# The near-zero-mu0 additive rescue (design validation item 4)


@pytest.mark.parametrize("method", _METHODS)
def test_relative_request_returns_additive_output_and_honest_joint_set(method):
    """Near-zero control means the relative Fieller set need not be a
    bounded interval, but the additive contrast remains estimable."""
    est = _one(_oracle_src(n=1000), method)
    assert est.abs_diff is not None and est.abs_se is not None
    assert est.abs_lb == pytest.approx(est.abs_diff - _Z * est.abs_se, rel=1e-12)
    assert est.abs_ub == pytest.approx(est.abs_diff + _Z * est.abs_se, rel=1e-12)
    relative = est.relative_confidence_set
    assert relative is not None
    assert relative.geometry in ("disconnected", "one_sided", "all_real")


@pytest.mark.parametrize("method", _METHODS)
def test_absolute_request_ships_a_row_on_the_data_the_guard_refuses(method):
    """The additive effect remains available on near-zero-control data."""
    est = _absolute(_oracle_src(n=1000), method)
    assert est.require_lift().value == pytest.approx(_TRUE_TAU, abs=0.15)
    lift = est.require_lift()
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < lift.value < lift.ub


def test_one_call_serves_a_multi_metric_source_with_one_metric_absolute():
    """The defect the call-global opt-in had: `estimate_ate` iterates ALL
    metrics per method and does not catch the mu0 guard's ValueError, so a
    near-zero metric aborted every sibling. Keyed per metric, the near-zero
    metric goes absolute and the normal metric keeps its relative row -
    one call, no abort, no warning."""
    table = _oracle_table(n=1000)
    revenue = np.asarray(table["revenue"].to_numpy()) + 10.0
    table = table.append_column("healthy", pa.array(revenue))
    src = _src(table, metrics={"revenue": "mean", "healthy": "mean"})

    with pytest.warns() as record:
        rows = estimate_ate(
            src,
            _DESIGN,
            methods=[Method(name="iptw")],
            value_scale={"revenue": "absolute"},
        ).results
        # Nothing this call does should warn; assert against an intentional
        # sentinel so `pytest.warns` itself has something to catch.
        import warnings

        warnings.warn("sentinel", UserWarning, stacklevel=1)
    assert [str(w.message) for w in record] == ["sentinel"]

    by_metric = {row.metric: row for row in rows}
    assert set(by_metric) == {"revenue", "healthy"}
    assert by_metric["revenue"].value_scale == "absolute"
    assert by_metric["healthy"].value_scale == "relative"
    # Same underlying tau on both rows - one shifted by 10, one not.
    assert by_metric["healthy"].abs_diff == pytest.approx(
        by_metric["revenue"].require_lift().value, rel=1e-12
    )


def test_decision_mappings_resolve_per_metric():
    """`estimate_ate` runs once for ALL metrics, so its decision inputs are
    name-keyed rather than scalar - a per-metric guardrail must land on its
    own metric only, and metrics the mappings do not name keep the call's
    defaults."""
    table = _oracle_table(n=800, shift=10.0)
    table = table.append_column("other", pa.array(np.asarray(table["revenue"].to_numpy()) * 2.0))
    src = _src(table, metrics={"revenue": "mean", "other": "mean"})
    rows = {
        row.metric: row
        for row in estimate_ate(
            src,
            _DESIGN,
            methods=[Method(name="iptw")],
            null_lifts={"revenue": 0.05},
            null_abs={"other": 0.25},
            alternatives={"other": "greater"},
        ).results
    }
    assert rows["revenue"].null_lift == pytest.approx(0.05)
    assert rows["revenue"].null_abs is None
    assert rows["revenue"].alternative == "two-sided"
    assert rows["other"].null_abs == pytest.approx(0.25)
    assert rows["other"].alternative == "greater"
    assert rows["other"].null_lift == 0.0
    # Joint relative evidence records the requested alpha directly; it is
    # not the old scalar 1-2*alpha display level.
    other_set = rows["other"].relative_confidence_set
    revenue_set = rows["revenue"].relative_confidence_set
    assert other_set is not None and revenue_set is not None
    assert other_set.alpha == pytest.approx(0.05)
    assert revenue_set.alpha == pytest.approx(0.05)
    assert rows["other"].abs_lb is not None and rows["other"].abs_ub is not None


# Prior-scale contract (design validation item 12)


def test_prior_refused_on_a_mixed_scale_call():
    """One scalar prior cannot be a unitless relative lift on one row and a
    metric-units additive effect on another simultaneously."""
    table = _oracle_table(n=600)
    table = table.append_column("healthy", pa.array(np.asarray(table["revenue"].to_numpy()) + 10.0))
    src = _src(table, metrics={"revenue": "mean", "healthy": "mean"})
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_ate(
            src,
            _DESIGN,
            value_scale={"revenue": "absolute"},
            prior=Normal(mu=0.0, sigma=1.0),
        )
    assert exc_info.value.code == "estimation.adjust.prior_interpreted_call"


def test_uniform_absolute_call_honors_an_explicit_prior_in_additive_units():
    """A single-metric absolute call is scale-unambiguous, so an explicit
    prior is honored - and it must be the exact conjugate update in the
    metric's own units, not the skipped-update flat path."""
    src = _oracle_src(n=1000)
    flat = _absolute(src, "iptw")
    prior = Normal(mu=0.0, sigma=0.05)
    shrunk = _absolute(src, "iptw", prior=prior)

    se = _posterior_sd(flat)
    variance = 1.0 / (1.0 / prior.sigma**2 + 1.0 / se**2)
    expected_mu = variance * (prior.mu / prior.sigma**2 + flat.require_lift().value / se**2)
    assert shrunk.require_lift().value == pytest.approx(expected_mu, rel=1e-12)
    assert _posterior_sd(shrunk) == pytest.approx(np.sqrt(variance), rel=1e-12)
    assert abs(shrunk.require_lift().value) < abs(
        flat.require_lift().value
    )  # the prior actually bit


def test_uniform_absolute_multi_metric_prior_warns_about_incommensurable_units():
    """Scale-uniform but still hazardous: one Normal spanning several
    metrics' own additive units (ms on one row, dollars on another)."""
    table = _oracle_table(n=600)
    table = table.append_column("second", pa.array(np.asarray(table["revenue"].to_numpy())))
    src = _src(table, metrics={"revenue": "mean", "second": "mean"})
    with pytest.warns(IncrementWarning) as rec:
        estimate_ate(
            src,
            _DESIGN,
            value_scale={"revenue": "absolute", "second": "absolute"},
            prior=Normal(mu=0.0, sigma=1.0),
        )
    assert "estimation.adjust.prior_absolute_scale_spans_metrics" in warning_codes(rec)


def test_the_multi_metric_prior_advisory_names_the_estimate_ate_caller():
    table = _oracle_table(n=600)
    table = table.append_column("second", pa.array(np.asarray(table["revenue"].to_numpy())))
    src = _src(table, metrics={"revenue": "mean", "second": "mean"})
    with pytest.warns(IncrementWarning) as rec:
        estimate_ate(
            src,
            _DESIGN,
            value_scale={"revenue": "absolute", "second": "absolute"},
            prior=Normal(mu=0.0, sigma=1.0),
        )
    advisories = [
        w
        for w in rec
        if getattr(w.message, "code", None)
        == "estimation.adjust.prior_absolute_scale_spans_metrics"
    ]
    assert advisories
    assert all(w.filename == __file__ for w in advisories)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_the_skipped_metric_advisory_names_the_estimate_ate_caller():
    table = _oracle_table(n=200)
    table = table.append_column("orders", pa.array(np.ones(200)))
    src = _src(
        table,
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
        ],
    )
    with pytest.warns(IncrementWarning) as rec:
        estimate_ate(src, _DESIGN, value_scale={"revenue": "absolute"})
    advisories = [
        w
        for w in rec
        if getattr(w.message, "code", None) == "estimation.adjust.skip_unsupported_metric"
    ]
    assert advisories
    assert all(w.filename == __file__ for w in advisories)


def test_prior_shared_false_skips_the_mixed_scale_refusal():
    """Two metrics on different value_scale (one relative, one absolute)
    each carrying their OWN declared prior must not trip the cross-
    metric scale-commensurability refusal - that refusal exists to
    catch ONE scalar prior spanning several metrics' units, which does
    not apply when `readouts.run` resolved each metric's prior
    independently (`prior_shared=False`)."""
    table = _oracle_table(n=600)
    table = table.append_column("second", pa.array(np.asarray(table["revenue"].to_numpy()) + 10.0))
    src = _src(table, metrics={"revenue": "mean", "second": "mean"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = estimate_ate(
            src,
            _DESIGN,
            value_scale={"revenue": "absolute"},
            prior=Normal(mu=0.0, sigma=1.0),
            prior_shared=False,
        ).results
    assert {r.metric: r.value_scale for r in rows} == {
        "revenue": "absolute",
        "second": "relative",
    }


def test_prior_shared_default_true_still_raises_on_mixed_scale():
    """The default `prior_shared=True` (every direct `estimate_ate`
    caller's own scalar `prior=`) keeps refusing a mixed relative/
    absolute call - unchanged existing contract."""
    table = _oracle_table(n=600)
    table = table.append_column("second", pa.array(np.asarray(table["revenue"].to_numpy()) + 10.0))
    src = _src(table, metrics={"revenue": "mean", "second": "mean"})
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_ate(
            src,
            _DESIGN,
            value_scale={"revenue": "absolute"},
            prior=Normal(mu=0.0, sigma=1.0),
        )
    assert exc_info.value.code == "estimation.adjust.prior_interpreted_call"


def test_metrics_subset_scopes_the_unadjusted_absolute_refusal():
    """`metrics=` narrows what this call estimates, so an absolute request
    naming a metric OUTSIDE that subset must not trip the
    unadjusted-cannot-be-absolute refusal - the refusal compares
    `value_scale` against THIS call's `methods`, and a metric this call
    never touches is not this call's problem. Regression: grouped
    dispatch in `readouts.run` sends one call per resolved-methods group,
    so a sibling group's absolute metric spuriously refused the
    unadjusted group."""
    table = _oracle_table(n=600)
    # `second` = revenue + 10 has mu0 ~ 10, identified on the relative scale;
    # oracle `revenue` has E[Y(0)] ~ 0, hence the legitimate absolute ask.
    table = table.append_column("second", pa.array(np.asarray(table["revenue"].to_numpy()) + 10.0))
    src = _src(table, metrics={"revenue": "mean", "second": "mean"})
    second = next(m for m in src.context.metrics if m.name == "second")

    rows = estimate_ate(
        src,
        _DESIGN,
        methods=[Method(name="unadjusted")],
        metrics=[second],
        value_scale={"revenue": "absolute"},
    ).results
    assert [r.metric for r in rows] == ["second"]
    assert rows[0].value_scale == "relative"


def test_metrics_subset_scopes_the_shared_prior_scale_uniformity_refusal():
    """Same scoping for the shared-prior uniformity judgment: one scalar
    `prior=` spans only the rows THIS call emits, so an absolute metric
    outside `metrics=` must not make the call look mixed-scale."""
    table = _oracle_table(n=600)
    table = table.append_column("second", pa.array(np.asarray(table["revenue"].to_numpy()) + 10.0))
    src = _src(table, metrics={"revenue": "mean", "second": "mean"})
    second = next(m for m in src.context.metrics if m.name == "second")

    rows = estimate_ate(
        src,
        _DESIGN,
        methods=[Method(name="iptw")],
        metrics=[second],
        value_scale={"revenue": "absolute"},
        prior=Normal(mu=0.0, sigma=1.0),
    ).results
    assert [r.metric for r in rows] == ["second"]
    assert rows[0].value_scale == "relative"


# Refusal matrix (design validation item 13)


class TestRefusalMatrix:
    def test_mu0_exactly_zero_keeps_set_only_relative_and_additive_output(self):
        table = pa.table(
            {
                "user_id": [f"u{i}" for i in range(6)],
                "variant": ["T", "T", "T", "C", "C", "C"],
                "revenue": [3.0, 5.0, 4.0, 0.0, 0.0, 0.0],
                "x": [1.0] * 6,
            }
        )
        src = _src(table)
        (result,) = estimate_ate(src, _DESIGN, methods=[Method(name="iptw")]).results
        assert result.lift is None
        assert result.relative_confidence_set is not None
        assert result.relative_confidence_set.geometry == "empty"
        assert result.abs_diff == pytest.approx(4)
        assert result.abs_lb is not None and result.abs_lb > 0

    @pytest.mark.parametrize(
        "mapping",
        [
            {"value_scale": {"revenu": "absolute"}},
            {"null_lifts": {"revenu": 0.05}},
            {"null_abs": {"revenu": 0.05}},
            {"alternatives": {"revenu": "greater"}},
        ],
    )
    def test_every_per_metric_mapping_key_must_name_a_declared_metric(self, mapping):
        """A typo'd key would silently drop the request it meant to make -
        the same anti-typo rationale the margin override maps carry."""
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(_oracle_src(n=200), _DESIGN, **mapping)
        assert exc_info.value.code == "estimation.adjust.names_metrics_source"

    def test_value_scale_value_must_be_a_known_scale(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(_oracle_src(n=200), _DESIGN, value_scale={"revenue": "abs"})  # ty: ignore[invalid-argument-type]  - the bad literal IS the case under test
        assert exc_info.value.code == "estimation.adjust.value_scale_relative"

    @pytest.mark.parametrize(
        ("specs", "name", "kind"),
        [
            (
                [
                    MetricSpec(name="revenue", type="mean"),
                    MetricSpec(name="p90", type="quantile", value_column="revenue", quantile=0.9),
                ],
                "p90",
                "quantile",
            ),
            (
                [
                    MetricSpec(name="revenue", type="mean"),
                    MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
                ],
                "rpo",
                "ratio",
            ),
        ],
    )
    def test_value_scale_on_a_metric_no_adjustment_supports_refuses_by_name(
        self, specs, name, kind
    ):
        """An explicitly opted-in metric must never degrade into the
        skip-with-warning channel reserved for capability gaps: the caller
        named it on purpose and would otherwise get a silently missing row."""
        table = _oracle_table(n=200)
        table = table.append_column("orders", pa.array(np.ones(200)))
        src = _src(table, metrics=specs)
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(src, _DESIGN, value_scale={name: "absolute"})
        assert exc_info.value.code == "estimation.adjust.value_scale_names"
        assert exc_info.value.context["metric_type"] == kind

    def test_observational_quantile_refusal_precedes_shared_prior_advisories(self):
        """A quantile in the call is refused by its stable code before the shared-prior
        advisory for the absolute-scale means can fire, so warnings-as-errors still see it."""
        from increment.errors import UnsupportedRequestError

        table = _oracle_table(n=200)
        table = table.append_column("signups", pa.array(table["revenue"].to_numpy() * 0.5))
        src = _src(
            table,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="signups", type="mean"),
                MetricSpec(name="p90", type="quantile", value_column="revenue", quantile=0.9),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UnsupportedRequestError) as exc_info:
                estimate_ate(
                    src,
                    _DESIGN,
                    value_scale={"revenue": "absolute", "signups": "absolute"},
                    prior=Normal(mu=0.0, sigma=10.0),
                )
        assert exc_info.value.code == "readout.observational.quantile"

    def test_observational_quantile_refusal_precedes_method_scale_judgment(self):
        """Direct estimate_ate agrees with the readout seam: a quantile no listed method can
        estimate is refused before a shared prior is judged against those methods' scales."""
        from increment.errors import UnsupportedRequestError

        src = _src(
            _oracle_table(n=200),
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="p90", type="quantile", value_column="revenue", quantile=0.9),
            ],
        )
        with pytest.raises(UnsupportedRequestError) as exc_info:
            estimate_ate(
                src,
                _DESIGN,
                methods=[Method(name="unadjusted"), Method(name="iptw")],
                prior=Normal(mu=0.0, sigma=10.0),
            )
        assert exc_info.value.code == "readout.observational.quantile"

    def test_percentile_winsorization_refusal_precedes_shared_prior_advisories(self):
        """The static percentile-winsorization refusal is not pre-empted by the shared-prior
        advisory, so warnings-as-errors still see the coded capability error."""
        from increment.errors import CapabilityError

        table = _oracle_table(n=200)
        table = table.append_column("signups", pa.array(table["revenue"].to_numpy() * 0.5))
        src = _src(
            table,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="signups", type="mean"),
                MetricSpec(
                    name="capped",
                    type="mean",
                    value_column="revenue",
                    winsorization={"upper_percentile": 0.99},
                ),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(CapabilityError) as exc_info:
                estimate_ate(
                    src,
                    _DESIGN,
                    value_scale={
                        "revenue": "absolute",
                        "signups": "absolute",
                        "capped": "absolute",
                    },
                    prior=Normal(mu=0.0, sigma=10.0),
                )
        assert exc_info.value.code == "adjust.winsorization.percentile_unsupported"

    def test_absolute_row_cannot_be_targeted_by_null_abs_at_estimate_ate(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(
                _oracle_src(n=200),
                _DESIGN,
                value_scale={"revenue": "absolute"},
                null_abs={"revenue": 0.1},
            )
        assert (
            exc_info.value.code
            == "estimation.adjust.resolve_value_scales_null_abs_on_absolute_metric"
        )
        assert exc_info.value.context["metric"] == "revenue"

    def test_infer_ate_carries_the_same_invariant_guard(self):
        """The entry check is at `estimate_ate`, but `infer_ate` is reachable
        directly - without this guard a future caller could stamp an
        absolute margin on a row whose abs fields are None, which
        `_stat_sig` reads as a silent False."""
        scores = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=100.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="T",
                method="iptw",
                method_role="decision",
                point=0.3,
                scores=scores,
                value_scale="absolute",
                null_abs=0.1,
            )
        assert exc_info.value.code == "estimation.inference.infer_ate_null_abs_on_absolute_metric"

    def test_infer_ate_refuses_a_zero_additive_standard_error(self):
        """infer_lift refuses abs_se <= 0 (no additive interval without additive
        uncertainty); infer_ate must refuse the same input instead of publishing a
        zero-width interval that reads stat_sig=True and crashes p_value()."""
        scores = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=4.0)

        with pytest.raises(InvalidRequestError) as lift_exc:
            infer_lift(
                metric="m",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05 - 0.0,
                se_t=0.02,
                se_c=0.02,
                abs_diff=2.0,
                abs_se=0.0,
                null_abs=1.0,
            )
        assert lift_exc.value.code == "estimation.inference.infer_lift_abs_se_positive"

        with pytest.raises(InvalidRequestError) as ate_exc:
            infer_ate(
                metric="m",
                group_id="T",
                method="iptw",
                method_role="decision",
                point=0.05,
                scores=scores,
                abs_diff=2.0,
                abs_se=0.0,
                null_abs=1.0,
            )
        assert ate_exc.value.code == "estimation.inference.infer_lift_abs_se_positive"

    def test_absolute_row_cannot_be_targeted_by_a_nonzero_null_lift(self):
        """The mirror of the null_abs refusal: a RELATIVE null on an
        additive row would be compared against an additive interval by
        _stat_sig/prob_favorable - a silent scale mismatch."""
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(
                _oracle_src(n=200),
                _DESIGN,
                value_scale={"revenue": "absolute"},
                null_lifts={"revenue": 0.05},
            )
        assert (
            exc_info.value.code
            == "estimation.adjust.resolve_value_scales_null_lift_on_absolute_metric"
        )
        assert exc_info.value.context["metric"] == "revenue"

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_absolute_row_accepts_a_zero_null_lift(self):
        """Testing against no effect is scale-agnostic, so the explicit
        zero stays legal (and is what every absolute row already carries)."""
        rows = estimate_ate(
            _oracle_src(n=200),
            _DESIGN,
            methods=[Method(name="aipw")],
            value_scale={"revenue": "absolute"},
            null_lifts={"revenue": 0.0},
        ).results
        assert [r.null_lift for r in rows] == [0.0]

    def test_infer_ate_guards_a_nonzero_null_lift_on_an_absolute_row(self):
        scores = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=100.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="T",
                method="iptw",
                method_role="decision",
                point=0.3,
                scores=scores,
                value_scale="absolute",
                null_lift=0.05,
            )
        assert exc_info.value.code == "estimation.inference.infer_ate_null_lift_on_absolute_metric"

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_prior_warning_counts_only_estimable_metrics(self):
        """The multi-metric prior warning must count row-producing metrics:
        with two absolute means plus an always-skipped ratio it names 2, not
        3, and lists only the two mean names."""
        table = _oracle_table(n=200)
        table = table.append_column("orders", pa.array(np.ones(200)))
        table = table.append_column("signups", pa.array(table["revenue"].to_numpy() * 0.5))
        src = _src(
            table,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="signups", type="mean"),
                MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
            ],
        )
        with pytest.warns(IncrementWarning) as caught:
            estimate_ate(
                src,
                _DESIGN,
                methods=[Method(name="aipw")],
                value_scale={"revenue": "absolute", "signups": "absolute"},
                prior=Normal(mu=0.0, sigma=10.0),
            )
        spans = [
            w.message
            for w in caught
            if isinstance(w.message, IncrementWarning)
            and w.message.code == "estimation.adjust.prior_absolute_scale_spans_metrics"
        ]
        assert spans
        assert "rpo" not in cast("list[str]", spans[0].context["names"])

    def test_unadjusted_method_cannot_honor_value_scale(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(
                _oracle_src(n=200),
                _DESIGN,
                methods=[Method(name="unadjusted")],
                value_scale={"revenue": "absolute"},
            )
        assert exc_info.value.code == "estimation.adjust.method_name_unadjusted"

    def test_unadjusted_method_accepts_an_all_relative_value_scale(self):
        """The hazard is an absolute-native unadjusted row, not the mapping's
        mere presence: an all-'relative' mapping is the default no-op and
        must not be refused with a message asserting an absolute request.
        Needs a shifted (positive-mean) fixture - the oracle DGP's
        E[Y(0)]=0 makes the unadjusted log-scale path unestimable anyway."""
        rows = estimate_ate(
            _src(_oracle_table(n=200, shift=10.0)),
            _DESIGN,
            methods=[Method(name="unadjusted")],
            value_scale={"revenue": "relative"},
        ).results
        assert rows and all(r.value_scale == "relative" for r in rows)

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_prior_survives_an_always_skipped_ratio_metric(self):
        """Scale-uniformity is judged over metrics that can produce a row.
        A declared ratio metric is skipped by every adjustment, so it must
        not drag a uniform absolute call into the mixed-scale refusal."""
        table = _oracle_table(n=200)
        table = table.append_column("orders", pa.array(np.ones(200)))
        src = _src(
            table,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
            ],
        )
        with pytest.warns(IncrementWarning) as rec:
            rows = estimate_ate(
                src,
                _DESIGN,
                methods=[Method(name="aipw")],
                value_scale={"revenue": "absolute"},
                prior=Normal(mu=0.0, sigma=10.0),
            ).results
        assert "estimation.adjust.skip_unsupported_metric" in warning_codes(rec)
        assert [(r.metric, r.value_scale) for r in rows] == [("revenue", "absolute")]

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_ratio_metric_without_a_value_scale_key_still_skips_with_a_warning(self):
        """The abs channel does NOT extend estimator coverage: an
        un-opted-in ratio metric keeps the pre-existing capability-gap
        behaviour (skip the metric, keep its siblings)."""
        table = _oracle_table(n=200)
        table = table.append_column("orders", pa.array(np.ones(200)))
        src = _src(
            table,
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
            ],
        )
        with pytest.warns(IncrementWarning) as rec:
            rows = estimate_ate(src, _DESIGN, value_scale={"revenue": "absolute"}).results
        assert "estimation.adjust.skip_unsupported_metric" in warning_codes(rec)
        assert [r.metric for r in rows] == ["revenue"]

    def test_zero_variance_influence_function_refuses(self):
        scores = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=0.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="T",
                method="iptw",
                method_role="decision",
                point=0.3,
                scores=scores,
                value_scale="absolute",
            )
        assert exc_info.value.code == "estimation.inference.degenerate_data_zero"

    def test_unknown_value_scale_refused_at_infer_ate(self):
        scores = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=100.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="T",
                method="iptw",
                method_role="decision",
                point=0.3,
                scores=scores,
                value_scale="ABSOLUTE",  # ty: ignore[invalid-argument-type]  - the bad literal IS the case under test)
            )
        assert exc_info.value.code == "estimation.inference.unknown_value_scale"


# Note composition and serialization (design validation items 14, 15)


def test_absolute_note_composes_with_the_impute_caveat_and_never_replaces_it():
    """Two independent caveats ride the same field. A plain overwrite would
    drop the impute-indicator disclosure - the reason the absolute caveat
    is appended, not assigned."""
    table = _oracle_table(n=600)
    x = np.array(table["x"].to_numpy(), dtype=float)
    x[:20] = np.nan
    table = table.set_column(table.schema.get_field_index("x"), "x", pa.array(x))
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",), missing="impute-indicator"),
    )
    (est,) = estimate_ate(
        _src(table),
        design,
        methods=[Method(name="iptw")],
        value_scale={"revenue": "absolute"},
    ).results
    assert est.note is not None
    impute_fragment = "missing covariate values pooled-mean imputed"
    absolute_fragment = "absolute"
    assert impute_fragment in est.note
    assert absolute_fragment in est.note
    assert est.note.index(impute_fragment) < est.note.index(absolute_fragment)


def test_absolute_row_round_trips_through_serialization():
    est = _absolute(_oracle_src(n=600), "iptw")
    restored = LiftEstimate.model_validate(est.model_dump())
    assert restored == est
    assert restored.value_scale == "absolute"
    assert restored.scale == "linear"


# Monte Carlo calibration. Oracle DGP is the blind spot for the historic
# AIPW missing-centering bug (SE 1.9-3x off); flanks add misspecification. IPTW skips the flanks - a misspecified propensity makes tau inconsistent, confounding bias with SE error.


def _replicate(flank: str, methods: tuple[str, ...], *, n: int, reps: int) -> dict:
    points = {m: [] for m in methods}
    ses = {m: [] for m in methods}
    covered = dict.fromkeys(methods, 0)
    for rep in range(reps):
        src = _src(_oracle_table(n, _SEED_BASE + rep, flank=flank))
        for method in methods:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
                    category=UserWarning,
                )
                est = _absolute(src, method)
            lift = est.require_lift()
            assert lift.lb is not None and lift.ub is not None
            points[method].append(lift.value)
            ses[method].append(_posterior_sd(est))
            covered[method] += lift.lb < _TRUE_TAU < lift.ub
    out = {}
    for method in methods:
        drawn = np.array(points[method])
        sd = drawn.std(ddof=1)
        out[method] = {
            "bias": float(drawn.mean() - _TRUE_TAU),
            "mcse": float(sd / np.sqrt(reps)),
            "se_over_sd": float(np.mean(ses[method]) / sd),
            "coverage": covered[method] / reps,
        }
    return out


@pytest.fixture(scope="module")
def oracle_replications():
    return _replicate("oracle", _METHODS, n=_N, reps=_REPS)


@pytest.fixture(scope="module")
def flank_replications():
    return {
        flank: _replicate(flank, ("dml", "aipw"), n=_N, reps=_REPS)
        for flank in ("wrong-m", "wrong-e")
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.xdist_group("abs_scale_recovery")
class TestAbsoluteChannelCalibration:
    @pytest.mark.parametrize("method", _METHODS)
    def test_recovers_the_additive_effect(self, oracle_replications, method):
        """Measured |bias|/MCSE at the pinned seeds: iptw 0.46, dml 0.04,
        aipw 0.21. The 1.5x allowance is float/BLAS drift headroom, not
        slack - a real bias shows up at many MCSE."""
        stats = oracle_replications[method]
        assert abs(stats["bias"]) < 1.5 * stats["mcse"]

    @pytest.mark.parametrize("method", ("dml", "aipw"))
    def test_reported_se_matches_the_empirical_spread(self, oracle_replications, method):
        low, high = _SE_SD_BAND
        assert low <= oracle_replications[method]["se_over_sd"] <= high

    def test_iptw_se_is_conservative_not_anticonservative(self, oracle_replications):
        """ACCEPTED and documented, not a defect: the plug-in IF ignores
        propensity estimation and over-states variance (measured 1.17 here,
        1.09-1.18 across seeds). It is inherited, not introduced - the
        relative channel uses the identical psi. The contract is that it
        errs wide, so coverage stays at or above nominal."""
        stats = oracle_replications["iptw"]
        assert stats["se_over_sd"] >= 1.0
        assert stats["coverage"] >= 0.95

    @pytest.mark.parametrize("method", _METHODS)
    def test_interval_coverage_is_near_nominal(self, oracle_replications, method):
        low, high = _COVERAGE_BAND
        assert low <= oracle_replications[method]["coverage"] <= high

    @pytest.mark.parametrize("flank", ("wrong-m", "wrong-e"))
    @pytest.mark.parametrize("method", ("dml", "aipw"))
    def test_calibrated_under_misspecified_nuisances(self, flank_replications, flank, method):
        stats = flank_replications[flank][method]
        low_ratio, high_ratio = _SE_SD_BAND
        low_cov, high_cov = _COVERAGE_BAND
        assert abs(stats["bias"]) < 1.5 * stats["mcse"]
        assert low_ratio <= stats["se_over_sd"] <= high_ratio
        assert low_cov <= stats["coverage"] <= high_cov


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_absolute_channel_calibration_smoke_small_n():
    """NOT a statistical claim - a handful of small-n replications kept in
    the fast suite so a broken absolute-channel pipeline is caught on every
    run even though the Monte Carlo suite above is excluded from it."""
    stats = _replicate("oracle", ("iptw", "aipw"), n=400, reps=8)
    for method in ("iptw", "aipw"):
        assert abs(stats[method]["bias"]) < 0.2
        assert 0.5 <= stats[method]["se_over_sd"] <= 2.0


@pytest.mark.parametrize("method", _METHODS)
def test_informative_prior_retains_weak_denominator_guard(method):
    source = _oracle_src(n=1000)
    flat = _one(source, method)
    assert flat.relative_confidence_set is not None and flat.abs_se is not None
    with pytest.raises(InvalidRequestError) as raised:
        _one(source, method, prior=Normal(mu=0.0, sigma=1.0))
    assert raised.value.code == "estimation.adjust_common.statistically_indistinguishable_from"
    denominator = raised.value.context["denominator"]
    se = raised.value.context["se"]
    assert isinstance(denominator, (int, float)) and isinstance(se, (int, float))
    assert abs(denominator) <= 4 * se


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("offset,refuses", [(0, True), (-1, True), (1, False)])
def test_informative_prior_denominator_guard_includes_four_se_boundary(sign, offset, refuses):
    import math

    import narwhals as nw

    from increment.estimation._adjust.common import (
        AdjustedContrastRequest,
        _refuse_near_zero_adjustment_denominator,
    )

    source = _oracle_src(n=20)
    metric = source.context.metrics[0]
    request = AdjustedContrastRequest(
        frame=nw.from_native(source.unit_frame(metric), eager_only=True),
        metric=metric,
        design=_DESIGN,
        control_group="C",
        treatment_group="T",
        control_native="C",
        treatment_native="T",
        covariates=["x"],
        prior=Normal(mu=0, sigma=1),
        alpha=0.05,
        alternative="two-sided",
        null_lift=0,
        null_abs=None,
        value_scale="relative",
        preferred_direction=None,
        cluster=None,
        method="IPTW",
    )
    magnitude = 4.0 if offset == 0 else math.nextafter(4.0, math.inf if offset > 0 else 0.0)
    if refuses:
        with pytest.raises(InvalidRequestError) as raised:
            _refuse_near_zero_adjustment_denominator(
                request, sign * magnitude, 1.0, label="control"
            )
        assert raised.value.code == "estimation.adjust_common.statistically_indistinguishable_from"
    else:
        _refuse_near_zero_adjustment_denominator(request, sign * magnitude, 1.0, label="control")
