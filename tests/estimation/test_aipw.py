"""Tests for the AIPW (doubly-robust, cross-fit) estimator: dispatch,
refusals, the design-doc identity invariants, and parameter recovery.

See `The Math` (design review) for the derivation these tests defend: a
Hajek-stabilized per-arm mean with a centered-residual influence
function, trim-first ordering, and the reserved-label / kwarg-routing
contract on `Method`.
"""

from __future__ import annotations

import itertools
import math
import re
import warnings

import numpy as np
import pyarrow as pa
import pytest
from scipy import stats
from scipy.special import expit

from increment import Analysis, IdentificationError
from increment.errors import (
    IncrementRuntimeWarning,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation._adjust.aipw import aipw_estimate
from increment.estimation._adjust.iptw import iptw_estimate
from increment.estimation._adjust.learners import LogisticPropensity, RidgeOutcome
from increment.estimation._adjust.overlap import _fold_ids
from increment.estimation.adjust import estimate_ate
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.engine import Method
from increment.frame import from_unit_summary
from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational
from tests.analysis_factory import lift_rows
from tests.warning_codes import warning_codes

# Deterministic (oracle) learners: zero fitting noise, so an identity test
# isolates the Hajek/IF algebra itself from estimation variability.


class _OracleConstant:
    """A propensity 'model' that ignores X and fit(), always predicting a
    fixed constant - eliminates propensity estimation noise."""

    def __init__(self, e0: float):
        self.e0 = e0

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.full(X.shape[0], self.e0)


class _OracleLinear:
    """An outcome 'model' that ignores fit() and always predicts a fixed
    affine function of the first covariate - eliminates outcome-model
    estimation noise."""

    def __init__(self, intercept: float, slope: float):
        self.intercept = intercept
        self.slope = slope

    def fit(self, X, d):
        pass

    def predict(self, X):
        return self.intercept + self.slope * X[:, 0]


class _ZeroOutcome:
    """m(X) == 0 everywhere: the design-doc identity (i) fixture - AIPW
    with a null outcome model must reduce verbatim to Hajek IPTW."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])


class _HalfPropensity:
    """Constant e=0.5 propensity stub, matching test_dml.py's fixture,
    keeps the overlap gate quiet so a test can reach the guard under study
    deterministically, or badly misspecify the propensity on purpose (the
    'wrong-e' double-robustness flank) while leaving X unused entirely."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.full(len(X), 0.5)


class _LargeOffsetOutcome:
    """Oracle arm means for the large-effect absolute-scale regression."""

    def fit(self, X, d):
        self.value = 1e12 if float(np.mean(d)) > 1e9 else 0.0

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def _aipw_metric(src, name):
    return next(m for m in src.context.metrics if m.name == name)


def _aipw_src(n: int = 60):
    """Enough units for the default folds=5 (needs n >= 2*folds=10), a
    covariate `x` that mildly separates the arms without threatening
    overlap, and both arms present. Mirrors test_dml.py's `_dml_src`."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    d = (rng.random(n) < expit(0.3 * x)).astype(int)
    y = 1.0 + 0.8 * x + 0.5 * d + rng.normal(scale=0.3, size=n)
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x": x,
        }
    )
    return from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )


_AIPW_DESIGN = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))


@pytest.mark.parametrize(
    ("value_scale", "sizes"),
    [
        ("absolute", (5, 10, 15, 20, 25) * 4),
        ("relative", (5,) * 19 + (1000,)),
    ],
)
def test_constant_arm_outcomes_have_no_cluster_size_uncertainty(value_scale, sizes):
    rows = [
        {
            "user_id": f"{arm}:{cluster}:{member}",
            "variant": arm,
            "store": f"{arm}:{cluster}",
            "revenue": 5.0 + 2.0 * (arm == "T"),
            "x": 0.0,
        }
        for arm in ("C", "T")
        for cluster, size in enumerate(sizes)
        for member in range(size)
    ]
    src = from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        cluster="store",
        design=_AIPW_DESIGN,
    )
    try:
        result = aipw_estimate(
            src, _aipw_metric(src, "revenue"), _AIPW_DESIGN, value_scale=value_scale
        )[0]
    except InvalidRequestError as error:
        assert value_scale == "absolute"
        assert error.code == "estimation.inference.degenerate_data_zero"
        return
    expected = 0.4 if value_scale == "relative" else 2.0
    lift = result.require_lift()
    assert lift.value == pytest.approx(expected)
    if value_scale == "relative":
        assert result.relative_confidence_set is None
        assert result.relative_unavailable_reason == "zero_relative_variance"
        assert (lift.lb, lift.ub) == (None, None)
        assert result.stat_sig() is False
        assert result.abs_diff == pytest.approx(2.0)
    else:
        assert lift.lb == pytest.approx(expected, abs=1e-12)
        assert lift.ub == pytest.approx(expected, abs=1e-12)


def _unexpected_learner_factory():
    raise AssertionError("learner factory should not run before fold validation")


# Dispatch / refusals - mirrors test_dml.py's TestDmlEstimate, adapted for
# the second per-arm outcome model AIPW carries.


class TestAipwEstimate:
    def test_too_few_units_for_folds_raises(self):
        src = _aipw_src(n=6)  # < 2*folds=10 at the default folds=5
        with pytest.raises(InvalidRequestError) as exc_info:
            aipw_estimate(src, _aipw_metric(src, "revenue"), _AIPW_DESIGN, folds=5)
        assert exc_info.value.code == "estimation.adjust_common.needs_least_folds"

    def test_too_few_units_in_an_arm_for_folds_raises(self):
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(10)],
                "variant": ["T"] + ["C"] * 9,
                "revenue": np.arange(10, dtype=float),
                "x": [np.nan] + list(np.linspace(-1.0, 1.0, 9)),
            }
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            aipw_estimate(
                src,
                _aipw_metric(src, "revenue"),
                _AIPW_DESIGN.model_copy(
                    update={"adjustment": AdjustmentSet(covariates=("x",), missing="allow")}
                ),
                propensity_learner=_unexpected_learner_factory,
                outcome_learner=_unexpected_learner_factory,
                folds=3,
            )
        assert exc_info.value.code == "estimation.adjust_overlap.arm_unit_but"

    @pytest.mark.filterwarnings("ignore:estimate_ate.*skipping metric:UserWarning")
    def test_ratio_metric_refused(self):
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(20)],
                "variant": ["T"] * 10 + ["C"] * 10,
                "spend": list(range(20)),
                "orders": [1] * 20,
                "x": [0.0] * 20,
            }
        )
        design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
        analysis = Analysis.from_unit_summary(
            tbl,
            unit="user_id",
            group="variant",
            metrics=[
                {"name": "rpo", "type": "ratio", "numerator": "spend", "denominator": "orders"}
            ],
            design=design,
        )
        with pytest.raises(UnsupportedRequestError) as exc_info:
            analysis.run(decision_method=Method(name="aipw", folds=2))
        assert exc_info.value.code == "estimation.adjust_common.supported_ratio_metric"

    def test_control_absent_raises_and_lists_available_groups(self):
        src = _aipw_src()
        design = Observational(control_group="ZZ", adjustment=AdjustmentSet(covariates=("x",)))
        with pytest.raises(InvalidRequestError) as exc_info:
            aipw_estimate(src, _aipw_metric(src, "revenue"), design)
        assert exc_info.value.code == "estimation.adjust_common.control_group_found"

    def test_missing_pattern_refused_with_dml_parity_message(self):
        rng = np.random.default_rng(4)
        n = 40
        x = rng.normal(size=n)
        x[3] = np.nan  # forces the missing-covariate policy branch to run
        d = (rng.random(n) < expit(0.3 * x[~np.isnan(x)].mean())).astype(int)
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": np.where(d == 1, "T", "C"),
                "revenue": 1.0 + rng.normal(scale=0.3, size=n),
                "x": x,
            }
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        design = Observational(
            control_group="C",
            adjustment=AdjustmentSet(covariates=("x",), missing="pattern"),
        )
        with pytest.raises(IdentificationError) as exc_info:
            estimate_ate(src, design, methods=[Method(name="aipw", folds=2)])
        assert exc_info.value.code == "adjust.identification.pattern_requires_iptw"

    def test_overlap_refusal_on_near_separating_covariate(self):
        n = 200
        rng = np.random.default_rng(2)
        x = rng.normal(size=n)
        d = (rng.random(n) < expit(6.0 * x)).astype(int)  # near-perfect separation
        y = 1.0 + 0.5 * d + rng.normal(scale=0.2, size=n)
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": np.where(d == 1, "T", "C"),
                "revenue": y,
                "x": x,
            }
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
        with pytest.raises(IdentificationError) as exc_info:
            estimate_ate(src, design, methods=[Method(name="aipw", folds=5)])
        assert exc_info.value.code == "adjust.identification.overlap_gate_refused"

    def test_registry_dispatch_via_estimate_ate(self):
        src = _aipw_src()
        ests = estimate_ate(src, _AIPW_DESIGN, methods=[Method(name="aipw")]).results
        assert len(ests) == 1
        assert ests[0].method == "aipw"

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_arm_stratification_avoids_degenerate_hash_refusal(self):
        import zlib

        ids = [f"u{i}" for i in range(4000) if zlib.crc32(f"u{i}".encode()) % 2 == 0][:12]
        assert len(ids) == 12
        tbl = pa.table(
            {
                "user_id": ids,
                "variant": ["T", "C"] * 6,
                "revenue": [float(i) for i in range(12)],
                "x": [0.1 * i for i in range(12)],
            }
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        estimates = aipw_estimate(src, _aipw_metric(src, "revenue"), _AIPW_DESIGN, folds=2)
        assert len(estimates) == 1

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_near_zero_control_preserves_relative_set_and_additive_inference(self):
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(12)],
                "variant": ["T", "C"] * 6,
                "revenue": [
                    3.0,
                    0.001,
                    5.0,
                    -0.002,
                    4.0,
                    0.0015,
                    4.5,
                    -0.001,
                    3.5,
                    0.002,
                    4.2,
                    -0.0005,
                ],
                "x": [0.3, -0.1, 0.2, 0.1, -0.2, 0.0, 0.15, -0.05, 0.25, 0.05, -0.15, 0.1],
            }
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        (result,) = aipw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            _AIPW_DESIGN,
            propensity_learner=_HalfPropensity,
            folds=2,
        )
        assert result.relative_confidence_set is not None
        assert result.relative_confidence_set.geometry in ("disconnected", "all_real", "one_sided")
        assert result.abs_lb is not None and result.abs_lb > 0
        assert result.abs_diff is not None and result.abs_diff > 3

    def test_balance_gate_message_names_propensity_and_outcome_defense(self):
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(12)],
                "variant": ["T"] * 6 + ["C"] * 6,
                "revenue": [3.0, 5.0, 4.0, 4.5, 3.5, 4.2, 1.0, 2.0, 1.5, 1.2, 1.8, 1.4],
                "x": [2.0, 2.1, 1.9, 2.05, 1.95, 2.0, 0.1, 0.0, -0.1, 0.05, -0.05, 0.0],
            }
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        design = Observational(
            control_group="C",
            adjustment=AdjustmentSet(covariates=("x",)),
            gate=IdentificationGate(max_smd=0.05),
        )
        with pytest.raises(IdentificationError) as exc_info:
            aipw_estimate(
                src,
                _aipw_metric(src, "revenue"),
                design,
                propensity_learner=_HalfPropensity,
                folds=2,
            )
        assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


# Method routing (engine.py's reserved-label tuple, _adjust_kwargs' explicit
# per-method mapping).


class TestMethodRouting:
    def test_aipw_reserved_under_estimate_lift(self):
        from increment.estimation.engine import estimate_lift
        from increment.semantics.models import MeanMetric

        row = centered_row_from_raw_sums(
            {
                "experiment_id": "e1",
                "metric": "revenue",
                "group_id": "C",
                "n": 10,
                "sum_y": 10.0,
                "sum_y2": 12.0,
            }
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                [MeanMetric(name="revenue", fact="revenue", entity="user")],
                [row],
                "C",
                methods=[Method(name="aipw")],
            )
        assert exc_info.value.code == "estimation.engine.method_name_observational"

    def test_registry_registered_adjustment_is_reserved_on_the_randomized_path(self):
        """Any registered adjustment name is an observational dispatch key the
        randomized moments path never applies; the mislabel guard must cover
        registered names, not only the three built-ins."""
        from increment.estimation.adjust import ADJUSTMENTS
        from increment.estimation.engine import estimate_lift
        from increment.semantics.models import MeanMetric

        original = dict(ADJUSTMENTS._entries)
        ADJUSTMENTS.register("regression_probe_adjustment", lambda *a, **k: [])
        try:
            row = centered_row_from_raw_sums(
                {
                    "experiment_id": "e1",
                    "metric": "revenue",
                    "group_id": "C",
                    "n": 10,
                    "sum_y": 10.0,
                    "sum_y2": 12.0,
                }
            )
            with pytest.raises(InvalidRequestError) as exc_info:
                estimate_lift(
                    [MeanMetric(name="revenue", fact="revenue", entity="user")],
                    [row],
                    "C",
                    methods=[Method(name="regression_probe_adjustment")],
                )
            assert exc_info.value.code == "estimation.engine.method_name_observational"
        finally:
            ADJUSTMENTS._entries.clear()
            ADJUSTMENTS._entries.update(original)

    def test_iptw_rejects_outcome_learner(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Method(name="iptw", outcome_learner=RidgeOutcome)
        assert exc_info.value.code == "estimation.engine.method.name_iptw_does"

    def test_iptw_rejects_folds(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Method(name="iptw", folds=3)
        assert exc_info.value.code == "estimation.engine.method.name_iptw_does"

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_dml_and_aipw_thread_factories_and_folds(self):
        """`Method(name="aipw")` forwards the configured learners and folds to
        the registered adjustment: the configuration runs on a sample the
        requested fold count fits, and the same call refuses the fold guard
        on one it does not."""
        configured = Method(
            name="aipw",
            propensity_learner=LogisticPropensity,
            outcome_learner=RidgeOutcome,
            folds=7,
        )
        (estimate,) = estimate_ate(_aipw_src(n=40), _AIPW_DESIGN, methods=[configured]).results
        assert estimate.method == "aipw"
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(_aipw_src(n=12), _AIPW_DESIGN, methods=[configured])
        assert exc_info.value.code == "estimation.adjust_common.needs_least_folds"

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_unset_fields_thread_nothing(self):
        """A `Method` with no nuisance fields runs the registered adjustment on
        its own defaults."""
        src = _aipw_src(n=40)
        for name in ("aipw", "dml", "iptw"):
            (estimate,) = estimate_ate(src, _AIPW_DESIGN, methods=[Method(name=name)]).results
            assert estimate.method == name

    def test_estimate_ate_end_to_end_with_custom_folds(self):
        """The whole point of ``Method.folds`` - folds actually reaches the
        cross-fit loop through the documented Method-based entry point,
        not just through aipw_estimate's own kwarg."""
        src = _aipw_src(n=20)  # < 2*folds=10 at the default 5, fine at folds=2
        (est,) = estimate_ate(src, _AIPW_DESIGN, methods=[Method(name="aipw", folds=2)]).results
        assert est.method == "aipw"
        # folds=5 (default) would refuse only if 2*folds > n; folds=9
        # (2*9=18<=20) passes while folds=11 must not:
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(src, _AIPW_DESIGN, methods=[Method(name="aipw", folds=11)])
        assert exc_info.value.code == "estimation.adjust_common.needs_least_folds"


# Design-doc identity invariants. Oracle learners carry zero estimation
# noise, so these isolate the Hajek/IF algebra itself.


def _identity_table(n: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e_true = expit(0.6 * x)
    d = (rng.random(n) < e_true).astype(int)
    y = 3.0 + 1.5 * x + 2.0 * d + rng.normal(scale=1.0, size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x": x,
        }
    )


class TestAipwIdentities:
    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_null_outcome_model_matches_iptw_point_and_se(self):
        """Design invariant (i): at m == 0, AIPW's psi reduces VERBATIM to
        Hajek IPTW's psi - point AND SE - when both use the SAME
        propensity. Uses a shared oracle propensity so cross-fitting
        (AIPW) vs a single in-sample fit (IPTW) cannot introduce any
        numerical difference of its own."""
        src = _aipw_src(n=100)
        design = _AIPW_DESIGN

        def propensity_factory():
            return _OracleConstant(0.5)

        (aipw_est,) = aipw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            design,
            propensity_learner=propensity_factory,
            outcome_learner=lambda: _ZeroOutcome(),
            folds=4,
        )
        (iptw_est,) = iptw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            design,
            learner=propensity_factory,
        )
        assert aipw_est.require_lift().value == pytest.approx(
            iptw_est.require_lift().value, rel=0, abs=1e-9
        )
        assert aipw_est.require_lift().lb == pytest.approx(
            iptw_est.require_lift().lb, rel=0, abs=1e-9
        )
        assert aipw_est.require_lift().ub == pytest.approx(
            iptw_est.require_lift().ub, rel=0, abs=1e-9
        )

    def test_absolute_large_effect_centers_scores_before_aggregation(self):
        """Absolute AIPW must retain residual variation beside a large tau."""
        n = 20
        tau = 1e12
        d = np.tile([1, 0], n // 2)
        treatment_residual = np.linspace(-0.45, 0.45, n // 2)
        control_residual = np.linspace(-0.2, 0.2, n // 2)
        y = np.empty(n)
        y[d == 1] = tau + treatment_residual
        y[d == 0] = control_residual
        src = from_unit_summary(
            pa.table(
                {
                    "user_id": [f"u{i}" for i in range(n)],
                    "variant": np.where(d == 1, "T", "C"),
                    "revenue": y,
                    "x": np.zeros(n),
                }
            ),
            unit="user_id",
            group="variant",
            control="C",
            metrics={"revenue": "mean"},
        )
        design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
        (estimate,) = aipw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            design,
            propensity_learner=lambda: _OracleConstant(0.5),
            outcome_learner=_LargeOffsetOutcome,
            folds=2,
            value_scale="absolute",
        )

        treatment_residual = y[d == 1] - tau
        control_residual = y[d == 0]
        b1 = float(treatment_residual.mean())
        b0 = float(control_residual.mean())
        scores = np.empty(n)
        scores[d == 1] = tau + 2.0 * treatment_residual - b1 - b0
        scores[d == 0] = tau + b1 - 2.0 * control_residual + b0
        expected_se = math.sqrt(float(np.var(scores)) / n)
        assert estimate.value_scale == "absolute"
        assert estimate.require_lift().value == pytest.approx(tau, abs=1e-3, rel=0)
        assert estimate.require_lift().log_se == pytest.approx(expected_se, rel=0.0, abs=1e-12)

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_constant_propensity_matches_regression_adjusted_estimator(self):
        """Design invariant (ii): at constant e, AIPW equals the classic
        regression-adjusted (G-computation-with-residual-correction)
        estimator. Independently recomputed here from the SAME oracle
        outcome functions and raw table (not by calling into adjust.py's
        internals) - a genuinely separate code path, not a tautology."""
        n = 300
        tbl = _identity_table(n, seed=11)
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        design = _AIPW_DESIGN
        e0 = 0.42
        m1 = _OracleLinear(intercept=5.0, slope=1.5)  # matches the true DGP exactly
        m0 = _OracleLinear(intercept=3.0, slope=1.5)

        (aipw_est,) = aipw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            design,
            propensity_learner=lambda: _OracleConstant(e0),
            outcome_learner=itertools.cycle([m1, m0]).__next__,  # m1,m0 alternate per fold
            folds=3,
        )

        # Independent recomputation: mu_a = mean_i(m_a(x_i)) + mean_{arm a}(y_i - m_a(x_i)).
        x = tbl["x"].to_numpy()
        y = tbl["revenue"].to_numpy()
        d = (tbl["variant"].to_pylist() == np.array(["T"] * n)).astype(float)
        m1_hat = m1.predict(x.reshape(-1, 1))
        m0_hat = m0.predict(x.reshape(-1, 1))
        mu1_expected = m1_hat.mean() + (y[d == 1] - m1_hat[d == 1]).mean()
        mu0_expected = m0_hat.mean() + (y[d == 0] - m0_hat[d == 0]).mean()
        lift_expected = (mu1_expected - mu0_expected) / mu0_expected

        assert aipw_est.require_lift().value == pytest.approx(lift_expected, rel=1e-9, abs=1e-9)

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_location_shift_invariance(self):
        """Design invariant (iii): Y -> Y+C shifts each mu_a by exactly C
        and leaves tau_abs invariant - checked here through the relative
        lift's own known transformation (tau_abs/mu0 -> tau_abs/(mu0+C)),
        since the observational path does not expose raw mu_a/tau_abs."""
        n = 300
        tbl = _identity_table(n, seed=13)
        design = _AIPW_DESIGN
        e0 = 0.5
        m1_base, m0_base = 5.0, 3.0
        slope = 1.5

        def _run(intercept_shift: float, y_shift: float):
            tbl2 = tbl.set_column(
                tbl.schema.get_field_index("revenue"),
                "revenue",
                pa.array(tbl["revenue"].to_numpy() + y_shift),
            )
            src2 = from_unit_summary(
                tbl2, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
            )
            m1 = _OracleLinear(intercept=m1_base + intercept_shift, slope=slope)
            m0 = _OracleLinear(intercept=m0_base + intercept_shift, slope=slope)
            (est,) = aipw_estimate(
                src2,
                _aipw_metric(src2, "revenue"),
                design,
                propensity_learner=lambda: _OracleConstant(e0),
                outcome_learner=itertools.cycle([m1, m0]).__next__,
                folds=3,
            )
            return est.require_lift().value

        lift_orig = _run(intercept_shift=0.0, y_shift=0.0)

        # Recompute mu0_orig independently to predict the shifted lift.
        x = tbl["x"].to_numpy()
        y = tbl["revenue"].to_numpy()
        d = (tbl["variant"].to_pylist() == np.array(["T"] * n)).astype(float)
        m0_hat = _OracleLinear(m0_base, slope).predict(x.reshape(-1, 1))
        mu0_orig = m0_hat.mean() + (y[d == 0] - m0_hat[d == 0]).mean()
        tau_abs = lift_orig * mu0_orig

        C = 250.0
        lift_shifted = _run(intercept_shift=C, y_shift=C)
        lift_shifted_expected = tau_abs / (mu0_orig + C)
        assert lift_shifted == pytest.approx(lift_shifted_expected, rel=1e-6, abs=1e-6)


# Trim-first ordering: overlap trimming must precede Hajek normalizers/mu/IF/SE;
# score-then-trim would bake full-sample normalizers into kept-unit scores.


class TestTrimOrdering:
    def test_trimmed_population_is_labelled_and_location_equivariant(self):
        n = 400
        rng = np.random.default_rng(17)
        x = rng.normal(size=n)
        e_true = expit(1.2 * x)
        d = (rng.random(n) < e_true).astype(int)
        y = 2.0 + x + 1.0 * d + rng.normal(scale=0.5, size=n)
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": np.where(d == 1, "T", "C"),
                "revenue": y,
                "x": x,
            }
        )
        design = Observational(
            control_group="C",
            adjustment=AdjustmentSet(covariates=("x",)),
            gate=IdentificationGate(overlap="trim", min_propensity=0.1),
        )
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        (est,) = aipw_estimate(src, _aipw_metric(src, "revenue"), design, folds=4)
        assert est.population is not None
        # The label records the gate's trimming bound and a strictly reduced
        # unit count, so trimming is visible on the estimate itself.
        retained, total = (int(value) for value in re.findall(r"(\d+) of (\d+)", est.population)[0])
        assert 0 < retained < total
        assert "0.1" in est.population and "0.9" in est.population


# Fold-leakage spy: a 1-NN perfect memorizer as every learner would
# reproduce e_hat==d and m_hat==y exactly if it scored its own training rows.


class _FingerprintNN1:
    """A 1-nearest-neighbor 'model' that fingerprints every fit/predict
    covariate set so a test can assert zero train/predict index overlap
    per fold, AND perfectly memorizes its training labels - under
    leakage this reproduces the training label exactly for every row it
    scores, catastrophically biasing theta if cross-fitting were broken.
    """

    fit_sets: list[frozenset[bytes]] = []
    predict_sets: list[frozenset[bytes]] = []

    def __init__(self):
        self._Xtrain: np.ndarray | None = None
        self._ytrain: np.ndarray | None = None

    def fit(self, X, d):
        self._Xtrain = X
        self._ytrain = np.asarray(d, dtype=float)
        type(self).fit_sets.append(frozenset(row.tobytes() for row in X))

    def predict(self, X):
        type(self).predict_sets.append(frozenset(row.tobytes() for row in X))
        assert self._Xtrain is not None and self._ytrain is not None
        out = np.empty(X.shape[0])
        for i, row in enumerate(X):
            dists = np.sum((self._Xtrain - row) ** 2, axis=1)
            out[i] = self._ytrain[np.argmin(dists)]
        return out


class TestFoldLeakage:
    def test_train_predict_indices_never_overlap(self):
        _FingerprintNN1.fit_sets = []
        _FingerprintNN1.predict_sets = []
        src = _aipw_src(n=100)
        aipw_estimate(
            src,
            _aipw_metric(src, "revenue"),
            _AIPW_DESIGN,
            propensity_learner=LogisticPropensity,
            outcome_learner=_FingerprintNN1,
            folds=5,
        )
        # Both outcome models are fit and scored in every fold: the spy must
        # have recorded a fit/predict pair per fold, in step.
        assert len(_FingerprintNN1.fit_sets) == len(_FingerprintNN1.predict_sets) == 10
        # A row EVER trained-on by the model that later scores it in the
        # SAME fold call is leakage - checked pairwise per outcome model.
        for fit_rows, predict_rows in zip(
            _FingerprintNN1.fit_sets, _FingerprintNN1.predict_sets, strict=True
        ):
            assert fit_rows.isdisjoint(predict_rows)
        all_predict = frozenset().union(*_FingerprintNN1.predict_sets)
        assert len(all_predict) > 0


def _cluster_aipw_src():
    rows = {"user_id": [], "variant": [], "revenue": [], "x": [], "store": []}
    uid = 0
    for arm_index, arm in enumerate(("C", "T")):
        for cluster_index in range(5):
            cluster = f"{arm}{cluster_index}"
            for _ in range(2):
                rows["user_id"].append(f"u{uid}")
                rows["variant"].append(arm)
                rows["revenue"].append(float(uid + arm_index))
                rows["x"].append(float(arm_index * 10 + cluster_index))
                rows["store"].append(cluster)
                uid += 1
    return from_unit_summary(
        pa.table(rows),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        cluster="store",
    )


def test_cluster_atomicity_reaches_recording_outcome_learners():
    _FingerprintNN1.fit_sets = []
    _FingerprintNN1.predict_sets = []
    src = _cluster_aipw_src()
    with pytest.warns(IncrementWarning) as rec:
        with pytest.warns(IncrementRuntimeWarning) as rec_runtime:
            aipw_estimate(
                src,
                _aipw_metric(src, "revenue"),
                _AIPW_DESIGN,
                propensity_learner=_HalfPropensity,
                outcome_learner=_FingerprintNN1,
                folds=5,
            )
    assert "estimation.adjust_common.covariate_balance_advisory" in warning_codes(rec)
    assert "estimation.engine.small_total_clusters" in warning_codes(rec_runtime)
    assert len(_FingerprintNN1.fit_sets) == len(_FingerprintNN1.predict_sets) == 10
    for fit_rows, predict_rows in zip(
        _FingerprintNN1.fit_sets, _FingerprintNN1.predict_sets, strict=True
    ):
        assert fit_rows.isdisjoint(predict_rows)


# Parameter recovery: one Monte Carlo suite over a heterogeneous-effect DGP
# with a closed-form true ATE-lift. Marked `parameter_recovery`.


def _het_table(n: int, seed: int) -> pa.Table:
    """Heterogeneous treatment effect, asymmetric propensity (X not
    centered at the propensity model's own pivot) - the combination that
    makes DML's propensity-variance-weighted PLR target genuinely differ
    from the unit-weighted ATE AIPW/IPTW report (design doc "The Math").

    True ATE = E[tau(X)] = 1.0 + 0.6*E[X] = 1.6 (X ~ N(1, 1)).
    True E[Y(0)] = E[4 + X] = 5.0.  True ATE-lift = 1.6 / 5.0 = 0.32.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=1.0, scale=1.0, size=n)
    e = expit(1.5 * x)
    d = (rng.random(n) < e).astype(int)
    tau = 1.0 + 0.6 * x
    y0 = 4.0 + 1.0 * x + rng.normal(scale=1.0, size=n)
    y1 = y0 + tau
    y = np.where(d == 1, y1, y0)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x": x,
        }
    )


_TRUE_ATE_LIFT = 1.6 / 5.0
_HET_DESIGN = Observational(
    control_group="C",
    adjustment=AdjustmentSet(covariates=("x",)),
    gate=IdentificationGate(overlap="trim"),
)
_RECOVERY_REPS = 100
_RECOVERY_N = 1500

#: Coverage-guard replications, separate from the five-variant recovery fixture
#: whose count sets other checks' tolerances. 400 one-interval draws (~10s)
#: reject at <= 367 hits against the 0.94 baseline: 90% power at true coverage
#: 0.90 and a 0.4% false-alarm rate at nominal 0.95.
_COVERAGE_REPS = 400


@pytest.fixture(scope="module")
def _aipw_recovery_replications():
    """One replication loop, several method variants per draw:

    - `aipw`: right propensity (fitted), right outcome (fitted) - the
      primary recovery + coverage + SE-calibration check.
    - `aipw_wrong_m`: right propensity, m forced to 0 (validation item 2's
      wrong-model/right-propensity flank - the boundary case that
      degenerates to Hajek IPTW, still unbiased under a correct
      propensity, and validation item 3's wrong-m SE-calibration flank).
    - `aipw_wrong_e`: propensity forced constant (ignores all
      confounding), right outcome (fitted) - validation item 2's
      right-model/wrong-propensity flank.
    - `dml`: for the divergence-from-ATE check (validation item 1).
      DML's PLR target is NOT this DGP's unit-weighted ATE.
    - `naive`: unadjusted, for the "confounding is real" sanity check.
    """
    aipw_pts, aipw_los, aipw_his = [], [], []
    wrong_m_pts, wrong_e_pts = [], []
    dml_pts, naive_pts = [], []
    hits = 0
    for i in range(_RECOVERY_REPS):
        tbl = _het_table(_RECOVERY_N, seed=200 + i)
        src = from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        an = Analysis.from_unit_summary(
            tbl, unit="user_id", group="variant", metrics={"revenue": "mean"}, design=_HET_DESIGN
        )
        an_naive = Analysis.from_unit_summary(
            tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
                category=UserWarning,
            )
            (aipw_est,) = lift_rows(an.run(decision_method=Method(name="aipw")))
            (wrong_m,) = aipw_estimate(
                src,
                _aipw_metric(src, "revenue"),
                _HET_DESIGN,
                outcome_learner=_ZeroOutcome,
            )
            (wrong_e,) = aipw_estimate(
                src,
                _aipw_metric(src, "revenue"),
                _HET_DESIGN,
                propensity_learner=_HalfPropensity,
            )
            (dml_est,) = lift_rows(an.run(decision_method=Method(name="dml")))
            (naive_est,) = lift_rows(an_naive.run())
        aipw_lift = aipw_est.require_lift()
        aipw_pts.append(aipw_lift.value)
        assert aipw_lift.lb is not None and aipw_lift.ub is not None
        aipw_los.append(aipw_lift.lb)
        aipw_his.append(aipw_lift.ub)
        hits += aipw_lift.lb < _TRUE_ATE_LIFT < aipw_lift.ub
        wrong_m_pts.append(wrong_m.require_lift().value)
        wrong_e_pts.append(wrong_e.require_lift().value)
        dml_pts.append(dml_est.require_lift().value)
        naive_pts.append(naive_est.require_lift().value)

    aipw_pts = np.array(aipw_pts)
    z = 1.959963984540054  # alpha=0.05 two-sided
    se_reported = (np.array(aipw_his) - np.array(aipw_los)) / (2 * z)
    return {
        "aipw": aipw_pts,
        "wrong_m": np.array(wrong_m_pts),
        "wrong_e": np.array(wrong_e_pts),
        "dml": np.array(dml_pts),
        "naive": np.array(naive_pts),
        "se_reported_mean": float(se_reported.mean()),
        "hits": hits,
        "coverage": hits / _RECOVERY_REPS,
    }


@pytest.fixture(scope="module")
def _aipw_coverage_replications():
    """Coverage count only, at ``_COVERAGE_REPS`` draws.

    Deliberately separate from ``_aipw_recovery_replications``: one interval
    per draw instead of five method variants, so the replication count needed
    to give the coverage guard real power costs seconds rather than minutes and
    cannot shift the tolerances the other Monte-Carlo checks are tuned to.
    Seeds are disjoint from that fixture's, so the baseline it is tested
    against is supported by independent draws.
    """
    hits = 0
    for i in range(_COVERAGE_REPS):
        tbl = _het_table(_RECOVERY_N, seed=9000 + i)
        an = Analysis.from_unit_summary(
            tbl, unit="user_id", group="variant", metrics={"revenue": "mean"}, design=_HET_DESIGN
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
                category=UserWarning,
            )
            (est,) = lift_rows(an.run(decision_method=Method(name="aipw")))
        est_lift = est.require_lift()
        assert est_lift.lb is not None and est_lift.ub is not None
        hits += est_lift.lb < _TRUE_ATE_LIFT < est_lift.ub
    return {"hits": hits, "coverage": hits / _COVERAGE_REPS}


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.xdist_group("aipw_recovery")
class TestAipwParameterRecovery:
    def test_bias_within_two_monte_carlo_se(self, _aipw_recovery_replications):
        pts = _aipw_recovery_replications["aipw"]
        mc_se = pts.std(ddof=1) / np.sqrt(len(pts))
        assert abs(pts.mean() - _TRUE_ATE_LIFT) < 2 * mc_se

    def test_coverage_does_not_regress_below_nominal(self, _aipw_coverage_replications):
        """Regression guard against nominal coverage, on the dedicated
        ``_COVERAGE_REPS``-draw fixture. AIPW's relative interval comes from
        the joint influence-function covariance of the additive effect and
        the control mean, and on this heterogeneous DGP at n=1500 it covers
        at nominal within Monte Carlo error: 372/400 = 0.930 on this
        fixture's own seeds and 143/150 = 0.953 on an independent probe
        through both ``aipw_estimate`` and ``Analysis.run`` (pooled 0.936,
        1.5 Monte Carlo standard deviations under nominal). An earlier
        reading attributed a deficit to the first-order ratio-denominator
        skew documented on ``RatioVarianceModel``; that mechanism does not
        apply here (the control mean's relative standard error is about
        0.02, far below the 0.15 at which that advisory fires), and the
        pooled measurement is not a deficit.

        One-sided exact binomial against a 0.94 baseline. At 400 draws the
        Monte Carlo standard deviation of an estimated coverage under nominal
        0.95 is sqrt(0.95 * 0.05 / 400) = 0.0109, so the rejection boundary
        (<= 367 hits, 91.75%) sits three of those below nominal: a rejection
        is a real undercoverage regression, not sampling noise (false-alarm
        rate 0.4% under nominal; 90% power against a true coverage of 0.90).
        """
        hits = _aipw_coverage_replications["hits"]
        baseline = 0.94
        pvalue = stats.binomtest(hits, _COVERAGE_REPS, baseline, alternative="less").pvalue
        assert pvalue > 0.05, (
            f"hits={hits}/{_COVERAGE_REPS} (coverage={hits / _COVERAGE_REPS:.4f}) is "
            f"significantly below the {baseline} floor (one-sided binomial p={pvalue:.4g}); "
            f"at {_COVERAGE_REPS} draws nominal 0.95 has Monte Carlo sd "
            f"{math.sqrt(0.95 * 0.05 / _COVERAGE_REPS):.4f}, so the rejection boundary of "
            "367 hits lies 3 sd below nominal -- a real interval undercoverage "
            "regression, not sampling noise"
        )

    def test_se_calibration_reported_vs_empirical_spread(self, _aipw_recovery_replications):
        """Validation item 3: IF-SE / empirical-SD in a sane band. Uses
        the CI-derived SE (see fixture) rather than ScoreStats.se()
        directly - an equivalent quantity through the same conjugate
        update, checked at the public-API surface."""
        empirical_sd = _aipw_recovery_replications["aipw"].std(ddof=1)
        ratio = _aipw_recovery_replications["se_reported_mean"] / empirical_sd
        assert 0.8 <= ratio <= 1.25

    def test_double_robustness_wrong_m_right_e_unbiased(self, _aipw_recovery_replications):
        """Validation item 2, flank 1: propensity right, outcome model
        forced to m==0 (AIPW degenerates to Hajek IPTW here) - still
        unbiased for the ATE."""
        pts = _aipw_recovery_replications["wrong_m"]
        mc_se = pts.std(ddof=1) / np.sqrt(len(pts))
        assert abs(pts.mean() - _TRUE_ATE_LIFT) < 3 * mc_se

    def test_double_robustness_right_m_wrong_e_unbiased(self, _aipw_recovery_replications):
        """Validation item 2, flank 2: propensity forced constant
        (ignores confounding entirely), outcome model fitted - still
        unbiased, unlike IPTW/DML under the same wrong propensity."""
        pts = _aipw_recovery_replications["wrong_e"]
        mc_se = pts.std(ddof=1) / np.sqrt(len(pts))
        assert abs(pts.mean() - _TRUE_ATE_LIFT) < 3 * mc_se

    def test_naive_confounding_is_real(self, _aipw_recovery_replications):
        naive_bias = abs(_aipw_recovery_replications["naive"].mean() - _TRUE_ATE_LIFT)
        assert naive_bias > 0.05

    def test_dml_targets_a_different_estimand_under_heterogeneity(
        self, _aipw_recovery_replications
    ):
        """Validation item 1: on this heterogeneous-effect DGP, DML's
        partially-linear (propensity-variance-weighted) theta and AIPW's
        unit-weighted ATE are genuinely different quantities - not two
        noisy estimates of the same one. This is the test dml.py's own
        estimator cannot pass (by construction, per the design doc)."""
        aipw_pts = _aipw_recovery_replications["aipw"]
        dml_pts = _aipw_recovery_replications["dml"]
        combined_se = np.sqrt(
            aipw_pts.std(ddof=1) ** 2 / len(aipw_pts) + dml_pts.std(ddof=1) ** 2 / len(dml_pts)
        )
        assert abs(aipw_pts.mean() - dml_pts.mean()) > 3 * combined_se
        # AIPW itself stays on target; DML does not have to (and does not
        # need testing against _TRUE_ATE_LIFT here - that is the point).
        aipw_mc_se = aipw_pts.std(ddof=1) / np.sqrt(len(aipw_pts))
        assert abs(aipw_pts.mean() - _TRUE_ATE_LIFT) < 3 * aipw_mc_se


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_aipw_recovery_smoke_single_replication_is_a_real_interval():
    """NOT a statistical claim - a single replication at small n, run in
    the fast suite, so a broken recovery pipeline is still caught on
    every run even though the full Monte Carlo suite above is excluded
    from it."""
    tbl = _het_table(300, seed=1)
    an = Analysis.from_unit_summary(
        tbl, unit="user_id", group="variant", metrics={"revenue": "mean"}, design=_HET_DESIGN
    )
    (aipw_est,) = lift_rows(an.run(decision_method=Method(name="aipw")))
    aipw_lift = aipw_est.require_lift()
    assert aipw_lift.lb is not None and aipw_lift.ub is not None
    assert aipw_lift.lb < aipw_lift.value < aipw_lift.ub


def test_fold_ids_reused_helper_still_importable():
    """Guards the shared `_fold_ids` contract AIPW reuses from the DML/IPTW
    module: deterministic, row-order-invariant, and arm-stratified, so the
    cross-estimator fold assignment stays reproducible."""
    unit_ids = np.array([f"u{i}" for i in range(12)])
    arms = np.array([0, 1] * 6)
    labels = _fold_ids(unit_ids, arms, 2)
    # Identical inputs assign identical folds.
    assert np.array_equal(labels, _fold_ids(unit_ids, arms, 2))
    # Fold membership is ranked by identity, not by input position.
    order = np.array([11, 4, 9, 1, 6, 2, 8, 0, 7, 3, 10, 5])
    assert np.array_equal(_fold_ids(unit_ids[order], arms[order], 2), labels[order])
    # Each arm's units reach every fold, so no fold trains on one arm alone.
    for arm in (0, 1):
        assert len(set(labels[arms == arm].tolist())) == 2


# Categorical adjustment: the level encoding is a fitted transform, learned
# inside every cross-fit training split, never from the cohort.


def _rare_level_units(n: int = 300):
    from tests.categorical_cases import categorical_units

    units = categorical_units(n, seed=21)
    # One unit carries a level nobody else has: whichever fold holds it out
    # trains on a level set without it.
    units["region"] = units["region"].astype(object)
    units["region"][5] = "island"
    return units


def _recording_factory(designs: list[tuple[str, np.ndarray, np.ndarray]], label: str):
    class Recording:
        def fit(self, X, d):
            self._train = np.asarray(X, dtype=float).copy()
            self._value = float(np.mean(d))

        def predict(self, X):
            designs.append((label, self._train, np.asarray(X, dtype=float).copy()))
            return np.full(np.asarray(X).shape[0], self._value)

    return Recording


def test_aipw_fits_the_level_encoding_inside_each_training_split():
    """The fold holding out the only ``island`` unit trains on two level
    indicators, and its held-out design has the same two columns with the
    island row all zero: the encoding never learns the held-out level."""
    from tests.categorical_cases import raw_table

    units = _rare_level_units()
    designs: list[tuple[str, np.ndarray, np.ndarray]] = []
    src = from_unit_summary(
        raw_table(units),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("spend", "region")),
        gate=IdentificationGate(overlap="trim"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate_ate(
            src,
            design,
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=_recording_factory(designs, "propensity"),
                    outcome_learner=_recording_factory(designs, "outcome"),
                    folds=3,
                )
            ],
        )
    propensity = [(train, test) for label, train, test in designs if label == "propensity"]
    assert len(propensity) == 3
    widths = sorted(train.shape[1] for train, _ in propensity)
    # Two folds train with island present (three indicators), one without.
    assert widths == [3, 4, 4]
    for train, test in propensity:
        assert test.shape[1] == train.shape[1]
        assert not np.isnan(train).any() and not np.isnan(test).any()
    (narrow_train, narrow_test) = next((tr, te) for tr, te in propensity if tr.shape[1] == 3)
    assert narrow_train.shape[0] + narrow_test.shape[0] == 300
    # The held-out island row carries no indicator at all: the reference.
    (island_row,) = np.flatnonzero(narrow_test[:, 0] == units["spend"][5])
    np.testing.assert_array_equal(narrow_test[island_row, 1:], [0.0, 0.0])
    # Where island is a training level it gets its own single-unit column.
    for train, _ in propensity:
        if train.shape[1] == 4:
            assert train[:, 1:].sum(axis=0).min() == 1.0


def _unseen_level_run(units, **method_kwargs):
    """The AIPW estimate over *units* plus every unseen-level advisory it raised."""
    from tests.categorical_cases import raw_table

    src = from_unit_summary(
        raw_table(units),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("spend", "region")),
        gate=IdentificationGate(overlap="trim"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (est,) = estimate_ate(src, design, methods=[Method(name="aipw", **method_kwargs)]).results
    advisories = [
        w.message
        for w in caught
        if getattr(w.message, "code", None) == "estimation.adjust_common.unseen_level_advisory"
    ]
    return est, advisories


def test_aipw_discloses_the_level_its_heldout_fold_never_trained_on():
    """The fold holding out the only ``island`` unit scores it with three
    fits whose training rows lacked the level -- the propensity and both
    arm outcome models -- each reading it as the reference. That is
    disclosed once per cohort: the coded advisory carries the covariate,
    level, the one row and those three fits, and the estimate note names
    the level. A level present in every training split raises nothing."""
    from tests.categorical_cases import categorical_units

    est, advisories = _unseen_level_run(_rare_level_units(), folds=3)
    (advisory,) = advisories
    assert isinstance(advisory, IncrementWarning)
    assert advisory.context["unseen"] == (("region", "island", 1, 3),)
    assert advisory.context["n"] == 300
    assert est.note is not None and "region=island" in est.note

    _, advisories = _unseen_level_run(categorical_units(300, seed=21), folds=3)
    assert advisories == []
