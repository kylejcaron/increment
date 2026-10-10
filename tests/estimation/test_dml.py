"""Tests for the DML estimator: learners, folds, scores, dispatch."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pyarrow as pa
import pytest
from scipy.special import expit

from increment import Analysis, IdentificationError
from increment.errors import InvalidRequestError
from increment.estimation._adjust.dml import _dml_theta
from increment.estimation._adjust.learners import RidgeOutcome
from increment.estimation._adjust.overlap import _fold_ids
from increment.estimation.adjust import estimate_ate
from increment.estimation.armstats import ScoreStats
from increment.estimation.engine import Method
from increment.frame import from_unit_summary
from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational
from tests.analysis_factory import lift_rows
from tests.estimation.test_adjust import _CellMean


class TestRidgeOutcome:
    def test_recovers_linear_signal(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(500, 3))
        y = 2.0 + X @ np.array([1.0, -0.5, 0.25]) + rng.normal(scale=0.01, size=500)
        m = RidgeOutcome()
        m.fit(X, y)
        pred = m.predict(X)
        assert np.corrcoef(pred, y)[0, 1] > 0.999

    def test_zero_variance_column_does_not_crash(self):
        X = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
        y = 3.0 * X[:, 1]
        m = RidgeOutcome()
        m.fit(X, y)
        assert np.isfinite(m.predict(X)).all()

    def test_predict_before_fit_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            RidgeOutcome().predict(np.ones((3, 2)))
        assert (
            exc_info.value.code
            == "estimation.adjust_learners.ridge_outcome.ridgeoutcome_predict_called"
        )


class TestFoldIds:
    def test_row_order_invariant(self):
        # The SAME unit must land in the SAME fold regardless of row order -
        # warehouse fetch order is unstable, so a positional split would be nondeterministic.
        ids = np.array([f"u{i}" for i in range(200)])
        arms = np.array([0, 1] * 100)
        folds = _fold_ids(ids, arms, 5)
        perm = np.random.default_rng(0).permutation(200)
        shuffled = _fold_ids(ids[perm], arms[perm], 5)
        assert (shuffled == folds[perm]).all()

    def test_deterministic_across_calls(self):
        ids = np.array([f"u{i}" for i in range(50)])
        arms = np.array([0, 1] * 25)
        assert (_fold_ids(ids, arms, 3) == _fold_ids(ids, arms, 3)).all()

    def test_balances_each_arm_across_folds(self):
        ids = np.array([f"unit-{i}" for i in range(23)])
        arms = np.array([0] * 11 + [1] * 12)
        folds = _fold_ids(ids, arms, 5)
        for arm in (0, 1):
            counts = np.bincount(folds[arms == arm], minlength=5)
            assert counts.max() - counts.min() <= 1
            assert (counts >= 1).all()

    def test_k_below_two_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            _fold_ids(np.array(["a", "b"]), np.array([0, 1]), 1)
        assert exc_info.value.code == "estimation.adjust_overlap.cross_fitting_needs"

    def test_too_few_units_in_an_arm_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            _fold_ids(np.array(["a", "b", "c", "d"]), np.array([0, 0, 0, 1]), 3)
        assert exc_info.value.code == "estimation.adjust_overlap.arm_unit_but"
        assert exc_info.value.context["arm"] == 1

    def test_numeric_dtype_rejected(self):
        # str(np.int64(1)) and str(np.float64(1.0)) hash to different folds
        # for the same logical id, so a numeric dtype must be rejected, not silently cast.
        for numeric in (np.array([1, 2, 3], dtype=np.int64), np.array([1.0, 2.0, 3.0])):
            with pytest.raises(InvalidRequestError) as exc_info:
                _fold_ids(numeric, np.array([0, 1, 0]), 2)
            assert exc_info.value.code == "estimation.adjust_overlap.fold_ids_needs"

    def test_cluster_folds_are_atomic_and_order_invariant(self):
        ids = np.array([f"u{i}" for i in range(24)])
        arms = np.array([0] * 12 + [1] * 12)
        clusters = np.array([f"c{arm}-{j // 2}" for arm in (0, 1) for j in range(12)])
        folds = _fold_ids(ids, arms, 3, cluster_ids=clusters)
        perm = np.random.default_rng(9).permutation(ids.size)
        shuffled = _fold_ids(ids[perm], arms[perm], 3, cluster_ids=clusters[perm])
        assert (shuffled == folds[perm]).all()
        for cluster in np.unique(clusters):
            assert np.unique(folds[clusters == cluster]).size == 1

    def test_cluster_folds_balance_mixed_treatment_clusters_without_arm_purity(self):
        # A mixed-treatment observational cluster - members spanning both
        # arms - is never split to force arm purity; it is placed to
        # balance the aggregate per-arm counts across folds instead of
        # refusing (unlike the arm-pure branch, which does refuse below).
        ids = np.array([f"u{i}" for i in range(10)])
        arms = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1])
        clusters = np.array(
            ["shared1", "shared1", "shared2", "shared2", "a1", "a2", "a3", "b1", "b2", "b3"]
        )
        folds = _fold_ids(ids, arms, 2, cluster_ids=clusters)
        for cluster in np.unique(clusters):
            assert np.unique(folds[clusters == cluster]).size == 1
        perm = np.random.default_rng(3).permutation(ids.size)
        shuffled = _fold_ids(ids[perm], arms[perm], 2, cluster_ids=clusters[perm])
        assert (shuffled == folds[perm]).all()

    def test_cluster_folds_balance_a_single_mixed_cluster_with_sufficient_pure_support(self):
        # The mixed cluster must fill the arm support left by pure groups.
        ids = np.array(["c", "t", "m0", "m1"])
        arms = np.array([0, 1, 0, 1])
        clusters = np.array(["c", "t", "shared", "shared"])
        folds = _fold_ids(ids, arms, 2, cluster_ids=clusters)
        for cluster in np.unique(clusters):
            assert np.unique(folds[clusters == cluster]).size == 1
        for f in range(2):
            assert set(arms[folds != f].tolist()) == {0, 1}
        perm = np.random.default_rng(4).permutation(ids.size)
        shuffled = _fold_ids(ids[perm], arms[perm], 2, cluster_ids=clusters[perm])
        assert (shuffled == folds[perm]).all()

    def test_singleton_cluster_folds_are_invariant_to_member_replication(self):
        ids = np.array([f"u{i}" for i in range(12)])
        arms = np.array([0] * 6 + [1] * 6)
        clusters = np.array([f"cluster-{(i * 7) % 13}" for i in range(12)])
        folds = _fold_ids(ids, arms, 3, cluster_ids=clusters)
        repeated = _fold_ids(
            np.repeat(ids, 3),
            np.repeat(arms, 3),
            3,
            cluster_ids=np.repeat(clusters, 3),
        )
        np.testing.assert_array_equal(folds, _fold_ids(ids, arms, 3))
        np.testing.assert_array_equal(repeated, np.repeat(folds, 3))

    def test_cluster_folds_refuse_when_an_arm_has_no_atomic_split(self):
        # Genuine insufficiency: arm 1 has zero pure clusters, and its only
        # representation is a single mixed cluster - unsplittable, so with
        # k=2 folds one fold necessarily holds ALL of arm 1's rows and the
        # other fold's training set would have none.
        ids = np.array([f"u{i}" for i in range(6)])
        arms = np.array([0, 0, 0, 0, 0, 1])
        clusters = np.array(["a1", "a2", "a3", "a4", "shared", "shared"])
        with pytest.raises(InvalidRequestError) as exc_info:
            _fold_ids(ids, arms, 2, cluster_ids=clusters)
        assert exc_info.value.code == "estimation.adjust_overlap.arm_cluster_but"
        assert exc_info.value.context["arm"] == 1

    def test_cluster_folds_require_k_clusters_per_arm(self):
        ids = np.array([f"u{i}" for i in range(8)])
        arms = np.array([0] * 4 + [1] * 4)
        clusters = np.array(["a0", "a0", "a1", "a1", "b0", "b0", "b1", "b1"])
        with pytest.raises(InvalidRequestError) as exc_info:
            _fold_ids(ids, arms, 3, cluster_ids=clusters)
        assert exc_info.value.code == "estimation.adjust_overlap.arm_cluster_but"
        assert exc_info.value.context["clusters"] == 2


# _dml_theta: the pooled DML2 estimating equation, tested white-box against a
# hand-computed example, exercising the sandwich seam (ScoreStats.sum_d_tilde2) directly.


class TestDmlTheta:
    def test_hand_computed_theta_and_sandwich_se(self):
        # d=[1,1,0,0], e_hat=.5 -> d_tilde=[.5,.5,-.5,-.5]; y=[3,5,1,1], m_hat=2 -> y_tilde=[1,3,-1,-1]
        # sum(d_tilde*y_tilde)=3.0, sum(d_tilde**2)=1.0 -> theta_hat=3.0
        d = np.array([1.0, 1.0, 0.0, 0.0])
        e_hat = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.array([3.0, 5.0, 1.0, 1.0])
        m_hat = np.array([2.0, 2.0, 2.0, 2.0])
        theta_hat, psi, scores = _dml_theta(d - e_hat, y - m_hat, metric_name="m", contrast="T")
        assert theta_hat == pytest.approx(3.0)
        # psi_i = d_tilde_i * (y_tilde_i - theta*d_tilde_i)
        # = [.5*(1-1.5), .5*(3-1.5), -.5*(-1+1.5), -.5*(-1+1.5)] = [-0.25, 0.75, -0.25, -0.25]
        assert psi == pytest.approx([-0.25, 0.75, -0.25, -0.25])
        assert isinstance(scores, ScoreStats)
        assert scores.sum_d_tilde2 == pytest.approx(1.0)
        assert scores.sum_psi == pytest.approx(psi.sum())
        # Sandwich SE uses sum_d_tilde2 (armstats.py:151-160) as the normalizer, not n.
        expected_se = math.sqrt((psi**2).sum()) / 1.0
        assert scores.se() == pytest.approx(expected_se)

    def test_estimating_equation_invariant_sum_psi_is_zero(self):
        # sum(psi)==0 at theta_hat IS the estimating equation theta_hat solves,
        # not a coincidence of the hand example; verified here on random data.
        rng = np.random.default_rng(11)
        n = 200
        d = rng.integers(0, 2, size=n).astype(float)
        e_hat = np.clip(rng.normal(0.5, 0.1, size=n), 0.05, 0.95)
        y = rng.normal(size=n)
        m_hat = rng.normal(size=n)
        _, psi, scores = _dml_theta(d - e_hat, y - m_hat, metric_name="m", contrast="T")
        assert abs(psi.sum()) < 1e-8 * np.abs(psi).sum()
        assert abs(scores.sum_psi) < 1e-8 * np.abs(psi).sum()

    def test_zero_residual_treatment_variation_raises(self):
        # e_hat == d exactly -> d_tilde is all zeros -> theta is not
        # identified (the propensity model perfectly predicts treatment).
        d = np.array([1.0, 0.0, 1.0, 0.0])
        e_hat = d.copy()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m_hat = np.zeros(4)
        with pytest.raises(InvalidRequestError) as exc_info:
            _dml_theta(d - e_hat, y - m_hat, metric_name="m", contrast="T")
        assert exc_info.value.code == "estimation.adjust_dml.dml_residualized_treatment"


# DML end-to-end through the public estimate_ate dispatch on a frame-backed
# source, mirroring tests/estimation/test_adjust.py's IPTW fixture idiom.


def _dml_src(n: int = 60):
    """Enough units for the default folds=5 (needs n >= 2*folds=10), a
    covariate `z` that mildly separates the arms without threatening
    overlap, and both arms present."""
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


class _HalfPropensity:
    """Constant e=0.5 propensity stub: keeps the overlap gate quiet so a
    test can reach the guard under study deterministically."""

    def fit(self, X, d):
        pass

    def predict(self, X):

        return np.full(len(X), 0.5)


def _unexpected_learner_factory():
    raise AssertionError("learner factory should not run before fold validation")


_DML_DESIGN = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))


class TestDmlEstimate:
    def test_too_few_units_for_folds_raises(self):
        src = _dml_src(n=6)  # < 2*folds=10 at the default folds=5
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(src, _DML_DESIGN, methods=[Method(name="dml")])
        assert exc_info.value.code == "estimation.adjust_common.needs_least_folds"
        assert exc_info.value.context["needed"] == 10

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
        design = _DML_DESIGN.model_copy(
            update={"adjustment": AdjustmentSet(covariates=("x",), missing="allow")}
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(
                src,
                design,
                methods=[
                    Method(
                        name="dml",
                        folds=3,
                        propensity_learner=_unexpected_learner_factory,
                        outcome_learner=_unexpected_learner_factory,
                    )
                ],
            )
        assert exc_info.value.code == "estimation.adjust_overlap.arm_unit_but"

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
        src = from_unit_summary(
            tbl,
            unit="user_id",
            group="variant",
            control="C",
            metrics=[
                {"name": "rpo", "type": "ratio", "numerator": "spend", "denominator": "orders"}
            ],
        )
        design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            computation = estimate_ate(src, design, methods=[Method(name="dml", folds=2)])
        assert computation.results == ()
        (failure,) = computation.failures.values()
        # See the typed ratio refusal contract at docs/guides/observational.md:941-943.
        assert failure.code == "estimation.adjust_common.supported_ratio_metric"

    def test_control_absent_raises_and_lists_available_groups(self):
        src = _dml_src()
        design = Observational(control_group="ZZ", adjustment=AdjustmentSet(covariates=("x",)))
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_ate(src, design, methods=[Method(name="dml")])
        assert exc_info.value.code == "estimation.adjust_common.control_group_found"
        assert exc_info.value.context["control_group"] == "ZZ"
        available = exc_info.value.context["available"]
        assert isinstance(available, (list, tuple))
        assert "T" in available

    def test_row_shuffle_gives_order_stable_estimate(self):
        # Fold membership keys on unit_id, not row position: same DGP in two row
        # orders matches to float precision; a positional-split bug would shift ~1e-2, far looser.
        def _dml_src_permuted(order: np.ndarray):
            rng2 = np.random.default_rng(4)
            n_units = 60
            x = rng2.normal(size=n_units)
            d = (rng2.random(n_units) < expit(0.3 * x)).astype(int)
            y = 1.0 + 0.8 * x + 0.5 * d + rng2.normal(scale=0.3, size=n_units)
            tbl = pa.table(
                {
                    "user_id": np.array([f"u{i}" for i in range(n_units)])[order],
                    "variant": np.where(d == 1, "T", "C")[order],
                    "revenue": y[order],
                    "x": x[order],
                }
            )
            return from_unit_summary(
                tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
            )

        identity_order = np.arange(60)
        src1 = _dml_src_permuted(identity_order)
        (first,) = estimate_ate(src1, _DML_DESIGN, methods=[Method(name="dml", folds=3)]).results

        shuffled_order = np.random.default_rng(1).permutation(60)
        src2 = _dml_src_permuted(shuffled_order)
        (second,) = estimate_ate(src2, _DML_DESIGN, methods=[Method(name="dml", folds=3)]).results
        assert first.require_lift().value == pytest.approx(
            second.require_lift().value, rel=0, abs=1e-9
        )
        assert first.require_lift().lb == pytest.approx(second.require_lift().lb, rel=0, abs=1e-9)
        assert first.require_lift().ub == pytest.approx(second.require_lift().ub, rel=0, abs=1e-9)

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
            estimate_ate(src, design, methods=[Method(name="dml", folds=5)])
        assert exc_info.value.code == "adjust.identification.overlap_gate_refused"

    def test_registry_dispatch_via_estimate_ate(self):
        src = _dml_src()
        ests = estimate_ate(src, _DML_DESIGN, methods=[Method(name="dml")]).results
        assert len(ests) == 1
        assert ests[0].method == "dml"

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_arm_stratification_avoids_degenerate_hash_refusal(self):
        """Arm stratification must populate every fold even when raw hashes
        all land in one modulo bucket."""
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
        estimates = estimate_ate(
            src,
            _DML_DESIGN,
            methods=[Method(name="dml", folds=2)],
            value_scale={"revenue": "absolute"},
        ).results
        assert len(estimates) == 1

    @pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
    def test_near_zero_control_preserves_joint_set_and_additive_inference(self):
        """A weak denominator need not erase an identified additive effect."""
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
        (result,) = estimate_ate(
            src,
            _DML_DESIGN,
            methods=[Method(name="dml", propensity_learner=_HalfPropensity, folds=2)],
        ).results
        assert result.relative_confidence_set is not None
        assert result.relative_confidence_set.geometry in ("disconnected", "all_real")
        assert result.abs_diff is not None and result.abs_diff > 3
        assert result.abs_lb is not None and result.abs_lb > 0

    def test_balance_gate_message_names_the_propensity_diagnostic(self):
        """DML residualizes, it never weights - the balance refusal is a
        propensity-model quality check under IPW weights, and its message
        must say that instead of describing an estimator that isn't
        running."""
        tbl = pa.table(
            {
                "user_id": [f"u{i}" for i in range(12)],
                "variant": ["T"] * 6 + ["C"] * 6,
                "revenue": [3.0, 5.0, 4.0, 4.5, 3.5, 4.2, 1.0, 2.0, 1.5, 1.2, 1.8, 1.4],
                # x tracks the arm perfectly: SMD is huge, overlap is fine
                # (constant stub propensities), so ONLY the balance gate can fire.
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
            estimate_ate(
                src,
                design,
                methods=[Method(name="dml", propensity_learner=_HalfPropensity, folds=2)],
            )
        assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


class _TruePropensity:
    """The denominator cohort's true propensity: 0.2 at x == 0, 0.8 at x == 1."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.where(np.asarray(X)[:, 0] == 1.0, 0.8, 0.2)


class _NullOutcome:
    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.zeros(np.asarray(X).shape[0])


def _denominator_src():
    """Heterogeneous effects: at x = (0, 1) the propensity is (.2, .8), the
    control mean (2, 6) and the effect (1, 3), with 50 noiseless units at each
    x. E[Y(0)] = 4, while the unweighted control mean is 2.8. x is spread
    evenly over the five cross-fitting folds, so every training fold keeps
    each (arm, x) share and saturated learners fit population nuisances."""
    ids = np.array([f"u{i:03d}" for i in range(100)])
    d = np.repeat([0, 1], 50)
    fold = _fold_ids(ids, d, 5)
    x = np.zeros(100)
    for arm, per_fold in ((0, 2), (1, 8)):
        for j in range(5):
            x[np.flatnonzero((d == arm) & (fold == j))[:per_fold]] = 1.0
    y = np.where(x == 0.0, 2.0, 6.0) + d * np.where(x == 0.0, 1.0, 3.0)
    return from_unit_summary(
        pa.table(
            {
                "user_id": ids,
                "variant": np.where(d == 1, "T", "C"),
                "revenue": y,
                "x": x,
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
@pytest.mark.parametrize(
    ("propensity", "outcome", "slope"),
    [(_HalfPropensity, _CellMean, 1.28), (_TruePropensity, _NullOutcome, 2.0)],
    ids=["wrong-propensity-exact-control-regression", "true-propensity-null-regression"],
)
def test_relative_denominator_is_the_augmented_control_mean(propensity, outcome, slope):
    """A relative row divides the unchanged PLR slope by
    mean(m0) + sum w0 (Y - m0) / sum w0 over the cohort. An exact control-arm
    regression under a wrong constant propensity, or the true propensity under
    a null regression, both give E[Y(0)] = 4; the fixed-propensity Hajek
    control mean under e = .5 gives 2.8, and reusing the exact pooled
    regression as the control regression gives 4.66. The slope keeps its own
    robustness profile: 1.28 under the wrong propensity (relative 0.32, where
    the relative ATE is 0.5) and 2.0 under the true one."""
    src = _denominator_src()
    method = Method(name="dml", propensity_learner=propensity, outcome_learner=outcome, folds=5)
    (relative,) = estimate_ate(src, _DML_DESIGN, methods=[method]).results
    (absolute,) = estimate_ate(
        src, _DML_DESIGN, methods=[method], value_scale={"revenue": "absolute"}
    ).results
    assert relative.relative_confidence_set is not None
    assert relative.relative_confidence_set.reference.c == pytest.approx(4.0, rel=1e-12)
    assert absolute.require_lift().value == pytest.approx(slope, rel=1e-12)
    assert relative.abs_diff == pytest.approx(absolute.require_lift().value, rel=1e-14)
    assert relative.require_lift().value == pytest.approx(slope / 4.0, rel=1e-12)


class _SharedStreamOutcome:
    """Outcome learner whose every fit draws from one shared random stream,
    like an ensemble consuming process-wide random state."""

    stream = np.random.default_rng(0)

    def fit(self, X, d):
        self._level = float(np.mean(d)) + type(self).stream.normal()

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self._level)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_relative_control_fits_never_perturb_the_slope_fits():
    """The control-arm regression is fitted only after every slope fit, on the
    same folds, so a learner drawing on shared state sees the slope's fit
    schedule unchanged: the relative row's slope equals the absolute row's."""
    src = _dml_src(n=60)
    method = Method(
        name="dml",
        propensity_learner=_HalfPropensity,
        outcome_learner=_SharedStreamOutcome,
        folds=3,
    )
    _SharedStreamOutcome.stream = np.random.default_rng(7)
    (relative,) = estimate_ate(src, _DML_DESIGN, methods=[method]).results
    _SharedStreamOutcome.stream = np.random.default_rng(7)
    (absolute,) = estimate_ate(
        src, _DML_DESIGN, methods=[method], value_scale={"revenue": "absolute"}
    ).results
    assert relative.abs_diff == pytest.approx(absolute.require_lift().value, rel=1e-14)


# Parameter recovery: does DML recover a known effect under confounding that
# biases the naive comparison, through the full Analysis(design=Observational(...)) entry point.


def _confounded_table(n: int, seed: int, effect: float = 2.0) -> pa.Table:
    """x drives BOTH treatment assignment (P(D=1|x)=expit(0.8x)) and the
    outcome, so the naive (unweighted) comparison is biased; DML residualizes
    it out. Additive ATE is exactly `effect`; the relative lift divides by
    the augmented control-arm mean, a consistent estimate of E[Y(0)]=10.0
    despite the selection on x - so `_TRUE_RELATIVE_LIFT = effect / 10.0` is
    exact, not selection-shifted.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = (rng.random(n) < expit(0.8 * x)).astype(int)
    y = 10.0 + 3.0 * x + effect * d + rng.normal(scale=0.5, size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x": x,
        }
    )


_OBS_DML = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
_TRUE_ADDITIVE_ATE = 2.0
_TRUE_RELATIVE_LIFT = _TRUE_ADDITIVE_ATE / 10.0


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_dml_recovers_theta_where_naive_is_biased():
    an_dml = Analysis.from_unit_summary(
        _confounded_table(600, seed=3),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=_OBS_DML,
    )
    (dml,) = lift_rows(an_dml.run(decision_method=Method(name="dml")))

    an_naive = Analysis.from_unit_summary(
        _confounded_table(600, seed=3),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    (naive,) = lift_rows(an_naive.run())

    dml_lift = dml.require_lift()
    naive_lift = naive.require_lift()
    assert abs(dml_lift.value - _TRUE_RELATIVE_LIFT) < abs(naive_lift.value - _TRUE_RELATIVE_LIFT)
    assert abs(naive_lift.value - _TRUE_RELATIVE_LIFT) > 0.05  # confounding is real
    assert dml_lift.lb is not None and dml_lift.ub is not None
    assert dml_lift.lb < _TRUE_RELATIVE_LIFT < dml_lift.ub


# Multi-seed bias/coverage check reusing _confounded_table's DGP, design, and
# ground truth; marked parameter_recovery. Smoke variant below stays in the fast suite.

_RECOVERY_REPS = 200
# n=800: DML's ridge/logistic regularization bias is orthogonality's
# second-order remainder (~-0.0012 at n=400, ~-0.0002 at n=800), safely inside the 2-MC-SE bias check.
_RECOVERY_N = 800
# overlap="trim": at n=800 this DGP's e(x) occasionally produces near-0/1
# propensities that would trip the default refuse-gate and abort a replication.
_RECOVERY_GATE = Observational(
    control_group="C",
    adjustment=AdjustmentSet(covariates=("x",)),
    gate=IdentificationGate(overlap="trim"),
)


@pytest.fixture(scope="module")
def _dml_recovery_replications():
    """Run DML and the naive estimator once per replication, fresh seed
    each time (seed=100+i), and return per-replication errors plus DML's
    nominal-CI hit rate. Shared across the bias and coverage tests below
    via a module-scoped fixture so the 200-replication Monte Carlo loop
    (the expensive part) runs exactly once per test session."""
    dml_errors = []
    naive_errors = []
    hits = 0
    for i in range(_RECOVERY_REPS):
        table = _confounded_table(_RECOVERY_N, seed=100 + i)
        an_dml = Analysis.from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=_RECOVERY_GATE,
        )
        an_naive = Analysis.from_unit_summary(
            table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
                category=UserWarning,
            )
            (dml,) = lift_rows(an_dml.run(decision_method=Method(name="dml")))
            (naive,) = lift_rows(an_naive.run())
        dml_lift = dml.require_lift()
        naive_lift = naive.require_lift()
        dml_errors.append(dml_lift.value - _TRUE_RELATIVE_LIFT)
        naive_errors.append(naive_lift.value - _TRUE_RELATIVE_LIFT)
        assert dml_lift.lb is not None and dml_lift.ub is not None
        hits += dml_lift.lb < _TRUE_RELATIVE_LIFT < dml_lift.ub
    return {
        "dml_errors": np.array(dml_errors),
        "naive_errors": np.array(naive_errors),
        "coverage": hits / _RECOVERY_REPS,
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.xdist_group("dml_recovery")
class TestDmlParameterRecovery:
    def test_bias_within_two_monte_carlo_se_while_naive_is_not(self, _dml_recovery_replications):
        dml_errors = _dml_recovery_replications["dml_errors"]
        naive_errors = _dml_recovery_replications["naive_errors"]
        n = len(dml_errors)
        dml_mc_se = dml_errors.std(ddof=1) / math.sqrt(n)
        naive_mc_se = naive_errors.std(ddof=1) / math.sqrt(n)
        # DML: mean error is statistically indistinguishable from zero.
        assert abs(dml_errors.mean()) < 2 * dml_mc_se
        # Naive: confounding leaves a bias that swamps its own Monte Carlo SE -
        # not near-unbiased at the tolerance DML just passed; the actual comparative claim.
        assert abs(naive_errors.mean()) > 10 * naive_mc_se

    def test_ci_coverage_near_nominal(self, _dml_recovery_replications):
        # Band ~2 MC-SEs of a 200-rep coverage estimate (SE~=0.0154 -> [0.919,0.981],
        # rounded to [0.91,0.99]) - loose enough to not flip on an unrelated BLAS/numpy bump.
        coverage = _dml_recovery_replications["coverage"]
        assert 0.91 <= coverage <= 0.99


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_dml_recovery_smoke_single_replication_is_a_real_interval():
    """NOT a statistical claim - a single replication at small n, run in
    the default fast suite so a broken recovery pipeline (crash, NaN,
    degenerate interval) is caught immediately rather than only on the
    slow `parameter_recovery` suite. Asserts only that the machinery runs
    and produces a genuine finite interval around a finite point.

    Uses `_RECOVERY_GATE` (overlap="trim"), not `_OBS_DML` (default
    overlap="refuse") - at n=60 the cross-fitted propensities have
    little positivity margin, so the default gate could occasionally
    raise `IdentificationError` instead of exercising these assertions, a
    hard fast-suite failure rather than the pipeline-health signal this
    test exists to be.
    """
    table = _confounded_table(60, seed=999)
    an = Analysis.from_unit_summary(
        table, unit="user_id", group="variant", metrics={"revenue": "mean"}, design=_RECOVERY_GATE
    )
    (dml,) = lift_rows(an.run(decision_method=Method(name="dml")))
    dml_lift = dml.require_lift()
    assert math.isfinite(dml_lift.value)
    assert dml_lift.lb is not None and dml_lift.ub is not None
    assert math.isfinite(dml_lift.lb) and math.isfinite(dml_lift.ub)
    assert dml_lift.lb < dml_lift.value < dml_lift.ub


def test_dml_pooled_outcome_and_propensity_fits_share_each_split_encoding():
    """DML's pooled partialling-out regression and its propensity are fitted
    on the same training rows per fold: both see the same level indicator
    set there, and the fold holding out the lone ``island`` unit fits one
    indicator fewer than the folds that train on it."""
    from tests.categorical_cases import categorical_units, raw_table

    units = categorical_units(300, seed=21)
    units["region"] = units["region"].astype(object)
    units["region"][5] = "island"
    widths: dict[str, list[int]] = {"propensity": [], "outcome": []}

    def factory(label: str):
        class Recording:
            def fit(self, X, d):
                widths[label].append(np.asarray(X).shape[1])
                self._value = float(np.mean(d))

            def predict(self, X):
                assert np.asarray(X).shape[1] == widths[label][-1]
                return np.full(np.asarray(X).shape[0], self._value)

        return Recording

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
                    name="dml",
                    propensity_learner=factory("propensity"),
                    outcome_learner=factory("outcome"),
                    folds=3,
                )
            ],
        )
    assert sorted(widths["propensity"]) == [3, 4, 4]
    # Pooled pair fits per fold, then the deferred control-arm fits on the
    # same folds: the control arm may or may not hold the island unit.
    assert widths["outcome"][:3] == widths["propensity"]
    assert len(widths["outcome"]) == 6
    assert all(width in (3, 4) for width in widths["outcome"][3:])
