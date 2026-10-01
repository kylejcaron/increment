"""Unit tests for the CATE design assembly."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from increment.errors import InvalidRequestError
from increment.estimation.cate import CateResult, Covariate, DesignSpec, fit_cate, prune_gram

COLS = {
    "spend": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    "platform": np.array(["ios", "android", "ios", "web", "ios", "android"]),
}


class TestDesignSpec:
    def test_continuous_is_standardized(self):
        spec = DesignSpec.fit(COLS, [Covariate(name="spend")])
        z, names = spec.transform(COLS)
        assert names == ("spend",)
        assert z[:, 0].mean() == pytest.approx(0.0)
        assert z[:, 0].std(ddof=1) == pytest.approx(1.0)

    def test_transform_reuses_fitted_moments(self):
        """Scoring new units MUST use the training mean/std, not its own."""
        spec = DesignSpec.fit(COLS, [Covariate(name="spend")])
        z, _ = spec.transform({"spend": np.array([3.5, 3.5])})
        assert z[0, 0] == pytest.approx((3.5 - 3.5) / np.std(COLS["spend"], ddof=1))

    def test_categorical_one_hot_drops_the_modal_level(self):
        spec = DesignSpec.fit(COLS, [Covariate(name="platform", kind="categorical")])
        z, names = spec.transform(COLS)
        assert names == ("platform=android", "platform=web")  # ios is modal -> reference
        assert z[:, 0].tolist() == [0, 1, 0, 0, 0, 1]

    def test_unseen_level_maps_to_reference(self):
        spec = DesignSpec.fit(COLS, [Covariate(name="platform", kind="categorical")])
        z, _ = spec.transform({"platform": np.array(["tv"])})
        assert z[0].tolist() == [0.0, 0.0]

    def test_zero_variance_continuous_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            DesignSpec.fit({"spend": np.ones(4)}, [Covariate(name="spend")])
        assert exc_info.value.code == "estimation.cate.design.covariate_zero_variance"

    def test_missing_column_raises_by_name(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            DesignSpec.fit(COLS, [Covariate(name="tenure")])
        assert exc_info.value.code == "estimation.cate.covariate_missing_from"

    def test_nulls_raise_by_name(self):
        cols = {"spend": np.array([1.0, np.nan, 3.0])}
        with pytest.raises(InvalidRequestError) as exc_info:
            DesignSpec.fit(cols, [Covariate(name="spend")])
        assert exc_info.value.code == "estimation.cate.covariate_nulls_non"

    def test_mixed_covariates_keep_block_order(self):
        """Names MUST line up with the horizontal-stack order of the blocks."""
        spec = DesignSpec.fit(
            COLS, [Covariate(name="spend"), Covariate(name="platform", kind="categorical")]
        )
        z, names = spec.transform(COLS)
        assert names == ("spend", "platform=android", "platform=web")
        expected_spend = (COLS["spend"] - 3.5) / np.std(COLS["spend"], ddof=1)
        assert z[:, 0] == pytest.approx(expected_spend)
        assert z[:, 1].tolist() == [0, 1, 0, 0, 0, 1]  # android
        assert z[:, 2].tolist() == [0, 0, 0, 1, 0, 0]  # web

    def test_mixed_covariates_round_trip_on_new_rows(self):
        spec = DesignSpec.fit(
            COLS, [Covariate(name="spend"), Covariate(name="platform", kind="categorical")]
        )
        z, names = spec.transform(
            {"spend": np.array([3.5, 6.0]), "platform": np.array(["web", "ios"])}
        )
        scale = np.std(COLS["spend"], ddof=1)
        assert names == ("spend", "platform=android", "platform=web")
        assert z[0].tolist() == pytest.approx([0.0, 0.0, 1.0])
        assert z[1].tolist() == pytest.approx([(6.0 - 3.5) / scale, 0.0, 0.0])

    def test_ragged_columns_raise(self):
        spec = DesignSpec.fit(
            COLS, [Covariate(name="spend"), Covariate(name="platform", kind="categorical")]
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            spec.transform({"spend": np.array([1.0, 2.0]), "platform": np.array(["ios"])})
        assert exc_info.value.code == "estimation.cate.design.covariate_rows_expected"

    def test_object_column_nulls_raise_by_name(self):
        cols = {"platform": np.array(["ios", None, "web"], dtype=object)}
        with pytest.raises(InvalidRequestError) as exc_info:
            DesignSpec.fit(cols, [Covariate(name="platform", kind="categorical")])
        assert exc_info.value.code == "estimation.cate.covariate_nulls"

    def test_int_knots_land_on_interior_quantiles(self):
        spec = DesignSpec.fit(COLS, [Covariate(name="spend", knots=3)])
        assert spec.transforms[0].knots == pytest.approx(
            tuple(np.quantile(COLS["spend"], [0.25, 0.5, 0.75]))
        )
        assert spec.column_names() == ("spend", "spend>k1", "spend>k2", "spend>k3")

    def test_explicit_knots_are_used_as_given(self):
        spec = DesignSpec.fit(COLS, [Covariate(name="spend", knots=(2.0, 5.5))])
        assert spec.transforms[0].knots == (2.0, 5.5)
        assert spec.column_names() == ("spend", "spend>k1", "spend>k2")

    def test_hinges_are_flat_below_the_knot_and_share_the_base_column_scale(self):
        spec = DesignSpec.fit(COLS, [Covariate(name="spend", knots=(2.0, 5.0))])
        z, names = spec.transform(COLS)
        scale = COLS["spend"].std(ddof=1)
        assert names == ("spend", "spend>k1", "spend>k2")
        assert z[:, 1] == pytest.approx(np.maximum(COLS["spend"] - 2.0, 0.0) / scale)
        assert z[:, 2] == pytest.approx(np.maximum(COLS["spend"] - 5.0, 0.0) / scale)

    def test_transform_reuses_fitted_knots(self):
        """Scoring new units MUST NOT re-derive the knots from their own column."""
        spec = DesignSpec.fit(COLS, [Covariate(name="spend", knots=2)])
        new = {"spend": np.array([10.0, 20.0, 30.0, 40.0])}
        knots = spec.transforms[0].knots
        assert knots != DesignSpec.fit(new, [Covariate(name="spend", knots=2)]).transforms[0].knots
        mean, scale = COLS["spend"].mean(), COLS["spend"].std(ddof=1)
        z, _ = spec.transform(new)
        assert z[0] == pytest.approx([(10.0 - mean) / scale, *((10.0 - k) / scale for k in knots)])


class TestCovariateKnots:
    def test_categorical_with_knots_is_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Covariate(name="platform", kind="categorical", knots=3)
        assert exc_info.value.code == "estimation.cate.covariate.categorical_so_knots"

    def test_non_positive_knot_count_is_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Covariate(name="spend", knots=0)
        assert exc_info.value.code == "estimation.cate.covariate.needs_least_one"

    def test_empty_knot_tuple_is_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Covariate(name="spend", knots=())
        assert exc_info.value.code == "estimation.cate.covariate.was_empty_knot"

    def test_repeated_or_unordered_knots_are_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Covariate(name="spend", knots=(2.0, 2.0))
        assert exc_info.value.code == "estimation.cate.covariate.needs_strictly_increasing"

    def test_non_finite_knots_are_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Covariate(name="spend", knots=(1.0, float("inf")))
        assert exc_info.value.code == "estimation.cate.covariate.non_finite_knots"


class TestPruneGram:
    def test_full_rank_keeps_everything(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((50, 4))
        zz, kept, pruned = prune_gram(z.T @ z, ["a", "b", "c", "d"])
        assert kept == (0, 1, 2, 3)
        assert pruned == ()

    def test_exact_collinearity_prunes_the_later_column_by_name(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((50, 3))
        z = np.column_stack([x, x[:, 0] + x[:, 1]])  # col 3 = col 0 + col 1
        zz, kept, pruned = prune_gram(z.T @ z, ["a", "b", "c", "a_plus_b"])
        assert pruned == ("a_plus_b",)
        assert kept == (0, 1, 2)

    def test_protected_columns_are_never_pruned(self):
        z = np.column_stack([np.ones(10), np.ones(10)])  # intercept duplicated
        with pytest.raises(InvalidRequestError) as exc_info:
            prune_gram(z.T @ z, ["intercept", "d"], protect=2)
        assert exc_info.value.code == "estimation.cate.protected_column_collinear"

    def test_non_square_gram_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            prune_gram(np.ones((2, 3)), ["a", "b", "c"])
        assert exc_info.value.code == "estimation.cate.zz_square_gram"

    def test_name_count_must_match_gram_dimension(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            prune_gram(np.eye(2), ["a", "b", "c"])
        assert exc_info.value.code == "estimation.cate.names_entries_gram"


def _sim(n=400, seed=0, tau=0.5, gamma=0.3):
    """y = x0 + D*(tau + gamma*x0) + eps, one binary covariate too."""
    r = np.random.default_rng(seed)
    x = r.standard_normal(n)
    plat = np.where(r.random(n) < 0.4, "ios", "android")
    d = (r.random(n) < 0.5).astype(float)
    y = x + d * (tau + gamma * (x - x.mean()) / x.std(ddof=1)) + r.standard_normal(n)
    return y, d, {"spend": x, "platform": plat}


def _sim_partial(n=400, seed=5):
    """A categorical whose 'android' level duplicates an earlier 0/1 column.

    Only that one level is collinear, so pruning takes 'd:platform=android'
    and leaves 'd:platform=web' standing: the partially-pruned case.
    """
    r = np.random.default_rng(seed)
    u = r.random(n)
    plat = np.where(u < 0.45, "ios", np.where(u < 0.75, "android", "web"))
    x = r.standard_normal(n)
    d = (r.random(n) < 0.5).astype(float)
    y = x + d * (0.5 + 0.3 * (plat == "web")) + r.standard_normal(n)
    return y, d, {"is_android": (plat == "android").astype(float), "platform": plat}


class TestFitCate:
    def test_cluster_identity_must_align_with_outcomes(self):
        y, d, cols = _sim()
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(y, d, cols, interact=[], cluster_ids=np.arange(y.size - 1))
        assert exc_info.value.code == "estimation.cate.cluster_ids_shape"

    def test_cluster_intervention_requires_cluster_identity(self):
        y, d, cols = _sim()
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(y, d, cols, interact=[], intervention_grain="cluster")
        assert exc_info.value.code == "estimation.cate.intervention_grain_without_cluster"

    def test_no_covariates_reproduces_difference_in_means(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, {}, interact=[])
        dim = y[d == 1].mean() - y[d == 0].mean()
        assert res.ate == pytest.approx(dim, rel=1e-12)

    def test_ate_equals_full_ols_d_coefficient(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        # Reference fit: numpy lstsq on the identical centered design.
        xs = (cols["spend"] - cols["spend"].mean()) / cols["spend"].std(ddof=1)
        z = np.column_stack([np.ones(len(y)), d, xs, d * (xs - xs.mean())])
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        assert res.ate == pytest.approx(beta[1], abs=1e-10)

    def test_cate_at_the_mean_is_the_ate(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        est = res.cate({"spend": float(cols["spend"].mean())})
        assert est.value == pytest.approx(res.ate, abs=1e-9)

    def test_contrast_matches_cate_difference(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        hi, lo = res.cate({"spend": 5.0}), res.cate({"spend": 1.0})
        diff = res.contrast({"spend": 5.0}, {"spend": 1.0})
        assert diff.value == pytest.approx(hi.value - lo.value, abs=1e-9)

    def test_score_is_vectorized_cate(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        s = res.score({"spend": np.array([1.0, 5.0])}, deploy_grain="unit")
        assert s[0] == pytest.approx(res.cate({"spend": 1.0}).value, abs=1e-9)

    def test_cate_on_a_categorical_matches_an_independent_ols_prediction(self):
        """String levels reach the contrast vector by a different path."""
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="platform", kind="categorical")])
        assert {i.name for i in res.interactions} == {"d:platform=ios"}
        # Reference fit: 'android' is modal and so is the reference level,
        # leaving one 'ios' indicator, centered in both blocks.
        ios = (cols["platform"] == "ios").astype(float)
        centered = ios - ios.mean()
        z = np.column_stack([np.ones(len(y)), d, centered, d * centered])
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        assert res.cate({"platform": "ios"}).value == pytest.approx(
            beta[1] + beta[3] * (1.0 - ios.mean()), abs=1e-10
        )
        assert res.cate({"platform": "android"}).value == pytest.approx(
            beta[1] - beta[3] * ios.mean(), abs=1e-10
        )

    def test_cate_basis_identity_survives_colliding_display_labels_and_pruning(self):
        rng = np.random.default_rng(7)
        n = 400
        d = np.arange(n) % 2
        platform = np.where((np.arange(n) // 2) % 2 == 0, "A", "B")
        z = rng.normal(size=n)
        y = 10 + 2 * d + 3 * d * (platform == "B") + 5 * d * z + rng.normal(0, 0.1, n)
        interact = [
            Covariate(name="x", kind="categorical"),
            Covariate(name="x=B"),
            Covariate(name="x_copy"),
        ]
        result = fit_cate(y, d, {"x": platform, "x=B": z, "x_copy": z}, interact=interact)
        renamed = fit_cate(
            y,
            d,
            {"x": platform, "z": z, "z_copy": z},
            interact=[
                Covariate(name="x", kind="categorical"),
                Covariate(name="z"),
                Covariate(name="z_copy"),
            ],
        )
        point = result.cate({"x": "B", "x=B": 1.0})
        assert point.value == pytest.approx(10.01076828893449)
        assert point.value == pytest.approx(renamed.cate({"x": "B", "z": 1.0}).value)
        assert result.contrast(
            {"x": "B", "x=B": 1.0}, {"x": "A", "x=B": 0.0}
        ).value == pytest.approx(renamed.contrast({"x": "B", "z": 1.0}, {"x": "A", "z": 0.0}).value)
        scored = result.score({"x": platform, "x=B": z}, deploy_grain="unit")
        np.testing.assert_allclose(
            scored, renamed.score({"x": platform, "z": z}, deploy_grain="unit")
        )
        from increment.estimation.cate import CateScoreState

        restored = CateScoreState.model_validate(result.score_state.model_dump())
        np.testing.assert_allclose(restored.score({"x": platform, "x=B": z}), scored)

    @pytest.mark.parametrize("ard", [False, True])
    @pytest.mark.parametrize("clustered", [False, True])
    def test_hinge_and_repeated_display_labels_preserve_ard_cluster_predictions(
        self, ard, clustered
    ):
        rng = np.random.default_rng(17)
        n = 400
        d = np.arange(n) % 2
        category = np.where((np.arange(n) // 2) % 2 == 0, "A", "B>k1")
        z = rng.normal(size=n)
        y = 4 + 2 * d + 3 * d * (category == "B>k1") + 5 * d * z + rng.normal(0, 0.1, n)
        ids = np.arange(n) // 4 if clustered else None
        kwargs = {"cluster_ids": ids} if ids is not None else {}
        result = fit_cate(
            y,
            d,
            {"x": category, "x=B": z},
            interact=[Covariate(name="x", kind="categorical"), Covariate(name="x=B", knots=(0.0,))],
            ard=ard,
            **kwargs,
        )
        renamed = fit_cate(
            y,
            d,
            {"x": category, "z": z},
            interact=[Covariate(name="x", kind="categorical"), Covariate(name="z", knots=(0.0,))],
            ard=ard,
            **kwargs,
        )
        a = {"x": "B>k1", "x=B": 1.0}
        b = {"x": "A", "x=B": -1.0}
        a_renamed = {"x": "B>k1", "z": 1.0}
        b_renamed = {"x": "A", "z": -1.0}
        assert result.cate(a).value == pytest.approx(renamed.cate(a_renamed).value)
        assert result.contrast(a, b).value == pytest.approx(
            renamed.contrast(a_renamed, b_renamed).value
        )
        scored = result.score({"x": category, "x=B": z}, deploy_grain="unit")
        np.testing.assert_allclose(
            scored, renamed.score({"x": category, "z": z}, deploy_grain="unit")
        )
        from increment.estimation.cate import CateScoreState

        restored = CateScoreState.model_validate_json(result.score_state.model_dump_json())
        np.testing.assert_allclose(restored.score({"x": category, "x=B": z}), scored)

    def test_contrast_across_two_covariates_matches_an_independent_fit(self):
        """Two points may differ in every covariate at once."""
        y, d, cols = _sim()
        res = fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="spend"), Covariate(name="platform", kind="categorical")],
        )
        spend = cols["spend"]
        scale = spend.std(ddof=1)
        basis = np.column_stack(
            [(spend - spend.mean()) / scale, (cols["platform"] == "ios").astype(float)]
        )
        centered = basis - basis.mean(axis=0)
        z = np.column_stack([np.ones(len(y)), d, centered, d[:, None] * centered])
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        # The treatment entries and the grand-mean centering both cancel, so
        # the contrast is the interaction block against the raw basis gap.
        gap = np.array([(2.0 - spend.mean()) / scale, 1.0]) - np.array(
            [(-1.0 - spend.mean()) / scale, 0.0]
        )
        got = res.contrast(
            {"spend": 2.0, "platform": "ios"}, {"spend": -1.0, "platform": "android"}
        )
        assert got.value == pytest.approx(float(beta[4:] @ gap), abs=1e-10)
        assert got.lb is not None and got.ub is not None
        assert got.lb < got.value < got.ub

    def test_intervals_use_the_t_distribution_at_n_minus_p_degrees_of_freedom(self):
        """HC2 sandwich intervals must use a Student-t reference, not a normal
        one: with only a handful of columns spent on n=30 rows, the two
        references disagree by several percent and only the t-based width
        matches a hand-built critical value.
        """
        from scipy.stats import norm
        from scipy.stats import t as t_dist

        y, d, cols = _sim(n=30)
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        p = len(res.columns)
        dof = res.n - p
        assert dof < 30  # a moderate design width, where t and normal visibly diverge

        expected_half = float(t_dist.ppf(0.975, dof)) * res.se
        stale_half = float(norm.ppf(0.975)) * res.se
        assert (res.ub - res.ate) == pytest.approx(expected_half, rel=1e-10)
        assert (res.ub - res.ate) > stale_half

        effect = res.interactions[0]
        expected_interaction_half = float(t_dist.ppf(0.975, dof)) * effect.se
        stale_interaction_half = float(norm.ppf(0.975)) * effect.se
        assert (effect.ub - effect.coef) == pytest.approx(expected_interaction_half, rel=1e-10)
        assert (effect.ub - effect.coef) > stale_interaction_half

        # The cate()/contrast() path (CateResult._estimate) must agree: at the
        # covariate mean the loading vector matches the ATE's exactly, so the
        # two intervals should coincide only if _estimate used the same
        # Student-t critical value at the same degrees of freedom.
        point = res.cate({"spend": float(cols["spend"].mean())})
        assert point.lb == pytest.approx(res.lb, rel=1e-10)
        assert point.ub == pytest.approx(res.ub, rel=1e-10)

    def test_cate_names_a_missing_covariate(self):
        y, d, cols = _sim()
        res = fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="spend"), Covariate(name="platform", kind="categorical")],
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            res.cate({"spend": 1.0})
        assert exc_info.value.code == "estimation.cate.cate.covariate_missing_from"

    def test_adjust_only_covariates_do_not_enter_the_interaction_block(self):
        y, d, cols = _sim()
        res = fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="spend")],
            adjust=[Covariate(name="platform", kind="categorical")],
        )
        assert {i.name for i in res.interactions} == {"d:spend"}

    def test_collinear_interaction_is_pruned_and_named(self):
        y, d, cols = _sim()
        cols = dict(cols) | {"spend_copy": cols["spend"].copy()}
        res = fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="spend"), Covariate(name="spend_copy")],
        )
        assert "d:spend_copy" in res.pruned
        # contrasts still work by name after pruning
        res.cate({"spend": 1.0, "spend_copy": 1.0})

    def test_pruned_covariate_is_no_longer_required(self):
        """A dropped column contributes nothing, so callers need not supply it."""
        y, d, cols = _sim()
        cols = dict(cols) | {"spend_copy": cols["spend"].copy()}
        res = fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="spend"), Covariate(name="spend_copy")],
        )
        assert "d:spend_copy" in res.pruned
        lean, full = res.cate({"spend": 1.0}), res.cate({"spend": 1.0, "spend_copy": 4.0})
        assert (lean.value, lean.lb, lean.ub) == (full.value, full.lb, full.ub)
        assert res.contrast({"spend": 5.0}, {"spend": 1.0}).value == pytest.approx(
            res.contrast(
                {"spend": 5.0, "spend_copy": 0.0}, {"spend": 1.0, "spend_copy": 9.0}
            ).value,
            abs=1e-12,
        )
        assert res.score({"spend": np.array([1.0, 5.0])}, deploy_grain="unit") == pytest.approx(
            res.score(
                {"spend": np.array([1.0, 5.0]), "spend_copy": np.array([0.0, 9.0])},
                deploy_grain="unit",
            )
        )

    def test_partially_pruned_categorical_is_still_required(self):
        """Losing one one-hot level MUST NOT excuse the covariate as a whole."""
        y, d, cols = _sim_partial()
        res = fit_cate(
            y,
            d,
            cols,
            interact=[
                Covariate(name="is_android"),
                Covariate(name="platform", kind="categorical"),
            ],
        )
        assert "d:platform=android" in res.pruned
        assert {i.name for i in res.interactions} == {"d:is_android", "d:platform=web"}
        with pytest.raises(InvalidRequestError) as exc_info:
            res.cate({"is_android": 0.0})
        assert exc_info.value.code == "estimation.cate.cate.covariate_missing_from"
        # Independent expectation: refit the surviving design via lstsq and
        # predict from ITS coefficients, so a corrupted beta cannot cancel out.
        flag = cols["is_android"]
        scale = flag.std(ddof=1)
        basis = np.column_stack(
            [(flag - flag.mean()) / scale, (cols["platform"] == "web").astype(float)]
        )
        means = basis.mean(axis=0)
        z = np.column_stack([np.ones(len(y)), d, basis - means, d[:, None] * (basis - means)])
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        point = np.array([-flag.mean() / scale, 1.0]) - means
        assert res.ate == pytest.approx(beta[1], abs=1e-10)
        assert res.cate({"is_android": 0.0, "platform": "web"}).value == pytest.approx(
            beta[1] + float(beta[4:] @ point), abs=1e-10
        )

    def test_unused_columns_are_ignored(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        point = res.cate({"spend": 1.0, "platform": "ios", "unused": 7.0})
        assert point.value == pytest.approx(res.cate({"spend": 1.0}).value, abs=1e-12)
        # The demo path scores whole frames, most of whose columns are not modelled.
        assert res.score(
            dict(cols) | {"unused": np.zeros(len(y))}, deploy_grain="unit"
        ) == pytest.approx(res.score({"spend": cols["spend"]}, deploy_grain="unit"))

    def test_nonfinite_outcome_uses_cate_refusal_code(self):
        y, d, cols = _sim()
        y = y.copy()
        y[0] = np.nan
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        assert exc_info.value.code == "estimation.cate.outcome_nulls_non"

    def test_treatment_shape_mismatch_is_refused(self):
        y, d, cols = _sim()
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(y, d[:-1], cols, interact=[Covariate(name="spend")])
        assert exc_info.value.code == "estimation.cate.treatment_shape_expected"

    def test_nonbinary_d_raises(self):
        y, d, cols = _sim()
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(y, d + 0.5, cols, interact=[Covariate(name="spend")])
        assert exc_info.value.code == "estimation.cate.treatment_binary"

    def test_wide_design_refused(self):
        r = np.random.default_rng(2)
        n = 60
        cols = {f"x{j}": r.standard_normal(n) for j in range(12)}
        y, d = r.standard_normal(n), (r.random(n) < 0.5).astype(float)
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(y, d, cols, interact=[Covariate(name=f"x{j}") for j in range(12)])
        assert exc_info.value.code == "estimation.cate.design_too_wide"

    def test_thin_one_hot_cell_is_refused(self):
        # A level with exactly 1 treated + 1 control unit drives HC2 leverage
        # to 1 (w_i = 0/0) and the sandwich ~10x anticonservative. Refuse.
        y, d, cols = _sim(n=200, seed=3)
        plat = cols["platform"].copy()
        plat[:2] = "tv"  # exactly one treated + one control 'tv' unit
        d[0], d[1] = 1.0, 0.0
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(
                y,
                d,
                dict(cols) | {"platform": plat},
                interact=[Covariate(name="platform", kind="categorical")],
            )
        assert exc_info.value.code == "estimation.cate.thin_cells.one_hot_level_interacted_min"

    def test_result_is_frozen_and_reports_unadjusted_se(self):
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        assert isinstance(res, CateResult)
        assert res.se_unadjusted > 0 and res.lb < res.ate < res.ub
        with pytest.raises((TypeError, ValueError)):
            res.ate = 0.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_cate_beyond_the_last_knot_matches_an_independent_fit(self):
        """Past the last knot every hinge is live, so the whole basis is exercised."""
        y, d, cols = _sim()
        knots = (-0.5, 0.5)
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend", knots=knots)])
        assert [i.name for i in res.interactions] == ["d:spend", "d:spend>k1", "d:spend>k2"]
        # Reference fit: same standardized base column plus one hinge per
        # knot, through lstsq, so a corrupted beta cannot cancel out.
        spend = cols["spend"]
        mean, scale = spend.mean(), spend.std(ddof=1)
        basis = np.column_stack(
            [(spend - mean) / scale, *(np.maximum(spend - k, 0.0) / scale for k in knots)]
        )
        means = basis.mean(axis=0)
        design = np.column_stack([np.ones(len(y)), d, basis - means, d[:, None] * (basis - means)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        point = 2.0
        row = np.array([(point - mean) / scale, *((point - k) / scale for k in knots)]) - means
        assert res.ate == pytest.approx(beta[1], abs=1e-10)
        assert res.cate({"spend": point}).value == pytest.approx(
            beta[1] + float(beta[5:] @ row), abs=1e-10
        )

    def test_wald_df_grows_with_the_basis(self):
        y, d, cols = _sim()
        linear = fit_cate(y, d, cols, interact=[Covariate(name="spend")])
        hinged = fit_cate(y, d, cols, interact=[Covariate(name="spend", knots=4)])
        assert (linear.heterogeneity.df, hinged.heterogeneity.df) == (1, 5)

    def test_hinges_in_the_adjustment_block_stay_out_of_the_interactions(self):
        y, d, cols = _sim()
        res = fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="platform", kind="categorical")],
            adjust=[Covariate(name="spend", knots=2)],
        )
        assert {i.name for i in res.interactions} == {"d:platform=ios"}
        assert res.main_spec.column_names() == ("spend", "spend>k1", "spend>k2", "platform=ios")

    def test_conflicting_knots_across_the_two_blocks_are_refused(self):
        y, d, cols = _sim()
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cate(
                y,
                d,
                cols,
                interact=[Covariate(name="spend", knots=2)],
                adjust=[Covariate(name="spend", knots=4)],
            )
        assert exc_info.value.code == "estimation.cate.covariate_declared_knots"

    def test_a_knot_past_the_data_is_pruned_by_name(self):
        """A dead hinge column is the existing rank pruning's problem, not a new one."""
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend", knots=(0.0, 99.0))])
        assert res.pruned == ("spend>k2", "d:spend>k2")
        assert res.cate({"spend": 1.0}).value == pytest.approx(
            res.score({"spend": np.array([1.0])}, deploy_grain="unit")[0], abs=1e-12
        )

    def test_tiny_alpha_produces_a_finite_interval_instead_of_nan(self):
        """``t.ppf(1 - alpha/2, dof)`` rounds ``1 - alpha/2`` to exactly 1.0
        in binary64 once alpha is small enough, silently returning an
        infinite critical value. That hit two sites: the fit-time
        interval/interaction bounds, and ``CateResult._estimate`` (used by
        ``cate``/``contrast``), which used to rebuild the tail from the
        already-rounded ``level`` field instead of carrying alpha itself.
        Both must stay finite even at alpha=1e-20.
        """
        y, d, cols = _sim()
        res = fit_cate(y, d, cols, interact=[Covariate(name="spend")], alpha=1e-20)
        assert math.isfinite(res.lb) and math.isfinite(res.ub)
        assert res.lb < res.ate < res.ub
        for effect in res.interactions:
            assert math.isfinite(effect.lb) and math.isfinite(effect.ub)
            assert effect.lb < effect.coef < effect.ub

        point = res.cate({"spend": float(cols["spend"].mean())})
        assert point.lb is not None and point.ub is not None
        assert math.isfinite(point.lb) and math.isfinite(point.ub)
        assert point.lb < point.value < point.ub


_ARD_INTERACT = [Covariate(name=f"x{j}") for j in range(8)]


def _sim_noise_plus_one_signal(n=1000, p=8, seed=9):
    """``d:x0`` is a strong real modifier; ``d:x1``..``d:x7`` are pure noise.

    Seed-pinned deliberately. Whether a NULL interaction shrinks to zero is a
    property of its realized |z|, not of the truth, so a chance-significant
    noise column legitimately survives partway - the shrinkage boundary sits
    near |z| ~ 1. At this draw every noise interaction lands at |z| < 1.0, so
    the ``|z| < 1.5`` selection below covers the whole noise block and the
    assertion has no cherry-picked exemptions.
    """
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, p))
    d = (r.random(n) < 0.5).astype(float)
    y = x @ np.linspace(1.0, 0.2, p) + d * (0.5 + 0.8 * x[:, 0]) + r.standard_normal(n)
    return y, d, {f"x{j}": x[:, j] for j in range(p)}


class TestArd:
    """Automatic relevance determination on the interaction block."""

    def test_noise_interactions_shrink_and_the_signal_survives(self):
        """Measured at this draw: noise to <= 6e-6 of OLS, signal at 0.99x."""
        y, d, cols = _sim_noise_plus_one_signal()
        res = fit_cate(y, d, cols, interact=_ARD_INTERACT, ard=True)
        signal, noise = None, []
        for effect in res.interactions:
            assert effect.ard_coef is not None
            ratio = abs(effect.ard_coef / effect.coef)
            if effect.name == "d:x0":
                signal = ratio
            elif abs(effect.coef / effect.se) < 1.5:
                noise.append((effect.name, ratio))
        assert len(noise) == 7, f"the pinned draw stopped covering the noise block: {noise}"
        assert max(r for _, r in noise) < 0.10, (
            f"a null interaction kept more than a tenth of its OLS magnitude: {noise}"
        )
        assert signal is not None and signal >= 0.70, (
            f"the real modifier was shrunk to {signal:.2f}x its OLS coefficient"
        )

    def test_ard_leaves_the_ate_the_se_and_the_wald_test_alone(self):
        """The reported test pays full price for its df, penalized or not."""
        y, d, cols = _sim_noise_plus_one_signal()
        plain = fit_cate(y, d, cols, interact=_ARD_INTERACT)
        shrunk = fit_cate(y, d, cols, interact=_ARD_INTERACT, ard=True)
        assert (shrunk.ate, shrunk.se, shrunk.lb, shrunk.ub) == (
            plain.ate,
            plain.se,
            plain.lb,
            plain.ub,
        )
        assert shrunk.heterogeneity == plain.heterogeneity
        assert [i.coef for i in shrunk.interactions] == [i.coef for i in plain.interactions]
        assert [i.se for i in shrunk.interactions] == [i.se for i in plain.interactions]
        assert all(i.ard_coef is None for i in plain.interactions)

    def test_convergence_is_deterministic(self):
        y, d, cols = _sim_noise_plus_one_signal()
        first = fit_cate(y, d, cols, interact=_ARD_INTERACT, ard=True)
        second = fit_cate(y, d, cols, interact=_ARD_INTERACT, ard=True)
        assert [i.ard_coef for i in first.interactions] == [i.ard_coef for i in second.interactions]

    def test_the_ard_point_ships_without_an_interval(self):
        """An ARD-shrunken point paired with the unpenalized HC2
        half-width asserted a nominal level it did not have. The fix mirrors
        ``InteractionEffect.ard_coef``: the shrunken point ships alone."""
        y, d, cols = _sim_noise_plus_one_signal()
        plain = fit_cate(y, d, cols, interact=_ARD_INTERACT)
        shrunk = fit_cate(y, d, cols, interact=_ARD_INTERACT, ard=True)
        point = {f"x{j}": 1.0 for j in range(8)}
        a, b = shrunk.cate(point), plain.cate(point)
        assert a.value != b.value
        assert a.lb is None and a.ub is None and a.level is None
        assert b.lb is not None and b.ub is not None and b.level is not None
        diff = shrunk.contrast(point, {f"x{j}": 0.0 for j in range(8)})
        assert diff.lb is None and diff.ub is None and diff.level is None
        # Independent expectation: score() is the ARD coefficients against the
        # same standardized, grand-mean-centered basis, assembled here directly.
        basis = np.column_stack(
            [(cols[f"x{j}"] - cols[f"x{j}"].mean()) / cols[f"x{j}"].std(ddof=1) for j in range(8)]
        )
        gamma = np.array([i.ard_coef for i in shrunk.interactions])
        assert shrunk.score(cols, deploy_grain="unit") == pytest.approx(
            shrunk.ate + basis @ gamma, abs=1e-12
        )
        assert shrunk.score({f"x{j}": np.array([1.0]) for j in range(8)}, deploy_grain="unit")[
            0
        ] == pytest.approx(a.value, abs=1e-12)


class TestCateResultConfidenceMetadataIsConsistent:
    """Bounding level and alpha independently still allowed a public result to
    serialize contradictory confidence metadata."""

    @staticmethod
    def _result():
        y, d, cols = _sim()
        return fit_cate(y, d, cols, interact=[Covariate(name="spend")])

    def test_a_level_contradicting_alpha_is_refused(self):
        """Re-validating a mutated copy is what a deserializing caller does."""
        fields = dict(self._result().__dict__)
        fields["level"] = 0.5
        with pytest.raises(InvalidRequestError) as exc_info:
            CateResult.model_validate(fields)
        assert exc_info.value.code == "estimation.cate.cate.level_contradicts_alpha"

    def test_the_fitted_result_is_self_consistent(self):
        result = self._result()
        assert result.level == pytest.approx(math.fsum((1.0, -result.alpha)))


def _cluster_cell_data():
    """Eight declared clusters, all with zero within-cluster outcome variance."""
    sizes = np.array([2, 4, 2, 4, 2, 4, 4, 2])
    groups = np.repeat(np.arange(8), sizes)
    y = np.repeat([0.0, 2.0, 0.0, 4.0, 1.0, 7.0, 4.0, 12.0], sizes)
    d = np.repeat([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], sizes)
    segment = np.repeat(["A", "A", "B", "B", "A", "A", "B", "B"], sizes)
    return y, d, {"segment": segment}, groups


def _cell_fit(*, weighting="member_count", ard=False):
    y, d, cols, groups = _cluster_cell_data()
    return fit_cate(
        y,
        d,
        cols,
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=groups,
        cluster_weight=weighting,
        ard=ard,
    )


def _independent_cluster_covariance(z, y, weights, groups):
    """Reference in raw row/cluster coordinates, with no production helpers."""
    h = sum(w * np.outer(row, row) for w, row in zip(weights, z, strict=True))
    beta = np.linalg.solve(h, z.T @ (weights * y))
    errors = y - z @ beta
    labels = np.unique(groups)
    meat = np.zeros_like(h)
    for label in labels:
        rows = groups == label
        score = sum(
            w * row * error
            for w, row, error in zip(weights[rows], z[rows], errors[rows], strict=True)
        )
        meat += np.outer(score, score)
    bread = np.linalg.inv(h)
    return beta, len(labels) / (len(labels) - 1) * bread @ meat @ bread


def _mackay_ard_reference(other, interactions, y, sigma_sq):
    """Reference ARD sweep: one evidence-maximizing precision per column.

    Written from the algorithm, independently of the production sweep: the
    other blocks are partialled out first, then each column's precision and
    the coefficients are re-solved together until the coefficients settle.
    """
    basis = np.linalg.qr(other)[0]
    z = interactions - basis @ (basis.T @ interactions)
    r = y - basis @ (basis.T @ y)
    ztz, ztr = z.T @ z, z.T @ r
    if not sigma_sq > 0.0:
        return np.linalg.solve(ztz, ztr)
    alpha, mu = np.ones(z.shape[1]), np.zeros(z.shape[1])
    for _ in range(200):
        posterior = sigma_sq * np.linalg.inv(ztz + sigma_sq * np.diag(alpha))
        previous, mu = mu, posterior @ ztr / sigma_sq
        if float(np.max(np.abs(mu - previous))) < 1e-8:
            break
        claimed = 1.0 - alpha * np.diag(posterior)
        alpha = claimed / np.maximum(mu**2, claimed / 1e8)
    return mu


@pytest.mark.parametrize(
    "weighting, means, variances, ate, ate_variance, contrast, contrast_variance",
    [
        ("equal", [1, 2, 4, 8], np.array([4, 16, 36, 64]) / 7, 4.5, 30 / 7, 3, 120 / 7),
        (
            "member_count",
            [4 / 3, 8 / 3, 5, 20 / 3],
            np.array([1, 4, 9, 16]) * 256 / 567,
            23 / 6,
            640 / 189,
            1 / 3,
            2560 / 189,
        ),
    ],
)
def test_cluster_cell_sandwich_independent_oracle(
    weighting, means, variances, ate, ate_variance, contrast, contrast_variance
):
    from fractions import Fraction

    from scipy.stats import f, t

    # Control-A: scores +/-8/3, bread 1/6, and ALL eight clusters counted.
    raw_variance = Fraction(8, 7) * 2 * Fraction(8, 3) ** 2 * Fraction(1, 6) ** 2
    assert raw_variance == Fraction(256, 567)
    assert 30 * raw_variance == Fraction(2560, 189)
    y, d, cols, groups = _cluster_cell_data()
    weights = np.ones(y.size) if weighting == "member_count" else 1 / np.bincount(groups)[groups]
    cells = 2 * d.astype(int) + (cols["segment"] == "B").astype(int)
    beta, covariance = _independent_cluster_covariance(np.eye(4)[cells], y, weights, groups)
    assert beta == pytest.approx(means)
    assert covariance == pytest.approx(np.diag(variances), abs=1e-12)

    result = _cell_fit(weighting=weighting)
    cell_loadings = np.array(
        [
            [1, 0, -0.5, 0],
            [1, 0, 0.5, 0],
            [1, 1, -0.5, -0.5],
            [1, 1, 0.5, 0.5],
        ]
    )
    assert cell_loadings @ result.beta == pytest.approx(means)
    assert cell_loadings @ result.vcov @ cell_loadings.T == pytest.approx(covariance, abs=1e-12)
    assert (result.dimension, result.n_clusters, result.reference_df) == (4, 8, 7)
    assert result.ate == pytest.approx(ate)
    assert result.se**2 == pytest.approx(ate_variance)
    assert result.ub - result.ate == pytest.approx(t.isf(0.025, 7) * math.sqrt(ate_variance))
    difference = result.contrast({"segment": "B"}, {"segment": "A"})
    assert difference.value == pytest.approx(contrast)
    assert difference.ub - difference.value == pytest.approx(
        t.isf(0.025, 7) * math.sqrt(contrast_variance)
    )
    for segment, cell in [("A", 0), ("B", 1)]:
        effect = result.cate({"segment": segment})
        assert effect.value == pytest.approx(means[cell + 2] - means[cell])
        assert effect.ub - effect.value == pytest.approx(
            t.isf(0.025, 7) * math.sqrt(variances[cell] + variances[cell + 2])
        )
    wald = result.heterogeneity
    assert wald.statistic == pytest.approx(contrast**2 / contrast_variance)
    assert wald.p_value == pytest.approx(f.sf(wald.statistic, 1, result.reference_df))
    assert wald.p_value == pytest.approx(2 * t.sf(math.sqrt(wald.statistic), 7))


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("ard", [False, True])
def test_cluster_effects_and_uncertainty_ignore_large_outcome_offset(weighting, ard):
    y, d, cols, groups = _cluster_cell_data()
    original = _cell_fit(weighting=weighting, ard=ard)
    shifted = fit_cate(
        y + 1e15,
        d,
        cols,
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=groups,
        cluster_weight=weighting,
        ard=ard,
    )
    assert shifted.ate == pytest.approx(original.ate)
    assert (shifted.lb, shifted.ub) == pytest.approx((original.lb, original.ub))
    assert np.asarray(shifted.vcov) == pytest.approx(np.asarray(original.vcov))
    assert shifted.se_unadjusted == pytest.approx(original.se_unadjusted)
    assert shifted.score(cols, deploy_grain="unit") == pytest.approx(
        original.score(cols, deploy_grain="unit"), abs=1e-9
    )
    assert shifted.contrast({"segment": "B"}, {"segment": "A"}).value == pytest.approx(
        original.contrast({"segment": "B"}, {"segment": "A"}).value, abs=1e-9
    )


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_unadjusted_se_uses_its_stored_two_column_sandwich(weighting):
    y, d, _, groups = _cluster_cell_data()
    weights = np.ones(y.size) if weighting == "member_count" else 1 / np.bincount(groups)[groups]
    _, expected = _independent_cluster_covariance(
        np.column_stack([np.ones(y.size), d]), y, weights, groups
    )
    result = _cell_fit(weighting=weighting)
    unadjusted = fit_cate(y, d, {}, interact=[], cluster_ids=groups, cluster_weight=weighting)
    assert np.asarray(result.unadjusted_vcov) == pytest.approx(expected)
    assert result.se_unadjusted**2 == pytest.approx(expected[1, 1])
    assert result.se_unadjusted == pytest.approx(unadjusted.se)
    assert np.asarray(unadjusted.vcov) == pytest.approx(expected)
    assert result.se_reduction == pytest.approx(1 - result.se / unadjusted.se)


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_fixed_design_member_cloning_preserves_covariance_and_ard(weighting):
    y, d, cols, groups = _cluster_cell_data()
    original = _cell_fit(weighting=weighting, ard=True)
    cloned = fit_cate(
        np.tile(y, 3),
        np.tile(d, 3),
        {"segment": np.tile(cols["segment"], 3)},
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=np.tile(groups, 3),
        cluster_weight=weighting,
        ard=True,
    )
    assert cloned.n_clusters == original.n_clusters == 8
    assert cloned.beta == pytest.approx(original.beta)
    assert np.asarray(cloned.vcov) == pytest.approx(np.asarray(original.vcov))
    assert cloned.beta_ard == pytest.approx(original.beta_ard, abs=1e-9)
    assert cloned.score(cols, deploy_grain="unit") == pytest.approx(
        original.score(cols, deploy_grain="unit"), abs=1e-9
    )
    assert cloned.se_unadjusted == pytest.approx(original.se_unadjusted)


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_new_ids_on_clones_follow_declared_independence_formula(weighting):
    """New IDs declare extra independent clusters; the specified CR1 scale changes."""
    y, d, cols, groups = _cluster_cell_data()
    original = _cell_fit(weighting=weighting)
    cloned = fit_cate(
        np.tile(y, 2),
        np.tile(d, 2),
        {"segment": np.tile(cols["segment"], 2)},
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=np.concatenate([groups, groups + 8]),
        cluster_weight=weighting,
    )
    assert (cloned.n_clusters, cloned.reference_df) == (16, 15)
    assert cloned.beta == pytest.approx(original.beta)
    assert np.asarray(cloned.vcov) == pytest.approx(np.asarray(original.vcov) * 7 / 15)


def test_cluster_ard_precision_uses_interaction_covariance_trace():
    from increment.estimation.cate import _cluster_ard_variance

    other = np.ones((4, 1))
    interactions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    # G = diag(2, 2), hence trace(G V) / q = (8 + 18) / 2.
    assert _cluster_ard_variance(other, interactions, covariance) == pytest.approx(13)
    assert _cluster_ard_variance(
        np.tile(other, (2, 1)), np.tile(interactions, (2, 1)), covariance
    ) == pytest.approx(26)


def test_cluster_singletons_use_cr1_hc0_and_all_count():
    y, d, cols, _ = _cluster_cell_data()
    groups = np.arange(y.size)
    b = (cols["segment"] == "B").astype(float) - 0.5
    z = np.column_stack([np.ones(y.size), d, b, d * b])
    _, covariance = _independent_cluster_covariance(z, y, np.ones(y.size), groups)
    result = fit_cate(
        y,
        d,
        cols,
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=groups,
    )
    independent = fit_cate(y, d, cols, interact=[Covariate(name="segment", kind="categorical")])
    assert (result.n_clusters, result.reference_df) == (24, 23)
    assert np.asarray(result.vcov) == pytest.approx(covariance)
    assert not np.allclose(result.vcov, independent.vcov)


def test_cluster_six_declared_clusters_have_five_reference_degrees():
    from scipy.stats import t

    y, d, _, groups = _cluster_cell_data()
    keep = groups < 6
    result = fit_cate(y[keep], d[keep], {}, interact=[], cluster_ids=groups[keep])
    assert (result.n_clusters, result.reference_df, result.dimension) == (6, 5, 2)
    projection = result.cate({})
    assert projection.ub is not None
    assert projection.ub - result.ate == pytest.approx(t.isf(0.025, 5) * result.se)


def test_cluster_fewer_than_two_declared_clusters_refuses():
    y, d, cols, groups = _cluster_cell_data()
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(y, d, cols, interact=[], cluster_ids=np.zeros_like(groups))
    assert caught.value.code == "estimation.cate.insufficient_clusters"
    assert caught.value.context["count"] == 1


def test_cluster_equal_weighting_requires_ids_and_invalid_weighting_refuses():
    from typing import Literal, cast

    y, d, _, _ = _cluster_cell_data()
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(y, d, {}, interact=[], cluster_weight="equal")
    assert caught.value.code == "estimation.cate.equal_weighting_without_cluster"

    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(
            y,
            d,
            {},
            interact=[],
            cluster_weight=cast('Literal["member_count", "equal"]', "implicit"),
        )
    assert caught.value.code == "estimation.cate.cluster_weight"


def test_cluster_rank_deficiency_is_not_silently_pruned():
    y, d, cols = _sim()
    cols["duplicate"] = cols["spend"].copy()
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="spend"), Covariate(name="duplicate")],
            cluster_ids=np.arange(y.size) // 4,
        )
    assert caught.value.code == "estimation.cate.rank_deficient"


def test_cluster_single_cluster_cell_direction_refuses_despite_many_rows():
    y, d, cols, groups = _cluster_cell_data()
    groups[groups == 1] = 0
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(
            y, d, cols, interact=[Covariate(name="segment", kind="categorical")], cluster_ids=groups
        )
    assert caught.value.code == "estimation.cate.single_cluster_direction"


def test_cluster_normalized_block_eigenvalue_one_refuses_only_affected_direction():
    rng = np.random.default_rng(102)
    groups = np.repeat(np.arange(8), 10)
    d = np.tile([0.0, 1.0], 40)
    isolated = (groups == 0).astype(float)
    y = d + isolated + rng.normal(size=80)
    result = fit_cate(
        y,
        d,
        {"isolated": isolated},
        interact=[],
        adjust=[Covariate(name="isolated")],
        cluster_ids=groups,
    )
    # One cluster pins one direction, and it loads only on the isolated
    # adjustment column: the treatment contrast is unaffected, so the fit
    # and its projection stay available while that direction is unsupported.
    assert len(result.unsupported_directions) == 1
    pinned = dict(zip(result.columns, result.unsupported_directions[0], strict=True))
    assert pinned["d"] == pytest.approx(0.0, abs=1e-9)
    assert abs(pinned["isolated"]) > 0.5
    assert result.se > 0
    assert np.linalg.matrix_rank(result.vcov) == result.dimension - 1
    projection = result.cate({})
    assert projection.ub is not None and projection.lb is not None
    assert projection.ub > projection.lb


def test_cluster_singular_interaction_wald_refuses_even_with_positive_scalar_variances():
    rng = np.random.default_rng(133)
    n = 120
    d = np.tile([0.0, 1.0], n // 2)
    cols = {"x": rng.normal(size=n), "v": rng.normal(size=n)}
    y = d + cols["x"] + rng.normal(size=n)
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(
            y,
            d,
            cols,
            interact=[Covariate(name="x"), Covariate(name="v")],
            cluster_ids=np.repeat([0, 1], n // 2),
        )
    assert caught.value.code == "estimation.cate.singular_wald"


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_cluster_invalid_covariance_refuses(covariance):
    from increment.estimation.cate import _check_covariance

    with pytest.raises(InvalidRequestError) as caught:
        _check_covariance(covariance)
    assert caught.value.code == "estimation.cate.invalid_covariance"


@pytest.mark.parametrize("sizes", [(2,) * 8, (1, 2, 3, 4, 5, 6)])
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("reverse", [False, True])
def test_cluster_unavailable_nonnullable_uncertainty_refuses(sizes, weighting, scale, reverse):
    groups = np.repeat(np.arange(len(sizes)), sizes)
    d = (groups % 2).astype(float)
    y = scale * d
    order = slice(None, None, -1 if reverse else 1)
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(
            y[order],
            d[order],
            {},
            interact=[],
            cluster_ids=groups[order],
            cluster_weight=weighting,
        )
    assert caught.value.code == "estimation.cate.uncertainty_unavailable"


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_adjusted_perfect_fit_refuses(weighting):
    groups = np.repeat(np.arange(6), [1, 2, 3, 4, 5, 6])
    d = (groups % 2).astype(float)
    x = np.linspace(-1, 1, len(d))
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(
            d + x,
            d,
            {"x": x},
            interact=[Covariate(name="x")],
            cluster_ids=groups,
            cluster_weight=weighting,
        )
    assert caught.value.code == "estimation.cate.uncertainty_unavailable"


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_cluster_small_resolvable_noise_preserves_uncertainty(weighting, scale):
    groups = np.repeat(np.arange(6), [1, 2, 3, 4, 5, 6])
    d = (groups % 2).astype(float)
    y = scale * (d + 1e-7 * groups)
    result = fit_cate(y, d, {}, interact=[], cluster_ids=groups, cluster_weight=weighting)
    weights = np.ones(len(d)) if weighting == "member_count" else 1 / (groups + 1)
    _, covariance = _independent_cluster_covariance(
        np.column_stack([np.ones(len(d)), d]), y, weights, groups
    )
    assert result.se / scale == pytest.approx(np.sqrt(covariance[1, 1]) / scale, rel=1e-7)
    assert result.ub > result.lb


def test_cluster_covariance_and_metadata_are_immutable_and_owned():
    from typing import cast

    from pydantic import ValidationError

    result = _cell_fit()
    fields = dict(result.__dict__)
    covariance = [list(row) for row in result.vcov]
    fields["vcov"] = covariance
    copied = CateResult.model_validate(fields)
    covariance[0][0] = -100
    assert copied.vcov == result.vcov
    array = np.asarray(copied.vcov)
    array[0, 0] = -200
    assert copied.vcov == result.vcov
    with pytest.raises(TypeError):
        cast("list[float]", copied.vcov[0])[0] = -300
    with pytest.raises(ValidationError):
        cast("Any", copied).reference_df = 99
    for key, value in [("dimension", 3), ("reference_df", 23), ("n_clusters", 7)]:
        fields = dict(result.__dict__)
        fields[key] = value
        with pytest.raises(InvalidRequestError) as caught:
            CateResult.model_validate(fields)
        assert caught.value.code == "estimation.cate.inference_metadata"


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_weighted_centering_and_multivariate_ard_share_the_regression_target(weighting):
    sizes = np.array([8, 8, 4, 6, 3, 5] * 2)
    groups = np.repeat(np.arange(12), sizes)
    segment = np.repeat(["A", "A", "B", "B", "C", "C"] * 2, sizes)
    y = np.repeat([0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 1.0, 7.0, 4.0, 12.0, 9.0, 19.0], sizes)
    d = (groups >= 6).astype(float)
    weights = np.ones(y.size) if weighting == "member_count" else 1 / sizes[groups]
    basis = np.column_stack([segment == "B", segment == "C"]).astype(float)
    center = np.average(basis, axis=0, weights=weights)
    basis -= center
    z = np.column_stack([np.ones(y.size), d, basis, d[:, None] * basis])
    beta, covariance = _independent_cluster_covariance(z, y, weights, groups)
    root = np.sqrt(weights)
    other, interactions = root[:, None] * z[:, :4], root[:, None] * z[:, 4:]
    residualized = interactions - other @ np.linalg.lstsq(other, interactions, rcond=None)[0]
    gram = residualized.T @ residualized
    sigma_sq = np.trace(gram @ covariance[4:, 4:]) / 2
    expected_ard = _mackay_ard_reference(other, interactions, root * y, sigma_sq)
    result = fit_cate(
        y,
        d,
        {"segment": segment},
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=groups,
        cluster_weight=weighting,
        ard=True,
    )
    assert result.interaction_means == pytest.approx(center)
    assert result.beta == pytest.approx(beta)
    assert np.asarray(result.vcov) == pytest.approx(covariance)
    assert np.asarray(result.beta_ard)[4:] == pytest.approx(expected_ard, abs=1e-9)
    assert result.score({"segment": segment}, deploy_grain="unit") == pytest.approx(
        result.ate + basis @ expected_ard
    )
    if weighting == "equal":
        assert center == pytest.approx([1 / 3, 1 / 3])
    else:
        assert center == pytest.approx([20 / 68, 16 / 68])


def test_cluster_permutation_preserves_weighted_covariance_and_contrasts():
    y, d, cols, groups = _cluster_cell_data()
    order = np.random.default_rng(209).permutation(y.size)
    original = _cell_fit(weighting="equal")
    permuted = fit_cate(
        y[order],
        d[order],
        {"segment": cols["segment"][order]},
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=groups[order],
        cluster_weight="equal",
    )
    assert permuted.beta == pytest.approx(original.beta)
    assert np.asarray(permuted.vcov) == pytest.approx(np.asarray(original.vcov))
    assert permuted.reference_df == original.reference_df
    contrast = permuted.contrast({"segment": "B"}, {"segment": "A"})
    expected = original.contrast({"segment": "B"}, {"segment": "A"})
    assert contrast.value == pytest.approx(expected.value)
    assert contrast.lb == pytest.approx(expected.lb)
    assert contrast.ub == pytest.approx(expected.ub)


def test_cluster_zero_score_cluster_is_still_counted():
    y, d, _, groups = _cluster_cell_data()
    # The additional control cluster sits exactly at the control mean (2).
    y = np.concatenate([y, [2.0, 2.0, 2.0]])
    d = np.concatenate([d, [0.0, 0.0, 0.0]])
    groups = np.concatenate([groups, [8, 8, 8]])
    z = np.column_stack([np.ones(y.size), d])
    beta, covariance = _independent_cluster_covariance(z, y, np.ones(y.size), groups)
    assert np.sum((y - z @ beta)[groups == 8]) == pytest.approx(0, abs=1e-12)
    result = fit_cate(y, d, {}, interact=[], cluster_ids=groups)
    assert (result.n_clusters, result.reference_df) == (9, 8)
    assert np.asarray(result.vcov) == pytest.approx(covariance)


def test_cluster_no_declared_clusters_refuses_without_clamping_degrees():
    with pytest.raises(InvalidRequestError) as caught:
        fit_cate(np.array([]), np.array([]), {}, interact=[], cluster_ids=np.array([]))
    assert caught.value.code == "estimation.cate.insufficient_clusters"
    assert caught.value.context["count"] == 0


def test_cluster_unadjusted_covariance_is_owned_and_is_the_se_source():
    result = _cell_fit()
    fields = dict(result.__dict__)
    covariance = [list(row) for row in result.unadjusted_vcov]
    fields["unadjusted_vcov"] = covariance
    copied = CateResult.model_validate(fields)
    expected_se = copied.se_unadjusted
    covariance[1][1] = 999
    assert copied.se_unadjusted == expected_se
    assert copied.model_dump()["se_unadjusted"] == expected_se
    fields["unadjusted_vcov"] = np.asarray(result.unadjusted_vcov) * 4
    projected = CateResult.model_validate(fields)
    assert projected.se_unadjusted == pytest.approx(2 * result.se_unadjusted)


def test_unclustered_hc2_and_ard_keep_the_original_row_precision():
    y, d, cols = _sim(n=80, seed=203)
    x = (cols["spend"] - cols["spend"].mean()) / cols["spend"].std(ddof=1)
    x -= x.mean()
    z = np.column_stack([np.ones(y.size), d, x, d * x])
    bread = np.linalg.inv(z.T @ z)
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    errors = y - z @ beta
    leverage = np.sum((z @ bread) * z, axis=1)
    meat = z.T @ ((errors**2 / (1 - leverage))[:, None] * z)
    covariance = bread @ meat @ bread
    sigma_sq = errors @ errors / (y.size - z.shape[1])
    expected_ard = _mackay_ard_reference(z[:, :3], z[:, 3:], y, sigma_sq)
    result = fit_cate(y, d, cols, interact=[Covariate(name="spend")], ard=True)
    explicit = fit_cate(
        y,
        d,
        cols,
        interact=[Covariate(name="spend")],
        ard=True,
        cluster_ids=None,
        cluster_weight="member_count",
    )
    assert result.model_dump() == explicit.model_dump()
    assert result.n_clusters is None
    assert result.reference_df == 76
    assert result.beta == pytest.approx(beta)
    assert np.asarray(result.vcov) == pytest.approx(covariance)
    assert result.beta_ard is not None
    assert result.beta_ard[3:] == pytest.approx(expected_ard, abs=1e-9)
    assert result.se_unadjusted**2 == pytest.approx(
        y[d == 1].var(ddof=1) / np.sum(d == 1) + y[d == 0].var(ddof=1) / np.sum(d == 0)
    )


def test_cluster_scores_are_keyed_canonical_immutable_and_use_stored_intervention():
    from pydantic import ValidationError

    from increment.results import ClusterScore

    y, d, cols, ids = _cluster_cell_data()
    fit = fit_cate(
        y,
        d,
        cols,
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=ids,
        intervention_grain="cluster",
    )
    unseen = {"segment": np.array(["A", "B", "B", "A", "A"])}
    unseen_ids = np.array(["z", "a", "z", "a", "z"])
    scores: tuple[ClusterScore, ...] = fit.score(
        unseen, cluster_ids=unseen_ids, deploy_grain="cluster"
    )
    unit = fit.score_state.score(unseen)
    assert [row.cluster_id for row in scores] == ["a", "z"]
    assert [row.member_count for row in scores] == [2, 3]
    assert [row.score for row in scores] == pytest.approx(
        [unit[[1, 3]].mean(), unit[[0, 2, 4]].mean()]
    )
    assert fit.score(unseen, cluster_ids=unseen_ids) == scores
    permutation = np.array([4, 2, 3, 1, 0])
    assert (
        fit.score({"segment": unseen["segment"][permutation]}, cluster_ids=unseen_ids[permutation])
        == scores
    )
    with pytest.raises(ValidationError):
        cast("Any", scores[0]).member_count = 99
    with pytest.raises(InvalidRequestError) as caught:
        fit.score(unseen, cluster_ids=unseen_ids, deploy_grain="unit")
    assert caught.value.code == "estimation.targeting.unsupported_unit_deployment"
    dependence = fit_cate(
        y,
        d,
        cols,
        interact=[Covariate(name="segment", kind="categorical")],
        cluster_ids=ids,
    )
    aligned: np.ndarray = dependence.score(unseen, cluster_ids=unseen_ids, deploy_grain="unit")
    default = dependence.score(unseen, cluster_ids=unseen_ids)
    assert isinstance(default, np.ndarray)
    assert default == pytest.approx(aligned)


@pytest.mark.parametrize("ard", [False, True])
def test_portable_score_state_keeps_fitted_knots_categories_and_shrinkage(ard):
    from increment.results import CateScoreState

    rng = np.random.default_rng(412)
    x = rng.normal(size=240)
    category = np.tile(["a", "b", "c"], 80)
    d = np.tile([0.0, 1.0], 120)
    y = x + d * (2 + x + 2 * np.maximum(x, 0) + (category == "b")) + rng.normal(size=240)
    fit = fit_cate(
        y,
        d,
        {"x": x, "category": category},
        interact=[
            Covariate(name="x", knots=1),
            Covariate(name="category", kind="categorical"),
        ],
        ard=ard,
    )
    state = CateScoreState.model_validate_json(fit.score_state.model_dump_json())
    unseen = {"x": np.array([-10.0, 4.0, 8.0]), "category": np.array(["b", "c", "new"])}
    expected = [
        fit.cate({"x": float(v), "category": str(c)}).value
        for v, c in zip(unseen["x"], unseen["category"], strict=True)
    ]
    np.testing.assert_allclose(state.score(unseen), expected)
    np.testing.assert_allclose(fit.score(unseen, deploy_grain="unit"), expected)


@pytest.mark.parametrize("intervention_grain", ["unit", "cluster"])
def test_constant_cate_scores_use_empty_column_deployment_roster(intervention_grain):
    y, d, _, ids = _cluster_cell_data()
    fit = fit_cate(
        y,
        d,
        {},
        interact=[],
        cluster_ids=ids,
        intervention_grain=intervention_grain,
    )
    roster = np.array(["z", "a", "z"])
    records = fit.score({}, cluster_ids=roster, deploy_grain="cluster")
    assert [(row.cluster_id, row.member_count) for row in records] == [("a", 1), ("z", 2)]
    assert [row.score for row in records] == pytest.approx([fit.ate, fit.ate])
    if intervention_grain == "unit":
        unit_scores = fit.score({}, cluster_ids=roster)
        assert unit_scores == pytest.approx(np.full(3, fit.ate))
    else:
        assert fit.score({}, cluster_ids=roster) == records
    with pytest.raises(InvalidRequestError) as caught:
        fit.score({}, cluster_ids=roster.reshape(1, 3), deploy_grain="cluster")
    assert caught.value.code == "estimation.targeting.deployment_cluster_ids_shape"
