"""Unit tests for offline seeded splits and out-of-fold selection."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from increment.errors import InvalidRequestError
from increment.estimation.crossfit import (
    check_cluster_atomic,
    fold_assignments,
    outer_split,
    select_out_of_fold,
)

IDS = np.array([f"u{i}" for i in range(40)])


class TestOuterSplit:
    def test_deterministic_given_seed(self):
        assert np.array_equal(outer_split(IDS, seed=7), outer_split(IDS, seed=7))

    def test_a_different_seed_moves_the_split(self):
        assert not np.array_equal(outer_split(IDS, seed=7), outer_split(IDS, seed=8))

    def test_default_test_size_is_half(self):
        assert outer_split(IDS, seed=7).sum() == 20

    def test_stratification_balances_arms_exactly(self):
        d = np.array([0.0, 1.0] * 20)
        mask = outer_split(IDS, seed=7, stratify=d)
        assert d[mask].sum() == 10.0 and (1.0 - d[mask]).sum() == 10.0

    def test_test_size_out_of_bounds_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            outer_split(IDS, test_size=1.0, seed=7)
        assert exc_info.value.code == "estimation.crossfit.test_size"
        assert exc_info.value.context == {"test_size": 1.0}

    def test_stratify_shape_mismatch_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            outer_split(IDS, seed=7, stratify=np.zeros(5))
        assert exc_info.value.code == "estimation.crossfit.stratify_shape_expected"


class TestFoldAssignments:
    def test_empty_atomic_groups_cannot_supply_requested_folds(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            fold_assignments(np.array([], dtype=str), n_folds=3, seed=7)
        assert exc_info.value.code == "estimation.crossfit.insufficient_clusters"
        assert exc_info.value.context["n_clusters"] == 0
        assert exc_info.value.context["n_folds"] == 3

    @pytest.mark.parametrize("explicit_clusters", [False, True])
    @pytest.mark.parametrize("strata", ["none", "pure", "mixed"])
    @pytest.mark.parametrize("support", [2, 3])
    def test_atomic_group_support_per_stratum(self, explicit_clusters, strata, support):
        groups = np.repeat([f"g{i}" for i in range(support)], 2)
        arm = None
        if strata != "none":
            arm = np.tile([0.0, 1.0] if strata == "mixed" else [1.0, 1.0], support)
            groups = np.concatenate([groups, np.repeat(["c0", "c1", "c2"], 2)])
            arm = np.concatenate([arm, np.zeros(6)])
        unit_ids = np.array([f"u{i}" for i in range(groups.size)]) if explicit_clusters else groups
        kwargs = {
            "n_folds": 3,
            "seed": 7,
            "stratify": arm,
            "cluster_ids": groups if explicit_clusters else None,
        }
        if support < 3:
            with pytest.raises(InvalidRequestError) as exc_info:
                fold_assignments(unit_ids, **kwargs)
            assert exc_info.value.code == "estimation.crossfit.insufficient_clusters"
            assert exc_info.value.context["n_clusters"] == support
            assert exc_info.value.context["n_folds"] == 3
        else:
            labels = fold_assignments(unit_ids, **kwargs)
            assert set(labels.tolist()) == {0, 1, 2}
            for group in np.unique(groups):
                assert np.unique(labels[groups == group]).size == 1
            if arm is not None:
                for fold in range(3):
                    assert set(arm[labels == fold].tolist()) == {0.0, 1.0}
                    assert set(arm[labels != fold].tolist()) == {0.0, 1.0}

    def test_deterministic_given_seed(self):
        a = fold_assignments(IDS, n_folds=4, seed=7)
        assert np.array_equal(a, fold_assignments(IDS, n_folds=4, seed=7))

    def test_labels_cover_the_range_with_balanced_sizes(self):
        labels = fold_assignments(IDS, n_folds=4, seed=7)
        counts = np.bincount(labels, minlength=4)
        assert set(labels.tolist()) == {0, 1, 2, 3}
        assert counts.max() - counts.min() <= 1

    def test_stratification_balances_arms_per_fold(self):
        d = np.array([0.0, 1.0] * 20)
        labels = fold_assignments(IDS, n_folds=4, seed=7, stratify=d)
        for k in range(4):
            assert d[labels == k].sum() == 5.0

    def test_fewer_than_two_folds_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            fold_assignments(IDS, n_folds=1, seed=7)
        assert exc_info.value.code == "estimation.crossfit.n_folds_least"
        assert exc_info.value.context == {"n_folds": 1}


class TestSelectOutOfFold:
    def test_models_never_see_their_own_validation_fold(self):
        folds = fold_assignments(IDS, n_folds=4, seed=7)
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def fit(train):
            return train.copy()

        def evaluate(model, val, index):
            seen.append((model, val))
            return np.zeros(int(val.sum()))

        select_out_of_fold(1, folds, fit, evaluate)
        for train, val in seen:
            assert not (train & val).any()
            assert (train | val).all()

    def test_pools_contributions_and_picks_the_argmax(self):
        folds = fold_assignments(IDS, n_folds=4, seed=7)
        payoff = (0.1, 0.5, 0.3)

        def evaluate(model, val, index):
            return np.full(int(val.sum()), payoff[index])

        sel = select_out_of_fold(3, folds, lambda train: None, evaluate)
        assert sel.best_index == 1
        assert sel.estimates == pytest.approx(payoff)
        assert sel.n_units == (40, 40, 40)
        assert sel.n_folds == 4

    def test_ses_is_the_pooled_standard_error_of_the_mean(self):
        # Deterministic 0..9 per-row contribution per fold makes the pooled
        # sample four exact repeats of 0..9, checkable against numpy std/sqrt(n).
        folds = fold_assignments(IDS, n_folds=4, seed=7)

        def evaluate(model, val, index):
            return np.arange(int(val.sum()), dtype=float)

        sel = select_out_of_fold(1, folds, lambda train: None, evaluate)
        pooled = np.tile(np.arange(10, dtype=float), 4)
        expected_se = pooled.std(ddof=1) / np.sqrt(pooled.size)
        assert sel.ses[0] == pytest.approx(expected_se)

    def test_ties_break_to_the_lowest_index(self):
        folds = fold_assignments(IDS, n_folds=2, seed=7)

        def evaluate(model, val, index):
            return np.full(int(val.sum()), 1.0)

        assert select_out_of_fold(3, folds, lambda train: None, evaluate).best_index == 0

    def test_a_wrong_contribution_shape_is_refused(self):
        folds = fold_assignments(IDS, n_folds=2, seed=7)
        with pytest.raises(InvalidRequestError) as exc_info:
            select_out_of_fold(1, folds, lambda train: None, lambda m, v, i: np.zeros(3))
        assert exc_info.value.code == "estimation.crossfit.candidate_returned_contribution"

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_candidate_contributions_cannot_win_selection(self, bad_value):
        folds = fold_assignments(IDS, n_folds=2, seed=7)

        def evaluate(model, val, index):
            contribution = np.full(int(val.sum()), 1.0)
            if index == 1:
                contribution[0] = bad_value
            return contribution

        with pytest.raises(InvalidRequestError) as exc_info:
            select_out_of_fold(2, folds, lambda train: None, evaluate)

        error = exc_info.value
        assert error.code == "estimation.crossfit.candidate_contribution_nonfinite"
        assert error.context["index"] == 1
        assert error.context["fold"] == 0
        assert error.context["n_bad"] == 1
        assert error.context["n_total"] == 20
        assert isinstance(error.context["route"], str)
        assert error.context["route"].strip()
        for cloned in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
            assert cloned.code == error.code
            assert cloned.context == error.context
            assert cloned.context["route"] == error.context["route"]

    def test_finite_candidate_selection_remains_argmax(self):
        folds = fold_assignments(IDS, n_folds=2, seed=7)

        def evaluate(model, val, index):
            return np.full(int(val.sum()), (0.25, 0.75)[index])

        selection = select_out_of_fold(2, folds, lambda train: None, evaluate)
        assert selection.best_index == 1
        assert selection.estimates == pytest.approx((0.25, 0.75))

    def test_fewer_than_two_fold_labels_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            select_out_of_fold(1, np.zeros(40, dtype=int), lambda t: None, lambda m, v, i: v)
        assert exc_info.value.code == "estimation.crossfit.selection_needs_least"

    def test_an_empty_candidate_grid_is_refused(self):
        folds = fold_assignments(IDS, n_folds=2, seed=7)
        with pytest.raises(InvalidRequestError) as exc_info:
            select_out_of_fold(0, folds, lambda t: None, lambda m, v, i: v)
        assert exc_info.value.code == "estimation.crossfit.n_candidates_least"


class TestCanonicalOrderInvariance:
    """The original finding: swapping which row a unit occupies must never
    change which side of the split it lands on."""

    def test_row_permutation_leaves_every_units_split_unchanged(self):
        ids = np.array([f"u{i}" for i in range(100)])
        arm = np.repeat([0.0, 1.0], 50)
        first = outer_split(ids, seed=7, stratify=arm)
        i = np.flatnonzero(first & (arm == 0))[0]
        j = np.flatnonzero(~first & (arm == 0))[0]
        order = np.arange(ids.size)
        order[i], order[j] = order[j], order[i]
        second = outer_split(ids[order], seed=7, stratify=arm[order])
        first_by_id = dict(zip(ids.tolist(), first.tolist(), strict=True))
        second_by_id = dict(zip(ids[order].tolist(), second.tolist(), strict=True))
        assert first_by_id == second_by_id

    def test_uuid_shaped_ids_are_also_row_order_invariant(self):
        rng = np.random.default_rng(0)
        ids = np.array([f"{rng.integers(0, 2**32):08x}-{i:04d}" for i in range(60)])
        arm = rng.integers(0, 2, size=ids.size).astype(float)
        first = outer_split(ids, seed=3, stratify=arm)
        order = rng.permutation(ids.size)
        second = outer_split(ids[order], seed=3, stratify=arm[order])
        first_by_id = dict(zip(ids.tolist(), first.tolist(), strict=True))
        second_by_id = dict(zip(ids[order].tolist(), second.tolist(), strict=True))
        assert first_by_id == second_by_id

    def test_fold_assignments_are_row_order_invariant_across_seeds(self):
        ids = np.array([f"u{i}" for i in range(60)])
        rng = np.random.default_rng(1)
        order = rng.permutation(ids.size)
        for seed in (1, 2, 3):
            first = fold_assignments(ids, n_folds=4, seed=seed)
            second = fold_assignments(ids[order], n_folds=4, seed=seed)
            first_by_id = dict(zip(ids.tolist(), first.tolist(), strict=True))
            second_by_id = dict(zip(ids[order].tolist(), second.tolist(), strict=True))
            assert first_by_id == second_by_id

    def test_uniform_member_replication_broadcasts_to_every_repeated_row(self):
        # A repeated unit id (e.g. panel replication) is its own singleton
        # cluster even with no explicit cluster_ids: every row sharing it
        # must land on the same side.
        rep_ids = np.tile(np.array([f"u{i}" for i in range(10)]), 3)
        mask = outer_split(rep_ids, seed=9)
        for u in np.unique(rep_ids):
            assert len(set(mask[rep_ids == u].tolist())) == 1

    def test_nested_outer_split_then_fold_assignments_stay_cluster_atomic(self):
        n_clusters = 24
        cluster_ids = np.repeat([f"c{i}" for i in range(n_clusters)], 5)
        unit_ids = np.array([f"u{i}" for i in range(cluster_ids.size)])
        arm = np.tile([0.0, 1.0, 0.0, 1.0, 0.0], n_clusters)
        outer = outer_split(unit_ids, seed=4, stratify=arm, cluster_ids=cluster_ids)
        for c in np.unique(cluster_ids):
            assert len(set(outer[cluster_ids == c].tolist())) == 1
        inner = ~outer
        folds = fold_assignments(
            unit_ids[inner],
            n_folds=3,
            seed=4,
            stratify=arm[inner],
            cluster_ids=cluster_ids[inner],
        )
        for c in np.unique(cluster_ids[inner]):
            assert len(set(folds[cluster_ids[inner] == c].tolist())) == 1


class TestClusterBroadcast:
    """`cluster_ids` broadcasts a canonical, seeded split/fold decision to
    every member of a cluster, without ever splitting one."""

    def test_pure_cluster_randomized_split_never_splits_a_cluster(self):
        cluster_ids = np.repeat([f"c{i}" for i in range(20)], 5)
        unit_ids = np.array([f"u{i}" for i in range(cluster_ids.size)])
        arm = np.repeat([0.0, 1.0], 50)  # cluster-randomized: whole clusters share one arm
        mask = outer_split(unit_ids, seed=3, stratify=arm, cluster_ids=cluster_ids)
        for c in np.unique(cluster_ids):
            assert len(set(mask[cluster_ids == c].tolist())) == 1
        arm_by_cluster = {c: arm[cluster_ids == c][0] for c in np.unique(cluster_ids)}
        test_clusters = {c for c in np.unique(cluster_ids) if mask[cluster_ids == c][0]}
        assert sum(arm_by_cluster[c] == 0.0 for c in test_clusters) == 5
        assert sum(arm_by_cluster[c] == 1.0 for c in test_clusters) == 5

    def test_mixed_treatment_clusters_are_never_split_and_stay_arm_balanced(self):
        n_clusters, per_cluster = 30, 10
        cluster_ids = np.repeat([f"m{i}" for i in range(n_clusters)], per_cluster)
        # every cluster holds an identical treated/control mix -- no cluster
        # is arm-pure, so this exercises the mixed-treatment balancing path.
        arm = np.tile([0.0, 1.0] * (per_cluster // 2), n_clusters)
        unit_ids = np.array([f"v{i}" for i in range(cluster_ids.size)])
        mask = outer_split(unit_ids, seed=5, stratify=arm, cluster_ids=cluster_ids)
        for c in np.unique(cluster_ids):
            assert len(set(mask[cluster_ids == c].tolist())) == 1
        assert mask.sum() == mask.size // 2
        assert arm[mask].mean() == pytest.approx(0.5)
        assert arm[~mask].mean() == pytest.approx(0.5)

    def test_fold_assignments_balance_mixed_clusters_across_folds(self):
        n_clusters, per_cluster = 20, 10
        cluster_ids = np.repeat([f"m{i}" for i in range(n_clusters)], per_cluster)
        arm = np.tile([0.0, 1.0] * (per_cluster // 2), n_clusters)
        unit_ids = np.array([f"v{i}" for i in range(cluster_ids.size)])
        labels = fold_assignments(
            unit_ids, n_folds=4, seed=11, stratify=arm, cluster_ids=cluster_ids
        )
        for c in np.unique(cluster_ids):
            assert len(set(labels[cluster_ids == c].tolist())) == 1
        for k in range(4):
            assert arm[labels == k].mean() == pytest.approx(0.5)

    def test_fold_assignments_balances_a_single_mixed_cluster_with_sufficient_pure_support(self):
        # Neither pure nor mixed groups suffice alone; together each arm
        # can populate both folds without splitting a cluster.
        unit_ids = np.array(["c", "t", "m0", "m1"])
        arm = np.array([0.0, 1.0, 0.0, 1.0])
        cluster_ids = np.array(["c", "t", "mixed", "mixed"])
        labels = fold_assignments(
            unit_ids, n_folds=2, seed=1, stratify=arm, cluster_ids=cluster_ids
        )
        for c in np.unique(cluster_ids):
            assert len(set(labels[cluster_ids == c].tolist())) == 1
        for f in range(2):
            assert set(arm[labels != f].tolist()) == {0.0, 1.0}

    def test_fold_assignments_refuses_when_a_value_has_no_atomic_split(self):
        # Genuine insufficiency: value 1.0 has zero pure clusters and its
        # only representation is a single mixed cluster - unsplittable, so
        # with n_folds=2 one fold necessarily holds ALL of value 1.0's rows.
        unit_ids = np.array([f"u{i}" for i in range(6)])
        arm = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        cluster_ids = np.array(["a1", "a2", "a3", "a4", "shared", "shared"])
        with pytest.raises(InvalidRequestError) as exc_info:
            fold_assignments(unit_ids, n_folds=2, seed=1, stratify=arm, cluster_ids=cluster_ids)
        assert exc_info.value.code == "estimation.crossfit.insufficient_clusters"
        assert exc_info.value.context["n_clusters"] == 1
        assert exc_info.value.context["n_folds"] == 2

    def test_insufficient_independent_clusters_is_refused(self):
        cluster_ids = np.array(["a", "a", "b", "b", "c", "c"])
        unit_ids = np.array([f"u{i}" for i in range(6)])
        with pytest.raises(InvalidRequestError) as exc_info:
            fold_assignments(unit_ids, n_folds=5, seed=1, cluster_ids=cluster_ids)
        assert exc_info.value.code == "estimation.crossfit.insufficient_clusters"
        assert exc_info.value.context["n_clusters"] == 3
        assert exc_info.value.context["n_folds"] == 5

    def test_cluster_ids_shape_mismatch_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            outer_split(IDS, seed=7, cluster_ids=np.array(["a", "b"]))
        assert exc_info.value.code == "estimation.crossfit.cluster_ids_shape_expected"

    def test_mixed_native_identity_collision_is_refused(self):
        ids = np.array([1, "1", 2], dtype=object)
        with pytest.raises(InvalidRequestError) as exc_info:
            outer_split(ids, seed=0)
        assert exc_info.value.code == "estimation.crossfit.identity_collision"


class TestCheckClusterAtomic:
    def test_refuses_a_split_cluster_naming_it_and_its_two_values(self):
        cluster_ids = np.array(["a", "a", "b", "b"])
        mask = np.array([True, False, False, False])
        with pytest.raises(InvalidRequestError) as exc_info:
            check_cluster_atomic(cluster_ids, mask)
        assert exc_info.value.code == "estimation.crossfit.cluster_split"
        assert exc_info.value.context["cluster"] == "a"
        values = exc_info.value.context["values"]
        assert isinstance(values, tuple)
        assert set(values) == {True, False}

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            check_cluster_atomic(np.array(["a", "b"]), np.array([True, False, True]))
        assert exc_info.value.code == "estimation.crossfit.cluster_atomic_shape"


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.crossfit.stratify_shape_expected",
            lambda: fold_assignments(np.zeros(5), n_folds=2, seed=0, stratify=np.array([1, 2, 3])),
        ),  # estimation/crossfit.py::fold_assignments
        (
            "estimation.crossfit.fold_assignments_unit_ids_shape",
            lambda: outer_split(np.array([[1, 2], [3, 4]]), test_size=0.5, seed=0),
        ),  # estimation/crossfit.py::outer_split
        (
            "estimation.crossfit.fold_assignments_unit_ids_shape",
            lambda: fold_assignments(np.array([[1, 2]]), n_folds=2, seed=0),
        ),  # estimation/crossfit.py::fold_assignments
        (
            "estimation.crossfit.n_candidates_least",
            lambda: select_out_of_fold(0, np.zeros(4), lambda t: None, lambda m, v, i: v),
        ),  # estimation/crossfit.py::select_out_of_fold
    ],
)
def test_crossfit_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
