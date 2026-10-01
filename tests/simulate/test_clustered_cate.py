"""Independent clustered causal-calibration witnesses.

The bounded replication test checks execution/accounting, not rate calibration.
``assert_scientific_tables`` consumes a complete, prospectively allocated family
and refuses inconclusive precision; it never chooses repetitions from outputs.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from itertools import combinations, product
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from increment.cate import estimate_cate, select_targeting_rule, targeting_rule, validate_cate
from increment.errors import CodedError, IncrementWarning, InvalidRequestError
from increment.estimation.engine import Method
from increment.estimation.targeting import ClusterBootstrap, TargetingRule
from increment.frame import MetricSpec, from_unit_summary
from increment.semantics.design import Randomized
from increment.simulate.cluster_dgp import (
    ClusteredCATEResult,
    ClusteredCATEScenario,
    ClusteredCell,
    ClusteredEvidenceArtifact,
    ClusteredRegistration,
    PointAccuracy,
    _action_truth,
    _causal_weighted_cut,
    _clustered_sizing_plan,
    _point_accuracy_observations,
    _rank_truth,
    _record,
    _statistics,
    _validation_targets,
    cell_inventory,
    child_rng,
    clustered_sizing_inventory,
    clustered_source,
    clustered_table_payload,
    evaluate_clustered_replication,
    evaluation_policy_truth,
    honest_clustered_population,
    policy_truth,
    reduce_clustered_observations,
    simulate_clustered_cate,
    validation_support_ceiling,
)
from tests.warning_codes import warning_codes


def _scenario(**overrides):
    return ClusteredCATEScenario.model_validate(
        {"n_clusters": 40, "members_per_cluster": 5, **overrides}
    )


def _source(table, *, grain="cluster"):
    return from_unit_summary(
        table,
        unit="unit_id",
        group="group_id",
        control="control",
        cluster="cluster_id",
        intervention_grain=grain,
        metrics={"y": "mean"},
        design=Randomized(control_group="control"),
    )


def _balanced_population(*, stream="training", grain="cluster", clones=1):
    """Paired clusters: arm-specific nuisance means are exactly zero.

    Y0=10+.3x+c+.2hx; Y1=Y0+2+.5x. c cycles -2..2 and h cycles
    -1.5..1.5 independently in each arm, so the public slope/ATE have exact
    targets and strictly positive cluster uncertainty in both directions.
    """
    rows, y0, y1, tau, ids = [], [], [], [], []
    for g in range(40):
        q = g // 2
        c, h = q % 5 - 2, q // 5 - 1.5
        for j, x in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
            a = 10 + 0.3 * x + c + 0.2 * h * x
            effect = 2 + 0.5 * x
            for clone in range(clones):
                rows.append(
                    (
                        f"g{g}-u{j}-{clone}",
                        f"g{g}",
                        "treatment" if g % 2 else "control",
                        a + effect if g % 2 else a,
                        0.0,
                        x,
                    )
                )
                y0.append(a)
                y1.append(a + effect)
                tau.append(effect)
                ids.append(g)
    return ClusteredCATEResult(
        _scenario(dimension=1, intervention_grain=grain, clone_factor=clones),
        stream,
        0,
        tuple(rows),
        ("unit_id", "cluster_id", "group_id", "y", "z", "x0"),
        tuple(y0),
        tuple(y1),
        tuple(tau),
        tuple(ids),
        (5 * clones,) * 40,
    )


def _four_cluster_oracle():
    rows, a, b, effects, ids = [], [], [], [], []
    for g, (size, x) in enumerate(zip((1, 2, 3, 4), (-1.0, 0.0, 1.0, 2.0), strict=True)):
        for j in range(size):
            effect = 2 + x / 2
            rows.append((f"o{g}-{j}", f"g{g}", "control", 10.0, 0.0, x))
            a.append(10.0)
            b.append(10 + effect)
            effects.append(effect)
            ids.append(g)
    return ClusteredCATEResult(
        _scenario(n_clusters=4, dimension=1, size_mode="variable", member_counts=(1, 2, 3, 4)),
        "oracle",
        0,
        tuple(rows),
        ("unit_id", "cluster_id", "group_id", "y", "z", "x0"),
        tuple(a),
        tuple(b),
        tuple(effects),
        tuple(ids),
        (1, 2, 3, 4),
    )


def test_frozen_scenario_result_arrays_and_caller_ownership():
    counts = [2, 3, 4, 5]
    scenario = _scenario(n_clusters=4, size_mode="variable", member_counts=counts)
    counts[0] = 900
    assert scenario.member_counts == (2, 3, 4, 5)
    with pytest.raises(ValidationError) as frozen:
        cast(Any, scenario).effect = 9
    assert frozen.value.errors()[0]["type"] == "frozen_instance"
    result = simulate_clustered_cate(scenario)
    before = result.truth
    with pytest.raises(FrozenInstanceError):
        cast(Any, result).y1 = ()
    arrays = result.potential_outcomes
    with pytest.raises(TypeError):
        cast(Any, arrays)["y1"] = np.zeros(len(result.rows))
    with pytest.raises(ValueError):
        arrays["y1"][0] = -999
    with pytest.raises(ValueError):
        arrays["y1"].setflags(write=True)
    rows = [list(row) for row in result.rows]
    copied = replace(result, rows=cast(Any, rows))
    rows[0][3] = -900
    assert copied.table.equals(result.table)
    external = result.table["y"].to_numpy().copy()
    external[:] = 0
    assert result.truth == before
    with pytest.raises(InvalidRequestError) as caught:
        replace(result, y1=(0.0,))
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


@pytest.mark.parametrize(
    ("bad", "code"),
    [
        ({"members": 100}, "model.field.unknown"),
        ({"clusters": 40}, "model.field.unknown"),
        ({"dimension": 1.5}, "model.field.type"),
        ({"effect": math.inf}, "model.field.nonfinite"),
    ],
)
def test_unknown_or_invalid_scenario_controls_are_not_silently_ignored(bad, code):
    with pytest.raises(InvalidRequestError) as raised:
        ClusteredCATEScenario(**bad)
    assert raised.value.code == code


@pytest.mark.parametrize(
    "bad",
    [
        {"dimension": 0, "leverage": 8},
        {"n_clusters": 4, "size_mode": "variable", "member_counts": (2, 3)},
        {"declare_clusters": False, "intervention_grain": "cluster"},
        {"assignment": "cluster", "omit_confounder": True},
    ],
)
def test_invalid_combinations_refuse_with_immutable_context(bad):
    with pytest.raises(InvalidRequestError) as caught:
        ClusteredCATEScenario(**bad)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"
    assert caught.value.context["reason"]
    with pytest.raises(TypeError):
        cast(Any, caught.value.context)["reason"] = "changed"


def test_rng_addresses_and_common_member_streams_are_stable():
    s = _scenario(dimension=1)
    baseline = simulate_clustered_cate(s)
    wider = simulate_clustered_cate(
        ClusteredCATEScenario.model_validate({**s.model_dump(), "dimension": 10})
    )
    for name in ("x0", "y", "group_id"):
        assert baseline.table[name].equals(wider.table[name])
    larger = simulate_clustered_cate(
        ClusteredCATEScenario.model_validate({**s.model_dump(), "members_per_cluster": 20})
    )
    for g in range(s.n_clusters):
        old = baseline.table.slice(g * 5, 5)
        new = larger.table.slice(g * 20, 5)
        for name in ("x0", "y", "group_id"):
            assert old[name].equals(new[name])
    addresses = [
        child_rng(17, role, rep, component).random(8).tobytes()
        for role, rep, component in product(
            ("training", "holdout", "oracle"),
            (0, 1),
            ("population", "covariate", "outcome", "assignment"),
        )
    ]
    assert len(set(addresses)) == len(addresses)
    for role in ("training", "holdout", "oracle"):
        first = simulate_clustered_cate(s, stream=role, replication=3)
        assert first == simulate_clustered_cate(s, stream=role, replication=3)
    assert not baseline.table.equals(simulate_clustered_cate(s, stream="oracle").table)


def test_independent_potential_outcome_equations_and_informative_targets():
    s = _scenario(size_mode="informative", effect=1, heterogeneity=0, size_effect=2)
    result = simulate_clustered_cate(s)
    z = np.asarray(result.table["z"])
    np.testing.assert_allclose(result.potential_outcomes["tau"], 1 + 2 * z)
    expected_member = np.mean(1 + 2 * z)
    expected_equal = np.mean(
        [1 + 2 * z[np.asarray(result.cluster_index) == g][0] for g in range(s.n_clusters)]
    )
    assert result.truth.member_ate == pytest.approx(expected_member)
    assert result.truth.equal_cluster_ate == pytest.approx(expected_equal)
    assert expected_member > expected_equal
    d = np.asarray(result.table["group_id"]) == "treatment"
    np.testing.assert_array_equal(np.asarray(result.table["y"]), np.where(d, result.y1, result.y0))


@pytest.mark.parametrize("weight", ["member_count", "equal"])
def test_public_fit_matches_hand_derived_nonzero_effect_and_cluster_sandwich(weight):
    data = _balanced_population()
    fit = estimate_cate(
        _source(data.table), "y", control="control", interact=["x0"], cluster_weight=weight
    )
    # Each arm has 20 independent c's, mean(c²)=2; CR1=40/39.
    assert fit.ate == pytest.approx(2)
    assert fit.se**2 == pytest.approx(40 / 39 * (2 / 20 + 2 / 20))
    effect = fit.contrast({"x0": 2}, {"x0": -2})
    assert effect.value == pytest.approx(2)
    assert effect.lb is not None and effect.ub is not None and effect.lb < effect.value < effect.ub
    assert fit.lb < 2 < fit.ub
    assert fit.n_clusters == 40 and fit.reference_df == 39


def test_exact_small_balanced_randomization_enumeration_has_unbiased_public_ate():
    values = []
    for treated in combinations(range(4), 2):
        rows = [
            {
                "unit_id": f"g{g}-{j}",
                "cluster_id": f"g{g}",
                "group_id": "treatment" if g in treated else "control",
                "y": 1 + g + 0.2 * j + 2 * (g in treated),
            }
            for g in range(4)
            for j in range(2)
        ]
        fit = estimate_cate(
            _source(pa.Table.from_pylist(rows)), "y", control="control", interact=[]
        )
        expected = 2 + np.mean(list(treated)) - np.mean([g for g in range(4) if g not in treated])
        assert fit.ate == pytest.approx(expected)
        assert fit.se > 0 and fit.lb < fit.ub
        values.append(fit.ate)
    assert np.mean(values) == pytest.approx(2)
    assert sorted(values) == pytest.approx([0, 1, 2, 2, 3, 4])


def test_assignment_support_is_not_inference_availability_and_never_redraws():
    s = _scenario(n_clusters=2, members_per_cluster=1)
    draws = child_rng(s.seed, "training", 0, "assignment/cluster").random(s.n_clusters)
    drawn = simulate_clustered_cate(s)
    groups = np.asarray(drawn.table["group_id"])
    # The schedule is drawn once and never redrawn to rescue arm support.
    np.testing.assert_array_equal(groups == "treatment", draws < s.treatment_ratio)
    with pytest.raises(InvalidRequestError) as caught:
        estimate_cate(clustered_source(drawn), "y", control="control", interact=[])
    assert caught.value.code == "estimation.cate.each_arm_needs"
    assert caught.value.context == {
        "n_treated": int((groups == "treatment").sum()),
        "n_control": int((groups == "control").sum()),
    }


@pytest.mark.slow
@pytest.mark.parametrize("weight,share,effect", [("member_count", 0.4, 3.0), ("equal", 0.5, 2.75)])
def test_public_frozen_rule_predicts_independent_oracle_and_honors_budgets(weight, share, effect):
    data = _balanced_population()
    rule = targeting_rule(
        _source(data.table),
        "y",
        control="control",
        interact=["x0"],
        fraction=0.5,
        cluster_weight=weight,
        bootstrap=ClusterBootstrap(seed=57, repetitions=99),
    )
    assert rule.deploy_grain == "cluster"
    oracle = _four_cluster_oracle()
    truth = policy_truth(rule, oracle, cost=1)
    actual = truth.member_effect if weight == "member_count" else truth.equal_cluster_effect
    fraction = truth.member_share if weight == "member_count" else truth.equal_cluster_share
    assert actual == pytest.approx(effect)
    assert fraction == pytest.approx(share)
    expected_actions = np.asarray(oracle.cluster_index) >= (3 if weight == "member_count" else 2)
    np.testing.assert_array_equal(truth.actions, expected_actions)
    expected_net = share * (effect - 1)
    assert (
        truth.member_net_benefit if weight == "member_count" else truth.equal_cluster_net_benefit
    ) == pytest.approx(expected_net)
    # Historical assignment is all control, so its outcome mean is ten, never the policy effect.
    assert np.mean(np.asarray(oracle.table["y"])) == 10
    assert truth.member_effect is not None
    assert truth.member_outcome == pytest.approx(10 + truth.member_share * truth.member_effect)
    restored = TargetingRule.model_validate_json(truth.rule_json)
    np.testing.assert_array_equal(
        restored.predict(oracle.covariates, cluster_ids=np.asarray(oracle.table["cluster_id"])),
        expected_actions,
    )
    for labels in (
        np.array(oracle.cluster_index) * 1009 - 71,
        np.array([f"店舗/{g}" for g in oracle.cluster_index]),
    ):
        np.testing.assert_array_equal(
            restored.predict(oracle.covariates, cluster_ids=labels), expected_actions
        )
        reversed_cols = {name: values[::-1] for name, values in oracle.covariates.items()}
        np.testing.assert_array_equal(
            restored.predict(reversed_cols, cluster_ids=labels[::-1]), expected_actions[::-1]
        )
        cloned_cols = {name: np.repeat(values, 5) for name, values in oracle.covariates.items()}
        np.testing.assert_array_equal(
            restored.predict(cloned_cols, cluster_ids=np.repeat(labels, 5)),
            np.repeat(expected_actions, 5),
        )
    tied_cols = {**oracle.covariates, "x0": np.zeros(10)}
    tied_ids = np.array([("z", "ä", "2", "10")[g] for g in oracle.cluster_index])
    np.testing.assert_array_equal(
        restored.predict(tied_cols, cluster_ids=tied_ids), expected_actions
    )
    collision = np.array([1, "1", *[f"u{i}" for i in range(8)]], dtype=object)
    with pytest.raises(InvalidRequestError) as identity:
        restored.predict(oracle.covariates, cluster_ids=collision)
    assert identity.value.code == "estimation.crossfit.identity_collision"
    with pytest.raises(InvalidRequestError) as caught:
        policy_truth(rule, data)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_partial_cluster_unit_policy_equal_weights_use_original_roster_sizes():
    oracle = _four_cluster_oracle()
    selected = np.zeros(10, dtype=bool)
    selected[[0, 6]] = True  # One member from size-one and size-four clusters.
    truth = _action_truth(selected, oracle, cost=1, rule_json="hand-derived-actions")
    assert truth.member_effect == pytest.approx((1.5 + 3) / 2)
    assert truth.equal_cluster_effect == pytest.approx((1.5 + 3 / 4) / (1 + 1 / 4))
    assert truth.member_share == pytest.approx(2 / 10)
    assert truth.equal_cluster_share == pytest.approx((1 + 1 / 4) / 4)
    empty = _action_truth(np.zeros(10, dtype=bool), oracle, cost=1, rule_json="empty")
    assert empty.member_effect is empty.equal_cluster_effect is None
    assert empty.member_net_benefit == empty.equal_cluster_net_benefit == 0
    assert empty.unavailable_reason == "simulate.cluster_dgp.empty_oracle_policy"


def _policy_value_witness(grain, weight):
    """Balance nuisance sums on BOTH evaluation partitions before fitting.

    Each cluster contains x=-2,-1,0,1,2, with tau=2+x/2. Center c,h within
    arm/partition/prefix intersections: Y0=10+c+h*x, Y1=Y0+tau. Thus selected
    arm nuisance means vanish exactly, while resampled nuisance means vary.
    """
    from increment.estimation.crossfit import outer_split
    from increment.estimation.targeting import _holdout_mask

    k = 400
    labels = np.array([f"g{g:04d}" for g in range(k)])
    d = np.arange(k) % 2
    held = _holdout_mask(labels, cluster_ids=labels)
    outer = outer_split(labels, test_size=0.5, seed=12, stratify=d, cluster_ids=labels)
    prefixes = []
    for mask in (held, outer):
        chosen = sorted(labels[mask])[: int(mask.sum()) // 2]
        prefixes.append(np.isin(labels, chosen))
    buckets = {}
    for g in range(k):
        key = (d[g], held[g], outer[g], prefixes[0][g], prefixes[1][g])
        buckets.setdefault(key, []).append(g)
    c, h = np.zeros(k), np.zeros(k)
    for indices in buckets.values():
        centered = np.arange(len(indices)) - (len(indices) - 1) / 2
        c[indices] = centered / 1000
        h[indices] = centered[::-1] / 1000
    rows, y0, y1, tau, groups = [], [], [], [], []
    for g in range(k):
        for j, x in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
            a, effect = 10 + c[g] + h[g] * x, 2 + x / 2
            rows.append(
                (
                    f"{labels[g]}/{j}",
                    labels[g],
                    "treatment" if d[g] else "control",
                    a + effect if d[g] else a,
                    0.0,
                    x,
                )
            )
            y0.append(a)
            y1.append(a + effect)
            tau.append(effect)
            groups.append(g)
    return ClusteredCATEResult(
        _scenario(n_clusters=k, dimension=1, intervention_grain=grain),
        "training",
        0,
        tuple(rows),
        ("unit_id", "cluster_id", "group_id", "y", "z", "x0"),
        tuple(y0),
        tuple(y1),
        tuple(tau),
        tuple(groups),
        (5,) * k,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "grain,weight", tuple(product(("unit", "cluster"), ("member_count", "equal")))
)
def test_public_policy_and_selection_values_match_nonzero_causal_targets(grain, weight):
    data = _policy_value_witness(grain, weight)
    options: dict[str, Any] = {
        "control": "control",
        "interact": ["x0"],
        "cluster_weight": weight,
        "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
        "include_evaluation_population": True,
    }
    source = _source(data.table, grain=grain)
    fixed = targeting_rule(source, "y", fraction=0.5, **options)
    selected = select_targeting_rule(
        source,
        "y",
        fractions=(0.0, 0.5, 1.0),
        n_folds=2,
        seed=12,
        cost_per_treated=2 if grain == "unit" else 0,
        **options,
    ).rule
    for candidate in (fixed, selected):
        # Every cluster's mean tau is 2; x>=0 has mean tau=5/2.
        expected_effect, expected_uplift = (2.5, 0.5) if grain == "unit" else (2.0, 0.0)
        assert candidate.validation.passed
        assert candidate.policy_value is not None and candidate.uplift_vs_average is not None
        assert candidate.policy_value.value == pytest.approx(expected_effect, abs=1e-10)
        assert candidate.uplift_vs_average.value == pytest.approx(expected_uplift, abs=1e-10)
        for point in (candidate.policy_value, candidate.uplift_vs_average):
            assert point.lb is point.ub is None
        causal, average = evaluation_policy_truth(candidate, data)
        target = causal.member_effect if weight == "member_count" else causal.equal_cluster_effect
        assert target is not None
        assert target == pytest.approx(expected_effect)
        assert target - average == pytest.approx(expected_uplift)


@pytest.mark.slow
@pytest.mark.parametrize("clones", [1, 5])
@pytest.mark.parametrize("order", ["forward", "reverse", "interleaved"])
def test_weighted_gates_tied_boundary_permutations_and_clones(clones, order):
    from increment.estimation.targeting import _holdout_mask, _weighted_bins

    # Four clusters, three members each: exact block masses are 2 and 2.
    score = np.repeat([0.0, 0.0, 1.0, 1.0], 3 * clones)
    tau = score.copy()
    weights = np.full(score.size, 1 / (3 * clones))
    positions = np.arange(score.size)
    if order == "reverse":
        positions = positions[::-1]
    elif order == "interleaved":
        positions = positions.reshape(4, -1).T.ravel()
    score, tau, weights = score[positions], tau[positions], weights[positions]
    cut = _causal_weighted_cut(score, weights, 0.5)
    assert cut == 0
    public_groups = _weighted_bins(score, weights, 2)
    np.testing.assert_array_equal(public_groups, score.astype(int))
    for group, chosen in ((0, score <= cut), (1, score > cut)):
        np.testing.assert_array_equal(public_groups == group, chosen)
        assert np.average(tau[chosen], weights=weights[chosen]) == group
    labels = np.array([f"boundary/{g}" for g in range(100)])
    held = _holdout_mask(labels, cluster_ids=labels)
    labels = np.r_[labels[held][:4], labels[~held][:12]]
    assert len(labels) == 16
    rows, y0, y1, effects, indices = [], [], [], [], []
    for g, label in enumerate(labels):
        effect = float((g // 2) % 2)
        nuisance = 0 if g < 4 else (g // 4) / 10
        for j in range(3 * clones):
            a = 10 + nuisance + ((j // clones) - 1) / 10
            rows.append(
                (
                    f"{label}/{j}",
                    label,
                    "treatment" if g % 2 else "control",
                    a + effect if g % 2 else a,
                    0.0,
                    effect,
                )
            )
            y0.append(a)
            y1.append(a + effect)
            effects.append(effect)
            indices.append(g)
    permutation = np.arange(len(rows))
    if order == "reverse":
        permutation = permutation[::-1]
    elif order == "interleaved":
        permutation = permutation.reshape(16, -1).T.ravel()
    data = ClusteredCATEResult(
        _scenario(n_clusters=16, members_per_cluster=3, dimension=1, clone_factor=clones),
        "training",
        0,
        tuple(rows[i] for i in permutation),
        ("unit_id", "cluster_id", "group_id", "y", "z", "x0"),
        tuple(y0[i] for i in permutation),
        tuple(y1[i] for i in permutation),
        tuple(effects[i] for i in permutation),
        tuple(indices[i] for i in permutation),
        (3 * clones,) * 16,
    )
    options: dict[str, Any] = {
        "control": "control",
        "interact": ["x0"],
        "cluster_weight": "equal",
        "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
    }
    source = _source(data.table, grain="unit")
    validation = validate_cate(source, "y", n_groups=2, **options)
    rule = targeting_rule(source, "y", fraction=0.5, **options)
    independent = _validation_targets(data, "equal", rule)
    assert validation.n_holdout == 12 * clones
    for group, target in zip(validation.groups, (0.0, 1.0), strict=True):
        assert group.n == 6 * clones
        assert group.effect == pytest.approx(target)
        assert independent[f"validation.group{group.group}"] == target


@pytest.mark.slow
@pytest.mark.parametrize("k", [2, 4, 100])
def test_uniform_order_statistic_finite_batch_target(k):
    from fractions import Fraction

    # E[Z_(j)]=2j/(K+1)-1 on Uniform(-1,1); average the top K/2 ranks.
    top = sum((2 * Fraction(j, k + 1) - 1 for j in range(k // 2 + 1, k + 1)), Fraction(0)) / (
        k // 2
    )
    assert top == Fraction(k, 2 * (k + 1))
    assert Fraction(1, 2) - top == Fraction(1, 2 * (k + 1))
    assert (1 + top) != Fraction(3, 2)  # Finite draws are not population truth.
    rule = targeting_rule(
        _source(_balanced_population().table),
        "y",
        control="control",
        interact=["x0"],
        fraction=0.5,
        cluster_weight="equal",
        bootstrap=ClusterBootstrap(seed=57, repetitions=99),
    )
    x = tuple(float(2 * Fraction(j, k + 1) - 1) for j in range(1, k + 1))
    rows = tuple((f"u{g}", f"c{g}", "control", 10.0, 0.0, value) for g, value in enumerate(x))
    oracle = ClusteredCATEResult(
        _scenario(n_clusters=k, members_per_cluster=1, dimension=1),
        "oracle",
        0,
        rows,
        ("unit_id", "cluster_id", "group_id", "y", "z", "x0"),
        (10.0,) * k,
        tuple(11 + value for value in x),
        tuple(1 + value for value in x),
        tuple(range(k)),
        (1,) * k,
    )
    # Expected order statistics form an exact quadrature for this linear target.
    target = policy_truth(rule, oracle)
    assert target.equal_cluster_effect == pytest.approx(float(1 + top))
    assert target.equal_cluster_share == 0.5


def test_point_accuracy_and_nonvacuity_inventory_detects_changed_or_missing_public_values():
    from increment.simulate.cluster_dgp import ClusteredObservation
    from increment.simulate.runner import _KeyOutcome

    s = _scenario(n_clusters=200)
    cell = ClusteredCell(
        "point-accounting", s, "member_count", _statistics(s, "member_count"), "calibration"
    )
    # This deterministic algebra check uses a rounding tolerance, not a campaign band.
    plans = tuple(
        PointAccuracy(cell.name, f"{name}.{quantity}", 1e-10, 0.5)
        for name, quantity in product(("policy", "selection"), ("effect", "uplift"))
    )
    registration = ClusteredRegistration((cell,), 2, 99, 57, 12, point_accuracy=plans)
    for plan in plans:
        correct = ClusteredObservation(
            plan.statistic,
            2.5,
            _KeyOutcome(
                status="ok",
                point=2.5,
                interval_reason="simulate.cluster_dgp.gated_policy_point_only",
            ),
        )
        changed = replace(correct, outcome=replace(correct.outcome, point=102.5))
        closed = replace(
            correct,
            outcome=_KeyOutcome(
                status="excluded", reason="simulate.cluster_dgp.policy_gate_closed"
            ),
        )
        for observation, error, missing in ((correct, 0, 0), (changed, 1, 0), (closed, None, 1)):
            rows = {
                o.statistic: o.outcome
                for o in _point_accuracy_observations(registration, cell, (observation,))
            }
            assert rows[f"{plan.statistic}.unavailable"].point == missing
            assert rows[f"{plan.statistic}.accuracy"].point == error
            if missing:
                assert rows[f"{plan.statistic}.accuracy"].reason == closed.outcome.reason


@pytest.mark.slow
@pytest.mark.parametrize(
    "weight,grain", tuple(product(("member_count", "equal"), ("unit", "cluster")))
)
def test_exact_policy_target_preserves_evaluation_roster(weight, grain):
    s = _scenario(
        n_clusters=200,
        dimension=1,
        size_mode="variable",
        member_counts=(3, 7, 11, 5) * 50,
        intervention_grain=grain,
    )
    data = honest_clustered_population(s)
    options: dict[str, Any] = {
        "control": "control",
        "interact": ["x0"],
        "cluster_weight": weight,
        "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
        "include_evaluation_population": True,
    }
    source = clustered_source(data)
    rules = (
        targeting_rule(source, "y", fraction=0.5, **options),
        select_targeting_rule(source, "y", fractions=(0.5,), seed=12, n_folds=2, **options).rule,
    )
    positions = {str(value): index for index, value in enumerate(data.table["unit_id"])}
    for rule in rules:
        truth, average = evaluation_policy_truth(rule, data)
        snapshot = rule.validation.evaluation_population
        assert snapshot is not None
        held = np.asarray([positions[value] for value in snapshot.unit_ids], dtype=np.intp)
        assert len(truth.actions) == len(held) == rule.validation.n_holdout
        assert data.sizes == s.member_counts
        weights = data.weights(weight)[held]
        tau = np.asarray(data.y1)[held] - np.asarray(data.y0)[held]
        actions = np.asarray(truth.actions)
        expected = np.average(tau[actions], weights=weights[actions]) if actions.any() else None
        actual = truth.member_effect if weight == "member_count" else truth.equal_cluster_effect
        assert actual == pytest.approx(expected) if expected is not None else actual is None
        assert average == pytest.approx(np.average(tau, weights=weights))
        reversed_data = replace(
            data,
            rows=data.rows[::-1],
            y0=data.y0[::-1],
            y1=data.y1[::-1],
            conditional_tau=data.conditional_tau[::-1],
            cluster_index=data.cluster_index[::-1],
        )
        reversed_truth, reversed_average = evaluation_policy_truth(rule, reversed_data)
        assert reversed_truth == truth
        assert reversed_average == pytest.approx(average)
        altered_snapshot = snapshot.model_copy(
            update={"base_weights": tuple(2 * value for value in snapshot.base_weights)}
        )
        altered = rule.model_copy(
            update={
                "validation": rule.validation.model_copy(
                    update={"evaluation_population": altered_snapshot}
                )
            }
        )
        with pytest.raises(InvalidRequestError) as caught:
            evaluation_policy_truth(altered, data)
        assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_independent_rank_integrals_have_hand_computed_tied_targets():
    # Half the population has effect 3 at the top score; half has effect 1.
    score = np.array([1.0, 1.0, 0.0, 0.0])
    tau = np.array([3.0, 3.0, 1.0, 1.0])
    autoc, qini = _rank_truth(score, tau, np.ones(4), clustered=True)
    assert autoc == pytest.approx(math.log(2))
    assert qini == pytest.approx(0.25)
    np.testing.assert_allclose(
        _rank_truth(score, np.full(4, 7.0), np.ones(4), clustered=True), [0, 0], atol=1e-14
    )


@pytest.mark.slow
@pytest.mark.parametrize("grain", ["unit", "cluster"])
def test_public_validation_selection_and_predictions_have_available_evidence(grain):
    data = _balanced_population(grain=grain)
    source = _source(data.table, grain=grain)
    options: dict[str, Any] = {
        "control": "control",
        "interact": ["x0"],
        "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
        "include_evaluation_population": True,
    }
    validation = validate_cate(source, "y", n_groups=2, **options)
    assert validation.holdout_ate is not None and validation.holdout_ate.lb is not None
    assert validation.holdout_ate.ub is not None
    assert validation.holdout_ate_se is not None and validation.holdout_ate_se > 0
    assert validation.holdout_ate.lb < validation.holdout_ate.ub
    assert validation.autoc.estimate is not None
    assert validation.autoc.se is not None and validation.autoc.se > 0
    assert validation.qini.se is not None and validation.qini.se > 0
    selected = select_targeting_rule(
        source, "y", fractions=(0.0, 0.5, 1.0), n_folds=2, seed=12, **options
    )
    assert selected.rule.deploy_grain == grain
    oracle = _four_cluster_oracle()
    truth = policy_truth(selected.rule, oracle)
    d = np.array(truth.actions)
    expected = np.mean(np.asarray(oracle.tau)[d]) if d.any() else None
    if expected is None:
        assert truth.member_effect is None
    else:
        assert truth.member_effect == pytest.approx(expected)
    np.testing.assert_array_equal(
        TargetingRule.model_validate_json(selected.rule.model_dump_json()).predict(
            oracle.covariates, cluster_ids=np.asarray(oracle.table["cluster_id"])
        ),
        d,
    )


@pytest.mark.slow
@pytest.mark.parametrize("grain", ["unit", "cluster"])
def test_observational_mixed_clusters_adjust_all_confounders_and_keep_unit_policies(grain):
    s = _scenario(
        n_clusters=200,
        members_per_cluster=20,
        assignment="observational",
        effect=2,
        heterogeneity=0,
        size_effect=0,
        intervention_grain=grain,
    )
    data = simulate_clustered_cate(s)
    source = clustered_source(data)
    assert source.context.design.adjustment.covariates == ("z", "x0", "x1")
    groups = np.asarray(data.table["group_id"])
    assert any(
        np.unique(groups[np.array(data.cluster_index) == g]).size == 2 for g in range(s.n_clusters)
    )
    bootstrap = ClusterBootstrap(seed=57, repetitions=99)
    validation = validate_cate(
        source, "y", control="control", interact=["x0"], n_groups=2, bootstrap=bootstrap
    )
    assert validation.population is None
    assert validation.holdout_ate is not None
    assert validation.holdout_ate.lb is not None and validation.holdout_ate.ub is not None
    assert validation.holdout_ate.lb < validation.holdout_ate.value < validation.holdout_ate.ub
    assert validation.holdout_ate_se is not None and validation.holdout_ate_se > 0
    # This is a nonvacuity smoke witness; the registered cells assess calibration.
    assert abs(validation.holdout_ate.value - 2) < 4 * validation.holdout_ate_se
    np.testing.assert_allclose(data.tau, 2)
    rule = targeting_rule(
        source, "y", control="control", interact=["x0"], fraction=0.5, bootstrap=bootstrap
    )
    assert rule.deploy_grain == grain
    actions = rule.predict(data.covariates, cluster_ids=np.asarray(data.table["cluster_id"]))
    if grain == "unit":
        assert any(
            np.unique(actions[np.array(data.cluster_index) == g]).size == 2
            for g in range(s.n_clusters)
        )
    else:
        assert all(
            np.unique(actions[np.array(data.cluster_index) == g]).size == 1
            for g in range(s.n_clusters)
        )
    with pytest.raises(InvalidRequestError) as caught:
        estimate_cate(source, "y", control="control", interact=["x0"])
    assert caught.value.code == "cate.identification.randomized_only"


@pytest.mark.parametrize("style", ["plain", "unicode", "integer"])
@pytest.mark.parametrize("weight", ["member_count", "equal"])
def test_arbitrary_ids_row_order_and_exact_uniform_cloning_preserve_public_fit(style, weight):
    data = _balanced_population()
    table = data.table
    if style != "plain":
        labels = [
            1009 * g - 71 if style == "integer" else f"店舗/é:{g} | α" for g in data.cluster_index
        ]
        table = table.set_column(1, "cluster_id", pa.array(labels))
    options: dict[str, Any] = {"control": "control", "interact": ["x0"], "cluster_weight": weight}
    fit = estimate_cate(_source(table), "y", **options)
    reversed_fit = estimate_cate(
        _source(table.take(pa.array(np.arange(len(data.rows))[::-1]))), "y", **options
    )
    cloned = _balanced_population(clones=5).table
    if style != "plain":
        cloned = cloned.set_column(1, "cluster_id", pa.array(np.repeat(labels, 5)))
    clone_fit = estimate_cate(_source(cloned), "y", **options)
    for actual in (reversed_fit, clone_fit):
        assert actual.ate == pytest.approx(2)
        assert actual.lb == pytest.approx(fit.lb)
        assert actual.ub == pytest.approx(fit.ub)
        # Cloning changes the fitted sample-SD basis, not effects in original units.
        for x0 in (-0.75, 0.75):
            observed = actual.cate({"x0": x0})
            expected = fit.cate({"x0": x0})
            assert (observed.value, observed.lb, observed.ub) == pytest.approx(
                (expected.value, expected.lb, expected.ub)
            )
        assert actual.reference_df == 39


@pytest.mark.slow
@pytest.mark.parametrize("weight", ["member_count", "equal"])
def test_uniform_cloning_preserves_validation_and_deployment_not_just_fit(weight):
    records = []
    for clones in (1, 5):
        data = _balanced_population(clones=clones)
        source = _source(data.table)
        options: dict[str, Any] = {
            "control": "control",
            "interact": ["x0"],
            "cluster_weight": weight,
            "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
        }
        validation = validate_cate(source, "y", n_groups=2, **options)
        rule = targeting_rule(source, "y", fraction=0.5, **options)
        records.append((validation, rule, policy_truth(rule, _four_cluster_oracle())))
    a, b = records
    assert a[0].holdout_ate is not None and b[0].holdout_ate is not None
    for attribute in ("value", "lb", "ub"):
        assert getattr(a[0].holdout_ate, attribute) == pytest.approx(
            getattr(b[0].holdout_ate, attribute)
        )
    for name in ("autoc", "qini"):
        left, right = getattr(a[0], name), getattr(b[0], name)
        assert left.estimate == pytest.approx(right.estimate)
        assert left.se == pytest.approx(right.se)
        assert left.unavailable_reason == right.unavailable_reason
    assert a[2].actions == b[2].actions
    assert a[2].member_effect == pytest.approx(b[2].member_effect)
    assert a[2].equal_cluster_effect == pytest.approx(b[2].equal_cluster_effect)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("missing_identity", "source.frame.cluster_labels"),
        ("duplicate_unit", "source.frame.duplicate_units"),
        ("mixed_randomized", "source.frame.cluster_labels"),
    ],
)
def test_incomplete_or_malformed_roster_has_exact_construction_refusal(fault, code):
    table = _balanced_population().table
    if fault == "missing_identity":
        ids = table["cluster_id"].to_pylist()
        ids[0] = None
        table = table.set_column(1, "cluster_id", pa.array(ids))
    elif fault == "duplicate_unit":
        table = pa.concat_tables([table, table.slice(0, 1)])
    else:
        groups = table["group_id"].to_pylist()
        groups[0] = "treatment"
        table = table.set_column(2, "group_id", pa.array(groups))
    with pytest.raises(InvalidRequestError) as caught:
        _source(table)
    assert caught.value.code == code


@pytest.mark.slow
def test_shared_reducer_separates_real_point_only_interval_and_failed_results():
    data = _balanced_population()
    source = _source(data.table)
    fit = estimate_cate(source, "y", control="control", interact=["x0"])
    validation = validate_cate(
        source,
        "y",
        control="control",
        interact=[],
        n_groups=2,
        bootstrap=ClusterBootstrap(seed=57, repetitions=99),
    )
    assert validation.autoc.estimate == pytest.approx(0)
    assert validation.autoc.lb is validation.autoc.ub is None
    assert (
        validation.autoc.unavailable_reason == "estimation.targeting.degenerate_rank_distribution"
    )
    with pytest.raises(InvalidRequestError) as caught:
        estimate_cate(
            _source(data.table.slice(0, 1).append_column("unused", pa.array([0]))),
            "y",
            control="control",
            interact=[],
        )
    records = (
        _record("effect", fit, 2),
        _record("effect", validation.autoc, 0),
        _record("effect", caught.value, 2),
    )
    result = reduce_clustered_observations(records)["effect"]
    assert result.attempted == 3
    assert result.point_estimable == 2 and result.interval_estimable == 1 and result.failed == 1
    assert result.bias == pytest.approx(0, abs=1e-12)
    assert result.coverage_conditional == 1 and result.coverage_unconditional == pytest.approx(
        1 / 3
    )
    assert result.coverage_conditional_mcse is None
    assert result.coverage_unconditional_mcse == pytest.approx(math.sqrt((1 / 3) * (2 / 3) / 3))
    assert result.interval_unavailable_reasons == {
        "estimation.targeting.degenerate_rank_distribution": 1
    }
    assert result.failure_reasons == {"cate.contrasts_exactly_two": 1}


def _compliance_source(order):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        one_sided=True,
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="Only uptake changes Y"
        ),
        min_first_stage_z=0.001,
    )
    pairs = [(2, 0), (4, 1), (7, 5), (12, 11)] * 10
    rows = []
    for group in ("control", "treatment"):
        for g, (size, take) in enumerate(pairs):
            for j in range(size):
                uptake = int(group == "treatment" and j < take)
                y = 2 + 2 * uptake + j % 2
                rows.append(
                    {
                        "unit_id": f"{group}/{g}/{j}",
                        "cluster_id": f"{group}/{g}",
                        "group_id": group,
                        "clicked": uptake,
                        "full": float(y),
                        "drop": float(y) if j in (0, size - 1) else None,
                    }
                )
    with pytest.warns(IncrementWarning) as rec:
        source = from_unit_summary(
            pa.Table.from_pylist(rows),
            unit="unit_id",
            group="group_id",
            control="control",
            cluster="cluster_id",
            design=design,
            experiment_id="c12-uptake",
            metrics=[
                MetricSpec(name=name, missing="drop" if name == "drop" else "error")
                for name in order
            ],
        )
    assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    return source, design, pairs


@pytest.mark.slow
def test_unequal_size_design_uptake_metric_order_and_complete_wire_parity(tmp_path):
    import pyarrow.parquet as pq
    from scipy.stats import t

    from increment import Analysis, readouts
    from increment.sources import ComplianceSummary, MomentsSource

    sources = [_compliance_source(order) for order in (("full", "drop"), ("drop", "full"))]
    reference = None
    for i, (source, design, pairs) in enumerate(sources):
        summary = source.compliance_summary(design)
        m, u = np.array(pairs, dtype=float).T
        rate = u.sum() / m.sum()
        variance = np.sum((u - rate * m) ** 2) / ((len(m) - 1) * len(m) * m.mean() ** 2)
        assert rate == pytest.approx(17 / 25)
        assert rate != pytest.approx(np.mean(u / m))
        arm = summary.arm("treatment")
        assert arm is not None
        assert arm.n_units == 250 and arm.n_clusters == 40
        assert arm.uptake_total == 170
        assert arm.ref_uptake == pytest.approx(u.mean())
        assert arm.ref_size == pytest.approx(m.mean())
        assert arm.cluster_uptake1 == pytest.approx(0)
        assert arm.cluster_size1 == pytest.approx(0)
        assert arm.cluster_uptake2 == pytest.approx(np.sum((u - u.mean()) ** 2))
        assert arm.cluster_size2 == pytest.approx(np.sum((m - m.mean()) ** 2))
        assert arm.cluster_cross == pytest.approx(np.sum((u - u.mean()) * (m - m.mean())))
        wire = summary.to_wire()
        assert ComplianceSummary.from_wire(wire, study_id=source.context.study_id) == summary
        path = tmp_path / f"uptake-{i}.parquet"
        source.export_moments(path)
        rows = pq.read_table(path).to_pylist()
        cube = MomentsSource(
            rows, metrics=source.context.metrics, study_id=source.context.study_id, design=design
        )
        assert cube.compliance_summary(design).to_wire() == wire
        assert cube.unit_counts() == source.unit_counts() == {"control": 250, "treatment": 250}
        assert cube.cluster_counts() == source.cluster_counts() == {"control": 40, "treatment": 40}
        for metric in source.context.metrics:
            assert list(cube.moments(metric)) == list(source.moments(metric))
        live = readouts.run(source, estimands=["compliance", "late"])
        replay = readouts.run(cube, estimands=["compliance", "late"])

        def keyed(results):
            return {
                (r.metric, r.estimand, r.value_scale): r.model_dump(mode="json") for r in results
            }

        assert keyed(live) == keyed(replay)
        compliance = next(row for row in live if row.estimand == "compliance").require_lift()
        # Welch-Satterthwaite collapses to df=39 here (single treatment arm,
        # 40 clusters -> 40-1), not the naive pooled 2*(40-1)=78: control-arm
        # uptake variance is structurally zero in this fixture.
        half = t.isf(0.025, 39) * math.sqrt(variance)
        assert compliance.value == pytest.approx(rate)
        assert compliance.lb == pytest.approx(rate - half)
        assert compliance.ub == pytest.approx(rate + half)
        assert compliance.lb is not None and compliance.ub is not None
        assert compliance.lb < compliance.value < compliance.ub
        assert any(row.estimand == "late" and row.require_lift().lb is not None for row in live)
        portable = Analysis.from_moments(
            rows, metrics={m.name: "mean" for m in source.context.metrics}, design=design
        )
        assert keyed(portable.run(estimands=["compliance", "late"])) == keyed(live)
        if reference is None:
            reference = (wire, keyed(live))
        else:
            assert (wire, keyed(live)) == reference


@pytest.mark.slow
@pytest.mark.parametrize("k,members,grain", list(product((16, 80), (5, 100), ("unit", "cluster"))))
def test_original_varying_noise_witness_retained_with_independent_truth(k, members, grain):
    s = _scenario(
        n_clusters=k,
        members_per_cluster=members,
        dimension=1,
        witness="original_varying_noise",
        intervention_grain=grain,
    )
    data = simulate_clustered_cate(s)
    x = np.asarray(data.table["x0"])
    np.testing.assert_allclose(data.potential_outcomes["tau"], 0.12 * x)
    assert data.truth.member_ate == pytest.approx(0 if k == 80 else -0.192)
    source = clustered_source(data)
    options: dict[str, Any] = {"control": "control", "interact": ["x0"]}
    fit = estimate_cate(source, "y", **options)
    # Independent WLS/CR1 equation, not the estimator's returned coefficients as truth.
    d = (np.asarray(data.table["group_id"]) == "treatment").astype(float)
    z = np.column_stack([np.ones(x.size), d, x - x.mean(), d * (x - x.mean())])
    bread = np.linalg.inv(z.T @ z)
    beta = bread @ z.T @ np.asarray(data.table["y"])
    residual = np.asarray(data.table["y"]) - z @ beta
    g = np.asarray(data.cluster_index)
    scores = np.stack([np.sum(z[g == i] * residual[g == i, None], axis=0) for i in range(k)])
    covariance = k / (k - 1) * bread @ scores.T @ scores @ bread
    assert fit.ate == pytest.approx(beta[1])
    assert fit.se**2 == pytest.approx(covariance[1, 1])
    assert fit.se > 0
    bootstrap = ClusterBootstrap(seed=0, repetitions=999)
    validation = validate_cate(source, "y", bootstrap=bootstrap, **options)
    assert validation.holdout_ate_se is not None and validation.holdout_ate_se > 0
    assert validation.autoc.se is not None and validation.autoc.se > 0
    selection = select_targeting_rule(
        source,
        "y",
        fractions=(0.0, 0.5, 1.0),
        seed=0,
        n_folds=4 if k == 16 else 2,
        bootstrap=bootstrap,
        **options,
    )
    actions = selection.rule.predict(
        data.covariates, cluster_ids=np.asarray(data.table["cluster_id"])
    )
    restored = TargetingRule.model_validate_json(selection.rule.model_dump_json())
    np.testing.assert_array_equal(
        restored.predict(data.covariates, cluster_ids=np.asarray(data.table["cluster_id"])), actions
    )


def test_manifest_controls_statistics_intersections_and_prospective_seeds():
    cells = cell_inventory()
    scenarios = [c.scenario for c in cells]
    assert {s.icc for s in scenarios} >= {0, 0.2, 0.5}
    assert {s.members_per_cluster for s in scenarios} >= {5, 20, 100}
    assert {s.size_mode for s in scenarios} == {"fixed", "variable", "informative"}
    assert {s.dimension for s in scenarios} >= {0, 2, 10}
    assert {s.leverage for s in scenarios} >= {0, 8}
    intersections = {(s.n_clusters, s.treatment_ratio, s.skew) for s in scenarios}
    assert set(product((4, 10, 20, 40, 200), (0.5, 0.75, 0.9), (0, 1))) <= intersections
    for grain, weight in product(("unit", "cluster"), ("member_count", "equal")):
        assert any(
            c.scenario.size_mode == "informative"
            and c.weighting == weight
            and c.scenario.intervention_grain == grain
            for c in cells
        )
        assert any(
            c.scenario.assignment == "observational"
            and c.weighting == weight
            and c.scenario.intervention_grain == grain
            for c in cells
        )
    for cell in cells:
        rebuilt = ClusteredCATEScenario.model_validate(cell.scenario.model_dump())
        assert rebuilt == cell.scenario
        expected = {
            "validation.ate",
            "validation.autoc",
            "validation.qini",
            "validation.autoc.reject",
            "validation.qini.reject",
            "validation.group1",
            "validation.group2",
            "policy.effect",
            "policy.uplift",
            "selection.effect",
            "selection.uplift",
        }
        expected.update(
            f"{name}.{quantity}.{gate}"
            for name, quantity, gate in product(
                ("policy", "selection"), ("effect", "uplift"), ("accuracy", "unavailable")
            )
        )
        if cell.scenario.assignment != "observational":
            expected.add("fit.ate")
        assert {stat.name for stat in cell.statistics} == expected
        assert all(
            {
                "point_estimable",
                "interval_estimable",
                "bias_mcse",
                "coverage_conditional",
                "coverage_unconditional",
                "interval_unavailable_reasons",
            }
            <= set(stat.accounting)
            for stat in cell.statistics
        )
    mutable = list(cells)
    plans = tuple(
        PointAccuracy(cell.name, stat.name.removesuffix(".accuracy"), 0.01, 0.5)
        for cell in cells
        if cell.purpose == "calibration"
        for stat in cell.statistics
        if stat.gate == "point_accuracy"
    )
    registration = ClusteredRegistration(cast(Any, mutable), 2, 99, 57, 12, point_accuracy=plans)
    fingerprint = registration.fingerprint
    mutable.clear()
    assert registration.fingerprint == fingerprint
    assert registration.seed_inventory == tuple((c.name, c.scenario.seed, range(2)) for c in cells)
    assert all(callable(globals()[node.split("::")[1]]) for node in registration.witness_inventory)
    assert replace(registration, bootstrap_seed=58).fingerprint != fingerprint
    assert registration.prospective_precision == ()
    # Every declared accuracy is answered, so the only release refusal left is MC precision.
    allocated = replace(registration, global_gated_statistics=10_000, replications=2 * 10**6)
    assert all(margin <= limit for _, margin, limit, _ in allocated.prospective_precision)
    with pytest.raises(InvalidRequestError) as insufficient:
        replace(registration, global_gated_statistics=10_000)
    assert insufficient.value.code == "simulate.cluster_dgp.invalid_scenario"


@pytest.mark.slow
@pytest.mark.parametrize("members", [5, 20, 100])
def test_inventory_member_size_labels_execute_the_actual_scenario(members):
    cell = next(c for c in cell_inventory() if c.name == f"icc=0.2/m={members}/member_count")
    data = simulate_clustered_cate(cell.scenario)
    assert data.sizes == (members,) * 200
    assert len(data.y0) == 200 * members
    assert data.truth.member_ate == pytest.approx(
        np.mean(data.potential_outcomes["y1"] - data.potential_outcomes["y0"])
    )


def _homogeneous_calibration_cell():
    """One frozen homogeneous calibration cell from the public inventory."""
    name = "homogeneous/effect=0.5/member_count"
    return next(cell for cell in cell_inventory() if cell.name == name)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_bounded_public_replications_account_every_statistic_and_cannot_pass_all_null():
    cell = _homogeneous_calibration_cell()
    registration = ClusteredRegistration((cell,), 2, 99, 57, 12)
    attempts = [evaluate_clustered_replication(registration, cell.name, r) for r in range(2)]
    report = reduce_clustered_observations(tuple(o for run in attempts for o in run.observations))
    assert set(report) == {spec.name for spec in cell.statistics}
    for stats in report.values():
        assert stats.attempted == 2
        assert stats.point_estimable + stats.excluded + stats.failed == 2
        assert sum(stats.failure_reasons.values()) == stats.failed
        assert sum(stats.exclusion_reasons.values()) == stats.excluded
        assert (
            sum(stats.interval_unavailable_reasons.values())
            == stats.point_estimable - stats.interval_estimable
        )
    for name in ("fit.ate", "validation.ate"):
        stats = report[name]
        assert stats.point_estimable == stats.interval_estimable == 2
        assert stats.bias is not None and math.isfinite(stats.bias)
        assert stats.bias_mcse is not None and stats.bias_mcse >= 0
    assert all(run.registration == registration.fingerprint for run in attempts)
    assert all(run.policy_oracles for run in attempts)


def _accounting_evidence_table():
    from increment.simulate.cluster_dgp import ClusteredObservation
    from increment.simulate.runner import _KeyOutcome

    return reduce_clustered_observations(
        [
            ClusteredObservation(
                "fit.ate",
                1.0,
                _KeyOutcome(status="ok", point=1.0, interval_reason="point_only"),
            ),
            ClusteredObservation("fit.ate", 1.0, _KeyOutcome(status="failed", reason="failed")),
            ClusteredObservation("fit.ate", 1.0, _KeyOutcome(status="excluded", reason="excluded")),
        ]
    )


def test_clustered_evidence_rejects_registration_fingerprint_mismatch():
    cell = _homogeneous_calibration_cell()
    registration = ClusteredRegistration((cell,), 2, 99, 57, 12)
    table = _accounting_evidence_table()
    evidence = ClusteredEvidenceArtifact(registration.fingerprint, {cell.name: table})
    assert evidence.validated_tables(registration)[cell.name] == table
    changed = replace(registration, bootstrap_seed=58)
    with pytest.raises(CodedError) as caught:
        evidence.validated_tables(changed)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"
    assert evidence.payload()["registration"] == registration.fingerprint


def test_clustered_evidence_payload_reader_round_trips_and_validates_accounting():
    import json

    cell = _homogeneous_calibration_cell()
    registration = ClusteredRegistration((cell,), 2, 99, 57, 12)
    evidence = ClusteredEvidenceArtifact(
        registration.fingerprint, {cell.name: _accounting_evidence_table()}
    )
    restored = ClusteredEvidenceArtifact.from_payload(json.loads(json.dumps(evidence.payload())))
    assert restored.validated_tables(registration) == evidence.tables
    changed = replace(registration, bootstrap_seed=58)
    with pytest.raises(CodedError) as caught:
        restored.validated_tables(changed)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"

    malformed = cast(dict[str, Any], evidence.payload())
    malformed["tables"][cell.name]["fit.ate"]["failed"] = 2
    with pytest.raises(CodedError) as caught:
        ClusteredEvidenceArtifact.from_payload(malformed)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def _available_interval_evidence_payload():
    from increment.simulate.cluster_dgp import ClusteredObservation
    from increment.simulate.runner import _KeyOutcome

    table = reduce_clustered_observations(
        [
            ClusteredObservation(
                "fit.ate",
                0.0,
                _KeyOutcome(status="ok", point=0.0, lb=-1.0, ub=1.0),
            ),
            ClusteredObservation(
                "fit.ate",
                0.0,
                _KeyOutcome(status="ok", point=0.0, lb=-1.0, ub=1.0),
            ),
            ClusteredObservation("fit.ate", 0.0, _KeyOutcome(status="failed", reason="failed")),
        ]
    )
    return cast(
        dict[str, Any],
        ClusteredEvidenceArtifact("registered", {"cell": table}).payload(),
    )


def test_clustered_evidence_reader_rejects_interval_rate_denominator_contradiction():
    payload = _available_interval_evidence_payload()
    row = payload["tables"]["cell"]["fit.ate"]
    row["coverage_unconditional"] = 1.0
    row["coverage_unconditional_mcse"] = 0.0
    with pytest.raises(CodedError) as caught:
        ClusteredEvidenceArtifact.from_payload(payload)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_clustered_evidence_reader_rejects_set_rate_denominator_contradiction():
    payload = _available_interval_evidence_payload()
    row = payload["tables"]["cell"]["fit.ate"]
    row["set_coverage_unconditional"] = 1.0
    row["set_coverage_unconditional_mcse"] = 0.0
    with pytest.raises(CodedError) as caught:
        ClusteredEvidenceArtifact.from_payload(payload)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


@pytest.mark.parametrize(
    "field",
    (
        "coverage_conditional_mcse",
        "coverage_unconditional_mcse",
        "set_coverage_conditional_mcse",
        "set_coverage_unconditional_mcse",
    ),
)
def test_clustered_evidence_reader_rejects_mismatched_coverage_mcse(field):
    payload = _available_interval_evidence_payload()
    row = payload["tables"]["cell"]["fit.ate"]
    row[field] = 0.25
    with pytest.raises(CodedError) as caught:
        ClusteredEvidenceArtifact.from_payload(payload)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_clustered_evidence_reader_rejects_unconditional_hit_with_no_available_intervals():
    payload = cast(
        dict[str, Any],
        ClusteredEvidenceArtifact("registered", {"cell": _accounting_evidence_table()}).payload(),
    )
    row = payload["tables"]["cell"]["fit.ate"]
    row["coverage_unconditional"] = 1.0
    row["coverage_unconditional_mcse"] = 0.0
    with pytest.raises(CodedError) as caught:
        ClusteredEvidenceArtifact.from_payload(payload)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_clustered_evidence_reader_preserves_rounded_large_count_rates():
    payload = _available_interval_evidence_payload()
    row = payload["tables"]["cell"]["fit.ate"]
    attempted = 2**60
    row.update(
        attempted=attempted,
        point_estimable=attempted - 1,
        interval_estimable=attempted - 1,
        confidence_set_estimable=attempted - 1,
    )
    for prefix in ("coverage", "set_coverage"):
        row[f"{prefix}_unconditional"] = (attempted - 1) / attempted
        row[f"{prefix}_unconditional_mcse"] = 0.0
    restored = ClusteredEvidenceArtifact.from_payload(payload).tables["cell"]["fit.ate"]
    assert restored.attempted == attempted
    assert restored.interval_estimable == attempted - 1
    assert restored.coverage_unconditional == 1.0


def test_clustered_evidence_payload_reader_owns_nested_reason_maps():
    evidence = ClusteredEvidenceArtifact("registered", {"cell": _accounting_evidence_table()})
    payload = cast(dict[str, Any], evidence.payload())
    restored = ClusteredEvidenceArtifact.from_payload(payload)
    payload["tables"]["cell"]["fit.ate"]["failure_reasons"]["forged"] = 1
    assert restored.tables["cell"]["fit.ate"].failure_reasons == {"failed": 1}
    with pytest.raises(TypeError):
        cast(Any, restored.tables["cell"]["fit.ate"].failure_reasons)["forged"] = 1


def test_clustered_evidence_reader_preserves_reducer_availability_boundaries():
    import json

    from increment.simulate.runner import _KeyOutcome, _reduce_key

    outcomes = {
        "empty": [],
        "singleton": [_KeyOutcome(status="ok", point=1.0, lb=0.5, ub=1.5)],
        "failed": [_KeyOutcome(status="failed", reason="unavailable")],
        "set_only": [_KeyOutcome(status="ok", set_lower=-1.0, set_upper=None)],
    }
    table = {name: _reduce_key(rows, truth=1.0) for name, rows in outcomes.items()}
    evidence = ClusteredEvidenceArtifact("registered", {"cell": table})
    payload = json.loads(json.dumps(evidence.payload(), allow_nan=False))
    assert ClusteredEvidenceArtifact.from_payload(payload).tables["cell"] == table


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failed", True),
        ("bias", 10**400),
        ("failure_reasons", {"failed": 2}),
        ("coverage_conditional", 0.5),
    ],
)
def test_clustered_evidence_reader_refuses_malformed_accounting_fields(field, value):
    evidence = ClusteredEvidenceArtifact("registered", {"cell": _accounting_evidence_table()})
    payload = cast(dict[str, Any], evidence.payload())
    payload["tables"]["cell"]["fit.ate"][field] = value
    with pytest.raises(CodedError) as caught:
        ClusteredEvidenceArtifact.from_payload(payload)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_clustered_evidence_owns_nested_accounting_maps():
    table = dict(_accounting_evidence_table())
    row = table["fit.ate"]
    caller_reasons = dict(row.failure_reasons)
    table["fit.ate"] = replace(row, failure_reasons=caller_reasons)
    evidence = ClusteredEvidenceArtifact("registered", {"cell": table})
    caller_reasons["changed"] = 1
    table.clear()
    retained = evidence.tables["cell"]["fit.ate"]
    assert retained.failure_reasons == {"failed": 1}
    assert retained.bias == 0.0
    with pytest.raises(TypeError):
        cast(Any, retained.failure_reasons)["changed"] = 1


@pytest.mark.slow
def test_sizing_power_buffer_admits_a_feasible_registration_and_rejects_coarse_margins():
    cell = next(c for c in cell_inventory() if c.purpose == "availability")
    global_count = 10_000
    eta = 0.01 / (2 * global_count)
    repetitions = math.ceil(math.log(1 / eta) / (2 * 0.0025**2))
    registration = ClusteredRegistration(
        (cell,), repetitions, 99, 57, 12, global_gated_statistics=global_count
    )
    precision = registration.prospective_precision
    sizing_count = len(clustered_sizing_inventory())
    assert len(precision) == sizing_count
    sizing = precision[-sizing_count:]
    # The buffer above the 0.8 acceptance threshold is a declared design constant,
    # so the sizing gate allows its full margin instead of a rounding residue.
    assert all(power >= 0.85 for power, _, _, _ in sizing)
    assert all(limit == 0.0025 for _, _, limit, _ in sizing)
    assert all(margin <= limit for _, margin, limit, _ in precision)

    with pytest.raises(InvalidRequestError) as caught:
        ClusteredRegistration(
            (cell,), repetitions // 4, 99, 57, 12, global_gated_statistics=global_count
        )
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def assert_scientific_tables(
    registration,
    evidence,
    *,
    global_gated_statistics,
    sizing_evidence,
    i13_evidence,
    i14_evidence,
):
    """Main's complete-family release assertion; no local error-budget reset.

    Evidence must carry the exact registration fingerprint; bare tables are
    intentionally rejected so changed registrations cannot be certified.
    """
    tables = evidence.validated_tables(registration)
    from tests.mc import binomial_error_upper_bound, family_eta, scientific_delta

    gates = ("coverage", "size", "point_accuracy", "point_availability")
    local_count = sum(
        spec.gate in gates
        for cell in registration.cells
        if cell.purpose == "calibration"
        for spec in cell.statistics
    )
    local_count += sum(len(c.statistics) for c in clustered_sizing_inventory())
    assert registration.global_gated_statistics == global_gated_statistics
    assert global_gated_statistics >= local_count
    eta = family_eta(registration.family_mc_error, global_gated_statistics)
    plans = {(p.cell, p.statistic): p for p in registration.point_accuracy}
    for cell in registration.cells:
        if cell.purpose != "calibration":
            continue
        if not cell.scenario.interactions:
            # A constant fitted score has no rank variation and cannot open
            # either policy gate; this does not exempt the full-fit ATE.
            for name, quantity in product(("policy", "selection"), ("effect", "uplift")):
                row = tables[cell.name][f"{name}.{quantity}"]
                assert row.attempted == registration.replications
                assert row.point_estimable == 0
                assert row.failed + row.excluded == row.attempted
        for spec in cell.statistics:
            if spec.gate not in gates:
                continue
            row = tables[cell.name][spec.name]
            assert row.attempted == registration.replications
            assert row.point_estimable > 0, (cell.name, spec.name, row)
            q = spec.nominal_error
            denominator = row.attempted
            if spec.gate == "point_availability":
                plan = plans[cell.name, spec.name.removesuffix(".unavailable")]
                q = 1 - plan.minimum_availability
            assert q is not None
            if spec.gate == "coverage":
                assert row.interval_estimable > 0, (cell.name, spec.name, row)
                errors = row.attempted - round(row.coverage_unconditional * row.attempted)
            elif spec.gate == "point_accuracy":
                denominator = row.point_estimable
                errors = round(row.bias * denominator)
                base = tables[cell.name][spec.name.removesuffix(".accuracy")]
                assert row.point_estimable == base.point_estimable
                assert row.failure_reasons == base.failure_reasons
                assert row.exclusion_reasons == base.exclusion_reasons
            else:
                errors = round(row.bias * row.point_estimable) + row.excluded + row.failed
            observed = errors / denominator
            upper = binomial_error_upper_bound(errors, denominator, eta)
            delta = scientific_delta(q)
            assert upper - observed <= min(0.0025, delta / 2), (
                cell.name,
                spec.name,
                "MC precision",
                row,
            )
            limit = q if spec.gate == "point_availability" else q + delta
            assert upper <= limit, (cell.name, spec.name, upper, row)
    assert_clustered_sizing_evidence(registration, sizing_evidence, eta=eta)
    assert_existing_estimator_evidence(registration, i13_evidence, i14_evidence)


@pytest.mark.slow
@pytest.mark.parametrize("p", [0.5, 0.75, 0.9])
def test_public_scalar_power_and_estimator_share_independent_cluster_variance_equation(p):
    from scipy.stats import nct, t

    from increment import readouts
    from increment.power import Baseline, PowerDesign, achieved_power, required_sample_size
    from tests.power._procedures import make_procedure

    # Each arm's cluster means have sample variance exactly two; the member
    # deviations sum to zero. This pins the variance equation, not MC power.
    kc, kt, m = 20, round(20 * p / (1 - p)), 5
    rows = []
    for label, k, mean in (("control", kc, 10), ("treatment", kt, 12)):
        for g in range(k):
            cluster_noise = (g % 5 - 2) * math.sqrt((k - 1) / k)
            for j in range(m):
                rows.append(
                    {
                        "unit_id": f"{label}/{g}/{j}",
                        "cluster_id": f"{label}/{g}",
                        "group_id": label,
                        "y": mean + cluster_noise + 0.3 * (j - 2),
                    }
                )
    source = _source(pa.Table.from_pylist(rows))
    (estimate,) = readouts.run(source, decision_method=Method(name="unadjusted"))
    se2 = 2 / (kt * 12**2) + 2 / (kc * 10**2)
    lift = estimate.require_lift()
    assert lift.value == pytest.approx(0.2)
    assert estimate.relative_confidence_set is not None
    reference = estimate.relative_confidence_set.reference
    assert (reference.a, reference.c) == pytest.approx((2.0, 10.0))
    assert reference.var_a == pytest.approx(2 / kt + 2 / kc)
    assert reference.var_c == pytest.approx(2 / kc)
    assert reference.cov_ac == pytest.approx(-2 / kc)
    # Planning's log-delta variance is a projection, not a persisted posterior.
    gradient_a, gradient_c = 1 / 12, -2 / (12 * 10)
    projected = (
        gradient_a**2 * reference.var_a
        + gradient_c**2 * reference.var_c
        + 2 * gradient_a * gradient_c * reference.cov_ac
    )
    assert projected == pytest.approx(se2)
    assert lift.lb is not None and lift.ub is not None
    baseline = Baseline(mean=10, var=2 / (0.5 + 0.5 / m), avg_cluster_size=m, cluster_icc=0.5)
    procedure = make_procedure(
        dependence="cluster", identification="randomized", population="assigned"
    )
    design = PowerDesign(allocation=p)
    planned = achieved_power(kt * m, 0.2, baseline, procedure, design)
    reference_df = min(kt - 1, kc - 1)
    assert reference.df == reference_df
    critical = t.isf(0.025, reference_df)
    nc = math.log1p(0.2) / math.sqrt(se2)
    independent_power = nct.sf(critical, reference_df, nc) + nct.cdf(-critical, reference_df, nc)
    assert planned.power == pytest.approx(independent_power)
    sized = required_sample_size(0.2, baseline, procedure, design)
    replay = achieved_power(sized.n_per_arm, 0.2, baseline, procedure, design)
    assert replay.power == pytest.approx(sized.power)
    assert replay.power >= design.power


def _clustered_sizing_attempt(cell, plan, replication):
    from increment import readouts
    from increment.simulate.cluster_dgp import ClusteredObservation
    from increment.simulate.runner import _KeyOutcome

    # Respect the exact recommended arm sizes, including a final partial roster.
    rows = []
    for arm, n in (("treatment", plan.n_per_arm), ("control", plan.n_total - plan.n_per_arm)):
        k = math.ceil(n / 5)
        u_rng = child_rng(cell.seed, "holdout", replication, f"sizing/{arm}/cluster")
        raw = u_rng.normal(size=k)
        u = (raw + cell.skew * (raw**2 - 1)) / math.sqrt(1 + 2 * cell.skew**2)
        z = child_rng(cell.seed, "holdout", replication, f"sizing/{arm}/effect").uniform(-1, 1, k)
        for g in range(k):
            m = min(5, n - 5 * g)
            noise_rng = child_rng(cell.seed, "holdout", replication, f"sizing/{arm}/unit/{g}")
            raw = noise_rng.normal(size=m)
            noise = (raw + cell.skew * (raw**2 - 1)) / math.sqrt(1 + 2 * cell.skew**2)
            for j in range(m):
                y0 = 10 + u[g] + noise[j] + cell.heterogeneity * z[g]
                y1 = 10 * (1 + cell.relative_effect) + u[g] + noise[j] - cell.heterogeneity * z[g]
                rows.append(
                    {
                        "unit_id": f"{arm}/{g}/{j}",
                        "cluster_id": f"{arm}/{g}",
                        "group_id": arm,
                        "y": y1 if arm == "treatment" else y0,
                    }
                )
    try:
        (result,) = readouts.run(
            _source(pa.Table.from_pylist(rows)), decision_method=Method(name="unadjusted")
        )
        p = result.p_value()
    except CodedError as exc:
        outcome = _KeyOutcome(status="failed", reason=exc.code)
    else:
        if p is None or not math.isfinite(p):
            outcome = _KeyOutcome(
                status="excluded", reason="simulate.cluster_dgp.sizing_p_value_unavailable"
            )
        else:
            outcome = _KeyOutcome(
                status="ok",
                point=float(p < 0.05),
                interval_reason="simulate.cluster_dgp.rejection_indicator",
            )
    return ClusteredObservation("sizing.reject", 0.0, outcome)


def evaluate_clustered_sizing(registration):
    """Execute runtime power at public sizing; caller supplies frozen R and M.

    The public solver is called before outcomes, then independent outcomes are
    sent through the public estimator at precisely its recommended sample sizes.
    No analytical inverse roundtrip is counted as observed rejection evidence.
    """
    records = {}
    for cell in clustered_sizing_inventory():
        plan = _clustered_sizing_plan(cell)
        records[cell.name] = {
            "registration": registration.fingerprint,
            "plan": plan.model_dump(mode="json"),
            "statistics": cell.statistics,
            "table": reduce_clustered_observations(
                tuple(
                    _clustered_sizing_attempt(cell, plan, r)
                    for r in range(registration.replications)
                )
            ),
        }
    return records


def assert_clustered_sizing_evidence(registration, evidence, *, eta):
    from tests.mc import binomial_error_upper_bound, coverage_lower_bound

    cells = clustered_sizing_inventory()
    assert set(evidence) == {cell.name for cell in cells}
    for cell in cells:
        record = evidence[cell.name]
        assert record["registration"] == registration.fingerprint
        assert tuple(record["statistics"]) == cell.statistics
        plan = _clustered_sizing_plan(cell)
        assert record["plan"] == plan.model_dump(mode="json")
        row = record["table"]["sizing.reject"]
        n = row.attempted
        assert n == registration.replications and row.point_estimable > 0
        rejected = round(row.bias * row.point_estimable)
        missing = row.excluded + row.failed
        lower = coverage_lower_bound(rejected, n, eta)
        upper = binomial_error_upper_bound(rejected + missing, n, eta)
        assert rejected / n - lower <= 0.0025, (cell.name, "lower precision", row)
        assert upper - (rejected + missing) / n <= 0.0025, (cell.name, "upper precision", row)
        # Runtime power must agree with planned power within +/- .005; missing decisions cannot
        # help either side of the bound or the availability gate.
        assert lower >= plan.power - 0.005 and upper <= plan.power + 0.005, (cell.name, plan, row)
        assert lower >= 0.8, (cell.name, "useful power", row)
        available = coverage_lower_bound(row.point_estimable, n, eta)
        assert available >= 0.99
        assert row.point_estimable / n - available <= 0.0025


def assert_existing_estimator_evidence(registration, i13_evidence, i14_evidence):
    """Consume full existing inventories and retain unresolved upstream gates."""
    from tests._unit_cycle_design import DESIGNS, MANIFEST
    from tests.estimation._i13_manifest import ACCEPTANCE_MANIFEST, I13_FAMILY_ALPHA

    assert (
        math.fsum((registration.family_mc_error, I13_FAMILY_ALPHA, MANIFEST["family_alpha"]))
        <= 0.01
    )
    assert set(i13_evidence) == {cell.name for cell in ACCEPTANCE_MANIFEST}
    for cell in ACCEPTANCE_MANIFEST:
        verdict = i13_evidence[cell.name].r03_gate(cell)
        assert verdict["passes"], (cell.name, verdict)
        assert verdict["acceptance_complete"], (cell.name, verdict)
    assert set(i14_evidence) == {cell.id for _, cell in DESIGNS}
    spent = 0.0
    for index, cell in DESIGNS:
        record = i14_evidence[cell.id]
        assert record["cell_index"] == index and record["cell"] == cell.id
        assert record["status"] == "certified", (cell.id, record)
        spent += record["mc_error"]
        for lane in ("null", "nonzero", "selected"):
            assert lane in record["reports"] and lane in record["gates"]
            for gate in record["gates"][lane].values():
                assert gate["status"] == "certified", (cell.id, lane, gate)
        # Existing comparison diagnostics explicitly deny equivalence. Preserve
        # the original sizing obligation as incomplete until it has evidence.
        agreement = record.get("mde_equivalence")
        assert agreement is not None and agreement["status"] == "certified", (
            cell.id,
            "missing sizing equivalence",
        )
        assert agreement["margin"] <= 0.0025
    assert spent <= MANIFEST["family_alpha"]


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_sizing_inventory_and_actual_rejection_accounting():
    cells = clustered_sizing_inventory()
    assert {(c.allocation, c.skew, c.heterogeneity, c.relative_effect) for c in cells} == set(
        product((0.5, 0.75, 0.9), (0.0, 1.0), (0.0, 0.5), (0.1, 0.2))
    )
    for cell in (cells[0], cells[-1]):
        plan = _clustered_sizing_plan(cell)
        attempts = tuple(_clustered_sizing_attempt(cell, plan, r) for r in range(2))
        row = reduce_clustered_observations(attempts)["sizing.reject"]
        assert row.attempted == 2
        assert row.point_estimable + row.excluded + row.failed == 2
        assert row.point_estimable == 2
        assert all(a.outcome.point in (0.0, 1.0) for a in attempts)
        assert row.interval_estimable == 0
        assert row.interval_unavailable_reasons == {"simulate.cluster_dgp.rejection_indicator": 2}


def test_public_hash_halves_receive_distinct_role_draws_without_assignment_conditioning():
    s = _scenario(size_mode="informative")
    training = simulate_clustered_cate(s)
    holdout = simulate_clustered_cate(s, stream="holdout")
    combined = honest_clustered_population(s)
    expected = {
        row[0]: (row, a, b)
        for population in (training, holdout)
        for row, a, b in zip(population.rows, population.y0, population.y1, strict=True)
    }
    assert len(expected) == len(training.rows) + len(holdout.rows)
    # The role prefix is the published half identity.
    held = np.array([cast(str, row[0]).startswith("holdout/") for row in combined.rows])
    assert held.any() and not held.all()
    for row, a, b, is_holdout in zip(combined.rows, combined.y0, combined.y1, held, strict=True):
        assert isinstance(row[0], str)
        assert row[0].startswith("holdout/" if is_holdout else "training/")
        assert (row, a, b) == expected[row[0]]
    clusters = np.asarray(combined.table["cluster_id"])
    assert not set(clusters[held]) & set(clusters[~held])


def test_assignment_support_ceiling_is_an_executable_prospective_boundary():
    small = _scenario(n_clusters=40, treatment_ratio=0.9)
    ceiling = validation_support_ceiling(small)
    assert ceiling is not None
    assert ceiling <= 1 - 0.9**20
    assert ceiling < 0.945
    cell = next(c for c in cell_inventory() if c.name == "K=40/p=0.9/skew=0.0/member_count")
    assert cell.purpose == "calibration"
    stats = {stat.name: stat for stat in cell.statistics}
    assert stats["fit.ate"].gate == "coverage"
    assert stats["validation.ate"].gate == "availability"
    assert stats["selection.effect.accuracy"].gate == "point_accuracy"
    assert cell.validation_support_ceiling == ceiling
    assert (
        cell.applicability_reason
        == "simulate.cluster_dgp.unconditional_coverage_exceeds_assignment_support"
    )
    large = _scenario(n_clusters=200, treatment_ratio=0.9)
    large_ceiling = validation_support_ceiling(large)
    assert large_ceiling is not None and large_ceiling > 0.945
    # The ceiling is necessary support only; it never asserts actual estimability.
    assert validation_support_ceiling(_scenario(assignment="observational")) is None


def test_shared_accounting_serialization_preserves_nulls_and_owned_reason_maps():
    import json

    rows = pa.table(
        {
            "unit_id": ["u0", "u1"],
            "cluster_id": ["c0", "c1"],
            "group_id": ["control", "treatment"],
            "y": [1.0, 3.0],
        }
    )
    with pytest.raises(InvalidRequestError) as caught:
        estimate_cate(_source(rows), "y", control="control", interact=[])
    table = reduce_clustered_observations((_record("ate", caught.value, 2),))
    payload = clustered_table_payload(table)
    restored = json.loads(json.dumps(payload, allow_nan=False))
    assert restored["ate"]["bias"] is None
    assert restored["ate"]["bias_mcse"] is None
    assert restored["ate"]["coverage_conditional"] is None
    assert restored["ate"]["coverage_unconditional"] == 0
    assert restored["ate"]["failure_reasons"] == {"estimation.cate.each_arm_needs": 1}
    restored["ate"]["failure_reasons"].clear()
    assert table["ate"].failure_reasons == {"estimation.cate.each_arm_needs": 1}
    with pytest.raises(TypeError):
        cast(Any, table["ate"].failure_reasons)["fabricated"] = 1


def test_overlap_policy_oracle_uses_retained_cluster_mass():
    from increment.estimation.cate import Covariate
    from increment.estimation.targeting import targeting_rule_arrays

    data = _balanced_population(grain="unit")
    table = data.table
    ids = np.asarray(table["cluster_id"])
    x = np.asarray(table["x0"])

    def score(y, d, X, unit_ids, clusters):
        cutoff = np.array([int(label[1:]) % 3 - 1 for label in clusters])
        return 2 + 0.5 * X[:, 0], X[:, 0] >= cutoff

    rule = targeting_rule_arrays(
        np.asarray(table["y"]),
        (np.asarray(table["group_id"]) == "treatment").astype(float),
        {"x0": x},
        np.asarray(table["unit_id"]),
        cluster_ids=ids,
        cluster_weight="equal",
        interact=[Covariate(name="x0")],
        adjustment=("x0",),
        psi_fn=score,
        arm_summary="score",
        fraction=0.5,
        n_groups=2,
        bootstrap_repetitions=99,
        include_evaluation_population=True,
    )
    snapshot = rule.validation.evaluation_population
    assert snapshot is not None
    positions = {str(value): index for index, value in enumerate(table["unit_id"])}
    held = np.array([positions[value] for value in snapshot.unit_ids], dtype=np.intp)
    _, inverse, counts = np.unique(ids[held], return_inverse=True, return_counts=True)
    assert len(np.unique(counts)) > 1  # unequal retained sizes make inverse-mass weighting bite
    expected_weights = 1.0 / counts[inverse]
    tau = np.asarray(data.tau)[held]
    truth, average = evaluation_policy_truth(rule, data)
    selected = np.asarray(truth.actions)
    assert selected.any()
    assert average == pytest.approx(np.average(tau, weights=expected_weights))
    assert average != pytest.approx(np.average(tau, weights=data.weights("equal")[held]))
    assert truth.equal_cluster_effect == pytest.approx(
        np.average(tau[selected], weights=expected_weights[selected])
    )


def _affine_witness(
    *, dimension=1, ratio=2.0, clones=1, reverse=False, observational=False, heterogeneity=3
):
    """Fixed innovations in the random law, retaining each clone's parent."""
    base = _balanced_population(grain="unit", clones=clones)
    scenario = _scenario(
        dimension=dimension,
        treated_noise_ratio=ratio,
        clone_factor=clones,
        intervention_grain=base.scenario.intervention_grain,
        effect=2,
        heterogeneity=heterogeneity,
        size_effect=0,
        outcome_scale=0.2,
        assignment="observational" if observational else "cluster",
    )
    rows, y0, y1, tau, ck, uk = [], [], [], [], [], []
    for i, row in enumerate(base.rows):
        g, j = base.cluster_index[i], (i // clones) % 5
        x = float(row[-1]) if dimension else 0.0
        u, e = (g // 2 % 5) / 5, (((g // 2 + 3) * (j + 1)) % 17 - 8) / 5
        noise = math.sqrt(scenario.icc) * u + math.sqrt(1 - scenario.icc) * e
        z = float(g // 2 % 3 - 1)
        mu, effect = 1 + 0.8 * z + 0.6 * x, scenario.effect + scenario.heterogeneity * x
        a, b = (
            mu + scenario.outcome_scale * noise,
            mu + effect + scenario.outcome_scale * ratio * noise,
        )
        rows.append((*row[:3], b if row[2] == "treatment" else a, z, *([x] if dimension else [])))
        y0.append(a)
        y1.append(b)
        tau.append(effect)
        role = "training" if g < 20 else "holdout"
        ck.append(f"{role}/0/{g}")
        uk.append(f"{role}/0/{g}/{j}")
    order = list(range(len(rows)))
    if reverse:
        order.reverse()
    return ClusteredCATEResult(
        scenario,
        "training",
        0,
        tuple(rows[i] for i in order),
        ("unit_id", "cluster_id", "group_id", "y", "z", *(["x0"] if dimension else [])),
        tuple(y0[i] for i in order),
        tuple(y1[i] for i in order),
        tuple(tau[i] for i in order),
        tuple(base.cluster_index[i] for i in order),
        base.sizes,
        tuple(ck[i] for i in order),
        tuple(uk[i] for i in order),
    )


@pytest.mark.parametrize(
    "dimension,ratio,statistic", tuple(product((0, 1), (0.4, 2.0), ("effect", "uplift")))
)
def test_affine_realized_target_and_target_noise(dimension, ratio, statistic):
    from increment.simulate.cluster_dgp import reconstruct_affine_point

    data = _affine_witness(dimension=dimension, ratio=ratio)
    selected = (
        np.asarray(data.table["x0"]) >= 0
        if dimension
        else np.asarray(data.cluster_index) // 2 % 2 == 0
    )
    point = reconstruct_affine_point(data, selected, statistic=statistic)
    assert point.bias is not None
    h, ell = np.asarray(point.target_coefficients), np.asarray(point.coefficients)
    assert point.target == pytest.approx(h @ (np.asarray(data.y1) - np.asarray(data.y0)))
    if statistic == "effect":
        assert point.target != pytest.approx(h @ data.conditional_tau)
    d = (np.asarray(data.table["group_id"]) == "treatment").astype(float)
    x = np.asarray(data.table["x0"]) if dimension else np.zeros(len(data.rows))
    mu = 1 + 0.8 * np.asarray(data.table["z"]) + 0.6 * x + d * np.asarray(data.conditional_tau)
    assert point.bias == pytest.approx(ell @ mu - h @ data.conditional_tau)
    di = data.scenario.outcome_scale * (ell * (1 + d * (ratio - 1)) - h * (ratio - 1))
    terms = dict(point.innovation_coefficients)
    assert data.innovation_unit is not None and data.innovation_cluster is not None
    for key, coefficient in zip(data.innovation_unit, di, strict=True):
        assert terms[f"unit:{key}"] == pytest.approx(math.sqrt(1 - data.scenario.icc) * coefficient)
    noise_error = 0.0
    for key, coefficient in terms.items():
        address = key.split("/")
        if key.startswith("cluster:"):
            innovation = (int(address[-1]) // 2 % 5) / 5
        else:
            innovation = (((int(address[-2]) // 2 + 3) * (int(address[-1]) + 1)) % 17 - 8) / 5
        noise_error += coefficient * innovation
    assert point.error == pytest.approx(point.bias + noise_error, abs=1e-12)


@pytest.mark.parametrize("statistic", ("effect", "uplift"))
def test_affine_clone_parent_aggregation_and_order(statistic):
    from increment.simulate.cluster_dgp import reconstruct_affine_point

    points = []
    for clones, reverse in ((1, False), (3, False), (3, True)):
        data = _affine_witness(clones=clones, reverse=reverse)
        selected = np.asarray(data.table["x0"]) >= 0
        points.append(reconstruct_affine_point(data, selected, statistic=statistic))
    original = points[0]
    assert any("training/" in key for key, _ in original.innovation_coefficients)
    assert any("holdout/" in key for key, _ in original.innovation_coefficients)
    for point in points[1:]:
        assert tuple(dict(point.innovation_coefficients)) == tuple(
            dict(original.innovation_coefficients)
        )
        np.testing.assert_allclose(
            list(dict(point.innovation_coefficients).values()),
            list(dict(original.innovation_coefficients).values()),
            atol=1e-15,
        )
        assert point.q_h == pytest.approx(original.q_h)
        assert point.m4_h == pytest.approx(original.m4_h)
        assert point.error == pytest.approx(original.error)
    missing = replace(data, innovation_unit=None)
    with pytest.raises(InvalidRequestError) as caught:
        reconstruct_affine_point(missing, selected, statistic=statistic)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"
    assert "reason" in caught.value.context


@pytest.mark.slow
@pytest.mark.parametrize("fraction", (0.0, 0.5, 1.0))
def test_affine_public_reporting_actual_fractions_and_mutation(fraction):
    from increment.simulate.cluster_dgp import check_policy_reconstruction

    data = _affine_witness()
    source = clustered_source(data)
    options: dict[str, Any] = {
        "control": "control",
        "interact": ["x0"],
        "cluster_weight": "equal",
        "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
        "include_evaluation_population": True,
    }
    rules = (
        targeting_rule(source, "y", fraction=fraction, **options),
        select_targeting_rule(
            source, "y", fractions=(fraction,), n_folds=2, seed=12, **options
        ).rule,
    )
    for rule in rules:
        assert rule.validation.passed
        assert rule.fraction == fraction
        if fraction == 0.5:
            assert rule.achieved_fraction == pytest.approx(0.6)
        points = check_policy_reconstruction(rule, data)
        if fraction == 0:
            assert not points
            assert rule.unavailable_reason == "estimation.targeting.empty_group"
            assert rule.policy_value is rule.uplift_vs_average is None
            continue
        assert len(points) == 2
        if fraction == 1:
            assert points[1].reconstructed == points[1].target == points[1].roundoff_bound == 0
        for field, point in zip(("policy_value", "uplift_vs_average"), points, strict=True):
            public = getattr(rule, field)
            assert public is not None
            mutation = public.model_copy(update={"value": public.value + 0.01})
            with pytest.raises(InvalidRequestError) as caught:
                check_policy_reconstruction(rule.model_copy(update={field: mutation}), data)
            assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"
            assert point.roundoff_bound < 0.01
        assert data.innovation_cluster is not None and data.innovation_unit is not None
        reversed_data = replace(
            data,
            rows=data.rows[::-1],
            y0=data.y0[::-1],
            y1=data.y1[::-1],
            conditional_tau=data.conditional_tau[::-1],
            cluster_index=data.cluster_index[::-1],
            innovation_cluster=data.innovation_cluster[::-1],
            innovation_unit=data.innovation_unit[::-1],
        )
        assert check_policy_reconstruction(rule, reversed_data) == points


@pytest.mark.slow
def test_affine_retained_dr_predictions_and_partial_cluster_weights():
    import functools

    from increment.estimation.cate import Covariate
    from increment.estimation.targeting import _dr_psi, targeting_rule_arrays
    from increment.semantics.design import IdentificationGate
    from increment.simulate.cluster_dgp import _evaluation_roster, check_policy_reconstruction

    data = _affine_witness(observational=True)

    class Propensity:
        def fit(self, X, d):
            pass

        def predict(self, X):
            # Remove different member counts while retaining both assignment arms.
            return np.where(X[:, 0] < X[:, 1], 0.001, 0.5)

    class Outcome:
        def fit(self, X, d):
            self.mean = float(np.mean(d))

        def predict(self, X):
            return np.full(len(X), self.mean)

    table = data.table
    fn = functools.partial(
        _dr_psi,
        propensity_learner=Propensity,
        outcome_learner=Outcome,
        folds=2,
        seed=3,
        gate=IdentificationGate(overlap="trim"),
    )
    rule = targeting_rule_arrays(
        np.asarray(table["y"]),
        (np.asarray(table["group_id"]) == "treatment").astype(float),
        {"x0": np.asarray(table["x0"]), "z": np.asarray(table["z"])},
        np.asarray(table["unit_id"]),
        cluster_ids=np.asarray(table["cluster_id"]),
        cluster_weight="equal",
        interact=[Covariate(name="x0")],
        adjustment=("x0", "z"),
        psi_fn=fn,
        arm_summary="score",
        fraction=0.5,
        n_groups=2,
        bootstrap_seed=57,
        bootstrap_repetitions=99,
        include_evaluation_population=True,
    )
    snapshot = rule.validation.evaluation_population
    assert snapshot is not None and snapshot.retention == "overlap_trimmed"
    roster = _evaluation_roster(rule, data)
    assert roster.nuisances is not None
    assert len(roster.rows) < rule.validation.n_train + rule.validation.n_holdout
    assert roster.cluster_ids is not None
    _, counts = np.unique(roster.cluster_ids, return_counts=True)
    assert len(set(counts)) > 1
    assert tuple(roster.base_weights) != tuple(data.weights("equal")[list(roster.indices)])
    points = check_policy_reconstruction(rule, data)
    assert len(points) == 2
    d = (np.asarray(roster.table["group_id"]) == "treatment").astype(float)
    y = np.asarray(roster.table["y"])
    e, m1, m0 = (np.asarray(v) for v in roster.nuisances)
    scores = m1 - m0 + d * (y - m1) / e - (1 - d) * (y - m0) / (1 - e)
    selected, weights = np.asarray(roster.actions), np.asarray(roster.base_weights)
    expected = np.average(scores[selected], weights=weights[selected])
    assert points[0].reconstructed == pytest.approx(expected)
    assert points[1].reconstructed == pytest.approx(expected - np.average(scores, weights=weights))
    assert points[0].target == pytest.approx(
        np.average(np.asarray(roster.tau)[selected], weights=weights[selected])
    )


@pytest.mark.parametrize(
    "bad_actions,statistic",
    (([1] * 200, "effect"), ([True] * 200, "other")),
)
def test_affine_rejects_implicit_actions_and_unknown_statistic(bad_actions, statistic):
    from increment.simulate.cluster_dgp import reconstruct_affine_point

    with pytest.raises(InvalidRequestError) as caught:
        reconstruct_affine_point(_affine_witness(), bad_actions, statistic=statistic)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"


def test_affine_archived_witness_keeps_deterministic_reconstruction():
    from increment.simulate.cluster_dgp import reconstruct_affine_point

    scenario = ClusteredCATEScenario(
        witness="original_varying_noise",
        n_clusters=16,
        members_per_cluster=5,
        dimension=1,
    )
    data = simulate_clustered_cate(scenario, stream="training", replication=0)
    selected = np.asarray(data.cluster_index) >= 8
    point = reconstruct_affine_point(data, selected, statistic="effect")
    group = np.asarray(data.table["group_id"])
    y = np.asarray(data.table["y"])
    expected = (
        y[selected & (group == "treatment")].mean() - y[selected & (group == "control")].mean()
    )
    assert abs(expected - point.reconstructed) <= point.roundoff_bound
    assert point.target == pytest.approx(np.mean(np.asarray(data.tau)[selected]))
    assert point.bias is point.q_h is point.m4_h is None
    assert point.innovation_coefficients == ()


def test_affine_rank_predicate_studentizes_roots_and_hulls_delete_one_t():
    from fractions import Fraction

    from scipy.stats import t

    from increment.simulate._cluster_reconstruction import _cluster_rank_gate, _support

    def gate(samples, deleted=(1.0, 3.0), *, scale=1.0, scales=None, strata=(2,), alpha=0.6, **kw):
        scales = (None,) * len(samples) if scales is None else scales
        return _cluster_rank_gate(
            2.0, samples, alpha, scale=scale, scales=scales, deleted=deleted, strata=strata, **kw
        )

    # Roots (T*-T)/s = (9, 10, 11) all reach T/s = 2: no evidence, unlike mean centering.
    assert gate((11.0, 12.0, 13.0)) == (1.0, None, 1.0)
    # Roots (-1, 0.5, 1.5) leave the tail empty; the delete-one t tail at 2/1 (df 1) is smaller.
    assert gate((1.0, 2.5, 3.5)) == (0.25, None, 1.0)
    # Deleting to (-1, 5) gives s_J = 3, and the t tail at 2/3 then sets the p-value.
    p, reason, df = gate((1.0, 2.5, 3.5), deleted=(-1.0, 5.0))
    assert (p, reason, df) == (pytest.approx(t.sf(2 / 3, 1)), None, 1.0)
    # A replicate scale divides its own root: 0.5/0.25 = 2 now counts, giving 2/4.
    assert gate((1.0, 2.5, 3.5), scales=(None, 0.25, None)) == (0.5, None, 1.0)
    # Two draws round 1/3 upward so the Monte Carlo p-value never understates the tail.
    p, reason, df = gate((-5.0, -6.0), alpha=0.9)
    assert reason is None and Fraction(p) > Fraction(1, 3) and p == math.nextafter(1 / 3, 1)
    # The reference df is the smallest resampling stratum less one.
    assert gate((1.0, 2.5, 3.5), deleted=(1.0, 3.0) * 3, strata=(3, 3))[2] == 2.0
    # An unresolved tail keeps the hulled p-value, and a jackknife failure replaces it.
    p, reason, df = gate((1.0, 2.5, 3.5), deleted=(-1.0, 5.0), alpha=0.1)
    assert (p, reason, df) == (
        pytest.approx(t.sf(2 / 3, 1)),
        "estimation.targeting.bootstrap_tail_resolution",
        1.0,
    )
    assert gate((1.0, 2.5, 3.5), deleted=(1.0, 3.0, 2.0), strata=(1, 2), alpha=0.1) == (
        None,
        "estimation.targeting.insufficient_clusters",
        None,
    )
    assert gate((1.0, 2.5, 3.5), deleted=(1.0, None)) == (
        None,
        "estimation.targeting.jackknife_unavailable_replicate",
        None,
    )
    assert gate((1.0, 2.5, 3.5), deleted=(2.0, 2.0)) == (
        None,
        "estimation.targeting.degenerate_cluster_variance",
        None,
    )
    assert gate((1.0, 2.5, 3.5), scale=None) == (
        None,
        "estimation.targeting.degenerate_cluster_variance",
        1.0,
    )
    assert gate((12.0, 12.0, 12.0)) == (
        None,
        "estimation.targeting.bootstrap_zero_variance",
        1.0,
    )
    neighbor = math.nextafter(12.0, math.inf)
    assert gate((12.0, neighbor, 12.0)) == (1.0, None, 1.0)
    support = _support(np.array(["a", "b", "b"]), np.array([0, 1, 1]), False)
    assert support == "estimation.targeting.insufficient_arm_clusters"
    assert gate((11.0, 12.0, 13.0), reason=support) == (None, support, 1.0)
    assert gate((11.0, None, 13.0)) == (
        None,
        "estimation.targeting.bootstrap_unavailable_replicate",
        1.0,
    )
    assert _cluster_rank_gate(
        math.inf, (1.0, 2.0), 0.6, scale=1.0, scales=(None, None), deleted=(1.0, 3.0), strata=(2,)
    ) == (None, "estimation.targeting.nonfinite_statistic", None)


@pytest.mark.slow
@pytest.mark.parametrize(
    "heterogeneity,seed,alpha",
    [(0.1, 1, 0.015), (0.0, 0, 0.47), (0.1, 0, 0.05), (0.0, 5, 0.2), (0.3, 0, 0.015)],
)
def test_affine_reporting_gate_matches_public_bootstrap_t_and_jackknife(heterogeneity, seed, alpha):
    from increment.simulate._cluster_reconstruction import reconstruct_reporting
    from increment.simulate.cluster_dgp import _evaluation_roster, check_policy_reconstruction

    data = _affine_witness(heterogeneity=heterogeneity)
    rule = targeting_rule(
        clustered_source(data),
        "y",
        fraction=0.5,
        control="control",
        interact=["x0"],
        cluster_weight="equal",
        bootstrap=ClusterBootstrap(seed=seed, repetitions=99),
        include_evaluation_population=True,
        alpha=alpha,
    )
    public = rule.validation.autoc
    gate = reconstruct_reporting(rule, _evaluation_roster(rule, data))
    assert gate.rank_reason == public.unavailable_reason
    assert gate.reference_df == public.reference_df
    assert gate.rank_passed == rule.validation.passed
    assert public.p_value is not None and gate.p_value is not None
    # Both gates evaluate max(p_B, S_df(T/s_J)) from the same rows in another
    # summation order. p_B is an exact rational; delete-one differences cancel
    # all but O(1/K) of each statistic, so T/s_J carries relative rounding of
    # order K*gamma_n, and log S_df has slope below df in log x.
    assert rule.validation.evaluation_population is not None
    assert public.n_clusters is not None and public.reference_df is not None
    gamma = len(rule.validation.evaluation_population.unit_ids) * np.finfo(float).eps
    assert gate.p_value == pytest.approx(
        public.p_value, rel=public.reference_df * public.n_clusters * gamma
    )
    check_policy_reconstruction(rule, data)


def test_affine_unclustered_gate_uses_discrete_tied_rank_influence():
    from scipy.stats import norm

    from increment.simulate._cluster_reconstruction import _unclustered_rank_gate

    score, psi = np.array([1.0, 1.0, 0.0]), np.array([3.0, 2.0, 0.0])
    phi = np.array([1.0, 2 / 3, 0.0])
    estimate, p = _unclustered_rank_gate(score, psi)
    assert estimate == pytest.approx(5 / 9)
    assert p == pytest.approx(norm.sf(phi.mean() / (phi.std(ddof=1) / math.sqrt(3))))
    assert _unclustered_rank_gate(score, np.zeros(3)) == (0, 1)


def test_affine_bootstrap_occurrences_recenter_ipw_with_each_draw():
    from types import SimpleNamespace

    from increment.simulate._cluster_reconstruction import reconstruct_reporting
    from increment.simulate.cluster_dgp import _EvaluationRoster

    data = _affine_witness()
    y0 = (0.0,) * len(data.rows)
    y1 = tuple(4.0 if g < 2 else 0.0 for g in data.cluster_index)
    rows = tuple(
        (*row[:3], y1[i] if row[2] == "treatment" else 0.0, *row[4:])
        for i, row in enumerate(data.rows)
    )
    data = replace(data, rows=rows, y0=y0, y1=y1, conditional_tau=y1)
    roster = _EvaluationRoster(
        data,
        tuple(range(20)),
        (True,) * 10 + (False,) * 10,
        (0.2,) * 20,
        None,
        tuple(f"g{g}" for g in range(4) for _ in range(5)),
        (1.0,) * 10 + (0.0,) * 10,
    )
    specification = SimpleNamespace(
        validation=SimpleNamespace(alpha=0.05, bootstrap_seed=57, bootstrap_repetitions=99),
        fraction=0.5,
        cluster_weight="equal",
    )
    gate = reconstruct_reporting(cast(Any, specification), roster)
    unequal_top_masses = set()
    for sources, statistic in zip(gate.bootstrap_sources, gate.bootstrap_statistics, strict=True):
        c, t, c0, t0 = (sources.count(f"g{g}") for g in range(4))
        # Two controls and two treatments are drawn; repeats remain occurrences.
        assert c + c0 == t + t0 == 2
        top = c + t
        unequal_top_masses.add(top)
        if top in (0, 4):
            expected = 0.0
        else:
            top_psi = (t * 2 * (4 - t) + c * 2 * t) / top
            bottom_psi = (c0 - t0) * 2 * t / (4 - top)
            share = top / 4
            expected = -share * math.log(share) * (top_psi - bottom_psi)
        assert statistic == pytest.approx(expected, abs=1e-14)
    assert {1, 3} <= unequal_top_masses


@pytest.mark.slow
def test_affine_campaign_rejects_additive_public_point_mutation(monkeypatch):
    import increment.cate as public

    data = _affine_witness()
    cell = ClusteredCell(
        "affine-mutation",
        data.scenario,
        "member_count",
        _statistics(data.scenario, "member_count"),
        "calibration",
    )
    registration = ClusteredRegistration((cell,), 2, 99, 57, 12)
    original = public.targeting_rule

    def mutated(*args, **kwargs):
        try:
            rule = original(*args, **kwargs)
        except CodedError as exc:
            pytest.fail(f"Campaign fixture refused before point mutation: {exc}")
        assert rule.policy_value is not None
        return rule.model_copy(
            update={
                "policy_value": rule.policy_value.model_copy(
                    update={"value": rule.policy_value.value + 0.01}
                ),
            }
        )

    monkeypatch.setattr(public, "targeting_rule", mutated)
    monkeypatch.setattr(
        "increment.simulate.cluster_dgp.honest_clustered_population",
        lambda *args, **kwargs: data,
    )
    with pytest.raises(InvalidRequestError) as caught:
        evaluate_clustered_replication(registration, cell.name, 0)
    assert caught.value.code == "simulate.cluster_dgp.invalid_scenario"
