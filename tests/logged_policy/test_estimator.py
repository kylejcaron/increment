"""Trajectory Hajek IPW contrast: exact hand fixtures, the T=1 IPW identity, and gates.

Exact numbers below were derived by rational arithmetic and are asserted to
float precision; the forty-unit fixture lives in ``tests/logged_policy/fixtures.py``.
"""

from __future__ import annotations

import math
import pickle
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from scipy.stats import t as t_dist

from increment.errors import CapabilityError, InvalidRequestError
from increment.logged_policy import (
    ESS_FLOOR,
    INDEPENDENT_UNIT_FLOOR,
    REFERENCE_POLICY_V1,
    TARGET_POLICY_V1,
    LoggedTrace,
    PolicyRegistry,
    PolicyValueContrast,
    TabularPolicy,
    estimate_policy_contrast,
)
from increment.logged_policy.estimator import _mean_contrast_over_time
from tests.logged_policy.fixtures import (
    FORTY_ESTIMATE,
    FORTY_LB,
    FORTY_MAX_WEIGHT_BY_TIME,
    FORTY_P_VALUE,
    FORTY_REFERENCE_VALUE_BY_TIME,
    FORTY_SE,
    FORTY_TARGET_ESS_BY_TIME,
    FORTY_TARGET_VALUE_BY_TIME,
    FORTY_UB,
    FORTY_UNITS,
    HOUR,
    REGISTRY,
    fixed_rows,
)

ALWAYS_B = TabularPolicy(policy_id="always", version="B", default={"A": 0.0, "B": 1.0})
ALWAYS_A = TabularPolicy(policy_id="always", version="A", default={"A": 1.0, "B": 0.0})


def _trace_for_route(rows, registry, admission):
    trace = LoggedTrace.from_records(rows, registry=registry)
    if admission == "registry":
        return trace
    return LoggedTrace.from_records(
        trace.records,
        logging_distributions=[
            dict(zip(record.candidate_actions, law, strict=True))
            for record, law in zip(trace.records, trace.logging_distributions, strict=True)
        ],
        horizon=trace.horizon,
    )


# Three units, T=2: refused by the independent-unit floor with its diagnostics.
THREE_UNITS = fixed_rows(
    [
        ("v1", 1, 0, "B", 0.5),
        ("v1", 2, 0, "A", 0.3),
        ("v2", 1, 0, "A", 0.2),
        ("v2", 2, 0, "B", 0.6),
        ("v3", 1, 1, "A", 0.4),
        ("v3", 2, 1, "B", 0.1),
    ]
)


def test_three_unit_fixture_reports_exact_diagnostics_in_the_cluster_floor_refusal():
    trace = LoggedTrace.from_records(THREE_UNITS, registry=REGISTRY)
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.inference.cluster_floor"
    context = raised.value.context
    assert context["n_units"] == 3
    assert context["minimum_total_units"] == INDEPENDENT_UNIT_FLOOR.minimum_total_units == 20
    # target ratios: t=1 (1.6, 0.4, 1.6); t=2 products (0.64, 0.64, 0.64)
    # ESS_1 = 3.6^2 / 5.28 = 27/11; ESS_2 = 1.92^2 / (3 * 0.4096) = 3 exactly
    assert context["ess_by_time"] == pytest.approx((27 / 11, 3.0), rel=0, abs=1e-12)
    # t=1 max over policies max(1.6, 1.0); t=2 max(0.64, 1.0)
    assert context["max_weight_by_time"] == pytest.approx((1.6, 1.0), rel=0, abs=1e-12)
    assert context["min_propensity_by_time"] == (0.5, 0.5)


@pytest.mark.parametrize("admission", ["registry", "records", "frame"])
def test_forty_unit_fixture_matches_the_exact_rational_solution(admission):
    rows = list(reversed(FORTY_UNITS))
    if admission == "registry":
        trace = LoggedTrace.from_records(rows, registry=REGISTRY)
    elif admission == "records":
        trace = LoggedTrace.from_records(
            rows, logging_distributions=[{"A": 0.5, "B": 0.5} for _ in rows]
        )
    else:
        import polars as pl

        frame_rows = []
        for row in rows:
            flat = dict(row)
            flat.update(flat.pop("pre_decision_context"))
            flat["law"] = {"A": 0.5, "B": 0.5}
            frame_rows.append(flat)
        trace = LoggedTrace.from_frame(
            pl.DataFrame(frame_rows),
            logging_distribution_column="law",
            context_columns=("x", "prior"),
        )
    assert trace.records == LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY).records
    result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert isinstance(result, PolicyValueContrast)
    assert result.target_value_by_time == pytest.approx(FORTY_TARGET_VALUE_BY_TIME, abs=1e-12)
    assert result.reference_value_by_time == pytest.approx(FORTY_REFERENCE_VALUE_BY_TIME, abs=1e-12)
    assert result.estimate == pytest.approx(FORTY_ESTIMATE, abs=1e-12)
    assert result.target_ess_by_time == pytest.approx(FORTY_TARGET_ESS_BY_TIME, abs=1e-12)
    assert result.reference_ess_by_time == (40.0, 40.0)
    assert result.ess_by_time == pytest.approx(FORTY_TARGET_ESS_BY_TIME, abs=1e-12)
    assert result.max_weight_by_time == pytest.approx(FORTY_MAX_WEIGHT_BY_TIME, abs=1e-12)
    assert result.min_propensity_by_time == (0.5, 0.5)
    assert result.min_policy_probability_by_time == (0.2, 0.2)
    assert result.se == pytest.approx(FORTY_SE, abs=1e-12)
    q = t_dist.isf(0.025, 39)
    assert result.reference_df == 39 and result.reference_kind == "t"
    assert result.lb == pytest.approx(result.estimate - q * result.se, abs=1e-12)
    assert result.ub == pytest.approx(result.estimate + q * result.se, abs=1e-12)
    assert result.lb == pytest.approx(FORTY_LB, abs=1e-12)
    assert result.ub == pytest.approx(FORTY_UB, abs=1e-12)
    assert result.p_value == pytest.approx(FORTY_P_VALUE, abs=1e-12)
    assert (result.horizon, result.n_units, result.alpha, result.null) == (2, 40, 0.05, 0.0)
    assert (result.target_id, result.target_version) == ("target-policy", "v1")
    assert (result.reference_id, result.reference_version) == ("reference-policy", "v1")
    assert result.logging_policy_versions == ("fixed-random/v1",)
    assert result.update_batches == ("fixed-b000",)
    assert (result.estimand, result.estimator) == ("delta_T", "trajectory_hajek_ipw")


def test_unit_influence_contributions_reproduce_the_reported_se():
    """Recompute phi_i by hand from the definition and match ``se`` exactly."""
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    n, horizon = 40, 2
    y = np.array([r.reward for r in trace.records]).reshape(n, horizon)
    phi = np.zeros(n)
    for policy, sign in ((TARGET_POLICY_V1, 1.0), (REFERENCE_POLICY_V1, -1.0)):
        ratio = np.array(
            [
                policy.probability(r.chosen_action, r.pre_decision_context) / 0.5
                for r in trace.records
            ]
        ).reshape(n, horizon)
        w = np.cumprod(ratio, axis=1)
        for t in range(horizon):
            v_t = (w[:, t] * y[:, t]).sum() / w[:, t].sum()
            phi += sign * w[:, t] * (y[:, t] - v_t) / w[:, t].mean() / horizon
    assert abs(phi.sum()) < 1e-12
    assert result.se == pytest.approx(math.sqrt((phi**2).sum() / (n * (n - 1))), abs=1e-14)


def test_t1_contrast_of_deterministic_policies_is_the_ipw_difference_in_means():
    rng = np.random.default_rng(11)
    actions = rng.choice(["A", "B"], size=80)
    rewards = rng.random(80).round(3)
    rows = fixed_rows(
        [
            (f"u{i:02d}", 1, int(rng.integers(0, 2)), a, float(y))
            for i, (a, y) in enumerate(zip(actions, rewards, strict=True))
        ]
    )
    trace = LoggedTrace.from_records(rows, registry=REGISTRY)
    result = estimate_policy_contrast(trace, ALWAYS_B, ALWAYS_A)
    y_b = rewards[actions == "B"]
    y_a = rewards[actions == "A"]
    assert result.estimate == pytest.approx(y_b.mean() - y_a.mean(), abs=1e-12)
    # phi_i = (n/n_B) 1[a=B](y - mean_B) - (n/n_A) 1[a=A](y - mean_A)
    n, n_b, n_a = 80, y_b.size, y_a.size
    assert min(n_b, n_a) >= ESS_FLOOR
    s_b, s_a = ((y_b - y_b.mean()) ** 2).sum(), ((y_a - y_a.mean()) ** 2).sum()
    expected_se = math.sqrt(n / (n - 1) * (s_b / n_b**2 + s_a / n_a**2))
    assert result.se == pytest.approx(expected_se, abs=1e-12)
    assert result.ess_by_time == pytest.approx((min(n_b, n_a),), abs=1e-12)
    assert result.target_max_weight_by_time == (2.0,) and result.reference_max_weight_by_time == (
        2.0,
    )


def test_zero_sandwich_se_with_nonzero_contrast_has_point_interval_and_zero_p_value():
    rows = fixed_rows(
        [
            (f"u{i:02d}", 1, 0, "A" if i % 2 == 0 else "B", 0.0 if i % 2 == 0 else 1.0)
            for i in range(40)
        ]
    )
    trace = LoggedTrace.from_records(rows, registry=REGISTRY)
    result = estimate_policy_contrast(trace, ALWAYS_B, ALWAYS_A)
    assert result.estimate == 1.0
    assert result.se == 0.0
    assert (result.lb, result.ub, result.p_value) == (1.0, 1.0, 0.0)


@pytest.mark.parametrize("admission", ["registry", "recorded"])
def test_adaptively_logged_trace_is_refused_before_any_support_or_floor_gate(admission):
    """Three epsilon-greedy versions in one trace: refused as adaptively
    logged before the ESS floor (which this fixture would also trip) and the
    ten-unit floor (three units) are reached."""
    eg = {
        v: TabularPolicy(policy_id="epsilon-greedy", version=v, probabilities=p)
        for v, p in {
            "v1": {0: {"A": 0.9, "B": 0.1}, 1: {"A": 0.1, "B": 0.9}},
            "v2": {0: {"A": 0.1, "B": 0.9}, 1: {"A": 0.1, "B": 0.9}},
            "v3": {0: {"A": 0.1, "B": 0.9}, 1: {"A": 0.9, "B": 0.1}},
        }.items()
    }
    rows = []
    spec = [
        ("u01", 1, "A", 0.9, "v1", 0, None, 0.20),
        ("u02", 1, "B", 0.9, "v1", 1, None, 0.50),
        ("u03", 1, "B", 0.1, "v1", 0, None, 0.40),
        ("u01", 2, "B", 0.9, "v2", 1, 0.20, 0.60),
        ("u02", 2, "B", 0.9, "v2", 0, 0.50, 0.30),
        ("u03", 2, "A", 0.1, "v2", 1, 0.40, 0.20),
        ("u01", 3, "B", 0.9, "v3", 0, 0.60, 0.70),
        ("u02", 3, "A", 0.9, "v3", 1, 0.30, 0.10),
        ("u03", 3, "A", 0.1, "v3", 0, 0.20, 0.50),
    ]
    day = datetime(2026, 1, 2, tzinfo=UTC)
    for unit, t, action, propensity, version, x, prior, reward in spec:
        start = day + (t - 1) * 2 * HOUR
        rows.append(
            {
                "decision_time": start,
                "unit_id": unit,
                "decision_index": t,
                "candidate_actions": ("A", "B"),
                "chosen_action": action,
                "propensity": propensity,
                "logging_policy_id": "epsilon-greedy",
                "logging_policy_version": version,
                "pre_decision_context": {"x": x, "prior": prior},
                "update_batch": f"eg-b00{version[1]}",
                "reward_observation_boundary": start + HOUR,
                "reward": reward,
            }
        )
    trace = _trace_for_route(rows, PolicyRegistry(eg.values()), admission)
    versions = ("epsilon-greedy/v1", "epsilon-greedy/v2", "epsilon-greedy/v3")
    assert trace.logging_policy_versions == versions
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.inference.adaptive_logging_unsupported"
    assert raised.value.context == {
        "logging_policy_versions": versions,
        "logging_policy_keys": tuple(("epsilon-greedy", version) for version in ("v1", "v2", "v3")),
        "update_batches": ("eg-b001", "eg-b002", "eg-b003"),
    }
    clone = pickle.loads(pickle.dumps(raised.value))
    assert clone.code == raised.value.code
    assert dict(clone.context) == dict(raised.value.context)


@pytest.mark.parametrize("admission", ["registry", "recorded"])
def test_unit_floor_is_refused_before_the_ess_floor_it_implies(admission):
    """Three units under the fixed logger, one of which played ``B``: the
    always-``B`` target's weights are ``(2, 0, 0)`` and ``ESS_1 = 1``, but the
    unit count is the shortfall named, with the ESS carried as a diagnostic."""
    rows = fixed_rows([("v1", 1, 0, "B", 0.5), ("v2", 1, 0, "A", 0.2), ("v3", 1, 1, "A", 0.4)])
    trace = _trace_for_route(rows, REGISTRY, admission)
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, ALWAYS_B, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.inference.cluster_floor"
    assert raised.value.context["n_units"] == 3
    assert raised.value.context["ess_by_time"] == (1.0,)


@pytest.mark.parametrize("admission", ["registry", "recorded"])
def test_ess_floor_refuses_concentrated_weights_with_both_ess_tuples(admission):
    """Forty units, three of which played ``B``: the always-``B`` target's
    weights are ``2`` on those three and zero elsewhere, so ``ESS_1 = 3``."""
    rows = fixed_rows(
        [(f"u{i:02d}", 1, 0, "B" if i < 3 else "A", 0.1 * (i % 7)) for i in range(40)]
    )
    trace = _trace_for_route(rows, REGISTRY, admission)
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, ALWAYS_B, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.support.ess_floor"
    context = raised.value.context
    assert (context["policy_id"], context["version"], context["decision_index"]) == (
        "always",
        "B",
        1,
    )
    assert context["ess"] == 3.0
    assert context["floor"] == ESS_FLOOR == 20.0
    assert context["target_ess_by_time"] == (3.0,)
    assert context["reference_ess_by_time"] == (40.0,)


@pytest.mark.parametrize("admission", ["registry", "recorded"])
def test_positivity_refuses_an_action_the_logger_never_plays(admission):
    """With a binary candidate set a proper logging law always supports both
    actions, so positivity is exercised on a three-action set where the
    logger gives ``C`` exactly zero probability."""
    logger = TabularPolicy(
        policy_id="two-of-three", version="v1", default={"A": 0.5, "B": 0.5, "C": 0.0}
    )
    wants_c = TabularPolicy(
        policy_id="wants-c", version="v1", default={"A": 0.45, "B": 0.45, "C": 0.1}
    )
    rows = fixed_rows([(f"u{i:02d}", 1, 0, "A" if i % 2 else "B", 0.5) for i in range(24)])
    for row in rows:
        row.update(candidate_actions=("A", "B", "C"), logging_policy_id="two-of-three")
    trace = _trace_for_route(rows, PolicyRegistry([logger]), admission)
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, wants_c, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.support.action_unsupported"
    context = raised.value.context
    assert context["unit_id"] == "u00"
    assert context["decision_index"] == 1
    assert context["policy_id"] == "wants-c"
    assert context["action"] == "C"
    assert context["policy_probability"] == 0.1
    # The uniform reference never asks for C, so the same trace supports it.
    result = estimate_policy_contrast(trace, REFERENCE_POLICY_V1, REFERENCE_POLICY_V1)
    assert result.estimate == 0.0 and result.se == 0.0 and result.p_value == 1.0


def test_alpha_gate_and_cluster_floor_boundary():
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    for alpha in (0.0, 1.0, -0.1, math.nan, 1):
        with pytest.raises(InvalidRequestError) as raised:
            estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, alpha=alpha)
        assert raised.value.code == "logged_policy.inference.alpha"
    nineteen = LoggedTrace.from_records(FORTY_UNITS[:19] + FORTY_UNITS[40:59], registry=REGISTRY)
    assert nineteen.n_units == 19
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(nineteen, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.inference.cluster_floor"
    assert raised.value.context["n_units"] == 19


def test_extreme_alpha_uses_the_upper_tail_directly():
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    tight = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, alpha=1e-12)
    assert math.isfinite(tight.lb) and math.isfinite(tight.ub)
    assert tight.ub - tight.lb == pytest.approx(2 * t_dist.isf(5e-13, 39) * tight.se, rel=1e-12)


def test_policy_that_does_not_sum_to_one_is_refused_at_the_support_gate():
    class Broken:
        policy_id = "broken"
        version = "v1"

        def probability(self, action: str, context) -> float:
            return 0.7

    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, Broken(), REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.support.policy_not_normalized"


def shifted(rows: list[dict[str, Any]], *, offset: float = 0.0, scale: float = 1.0):
    return [{**row, "reward": offset + scale * row["reward"]} for row in rows]


# FORTY_UNITS with every reward rounded to a multiple of 1/8, so adding a
# power-of-two offset below 2**50 leaves each reward exactly representable.
DYADIC_FORTY_UNITS = [{**row, "reward": round(row["reward"] * 8) / 8} for row in FORTY_UNITS]


def test_large_common_reward_offset_leaves_the_contrast_and_se_unchanged():
    """Rewards ``2**49 + y`` (``ulp = 0.125``) are exact, but the products
    ``W_i Y_i`` with ``W_i in {1.6, 0.4, ...}`` are not: a raw weighted sum
    rounds the contrast away, centered aggregation keeps it to float precision."""
    offset = 2.0**49
    assert all(row["reward"] + offset - offset == row["reward"] for row in DYADIC_FORTY_UNITS)
    base = estimate_policy_contrast(
        LoggedTrace.from_records(DYADIC_FORTY_UNITS, registry=REGISTRY),
        TARGET_POLICY_V1,
        REFERENCE_POLICY_V1,
    )
    lifted = estimate_policy_contrast(
        LoggedTrace.from_records(shifted(DYADIC_FORTY_UNITS, offset=offset), registry=REGISTRY),
        TARGET_POLICY_V1,
        REFERENCE_POLICY_V1,
    )
    assert base.estimate != 0.0
    assert lifted.estimate == pytest.approx(base.estimate, abs=1e-12)
    assert lifted.se == pytest.approx(base.se, abs=1e-12)
    assert (lifted.lb, lifted.ub, lifted.p_value) == pytest.approx(
        (base.lb, base.ub, base.p_value), abs=1e-12
    )
    for a, b in (
        (lifted.target_value_by_time, base.target_value_by_time),
        (lifted.reference_value_by_time, base.reference_value_by_time),
    ):
        assert a == pytest.approx(tuple(v + offset for v in b), abs=math.ulp(offset))
    assert lifted.ess_by_time == base.ess_by_time


def test_rewards_near_float_max_give_the_scaled_exact_solution():
    """Every reward scaled by ``float_max / 4``: the raw sum ``sum_i W_i Y_i``
    overflows, but the contrast and its interval are representable and equal
    the unscaled solution times the scale."""
    scale = sys.float_info.max / 4
    trace = LoggedTrace.from_records(shifted(FORTY_UNITS, scale=scale), registry=REGISTRY)
    result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert result.estimate == pytest.approx(scale * FORTY_ESTIMATE, rel=1e-12)
    assert result.se == pytest.approx(scale * FORTY_SE, rel=1e-12)
    assert result.target_value_by_time == pytest.approx(
        tuple(scale * v for v in FORTY_TARGET_VALUE_BY_TIME), rel=1e-12
    )
    assert result.lb == pytest.approx(scale * FORTY_LB, rel=1e-12)
    assert result.ub == pytest.approx(scale * FORTY_UB, rel=1e-12)
    assert result.p_value == pytest.approx(FORTY_P_VALUE, abs=1e-12)
    assert result.target_ess_by_time == pytest.approx(FORTY_TARGET_ESS_BY_TIME, abs=1e-12)


def test_contrast_beyond_float_range_is_refused_not_returned_as_inf():
    """``A`` rewards at ``-float_max`` and ``B`` rewards at ``+float_max``:
    both policy values are representable but their difference is not."""
    top = sys.float_info.max
    rows = fixed_rows(
        [
            (f"u{i:02d}", 1, 0, "A" if i % 2 == 0 else "B", -top if i % 2 == 0 else top)
            for i in range(40)
        ]
    )
    trace = LoggedTrace.from_records(rows, registry=REGISTRY)
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, ALWAYS_B, ALWAYS_A)
    assert raised.value.code == "logged_policy.inference.unrepresentable"
    context = raised.value.context
    assert context["nonfinite"] == ("estimate",)
    assert (context["target_value"], context["reference_value"]) == (top, -top)
    assert context["estimate"] == math.inf and context["se"] == 0.0


def test_opposing_extreme_period_contrasts_with_a_representable_average_are_estimated():
    """Each period's contrast is beyond the float range with opposite signs
    while ``Delta_2`` and its SE are representable; the result must equal
    the unit-scale design times the scale."""
    top = sys.float_info.max
    # Units u00-u19 always play B, u20-u39 always play A; period 2 flips the sign
    # pattern with a different spread so the average contrast and SE are nonzero.
    b_rewards = ((0.9, 0.8, 0.7, 0.8, 0.9), (-0.7, -0.9, -0.8, -0.9, -0.7))
    a_rewards = ((-0.9, -0.8, -0.7, -0.8, -0.9), (0.9, 0.7, 0.8, 0.8, 0.9))
    design = [
        (
            f"u{i:02d}",
            t,
            0,
            "B" if i < 20 else "A",
            (b_rewards if i < 20 else a_rewards)[t - 1][i % 5],
        )
        for i in range(40)
        for t in (1, 2)
    ]
    base = estimate_policy_contrast(
        LoggedTrace.from_records(fixed_rows(design), registry=REGISTRY), ALWAYS_B, ALWAYS_A
    )
    assert base.estimate == pytest.approx(0.01, abs=1e-12) and base.se > 0.0
    scaled = estimate_policy_contrast(
        LoggedTrace.from_records(shifted(fixed_rows(design), scale=top), registry=REGISTRY),
        ALWAYS_B,
        ALWAYS_A,
    )
    contrasts = [
        v_t - v_r
        for v_t, v_r in zip(
            scaled.target_value_by_time, scaled.reference_value_by_time, strict=True
        )
    ]
    assert contrasts == [math.inf, -math.inf]
    assert scaled.estimate == pytest.approx(top * base.estimate, rel=1e-12)
    assert scaled.se == pytest.approx(top * base.se, rel=1e-12)
    assert (scaled.lb, scaled.ub) == pytest.approx((top * base.lb, top * base.ub), rel=1e-12)
    assert scaled.p_value == pytest.approx(base.p_value, abs=1e-12)
    assert scaled.target_value_by_time == pytest.approx(
        tuple(top * v for v in base.target_value_by_time), rel=1e-12
    )


def test_shared_near_max_period_does_not_erase_an_ordinary_contrast_elsewhere():
    """Period 1 puts both policies at the same value near ``float_max`` (with
    spread inside each group); period 2 carries an ordinary contrast of
    ``0.4``. ``Delta_2 = 0.2`` must survive, and the SE is the period-1
    spread, which dominates period 2 by ``1e308``."""
    top = sys.float_info.max
    period_one = (0.8, 0.8, 0.8, 1.0, 1.0)  # group mean 0.88, midpoint 0.9, both groups
    b_period_two = (0.5, 0.6, 0.7, 0.8, 0.9)  # mean 0.7
    a_period_two = (0.1, 0.2, 0.3, 0.4, 0.5)  # mean 0.3
    design = [
        (
            f"u{i:02d}",
            t,
            0,
            "B" if i < 20 else "A",
            top * period_one[i % 5]
            if t == 1
            else (b_period_two if i < 20 else a_period_two)[i % 5],
        )
        for i in range(40)
        for t in (1, 2)
    ]
    result = estimate_policy_contrast(
        LoggedTrace.from_records(fixed_rows(design), registry=REGISTRY), ALWAYS_B, ALWAYS_A
    )
    assert result.target_value_by_time[0] == result.reference_value_by_time[0]
    assert result.estimate == pytest.approx(0.2, rel=1e-12)
    # phi_i = (n / n_g) (y_i1 - 0.88) / 2 per group at the top scale; period 2 is
    # below float resolution next to it. sum phi^2 = 8 * (3 * 0.08^2 + 2 * 0.12^2).
    expected_se = top * math.sqrt(8 * (3 * 0.08**2 + 2 * 0.12**2) / (40 * 39))
    assert result.se == pytest.approx(expected_se, rel=1e-12)
    assert math.isfinite(result.lb) and math.isfinite(result.ub)


def test_adjacent_same_sign_period_values_keep_their_exact_difference():
    """``1.0`` against its predecessor float at every one of three periods:
    the mean contrast is exactly ``2**-53``. Scaling the two sides by ``1/T``
    before subtracting rounds that difference to zero."""
    below_one = math.nextafter(1.0, 0.0)
    assert 1.0 - below_one == 2.0**-53
    rows = np.array([[1.0] * 3, [below_one] * 3])
    assert _mean_contrast_over_time(rows, rows[::-1]).tolist() == [2.0**-53, -(2.0**-53)]


def test_nested_residual_between_cancelling_extremes_is_exact():
    """Terms ``[A, 1, -A, -1, 2**-54]`` (``A = 2**100``) cancel in two nested
    levels; the surviving ``2**-54`` must reach the mean exactly rather than
    being rounded away by the first level's magnitude."""
    big = 2.0**100
    values = np.array([big, 1.0, -big, -1.0, 2.0**-54])
    assert float(_mean_contrast_over_time(values, np.zeros(5))) == 2.0**-54 / 5
    assert float(_mean_contrast_over_time(np.zeros(5), values)) == -(2.0**-54) / 5


def test_same_sign_residual_far_below_the_larger_value_survives():
    """``1.0`` against ``2**-60`` in period 1 and ``-1.0`` against ``0`` in
    period 2: a float subtraction of the same-sign pair rounds to ``1.0`` and
    the mean collapses to zero; the exact mean is ``-2**-61``."""
    target = np.array([1.0, -1.0])
    reference = np.array([2.0**-60, 0.0])
    assert float(_mean_contrast_over_time(target, reference)) == -(2.0**-61)
    assert float(_mean_contrast_over_time(reference, target)) == 2.0**-61


def test_subnormal_period_values_are_not_flushed_before_the_mean():
    """The smallest positive float at three periods averages to itself, and
    three of them next to a cancelling ``+float_max / -float_max`` pair
    average to one; scaling such terms down first flushes them to zero."""
    smallest = math.nextafter(0.0, 1.0)
    assert smallest == 5e-324
    assert float(_mean_contrast_over_time(np.full(3, smallest), np.zeros(3))) == smallest
    top = sys.float_info.max
    values = np.array([top, -top, 3 * smallest])
    assert float(_mean_contrast_over_time(values, np.zeros(3))) == smallest
    assert float(_mean_contrast_over_time(np.zeros(3), values)) == -smallest


def test_input_record_order_does_not_change_the_result():
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    reversed_trace = LoggedTrace.from_records(FORTY_UNITS[::-1], registry=REGISTRY)
    interleaved = LoggedTrace.from_records(FORTY_UNITS[1::2] + FORTY_UNITS[::2], registry=REGISTRY)
    result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert estimate_policy_contrast(reversed_trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1) == result
    assert estimate_policy_contrast(interleaved, TARGET_POLICY_V1, REFERENCE_POLICY_V1) == result


def test_smallest_positive_alpha_is_refused_because_its_half_tail_underflows():
    """``alpha`` is in the request domain, so the shared tail helper owns the
    refusal: its half tail is exactly zero and no quantile exists for it."""
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    smallest = sys.float_info.min * sys.float_info.epsilon
    assert smallest > 0.0 and smallest / 2.0 == 0.0
    with pytest.raises(InvalidRequestError) as raised:
        estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, alpha=smallest)
    assert raised.value.code == "estimation.tails.unresolvable"
    assert raised.value.context["tail_or_alpha"] == 0.0


def test_alpha_far_below_float_epsilon_still_resolves_a_finite_interval():
    """The ``1e-300`` half tail on 39 degrees of freedom is about ``3e8``:
    the interval is reported with that critical value, not refused."""
    from increment.estimation._tails import student_t_isf

    critical = student_t_isf(0.5e-300, 39)
    assert math.isfinite(critical) and critical > 1e8
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, alpha=1e-300)
    assert result.ub - result.lb == pytest.approx(2 * critical * result.se, rel=1e-12)


def test_bounds_that_overflow_from_a_wide_tail_are_refused():
    """A representable contrast and SE whose ``t`` critical multiple overflows
    is the shared Wald helper's hazard, not a contrast-representation failure."""
    scale = sys.float_info.max / 4
    trace = LoggedTrace.from_records(shifted(FORTY_UNITS, scale=scale), registry=REGISTRY)
    with pytest.raises(InvalidRequestError) as raised:
        estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, alpha=1e-100)
    assert raised.value.code == "estimation.tails.unresolvable"
    assert raised.value.context["tail_or_alpha"] == pytest.approx(t_dist.isf(0.5e-100, 39))


def test_recorded_full_law_matches_registry_for_unsorted_records_and_frame():
    recorded = [{"A": 0.5, "B": 0.5} for _ in FORTY_UNITS]
    shuffled = list(reversed(FORTY_UNITS))
    shuffled_laws = list(reversed(recorded))
    registry_trace = LoggedTrace.from_records(shuffled, registry=REGISTRY)
    recorded_trace = LoggedTrace.from_records(
        shuffled,
        logging_distributions=shuffled_laws,
    )
    expected = estimate_policy_contrast(registry_trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    actual = estimate_policy_contrast(recorded_trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    for field in (
        "estimate",
        "se",
        "lb",
        "ub",
        "target_value_by_time",
        "reference_value_by_time",
        "target_ess_by_time",
        "reference_ess_by_time",
    ):
        assert getattr(actual, field) == pytest.approx(getattr(expected, field), abs=1e-12)

    import polars as pl

    frame_rows = []
    for row in shuffled:
        context = row["pre_decision_context"]
        frame_rows.append(
            {
                **row,
                "candidate_actions": "A,B",
                "x": context["x"],
                "prior": context["prior"],
                "logging_distribution": {"A": 0.5, "B": 0.5},
            }
        )
    frame_trace = LoggedTrace.from_frame(
        pl.DataFrame(frame_rows),
        logging_distribution_column="logging_distribution",
        context_columns=("x", "prior"),
    )
    frame_result = estimate_policy_contrast(frame_trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert frame_result.estimate == pytest.approx(expected.estimate, abs=1e-12)
    assert frame_result.target_ess_by_time == pytest.approx(expected.target_ess_by_time, abs=1e-12)
