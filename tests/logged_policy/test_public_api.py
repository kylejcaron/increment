"""Root and results exports of the logged-policy family, end to end on two fixtures.

Fixture 1 (four fixed-random units) passes every trace and support gate and
is refused by the independent-unit floor; the shared forty-unit fixture
yields the exact contrast derived in ``tests/logged_policy/fixtures.py``.
Fixture 2 (three units logged under three policy versions) is refused as
adaptively logged.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import increment
from increment import LoggedTrace, PolicyRegistry, TabularPolicy, estimate_policy_contrast
from increment.errors import CapabilityError
from increment.logged_policy import REFERENCE_POLICY_V1, TARGET_POLICY_V1
from increment.results import PolicyValueContrast
from tests.logged_policy.fixtures import (
    FORTY_ESTIMATE,
    FORTY_LB,
    FORTY_P_VALUE,
    FORTY_SE,
    FORTY_TARGET_ESS_BY_TIME,
    FORTY_UB,
    FORTY_UNITS,
)

HOUR = timedelta(hours=1)


def rows(
    day: int, spec: list[tuple[str, int, str, float, str, int, float | None, float]], policy_id: str
) -> list[dict[str, Any]]:
    """``(unit, t, chosen, propensity, version, x, prior, reward)`` at 2h periods on ``day``."""
    base = datetime(2026, 1, day, tzinfo=UTC)
    batch = {"fixed-random": "fixed-b000"}
    out = []
    for unit, t, chosen, propensity, version, x, prior, reward in spec:
        start = base + (t - 1) * 2 * HOUR
        out.append(
            {
                "decision_time": start,
                "unit_id": unit,
                "decision_index": t,
                "candidate_actions": ("A", "B"),
                "chosen_action": chosen,
                "propensity": propensity,
                "logging_policy_id": policy_id,
                "logging_policy_version": version,
                "pre_decision_context": {"x": x, "prior": prior},
                "update_batch": batch.get(policy_id, f"eg-b00{version[1]}"),
                "reward_observation_boundary": start + HOUR,
                "reward": reward,
            }
        )
    return out


FIXTURE_1 = rows(
    1,
    [
        ("u01", 1, "A", 0.5, "v1", 0, None, 0.20),
        ("u02", 1, "B", 0.5, "v1", 1, None, 0.40),
        ("u03", 1, "B", 0.5, "v1", 0, None, 0.60),
        ("u04", 1, "A", 0.5, "v1", 1, None, 0.10),
        ("u01", 2, "B", 0.5, "v1", 1, 0.20, 0.30),
        ("u02", 2, "A", 0.5, "v1", 0, 0.40, 0.20),
        ("u03", 2, "A", 0.5, "v1", 1, 0.60, 0.50),
        ("u04", 2, "B", 0.5, "v1", 0, 0.10, 0.70),
    ],
    "fixed-random",
)
FIXTURE_2 = rows(
    2,
    [
        ("u01", 1, "A", 0.9, "v1", 0, None, 0.20),
        ("u02", 1, "B", 0.9, "v1", 1, None, 0.50),
        ("u03", 1, "B", 0.1, "v1", 0, None, 0.40),
        ("u01", 2, "B", 0.9, "v2", 1, 0.20, 0.60),
        ("u02", 2, "B", 0.9, "v2", 0, 0.50, 0.30),
        ("u03", 2, "A", 0.1, "v2", 1, 0.40, 0.20),
        ("u01", 3, "B", 0.9, "v3", 0, 0.60, 0.70),
        ("u02", 3, "A", 0.9, "v3", 1, 0.30, 0.10),
        ("u03", 3, "A", 0.1, "v3", 0, 0.20, 0.50),
    ],
    "epsilon-greedy",
)
REGISTRY = PolicyRegistry(
    [
        TabularPolicy(policy_id="fixed-random", version="v1", default={"A": 0.5, "B": 0.5}),
        TabularPolicy(
            policy_id="epsilon-greedy",
            version="v1",
            probabilities={0: {"A": 0.9, "B": 0.1}, 1: {"A": 0.1, "B": 0.9}},
        ),
        TabularPolicy(
            policy_id="epsilon-greedy",
            version="v2",
            probabilities={0: {"A": 0.1, "B": 0.9}, 1: {"A": 0.1, "B": 0.9}},
        ),
        TabularPolicy(
            policy_id="epsilon-greedy",
            version="v3",
            probabilities={0: {"A": 0.1, "B": 0.9}, 1: {"A": 0.9, "B": 0.1}},
        ),
    ]
)


def test_fixture1_passes_trace_and_support_gates_but_hits_the_unit_floor():
    trace = LoggedTrace.from_records(FIXTURE_1, registry=REGISTRY)
    assert trace.horizon == 2 and trace.n_units == 4
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.inference.cluster_floor"
    context = raised.value.context
    assert context["n_units"] == 4 and context["minimum_total_units"] == 20
    # target cumulative weights: t=1 (0.4, 0.4, 1.6, 1.6), t=2 (0.16, 0.16, 2.56, 2.56)
    assert context["ess_by_time"] == pytest.approx((50 / 17, 578 / 257), abs=1e-12)
    assert context["max_weight_by_time"] == pytest.approx((1.6, 2.56), abs=1e-12)
    assert context["min_propensity_by_time"] == (0.5, 0.5)


def test_forty_unit_fixture_yields_the_exact_contrast_through_the_root_export():
    trace = LoggedTrace.from_records(FORTY_UNITS, registry=REGISTRY)
    result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert isinstance(result, PolicyValueContrast)
    assert result.estimate == pytest.approx(FORTY_ESTIMATE, abs=1e-12)
    assert result.se == pytest.approx(FORTY_SE, abs=1e-12)
    assert (result.lb, result.ub) == pytest.approx((FORTY_LB, FORTY_UB), abs=1e-12)
    assert result.p_value == pytest.approx(FORTY_P_VALUE, abs=1e-12)
    assert result.reference_df == 39 and result.n_units == 40 and result.horizon == 2
    assert result.ess_by_time == pytest.approx(FORTY_TARGET_ESS_BY_TIME, abs=1e-12)
    assert result.logging_policy_versions == ("fixed-random/v1",)


def test_fixture2_is_refused_as_adaptively_logged_from_the_root_export():
    trace = LoggedTrace.from_records(FIXTURE_2, registry=REGISTRY)
    versions = ("epsilon-greedy/v1", "epsilon-greedy/v2", "epsilon-greedy/v3")
    assert trace.logging_policy_versions == versions
    with pytest.raises(CapabilityError) as raised:
        estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
    assert raised.value.code == "logged_policy.inference.adaptive_logging_unsupported"
    assert raised.value.context["logging_policy_versions"] == versions


def test_root_and_results_exports_resolve_lazily_to_the_family_modules():
    from increment import logged_policy

    assert increment.estimate_policy_contrast is logged_policy.estimate_policy_contrast
    assert increment.LoggedTrace is logged_policy.LoggedTrace
    assert increment.PolicyRegistry is logged_policy.PolicyRegistry
    assert increment.TabularPolicy is logged_policy.TabularPolicy
    assert PolicyValueContrast is logged_policy.PolicyValueContrast
    for name in ("estimate_policy_contrast", "LoggedTrace", "PolicyRegistry", "TabularPolicy"):
        assert name in increment.__all__
    assert "PolicyValueContrast" not in increment.__all__


def test_builtin_integer_keyed_policies_persist_in_python_mode_only():
    restored = pickle.loads(pickle.dumps(TARGET_POLICY_V1))
    assert restored.probability("B", {"x": 0}) == TARGET_POLICY_V1.probability("B", {"x": 0})
    assert restored.probability("B", {"x": 1}) == TARGET_POLICY_V1.probability("B", {"x": 1})
    with pytest.raises(CapabilityError) as raised:
        TARGET_POLICY_V1.model_dump_json()
    assert raised.value.code == "logged_policy.policy.json_context_key"
    json.loads(REFERENCE_POLICY_V1.model_dump_json(warnings="error"))
