"""Hand fixtures under the fixed 0.5/0.5 logger and their exact rational solutions.

The forty-unit, two-period fixture balances the target's per-step likelihood
ratios (twenty rows at 1.6 and twenty at 0.4 in each period, ten of each
product at the second) so both policies clear the effective-sample-size
floor: target ESS ``(500/17, 17956/835)``, reference ESS ``(40, 40)``. Its
policy values and contrast were derived with exact fractions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from increment.logged_policy import PolicyRegistry, TabularPolicy

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
FIXED = TabularPolicy(policy_id="fixed-random", version="v1", default={"A": 0.5, "B": 0.5})
REGISTRY = PolicyRegistry([FIXED])


def fixed_rows(rows: list[tuple[str, int, int, str, float]]) -> list[dict[str, Any]]:
    """``(unit, t, x, action, reward)`` under the fixed 0.5/0.5 logger, 2h periods."""
    out = []
    prior: dict[str, float | None] = {}
    for unit, t, x, action, reward in rows:
        start = T0 + (t - 1) * 2 * HOUR
        out.append(
            {
                "decision_time": start,
                "unit_id": unit,
                "decision_index": t,
                "candidate_actions": ("A", "B"),
                "chosen_action": action,
                "propensity": 0.5,
                "logging_policy_id": "fixed-random",
                "logging_policy_version": "v1",
                "pre_decision_context": {"x": x, "prior": prior.get(unit)},
                "update_batch": "fixed-b000",
                "reward_observation_boundary": start + HOUR,
                "reward": reward,
            }
        )
        prior[unit] = reward
    return out


# (unit, x_1, a_1, y_1, x_2, a_2, y_2)
_FORTY = [
    ("u01", 0, "A", 0.2, 1, "B", 0.3),
    ("u02", 1, "B", 0.4, 0, "A", 0.2),
    ("u03", 0, "B", 0.6, 1, "A", 0.5),
    ("u04", 1, "A", 0.1, 0, "B", 0.7),
    ("u05", 0, "A", 0.3, 0, "B", 0.5),
    ("u06", 1, "B", 0.2, 1, "A", 0.4),
    ("u07", 0, "B", 0.8, 1, "B", 0.1),
    ("u08", 1, "A", 0.6, 0, "A", 0.3),
    ("u09", 0, "B", 0.4, 0, "B", 0.9),
    ("u10", 1, "A", 0.5, 1, "A", 0.2),
    ("u11", 1, "A", 0.25, 0, "B", 0.25),
    ("u12", 0, "B", 0.45, 1, "A", 0.15),
    ("u13", 1, "B", 0.65, 0, "A", 0.45),
    ("u14", 0, "A", 0.15, 1, "B", 0.65),
    ("u15", 1, "A", 0.35, 1, "B", 0.45),
    ("u16", 0, "B", 0.25, 0, "A", 0.35),
    ("u17", 1, "B", 0.85, 0, "B", 0.05),
    ("u18", 0, "A", 0.65, 1, "A", 0.25),
    ("u19", 1, "B", 0.45, 1, "B", 0.85),
    ("u20", 0, "A", 0.55, 0, "A", 0.15),
    ("u21", 0, "B", 0.3, 1, "A", 0.2),
    ("u22", 1, "A", 0.5, 0, "B", 0.1),
    ("u23", 0, "A", 0.7, 1, "B", 0.4),
    ("u24", 1, "B", 0.2, 0, "A", 0.6),
    ("u25", 0, "B", 0.4, 0, "A", 0.4),
    ("u26", 1, "A", 0.3, 1, "B", 0.3),
    ("u27", 0, "A", 0.85, 1, "A", 0.0),
    ("u28", 1, "B", 0.7, 0, "B", 0.2),
    ("u29", 0, "A", 0.5, 0, "A", 0.8),
    ("u30", 1, "B", 0.6, 1, "B", 0.1),
    ("u31", 1, "B", 0.35, 0, "A", 0.15),
    ("u32", 0, "A", 0.55, 1, "B", 0.05),
    ("u33", 1, "A", 0.75, 0, "B", 0.35),
    ("u34", 0, "B", 0.25, 1, "A", 0.55),
    ("u35", 1, "B", 0.45, 1, "A", 0.35),
    ("u36", 0, "A", 0.35, 0, "B", 0.25),
    ("u37", 1, "A", 0.95, 0, "A", 0.25),
    ("u38", 0, "B", 0.75, 1, "B", 0.15),
    ("u39", 1, "A", 0.55, 1, "A", 0.75),
    ("u40", 0, "B", 0.65, 0, "B", 0.05),
]
FORTY_UNITS = fixed_rows(
    [(u, 1, x1, a1, y1) for u, x1, a1, y1, _, _, _ in _FORTY]
    + [(u, 2, x2, a2, y2) for u, _, _, _, x2, a2, y2 in _FORTY]
)

# Exact solution for TARGET_POLICY_V1 against REFERENCE_POLICY_V1.
FORTY_TARGET_VALUE_BY_TIME = (969 / 2000, 971 / 2680)
FORTY_REFERENCE_VALUE_BY_TIME = (387 / 800, 137 / 400)
FORTY_ESTIMATE = 5511 / 536000
FORTY_TARGET_ESS_BY_TIME = (500 / 17, 17956 / 835)
FORTY_MAX_WEIGHT_BY_TIME = (1.6, 2.56)
# sqrt(169462815424917 / 419143316800000000), the CR1 sandwich on the exact influences.
FORTY_SE = 0.02010740084827828
FORTY_LB = -0.030389340703441493
FORTY_UB = 0.050952773539262385
FORTY_P_VALUE = 0.6119960680817847
