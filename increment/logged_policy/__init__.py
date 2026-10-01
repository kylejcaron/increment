"""Logged-policy evidence family: off-policy contrast of two pre-registered stochastic policies.

The estimand is the finite-horizon dynamic policy-value contrast ``Delta_T``
and the only estimator is the trajectory-level self-normalized IPW with
cumulative likelihood ratios and a unit-clustered sandwich interval. The
interval is asymptotic in the number of independent units and calibrated
only for a trace logged under one fixed policy version; a trace whose
logging policy was updated while it was collected is refused.
"""

from increment.logged_policy._refusals import LOGGED_POLICY_REFUSALS
from increment.logged_policy.estimator import (
    ESS_FLOOR,
    INDEPENDENT_UNIT_FLOOR,
    PolicyValueContrast,
    estimate_policy_contrast,
)
from increment.logged_policy.policy import (
    REFERENCE_POLICY_V1,
    TARGET_POLICY_V1,
    PolicyRegistry,
    StochasticPolicy,
    TabularPolicy,
)
from increment.logged_policy.trace import PROPENSITY_FLOOR, DecisionRecord, LoggedTrace

__all__ = [
    "ESS_FLOOR",
    "INDEPENDENT_UNIT_FLOOR",
    "LOGGED_POLICY_REFUSALS",
    "PROPENSITY_FLOOR",
    "REFERENCE_POLICY_V1",
    "TARGET_POLICY_V1",
    "DecisionRecord",
    "LoggedTrace",
    "PolicyRegistry",
    "PolicyValueContrast",
    "StochasticPolicy",
    "TabularPolicy",
    "estimate_policy_contrast",
]
