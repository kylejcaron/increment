"""Synthetic data generator and end-to-end ground-truth evaluation.

Usage
-----
    uv run python -m increment.simulate --replications 200
"""

from increment.simulate.bandit_loggers import (
    LoggedRun,
    RewardModel,
    simulate_logged_run,
    simulate_logged_trace,
    true_policy_value,
)
from increment.simulate.dgp import (
    Scenario,
    SwitchbackScenario,
    simulate_raw_logs,
    simulate_switchback_panel,
)
from increment.simulate.runner import (
    EvalResult,
    SwitchbackEvalResult,
    run_end_to_end,
    run_switchback_end_to_end,
)

__all__ = [
    "LoggedRun",
    "RewardModel",
    "simulate_logged_run",
    "simulate_logged_trace",
    "true_policy_value",
    "Scenario",
    "SwitchbackScenario",
    "simulate_raw_logs",
    "simulate_switchback_panel",
    "EvalResult",
    "SwitchbackEvalResult",
    "run_end_to_end",
    "run_switchback_end_to_end",
]
