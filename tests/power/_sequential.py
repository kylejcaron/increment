"""Drive the sequential minimum-detectable-effect solver below the public
API, from the same public inputs the solvers accept."""

from __future__ import annotations

from typing import cast

from increment.estimation.arm_contract import RelativeDecisionPolicy
from increment.power._search import _MdeRefusal
from increment.power.core import (
    _ArmPlan,
    _assigned_minimum_per_arm,
    _bounded_preflight,
    _compute_arms,
    _MdeSearch,
    _planned_looks,
    _prepare_solver,
    _sequential_side,
)
from increment.power.sequential import _SequentialMde, _SequentialMdeSearch


def sequential_search(
    n_per_arm, baseline, procedure, design, planned_looks
) -> _SequentialMdeSearch:
    procedure, baseline = _prepare_solver(procedure, baseline)
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    looks = _planned_looks(procedure, planned_looks)
    assert looks is not None
    bounded = _bounded_preflight(procedure, baseline, null_lift=decision.null_lift)
    n_T, n_C = _compute_arms(
        n_per_arm, design, minimum_per_arm=_assigned_minimum_per_arm(procedure, baseline)
    )
    plan = _ArmPlan(
        procedure, baseline, n_T * baseline.trigger_rate, n_C * baseline.trigger_rate, None, bounded
    )
    alpha_seq, exit_side, e_value_dual = _sequential_side(procedure)
    search = _MdeSearch.build(
        plan,
        target=design.power,
        null_lift=decision.null_lift,
        alternative=decision.alternative,
    )
    return _SequentialMdeSearch.build(
        looks,
        search,
        alpha_seq=alpha_seq,
        exit_side=exit_side,
        e_value_dual=e_value_dual,
    )


def sequential_solve(
    n_per_arm, baseline, procedure, design, planned_looks
) -> _SequentialMde | _MdeRefusal:
    return sequential_search(n_per_arm, baseline, procedure, design, planned_looks).solve()


def assert_sequential_solved(outcome: _SequentialMde | _MdeRefusal) -> _SequentialMde:
    assert isinstance(outcome, _SequentialMde)
    return outcome
