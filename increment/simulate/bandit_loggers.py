"""Adaptive logging-policy simulators that emit exact-propensity decision traces.

Every logger is batched: its action distribution is a table over the binary
context ``x`` that is frozen between update batches, each batch registers a
new immutable policy version, and every record's ``propensity`` is the exact
table probability used to draw the action. Updates are fitted only from
rewards whose observation boundary closed before the next decision, and
nothing is clipped: a posterior that drives a propensity under the floor
refuses at record construction exactly as a real trace would.

Rewards are Bernoulli with a context-dependent mean and later contexts follow
the previous action, so the context path of a context-only policy is a
two-state Markov chain and its value has the closed form in
:func:`true_policy_value`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import betaln

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.logged_policy import (
    DecisionRecord,
    LoggedTrace,
    PolicyRegistry,
    StochasticPolicy,
    TabularPolicy,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "simulate.bandit_loggers.count_positive": "{name} must be an integer >= 1, got {value!r}",
        "simulate.bandit_loggers.epsilon": "epsilon must lie in (0, 1], got {epsilon!r}",
        "simulate.bandit_loggers.reward_mean_out_of_range": "reward mean for action={action!r}, x={x} is {mean!r}, outside [0, 1]; Bernoulli rewards need every cell mean inside the unit interval",
    },
)
_raise = raiser(_REFUSALS)

Logger = Literal["fixed_random", "epsilon_greedy", "thompson", "contextual_thompson"]
ACTIONS: tuple[str, str] = ("A", "B")
CONTEXTS: tuple[int, int] = (0, 1)
START = datetime(2026, 1, 1, tzinfo=UTC)
PERIOD = timedelta(hours=2)
REWARD_DELAY = timedelta(hours=1)

# (x, action) -> (successes, trials) over closed rewards; a table is a pure function of it.
type _Counts = Mapping[tuple[int, str], tuple[int, int]]
type _Table = dict[int, dict[str, float]]


class RewardModel(CodedModel, BaseModel):
    """Bernoulli reward with mean ``base[x] + sign[x] * effect * 1[action == "B"]``.

    ``sign`` is ``+1`` at ``x = 0`` and ``-1`` at ``x = 1``, so action ``B``
    helps in one context and hurts in the other; a context-aware target
    policy therefore beats the uniform reference by ``0.3 * effect`` while
    a context-blind logger sees the two actions as equivalent on average.

    The first context is ``Bernoulli(context_probability)``. Later contexts
    follow the previous action: ``P(x_t = 1 | a_{t-1}) = clip(0.5 + 0.5 *
    context_dependence * 1[a_{t-1} == "B"], 0.05, 0.95)``, so ``B`` pushes
    the next context toward ``x = 1`` and ``A`` leaves it uniform. With
    ``context_dependence = 0`` contexts are independent draws and every
    policy value is stationary in ``t``; above zero, later contexts depend
    on earlier actions, the context marginal under the target policy
    diverges from the logged one from ``t = 3`` on, and only the cumulative
    likelihood ratio identifies ``V_t`` for ``t >= 2``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_by_context: tuple[float, float] = (0.30, 0.50)
    effect: float = 0.1
    context_probability: float = Field(default=0.5, gt=0.0, lt=1.0)
    context_dependence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _means_in_unit_interval(self) -> RewardModel:
        for x in CONTEXTS:
            for action in ACTIONS:
                mean = self.mean(action, x)
                if not (math.isfinite(mean) and 0.0 <= mean <= 1.0):
                    _raise(
                        "simulate.bandit_loggers.reward_mean_out_of_range",
                        action=action,
                        x=x,
                        mean=mean,
                    )
        return self

    def mean(self, action: str, x: int) -> float:
        sign = 1.0 if x == 0 else -1.0
        return self.base_by_context[x] + sign * self.effect * (action == "B")

    def next_context_probability(self, action: str) -> float:
        """``P(x_{t+1} = 1 | a_t = action)``."""
        raw = 0.5 + 0.5 * self.context_dependence * (action == "B")
        return min(0.95, max(0.05, raw))

    def initial_context_distribution(self) -> tuple[float, float]:
        return (1.0 - self.context_probability, self.context_probability)


def true_policy_value(
    policy: StochasticPolicy, reward_model: RewardModel, *, horizon: int = 1
) -> float:
    """Exact ``V_T(pi)`` for a policy that reads only ``x``.

    The context path is a two-state Markov chain with kernel
    ``K[x, x'] = sum_a pi(a | x) P(x' | a)``; ``V_T = (1/T) sum_t sum_x
    P(x_t = x) sum_a pi(a | x) mu(a, x)`` with ``P(x_1)`` the initial
    distribution and ``P(x_{t+1}) = P(x_t) K``. When both
    ``context_dependence = 0`` and ``context_probability = 0.5``, every row of
    ``K`` equals the initial distribution and the value is horizon-invariant.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        _raise("simulate.bandit_loggers.count_positive", name="horizon", value=horizon)
    pi = {(x, a): policy.probability(a, {"x": x, "prior": None}) for x in CONTEXTS for a in ACTIONS}
    step_value = {
        x: math.fsum(pi[(x, a)] * reward_model.mean(a, x) for a in ACTIONS) for x in CONTEXTS
    }
    kernel = {
        x: math.fsum(pi[(x, a)] * reward_model.next_context_probability(a) for a in ACTIONS)
        for x in CONTEXTS
    }
    p_one = reward_model.initial_context_distribution()[1]
    values = []
    for _ in range(horizon):
        values.append((1.0 - p_one) * step_value[0] + p_one * step_value[1])
        p_one = (1.0 - p_one) * kernel[0] + p_one * kernel[1]
    return math.fsum(values) / horizon


@dataclass(frozen=True, slots=True)
class LoggedRun:
    """A simulated trace with the registry that reproduces its logging law."""

    trace: LoggedTrace
    registry: PolicyRegistry
    reward_model: RewardModel


def beta_exceedance(alpha_x: int, beta_x: int, alpha_y: int, beta_y: int) -> float:
    """``P(X > Y)`` for ``X ~ Beta(alpha_x, beta_x)`` and ``Y ~ Beta(alpha_y, beta_y)``.

    Exact finite sum for integer ``alpha_x``::

        sum_{i=0}^{alpha_x - 1} B(alpha_y + i, beta_x + beta_y)
                                / ((beta_x + i) B(1 + i, beta_x) B(alpha_y, beta_y))

    The terms span many orders of magnitude, so each is formed in log space
    with ``betaln`` and exponentiated once; ``fsum`` then adds them with a
    single rounding. Rounding can carry the sum a few ulps past one, which
    the final bound removes.
    """
    log_terms = [
        betaln(alpha_y + i, beta_x + beta_y)
        - math.log(beta_x + i)
        - betaln(1 + i, beta_x)
        - betaln(alpha_y, beta_y)
        for i in range(alpha_x)
    ]
    return float(min(1.0, math.fsum(math.exp(v) for v in log_terms)))


@dataclass
class _Learner:
    """Batched learner: one frozen table per version, refitted from the closed rewards.

    Decisions are time-major, so rewards arrive with non-decreasing
    observation boundaries and the open queue closes from its front.
    """

    policy_id: str
    batch_prefix: str
    table_from_counts: Callable[[_Counts], _Table]
    first_batch: int = 1
    versions: list[TabularPolicy] = field(default_factory=list)
    _open: deque[tuple[datetime, int, str, float]] = field(default_factory=deque)
    _counts: dict[tuple[int, str], tuple[int, int]] = field(
        default_factory=lambda: {(x, a): (0, 0) for x in CONTEXTS for a in ACTIONS}
    )

    def observe(self, boundary: datetime, x: int, action: str, reward: float) -> None:
        self._open.append((boundary, x, action, reward))

    def fit(self, before: datetime) -> tuple[TabularPolicy, str]:
        """Register the next version and its batch id, fitted from every reward closed before ``before``."""
        while self._open and self._open[0][0] < before:
            _, x, action, reward = self._open.popleft()
            successes, trials = self._counts[(x, action)]
            self._counts[(x, action)] = (successes + int(reward), trials + 1)
        number = len(self.versions) + 1
        policy = TabularPolicy(
            policy_id=self.policy_id,
            version=f"v{number}",
            probabilities=self.table_from_counts(self._counts),
        )
        self.versions.append(policy)
        return policy, f"{self.batch_prefix}{number - 1 + self.first_batch:03d}"


def _uniform_table(_: _Counts) -> _Table:
    return {x: {"A": 0.5, "B": 0.5} for x in CONTEXTS}


def _epsilon_greedy_table(epsilon: float) -> Callable[[_Counts], _Table]:
    explore, exploit = epsilon / 2.0, 1.0 - epsilon / 2.0

    def table(counts: _Counts) -> _Table:
        out: _Table = {}
        for x in CONTEXTS:
            # Beta(1, 1)-posterior means; ties go to A.
            means = {a: (counts[(x, a)][0] + 1) / (counts[(x, a)][1] + 2) for a in ACTIONS}
            greedy = "B" if means["B"] > means["A"] else "A"
            out[x] = {a: (exploit if a == greedy else explore) for a in ACTIONS}
        return out

    return table


def _thompson_table(*, contextual: bool) -> Callable[[_Counts], _Table]:
    def table(counts: _Counts) -> _Table:
        out: _Table = {}
        for x in CONTEXTS:
            pool = (x,) if contextual else CONTEXTS
            s = {a: sum(counts[(c, a)][0] for c in pool) for a in ACTIONS}
            n = {a: sum(counts[(c, a)][1] for c in pool) for a in ACTIONS}
            p_b = beta_exceedance(1 + s["B"], 1 + n["B"] - s["B"], 1 + s["A"], 1 + n["A"] - s["A"])
            out[x] = {"A": 1.0 - p_b, "B": p_b}
        return out

    return table


_LEARNERS: Mapping[Logger, Callable[[float], _Learner]] = {
    "fixed_random": lambda _: _Learner("fixed-random", "fixed-b", _uniform_table, first_batch=0),
    "epsilon_greedy": lambda epsilon: _Learner(
        "epsilon-greedy", "eg-b", _epsilon_greedy_table(epsilon)
    ),
    "thompson": lambda _: _Learner("thompson", "ts-b", _thompson_table(contextual=False)),
    "contextual_thompson": lambda _: _Learner(
        "contextual-thompson", "cts-b", _thompson_table(contextual=True)
    ),
}


def _validate(n_units: int, horizon: int, batch_size: int, epsilon: float) -> None:
    for name, value in (("n_units", n_units), ("horizon", horizon), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            _raise("simulate.bandit_loggers.count_positive", name=name, value=value)
    if not (math.isfinite(epsilon) and 0.0 < epsilon <= 1.0):
        _raise("simulate.bandit_loggers.epsilon", epsilon=epsilon)


def simulate_logged_run(
    logger: Logger,
    *,
    n_units: int,
    horizon: int,
    seed: int,
    epsilon: float = 0.2,
    batch_size: int = 50,
    reward_effect: float = 0.1,
    context_dependence: float = 0.0,
) -> LoggedRun:
    """Simulate a complete-horizon trace under one batched adaptive logger.

    Decisions run time-major: every unit decides at period ``t`` (at
    ``START + (t - 1) * PERIOD``) before any unit decides at ``t + 1``. A
    new policy version is fitted after every ``batch_size`` decisions from
    the rewards whose boundary (``decision_time + REWARD_DELAY``) closed
    before the next decision; an update after the final decision is skipped.
    ``fixed_random`` never updates and stays at ``fixed-random/v1``. The
    random stream draws, per decision, the context, the action, and the
    reward, in that order.
    """
    _validate(n_units, horizon, batch_size, epsilon)
    reward_model = RewardModel(effect=reward_effect, context_dependence=context_dependence)
    rng = np.random.default_rng(seed)
    learner = _LEARNERS[logger](epsilon)
    policy, batch = learner.fit(START)
    adaptive = logger != "fixed_random"
    records: list[DecisionRecord] = []
    prior_reward: list[float | None] = [None] * n_units
    p_context_one = [reward_model.context_probability] * n_units
    total = n_units * horizon
    unit_width = max(5, len(str(n_units)))
    for t in range(1, horizon + 1):
        decision_time = START + (t - 1) * PERIOD
        boundary = decision_time + REWARD_DELAY
        for i in range(n_units):
            x = int(rng.random() < p_context_one[i])
            context = {"x": x, "prior": prior_reward[i]}
            distribution = policy.distribution(context)
            action = "B" if rng.random() < distribution["B"] else "A"
            reward = float(rng.random() < reward_model.mean(action, x))
            records.append(
                DecisionRecord(
                    decision_time=decision_time,
                    unit_id=f"u{i + 1:0{unit_width}d}",
                    decision_index=t,
                    candidate_actions=ACTIONS,
                    chosen_action=action,
                    propensity=distribution[action],
                    logging_policy_id=policy.policy_id,
                    logging_policy_version=policy.version,
                    pre_decision_context=context,
                    update_batch=batch,
                    reward_observation_boundary=boundary,
                    reward=reward,
                )
            )
            learner.observe(boundary, x, action, reward)
            prior_reward[i] = reward
            p_context_one[i] = reward_model.next_context_probability(action)
            decided = len(records)
            if adaptive and decided % batch_size == 0 and decided < total:
                next_time = decision_time + PERIOD if i == n_units - 1 else decision_time
                policy, batch = learner.fit(next_time)
    registry = PolicyRegistry(learner.versions)
    trace = LoggedTrace.from_records(records, registry=registry, horizon=horizon)
    return LoggedRun(trace=trace, registry=registry, reward_model=reward_model)


def simulate_logged_trace(
    logger: Logger,
    *,
    n_units: int,
    horizon: int,
    seed: int,
    epsilon: float = 0.2,
    batch_size: int = 50,
    reward_effect: float = 0.1,
    context_dependence: float = 0.0,
) -> LoggedTrace:
    """The admitted trace from :func:`simulate_logged_run` with the same arguments."""
    return simulate_logged_run(
        logger,
        n_units=n_units,
        horizon=horizon,
        seed=seed,
        epsilon=epsilon,
        batch_size=batch_size,
        reward_effect=reward_effect,
        context_dependence=context_dependence,
    ).trace
