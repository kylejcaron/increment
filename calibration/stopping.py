"""Stopping rules for calibration certification.

A calibration cell is a Bernoulli stream: every replication either misses
(the interval failed to cover, or the test erred) or it does not. The
campaign has to decide, per cell, between an acceptable true miss rate
``p <= q`` and an unacceptable one ``p >= q + delta``.

Two rules decide that from the *same* frozen design ``(q, delta, eta, n)``:

``FixedStopping``
    Draw exactly ``n`` replications, then accept iff the exact one-sided
    Clopper-Pearson upper bound on the miss rate is inside tolerance.
    The reported object is that bound.

``SequentialStopping`` (rule id ``sprt-v1``)
    Wald's sequential probability ratio test on the per-draw miss
    indicator. Its two error probabilities are *not* chosen freely: they
    are the fixed design's own exact error probabilities at its declared
    operating point ``k*``, so the sequential rule inherits the fixed
    rule's guarantee instead of inventing a new one. The reported object
    becomes a sequential decision rather than a confidence bound.

The design's operating point
----------------------------
The fixed design sizes ``n`` against one declared count,

    k* = ceil(n * (q + delta/2)),

the count at which the observed miss rate has consumed exactly half the
tolerance. ``k*`` is where the design's operating characteristic is
*stated*:

    alpha = P(K >  k* | p = q)          false failure of a nominal cell
    beta  = P(K <= k* | p = q + delta)  false acceptance at the tolerance edge

Both must fit the per-bound Monte-Carlo budget ``eta``, and for ``beta``
that requirement is exactly the requirement that the Clopper-Pearson
certificate still passes at ``k*``. ``CP_U(k, n, eta)`` is by construction
the rate ``u`` solving ``P_u(K <= k) = eta``, and ``P_p(K <= k)`` falls as
``p`` rises, so

    CP_U(k*, n, eta) <= q + delta   <=>   P_{q+delta}(K <= k*) <= eta,

i.e. the bound certifies at the operating point precisely when ``beta``
is inside budget. :meth:`FixedDesign.resolve` enforces the single
condition ``max(alpha, beta) <= eta`` and refuses any design that misses
it rather than silently re-pointing ``k*``.

The same duality fixes ``FixedStopping``'s acceptance region: the largest
count whose bound still certifies is the largest count whose
tolerance-edge acceptance probability is still within ``eta``. That edge
is exposed as ``certifiable_count`` and is the one boundary here that
moves with ``eta``. The sequential rule inherits ``(alpha, beta)`` as
declared at ``k*``, which is what the frozen design publishes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from tests.mc import binomial_error_upper_bound

# Rule ids implemented here. A profile that declares `stopping: sequential`
# must name one of these; the enumeration lives with the code that executes
# it so configuration can never advertise a procedure nothing can run.
SEQUENTIAL_RULES = frozenset({"sprt-v1"})

# Doubling bound for the Wald-exponent bracket. The root is |h| <= 1 for any
# rate between the two hypotheses and grows only for rates far outside them,
# so exceeding this means the increments are degenerate, not that the search
# needs more room.
_EXPONENT_LIMIT = 1e6

Decision = Literal["continue", "accept", "reject", "unresolved"]


class StoppingError(ValueError):
    """A stopping design is infeasible, or a rule was driven past its decision."""


def _certifiable_count(repetitions: int, limit: float, eta: float, *, known: int) -> int:
    """Largest miss count whose exact CP upper bound is still ``<= limit``.

    ``binomial_error_upper_bound`` is increasing in ``k``, so the acceptance
    region of a CP-gated fixed rule is the prefix ``k <= certifiable_count``
    and bisection finds its edge exactly. *known* is a count already shown to
    certify, which both seeds the search and rules out an empty region.
    """
    low, high = known, repetitions
    while low < high:
        middle = (low + high + 1) // 2
        if binomial_error_upper_bound(middle, repetitions, eta) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True, slots=True)
class FixedDesign:
    """A resolved fixed repetition design and the error rates it declares.

    Every field is derived from ``(nominal_error, tolerance, eta,
    repetitions)``; nothing here is tabulated.
    """

    nominal_error: float
    tolerance: float
    eta: float
    repetitions: int
    acceptance_count: int
    certifiable_count: int
    type_i_error: float
    type_ii_error: float

    @property
    def error_upper_limit(self) -> float:
        """The largest miss rate the certificate may report, ``q + delta``."""
        return self.nominal_error + self.tolerance

    def certifies(self, misses: int) -> bool:
        """Whether *misses* out of ``repetitions`` still certifies the cell."""
        if not 0 <= misses <= self.repetitions:
            raise StoppingError(f"misses must be in [0, {self.repetitions}], got {misses}")
        bound = binomial_error_upper_bound(misses, self.repetitions, self.eta)
        return bound <= self.error_upper_limit

    @classmethod
    def resolve(
        cls, *, nominal_error: float, tolerance: float, eta: float, repetitions: int
    ) -> FixedDesign:
        """Derive ``k*``, the certifiable boundary, and the exact error rates.

        Refuses any ``(q, delta, eta, n)`` whose operating point does not fit
        the Monte-Carlo budget. Both exact error probabilities at ``k*`` must
        satisfy ``<= eta``; the ``beta`` half of that is, by Clopper-Pearson
        duality, exactly the statement that the reported bound still
        certifies there.
        """
        from scipy.stats import binom

        if not 0.0 < nominal_error < 1.0:
            raise StoppingError(f"nominal_error must be in (0, 1), got {nominal_error}")
        if tolerance <= 0.0 or nominal_error + tolerance >= 1.0:
            raise StoppingError(f"tolerance must keep q + delta in (q, 1), got {tolerance}")
        if not 0.0 < eta < 1.0:
            raise StoppingError(f"eta must be in (0, 1), got {eta}")
        if repetitions < 1:
            raise StoppingError(f"repetitions must be >= 1, got {repetitions}")

        limit = nominal_error + tolerance
        acceptance_count = math.ceil(repetitions * (nominal_error + tolerance / 2))
        if acceptance_count > repetitions:
            raise StoppingError(
                f"operating point {acceptance_count} exceeds {repetitions} repetitions"
            )
        type_i_error = float(binom.sf(acceptance_count, repetitions, nominal_error))
        type_ii_error = float(binom.cdf(acceptance_count, repetitions, limit))
        if max(type_i_error, type_ii_error) > eta:
            raise StoppingError(
                f"operating point {acceptance_count} of {repetitions} misses the budget "
                f"{eta}: false failure {type_i_error}, tolerance-edge acceptance "
                f"{type_ii_error}"
            )
        return cls(
            nominal_error=nominal_error,
            tolerance=tolerance,
            eta=eta,
            repetitions=repetitions,
            acceptance_count=acceptance_count,
            certifiable_count=_certifiable_count(repetitions, limit, eta, known=acceptance_count),
            type_i_error=type_i_error,
            type_ii_error=type_ii_error,
        )

    def parameters(self) -> dict[str, Any]:
        """Resolved design numbers, for the evidence bundle."""
        return {
            "nominal_error": self.nominal_error,
            "tolerance": self.tolerance,
            "eta": self.eta,
            "repetitions": self.repetitions,
            "error_upper_limit": self.error_upper_limit,
            "acceptance_count": self.acceptance_count,
            "certifiable_count": self.certifiable_count,
            "type_i_error": self.type_i_error,
            "type_ii_error": self.type_ii_error,
        }


class StoppingRule(ABC):
    """One decision procedure over a stream of per-replication miss flags.

    Feed outcomes one at a time with :meth:`observe`; it returns
    ``continue`` until the rule resolves. The draw cap is enforced here
    rather than by each rule: once ``max_draws`` outcomes have been
    consumed without an early decision the rule is forced to settle, and a
    settled rule refuses further observations rather than drifting.
    """

    rule: str

    def __init__(self, design: FixedDesign, *, max_draws: int) -> None:
        if max_draws < 1:
            raise StoppingError(f"max_draws must be >= 1, got {max_draws}")
        self.design = design
        self.max_draws = max_draws
        self.draws = 0
        self.misses = 0
        self.decision: Decision = "continue"

    @property
    def resolved(self) -> bool:
        """Whether the rule has stopped."""
        return self.decision != "continue"

    def observe(self, *, miss: bool) -> Decision:
        """Record one replication outcome and return the current decision."""
        if self.resolved:
            raise StoppingError(f"{self.rule} already decided {self.decision!r}")
        self.draws += 1
        self.misses += int(miss)
        decision = self._step(miss=miss)
        if decision == "continue" and self.draws >= self.max_draws:
            decision = self._exhaust()
        self.decision = decision
        return decision

    def run(self, stream: Iterable[bool]) -> Decision:
        """Drive the rule over an iterable of miss flags until it resolves."""
        for miss in stream:
            if self.resolved:
                break
            self.observe(miss=bool(miss))
        return self.decision

    @abstractmethod
    def _step(self, *, miss: bool) -> Decision:
        """The rule's own verdict after one more outcome, or ``continue``."""

    @abstractmethod
    def _exhaust(self) -> Decision:
        """How the rule settles once the draw cap is reached."""

    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...


class FixedStopping(StoppingRule):
    """Draw exactly ``n``, then certify with the exact Clopper-Pearson bound."""

    rule = "fixed"

    def __init__(self, design: FixedDesign) -> None:
        super().__init__(design, max_draws=design.repetitions)

    def _step(self, *, miss: bool) -> Decision:
        del miss  # a fixed rule never decides early; only the final count counts
        return "continue"

    def _exhaust(self) -> Decision:
        return "accept" if self.design.certifies(self.misses) else "reject"

    def parameters(self) -> dict[str, Any]:
        """Resolved rule parameters, for the evidence bundle."""
        return {
            "rule": self.rule,
            "max_draws": self.max_draws,
            **self.design.parameters(),
        }


class SequentialStopping(StoppingRule):
    """Wald SPRT whose error probabilities are the fixed design's own.

    Hypotheses are the two rates the tolerance already names: ``H0: p = q``
    (acceptable) against ``H1: p = q + delta`` (the tolerance edge). The
    per-draw log-likelihood ratio takes one of two values,

        miss -> log(p1/p0),   hit -> log((1 - p1)/(1 - p0)),

    and the cumulative statistic is compared with Wald's boundaries formed
    from ``alpha = design.type_i_error`` and ``beta = design.type_ii_error``:

        accept below log(beta / (1 - alpha)),
        reject above log((1 - beta) / alpha).

    Wald's inequalities bound the realised error probabilities of the
    truncation-free test by ``alpha / (1 - beta)`` and ``beta / (1 - alpha)``.
    Both are reported: the design already holds ``alpha`` and ``beta`` inside
    its ``eta`` budget, and these say how much the sequential construction
    can inflate them.
    """

    rule = "sprt-v1"

    def __init__(self, design: FixedDesign) -> None:
        super().__init__(design, max_draws=design.repetitions)
        alpha = design.type_i_error
        beta = design.type_ii_error
        p0 = design.nominal_error
        p1 = design.error_upper_limit
        self.alpha = alpha
        self.beta = beta
        self.miss_increment = math.log(p1 / p0)
        self.hit_increment = math.log1p(-p1) - math.log1p(-p0)
        self.accept_boundary = math.log(beta) - math.log1p(-alpha)
        self.reject_boundary = math.log1p(-beta) - math.log(alpha)
        self.type_i_guarantee = alpha / (1.0 - beta)
        self.type_ii_guarantee = beta / (1.0 - alpha)
        self.statistic = 0.0

    @property
    def indifference_rate(self) -> float:
        """Miss rate at which the statistic has zero drift.

        ``p* = -hit_increment / (miss_increment - hit_increment)`` lies
        strictly between ``q`` and ``q + delta``; it is where the test is
        slowest and where the draw cap actually binds.
        """
        return -self.hit_increment / (self.miss_increment - self.hit_increment)

    def _step(self, *, miss: bool) -> Decision:
        self.statistic += self.miss_increment if miss else self.hit_increment
        if self.statistic <= self.accept_boundary:
            return "accept"
        if self.statistic >= self.reject_boundary:
            return "reject"
        return "continue"

    def _exhaust(self) -> Decision:
        # The cap is the fixed design's budget: a sequential rule may never
        # cost more than the rule it replaces. Spending it without crossing a
        # boundary is a real outcome, so say so instead of accepting by
        # default -- the campaign reports such a cell as uncertified.
        return "unresolved"

    def acceptance_probability(self, rate: float) -> float:
        """Wald's ``L(p)``: probability the test stops on the accept boundary.

        This is the rule's power curve, and it is where the inherited
        guarantee becomes checkable: ``L(q)`` is ``1 - alpha`` and
        ``L(q + delta)`` is ``beta``, the very numbers the fixed design
        declared at its operating point.
        """
        if not 0.0 <= rate <= 1.0:
            raise StoppingError(f"rate must be in [0, 1], got {rate}")
        if rate == 0.0:
            return 1.0
        if rate == 1.0:
            return 0.0
        a, b = self.accept_boundary, self.reject_boundary
        h = self._wald_exponent(rate)
        if h == 0.0:
            # Zero drift: the walk hits either boundary in proportion to its
            # distance from the other one.
            return b / (b - a)
        # Both branches divide through by the larger exponential so neither
        # boundary term can overflow for a rate far outside the hypotheses.
        if h > 0.0:
            return math.expm1(-h * b) / math.expm1(h * (a - b))
        return (math.exp(h * (b - a)) - math.exp(-h * a)) / math.expm1(h * (b - a))

    def _drift(self, rate: float) -> float:
        """``E_p[lambda]``, written so that it vanishes exactly at ``p*``."""
        return self.hit_increment + rate * (self.miss_increment - self.hit_increment)

    def _wald_exponent(self, rate: float) -> float:
        """Nonzero root ``h`` of ``E_p[exp(h * lambda)] = 1``, or 0 at ``p*``.

        The moment generating function of the log-likelihood increment is
        strictly convex with a root at ``h = 0``; its derivative there is the
        drift ``E_p[lambda]``, so the second root sits opposite the drift's
        sign. Bracket by doubling away from zero, then solve.
        """
        from scipy.optimize import brentq

        drift = self._drift(rate)
        if drift == 0.0:
            return 0.0

        def mgf(h: float) -> float:
            miss = rate * math.exp(h * self.miss_increment)
            hit = (1.0 - rate) * math.exp(h * self.hit_increment)
            return miss + hit - 1.0

        step = -math.copysign(1.0, drift)
        near, far = step * 1e-9, step
        while mgf(far) < 0.0:
            near, far = far, far * 2.0
            if abs(far) > _EXPONENT_LIMIT:
                raise StoppingError(f"no Wald exponent bracketed for rate {rate}")
        low, high = (near, far) if near < far else (far, near)
        return float(brentq(mgf, low, high, xtol=1e-14, rtol=1e-14))

    def expected_draws(self, rate: float) -> float:
        """Wald's average sample number at true miss rate *rate*.

        Untruncated: this is the expectation of the unbounded SPRT and takes
        no account of ``max_draws``, so it is an upper reference. At the
        indifference rate it legitimately exceeds the cap, which is exactly
        why the cap resolves to ``unresolved`` rather than to a decision.
        """
        if not 0.0 <= rate <= 1.0:
            raise StoppingError(f"rate must be in [0, 1], got {rate}")
        a, b = self.accept_boundary, self.reject_boundary
        drift = self._drift(rate)
        if drift == 0.0 or rate == self.indifference_rate:
            # Zero drift: the walk is driven by its variance alone, and the
            # expected first passage of a driftless walk across (a, b) is
            # -a*b / E[lambda^2].
            variance = rate * self.miss_increment**2 + (1.0 - rate) * self.hit_increment**2
            return -a * b / variance
        accept_probability = self.acceptance_probability(rate)
        return (accept_probability * a + (1.0 - accept_probability) * b) / drift

    def parameters(self) -> dict[str, Any]:
        """Resolved rule parameters, for the evidence bundle."""
        return {
            "rule": self.rule,
            "max_draws": self.max_draws,
            "alpha": self.alpha,
            "beta": self.beta,
            "type_i_guarantee": self.type_i_guarantee,
            "type_ii_guarantee": self.type_ii_guarantee,
            "accept_boundary": self.accept_boundary,
            "reject_boundary": self.reject_boundary,
            "miss_increment": self.miss_increment,
            "hit_increment": self.hit_increment,
            "indifference_rate": self.indifference_rate,
            "expected_draws_at_nominal": self.expected_draws(self.design.nominal_error),
            "expected_draws_at_tolerance": self.expected_draws(self.design.error_upper_limit),
            **self.design.parameters(),
        }


def build_rule(
    design: FixedDesign, *, stopping: str, sequential_rule: str | None = None
) -> StoppingRule:
    """Construct the rule a profile declares, or refuse the declaration."""
    if stopping == "fixed":
        if sequential_rule is not None:
            raise StoppingError("fixed stopping must not declare a sequential rule")
        return FixedStopping(design)
    if stopping == "sequential":
        if sequential_rule not in SEQUENTIAL_RULES:
            raise StoppingError(
                f"unknown sequential rule {sequential_rule!r}; "
                f"implemented: {sorted(SEQUENTIAL_RULES)}"
            )
        return SequentialStopping(design)
    raise StoppingError(f"unknown stopping rule {stopping!r}")
