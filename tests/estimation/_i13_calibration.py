"""Prospective accounting for the cluster-randomized calibration manifest.

A smoke run can never certify this manifest. It reserves .001 of the release's
shared .01 Monte Carlo decision-error budget; the release ledger must register
that reservation alongside the other campaigns' allocations before acceptance.
Bias uses an explicitly asymptotic Student Monte Carlo interval: unlike the
Bernoulli gates, it does not claim a finite-replication distribution-free bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from increment.errors import CodedError
from increment.estimation.inference import LiftGuardError
from tests.estimation._i13_manifest import I13_ETA, AcceptanceCell, DGPSample, ManifestCell
from tests.mc import binomial_error_upper_bound, coverage_lower_bound, mcse, scientific_delta

if TYPE_CHECKING:
    from increment.estimation.results import RelativeConfidenceSet


@dataclass(frozen=True)
class IntervalResult:
    point: float | None
    lb: float | None
    ub: float | None
    open_side: Literal["lower", "upper"] | None = None
    p_value: float | None = None
    unavailable_reason: str | None = None
    confidence_set: RelativeConfidenceSet | None = None
    decision_unavailable_reason: str | None = None

    @property
    def point_available(self) -> bool:
        return self.point is not None and math.isfinite(self.point)

    @property
    def interval_available(self) -> bool:
        if self.confidence_set is not None:
            return self.confidence_set.geometry != "unavailable"
        if not self.point_available:
            return False
        if self.open_side == "lower":
            return self.lb is None and self.ub is not None and math.isfinite(self.ub)
        if self.open_side == "upper":
            return self.ub is None and self.lb is not None and math.isfinite(self.lb)
        return (
            self.lb is not None
            and self.ub is not None
            and math.isfinite(self.lb)
            and math.isfinite(self.ub)
            and self.lb <= self.ub
        )

    def contains(self, truth: float) -> bool:
        if self.confidence_set is not None:
            return self.confidence_set.contains(truth) is True
        if not self.interval_available:
            return False
        return (self.lb is None or self.lb <= truth) and (self.ub is None or truth <= self.ub)


@dataclass(frozen=True)
class ReplicationPlan:
    reps: int
    passing_count: int
    boundary_margin: float
    nominal_false_failure: float
    eta: float


def _passing_count(reps: int, limit: float, eta: float) -> int:
    lo, hi = -1, reps
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binomial_error_upper_bound(mid, reps, eta) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


@cache
def prospective_plan(
    q: float = 0.05,
    eta: float = I13_ETA,
    *,
    limit: float | None = None,
    margin_limit: float | None = None,
) -> ReplicationPlan:
    """Fixed 1000-step search, with precision at the unacceptable boundary.

    Certification probability is the exact binomial tail at the nominal law,
    not an MCSE criterion. No generated outcomes enter this calculation.
    """
    from scipy.stats import binom

    delta = scientific_delta(q) if q > 0 else 0.005
    limit = q + delta if limit is None else limit
    margin_limit = delta / 2 if margin_limit is None else margin_limit
    if not 0 <= q < limit < 1 or not 0 < eta < 1 or margin_limit <= 0:
        raise ValueError("invalid prospective Bernoulli gate")
    reps = 1000
    while True:
        boundary_count = math.ceil(limit * reps)
        margin = binomial_error_upper_bound(boundary_count, reps, eta) - boundary_count / reps
        kcrit = _passing_count(reps, limit, eta)
        false_failure = float(binom.sf(kcrit, reps, q))
        if margin <= margin_limit and false_failure <= eta:
            return ReplicationPlan(reps, kcrit, margin, false_failure, eta)
        reps += 1000


@cache
def manifest_replications() -> int:
    """Common fixed R covers each distinct Bernoulli gate's precision/power."""
    from scipy.stats import binom

    gates = ((0.05, 0.055), (0.0, 0.01), (0.10, 0.20), (0.05, 0.10))
    reps = max(prospective_plan(q, limit=limit, margin_limit=0.0025).reps for q, limit in gates)
    while True:
        passes = True
        for q, limit in gates:
            boundary_count = math.ceil(limit * reps)
            margin = (
                binomial_error_upper_bound(boundary_count, reps, I13_ETA) - boundary_count / reps
            )
            critical = _passing_count(reps, limit, I13_ETA)
            passes &= margin <= 0.0025 and float(binom.sf(critical, reps, q)) <= I13_ETA
        if passes:
            return reps
        reps += 1000


def required_reps_for_margin(margin: float = 0.0025, p: float = 0.05) -> int:
    if margin != scientific_delta(p) / 2:
        raise ValueError("The precision budget must be delta/2, fixed before observing results")
    return prospective_plan(p).reps


@dataclass
class R03Result:
    cell_name: str
    attempted: int = 0
    point_estimable: int = 0
    interval_estimable: int = 0
    excluded: int = 0
    failed: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    failure_reasons: dict[str, int] = field(default_factory=dict)
    interval_unavailable_reasons: dict[str, int] = field(default_factory=dict)
    decision_unavailable_reasons: dict[str, int] = field(default_factory=dict)
    hits: int = 0
    rejections: int = 0
    rejection_available: int = 0
    finite_widths: int = 0
    acceptable_widths: int = 0
    biases: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)

    @property
    def estimable(self) -> int:
        return self.interval_estimable

    @property
    def coverage_conditional(self) -> float | None:
        return self.hits / self.interval_estimable if self.interval_estimable else None

    @property
    def coverage_unconditional(self) -> float | None:
        return self.hits / self.attempted if self.attempted else None

    @property
    def coverage_conditional_mcse(self) -> float | None:
        rate = self.coverage_conditional
        return (
            mcse(rate, self.interval_estimable)
            if rate is not None and self.interval_estimable >= 2
            else None
        )

    @property
    def coverage_unconditional_mcse(self) -> float | None:
        rate = self.coverage_unconditional
        return mcse(rate, self.attempted) if rate is not None and self.attempted >= 2 else None

    @property
    def bias(self) -> float | None:
        return math.fsum(self.biases) / len(self.biases) if self.biases else None

    @property
    def bias_mcse(self) -> float | None:
        if len(self.biases) < 2:
            return None
        center = self.bias
        assert center is not None
        return math.sqrt(
            math.fsum((x - center) ** 2 for x in self.biases)
            / (len(self.biases) * (len(self.biases) - 1))
        )

    def r03_gate(self, cell: AcceptanceCell, *, alpha: float = 0.05) -> dict[str, object]:
        from scipy.stats import t

        if alpha != 0.05:
            raise ValueError("The I13 acceptance manifest freezes alpha=.05")
        n = self.attempted
        if not n:
            return {"passes": False, "reason": "no attempts"}
        if cell.refusal_code is not None:
            return {
                "passes": self.failed == 0
                and self.point_estimable == 0
                and self.exclusion_reasons == {cell.refusal_code: n},
                "expected_refusal": cell.refusal_code,
                "exclusions": self.exclusion_reasons,
                "failures": self.failure_reasons,
                "attempted": n,
                "acceptance_complete": False,
            }
        eta = I13_ETA
        delta = scientific_delta(alpha)
        misses = n - self.hits
        error_ub = binomial_error_upper_bound(misses, n, eta)
        checks: dict[str, bool] = {
            "coverage": error_ub <= alpha + delta,
            "coverage_precision": error_ub - misses / n <= delta / 2,
            "prospective_reps": n == manifest_replications(),
            "point_availability": coverage_lower_bound(self.point_estimable, n, eta) >= 0.99,
            "interval_availability": coverage_lower_bound(self.interval_estimable, n, eta) >= 0.99,
            "finite_width": coverage_lower_bound(self.finite_widths, n, eta) >= 0.99,
            "no_crashes": self.failed == 0,
            "support_reference_resolved": cell.support == "ordinary_available",
        }
        for name, count in (
            ("point", self.point_estimable),
            ("interval", self.interval_estimable),
            ("finite_width", self.finite_widths),
        ):
            checks[f"{name}_precision"] = count / n - coverage_lower_bound(count, n, eta) <= 0.0025
        bias_radius = None
        if self.bias_mcse is not None:
            bias_radius = float(t.isf(eta, len(self.biases) - 1)) * self.bias_mcse
        checks["bias"] = (
            self.bias is not None
            and bias_radius is not None
            and abs(self.bias) + bias_radius <= cell.bias_tolerance
        )
        if "null_rejection" in cell.quantities:
            # Missing tests are scored as failures for the unconditional validity gate.
            errors = self.rejections + n - self.rejection_available
            upper = binomial_error_upper_bound(errors, n, eta)
            checks["null_rejection"] = upper <= alpha + delta
            checks["null_precision"] = upper - errors / n <= delta / 2
        if cell.useful:
            power_lb = coverage_lower_bound(self.rejections, n, eta)
            width_lb = coverage_lower_bound(self.acceptable_widths, n, eta)
            checks["power"] = power_lb >= 0.80
            checks["power_precision"] = self.rejections / n - power_lb <= 0.0025
            checks["width_limit"] = width_lb >= 0.90
            checks["width_precision"] = self.acceptable_widths / n - width_lb <= 0.0025
        return {
            "passes": all(checks.values()),
            "checks": checks,
            "acceptance_complete": False,
            "remaining_prerequisites": (
                "finite-sample nuisance/residual geometry; bias-tail bound; "
                "R09 family reservation; full scientific evidence"
            ),
            "coverage_lower_bound": 1.0 - error_ub,
            "mc_margin": error_ub - misses / n,
            "bias": self.bias,
            "bias_mcse": self.bias_mcse,
            "bias_radius": bias_radius,
            "bias_reference": "asymptotic Student MC; finite-replication bound unresolved",
            "conditional_coverage": self.coverage_conditional,
            "unconditional_coverage": self.coverage_unconditional,
            "conditional_coverage_mcse": self.coverage_conditional_mcse,
            "unconditional_coverage_mcse": self.coverage_unconditional_mcse,
            "mean_finite_width": math.fsum(self.widths) / len(self.widths) if self.widths else None,
            "exclusions": self.exclusion_reasons,
            "failures": self.failure_reasons,
            "interval_unavailable_reasons": self.interval_unavailable_reasons,
            "point_estimable": self.point_estimable,
            "interval_estimable": self.interval_estimable,
            "attempted": n,
            "eta": eta,
            "rejection_rule": cell.rejection_rule,
        }


def run_cell(
    cell: ManifestCell,
    reps: int,
    *,
    truth_of: Callable[[DGPSample], float],
    estimate_of: Callable[[DGPSample, int], IntervalResult],
    alpha: float = 0.05,
    width_limit: float | None = None,
    observer: Callable[[int, str, dict[str, Any]], None] | None = None,
) -> R03Result:
    if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
        raise ValueError("reps must be a positive integer, excluding Boolean values")
    result = R03Result(cell.name)
    for i in range(reps):
        result.attempted += 1
        if observer is not None:
            observer(i, "started", {"seed": cell.dgp.seed * 100_003 + i})
        try:
            sample = replace(cell.dgp, seed=cell.dgp.seed * 100_003 + i).draw()
            truth = truth_of(sample)
            interval = estimate_of(sample, i)
        except (CodedError, LiftGuardError) as exc:
            result.excluded += 1
            reason = "estimation.engine.lift_guard" if isinstance(exc, LiftGuardError) else exc.code
            result.exclusion_reasons[reason] = result.exclusion_reasons.get(reason, 0) + 1
            if observer is not None:
                observer(i, "excluded", {"reason": reason, "message": str(exc)})
            continue
        except Exception as exc:
            result.failed += 1
            reason = f"{type(exc).__name__}: {exc}"
            result.failure_reasons[reason] = result.failure_reasons.get(reason, 0) + 1
            if observer is not None:
                observer(i, "failed", {"reason": reason})
            continue
        if not interval.point_available:
            result.excluded += 1
            reason = interval.unavailable_reason or "returned_point_unavailable"
            result.exclusion_reasons[reason] = result.exclusion_reasons.get(reason, 0) + 1
        if not interval.interval_available:
            reason = interval.unavailable_reason or "returned_interval_unavailable"
            result.interval_unavailable_reasons[reason] = (
                result.interval_unavailable_reasons.get(reason, 0) + 1
            )
        if interval.point_available:
            result.point_estimable += 1
            assert interval.point is not None
            result.biases.append(interval.point - truth)
        if interval.interval_available:
            result.interval_estimable += 1
            result.hits += int(interval.contains(truth))
            if interval.lb is not None and interval.ub is not None:
                width = interval.ub - interval.lb
                result.finite_widths += int(math.isfinite(width))
                if math.isfinite(width):
                    result.widths.append(width)
                result.acceptable_widths += int(width_limit is not None and width <= width_limit)
        # Two-sided zero-null decision evaluated separately from coverage of truth.
        if interval.p_value is not None and math.isfinite(interval.p_value):
            result.rejection_available += 1
            result.rejections += int(interval.p_value < alpha)
        elif interval.decision_unavailable_reason is not None:
            reason = interval.decision_unavailable_reason
            result.decision_unavailable_reasons[reason] = (
                result.decision_unavailable_reasons.get(reason, 0) + 1
            )
        elif interval.interval_available:
            result.rejection_available += 1
            result.rejections += int(not interval.contains(0.0))
        if observer is not None:
            observer(i, "completed", {"truth": truth, "interval": interval})
    return result
