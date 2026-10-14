"""Fixed-composition pilot planning for longitudinal change-score contrasts.

Power uses a plug-in noncentral-t approximation, not a finite-sample guarantee.
New independent grains replicate the pilot's allocation, cohort proportions,
cluster-size distribution, covariance and schedule under a homogeneous shift.
Additional timepoints require a new pilot reduction; they never increment N.
"""

from __future__ import annotations

import math
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, model_validator
from scipy.optimize import brentq

from increment.errors import CodedModel, InvalidRequestError, RefusalSpec, refuse
from increment.estimation.longitudinal import LongitudinalSummary, fit_longitudinal
from increment.power._noncentral_t import _scalar_power_from_nc
from increment.semantics.longitudinal import LongitudinalDesign

LONGITUDINAL_PLANNING = RefusalSpec(
    "power.longitudinal.input", InvalidRequestError, template="{message}"
)


def _invalid(message: str, **context: object) -> NoReturn:
    refuse(LONGITUDINAL_PLANNING, message=message, **context)


class LongitudinalBaseline(CodedModel, BaseModel):
    """Pilot variance per independent grain and the composition it conditions on.

    Construct with from_summary to retain the fitted design and exact pilot SE.
    Direct construction is reserved for trusted, independently derived pilots.
    Planning excludes clusters crossing cohorts; inference can retain them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    variance_per_independent: float
    pilot_independent: int
    pilot_units: int
    independent_per_cohort_arm: tuple[int, ...]
    design: LongitudinalDesign
    covariates: tuple[str, ...]
    guarantee: Literal["planning approximation"] = "planning approximation"

    @model_validator(mode="after")
    def _check(self) -> LongitudinalBaseline:
        if not math.isfinite(self.variance_per_independent) or self.variance_per_independent <= 0:
            _invalid("Pilot variance must be finite and positive; obtain a nondegenerate pilot.")
        if len(self.independent_per_cohort_arm) != 2 * len(self.design.cohorts) or any(
            n < _minimum_grains(self) for n in self.independent_per_cohort_arm
        ):
            _invalid(
                "Pilot must retain enough independent grains per cohort/arm for its adjustment design."
            )
        if (
            sum(self.independent_per_cohort_arm) != self.pilot_independent
            or self.pilot_units < self.pilot_independent
        ):
            _invalid("Pilot independent counts and unit counts are inconsistent.")
        return self

    @classmethod
    def from_summary(cls, summary: LongitudinalSummary) -> LongitudinalBaseline:
        """Fit a pilot with the same estimator and retained independent scores."""
        result = fit_longitudinal(summary)
        if result.standard_error is None:
            _invalid("Pilot uncertainty is unavailable: zero_variance.")
        stride = 1 + len(summary.covariates)
        for g in summary.clusters:
            participating = sum(
                any(g.gram[(2 * j + a) * stride][(2 * j + a) * stride] > 0 for a in (0, 1))
                for j in range(len(summary.design.cohorts))
            )
            if participating != 1:
                _invalid(
                    "Planning a cluster spanning cohorts requires a new participation model; split the planning cohorts or use an independent pilot design."
                )
        counts = tuple(
            sum(g.gram[j * stride][j * stride] > 0 for g in summary.clusters)
            for j in range(2 * len(summary.design.cohorts))
        )
        assert result.standard_error is not None
        return cls(
            variance_per_independent=result.standard_error**2 * result.n_independent,
            pilot_independent=result.n_independent,
            pilot_units=result.n_units,
            independent_per_cohort_arm=counts,
            design=summary.design,
            covariates=summary.covariates,
        )

    def standard_error(self, n_independent: int) -> float:
        """Project variance for independent grains at the unchanged pilot composition."""
        if type(n_independent) is not int or not 4 <= n_independent <= 2**53:
            _invalid("n_independent must be an integer from 4 through 2**53, never panel rows.")
        minimum = _minimum_grains(self)
        if any(
            n_independent * n / self.pilot_independent < minimum
            for n in self.independent_per_cohort_arm
        ):
            _invalid(
                "Projected cohort/arm composition has too few independent grains for the adjustment design."
            )
        se = math.sqrt(self.variance_per_independent) / math.sqrt(n_independent)
        if not math.isfinite(se) or se <= 0:
            _invalid("Projected standard error exceeds numerical resolution; rescale outcomes.")
        return se


def _minimum_grains(baseline: LongitudinalBaseline) -> int:
    # Unit rows need independent replication of every direction; clusters can
    # replicate those directions within each grain, as certified by the pilot.
    return len(baseline.covariates) + 2 if baseline.design.independence_grain == "unit" else 2


def _request(*, alpha: float, delta: float) -> None:
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1 or alpha / 2 == 0:
        _invalid(
            "alpha must be finite, strictly between zero and one, and have a representable half-tail."
        )
    if isinstance(delta, bool) or not math.isfinite(delta):
        _invalid("delta must be a finite additive effect.")


def longitudinal_achieved_power(
    baseline: LongitudinalBaseline, *, n_independent: int, delta: float, alpha: float = 0.05
) -> float:
    """Two-sided plug-in power at a fixed independent-grain count."""
    _request(alpha=alpha, delta=delta)
    se = baseline.standard_error(n_independent)
    power = _scalar_power_from_nc(
        delta / se, alternative="two-sided", tail_alpha=alpha / 2, dof=n_independent - 1
    )
    if not math.isfinite(power):
        _invalid("The noncentral-t power calculation is numerically unresolved.")
    return power


def _target(power: float, alpha: float) -> None:
    if isinstance(power, bool) or not math.isfinite(power) or not alpha < power < 1:
        _invalid("target_power must be finite and strictly between alpha and one.")


def longitudinal_minimum_detectable_effect(
    baseline: LongitudinalBaseline,
    *,
    n_independent: int,
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Smallest positive homogeneous additive shift attaining the target power."""
    _request(alpha=alpha, delta=0)
    _target(target_power, alpha)
    se = baseline.standard_error(n_independent)

    def objective(nc: float) -> float:
        value = _scalar_power_from_nc(
            nc, alternative="two-sided", tail_alpha=alpha / 2, dof=n_independent - 1
        )
        if not math.isfinite(value):
            _invalid("The noncentral-t MDE calculation is numerically unresolved.")
        return value - target_power

    high = 1.0
    while objective(high) < 0:
        high *= 2
        if not math.isfinite(high):
            _invalid("The requested MDE exceeds numerical resolution.")
    result = float(brentq(objective, 0.0, high)) * se
    if not math.isfinite(result) or result <= 0:
        _invalid("The requested MDE exceeds numerical resolution.")
    return result


def longitudinal_required_independent(
    baseline: LongitudinalBaseline, *, delta: float, target_power: float = 0.8, alpha: float = 0.05
) -> int:
    """Minimum independent grains under the fixed-composition approximation."""
    _request(alpha=alpha, delta=delta)
    _target(target_power, alpha)
    if delta == 0:
        _invalid("A zero effect cannot attain power above alpha.")
    minimum = _minimum_grains(baseline)
    low = max(
        4,
        math.ceil(
            max(
                minimum * baseline.pilot_independent / n
                for n in baseline.independent_per_cohort_arm
            )
        ),
    )
    high = low

    def attained(n: int) -> bool:
        return (
            longitudinal_achieved_power(baseline, n_independent=n, delta=delta, alpha=alpha)
            >= target_power
        )

    while not attained(high):
        if high == 2**53:
            _invalid("Target power is unattained within exactly representable independent counts.")
        high = min(2**53, high * 2)
    while low < high:
        middle = (low + high) // 2
        if attained(middle):
            high = middle
        else:
            low = middle + 1
    return low
