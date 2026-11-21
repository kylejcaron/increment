"""Person-weighted untreated contrasts for stratified two-stage randomization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from increment._spillover_design import SpilloverDesign
from increment.errors import InvalidRequestError, RefusalSpec, refuse

Z95: Final = 1.959963984540054  # Upper 0.025 standard Normal critical value.

_AVAILABLE, _TOO_FEW, _NO_VARIATION, _ZERO_VARIANCE, _NONFINITE = range(5)
_REASON_TEXT = {
    _AVAILABLE: None,
    _TOO_FEW: "too_few_groups",
    _NO_VARIATION: "no_outcome_variation",
    _ZERO_VARIANCE: "zero_variance",
    _NONFINITE: "nonfinite_variance",
}
_INVALID_COUNTS = RefusalSpec(
    "spillover.counts.invalid",
    InvalidRequestError,
    template="Invalid spillover counts: {requirement}.",
)
# Bound working arrays independently of the number of assignments supplied.
_WORK_CELLS = 262_144


def reason_text(code: int | np.integer) -> str | None:
    """Decode the availability of a single observed untreated contrast."""
    return _REASON_TEXT[int(code)]


@dataclass(frozen=True, slots=True)
class SpilloverBatch:
    """One estimate, nullable standard error and availability code per assignment."""

    estimate: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    reason: NDArray[np.uint8]

    @property
    def available(self) -> NDArray[np.bool_]:
        return self.reason == _AVAILABLE


def _arm_moments(
    totals: NDArray[np.float64], mask: NDArray[np.bool_], count: int, *, estimate_only: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    values = totals[mask].reshape(len(totals), count)
    # Center before averaging: an identical noninteger total repeated in an arm has exactly zero spread.
    origin = values[:, 0].copy()
    values -= origin[:, None]
    offset = values.mean(axis=1)
    origin += offset
    if estimate_only:
        return origin, None
    values -= offset[:, None]
    sum_squares = np.einsum("ij,ij->i", values, values)
    return origin, sum_squares / (count * (count - 1))


@dataclass(frozen=True, slots=True)
class _Stratum:
    columns: slice
    n_high: int
    n_low: int
    weight: float


def _strata(design: SpilloverDesign) -> tuple[_Stratum, ...]:
    people = sum(len(g.members) for g in design.groups)
    start = 0
    strata = []
    for stratum, groups in Counter(g.stratum for g in design.groups).items():
        n_high = design.high_groups[stratum]
        strata.append(
            _Stratum(slice(start, start + groups), n_high, groups - n_high, groups / people)
        )
        start += groups
    return tuple(strata)


def _summarize_rows(
    high: NDArray[np.bool_],
    successes: np.ndarray,
    sizes: NDArray[np.int64],
    low_untreated: NDArray[np.int64],
    high_untreated: NDArray[np.int64],
    strata: tuple[_Stratum, ...],
    estimate: NDArray[np.float64],
    se: NDArray[np.float64],
    reason: NDArray[np.uint8],
) -> None:
    untreated = np.where(high, high_untreated, low_untreated)
    if np.any(successes < 0) or np.any(successes > untreated):
        refuse(
            _INVALID_COUNTS,
            requirement="event counts must lie between zero and the untreated count",
        )
    # Convert before multiplying: n*y may exceed the capacity of the *input* integer dtype.
    totals = np.multiply(successes, sizes, dtype=np.float64)
    totals /= untreated
    scarce = any(min(s.n_low, s.n_high) < 2 for s in strata)
    for s in strata:
        policy = high[:, s.columns]
        if np.any(policy.sum(axis=1) != s.n_high):
            refuse(
                _INVALID_COUNTS,
                requirement="each stratum must have its declared number of high groups",
            )
        values = totals[:, s.columns]
        high_mean, high_var = _arm_moments(values, policy, s.n_high, estimate_only=scarce)
        low_mean, low_var = _arm_moments(values, ~policy, s.n_low, estimate_only=scarce)
        estimate += s.weight * (high_mean - low_mean)
        if high_var is not None and low_var is not None:
            se += s.weight**2 * (high_var + low_var)

    if scarce:
        reason.fill(_TOO_FEW)
        se.fill(np.nan)
        return
    # With explicit rosters, totals are bounded by group sizes and even their squared sums fit float64.
    # A nonfinite variance is an arithmetic failure, not lack of events in an observed assignment.
    reason[~np.isfinite(se)] = _NONFINITE
    reason[se == 0] = _ZERO_VARIANCE
    reason[~successes.any(axis=1) | ~(successes < untreated).any(axis=1)] = _NO_VARIATION
    se[reason != _AVAILABLE] = np.nan
    np.sqrt(se, out=se)


def summarize_counts(
    design: SpilloverDesign, high: NDArray[np.bool_], successes: np.ndarray
) -> SpilloverBatch:
    """Estimate high minus low from batch-by-group policies and integer untreated event counts.

    Parameters
    ----------
    design : SpilloverDesign
        Frozen roster and policy-specific treatment counts.
    high : NDArray[np.bool_]
        Boolean policy assignments in canonical design order.
    successes : np.ndarray
        Counts of events among the untreated under each realized policy.

    Returns
    -------
    SpilloverBatch
        Person-weighted contrasts and their within-policy, between-group uncertainty.

    Notes
    -----
    For group j, T_j = n_j * successes_j / untreated_j estimates its untreated outcome total.
    Stratum s contributes (G_s / N) times the difference of the two policy means of T_j, where
    N is roster size. The variance contribution is (G_s / N)^2 times the sum of each arm's
    sample variance divided by its group count (Jiang, Imai & Malani, Biometrics 2023).
    Inference assumes outcomes depend on the number, not the identities, of treated peers.
    """
    g = len(design.groups)
    if high.ndim != 2 or high.shape[1] != g or high.shape != successes.shape:
        refuse(
            _INVALID_COUNTS, requirement="supply aligned assignment-by-group arrays in design order"
        )
    if high.dtype != np.bool_ or successes.dtype.kind not in "iu":
        refuse(_INVALID_COUNTS, requirement="supply boolean policies and integer event counts")
    sizes = np.fromiter((len(group.members) for group in design.groups), np.int64, count=g)
    low_untreated = sizes - np.fromiter(
        (group.low_treated for group in design.groups), np.int64, count=g
    )
    high_untreated = sizes - np.fromiter(
        (group.high_treated for group in design.groups), np.int64, count=g
    )
    strata = _strata(design)
    n = high.shape[0]
    result = SpilloverBatch(np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.uint8))
    batch_rows = max(1, _WORK_CELLS // g)
    for start in range(0, n, batch_rows):
        rows = slice(start, start + batch_rows)
        _summarize_rows(
            high[rows],
            successes[rows],
            sizes,
            low_untreated,
            high_untreated,
            strata,
            result.estimate[rows],
            result.standard_error[rows],
            result.reason[rows],
        )
    return result
