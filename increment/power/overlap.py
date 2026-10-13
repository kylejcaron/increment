"""Planning for two overlapping experiments: interaction and launch contrasts.

The analytic solvers use the estimator's own variance,
``sum_k w_k^2 sigma_k^2 / (N pi_k)``, at the expected joint-cell allocation
``pi_k`` of the jointly eligible population ``N``, with a normal reference:
a planning approximation that omits the Welch t correction and the chance
that a cell is sparse. :func:`simulate_overlap_power` instead draws
independent Bernoulli assignments and runs the runtime estimator, including
its Welch interval, multiplicity share, and unavailable cells.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from increment._overlap.admission import Admission, EligibilityAudit
from increment._overlap.contrasts import CELLS, Cell, cell_probabilities, contrast_weights
from increment._overlap.design import OverlapDesign
from increment._overlap.frame import cell_moments
from increment._overlap.refusals import _raise
from increment.overlap.cells import OverlapCells
from increment.overlap.estimator import _validated_contrasts, estimate_overlap_effects

_MIN_TOTAL = 8


def _sds(sd: float | Mapping[Cell, float]) -> dict[Cell, float]:
    values = dict(sd) if isinstance(sd, Mapping) else dict.fromkeys(CELLS, sd)
    for cell in CELLS:
        v = values.get(cell)
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            _raise("overlap.planning.sd", cell=cell, sd=v)
    return {c: float(values[c]) for c in CELLS}


def _checked(design: OverlapDesign, n_total: object, alpha: float, family_size: object) -> float:
    if not (isinstance(n_total, int) and not isinstance(n_total, bool) and n_total >= _MIN_TOTAL):
        _raise("overlap.planning.n_total", n_total=n_total)
    if not (isinstance(alpha, float) and math.isfinite(alpha) and 0.0 < alpha < 1.0):
        _raise("overlap.request.alpha", alpha=alpha)
    if not (
        isinstance(family_size, int) and not isinstance(family_size, bool) and family_size >= 1
    ):
        _raise("overlap.planning.family_size", family_size=family_size)
    for cell, probability in cell_probabilities(
        design.first.treatment_probability, design.second.treatment_probability
    ).items():
        if probability <= 0.0:
            _raise("overlap.planning.cell_probability", cell=cell, probability=probability)
    return alpha / family_size


def overlap_contrast_se(
    design: OverlapDesign, contrast: str, *, n_total: int, sd: float | Mapping[Cell, float]
) -> float:
    """Expected standard error of one contrast at the design's joint-cell allocation.

    Parameters
    ----------
    design : OverlapDesign
        The two experiments; their treatment probabilities set the allocation.
    contrast : str
        A label from ``increment.overlap.CONTRASTS``.
    n_total : int
        Jointly eligible units.
    sd : float | Mapping[Cell, float]
        Outcome standard deviation, common or per ``(first_treated, second_treated)`` cell.

    Returns
    -------
    float
        ``sqrt(sum_k w_k^2 sigma_k^2 / (n_total pi_k))``.
    """
    (contrast,) = _validated_contrasts((contrast,))
    _checked(design, n_total, 0.05, 1)
    sds = _sds(sd)
    p1, p2 = design.first.treatment_probability, design.second.treatment_probability
    pi = cell_probabilities(p1, p2)
    weights = contrast_weights(contrast, p1, p2)
    return math.sqrt(math.fsum(w * w * sds[c] ** 2 / (n_total * pi[c]) for c, w in weights.items()))


def _two_sided_power(z: float, level: float) -> float:
    crit = float(norm.isf(level / 2.0))
    return float(norm.sf(crit - z) + norm.sf(crit + z))


def overlap_contrast_power(
    design: OverlapDesign,
    contrast: str,
    *,
    effect: float,
    n_total: int,
    sd: float | Mapping[Cell, float],
    alpha: float = 0.05,
    family_size: int = 1,
) -> float:
    """Two-sided power to detect ``effect`` on one contrast at ``alpha / family_size``.

    Parameters
    ----------
    design : OverlapDesign
        The two experiments; their treatment probabilities set the allocation.
    contrast : str
        A label from ``increment.overlap.CONTRASTS``.
    effect : float
        True value of the contrast on the outcome scale.
    n_total : int
        Jointly eligible units.
    sd : float | Mapping[Cell, float]
        Outcome standard deviation, common or per cell.
    alpha : float
        Familywise level.
    family_size : int
        Bonferroni family size the runtime will use (primary metrics times declared contrasts).

    Returns
    -------
    float
        Normal-approximation power.
    """
    level = _checked(design, n_total, alpha, family_size)
    if not (isinstance(effect, (int, float)) and math.isfinite(effect)):
        _raise("overlap.planning.effect", effect=effect)
    se = overlap_contrast_se(design, contrast, n_total=n_total, sd=sd)
    return _two_sided_power(abs(effect) / se, level)


def overlap_minimum_detectable_effect(
    design: OverlapDesign,
    contrast: str,
    *,
    n_total: int,
    sd: float | Mapping[Cell, float],
    alpha: float = 0.05,
    power: float = 0.8,
    family_size: int = 1,
) -> float:
    """Smallest absolute contrast detected with ``power`` at ``alpha / family_size``.

    Parameters
    ----------
    design : OverlapDesign
        The two experiments; their treatment probabilities set the allocation.
    contrast : str
        A label from ``increment.overlap.CONTRASTS``.
    n_total : int
        Jointly eligible units.
    sd : float | Mapping[Cell, float]
        Outcome standard deviation, common or per cell.
    alpha : float
        Familywise level.
    power : float
        Target power.
    family_size : int
        Bonferroni family size the runtime will use.

    Returns
    -------
    float
        The effect at which :func:`overlap_contrast_power` equals ``power``.
    """
    level = _checked(design, n_total, alpha, family_size)
    if not (isinstance(power, float) and math.isfinite(power) and level / 2.0 < power < 1.0):
        _raise("overlap.planning.power", power=power)
    se = overlap_contrast_se(design, contrast, n_total=n_total, sd=sd)
    crit = float(norm.isf(level / 2.0))
    upper = crit + float(norm.isf(1.0 - power)) + 1.0
    z = brentq(lambda x: _two_sided_power(x, level) - power, 0.0, upper, xtol=1e-12)
    return float(z * se)


@dataclass(frozen=True, slots=True)
class OverlapPowerSimulation:
    """Monte Carlo rejection and availability rates of one contrast under the runtime procedure."""

    contrast: str
    replications: int
    rejection_rate: float
    rejection_mcse: float
    availability_rate: float


def simulate_overlap_power(
    design: OverlapDesign,
    contrast: str,
    *,
    cell_means: Mapping[Cell, float],
    n_total: int,
    sd: float | Mapping[Cell, float],
    alpha: float = 0.05,
    family_size: int = 1,
    replications: int = 1000,
    seed: int = 0,
) -> OverlapPowerSimulation:
    """Design-matched power: draw independent assignments and run the runtime estimator.

    Each replication assigns ``n_total`` jointly eligible units by independent
    Bernoulli draws at the declared probabilities, draws normal outcomes with
    the given cell means and standard deviations, and tests ``contrast`` as a
    primary row at ``alpha / family_size``. Replications with an unavailable
    row count as non-rejections.

    Parameters
    ----------
    design : OverlapDesign
        The two experiments; their treatment probabilities set the allocation.
    contrast : str
        A label from ``increment.overlap.CONTRASTS``.
    cell_means : Mapping[Cell, float]
        True mean per ``(first_treated, second_treated)`` cell.
    n_total : int
        Jointly eligible units.
    sd : float | Mapping[Cell, float]
        Outcome standard deviation, common or per cell.
    alpha : float
        Familywise level.
    family_size : int
        Bonferroni family size the runtime will use.
    replications : int
        Monte Carlo replications.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    OverlapPowerSimulation
        Rejection rate with its Monte Carlo standard error, and availability rate.
    """
    level = _checked(design, n_total, alpha, family_size)
    (contrast,) = _validated_contrasts((contrast,))
    sds = _sds(sd)
    # Cell index ``first + 2 * second`` matches CELLS order (00, 10, 01, 11).
    mean_by_index = np.array([float(cell_means[c]) for c in CELLS])
    sd_by_index = np.array([sds[c] for c in CELLS])
    rng = np.random.default_rng(seed)
    p1, p2 = design.first.treatment_probability, design.second.treatment_probability
    audit = EligibilityAudit(
        units=n_total,
        jointly_eligible=n_total,
        first_only_eligible=0,
        second_only_eligible=0,
        neither_eligible=0,
    )
    rejections = available = 0
    for _ in range(replications):
        first = (rng.random(n_total) < p1).astype(np.int8)
        second = (rng.random(n_total) < p2).astype(np.int8)
        index = first + 2 * second
        y = mean_by_index[index] + sd_by_index[index] * rng.standard_normal(n_total)
        cells = OverlapCells._admit(
            Admission(
                design=design,
                metrics=("y",),
                moments=cell_moments(design, "y", first, second, y),
                audit=audit,
                outcome_window=dt.timedelta(0),
            )
        )
        row = estimate_overlap_effects(
            cells, contrasts=(contrast,), roles={"y": "unassigned"}, alpha=level
        ).rows[0]
        if row.available:
            available += 1
            rejections += bool(row.p_value is not None and row.p_value <= level)
    rate = rejections / replications
    return OverlapPowerSimulation(
        contrast=contrast,
        replications=replications,
        rejection_rate=rate,
        rejection_mcse=math.sqrt(rate * (1.0 - rate) / replications),
        availability_rate=available / replications,
    )
