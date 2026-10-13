"""Fixed-horizon joint-cell contrasts for two independently randomized binary experiments.

Each declared contrast is ``sum_k w_k * ybar_k`` over the four joint
assignment cells of the jointly eligible population. Cells are independent
samples under independent unit randomization, so the variance is
``sum_k w_k^2 s_k^2 / n_k`` with Welch-Satterthwaite degrees of freedom: the
heteroskedasticity-consistent (HC2) variance of the saturated regression on
cell indicators. The interval is asymptotic in the per-cell counts.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict
from scipy.stats import t as t_dist

from increment._overlap.admission import CellMoments, EligibilityAudit, cell_arms
from increment._overlap.contrasts import (
    CELLS,
    CONTRASTS,
    Cell,
    cell_probabilities,
    contrast_weights,
)
from increment._overlap.design import OverlapDesign
from increment._overlap.refusals import _raise
from increment.errors import CodedModel
from increment.estimation._tails import student_t_isf, two_sided_critical_value, wald_bounds
from increment.estimation.diagnostics import sample_ratio_mismatch
from increment.estimation.family import bh_select
from increment.overlap.cells import OverlapCells

OverlapRole = Literal["primary", "secondary", "unassigned"]
UnavailableReason = Literal["cell_empty", "cell_variance_unidentified"]
MultiplicityStatus = Literal["bonferroni_fwer", "bh_fdr", "exploratory_unadjusted"]
ASSIGNMENT_CHECK_ALPHA = 0.001


class OverlapEffect(CodedModel, BaseModel):
    """One declared contrast for one metric on the jointly eligible population.

    ``weights`` are the cell-mean weights in ``cells`` order, so the
    estimand is reproducible from the published cell moments. An unavailable
    row keeps its identity and multiplicity fields with numeric nulls and
    names the cells that lack support; it is never filled by extrapolation.

    Intervals for ``primary`` rows are Bonferroni-simultaneous at
    ``test_level``; ``secondary`` and ``unassigned`` rows report marginal
    intervals at ``alpha``. ``significant`` is judged against the row's own
    multiplicity procedure and is null when the row is unavailable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    role: OverlapRole
    contrast: str
    first_experiment: str
    second_experiment: str
    cells: tuple[Cell, ...] = CELLS
    weights: tuple[float, ...]
    available: bool
    unavailable_reason: UnavailableReason | None = None
    unsupported_cells: tuple[Cell, ...] = ()
    estimate: float | None = None
    se: float | None = None
    df: float | None = None
    lb: float | None = None
    ub: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    significant: bool | None = None
    multiplicity_status: MultiplicityStatus
    family_size: int
    test_level: float
    estimand: Literal["joint_cell_mean_contrast"] = "joint_cell_mean_contrast"
    estimator: Literal["cell_means_welch"] = "cell_means_welch"


class AssignmentCheck(CodedModel, BaseModel):
    """Fixed-look Pearson check of the four joint cell counts against the declared law.

    ``cells`` labels each cell ``"<first arm>|<second arm>"`` in ``(00, 10, 01,
    11)`` order. Under independent randomization the expected shares are
    products of the two declared treatment probabilities, so a mismatch flags
    a broken assignment law or a join that dropped units by arm, not merely
    an unbalanced experiment. ``mismatch`` compares ``p_value`` with ``alpha``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cells: tuple[str, ...]
    observed: tuple[int, ...]
    expected_shares: tuple[float, ...]
    chi2_stat: float
    df: int
    p_value: float
    alpha: float
    mismatch: bool
    low_expected_count: bool


class OverlapReadout(CodedModel, BaseModel):
    """Joint-effect evidence for two overlapping experiments.

    Carries the design and assignment law (provenance), the eligibility
    audit, the per-cell moments every row is computed from, the joint
    assignment check, and one row per declared ``(metric, contrast)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: OverlapDesign
    contrasts: tuple[str, ...]
    rows: tuple[OverlapEffect, ...]
    cells: tuple[CellMoments, ...]
    audit: EligibilityAudit
    outcome_window: dt.timedelta
    assignment_check: AssignmentCheck
    alpha: float
    q: float
    population: Literal["jointly_eligible"] = "jointly_eligible"

    def row(self, metric: str, contrast: str) -> OverlapEffect:
        """The row for ``(metric, contrast)``.

        Parameters
        ----------
        metric : str
            Metric name.
        contrast : str
            Declared contrast label.

        Returns
        -------
        OverlapEffect
            The matching row.

        Raises
        ------
        KeyError
            When ``(metric, contrast)`` was not estimated.
        """
        for r in self.rows:
            if r.metric == metric and r.contrast == contrast:
                return r
        raise KeyError((metric, contrast))


def _cell_key(design: OverlapDesign, cell: Cell) -> str:
    return "|".join(cell_arms(design, cell))


def _assignment_check(cells: OverlapCells) -> AssignmentCheck:
    design = cells.design
    probabilities = cell_probabilities(
        design.first.treatment_probability, design.second.treatment_probability
    )
    labels = tuple(_cell_key(design, c) for c in CELLS)
    observed = tuple(cells.cell_count(c) for c in CELLS)
    srm = sample_ratio_mismatch(
        dict(zip(labels, observed, strict=True)),
        {label: probabilities[c] for label, c in zip(labels, CELLS, strict=True)},
        alpha=ASSIGNMENT_CHECK_ALPHA,
        inference="fixed",
    )
    return AssignmentCheck(
        cells=labels,
        observed=observed,
        expected_shares=tuple(probabilities[c] for c in CELLS),
        chi2_stat=srm.chi2_stat,
        df=srm.df,
        p_value=srm.fixed_p_value,
        alpha=srm.alpha,
        mismatch=srm.is_srm,
        low_expected_count=srm.low_expected_count,
    )


def _validated_roles(
    cells: OverlapCells, roles: Mapping[str, str] | None
) -> dict[str, OverlapRole]:
    supplied = dict(roles or {})
    for metric, role in supplied.items():
        if metric not in cells.metrics:
            _raise("overlap.request.metric_unknown", metric=metric, available=cells.metrics)
        if role == "guardrail":
            _raise("overlap.multiplicity.guardrail_unfinished", metric=metric)
        if role not in ("primary", "secondary", "unassigned"):
            _raise("overlap.request.role_unknown", metric=metric, role=role)
    out: dict[str, OverlapRole] = {}
    for m in cells.metrics:
        role = supplied.get(m, "unassigned")
        out[m] = (
            "primary" if role == "primary" else "secondary" if role == "secondary" else "unassigned"
        )
    return out


def _validated_contrasts(contrasts: Sequence[str]) -> tuple[str, ...]:
    declared = tuple(contrasts)
    if not declared:
        _raise("overlap.request.contrasts_empty", allowed=CONTRASTS)
    seen: set[str] = set()
    for c in declared:
        if c not in CONTRASTS:
            _raise("overlap.request.contrast_unknown", contrast=c, allowed=CONTRASTS)
        if c in seen:
            _raise("overlap.request.contrast_duplicate", contrast=c)
        seen.add(c)
    return declared


def _level(value: object, code: str, key: str) -> float:
    if not (isinstance(value, float) and math.isfinite(value) and 0.0 < value < 1.0):
        _raise(code, **{key: value})
    return value


@dataclass(frozen=True, slots=True)
class _Fit:
    estimate: float
    se: float
    df: float
    p_value: float


@dataclass(frozen=True, slots=True)
class _Gap:
    reason: UnavailableReason
    cells: tuple[Cell, ...]


def _supported(m: CellMoments) -> tuple[float, float, int]:
    if m.mean is None or m.variance is None:
        raise AssertionError(f"cell {m.cell} reached inference without a mean and variance")
    return m.mean, m.variance, m.n


def _fsum(values: Iterable[float]) -> float:
    """``math.fsum`` of terms scaled by their largest magnitude, so a sum beyond the
    float range comes back as a signed infinity instead of an ``OverflowError``."""
    terms = list(values)
    scale = max((abs(t) for t in terms), default=0.0)
    if scale == 0.0 or not math.isfinite(scale):
        return math.fsum(terms)
    return math.fsum(t / scale for t in terms) * scale


def _compute(
    cells: OverlapCells, metric: str, contrast: str, weights: Mapping[Cell, float]
) -> _Fit | _Gap:
    used = [c for c in CELLS if weights.get(c, 0.0) != 0.0]
    moments = {c: cells.moment(metric, c) for c in used}
    empty = tuple(c for c in used if moments[c].n == 0)
    thin = tuple(c for c in used if moments[c].n == 1)
    if empty or thin:
        return _Gap("cell_empty" if empty else "cell_variance_unidentified", empty + thin)
    stats = {c: _supported(moments[c]) for c in used}
    # The weights sum to zero, so centering the means on their midpoint leaves the
    # contrast unchanged while bounding each term by half the means' range.
    means = [stats[c][0] for c in used]
    center = min(means) / 2.0 + max(means) / 2.0
    estimate = _fsum(weights[c] * (stats[c][0] - center) for c in used)
    parts = [weights[c] ** 2 * stats[c][1] / stats[c][2] for c in used]
    var = _fsum(parts)
    se = math.sqrt(var)
    if var > 0.0:
        # Scaling by the largest part keeps the squared terms inside the float range.
        scale = max(parts)
        denominator = math.fsum(
            (v / scale) ** 2 / (stats[c][2] - 1) for v, c in zip(parts, used, strict=True)
        )
        df = (var / scale) ** 2 / denominator
    else:
        df = float(sum(stats[c][2] - 1 for c in used))
    nonfinite = tuple(
        name for name, v in (("estimate", estimate), ("se", se), ("df", df)) if not math.isfinite(v)
    )
    if nonfinite:
        _raise(
            "overlap.inference.unrepresentable",
            contrast=contrast,
            metric=metric,
            nonfinite=nonfinite,
        )
    if se > 0.0:
        p_value = float(2.0 * t_dist.sf(abs(estimate) / se, df))
    else:
        p_value = 1.0 if estimate == 0.0 else 0.0
    return _Fit(estimate, se, df, p_value)


def _bh_adjusted(p_values: Sequence[float]) -> list[float]:
    """Step-up BH adjusted p-values ``min_{j>=i} m p_(j) / j``, capped at one."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        adjusted[i] = min(1.0, running)
    return adjusted


def _secondary_decisions(
    members: Sequence[tuple[tuple[str, str], _Fit | _Gap]], q: float
) -> dict[tuple[str, str], tuple[float, bool]]:
    """BH at ``q`` over the secondary family; unavailable members stay in at ``p = 1``."""
    if not members:
        return {}
    ps = [fit.p_value if isinstance(fit, _Fit) else 1.0 for _, fit in members]
    selected, _ = bh_select(ps, q)
    chosen = set(selected)
    return {
        key: (adjusted, i in chosen)
        for i, ((key, fit), adjusted) in enumerate(zip(members, _bh_adjusted(ps), strict=True))
        if isinstance(fit, _Fit)
    }


def _row(
    design: OverlapDesign,
    key: tuple[str, str],
    fit: _Fit | _Gap,
    role: OverlapRole,
    family_size: int,
    weights: Mapping[Cell, float],
    alpha: float,
    secondary: tuple[float, bool] | None,
) -> OverlapEffect:
    metric, contrast = key
    size = family_size if role != "unassigned" else 1
    status: MultiplicityStatus
    if role == "primary":
        status, level = "bonferroni_fwer", alpha / size
    elif role == "secondary":
        status, level = "bh_fdr", alpha
    else:
        status, level = "exploratory_unadjusted", alpha
    if isinstance(fit, _Gap):
        estimate = se = df = lb = ub = p_value = adjusted = None
        significant = None
        reason, unsupported = fit.reason, fit.cells
    else:
        estimate, se, df, p_value = fit.estimate, fit.se, fit.df, fit.p_value
        reason, unsupported = None, ()
        if fit.se > 0.0:
            what = f"overlap contrast {contrast!r} for {metric!r}"
            crit = two_sided_critical_value(student_t_isf, level, fit.df, what=what)
            lb, ub = wald_bounds(fit.estimate, crit, fit.se, what=what)
        else:
            lb = ub = fit.estimate
        if role == "primary":
            adjusted, significant = min(1.0, fit.p_value * size), fit.p_value <= level
        elif secondary is not None:
            adjusted, significant = secondary
        else:
            adjusted, significant = fit.p_value, fit.p_value <= alpha
    return OverlapEffect(
        metric=metric,
        role=role,
        contrast=contrast,
        first_experiment=design.first.name,
        second_experiment=design.second.name,
        weights=tuple(weights.get(c, 0.0) for c in CELLS),
        available=reason is None,
        unavailable_reason=reason,
        unsupported_cells=unsupported,
        estimate=estimate,
        se=se,
        df=df,
        lb=lb,
        ub=ub,
        p_value=p_value,
        adjusted_p_value=adjusted,
        significant=significant,
        multiplicity_status=status,
        family_size=size,
        test_level=level,
    )


def estimate_overlap_effects(
    cells: OverlapCells,
    *,
    contrasts: Sequence[str],
    roles: Mapping[str, str] | None = None,
    alpha: float = 0.05,
    q: float = 0.10,
) -> OverlapReadout:
    """Estimate prespecified joint-cell contrasts for two overlapping experiments.

    Parameters
    ----------
    cells : OverlapCells
        Admitted cell moments from ``OverlapCells.from_frame`` or ``from_table``.
    contrasts : Sequence[str]
        The declared contrast family, from ``increment.overlap.CONTRASTS``.
        Only declared contrasts are estimated and corrected for; there is no
        search over launch combinations.
    roles : Mapping[str, str] | None
        Metric roles. ``primary`` metrics form one Bonferroni family over
        every declared contrast at ``alpha``; ``secondary`` metrics one BH
        family at ``q``; undeclared metrics are ``unassigned`` and reported
        ``exploratory_unadjusted``.
    alpha : float
        Familywise level for primary rows and the interval level otherwise.
    q : float
        False-discovery level for the secondary family.

    Returns
    -------
    OverlapReadout
        One row per ``(metric, contrast)`` plus provenance and diagnostics.
    """
    declared = _validated_contrasts(contrasts)
    role_of = _validated_roles(cells, roles)
    alpha = _level(alpha, "overlap.request.alpha", "alpha")
    q = _level(q, "overlap.request.q", "q")
    design = cells.design
    p1, p2 = design.first.treatment_probability, design.second.treatment_probability
    weights = {c: contrast_weights(c, p1, p2) for c in declared}
    computed = {(m, c): _compute(cells, m, c, weights[c]) for m in cells.metrics for c in declared}
    family = {
        role: [k for k in computed if role_of[k[0]] == role]
        for role in ("primary", "secondary", "unassigned")
    }
    decisions = _secondary_decisions([(k, computed[k]) for k in family["secondary"]], q)
    rows = [
        _row(
            design,
            key,
            fit,
            role_of[key[0]],
            len(family[role_of[key[0]]]),
            weights[key[1]],
            alpha,
            decisions.get(key),
        )
        for key, fit in computed.items()
    ]
    return OverlapReadout(
        design=design,
        contrasts=declared,
        rows=tuple(rows),
        cells=cells.moments,
        audit=cells.audit,
        outcome_window=cells.outcome_window,
        assignment_check=_assignment_check(cells),
        alpha=alpha,
        q=q,
    )
