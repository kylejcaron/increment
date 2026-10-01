"""Independent arithmetic and reporting checks for clustered CATE results.

These checks avoid producer reducers and final gate functions, so they do not
certify reporting probabilities or conditional sampling-error thresholds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from increment.errors import refuse
from increment.simulate.cluster_dgp import _INVALID, _EvaluationRoster, _rank_truth

if TYPE_CHECKING:
    from collections.abc import Sequence

    from increment.estimation.targeting import TargetingRule

_REASON = "estimation.targeting."


class _Interval:
    """Outward binary64 bounds, including arbitrary-order reduction error."""

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=float)

    @staticmethod
    def rounded(lo, hi):
        return _Interval(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))

    def __add__(self, other):
        if not isinstance(other, _Interval):
            other = _Interval(other)
        return self.rounded(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self):
        return _Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -(other if isinstance(other, _Interval) else _Interval(other))

    def __mul__(self, other):
        if not isinstance(other, _Interval):
            other = _Interval(other)
        products = np.array(
            [
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            ]
        )
        return self.rounded(products.min(axis=0), products.max(axis=0))

    def __truediv__(self, other):
        if not isinstance(other, _Interval):
            other = _Interval(other)
        if np.any((other.lo <= 0) & (other.hi >= 0)):
            refuse(_INVALID, reason="point arithmetic denominator is not separated from zero")
        return self * self.rounded(1 / other.hi, 1 / other.lo)

    def __getitem__(self, key):
        return _Interval(self.lo[key], self.hi[key])

    def sum(self):
        # gamma_n bounds n additions; a subnormal allowance covers underflow.
        n = self.lo.size
        nu = n * np.finfo(float).eps
        if nu >= 1:
            refuse(_INVALID, reason="point reduction exceeds the arithmetic bound")
        magnitude = math.fsum(float(x) for x in np.maximum(abs(self.lo), abs(self.hi)).flat)
        error = math.nextafter(nu / (1 - nu) * magnitude + n * math.ulp(0.0), math.inf)
        return self.rounded(np.sum(self.lo) - error, np.sum(self.hi) + error)


def point_roundoff_bound(  # noqa: PLR0915
    y: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    actions: np.ndarray,
    statistic: str,
    nuisances: tuple[Sequence[float], ...] | None,
    reconstructed: float,
    *,
    clustered: bool,
    scores: np.ndarray | None,
) -> float:
    """Enclose public arithmetic using its primitive operations, not its result.

    Each primitive rounds outwards by one ulp. Reductions allow gamma_n times
    the absolute input mass. This includes centered means, exact cancellation
    fallbacks and both association orders of a contrast. No outcome-unit band
    or observed discrepancy is used to choose the enclosure.
    """
    if statistic == "uplift" and actions.all():
        return 0.0
    iy, iw, id_ = _Interval(y), _Interval(w), _Interval(d)

    def mean(values, weights):
        return values[0] + ((weights / weights.sum()) * (values - values[0])).sum()

    def contrast(values, first, second):
        a, b = values[first], values[second]
        wa, wb = iw[first], iw[second]
        return (
            a[0]
            - b[0]
            + ((wa / wa.sum()) * (a - a[0])).sum()
            - ((wb / wb.sum()) * (b - b[0])).sum()
        )

    if nuisances is None and statistic == "effect":
        first, second = actions & (d == 1), actions & (d == 0)
        result = (
            contrast(iy, first, second)
            if clustered
            else iy[first].sum() / first.sum() - iy[second].sum() / second.sum()
        )
    else:
        if nuisances is not None:
            e, m1, m0 = (_Interval(v) for v in nuisances)
            psi = (
                m1
                - m0
                + id_ * (iy - m1) / e
                - (_Interval(1) - id_) * (iy - m0) / (_Interval(1) - e)
            )
        else:
            p = (iw * id_).sum() / iw.sum()
            if clustered:
                centered = iy - iy[0]
                residual = centered - (iw * centered).sum() / iw.sum()
            else:
                residual = iy - iy.sum() / y.size
            psi = residual * (id_ - p) / (p * (_Interval(1) - p))
        if statistic == "effect":
            result = (
                mean(psi[actions], iw[actions]) if clustered else psi[actions].sum() / actions.sum()
            )
        elif clustered:
            result = contrast(psi, actions, np.ones(y.size, dtype=bool))
        else:
            # The discrete TOC uses one spike; tied ranks share its coefficient.
            if scores is None:
                result = psi[actions].sum() / actions.sum() - psi.sum() / y.size
            else:
                n, j = y.size, int(actions.sum())
                u = np.zeros(n)
                u[j - 1] = n
                tail = np.r_[np.full(j, float(n / j)), np.zeros(n - j)]
                tail_interval = _Interval.rounded(tail, tail)
                rank = (tail_interval - _Interval(u).sum() / n) / n
                rank = rank - rank.sum() / n
                order = np.argsort(-scores, kind="stable")
                sorted_scores = scores[order]
                starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, n]
                lo, hi = np.empty(n), np.empty(n)
                for a, b in zip(starts[:-1], starts[1:], strict=True):
                    tied = rank[a:b].sum() / (b - a)
                    lo[order[a:b]], hi[order[a:b]] = tied.lo, tied.hi
                result = (_Interval(lo, hi) * n * psi).sum() / n
    # Enclose the separately evaluated affine expansion as well as the public
    # centered arithmetic. Their union width bounds their numerical difference.
    q = iw / iw.sum()
    v = q * actions / (q * actions).sum()
    h = v if statistic == "effect" else v - q
    if nuisances is not None:
        e, m1, m0 = (_Interval(values) for values in nuisances)
        k = id_ / e - (_Interval(1) - id_) / (_Interval(1) - e)
        r = m1 - m0 - id_ * m1 / e + (_Interval(1) - id_) * m0 / (_Interval(1) - e)
        affine = (h * r).sum() + (h * k * iy).sum()
    elif statistic == "effect":
        wt, wc = iw * actions * id_, iw * actions * (_Interval(1) - id_)
        affine = ((wt / wt.sum() - wc / wc.sum()) * iy).sum()
    else:
        p = (q * id_).sum()
        k = (id_ - p) / (p * (_Interval(1) - p))
        affine = ((h * k - q * (h * k).sum()) * iy).sum()
    low = min(float(result.lo), float(affine.lo))
    high = max(float(result.hi), float(affine.hi))
    bound = math.nextafter(high - low, math.inf)
    if not math.isfinite(bound) or not math.isfinite(reconstructed):
        refuse(_INVALID, reason="point reconstruction has no finite arithmetic enclosure")
    if not low <= reconstructed <= high:
        refuse(_INVALID, reason="independent point escaped its arithmetic enclosure")
    return math.nextafter(bound, math.inf)


def _psi(y, d, weights, nuisances):
    if nuisances is not None:
        e, m1, m0 = (np.asarray(v) for v in nuisances)
        return m1 - m0 + d * (y - m1) / e - (1 - d) * (y - m0) / (1 - e)
    p = math.fsum(float(a * b) for a, b in zip(weights, d, strict=True)) / math.fsum(weights)
    if not 0 < p < 1:
        refuse(_INVALID, reason="reporting reconstruction requires both assignment arms")
    centered = y - y[0]
    mean = math.fsum(float(a * b) for a, b in zip(weights, centered, strict=True)) / math.fsum(
        weights
    )
    return (centered - mean) * (d - p) / (p * (1 - p))


def _support(ids, d, observational, *, bootstrap=False):
    prefix = _REASON + ("bootstrap_" if bootstrap else "")
    minimum = 1 if bootstrap else 2
    if observational:
        return prefix + "insufficient_clusters" if len(set(ids)) < minimum else None
    for arm in (0, 1):
        count = len(set(ids[d == arm]))
        if count < minimum:
            return prefix + ("empty_arm" if count == 0 else "insufficient_arm_clusters")
    return None


def _weights(ids, weighting):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return np.ones(len(ids)) if weighting == "member_count" else 1 / counts[inverse]


def _rank_reference(score, psi, weights):
    """Fixed-rank influence products whose weighted mean is the AUTOC.

    Tied block b, in descending score order, has mass m_b and prior mass a_b.
    Its rank weight averages -log(q)-1 over (a_b, a_b+m_b]:
    w_b = -log(a_b+m_b) - (a_b/m_b) log((a_b+m_b)/a_b), and w_0 = -log(m_0).
    The product (w_b(i) - mean w)(psi_i - mean psi) is the row's contribution.
    """
    order = np.argsort(-score, kind="stable")
    ordered, p = score[order], weights[order] / math.fsum(weights)
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    mass = np.add.reduceat(p, starts)
    upper = np.cumsum(mass)
    lower = upper - mass
    rank_weight = -np.log(upper)
    rank_weight[1:] -= lower[1:] / mass[1:] * np.log1p(mass[1:] / lower[1:])
    block = np.repeat(np.arange(mass.size), np.diff(np.r_[starts, ordered.size]))
    centered = rank_weight[block] - math.fsum(mass * rank_weight)
    residual = psi[order] - math.fsum(p * psi[order])
    reference = np.empty(score.size)
    reference[order] = centered * residual
    return reference


def _cluster_scale(values, weights, ids):
    """Cluster-robust scale of a weighted mean, or None when it carries no information.

    With p_i the normalized weights and u_g = sum_{i in g} p_i (v_i - mean v),
    the scale is sqrt(K/(K-1) sum_g u_g^2). Fewer than two clusters, a zero
    spread, or an unrepresentable value all leave the scale unavailable.
    """
    labels, inverse = np.unique(ids, return_inverse=True)
    k = labels.size
    if k < 2:
        return None
    p = weights / math.fsum(weights)
    centered = values - math.fsum(p * values)
    u = np.bincount(inverse, weights=p * centered, minlength=k)
    scale = math.hypot(*u) * math.sqrt(k / (k - 1))
    return scale if math.isfinite(scale) and scale > 0 else None


def _delete_one_summary(autoc, deleted, strata):
    """Delete-one-cluster t scale centered on the full statistic, and its reference df.

    s_J = sqrt((K-1)/K sum_g (T - T_(-g))^2); the smallest resampling stratum
    sets df = min_h(K_h - 1). Returns (scale, df, reason).
    """
    k = len(deleted)
    if k < 2 or min(strata) < 2:
        return None, None, _REASON + "insufficient_clusters"
    if not math.isfinite(autoc) or any(v is None or not math.isfinite(v) for v in deleted):
        return None, None, _REASON + "jackknife_unavailable_replicate"
    factor = Fraction(math.sqrt((k - 1) / k))
    try:
        differences = [float(factor * (Fraction(autoc) - Fraction(v))) for v in deleted]
    except OverflowError:
        return None, None, _REASON + "nonfinite_statistic"
    scale = math.hypot(*differences)
    if not math.isfinite(scale):
        return None, None, _REASON + "nonfinite_statistic"
    if scale <= 0:
        return None, None, _REASON + "degenerate_cluster_variance"
    return scale, float(min(strata) - 1), None


def _bootstrap_t_summary(autoc, samples, alpha, *, scale, scales, reason):
    """Bootstrap-t p-value from roots (T* - T)/s*, compared with T/s exactly.

    p = (1 + #{root >= T/s}) / (B+1), rounded upward. The two-sided interval
    inverts the alpha/2 order statistics of the roots on the original scale;
    when (B+1) alpha/2 < 1 the tail is unresolved and only p is reported.
    """
    if not math.isfinite(autoc):
        reason = _REASON + "nonfinite_statistic"
    elif reason is None and scale is None:
        reason = _REASON + "degenerate_cluster_variance"
    elif reason is None and (
        len(samples) < 2 or any(v is None or not math.isfinite(v) for v in samples)
    ):
        reason = _REASON + "bootstrap_unavailable_replicate"
    if reason is not None:
        return None, reason
    # Exact binary-float centering resolves neighboring values and zero spread.
    exact = [Fraction(float(v)) for v in samples]
    center = sum(exact, Fraction()) / len(exact)
    spread = max(abs(v - center) for v in exact)
    if spread == 0:
        return None, _REASON + "bootstrap_zero_variance"
    normalized = math.hypot(*(float((v - center) / spread) for v in exact))
    try:
        # The runtime reports this bootstrap SD as `se`; only its overflow matters here.
        float(spread * Fraction(normalized / math.sqrt(len(exact) - 1)))
        roots = sorted(
            (Fraction(float(v)) - Fraction(autoc)) / Fraction(scale if s is None else s)
            for v, s in zip(samples, scales, strict=True)
        )
        for root in roots:
            float(root)
    except OverflowError:
        return None, _REASON + "nonfinite_statistic"
    threshold = Fraction(autoc) / Fraction(scale)
    probability = Fraction(1 + sum(root >= threshold for root in roots), len(roots) + 1)
    p_value = float(probability)
    if Fraction(p_value) < probability:
        p_value = math.nextafter(p_value, math.inf)
    tail_count = (len(roots) + 1) * Fraction(alpha) / 2
    if tail_count < 1:
        return p_value, _REASON + "bootstrap_tail_resolution"
    tail = math.floor(tail_count) - 1
    try:
        for root in (roots[-tail - 1], roots[tail]):
            float(Fraction(autoc) - Fraction(scale) * root)
    except OverflowError:
        return None, _REASON + "nonfinite_statistic"
    return p_value, None


def _cluster_rank_gate(autoc, samples, alpha, *, scale, scales, deleted, strata, reason=None):
    """p = max(bootstrap-t p, delete-one-cluster t p); either component alone can reject.

    Returns (p, reason, df). A bootstrap reason other than an unresolved tail
    stands; a jackknife reason then replaces the p-value. The t interval only
    needs to be representable, since the gate reads p alone.
    """
    from scipy.stats import t as student_t

    p_value, reason = _bootstrap_t_summary(
        autoc, samples, alpha, scale=scale, scales=scales, reason=reason
    )
    jackknife_scale, df, jackknife_reason = _delete_one_summary(autoc, deleted, strata)
    if reason not in (None, _REASON + "bootstrap_tail_resolution"):
        return None, reason, df
    if jackknife_reason is not None:
        return None, jackknife_reason, df
    assert p_value is not None and jackknife_scale is not None and df is not None
    p_value = max(p_value, float(student_t.sf(autoc / jackknife_scale, df)))
    if reason is not None:
        return p_value, reason, df
    half = float(student_t.isf(alpha / 2, df)) * jackknife_scale
    if not (math.isfinite(autoc - half) and math.isfinite(autoc + half)):
        return None, _REASON + "nonfinite_statistic", df
    return p_value, None, df


def _unclustered_rank_gate(score, psi):
    from scipy.stats import norm

    # The unclustered rank gate uses a normal, not a Student t, tail.
    n = len(score)
    order = np.argsort(-score, kind="stable")
    rank = np.empty(n)
    harmonic = 0.0
    for j in range(n, 0, -1):
        harmonic = math.fsum((harmonic, 1 / j))
        rank[j - 1] = harmonic - 1
    rank -= rank.mean()
    starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1, n]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        rank[a:b] = math.fsum(rank[a:b]) / (b - a)
    phi = rank * psi[order]
    autoc = float(phi.mean())
    se = float(phi.std(ddof=1)) / math.sqrt(n) if n > 1 else math.nan
    return autoc, float(norm.sf(autoc / se)) if se > 0 else 1.0


class _Rows(NamedTuple):
    """Evaluation arrays with the assignment and weighting the rank statistic needs."""

    score: np.ndarray
    y: np.ndarray
    d: np.ndarray
    psi: np.ndarray
    ids: np.ndarray
    observational: bool
    weighting: Literal["member_count", "equal"]

    def rank(self, rows, instances, *, scale):
        """AUTOC over retained rows, each instance its own cluster, with its reference scale."""
        rw = _weights(instances, self.weighting)
        rp = self.psi[rows] if self.observational else _psi(self.y[rows], self.d[rows], rw, None)
        autoc = _rank_truth(self.score[rows], rp - rp[0], rw, clustered=True)[0]
        if not scale:
            return autoc, None
        return autoc, _cluster_scale(_rank_reference(self.score[rows], rp, rw), rw, instances)


def _resampled_ranks(frame, labels, members, pools, seed, repetitions):
    """Whole-cluster draws within each pool; a repeated source is a new occurrence."""
    rng = np.random.default_rng(seed)
    samples: list[float | None] = []
    scales: list[float | None] = []
    sources: list[tuple[str, ...]] = []
    failure = None
    for _ in range(repetitions if len(labels) >= 2 else 0):
        draw = np.concatenate([rng.choice(pool, size=len(pool), replace=True) for pool in pools])
        sources.append(tuple(str(labels[g]) for g in draw))
        rows = np.concatenate([members[g] for g in draw])
        # Every occurrence has its own mass, even when a source repeats.
        instances = np.concatenate([np.full(len(members[g]), j) for j, g in enumerate(draw)])
        support = _support(frame.ids[rows], frame.d[rows], frame.observational, bootstrap=True)
        if support:
            samples.append(None)
            scales.append(None)
            failure = support
            continue
        autoc, scale = frame.rank(rows, instances, scale=True)
        samples.append(autoc)
        scales.append(scale)
    return samples, scales, sources, failure


def _deleted_ranks(frame, members):
    """The full statistic with each source cluster deleted once."""
    deleted: list[float | None] = []
    for omitted in range(len(members) if len(members) >= 2 else 0):
        rows = np.concatenate([m for g, m in enumerate(members) if g != omitted])
        if _support(frame.ids[rows], frame.d[rows], frame.observational, bootstrap=True):
            deleted.append(None)
            continue
        deleted.append(frame.rank(rows, frame.ids[rows], scale=False)[0])
    return deleted


@dataclass(frozen=True, slots=True)
class ReportingReconstruction:
    """An observed predicate evaluation, never a conditioning-state certificate."""

    rank_passed: bool
    reported: bool
    recommendation: Literal["target", "simple"]
    rank_reason: str | None
    p_value: float | None
    reference_df: float | None
    autoc: float
    fraction: float
    selected_fraction: float
    bootstrap_sources: tuple[tuple[str, ...], ...]
    bootstrap_statistics: tuple[float | None, ...]
    deleted_statistics: tuple[float | None, ...]


def reconstruct_reporting(
    rule: TargetingRule, roster: _EvaluationRoster
) -> ReportingReconstruction:
    """Rebuild AUTOC support and its bootstrap-t, delete-one-cluster or normal gate.

    Only named design/fitted-state fields are read from the rule. Gate decisions,
    public points and evaluation outcomes are not conditioning information.
    """
    y = np.asarray(roster.table["y"], dtype=float)
    d = (np.asarray(roster.table["group_id"]) == "treatment").astype(float)
    score, w = np.asarray(roster.scores), np.asarray(roster.base_weights)
    observational = roster.scenario.assignment == "observational"
    if observational and roster.nuisances is None:
        refuse(_INVALID, reason="reporting reconstruction requires frozen observational nuisances")
    psi = _psi(y, d, w, roster.nuisances)
    ids = None if roster.cluster_ids is None else np.asarray(roster.cluster_ids)
    autoc = _rank_truth(score, psi - psi[0], w, clustered=ids is not None)[0]
    samples: list[float | None] = []
    deleted: list[float | None] = []
    sources: list[tuple[str, ...]] = []
    reason = df = None
    alpha = rule.validation.alpha
    if ids is None:
        autoc, p_value = _unclustered_rank_gate(score, psi)
    else:
        reason = _support(ids, d, observational)
        if reason is None and len(set(score)) < 2:
            reason = _REASON + "degenerate_rank_distribution"
        frame = _Rows(score, y, d, psi, ids, observational, rule.cluster_weight)
        labels = sorted(set(ids))
        members = [np.flatnonzero(ids == label) for label in labels]
        pure = all(len(set(d[rows])) == 1 for rows in members)
        pools: dict[float, list[int]] = {}
        for g, rows in enumerate(members):
            arm = float(d[rows[0]]) if not observational and pure else 0.0
            pools.setdefault(arm, []).append(g)
        seed, repetitions = rule.validation.bootstrap_seed, rule.validation.bootstrap_repetitions
        if seed is None or repetitions is None:
            refuse(
                _INVALID, reason="reporting reconstruction requires declared bootstrap addresses"
            )
        samples, scales, sources, failure = _resampled_ranks(
            frame, labels, members, list(pools.values()), seed, repetitions
        )
        deleted = _deleted_ranks(frame, members)
        p_value, reason, df = _cluster_rank_gate(
            autoc,
            samples,
            alpha,
            scale=_cluster_scale(_rank_reference(score, psi, w), w, ids),
            scales=scales,
            deleted=deleted,
            strata=tuple(len(pool) for pool in pools.values()),
            reason=reason or failure,
        )
    rank_passed = p_value is not None and p_value < alpha
    selected = np.asarray(roster.actions)
    defined = selected.any() and (
        observational or (np.any(d[selected] == 0) and np.any(d[selected] == 1))
    )
    if ids is None and rank_passed and selected.any() and not observational:
        if any(np.count_nonzero(selected & (d == arm)) < 2 for arm in (0, 1)):
            refuse(
                _INVALID, reason="unclustered reported policy lacks its required two units per arm"
            )
    recommendation: Literal["target", "simple"] = (
        "target" if rank_passed and (not selected.any() or defined) else "simple"
    )
    return ReportingReconstruction(
        rank_passed,
        bool(rank_passed and defined),
        recommendation,
        reason,
        p_value,
        df,
        autoc,
        rule.fraction,
        float(np.average(selected, weights=w)),
        tuple(sources),
        tuple(samples),
        tuple(deleted),
    )
