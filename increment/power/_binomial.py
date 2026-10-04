"""Planning power for the runtime's exact binomial risk-ratio decision.

A binomial-eligible conversion or retention contrast (unadjusted, unclustered,
unit-grain raw binary counts, fixed horizon; see
``increment.estimation.engine._binomial_eligible``) is decided at runtime by
the Berger-Boos test in ``increment.estimation.binomial_rr``: for observed
counts ``(i, j)`` at analyzed arm sizes ``(n_c, n_t)`` it reports the finite
nuisance-search certificates ``p_+ = p_plus(r0, i, n_c, j, n_t, beta, tail=u)`` and
``p_- = p_minus(...)`` with ``r0 = 1 + null_lift`` and ``beta =
nuisance_beta(alpha_d)``, ``alpha_d`` the compiled decision alpha, and rejects
when the directional certificate is strictly below its tail allocation ``u``
(``compiled_tail_alpha``, the level its refinement is read against): ``p_+ < u`` for
"greater", ``p_- < u`` for "less",
and either for "two-sided". A count pair whose runtime row fails to build is a
``DecisionFailure`` and never a rejection.

For fixed ``(n_c, n_t, r0, beta, u, alternative)`` that event ``D(i, j)`` is a
set of count pairs -- the rejection geometry -- independent of the true rates,
so the planning power is the polynomial

    power(p_c, p_t) = sum_i sum_j Bin(i; n_c, p_c) Bin(j; n_t, p_t) D(i, j).

It is integrated over outer count windows each omitting at most
``_OUTER_TAIL`` of its arm's mass; the omitted mass is measured, not
renormalized, and bounds the reported (central) value from above only.

Two routes classify ``D``:

* ``exact``: replays the runtime's own nuisance search -- the same
  Clopper-Pearson domains, support windows, endpoint and coordinate-corner
  evaluations, largest-bound-first split order with its tie order,
  tail-floored relative-gap stop rule, split cap and floating-point floor -- over many count
  pairs at once. Its tails use the runtime's binomial special functions
  elementwise; only the summation order differs, within the per-row
  allowance ``delta``. Every comparison of the replay is carried out with
  that allowance: one it cannot settle hands the count pair to the unchanged
  runtime functions. Two proved exits stop a replay once its Boolean outcome
  is fixed: an achieved endpoint already at the tail allocation cannot
  reject, and once every current leaf's reachable bound (``_eventual_bound``)
  is below it the runtime must reject.
* ``approximate``: replays the same search with a continuity-corrected Normal
  tail for the conditional sum and exact single-binomial tails when either
  arm is deterministic, keeping every allowance and cap the runtime adds. It
  is the certificate-aware approximation of the exact route: no fitted
  offsets and no threshold compression. Its exits are the same non-rejection
  stop and a rejection stop from interval bounds of the Normal tail over each
  leaf (``_SurrogateTails.reach``).

The route is a deterministic function of the geometry: ``exact`` when the
retained (control, treatment) cell count at the null rate is within
``EXACT_CELL_BUDGET``, else ``approximate``. A geometry the runtime refuses in full (an arm
above its ceiling, or a tail level its float margin dominates) has power exactly zero and
is never replayed. A geometry never stores more than ``PLANNING_CELL_CEILING`` cells: a
decision whose null rectangle exceeds it is not planned, and an evaluation whose own
rectangle (the control window at the control rate by the treatment window at the alternative
rate), or the union the geometry would hold with it, exceeds it raises ``ReplayBoundExceeded``
before any mask is allocated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr as _ndtr
from scipy.stats import binom as _binom

from increment._literals import Alternative
from increment.estimation import binomial_rr as _rr

Kind = Literal["plus", "minus"]
Route = Literal["exact", "approximate"]

_EPS = float(np.finfo(np.float64).eps)

#: Per-side mass each outer count window may omit. The omitted mass is
#: measured afterwards and reported as an upper allowance, never renormalized.
_OUTER_TAIL = 5e-13

#: Descendant term growth: a child's width is half its parent's up to one
#: rounding of the midpoint, and its curvature bound no larger than the
#: parent's up to rounding of the same arithmetic.
_TERM_GROWTH = 1.0 + 1e-9

#: Rounding room between the surrogate's pointwise tails and their interval
#: bounds (both float64 evaluations of the same Normal tail formula).
_SURROGATE_SLACK = 1e-12

#: Exact-route budget in retained (control, treatment) cells at the null rate.
#: Benchmarks across arm sizes, rates, and one- and two-sided tests keep the
#: slowest measured complete `achieved_power` call near two seconds.
EXACT_CELL_BUDGET = 16_000

#: Planning bound in (control, treatment) count cells a geometry may store, about 10-15
#: CPU-microseconds each. It bounds every evaluation's rectangle (the control window by the
#: treatment window at the null or the alternative rate) and the union a geometry holds across
#: them; the runtime's own arm ceiling is unaffected by it.
PLANNING_CELL_CEILING = 10_000_000


@dataclass(frozen=True, slots=True)
class BinomialDecision:
    """The runtime decision at fixed analyzed counts: the geometry's key."""

    n_c: int
    n_t: int
    null_ratio: float
    beta: float
    tail_alpha: float
    alternative: Alternative

    @property
    def kinds(self) -> tuple[Kind, ...]:
        if self.alternative == "greater":
            return ("plus",)
        if self.alternative == "less":
            return ("minus",)
        return ("plus", "minus")


# --- Groups: one control count and direction over a treatment-count range ----


@dataclass(slots=True)
class _Groups:
    """Per-group constants of one classification batch (structure of arrays)."""

    kind: np.ndarray  # 0 plus, 1 minus
    x_c: np.ndarray
    a: np.ndarray
    hi: np.ndarray
    wlo: np.ndarray
    width: np.ndarray
    omitted: np.ndarray
    margin: np.ndarray
    j0: np.ndarray
    j1: np.ndarray
    dmin: np.ndarray
    dmax: np.ndarray
    offsets: np.ndarray  # (G, Wmax) threshold offsets, padded with zero


def _threshold_offsets(kind: Kind, n_c: int, n_t: int, x_c: int, s: np.ndarray) -> np.ndarray:
    """Treatment thresholds of the runtime tails minus the treatment count.

    The runtime's threshold for control count ``s`` and ``k = n_c j - n_t x_c`` is
    `binomial_rr._count_threshold` of ``k + n_t s = n_c j + n_t (s - x_c)``, which is ``j`` plus
    the same threshold of ``n_t (s - x_c)``: the shift by a multiple of ``n_c`` is exact in
    integers. The offsets are therefore exact at every admitted arm size.
    """
    return _rr._count_threshold(kind, n_c, n_t * (s - x_c))


# --- Replay engine ----------------------------------------------------------


@dataclass(slots=True)
class _Leaves:
    """Per-row leaf arrays of the replayed nuisance search."""

    u: np.ndarray
    v: np.ndarray
    pu: np.ndarray  # point ids of the endpoints (exact tails only)
    pv: np.ndarray
    fu: np.ndarray
    fv: np.ndarray
    bound: np.ndarray  # runtime (capped) bound
    raw: np.ndarray  # bound before the cap at one
    reach: np.ndarray  # largest value reportable from inside the leaf
    count: np.ndarray

    _FIELDS = ("u", "v", "pu", "pv", "fu", "fv", "bound", "raw", "reach")

    @staticmethod
    def _blank(name: str, rows: int, capacity: int) -> np.ndarray:
        if name in ("pu", "pv"):
            return np.zeros((rows, capacity), np.int64)
        blank = np.zeros((rows, capacity))
        if name in ("bound", "reach"):
            blank.fill(-np.inf)
        return blank

    @classmethod
    def empty(cls, rows: int, capacity: int) -> _Leaves:
        arrays = {name: cls._blank(name, rows, capacity) for name in cls._FIELDS}
        return cls(**arrays, count=np.ones(rows, np.int64))

    def take(self, rows: np.ndarray, *, capacity: int) -> _Leaves:
        taken = _Leaves.empty(rows.size, capacity)
        width = self.u.shape[1]
        for name in self._FIELDS:
            getattr(taken, name)[:, :width] = getattr(self, name)[rows]
        taken.count = self.count[rows]
        return taken

    def grow(self, capacity: int) -> None:
        """Widen every field to *capacity* slots, one field at a time so a field's old array is
        released before the next is replaced and the transient is one field, not a second copy."""
        width = self.u.shape[1]
        for name in self._FIELDS:
            grown = self._blank(name, self.u.shape[0], capacity)
            grown[:, :width] = getattr(self, name)
            setattr(self, name, grown)


def _curvature(u: np.ndarray, v: np.ndarray, r: float, n_c: int, n_t: int) -> np.ndarray:
    """``binomial_rr._i_bound`` elementwise, with the runtime's operation order."""

    def at(q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            control = np.where((q > 0.0) & (q < 1.0), n_c / (q * (1.0 - q)), np.inf)
            rq = r * q
            treatment = np.where((q <= 0.0) | (rq >= 1.0), np.inf, (n_t * r) / (q * (1.0 - rq)))
        return control + treatment

    return np.maximum(at(u), at(v))


def _quad_term(imax: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``imax * (v - u) ** 2 / 4.0`` as the runtime evaluates it (C ``pow``)."""
    finite = np.isfinite(imax)
    width = v - u
    with np.errstate(invalid="ignore", over="ignore"):
        term = imax * np.power(width, np.full_like(width, 2.0)) / 4.0
    return np.where(finite, term, np.inf)


class _Outcome:
    SEARCHING = -1
    ACCEPT = 0
    REJECT = 1
    AMBIGUOUS = 2
    FINISHED = 3


@dataclass(slots=True)
class _Batch:
    """All count pairs of one classification call, grouped contiguously."""

    decision: BinomialDecision
    groups: _Groups
    group: np.ndarray  # row -> group
    j: np.ndarray  # row -> treatment count
    delta: np.ndarray  # per-row summation allowance (zero for the approximate route)
    guard: np.ndarray  # per-row reachable-bound allowance (exact route only)


def _eventual_bound(
    bound: np.ndarray, mono: np.ndarray, term: np.ndarray, guard: np.ndarray
) -> np.ndarray:
    """Largest value the runtime can later report from inside a leaf.

    Coordinate monotonicity of the full tail bounds every descendant corner,
    and every endpoint tail, by the leaf's own corner plus ``guard``: one
    support omission plus twice the runtime's per-tail margin, which covers a
    computed tail's distance from its exact windowed sum. The runtime bound is
    a certified upper bound on the leaf's supremum, so every later endpoint is
    at most ``bound + guard``; a descendant's curvature term is at most a
    quarter of the leaf's, so every descendant quadratic bound is at most
    ``bound + guard + term / 4``.
    """
    later = np.minimum(mono + guard, bound + guard + term * (0.25 * _TERM_GROWTH))
    return np.maximum(bound, later)


def _settle(
    status: np.ndarray,
    act: np.ndarray,
    best: np.ndarray,
    lv: _Leaves,
    batch: _Batch,
    rows: np.ndarray,
    leaves: int,
) -> None:
    """Boolean exits: an achieved endpoint at or above the tail allocation
    already fixes non-rejection (the reported value never falls below it);
    every current leaf's reachable bound below it fixes rejection. Each row of
    *act* holds *leaves* leaves."""
    decision = batch.decision
    beta, u_alpha = decision.beta, decision.tail_alpha
    dl = batch.delta[rows[act]]
    accept = beta + best[act] - dl >= u_alpha
    status[act[accept]] = _Outcome.ACCEPT
    reach = lv.reach[:, :leaves][act].max(axis=1)
    reject = ~accept & (beta + np.maximum(reach, best[act]) + dl < u_alpha)
    status[act[reject]] = _Outcome.REJECT


def _leaf_bounds(
    fu: np.ndarray,
    fv: np.ndarray,
    mono_raw: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    decision: BinomialDecision,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(bound, raw bound, mono, term)`` of leaves, as ``_certified_sup``'s
    ``bound`` forms them."""
    mono = np.minimum(1.0, mono_raw)
    imax = _curvature(u, v, decision.null_ratio, decision.n_c, decision.n_t)
    term = _quad_term(imax, u, v)
    finite = np.isfinite(imax)
    quad = np.maximum(fu, fv) + term
    bound = np.where(finite, np.minimum(1.0, np.minimum(mono, quad)), mono)
    raw = np.where(finite, np.minimum(mono_raw, quad), mono_raw)
    degenerate = v <= u
    if degenerate.any():
        top = np.maximum(fu, fv)
        bound = np.where(degenerate, top, bound)
        raw = np.where(degenerate, top, raw)
        mono = np.where(degenerate, top, mono)
        term = np.where(degenerate, 0.0, term)
    return bound, raw, mono, term


@dataclass(slots=True)
class _Search:
    """The rows of one replay still being searched, with their leaves; every array is indexed
    by position in ``rows``, the batch row each of them replays."""

    rows: np.ndarray
    lv: _Leaves
    best: np.ndarray
    delta: np.ndarray
    guard: np.ndarray
    status: np.ndarray
    final: np.ndarray

    def take(self, ids: np.ndarray) -> _Search:
        """A copy of the search rows *ids*, so a chunk can deepen on its own."""
        return _Search(
            self.rows[ids],
            self.lv.take(ids, capacity=self.lv.u.shape[1]),
            self.best[ids],
            self.delta[ids],
            self.guard[ids],
            self.status[ids],
            self.final[ids],
        )


def _finish(search: _Search, done: np.ndarray, top: np.ndarray, beta: float) -> None:
    best = search.best[done]
    search.final[done] = np.minimum(1.0, beta + np.minimum(1.0, np.maximum(top, best)))
    search.status[done] = _Outcome.FINISHED


def _advance(
    batch: _Batch,
    tails: _ExactTails | _SurrogateTails,
    search: _Search,
    splits: range,
    *,
    exact: bool,
    rule: _rr._StopRule,
) -> None:
    """Runs the search iterations *splits* of every row still searching, in place."""
    decision = batch.decision
    beta = decision.beta
    lv, rows, best, delta, guard = search.lv, search.rows, search.best, search.delta, search.guard
    status = search.status
    for iteration in splits:
        act = np.flatnonzero(status == _Outcome.SEARCHING)
        if act.size == 0:
            break
        if iteration + 2 > lv.u.shape[1]:
            lv.grow(min(splits.stop + 1, 2 * lv.u.shape[1]))  # no wider than the stage needs
        leaves = iteration + 1  # every row still searching has split once per past iteration
        bnd = lv.bound[:, :leaves][act]
        top = bnd.max(axis=1)
        k = np.argmin(np.where(bnd == top[:, None], lv.u[:, :leaves][act], np.inf), axis=1)
        lower = np.maximum(0.0, best[act] - guard[act])
        gap = np.maximum(top, best[act]) - lower
        allowed = rule.gap_fraction * np.maximum(beta + lower, decision.tail_alpha) + guard[act]
        stop = gap <= allowed
        if exact:
            dl = delta[act][:, None]
            others = bnd.copy()
            others[np.arange(act.size), k] = -np.inf
            runner = others.max(axis=1)
            capped_ties = np.all((bnd != 1.0) | (lv.raw[:, :leaves][act] >= 1.0 + dl), axis=1)
            order = (top - runner <= 2.0 * dl[:, 0]) & ~(
                (top == 1.0) & (runner == 1.0) & capped_ties
            )
            unsettled = (order & ~stop) | (
                np.abs(gap - allowed) <= (2.0 + rule.gap_fraction) * dl[:, 0] + 4.0 * _EPS
            )
            status[act[unsettled]] = _Outcome.AMBIGUOUS
            act, k, top, stop = act[~unsettled], k[~unsettled], top[~unsettled], stop[~unsettled]
        _finish(search, act[stop], top[stop], beta)
        act, k, top = act[~stop], k[~stop], top[~stop]
        u = lv.u[act, k]
        v = lv.v[act, k]
        mid = (u + v) / 2.0
        floor = (mid <= u) | (mid >= v)
        _finish(search, act[floor], top[floor], beta)
        act, k, u, v, mid = act[~floor], k[~floor], u[~floor], v[~floor], mid[~floor]
        if act.size == 0:
            continue
        pu, pv = lv.pu[act, k], lv.pv[act, k]
        fm_raw, left_raw, right_raw, pm = tails.split(batch, rows[act], u, mid, v, pu, pv)
        fm = np.minimum(1.0, fm_raw)
        fu, fv = lv.fu[act, k], lv.fv[act, k]
        best[act] = np.maximum(best[act], fm)
        # The left child replaces the split leaf, the right child takes the next slot.
        owner = np.concatenate([act, act])
        slot = np.concatenate([k, lv.count[act]])
        cu, cv = np.concatenate([u, mid]), np.concatenate([mid, v])
        cfu, cfv = np.concatenate([fu, fm]), np.concatenate([fm, fv])
        lv.u[owner, slot], lv.v[owner, slot] = cu, cv
        lv.pu[owner, slot], lv.pv[owner, slot] = np.concatenate([pu, pm]), np.concatenate([pm, pv])
        lv.fu[owner, slot], lv.fv[owner, slot] = cfu, cfv
        leaf = _leaf_bounds(cfu, cfv, np.concatenate([left_raw, right_raw]), cu, cv, decision)
        lv.bound[owner, slot], lv.raw[owner, slot] = leaf[0], leaf[1]
        lv.reach[owner, slot] = tails.reach(rows[owner], cu, cv, leaf[0], leaf[2], leaf[3])
        lv.count[act] += 1
        _settle(status, act, best, lv, batch, rows, leaves + 1)


def _finish_unsettled(search: _Search, beta: float) -> None:
    """Rows still searching after their last split report their largest remaining bound."""
    rest = np.flatnonzero(search.status == _Outcome.SEARCHING)
    if rest.size:
        lv = search.lv
        leaves = int(lv.count[rest].max())
        _finish(search, rest, lv.bound[:, :leaves][rest].max(axis=1), beta)


# One vectorized pass mirrors the runtime's single heap loop step for step.
def _replay(batch: _Batch, tails: _ExactTails | _SurrogateTails, *, exact: bool) -> np.ndarray:
    """Outcome per row: ACCEPT, REJECT, or (exact route) AMBIGUOUS.

    Replays ``binomial_rr._certified_sup`` for every row at once. Each step
    takes the leaf with the largest bound, ties to the smaller left endpoint
    (the heap's tuple order for disjoint leaves); stops when the gap to the
    best evaluated point, deflated as ``tail_lower_enclosure`` deflates it,
    is within ``NUISANCE_STOP.gap_fraction`` of the larger of ``beta`` plus that deflated
    value and the tail allocation, plus the certification noise (the deflation itself),
    at the floating-point floor or after ``NUISANCE_STOP.max_iter``
    splits (read when the replay runs); and reports ``min(1, beta + min(1,
    max(remaining bounds, best endpoint)))``. The runtime's test after its
    last split only sets its ``stopped`` flag, never the reported value. On
    the exact route a comparison within the summation allowance marks the row
    AMBIGUOUS.

    The approximate route runs the first ``_COMMON_SPLITS`` splits of every row together;
    rows still searching then continue in chunks sized for the full split cap
    (`_chunk_rows`), so the few rows that need deep searches never keep the whole batch's
    leaf arrays at that depth. The root's arrays are released once the searching rows have
    copied theirs, the chunks' copies are taken before the first stage's arrays are released,
    and a chunk deepens beside the copies still waiting; `_batch_rows` and `_chunk_rows` size
    the stages so that none of those moments holds more than the leaf budget, whatever share
    of the rows survive. The exact route keeps one pass: its tails register each nuisance
    point once for the treatment counts of the rows then active, so rows may leave a search
    but never join one.
    """
    decision = batch.decision
    beta, u_alpha = decision.beta, decision.tail_alpha
    n_rows = batch.j.size
    outcome = np.full(n_rows, _Outcome.SEARCHING, np.int64)
    if n_rows == 0:
        return outcome
    groups = batch.groups
    a = groups.a[batch.group]
    hi = groups.hi[batch.group]
    fa_raw, fb_raw, mono_raw = tails.root(batch)
    fa = np.minimum(1.0, fa_raw)
    fb = np.minimum(1.0, fb_raw)
    lv = _Leaves.empty(n_rows, 1)
    lv.u[:, 0], lv.v[:, 0] = a, hi
    lv.pu[:, 0], lv.pv[:, 0] = tails.root_points(batch)
    lv.fu[:, 0], lv.fv[:, 0] = fa, fb
    bounds = _leaf_bounds(fa, fb, mono_raw, a, hi, decision)
    lv.bound[:, 0], lv.raw[:, 0] = bounds[0], bounds[1]
    everyone = np.arange(n_rows)
    lv.reach[:, 0] = tails.reach(everyone, a, hi, bounds[0], bounds[2], bounds[3])
    best = np.maximum(fa, fb)
    _settle(outcome, everyone, best, lv, batch, everyone, 1)
    rows = np.flatnonzero(outcome == _Outcome.SEARCHING)
    if rows.size == 0:
        return outcome
    rule = _rr.NUISANCE_STOP
    search = _Search(
        rows,
        lv.take(rows, capacity=_START_SLOTS),
        best[rows],
        batch.delta[rows],
        batch.guard[rows],
        np.full(rows.size, _Outcome.SEARCHING, np.int64),
        np.full(rows.size, np.nan),
    )
    del lv  # the root's arrays are not needed again
    common = _common_splits(rule.max_iter, exact=exact)
    _advance(batch, tails, search, range(common), exact=exact, rule=rule)
    if common == rule.max_iter:
        _finish_unsettled(search, beta)
    else:
        survivors = np.flatnonzero(search.status == _Outcome.SEARCHING)
        size = _chunk_rows(rule.max_iter)
        chunks = [survivors[start : start + size] for start in range(0, survivors.size, size)]
        deeper: list[_Search | None] = [search.take(ids) for ids in chunks]
        search.lv = _Leaves.empty(0, 1)  # the first stage's arrays are not needed again
        for index, ids in enumerate(chunks):
            rest, deeper[index] = deeper[index], None  # release each chunk once it is done
            assert rest is not None
            _advance(batch, tails, rest, range(common, rule.max_iter), exact=exact, rule=rule)
            _finish_unsettled(rest, beta)
            search.status[ids], search.final[ids] = rest.status, rest.final
    status, final = search.status, search.final
    finished = status == _Outcome.FINISHED
    if exact:
        close = finished & (np.abs(final - u_alpha) <= search.delta + 4.0 * _EPS)
        status[close] = _Outcome.AMBIGUOUS
        finished &= ~close
    status[finished] = np.where(final[finished] < u_alpha, _Outcome.REJECT, _Outcome.ACCEPT)
    outcome[rows] = status
    return outcome


# --- Tail sources ---------------------------------------------------------------


class _ExactTails:
    """The runtime's windowed joint-binomial tails, shared across count pairs.

    Every nuisance point a replay visits is registered per group once, with
    its treatment SF (plus) or CDF (minus) table over the thresholds the
    group's active rows can reach, and its control PMF over the group's
    summed support (shared by groups with the same support) -- the runtime's
    own special functions, elementwise.
    """

    def __init__(self, batch: _Batch) -> None:
        self.batch = batch
        groups = batch.groups
        self.n_c = batch.decision.n_c
        self.n_t = batch.decision.n_t
        self.r = batch.decision.null_ratio
        self.wmax = int(groups.width.max())
        span = (groups.j1 - groups.j0) + (groups.dmax - groups.dmin) + 1
        self.mmax = int(span.max())
        self.index: dict[tuple[int, float], int] = {}
        self.size = 0
        self.table = np.zeros((64, self.mmax))
        self.base = np.zeros(64, np.int64)
        self.point_pmf = np.zeros(64, np.int64)
        self.pmf_index: dict[tuple[int, int, float], int] = {}
        self.pmf = np.zeros((64, self.wmax))
        starts = np.searchsorted(batch.group, np.arange(groups.kind.size + 1))
        self.starts = starts
        self.root_ids = np.zeros((groups.kind.size, 2), np.int64)

    @staticmethod
    def _grown(array: np.ndarray, used: int, need: int) -> np.ndarray:
        if need <= array.shape[0]:
            return array
        grown = np.zeros((max(need, 2 * array.shape[0]), *array.shape[1:]), array.dtype)
        grown[:used] = array[:used]
        return grown

    def _pmf_rows(self, gg: np.ndarray, qq: np.ndarray) -> np.ndarray:
        """Control-PMF rows of new points, computing supports not seen yet."""
        groups = self.batch.groups
        rows = np.empty(qq.size, np.int64)
        fresh: list[int] = []
        used = len(self.pmf_index)
        for n, key in enumerate(
            zip(groups.wlo[gg].tolist(), groups.width[gg].tolist(), qq.tolist(), strict=True)
        ):
            row = self.pmf_index.get(key)
            if row is None:
                row = self.pmf_index[key] = used + len(fresh)
                fresh.append(n)
            rows[n] = row
        if fresh:
            sel = np.asarray(fresh)
            self.pmf = self._grown(self.pmf, used, used + sel.size)
            s = groups.wlo[gg[sel]][:, None] + np.arange(self.wmax)[None, :]
            inside = np.arange(self.wmax)[None, :] < groups.width[gg[sel]][:, None]
            pmf = _rr._fast_binom_pmf(s, self.n_c, qq[sel, None])
            self.pmf[used : used + sel.size] = np.where(inside, pmf, 0.0)
        return rows

    def _register(
        self, group: np.ndarray, q: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        """Point ids of ``(group, q)``, creating missing points whose tables
        cover treatment counts ``[lo, hi]`` of their group (one range per
        group per call)."""
        keys, first, inverse = np.unique(
            group.astype(np.float64) + 1j * q, return_index=True, return_inverse=True
        )
        ids = np.empty(keys.size, np.int64)
        fresh: list[int] = []
        for n, key in enumerate(zip(group[first].tolist(), q[first].tolist(), strict=True)):
            pid = self.index.get(key)
            if pid is None:
                pid = self.index[key] = self.size + len(fresh)
                fresh.append(n)
            ids[n] = pid
        if not fresh:
            return ids[inverse]
        sel = first[np.asarray(fresh)]
        count = sel.size
        need = self.size + count
        self.table = self._grown(self.table, self.size, need)
        self.base = self._grown(self.base, self.size, need)
        self.point_pmf = self._grown(self.point_pmf, self.size, need)
        new = slice(self.size, need)
        groups = self.batch.groups
        gg = group[sel]
        qq = q[sel]
        self.point_pmf[new] = self._pmf_rows(gg, qq)
        base = lo[sel] + groups.dmin[gg]
        span = int((hi[sel] - lo[sel] + groups.dmax[gg] - groups.dmin[gg]).max()) + 1
        m = base[:, None] + np.arange(span)[None, :]
        plus = groups.kind[gg] == 0
        p = np.where(plus, np.minimum(self.r * qq, 1.0), self.r * qq)
        table = np.empty((count, span))
        if plus.any():
            table[plus] = _rr._fast_binom_sf(m[plus], self.n_t, p[plus, None])
        if (~plus).any():
            table[~plus] = _rr._fast_binom_cdf(m[~plus], self.n_t, p[~plus, None])
        self.table[new, :span] = table
        self.base[new] = base
        self.size = need
        return ids[inverse]

    def _gather(self, rows: np.ndarray, qid: np.ndarray, tid: np.ndarray) -> np.ndarray:
        batch = self.batch
        g = batch.group[rows]
        idx = (batch.j[rows] - self.base[tid])[:, None] + batch.groups.offsets[g]
        idx += (tid * self.table.shape[1])[:, None]
        tails = np.take(self.table.reshape(-1), idx)
        values = np.einsum("ij,ij->i", self.pmf[self.point_pmf[qid]], tails)
        return (values + batch.groups.omitted[g]) + batch.groups.margin[g]

    def root(self, batch: _Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups = batch.groups
        count = groups.kind.size
        gid = np.arange(count)
        ids = self._register(
            np.concatenate([gid, gid]),
            np.concatenate([groups.a, groups.hi]),
            np.concatenate([groups.j0, groups.j0]),
            np.concatenate([groups.j1, groups.j1]),
        )
        self.root_ids[:, 0] = ids[:count]
        self.root_ids[:, 1] = ids[count:]
        fa = np.empty(batch.j.size)
        fb = np.empty(batch.j.size)
        mono = np.empty(batch.j.size)
        for n in range(count):
            start, stop = self.starts[n], self.starts[n + 1]
            if start == stop:
                continue
            ia, ib = self.root_ids[n]
            kernel_a = self._kernel(n, ia)
            kernel_b = self._kernel(n, ib)
            corner = (ia, ib) if groups.kind[n] == 0 else (ib, ia)
            fa[start:stop] = self._correlate(n, kernel_a, ia)
            fb[start:stop] = self._correlate(n, kernel_b, ib)
            mono[start:stop] = self._correlate(
                n, kernel_a if corner[0] == ia else kernel_b, corner[1]
            )
        return fa, fb, mono

    def reach(
        self,
        rows: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        bound: np.ndarray,
        mono: np.ndarray,
        term: np.ndarray,
    ) -> np.ndarray:
        del u, v
        return _eventual_bound(bound, mono, term, self.batch.guard[rows])

    def _kernel(self, g: int, pid: int) -> np.ndarray:
        groups = self.batch.groups
        width = int(groups.width[g])
        span = int(groups.dmax[g] - groups.dmin[g]) + 1
        d = groups.offsets[g, :width] - groups.dmin[g]
        pmf = self.pmf[self.point_pmf[pid], :width]
        if self.n_t >= self.n_c:
            kernel = np.zeros(span)
            kernel[d] = pmf
            return kernel
        return np.bincount(d, weights=pmf, minlength=span)

    def _correlate(self, g: int, kernel: np.ndarray, tid: int) -> np.ndarray:
        groups = self.batch.groups
        length = int(groups.j1[g] - groups.j0[g]) + int(groups.dmax[g] - groups.dmin[g]) + 1
        raw = np.correlate(self.table[tid, :length], kernel, "valid")
        return (raw + groups.omitted[g]) + groups.margin[g]

    def root_points(self, batch: _Batch) -> tuple[np.ndarray, np.ndarray]:
        ids = self.root_ids[batch.group]
        return ids[:, 0], ids[:, 1]

    def split(
        self,
        batch: _Batch,
        rows: np.ndarray,
        u: np.ndarray,
        mid: np.ndarray,
        v: np.ndarray,
        pu: np.ndarray,
        pv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del u, v
        g = batch.group[rows]
        j = batch.j[rows]
        order = np.lexsort((j, g))
        lo = np.empty(rows.size, np.int64)
        hi = np.empty(rows.size, np.int64)
        # Each group's still-active treatment range bounds every later use.
        gs = g[order]
        first = np.r_[True, gs[1:] != gs[:-1]]
        last = np.r_[gs[1:] != gs[:-1], True]
        seg = np.cumsum(first) - 1
        lo[order] = j[order][first][seg]
        hi[order] = j[order][last][seg]
        pm = self._register(g, mid, lo, hi)
        plus = batch.groups.kind[g] == 0
        fm = self._gather(rows, pm, pm)
        left = self._gather(rows, np.where(plus, pu, pm), np.where(plus, pm, pu))
        right = self._gather(rows, np.where(plus, pm, pv), np.where(plus, pv, pm))
        return fm, left, right, pm


class _SurrogateTails:
    """Continuity-corrected Normal tails of ``K = n_c X_t - n_t X_c``, exact
    single-binomial tails when an arm is deterministic, with the runtime's
    support-omission and margin allowances and cap."""

    def __init__(self, batch: _Batch) -> None:
        self.batch = batch
        decision = batch.decision
        self.n_c = decision.n_c
        self.n_t = decision.n_t
        self.r = decision.null_ratio
        self.h = math.gcd(self.n_c, self.n_t) / (self.n_c * self.n_t)

    def _tail(self, rows: np.ndarray, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        batch = self.batch
        groups = batch.groups
        g = batch.group[rows]
        n_c, n_t = self.n_c, self.n_t
        x_c = groups.x_c[g]
        j = batch.j[rows]
        plus = groups.kind[g] == 0
        k = n_c * j - n_t * x_c
        d = j / n_t - x_c / n_c
        mean = p - q
        var = p * (1.0 - p) / n_t + q * (1.0 - q) / n_c
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(plus, d - self.h / 2.0 - mean, d + self.h / 2.0 - mean) / np.sqrt(var)
            tail = np.where(plus, _ndtr(-z), _ndtr(z))
        det_t = (p <= 0.0) | (p >= 1.0)
        det_c = (q <= 0.0) | (q >= 1.0)
        if det_t.any() or det_c.any():
            tail = np.where(det_t | det_c, self._deterministic(k, q, p, plus, det_t, det_c), tail)
        return (tail + groups.omitted[g]) + groups.margin[g]

    def _deterministic(self, k, q, p, plus, det_t, det_c):
        """Exact tails with integer thresholds when an arm is a point mass."""
        n_c, n_t = self.n_c, self.n_t
        out = np.zeros(k.size)
        xt = np.where(p >= 1.0, n_t, 0)
        xc = np.where(q >= 1.0, n_c, 0)
        both = det_t & det_c
        stat = n_c * xt - n_t * xc
        out = np.where(both, np.where(plus, stat >= k, stat <= k).astype(float), out)
        only_t = det_t & ~det_c
        if only_t.any():
            # Plus: X_c <= floor((n_c xt - k) / n_t); minus: X_c >= ceil(...).
            num = n_c * xt - k
            floor_ = num // n_t
            ceil_ = -((-num) // n_t)
            with np.errstate(invalid="ignore"):
                cdf = _rr._fast_binom_cdf(floor_, n_c, q)
                sf = _rr._fast_binom_sf(ceil_ - 1, n_c, q)
            cdf = np.where(floor_ < 0, 0.0, np.where(floor_ >= n_c, 1.0, cdf))
            sf = np.where(ceil_ <= 0, 1.0, np.where(ceil_ > n_c, 0.0, sf))
            out = np.where(only_t, np.where(plus, cdf, sf), out)
        only_c = det_c & ~det_t
        if only_c.any():
            # Plus: X_t >= ceil((k + n_t xc) / n_c); minus: X_t <= floor(...).
            num = k + n_t * xc
            ceil_ = -((-num) // n_c)
            floor_ = num // n_c
            with np.errstate(invalid="ignore"):
                sf = _rr._fast_binom_sf(ceil_ - 1, n_t, p)
                cdf = _rr._fast_binom_cdf(floor_, n_t, p)
            sf = np.where(ceil_ <= 0, 1.0, np.where(ceil_ > n_t, 0.0, sf))
            cdf = np.where(floor_ < 0, 0.0, np.where(floor_ >= n_t, 1.0, cdf))
            out = np.where(only_c, np.where(plus, sf, cdf), out)
        return out

    def _p(self, rows: np.ndarray, q: np.ndarray) -> np.ndarray:
        plus = self.batch.groups.kind[self.batch.group[rows]] == 0
        return np.where(plus, np.minimum(self.r * q, 1.0), self.r * q)

    def root(self, batch: _Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.arange(batch.j.size)
        groups = batch.groups
        a = groups.a[batch.group]
        hi = groups.hi[batch.group]
        plus = groups.kind[batch.group] == 0
        fa = self._tail(rows, a, self._p(rows, a))
        fb = self._tail(rows, hi, self._p(rows, hi))
        mono = self._tail(
            rows, np.where(plus, a, hi), np.where(plus, self._p(rows, hi), self._p(rows, a))
        )
        return fa, fb, mono

    def reach(
        self,
        rows: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        bound: np.ndarray,
        mono: np.ndarray,
        term: np.ndarray,
    ) -> np.ndarray:
        """Largest surrogate value the replay can later report from inside
        each leaf ``[u, v]`` of batch row ``rows``: interval bounds of the
        Normal tail over the leaf's nuisance rectangle (every descendant
        corner) and along its null path (every endpoint, hence every
        descendant quadratic bound up to a quarter of the leaf's curvature
        term). Leaves touching a deterministic arm are left unbounded."""
        del mono
        batch = self.batch
        groups = batch.groups
        g = batch.group[rows]
        n_c, n_t, r = self.n_c, self.n_t, self.r
        plus = groups.kind[g] == 0
        d = batch.j[rows] / n_t - groups.x_c[g] / n_c
        c = np.where(plus, d - self.h / 2.0, d + self.h / 2.0)
        usable = (u > 0.0) & (v < 1.0) & (r * v < 1.0)
        u = np.where(usable, u, 0.25)
        v = np.where(usable, v, 0.5)
        pu, pv = r * u, r * v

        def spread(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            at_lo, at_hi = lo * (1.0 - lo), hi * (1.0 - hi)
            top = np.where((lo <= 0.5) & (hi >= 0.5), 0.25, np.maximum(at_lo, at_hi))
            return np.minimum(at_lo, at_hi), top

        q_min, q_max = spread(u, v)
        p_min, p_max = spread(pu, pv)
        rect_lo = p_min / n_t + q_min / n_c
        rect_hi = p_max / n_t + q_max / n_c
        alpha_ = r / n_t + 1.0 / n_c
        gamma_ = r * r / n_t + 1.0 / n_c

        def path(q: np.ndarray) -> np.ndarray:
            return alpha_ * q - gamma_ * q * q

        path_hi = path(np.clip(alpha_ / (2.0 * gamma_), u, v))
        path_lo = np.minimum(path(u), path(v))

        def extreme(num: np.ndarray, var_lo: np.ndarray, var_hi: np.ndarray, *, low: bool):
            # Extreme of num / sqrt(var) over var in [var_lo, var_hi].
            favours_hi = num >= 0.0 if low else num <= 0.0
            return num / np.sqrt(np.where(favours_hi, var_hi, var_lo))

        with np.errstate(divide="ignore", invalid="ignore"):
            # Plus tail Phi(-z): bound z below. Minus tail Phi(z): bound z above.
            rect_plus = extreme(c - pv + u, rect_lo, rect_hi, low=True)
            rect_minus = extreme(c - pu + v, rect_lo, rect_hi, low=False)
            slope = c - (r - 1.0) * np.where(r >= 1.0, v, u)
            slope_minus = c - (r - 1.0) * np.where(r >= 1.0, u, v)
            path_plus = extreme(slope, path_lo, path_hi, low=True)
            path_minus = extreme(slope_minus, path_lo, path_hi, low=False)
        corner = np.where(plus, _ndtr(-rect_plus), _ndtr(rect_minus))
        along = np.where(plus, _ndtr(-path_plus), _ndtr(path_minus))
        allow = (groups.omitted[g] + groups.margin[g]) + _SURROGATE_SLACK
        later = np.minimum(corner, along + term * (0.25 * _TERM_GROWTH)) + allow
        later = np.where(np.isnan(later), np.inf, later)
        return np.where(usable, np.maximum(bound, later), np.inf)

    def root_points(self, batch: _Batch) -> tuple[np.ndarray, np.ndarray]:
        zero = np.zeros(batch.j.size, np.int64)
        return zero, zero

    def split(self, batch, rows, u, mid, v, pu, pv):
        del pu, pv
        plus = batch.groups.kind[batch.group[rows]] == 0
        three = np.concatenate([rows, rows, rows])
        p_mid, p_u, p_v = np.split(self._p(three, np.concatenate([mid, u, v])), 3)
        q = np.concatenate([mid, np.where(plus, u, mid), np.where(plus, mid, v)])
        p = np.concatenate([p_mid, np.where(plus, p_mid, p_u), np.where(plus, p_v, p_mid)])
        fm, left, right = np.split(self._tail(three, q, p), 3)
        return fm, left, right, np.zeros(rows.size, np.int64)


# --- Classification ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Request:
    x_c: int
    kind: Kind
    j0: int
    j1: int


def _runtime_rejects(decision: BinomialDecision, kind: Kind, x_c: int, x_t: int) -> bool:
    """The unchanged runtime decision of one directional test, read against its tail level."""
    test = _rr.p_plus if kind == "plus" else _rr.p_minus
    try:
        p = test(
            decision.null_ratio,
            x_c,
            decision.n_c,
            x_t,
            decision.n_t,
            decision.beta,
            tail=decision.tail_alpha,
        )
    except _rr.BinomialDataError:
        return False
    return p < decision.tail_alpha


#: Bytes of leaf arrays (`_Leaves`) one replay may hold at once, whatever the data: every
#: `_Leaves` alive together, counting the field `_Leaves.grow` holds beside its replacement.
#: A search's other arrays (its per-row bookkeeping, one iteration's temporaries) are not counted.
_LEAF_BUDGET_BYTES = 350e6
#: Splits the approximate route runs for every row of a batch together.
_COMMON_SPLITS = 63
#: Leaf slots a search's arrays start with; `_advance` widens them only as far as its splits need.
_START_SLOTS = 8


def _stage_slots(splits: int) -> int:
    """Leaf slots a search's arrays hold after a stage of *splits* splits: a row holds
    ``splits + 1`` leaves after that many splits, and the arrays never start narrower than
    `_START_SLOTS`."""
    return max(_START_SLOTS, splits + 1)


def _leaf_bytes(slots: int) -> int:
    """Bytes of one row's leaf arrays of *slots* slots: an eight-byte value per field
    (`_Leaves._FIELDS`) and slot, and the row's eight-byte count."""
    return 8 * (len(_Leaves._FIELDS) * slots + 1)


def _widening_bytes(slots: int) -> int:
    """Most bytes of one row's leaf arrays while `_Leaves.grow` widens them to *slots* slots.
    `grow` replaces one field at a time, so beside the fields it has widened it holds the one
    it is replacing, which is narrower than its replacement."""
    return _leaf_bytes(slots) + 8 * slots


def _common_splits(max_iter: int, *, exact: bool) -> int:
    """Splits every row of a batch runs together before any continue alone: all *max_iter* on the
    exact route, whose rows are never copied, at most ``_COMMON_SPLITS`` on the approximate."""
    return max_iter if exact else min(max_iter, _COMMON_SPLITS)


def _chunk_rows(max_iter: int) -> int:
    """Rows of a chunk the approximate route deepens to *max_iter* splits while the copies of
    the other chunks wait: half the leaf budget, the copies keeping the other half
    (`_batch_rows`)."""
    return max(1, int(_LEAF_BUDGET_BYTES // 2 // _widening_bytes(_stage_slots(max_iter))))


def _batch_rows(*, exact: bool) -> int:
    """Rows per classification batch, so the replay's leaf arrays stay within the leaf budget
    however many of its rows keep searching.

    A one-stage replay (the exact route, or an approximate one capped at its common splits)
    peaks either while copying the root into search arrays or widening to the last width.
    A two-stage replay peaks in one of two moments. Having copied the rows still searching
    beside the first stage's arrays, which the copies equal when every row survives, it holds both and the
    one field `_Leaves.take` gathers of the chunk it is copying: the batch takes half the
    budget less that gather. Then the copies wait beside a chunk that deepens, which takes the
    other half (`_chunk_rows`). The moments before these, the root's arrays beside the first
    rows' and the first stage widening, hold less."""
    max_iter = _rr.NUISANCE_STOP.max_iter
    common = _common_splits(max_iter, exact=exact)
    if common == max_iter:
        slots = _stage_slots(max_iter)
        root_copy = _leaf_bytes(1) + _leaf_bytes(_START_SLOTS) + 8
        return max(1, int(_LEAF_BUDGET_BYTES // max(root_copy, _widening_bytes(slots))))
    first = _stage_slots(common)
    gather = 8 * first * _chunk_rows(max_iter)
    return max(1, int((_LEAF_BUDGET_BYTES - gather) // (2 * _leaf_bytes(first))))


def classify(
    decision: BinomialDecision, route: Route, requests: Sequence[_Request]
) -> list[np.ndarray]:
    """Directional rejection masks for each request's treatment-count range.

    Every count pair is decided by its own replay, with one certified
    exception on the approximate route: a count the replay's first step would
    settle (its root non-rejection or rejection exit) is given that exit when
    a neighbouring count's computed margin, through the step's monotonicity in
    the treatment count, forces it (``_root_settled``; it assumes each
    computed tail lies within ``_ROOT_ROUNDING`` of its exact-arithmetic
    value). Every other count is replayed."""
    results: list[np.ndarray | None] = []
    slots: list[list[int]] = []
    live: list[tuple[int, _Request, float, float, tuple[int, int, float]]] = []
    pending = 0
    batch_rows = _batch_rows(exact=route == "exact")
    for req in requests:
        size = req.j1 - req.j0 + 1
        try:
            a, b = _rr.clopper_pearson(req.x_c, decision.n_c, decision.beta)
        except _rr.BinomialDataError:
            # The runtime row cannot be built: a decision failure, never a rejection.
            results.append(np.zeros(size, bool))
            slots.append([len(results) - 1])
            continue
        hi = b
        if req.kind == "minus":
            hi = b if decision.null_ratio <= 0.0 else min(b, 1.0 / decision.null_ratio)
            if hi < a:
                # Empty nuisance domain: the runtime reports beta alone.
                results.append(np.full(size, min(1.0, decision.beta) < decision.tail_alpha))
                slots.append([len(results) - 1])
                continue
        window = _rr._support_window(decision.n_c, a, hi)
        mine: list[int] = []
        # A request larger than a batch is replayed in pieces of at most a batch: each count
        # pair has its own replay, so a piece decides exactly as the whole would.
        for j0 in range(req.j0, req.j1 + 1, batch_rows):
            piece = _Request(req.x_c, req.kind, j0, min(req.j1, j0 + batch_rows - 1))
            rows = piece.j1 - piece.j0 + 1
            if live and pending + rows > batch_rows:
                _classify_live(decision, route, live, results)
                live, pending = [], 0
            results.append(None)
            mine.append(len(results) - 1)
            live.append((len(results) - 1, piece, a, hi, window))
            pending += rows
        slots.append(mine)
    if live:
        _classify_live(decision, route, live, results)
    return [
        np.concatenate([results[slot] for slot in mine]) if mine else np.zeros(0, bool)
        for mine in slots
    ]


def _classify_live(decision, route, live, results) -> None:
    n_c, n_t = decision.n_c, decision.n_t
    count = len(live)
    exact = route == "exact"
    wlo = np.array([w[0] for *_, w in live], np.int64)
    whi = np.array([w[1] for *_, w in live], np.int64)
    a = np.array([item[2] for item in live])
    hi = np.array([item[3] for item in live])
    kinds = [item[1].kind for item in live]
    x_c = np.array([item[1].x_c for item in live], np.int64)
    widths = whi - wlo + 1
    tmax = int(widths.max()) if exact else 1
    offsets = np.zeros((count, tmax), np.int64)
    dmin = np.zeros(count, np.int64)
    dmax = np.zeros(count, np.int64)
    if exact:
        for n in range(count):
            s = np.arange(wlo[n], whi[n] + 1, dtype=np.int64)
            d = _threshold_offsets(kinds[n], n_c, n_t, int(x_c[n]), s)
            offsets[n, : d.size] = d
            dmin[n], dmax[n] = d[0], d[-1]
    groups = _Groups(
        kind=np.array([0 if kind == "plus" else 1 for kind in kinds], np.int64),
        x_c=x_c,
        a=a,
        hi=hi,
        wlo=wlo,
        width=widths,
        omitted=np.array([w[2] for *_, w in live]),
        margin=np.array([_rr._eps_margin(int(w), n_c, n_t) for w in widths]),
        j0=np.array([item[1].j0 for item in live], np.int64),
        j1=np.array([item[1].j1 for item in live], np.int64),
        dmin=dmin,
        dmax=dmax,
        offsets=offsets,
    )
    sizes = groups.j1 - groups.j0 + 1
    group = np.repeat(np.arange(count), sizes)
    j = np.arange(group.size) - np.repeat(np.cumsum(sizes) - sizes, sizes) + groups.j0[group]
    settled = np.zeros(j.size, bool)
    reject = np.zeros(j.size, bool)
    if not exact:
        settled, reject = _root_settled(decision, groups, group, j)
    band = np.flatnonzero(~settled)
    if exact:
        # Two summation orders of ``width`` nonnegative products (each at most one, total at most
        # two) are each within ``(width - 1) * eps / 2`` of the exact total, so differ by at most
        # ``width * eps``; adding the omitted mass and margin rounds once each. Both orders sum
        # the runtime's own support window.
        delta = ((2.0 * widths + 8.0) * _EPS)[group[band]]
    else:
        delta = np.zeros(band.size)
    guard = (groups.omitted + 2.0 * groups.margin)[group[band]]
    batch = _Batch(decision, groups, group[band], j[band], delta, guard)
    tails = _ExactTails(batch) if exact else _SurrogateTails(batch)
    outcome = _replay(batch, tails, exact=exact)
    reject[band] = outcome == _Outcome.REJECT
    for row in np.flatnonzero(outcome == _Outcome.AMBIGUOUS):
        req = live[batch.group[row]][1]
        reject[band[row]] = _runtime_rejects(decision, req.kind, req.x_c, int(batch.j[row]))
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    for n, (index, *_rest) in enumerate(live):
        results[index] = reject[bounds[n] : bounds[n + 1]]


#: Floor of the gap between the replay's computed root tails and reachable bound and their exact
#: values: a few roundings of arguments below 1e3 through ``ndtr`` (derivative at most one). A
#: point-mass arm's tails use the guarded primitives, whose error grows with the arm, so
#: `_root_settled` raises this to the decision's margin and infers a root exit only beyond twice it.
_ROOT_ROUNDING = 5e-11


def _root_margins(
    decision: BinomialDecision, groups: _Groups, g: np.ndarray, j: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The approximate replay's two root exits at ``(group, count)`` as
    margins: ``beta + best - u`` (non-rejection when ``>= 0``) and ``u -
    beta - max(reach, best)`` (rejection when ``> 0``), computed exactly as
    ``_replay`` computes them before its first split."""
    zeros = np.zeros(j.size)
    batch = _Batch(decision, groups, g, j, zeros, zeros)
    tails = _SurrogateTails(batch)
    fa_raw, fb_raw, mono_raw = tails.root(batch)
    fa, fb = np.minimum(1.0, fa_raw), np.minimum(1.0, fb_raw)
    a, hi = groups.a[g], groups.hi[g]
    bound, _, mono, term = _leaf_bounds(fa, fb, mono_raw, a, hi, decision)
    reach = tails.reach(np.arange(j.size), a, hi, bound, mono, term)
    best = np.maximum(fa, fb)
    beta, u_alpha = decision.beta, decision.tail_alpha
    return beta + best - u_alpha, u_alpha - (beta + np.maximum(reach, best))


def _root_settled(
    decision: BinomialDecision, groups: _Groups, group: np.ndarray, j: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(settled, reject)`` masks over the cells ``(group, j)`` whose
    approximate-route decision is fixed by the replay's own root exits,
    inferred without replaying them.

    For a fixed control count the surrogate's root endpoint tails, its
    coordinate corner and its root reachable bound are, in exact arithmetic,
    each a monotone function of the treatment count: every term is a Normal
    or binomial tail whose standardized argument (or integer threshold) moves
    with ``j`` alone, nonincreasing in ``j`` for plus and nondecreasing for
    minus, and ``min``/``max``/sums of such functions keep that direction. So
    the non-rejection margin is monotone one way and the rejection margin
    the other. Bisection over ``j`` finds, for each group, an evaluated count
    whose computed margin is at least ``2 * _ROOT_ROUNDING`` (non-rejection)
    or strictly above it (rejection); every count on its far side then has
    exact margin at least (above) ``_ROOT_ROUNDING`` and computed margin at
    least zero (above zero), matching the replay's own exits -- non-rejection
    at a margin of zero or more, rejection only at a strictly positive one --
    so the replay there would take the same root exit.
    Counts not reached this way are replayed. Rows where both inferred sets
    would meet (impossible in exact arithmetic) are replayed in full.
    """
    count = groups.kind.size
    lo, hi = groups.j0, groups.j1
    rows = np.arange(count)
    plus = groups.kind == 0
    need = 2.0 * max(_ROOT_ROUNDING, _rr._eps_margin(1, decision.n_c, decision.n_t))

    def edge(which: int, upper: np.ndarray) -> np.ndarray:
        """Per group, the evaluated count nearest the other end at which
        margin ``which`` reaches ``need`` (non-rejection) or exceeds it
        (rejection), for a margin true on the upper (``upper``) or lower set;
        a sentinel outside the range if none."""
        true_end = np.where(upper, hi + 1, lo - 1)
        false_end = np.where(upper, lo - 1, hi + 1)
        active = np.abs(true_end - false_end) > 1
        while active.any():
            idx = rows[active]
            mid = (true_end[idx] + false_end[idx]) // 2
            margin = _root_margins(decision, groups, idx, mid)[which]
            holds = margin > need if which == 1 else margin >= need
            true_end[idx] = np.where(holds, mid, true_end[idx])
            false_end[idx] = np.where(holds, false_end[idx], mid)
            active = np.abs(true_end - false_end) > 1
        return true_end

    accept_edge = edge(0, ~plus)
    reject_edge = edge(1, plus)
    at_plus = plus[group]
    accepted = np.where(at_plus, j <= accept_edge[group], j >= accept_edge[group])
    rejected = np.where(at_plus, j >= reject_edge[group], j <= reject_edge[group])
    clash = np.zeros(count, bool)
    np.logical_or.at(clash, group, accepted & rejected)
    accepted &= ~clash[group]
    rejected &= ~clash[group]
    return accepted | rejected, rejected


# --- Geometry, integration, and routing -----------------------------------------


@dataclass(frozen=True, slots=True)
class _Window:
    """Outer count window of ``Bin(n, p)`` with its PMF and measured omitted mass."""

    lo: int
    hi: int
    omitted: float
    weights: np.ndarray

    @property
    def size(self) -> int:
        return self.hi - self.lo + 1


def _window_bounds(n: int, p: float) -> tuple[int, int]:
    if p <= 0.0:
        return 0, 0
    if p >= 1.0:
        return n, n
    return max(0, int(_binom.ppf(_OUTER_TAIL, n, p))), min(n, int(_binom.isf(_OUTER_TAIL, n, p)))


def _window(n: int, p: float) -> _Window:
    lo, hi = _window_bounds(n, p)
    if p <= 0.0 or p >= 1.0:
        return _Window(lo, hi, 0.0, np.ones(1))
    below = float(_rr._fast_binom_cdf(np.asarray(lo - 1), n, p)) if lo > 0 else 0.0
    above = float(_rr._fast_binom_sf(np.asarray(hi), n, p)) if hi < n else 0.0
    weights = _rr._fast_binom_pmf(np.arange(lo, hi + 1), n, p)
    return _Window(lo, hi, below + above, weights)


def solver_floor() -> float:
    """The nuisance budget below which the runtime's Clopper-Pearson endpoint solver refuses."""
    return _rr._CP_BETA_FLOOR


def solver_refuses(decision: BinomialDecision) -> bool:
    """Whether the nuisance budget is below `solver_floor`, which refuses the control arm's
    Clopper-Pearson interval at every count once it has two or more units (only ``n = 1`` has
    closed-form endpoints)."""
    return decision.beta < solver_floor() and decision.n_c > 1


def refused(decision: BinomialDecision) -> bool:
    """Whether the runtime refuses every count pair of this decision, so none rejects: an arm
    above the finite-sample ceiling, a nuisance budget below the solver's floor, or a tail
    level the float margin dominates (`binomial_rr.margin_dominates_tail`, which
    `confidence_interval` applies)."""
    if max(decision.n_c, decision.n_t) > _rr.FINITE_SAMPLE_MAX_ARM_SIZE or solver_refuses(decision):
        return True
    return _rr.margin_dominates_tail(decision.tail_alpha, decision.beta, decision.n_c, decision.n_t)


def window_cells(decision: BinomialDecision, p_c: float, p_t: float | None = None) -> int:
    """Retained (control, treatment) cells of the windows at control rate ``p_c`` and treatment
    rate ``p_t`` (the null rate when omitted), whether or not the runtime refuses the decision:
    the rectangle an evaluation at those rates classifies. It grows with the arm sizes up to
    the integer edges of its windows; `replay_cells` drops to zero once the runtime refuses."""
    lo_c, hi_c = _window_bounds(decision.n_c, p_c)
    rate = min(1.0, decision.null_ratio * p_c) if p_t is None else p_t
    lo_t, hi_t = _window_bounds(decision.n_t, rate)
    return (hi_c - lo_c + 1) * (hi_t - lo_t + 1)


class ReplayBoundExceeded(Exception):
    """An evaluation would leave a geometry storing ``cells`` count cells, beyond its bound.
    ``p_t`` is the evaluation's treatment rate, when it has one."""

    def __init__(self, cells: int, p_t: float | None = None) -> None:
        super().__init__(cells)
        self.cells = cells
        self.p_t = p_t


def replay_cells(decision: BinomialDecision, p_c: float) -> int:
    """Cells the planner replays: `window_cells`, or zero for a decision the runtime refuses in
    full, which is not replayed."""
    return 0 if refused(decision) else window_cells(decision, p_c)


def route_for(cells: int) -> Route:
    """Deterministic route of a geometry with ``cells`` retained cells at the null rate: exact
    within the cell budget, else approximate."""
    return "exact" if cells <= EXACT_CELL_BUDGET else "approximate"


@dataclass(frozen=True, slots=True)
class BinomialPower:
    """Rejection probability at one alternative, split into the part from
    plus-direction rejections (nondecreasing in the treatment rate) and the
    remainder from minus-direction rejections (nonincreasing in it).
    ``power`` sums the retained cells; ``omitted`` bounds what the windows
    left out, so the runtime's rejection probability lies in ``[power, power
    + omitted]``."""

    plus: float
    minus: float
    omitted: float

    @property
    def power(self) -> float:
        return min(1.0, self.plus + self.minus)


@dataclass(slots=True)
class _Segment:
    """Classified cells over one contiguous run of treatment counts
    ``[j0, j0 + cols)``, for every stored control count."""

    j0: int
    plus: np.ndarray
    minus: np.ndarray
    known: np.ndarray

    @property
    def j1(self) -> int:
        return self.j0 + self.plus.shape[1] - 1


class RejectionGeometry:
    """The classified rejection set of one runtime decision, grown on demand.

    Control counts ``[x0, x0 + rows)`` index every block; treatment counts
    are stored in disjoint column segments, merged whenever a request
    overlaps or touches them, so distant windows (a rate near one, say)
    never force the counts between them to be classified. The stored cells
    (every row by every segment's columns) never exceed ``max_cells``: a request
    that would exceed it raises `ReplayBoundExceeded` before anything is allocated.
    """

    def __init__(
        self, decision: BinomialDecision, route: Route, max_cells: int = PLANNING_CELL_CEILING
    ) -> None:
        self.decision = decision
        self.route = route
        self.max_cells = max_cells
        self.refused = refused(decision)
        self.x0 = 0
        self.rows = 0
        self.segments: list[_Segment] = []
        # Effect searches already solved on this geometry, keyed by their
        # control rate, compliance, and target: a curve's companion effects.
        self.effects: dict[tuple[float, float, float], object] = {}

    def _row_span(self, x_lo: int, x_hi: int) -> tuple[int, int]:
        """First and last control count of the stored rows extended to ``[x_lo, x_hi]``."""
        if not self.rows:
            return x_lo, x_hi
        return min(self.x0, x_lo), max(self.x0 + self.rows - 1, x_hi)

    def _touching(self, j_lo: int, j_hi: int) -> list[_Segment]:
        """The segments that overlap or touch ``[j_lo, j_hi]``."""
        return [s for s in self.segments if s.j0 <= j_hi + 1 and s.j1 >= j_lo - 1]

    def _stored_after(self, x_lo: int, x_hi: int, j_lo: int, j_hi: int) -> int:
        """Cells stored once ``[x_lo, x_hi] x [j_lo, j_hi]`` is covered: every segment spans all
        rows, and the one covering the request absorbs the segments it touches."""
        x0, x1 = self._row_span(x_lo, x_hi)
        touching = self._touching(j_lo, j_hi)
        merged = max([j_hi, *(s.j1 for s in touching)]) - min([j_lo, *(s.j0 for s in touching)]) + 1
        apart = sum(s.j1 - s.j0 + 1 for s in self.segments if all(s is not t for t in touching))
        return (x1 - x0 + 1) * (apart + merged)

    def _cover_rows(self, x_lo: int, x_hi: int) -> None:
        x0, x1 = self._row_span(x_lo, x_hi)
        if (x0, x1 - x0 + 1) == (self.x0, self.rows):
            return
        top = self.x0 - x0
        for segment in self.segments:
            for name in ("plus", "minus", "known"):
                old = getattr(segment, name)
                new = np.zeros((x1 - x0 + 1, old.shape[1]), bool)
                new[top : top + self.rows] = old
                setattr(segment, name, new)
        self.x0, self.rows = x0, x1 - x0 + 1

    def _merged(self, j_lo: int, j_hi: int) -> _Segment:
        """The one segment covering ``[j_lo, j_hi]``, absorbing every
        segment that overlaps or touches it."""
        touching = self._touching(j_lo, j_hi)
        if len(touching) == 1 and touching[0].j0 <= j_lo and touching[0].j1 >= j_hi:
            return touching[0]
        j0 = min([j_lo, *(s.j0 for s in touching)])
        j1 = max([j_hi, *(s.j1 for s in touching)])
        shape = (self.rows, j1 - j0 + 1)
        merged = _Segment(j0, np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool))
        for segment in touching:
            cols = slice(segment.j0 - j0, segment.j1 - j0 + 1)
            merged.plus[:, cols] = segment.plus
            merged.minus[:, cols] = segment.minus
            merged.known[:, cols] = segment.known
        self.segments = sorted(
            [s for s in self.segments if all(s is not t for t in touching)] + [merged],
            key=lambda s: s.j0,
        )
        return merged

    def _containing(self, j_lo: int, j_hi: int) -> _Segment | None:
        for segment in self.segments:
            if segment.j0 <= j_lo and segment.j1 >= j_hi:
                return segment
        return None

    def ensure(self, x_lo: int, x_hi: int, j_lo: int, j_hi: int) -> None:
        """Classify every cell of ``[x_lo, x_hi] x [j_lo, j_hi]`` not yet known."""
        if self.refused:
            return
        stored = self._stored_after(x_lo, x_hi, j_lo, j_hi)
        if stored > self.max_cells:
            raise ReplayBoundExceeded(stored)
        self._cover_rows(x_lo, x_hi)
        segment = self._merged(j_lo, j_hi)
        rows = slice(x_lo - self.x0, x_hi - self.x0 + 1)
        cols = slice(j_lo - segment.j0, j_hi - segment.j0 + 1)
        unknown = ~segment.known[rows, cols]
        if not unknown.any():
            return
        # Runs of unknown treatment counts, row by row.
        edges = np.diff(np.pad(unknown.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        starts = np.argwhere(edges == 1)
        stops = np.argwhere(edges == -1)
        requests = [
            _Request(x_lo + int(row), kind, j_lo + int(start), j_lo + int(stop) - 1)
            for (row, start), (_, stop) in zip(starts, stops, strict=True)
            for kind in self.decision.kinds
        ]
        for req, mask in zip(requests, classify(self.decision, self.route, requests), strict=True):
            target = segment.plus if req.kind == "plus" else segment.minus
            row = req.x_c - self.x0
            target[row, req.j0 - segment.j0 : req.j1 - segment.j0 + 1] = mask
        segment.known[rows, cols] = True

    def cells(self, x_lo: int, x_hi: int, j_lo: int, j_hi: int) -> tuple[np.ndarray, np.ndarray]:
        """Plus and minus rejection masks over ``[x_lo, x_hi] x [j_lo, j_hi]``."""
        if self.refused:
            shape = (x_hi - x_lo + 1, j_hi - j_lo + 1)
            return np.zeros(shape, bool), np.zeros(shape, bool)
        self.ensure(x_lo, x_hi, j_lo, j_hi)
        segment = self._containing(j_lo, j_hi)
        assert segment is not None
        rows = slice(x_lo - self.x0, x_hi - self.x0 + 1)
        cols = slice(j_lo - segment.j0, j_hi - segment.j0 + 1)
        return segment.plus[rows, cols], segment.minus[rows, cols]

    def evaluate(self, p_c: float, p_t: float) -> BinomialPower:
        """Rejection probability at control rate ``p_c``, treatment rate ``p_t``."""
        decision = self.decision
        if self.refused:
            return BinomialPower(0.0, 0.0, 0.0)
        wc = _window(decision.n_c, p_c)
        wt = _window(decision.n_t, p_t)
        try:
            plus, minus = self.cells(wc.lo, wc.hi, wt.lo, wt.hi)
        except ReplayBoundExceeded as exceeded:
            raise ReplayBoundExceeded(exceeded.cells, p_t) from None
        return BinomialPower(
            float(wc.weights @ plus @ wt.weights),
            float(wc.weights @ (minus & ~plus) @ wt.weights),
            wc.omitted + wt.omitted,
        )

    def closure_bound(self, p_c: float, p_lo: float, p_hi: float) -> float:
        """Upper bound on ``evaluate(p_c, p).power`` for every treatment rate
        ``p`` in ``[p_lo, p_hi]``.

        Every such evaluation sums treatment counts inside ``[L, H]``, the
        lower edge of ``p_lo``'s window and the upper edge of ``p_hi``'s.
        Within it each control row's rejections lie inside the upper-set
        closure of its plus rejections, ``j >= t``, and the lower-set closure
        of its minus rejections, ``j <= s``. ``P(t <= X <= H)`` has derivative
        ``n (b(t - 1) - b(H))`` in the rate (``b`` the ``Bin(n - 1)`` PMF),
        whose sign changes at most once, upward to downward, so it is
        nondecreasing on the interval when ``b(t - 1) >= b(H)`` at ``p_hi``;
        symmetrically ``P(L <= X <= s)`` is nonincreasing when ``b(s) >=
        b(L - 1)`` at ``p_lo``. A row failing its check, or with unclassified
        counts in ``[L, H]``, is bounded by its untruncated tail instead.
        """
        decision = self.decision
        n_t = decision.n_t
        if self.refused:
            return 0.0
        wc = _window(decision.n_c, p_c)
        low, high = _window(n_t, p_lo).lo, _window(n_t, p_hi).hi
        rows = np.arange(wc.lo, wc.hi + 1) - self.x0
        # Rows without classified cells across [low, high]: every count may reject.
        t = np.full(rows.size, low, np.int64)
        s = np.full(rows.size, high, np.int64)
        covered = np.zeros(rows.size, bool)
        segment = self._containing(low, high)
        inside = (rows >= 0) & (rows < self.rows)
        if segment is not None and inside.any():
            r = rows[inside]
            cols = slice(low - segment.j0, high - segment.j0 + 1)
            j = np.arange(low, high + 1)
            width = j.size
            plus = segment.plus[r, cols]
            minus = segment.minus[r, cols]
            known = segment.known[r, cols].all(axis=1)
            first = np.where(plus.any(axis=1), j[np.argmax(plus, axis=1)], high + 1)
            last_index = width - 1 - np.argmax(minus[:, ::-1], axis=1)
            last = np.where(minus.any(axis=1), j[last_index], low - 1)
            covered[inside] = known
            t[inside] = np.where(known, first, low)
            s[inside] = np.where(known, last, high)
        if "plus" not in decision.kinds:
            t[:] = high + 1
        if "minus" not in decision.kinds:
            s[:] = low - 1
        with np.errstate(invalid="ignore"):
            tail_up = np.where(t <= 0, 1.0, _rr._fast_binom_sf(t - 1, n_t, p_hi))
            high_up = _rr._fast_binom_sf(np.asarray(high), n_t, p_hi) if high < n_t else 0.0
            inner_up = tail_up - high_up
            rising = _rr._fast_binom_pmf(t - 1, n_t - 1, p_hi) >= _rr._fast_binom_pmf(
                np.array(high), n_t - 1, p_hi
            )
            up = np.where(t > high, 0.0, np.where(covered & rising, inner_up, tail_up))
            tail_down = np.where(s >= n_t, 1.0, _rr._fast_binom_cdf(s, n_t, p_lo))
            low_down = _rr._fast_binom_cdf(np.asarray(low - 1), n_t, p_lo) if low > 0 else 0.0
            inner_down = tail_down - low_down
            falling = _rr._fast_binom_pmf(s, n_t - 1, p_lo) >= _rr._fast_binom_pmf(
                np.array(low - 1), n_t - 1, p_lo
            )
            down = np.where(s < low, 0.0, np.where(covered & falling, inner_down, tail_down))
        rows_bound = np.minimum(1.0, np.maximum(up, 0.0) + np.maximum(down, 0.0))
        return min(1.0, float(wc.weights @ rows_bound))
