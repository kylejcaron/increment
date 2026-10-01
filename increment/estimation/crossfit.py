"""Offline cross-validation: seeded splits and out-of-fold candidate selection.

Analysis-time randomization, not experiment randomization: a seeded
``np.random.default_rng`` shuffle is the right tool, and ``seed`` is
required everywhere - rerunning an unseeded split until the answer
looks good is the exact selection failure this module exists to prevent.

Splits are keyed on canonical identity, never row position: units (or,
when ``cluster_ids`` is supplied, whole clusters) are ranked by a
row-order-independent canonical ordering of their unique ids before the
seeded shuffle runs, so swapping which row a unit occupies never changes
which side of a split it lands on. Never Python's process-randomized
``hash()`` - only content-stable string canonicalization and numpy's
seeded ``Generator``.

The selection harness is deliberately generic: candidates are indices,
models are opaque, and the caller supplies ``fit``/``evaluate``
callables returning per-unit objective contributions, pooled across
folds into a mean, plug-in SE, and sample size per candidate - reusable
for any grid-over-candidates selection.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from increment import _identity
from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.crossfit.stratify_shape_expected": "stratify has shape {s}, expected ({n},) to match unit_ids",
        "estimation.crossfit.fold_assignments_unit_ids_shape": "unit_ids must be 1-d, got shape {unit_ids}",
        "estimation.crossfit.test_size": "test_size must be in (0, 1), got {test_size}",
        "estimation.crossfit.n_folds_least": "n_folds must be at least 2, got {n_folds}",
        "estimation.crossfit.n_candidates_least": "n_candidates must be at least 1, got {n_candidates}",
        "estimation.crossfit.selection_needs_least": "selection needs at least 2 fold labels, got {n_labels}",
        "estimation.crossfit.candidate_returned_contribution": "candidate {index} returned a contribution of shape {shape} for a validation fold of {n_val} rows",
        "estimation.crossfit.candidate_contribution_nonfinite": RefusalSpec(
            "estimation.crossfit.candidate_contribution_nonfinite",
            InvalidRequestError,
            template="candidate {index} on fold {fold} returned {n_bad} non-finite contribution(s) out of {n_total}. {route}",
        ),
        "estimation.crossfit.cluster_ids_shape_expected": "cluster_ids has shape {cluster_ids}, expected {unit_ids} to match unit_ids",
        "estimation.crossfit.insufficient_clusters": "only {n_clusters} atomic group(s) available in a stratum for n_folds={n_folds} -- cross-fitting needs at least one group per stratum per fold; groups are distinct cluster_ids when supplied, otherwise distinct unit_ids; lower n_folds or supply more groups",
        "estimation.crossfit.cluster_atomic_shape": "cluster_ids has shape {cluster_ids}, expected {assignment} to match the supplied assignment",
        "estimation.crossfit.cluster_split": "cluster {cluster!r} is split across the supplied assignment (values {values[0]!r} and {values[1]!r} both appear among its members) -- a cluster must land entirely on one side of a split, or entirely in one fold",
    },
)
N_FOLDS_LEAST = _REFUSALS["estimation.crossfit.n_folds_least"]
_raise = raiser(_REFUSALS)


def _strata(n: int, stratify: np.ndarray | None) -> np.ndarray:
    """One stratum label per row; a single stratum when none are given."""
    if stratify is None:
        return np.zeros(n, dtype=np.int64)
    s = np.asarray(stratify)
    if s.shape != (n,):
        _raise("estimation.crossfit.stratify_shape_expected", s=s.shape, n=n)
    return s


def check_cluster_atomic(
    cluster_ids: np.ndarray, assignment: np.ndarray, *, what: str = "cluster_ids"
) -> None:
    """Refuse a caller-supplied mask or fold-label array that splits a cluster.

    ``assignment`` is any per-row array sharing a value means "same side" -
    a boolean train/test mask or an integer fold-label array. Every row
    sharing a canonical cluster id must carry the same assignment value;
    otherwise the cluster's declared dependence leaks across train/test or
    across folds. Raises naming the offending cluster and its two observed
    assignment values. An empty *cluster_ids* has nothing to split -
    returns rather than indexing a boundary array of size zero.
    """
    cluster_ids = np.asarray(cluster_ids)
    assignment = np.asarray(assignment)
    if cluster_ids.shape != assignment.shape:
        _raise(
            "estimation.crossfit.cluster_atomic_shape",
            cluster_ids=cluster_ids.shape,
            assignment=assignment.shape,
        )
    canon = _identity.canonical_id_strings(cluster_ids, what=what)
    if canon.size == 0:
        return
    order = np.argsort(canon)
    sorted_canon = canon[order]
    sorted_assignment = assignment[order]
    boundary = np.empty(sorted_canon.shape, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_canon[1:] != sorted_canon[:-1]
    changed = np.zeros(sorted_canon.shape, dtype=bool)
    changed[1:] = sorted_assignment[1:] != sorted_assignment[:-1]
    split = changed & ~boundary
    if split.any():
        i = int(np.flatnonzero(split)[0])
        _raise(
            "estimation.crossfit.cluster_split",
            cluster=sorted_canon[i],
            values=(sorted_assignment[i - 1].item(), sorted_assignment[i].item()),
        )


def _grouped_strata(
    unit_ids: np.ndarray, stratify: np.ndarray | None, cluster_ids: np.ndarray | None
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, dict[int, dict[int, int]], int]:
    """Canonical group index per row, its count, each group's canonical id,
    each group's *pure* stratum value (``-1`` when it spans more than one),
    and each *mixed* group's ``{stratum_value: count}``.

    A cluster (or, absent an explicit ``cluster_ids``, a repeated unit id)
    is the unit of assignment; ``stratify`` still keys the strata. Groups
    are numbered by rank in the canonically sorted array of unique ids, so
    the numbering - and everything downstream that shuffles it - never
    depends on row order.

    The per-group/stratum breakdown is sized to the ``(group, value)``
    pairs actually observed - at most one per row - never the full
    ``n_groups`` x (distinct stratum values) cross product a dense matrix
    would allocate: a caller stratifying many rows on a high-cardinality
    column pays for what is actually there, not for every group's absence
    from every value it never touches.
    """
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape != unit_ids.shape:
            _raise(
                "estimation.crossfit.cluster_ids_shape_expected",
                cluster_ids=cluster_ids.shape,
                unit_ids=unit_ids.shape,
            )
    ids_for_groups = unit_ids if cluster_ids is None else cluster_ids
    what = "unit_ids" if cluster_ids is None else "cluster_ids"
    canon = _identity.canonical_id_strings(ids_for_groups, what=what)
    group_canon = np.unique(canon)
    groups = np.searchsorted(group_canon, canon).astype(np.int64)
    n_groups = int(group_canon.size)
    strat_raw = _strata(unit_ids.size, stratify)
    _, strat_codes = np.unique(strat_raw, return_inverse=True)
    strat_codes = np.asarray(strat_codes, dtype=np.int64).reshape(-1)
    n_values = int(strat_codes.max()) + 1 if strat_codes.size else 0

    pairs, pair_counts = np.unique(np.stack([groups, strat_codes]), axis=1, return_counts=True)
    pair_groups = np.argsort(pairs[0], kind="stable")
    pair_groups, pair_values, pair_counts = (
        pairs[0][pair_groups],
        pairs[1][pair_groups],
        pair_counts[pair_groups],
    )
    group_offsets = np.searchsorted(pair_groups, np.arange(n_groups + 1))
    n_distinct = np.diff(group_offsets)
    purity = np.full(n_groups, -1, dtype=np.int64)
    pure_groups = np.flatnonzero(n_distinct == 1)
    purity[pure_groups] = pair_values[group_offsets[pure_groups]]
    mixed_rows: dict[int, dict[int, int]] = {}
    for g in np.flatnonzero(n_distinct > 1).tolist():
        start, end = int(group_offsets[g]), int(group_offsets[g + 1])
        mixed_rows[g] = dict(
            zip(pair_values[start:end].tolist(), pair_counts[start:end].tolist(), strict=True)
        )
    return groups, n_groups, group_canon, purity, mixed_rows, n_values


def _pure_groups(purity: np.ndarray, n_values: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield canonical pure-group indices without a group-by-stratum scan."""
    order = np.argsort(purity, kind="stable")
    offsets = np.searchsorted(purity[order], np.arange(n_values + 1))
    for value, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        if start != stop:
            yield value, order[start:stop]


def _assigned_strata_counts(
    groups: np.ndarray, purity: np.ndarray, destinations: np.ndarray, n_dest: int, n_values: int
) -> np.ndarray:
    """Count rows already assigned through pure groups."""
    counts = np.zeros((n_dest, n_values), dtype=float)
    pure = purity >= 0
    sizes = np.bincount(groups, minlength=purity.size)
    np.add.at(counts, (destinations[pure], purity[pure]), sizes[pure])
    return counts


def _balance_destinations(
    order: np.ndarray,
    rows: dict[int, dict[int, int]],
    weights: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Fill missing stratum support, then balance normalized row counts.

    Pure groups contribute through ``initial``. Fractions preserve uniform
    member-replication invariance; each group touches only its observed strata.
    With binary strata, all mixed groups cover both arms, so prioritizing
    missing support fills every feasible fold after pure round-robin placement.
    """
    totals = initial.sum(axis=0)
    for group in order:
        for value, count in rows[int(group)].items():
            totals[value] += count
    running = initial / totals
    dest = np.empty(order.size, dtype=np.int64)
    for pos, group in enumerate(order):
        row = rows[int(group)]
        values = np.fromiter(row, dtype=np.int64, count=len(row))
        fractions = np.fromiter(row.values(), dtype=float, count=len(row)) / totals[values]
        active = running[:, values]
        missing = (active == 0.0).sum(axis=1)
        score = ((weights[:, None] - active) * fractions).sum(axis=1)
        score[missing != missing.max()] = -np.inf
        best = int(np.argmax(score))
        dest[pos] = best
        running[best, values] += fractions
    return dest


def outer_split(
    unit_ids: np.ndarray,
    *,
    test_size: float = 0.5,
    seed: int,
    stratify: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of the outer test set: ``True`` rows are held out untouched.

    Units - or, when *cluster_ids* is supplied, whole clusters - are
    ranked by a canonical, row-order-independent ordering of their unique
    ids, then a ``seed``-keyed generator shuffles that ranking and the
    first ``round(test_size * n)`` land in the test set, so arms stay
    balanced when *stratify* is the treatment column. Without
    *cluster_ids*, a repeated unit id (e.g. panel replication) is its own
    singleton cluster: every row sharing it lands on the same side.

    For cluster-randomized data (every member of a cluster shares one
    *stratify* value), whole clusters are stratified by that value
    exactly like the unclustered case. A cluster spanning more than one
    *stratify* value (a mixed-treatment observational cluster) is never
    split to force arm purity; instead it is placed to balance the
    aggregate per-value counts across train/test.
    """
    unit_ids = np.asarray(unit_ids)
    if unit_ids.ndim != 1:
        _raise("estimation.crossfit.fold_assignments_unit_ids_shape", unit_ids=unit_ids.shape)
    if not 0.0 < test_size < 1.0:
        _raise("estimation.crossfit.test_size", test_size=test_size)
    groups, n_groups, _group_canon, purity, mixed_rows, n_values = _grouped_strata(
        unit_ids, stratify, cluster_ids
    )
    rng = np.random.default_rng(seed)
    dest_by_group = np.zeros(n_groups, dtype=np.int64)
    for _value, pure_ids in _pure_groups(purity, n_values):
        shuffled = pure_ids.copy()
        rng.shuffle(shuffled)
        n_test = int(round(test_size * shuffled.size))
        dest_by_group[shuffled[:n_test]] = 1
    mixed_ids = np.flatnonzero(purity == -1)
    if mixed_ids.size:
        shuffled = mixed_ids.copy()
        rng.shuffle(shuffled)
        weights = np.array([1.0 - test_size, test_size])
        initial = _assigned_strata_counts(groups, purity, dest_by_group, 2, n_values)
        dest_by_group[shuffled] = _balance_destinations(shuffled, mixed_rows, weights, initial)
    return dest_by_group[groups].astype(bool)


def fold_assignments(
    unit_ids: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    stratify: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Fold label ``0..n_folds-1`` per row, shuffled within each stratum.

    Units - or, when *cluster_ids* is supplied, whole clusters - are
    ranked by a canonical, row-order-independent ordering of their unique
    ids, then round-robin assigned after the seeded shuffle, keeping fold
    sizes within one of each other inside every stratum so no fold ends
    up arm-starved by chance. Without *cluster_ids*, a repeated unit id is
    its own singleton cluster: every row sharing it lands in the same
    fold.

    Cluster-randomized strata (every member shares one *stratify* value)
    round-robin whole clusters exactly like the unclustered case. A
    mixed-treatment observational cluster is never split to force arm
    purity; instead it is placed to balance the aggregate per-value
    counts across folds. The support check counts pure and mixed groups
    together for each stratum. With binary strata, sufficient combined
    support gives both arms in every fold. With more strata, mixed-group
    balancing is greedy and does not guarantee coverage in every fold.
    """
    unit_ids = np.asarray(unit_ids)
    if unit_ids.ndim != 1:
        _raise("estimation.crossfit.fold_assignments_unit_ids_shape", unit_ids=unit_ids.shape)
    if n_folds < 2:
        _raise("estimation.crossfit.n_folds_least", n_folds=n_folds)
    groups, n_groups, _group_canon, purity, mixed_rows, n_values = _grouped_strata(
        unit_ids, stratify, cluster_ids
    )
    mixed_ids = np.flatnonzero(purity == -1)
    touching = np.bincount(purity[purity >= 0], minlength=n_values).astype(np.int64)
    for g in mixed_ids.tolist():
        for v in mixed_rows[g]:
            touching[v] += 1
    minimum_support = int(touching.min(initial=n_groups))
    if minimum_support < n_folds:
        _raise(
            "estimation.crossfit.insufficient_clusters",
            n_clusters=minimum_support,
            n_folds=n_folds,
        )
    rng = np.random.default_rng(seed)
    dest_by_group = np.empty(n_groups, dtype=np.int64)
    for _value, pure_ids in _pure_groups(purity, n_values):
        shuffled = pure_ids.copy()
        rng.shuffle(shuffled)
        dest_by_group[shuffled] = np.arange(shuffled.size, dtype=np.int64) % n_folds
    if mixed_ids.size:
        shuffled = mixed_ids.copy()
        rng.shuffle(shuffled)
        weights = np.full(n_folds, 1.0 / n_folds)
        initial = _assigned_strata_counts(groups, purity, dest_by_group, n_folds, n_values)
        dest_by_group[shuffled] = _balance_destinations(shuffled, mixed_rows, weights, initial)
    return dest_by_group[groups]


class Selection(BaseModel):
    """Pooled out-of-fold objective per candidate, and which one won.

    Candidates are indices: the harness never sees candidate values,
    so the caller keeps whatever type its grid holds. ``ses`` are
    plug-in standard errors of the pooled per-unit contributions -
    descriptive, not a license to re-select.
    """

    model_config = ConfigDict(frozen=True)

    estimates: tuple[float, ...]
    ses: tuple[float, ...]
    n_units: tuple[int, ...]
    best_index: int
    n_folds: int


def select_out_of_fold(
    n_candidates: int,
    folds: np.ndarray,
    fit: Callable[[np.ndarray], Any],
    evaluate: Callable[[Any, np.ndarray, int], np.ndarray],
) -> Selection:
    """Fit per fold, score every candidate out-of-fold, return the argmax.

    ``fit`` receives the boolean train mask for each fold; ``evaluate``
    must return one objective contribution per row of the validation
    mask. Contributions pool across folds so each candidate's estimate
    is a plain mean over every inner unit, scored by a model that
    never saw it. Ties break to the lowest index.
    """
    folds = np.asarray(folds)
    if n_candidates < 1:
        _raise("estimation.crossfit.n_candidates_least", n_candidates=n_candidates)
    labels = np.unique(folds)
    if labels.size < 2:
        _raise("estimation.crossfit.selection_needs_least", n_labels=labels.size)
    pooled: list[list[np.ndarray]] = [[] for _ in range(n_candidates)]
    for label in labels:
        val = folds == label
        model = fit(~val)
        for index in range(n_candidates):
            contrib = np.asarray(evaluate(model, val, index), dtype=float)
            if contrib.shape != (int(val.sum()),):
                _raise(
                    "estimation.crossfit.candidate_returned_contribution",
                    index=index,
                    shape=contrib.shape,
                    n_val=int(val.sum()),
                )
            finite = np.isfinite(contrib)
            if not finite.all():
                _raise(
                    "estimation.crossfit.candidate_contribution_nonfinite",
                    index=index,
                    fold=label,
                    n_bad=int((~finite).sum()),
                    n_total=int(contrib.size),
                    route="Correct the evaluation callback to return finite contributions for every validation row.",
                )
            pooled[index].append(contrib)
    estimates, ses, n_units = [], [], []
    for index in range(n_candidates):
        x = np.concatenate(pooled[index])
        estimates.append(float(x.mean()))
        ses.append(float(x.std(ddof=1) / math.sqrt(x.size)))
        n_units.append(int(x.size))
    return Selection(
        estimates=tuple(estimates),
        ses=tuple(ses),
        n_units=tuple(n_units),
        best_index=int(np.argmax(estimates)),
        n_folds=int(labels.size),
    )


ESTIMATION_CROSSFIT_FOLD_ASSIGNMENTS_UNIT_IDS_SHAPE = _REFUSALS[
    "estimation.crossfit.fold_assignments_unit_ids_shape"
]
