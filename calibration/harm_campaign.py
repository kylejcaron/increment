"""Synthetic decision study for bounds on the fraction harmed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from functools import cache
from itertools import product
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pydantic
import scipy
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import beta

from calibration.harm_bounds import (
    Band,
    Cell,
    Counts4,
    Interval,
    LeafMap,
    StudyError,
    learn,
    summarize,
)
from calibration.harm_oracle import _Probability as _OracleProbability
from calibration.harm_oracle import _Report as _OracleReport
from calibration.harm_oracle import potential_table_bounds

_BASE = "8fc12df340d4a0b36f7d4039b2a37f22fcb4eb8e"
_PROTOCOL_BASE = "77ab667b227b2005e18bcc3d0f56c23b6aaec651"
_PROTOCOL = "calibration/protocols/harm_study_v1.md"
_PROTOCOL_DIGEST = "516a7ac0697769f2b477e22f27b257fd75405eeb8a9515f1ed2f0f5ed7e585ac"
_ORACLE_SOURCES = ("calibration/harm_bounds.py", "calibration/harm_oracle.py", "uv.lock")
_ARTIFACTS = ("cells.jsonl", "decisions.jsonl", "gates.json")
_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_SEED = 20261119082
_NS = (160, 4096)
_ES = (0.5, 0.1)
_GATES = 88
_BATCH = 128
_FULL_TRIALS = 20_000
_SMOKE_TRIALS = 8
_TAU = (0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.32, 0.50)
_TAUS = np.asarray(_TAU)
_AT_32, _AT_50 = _TAU.index(0.32), _TAU.index(0.50)
_EQUIV = 1e-12
# Only a few rounding units for the arithmetic in the projections, much less than the decision margin.
_ULPS = 8 * math.ulp(1.0)
_U0: LeafMap = (0, 0, 0, 0)
_F: LeafMap = (0, 0, 1, 1)
_Q: LeafMap = (0, 1, 2, 3)
_STATIC_MAPS = (_U0, _F, _Q)
_METHODS = ("U0", "F", "Q", "H", "M")
_MODES = ("known", "estimated")
_PIPELINES = ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3))
_TARGETS = (*_METHODS, "P0", "P1", "P2", "P3", "P1∩H")
_P0, _P1, _P1H = (_TARGETS.index(p) for p in ("P0", "P1", "P1∩H"))
_ESTIMATED = _MODES.index("estimated")
_LOSS = ((10, 1, 0.25), (1, 10, 0.25))
_ERRORS = ("simultaneous_noncoverage", "any_false_clearance", "any_false_rule_out")
_POWER = ("D2_P0_rule_out_50", "D3_P0_clear_50", "D4_P1_increment_32", "D5_P1H_increment_32")
_T, _V = (43, 47, 0, 10), (5, 0, 25, 70)
_JOINT = (
    ((50, 0, 0, 50),) * 4,
    ((0, 50, 50, 0),) * 4,
    ((0, 0, 100, 0),) * 4,
    ((0, 100, 0, 0),) * 4,
    (_T, _T, _V, _V),
    (_T, _V, _T, _V),
    ((98, 1, 1, 0), (0, 1, 1, 98), (1, 48, 49, 2), (2, 49, 48, 1)),
)
_QUARTER = Fraction(1, 4)
_BALANCED = (_QUARTER, _QUARTER, _QUARTER, _QUARTER)
_WEIGHTS = (_BALANCED,) * 6 + (
    (Fraction(49, 100), Fraction(49, 100), Fraction(1, 100), Fraction(1, 100)),
)
_TRUTH_ANCHORS = (
    ("0", "0", ".5", "0", ".5", "0", ".5"),
    (".5", "0", ".5", "0", ".5", "0", ".5"),
    ("1", "1", "1", "1", "1", "1", "1"),
    ("0", "0", "0", "0", "0", "0", "0"),
    (".125", "0", ".365", ".125", ".20", ".125", ".20"),
    (".125", "0", ".365", "0", ".365", ".125", ".20"),
    (".0195", "0", ".5", "0", ".5", ".0001", ".0197"),
)
# The multinomial is drawn before masking, not from a lower-dimensional observed law.
_SUPPORT = tuple(product(range(4), range(2), range(2), range(2)))
_FACTUAL = tuple(
    tuple(i for i, (g, y0, y1, a) in enumerate(_SUPPORT) if (g, a, (y0, y1)[a]) == key)
    for key in product(range(4), range(2), range(2))
)


type _RunMode = Literal["smoke", "full"]
type _GateKind = Literal["error", "power"]
type _GateStatus = Literal["green", "red", "inconclusive"]
type _RunStatus = Literal["green", "red", "inconclusive", "smoke", "execution_failure"]
type _Number = NDArray[np.float64]
type _Flag = NDArray[np.bool_]
type _Count = NDArray[np.int64]
type _Truth = tuple[Fraction, Fraction, Fraction]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)


class _Coordinate(_Frozen):
    dgp: int
    n_index: int
    e_index: int
    n: int
    e: float


class _Diagnostic(_Frozen):
    reason: str
    context: tuple[tuple[str, str | int], ...]


class _GateCounter(_Frozen):
    coordinate: _Coordinate
    name: str
    kind: _GateKind
    count: int
    attempted: int
    interval: Interval
    status: _GateStatus


class _Gates(_Frozen):
    schema_version: Literal[1] = 1
    status: _RunStatus
    certification_eligible: bool
    counters: tuple[_GateCounter, ...]


class _Summary(_Frozen):
    n: int
    mean: float | None
    mcse: float | None


class _EndpointError(_Frozen):
    n: int
    bias: float | None
    rmse: float | None
    bias_mcse: float | None


class _Spans(_Frozen):
    true_id: _Summary
    estimated_id: _Summary
    outer: _Summary
    inner: _Summary
    sampling_excess: _Summary


class _CountByReason(_Frozen):
    reason: str
    count: int


class _CountByMap(_Frozen):
    leaf_map: LeafMap
    count: int


class _HWidthComparison(_Frozen):
    wider_than_q: int
    with_coarser_true_id: int
    with_same_true_id: int


class _MethodRow(_Frozen):
    method: str
    mode: str
    attempted: int
    valid_bands: int
    plugin_trials: int
    null_plugin: int
    null_plugin_reasons: tuple[_CountByReason, ...]
    empty_arm_leaves: int
    empty_membership_leaves: int
    l_covered: int
    u_covered: int
    jointly_covered: int
    h_covered: int
    plugin_l_error: _EndpointError
    plugin_u_error: _EndpointError
    spans: _Spans
    paired_spans_vs_u0: _Spans
    paired_spans_vs_f: _Spans
    leaf_maps: tuple[_CountByMap, ...]
    training_n: int
    estimation_n: int
    h_vs_q: _HWidthComparison | None


class _CellRow(_Frozen):
    coordinate: _Coordinate
    attempted: int
    evaluated: int
    h: float
    id_u0: Interval
    id_f: Interval
    id_q: Interval
    methods: tuple[_MethodRow, ...]


class _Tally(_Frozen):
    evaluated: int
    clear: int
    rule_out: int
    inconclusive: int
    lower_equal: int
    upper_equal: int


class _Rates(_Frozen):
    abstention: float
    false_clear: float
    false_rule_out: float
    false_id_clear: float
    false_id_rule_out: float


class _Loss(_Frozen):
    costs: tuple[int, int, float]
    proposed: _Summary
    vs_never_decide: _Summary
    vs_p0: _Summary


class _DecisionRow(_Frozen):
    coordinate: _Coordinate
    mode: str
    target: str
    tau: float
    attempted: int
    evaluated: int
    clear: int
    rule_out: int
    inconclusive: int
    false_clear: int
    false_rule_out: int
    false_id_clear: int
    false_id_rule_out: int
    id_abstain: int
    sampling_abstain: int
    diagnostic_abstain: int
    diagnostic_abstain_reason: Literal["discordant_bands"] | None
    rates_per_attempt: _Rates
    outer_equal: tuple[int, int]
    true_policy: _Tally
    full_q_id: _Tally
    method_id: _Tally | None
    plugin: _Tally | None
    losses: tuple[_Loss, ...]


class _Receipt(_Frozen):
    path: str
    sha256: str


class _Rng(_Frozen):
    generator: Literal["PCG64"] = "PCG64"
    seed_sequence: tuple[str | int, ...] = (_SEED, "dgp", "n_index", "e_index", "trial", "split")
    dgps: tuple[int, ...] = tuple(range(len(_JOINT)))
    n_values: tuple[int, ...] = _NS
    e_values: tuple[float, ...] = _ES
    splits: tuple[str, str] = ("train", "evaluation")
    support_order: tuple[str, ...] = ("S", "B", "Y0", "Y1", "A")
    batch_size: int = _BATCH


class _Manifest(_Frozen):
    schema_version: Literal[1] = 1
    mode: _RunMode
    certification_eligible: bool
    status: _RunStatus
    exit_code: int
    base_revision: str = _BASE
    source_revision: str | None
    protocol_sha256: str | None
    oracle_sha256: str | None
    sources: tuple[_Receipt, ...]
    inputs: tuple[_Receipt, ...]
    artifacts: tuple[_Receipt, ...]
    invocation: tuple[str, ...]
    python: str
    numpy: str
    scipy: str
    pydantic: str
    thread_environment: tuple[tuple[str, str | None], ...]
    rng: _Rng = _Rng()
    started_at: datetime
    ended_at: datetime
    trials_per_cell: int
    completed_cells: int
    failures: tuple[_Diagnostic, ...]


def coordinates() -> tuple[_Coordinate, ...]:
    """Return the frozen roster in cell order."""
    return tuple(
        _Coordinate(dgp=d, n_index=i, e_index=j, n=n, e=e)
        for d, (i, n), (j, e) in product(range(len(_JOINT)), enumerate(_NS), enumerate(_ES))
    )


@cache
def population_truth(d: int, leaf_map: LeafMap) -> _Truth:
    """Enumerate rational conditional joint tables; return (h, L, U)."""
    weights, joint = _WEIGHTS[d], _JOINT[d]
    mass = tuple(
        tuple(w * Fraction(k, 100) for k in law) for w, law in zip(weights, joint, strict=True)
    )
    h = sum((m[2] for m in mass), Fraction(0))
    lower = upper = Fraction(0)
    for leaf in sorted(set(leaf_map)):
        c00, c01, c10, c11 = (
            sum((m[j] for m, g in zip(mass, leaf_map, strict=True) if g == leaf), Fraction(0))
            for j in range(4)
        )
        w = c00 + c01 + c10 + c11
        lo, hi = potential_table_bounds((c10 + c11) / w, (c01 + c11) / w)
        lower += w * lo
        upper += w * hi
    return h, lower, upper


def _error(reason: str, stage: str, coordinate: str, **details: str | int) -> StudyError:
    return StudyError(reason, (("stage", stage), ("coordinate", coordinate), *details.items()))


def counter_gate(k: int, n: int, kind: _GateKind) -> tuple[Interval, _GateStatus]:
    """Invert exact binomial tails with the 88-counter family allocation."""
    if not 0 <= k <= n or n <= 0:
        raise _error("invalid_counter", "gate", kind, count=k, attempted=n)
    gamma = 0.01 / (2 * _GATES)
    lo = 0.0 if k == 0 else float(beta.ppf(gamma, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.isf(gamma, k + 1, n - k))
    if not 0 <= lo <= hi <= 1:
        raise _error("invalid_counter_interval", "gate", kind, count=k, attempted=n)
    status: _GateStatus
    if kind == "error":
        status = "green" if hi <= 0.05 else "red" if lo > 0.05 else "inconclusive"
    else:
        status = "green" if lo >= 0.80 else "red" if hi < 0.80 else "inconclusive"
    return (lo, hi), status


@cache
def _probabilities(d: int, e: float) -> tuple[float, ...]:
    return tuple(
        float(_WEIGHTS[d][g]) * (_JOINT[d][g][2 * y0 + y1] / 100) * (e if a else 1 - e)
        for g, y0, y1, a in _SUPPORT
    )


def draw_counts(coord: _Coordinate, trial: int, split: int) -> Counts4:
    """Draw one split multinomial, then reduce it to factual arm/atom counts."""
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([_SEED, coord.dgp, coord.n_index, coord.e_index, trial, split])
        )
    )
    training_n = coord.n // 3
    draw = rng.multinomial(
        training_n if split == 0 else coord.n - training_n, _probabilities(coord.dgp, coord.e)
    )
    a, b, c, d, e, f, g, h, i, j, k, y, m, n, o, p = (int(draw[x] + draw[z]) for x, z in _FACTUAL)
    return (
        Cell(a + b, b, c + d, d),
        Cell(e + f, f, g + h, h),
        Cell(i + j, j, k + y, y),
        Cell(m + n, n, o + p, p),
    )


def _add_cells(a: Cell, b: Cell) -> Cell:
    return Cell(a.n0 + b.n0, a.y0 + b.y0, a.n1 + b.n1, a.y1 + b.y1)


@dataclass(frozen=True, slots=True)
class _Trial:
    index: int
    maps: tuple[LeafMap, ...]
    ids: tuple[Interval, ...]
    bands: tuple[tuple[Band, ...], ...]


@cache
def _identification(d: int, leaf_map: LeafMap) -> Interval:
    h, lo, hi = population_truth(d, leaf_map)
    _, qlo, qhi = population_truth(d, _Q)
    if not 0 <= lo <= qlo <= h <= qhi <= hi <= 1:
        raise _error("oracle_inconsistent", "truth", str(leaf_map), dgp=d)
    return float(lo), float(hi)


def _trial(coord: _Coordinate, index: int) -> _Trial:
    train, evaluation = draw_counts(coord, index, 0), draw_counts(coord, index, 1)
    a, b, c, d = (_add_cells(x, y) for x, y in zip(train, evaluation, strict=True))
    maps = (*_STATIC_MAPS, learn(train, omit_b=False), learn(train, omit_b=True))
    ids = tuple(_identification(coord.dgp, leaf_map) for leaf_map in maps)
    train_eval = ((a, b, c, d),) * 3 + (evaluation, evaluation)
    # The interval kernel caches by (k, n, K). A block uses at most 128 * 3 * (1+2+4+4+2)
    # distinct keys; both mass modes therefore share grouped inversions in its 16,384-key cache.
    bands = tuple(
        tuple(
            summarize(counts, leaf_map, atom_weights=w)
            for counts, leaf_map in zip(train_eval, maps, strict=True)
        )
        for w in (_WEIGHTS[coord.dgp], None)
    )
    return _Trial(index, maps, ids, bands)


@dataclass(slots=True)
class _Moments:
    n: _Count
    mean: _Number
    m2: _Number

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> _Moments:
        return cls(np.zeros(shape, dtype=np.int64), np.zeros(shape), np.zeros(shape))

    def add(self, x: _Number, present: _Flag) -> None:
        count = present.sum(axis=0, dtype=np.int64)
        total = self.n + count
        mean = np.sum(x, axis=0, where=present) / np.maximum(count, 1)
        delta = mean - self.mean
        self.m2 += np.sum(
            (x - mean) ** 2, axis=0, where=present
        ) + delta**2 * self.n * count / np.maximum(total, 1)
        self.mean += delta * count / np.maximum(total, 1)
        self.n = total

    def summary(self, key: tuple[int, ...]) -> _Summary:
        n = int(self.n[key])
        mean = float(self.mean[key]) if n else None
        mcse = math.sqrt(float(self.m2[key]) / (n * (n - 1))) if n > 1 else None
        return _Summary(n=n, mean=mean, mcse=mcse)

    def error(self, key: tuple[int, ...]) -> _EndpointError:
        s = self.summary(key)
        rmse = math.sqrt(float(self.mean[key] ** 2 + self.m2[key] / s.n)) if s.n else None
        return _EndpointError(n=s.n, bias=s.mean, rmse=rmse, bias_mcse=s.mcse)

    def spans(self, mode: int, method: int) -> _Spans:
        a, b, c, d, e = (self.summary((mode, method, i)) for i in range(5))
        return _Spans(true_id=a, estimated_id=b, outer=c, inner=d, sampling_excess=e)


@dataclass(frozen=True, slots=True)
class _Block:
    ids: _Number
    outer: _Number
    l_ci: _Number
    u_ci: _Number
    plugin: _Number
    present: _Flag
    empty_arms: _Count
    empty_leaves: _Count
    valid: _Flag

    @classmethod
    def of(cls, trials: list[_Trial]) -> _Block:
        raw = tuple(t.bands for t in trials)
        ids = np.array([t.ids for t in trials], dtype=np.float64)[:, None]
        outer = np.array([[[b.outer for b in group] for group in t] for t in raw], dtype=np.float64)
        l_ci = np.array([[[b.L_ci for b in group] for group in t] for t in raw], dtype=np.float64)
        u_ci = np.array([[[b.U_ci for b in group] for group in t] for t in raw], dtype=np.float64)
        plugin = np.array(
            [[[b.plugin or (0.0, 0.0) for b in group] for group in t] for t in raw],
            dtype=np.float64,
        )
        present = np.array(
            [[[b.plugin is not None for b in group] for group in t] for t in raw], dtype=np.bool_
        )
        empty_arms = np.array(
            [
                [
                    [sum(int(c.n0 == 0) + int(c.n1 == 0) for c in b.leaf_counts) for b in group]
                    for group in t
                ]
                for t in raw
            ],
            dtype=np.int64,
        )
        empty_leaves = np.array(
            [
                [[sum(c.n0 + c.n1 == 0 for c in b.leaf_counts) for b in group] for group in t]
                for t in raw
            ],
            dtype=np.int64,
        )
        valid = np.logical_and.reduce(
            tuple(_valid_interval(v) for v in (outer, l_ci, u_ci, plugin))
        )
        valid &= (outer[..., 0] == l_ci[..., 0]) & (outer[..., 1] == u_ci[..., 1])
        valid &= (l_ci[..., 0] <= u_ci[..., 0] + _ULPS) & (l_ci[..., 1] <= u_ci[..., 1] + _ULPS)
        reason_matches = np.array(
            [
                [
                    [
                        b.reason == (None if b.plugin is not None else "empty_arm_leaf")
                        for b in group
                    ]
                    for group in t
                ]
                for t in raw
            ],
            dtype=np.bool_,
        )
        valid &= reason_matches & (present == (empty_arms == 0))
        return cls(ids, outer, l_ci, u_ci, plugin, present, empty_arms, empty_leaves, valid)

    def select(self, keep: _Flag) -> _Block:
        return _Block(
            self.ids[keep],
            self.outer[keep],
            self.l_ci[keep],
            self.u_ci[keep],
            self.plugin[keep],
            self.present[keep],
            self.empty_arms[keep],
            self.empty_leaves[keep],
            self.valid[keep],
        )


def _valid_interval(x: _Number) -> _Flag:
    lo, hi = x[..., 0], x[..., 1]
    return (0 <= lo) & (lo <= hi) & (hi <= 1)


def _contains(x: _Number, v: _Number | float) -> _Flag:
    return (x[..., 0] <= v + _ULPS) & (v <= x[..., 1] + _ULPS)


@dataclass(frozen=True, slots=True)
class _Actions:
    clear: _Flag
    rule_out: _Flag
    inconclusive: _Flag
    lower_equal: _Flag
    upper_equal: _Flag

    @classmethod
    def of(cls, bounds: _Number, *, present: _Flag, margin: float = 0.0) -> _Actions:
        lo, hi = bounds[..., :1], bounds[..., 1:]
        has = present[..., None]
        clear, rule_out = has & (_TAUS > hi + margin), has & (lo > _TAUS + margin)
        return cls(
            clear, rule_out, has & ~(clear | rule_out), has & (lo == _TAUS), has & (hi == _TAUS)
        )

    def counts(self) -> _Count:
        return np.stack(
            (self.clear, self.rule_out, self.inconclusive, self.lower_equal, self.upper_equal)
        ).sum(axis=1)


def _tally(counts: _Count, key: tuple[int, ...]) -> _Tally:
    clear, rule_out, abstain, lower, upper = map(int, counts[(slice(None), *key)])
    return _Tally(
        evaluated=clear + rule_out + abstain,
        clear=clear,
        rule_out=rule_out,
        inconclusive=abstain,
        lower_equal=lower,
        upper_equal=upper,
    )


def _width(bounds: _Number) -> _Number:
    return bounds[..., 1] - bounds[..., 0]


class _MethodTotals:
    def __init__(self) -> None:
        self.covered = np.zeros((4, 2, 5), dtype=np.int64)
        self.empty = np.zeros((2, 2, 5), dtype=np.int64)
        self.spans = _Moments.zeros((2, 5, 5))
        self.paired = tuple(_Moments.zeros((2, 5, 5)) for _ in range(2))
        self.errors = _Moments.zeros((2, 5, 2))
        self.maps: tuple[Counter[LeafMap], ...] = tuple(Counter() for _ in _METHODS)
        self.h_comparison = np.zeros((3, 2), dtype=np.int64)

    def add(self, block: _Block, trials: list[_Trial], h: float) -> _Flag:
        for i, counts in enumerate(self.maps):
            counts.update(t.maps[i] for t in trials)
        l = _contains(block.l_ci, block.ids[..., 0])
        u = _contains(block.u_ci, block.ids[..., 1])
        joint = l & u
        self.covered += np.stack((l, u, joint, _contains(block.outer, h))).sum(axis=1)
        self.empty += np.stack((block.empty_arms, block.empty_leaves)).sum(axis=1)
        outer = _width(block.outer)
        true = np.broadcast_to(_width(block.ids), outer.shape)
        inner = np.maximum(0.0, block.u_ci[..., 0] - block.l_ci[..., 1])
        spans = np.stack((true, _width(block.plugin), outer, inner, outer - true), axis=-1)
        present = np.ones(spans.shape, dtype=np.bool_)
        present[..., 1] = block.present
        self.spans.add(spans, present)
        for i, paired in enumerate(self.paired):
            paired.add(spans - spans[:, :, i : i + 1], present & present[:, :, i : i + 1])
        self.errors.add(block.plugin - block.ids, block.present[..., None])
        worse = outer[..., 3] > outer[..., 2] + _ULPS
        coarser = true[..., 3] > true[..., 2] + _ULPS
        self.h_comparison += np.stack((worse, worse & coarser, worse & ~coarser)).sum(axis=1)
        return ~joint.all(axis=(1, 2))

    def rows(self, coord: _Coordinate, attempted: int, evaluated: int) -> tuple[_MethodRow, ...]:
        return tuple(
            self._row(mode, method, coord, attempted, evaluated)
            for mode, method in product(range(len(_MODES)), range(len(_METHODS)))
        )

    def _row(
        self, mode: int, method: int, coord: _Coordinate, attempted: int, evaluated: int
    ) -> _MethodRow:
        lo, hi, joint, h = map(int, self.covered[:, mode, method])
        arms, leaves = map(int, self.empty[:, mode, method])
        l_error = self.errors.error((mode, method, 0))
        nulls = evaluated - l_error.n
        training = coord.n // 3 if method >= 3 else 0
        wider, coarser, same = map(int, self.h_comparison[:, mode])
        return _MethodRow(
            method=_METHODS[method],
            mode=_MODES[mode],
            attempted=attempted,
            valid_bands=evaluated,
            plugin_trials=l_error.n,
            null_plugin=nulls,
            null_plugin_reasons=(_CountByReason(reason="empty_arm_leaf", count=nulls),)
            if nulls
            else (),
            empty_arm_leaves=arms,
            empty_membership_leaves=leaves,
            l_covered=lo,
            u_covered=hi,
            jointly_covered=joint,
            h_covered=h,
            plugin_l_error=l_error,
            plugin_u_error=self.errors.error((mode, method, 1)),
            spans=self.spans.spans(mode, method),
            paired_spans_vs_u0=self.paired[0].spans(mode, method),
            paired_spans_vs_f=self.paired[1].spans(mode, method),
            training_n=training,
            estimation_n=coord.n - training,
            leaf_maps=tuple(
                _CountByMap(leaf_map=m, count=count)
                for m, count in sorted(self.maps[method].items())
            ),
            h_vs_q=_HWidthComparison(
                wider_than_q=wider, with_coarser_true_id=coarser, with_same_true_id=same
            )
            if method == 3
            else None,
        )


class _DecisionTotals:
    def __init__(self) -> None:
        shape = (len(_MODES), len(_TARGETS), len(_TAU))
        self.band = np.zeros((5, *shape), dtype=np.int64)
        self.policy = np.zeros((5, len(_TAU)), dtype=np.int64)
        self.ids = np.zeros((5, len(_METHODS), len(_TAU)), dtype=np.int64)
        self.plugin = np.zeros((5, len(_MODES), len(_METHODS), len(_TAU)), dtype=np.int64)
        self.errors = np.zeros((4, *shape), dtype=np.int64)
        self.abstain = np.zeros((3, *shape), dtype=np.int64)
        self.loss = _Moments.zeros((len(_LOSS), 3, *shape))

    def add(
        self,
        block: _Block,
        h: float,
        q: Interval,
        endpoint_miss: _Flag,
    ) -> tuple[_Count, _Flag, _Count]:
        outer = np.concatenate((block.outer, _intersections(block.outer)), axis=2)
        discordant = outer[..., 0] > outer[..., 1]
        states = _Actions.of(outer, present=~discordant, margin=_EQUIV)
        method_id = _Actions.of(
            block.ids[:, 0], present=np.ones(block.ids[:, 0].shape[:-1], dtype=np.bool_)
        )
        self.ids += method_id.counts()
        self.plugin += _Actions.of(block.plugin, present=block.present, margin=_EQUIV).counts()
        self.policy += _Actions.of(
            np.full((len(outer), 2), h), present=np.ones(len(outer), dtype=np.bool_)
        ).counts()
        self.band += states.counts()
        false_clear = states.clear & (h >= _TAUS)
        false_out = states.rule_out & (h <= _TAUS)
        id_low = np.concatenate(
            (block.ids[..., 0], np.full((len(outer), 1, len(_PIPELINES)), q[0])), axis=2
        )
        id_high = np.concatenate(
            (block.ids[..., 1], np.full((len(outer), 1, len(_PIPELINES)), q[1])), axis=2
        )
        false_id_clear = states.clear & (id_high[..., None] >= _TAUS)
        false_id_out = states.rule_out & (id_low[..., None] <= _TAUS)
        self.errors += np.stack((false_clear, false_out, false_id_clear, false_id_out)).sum(axis=1)
        straddles = (id_low[..., None] <= _TAUS) & (_TAUS <= id_high[..., None])
        self.abstain += np.stack(
            (
                states.inconclusive & straddles,
                states.inconclusive & ~straddles,
                np.broadcast_to(discordant[..., None], states.clear.shape),
            )
        ).sum(axis=1)
        abstains = states.inconclusive | discordant[..., None]
        self._add_loss(false_clear, false_out, abstains)
        missed = endpoint_miss | discordant.any(axis=(1, 2))
        clear_any, out_any, id_any = (
            x.any(axis=(1, 2, 3)) for x in (false_clear, false_out, false_id_clear | false_id_out)
        )
        unexpected = (clear_any | out_any | id_any) & ~missed
        power = np.array(
            [
                states.rule_out[:, _ESTIMATED, _P0, _AT_50].sum(),
                states.clear[:, _ESTIMATED, _P0, _AT_50].sum(),
                (
                    abstains[:, _ESTIMATED, _P0, _AT_32] & states.clear[:, _ESTIMATED, _P1, _AT_32]
                ).sum(),
                (
                    abstains[:, _ESTIMATED, _P1, _AT_32] & states.clear[:, _ESTIMATED, _P1H, _AT_32]
                ).sum(),
            ],
            dtype=np.int64,
        )
        return (
            np.array((missed.sum(), clear_any.sum(), out_any.sum()), dtype=np.int64),
            unexpected,
            power,
        )

    def _add_loss(self, false_clear: _Flag, false_out: _Flag, abstains: _Flag) -> None:
        groups = []
        for c_clear, c_out, c_abs in _LOSS:
            loss = c_clear * false_clear + c_out * false_out + c_abs * abstains
            groups.append(np.stack((loss, loss - c_abs, loss - loss[:, :, _P0 : _P0 + 1]), axis=1))
        values = np.stack(groups, axis=1)
        self.loss.add(values, np.ones(values.shape, dtype=np.bool_))

    def rows(self, coord: _Coordinate, attempted: int, evaluated: int) -> tuple[_DecisionRow, ...]:
        return tuple(
            self._row(coord, (mode, target, i), attempted, evaluated)
            for mode, target, i in product(
                range(len(_MODES)), range(len(_TARGETS)), range(len(_TAU))
            )
        )

    def _row(
        self, coord: _Coordinate, key: tuple[int, int, int], attempted: int, evaluated: int
    ) -> _DecisionRow:
        mode, target, i = key
        band = _tally(self.band, key)
        clear, rule_out, id_clear, id_out = map(int, self.errors[(slice(None), *key)])
        id_abs, sampling_abs, diagnostic_abs = map(int, self.abstain[(slice(None), *key)])
        abstention = id_abs + sampling_abs + diagnostic_abs
        return _DecisionRow(
            coordinate=coord,
            mode=_MODES[mode],
            target=_TARGETS[target],
            tau=_TAU[i],
            attempted=attempted,
            evaluated=evaluated,
            clear=band.clear,
            rule_out=band.rule_out,
            inconclusive=abstention,
            false_clear=clear,
            false_rule_out=rule_out,
            false_id_clear=id_clear,
            false_id_rule_out=id_out,
            id_abstain=id_abs,
            sampling_abstain=sampling_abs,
            diagnostic_abstain=diagnostic_abs,
            diagnostic_abstain_reason="discordant_bands" if diagnostic_abs else None,
            rates_per_attempt=_Rates(
                abstention=abstention / attempted,
                false_clear=clear / attempted,
                false_rule_out=rule_out / attempted,
                false_id_clear=id_clear / attempted,
                false_id_rule_out=id_out / attempted,
            ),
            outer_equal=(band.lower_equal, band.upper_equal),
            true_policy=_tally(self.policy, (i,)),
            full_q_id=_tally(self.ids, (2, i)),
            method_id=_tally(self.ids, (target, i)) if target < 5 else None,
            plugin=_tally(self.plugin, key) if target < 5 else None,
            losses=tuple(
                _Loss(
                    costs=costs,
                    proposed=self.loss.summary((j, 0, *key)),
                    vs_never_decide=self.loss.summary((j, 1, *key)),
                    vs_p0=self.loss.summary((j, 2, *key)),
                )
                for j, costs in enumerate(_LOSS)
            ),
        )


def _intersections(outer: _Number) -> _Number:
    return np.stack(
        tuple(
            np.stack(
                (outer[:, :, methods, 0].max(axis=2), outer[:, :, methods, 1].min(axis=2)), axis=-1
            )
            for methods in _PIPELINES
        ),
        axis=2,
    )


def _diagnostic(exc: Exception, stage: str, coordinate: str, **details: str | int) -> _Diagnostic:
    if isinstance(exc, StudyError):
        # Preserve the exact numerical coordinate alongside the trial or file that generated it.
        context = (
            *exc.context,
            ("campaign_stage", stage),
            ("campaign_coordinate", coordinate),
            *details.items(),
        )
        return _Diagnostic(reason=exc.reason, context=context)
    return _Diagnostic(
        reason="execution_failure",
        context=(
            ("stage", stage),
            ("coordinate", coordinate),
            ("exception", type(exc).__name__),
            *details.items(),
        ),
    )


def _counter(coord: _Coordinate, name: str, k: int, n: int, kind: _GateKind) -> _GateCounter:
    interval, status = counter_gate(k, n, kind)
    return _GateCounter(
        coordinate=coord,
        name=name,
        kind=kind,
        count=k,
        attempted=n,
        interval=interval,
        status=status,
    )


def evaluate_cell(
    coord: _Coordinate, attempts: int
) -> tuple[_CellRow, tuple[_DecisionRow, ...], tuple[_GateCounter, ...], tuple[_Diagnostic, ...]]:
    """Aggregate one cell, keeping at most one block of trial results in memory."""
    if attempts <= 0:
        raise _error("invalid_trial_count", "cell", str(coord.dgp), attempted=attempts)
    h = float(population_truth(coord.dgp, _Q)[0])
    u0, f, q = (_identification(coord.dgp, m) for m in _STATIC_MAPS)
    methods, decisions = _MethodTotals(), _DecisionTotals()
    errors, power = np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64)
    evaluated = 0
    failures = []
    coordinate = f"D{coord.dgp}/n{coord.n_index}/e{coord.e_index}"
    for start in range(0, attempts, _BATCH):
        trials = []
        for trial in range(start, min(start + _BATCH, attempts)):
            try:
                trials.append(_trial(coord, trial))
            except Exception as exc:
                failures.append(_diagnostic(exc, "trial", coordinate, trial=trial))
        if not trials:
            continue
        block = _Block.of(trials)
        keep = block.valid.all(axis=(1, 2))
        if not keep.all():
            for t, mode, method in np.argwhere(~block.valid):
                exc = _error("invalid_band", "aggregation", _METHODS[method], mode=_MODES[mode])
                failures.append(_diagnostic(exc, "trial", coordinate, trial=trials[t].index))
            trials = [t for t, ok in zip(trials, keep, strict=True) if ok]
            if not trials:
                continue
            block = block.select(keep)
        evaluated += len(trials)
        miss = methods.add(block, trials, h)
        bad, unexpected, wins = decisions.add(block, h, q, miss)
        errors += bad
        power += wins
        for i in np.flatnonzero(unexpected):
            exc = _error("false_action_without_bound_miss", "aggregation", coordinate)
            failures.append(_diagnostic(exc, "trial", coordinate, trial=trials[i].index))
    cell = _CellRow(
        coordinate=coord,
        attempted=attempts,
        evaluated=evaluated,
        h=h,
        id_u0=u0,
        id_f=f,
        id_q=q,
        methods=methods.rows(coord, attempts, evaluated),
    )
    counters: tuple[_GateCounter, ...] = ()
    if evaluated == attempts:
        counters = tuple(
            _counter(coord, name, int(k), attempts, "error")
            for name, k in zip(_ERRORS, errors, strict=True)
        )
        if coord.n == 4096 and coord.e == 0.5 and 2 <= coord.dgp <= 5:
            j = coord.dgp - 2
            counters += (_counter(coord, _POWER[j], int(power[j]), attempts, "power"),)
    return cell, decisions.rows(coord, attempts, evaluated), counters, tuple(failures)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt(root: Path, path: str) -> _Receipt:
    return _Receipt(path=path, sha256=_hash((root / path).read_bytes()))


def _verify_truth() -> None:
    for d, expected in enumerate(_TRUTH_ANCHORS):
        h, *_ = population_truth(d, _Q)
        endpoints = (v for m in _STATIC_MAPS for v in population_truth(d, m)[1:])
        if (h, *endpoints) != tuple(map(Fraction, expected)):
            raise _error("oracle_truth_mismatch", "truth", f"D{d}")


def _exact_probability(p: _OracleProbability) -> Fraction:
    exact = Fraction(p.exact)
    if not 0 <= exact <= 1 or p.value != float(exact):
        raise ValueError("invalid exact probability")
    return exact


def _verify_oracle(path: Path, protocol: _Receipt, sources: tuple[_Receipt, ...]) -> _Receipt:
    name = str(path.resolve())
    try:
        data = path.read_bytes()
        report = _OracleReport.model_validate_json(data)
    except (OSError, ValueError) as exc:
        raise _error(
            "invalid_oracle_receipt", "provenance", name, exception=type(exc).__name__
        ) from exc
    manifest = report.manifest
    if (
        manifest is None
        or manifest.base_revision != _PROTOCOL_BASE
        or manifest.protocol_sha256 != protocol.sha256
        or tuple((r.path, r.sha256) for r in manifest.inputs) != ((protocol.path, protocol.sha256),)
    ):
        raise _error("oracle_protocol_mismatch", "provenance", name)
    expected = tuple((r.path, r.sha256) for r in sources if r.path in _ORACLE_SOURCES)
    if tuple((r.path, r.sha256) for r in manifest.sources) != expected:
        raise _error("oracle_source_mismatch", "provenance", name)
    if (
        report.status != "passed"
        or report.mismatches != 0
        or report.failures
        or (report.tables, report.marginal_pairs, report.stratum_table_pairs, report.simplex_cases)
        != (165, 81, 1225, 5)
        or report.largest_cp_error is None
        or not 0 <= report.largest_cp_error <= 1e-11
    ):
        raise _error("oracle_checks_failed", "provenance", name)
    grid = (0.0, 0.01, 0.2, 0.4, 0.5, 0.99, 1.0)
    roster = tuple(
        (n0, n1, p, q) for (n0, n1), p, q in product(((4, 4), (9, 11), (25, 20)), grid, grid)
    )
    rows = report.sampling_cells
    if (
        tuple((c.coordinate.n0, c.coordinate.n1, c.coordinate.p, c.coordinate.q) for c in rows)
        != roster
    ):
        raise _error("oracle_coordinate_mismatch", "provenance", name)
    for coord, row in zip(roster, rows, strict=True):
        try:
            box, lo, hi, joint = map(_exact_probability, (row.box, row.L, row.U, row.joint))
            if (
                not 1 - Fraction(4, 600) <= box <= joint <= min(lo, hi)
                or tuple(a.tau for a in row.decisions) != _TAU
                or any(
                    _exact_probability(v) > 1 - joint
                    for a in row.decisions
                    for v in (a.false_clear, a.false_rule_out)
                )
                or (coord == (25, 20, 0.2, 0.4) and abs(float(box) - 0.996327663382) > 1e-11)
            ):
                raise ValueError("inconsistent sampling audit")
        except (ValueError, ZeroDivisionError) as exc:
            raise _error(
                "oracle_sampling_mismatch", "provenance", name, sampling_coordinate=str(coord)
            ) from exc
    return _Receipt(path=name, sha256=_hash(data))


def _status(
    mode: _RunMode, failures: list[_Diagnostic], counters: list[_GateCounter]
) -> _RunStatus:
    if failures:
        return "execution_failure"
    if mode == "smoke":
        return "smoke"
    if any(c.status == "red" for c in counters):
        return "red"
    return "green" if all(c.status == "green" for c in counters) else "inconclusive"


def _write(handle: TextIO, record: _Frozen) -> None:
    handle.write(json.dumps(record.model_dump(mode="json"), allow_nan=False) + "\n")


def _run(mode: _RunMode, out: Path, oracle: Path, raw_args: tuple[str, ...]) -> int:
    root = Path(__file__).resolve().parents[1]
    started_at = datetime.now(UTC)
    threads = tuple((k, os.environ.get(k)) for k in _THREADS)
    trials = _FULL_TRIALS if mode == "full" else _SMOKE_TRIALS
    failures: list[_Diagnostic] = []
    counters: list[_GateCounter] = []
    sources: tuple[_Receipt, ...] = ()
    inputs: tuple[_Receipt, ...] = ()
    revision: str | None = None
    protocol_sha: str | None = None
    oracle_sha: str | None = None
    complete: list[_Coordinate] = []
    out.mkdir(parents=True, exist_ok=False)
    with (
        (out / _ARTIFACTS[0]).open("x", encoding="utf-8") as cells,
        (out / _ARTIFACTS[1]).open("x", encoding="utf-8") as decisions,
    ):
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            protocol = _receipt(root, _PROTOCOL)
            protocol_sha, inputs = protocol.sha256, (protocol,)
            sources = tuple(
                _receipt(root, p) for p in (*_ORACLE_SOURCES, "calibration/harm_campaign.py")
            )
            if protocol_sha != _PROTOCOL_DIGEST:
                raise _error("protocol_digest_mismatch", "provenance", _PROTOCOL)
            oracle_receipt = _verify_oracle(oracle, protocol, sources)
            oracle_sha, inputs = oracle_receipt.sha256, (*inputs, oracle_receipt)
            if any(value != "1" for _, value in threads):
                raise _error("single_thread_environment_required", "provenance", mode)
            _verify_truth()
            roster = coordinates()
            for coord in roster:
                try:
                    cell, rows, gates, diagnostics = evaluate_cell(coord, trials)
                    _write(cells, cell)
                    for row in rows:
                        _write(decisions, row)
                    failures.extend(diagnostics)
                    counters.extend(gates)
                    if not diagnostics and cell.attempted == cell.evaluated == trials:
                        complete.append(coord)
                except Exception as exc:
                    failures.append(
                        _diagnostic(exc, "cell", f"D{coord.dgp}/n{coord.n_index}/e{coord.e_index}")
                    )
            if not failures and (
                len(roster) != 28 or tuple(complete) != roster or len(counters) != _GATES
            ):
                raise _error(
                    "missing_campaign_coordinates",
                    "aggregation",
                    "roster",
                    completed=len(complete),
                    counters=len(counters),
                )
        except Exception as exc:
            failures.append(_diagnostic(exc, "campaign", mode))
    status = _status(mode, failures, counters)
    eligible = mode == "full" and status != "execution_failure"
    exit_code = (
        1 if status in ("red", "execution_failure") else 2 if status == "inconclusive" else 0
    )
    with (out / _ARTIFACTS[2]).open("x", encoding="utf-8") as gates_file:
        _write(
            gates_file,
            _Gates(status=status, certification_eligible=eligible, counters=tuple(counters)),
        )
    manifest = _Manifest(
        mode=mode,
        certification_eligible=eligible,
        status=status,
        exit_code=exit_code,
        source_revision=revision,
        protocol_sha256=protocol_sha,
        oracle_sha256=oracle_sha,
        sources=sources,
        inputs=inputs,
        artifacts=tuple(_receipt(out, path) for path in _ARTIFACTS),
        invocation=(sys.executable, "-m", "calibration.harm_campaign", *raw_args),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pydantic=pydantic.__version__,
        thread_environment=threads,
        started_at=started_at,
        ended_at=datetime.now(UTC),
        trials_per_cell=trials,
        completed_cells=len(complete),
        failures=tuple(failures),
    )
    with (out / "manifest.json").open("x", encoding="utf-8") as handle:
        _write(handle, manifest)
    print(
        f"mode={mode} status={status} completed_cells={len(complete)} trials_per_cell={trials} counters={len(counters)} failures={len(failures)}"
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Run the frozen synthetic study, writing a new receipt-bound packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        help="reviewed exact-check receipt (default: ../oracle.json relative to --out)",
    )
    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    return _run(
        args.mode,
        args.out,
        args.oracle if args.oracle is not None else args.out.parent / "oracle.json",
        raw_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
