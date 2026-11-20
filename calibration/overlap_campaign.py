"""Frozen overlap qualification on independent synthetic experiments."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import repeat
from math import inf, nextafter, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from scipy.special import ndtr
from scipy.stats import beta

from increment import __version__
from increment._overlap_evidence import MAX_EXACT_INTEGER, overlap_binary_moments
from increment.errors import CodedError
from increment.power.overlap import (
    JointDecision,
    OverlapDraw,
    PlanningRow,
    prepare_overlap_scenario,
)
from increment.semantics.overlap import (
    CONTRAST_IDS,
    CellId,
    OverlapCell,
    OverlapClaim,
    OverlapFactor,
    OverlapLaw,
    OverlapMetric,
    OverlapPlan,
)
from tests.oracles.overlap_design import (
    RegressionBatch,
    contrast_matrix,
    exact_contrast_rows,
    occupancy_probability,
    reference_support_violations,
    regress_raw_trials,
)

_SEED = 2026111983
_REPLICATIONS = 20_000
_BATCH = 64
_EMPIRICAL_ROWS = 256
_EXACT_GATES = 88
_ALPHA = 0.05
_MC_ALPHA = 0.01 / _EXACT_GATES
_PROTOCOL_DIGEST = "5863fb24ba484fcaf0ab9ce6338f0799f38b0c75371367da17135583375021ad"
_CELLS: tuple[CellId, ...] = ("00", "10", "01", "11")
_NAMES = ("value", "cost", "activity")
_ROW_KEYS = tuple((name, contrast) for name in _NAMES for contrast in CONTRAST_IDS)
_JOINT = 6
_INTERACTION = 7
_ROLE_JOINT = (0, 1)
_SCALE = np.asarray((1.0, 1.5, 1.0, 2.0))
_DIRECTION = np.asarray(((1,) * 7 + (0,), (-1,) * 8, (0,) * 8))
_THRESHOLDS = (Fraction(), Fraction(1, 4), Fraction())
_BOUNDARY = np.broadcast_to(np.asarray(_THRESHOLDS, dtype=float)[:, None], (3, 8))
_ZERO_MEANS = (Fraction(),) * 4
_MIXED_MEANS = tuple(map(Fraction, ("0", "0.3", "0.2", "0.8")))
_MIXED_RISK_INCREMENTS = tuple(map(Fraction, ("0", "0.02", "0.01", "0.04")))
_MIXED_COST = (Fraction(),) * 3 + (_THRESHOLDS[1],)


@dataclass(frozen=True)
class _Regime:
    name: str
    n: int
    weights_a: tuple[int, int]
    weights_b: tuple[int, int]
    dense: bool = True
    skew: bool = False
    risk: Fraction | None = None

    @property
    def a(self) -> float:
        return self.weights_a[1] / sum(self.weights_a)

    @property
    def b(self) -> float:
        return self.weights_b[1] / sum(self.weights_b)


_REGIMES = (
    _Regime("dense_normal", 512, (1, 1), (1, 1)),
    _Regime("unequal_normal", 4096, (3, 1), (1, 3)),
    _Regime("dense_skew", 4096, (1, 1), (1, 1), skew=True),
    _Regime("dense_binary", 4096, (1, 1), (1, 1), risk=Fraction("0.20")),
    _Regime("sparse_joint", 48, (9, 1), (4, 1), dense=False),
    _Regime("rare_binary", 512, (3, 1), (1, 3), dense=False, risk=Fraction("0.005")),
)


@dataclass(frozen=True)
class _Case:
    regime: _Regime
    mixed: bool
    stream: np.random.SeedSequence

    @property
    def name(self) -> str:
        return f"{self.regime.name}/{'mixed_null' if self.mixed else 'global_null'}"

    @property
    def means(self) -> NDArray[np.float64]:
        cost = np.asarray(_MIXED_COST if self.mixed else _ZERO_MEANS, dtype=float)
        risk = self.regime.risk
        if risk is None:
            value = np.asarray(_MIXED_MEANS if self.mixed else _ZERO_MEANS, dtype=float)
        else:
            # Keep the generator's original binary64 addition of risk and increment.
            value = float(risk) + np.asarray(
                _MIXED_RISK_INCREMENTS if self.mixed else _ZERO_MEANS, dtype=float
            )
        return np.column_stack((value, cost, value))

    @property
    def cell_means(self) -> tuple[tuple[Fraction, ...], ...]:
        cost = _MIXED_COST if self.mixed else _ZERO_MEANS
        risk = self.regime.risk
        if risk is None:
            value = _MIXED_MEANS if self.mixed else _ZERO_MEANS
        else:
            increments = _MIXED_RISK_INCREMENTS if self.mixed else _ZERO_MEANS
            value = tuple(risk + increment for increment in increments)
        return tuple(zip(value, cost, value, strict=True))


class CampaignRow(PlanningRow, Protocol):
    """Readout fields needed to measure band issuance and coverage."""

    @property
    def lower(self) -> float | None: ...

    @property
    def upper(self) -> float | None: ...


type EvaluateTrial = Callable[
    [OverlapPlan, OverlapDraw], tuple[tuple[CampaignRow, ...], JointDecision]
]


def _factor(name: str, weights: tuple[int, int]) -> OverlapFactor:
    return OverlapFactor(
        target=name,
        experiment=name,
        revision="1",
        off="off",
        on="on",
        off_weight=weights[0],
        on_weight=weights[1],
        arm_draw_id=name,
        layer=None,
    )


def _plan(case: _Case) -> OverlapPlan:
    kind = "mean" if case.regime.risk is None else "binary_mean"
    metrics = (
        OverlapMetric(
            name="value",
            unit="value",
            definition_sha256="1" * 64,
            kind=kind,
            role="primary",
            better="higher",
        ),
        OverlapMetric(
            name="cost",
            unit="cost",
            definition_sha256="2" * 64,
            kind="mean",
            role="guardrail",
            better="lower",
            margin_abs=0.25,
        ),
        OverlapMetric(
            name="activity",
            unit="activity",
            definition_sha256="3" * 64,
            kind=kind,
            role="secondary",
            better="higher",
        ),
    )
    claims = tuple(
        OverlapClaim(
            metric=metric,
            contrast=contrast,
            alternative=(
                "noninferiority"
                if metric == "cost"
                else "difference"
                if metric == "activity" or contrast == "interaction"
                else "benefit"
            ),
        )
        for metric in _NAMES
        for contrast in CONTRAST_IDS
    )
    return OverlapPlan.model_validate(
        {
            "study_id": case.name,
            "unit_namespace": "synthetic",
            "population_id": "synthetic-entrants",
            "eligibility_rule_sha256": "4" * 64,
            "enrollment_rule_sha256": "5" * 64,
            "registered_at": "2020-01-01T00:00:00Z",
            "entry_open": "2020-01-02T00:00:00Z",
            "start": "2020-01-03T00:00:00Z",
            "end": "2020-01-04T00:00:00Z",
            "target": "jointly_eligible",
            "a": _factor("a", case.regime.weights_a),
            "b": _factor("b", case.regime.weights_b),
            "law": OverlapLaw(
                scheme="independent_unit_bernoulli",
                origin="external",
                archive_sha256=("6" * 64, "7" * 64),
                gate_basis="preassigned_roster",
                admission="both_guaranteed",
                rule_source_id="synthetic",
                rule_version="1",
                serving_barrier_id="synthetic",
            ),
            "metrics": metrics,
            "claims": claims,
            "alpha": _ALPHA,
        }
    )


def _rng(stream: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream))


def _generate_outcomes(
    rng: np.random.Generator,
    regime: _Regime,
    means: NDArray[np.float64],
    cells: int | NDArray[np.int64],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    normals = rng.standard_normal((*shape, 2))
    z, u = normals[..., 0], normals[..., 1]
    v = -0.5 * z + sqrt(0.75) * u
    outcomes = np.empty((*shape, 3))
    if regime.skew:
        z_noise, v_noise = np.expm1(z - 0.5), np.expm1(v - 0.5)
    else:
        z_noise, v_noise = z, v
    outcomes[..., 1] = means[cells, 1] + _SCALE[cells] * v_noise
    if regime.risk is None:
        outcomes[..., 0] = means[cells, 0] + _SCALE[cells] * z_noise
        outcomes[..., 2] = means[cells, 2] + z_noise
    else:
        outcomes[..., 0] = outcomes[..., 2] = ndtr(z) < means[cells, 0]
    return outcomes


def _empirical_law(case: _Case, stream: np.random.SeedSequence) -> NDArray[np.float64]:
    rng, means = _rng(stream), case.means
    return np.stack(
        [_generate_outcomes(rng, case.regime, means, j, (_EMPIRICAL_ROWS,)) for j in range(4)]
    )


def _normal_cell(metric: OverlapMetric, cell: CellId, values: NDArray[np.float64]) -> OverlapCell:
    n = len(values)
    success = int(values.sum()) if metric.kind == "binary_mean" else None
    anchor = delta = m2 = None
    if success is not None and n:
        anchor, delta, m2 = overlap_binary_moments(n, success)
    elif n:
        anchor = float(values.min())
        centered = values - anchor
        delta = float(centered.mean())
        residual = centered - delta
        m2 = float(residual @ residual)
    return OverlapCell(
        metric=metric.name,
        cell=cell,
        n_assigned=n,
        n_observed=n,
        n_missing=0,
        n_pending=0,
        n_success=success,
        anchor=anchor,
        mean_delta=delta,
        m2=m2,
    )


def _parametric_draws(
    case: _Case, plan: OverlapPlan, stream: np.random.SeedSequence
) -> Iterator[OverlapDraw]:
    rng, means = _rng(stream), case.means
    p = [float(probability) for probability in plan.cell_probabilities]
    for _ in range(_REPLICATIONS):
        n0, n1, n2, n3 = map(int, rng.multinomial(case.regime.n, p))
        counts = n0, n1, n2, n3
        vectors = [
            _generate_outcomes(rng, case.regime, means, j, (n,)) for j, n in enumerate(counts)
        ]
        yield OverlapDraw(
            counts,
            tuple(
                _normal_cell(metric, cell, values[:, i])
                for i, metric in enumerate(plan.metrics)
                for cell, values in zip(_CELLS, vectors, strict=True)
            ),
        )


@dataclass(frozen=True)
class _Trials:
    regression: RegressionBatch
    joint: NDArray[np.bool_]
    invalid_ready: int
    invalid_approvals: int


def _allocate_batch(reps: int) -> RegressionBatch:
    shape = (reps, 3, 8)
    return RegressionBatch(
        np.empty((reps, 4), dtype=np.int64),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.zeros(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.bool_),
        {},
    )


def _fill_runtime_trial(
    data: RegressionBatch, index: int, draw: OverlapDraw, rows: tuple[CampaignRow, ...]
) -> None:
    data.counts[index] = draw.counts
    for k, (row, (name, contrast)) in enumerate(zip(rows, _ROW_KEYS, strict=True)):
        if (row.metric, row.contrast) != (name, contrast):
            raise ValueError("Readout did not preserve the registered row order")
        j, c = divmod(k, 8)
        if row.estimate is not None:
            data.estimate[index, j, c] = row.estimate
        if row.se is not None:
            data.se[index, j, c] = row.se
        if row.lower is not None and row.upper is not None:
            data.lo[index, j, c] = row.lower
            data.hi[index, j, c] = row.upper
        data.favorable[index, j, c] = row.decision == (
            "reject" if _DIRECTION[j, c] == 0 else "clear"
        )
        data.ready[index, j, c] = row.decision in (
            "clear",
            "reject",
            "loss_beyond_tolerance",
            "inconclusive",
        )
        for _, reason in row.reasons:
            if reason not in data.reasons:
                data.reasons[reason] = np.zeros_like(data.ready)
            data.reasons[reason][index, j, c] = True


def _runtime_trials(
    plan: OverlapPlan,
    draws: Iterator[OverlapDraw],
    contrast: NDArray[np.float64],
    evaluate: EvaluateTrial,
    attempts: dict[str, int],
    key: str,
) -> _Trials:
    data = _allocate_batch(_REPLICATIONS)
    joint = np.zeros(_REPLICATIONS, dtype=np.bool_)
    supported = np.empty((_REPLICATIONS, 3, 4), dtype=np.bool_)
    for index in range(_REPLICATIONS):
        attempts[key] += 1
        draw = next(draws)
        rows, verdict = evaluate(plan, draw)
        _fill_runtime_trial(data, index, draw, rows)
        joint[index] = verdict == "criteria_met"
        supported[index].flat[:] = [
            cell.n_observed >= 30
            and cell.m2 is not None
            and cell.m2 > 0
            and (cell.n_success is None or 10 <= cell.n_success <= cell.n_observed - 10)
            for cell in draw.cells
        ]
    unsupported = (~supported[:, :, None, :] & (contrast[None, None, :, :] != 0)).any(axis=-1)
    false_joint = joint & ~data.favorable[:, _ROLE_JOINT, _JOINT].all(axis=-1)
    return _Trials(
        data,
        joint,
        int((data.ready & unsupported).sum()),
        int((data.favorable & unsupported).sum() + false_joint.sum()),
    )


def _reference_trials(
    case: _Case,
    stream: np.random.SeedSequence,
    contrast: NDArray[np.float64],
    attempts: dict[str, int],
    key: str,
    empirical: NDArray[np.float64] | None = None,
) -> _Trials:
    data = _allocate_batch(_REPLICATIONS)
    rng, means, regime = _rng(stream), case.means, case.regime
    binary = () if regime.risk is None else (0, 2)
    invalid_ready = invalid_approvals = 0
    for first in range(0, _REPLICATIONS, _BATCH):
        n = min(_REPLICATIONS - first, _BATCH)
        shape = (n, regime.n)
        attempts[key] += n
        uniforms = rng.random((*shape, 2))
        assignments = (uniforms[..., 0] < regime.a).astype(np.int64) + 2 * (
            uniforms[..., 1] < regime.b
        )
        if empirical is None:
            observations = _generate_outcomes(rng, regime, means, assignments, shape)
        else:
            observations = empirical[assignments, rng.integers(_EMPIRICAL_ROWS, size=shape)]
        ref = regress_raw_trials(
            assignments,
            observations,
            contrast=contrast,
            binary_metrics=binary,
            raw_boundary=_BOUNDARY,
            direction=_DIRECTION,
            alpha=_ALPHA,
        )
        bad_ready, bad_approvals = reference_support_violations(
            assignments,
            observations,
            contrast=contrast,
            binary_metrics=binary,
            ready=ref.ready,
            favorable=ref.favorable,
        )
        invalid_ready += bad_ready
        invalid_approvals += bad_approvals
        batch = slice(first, first + n)
        for dest, src in zip(
            (data.counts, data.estimate, data.se, data.lo, data.hi, data.favorable, data.ready),
            (ref.counts, ref.estimate, ref.se, ref.lo, ref.hi, ref.favorable, ref.ready),
            strict=True,
        ):
            dest[batch] = src
        for reason, mask in ref.reasons.items():
            if reason not in data.reasons:
                data.reasons[reason] = np.zeros_like(data.ready)
            data.reasons[reason][batch] = mask
    joint = data.favorable[:, _ROLE_JOINT, _JOINT].all(axis=-1)
    return _Trials(data, joint, invalid_ready, invalid_approvals)


@dataclass(frozen=True)
class QualificationTruth:
    """Exact claim values and precomputed comparisons for qualification."""

    values: tuple[tuple[Fraction, ...], ...]
    null: NDArray[np.bool_]
    _max_lower: NDArray[np.float64]
    _min_upper: NDArray[np.float64]

    @classmethod
    def from_cell_means(
        cls,
        means: Sequence[Sequence[Fraction]],
        *,
        weights_a: tuple[int, int],
        weights_b: tuple[int, int],
    ) -> QualificationTruth:
        """Classify the registered directions against exact contrast values."""
        a = Fraction(weights_a[1], sum(weights_a))
        b = Fraction(weights_b[1], sum(weights_b))
        values = exact_contrast_rows(means, a, b)
        null = np.empty(_DIRECTION.shape, dtype=np.bool_)
        max_lower = np.empty(_DIRECTION.shape)
        min_upper = np.empty(_DIRECTION.shape)
        for j, (row, threshold) in enumerate(zip(values, _THRESHOLDS, strict=True)):
            for c, truth in enumerate(row):
                direction = int(_DIRECTION[j, c])
                difference = truth - threshold
                null[j, c] = difference == 0 if direction == 0 else direction * difference <= 0
                nearest = float(truth)
                # A finite float band contains an exact value iff its lower bound
                # is at most the floor float and its upper bound at least the ceiling.
                max_lower[j, c] = nearest if nearest <= truth else nextafter(nearest, -inf)
                min_upper[j, c] = nearest if nearest >= truth else nextafter(nearest, inf)
        return cls(values, null, max_lower, min_upper)

    @classmethod
    def from_empirical_vectors(
        cls,
        vectors: NDArray[np.float64],
        *,
        weights_a: tuple[int, int],
        weights_b: tuple[int, int],
    ) -> QualificationTruth:
        """Treat each observed binary64 outcome as its exact represented value."""
        means = tuple(
            tuple(
                sum((Fraction(float(x)) for x in column), Fraction()) / len(column)
                for column in cell.T
            )
            for cell in vectors
        )
        return cls.from_cell_means(means, weights_a=weights_a, weights_b=weights_b)

    def coverage(self, lo: NDArray[np.float64], hi: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return simultaneous coverage of the exact claims by finite band bounds."""
        return ((lo <= self._max_lower) & (self._min_upper <= hi)).all(axis=(1, 2))


class _Rate(TypedDict):
    count: int
    rate: float


def _rate(mask: NDArray[np.bool_]) -> _Rate:
    k = int(mask.sum())
    return {"count": k, "rate": k / _REPLICATIONS}


def _claim_summary(
    trials: RegressionBatch, truth: QualificationTruth, j: int, c: int
) -> dict[str, object]:
    value = truth.values[j][c]
    estimates, standard_errors = trials.estimate[:, j, c], trials.se[:, j, c]
    points = estimates[np.isfinite(estimates)]
    fitted = standard_errors[np.isfinite(standard_errors)]
    return {
        "metric": _NAMES[j],
        "contrast": CONTRAST_IDS[c],
        "truth": float(value),
        "threshold": float(_BOUNDARY[j, c]),
        "false_claim": bool(truth.null[j, c]),
        "n_point": len(points),
        "n_fit": len(fitted),
        "n_decision_ready": int(trials.ready[:, j, c].sum()),
        "n_abstain": _REPLICATIONS - int(trials.ready[:, j, c].sum()),
        "favorable": _rate(trials.favorable[:, j, c]),
        "bias_given_point": float(Fraction(float(points.mean())) - value) if len(points) else None,
        "empirical_sd_given_point": float(points.std(ddof=1)) if len(points) >= 2 else None,
        "mean_se_given_fit": float(fitted.mean()) if len(fitted) else None,
        "se_quantiles_given_fit_05_50_95": np.quantile(fitted, (0.05, 0.50, 0.95)).tolist()
        if len(fitted)
        else None,
        "reason_counts": {
            reason: int(mask[:, j, c].sum())
            for reason, mask in trials.reasons.items()
            if mask[:, j, c].any()
        },
    }


class _Summary(TypedDict):
    attempts: int
    invalid_ready: int
    invalid_approvals: int
    all_cells_at_least_two: _Rate
    all_bands_issued: _Rate
    all_bands_issued_and_cover: _Rate
    all_bands_issued_and_miss: _Rate
    not_all_bands_issued: _Rate
    fwer: _Rate
    false_compound_joint: _Rate
    compound_joint: _Rate
    primary_joint: _Rate
    interaction: dict[str, _Rate]
    claims: list[dict[str, object]]


def _summarize(trials: _Trials, truth: QualificationTruth) -> _Summary:
    d = trials.regression
    null = truth.null
    all_issued = (np.isfinite(d.lo) & np.isfinite(d.hi)).all(axis=(1, 2))
    cover = truth.coverage(d.lo, d.hi)
    return {
        "attempts": _REPLICATIONS,
        "invalid_ready": trials.invalid_ready,
        "invalid_approvals": trials.invalid_approvals,
        "all_cells_at_least_two": _rate((d.counts >= 2).all(axis=1)),
        "all_bands_issued": _rate(all_issued),
        "all_bands_issued_and_cover": _rate(all_issued & cover),
        "all_bands_issued_and_miss": _rate(all_issued & ~cover),
        "not_all_bands_issued": _rate(~all_issued),
        "fwer": _rate((d.favorable & null).any(axis=(1, 2))),
        "false_compound_joint": _rate(trials.joint & null[_ROLE_JOINT, _JOINT].any()),
        "compound_joint": _rate(trials.joint),
        "primary_joint": _rate(d.favorable[:, 0, _JOINT]),
        "interaction": {
            name: _rate(d.favorable[:, j, _INTERACTION]) for j, name in enumerate(_NAMES)
        },
        "claims": [_claim_summary(d, truth, j, c) for j in range(3) for c in range(8)],
    }


def _bounds(k: int, tail: float) -> tuple[float, float]:
    r = _REPLICATIONS
    lo = float(beta.ppf(tail, k, r - k + 1)) if k else 0.0
    hi = float(beta.isf(tail, k + 1, r - k)) if k < r else 1.0
    return lo, hi


def _parametric_gates(
    summary: _Summary,
    exact_occupancy: float | None,
) -> list[dict[str, object]]:
    gates: list[dict[str, object]] = []
    for name, rate in (
        ("fwer", summary["fwer"]),
        ("false_compound_joint", summary["false_compound_joint"]),
    ):
        upper = _bounds(rate["count"], _MC_ALPHA)[1]
        gates.append({"quantity": name, "upper": upper, "max": 0.06, "passed": upper <= 0.06})
    if exact_occupancy is None:
        for name, rate, limit in (
            ("all_bands_issued", summary["all_bands_issued"], 0.99),
            ("all_bands_issued_and_cover", summary["all_bands_issued_and_cover"], 0.94),
        ):
            lower = _bounds(rate["count"], _MC_ALPHA)[0]
            gates.append({"quantity": name, "lower": lower, "min": limit, "passed": lower >= limit})
    else:
        lo, hi = _bounds(summary["all_cells_at_least_two"]["count"], _MC_ALPHA / 2)
        gates.append(
            {
                "quantity": "occupancy",
                "interval": (lo, hi),
                "oracle_probability": exact_occupancy,
                "passed": lo <= exact_occupancy <= hi,
            }
        )
    return gates


def _matched_gates(runtime: _Summary, reference: _Summary) -> list[dict[str, object]]:
    gates: list[dict[str, object]] = []
    for name, first, second in (
        ("primary_joint", runtime["primary_joint"], reference["primary_joint"]),
        ("primary_interaction", runtime["interaction"]["value"], reference["interaction"]["value"]),
        ("compound_joint", runtime["compound_joint"], reference["compound_joint"]),
    ):
        x, y = first["rate"], second["rate"]
        allowed = 0.01 + 4 * sqrt((x * (1 - x) + y * (1 - y)) / _REPLICATIONS)
        gates.append(
            {
                "quantity": name,
                "runtime": x,
                "reference": y,
                "tolerance": allowed,
                "passed": abs(x - y) <= allowed,
            }
        )
    return gates


_RUN_KEYS = (
    "runtime_parametric",
    "reference_parametric",
    "runtime_empirical",
    "reference_empirical",
)


def _case_record(case: _Case, evaluate: EvaluateTrial) -> dict[str, object]:
    regime = case.regime
    attempts = dict.fromkeys(_RUN_KEYS, 0)
    summaries: dict[str, _Summary] = {}
    gates: dict[str, list[dict[str, object]]] = {}
    record: dict[str, object] = {
        "case": case.name,
        "n_target": regime.n,
        "a": regime.a,
        "b": regime.b,
        "spawn_key": case.stream.spawn_key,
        "attempts": attempts,
        "summaries": summaries,
        "gates": gates,
    }
    try:
        _run_case(case, evaluate, record, attempts, summaries)
    except Exception as error:
        fault: dict[str, object] = {"type": type(error).__name__}
        if isinstance(error, CodedError):
            fault.update(code=error.code, context=dict(error.context))
        else:
            fault["detail"] = str(error)
        record.update(fault=fault, passed=False)
        return record
    for key, summary in summaries.items():
        gates[key] = [
            {
                "quantity": "unsupported_decisions",
                "count": summary["invalid_ready"],
                "invalid_approvals": summary["invalid_approvals"],
                "passed": summary["invalid_ready"] == summary["invalid_approvals"] == 0,
            }
        ]
        if key.endswith("parametric"):
            exact_occupancy = (
                occupancy_probability(regime.n, _probabilities(regime), (2, 2, 2, 2))
                if not regime.dense
                else None
            )
            gates[key].extend(_parametric_gates(summary, exact_occupancy))
    gates["matched"] = _matched_gates(
        summaries["runtime_empirical"], summaries["reference_empirical"]
    )
    record["passed"] = all(gate["passed"] for records in gates.values() for gate in records)
    return record


def _probabilities(regime: _Regime) -> tuple[float, float, float, float]:
    a, b = regime.a, regime.b
    return (1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b


def _run_case(
    case: _Case,
    evaluate: EvaluateTrial,
    record: dict[str, object],
    attempts: dict[str, int],
    summaries: dict[str, _Summary],
) -> None:
    plan = _plan(case)
    seeds = case.stream.spawn(5)
    contrast = contrast_matrix(case.regime.a, case.regime.b)
    weights = {"weights_a": case.regime.weights_a, "weights_b": case.regime.weights_b}
    parametric_truth = QualificationTruth.from_cell_means(case.cell_means, **weights)
    empirical = _empirical_law(case, seeds[2])
    empirical_truth = QualificationTruth.from_empirical_vectors(empirical, **weights)
    seed = int(seeds[3].generate_state(1, dtype=np.uint64)[0]) & MAX_EXACT_INTEGER
    scenario = prepare_overlap_scenario(
        plan,
        n_target=case.regime.n,
        cell_outcomes={
            cell: pd.DataFrame(vectors, columns=list(_NAMES))
            for cell, vectors in zip(_CELLS, empirical, strict=True)
        },
    )
    record.update(
        runtime_empirical_seed=seed,
        scenario_id=scenario.scenario_id,
        family_sha256=plan.family_sha256,
    )
    for key, draws, truth in (
        ("runtime_parametric", _parametric_draws(case, plan, seeds[0]), parametric_truth),
        (
            "runtime_empirical",
            scenario.draws(replications=_REPLICATIONS, seed=seed),
            empirical_truth,
        ),
    ):
        summaries[key] = _summarize(
            _runtime_trials(plan, draws, contrast, evaluate, attempts, key), truth
        )
    for key, stream, law, truth in (
        ("reference_parametric", seeds[1], None, parametric_truth),
        ("reference_empirical", seeds[4], empirical, empirical_truth),
    ):
        summaries[key] = _summarize(
            _reference_trials(case, stream, contrast, attempts, key, law), truth
        )


def _checkpoint(path: Path, report: dict[str, object]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(report, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
    tmp_path.replace(path)


def _records(workers: int, evaluate: EvaluateTrial) -> Iterator[dict[str, object]]:
    cases = [
        _Case(regime, mixed, stream)
        for (regime, mixed), stream in zip(
            ((regime, mixed) for regime in _REGIMES for mixed in (False, True)),
            np.random.SeedSequence(_SEED).spawn(2 * len(_REGIMES)),
            strict=True,
        )
    ]
    if workers == 1:
        yield from map(_case_record, cases, repeat(evaluate))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_case_record, cases, repeat(evaluate))


def main(evaluate: EvaluateTrial, argv: Sequence[str] | None = None) -> int:
    """Run the frozen campaign once and checkpoint the outcome to a new JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("overlap-qualification.json"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 2 * len(_REGIMES):
        parser.error("--workers must be between 1 and the number of cases")
    digest = sha256(args.protocol.read_bytes()).hexdigest()
    if digest != _PROTOCOL_DIGEST:
        parser.error("protocol does not match the frozen overlap qualification")
    records: list[dict[str, object]] = []
    failures: list[str] = []
    report: dict[str, object] = {
        "protocol_sha256": digest,
        "status": "running",
        "seed": _SEED,
        "generator": "PCG64",
        "replications_per_algorithm": _REPLICATIONS,
        "exact_binomial_gates": _EXACT_GATES,
        "records": records,
        "failures": failures,
        "versions": {
            "increment": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    try:
        for record in _records(args.workers, evaluate):
            records.append(record)
            if not record["passed"]:
                failures.append(str(record["case"]))
            _checkpoint(args.out, report)
    except Exception as error:
        report["execution_fault"] = {"type": type(error).__name__, "detail": str(error)}
        failures.append("campaign_execution")
    report["status"] = "failed" if failures else "passed"
    _checkpoint(args.out, report)
    return int(bool(failures))
