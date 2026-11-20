"""Prospective power from paired empirical outcomes under independent assignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from math import frexp, isfinite, ldexp
from types import MappingProxyType
from typing import Literal, Protocol

import narwhals as nw
import numpy as np
from narwhals.exceptions import NarwhalsError
from narwhals.typing import IntoDataFrame
from numpy.typing import NDArray
from pydantic import FiniteFloat
from scipy.stats import beta

from increment._canonical import canonical_json_bytes
from increment._overlap_evidence import (
    CELL_IDS,
    MAX_EXACT_INTEGER,
    Count,
    Identity,
    Ordered,
    OverlapModel,
    Sha256,
    overlap_binary_moments,
)
from increment._overlap_refusals import NUMERIC_RANGE, PLAN_INVALID
from increment.errors import refuse
from increment.semantics.overlap import CellId, ContrastId, OverlapCell, OverlapMetric, OverlapPlan

type CellCounts = tuple[int, int, int, int]
type ClaimKey = tuple[str, ContrastId]
type JointDecision = Literal["not_requested", "insufficient_evidence", "not_met", "criteria_met"]


@dataclass(frozen=True)
class OverlapDraw:
    """Four assigned counts and complete metric-major statistics, not a source receipt."""

    counts: CellCounts
    cells: tuple[OverlapCell, ...]


class PlanningRow(Protocol):
    """The decision fields used to summarize prospective draws."""

    @property
    def metric(self) -> str: ...

    @property
    def contrast(self) -> ContrastId: ...

    @property
    def estimate(self) -> float | None: ...

    @property
    def se(self) -> float | None: ...

    @property
    def decision(self) -> str | None: ...

    @property
    def reasons(self) -> tuple[tuple[CellId | None, str], ...]: ...


type EvaluateDraw = Callable[[OverlapDraw], tuple[tuple[PlanningRow, ...], JointDecision]]


class OverlapPowerClaim(OverlapModel):
    """Unconditional favorable decisions out of all draws for one declared claim."""

    metric: Identity
    contrast: ContrastId
    truth: FiniteFloat
    threshold: FiniteFloat
    n_point: Count
    n_fit: Count
    n_decision_ready: Count
    n_reject: Count
    rejection_rate: FiniteFloat
    power: FiniteFloat | None
    rate_kind: Literal["power", "type_i_error"]
    mc_interval: Ordered[tuple[FiniteFloat, FiniteFloat]]
    reason_counts: Ordered[tuple[tuple[str, Count], ...]]


class OverlapPowerResult(OverlapModel):
    """Design-conditional decision frequencies and their Monte Carlo uncertainty."""

    scenario_id: Sha256
    plan_sha256: Sha256
    family_sha256: Sha256
    population_id: Identity
    n_target: Count
    expected_counts: Ordered[tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]]
    seed: Count
    generator: Literal["PCG64"] = "PCG64"
    numpy_version: Identity
    replications: Count
    claims: Ordered[tuple[OverlapPowerClaim, ...]]
    joint_criteria_met_count: Count
    joint_criteria_met_rate: FiniteFloat
    model: Literal["iid_paired_empirical"] = "iid_paired_empirical"
    assumptions: tuple[str, ...] = (
        "Independent target entrants and the registered assignment probabilities",
        "Independent sampling with replacement of paired outcomes within assignment cell",
        "Outcomes match the registered definitions, units, and window",
        "Complete mature measurement and delivery as contemplated in the scenario",
    )

    @property
    def per_claim(self) -> Mapping[ClaimKey, OverlapPowerClaim]:
        """Claim summaries keyed by metric and contrast, in registered order."""
        return MappingProxyType({(row.metric, row.contrast): row for row in self.claims})

    @property
    def interaction_reject_count(self) -> Mapping[ClaimKey, int]:
        """Favorable decisions for the registered interaction claims."""
        return MappingProxyType(
            {
                (row.metric, row.contrast): row.n_reject
                for row in self.claims
                if row.contrast == "interaction"
            }
        )

    @property
    def interaction_reject_rate(self) -> Mapping[ClaimKey, float]:
        """Favorable interaction decisions divided by all replications."""
        return MappingProxyType(
            {
                (row.metric, row.contrast): row.rejection_rate
                for row in self.claims
                if row.contrast == "interaction"
            }
        )


@dataclass(frozen=True, eq=False, repr=False)
class _EmpiricalCell:
    vectors: NDArray[np.float64]
    multiplicity: NDArray[np.int64]
    proportions: NDArray[np.float64]

    @property
    def means(self) -> tuple[Fraction, ...]:
        total = int(self.multiplicity.sum())
        return tuple(
            sum(
                (
                    Fraction(float(x)) * int(f)
                    for x, f in zip(column, self.multiplicity, strict=True)
                ),
                Fraction(),
            )
            / total
            for column in self.vectors.T
        )


@dataclass(frozen=True)
class OverlapScenario:
    """Validated empirical outcome laws for a fixed prospective design."""

    plan: OverlapPlan
    n_target: int
    scenario_id: str
    _cells: tuple[_EmpiricalCell, ...] = field(repr=False)

    @property
    def expected_counts(self) -> tuple[float, float, float, float]:
        """Expected target occupants, not fixed sample sizes."""
        p0, p1, p2, p3 = self.plan.cell_probabilities
        n = self.n_target
        return float(n * p0), float(n * p1), float(n * p2), float(n * p3)

    def draws(self, *, replications: int, seed: int) -> Iterator[OverlapDraw]:
        """Iterate complete prospective statistics independently of the estimator."""
        _bounded_integer(replications, "replications", minimum=1)
        _bounded_integer(seed, "seed")
        rng = np.random.Generator(np.random.PCG64(seed))
        steps = _assignment_steps(self.plan)
        return (self._draw(rng, steps) for _ in range(replications))

    def _draw(self, rng: np.random.Generator, steps: _AssignmentSteps) -> OverlapDraw:
        counts = _joint_counts(rng, self.n_target, steps)
        selected = tuple(
            _draw_vectors(rng, n, data) if n else None
            for n, data in zip(counts, self._cells, strict=True)
        )
        cells = tuple(
            _draw_cell(metric, cell, n, data, column)
            for column, metric in enumerate(self.plan.metrics)
            for cell, n, data in zip(CELL_IDS, counts, selected, strict=True)
        )
        return OverlapDraw(counts, cells)


def _bounded_integer(value: int, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_EXACT_INTEGER:
        refuse(PLAN_INVALID, field=name, reason="invalid_value")


def _empirical_cell(
    frame: IntoDataFrame, metrics: tuple[OverlapMetric, ...], cell: CellId
) -> _EmpiricalCell:
    names = tuple(metric.name for metric in metrics)
    df = _native_frame(frame, cell)
    schema = df.schema
    if len(schema) != len(names) or schema.keys() != set(names):
        refuse(PLAN_INVALID, field="cell_outcomes", reason="metric_columns", cell=cell)
    if not 0 < len(df) <= MAX_EXACT_INTEGER:
        refuse(PLAN_INVALID, field="cell_outcomes", reason="invalid_row_count", cell=cell)
    for metric in metrics:
        dtype = schema[metric.name]
        if not dtype.is_numeric() and not (metric.kind == "binary_mean" and dtype == nw.Boolean):
            refuse(PLAN_INVALID, field="cell_outcomes", reason="nonnumeric_metric", cell=cell)
        # Non-null Booleans are already binary; numeric lanes must be checked
        # before casting so a rounded Decimal cannot masquerade as 0 or 1.
        if metric.kind == "binary_mean" and dtype != nw.Boolean:
            values = df.get_column(metric.name)
            if not ((values == 0) | (values == 1)).fill_null(False).all():
                refuse(PLAN_INVALID, field="cell_outcomes", reason="nonbinary_metric", cell=cell)
    try:
        data = df.select(nw.col(name).cast(nw.Float64) for name in names).to_numpy()
    except (TypeError, ValueError, OverflowError, NarwhalsError):
        pass
    else:
        return _validate_vectors(data, cell)
    refuse(PLAN_INVALID, field="cell_outcomes", reason="invalid_dataframe", cell=cell)


def _native_frame(frame: IntoDataFrame, cell: CellId) -> nw.DataFrame[IntoDataFrame]:
    try:
        return nw.from_native(frame, eager_only=True)
    except (TypeError, ValueError, NarwhalsError):
        pass
    refuse(PLAN_INVALID, field="cell_outcomes", reason="invalid_dataframe", cell=cell)


def _validate_vectors(data: NDArray[np.float64], cell: CellId) -> _EmpiricalCell:
    if not np.isfinite(data).all():
        refuse(PLAN_INVALID, field="cell_outcomes", reason="nonfinite_metric", cell=cell)
    # Lexicographic unique rows retain their original multiplicities.
    vectors, frequencies = np.unique(data, axis=0, return_counts=True)
    vectors[vectors == 0] = 0  # Both spellings of zero are one outcome.
    proportions = frequencies / len(data)
    for array in (vectors, frequencies, proportions):
        array.setflags(write=False)
    return _EmpiricalCell(vectors, frequencies, proportions)


def prepare_overlap_scenario(
    plan: OverlapPlan, *, n_target: int, cell_outcomes: Mapping[CellId, IntoDataFrame]
) -> OverlapScenario:
    """Validate one prespecified joint outcome distribution per assignment cell."""
    _bounded_integer(n_target, "n_target")
    if not isinstance(cell_outcomes, Mapping) or set(cell_outcomes) != set(CELL_IDS):
        refuse(PLAN_INVALID, field="cell_outcomes", reason="expected_four_cells")
    cells = tuple(_empirical_cell(cell_outcomes[cell], plan.metrics, cell) for cell in CELL_IDS)
    identity = canonical_json_bytes(
        {
            "plan_sha256": plan.plan_sha256,
            "n_target": n_target,
            "cell_distributions": [
                [cell.vectors.tolist(), cell.multiplicity.tolist()] for cell in cells
            ],
        }
    )
    scenario_id = sha256(b"increment.overlap.v1\0scenario\0" + identity).hexdigest()
    return OverlapScenario(plan, n_target, scenario_id, cells)


type _AssignmentSteps = tuple[tuple[tuple[int, float], ...], int]


def _assignment_steps(plan: OverlapPlan) -> _AssignmentSteps:
    probabilities = plan.cell_probabilities
    order = sorted(range(4), key=lambda j: (probabilities[j], j))
    remaining = Fraction(1)
    steps = []
    for j in order[:-1]:
        # Draw small cells first and compute each conditional ratio before float rounding.
        steps.append((j, float(probabilities[j] / remaining)))
        remaining -= probabilities[j]
    return tuple(steps), order[-1]


def _joint_counts(rng: np.random.Generator, n: int, steps: _AssignmentSteps) -> CellCounts:
    draws, last = steps
    counts = [0, 0, 0, 0]
    remaining = n
    for j, probability in draws:
        counts[j] = int(rng.binomial(remaining, probability))
        remaining -= counts[j]
    counts[last] = remaining
    return counts[0], counts[1], counts[2], counts[3]


type _PickedVectors = tuple[NDArray[np.float64], NDArray[np.int64]]


def _draw_vectors(rng: np.random.Generator, n: int, data: _EmpiricalCell) -> _PickedVectors:
    if len(data.vectors) == 1:
        return data.vectors, np.array([n], dtype=np.int64)
    frequency = rng.multinomial(n, data.proportions)
    occupied = frequency != 0
    return data.vectors[occupied], frequency[occupied]


def _continuous_moments(
    column: NDArray[np.float64], frequency: NDArray[np.int64], n: int, metric: str
) -> tuple[float, float, float]:
    anchor = float(column.min())
    scale = float(column.max()) - anchor
    if scale == 0:
        return anchor, 0.0, 0.0
    if not isfinite(scale):
        refuse(NUMERIC_RANGE, metric=metric, quantity="m2")
    with np.errstate(under="ignore"):
        normalized = (column - anchor) / scale
        center = float(normalized @ frequency) / n
        residual = normalized - center
        ss_normalized = float((residual * residual) @ frequency)
    mantissa, exponent = frexp(scale)
    try:
        m2 = ldexp(ss_normalized * mantissa * mantissa, 2 * exponent)
    except OverflowError:
        m2 = float("inf")
    delta = center * scale
    if not isfinite(m2) or m2 == 0 or delta == 0:
        refuse(NUMERIC_RANGE, metric=metric, quantity="mean_delta" if delta == 0 else "m2")
    return anchor, delta, m2


def _draw_cell(
    metric: OverlapMetric, cell: CellId, n: int, selected: _PickedVectors | None, column: int
) -> OverlapCell:
    binary = metric.kind == "binary_mean"
    successes = 0 if binary else None
    anchor = delta = m2 = None
    if selected is not None:
        vectors, frequency = selected
        values = vectors[:, column]
        if binary:
            successes = int(frequency[values == 1].sum())
            anchor, delta, m2 = overlap_binary_moments(n, successes)
        else:
            anchor, delta, m2 = _continuous_moments(values, frequency, n, metric.name)
    return OverlapCell(
        metric=metric.name,
        cell=cell,
        n_assigned=n,
        n_observed=n,
        n_missing=0,
        n_pending=0,
        n_success=successes,
        anchor=anchor,
        mean_delta=delta,
        m2=m2,
    )


def _overlap_draws(
    plan: OverlapPlan,
    *,
    n_target: int,
    cell_outcomes: Mapping[CellId, IntoDataFrame],
    replications: int,
    seed: int,
) -> Iterator[OverlapDraw]:
    return prepare_overlap_scenario(plan, n_target=n_target, cell_outcomes=cell_outcomes).draws(
        replications=replications, seed=seed
    )


def _contrast_truths(
    plan: OverlapPlan, means: tuple[Fraction, ...]
) -> Mapping[ContrastId, Fraction]:
    off_off, a_only, b_only, both = means
    a0, a1 = plan.a.arm_probabilities
    b0, b1 = plan.b.arm_probabilities
    a_off, a_on = a_only - off_off, both - b_only
    b_off, b_on = b_only - off_off, both - a_only
    return {
        "a_marginal": b0 * a_off + b1 * a_on,
        "b_marginal": a0 * b_off + a1 * b_on,
        "a_given_b_off": a_off,
        "a_given_b_on": a_on,
        "b_given_a_off": b_off,
        "b_given_a_on": b_on,
        "joint_launch": both - off_off,
        "interaction": a_on - a_off,
    }


@dataclass
class _ClaimTally:
    key: ClaimKey
    truth: float
    threshold: float
    alternative_true: bool
    favorable_decision: Literal["clear", "reject"]
    n_point: int = 0
    n_fit: int = 0
    n_decision_ready: int = 0
    n_reject: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def observe(self, row: PlanningRow) -> None:
        self.n_point += row.estimate is not None
        self.n_fit += row.se is not None
        self.n_decision_ready += row.decision in (
            "clear",
            "reject",
            "loss_beyond_tolerance",
            "inconclusive",
        )
        self.n_reject += row.decision == self.favorable_decision
        self.reasons.update({reason for _, reason in row.reasons})

    def summarize(self, replications: int) -> OverlapPowerClaim:
        metric, contrast = self.key
        rate = self.n_reject / replications
        return OverlapPowerClaim(
            metric=metric,
            contrast=contrast,
            truth=self.truth,
            threshold=self.threshold,
            n_point=self.n_point,
            n_fit=self.n_fit,
            n_decision_ready=self.n_decision_ready,
            n_reject=self.n_reject,
            rejection_rate=rate,
            power=rate if self.alternative_true else None,
            rate_kind="power" if self.alternative_true else "type_i_error",
            mc_interval=_mc_interval(self.n_reject, replications),
            reason_counts=tuple(sorted(self.reasons.items())),
        )


def _mc_interval(k: int, n: int) -> tuple[float, float]:
    lower = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    upper = float(beta.isf(0.025, k + 1, n - k)) if k < n else 1.0
    return lower, upper


def _claim_tallies(scenario: OverlapScenario) -> dict[ClaimKey, _ClaimTally]:
    plan = scenario.plan
    means = zip(*(cell.means for cell in scenario._cells), strict=True)
    truths = {m.name: _contrast_truths(plan, mu) for m, mu in zip(plan.metrics, means, strict=True)}
    metrics = {metric.name: metric for metric in plan.metrics}
    tallies = {}
    for claim in plan.claims:
        metric = metrics[claim.metric]
        key = (claim.metric, claim.contrast)
        truth = truths[claim.metric][claim.contrast]
        sign = 1 if metric.better == "higher" else -1
        threshold = -sign * metric.margin_abs if claim.alternative == "noninferiority" else 0.0
        two_sided = claim.alternative == "difference"
        try:
            reported_truth = float(truth)
        except OverflowError:
            refuse(NUMERIC_RANGE, metric=claim.metric, quantity="truth")
        if truth and reported_truth == 0:
            refuse(NUMERIC_RANGE, metric=claim.metric, quantity="truth")
        tallies[key] = _ClaimTally(
            key,
            reported_truth,
            threshold,
            truth != 0 if two_sided else sign * (truth - Fraction(threshold)) > 0,
            "reject" if two_sided else "clear",
        )
    return tallies


def evaluate_overlap_scenario(
    scenario: OverlapScenario, *, replications: int, seed: int, evaluate: EvaluateDraw
) -> OverlapPowerResult:
    """Evaluate every draw with the supplied fixed-family decision procedure."""
    draws = scenario.draws(replications=replications, seed=seed)
    tallies = _claim_tallies(scenario)
    joint_count = 0
    for draw in draws:
        rows, joint = evaluate(draw)
        joint_count += joint == "criteria_met"
        pending = set(tallies)
        for row in rows:
            key = (row.metric, row.contrast)
            if (tally := tallies.get(key)) is not None:
                if key not in pending:
                    refuse(PLAN_INVALID, field="claims", reason="duplicate_decision_evaluation")
                pending.remove(key)
                tally.observe(row)
        if pending:
            refuse(PLAN_INVALID, field="claims", reason="incomplete_decision_evaluation")
    return OverlapPowerResult(
        scenario_id=scenario.scenario_id,
        plan_sha256=scenario.plan.plan_sha256,
        family_sha256=scenario.plan.family_sha256,
        population_id=scenario.plan.population_id,
        n_target=scenario.n_target,
        expected_counts=scenario.expected_counts,
        seed=seed,
        numpy_version=np.__version__,
        replications=replications,
        claims=tuple(tally.summarize(replications) for tally in tallies.values()),
        joint_criteria_met_count=joint_count,
        joint_criteria_met_rate=joint_count / replications,
    )
