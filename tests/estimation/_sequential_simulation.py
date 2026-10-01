"""Sufficient-state simulation of the sequential certification campaign.

The deployed sequential statistic is a function of each retained cell's
sufficient state and nothing else. Reading ``_evaluate_sequential_diagnostic``
top to bottom, a look's evidence and geometry come from

    (model, cell, control SequentialArmState, treatment SequentialArmState, alpha)

and ``alpha`` is ``inference.allocated_alpha(cell.alpha, revealed_units)`` where
``revealed_units`` is the deterministic joint prefix length. The prefix digest,
the ancestry and the per-unit record proofs enter no numerical claim; they are
what ``capture_sequential_snapshot`` exists to verify, not what the statistic
reads.

Four identities follow, each pinned by ``test_sequential_simulation.py``:

1.  **State sufficiency.** Folding :func:`advance` over the revealed
    observations reproduces ``snapshot.states`` bit for bit, so a replication
    may carry the state and drop the prefix. This removes the per-unit digest
    and the superlinear record replay inside ``SequentialSnapshot``.

2.  **Decision locality.** ``SequentialInferenceResult.rejects`` is
    ``log_e >= (-log_interval(decision_alpha)).hi`` -- a function of the
    likelihood certificate alone. No rejection, no stopping predicate and no
    selection reads a confidence interval, so the interval inversion runs only
    where an acceptance gate reads geometry: the stopped look's availability
    counters and the selected-cell FCR reinversion.
    Intermediate-look inversion is not approximated away; it is never required.

3.  **Aggregate sampling (Bernoulli only).** The fixed-arm sampler gives every
    (arm, segment) block a deterministic unit count at every look, and each
    unit's value is ``1{U < p}`` for a fixed ``p``. A block's success increment
    is therefore exactly ``Binomial(m, p)`` when metric noise is independent,
    and exactly a ``Multinomial`` over the partition induced by the sorted
    metric rates when one uniform is shared across metrics. Drawing the
    increment rather than the units is an exact change of representation.

4.  **State recurrence (Bernoulli only).** Deterministic arm counts put a
    Bernoulli cell's state at look ``l`` on the finite grid
    ``{0..n_c} x {0..n_t}``. Evaluating the decision once per *distinct* state
    a batch of replications visits is exact for the same reason as (2): the
    decision is a function of the state.

Scope. Identities 1 and 2 hold for every law here; identities 3 and 4 hold for
the Bernoulli law alone, because the Gaussian (NIG) and Gaussian-ratio (NIW)
sufficient states are continuous and admit neither a finite grid nor an
aggregate increment law -- those two laws keep the deployed per-unit sampler and
gain only 1 and 2. ``scalar_mean`` is excluded outright: its sampler draws a
per-unit rare segment, so even its arm counts are random, and its selection and
disconnected asymptotic geometry do not share the likelihood route. Excluded
cases must stay on ``capture_sequential_snapshot`` -> ``estimate_sequential`` ->
``select_sequential_family``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from calibration.stopping import Decision, StoppingRule
    from increment.estimation._sequential_likelihood import Alternative, GaussianState
    from increment.sequential_state import SequentialArmState

F = Fraction

# Mirrors ``SequentialArmState.law``. Spelled out here because this module must
# import nothing numerical at module scope, and pinned to the deployed set by
# ``test_certified_laws_are_exactly_the_tested_ones``.
SequentialLaw = Literal["bernoulli", "gaussian", "gaussian_ratio", "scalar_mean"]

# One revealed metric value: a Bernoulli event, a scalar, or a joint
# numerator/denominator pair.
Observation = int | float | Fraction | tuple[float | Fraction | int, ...]

# Laws whose campaign record this module reproduces exactly.
SUFFICIENT_STATE_LAWS = frozenset({"bernoulli", "gaussian", "gaussian_ratio"})

# Laws whose per-look state increment has a closed aggregate law, so a batch of
# replications advances without materialising units.
AGGREGATE_SAMPLING_LAWS = frozenset({"bernoulli"})

# Statistics that are one Bernoulli event per replication whatever the roster
# size; the other gated statistics are proportions over the retained roster and
# are binary only for a one-cell family.
_ALWAYS_BINARY_STATISTICS = frozenset({"ever_null_rejection", "nonnull_discovery"})

# Replications advanced together by the vectorised route.
_REPLICATION_CHUNK = 2048

# Ceiling on retained per-state decisions. The reachable Bernoulli grid can hold
# more than a million states on a wide case, and every retained value is a pure
# function of its key, so dropping the lot costs recomputation and nothing else.
_MEMO_ENTRIES = 1 << 18


def certifies_exactly(case) -> bool:
    """Whether the sufficient-state route reproduces this case exactly.

    Deliberately free of numerical imports so a controller can resolve routes
    before paying for numpy or scipy.
    """
    return case.law in SUFFICIENT_STATE_LAWS


def curtailable_gates(case) -> tuple[str, ...]:
    """Names of this case's gates whose statistic is a Bernoulli sequence.

    A stopping rule from :mod:`calibration.stopping` consumes miss flags, so it
    may only curtail a statistic that is ``0`` or ``1`` per replication. The
    proportion statistics carry the retained roster in their denominator and so
    qualify exactly when that roster holds one cell.
    """
    from tests.estimation._sequential_acceptance import _coverage_binding, case_acceptance_gates

    return tuple(
        gate.name
        for gate in case_acceptance_gates(case, _coverage_binding(case))
        if gate.direction != "record"
        and (gate.statistic in _ALWAYS_BINARY_STATISTICS or case.family == 1)
    )


# --- the pure state transition ----------------------------------------------


def empty_state(
    metric: str, group_id: str, segment: Sequence[tuple[str, str]], law: SequentialLaw
) -> SequentialArmState:
    """The canonical zero state :func:`advance` folds onto, as capture builds it."""
    from increment.sequential_state import SequentialArmState

    if law == "bernoulli":
        return SequentialArmState(
            metric=metric,
            group_id=group_id,
            segment=tuple(segment),
            law=law,
            n=0,
            successes=0,
        )
    dimension = 2 if law == "gaussian_ratio" else 1
    zero = (F(0),) * dimension
    return SequentialArmState(
        metric=metric,
        group_id=group_id,
        segment=tuple(segment),
        law=law,
        n=0,
        mean=zero,
        scatter=(zero,) * dimension,
    )


def _exact(value: object) -> Fraction:
    """The deployed lift of one revealed observation to an exact rational."""
    from increment.sequential_state import sequential_refuse

    if isinstance(value, bool):
        return F(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            sequential_refuse("source.invalid", "finalized observations must be finite")
        return F(*value.as_integer_ratio())
    if isinstance(value, (int, F)):
        return F(value)
    sequential_refuse(
        "source.invalid", "observations must be exact integers, rationals or binary64"
    )


def advance(state: SequentialArmState, observation: Observation) -> SequentialArmState:
    """Fold one revealed observation into an arm's sufficient state.

    Pure: no raw record, no digest, no identity. This is exactly the transition
    ``_capture_sequential_diagnostic_snapshot`` applies to the arm a unit is
    assigned to, including the binary64-to-dyadic lift, so folding it over a
    reveal order reproduces ``snapshot.states`` exactly.
    """
    from increment.sequential_state import SequentialArmState, sequential_refuse

    dimension = 2 if state.law == "gaussian_ratio" else 1
    raw = observation if isinstance(observation, (tuple, list)) else (observation,)
    if len(raw) != dimension:
        sequential_refuse("source.invalid", "joint numerator/denominator observation required")
    vector = tuple(_exact(value) for value in raw)
    if state.law == "bernoulli":
        if vector[0] not in (0, 1):
            sequential_refuse("source.invalid", "Bernoulli observation is not binary")
        assert state.successes is not None
        return SequentialArmState(
            metric=state.metric,
            group_id=state.group_id,
            segment=state.segment,
            law=state.law,
            n=state.n + 1,
            successes=state.successes + int(vector[0]),
        )
    singleton = SequentialArmState(
        metric=state.metric,
        group_id=state.group_id,
        segment=state.segment,
        law=state.law,
        n=1,
        mean=vector,
        scatter=((F(0),) * dimension,) * dimension,
    )
    merged = cast("GaussianState", state.kernel()).merge(cast("GaussianState", singleton.kernel()))
    return SequentialArmState(
        metric=state.metric,
        group_id=state.group_id,
        segment=state.segment,
        law=state.law,
        n=merged.n,
        mean=merged.mean,
        scatter=merged.scatter,
    )


def retains(segment: Sequence[tuple[str, str]], group_id: str, record: Mapping) -> bool:
    """Whether a record is assigned to a (group, segment) state, as capture decides."""
    if record["group_id"] != group_id:
        return False
    segments = record.get("segments", {})
    return all(segments.get(key) == value for key, value in segment)


def fold_batches(case, batches: Sequence[Sequence[Mapping]]):
    """Per-look ``SequentialArmState`` tuples, folded with :func:`advance` alone.

    The specification of state accumulation: slow, obviously correct, and the
    reference the fast accumulators inside this module are pinned against.
    """
    resolved = geometry(case)
    states = [empty_state(key.metric, key.group_id, key.segment, case.law) for key in resolved.keys]
    history = []
    for rows in batches:
        for record in rows:
            for index, key in enumerate(resolved.keys):
                if retains(key.segment, key.group_id, record):
                    states[index] = advance(states[index], record["values"][key.metric])
        history.append(tuple(states))
    return tuple(history)


# --- resolved case geometry -------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CellPlan:
    """Everything a retained cell's decision needs that is not random."""

    index: int
    metric: str
    group_id: str
    control_group: str
    segment: tuple[tuple[str, str], ...]
    alternative: Alternative
    cell_alpha: Fraction
    declared_null: bool
    truth: Fraction
    control_key: int
    treatment_key: int
    in_family: bool
    model: Any
    cell: Any


@dataclass(frozen=True, slots=True)
class _StateKey:
    """One (metric, arm, segment) sufficient state tracked across looks.

    ``counts`` is the state's cumulative unit count at each look, which the
    fixed-arm sampler fixes in advance. ``rate`` is the Bernoulli success
    probability the sampler uses for this (metric, arm); the Gaussian laws draw
    per unit and carry none.
    """

    index: int
    metric: str
    group_id: str
    segment: tuple[tuple[str, str], ...]
    counts: tuple[int, ...]
    rate: float | None = None


@dataclass(frozen=True, slots=True)
class _Geometry:
    """Every deterministic quantity of one manifest case."""

    case: Any
    registration: Any
    policy: Any
    truths: Mapping[Any, Any]
    arms: tuple[str, ...]
    schedule: tuple[int, ...]
    keys: tuple[_StateKey, ...]
    cells: tuple[_CellPlan, ...]
    revealed: tuple[int, ...]
    alphas: tuple[tuple[Fraction, ...], ...]
    reject_thresholds: tuple[tuple[Fraction, ...], ...]
    family_order: tuple[int, ...]
    ebh_thresholds: tuple[Fraction, ...]
    blocks: tuple[tuple[str, int, str, tuple[int, ...]], ...]


_GEOMETRY_CACHE: dict[Any, _Geometry] = {}


def _segment_label(segment) -> str | None:
    if not segment:
        return None
    labels = dict(segment)
    if set(labels) != {"segment"}:
        raise ValueError(f"unsupported retained segment {segment!r}")
    return labels["segment"]


def _parity_share(count: int, label: str | None) -> int:
    """Units of one look's arm batch carried by a segment label.

    ``campaign_batch`` labels the ``j``-th unit of an arm ``A`` when ``j`` is
    even and ``B`` otherwise, restarting ``j`` per arm per look.
    """
    if label is None:
        return count
    if label == "A":
        return (count + 1) // 2
    if label == "B":
        return count // 2
    raise ValueError(f"unsupported segment label {label!r}")


def geometry(case) -> _Geometry:
    """Resolve the deterministic geometry of *case*, once per case."""
    cached = _GEOMETRY_CACHE.get(case)
    if cached is not None:
        return cached
    from increment.estimation._certified import log_interval
    from tests.estimation._sequential_acceptance import campaign_declaration

    if not certifies_exactly(case):
        raise ValueError(
            f"law {case.law!r} is outside the sufficient-state scope "
            f"{sorted(SUFFICIENT_STATE_LAWS)}; drive it on the raw public path"
        )
    registration, policy, truths, arms, schedule = campaign_declaration(case)
    models = {model.metric: model for model in registration.models}
    targets = {(cell.metric, cell.group_id): F(truths[cell][0]) for cell in registration.roster}
    control_group = registration.control_group
    drawn = (control_group, *arms)

    raw_keys = sorted(
        {
            (cell.metric, arm, cell.segment)
            for cell in registration.roster
            for arm in (control_group, cell.group_id)
        }
    )
    keys = []
    for index, (metric, arm, segment) in enumerate(raw_keys):
        running, counts = 0, []
        for size in schedule:
            if arm in drawn:
                allocation = case.allocation[0 if arm == control_group else 1]
                running += _parity_share(size * allocation, _segment_label(segment))
            counts.append(running)
        rate = None
        if case.law == "bernoulli":
            # The sampler compares one uniform against rate * declared ratio,
            # in binary64, so the aggregate draw must use that same float.
            rate = case.rate * float(targets.get((metric, arm), F(1)))
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"sampler rate {rate} for {(metric, arm)!r} leaves [0, 1]")
        keys.append(
            _StateKey(
                index=index,
                metric=metric,
                group_id=arm,
                segment=segment,
                counts=tuple(counts),
                rate=rate,
            )
        )
    lookup = {(k.metric, k.group_id, k.segment): k.index for k in keys}

    multiplier = case.allocation[0] + len(arms) * case.allocation[1]
    revealed = tuple(multiplier * sum(schedule[: i + 1]) for i in range(len(schedule)))
    cells = []
    for index, cell in enumerate(registration.roster):
        truth, declared_null = truths[cell]
        cells.append(
            _CellPlan(
                index=index,
                metric=cell.metric,
                group_id=cell.group_id,
                control_group=control_group,
                segment=cell.segment,
                alternative=cell.alternative,
                cell_alpha=cell.alpha,
                declared_null=bool(declared_null),
                truth=F(truth),
                control_key=lookup[cell.metric, control_group, cell.segment],
                treatment_key=lookup[cell.metric, cell.group_id, cell.segment],
                in_family=bool(cell.family),
                model=models[cell.metric],
                cell=cell,
            )
        )
    alphas = tuple(
        tuple(policy.allocated_alpha(plan.cell_alpha, count) for count in revealed)
        for plan in cells
    )
    reject_thresholds = tuple(tuple((-log_interval(alpha)).hi for alpha in row) for row in alphas)
    family_order = tuple(plan.index for plan in cells if plan.in_family)
    width = len(family_order)
    ebh_thresholds = ebh_step_thresholds(registration.q, width)

    # One draw block per (arm, parity) of a look's batch: every state on that
    # arm reads the same units, so the parity split must be drawn jointly.
    blocks = []
    for arm in drawn:
        allocation = case.allocation[0 if arm == control_group else 1]
        for parity in ("A", "B"):
            members = tuple(
                k.index
                for k in keys
                if k.group_id == arm and _segment_label(k.segment) in (None, parity)
            )
            if members:
                blocks.append((arm, allocation, parity, members))

    resolved = _Geometry(
        case=case,
        registration=registration,
        policy=policy,
        truths=truths,
        arms=tuple(arms),
        schedule=tuple(schedule),
        keys=tuple(keys),
        cells=tuple(cells),
        revealed=revealed,
        alphas=alphas,
        reject_thresholds=reject_thresholds,
        family_order=family_order,
        ebh_thresholds=ebh_thresholds,
        blocks=tuple(blocks),
    )
    _GEOMETRY_CACHE[case] = resolved
    return resolved


# --- decision kernels on a sufficient state ---------------------------------


@dataclass(frozen=True, slots=True)
class _Evidence:
    """A cell's likelihood verdict at one state; no interval is inverted."""

    log_e: Fraction | float
    rejects: bool
    positive: bool
    above: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _CellGeometry:
    """A cell's gate-visible geometry at one stopped state."""

    point_available: bool
    point_reason: str | None
    certified: bool
    unknown: bool
    missed: bool


def _point_ratio(law: str, control, treatment) -> tuple[float | None, str | None]:
    """``increment.estimation.sequential_result._point`` on raw kernels."""
    if not control.n or not treatment.n:
        return None, "missing arm in retained cell"
    if law == "bernoulli":
        mc, mt = F(control.successes, control.n), F(treatment.successes, treatment.n)
    elif law in ("gaussian", "scalar_mean"):
        mc, mt = control.mean[0], treatment.mean[0]
    else:
        if not control.mean[1] or not treatment.mean[1]:
            return None, "observed denominator mean is zero"
        mc, mt = control.mean[0] / control.mean[1], treatment.mean[0] / treatment.mean[1]
    if mc == 0:
        return None, "observed control mean is zero"
    try:
        value = float(mt / mc - 1)
    except OverflowError:
        return None, "relative point exceeds binary64 range"
    return (
        (value, None) if math.isfinite(value) else (None, "relative point exceeds binary64 range")
    )


def _certificate(plan: _CellPlan, control, treatment):
    """``checkpoint_certificate`` at one retained state of one cell."""
    from increment.estimation.sequential_result import checkpoint_certificate
    from increment.sequential_state import SequentialArmState, SequentialCheckpoint

    law = plan.model.law

    def arm(kernel, group_id: str) -> SequentialArmState:
        """Rebuild the portable arm state the simulated kernel folds from."""
        identity = {
            "metric": plan.cell.metric,
            "group_id": group_id,
            "segment": plan.cell.segment,
            "law": law,
            "n": kernel.n,
        }
        if law == "bernoulli":
            return SequentialArmState(**identity, successes=kernel.successes)
        return SequentialArmState(**identity, mean=kernel.mean, scatter=kernel.scatter)

    # The identity fields name the stop's provenance, which no computation reads:
    # the certificate is a function of the retained cell and its arm states alone.
    # This route drops the prefix, so the simulated stop names none.
    return checkpoint_certificate(
        SequentialCheckpoint(
            registration_id="0" * 64,
            prefix_id="0" * 64,
            filtration_id="simulated-joint-reveal",
            cell=plan.cell,
            model=plan.model,
            control=arm(control, plan.control_group),
            treatment=arm(treatment, plan.group_id),
            revealed_units=control.n + treatment.n,
        )
    )


def _log_e(certificate) -> Fraction | float:
    """``SequentialInferenceResult.log_e`` from the certificate alone."""
    if certificate.status == "zero":
        return float("-inf")
    if certificate.status == "infinite":
        return float("inf")
    assert certificate.log_e is not None
    return certificate.log_e.lo


# A verdict is a pure function of its key, so the scalar and vectorised routes,
# and every replication that reaches the same state, share one evaluation.
@lru_cache(maxsize=_MEMO_ENTRIES)
def evidence_at(
    plan: _CellPlan,
    control,
    treatment,
    *,
    reject_threshold: Fraction,
    ebh_thresholds: Sequence[Fraction] = (),
) -> _Evidence:
    """Evaluate one cell's likelihood verdict at a sufficient state."""
    log_e = _log_e(_certificate(plan, control, treatment))
    return _Evidence(
        log_e=log_e,
        rejects=log_e >= reject_threshold,
        positive=log_e > 0,
        above=tuple(log_e >= threshold for threshold in ebh_thresholds),
    )


def _bounds(plan: _CellPlan, control, treatment, alpha: Fraction):
    """``checkpoint_bounds`` without the checkpoint envelope."""
    from increment.estimation._sequential_inversion import (
        bernoulli_confidence_sequence,
        gaussian_confidence_sequence,
    )
    from increment.estimation.sequential_result import _beta_prior, _gaussian_prior

    model = plan.model
    if model.law == "bernoulli":
        return bernoulli_confidence_sequence(
            control,
            treatment,
            _beta_prior(model.control_prior),
            _beta_prior(model.treatment_prior),
            alpha=alpha,
            alternative=plan.alternative,
        )
    return gaussian_confidence_sequence(
        control,
        treatment,
        _gaussian_prior(model.control_prior),
        _gaussian_prior(model.treatment_prior),
        alpha=alpha,
        alternative=plan.alternative,
    )


@lru_cache(maxsize=_MEMO_ENTRIES)
def geometry_at(plan: _CellPlan, control, treatment, alpha: Fraction) -> _CellGeometry:
    """Invert one cell's confidence sequence and read what the gates read."""
    from tests.estimation._sequential_acceptance import CERTIFICATION_DESIGN

    point, reason = _point_ratio(plan.model.law, control, treatment)
    bounds = _bounds(plan, control, treatment, alpha)
    unknown = bounds.status in CERTIFICATION_DESIGN.unavailable_statuses
    truth = plan.truth
    missed = bool(
        not unknown
        and (
            bounds.empty
            or (bounds.lower is not None and truth < bounds.lower)
            or (bounds.upper is not None and truth > bounds.upper)
        )
    )
    return _CellGeometry(
        point_available=point is not None,
        point_reason=reason,
        certified=not unknown,
        unknown=unknown,
        missed=missed,
    )


def ebh_step_thresholds(q: Fraction, width: int) -> tuple[Fraction, ...]:
    """``e_bh_select``'s step thresholds for a ``width``-cell family at level ``q``."""
    from increment.estimation._certified import log_interval

    return tuple((-log_interval(q * k / width)).hi for k in range(1, width + 1))


def ebh_indices(above: Sequence[Sequence[bool]]) -> tuple[int, ...]:
    """``e_bh_select`` rewritten over precomputed threshold comparisons.

    ``e_bh_select`` returns, for the largest ``k`` whose ``k``-th largest value
    clears ``threshold_k``, every index clearing that threshold. The ``k``-th
    largest clears ``threshold_k`` exactly when at least ``k`` values do, so the
    clearing count decides ``k`` without sorting the evidence.
    """
    width = len(above)
    for k in range(width, 0, -1):
        clearing = tuple(i for i in range(width) if above[i][k - 1])
        if len(clearing) >= k:
            return clearing
    return ()


# --- exact accumulation ------------------------------------------------------


def _bernoulli_kernel(n: int, successes: int):
    """The evidence kernel a portable Bernoulli arm state carries."""
    from increment.sequential_state import SequentialArmState

    return SequentialArmState(
        metric="simulation", group_id="simulation", law="bernoulli", n=n, successes=successes
    ).kernel()


class _BernoulliMoments:
    """Exact Bernoulli event counts."""

    __slots__ = ("n", "successes")

    def __init__(self) -> None:
        self.n = 0
        self.successes = 0

    def extend(self, values) -> None:
        for value in values:
            if value not in (0, 1):
                from increment.sequential_state import sequential_refuse

                sequential_refuse("source.invalid", "Bernoulli observation is not binary")
            self.n += 1
            self.successes += int(value)

    def kernel(self):
        return _bernoulli_kernel(self.n, self.successes)


class _GaussianMoments:
    """Exact running moments of binary64 rows, carried in integer arithmetic.

    Every binary64 is a dyadic ``m * 2**-s``. Holding one common ``s`` turns the
    observations into integers, so the raw sums are exact integer sums and the
    centred scatter ``sum(x_i x_j) - n * mean_i * mean_j`` is formed once per
    look rather than once per unit. Chan's per-unit merge computes the same
    exact rational, which the equivalence test pins bit for bit.
    """

    __slots__ = ("dimension", "n", "shift", "sums", "cross")

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.n = 0
        self.shift = 0
        self.sums = [0] * dimension
        self.cross = [[0] * dimension for _ in range(dimension)]

    def _rescale(self, shift: int) -> None:
        step = shift - self.shift
        if step <= 0:
            return
        self.sums = [value << step for value in self.sums]
        self.cross = [[value << (2 * step) for value in row] for row in self.cross]
        self.shift = shift

    def extend(self, rows) -> None:
        dimension = self.dimension
        scaled_rows = []
        needed = self.shift
        for row in rows:
            row = row if isinstance(row, (tuple, list)) else (row,)
            if len(row) != dimension:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "source.invalid", "joint numerator/denominator observation required"
                )
            parts = []
            for value in row:
                numerator, denominator = float(value).as_integer_ratio()
                exponent = denominator.bit_length() - 1
                parts.append((numerator, exponent))
                if exponent > needed:
                    needed = exponent
            scaled_rows.append(parts)
        self._rescale(needed)
        shift, sums, cross = self.shift, self.sums, self.cross
        for parts in scaled_rows:
            values = [numerator << (shift - exponent) for numerator, exponent in parts]
            for i in range(dimension):
                vi = values[i]
                sums[i] += vi
                row_i = cross[i]
                for j in range(i, dimension):
                    row_i[j] += vi * values[j]
            self.n += 1
        for i in range(dimension):
            for j in range(i):
                cross[i][j] = cross[j][i]

    def kernel(self):
        from increment.estimation._sequential_likelihood import GaussianState

        dimension = self.dimension
        if not self.n:
            return GaussianState.empty(dimension)
        scale = 1 << self.shift
        mean = tuple(F(value, self.n * scale) for value in self.sums)
        square = scale * scale
        scatter = tuple(
            tuple(
                F(self.cross[i][j], square) - self.n * mean[i] * mean[j] for j in range(dimension)
            )
            for i in range(dimension)
        )
        return GaussianState(self.n, mean, scatter)


def _moments(law: str):
    if law == "bernoulli":
        return _BernoulliMoments()
    return _GaussianMoments(2 if law == "gaussian_ratio" else 1)


# --- one replication over sufficient state ----------------------------------


def _replication_record(
    *,
    ever: bool,
    selected: Sequence[int],
    geometries: Sequence[_CellGeometry],
    fcr: Sequence[tuple[bool, bool]],
    declared_null: Sequence[bool],
    retained: int,
    completed_looks: int,
) -> dict[str, Any]:
    """Project one replication onto the fields ``replication_statistics`` reads."""
    from tests.estimation._sequential_acceptance import CERTIFICATION_DESIGN

    declared = CERTIFICATION_DESIGN.declared_point_reasons
    denominator = max(len(selected), 1)
    false = sum(1 for i in selected if declared_null[i])
    nonnull = sum(1 for i in selected if not declared_null[i])
    unknown = sum(1 for is_unknown, _ in fcr if is_unknown)
    missed = sum(1 for is_unknown, miss in fcr if miss and not is_unknown)
    return {
        "nominal_ever_null_rejection": int(ever),
        "nominal_fdp": F(false, denominator),
        "nominal_fcp_lower": F(missed, denominator),
        "nominal_fcp_upper": F(missed + unknown, denominator),
        "nonnull_discovery": int(nonnull > 0),
        "selected_cells": len(selected),
        "retained_cells": retained,
        "available_points": sum(1 for g in geometries if g.point_available),
        "certified_intervals": sum(1 for g in geometries if g.certified),
        "undeclared_point_reasons": sum(
            1 for g in geometries if g.point_reason is not None and g.point_reason not in declared
        ),
        "completed_looks": completed_looks,
    }


def _selection(resolved: _Geometry, evidence: Sequence[_Evidence]):
    """The case's family selection at one look, as its law's route computes it."""
    from tests.estimation._sequential_acceptance import CERTIFICATION_DESIGN, _diagnostic_selection

    case = resolved.case
    if case.law == "bernoulli":
        order = resolved.family_order
        chosen = tuple(order[i] for i in ebh_indices([evidence[index].above for index in order]))
        realized = resolved.registration.q * len(chosen) / len(order) if chosen and order else None
        alpha = min(realized, CERTIFICATION_DESIGN.alpha) if realized is not None else None
        return chosen, alpha
    chosen, alpha = _diagnostic_selection(evidence)
    return tuple(chosen), alpha


def _reinversion_alpha(law: str, fcr_alpha: Fraction, decision_alpha: Fraction) -> Fraction:
    """The alpha the case's route reinverts a selected cell at.

    ``reinvert_selected`` floors the public Bernoulli replay at the stopped
    decision allocation; the private diagnostic route reinverts at ``fcr_alpha``
    itself.
    """
    if law == "bernoulli":
        return min(F(fcr_alpha), decision_alpha)
    return F(fcr_alpha)


def replicate(case, draw: Callable[[int, int, int], Sequence[Mapping]]) -> dict[str, Any]:
    """Run one replication over sufficient state from a per-look batch source.

    *draw* receives ``(look, size, offset)`` and returns that look's records, so
    a caller may supply the deployed ``campaign_batch`` or an enumerated fixture.
    Everything after the draw -- accumulation, evidence, freezing, selection,
    stopping and the stopped-look geometry -- is the sufficient-state route.
    """
    resolved = geometry(case)
    cells = resolved.cells
    accumulators = [_moments(case.law) for _ in resolved.keys]
    frozen: dict[int, tuple[Any, Any, Fraction]] = {}
    ever = False
    selected: tuple[int, ...] = ()
    fcr_alpha: Fraction | None = None
    states: list[tuple[Any, Any, Fraction]] = []
    offset = 0
    look = 0
    for look, size in enumerate(resolved.schedule):
        rows = draw(look, size, offset)
        offset += len(rows)
        for index, key in enumerate(resolved.keys):
            retained_values = [
                record["values"][key.metric]
                for record in rows
                if retains(key.segment, key.group_id, record)
            ]
            if retained_values:
                accumulators[index].extend(retained_values)
        kernels = [accumulator.kernel() for accumulator in accumulators]
        states, evidence = [], []
        for plan in cells:
            if plan.index in frozen:
                control, treatment, alpha = frozen[plan.index]
                threshold = _reject_threshold(alpha)
            else:
                control = kernels[plan.control_key]
                treatment = kernels[plan.treatment_key]
                alpha = resolved.alphas[plan.index][look]
                threshold = resolved.reject_thresholds[plan.index][look]
            states.append((control, treatment, alpha))
            evidence.append(
                evidence_at(
                    plan,
                    control,
                    treatment,
                    reject_threshold=threshold,
                    ebh_thresholds=resolved.ebh_thresholds if plan.in_family else (),
                )
            )
        ever = ever or any(
            row.rejects and plan.declared_null for row, plan in zip(evidence, cells, strict=True)
        )
        selected, fcr_alpha = _selection(resolved, evidence)
        if case.freeze and look + 1 < len(resolved.schedule):
            middle = look == len(resolved.schedule) // 2
            for slot, plan in enumerate(cells):
                if plan.index in frozen:
                    continue
                if evidence[slot].rejects or (middle and plan.index % 2 == 0):
                    frozen[plan.index] = states[slot]
        if _stops(case, look, evidence, selected):
            break
    geometries, fcr = [], []
    for slot, plan in enumerate(cells):
        control, treatment, alpha = states[slot]
        geometries.append(geometry_at(plan, control, treatment, alpha))
        if plan.index in selected:
            assert fcr_alpha is not None
            resolved_geometry = geometry_at(
                plan, control, treatment, _reinversion_alpha(case.law, fcr_alpha, alpha)
            )
            fcr.append((resolved_geometry.unknown, resolved_geometry.missed))
    return _replication_record(
        ever=ever,
        selected=selected,
        geometries=geometries,
        fcr=fcr,
        declared_null=[plan.declared_null for plan in cells],
        retained=len(cells),
        completed_looks=look + 1,
    )


def _reject_threshold(alpha: Fraction) -> Fraction:
    from increment.estimation._certified import log_interval

    return (-log_interval(alpha)).hi


def _stops(case, look: int, evidence, selected) -> bool:
    """``campaign_replication``'s stopping predicate over the same verdicts."""
    if case.stopping == "first_rejection":
        return any(row.rejects for row in evidence)
    if case.stopping == "first_discovery":
        return bool(selected)
    if case.stopping == "adaptive":
        return look + 1 >= 2 and any(row.positive for row in evidence[::2])
    return False


def campaign_source(case, rng):
    """A per-look batch source driving the deployed campaign sampler."""
    from tests.estimation._sequential_acceptance import campaign_batch

    resolved = geometry(case)

    def draw(look: int, size: int, offset: int):
        del look
        return campaign_batch(
            case, rng, resolved.registration, resolved.truths, resolved.arms, size, offset
        )

    return draw


# --- vectorised Bernoulli engine --------------------------------------------


class _BernoulliEngine:
    """Aggregate-increment, distinct-state Bernoulli route.

    Advances a whole batch of replications through the deterministic arm counts
    by drawing each block's success increment, then evaluates each look's
    decision once per distinct sufficient state rather than once per
    replication.

    The retained states are bounded: a campaign of a million replications on a
    wide grid can visit more states than fit in memory, and dropping retained
    values is free because every one of them is a pure function of its key.
    """

    def __init__(self, resolved: _Geometry, *, memo_entries: int = _MEMO_ENTRIES) -> None:
        self.geometry = resolved
        self.case = resolved.case
        self.evidence_calls = 0
        self.evidence_states = 0
        self.memo_entries = memo_entries
        self._evidence: dict[tuple, _Evidence] = {}
        self._geometry: dict[tuple, _CellGeometry] = {}
        self._counts = None

    # -- memoised decisions on a state ------------------------------------

    def _kernels(self, nc: int, sc: int, nt: int, st: int):
        return _bernoulli_kernel(nc, sc), _bernoulli_kernel(nt, st)

    def evidence(self, plan: _CellPlan, nc, sc, nt, st, look) -> _Evidence:
        threshold = self.geometry.reject_thresholds[plan.index][look]
        key = (plan.index, threshold, nc, sc, nt, st)
        found = self._evidence.get(key)
        if found is None:
            control, treatment = self._kernels(nc, sc, nt, st)
            found = evidence_at(
                plan,
                control,
                treatment,
                reject_threshold=threshold,
                ebh_thresholds=self.geometry.ebh_thresholds if plan.in_family else (),
            )
            if len(self._evidence) >= self.memo_entries:
                self._evidence.clear()
            self._evidence[key] = found
            self.evidence_states += 1
        return found

    def cell_geometry(self, plan: _CellPlan, nc, sc, nt, st, alpha) -> _CellGeometry:
        key = (plan.index, alpha, nc, sc, nt, st)
        found = self._geometry.get(key)
        if found is None:
            control, treatment = self._kernels(nc, sc, nt, st)
            found = geometry_at(plan, control, treatment, alpha)
            if len(self._geometry) >= self.memo_entries:
                self._geometry.clear()
            self._geometry[key] = found
        return found

    # -- batch execution ---------------------------------------------------

    def run(self, rng, size: int, increments=None) -> list[dict[str, Any]]:
        """Advance *size* replications; *increments* replaces the aggregate draw."""
        import numpy as np

        resolved = self.geometry
        cells = resolved.cells
        looks = len(resolved.schedule)
        counts = self._count_table()
        successes = np.zeros((len(resolved.keys), size), dtype=np.int64)
        frozen_look = np.full((len(cells), size), -1, dtype=np.int64)
        frozen_sc = np.zeros((len(cells), size), dtype=np.int64)
        frozen_st = np.zeros((len(cells), size), dtype=np.int64)
        ever = np.zeros(size, dtype=bool)
        active = np.arange(size)
        records: list[dict[str, Any] | None] = [None] * size
        shared = self.case.dependence == "shared"

        for look in range(looks):
            if not active.size:
                break
            if increments is None:
                self._draw(rng, look, successes, active, shared)
            else:
                for index in range(len(resolved.keys)):
                    successes[index, active] += np.asarray(increments[look][index])[active]
            state = self._cell_states(
                look, counts, successes, active, frozen_look, frozen_sc, frozen_st
            )
            rejects, positive, above = self._evaluate(look, state, active.size)
            null_mask = np.array([[plan.declared_null] for plan in cells])
            ever[active] |= (rejects & null_mask).any(axis=0)
            chosen = self._select(above, active.size)
            stop = self._stop(look, rejects, positive, chosen)
            self._freeze(look, looks, rejects, active, frozen_look, frozen_sc, frozen_st, state)
            final = stop | (look == looks - 1)
            if final.any():
                self._settle(look, state, chosen, ever[active], final, active, records)
            active = active[~final]
        settled = [record for record in records if record is not None]
        assert len(settled) == size
        return settled

    def _count_table(self):
        import numpy as np

        if self._counts is None:
            self._counts = np.array([key.counts for key in self.geometry.keys], dtype=np.int64)
        return self._counts

    def _draw(self, rng, look, successes, active, shared) -> None:
        import numpy as np

        resolved = self.geometry
        size = resolved.schedule[look]
        width = active.size
        for _arm, allocation, parity, members in resolved.blocks:
            units = _parity_share(size * allocation, parity)
            if not units:
                continue
            rates = {}
            for index in members:
                key = resolved.keys[index]
                rates.setdefault(key.metric, key.rate)
            names = sorted(rates)
            if shared and len(names) > 1:
                # One uniform per unit makes the metric indicators comonotone:
                # the count vector is a multinomial over the partition the
                # sorted rates induce, and each metric's successes are the units
                # falling below its own rate.
                values = np.array([rates[name] for name in names])
                order = np.argsort(values, kind="stable")
                edges = np.diff(np.concatenate(([0.0], values[order], [1.0])))
                draws = rng.multinomial(units, np.maximum(edges, 0.0), size=width)
                cumulative = np.cumsum(draws[:, :-1], axis=1)
                per_metric = {
                    names[slot]: cumulative[:, position] for position, slot in enumerate(order)
                }
            else:
                per_metric = {name: rng.binomial(units, rates[name], size=width) for name in names}
            for index in members:
                key = resolved.keys[index]
                successes[index, active] += per_metric[key.metric]

    def _cell_states(self, look, counts, successes, active, frozen_look, frozen_sc, frozen_st):
        import numpy as np

        rows = []
        for plan in self.geometry.cells:
            frozen = frozen_look[plan.index, active]
            held = frozen >= 0
            source = np.where(held, np.maximum(frozen, 0), look)
            rows.append(
                (
                    counts[plan.control_key][source],
                    np.where(
                        held, frozen_sc[plan.index, active], successes[plan.control_key, active]
                    ),
                    counts[plan.treatment_key][source],
                    np.where(
                        held, frozen_st[plan.index, active], successes[plan.treatment_key, active]
                    ),
                    source,
                )
            )
        return rows

    def _evaluate(self, look, state, width):
        import numpy as np

        resolved = self.geometry
        cells = resolved.cells
        depth = len(resolved.ebh_thresholds)
        rejects = np.empty((len(cells), width), dtype=bool)
        positive = np.empty((len(cells), width), dtype=bool)
        above = np.zeros((len(cells), width, depth), dtype=bool)
        for slot, plan in enumerate(cells):
            nc, sc, nt, st, source = state[slot]
            # A cell's state at a look lives on a bounded integer grid whose
            # arm counts the look already fixes, so (look, s_c, s_t) names it
            # and the distinct states a batch visits are far fewer than its
            # replications; the decision is evaluated once per distinct state.
            span_c = int(resolved.keys[plan.control_key].counts[-1]) + 1
            span_t = int(resolved.keys[plan.treatment_key].counts[-1]) + 1
            code = (source * span_c + sc) * span_t + st
            unique, first, inverse = np.unique(code, return_index=True, return_inverse=True)
            inverse = inverse.reshape(-1)
            self.evidence_calls += inverse.size
            unique_rejects = np.empty(unique.size, dtype=bool)
            unique_positive = np.empty(unique.size, dtype=bool)
            unique_above = np.zeros((unique.size, depth), dtype=bool)
            for column, position in enumerate(first):
                found = self.evidence(
                    plan,
                    int(nc[position]),
                    int(sc[position]),
                    int(nt[position]),
                    int(st[position]),
                    int(source[position]),
                )
                unique_rejects[column] = found.rejects
                unique_positive[column] = found.positive
                if plan.in_family:
                    unique_above[column] = found.above
            rejects[slot] = unique_rejects[inverse]
            positive[slot] = unique_positive[inverse]
            above[slot] = unique_above[inverse]
        return rejects, positive, above

    def _select(self, above, width):
        import numpy as np

        resolved = self.geometry
        order = resolved.family_order
        if not order:
            return None
        stack = above[list(order)]  # (m, width, m)
        depth = len(order)
        clearing = stack.sum(axis=0)
        feasible = clearing >= np.arange(1, depth + 1)
        any_feasible = feasible.any(axis=1)
        level = depth - 1 - feasible[:, ::-1].argmax(axis=1)
        mask = np.zeros((depth, width), dtype=bool)
        rows = np.nonzero(any_feasible)[0]
        if rows.size:
            mask[:, rows] = stack[:, rows, level[rows]]
        return mask

    def _stop(self, look, rejects, positive, chosen):
        import numpy as np

        case = self.case
        if case.stopping == "first_rejection":
            return rejects.any(axis=0)
        if case.stopping == "first_discovery":
            if chosen is None:
                return np.zeros(rejects.shape[1], dtype=bool)
            return chosen.any(axis=0)
        if case.stopping == "adaptive" and look + 1 >= 2:
            return positive[::2].any(axis=0)
        return np.zeros(rejects.shape[1], dtype=bool)

    def _freeze(self, look, looks, rejects, active, frozen_look, frozen_sc, frozen_st, state):
        if not self.case.freeze or look + 1 >= looks:
            return
        middle = look == len(self.geometry.schedule) // 2
        for slot, plan in enumerate(self.geometry.cells):
            eligible = rejects[slot] | bool(middle and plan.index % 2 == 0)
            fresh = (frozen_look[plan.index, active] < 0) & eligible
            if not fresh.any():
                continue
            columns = active[fresh]
            frozen_look[plan.index, columns] = look
            frozen_sc[plan.index, columns] = state[slot][1][fresh]
            frozen_st[plan.index, columns] = state[slot][3][fresh]

    def _settle(self, look, state, chosen, ever, final, active, records):
        import numpy as np

        from tests.estimation._sequential_acceptance import CERTIFICATION_DESIGN

        resolved = self.geometry
        cells = resolved.cells
        declared_null = [plan.declared_null for plan in cells]
        order = resolved.family_order
        width = len(order)
        q = resolved.registration.q
        for position in np.nonzero(final)[0]:
            selected = (
                tuple(order[i] for i in range(width) if chosen[i, position])
                if chosen is not None
                else ()
            )
            fcr_alpha = (
                min(q * len(selected) / width, CERTIFICATION_DESIGN.alpha) if selected else None
            )
            geometries, fcr = [], []
            for slot, plan in enumerate(cells):
                nc, sc, nt, st, source = state[slot]
                alpha = resolved.alphas[plan.index][int(source[position])]
                pin = (int(nc[position]), int(sc[position]), int(nt[position]), int(st[position]))
                geometries.append(self.cell_geometry(plan, *pin, alpha))
                if plan.index in selected:
                    assert fcr_alpha is not None
                    resolved_geometry = self.cell_geometry(
                        plan, *pin, _reinversion_alpha("bernoulli", fcr_alpha, alpha)
                    )
                    fcr.append((resolved_geometry.unknown, resolved_geometry.missed))
            records[active[position]] = _replication_record(
                ever=bool(ever[position]),
                selected=selected,
                geometries=geometries,
                fcr=fcr,
                declared_null=declared_null,
                retained=len(cells),
                completed_looks=look + 1,
            )


def simulate_bernoulli_increments(case, increments) -> list[dict[str, Any]]:
    """Run the vectorised Bernoulli engine on explicit per-key increments.

    ``increments[look][key_index][replication]`` is the success increment the
    aggregate draw would have produced. The equivalence test uses this to drive
    the vectorised engine over the same state path as the raw public route.
    """
    engine = _BernoulliEngine(geometry(case))
    width = len(increments[0][0])
    return engine.run(None, width, increments=increments)


# --- public campaign entry point --------------------------------------------


@dataclass(frozen=True, slots=True)
class GateStream:
    """One gate's Bernoulli sequence of replication outcomes."""

    gate: str
    statistic: str
    draws: int
    hits: int
    misses: int
    decision: str
    rule: str
    rule_parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Exact accumulators for one manifest case over its executed replications."""

    case_name: str
    law: str
    route: str
    requested_replications: int
    replications: int
    retained_cells: int
    available_points: int
    certified_intervals: int
    selected_cells: int
    nonnull_discoveries: int
    nominal_ever_null_rejections: int
    undeclared_point_reasons: int
    nominal_fdp_sum: Fraction
    nominal_fcp_lower_sum: Fraction
    nominal_fcp_upper_sum: Fraction
    completed_looks: int
    gate_statistics: Mapping[str, tuple[Fraction, int]]
    gate_streams: tuple[GateStream, ...]
    # Likelihood decisions the route needed, against the sufficient states it
    # actually pushed through the kernel. Their ratio is the measured state
    # recurrence, and equality means the law admits no reuse.
    evidence_evaluations: int
    distinct_evidence_states: int

    def means(self) -> dict[str, Fraction]:
        """Replication means of every measured statistic, exactly."""
        return {
            name: total / count for name, (total, count) in self.gate_statistics.items() if count
        }


def _fresh_rule(rule: StoppingRule) -> StoppingRule:
    """A rule of the same declared design, ready to consume its own stream."""
    from calibration.stopping import build_rule

    if rule.rule == "fixed":
        return build_rule(rule.design, stopping="fixed")
    return build_rule(rule.design, stopping="sequential", sequential_rule=rule.rule)


def _uncurtailed_floor(case, curtailed: Sequence[str], replications: int) -> int:
    """Draws the gates without a rule still require, capped by *replications*.

    Sized at the declared ``CERTIFICATION_LEDGER.per_decision_error``. A profile
    that certifies a subset of the gates spends a different per-decision error,
    so a caller running under such a profile must pass its own frozen ``floor``
    rather than inherit this one.
    """
    from tests.estimation._sequential_acceptance import (
        CERTIFICATION_LEDGER,
        _coverage_binding,
        case_acceptance_gates,
        gate_replications,
    )

    error = CERTIFICATION_LEDGER.per_decision_error
    floor = 0
    for gate in case_acceptance_gates(case, _coverage_binding(case)):
        if gate.name in curtailed or not gate.margin:
            continue
        floor = max(floor, gate_replications(gate, error))
    return min(floor, replications)


def simulate_case(  # noqa: PLR0915
    case,
    *,
    replications: int,
    seed: int,
    stopping: StoppingRule | Mapping[str, StoppingRule] | None = None,
    observer: Callable[[str, int, Mapping[str, Any] | None], object] | None = None,
    floor: int | None = None,
    chunk_size: int | None = None,
):
    """Advance *replications* of *case* through its sufficient state.

    ``seed`` names the whole run; replications are drawn in deterministic chunks
    of ``chunk_size`` derived from ``(seed, chunk index)``, so a curtailed run
    consumes a prefix of exactly the stream an uncurtailed run consumes, and a
    curtailment can only end the run on a chunk boundary.

    ``stopping`` is a mapping from gate name to a :mod:`calibration.stopping`
    rule, or one rule applied to every curtailable gate; each gate is driven by
    its own instance. The run ends when every supplied rule has resolved and the
    draws have reached ``floor``, which is what the gates left without a rule
    still require -- defaulting to the declared ledger's requirement, and
    overridable by a profile that froze its own.

    *observer*, when given, is called ``("started", index, None)`` before and
    ``("completed", index, record)`` after each replication, where ``record``
    carries the replication's own counters rather than their projection onto the
    gate statistics.
    """
    import numpy as np

    from calibration.stopping import StoppingRule as _Rule
    from tests.estimation._sequential_acceptance import replication_statistics

    if replications < 1:
        raise ValueError(f"replications must be >= 1, got {replications}")
    resolved = geometry(case)
    vectorised = case.law in AGGREGATE_SAMPLING_LAWS
    engine = _BernoulliEngine(resolved) if vectorised else None

    gates = {gate.name: gate for gate in _case_gates(case) if gate.name in curtailable_gates(case)}
    declared: dict[str, StoppingRule] = {}
    if isinstance(stopping, _Rule):
        declared = dict.fromkeys(gates, stopping)
    elif stopping is not None:
        declared = dict(stopping.items())
        unknown = sorted(set(declared) - set(gates))
        if unknown:
            raise ValueError(
                f"gates {unknown} are not curtailable for {case.name!r}; "
                f"curtailable: {sorted(gates)}"
            )
    rules = {name: _fresh_rule(rule) for name, rule in declared.items()}
    required = (
        _uncurtailed_floor(case, tuple(rules), replications)
        if floor is None
        else min(floor, replications)
    )
    decisions: dict[str, Decision] = dict.fromkeys(rules, "continue")
    draws = dict.fromkeys(rules, 0)
    hits = dict.fromkeys(rules, 0)
    misses = dict.fromkeys(rules, 0)

    counters = dict.fromkeys(
        (
            "retained_cells",
            "available_points",
            "certified_intervals",
            "selected_cells",
            "nonnull_discovery",
            "nominal_ever_null_rejection",
            "undeclared_point_reasons",
            "completed_looks",
        ),
        0,
    )
    sums = {"nominal_fdp": F(0), "nominal_fcp_lower": F(0), "nominal_fcp_upper": F(0)}
    statistics: dict[str, list] = {}
    evidence_calls = evidence_states = 0
    executed = 0
    chunk = 0
    width = _REPLICATION_CHUNK if chunk_size is None else chunk_size
    if width < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    while executed < replications:
        size = min(width, replications - executed)
        rng = np.random.default_rng([seed, chunk])
        if observer is not None:
            for index in range(executed, executed + size):
                observer("started", index, None)
        if engine is not None:
            batch = engine.run(rng, size)
            evidence_calls, evidence_states = engine.evidence_calls, engine.evidence_states
        else:
            source = campaign_source(case, rng)
            batch = [replicate(case, source) for _ in range(size)]
            # A continuous state never recurs, so every decision is its own
            # kernel evaluation; saying so keeps the two routes comparable.
            required_decisions = sum(
                record["completed_looks"] * len(resolved.cells) for record in batch
            )
            evidence_calls += required_decisions
            evidence_states += required_decisions
        for record in batch:
            values = replication_statistics(record)
            if observer is not None:
                observer("completed", executed, record)
            executed += 1
            for name in counters:
                counters[name] += record[name]
            for name in sums:
                sums[name] += record[name]
            for name, value in values.items():
                bucket = statistics.setdefault(name, [F(0), 0])
                bucket[0] += value
                bucket[1] += 1
            for name, rule in rules.items():
                if decisions[name] != "continue":
                    continue
                value = values.get(gates[name].statistic)
                if value is None:
                    continue
                if value not in (0, 1):
                    raise ValueError(
                        f"gate {name!r} statistic {gates[name].statistic!r} took the "
                        f"non-binary value {value}; a Bernoulli rule cannot curtail it"
                    )
                miss = value == 1 if gates[name].direction == "upper" else value == 0
                draws[name] += 1
                hits[name] += not miss
                misses[name] += miss
                decisions[name] = rule.observe(miss=miss)
        chunk += 1
        if (
            rules
            and executed >= required
            and all(decision != "continue" for decision in decisions.values())
        ):
            break

    return CaseOutcome(
        case_name=case.name,
        law=case.law,
        route="sufficient_state",
        requested_replications=replications,
        replications=executed,
        retained_cells=counters["retained_cells"],
        available_points=counters["available_points"],
        certified_intervals=counters["certified_intervals"],
        selected_cells=counters["selected_cells"],
        nonnull_discoveries=counters["nonnull_discovery"],
        nominal_ever_null_rejections=counters["nominal_ever_null_rejection"],
        undeclared_point_reasons=counters["undeclared_point_reasons"],
        nominal_fdp_sum=sums["nominal_fdp"],
        nominal_fcp_lower_sum=sums["nominal_fcp_lower"],
        nominal_fcp_upper_sum=sums["nominal_fcp_upper"],
        completed_looks=counters["completed_looks"],
        gate_statistics={name: (total, count) for name, (total, count) in statistics.items()},
        gate_streams=tuple(
            GateStream(
                gate=name,
                statistic=gates[name].statistic,
                draws=draws[name],
                hits=hits[name],
                misses=misses[name],
                decision=decisions[name],
                rule=rule.rule,
                rule_parameters=rule.parameters(),
            )
            for name, rule in rules.items()
        ),
        evidence_evaluations=evidence_calls,
        distinct_evidence_states=evidence_states,
    )


def _case_gates(case):
    from tests.estimation._sequential_acceptance import _coverage_binding, case_acceptance_gates

    return case_acceptance_gates(case, _coverage_binding(case))


__all__ = [
    "AGGREGATE_SAMPLING_LAWS",
    "SUFFICIENT_STATE_LAWS",
    "CaseOutcome",
    "GateStream",
    "Observation",
    "advance",
    "campaign_source",
    "certifies_exactly",
    "curtailable_gates",
    "ebh_indices",
    "ebh_step_thresholds",
    "empty_state",
    "evidence_at",
    "fold_batches",
    "geometry",
    "geometry_at",
    "replicate",
    "retains",
    "simulate_bernoulli_increments",
    "simulate_case",
]
