"""Execute the sequential certification campaign at a declared profile.

Run from the checkout with ``python -m scripts.run_sequential_campaign``.

A profile names the enumerated tolerance the acceptance gates are decided
against, the declared cells it certifies, and the stopping rule. Cells the
profile omits are reported uncertified rather than dropped, and a case is
certified only when it ran its derived replication requirement and every gate
it binds accepted. Per-replication accounting goes to ``calibration.journal``:
a hash-chained block log plus a capped exception log, never one record per
look.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, replace
from fractions import Fraction
from importlib.util import find_spec
from itertools import accumulate
from pathlib import Path

import yaml

from calibration.journal import DrawJournal
from calibration.journal import verify as verify_journal
from calibration.profile import PROFILES_PATH
from calibration.profile import load as load_profile
from calibration.stopping import FixedDesign, StoppingError, build_rule
from scripts.run_test_tier import run_with_budget
from tests.estimation._sequential_acceptance import (
    ACCEPTANCE_GATES,
    CERTIFICATION_DESIGN,
    CERTIFICATION_LEDGER,
    AcceptanceGate,
    _decidable,
    _roster_size,
    acceptance_verdicts,
    campaign_identity,
    campaign_schedule,
    campaign_science,
    case_replications,
    certification_ledger,
    gate_replications,
    gate_threshold,
    principal_manifest,
    replication_statistics,
)

CAMPAIGN = "sequential"
# Entries per checkpoint memo are sized from measured saturation on low-rate
# Bernoulli shapes; 4096 clears the knee within the worker's memory budget.
CHECKPOINT_MEMO_ENTRIES = 4096
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_sequential_campaign.py",
    "scripts/run_test_tier.py",
    "calibration/journal.py",
    "calibration/profile.py",
    "calibration/profiles.yaml",
    "calibration/stopping.py",
    "tests/estimation/_sequential_acceptance.py",
    "tests/estimation/_sequential_simulation.py",
    "tests/estimation/test_sequential_deployed_validity.py",
    "tests/asymptotic_cases.py",
    "tests/sequential_cases.py",
    "increment/__init__.py",
    "increment/errors.py",
    "increment/semantics/sequential.py",
    "increment/sequential_state.py",
    "increment/estimation/sequential.py",
    "increment/estimation/sequential_runtime.py",
    "increment/estimation/sequential_result.py",
    "increment/estimation/_sequential_likelihood.py",
    "increment/estimation/_sequential_inversion.py",
    "increment/estimation/_certified.py",
    "increment/estimation/asymptotic_mean.py",
    "increment/estimation/family.py",
    "increment/estimation/decision_types.py",
    "increment/estimation/results.py",
)

# Journal counters are exact per-replication integers, so sealing them into the
# hash chain makes gated proportions tamper-evident. Rational proportions use
# per-replication ratios and are bounded against verified draw counts instead.
_COUNTERS = (
    ("retained_cell", "retained_cells"),
    ("available_point", "available_points"),
    ("certified_interval", "certified_intervals"),
    ("selected_cell", "selected_cells"),
    ("undeclared_point_reason", "undeclared_point_reasons"),
    ("finite_look_cell", "finite_look_confidence_cells"),
    ("finite_look_miss", "finite_look_confidence_misses"),
    ("finite_look_unknown", "finite_look_confidence_unknown"),
)
# Per-replication 0/1 events, sealed as counters and reconciled against the
# published statistic sums, which must agree exactly because each of those is
# a mean of indicators.
_INDICATORS = {
    "nominal_ever_null_rejection": "ever_null_rejection",
    "nonnull_discovery": "nonnull_discovery",
}
_RATIONALS = ("nominal_fdp", "nominal_fcp_lower", "nominal_fcp_upper")
_REQUIRED_RECORD_FIELDS = tuple(field for _, field in _COUNTERS) + tuple(_INDICATORS) + _RATIONALS
_DECISION_STATUS = {
    "accept": "accepted",
    "reject": "refused",
    "unresolved": "unresolved",
    "continue": "undecided",
}


def _default(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def _json(value):
    return json.dumps(value, default=_default, sort_keys=True, allow_nan=False)


def _write(path, value):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _sources():
    """Digest every source the claim rests on; an absent file is itself recorded."""
    digests = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return digests


# --------------------------------------------------------------------------
# The profile's restatement of the declared design
# --------------------------------------------------------------------------


def rescaled_gate(gate: AcceptanceGate, ratio: Fraction) -> AcceptanceGate:
    """Restate one acceptance gate at a profile's tolerance.

    The declared design fixes each budgeted gate's Monte-Carlo margin as a
    constant fraction of its own tolerance: half of it for the scientific
    excess gates (``scientific_excess`` 1/200 against ``max_mc_margin``
    1/400), a quarter of it for the availability gates (slack 1/100 against
    the same margin). A profile scales BOTH numbers by the same ratio, so
    every gate keeps its declared shape and only its resolution moves.

    Scaling the tolerance alone would be worse than useless. ``gate_threshold``
    puts the accept threshold exactly ``margin`` away from the tolerated worst
    case, so ``gate_replications`` inverts a divergence ``KL(worst -+ margin ||
    worst)`` that depends on the MARGIN, not on the tolerance; the tolerance
    enters only by moving the operating point ``worst`` toward 1/2, where that
    divergence is smaller. Loosening the tolerance on its own therefore RAISES
    the replication count while weakening the claim. Scaling both together
    gives the intended ``n ~ 1/ratio**2``, and at ratio 1 reproduces the
    declared gate identically.
    """
    if ratio == 1 or not gate.margin:
        return gate
    tolerance, margin = gate.tolerance * ratio, gate.margin * ratio
    decided = _decidable(gate.direction, gate.target, tolerance, margin)
    return replace(
        gate,
        direction=gate.direction if decided else "record",
        tolerance=tolerance if decided else Fraction(0),
        margin=margin if decided else Fraction(0),
    )


def _miss_rate(gate: AcceptanceGate) -> Fraction:
    """The gate restated as a bound on a miss rate.

    An upper gate already bounds one; a lower gate on an achievement rate is
    the same statement about its complement, which is what a Bernoulli
    stopping rule consumes.
    """
    return gate.target if gate.direction == "upper" else Fraction(1) - gate.target


def _miss_flag(gate: AcceptanceGate, value: Fraction) -> bool:
    """One replication's miss indicator for a gate decided as a Bernoulli stream.

    Only gates whose replication statistic is a genuine 0/1 indicator are ever
    decided this way. A value in between means the premise is wrong, which is
    a broken design rather than a draw to smooth over.
    """
    if value not in (0, 1):
        raise RuntimeError(
            f"gate {gate.name!r} is decided as a Bernoulli stream but observed "
            f"the intermediate statistic {value}"
        )
    return bool(value == 1) if gate.direction == "upper" else bool(value == 0)


_EXACT_REQUIREMENT: dict[tuple, int | None] = {}


def _bernoulli_statistic(case, gate: AcceptanceGate) -> bool:
    """Whether this gate's replication statistic can only be 0 or 1.

    Every gated statistic is a proportion over the case's retained cells, so it
    is an indicator exactly when that roster holds a single cell.
    ``_miss_flag`` re-checks it on every observation rather than trusting the
    argument.
    """
    return bool(gate.margin) and case.family == 1


def _exact_requirement(gate: AcceptanceGate, eta: Fraction, ceiling: int) -> int | None:
    """Fewest quantised draws whose EXACT false-acceptance tail fits the budget.

    The declared ledger sizes every gate with the distribution-free
    Chernoff-Hoeffding divergence bound, and says why: several gated statistics
    are proportions over a random denominator, so no binomial interval covers
    them. That reason does not apply to a case whose retained roster holds one
    cell, where the denominator is the constant 1, and there the exact bound is
    tighter than the distribution-free one at the SAME decision boundary.

    Nothing else moves. The gate still accepts when the replication mean is
    within ``gate_threshold``, which for a count of misses is ``K <= floor(n *
    threshold)``, and the requirement is the smallest quantised ``n`` whose
    exact probability of that acceptance at the tolerated worst case is inside
    the per-decision budget. Deriving ``n`` by bisecting ``FixedDesign.resolve``
    instead would silently move the boundary to that design's midpoint
    operating point, which is a different and undeclared decision rule.
    """
    from scipy.stats import binom

    quantum = CERTIFICATION_DESIGN.replication_quantum
    threshold = gate_threshold(gate)
    if threshold is None:
        # A recorded gate has no decision boundary, so there is no
        # false-acceptance tail to size against.
        raise RuntimeError(f"gate {gate.name!r} is recorded and cannot be sized")
    worst = _miss_rate(gate) + gate.tolerance
    key = (threshold, worst, eta, ceiling)
    if key in _EXACT_REQUIREMENT:
        return _EXACT_REQUIREMENT[key]
    limit, rate = float(eta), float(worst)

    def fits(repetitions: int) -> bool:
        return bool(binom.cdf(math.floor(repetitions * threshold), repetitions, rate) <= limit)

    if not fits(ceiling):
        _EXACT_REQUIREMENT[key] = None
        return None
    low, high = 1, ceiling // quantum
    while low < high:
        middle = (low + high) // 2
        if fits(middle * quantum):
            high = middle
        else:
            low = middle + 1
    _EXACT_REQUIREMENT[key] = low * quantum
    return _EXACT_REQUIREMENT[key]


def _bernoulli_design(gate: AcceptanceGate, repetitions: int, eta: Fraction):
    """The fixed design a stopping rule for this gate would inherit, or a refusal.

    ``FixedDesign`` states its operating characteristic at ``k* = ceil(n * (q +
    delta/2))``, the midpoint of the tolerance interval. A rule built on it
    therefore decides the gate the campaign declared only where the gate's own
    boundary IS that midpoint, which is exactly the gates whose margin is half
    their tolerance -- the scientific-excess family, by construction of the
    declared design and preserved by any profile rescaling. The availability
    gates place their boundary three quarters of the way to the tolerance edge,
    so no rule from this design would be deciding their declared claim, and
    they keep their fixed count and their mean-against-threshold decision.
    """
    if 2 * gate.margin != gate.tolerance:
        return None, "declared boundary is not the fixed design's operating point"
    try:
        design = FixedDesign.resolve(
            nominal_error=float(_miss_rate(gate)),
            tolerance=float(gate.tolerance),
            eta=float(eta),
            repetitions=repetitions,
        )
    except StoppingError as error:
        return None, str(error)
    return design, None


def _gate_plan(case, gate: AcceptanceGate, eta: Fraction, profile):
    """One gate's frozen decision parameters, and the argument that sized them."""
    divergence = gate_replications(gate, eta)
    bernoulli = _bernoulli_statistic(case, gate)
    exact = _exact_requirement(gate, eta, divergence) if bernoulli else None
    repetitions = divergence if exact is None else exact
    design, refusal = (None, "statistic is a proportion over more than one retained cell")
    if not gate.margin:
        refusal = "gate is recorded, not decided"
    elif bernoulli:
        design, refusal = _bernoulli_design(gate, repetitions, eta)
    return {
        "gate": gate.name,
        "statistic": gate.statistic,
        "direction": gate.direction,
        "target": gate.target,
        "tolerance": gate.tolerance,
        "margin": gate.margin,
        "conditioning": gate.conditioning,
        "reference": gate.reference,
        "threshold": gate_threshold(gate),
        "divergence_repetitions": divergence,
        "repetitions": repetitions,
        "requirement_argument": (
            "chernoff-hoeffding-divergence" if exact is None else "exact-binomial-tail"
        ),
        "decided_by": (
            "replication-mean-against-threshold"
            if design is None
            else (profile.sequential_rule or profile.stopping)
        ),
        "bernoulli_design": None if design is None else design.parameters(),
        "bernoulli_refusal": refusal,
    }


def _rebuild_gate(plan) -> AcceptanceGate:
    """The frozen gate exactly as the plan published it."""
    return AcceptanceGate(
        plan["gate"],
        plan["statistic"],
        plan["direction"],
        Fraction(plan["target"]),
        Fraction(plan["tolerance"]),
        Fraction(plan["margin"]),
        plan["conditioning"],
        None if plan["reference"] is None else Fraction(plan["reference"]),
    )


def _route(case):
    """Which executed route carries this case's evidence, and why."""
    if find_spec("tests.estimation._sequential_simulation") is None:
        return "executed_public_path", "sufficient-state simulator absent from this checkout"
    from tests.estimation._sequential_simulation import certifies_exactly

    if certifies_exactly(case):
        return "executed_sufficient_state", "sufficient state certified exact for this case"
    return "executed_public_path", "sufficient-state route not certified exact for this case"


def cost_model(path=None):
    """The declared per-look cost coefficients this campaign ranks cells by."""
    document = yaml.safe_load((path or PROFILES_PATH).read_text(encoding="utf-8"))
    model = document.get(CAMPAIGN, {}).get("cost_model")
    if not isinstance(model, dict) or not isinstance(model.get("laws"), dict):
        raise RuntimeError("the sequential campaign declares no cost model")
    return model


def case_cost_seconds(case, model) -> float:
    """Modelled seconds for ONE replication of *case*.

    Per look the campaign pays for one evaluation per retained cell. Below the
    law's knee in units per arm the exact kernels are cheap; above it they are
    flat, and inverting the confidence sequence -- which costs an order of
    magnitude more than the certificate -- is paid only at the look a gate
    actually reads. The larger arm crosses the knee first, so its unit count is
    what indexes the knee.

    Modelled, never measured: the coefficients carry a per-case band, and the
    runner meters real spend against the budget rather than trusting this.
    """
    law = model["laws"][case.law]
    knee = law["knee"]
    stopped_look_only = model["bounds_at"] == "stopped-look" and case.law != "scalar_mean"
    revealed = list(accumulate(campaign_schedule(case)))
    scale = max(case.allocation)
    final = len(revealed) - 1
    seconds = 0.0
    for look, units in enumerate(revealed):
        if knee is not None and units * scale < knee:
            seconds += law["cheap"]
            continue
        seconds += law["certificate"]
        if look == final or not stopped_look_only:
            seconds += law["bounds"]
    return _roster_size(case) * seconds + _capture_seconds(case, model)


def _capture_seconds(case, model) -> float:
    """Modelled seconds spent building and hashing records for one replication.

    Zero on the sufficient-state route, which materialises no records at all.
    The two record-bearing modes exist because the anchors the coefficients
    were fitted against were timed on the raw path: ``pre-fix`` carries the
    cubic ancestor replay the snapshot identity used to perform, ``post-fix``
    the two irreducible passes per look that remain now that validation is
    flat in ancestor count but still re-derives a prefix digest over every
    record revealed so far.
    """
    mode = model["capture"]
    if mode == "none":
        return 0.0
    coefficients = model["capture_coefficients"][mode]
    arms = 1 if case.law == "scalar_mean" else (1 if case.family == 1 else 2)
    width = (
        sum(case.allocation)
        if case.law == "scalar_mean"
        else case.allocation[0] + arms * case.allocation[1]
    )
    per_look = [size * width for size in campaign_schedule(case)]
    revealed = list(accumulate(per_look))
    passes = (
        sum(revealed)
        if mode == "post-fix"
        else sum(total + sum(revealed[:look]) for look, total in enumerate(revealed))
    )
    return coefficients["per_pass"] * passes + coefficients["per_record"] * revealed[-1]


def manifest_records(model=None):
    """The manifest as the records a cell set selects over, each with its cost.

    A cell's modelled cost is always stated at the DECLARED ledger's
    replication requirement, never at the requirement of the tier doing the
    selecting. Otherwise a budget would choose the cells whose cost it then
    changes, and the ranking would depend on which tier asked.
    """
    model = cost_model() if model is None else model
    declared = CERTIFICATION_LEDGER.per_decision_error
    records = []
    for case, gates in ACCEPTANCE_GATES:
        seconds = case_cost_seconds(case, model)
        replications = case_replications(gates, declared)
        records.append(
            {
                **asdict(case),
                "cost_seconds_per_replication": seconds,
                "cost_replications": replications,
                "cost_core_hours": seconds * replications / 3600.0,
            }
        )
    return records


def campaign_plan(profile, *, repetitions=None):
    """Resolve the profile into the cells, gates and counts the campaign will run."""
    model = cost_model()
    records = manifest_records(model)
    cases = principal_manifest()
    certification = profile.certification(records)
    costs = {record["name"]: record for record in records}
    ratio = profile.tolerance_fraction / CERTIFICATION_DESIGN.scientific_excess
    gated = [
        (case, tuple(rescaled_gate(gate, ratio) for gate in gates))
        for case, gates in ACCEPTANCE_GATES
        if certification[case.name]
    ]
    ledger = certification_ledger(gated)
    eta = ledger.per_decision_error
    index_of = {case.name: index for index, case in enumerate(cases)}
    plans: list[dict] = []
    modelled_total = 0.0
    required_total = 0
    divergence_total = 0
    for case, gates in gated:
        entries = [_gate_plan(case, gate, eta, profile) for gate in gates]
        required = max((entry["repetitions"] for entry in entries), default=0)
        divergence = max((entry["divergence_repetitions"] for entry in entries), default=0)
        modelled = costs[case.name]["cost_core_hours"]
        modelled_total += modelled
        required_total += required
        divergence_total += divergence
        route, reason = _route(case)
        plans.append(
            {
                "case_index": index_of[case.name],
                "case_id": case.name,
                "route": route,
                "route_reason": reason,
                "modelled_core_hours": modelled,
                "modelled_seconds_per_replication": costs[case.name][
                    "cost_seconds_per_replication"
                ],
                "required_replications": required,
                "divergence_replications": divergence,
                "requested_replications": (
                    required if repetitions is None else min(required, repetitions)
                ),
                "truncated_diagnostic": repetitions is not None and repetitions < required,
                # Draws the gates that no rule decides still demand, which is
                # the floor a curtailed run may never stop below.
                "uncurtailed_floor": max(
                    (
                        entry["repetitions"]
                        for entry in entries
                        if entry["bernoulli_design"] is None
                    ),
                    default=0,
                ),
                "gates": entries,
            }
        )
    return {
        "profile": {
            "campaign": profile.campaign,
            "name": profile.name,
            "tolerance": profile.tolerance,
            "delta": profile.delta,
            "tolerance_fraction": profile.tolerance_fraction,
            "cells": profile.cells.name,
            "cell_rule": profile.cells.rule,
            "cell_axes": list(profile.cells.axes),
            "stopping": profile.stopping,
            "sequential_rule": profile.sequential_rule,
        },
        "tolerance_ratio": ratio,
        "declared_scientific_excess": CERTIFICATION_DESIGN.scientific_excess,
        "declared_mc_margin": CERTIFICATION_DESIGN.max_mc_margin,
        "declared_availability_slack": CERTIFICATION_DESIGN.availability_slack,
        "profile_margins": {
            "scientific_excess": CERTIFICATION_DESIGN.scientific_excess * ratio,
            "scientific_excess_margin": CERTIFICATION_DESIGN.max_mc_margin * ratio,
            "availability_slack": CERTIFICATION_DESIGN.availability_slack * ratio,
            "availability_margin": CERTIFICATION_DESIGN.max_mc_margin * ratio,
        },
        "ledger": asdict(ledger),
        "declared_ledger": asdict(CERTIFICATION_LEDGER),
        "reproduces_declared_ledger": ledger == CERTIFICATION_LEDGER,
        "budget_core_hours": profile.cells.budget,
        "cost_model": model,
        "modelled_core_hours": modelled_total,
        "modelled_core_hours_band": model["band"],
        "modelled_reach_scope": (
            "cost is modelled, not measured, so reach is a prediction; the run "
            "meters real core-hours against budget_core_hours and reports the "
            "cells the meter did not reach"
        ),
        "plan_replications": required_total,
        "divergence_replications": divergence_total,
        "certification": certification,
        "certified_case_count": sum(certification.values()),
        "uncertified_case_count": sum(1 for value in certification.values() if not value),
        "cases": plans,
    }


def plan_digest(plan) -> str:
    """Identity of the resolved plan, so a summary cannot re-derive a different one."""
    return hashlib.sha256(_json(plan).encode()).hexdigest()


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


class _Accountant:
    """Seal one case's replications into its journal and accumulate its gates.

    Integer outcomes become journal counters, so the chain covers both the
    numerator and the denominator of every gated proportion. The exact
    rational statistics are summed here and published with the case result.
    """

    def __init__(self, journal, *, case_id):
        self._journal = journal
        self._case_id = case_id
        self.attempted = 0
        self.completed = 0
        self.failed = 0
        self.sums: dict[str, Fraction] = {}
        self.counts: dict[str, int] = {}
        self.totals: dict[str, Fraction] = {name: Fraction(0) for name in _RATIONALS}
        self.integers: dict[str, int] = {field: 0 for _, field in _COUNTERS}
        self.integers.update(dict.fromkeys(_INDICATORS, 0))

    def started(self, index):
        self.attempted += 1
        self._journal.started(index)

    def completed_record(self, index, record) -> dict[str, Fraction]:
        missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise RuntimeError(f"replication record for {self._case_id!r} omits {missing}")
        tally = ["completed"]
        for name, field in _COUNTERS:
            count = int(record[field])
            self.integers[field] += count
            if count:
                tally.extend([name] * count)
        for name in _INDICATORS:
            value = int(record[name])
            self.integers[name] += value
            if value:
                tally.append(name)
        for name in _RATIONALS:
            self.totals[name] += Fraction(record[name])
        statistics = replication_statistics(record)
        for name, value in statistics.items():
            self.sums[name] = self.sums.get(name, Fraction(0)) + value
            self.counts[name] = self.counts.get(name, 0) + 1
        self.completed += 1
        # A declared unavailability is an expected, gated outcome whose counts
        # the chain already carries, and the draw itself replays from the
        # frozen seed. An UNDECLARED reason is what the design says cannot
        # happen, so that is the record worth keeping whole.
        self._journal.completed(
            index,
            {
                "case_id": self._case_id,
                "repetition": index,
                "status": "completed",
                "undeclared_point_reasons": record["undeclared_point_reasons"],
                # Absent on the sufficient-state route, which does not observe
                # per-look geometry; None says so rather than inventing an
                # empty census.
                "missing_point_reasons": record.get("missing_point_reasons"),
                "interval_statuses": record.get("interval_statuses"),
                "completed_looks": record.get("completed_looks"),
            },
            tally=tally,
            exceptional=bool(record["undeclared_point_reasons"]),
        )
        return statistics

    def failed_record(self, index, failure):
        self.failed += 1
        self._journal.completed(
            index,
            {"case_id": self._case_id, "repetition": index, "status": "failed", **failure},
            tally=["failed"],
            exceptional=True,
        )

    def result(self):
        return {
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "unfinished": self.attempted - self.completed - self.failed,
            **self.integers,
            **{f"{name}_sum": self.totals[name] for name in _RATIONALS},
            "statistic_sums": dict(self.sums),
            "statistic_counts": dict(self.counts),
        }


def _rules(plan, profile):
    """Live rules for the gates this plan decides as Bernoulli streams."""
    rules = {}
    for entry in plan["gates"]:
        if entry["bernoulli_design"] is None:
            continue
        parameters = entry["bernoulli_design"]
        design = FixedDesign.resolve(
            nominal_error=parameters["nominal_error"],
            tolerance=parameters["tolerance"],
            eta=parameters["eta"],
            repetitions=parameters["repetitions"],
        )
        rules[entry["gate"]] = build_rule(
            design, stopping=profile.stopping, sequential_rule=profile.sequential_rule
        )
    return rules


def _stream(name, rule, abandoned):
    return {
        "rule": rule.rule,
        "decision": rule.decision,
        "draws": rule.draws,
        "misses": rule.misses,
        "abandoned": abandoned.get(name),
    }


def _chunk_size(rules, requested):
    """How many replications the simulator may advance before it can stop.

    The simulator vectorises in batches and a curtailed run can only end on a
    batch boundary, so the batch IS the resolution of curtailment: a case
    overshoots its rule's stopping time by up to one batch. Where the rules
    stop sooner than the simulator's own batching, say so and bound the
    overshoot by one expected stopping time; where they do not, the default
    batching is already the tighter of the two and vectorises better, so
    leave it alone. This only ever narrows a batch, never widens one.
    """
    from tests.estimation._sequential_simulation import _REPLICATION_CHUNK

    scales = [
        rule.design.repetitions
        if getattr(rule, "expected_draws", None) is None
        else max(1, round(rule.expected_draws(rule.design.nominal_error)))
        for rule in rules.values()
    ]
    if not scales:
        return None
    chunk = max(1, min(min(scales), requested))
    return chunk if chunk < _REPLICATION_CHUNK else None


def _public_worker(case, directory, plan, profile, accountant):
    """Drive the public capture/estimate/select path, one replication at a time."""
    import numpy as np

    from tests.estimation._sequential_acceptance import campaign_declaration, campaign_replication

    rng = np.random.default_rng(case.seed)
    declaration = campaign_declaration(case)
    _write(
        directory / "experiment.json",
        {
            "case": asdict(case),
            "scientific_identity": campaign_identity((case,)),
            "science": campaign_science(case),
            "registration": json.loads(declaration[0].model_dump_json()),
            "numpy": np.__version__,
            "bit_generator": type(rng.bit_generator).__name__,
            "initial_rng_state": rng.bit_generator.state,
            "route": plan["route"],
            "interim_geometry_observed": False,
        },
    )
    gates = {entry["gate"]: _rebuild_gate(entry) for entry in plan["gates"]}
    rules = _rules(plan, profile)
    abandoned: dict[str, str] = {}
    floor = plan["uncurtailed_floor"]

    def checkpoint(event):
        del event  # look geometry replays from the frozen seed; nothing reads it

    for index in range(plan["requested_replications"]):
        accountant.started(index)
        try:
            record = campaign_replication(
                case, rng, declaration, checkpoint, interim_geometry=False
            )
        except Exception as exc:
            accountant.failed_record(
                index,
                {
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc),
                },
            )
            raise
        statistics = accountant.completed_record(index, record)
        for name, rule in rules.items():
            if rule.resolved or name in abandoned:
                continue
            value = statistics.get(gates[name].statistic)
            if value is None:
                abandoned[name] = "statistic unmeasured at an executed replication"
                continue
            rule.observe(miss=_miss_flag(gates[name], value))
        if rules and index + 1 >= floor and all(rule.resolved for rule in rules.values()):
            break
    return {name: _stream(name, rule, abandoned) for name, rule in rules.items()}


def _sufficient_state_worker(case, directory, plan, profile, accountant):
    """Drive the sufficient-state simulator, journalling every replication it runs."""
    from tests.estimation._sequential_simulation import simulate_case

    requested = plan["requested_replications"]
    rules = _rules(plan, profile)

    def observer(event, index, payload):
        if event == "started":
            accountant.started(index)
        elif event == "completed":
            accountant.completed_record(index, payload)
        else:
            accountant.failed_record(index, dict(payload or {}))

    outcome = simulate_case(
        case,
        replications=requested,
        seed=case.seed,
        stopping=rules or None,
        floor=min(plan["uncurtailed_floor"], requested),
        chunk_size=_chunk_size(rules, requested),
        observer=observer,
    )
    # The simulator drives its own copies of these rules, so its accumulators
    # are an independent count of the same draws. Any disagreement is an
    # engine defect, not a number to publish.
    disagreement = {
        field: [accountant.integers[field], getattr(outcome, field)]
        for _, field in _COUNTERS
        if getattr(outcome, field) != accountant.integers[field]
    }
    for record, published in (
        ("nominal_ever_null_rejection", outcome.nominal_ever_null_rejections),
        ("nonnull_discovery", outcome.nonnull_discoveries),
    ):
        if published != accountant.integers[record]:
            disagreement[record] = [accountant.integers[record], published]
    if disagreement:
        raise RuntimeError(
            f"sufficient-state accumulators disagree with the journalled draws: {disagreement}"
        )
    _write(
        directory / "experiment.json",
        {
            "case": asdict(case),
            "scientific_identity": campaign_identity((case,)),
            "science": campaign_science(case),
            "route": plan["route"],
            "interim_geometry_observed": False,
            "simulator_replications": outcome.replications,
            "simulator_evidence_evaluations": outcome.evidence_evaluations,
            "simulator_distinct_evidence_states": outcome.distinct_evidence_states,
        },
    )
    return {
        stream.gate: {
            "rule": stream.rule,
            "decision": stream.decision,
            "draws": stream.draws,
            "misses": stream.misses,
            "abandoned": None,
        }
        for stream in outcome.gate_streams
    }


def _raise_checkpoint_memo_bound() -> dict:
    """Give a campaign worker a memo large enough for a case's reachable states.

    The exact-kernel memos are keyed on the sufficient state, so their hit
    rate is set by how many distinct states a case actually reaches, not by
    how many replications it runs. The library default is sized for a process
    evaluating a handful of checkpoints; a campaign worker evaluates one
    case's whole state space, and on the measured access stream the low-rate
    Bernoulli shapes keep climbing past the default -- 76.96% at 256, 82.35%
    at 512, 90.21% at 1024, saturating at 93.55% from 2048 -- so the default
    throws away reuse the campaign has already paid to generate. 4096 entries
    reaches saturation for those shapes at roughly 112 MB.

    It buys nothing for the high-rate, large-batch shapes, whose states are
    almost all distinct; those would need tens of gigabytes to hit and are
    left to miss rather than pretended about. The bound is recorded in the
    case's evidence either way.
    """
    from increment.estimation.sequential_result import set_checkpoint_memo_size

    set_checkpoint_memo_size(CHECKPOINT_MEMO_ENTRIES)
    return {"entries": CHECKPOINT_MEMO_ENTRIES}


def _memo_usage() -> list[dict]:
    """Reuse the exact kernels actually got, per memo, as plain evidence.

    Recorded because it is the one number that says whether the bound above
    was worth its memory on this case: a hit rate near zero means the case's
    states are all distinct and no affordable bound would have helped.
    """
    from increment.estimation.sequential_result import checkpoint_memo_usage

    return [
        {
            "memo": usage.name,
            "hits": usage.hits,
            "misses": usage.misses,
            "entries": usage.entries,
            "bound": usage.bound,
        }
        for usage in checkpoint_memo_usage()
    ]


def _worker(directory: Path) -> int:
    plan = json.loads((directory / "plan.json").read_text())
    manifest = json.loads((directory.parent / "manifest.json").read_text())
    if _sources() != manifest["source_files"]:
        raise RuntimeError("campaign source changed after its manifest was frozen")
    if campaign_identity(principal_manifest()) != manifest["scientific_identity"]:
        raise RuntimeError("campaign scientific declarations changed")
    profile = load_profile(manifest["plan"]["profile"]["name"], campaign=CAMPAIGN)
    case = principal_manifest()[plan["case_index"]]
    if case.name != plan["case_id"]:
        raise RuntimeError("frozen plan does not address the case it names")
    memo = _raise_checkpoint_memo_bound()
    drive = _public_worker if plan["route"] == "executed_public_path" else _sufficient_state_worker
    journal = DrawJournal(directory, case_id=case.name)
    accountant = _Accountant(journal, case_id=case.name)
    try:
        streams = drive(case, directory, plan, profile, accountant)
    finally:
        totals = journal.close()
    _write(
        directory / "result.json",
        {
            **accountant.result(),
            "journal_digest": totals.digest,
            "journal_blocks": totals.blocks,
            "interim_geometry_observed": False,
            "checkpoint_memo": {**memo, "usage": _memo_usage()},
            "gate_streams": streams,
        },
    )
    return 0


def _worker_entry(directory: Path) -> int:
    try:
        return _worker(directory)
    except Exception as exc:
        _write(
            directory / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "code": getattr(exc, "code", None),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _controller(output: Path, case_seconds: float, work_seconds: float) -> int:
    """Run the selected cells cheapest first, stopping when the budget is spent.

    The cost model only ORDERS the cells and predicts how far a tier reaches.
    What actually bounds a tier is this meter: cells run in modelled-cost
    order and the controller stops admitting them once measured core-hours
    reach the profile's declared budget. A model error therefore costs reach,
    which the summary reports cell by cell, and can never certify a cell that
    did not run. Admission is checked before a cell starts and the meter is
    charged after it finishes, so a run overruns its budget by at most the cost
    of the one cell that crossed it.
    """
    manifest = json.loads((output / "manifest.json").read_text())
    plans = {entry["case_index"]: entry for entry in manifest["plan"]["cases"]}
    budget = manifest["plan"]["budget_core_hours"]
    order = sorted(
        manifest["selected_indices"], key=lambda index: plans[index]["modelled_core_hours"]
    )
    deadline = time.monotonic() + work_seconds
    spent, failed = 0.0, False
    for index in order:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124
        if budget is not None and spent >= budget:
            _write(
                output / f"case-{index:04d}-unspent.json",
                {"reason": "declared budget exhausted", "metered_core_hours": spent},
            )
            continue
        directory = output / f"case-{index:04d}"
        directory.mkdir()
        _write(directory / "plan.json", plans[index])
        started = time.monotonic()
        code = run_with_budget(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; "
                "from scripts.run_sequential_campaign import _worker_entry; "
                "raise SystemExit(_worker_entry(Path(sys.argv[1])))",
                str(directory),
            ],
            tier=f"sequential-case-{index}",
            budget_seconds=min(case_seconds, remaining),
        )
        elapsed = time.monotonic() - started
        spent += elapsed / 3600.0
        _write(
            directory / "execution.json",
            {
                "returncode": code,
                "elapsed_seconds": elapsed,
                "metered_core_hours": elapsed / 3600.0,
            },
        )
        failed |= code != 0
        if code in (130, 143):
            return code
    _write(output / "meter.json", {"metered_core_hours": spent, "budget_core_hours": budget})
    return 2 if failed else 0


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


def _verdicts(plan, result):
    """Decide every gate, saying which procedure decided it and on what evidence."""
    gates = [_rebuild_gate(entry) for entry in plan["gates"]]
    statistics = {
        name: Fraction(total) / result["statistic_counts"][name]
        for name, total in result["statistic_sums"].items()
        if result["statistic_counts"].get(name)
    }
    means = {verdict.name: verdict for verdict in acceptance_verdicts(gates, statistics)}
    streams = result["gate_streams"]
    records = []
    for entry in plan["gates"]:
        mean = means[entry["gate"]]
        stream = streams.get(entry["gate"])
        if stream is not None and stream["abandoned"] is None:
            records.append(
                {
                    "gate": entry["gate"],
                    "status": _DECISION_STATUS[stream["decision"]],
                    "decided_by": stream["rule"],
                    "draws": stream["draws"],
                    "misses": stream["misses"],
                    "value": mean.value,
                    "threshold": mean.threshold,
                }
            )
            continue
        status = mean.status
        if status in ("accepted", "refused") and result["completed"] < entry["repetitions"]:
            status = "insufficient_replications"
        records.append(
            {
                "gate": entry["gate"],
                "status": status,
                "decided_by": "replication-mean-against-threshold",
                "draws": result["completed"],
                "misses": None,
                "value": mean.value,
                "threshold": mean.threshold,
                "bernoulli_refusal": entry["bernoulli_refusal"],
                "bernoulli_abandoned": None if stream is None else stream["abandoned"],
            }
        )
    return records


def _missingness(result):
    """Deterministic unknown-draw bounds; never a Monte-Carlo confidence statement."""
    attempted = result["attempted"]
    if not attempted:
        return None
    unknown = result["failed"] + result["unfinished"]
    return {
        metric: {
            "lower": Fraction(result[lower]) / attempted,
            "upper": (Fraction(result[upper]) + unknown) / attempted,
        }
        for metric, lower, upper in (
            (
                "nominal_ever_null_rejection",
                "nominal_ever_null_rejection",
                "nominal_ever_null_rejection",
            ),
            ("nominal_fdp", "nominal_fdp_sum", "nominal_fdp_sum"),
            ("nominal_fcp", "nominal_fcp_lower_sum", "nominal_fcp_upper_sum"),
            ("nonnull_discovery", "nonnull_discovery", "nonnull_discovery"),
        )
    }


def _reconcile(result, totals):
    """Every integer the chain holds must equal the one the case published."""
    expected = {name: result[field] for name, field in _COUNTERS}
    expected.update({name: result[name] for name in _INDICATORS})
    expected["completed"] = result["completed"]
    expected["failed"] = result["failed"]
    disagreement = {
        name: [value, totals.counters.get(name, 0)]
        for name, value in expected.items()
        if value != totals.counters.get(name, 0)
    }
    if result["attempted"] != totals.started:
        disagreement["attempted"] = [result["attempted"], totals.started]
    for record, statistic in _INDICATORS.items():
        published = result["statistic_sums"].get(statistic)
        if published is not None and Fraction(published) != result[record]:
            disagreement[statistic] = [published, result[record]]
    for name in _RATIONALS:
        total = Fraction(result[f"{name}_sum"])
        if not 0 <= total <= result["completed"]:
            disagreement[f"{name}_sum"] = [str(total), result["completed"]]
    return disagreement


def _executed_record(output, plan):
    """Verify one case's chain, reconcile it with the published result, and decide."""
    directory = output / f"case-{plan['case_index']:04d}"
    record = {
        **{key: plan[key] for key in ("case_index", "case_id", "route", "route_reason")},
        "certified_by_profile": True,
        "required_replications": plan["required_replications"],
        "divergence_replications": plan["divergence_replications"],
        "requested_replications": plan["requested_replications"],
        "truncated_diagnostic": plan["truncated_diagnostic"],
        "executed": True,
    }
    execution = directory / "execution.json"
    record["worker_exit_code"] = (
        json.loads(execution.read_text())["returncode"] if execution.exists() else None
    )
    failure = directory / "failure.json"
    record["failure_evidence"] = failure.name if failure.exists() else None
    if not (directory / "journal.jsonl").exists():
        return {**record, "status": "incomplete", "uncertified_reason": "case did not start"}
    totals = verify_journal(directory, case_id=plan["case_id"])
    record["journal"] = {
        "blocks": totals.blocks,
        "started": totals.started,
        "completed": totals.completed,
        "exceptional": totals.exceptional,
        "exceptional_dropped": totals.exceptional_dropped,
        "digest": totals.digest,
        "counters": dict(totals.counters),
    }
    result_path = directory / "result.json"
    if not result_path.exists():
        return {
            **record,
            "status": "incomplete",
            "uncertified_reason": "worker stopped before publishing its result",
        }
    result = json.loads(result_path.read_text())
    disagreement = _reconcile(result, totals)
    if disagreement:
        return {
            **record,
            "status": "incomplete",
            "uncertified_reason": "published result disagrees with the verified chain",
            "chain_disagreement": disagreement,
        }
    record.update(
        attempted=result["attempted"],
        completed=result["completed"],
        failed=result["failed"],
        unfinished=result["unfinished"],
        gate_streams=result["gate_streams"],
        interim_geometry_observed=result["interim_geometry_observed"],
        missingness_bounds=_missingness(result),
        missingness_bounds_scope="deterministic unknown-draw bounds, not MC confidence",
        chain_verified_counters=[field for _, field in _COUNTERS] + sorted(_INDICATORS),
        unchained_statistics=[f"{name}_sum" for name in _RATIONALS],
    )
    verdicts = _verdicts(plan, result)
    record["gate_verdicts"] = verdicts
    refused = sorted(
        entry["gate"] for entry in verdicts if entry["status"] not in ("accepted", "recorded")
    )
    reason = None
    if record["worker_exit_code"] != 0:
        reason = "worker did not exit cleanly"
    elif result["failed"] or result["unfinished"]:
        reason = "case has failed or unfinished replications"
    elif plan["truncated_diagnostic"]:
        reason = "run was truncated below the derived replication requirement"
    elif refused:
        reason = f"gates not accepted: {refused}"
    record["status"] = "certified" if reason is None else "incomplete"
    record["uncertified_reason"] = reason
    return record


def _summary(output, manifest, controller_code):
    plan = manifest["plan"]
    plans = {entry["case_index"]: entry for entry in plan["cases"]}
    executed = set(manifest["selected_indices"]) & set(plans)
    records = []
    for index, case in enumerate(manifest["cases"]):
        if index in executed:
            records.append(_executed_record(output, plans[index]))
            continue
        certified_by_profile = plan["certification"][case["name"]]
        records.append(
            {
                "case_index": index,
                "case_id": case["name"],
                "certified_by_profile": certified_by_profile,
                "executed": False,
                "status": "uncertified",
                "uncertified_reason": (
                    "profile-certified cell was not selected for this run"
                    if certified_by_profile
                    else f"cell set {plan['profile']['cells']!r} of profile "
                    f"{plan['profile']['name']!r} does not certify this cell"
                ),
            }
        )
    source_unchanged = _sources() == manifest["source_files"]
    certified = [record for record in records if record["status"] == "certified"]
    complete = controller_code == 0 and source_unchanged and len(certified) == len(records)
    summary = {
        "status": "certified" if complete else "incomplete",
        "certified": complete,
        "campaign": CAMPAIGN,
        "profile": plan["profile"]["name"],
        "tolerance": plan["profile"]["tolerance"],
        "tolerance_ratio": plan["tolerance_ratio"],
        "profile_margins": plan["profile_margins"],
        "stopping": plan["profile"]["stopping"],
        "sequential_rule": plan["profile"]["sequential_rule"],
        "reproduces_declared_ledger": plan["reproduces_declared_ledger"],
        "interim_geometry_observed": False,
        "controller_exit_code": controller_code,
        "source_unchanged": source_unchanged,
        "original_case_count": len(records),
        "profile_certified_case_count": plan["certified_case_count"],
        "executed_case_count": len(executed),
        "certified_case_count": len(certified),
        "uncertified_case_count": len(records) - len(certified),
        "public_path_case_count": sum(
            1 for record in records if record.get("route") == "executed_public_path"
        ),
        "sufficient_state_case_count": sum(
            1 for record in records if record.get("route") == "executed_sufficient_state"
        ),
        "budget_core_hours": plan["budget_core_hours"],
        "modelled_core_hours": plan["modelled_core_hours"],
        "modelled_core_hours_band": plan["modelled_core_hours_band"],
        "modelled_reach_scope": plan["modelled_reach_scope"],
        "metered_core_hours": (
            json.loads((output / "meter.json").read_text())["metered_core_hours"]
            if (output / "meter.json").exists()
            else None
        ),
        "required_replications": sum(plans[index]["required_replications"] for index in executed),
        "executed_replications": sum(int(record.get("attempted", 0)) for record in records),
        "records": records,
    }
    _write(output / "summary.json", summary)
    print(_json({key: value for key, value in summary.items() if key != "records"}))
    return 0 if complete else 2


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="release")
    parser.add_argument("--case", type=int, action="append", dest="cases")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--case-seconds", type=float, default=300)
    return parser


def _indices(cases, plan):
    """The profile-certified cases this run executes, in manifest order."""
    certified = [entry["case_index"] for entry in plan["cases"]]
    if cases is None:
        return tuple(certified)
    if len(cases) != len(set(cases)):
        raise ValueError("case indices must be unique")
    outside = sorted(set(cases) - set(certified))
    if outside:
        raise ValueError(
            f"cases {outside} are not certified by profile {plan['profile']['name']!r}; "
            "a run cannot execute a cell its profile does not certify"
        )
    chosen = set(cases)
    return tuple(index for index in certified if index in chosen)


def main(argv=None):
    started = time.monotonic()
    parser = _parser()
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if any(
        not math.isfinite(value) or not 0 < value <= 3600
        for value in (args.budget_seconds, args.case_seconds)
    ):
        parser.error("budgets must be finite and in (0, 3600]")
    # The derived requirement, not the retained historical ladder, is what a
    # correctly sized run asks for. The declared ledger's binding gate needs
    # 1,195,000 replications, which is above every retained historical stage,
    # so a cap taken from that ladder would refuse the campaign's own design.
    if args.repetitions is not None and not (
        1 <= args.repetitions <= CERTIFICATION_LEDGER.max_case_replications
    ):
        parser.error(
            f"repetitions must lie in [1, {CERTIFICATION_LEDGER.max_case_replications}], "
            "the largest case requirement the declared ledger derives (binding gate "
            f"{CERTIFICATION_LEDGER.binding_gate})"
        )
    os.chdir(ROOT)
    try:
        profile = load_profile(args.profile, campaign=CAMPAIGN)
    except ValueError as exc:
        parser.error(str(exc))
    plan = campaign_plan(profile, repetitions=args.repetitions)
    cases = principal_manifest()
    try:
        indices = _indices(args.cases, plan)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 3,
        "campaign": CAMPAIGN,
        "cases": [asdict(case) for case in cases],
        "science": [campaign_science(case) for case in cases],
        "scientific_identity": campaign_identity(cases),
        "certification_design": asdict(CERTIFICATION_DESIGN),
        "plan": plan,
        "plan_digest": plan_digest(plan),
        "selected_indices": indices,
        "repetitions_override": args.repetitions,
        "budget_seconds": args.budget_seconds,
        "case_seconds": args.case_seconds,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_files": _sources(),
        "historical_sampler_revision": "5f3b92fe",
        "scalar_sampler_revision": "iid-scalar-joint-v2",
        "source_lanes": ["synthetic_finalized_records"],
        "fraction_encoding": "exact numerator/denominator strings",
    }
    _write(args.output / "manifest.json", manifest)
    remaining = args.budget_seconds - (time.monotonic() - started)
    code = 124
    if remaining > 0:
        code = run_with_budget(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; "
                "from scripts.run_sequential_campaign import _controller; "
                "raise SystemExit(_controller(Path(sys.argv[1]), float(sys.argv[2]), "
                "float(sys.argv[3])))",
                str(args.output),
                str(args.case_seconds),
                str(remaining),
            ],
            tier="sequential-certification-campaign",
            budget_seconds=remaining,
        )
    return _summary(args.output, manifest, code)


if __name__ == "__main__":
    raise SystemExit(main())
