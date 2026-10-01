"""Choose the default Beta prior weight for automatic Bernoulli monitoring.

Grid: declared control rate p0 in {0.02, 0.10, 0.30}; relative lift in
{0, 0.05, 0.10, 0.20}; prior Beta(w*p0, w*(1-p0)) for w in {2, 10, 50, 200}
plus the flat Beta(1, 1); 200 replications per cell; one look every 250 units
per arm up to 4000 per arm. A misspecification arm repeats lift {0, 0.10} for
every w with the true control rate at twice the declared one.

Each replication drives the registered route: a SequentialRegistration
carrying the candidate prior, capture_sequential_snapshot appending every
look, and checkpoint_certificate on the retained checkpoint, the evidence
estimate_sequential compares with 1/alpha in stat_sig(). The confidence
sequence is not inverted at every look (it is the same decision at fifteen
times the cost); the first replication of every cell runs estimate_sequential
at its stopping look and records whether the public decision agrees.

Decision rule (choose_weight): a weight qualifies when its null crossing rate
is at most alpha at every declared rate; among qualifying weights, for every
declared rate its power at the median lift (0.10) after 2000 units per arm
must be within five points of the best qualifying weight at that rate; the
smallest such weight wins. If no weight is within five points at every rate,
the smallest weight within five points of the best mean power wins.

Run outside pytest under a hard deadline:

    uv run python -m calibration.bernoulli_prior \\
        --output research/prior-weight --pilot
    uv run python -m calibration.bernoulli_prior \\
        --output research/prior-weight --budget-seconds 3600

Every completed cell is written to <output>/cells/<name>.json before the next
starts and is skipped on a rerun; summary.json and summary.md are written once
all cells exist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from decimal import Decimal
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

ALPHA = Fraction(1, 20)
DECLARED_RATES = ("0.02", "0.1", "0.3")
LIFTS = ("0", "0.05", "0.1", "0.2")
WEIGHTS = (2, 10, 50, 200)
MISSPECIFIED_LIFTS = ("0", "0.1")
LOOKS = 16
BATCH = 250
DECISION_LOOK = 8  # 2000 units per arm
SEED = 20260922
METRIC = "outcome"


def cell_grid(*, replications: int) -> list[dict]:
    cells = []
    for p0 in DECLARED_RATES:
        for lift in LIFTS:
            for weight in (*WEIGHTS, None):
                cells.append(_cell(p0, p0, lift, weight, replications))
    for p0 in DECLARED_RATES:
        true = str(Decimal(p0) * 2)
        for lift in MISSPECIFIED_LIFTS:
            for weight in WEIGHTS:
                cells.append(_cell(p0, true, lift, weight, replications))
    return cells


def _cell(declared, true, lift, weight, replications) -> dict:
    return {
        "declared_rate": declared,
        "true_rate": true,
        "lift": lift,
        "weight": weight,
        "replications": replications,
        "looks": LOOKS,
        "batch": BATCH,
        "seed": SEED,
    }


def cell_name(cell: dict) -> str:
    weight = "flat" if cell["weight"] is None else f"w{cell['weight']}"
    return f"p{cell['declared_rate']}-true{cell['true_rate']}-lift{cell['lift']}-{weight}"


def registration(declared_rate: str, weight: int | None):
    from increment import (
        JointReveal,
        PredictivePrior,
        SequentialCell,
        SequentialModel,
        SequentialRegistration,
    )

    if weight is None:
        prior = PredictivePrior(kind="beta", a=Fraction(1), b=Fraction(1))
    else:
        rate = Fraction(declared_rate)
        prior = PredictivePrior(kind="beta", a=weight * rate, b=weight * (1 - rate))
    model = SequentialModel(
        metric=METRIC,
        law="bernoulli",
        control_prior=prior,
        treatment_prior=prior,
        positive_population_control=True,
    )
    return SequentialRegistration(
        source_id="prior-weight-probe",
        definitions_id=f"prior-weight-probe-{declared_rate}-{weight}",
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id="prior-weight-probe-units-v1",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=0,
        ),
        models=(model,),
        roster=(SequentialCell(metric=METRIC, group_id="treatment", alpha=ALPHA),),
    )


def _records(rng, look: int, batch: int, control_rate: float, treatment_rate: float):
    control = rng.random(batch) < control_rate
    treatment = rng.random(batch) < treatment_rate
    rows = []
    for i in range(batch):
        rows.append(
            {
                "unit_id": f"{look:02d}-{i:04d}-control",
                "group_id": "control",
                "values": {METRIC: int(control[i])},
                "segments": {},
            }
        )
        rows.append(
            {
                "unit_id": f"{look:02d}-{i:04d}-treatment",
                "group_id": "treatment",
                "values": {METRIC: int(treatment[i])},
                "segments": {},
            }
        )
    return rows


def run_cell(cell: dict) -> dict:
    import numpy as np

    from increment import AlwaysValid, capture_sequential_snapshot, estimate_sequential
    from increment.estimation._certified import log_interval
    from increment.estimation.sequential_result import (
        SequentialCheckpoint,
        checkpoint_certificate,
        clear_checkpoint_memos,
    )

    reg = registration(cell["declared_rate"], cell["weight"])
    policy = AlwaysValid(registration=reg)
    threshold = (-log_interval(ALPHA)).hi
    control_rate = float(Fraction(cell["true_rate"]))
    treatment_rate = float(Fraction(cell["true_rate"]) * (1 + Fraction(cell["lift"])))
    rng = np.random.default_rng([cell["seed"], zlib.crc32(cell_name(cell).encode())])
    first_crossing = []
    agrees = None
    started = time.monotonic()
    for rep in range(cell["replications"]):
        snapshot = None
        crossing = None
        for look in range(1, cell["looks"] + 1):
            rows = _records(rng, look, cell["batch"], control_rate, treatment_rate)
            snapshot = capture_sequential_snapshot(
                reg,
                rows,
                source_id=reg.source_id,
                definitions_id=reg.definitions_id,
                finalized=True,
                previous=snapshot,
                append=snapshot is not None,
            )
            checkpoint = SequentialCheckpoint(
                registration_id=snapshot.registration_id,
                prefix_id=snapshot.prefix_id,
                filtration_id=reg.reveal.filtration_id,
                cell=reg.roster[0],
                model=reg.models[0],
                control=snapshot.arm(METRIC, "control"),
                treatment=snapshot.arm(METRIC, "treatment"),
                revealed_units=len(snapshot.records),
            )
            certificate = checkpoint_certificate(checkpoint)
            if certificate.status == "infinite":
                log_e = math.inf
            elif certificate.status == "zero" or certificate.log_e is None:
                log_e = -math.inf
            else:
                log_e = certificate.log_e.lo
            if log_e >= threshold:
                crossing = look
                break
        if rep == 0:
            assert snapshot is not None
            row = estimate_sequential(snapshot, policy).results[0]
            agrees = row.stat_sig() == (crossing is not None)
        first_crossing.append(crossing)
        clear_checkpoint_memos()
    cumulative = [
        sum(1 for c in first_crossing if c is not None and c <= look) / cell["replications"]
        for look in range(1, cell["looks"] + 1)
    ]
    return {
        **cell,
        "name": cell_name(cell),
        "cumulative_crossing": cumulative,
        "public_decision_agrees": agrees,
        "seconds": time.monotonic() - started,
    }


def _power_at_decision(cells, *, lift="0.1", look=DECISION_LOOK):
    power = {}
    for cell in cells:
        if cell["lift"] == lift and cell["true_rate"] == cell["declared_rate"]:
            power[(cell["declared_rate"], cell["weight"])] = cell["cumulative_crossing"][look - 1]
    return power


def _null_rate(cells):
    rates = {}
    for cell in cells:
        if cell["lift"] == "0" and cell["true_rate"] == cell["declared_rate"]:
            rates[(cell["declared_rate"], cell["weight"])] = cell["cumulative_crossing"][-1]
    return rates


def choose_weight(cells, *, tolerance: float = 0.05) -> int:
    power = _power_at_decision(cells)
    null = _null_rate(cells)
    qualifying = [w for w in WEIGHTS if all(null[(p0, w)] <= float(ALPHA) for p0 in DECLARED_RATES)]
    if not qualifying:
        raise SystemExit("no candidate weight keeps the null crossing rate within alpha")
    within_everywhere = [
        w
        for w in qualifying
        if all(
            power[(p0, w)] >= max(power[(p0, v)] for v in qualifying) - tolerance
            for p0 in DECLARED_RATES
        )
    ]
    if within_everywhere:
        return min(within_everywhere)
    mean_power = {w: sum(power[(p0, w)] for p0 in DECLARED_RATES) / 3 for w in qualifying}
    best = max(mean_power.values())
    return min(w for w in qualifying if mean_power[w] >= best - tolerance)


def summary_table(cells) -> str:
    by = {(c["declared_rate"], c["true_rate"], c["lift"], c["weight"]): c for c in cells}

    def value(p0, true, lift, weight, look):
        cell = by.get((p0, true, lift, weight))
        return "-" if cell is None else f"{cell['cumulative_crossing'][look - 1]:.3f}"

    lines = [
        "| p0 | prior | type I @4000 | power @2000, lift .05 | lift .10 | lift .20 "
        "| power @4000, lift .10 | true p0 = 2x: type I @4000 | power @2000, lift .10 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p0 in DECLARED_RATES:
        for weight in (None, *WEIGHTS):
            label = "Beta(1,1)" if weight is None else f"w={weight}"
            true = str(Decimal(p0) * 2)
            lines.append(
                f"| {p0} | {label} | {value(p0, p0, '0', weight, LOOKS)} "
                f"| {value(p0, p0, '0.05', weight, DECISION_LOOK)} "
                f"| {value(p0, p0, '0.1', weight, DECISION_LOOK)} "
                f"| {value(p0, p0, '0.2', weight, DECISION_LOOK)} "
                f"| {value(p0, p0, '0.1', weight, LOOKS)} "
                f"| {value(p0, true, '0', weight, LOOKS)} "
                f"| {value(p0, true, '0.1', weight, DECISION_LOOK)} |"
            )
    return "\n".join(lines) + "\n"


def _worker(output: Path, replications: int, processes: int) -> int:
    cells_dir = output / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        c
        for c in cell_grid(replications=replications)
        if not (cells_dir / f"{cell_name(c)}.json").exists()
    ]
    total = len(cell_grid(replications=replications))
    print(f"prior-weight probe: {len(pending)} of {total} cells pending", file=sys.stderr)
    done = total - len(pending)
    started = time.monotonic()
    with Pool(processes=processes) as pool:
        for result in pool.imap_unordered(run_cell, pending):
            path = cells_dir / f"{result['name']}.json"
            path.write_text(json.dumps(result, sort_keys=True) + "\n")
            done += 1
            elapsed = time.monotonic() - started
            print(
                f"prior-weight probe: {done}/{total} {result['name']} "
                f"agrees={result['public_decision_agrees']} {elapsed:.0f}s",
                file=sys.stderr,
            )
    return 0


def _summarise(output: Path, replications: int) -> int:
    cells_dir = output / "cells"
    cells = [json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))]
    expected = {cell_name(c) for c in cell_grid(replications=replications)}
    missing = sorted(expected - {c["name"] for c in cells})
    if missing:
        print(f"prior-weight probe incomplete: {len(missing)} cells missing", file=sys.stderr)
        return 2
    disagreements = [c["name"] for c in cells if c["public_decision_agrees"] is False]
    if disagreements:
        print(f"public decision disagreed with the certificate in {disagreements}", file=sys.stderr)
        return 3
    chosen = choose_weight(cells)
    table = summary_table(cells)
    (output / "summary.json").write_text(
        json.dumps(
            {"chosen_weight": chosen, "replications": replications, "cells": cells},
            sort_keys=True,
        )
        + "\n"
    )
    (output / "summary.md").write_text(f"chosen weight: {chosen}\n\n{table}")
    print(f"chosen weight: {chosen}")
    print(table)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--pilot", action="store_true", help="10 replications per cell")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 3600:
        parser.error("--budget-seconds must be finite and in (0, 3600]")
    replications = 10 if args.pilot else args.replications
    if args.worker:
        return _worker(args.output, replications, args.processes)
    from scripts.run_test_tier import run_with_budget

    code = run_with_budget(
        [
            sys.executable,
            "-m",
            "calibration.bernoulli_prior",
            "--worker",
            "--output",
            str(args.output),
            "--replications",
            str(replications),
            "--processes",
            str(args.processes),
        ],
        tier="bernoulli-prior-weight-probe",
        budget_seconds=args.budget_seconds,
    )
    if code != 0:
        print(
            f"prior-weight probe stopped with status {code}; partial cells retained",
            file=sys.stderr,
        )
        return code
    return _summarise(args.output, replications)


if __name__ == "__main__":
    raise SystemExit(main())
