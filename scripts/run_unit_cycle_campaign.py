"""Run the retained unit-cycle stress matrix with bounded, checkpointed execution.

Invoke with ``python -m scripts.run_unit_cycle_campaign``. Existing output
bundles are never overwritten. Restart a range deterministically with
``--start-design`` and the same source; failed outcomes remain in their bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

from scripts.run_test_tier import run_with_budget

ROOT = Path(__file__).resolve().parents[1]


def append_record(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")


def checkpoints(maximum):
    values = []
    count = 128
    while count <= maximum:
        values.append(count)
        count *= 4
    return tuple(values)


def command(args, *internal):
    result = [
        sys.executable,
        "-m",
        "scripts.run_unit_cycle_campaign",
        "--output",
        str(args.output),
        "--budget-seconds",
        str(args.budget_seconds),
        "--case-budget-seconds",
        str(args.case_budget_seconds),
        "--max-repetitions",
        str(args.max_repetitions),
        "--start-design",
        str(args.start_design),
        "--stop-design",
        str(args.stop_design),
        *internal,
    ]
    if args.smoke:
        result.append("--smoke")
    return result


def worker(args):
    from dataclasses import replace

    from tests._unit_cycle_design import CELLS, DESIGNS, MANIFEST
    from tests.simulate.test_unit_cycle_calibration import (
        calibrate_cell,
        test_representative_independent_analytic_proof,
        test_representative_public_parity,
    )
    from tests.simulate.unit_cycle_sizing import allocate, calibration_schedule

    design_index = args._case_index
    cell_index, cell = DESIGNS[design_index]
    path = args.output / f"design-{design_index:04d}.jsonl"
    levels = (args.max_repetitions,) if args.smoke else checkpoints(args.max_repetitions)
    schedule = calibration_schedule(
        total_error=allocate(MANIFEST["family_alpha"], 2),
        cases=len(DESIGNS),
        checkpoints=levels,
    )
    append_record(
        path,
        {
            "kind": "case_start",
            "design_index": design_index,
            "cell_index": cell_index,
            "cell": cell.id,
            "schedule": schedule,
            "scope": "diagnostic_smoke" if args.smoke else "full_matrix_design",
        },
    )
    try:
        for effect in MANIFEST["axes"]["effect"]:
            original_cell = replace(cell, effect=effect)
            original_index = CELLS.index(original_cell)
            properties = {}
            test_representative_independent_analytic_proof(original_cell, properties.__setitem__)
            test_representative_public_parity(original_index, original_cell)
            append_record(
                path,
                {
                    "kind": "analytic_and_public_parity",
                    "cell_index": original_index,
                    "cell": original_cell.id,
                    "analytic_proof": json.loads(properties["analytic_proof"]),
                },
            )
        for index, (count, error) in enumerate(
            zip(levels, schedule["checkpoint_error"], strict=True), 1
        ):

            def progress(record, *, checkpoint=index, repetitions=count):
                append_record(
                    path, {"checkpoint": checkpoint, "repetitions": repetitions, **record}
                )

            result = calibrate_cell(
                cell_index, repetitions=count, mc_error=error, progress=progress
            )
            append_record(path, {"checkpoint": index, **result})
            print(
                json.dumps(
                    {"design_index": design_index, "repetitions": count, "status": result["status"]}
                ),
                flush=True,
            )
            if result["status"] in {"certified", "failed"}:
                append_record(path, {"kind": "case_end", "status": result["status"]})
                return 0 if result["status"] == "certified" else 1
        append_record(path, {"kind": "case_end", "status": "inconclusive"})
        return 2
    except Exception as error:
        append_record(
            path,
            {
                "kind": "case_end",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def controller(args):
    from tests._unit_cycle_design import DESIGNS
    from tests.simulate.test_unit_cycle_calibration import RELEASE_CELLS

    started = time.monotonic()
    if args.smoke:
        lookup = {cell: index for index, (_, cell) in enumerate(DESIGNS)}
        indices = [lookup[cell] for cell in RELEASE_CELLS]
    else:
        indices = list(range(args.start_design, args.stop_design))
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "design_indices": indices,
                "original_design_count": len(DESIGNS),
                "original_cells_per_design": 2,
            }
        ),
        encoding="utf-8",
    )
    outcomes = {}
    grace = min(1.0, args.budget_seconds / 40)
    for index in indices:
        remaining = args.budget_seconds - (time.monotonic() - started) - 2 * grace
        if remaining <= 0:
            return 124
        code = run_with_budget(
            command(args, "--_case-index", str(index)),
            tier=f"unit-cycle-design-{index}",
            budget_seconds=min(args.case_budget_seconds, remaining),
            grace_seconds=grace,
        )
        status = {0: "certified", 1: "failed", 2: "inconclusive", 124: "timeout"}.get(
            code, "interrupted"
        )
        outcomes[index] = status
        append_record(
            args.output / "controller.jsonl",
            {
                "kind": "case_process_end",
                "design_index": index,
                "status": status,
                "exit_code": code,
            },
        )
        if code not in (0, 1, 2, 124):
            return code
        if status == "failed":
            return 1
    status = (
        "certified" if all(value == "certified" for value in outcomes.values()) else "inconclusive"
    )
    (args.output / "controller_end.json").write_text(
        json.dumps(
            {
                "status": status,
                "requested_designs": len(indices),
                "attempted_designs": len(outcomes),
                "full_matrix_covered": not args.smoke and len(indices) == len(DESIGNS),
            }
        ),
        encoding="utf-8",
    )
    return 0 if status == "certified" else 2


def provenance():
    paths = sorted((ROOT / "increment").rglob("*.py"))
    paths += [
        ROOT / "tests/_unit_cycle_design.py",
        ROOT / "tests/unit_cycle_prospective.json",
        ROOT / "tests/simulate/test_unit_cycle_calibration.py",
        ROOT / "tests/simulate/unit_cycle_sizing.py",
        ROOT / "scripts/run_test_tier.py",
        Path(__file__).resolve(),
    ]
    return {
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "python": sys.version,
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "polars")},
    }


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-repetitions", type=int)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--case-budget-seconds", type=float, default=60)
    parser.add_argument("--start-design", type=int, default=0)
    parser.add_argument("--stop-design", type=int)
    parser.add_argument("--_controller", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_case-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    manifest_path = ROOT / "tests/unit_cycle_prospective.json"
    original = json.loads(manifest_path.read_text())
    if args.stop_design is None:
        args.stop_design = original["model_design_count"]
    if args.max_repetitions is None:
        args.max_repetitions = 8 if args.smoke else 8192
    if not 1 <= args.budget_seconds <= 3600 or not 0 < args.case_budget_seconds <= 3600:
        parser.error("overall budget must be in [1,3600] and case budget in (0,3600]")
    if args.max_repetitions < (1 if args.smoke else 128):
        parser.error("full campaigns need at least128 repetitions; smoke counts must be positive")
    if not 0 <= args.start_design < args.stop_design <= original["model_design_count"]:
        parser.error("requested design range must be nonempty and inside the retained manifest")
    if args._case_index is not None and not 0 <= args._case_index < original["model_design_count"]:
        parser.error("invalid internal design index")
    args.output = args.output.resolve()
    if args._case_index is not None:
        return worker(args)
    if args._controller:
        return controller(args)
    if args.output.exists():
        parser.error("output bundle already exists; use a new path to preserve prior evidence")
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "scope": "diagnostic_smoke" if args.smoke else "declared_design_range",
                "scientific_acceptance_eligible": not args.smoke,
                "original_manifest": original,
                "configuration": {
                    "start_design": args.start_design,
                    "stop_design": args.stop_design,
                    "max_repetitions": args.max_repetitions,
                    "budget_seconds": args.budget_seconds,
                    "case_budget_seconds": args.case_budget_seconds,
                    "checkpoint_rule": "128 * 4**(j-1); diagnostic smoke uses its declared count",
                    "family_reservation": "half of original I14 family_alpha; divided over ALL2160 designs",
                    "time_spending": "case_error / (j*(j+1))",
                    "numerical_assumption": "SciPy beta inversions with outward binary64 rounding, not validated-arithmetic enclosures",
                },
                **provenance(),
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    outer_grace = min(5.0, args.budget_seconds / 10)
    remaining = args.budget_seconds - (time.monotonic() - started) - outer_grace
    code = (
        124
        if remaining <= 0
        else run_with_budget(
            command(args, "--_controller"),
            tier="unit-cycle-campaign",
            budget_seconds=remaining,
            grace_seconds=outer_grace,
        )
    )
    status = {0: "complete", 1: "failed", 2: "inconclusive", 124: "timeout"}.get(
        code, "interrupted"
    )
    ending = {
        "status": status,
        "exit_code": code,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "diagnostic_smoke" if args.smoke else "declared_design_range",
        "full_matrix_certified": code == 0
        and not args.smoke
        and args.start_design == 0
        and args.stop_design == original["model_design_count"],
    }
    (args.output / "campaign_end.json").write_text(json.dumps(ending), encoding="utf-8")
    print(json.dumps(ending), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
