"""Collect bounded, source-frozen cluster diagnostics without certifying the historical matrix."""

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
from dataclasses import asdict, is_dataclass, replace
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def _json(value):
    return json.dumps(value, default=_default, sort_keys=True, allow_nan=False)


def _write(path, value, *, replace=False):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
    if replace:
        os.replace(temporary, path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()


def _append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")


def _sources():
    files = [*ROOT.joinpath("increment").rglob("*.py")]
    files.extend(
        ROOT / name
        for name in (
            "calibration/cluster_diagnostics.py",
            "scripts/run_test_tier.py",
            "tests/estimation/_i13_manifest.py",
            "tests/estimation/_i13_adapters.py",
            "tests/estimation/_i13_calibration.py",
            "tests/mc.py",
        )
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def _versions():
    versions = {"python": platform.python_version()}
    for package in (
        "numpy",
        "scipy",
        "pydantic",
        "scikit-learn",
        "polars",
        "narwhals",
        "duckdb",
        "ibis-framework",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _command(function, payload):
    return [
        sys.executable,
        "-c",
        f"import json, sys; from calibration.cluster_diagnostics import {function}; "
        f"raise SystemExit({function}(json.loads(sys.argv[1])))",
        _json(payload),
    ]


def _worker(payload):
    if payload.get("paired_width"):
        return _width_worker(payload)
    from tests.estimation._i13_adapters import estimate_of, truth_of
    from tests.estimation._i13_calibration import run_cell
    from tests.estimation._i13_manifest import ACCEPTANCE_MANIFEST

    output = Path(payload["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    index = payload["index"]
    if index not in manifest["selected_indices"] or _sources() != manifest["source_files"]:
        raise ValueError("unselected case or changed campaign source")
    cell = ACCEPTANCE_MANIFEST[index]
    directory = output / f"case-{index:04d}"
    directory.mkdir()
    counts = {
        "attempted": 0,
        "completed": 0,
        "refused": 0,
        "failed": 0,
        "unfinished": 0,
        "point_available": 0,
        "interval_available": 0,
        "covered": 0,
        "nonvacuous_intervals": 0,
        "finite_widths": 0,
        "acceptable_widths": 0,
        "rejection_available": 0,
        "rejected": 0,
    }
    geometries = {}
    progress = {
        "index": index,
        "name": cell.name,
        "requested": payload["repetitions"],
        "status": "running",
        "counts": counts,
        "geometries": geometries,
        "qualification": "working_approximation",
    }
    _write(directory / "progress.json", progress)

    def observe(replication, phase, data):
        if phase == "started":
            counts["attempted"] += 1
            counts["unfinished"] += 1
        else:
            counts["unfinished"] -= 1
            if phase == "completed":
                interval = data["interval"]
                counts["completed"] += 1
                counts["point_available"] += int(interval.point_available)
                counts["interval_available"] += int(interval.interval_available)
                counts["covered"] += int(interval.contains(data["truth"]))
                geometry = (
                    interval.confidence_set.geometry
                    if interval.confidence_set is not None
                    else "unavailable"
                    if not interval.interval_available
                    else "one_sided"
                    if interval.open_side is not None
                    else "bounded"
                )
                geometries[geometry] = geometries.get(geometry, 0) + 1
                counts["nonvacuous_intervals"] += int(
                    interval.interval_available and geometry not in ("all_real", "empty")
                )
                width = (
                    interval.ub - interval.lb
                    if interval.interval_available
                    and interval.lb is not None
                    and interval.ub is not None
                    else None
                )
                finite_width = width is not None and math.isfinite(width)
                counts["finite_widths"] += int(finite_width)
                counts["acceptable_widths"] += int(finite_width and width <= cell.width_limit)
                data = {
                    **data,
                    "geometry": geometry,
                    "width": width if finite_width else None,
                    "width_unavailable_reason": None if finite_width else geometry,
                }
                if interval.p_value is not None and math.isfinite(interval.p_value):
                    counts["rejection_available"] += 1
                    counts["rejected"] += int(interval.p_value < 0.05)
                elif interval.interval_available and interval.decision_unavailable_reason is None:
                    counts["rejection_available"] += 1
                    counts["rejected"] += int(not interval.contains(0.0))
            else:
                counts["refused" if phase == "excluded" else "failed"] += 1
        _write(directory / "progress.json", progress, replace=True)
        _append(directory / "draws.jsonl", {"replication": replication, "phase": phase, **data})

    try:
        result = run_cell(
            cell.design,
            payload["repetitions"],
            truth_of=lambda sample: truth_of(cell, sample),
            estimate_of=lambda sample, index: estimate_of(cell, sample, index),
            width_limit=cell.width_limit,
            observer=observe,
        )
        original_fixed_count = (
            payload["historical_repetitions"] is not None
            and payload["repetitions"] == payload["historical_repetitions"]
        )
        useful_output = all(
            counts[name]
            for name in ("point_available", "nonvacuous_intervals", "rejection_available")
        )
        progress.update(
            status="diagnostic_completed" if useful_output else "unavailable",
            source_unchanged=_sources() == manifest["source_files"],
            historical_gate=result.r03_gate(cell) if original_fixed_count else None,
            historical_gate_scope="original fixed-R requirements, not the practical release contract",
            mc_uncertainty_reason=None
            if original_fixed_count
            else "diagnostic draw budget; no MC certification",
            bias=result.bias,
            mean_finite_width=math.fsum(result.widths) / len(result.widths)
            if result.widths
            else None,
            exclusion_reasons=result.exclusion_reasons,
            failure_reasons=result.failure_reasons,
            interval_unavailable_reasons=result.interval_unavailable_reasons,
            decision_unavailable_reasons=result.decision_unavailable_reasons,
        )
        if counts["failed"] or counts["refused"]:
            progress["status"] = "completed_with_failures"
        _write(directory / "result.json", progress)
        return (
            0
            if progress["status"] == "diagnostic_completed" and progress["source_unchanged"]
            else 2
        )
    except Exception as exc:
        _write(
            directory / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def _width_worker(payload):
    from increment.errors import CodedError
    from increment.estimation.inference import LiftGuardError
    from tests.estimation._i13_adapters import estimate_of
    from tests.estimation._i13_manifest import ACCEPTANCE_MANIFEST, I13_ETA
    from tests.mc import coverage_lower_bound

    output = Path(payload["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    index = payload["index"]
    if index not in manifest["selected_width_indices"] or _sources() != manifest["source_files"]:
        raise ValueError("unselected width pair or changed campaign source")
    small = ACCEPTANCE_MANIFEST[index]
    large = next(
        cell
        for cell in ACCEPTANCE_MANIFEST
        if cell.useful
        and cell.design.dgp.k_t == 80
        and cell.estimator == small.estimator
        and cell.scale == small.scale
        and cell.nuisance == small.nuisance
        and cell.folds == small.folds
    )
    directory = output / f"width-{index:04d}"
    directory.mkdir()
    counts = dict.fromkeys(
        (
            "attempted",
            "completed",
            "refused",
            "failed",
            "unfinished",
            "interval_available",
            "contracted",
        ),
        0,
    )
    progress = {
        "index": index,
        "name": small.name,
        "large_name": large.name,
        "requested": payload["repetitions"],
        "status": "running",
        "counts": counts,
    }
    _write(directory / "progress.json", progress)
    for replication in range(payload["repetitions"]):
        counts["attempted"] += 1
        counts["unfinished"] += 1
        _write(directory / "progress.json", progress, replace=True)
        _append(directory / "draws.jsonl", {"replication": replication, "phase": "started"})
        try:
            intervals = []
            for cell in (small, large):
                dgp = cell.design.dgp
                sample = replace(dgp, seed=dgp.seed * 100_003 + replication).draw()
                intervals.append(estimate_of(cell, sample, replication))
            a, b = intervals
            available = (
                a.interval_available
                and b.interval_available
                and a.lb is not None
                and a.ub is not None
                and b.lb is not None
                and b.ub is not None
            )
            contracted = bool(available and b.ub - b.lb <= 0.8 * (a.ub - a.lb))
            counts["interval_available"] += int(available)
            counts["contracted"] += int(contracted)
            event = {"phase": "completed", "intervals": intervals, "contracted": contracted}
        except LiftGuardError as exc:
            event = {"phase": "refused", "reason": type(exc).__name__}
        except CodedError as exc:
            event = {"phase": "refused", "reason": exc.code}
        except Exception as exc:
            event = {"phase": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        counts[event["phase"]] += 1
        counts["unfinished"] -= 1
        _write(directory / "progress.json", progress, replace=True)
        _append(directory / "draws.jsonl", {"replication": replication, **event})
        if event["phase"] == "failed":
            break
    n = payload["repetitions"]
    full = payload["historical_repetitions"] == n and counts["completed"] == n
    lower = coverage_lower_bound(counts["contracted"], n, I13_ETA) if full else None
    gate = (
        {
            "lower": lower,
            "margin": counts["contracted"] / n - lower,
            "passes": lower >= 0.90 and counts["contracted"] / n - lower <= 0.0025,
        }
        if lower is not None
        else None
    )
    complete = counts["completed"] == n and counts["interval_available"] > 0
    progress.update(
        status="diagnostic_completed" if complete else "completed_with_failures",
        source_unchanged=_sources() == manifest["source_files"],
        historical_gate=gate,
        historical_gate_scope="original paired-width threshold 0.8, lower 0.90, margin 0.0025",
        certified=False,
    )
    _write(directory / "result.json", progress)
    return 0 if complete and progress["source_unchanged"] else 2


def _jobs(manifest):
    return [
        (index, paired)
        for paired, key in ((False, "selected_indices"), (True, "selected_width_indices"))
        for index in manifest[key]
    ]


def _case_name(index, paired):
    return f"{'width' if paired else 'case'}-{index:04d}"


def _controller(payload):
    from scripts.run_test_tier import run_with_budget
    from tests.estimation._i13_calibration import manifest_replications

    output = Path(payload["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    deadline = time.monotonic() + max(0, payload["budget"] - 4)
    historical_repetitions = (
        manifest_replications()
        if payload["selection"] == "full" and payload["repetitions"] is None
        else None
    )
    repetitions = payload["repetitions"] or (
        4 if payload["selection"] == "representative" else historical_repetitions
    )
    _write(
        output / "design.json",
        {
            "historical_repetitions": historical_repetitions,
            "requested_repetitions": repetitions,
            "historical_repetition_rule": "tests.estimation._i13_calibration.manifest_replications()",
        },
    )
    failed = False
    for index, paired in _jobs(manifest):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or _sources() != manifest["source_files"]:
            failed = True
            break
        name = _case_name(index, paired)
        code = run_with_budget(
            _command(
                "_worker",
                {
                    "output": str(output),
                    "index": index,
                    "paired_width": paired,
                    "repetitions": repetitions,
                    "historical_repetitions": historical_repetitions,
                },
            ),
            tier=f"cluster-{name}",
            budget_seconds=min(payload["case_budget"], remaining),
        )
        _write(output / f"{name}-execution.json", {"worker_exit_code": code})
        failed |= code != 0
    return 2 if failed else 0


def _summary(output, manifest, controller_code):
    design_path = output / "design.json"
    design = json.loads(design_path.read_text()) if design_path.exists() else {}
    records = []
    for index, paired in _jobs(manifest):
        name = _case_name(index, paired)
        directory = output / name
        final, partial = directory / "result.json", directory / "progress.json"
        record: dict[str, Any] = (
            json.loads((final if final.exists() else partial).read_text())
            if final.exists() or partial.exists()
            else {"index": index, "status": "not_started"}
        )
        execution = output / f"{name}-execution.json"
        code = json.loads(execution.read_text())["worker_exit_code"] if execution.exists() else None
        counts: dict[str, Any] = record.get("counts", {})
        attempted = counts.get("attempted", 0)
        requested = record.get("requested", design.get("requested_repetitions"))
        if attempted != sum(
            counts.get(key, 0) for key in ("completed", "refused", "failed", "unfinished")
        ):
            raise RuntimeError(f"inconsistent attempt accounting for {name}")
        complete = (
            final.exists()
            and code == 0
            and record["status"] == "diagnostic_completed"
            and counts.get("completed", 0) == requested
            and not counts.get("failed", 0)
            and not counts.get("unfinished", 0)
        )
        if not complete and record["status"] != "not_started":
            record["status"] = "interrupted" if code in (None, 124) else "failed"
        record.update(
            worker_exit_code=code,
            evidence_directory=name,
            paired_width=paired,
            requested=requested,
            not_started=requested - attempted if requested is not None else None,
        )
        for quantity, hits, available in (
            (("contracting_width", "contracted", "interval_available"),)
            if paired
            else (
                ("coverage", "covered", "interval_available"),
                ("rejection", "rejected", "rejection_available"),
            )
        ):
            unknown = attempted - counts.get(available, 0)
            record[f"{quantity}_missingness_bounds"] = (
                {
                    "lower": str(Fraction(counts.get(hits, 0), attempted)),
                    "upper": str(Fraction(counts.get(hits, 0) + unknown, attempted)),
                }
                if attempted
                else None
            )
        record["bounds_scope"] = "descriptive missingness bounds, not MC confidence"
        records.append(record)
    unchanged = _sources() == manifest["source_files"]
    complete = (
        controller_code == 0
        and unchanged
        and all(record["status"] == "diagnostic_completed" for record in records)
    )
    _write(
        output / "summary.json",
        {
            "status": "diagnostic_completed" if complete else "incomplete",
            "certified": False,
            "claim_scope": "qualified working-reference diagnostic; no original matrix certification",
            "source_unchanged": unchanged,
            "selected_count": len(records),
            "package_versions": _versions(),
            "original_count": len(manifest["cells"]),
            "records": records,
        },
    )
    return 0 if complete else 2


def main(argv=None):
    from scripts.run_test_tier import run_with_budget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", choices=("representative", "full"), default="representative")
    parser.add_argument("--case", type=int, action="append", dest="indices")
    parser.add_argument("--width-case", type=int, action="append", dest="width_indices")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--case-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    budget = (
        args.budget_seconds
        if args.budget_seconds is not None
        else (180 if args.selection == "representative" else 3600)
    )
    if any(
        not math.isfinite(value) or not 0 < value <= 3600 for value in (budget, args.case_seconds)
    ):
        parser.error("budgets must be finite and in (0, 3600]")
    if args.repetitions is not None and args.repetitions <= 0:
        parser.error("repetitions must be positive")
    output = args.output.resolve()
    os.chdir(ROOT)
    # Heavy fixed-R design sizing runs only inside the guarded full campaign.
    from tests.estimation._i13_manifest import ACCEPTANCE_MANIFEST, I13_ETA, I13_FAMILY_ALPHA

    if args.indices is not None:
        indices = args.indices
    elif args.width_indices is not None:
        indices = []
    elif args.selection == "full":
        indices = list(range(len(ACCEPTANCE_MANIFEST)))
    else:
        indices = []
        for estimator in ("mean", "ratio", "late", "sitewide", "iptw", "aipw", "dml"):
            candidates = [
                (i, cell)
                for i, cell in enumerate(ACCEPTANCE_MANIFEST)
                if cell.estimator == estimator
            ]

            def priority(item, estimator=estimator):
                cell = item[1]
                return (
                    not cell.useful,
                    abs(cell.design.dgp.k_t - 20) + abs(cell.design.dgp.k_c - 20),
                    cell.scale != ("absolute" if estimator == "sitewide" else "relative"),
                    cell.nuisance != ("fitted" if estimator in ("iptw", "aipw", "dml") else "none"),
                    cell.name,
                )

            indices.append(min(candidates, key=priority)[0])
        indices.sort()
    if len(set(indices)) != len(indices) or any(
        not 0 <= index < len(ACCEPTANCE_MANIFEST) for index in indices
    ):
        parser.error("case indices must be unique and present in the frozen manifest")
    width_indices = (
        args.width_indices
        if args.width_indices is not None
        else (
            [
                i
                for i, cell in enumerate(ACCEPTANCE_MANIFEST)
                if "contracting_width" in cell.quantities
            ]
            if args.selection == "full"
            else []
        )
    )
    if len(set(width_indices)) != len(width_indices) or any(
        not 0 <= index < len(ACCEPTANCE_MANIFEST)
        or "contracting_width" not in ACCEPTANCE_MANIFEST[index].quantities
        for index in width_indices
    ):
        parser.error("width cases must be unique contracting-width entries")
    output.mkdir(parents=True, exist_ok=False)
    cells = [
        {
            **asdict(cell),
            "name": cell.name,
            "historical_support": cell.support,
            "historical_refusal_code": cell.refusal_code,
            "quantities": cell.quantities,
            "bias_tolerance": cell.bias_tolerance,
            "width_limit": cell.width_limit,
            "rejection_rule": cell.rejection_rule,
        }
        for cell in ACCEPTANCE_MANIFEST
    ]
    _write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "cells": cells,
            "selected_indices": indices,
            "selected_width_indices": width_indices,
            "selection": args.selection,
            "historical_family_alpha": I13_FAMILY_ALPHA,
            "historical_eta": I13_ETA,
            "source_files": _sources(),
            "package_versions": _versions(),
            "seed_rule": "design.dgp.seed * 100003 + zero_based_replication",
            "certification_claim": "none",
        },
    )
    code = run_with_budget(
        _command(
            "_controller",
            {
                "output": str(output),
                "selection": args.selection,
                "repetitions": args.repetitions,
                "budget": budget,
                "case_budget": args.case_seconds,
            },
        ),
        tier="cluster-diagnostics",
        budget_seconds=budget,
    )
    _write(output / "launcher.json", {"controller_exit_code": code, "budget_seconds": budget})
    print(
        _json({"controller_exit_code": code, "certified": False, "evidence_directory": str(output)})
    )
    manifest = json.loads((output / "manifest.json").read_text())
    return _summary(output, manifest, code)


if __name__ == "__main__":
    raise SystemExit(main())
