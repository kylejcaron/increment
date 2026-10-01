"""Collect bounded diagnostics for every frozen inference case; never certify the candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWAP_ARGUMENT = "aa_allocation_swap_equivariance"
SWAP_ARGUMENT_TEXT = (
    "An A/A cell and its arm-size-swapped twin are one simulation read two ways: "
    "on a fixed sample with the replicate stream held fixed, exchanging the arm "
    "roles negates both root series, both centering targets and the additive "
    "endpoints exactly, and leaves the studentization, cutoffs and pilots "
    "bitwise identical. The executed cell's draws are therefore draws of the "
    "twin's own declared law, and its gate outcomes are the twin's. "
    "tests.estimation.test_inference_calibration."
    "test_aa_allocation_swap_is_pathwise_equivariant pins the identity."
)
SEED_REASSIGNMENT_TEXT = (
    "Cells that differ only in the clipping quantile execute on one draw stream, "
    "so each unit runs under its first cell's declared seed. This moves "
    "provenance, not statistics: within a cell the draws remain iid, every exact "
    "Clopper-Pearson bound stays exact, and the Bonferroni union over cells is "
    "insensitive to the dependence this introduces across cells."
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _safe(item) for key, item in value.items()}
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                result[f"{key}_unavailable_reason"] = "nonfinite_numeric_value"
        return result
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if type(value).__module__.startswith("numpy"):
        return _safe(value.item())
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _publish(path: Path, value: Any) -> None:
    """Publish strict JSON exclusively; finalized evidence cannot be replaced."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_safe(value), stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _provenance() -> dict[str, Any]:
    paths = [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "tests/_i15_manifest.json"]
    for directory in ("increment", "tests", "scripts", "calibration"):
        paths.extend((ROOT / directory).rglob("*.py"))
    # The journal and the execution profiles decide what a run does and what it
    # claims, so their declarations belong inside the evidence hash too.
    paths.extend((ROOT / "calibration").rglob("*.yaml"))
    packages = {}
    for name in ("numpy", "scipy", "pydantic", "narwhals", "polars", "ibis-framework", "duckdb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
        },
    }


def _unit_case_ids(unit: Any) -> list[str]:
    """A unit's declared case ids; the plan is read back from JSON, so narrow it."""
    case_ids = unit["case_ids"]
    if not isinstance(case_ids, list):
        raise RuntimeError("unit plan must declare a list of case ids")
    return [str(case_id) for case_id in case_ids]


def _bound(value, status, index) -> float:
    """Order an endpoint on the extended reals; a finite status must carry a value."""
    if status != "finite":
        return -math.inf if index == 0 else math.inf
    if value is None:
        raise RuntimeError("a finite interval endpoint cannot be null")
    return float(value)


def _interval(interval, statuses=None) -> dict[str, Any]:
    values = (None, None) if interval is None else tuple(interval)
    if statuses is None:
        statuses = tuple(
            "undefined"
            if value is None or math.isnan(value)
            else "finite"
            if math.isfinite(value)
            else "unbounded"
            for value in values
        )
    bounds: tuple[float, ...] = tuple(
        _bound(value, status, index)
        for index, (value, status) in enumerate(zip(values, statuses, strict=True))
    )
    available = all(
        (status == "finite" and value is not None and math.isfinite(value))
        or (status == "unbounded" and (value is None or value == bound))
        for value, status, bound in zip(values, statuses, bounds, strict=True)
    )
    available = available and bounds[0] <= bounds[1]
    return {
        "values": _safe(values),
        "statuses": statuses,
        "available": available,
        "nonvacuous": available and "finite" in statuses,
    }


def _observation(case, observation) -> dict[str, Any]:
    relative = _interval(observation.get("interval"), observation.get("interval_status"))
    additive = _interval(
        observation.get("additive_interval"), observation.get("additive_interval_status")
    )
    point = observation.get("point")
    point_available = point is not None and math.isfinite(point)
    coverage = rejected = None
    if relative["available"]:
        lo, hi = (
            value if status == "finite" else -math.inf if index == 0 else math.inf
            for index, (value, status) in enumerate(
                zip(relative["values"], relative["statuses"], strict=True)
            )
        )
        directional = case["family"].startswith("one_sided")
        coverage = bool(lo <= case["truth"] and (directional or case["truth"] <= hi))
        rejected = bool(lo > 0 or (not directional and hi < 0))
    return {
        "observation": observation,
        "relative": relative,
        "additive": additive,
        "point_available": point_available,
        "point_unavailable_reason": None if point_available else "not_returned_by_adapter",
        "usable": point_available or relative["nonvacuous"] or additive["nonvacuous"],
        "covers_truth": coverage,
        "rejects_zero_by_interval": rejected,
        "p_value": None,
        "p_value_reason": "this adapter returns intervals, not a candidate p-value",
        "reference_p_value": observation.get("reference_p_value"),
        "reference_interval": observation.get("oracle"),
    }


def _swap_exact(event: str, detail: dict[str, Any]) -> bool:
    """Whether the arm-swapped twin's draw is exactly determined by this one.

    Every quantity a gate reads is bitwise identical or an exact negation on
    the log scale. The single asymmetry is representability: a lift-scale
    endpoint ``expm1(b)`` can be finite and above -1 while ``expm1(-b)`` is
    not. A refusal transfers unchanged, because every refusal predicate reads
    only the pooled cutoff, the per-arm means, the density and the
    studentization, none of which the role swap moves.
    """
    if event == "refused":
        return True
    if event != "completed":
        return False
    relative = detail.get("relative")
    if relative is None or tuple(relative["statuses"]) != ("finite", "finite"):
        return False
    for value in relative["values"]:
        if value is None or not math.isfinite(value) or value <= -1:
            return False
        try:
            mirrored = math.expm1(-math.log1p(value))
        except (OverflowError, ValueError):
            return False
        if not math.isfinite(mirrored) or mirrored <= -1:
            return False
    return True


def _observer(case, journal, *, seed=None, mirrored=False):
    """Drive the hash-chained journal; retain whole records only when notable."""

    def observe(event, index, observation):
        if event == "started":
            journal.started(index)
            return
        tally = [event]
        detail: dict[str, Any] = {}
        usable = False
        if observation is not None:
            if event == "completed":
                detail = _observation(case, observation)
                usable = bool(detail.get("usable"))
            else:
                detail = {"observation": observation, "usable": False}
        if usable:
            tally.append("usable")
        if mirrored and _swap_exact(event, detail):
            tally.append("swap_exact")
        journal.completed(
            index,
            _safe(
                {
                    "case_id": case["case_id"],
                    "draw": index,
                    "seed": case.get("seed") if seed is None else seed,
                    "stream": 0 if case.get("kind") == "exact" else index,
                    "status": event,
                    **detail,
                }
            ),
            tally=tally,
            # A completed, usable draw is reproducible from its seed; anything
            # else is what a reader would actually need to inspect.
            exceptional=event != "completed" or not usable,
        )

    return observe


def _single_case_observations(case_id, observations):
    for observation in observations:
        yield {case_id: observation}


def _unit_worker(cases, directory, requested, unit, journals):
    from tests.estimation.test_inference_calibration import (
        _campaign_observations,
        _run_shared_cases,
        _winsor_unit_observations,
    )

    seed = int(unit["seed"])
    mirrors = dict(unit.get("mirrors") or {})
    observers = {
        case["case_id"]: _observer(
            case, journals[case["case_id"]], seed=seed, mirrored=case["case_id"] in mirrors
        )
        for case in cases
    }
    stream = None

    def build(rng):
        nonlocal stream
        if stream is None:
            if cases[0]["family"] == "pooled_winsor":
                stream = _winsor_unit_observations(cases, rng)
            elif len(cases) == 1:
                stream = _single_case_observations(
                    cases[0]["case_id"], _campaign_observations(cases[0], rng)
                )
            else:
                raise RuntimeError("only clipping-quantile siblings share one draw stream")
        return next(stream)

    reports = _run_shared_cases(cases, build, seed=seed, repetitions=requested, observers=observers)
    for case in cases:
        case_id = case["case_id"]
        report = reports[case_id]
        report["certified"] = False
        report["declared_seed"] = case["seed"]
        report["executed_seed"] = seed
        report["seed_reassigned"] = seed != case["seed"]
        report["seed_reassignment_scope"] = SEED_REASSIGNMENT_TEXT if seed != case["seed"] else None
        report["shared_bootstrap_draw_with"] = [
            other["case_id"] for other in cases if other["case_id"] != case_id
        ]
        report["certifies_by_equivariance"] = mirrors.get(case_id)
        report["gate_scope"] = (
            "historical fixed-R criteria"
            if report["historical_repetitions_completed"]
            else "fixed diagnostic request; not the original historical gate"
        )
        _publish(directory / case_id / "calibration.json", report)


def _exact_worker(case, directory, requested, journal):
    import numpy as np
    from scipy.stats import norm

    from increment.errors import CodedError
    from tests._i15_design import exact_label_reference
    from tests.estimation.test_inference_calibration import (
        _winsor_array_region,
        _winsor_region_observation,
    )

    raw = np.asarray(case["dgp"]["raw_pooled_values"], dtype=float)
    assignments, reference_pvalues, cutoff = exact_label_reference(raw, 4, 0.99)
    observe = _observer(case, journal)
    rejected = unavailable = 0
    widths, log_widths = [], []
    for index, indices in enumerate(assignments[:requested]):
        observe("started", index, None)
        mask = np.zeros(len(raw), dtype=bool)
        mask[list(indices)] = True
        try:
            observation = _winsor_region_observation(
                _winsor_array_region(raw[mask], raw[~mask], 0.99, stream=0)
            )
        except CodedError as exc:
            observe("refused", index, {"reason": exc.code})
            unavailable += 1
            continue
        except Exception as exc:
            observe("failed", index, {"error_type": type(exc).__name__, "error": str(exc)})
            raise
        observe(
            "completed",
            index,
            {
                **observation,
                "reference_p_value": float(reference_pvalues[index]),
                "control_indices": list(indices),
            },
        )
        lo, hi = observation["interval"]
        if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
            unavailable += 1
            continue
        rejected += lo > 0 or hi < 0
        widths.append(hi - lo)
        log_widths.append(math.log1p(hi) - math.log1p(lo) if lo > -1 else math.inf)
    transformed = np.minimum(raw, cutoff)
    width_cap = (
        8
        * float(norm.isf(0.025))
        * math.sqrt(
            float(transformed.var(ddof=1)) * (1 / 4 + 1 / 4) / float(transformed.mean()) ** 2
        )
    )
    complete = requested == len(assignments) == 70
    reference_rejections = sum(float(value) <= 0.05 for value in reference_pvalues)
    _publish(
        directory / "calibration.json",
        {
            "case_id": case["case_id"],
            "certified": False,
            "executed_repetitions": requested,
            "historical_repetitions_completed": complete,
            "reference_null": "raw_distribution_exchangeability",
            "reference_rejections": reference_rejections,
            "reference_label_count": len(assignments),
            "production_auxiliary_stream": 0,
            "unavailable": unavailable,
            "rejected": rejected,
            "widths": widths,
            "log_widths": log_widths,
            "log_width_cap": width_cap,
            "gate_scope": "exact over all 70 labels conditional on fixed bootstrap randomness",
            "gates": {
                "independent_reference": reference_rejections == 2,
                "availability": unavailable == 0,
                "finite_positive_width": all(math.isfinite(w) and w > 0 for w in widths),
                "useful_width": all(0 < w <= width_cap for w in log_widths),
                "sharp_null_level": rejected / 70 <= 0.055,
            }
            if complete
            else None,
            "gate_unavailable_reason": None if complete else "incomplete_exact_label_enumeration",
        },
    )


def _worker(directory: Path) -> int:
    from calibration.journal import DrawJournal
    from tests._i15_design import load_manifest

    unit = json.loads((directory / "unit.json").read_text())
    by_id = {case["case_id"]: case for case in load_manifest()["cases"]}
    cases = [by_id[case_id] for case_id in unit["case_ids"]]
    requested = int(unit["requested"])
    with ExitStack() as stack:
        journals = {}
        for case in cases:
            case_directory = directory / case["case_id"]
            case_directory.mkdir()
            journals[case["case_id"]] = stack.enter_context(
                DrawJournal(case_directory, case_id=case["case_id"])
            )
        if cases[0].get("kind") == "exact":
            (case,) = cases
            _exact_worker(case, directory / case["case_id"], requested, journals[case["case_id"]])
        else:
            _unit_worker(cases, directory, requested, unit, journals)
    _publish(directory / "worker-complete.json", {"requested": requested})
    return 0


def _controller(output: Path, case_seconds: float, work_seconds: float) -> int:
    from scripts.run_test_tier import run_with_budget

    selection = json.loads((output / "selection.json").read_text())
    deadline = time.monotonic() + work_seconds
    failed = False
    for unit in selection["units"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124
        directory = output / unit["directory"]
        directory.mkdir()
        _publish(directory / "unit.json", unit)
        command = [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from scripts.run_i15_campaign import _worker; "
            "raise SystemExit(_worker(Path(sys.argv[1])))",
            str(directory),
        ]
        started = time.monotonic()
        code = run_with_budget(
            command,
            tier=f"i15-{unit['directory']}",
            budget_seconds=min(case_seconds * len(unit["case_ids"]), remaining),
        )
        _publish(
            directory / "execution.json",
            {
                "returncode": code,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        failed |= code != 0
        if code in (130, 143):
            return code
    return 2 if failed else 0


def _executed_record(output, item, unit):
    from calibration.journal import verify as verify_journal

    directory = output / item["directory"]
    totals = verify_journal(directory, case_id=item["case_id"])
    if totals.started > item["requested"]:
        raise RuntimeError(f"invalid attempt journal for {item['case_id']}")
    counts = {
        status: totals.counters.get(status, 0) for status in ("completed", "refused", "failed")
    }
    counts.update(
        attempted=totals.started,
        unfinished=totals.started - sum(counts.values()),
        not_started=item["requested"] - totals.started,
    )
    if counts["attempted"] != sum(
        counts[key] for key in ("completed", "refused", "failed", "unfinished")
    ):
        raise RuntimeError(f"unrecognized result status for {item['case_id']}")
    usable = totals.counters.get("usable", 0)
    unit_directory = output / unit["directory"]
    execution_path = unit_directory / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    worker_code = execution.get("returncode")
    complete = (unit_directory / "worker-complete.json").exists() and worker_code == 0
    complete &= (
        counts["attempted"] == item["requested"]
        and not counts["failed"]
        and not counts["unfinished"]
        and usable > 0
    )
    mirrors = dict(unit.get("mirrors") or {})
    twin = mirrors.get(item["case_id"])
    return {
        **item,
        **counts,
        "usable": usable,
        "certifies_by_equivariance": twin,
        # Counted only where it means something: a cell that carries a twin.
        "swap_exact": totals.counters.get("swap_exact", 0) if twin else None,
        "worker_exit_code": worker_code,
        "status": "diagnostic_completed" if complete else "incomplete",
        "historical_report": f"{item['directory']}/calibration.json"
        if (directory / "calibration.json").exists()
        else None,
    }


def _equivariance_record(item, source):
    """Certify a collapsed twin, or say exactly why its source cannot carry it."""
    reason = None
    if source is None or source["status"] != "diagnostic_completed":
        reason = "certifying_case_incomplete"
    elif source["certifies_by_equivariance"] != item["case_id"]:
        reason = "certifying_case_did_not_track_this_twin"
    elif source["requested"] != item["requested"]:
        reason = "certifying_case_ran_a_different_repetition_count"
    elif source["swap_exact"] != source["attempted"]:
        reason = "certifying_case_has_draws_whose_swap_image_is_unrepresentable"
    return {
        **item,
        "status": "incomplete" if reason else "certified_by_equivariance",
        "executed": False,
        "uncertified_reason": reason,
        "certifying_swap_exact_draws": None if source is None else source["swap_exact"],
        "certifying_attempted_draws": None if source is None else source["attempted"],
        "historical_report": None if source is None else source["historical_report"],
    }


def _summarize(output, selection, provenance, controller_code):
    units = {unit["unit"]: unit for unit in selection["units"]}
    executed = {}
    for item in selection["cases"]:
        if item["execution"] == "executed":
            executed[item["case_id"]] = _executed_record(output, item, units[item["unit"]])
    records = []
    for item in selection["cases"]:
        if item["execution"] == "executed":
            records.append(executed[item["case_id"]])
            continue
        records.append(_equivariance_record(item, executed.get(item["certified_by"]["case_id"])))
    final_provenance = _provenance()
    unchanged = final_provenance == provenance
    accepted = ("diagnostic_completed", "certified_by_equivariance")
    complete = (
        controller_code == 0 and unchanged and all(row["status"] in accepted for row in records)
    )
    certified_by_equivariance = sorted(
        row["case_id"] for row in records if row["status"] == "certified_by_equivariance"
    )
    summary = {
        "status": "diagnostic_completed" if complete else "incomplete",
        "certified": False,
        "claim_scope": "experimental model-conditioned evidence only",
        "original_case_count": selection["original_case_count"],
        "selected_case_count": len(records),
        "executed_case_count": sum(1 for row in records if row["execution"] == "executed"),
        "equivariance_certified_case_count": len(certified_by_equivariance),
        "simulation_unit_count": len(selection["units"]),
        "collapse": {
            "allocation_swap": {
                "argument": SWAP_ARGUMENT,
                "derivation": SWAP_ARGUMENT_TEXT,
                "case_ids": certified_by_equivariance,
            },
            "shared_bootstrap_draw": {
                "units": sum(1 for unit in selection["units"] if unit["shares_bootstrap_draw"]),
                "seed_reassignment": SEED_REASSIGNMENT_TEXT,
                "reassigned_case_ids": sorted(
                    case_id
                    for unit in selection["units"]
                    for case_id in unit["case_ids"][1:]
                    if unit["shares_bootstrap_draw"]
                ),
            },
        },
        "controller_exit_code": controller_code,
        "source_unchanged": unchanged,
        "records": records,
        "provenance": {"start": provenance, "end": final_provenance},
    }
    _publish(output / "summary.json", summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("records", "provenance", "collapse")
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 2


def _swapped_reference(reference):
    """Role-swap image of an A/A cell's independent reference."""
    return {
        **reference,
        "pool_weights": list(reversed(reference["pool_weights"])),
        "winsorized_means": list(reversed(reference["winsorized_means"])),
    }


def _is_swap_image(source, twin) -> bool:
    """Whether *twin* is exactly *source* with the two arm roles exchanged."""
    a, b = source["dgp"], twin["dgp"]
    return (
        source.get("kind") == twin.get("kind") == "aa"
        and a["control"] == a["treatment"] == b["control"] == b["treatment"]
        and (a["n_c"], a["n_t"]) == (b["n_t"], b["n_c"])
        and (a["quantile"], a["quantile_method"]) == (b["quantile"], b["quantile_method"])
        and source["family"] == twin["family"]
        # The swap negates the estimand, and A/A truth is its own negation.
        and source["truth"] == twin["truth"] == 0.0
        and source["design"] == twin["design"]
        and source["nonvacuity"] == twin["nonvacuity"]
        and source["availability"] == twin["availability"]
        and _swapped_reference(source["reference"]) == twin["reference"]
    )


def _draw_sharing_key(case):
    """Cells that differ only in the clipping quantile, or ``None`` if unique.

    The frozen replicate draw is a function of the per-arm log centers, the
    bandwidth and the seeded streams; ``raw.quantile`` first enters when a
    replicate is reduced to statistics.
    """
    if case.get("kind") not in ("aa", "alternative"):
        return None
    dgp = case["dgp"]
    return (
        case["kind"],
        dgp["baseline_n"],
        tuple(dgp["allocation"]),
        json.dumps(dgp["control"], sort_keys=True),
        json.dumps(dgp["treatment"], sort_keys=True),
    )


def _allocation_swap_partners(cases):
    """Map each collapsed A/A cell to the cell whose simulation certifies it.

    The cell with the smaller control arm executes; its twin is certified by
    the proved role-swap equivariance. A candidate that is not an exact swap
    image is a broken premise, not a cell to quietly execute twice.
    """

    def shape(dgp, *, swapped=False):
        sizes = (dgp["n_t"], dgp["n_c"]) if swapped else (dgp["n_c"], dgp["n_t"])
        return (*sizes, dgp["quantile"], json.dumps(dgp["control"], sort_keys=True))

    by_shape = {shape(case["dgp"]): case for case in cases if case.get("kind") == "aa"}
    partners = {}
    for case in cases:
        if case.get("kind") != "aa" or case["dgp"]["n_c"] <= case["dgp"]["n_t"]:
            continue
        source = by_shape.get(shape(case["dgp"], swapped=True))
        if source is None:
            continue
        if not _is_swap_image(source, case):
            raise ValueError(
                f"{case['case_id']} shares an arm-size swap with {source['case_id']} "
                "but is not its exact swap image"
            )
        partners[case["case_id"]] = source["case_id"]
    return partners


def _selection(args, manifest):
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    ids = args.case_ids
    if ids is None:
        ids = (
            list(by_id)
            if args.selection == "full"
            else [
                "normal-seed101",
                "ate-unequal-50-200",
                "binomial-seed707",
                "winsor-original-seed808",
                "winsor-aa-contamination-n50-1x1-p0.99",
                "winsor-aa-ln-s2.0-n50-1x4-p0.99",
                "winsor-alt-n2000-4x1-p0.99",
            ]
        )
    if len(ids) != len(set(ids)) or any(case_id not in by_id for case_id in ids):
        raise ValueError("case IDs must be unique entries of the preserved manifest")
    selected = [by_id[case_id] for case_id in ids]
    requests = {}
    for case_id in ids:
        original = int(by_id[case_id]["design"]["repetitions"])
        limit = args.max_draws if args.max_draws is not None else 3
        requested = (
            original
            if args.selection == "full" and not args.diagnostic_override
            else min(original, limit)
        )
        requests[case_id] = (requested, original)
    partners = {
        twin: source
        for twin, source in _allocation_swap_partners(selected).items()
        if source in requests and requests[source] == requests[twin]
    }
    groups: list[list[dict]] = []
    position: dict[Any, int] = {}
    for case in selected:
        if case["case_id"] in partners:
            continue
        key = _draw_sharing_key(case)
        if key is None:
            groups.append([case])
            continue
        if key not in position:
            position[key] = len(groups)
            groups.append([])
        groups[position[key]].append(case)
    units = []
    home = {}
    for number, group in enumerate(groups):
        case_ids = [case["case_id"] for case in group]
        counts = {requests[case_id][0] for case_id in case_ids}
        if len(counts) != 1:
            raise ValueError("cells sharing one draw stream must share their requested count")
        (requested,) = counts
        directory = f"unit-{number:04d}"
        units.append(
            {
                "unit": number,
                "directory": directory,
                "case_ids": case_ids,
                "requested": requested,
                "seed": group[0]["seed"],
                "seed_source_case_id": case_ids[0],
                "shares_bootstrap_draw": len(case_ids) > 1,
                "mirrors": {
                    source: twin for twin, source in partners.items() if source in set(case_ids)
                },
            }
        )
        for case_id in case_ids:
            home[case_id] = (number, directory)
    cases = []
    for case_id in ids:
        requested, original = requests[case_id]
        source = partners.get(case_id)
        number, directory = home[case_id if source is None else source]
        cases.append(
            {
                "case_id": case_id,
                "unit": number,
                "directory": None if source else f"{directory}/{case_id}",
                "requested": requested,
                "original_requested": original,
                "truncated_diagnostic": requested < original,
                "execution": "certified_by_equivariance" if source else "executed",
                "shares_bootstrap_draw_with": []
                if source
                else [other for other in _unit_case_ids(units[number]) if other != case_id],
                "certified_by": {
                    "case_id": source,
                    "argument": SWAP_ARGUMENT,
                    "evidence": f"{directory}/{source}/calibration.json",
                }
                if source
                else None,
            }
        )
    return {
        "selection": args.selection,
        "original_case_count": len(by_id),
        "units": units,
        "cases": cases,
    }


def main(argv=None):
    from scripts.run_test_tier import run_with_budget
    from tests._i15_design import load_manifest

    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("representative", "full"), default="representative")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--diagnostic-override", action="store_true")
    parser.add_argument("--max-draws", type=int)
    parser.add_argument("--overall-seconds", type=float, default=3600)
    parser.add_argument("--case-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(x) or not 0 < x <= 3600 for x in (args.overall_seconds, args.case_seconds)
    ):
        parser.error("budgets must be finite and in (0, 3600]")
    if args.max_draws is not None and args.max_draws <= 0:
        parser.error("max-draws must be positive")
    if args.selection == "full" and args.max_draws is not None and not args.diagnostic_override:
        parser.error(
            "full requests retain original repetitions; truncation requires --diagnostic-override"
        )
    args.output = args.output.resolve()
    os.chdir(ROOT)
    manifest = load_manifest()
    try:
        selection = _selection(args, manifest)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = _provenance()
    _publish(args.output / "manifest.json", manifest)
    _publish(
        args.output / "selection.json",
        {**selection, "overall_seconds": args.overall_seconds, "case_seconds": args.case_seconds},
    )
    _publish(args.output / "provenance.json", provenance)
    remaining = args.overall_seconds - (time.monotonic() - started)
    code = 124
    if remaining > 0:
        code = run_with_budget(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; from scripts.run_i15_campaign import _controller; "
                "raise SystemExit(_controller(Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])))",
                str(args.output),
                str(args.case_seconds),
                str(remaining),
            ],
            tier="i15-research-campaign",
            budget_seconds=remaining,
        )
    return _summarize(args.output, selection, provenance, code)


if __name__ == "__main__":
    raise SystemExit(main())
