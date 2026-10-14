"""Create advisory runtime summaries from pytest evidence receipts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_METADATA_FIELDS = (
    "python",
    "platform",
    "resolved_worker_count",
    "dist",
    "thread_limits",
)
_PHASES = ("setup", "call", "teardown")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_run(evidence_root: Path) -> Path:
    if (evidence_root / "run.json").is_file():
        return evidence_root
    runs = [path for path in evidence_root.glob("run-*") if (path / "run.json").is_file()]
    if not runs:
        raise ValueError(f"no run.json found under {evidence_root}")
    return max(runs, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _duration(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration must be a finite non-negative number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("duration must be a finite non-negative number") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError("duration must be a finite non-negative number")
    return result


def _read_run(
    evidence_root: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[str]]:
    run = _select_run(evidence_root)
    try:
        metadata = _load_json(run / "run.json")
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {run / 'run.json'}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{run / 'run.json'} must contain a JSON object")
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    journal = run / "reports.jsonl"
    if not journal.is_file():
        warnings.append("reports.jsonl is missing")
    else:
        for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as error:
                warnings.append(f"malformed journal record at line {line_number}: {error}")
                continue
            if not isinstance(event, dict):
                warnings.append(
                    f"malformed journal record at line {line_number}: expected an object"
                )
                continue
            event_kind = event.get("event")
            if not isinstance(event_kind, str):
                warnings.append(
                    f"malformed journal record at line {line_number}: event kind must be a string"
                )
                continue
            if event_kind in {"report", "start"}:
                if not isinstance(event.get("nodeid"), str) or not event["nodeid"]:
                    warnings.append(
                        f"malformed {event_kind} record at line {line_number}: nodeid missing"
                    )
                events.append(event)
            else:
                warnings.append(f"unrecognized journal event at line {line_number}: {event_kind!r}")
    terminal_nodeids = {
        event["nodeid"]
        for event in events
        if event.get("event") == "report"
        and isinstance(event.get("nodeid"), str)
        and (
            event.get("when") == "call"
            or event.get("when") == "setup"
            and event.get("outcome") in ("failed", "skipped")
        )
    }
    reported_nodeids = {
        event["nodeid"]
        for event in events
        if event.get("event") == "report"
        and isinstance(event.get("nodeid"), str)
        and event["nodeid"]
    }
    executed_count = metadata.get("executed_tests")
    if "executed_tests" not in metadata or executed_count is None:
        warnings.append(
            "executed_tests inventory is unavailable; whole-run completeness is unknown"
        )
    elif (
        isinstance(executed_count, bool)
        or not isinstance(executed_count, int)
        or executed_count < 0
    ):
        warnings.append("executed_tests inventory is malformed")
    elif executed_count != len(terminal_nodeids):
        warnings.append(
            f"executed_tests declares {executed_count} cases but the journal has "
            f"{len(terminal_nodeids)} terminal case identities"
        )
    if "selected_nodeids" in metadata:
        selected = metadata["selected_nodeids"]
        if not isinstance(selected, list) or any(
            not isinstance(nodeid, str) or not nodeid for nodeid in selected
        ):
            warnings.append("selected_nodeids inventory is malformed")
        elif len(selected) != len(set(selected)):
            warnings.append("selected_nodeids inventory contains duplicate identities")
        elif selected:
            selected_ids = set(selected)
            if selected_ids != reported_nodeids:
                warnings.append(
                    f"selected_nodeids and journal case identities differ "
                    f"({len(selected_ids)} selected, {len(reported_nodeids)} reported)"
                )
            if selected_ids != terminal_nodeids:
                warnings.append(
                    f"selected_nodeids and journal terminal identities differ "
                    f"({len(selected_ids)} selected, {len(terminal_nodeids)} terminal)"
                )
    return run, metadata, events, warnings


def _case_summary(
    case: dict[str, Any], threshold: float, priority_threshold: float
) -> dict[str, Any]:
    marks = sorted(case["markers"])
    phases = {phase: case["phases"][phase] for phase in _PHASES if phase in case["phases"]}
    missing_phases = [phase for phase in _PHASES if phase not in phases]
    if missing_phases:
        case["complete"] = False
        case["warnings"].append("missing phases: " + ", ".join(missing_phases))
    if case["outcomes"] and case["outcomes"] != {"passed"}:
        case["complete"] = False
        case["warnings"].append("case did not have exclusively passed phase outcomes")
    if case["started"] and not phases:
        case["complete"] = False
        case["warnings"].append("test started but has no phase reports")
    complete = case["complete"]
    total = sum(phases.values()) if complete else None
    return {
        "nodeid": case["nodeid"],
        "canonical_nodeid": case["canonical_nodeid"],
        "phases": phases,
        "total_seconds": total,
        "markers": marks,
        "complete": complete,
        "new": None,
        "newly_observed": False,
        "regression": None,
        "suggestion": (
            "slow"
            if complete and total is not None and total > threshold and "slow" not in marks
            else None
        ),
        "priority": bool(complete and total is not None and total > priority_threshold),
        "warnings": sorted(set(case["warnings"])),
    }


def _cases(
    events: list[dict[str, Any]], threshold: float, priority_threshold: float
) -> list[dict[str, Any]]:
    canonical_by_raw = {
        event["nodeid"]: event["canonical_nodeid"]
        for event in events
        if event.get("event") == "report"
        and isinstance(event.get("nodeid"), str)
        and isinstance(event.get("canonical_nodeid"), str)
        and event["canonical_nodeid"]
    }
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        raw_nodeid = event.get("nodeid")
        if not isinstance(raw_nodeid, str) or not raw_nodeid:
            continue
        is_start = event.get("event") == "start"
        canonical = (
            canonical_by_raw.get(raw_nodeid, raw_nodeid)
            if is_start
            else event.get("canonical_nodeid")
        )
        identity_complete = isinstance(canonical, str) and bool(canonical) and not is_start
        if not isinstance(canonical, str) or not canonical:
            canonical = raw_nodeid
        case = grouped.setdefault(
            canonical,
            {
                "nodeid": raw_nodeid,
                "canonical_nodeid": canonical,
                "markers": set(),
                "phases": {},
                "warnings": [],
                "complete": True,
                "outcomes": set(),
                "started": False,
            },
        )
        if raw_nodeid != case["nodeid"]:
            case["complete"] = False
            case["warnings"].append("inconsistent nodeid for canonical identity")
        if is_start:
            case["started"] = True
            continue
        if not identity_complete:
            case["complete"] = False
            case["warnings"].append("canonical identity missing; using nodeid fallback")
        markers = event.get("markers")
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers):
            case["complete"] = False
            case["warnings"].append("effective markers missing or malformed")
        else:
            case["markers"].update(markers)
        outcome = event.get("outcome")
        if isinstance(outcome, str):
            case["outcomes"].add(outcome)
        else:
            case["complete"] = False
            case["warnings"].append("outcome missing or malformed")
        phase = event.get("when")
        if phase not in _PHASES:
            case["complete"] = False
            case["warnings"].append(f"unknown phase {phase!r}")
            continue
        try:
            seconds = _duration(event.get("duration"))
        except ValueError as error:
            case["complete"] = False
            case["warnings"].append(f"invalid {phase} duration: {error}")
            continue
        previous = case["phases"].get(phase)
        if previous is None:
            case["phases"][phase] = seconds
        elif previous != seconds:
            case["complete"] = False
            case["warnings"].append(f"conflicting repeated {phase} events; counted once")
            case["phases"][phase] = max(previous, seconds)
    results = [_case_summary(case, threshold, priority_threshold) for case in grouped.values()]
    return sorted(results, key=lambda case: case["canonical_nodeid"])


def _metadata_unknown(metadata: Mapping[str, Any]) -> list[str]:
    unknown = [
        field for field in _METADATA_FIELDS if field not in metadata or metadata[field] is None
    ]
    if not isinstance(metadata.get("python"), str) or not metadata["python"]:
        unknown.append("python")
    if not isinstance(metadata.get("platform"), str) or not metadata["platform"]:
        unknown.append("platform")
    workers = metadata.get("resolved_worker_count")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        unknown.append("resolved_worker_count")
    if not isinstance(metadata.get("dist"), str) or not metadata["dist"]:
        unknown.append("dist")
    limits = metadata.get("thread_limits")
    if not isinstance(limits, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in limits.items()
    ):
        unknown.append("thread_limits")
    return sorted(set(unknown))


def _compatibility(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    mismatches = sorted(
        set(
            [field for field in _METADATA_FIELDS if current.get(field) != baseline.get(field)]
            + _metadata_unknown(current)
            + _metadata_unknown(baseline)
        )
    )
    return not mismatches, mismatches


def _baseline_report(
    cases: list[dict[str, Any]],
    baseline_root: Path,
    current_metadata: Mapping[str, Any],
    current_evidence_warnings: list[str],
) -> dict[str, Any]:
    try:
        _, baseline_metadata, events, baseline_evidence_warnings = _read_run(baseline_root)
    except (OSError, ValueError) as error:
        return {
            "comparable": False,
            "reason": str(error),
            "metadata": None,
            "inventory_complete": False,
        }
    comparable, mismatches = _compatibility(current_metadata, baseline_metadata)
    if baseline_metadata.get("status") != "finished" or baseline_metadata.get("exit_code") != 0:
        comparable = False
        mismatches.append("baseline run incomplete")
    if baseline_evidence_warnings:
        comparable = False
        mismatches.append("baseline evidence inventory/journal is incomplete or inconsistent")
    if current_evidence_warnings:
        comparable = False
        mismatches.append("current evidence inventory/journal is incomplete or inconsistent")
    baseline_cases = _cases(events, 5.0, 10.0)
    indexed = {case["canonical_nodeid"]: case for case in baseline_cases}
    selected = baseline_metadata.get("selected_nodeids")
    baseline_inventory_complete = (
        isinstance(selected, list)
        and bool(selected)
        and all(isinstance(nodeid, str) for nodeid in selected)
        and len(selected) == len(set(selected))
        and set(selected) == {case["nodeid"] for case in baseline_cases}
        and all(case["complete"] for case in baseline_cases)
    )
    for case in cases:
        old = indexed.get(case["canonical_nodeid"])
        if comparable and old is not None and case["complete"] and old["complete"]:
            case["new"] = False
            case["regression"] = case["total_seconds"] > old["total_seconds"]
        elif comparable and old is None and baseline_inventory_complete:
            case["new"] = None
            case["newly_observed"] = True
            case["warnings"].append(
                "absent from exact baseline selection; authored novelty is unknown"
            )
        else:
            case["new"] = None
            case["regression"] = None
            case["warnings"].append("baseline comparison cannot establish a new case or regression")
    return {
        "comparable": comparable,
        "reason": None
        if comparable
        else "incomparable metadata: " + ", ".join(sorted(set(mismatches))),
        "metadata": {field: baseline_metadata.get(field) for field in _METADATA_FIELDS},
        "inventory_complete": baseline_inventory_complete,
    }


def _git_test_inventory(base: str, cwd: Path) -> tuple[set[str], str]:
    """Return exact pytest function identities discoverable in tracked test sources."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    revision = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").strip()
    paths = git("ls-tree", "-r", "--name-only", revision, "--", "tests").splitlines()
    names: set[str] = set()
    for source_path in paths:
        if not source_path.endswith(".py"):
            continue
        try:
            source = git("show", f"{revision}:{source_path}")
            tree = ast.parse(source, filename=source_path)
        except (subprocess.CalledProcessError, SyntaxError, UnicodeError):
            continue

        def visit_body(
            body: list[ast.stmt],
            parents: tuple[str, ...] = (),
            path: str = source_path,
        ) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    visit_body(node.body, (*parents, node.name))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        names.add(f"{path}::{'::'.join((*parents, node.name))}")

        visit_body(tree.body)
    return names, revision


def _load_duration_baseline(path: Path) -> dict[str, float]:
    try:
        raw = _load_json(path)
        if not isinstance(raw, dict):
            raise ValueError("baseline must contain a JSON object")
        result = {}
        for nodeid, value in raw.items():
            if not isinstance(nodeid, str) or not nodeid:
                raise ValueError("baseline keys must be non-empty nodeids")
            try:
                result[nodeid] = _duration(value)
            except ValueError as error:
                raise ValueError(f"invalid duration for {nodeid!r}: {error}") from error
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(f"duration baseline: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _duration_guard(
    cases: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, float],
    *,
    run_complete: bool,
    python: str | None,
    ratio: float,
    floor: float,
    shard_ratio: float,
    shard_floor: float,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    compared_tests = 0
    unbaselined_count = 0
    unbaselined_seconds = 0.0
    incomparable_count = 0
    for case in cases:
        if not case["complete"]:
            continue
        current = case["total_seconds"]
        if current is None or not math.isfinite(current):
            continue
        if any(case["phases"].get(phase, 0) > 600 for phase in ("setup", "teardown")):
            incomparable_count += 1
            continue
        nodeid = case["nodeid"]
        if nodeid not in baseline:
            unbaselined_count += 1
            unbaselined_seconds += current
            continue
        previous = baseline[nodeid]
        compared_tests += 1
        filename = nodeid.split("::", 1)[0]
        file_data = files.setdefault(
            filename,
            {"file": filename, "baseline_seconds": 0.0, "current_seconds": 0.0, "top_tests": []},
        )
        file_data["baseline_seconds"] += previous
        file_data["current_seconds"] += current
        file_data["top_tests"].append(
            {
                "nodeid": nodeid,
                "baseline_seconds": previous,
                "current_seconds": current,
                "delta_seconds": current - previous,
            }
        )

    for data in files.values():
        data["top_tests"].sort(key=lambda test: test["delta_seconds"], reverse=True)
        data["top_tests"] = data["top_tests"][:3]
        delta = data["current_seconds"] - data["baseline_seconds"]
        data["regressed"] = (
            data["current_seconds"] > data["baseline_seconds"] * ratio and delta > floor
        )
        data["_delta"] = delta
    ordered_files = sorted(files.values(), key=lambda data: data["_delta"], reverse=True)
    regressed_files = [data["file"] for data in ordered_files if data["regressed"]]
    current_total = sum(data["current_seconds"] for data in files.values())
    shard_baseline = sum(data["baseline_seconds"] for data in files.values())
    shard_current = current_total
    shard_delta = shard_current - shard_baseline
    shard_regressed = shard_current > shard_baseline * shard_ratio and shard_delta > shard_floor
    reasons = []
    if not run_complete:
        reasons.append("run_incomplete")
    if compared_tests == 0:
        reasons.append("no_comparable_tests")
    denominator = compared_tests + unbaselined_count + incomparable_count
    if denominator and compared_tests < 0.5 * denominator:
        reasons.append("low_baseline_coverage")
    if unbaselined_seconds > 0.25 * (current_total + unbaselined_seconds):
        reasons.append("unbaselined_runtime_high")
    for data in ordered_files:
        data.pop("_delta")
    return {
        "status": "inconclusive" if reasons else "compared",
        "reasons": reasons,
        "baseline_path": None,
        "python": python,
        "thresholds": {
            "ratio": ratio,
            "floor_seconds": floor,
            "shard_ratio": shard_ratio,
            "shard_floor_seconds": shard_floor,
        },
        "compared_tests": compared_tests,
        "unbaselined": {"count": unbaselined_count, "seconds": unbaselined_seconds},
        "incomparable": {"count": incomparable_count},
        "shard_case_work": {
            "baseline_seconds": shard_baseline,
            "current_seconds": shard_current,
            "regressed": shard_regressed,
        },
        "files": ordered_files,
        "regressed_files": regressed_files,
    }


def _markdown_cell(identity: str) -> str:
    return identity.replace("|", r"\|")


def _duration_guard_markdown(guard: Mapping[str, Any]) -> list[str]:
    thresholds = guard["thresholds"]
    lines = [
        "### Runtime regression guard",
        "",
        f"- Status: **{guard['status']}**; reasons: {', '.join(guard['reasons']) or '—'}.",
        f"- Provisional thresholds: file ratio {thresholds['ratio']} and floor "
        f"{thresholds['floor_seconds']}s; shard ratio {thresholds['shard_ratio']} "
        f"and floor {thresholds['shard_floor_seconds']}s.",
        f"- Compared {guard['compared_tests']} tests; unbaselined "
        f"{guard['unbaselined']['count']} ({guard['unbaselined']['seconds']:.3f}s); "
        f"incomparable {guard['incomparable']['count']}.",
        "",
        "| Shard case work | Baseline seconds | Current seconds | Regressed |",
        "|---|---:|---:|---|",
        f"| Compared tests | {guard['shard_case_work']['baseline_seconds']:.3f} | "
        f"{guard['shard_case_work']['current_seconds']:.3f} | "
        f"{str(guard['shard_case_work']['regressed']).lower()} |",
        "",
        "| File | Baseline seconds | Current seconds | Regressed | Top tests (baseline → current, delta) |",
        "|---|---:|---:|---|---|",
    ]
    for item in guard["files"][:10]:
        top = "; ".join(
            f"`{_markdown_cell(test['nodeid'])}` ({test['baseline_seconds']:.3f} → "
            f"{test['current_seconds']:.3f}, {test['delta_seconds']:+.3f})"
            for test in item["top_tests"]
        )
        lines.append(
            f"| `{_markdown_cell(item['file'])}` | {item['baseline_seconds']:.3f} | "
            f"{item['current_seconds']:.3f} | {str(item['regressed']).lower()} | {top} |"
        )
    lines.extend(
        [
            "",
            "Record an intentional increase by refreshing `.github/fast-test-durations.json` "
            "in the same PR, with the reason stated (see CONTRIBUTING).",
            "",
        ]
    )
    return lines


def _markdown(report: Mapping[str, Any], json_path: str | None) -> str:
    cases = report["cases"]
    threshold = report["threshold_seconds"]
    relevant = [
        case
        for case in cases
        if not case["complete"]
        or case["priority"]
        or case["total_seconds"] is not None
        and case["total_seconds"] > threshold
        or case["regression"] is True
        or case["newly_observed"]
    ]
    relevant.sort(
        key=lambda case: (
            not case["priority"],
            case["total_seconds"] is None,
            -(case["total_seconds"] or 0),
            case["canonical_nodeid"],
        )
    )
    row_limit = 100
    rows = relevant[:row_limit]
    lines = [
        "# Test runtime report",
        "",
        f"- Run: `{report['run']['path']}` ({report['run']['status']})",
        f"- Complete: `{str(report['complete']).lower()}`",
        f"- {len(cases)} cases total; {sum(case['complete'] for case in cases)} complete; "
        f"{sum(not case['complete'] for case in cases)} incomplete; "
        f"{sum(case['total_seconds'] is not None and case['total_seconds'] > threshold for case in cases)} over threshold; "
        f"{sum(case['priority'] for case in cases)} priority.",
        f"- Showing {len(rows)} relevant of {len(cases)} cases; full JSON: `{json_path}`"
        if json_path
        else f"- Showing {len(rows)} relevant of {len(cases)} cases; full JSON not written (pass `--json PATH`).",
        f"- Advisory thresholds: slow > {threshold}s; priority > {report['priority_threshold_seconds']}s.",
        "",
        "| Test | Setup | Call | Teardown | Total | Markers | Suggestion | New | Newly observed | Regression | Priority | Complete |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---:|---|",
    ]
    for case in rows:
        phase_values = [case["phases"].get(name, "—") for name in _PHASES]
        lines.append(
            f"| `{_markdown_cell(case['canonical_nodeid'])}` | {phase_values[0]} | {phase_values[1]} | {phase_values[2]} | "
            f"{case['total_seconds'] if case['total_seconds'] is not None else 'unknown'} | "
            f"{', '.join(case['markers']) or '—'} | {case['suggestion'] or '—'} | "
            f"{case['new'] if case['new'] is not None else 'unknown'} | "
            f"{str(case['newly_observed']).lower()} | "
            f"{case['regression'] if case['regression'] is not None else 'unknown'} | "
            f"{str(case['priority']).lower()} | {str(case['complete']).lower()} |"
        )
    lines.append("")
    if report["duration_guard"] is not None:
        lines.extend(_duration_guard_markdown(report["duration_guard"]))
    for warning in report["warnings"][:20]:
        lines.append(f"> **Warning:** {warning}")
    if len(report["warnings"]) > 20:
        lines.append(
            f"> {len(report['warnings']) - 20} additional report warnings are in the full JSON."
        )
    for case in rows:
        for warning in case["warnings"][:5]:
            lines.append(f"> **Warning:** `{case['canonical_nodeid']}`: {warning}")
    if len(relevant) > row_limit:
        lines.append(
            f"> Showing top {row_limit} relevant cases; {len(relevant) - row_limit} additional cases are in the full JSON."
        )
    return "\n".join(lines) + "\n"


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Advisory slow threshold in seconds (default: 5).",
    )
    parser.add_argument(
        "--priority-threshold",
        type=float,
        default=10.0,
        help="Advisory priority threshold in seconds (default: 10).",
    )
    parser.add_argument("--base", help="Optional git revision for AST test-family inventory.")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline evidence root for comparable runtime deltas.",
    )
    parser.add_argument("--json", metavar="PATH", help="Write the complete JSON report to PATH.")
    parser.add_argument("--markdown", metavar="PATH", help="Write the Markdown report to PATH.")
    parser.add_argument(
        "--duration-baseline",
        type=Path,
        help="Optional nodeid-to-duration baseline for advisory runtime regression checks.",
    )
    parser.add_argument("--regression-ratio", type=float, default=1.5)
    parser.add_argument("--regression-floor-seconds", type=float, default=2.0)
    parser.add_argument("--shard-regression-ratio", type=float, default=1.25)
    parser.add_argument("--shard-regression-floor-seconds", type=float, default=30.0)
    return parser


def _github_warning(message: str) -> None:
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::warning::{escaped}")


def _emit_duration_guard_annotations(guard: Mapping[str, Any]) -> None:
    shard = guard["shard_case_work"]
    shard_summary = (
        f"; shard case work regressed from {shard['baseline_seconds']:.3f}s "
        f"to {shard['current_seconds']:.3f}s"
        if shard["regressed"]
        else ""
    )
    if guard["status"] == "inconclusive":
        summary = (
            f"Runtime regression comparison inconclusive ({', '.join(guard['reasons'])}); "
            f"{len(guard['regressed_files'])} regressed files; "
            f"{guard['compared_tests']} compared, "
            f"{guard['unbaselined']['count']} unbaselined, "
            f"{guard['incomparable']['count']} incomparable; refresh duration weights"
        )
    else:
        summary = (
            f"Runtime regression summary: {len(guard['regressed_files'])} regressed files; "
            f"{guard['compared_tests']} compared tests"
        )
    _github_warning(summary + shard_summary)
    for item in [file for file in guard["files"] if file["regressed"]][:5]:
        _github_warning(
            f"Runtime regression for {item['file']}: "
            f"{item['baseline_seconds']:.3f}s to {item['current_seconds']:.3f}s"
        )


def _duration_guard_report(
    path: Path,
    cases: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    run_complete: bool,
    ratio: float,
    floor: float,
    shard_ratio: float,
    shard_floor: float,
) -> dict[str, Any]:
    guard = _duration_guard(
        cases,
        _load_duration_baseline(path),
        run_complete=run_complete,
        python=metadata.get("python"),
        ratio=ratio,
        floor=floor,
        shard_ratio=shard_ratio,
        shard_floor=shard_floor,
    )
    guard["baseline_path"] = str(path)
    return guard


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.threshold < 0 or not math.isfinite(args.threshold):
        print("threshold must be finite and non-negative", file=sys.stderr)
        return 2
    if args.priority_threshold < 0 or not math.isfinite(args.priority_threshold):
        print("priority threshold must be finite and non-negative", file=sys.stderr)
        return 2
    duration_thresholds = (
        args.regression_ratio,
        args.regression_floor_seconds,
        args.shard_regression_ratio,
        args.shard_regression_floor_seconds,
    )
    if any(not math.isfinite(value) or value < 0 for value in duration_thresholds):
        print("duration regression thresholds must be finite and non-negative", file=sys.stderr)
        return 2
    try:
        run, metadata, events, journal_warnings = _read_run(args.evidence_root)
    except (OSError, ValueError) as error:
        print(f"runtime report: {error}", file=sys.stderr)
        return 2
    cases = _cases(events, args.threshold, args.priority_threshold)
    missing = _metadata_unknown(metadata)
    complete = (
        metadata.get("status") == "finished"
        and metadata.get("exit_code") == 0
        and not missing
        and not journal_warnings
        and bool(cases)
        and all(case["complete"] for case in cases)
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "path": str(run),
            "status": metadata.get("status"),
            "exit_code": metadata.get("exit_code"),
        },
        "complete": complete,
        "metadata": {field: metadata.get(field) for field in _METADATA_FIELDS},
        "threshold_seconds": args.threshold,
        "priority_threshold_seconds": args.priority_threshold,
        "cases": cases,
        "warnings": list(journal_warnings),
        "baseline": None,
        "base": None,
        "outputs": {"json": args.json, "markdown": args.markdown},
        "duration_guard": None,
    }
    if missing:
        report["warnings"].append("missing or unknown run metadata: " + ", ".join(missing))
    if metadata.get("status") != "finished" or metadata.get("exit_code") != 0:
        report["warnings"].append("source run is incomplete or did not finish successfully")
    if not cases:
        report["warnings"].append("no usable report events found")
    if args.duration_baseline:
        report["duration_guard"] = _duration_guard_report(
            args.duration_baseline,
            cases,
            metadata,
            run_complete=complete,
            ratio=args.regression_ratio,
            floor=args.regression_floor_seconds,
            shard_ratio=args.shard_regression_ratio,
            shard_floor=args.shard_regression_floor_seconds,
        )
    if args.baseline:
        report["baseline"] = _baseline_report(cases, args.baseline, metadata, journal_warnings)
        if not report["baseline"]["comparable"]:
            report["warnings"].append("baseline incomparable: " + report["baseline"]["reason"])
        elif not report["baseline"]["inventory_complete"]:
            report["warnings"].append(
                "baseline inventory is incomplete or unknown; authored novelty cannot be established"
            )
    if args.base:
        try:
            inventory, revision = _git_test_inventory(args.base, Path.cwd())
            family_matches = []
            for case in cases:
                family = case["canonical_nodeid"].partition("[")[0]
                present = family in inventory
                case["base_family_present"] = True if present else None
                family_matches.append(
                    {
                        "nodeid": case["canonical_nodeid"],
                        "family": family,
                        "present": True if present else None,
                        "absence_meaning": None if present else "AST miss does not prove absence",
                    }
                )
            report["base"] = {
                "revision": revision,
                "test_inventory_count": len(inventory),
                "family_comparison": family_matches,
                "comparison": "AST exact test-family membership only; misses are unknown and parameter variants are not inferred",
            }
        except (OSError, subprocess.CalledProcessError) as error:
            report["base"] = {"comparable": False, "reason": f"git inventory unavailable: {error}"}
            report["warnings"].append(report["base"]["reason"])
    if report["duration_guard"] is not None:
        _emit_duration_guard_annotations(report["duration_guard"])
    annotation_messages = list(report["warnings"])
    for case in cases:
        annotation_messages.extend(
            f"{case['canonical_nodeid']}: {warning}" for warning in case["warnings"]
        )
        if (
            case["complete"]
            and case["total_seconds"] is not None
            and case["total_seconds"] > args.threshold
        ):
            marker_state = (
                "already marked slow" if "slow" in case["markers"] else "missing slow marker"
            )
            observation = "newly observed; " if case["newly_observed"] else ""
            annotation_messages.append(
                f"Runtime {case['total_seconds']:.3f}s exceeds advisory threshold "
                f"({observation}{marker_state}) for {case['canonical_nodeid']}"
            )
    annotation_limit = 25
    for warning in annotation_messages[:annotation_limit]:
        _github_warning(warning)
    if len(annotation_messages) > annotation_limit:
        _github_warning(
            f"{len(annotation_messages) - annotation_limit} additional runtime warnings omitted; see full JSON"
        )
    json_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    markdown_text = _markdown(report, args.json)
    _write(args.json, json_text)
    _write(args.markdown, markdown_text)
    if not args.json and not args.markdown:
        sys.stdout.write(markdown_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
