"""Join the fraction-harmed study's aggregate evidence without reading trial inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product, zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from calibration import harm_campaign as campaign
from calibration import harm_case as case
from calibration.harm_bounds import StudyError, decide

if TYPE_CHECKING:
    from collections.abc import Iterable

    from calibration.harm_bounds import Interval

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (*campaign._ORACLE_SOURCES, "calibration/harm_campaign.py")
_PIPELINES = ("P0", "P1", "P2", "P3")
_ACTIONS = ("action_if_clear", "action_if_rule_out", "action_if_inconclusive")
_KIND = ("clear", "rule_out", "abstain")
_FIELD_FILES = ("input_manifest.json", "bounds.json", "thresholds.csv")
_EXIT = {"green": 0, "red": 1, "inconclusive": 2, "smoke": 0, "execution_failure": 1}
_METHODS = {"P0": "U0", "P1": "U0 ∩ F", "P2": "U0 ∩ F ∩ Q", "P3": "U0 ∩ F ∩ Q ∩ H"}
_LIMITS = (
    "The target is a population fraction losing a beneficial binary outcome, not individual harm.",
    "The simultaneous sampling guarantee is at least 95% in real arithmetic for the frozen "
    "family under independent sampling, constant two-arm assignment and complete fixed-window "
    "outcomes; the synthetic study does not establish transport, permissions or field utility.",
)

_RunStatus = campaign._RunStatus
type _Method = Literal["P0", "P1", "P2", "P3"]
type _Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _WireReceipts(case._Receipts):
    inputs: _Sha = Field(alias="input_manifest.json")
    bounds: _Sha = Field(alias="bounds.json")
    thresholds: _Sha = Field(alias="thresholds.csv")


class _WireDecision(case._FieldDecision):
    receipts: _WireReceipts


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class _IntendedAction(_Frozen):
    pipeline: _Method
    action_key: Literal["action_if_clear", "action_if_rule_out"]
    feasible_and_intended: bool


class _Acknowledgment(_Frozen):
    schema_version: Literal[1]
    decision_sha256: _Sha
    card_sha256: _Sha
    acknowledged_at: AwareDatetime
    intended_action: _IntendedAction | None
    all_admissible_baseline_features: bool
    comparison_adequate: bool


@dataclass(frozen=True, slots=True)
class _Recommendation:
    recommendation: Literal["implement", "defer", "reject"]
    method: _Method | None
    reasons: tuple[str, ...]


def _problem(reason: str, coordinate: str) -> StudyError:
    return StudyError(reason, (("stage", "recommendation"), ("coordinate", coordinate)))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, coordinate: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise _problem("missing_or_invalid_evidence_file", coordinate)
    try:
        return path.read_bytes()
    except OSError:
        raise _problem("evidence_read_error", coordinate) from None


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    members = dict(pairs)
    if len(members) != len(pairs):
        raise ValueError("duplicate key")
    return members


def _decode[T: BaseModel](model: type[T], raw: bytes, coordinate: str) -> T:
    try:
        items = json.loads(raw, object_pairs_hook=_unique)
        keys = {field.serialization_alias or name for name, field in model.model_fields.items()}
        if not isinstance(items, dict) or items.keys() != keys:
            raise ValueError("wrong fields")
        return model.model_validate_json(raw)
    except (ValueError, RecursionError):
        raise _problem("invalid_evidence_schema", coordinate) from None


def _checked_bytes(path: Path, expected: str) -> bytes:
    raw = _read(path, path.name)
    if _sha(raw) != expected:
        raise _problem("artifact_digest_mismatch", path.name)
    return raw


def _oracle_path(directory: Path, receipt: campaign._Receipt, override: Path | None) -> Path:
    if override is not None:
        return override
    # Don't follow an arbitrary path supplied by a manifest. The recorded path is
    # usable only for a local aggregate receipt; moved packets can pass --oracle.
    recorded = Path(receipt.path)
    if (
        recorded.is_absolute()
        and recorded.name == "oracle.json"
        and recorded.resolve().is_relative_to((_ROOT / ".test-evidence").resolve())
        and recorded.is_file()
    ):
        return recorded
    return directory.parent / "oracle.json"


def _check_oracle(directory: Path, manifest: campaign._Manifest, oracle: Path | None) -> None:
    if len(manifest.inputs) != 2 or manifest.oracle_sha256 is None:
        raise _problem("missing_oracle_receipt", "oracle.json")
    protocol, receipt = manifest.inputs
    if (
        protocol.path != campaign._PROTOCOL
        or protocol.sha256 != manifest.protocol_sha256
        or receipt.sha256 != manifest.oracle_sha256
    ):
        raise _problem("inconsistent_oracle_receipt", "oracle.json")
    oracle_path = _oracle_path(directory, receipt, oracle)
    if oracle_path.is_symlink() or not oracle_path.is_file():
        raise _problem("missing_or_invalid_evidence_file", "oracle.json")
    try:
        actual = campaign._verify_oracle(oracle_path, protocol, manifest.sources)
    except StudyError as exc:
        raise _problem(exc.reason, "oracle.json") from None
    if actual.sha256 != receipt.sha256:
        raise _problem("artifact_digest_mismatch", "oracle.json")


def _counter_roster() -> Iterable[tuple[campaign._Coordinate, str, campaign._GateKind]]:
    for coord in campaign.coordinates():
        for name in campaign._ERRORS:
            yield coord, name, "error"
        if coord.n_index == 1 and coord.e_index == 0 and 2 <= coord.dgp <= 5:
            yield coord, campaign._POWER[coord.dgp - 2], "power"


def _check_gates(manifest: campaign._Manifest, gates: campaign._Gates) -> None:
    status = campaign._status(manifest.mode, list(manifest.failures), list(gates.counters))
    complete = status != "execution_failure"
    eligible = complete and manifest.mode == "full"
    attempts = campaign._FULL_TRIALS if manifest.mode == "full" else campaign._SMOKE_TRIALS
    if (
        manifest.status != status
        or gates.status != status
        or manifest.certification_eligible != eligible
        or gates.certification_eligible != eligible
        or manifest.exit_code != _EXIT[status]
        or manifest.trials_per_cell != attempts
        or (complete and manifest.completed_cells != len(campaign.coordinates()))
    ):
        raise _problem("inconsistent_campaign_summary", "gates.json")
    expected = tuple(_counter_roster())
    actual = tuple((row.coordinate, row.name, row.kind) for row in gates.counters)
    if complete and actual != expected:
        raise _problem("invalid_campaign_counter_roster", "gates.json")
    if len(set(actual)) != len(actual) or any(key not in expected for key in actual):
        raise _problem("invalid_campaign_counter_roster", "gates.json")
    for row in gates.counters:
        if row.attempted != attempts or not 0 <= row.count <= row.attempted:
            raise _problem("invalid_campaign_counter", "gates.json")
        interval, state = campaign.counter_gate(row.count, row.attempted, row.kind)
        if row.status != state or any(
            not math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-15)
            for got, want in zip(row.interval, interval, strict=True)
        ):
            raise _problem("invalid_campaign_counter", "gates.json")


def _stream_keys[T: BaseModel](
    path: Path,
    expected: str,
    model: type[T],
    roster: Iterable[object] | None,
    key: _StreamKey[T],
) -> None:
    if path.is_symlink() or not path.is_file():
        raise _problem("missing_or_invalid_evidence_file", path.name)
    try:
        with path.open("rb") as stream:
            if roster is None:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            else:
                hasher = hashlib.sha256()
                for raw, item in zip_longest(stream, roster):
                    if raw is None or item is None or key(model.model_validate_json(raw)) != item:
                        raise _problem("invalid_campaign_result_roster", path.name)
                    hasher.update(raw)
                digest = hasher.hexdigest()
    except ValidationError:
        raise _problem("invalid_evidence_schema", path.name) from None
    except OSError:
        raise _problem("evidence_read_error", path.name) from None
    if digest != expected:
        raise _problem("artifact_digest_mismatch", path.name)


if TYPE_CHECKING:
    from collections.abc import Callable

    type _StreamKey[T] = Callable[[T], object]


def _read_campaign(
    directory: Path, protocol: str, oracle: Path | None
) -> tuple[campaign._Manifest, campaign._Gates, str]:
    root = _read(directory / "manifest.json", "manifest.json")
    manifest = _decode(campaign._Manifest, root, "manifest.json")
    if manifest.protocol_sha256 != protocol:
        raise _problem("protocol_digest_mismatch", "manifest.json")
    if tuple(row.path for row in manifest.artifacts) != campaign._ARTIFACTS:
        raise _problem("invalid_campaign_artifact_roster", "manifest.json")
    receipts = {row.path: row.sha256 for row in manifest.artifacts}
    gates = _decode(
        campaign._Gates,
        _checked_bytes(directory / "gates.json", receipts["gates.json"]),
        "gates.json",
    )
    _check_gates(manifest, gates)
    complete = manifest.status != "execution_failure"
    if complete:
        if (
            manifest.sources != tuple(campaign._receipt(_ROOT, path) for path in _SOURCE_FILES)
            or manifest.base_revision != campaign._BASE
            or not _revision(manifest.source_revision)
            or manifest.rng != campaign._Rng()
            or not _chronological(manifest.started_at, manifest.ended_at)
            or manifest.thread_environment != tuple((name, "1") for name in campaign._THREADS)
        ):
            raise _problem("inconsistent_campaign_provenance", "manifest.json")
        _check_oracle(directory, manifest, oracle)
    coords = campaign.coordinates()
    n = manifest.trials_per_cell
    _stream_keys(
        directory / "cells.jsonl",
        receipts["cells.jsonl"],
        campaign._CellRow,
        ((coord, n, n) for coord in coords) if complete else None,
        lambda row: (row.coordinate, row.attempted, row.evaluated),
    )
    _stream_keys(
        directory / "decisions.jsonl",
        receipts["decisions.jsonl"],
        campaign._DecisionRow,
        ((*key, n, n) for key in product(coords, campaign._MODES, campaign._TARGETS, campaign._TAU))
        if complete
        else None,
        lambda row: (row.coordinate, row.mode, row.target, row.tau, row.attempted, row.evaluated),
    )
    return manifest, gates, _sha(root)


def _revision(text: str | None) -> bool:
    return text is not None and re.fullmatch(r"[a-f0-9]{40}", text) is not None


def _chronological(start: datetime, end: datetime) -> bool:
    return start.utcoffset() is not None and end.utcoffset() is not None and start <= end


def _valid_interval(value: Interval) -> bool:
    return 0 <= value[0] <= value[1] <= 1


def _worthwhile(result: case._PipelineDecision) -> bool:
    return (
        result.decision != "inconclusive" and result.changes_action and result.gain_exceeds_minimum
    )


def _simplest(owner: case._OwnerDecision) -> _Method | None:
    for name in ("P0", "P1", "P2"):
        if _worthwhile(getattr(owner.pipelines, name)):
            return name
    return "P3" if owner.h_adds_decision else None


def _validate_owner(field: case._FieldDecision, inner: Interval | None) -> None:
    owner, bounds = field.owner_decision, field.bounds
    if owner is None or bounds is None:
        raise _problem("inconsistent_owner_evidence", "decision.json")
    irreducible = inner is not None and inner[0] + case._TOL < owner.tau < inner[1] - case._TOL
    simplest = _simplest(owner)
    if (
        owner.confirmation is not None
        or not 0.05 <= owner.max_false_action_risk <= 1
        or owner.current_action.key != "current_action"
        or tuple((a.key, a.kind) for a in owner.alternatives)
        != tuple(zip(_ACTIONS, _KIND, strict=True))
        or owner.irreducible_at_features != irreducible
        or owner.smallest_useful_simple_pipeline != (None if simplest == "P3" else simplest)
        or owner.h_adds_decision
        != (owner.pipelines.P2.decision == "inconclusive" and _worthwhile(owner.pipelines.P3))
    ):
        raise _problem("inconsistent_owner_evidence", "decision.json")
    for name, result in owner.pipelines:
        band = getattr(bounds, name)
        if band is None or not _valid_interval(band) or decide(band, owner.tau) != result.decision:
            raise _problem("inconsistent_owner_evidence", "decision.json")
        i = ("clear", "rule_out", "inconclusive").index(result.decision)
        ambiguity = (
            None
            if i != 2
            else (
                "irreducible_at_features"
                if irreducible
                else "sampling_or_unresolved_identification"
            )
        )
        if (
            result.action != owner.alternatives[i]
            or result.ambiguity != ambiguity
            or (result.guess_vs_wait_worst_excess is None) != (i != 2)
            or not math.isfinite(result.g_min)
            or not math.isfinite(result.g_max)
            or result.g_min > result.g_max
            or (
                result.best_gain_exceeds_minimum and result.action.kind == owner.current_action.kind
            )
            or (result.gain_exceeds_minimum and not result.best_gain_exceeds_minimum)
            or (result.gain_exceeds_minimum and result.g_min < owner.minimum_gain)
            or (not result.best_gain_exceeds_minimum and result.g_max > owner.minimum_gain)
            or (not result.gain_exceeds_minimum and result.g_min > owner.minimum_gain)
            or (result.best_gain_exceeds_minimum and result.g_max < owner.minimum_gain)
        ):
            raise _problem("inconsistent_owner_evidence", "decision.json")


def _check_bounds(field: case._FieldDecision, details: case._Bounds | None) -> Interval | None:
    if details is None:
        if field.bounds is not None:
            raise _problem("inconsistent_field_packet", "bounds.json")
        return None
    if field.bounds is None or details.historical_benchmark != (field.source == "hillstrom"):
        raise _problem("inconsistent_field_packet", "bounds.json")
    for _, method in details.methods:
        inner = (method.L_ci[1], method.U_ci[0])
        if (
            not all(_valid_interval(pair) for pair in (method.L_ci, method.U_ci, method.outer))
            or method.outer != (method.L_ci[0], method.U_ci[1])
            or method.inner_span != (inner if inner[0] < inner[1] else None)
        ):
            raise _problem("invalid_field_band", "bounds.json")
    for i, (name, pipe) in enumerate(details.pipelines):
        names = campaign._METHODS[: i + 1]
        bands = tuple(getattr(details.methods, key).outer for key in names)
        lo, hi = max(b[0] for b in bands), min(b[1] for b in bands)
        expected = (lo, hi) if lo <= hi else None
        if (
            pipe.methods != names
            or pipe.outer != expected
            or getattr(field.bounds, name) != expected
            or pipe.reason != ("discordant_bands" if expected is None else None)
        ):
            raise _problem("inconsistent_field_packet", "bounds.json")
    return details.methods.Q.inner_span


def _check_thresholds(raw: bytes, details: case._Bounds | None, field: case._FieldDecision) -> None:
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        if tuple(next(reader, ())) != case._THRESHOLD_HEADER:
            raise ValueError("wrong header")
        rows = {
            (scope, name): {}
            for scope, names in (
                ("method", campaign._METHODS),
                ("pipeline", _PIPELINES),
            )
            for name in names
        }
        for scope, name, threshold, lo, hi, decision, reason, origin in reader:
            if details is None:
                raise ValueError("unexpected results")
            tau = float(threshold)
            group = rows[scope, name]
            if not 0 < tau < 1 or tau in group:
                raise ValueError("wrong threshold")
            group[tau] = origin
            band = getattr(details.methods if scope == "method" else details.pipelines, name).outer
            if band is None:
                if lo or hi or decision != "inconclusive" or reason != "discordant_bands":
                    raise ValueError("wrong null result")
            elif (float(lo), float(hi)) != band or decision != decide(band, tau) or reason:
                raise ValueError("wrong result")
        if details is None:
            return
        grid = dict.fromkeys(case._THRESHOLDS, "protocol_grid")
        first = rows["method", "U0"]
        extra = first.keys() - grid.keys()
        owner = field.owner_decision
        if owner is not None:
            if owner.tau not in grid:
                grid[owner.tau] = "eligible_card"
        elif len(extra) == 1 and field.card_sha256 is not None:
            tau = next(iter(extra))
            origin = first[tau]
            if origin == "ineligible_card_sensitivity" or (
                origin == "eligible_card"
                and field.source == "owner_trial"
                and field.reason == "discordant_bands"
            ):
                grid[tau] = origin
        if any(row != grid for row in rows.values()):
            raise ValueError("wrong threshold roster")
    except (ValueError, csv.Error, KeyError):
        raise _problem("invalid_field_thresholds", "thresholds.csv") from None


def _read_field(
    directory: Path, protocol: str, bounds_sha: str, case_sha: str
) -> tuple[_WireDecision, str]:
    root = _read(directory / "decision.json", "decision.json")
    field = _decode(_WireDecision, root, "decision.json")
    if field.protocol_sha256 != protocol:
        raise _problem("protocol_digest_mismatch", "decision.json")
    run = field.reproduction
    if (
        run.protocol_base_revision != campaign._PROTOCOL_BASE
        or run.harm_bounds_sha256 != bounds_sha
        or run.harm_case_sha256 != case_sha
        or not _revision(run.source_revision)
        or not _chronological(run.started_at, run.ended_at)
    ):
        raise _problem("inconsistent_field_provenance", "decision.json")
    a, b, c = _FIELD_FILES
    inputs = _decode(case._InputReceipt, _checked_bytes(directory / a, field.receipts.inputs), a)
    bounds = _decode(case._BoundsPacket, _checked_bytes(directory / b, field.receipts.bounds), b)
    thresholds = _checked_bytes(directory / c, field.receipts.thresholds)
    if bounds.protocol_sha256 != protocol:
        raise _problem("protocol_digest_mismatch", b)
    if (
        (bounds.status, bounds.reason) != (field.status, field.reason)
        or (
            inputs.source,
            inputs.manifest_sha256,
            inputs.card_sha256,
            inputs.permission_verified,
            inputs.design_verified,
        )
        != (
            field.source,
            field.manifest_sha256,
            field.card_sha256,
            field.permission_verified,
            field.design_verified,
        )
        or inputs.digest_verified
        != (inputs.csv_sha256 is not None and inputs.csv_sha256 == inputs.expected_csv_sha256)
        or (inputs.digest_verified and not (field.permission_verified and field.design_verified))
        or (field.permission_verified and field.manifest_sha256 is None)
        or (field.design_verified and not field.permission_verified)
        or field.design_verified
        != (inputs.assignment is not None and inputs.treatment_probability is not None)
        or (
            field.design_verified
            and (
                (field.source == "owner_trial")
                != (
                    inputs.design_sha256 is not None
                    and inputs.assignment in ("bernoulli", "complete")
                )
            )
        )
        or (bounds.bounds is not None) != (field.status in ("completed", "research_only"))
        or (bounds.bounds is not None and not inputs.digest_verified)
        or (field.status == "completed")
        != (field.owner_decision is not None and field.reason is None)
        or (field.status != "completed" and (field.owner_decision is not None or not field.reason))
        or (
            field.status == "completed"
            and (field.source != "owner_trial" or field.card_sha256 is None)
        )
        or (field.status == "unavailable" and field.reason not in case._UNAVAILABLE)
        or (
            field.status == "research_only"
            and field.reason not in ("missing_decision_card", "discordant_bands")
        )
        or (
            field.reason == "missing_case_manifest"
            and inputs != case._InputReceipt(source=field.source)
        )
    ):
        raise _problem("inconsistent_field_packet", "decision.json")
    inner = _check_bounds(field, bounds.bounds)
    if field.owner_decision is not None:
        _validate_owner(field, inner)
    _check_thresholds(thresholds, bounds.bounds, field)
    return field, _sha(root)


def _validate_ack(
    ack: _Acknowledgment, field: case._FieldDecision, field_sha: str, now: datetime
) -> None:
    if (
        field.owner_decision is None
        or ack.decision_sha256 != field_sha
        or ack.card_sha256 != field.card_sha256
        or not field.reproduction.ended_at < ack.acknowledged_at <= now
    ):
        raise _problem("acknowledgment_binding_mismatch", "acknowledgment.json")


def _recommend(
    status: _RunStatus, field: case._FieldDecision, ack: _Acknowledgment | None
) -> _Recommendation:
    reasons = [] if status == "green" else [f"campaign_{status}"]
    owner = field.owner_decision
    if field.reason is not None:
        reasons.append("field_execution_failure" if field.status == "failed" else field.reason)
    if not field.permission_verified:
        reasons.append("missing_analysis_and_reporting_permission")
    if not field.design_verified:
        reasons.append("missing_verified_field_design")
    if field.source != "owner_trial":
        reasons.append("missing_matched_owner_trial")
    if owner is None:
        reasons.append("missing_eligible_owner_card")
    if ack is None:
        reasons.append("missing_post_readout_acknowledgment")
    if reasons or owner is None or ack is None:
        return _Recommendation("defer", None, tuple(reasons))
    method = _simplest(owner)
    if method is not None:
        result = getattr(owner.pipelines, method)
        accepted = ack.intended_action
        if (
            accepted is None
            or not accepted.feasible_and_intended
            or (accepted.pipeline, accepted.action_key) != (method, result.action.key)
        ):
            return _Recommendation("defer", None, ("acknowledgment_action_mismatch",))
        return _Recommendation("implement", method, ("matched_action_with_strict_worst_case_gain",))
    if owner.irreducible_at_features and ack.all_admissible_baseline_features:
        return _Recommendation("reject", None, ("tolerance_inside_full_feature_inner_interval",))
    conclusive = tuple(result for _, result in owner.pipelines if result.decision != "inconclusive")
    if (
        ack.comparison_adequate
        and len(conclusive) == len(_PIPELINES)
        and any(result.changes_action for result in conclusive)
        and not any(
            result.best_gain_exceeds_minimum for result in conclusive if result.changes_action
        )
    ):
        return _Recommendation("reject", None, ("best_compatible_gain_not_enough",))
    return _Recommendation(
        "defer",
        None,
        (
            "unresolved_at_owner_tolerance"
            if not conclusive
            else "insufficient_incremental_utility",
        ),
    )


_REMEDIES = {
    "campaign_smoke": "Complete the prespecified full calibration; smoke evidence cannot qualify.",
    "campaign_inconclusive": "Resolve the inconclusive safety or power counters at adequate precision.",
    "campaign_red": "Diagnose the measured safety or power failures before considering a recommendation. A red calibration counter is not on its own a scientific counterexample.",
    "missing_case_manifest": "Stage the independently verified aggregate evaluation inputs; no field bounds were produced.",
    "missing_analysis_and_reporting_permission": "Document authority to analyze the staged trial and report aggregates.",
    "missing_verified_field_design": "Document assignment, complete outcomes, unit identity and the same target population.",
    "missing_matched_owner_trial": "Use a representative trial prospectively matched to the actual decision; the historical example is benchmark-only.",
    "missing_eligible_owner_card": "Freeze the tolerance, action alternatives, loss scale and costs with an authorized owner before unblinding.",
    "missing_post_readout_acknowledgment": "Obtain independent post-readout confirmation tied to the exact decision and card receipts.",
    "acknowledgment_action_mismatch": "Confirm the feasibility and intent of the selected pipeline and card action for this exact readout.",
    "unresolved_at_owner_tolerance": "Distinguish sampling uncertainty (which more data may shrink) from partial identification (which it cannot).",
    "insufficient_incremental_utility": "Establish a changed conclusive action with strictly sufficient worst-case gain; learned adjustment needs an increment over fixed features.",
    "tolerance_inside_full_feature_inner_interval": "For this named use case, more data at the same feature resolution cannot resolve the demonstrated ambiguity. New admissible features or defensible identifying evidence could change this.",
    "best_compatible_gain_not_enough": "For this named use case, even the best compatible gain cannot meet the frozen criterion. A materially different workflow with prospectively documented costs could change this.",
}

_GAPS = (
    ("Documented need", "Matched action and gain in this readout; broader demand not established."),
    (
        "Approved design",
        "Population harmful fraction and projected exact-binomial sampling bands under the frozen assumptions; public design approval still required.",
    ),
    (
        "Interactions",
        "Raw fixed-horizon binary outcome only. No CUPED residuals, ratios, winsorization, clustering, sequential looks, subgroup declarations or general multiplicity.",
    ),
    ("Planning matches runtime", "Neither a public planner nor runtime parity is established."),
    ("Ingress parity", "No public ingress or six-builder parity is claimed."),
    (
        "Independent reference",
        "Exact table, simplex and binomial enumeration for the frozen design; public inventory still required.",
    ),
    (
        "Refusals and bounds",
        "Design, outcome, positivity, population and eligibility refusals and operational ceilings need public contracts.",
    ),
    (
        "Complete change",
        "Public implementation, docs, examples, compatibility and fixtures are separate work.",
    ),
)


def _markdown(payload: dict[str, object], result: _Recommendation) -> str:
    title = result.recommendation.upper()
    lines = [f"# Fraction-harmed study: {title}", ""]
    if result.method is not None:
        lines += [
            f"Scope only **{result.method}: {_METHODS[result.method]}** separately. This is not design approval or a launch decision.",
            "",
        ]
    lines += [
        "## Evidence and reasons",
        "",
        *(f"- `{reason}`. {_REMEDIES.get(reason, '')}".rstrip() for reason in result.reasons),
        "",
    ]
    if payload["field_status"] == "unavailable":
        lines += ["Field bounds, owner tolerance, and gain are unavailable, not zero.", ""]
    if payload["campaign_status"] is not None:
        lines += [
            f"Synthetic campaign: **{payload['campaign_status']}**, {payload['green_counters']}/{payload['total_counters']} counters green.",
            "",
        ]
    for name in ("protocol", "campaign_manifest", "oracle", "field_decision", "acknowledgment"):
        digest = payload[f"{name}_sha256"]
        if digest is not None:
            lines.append(f"- {name.replace('_', ' ')} SHA-256: `{digest}`")
    lines += ["", "## Limits", "", *(f"- {limit}" for limit in _LIMITS)]
    if result.recommendation == "implement":
        lines += [
            "",
            "## Method acceptance gaps",
            "",
            "| Requirement | Evidence and gap |",
            "| --- | --- |",
            *(f"| {name} | {gap} |" for name, gap in _GAPS),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Verify the frozen aggregate packets and write a recommendation in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--oracle", type=Path, help="Exact R1 aggregate receipt if moved since the campaign"
    )
    parser.add_argument(
        "--acknowledgment", type=Path, help="Independently authenticated post-readout receipt"
    )
    args = parser.parse_args(argv)
    payload: dict[str, object] = dict.fromkeys(
        (
            "protocol_sha256",
            "campaign_manifest_sha256",
            "oracle_sha256",
            "field_decision_sha256",
            "acknowledgment_sha256",
            "campaign_status",
            "field_status",
            "field_source",
            "green_counters",
            "total_counters",
            "non_green_counters",
            "field_bounds",
            "owner_tolerance",
            "minimum_gain",
            "g_min",
            "g_max",
            "diagnostics",
        )
    )
    exit_code = 1
    try:
        protocol = _sha(_read(_ROOT / campaign._PROTOCOL, "frozen_protocol"))
        if protocol != campaign._PROTOCOL_DIGEST:
            raise _problem("protocol_digest_mismatch", "frozen_protocol")
        payload["protocol_sha256"] = protocol
        manifest, gates, campaign_sha = _read_campaign(args.campaign, protocol, args.oracle)
        totals = Counter(row.status for row in gates.counters)
        payload.update(
            campaign_manifest_sha256=campaign_sha,
            oracle_sha256=manifest.oracle_sha256,
            campaign_status=manifest.status,
            green_counters=totals["green"],
            total_counters=len(gates.counters),
            non_green_counters=[
                c.model_dump(mode="json") for c in gates.counters if c.status != "green"
            ],
        )
        field, field_sha = _read_field(
            args.case,
            protocol,
            _sha(_read(_ROOT / "calibration/harm_bounds.py", "harm_bounds.py")),
            _sha(_read(_ROOT / "calibration/harm_case.py", "harm_case.py")),
        )
        payload.update(
            field_decision_sha256=field_sha,
            field_status=field.status,
            field_source=field.source,
            field_bounds=field.bounds.model_dump(mode="json") if field.bounds is not None else None,
        )
        ack = None
        if args.acknowledgment is not None:
            raw = _read(args.acknowledgment, "acknowledgment.json")
            ack = _decode(_Acknowledgment, raw, "acknowledgment.json")
            _validate_ack(ack, field, field_sha, datetime.now(UTC))
            payload["acknowledgment_sha256"] = _sha(raw)
        if manifest.status == "execution_failure" or field.status == "failed":
            coordinate = (
                "manifest.json" if manifest.status == "execution_failure" else "decision.json"
            )
            raise _problem("source_execution_failure", coordinate)
        result = _recommend(manifest.status, field, ack)
        owner = field.owner_decision
        if owner is not None:
            payload.update(owner_tolerance=owner.tau, minimum_gain=owner.minimum_gain)
            if result.method is not None:
                row = getattr(owner.pipelines, result.method)
                payload.update(g_min=row.g_min, g_max=row.g_max)
        exit_code = 2 if result.recommendation == "defer" else 0
    except StudyError as exc:
        result = _Recommendation("defer", None, (exc.reason,))
        payload["diagnostics"] = [{"reason": exc.reason, "context": exc.context}]
    except OSError:
        result = _Recommendation("defer", None, ("evidence_read_error",))
        payload["diagnostics"] = [
            {
                "reason": "evidence_read_error",
                "context": (("stage", "recommendation"), ("coordinate", "source_receipt")),
            }
        ]
    payload.update(
        schema_version=1,
        recommendation=result.recommendation,
        method=result.method,
        reasons=result.reasons,
        limits=_LIMITS,
        exit_code=exit_code,
        status="invalid_evidence" if exit_code == 1 else "assessed",
    )
    try:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "verdict.json").write_text(
            json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "verdict.md").write_text(_markdown(payload, result), encoding="utf-8")
    except OSError:
        print("could_not_write_verdict", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
