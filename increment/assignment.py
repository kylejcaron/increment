"""Reference fixed assignment and independent checks over assignment logs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, StrictBool

from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.readout_types import IntegrityResult
from increment.semantics.randomization import (
    BUCKETS,
    AssignmentLayer,
    AssignmentModel,
    AssignmentNamespace,
    AssignmentShare,
    AssignmentSpec,
    Text,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "assignment.input.identity": "unit_id must be a nonempty UTF-8 string; canonicalize identity before assignment",
        "assignment.input.eligibility": "eligible must be a boolean evaluated before assignment",
        "assignment.input.grain": "randomization_grain must be unit or cluster",
        "assignment.log.duplicate": "Assignment log repeats randomization unit {unit_id!r}; audit one enrollment per unit and contract",
        "assignment.contract.conflict": "Conflicting {scope} declarations for {name!r}; share the same versioned rules across experiments",
    },
)
_raise = raiser(_REFUSALS)

AssignmentStatus = Literal["assigned", "ineligible", "holdout", "outside_layer"]


class AssignmentRecord(AssignmentModel):
    """An enrollment decision; this records assignment, never actual exposure."""

    unit_id: Text
    eligible: StrictBool
    specification_version: Text
    contract_fingerprint: Text
    status: AssignmentStatus
    arm: Text | None = None


class AssignmentMismatch(AssignmentModel):
    unit_id: Text
    reasons: tuple[str, ...]
    logged: AssignmentRecord
    expected: AssignmentRecord


class AssignmentAudit(AssignmentModel):
    """Separate conformance, exclusion, and allocation evidence for supplied logs.

    No result proves complete logging, trustworthy identity/eligibility,
    exposure, or causal validity. Balance uses logged enrolled arms and the
    existing always-valid diagnostic, not the replayed arm distribution.
    """

    contract_fingerprint: Text
    checked_units: int = Field(ge=0)
    replay: Literal["matched", "failed", "not_checked"]
    exclusions: Literal["not_violated", "failed", "not_checked"]
    exclusion_units: tuple[str, ...]
    mismatches: tuple[AssignmentMismatch, ...]
    allocation: IntegrityResult


def assignment_bucket(domain: str, *parts: str) -> int:
    """SHA-256 over a compact UTF-8 JSON array; unsigned digest modulo 10,000.

    All components are strings, with no Unicode normalization. The domain
    and literal protocol prefix prevent tuple/concatenation ambiguity.
    """
    for part in (domain, *parts):
        _validate_identity(part)
    encoded = json.dumps(
        ["increment.assignment.v1", domain, *parts], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big") % BUCKETS


def _validate_identity(unit_id: object) -> None:
    if not isinstance(unit_id, str) or not unit_id:
        _raise("assignment.input.identity")
    try:
        unit_id.encode("utf-8")
    except UnicodeEncodeError:
        _raise("assignment.input.identity")


def _slot(shares: tuple[AssignmentShare, ...], bucket: int) -> str | None:
    edge = 0
    for share in sorted(shares, key=lambda s: s.name):
        edge += share.buckets
        if bucket < edge:
            return share.name
    return None


def assign(spec: AssignmentSpec, unit_id: str, *, eligible: bool = True) -> AssignmentRecord:
    """Assign a stable randomization unit under a pinned pre-enrollment contract."""
    _validate_identity(unit_id)
    if type(eligible) is not bool:
        _raise("assignment.input.eligibility")
    ns, layer = spec.layer.namespace, spec.layer
    identity = (ns.name, ns.version, ns.salt, ns.identity_key, unit_id)
    arm = None
    status: AssignmentStatus
    if not eligible:
        status = "ineligible"
    elif assignment_bucket("holdout", *identity) < ns.holdout_buckets:
        status = "holdout"
    elif (
        _slot(
            layer.experiments,
            assignment_bucket("layer", *identity, layer.name, layer.version, layer.salt),
        )
        != spec.experiment
    ):
        status = "outside_layer"
    else:
        status = "assigned"
        arm = _slot(
            spec.arms,
            assignment_bucket(
                "arm",
                *identity,
                layer.name,
                layer.version,
                layer.salt,
                spec.experiment,
                spec.version,
                spec.salt,
                spec.eligibility_version,
            ),
        )
    return AssignmentRecord(
        unit_id=unit_id,
        eligible=eligible,
        specification_version=spec.version,
        contract_fingerprint=spec.fingerprint,
        status=status,
        arm=arm,
    )


def audit_assignments(
    spec: AssignmentSpec,
    records: Iterable[AssignmentRecord],
    *,
    randomization_grain: Literal["unit", "cluster"] = "unit",
) -> AssignmentAudit:
    """Replay one contract's assignment-grain log, checking exclusions and balance.

    Duplicates refuse rather than inflating balance. An empty log is unchecked.
    Wrong contract versions, unknown arms, or inconsistent enrollment statuses
    leave balance unavailable rather than silently dropping invalid entries.
    """
    if randomization_grain not in ("unit", "cluster"):
        _raise("assignment.input.grain")
    seen: set[str] = set()
    mismatches = []
    excluded = []
    counts = dict.fromkeys(spec.allocation, 0)
    balance_available = True
    for logged in records:
        if logged.unit_id in seen:
            _raise("assignment.log.duplicate", unit_id=logged.unit_id)
        seen.add(logged.unit_id)
        expected = assign(spec, logged.unit_id, eligible=logged.eligible)
        reasons = tuple(
            field
            for field in ("specification_version", "contract_fingerprint", "status", "arm")
            if getattr(logged, field) != getattr(expected, field)
        )
        if reasons:
            mismatches.append(
                AssignmentMismatch(
                    unit_id=logged.unit_id, reasons=reasons, logged=logged, expected=expected
                )
            )
        if expected.status != "assigned" and (
            logged.arm is not None or logged.status == "assigned"
        ):
            excluded.append(logged.unit_id)
        if (
            logged.contract_fingerprint != spec.fingerprint
            or logged.specification_version != spec.version
            or logged.status != expected.status
            or (logged.status == "assigned") != (logged.arm is not None)
            or (logged.arm is not None and logged.arm not in counts)
        ):
            balance_available = False
        elif logged.arm is not None:
            counts[logged.arm] += 1
    allocation = assignment_integrity(
        spec.design(),
        counts if balance_available and sum(counts.values()) else None,
        randomization_grain=randomization_grain,
    )
    if allocation.status == "not_checked_missing_counts":
        allocation = IntegrityResult.model_validate(
            {
                **{name: getattr(allocation, name) for name in IntegrityResult.model_fields},
                "context": {
                    "reason": "assignment_log_invalid"
                    if not balance_available
                    else "no_enrolled_units"
                    if seen
                    else "assignment_log_empty"
                },
            }
        )
    return AssignmentAudit(
        contract_fingerprint=spec.fingerprint,
        checked_units=len(seen),
        replay="failed" if mismatches else "matched" if seen else "not_checked",
        exclusions="failed" if excluded else "not_violated" if seen else "not_checked",
        exclusion_units=tuple(excluded),
        mismatches=tuple(mismatches),
        allocation=allocation,
    )


def validate_assignment_specs(specs: Iterable[AssignmentSpec]) -> None:
    """Check shared namespace/layer definitions before deploying experiments together."""
    namespaces: dict[tuple[str, str], AssignmentNamespace] = {}
    layers: dict[tuple[str, str, str, str], dict[str, object]] = {}
    experiments: dict[tuple[str, str, str, str], str] = {}
    for spec in specs:
        ns, layer = spec.layer.namespace, spec.layer
        key = (ns.name, ns.version)
        if key in namespaces and namespaces[key] != ns:
            _raise("assignment.contract.conflict", scope="namespace", name=ns.name)
        namespaces[key] = ns
        layer_key = (*key, layer.name, layer.version)
        data = layer.model_dump(mode="json")
        data["experiments"] = sorted(data["experiments"], key=lambda s: s["name"])
        if layer_key in layers and layers[layer_key] != data:
            _raise("assignment.contract.conflict", scope="layer", name=layer.name)
        layers[layer_key] = data
        experiment_key = (*key, spec.experiment, spec.version)
        if experiment_key in experiments and experiments[experiment_key] != spec.fingerprint:
            _raise("assignment.contract.conflict", scope="experiment", name=spec.experiment)
        experiments[experiment_key] = spec.fingerprint


__all__ = [
    "AssignmentAudit",
    "AssignmentLayer",
    "AssignmentMismatch",
    "AssignmentNamespace",
    "AssignmentRecord",
    "AssignmentShare",
    "AssignmentSpec",
    "assign",
    "assignment_bucket",
    "audit_assignments",
    "validate_assignment_specs",
]
