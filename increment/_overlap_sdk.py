"""Normalize persisted SDK decisions against the original allocation archive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import NoReturn, Self, cast, final

from increment._assignment_refusals import refuse_assignment_replay_mismatch
from increment._overlap_evidence import _SURROGATE, FACTOR_IDS, FactorId, _utc_microsecond
from increment._overlap_refusals import EVIDENCE_INVALID, LAW_UNSUPPORTED, PLAN_INVALID
from increment.assignment import AllocationEvaluator, Decision, compile_allocation
from increment.assignment._archive import (
    ArchivedFactor,
    describe_archived_factor,
    has_archived_target,
)
from increment.errors import CodedError, refuse
from increment.semantics.overlap import (
    OverlapClaim,
    OverlapFactor,
    OverlapLaw,
    OverlapMetric,
    OverlapPlan,
    OverlapTarget,
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            refuse(PLAN_INVALID, field="config", reason="duplicate_key")
        result[key] = value
    return result


def _nonfinite(_: str) -> NoReturn:
    refuse(PLAN_INVALID, field="config", reason="nonfinite_number")


@final
@dataclass(frozen=True, slots=True, eq=False, init=False)
class OverlapSdkArchive:
    """A validated allocation snapshot. Construct it from exact archived bytes."""

    _sha256: str
    _evaluator: AllocationEvaluator

    def __new__(cls) -> Self:
        refuse(PLAN_INVALID, field="config", reason="archive_requires_bytes")

    @classmethod
    def from_bytes(cls, raw: bytes) -> Self:
        """Bind the original bytes and validate their protocol-v1 allocation rules."""
        if type(raw) is not bytes:
            refuse(PLAN_INVALID, field="config", reason="archive_requires_bytes")
        digest = sha256(raw).hexdigest()
        try:
            decoded = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_nonfinite
            )
        except CodedError:
            raise
        except (ValueError, RecursionError):
            pass
        else:
            evaluator = compile_allocation(decoded)
            archive = object.__new__(cls)
            object.__setattr__(archive, "_sha256", digest)
            object.__setattr__(archive, "_evaluator", evaluator)
            return archive
        refuse(PLAN_INVALID, field="config", reason="invalid_json")


def _archive(config: bytes | OverlapSdkArchive) -> OverlapSdkArchive:
    if type(config) is OverlapSdkArchive:
        return config
    return OverlapSdkArchive.from_bytes(config)


def _matches_record(submitted: object, expected: object) -> bool:
    if type(submitted) is not type(expected):
        return False
    if type(expected) is dict:
        actual = cast("dict[str, object]", submitted)
        reference = cast("dict[str, object]", expected)
        return actual.keys() == reference.keys() and all(
            _matches_record(actual[key], value) for key, value in reference.items()
        )
    return submitted == expected


def _is_identity(value: object) -> bool:
    return type(value) is str and bool(value) and _SURROGATE.search(value) is None


def sdk_overlap_decision(
    *,
    study_id: str,
    unit_id: str,
    decided_at: datetime,
    decision: Decision,
    config: bytes | OverlapSdkArchive,
) -> dict[str, object]:
    """Check an existing SDK decision against its archive and project one log row.

    A missing decision is not reconstructed. Reuse a validated archive when
    projecting a batch under the same allocation snapshot.
    """
    archive = _archive(config)
    if (
        not _is_identity(study_id)
        or not _is_identity(unit_id)
        or not isinstance(decided_at, datetime)
    ):
        refuse(EVIDENCE_INVALID, role="decisions", reason="invalid_value", count=1)
    try:
        instant = _utc_microsecond(decided_at)
    except ValueError:
        pass
    else:
        return _sdk_row(archive, study_id, unit_id, instant, decision)
    refuse(EVIDENCE_INVALID, role="decisions", reason="invalid_value", count=1)


def _sdk_row(
    archive: OverlapSdkArchive,
    study_id: str,
    unit_id: str,
    instant: datetime,
    decision: Decision,
) -> dict[str, object]:
    target = decision.get("target") if type(decision) is dict else None
    if not has_archived_target(target, archive._evaluator):
        refuse_assignment_replay_mismatch(details="decision")
    expected = archive._evaluator.decide(target, unit_id)
    if not _matches_record(decision, expected):
        refuse_assignment_replay_mismatch(details="decision")
    return {
        "study_id": study_id,
        "unit_id": unit_id,
        "target": target,
        "decided_at": instant,
        "revision": decision["revision"],
        "archive_sha256": archive._sha256,
        "selected_experiment": decision["experiment"],
        "status": decision["status"],
        "arm": decision["arm"],
        "assigned_weight": decision["arm_bps"],
    }


def _factor(
    side: FactorId,
    archive: OverlapSdkArchive,
    target: str,
    experiment: str,
    off: str,
    on: str,
) -> tuple[OverlapFactor, ArchivedFactor]:
    label = side.lower()
    configured = describe_archived_factor(archive._evaluator, target, experiment)
    if isinstance(configured, str):
        field = "target" if configured == "unknown_target" else "experiment"
        refuse(PLAN_INVALID, field=f"{label}_{field}", reason=configured)
    arms = dict(configured.arms)
    if len(arms) != 2:
        refuse(LAW_UNSUPPORTED, reason="nonbinary_experiment", factors=(side,))
    for field, arm_key in (("off", off), ("on", on)):
        if type(arm_key) is not str or arm_key not in arms:
            refuse(PLAN_INVALID, field=f"{label}_{field}", reason="unknown_arm")
    return (
        OverlapFactor(
            target=target,
            experiment=configured.experiment,
            revision=configured.revision,
            off=off,
            on=on,
            off_weight=arms[off],
            on_weight=arms[on],
            arm_draw_id=configured.arm_draw_id,
            layer=configured.layer,
        ),
        configured,
    )


def sdk_overlap_plan(  # noqa: PLR0913 — archive selectors and registered plan fields.
    *,
    a_config: bytes | OverlapSdkArchive,
    a_target: str,
    a_experiment: str,
    a_off: str,
    a_on: str,
    b_config: bytes | OverlapSdkArchive,
    b_target: str,
    b_experiment: str,
    b_off: str,
    b_on: str,
    study_id: str,
    unit_namespace: str,
    population_id: str,
    eligibility_rule_sha256: str,
    enrollment_rule_sha256: str,
    registered_at: datetime,
    entry_open: datetime,
    start: datetime,
    end: datetime,
    target: OverlapTarget,
    metrics: tuple[OverlapMetric, ...],
    claims: tuple[OverlapClaim, ...],
    alpha: float,
    serving_barrier_id: str,
) -> OverlapPlan:
    """Register the fixed-window design with the arm law derived from both archives."""
    a_archive = _archive(a_config)
    b_archive = (
        a_archive
        if type(a_config) is bytes and type(b_config) is bytes and a_config == b_config
        else _archive(b_config)
    )
    a, a_rules = _factor("A", a_archive, a_target, a_experiment, a_off, a_on)
    b, b_rules = _factor("B", b_archive, b_target, b_experiment, b_off, b_on)
    if (
        a_rules.holdout_domain is not None
        and a_rules.holdout_domain == b_rules.holdout_domain
        and a_rules.held_out_bps != b_rules.held_out_bps
    ):
        refuse(LAW_UNSUPPORTED, reason="shared_gate_mismatch", factors=FACTOR_IDS)
    if not (a_rules.can_assign and b_rules.can_assign):
        admission = "joint_gate_impossible"
    elif a_rules.always_assign and b_rules.always_assign:
        admission = "both_guaranteed"
    else:
        admission = "joint_gate_required"
    law = OverlapLaw(
        scheme="independent_unit_bernoulli",
        origin="sdk_v1",
        archive_sha256=(a_archive._sha256, b_archive._sha256),
        gate_basis="common_entry_archived_draws",
        admission=admission,
        rule_source_id="increment.assignment.protocol-v1",
        rule_version="1",
        serving_barrier_id=serving_barrier_id,
    )
    return OverlapPlan(
        study_id=study_id,
        unit_namespace=unit_namespace,
        population_id=population_id,
        eligibility_rule_sha256=eligibility_rule_sha256,
        enrollment_rule_sha256=enrollment_rule_sha256,
        registered_at=registered_at,
        entry_open=entry_open,
        start=start,
        end=end,
        target=target,
        a=a,
        b=b,
        law=law,
        metrics=metrics,
        claims=claims,
        alpha=alpha,
    )
