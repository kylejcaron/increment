"""Portable declarations for fixed, version-pinned parallel assignment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL
from increment.errors import CodedModel, InvalidRequestError, raiser, refusals
from increment.semantics.design import Randomized

BUCKETS = 10_000
Text = Annotated[str, Field(strict=True, min_length=1)]

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "assignment.contract.invalid": "Invalid assignment contract: {reason}",
        "assignment.contract.design_mismatch": "Assignment contract disagrees with analysis: {reason}",
        "assignment.contract.version_reused": "Changed assignment rules require a new {scope} version; retain the old contract for existing enrollments",
    },
)
_raise = raiser(_REFUSALS)


class AssignmentModel(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Reject duplicate object keys rather than accepting parser-dependent rules."""
        try:
            json.loads(json_data, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, RecursionError):
            pass  # Preserve Pydantic's ordinary malformed-JSON validation.
        return super().model_validate_json(json_data, **kwargs)

    @field_validator("*", mode="before")
    @classmethod
    def _validate_utf8(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                _raise("assignment.contract.invalid", reason="strings must encode as UTF-8")
        return value


class AssignmentShare(AssignmentModel):
    """A named, positive share of the 10,000 bucket space."""

    name: Text
    buckets: StrictInt = Field(gt=0, le=BUCKETS)


class AssignmentNamespace(AssignmentModel):
    """Shared identity and holdout policy across independent experiment layers."""

    name: Text
    version: Text
    salt: Text
    identity_key: Text
    holdout_buckets: StrictInt = Field(default=0, ge=0, lt=BUCKETS)


class AssignmentLayer(AssignmentModel):
    """Disjoint experiment slots; unallocated buckets enroll in no experiment."""

    namespace: AssignmentNamespace
    name: Text
    version: Text
    salt: Text
    experiments: tuple[AssignmentShare, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_slots(self) -> AssignmentLayer:
        _validate_shares(self.experiments, complete=False)
        return self


class AssignmentSpec(AssignmentModel):
    """Immutable fixed-allocation rules; enrollment pins this exact version.

    Eligibility is evaluated externally before assignment under the named
    eligibility version. Changing it, identity, allocation, or routing creates
    a new version for new enrollments; existing enrollments retain their spec.
    """

    schema_version: Literal[1] = 1
    algorithm: Literal["sha256-json-v1"] = "sha256-json-v1"
    experiment: Text
    version: Text
    salt: Text
    eligibility_version: Text
    layer: AssignmentLayer
    control_group: Text
    arms: tuple[AssignmentShare, ...] = Field(min_length=2)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_type(cls, value: object) -> object:
        if type(value) is not int:
            _raise("assignment.contract.invalid", reason="schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def _validate_arms(self) -> AssignmentSpec:
        _validate_shares(self.arms, complete=True)
        if {arm.name for arm in self.arms} & {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL}:
            _raise("assignment.contract.invalid", reason="accounting labels cannot be arms")
        if self.control_group not in {arm.name for arm in self.arms}:
            _raise("assignment.contract.invalid", reason="control_group is absent from arms")
        if self.experiment not in {slot.name for slot in self.layer.experiments}:
            _raise("assignment.contract.invalid", reason="experiment is absent from layer")
        return self

    def canonical_json(self) -> str:
        """Canonical portable bytes: sorted keys and name-sorted share arrays."""
        data = self.model_dump(mode="json")
        data["arms"] = sorted(data["arms"], key=lambda arm: arm["name"])
        data["layer"]["experiments"] = sorted(
            data["layer"]["experiments"], key=lambda slot: slot["name"]
        )
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """SHA-256 of canonical contract UTF-8 bytes, including every declaration."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def allocation(self) -> dict[str, float]:
        """Conditional arm allocation among this experiment's enrolled units."""
        return {arm.name: arm.buckets / BUCKETS for arm in sorted(self.arms, key=lambda a: a.name)}

    def design(self) -> Randomized:
        """Declare the same independent fixed law to any parallel analysis ingress."""
        return Randomized(
            control_group=self.control_group,
            allocation=self.allocation,
            allocation_scheme="independent",
        )

    def validate_analysis(
        self,
        *,
        experiment: str,
        identity_key: str,
        control_group: str,
        allocation: Mapping[str, float] | None,
        allocation_scheme: str | None,
        randomized: bool,
    ) -> None:
        """Refuse a conflicting analysis law before reading an assignment source."""
        reason = None
        if not randomized:
            reason = "only parallel randomized experiments accept this contract"
        elif experiment != self.experiment:
            reason = "experiment name differs"
        elif identity_key != self.layer.namespace.identity_key:
            reason = "identity_key must name the randomization-grain column"
        elif control_group != self.control_group:
            reason = "control group differs"
        elif allocation_scheme not in (None, "independent"):
            reason = "allocation scheme must be independent"
        elif allocation is not None:
            scale = max(allocation.values(), default=0)
            if scale <= 0 or any(not math.isfinite(v) or v <= 0 for v in allocation.values()):
                _raise("assignment.contract.design_mismatch", reason="invalid allocation weights")
            total = math.fsum(value / scale for value in allocation.values())
            if set(allocation) != set(self.allocation) or not all(
                math.isclose((allocation[name] / scale) / total, share, rel_tol=1e-12, abs_tol=0)
                for name, share in self.allocation.items()
            ):
                reason = "allocation differs"
        if reason is not None:
            _raise("assignment.contract.design_mismatch", reason=reason)

    def validate_successor(self, successor: AssignmentSpec) -> None:
        """Require new versions wherever rules changed; never migrate enrolled units."""
        if self.canonical_json() == successor.canonical_json():
            return
        if self.experiment != successor.experiment:
            _raise("assignment.contract.invalid", reason="successor names another experiment")
        if self.version == successor.version:
            _raise("assignment.contract.version_reused", scope="experiment")
        old_ns, new_ns = self.layer.namespace, successor.layer.namespace
        if old_ns != new_ns and (old_ns.name, old_ns.version) == (new_ns.name, new_ns.version):
            _raise("assignment.contract.version_reused", scope="namespace")
        old_layer = self.layer.model_dump(exclude={"namespace", "experiments"})
        new_layer = successor.layer.model_dump(exclude={"namespace", "experiments"})
        old_slots = sorted((s.name, s.buckets) for s in self.layer.experiments)
        new_slots = sorted((s.name, s.buckets) for s in successor.layer.experiments)
        if (old_layer != new_layer or old_slots != new_slots) and (
            self.layer.name,
            self.layer.version,
        ) == (successor.layer.name, successor.layer.version):
            _raise("assignment.contract.version_reused", scope="layer")


def _validate_shares(shares: tuple[AssignmentShare, ...], *, complete: bool) -> None:
    if len({share.name for share in shares}) != len(shares):
        _raise("assignment.contract.invalid", reason="share names must be unique")
    total = sum(share.buckets for share in shares)
    if total > BUCKETS or (complete and total != BUCKETS):
        _raise("assignment.contract.invalid", reason="invalid bucket total")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _raise("assignment.contract.invalid", reason="duplicate JSON object key")
        result[key] = value
    return result
