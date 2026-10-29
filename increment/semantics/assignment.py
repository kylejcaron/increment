"""Orthogonal assignment declarations for parallel and switchback studies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from increment.errors import CodedModel, InvalidRequestError, raiser, refusals


def refuse_assignment_invalid(reason: str, *, details: object = None) -> NoReturn:
    from increment._assignment_refusals import refuse_assignment_invalid as refuse

    refuse(reason, details=details)


def refuse_assignment_replay_mismatch(*, details: object = None) -> NoReturn:
    from increment._assignment_refusals import (
        refuse_assignment_replay_mismatch as refuse,
    )

    refuse(details=details)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "definition.assignment.reject_bool": "expected an int, got a bool",
        "definition.assignment.carryover_order_type": "carryover_order must be a non-boolean int",
        "definition.assignment.carryover_order_range": "carryover_order={carryover_order} must satisfy 0 <= carryover_order < observation_steps={observation_steps}; an order at or past observation_steps would discard the entire retained window",
    },
)
_raise = raiser(_REFUSALS)
ASSIGNMENT_REJECT_BOOL = _REFUSALS["definition.assignment.reject_bool"]


def _reject_bool(v: object) -> object:
    """``bool`` is an ``int`` subclass in Python; a declared step count
    must never silently accept ``True``/``False`` as ``1``/``0``."""
    if isinstance(v, bool):
        _raise("definition.assignment.reject_bool")
    return v


class _AssignmentModel(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParallelAssignment(_AssignmentModel):
    kind: Literal["parallel"] = "parallel"


class IndependentBernoulliOrder(_AssignmentModel):
    """One independent Bernoulli CT/TC order draw per unit-cycle.

    Every unit draws its own order independently of every other unit, which
    is what makes the unit the independent replicate: cross-unit replication
    exists for a unit-level reference distribution.
    """

    scheme: Literal["independent_bernoulli_order"] = "independent_bernoulli_order"
    probability_ct: float = Field(default=0.5, gt=0.0, lt=1.0)
    independence_unit: Literal["unit_cycle"] = "unit_cycle"


class SharedScheduleOrder(_AssignmentModel):
    """One independent Bernoulli CT/TC order draw per two-period block,
    shared by every unit in a fixed, complete unit roster.

    Unlike :class:`IndependentBernoulliOrder`, no unit draws its own order:
    the whole roster follows one realized order per block, so the
    independent replicate is the block, not the unit. A common roster
    duplicated uniformly adds no independent blocks and therefore cannot
    change the mean-unit retained-window effect scale or inferential sample
    size. The switchback panel path reduces the roster within each block
    and uses ``switchback_block_t`` with a ``block_t`` reference. At least
    two independent blocks are required; a one-unit roster is supported.
    """

    scheme: Literal["shared_schedule"] = "shared_schedule"
    probability_ct: float = Field(gt=0.0, lt=1.0)
    independence_unit: Literal["shared_block"] = "shared_block"


AssignmentSequence = Annotated[
    IndependentBernoulliOrder | SharedScheduleOrder, Field(discriminator="scheme")
]


class SwitchbackWindow(_AssignmentModel):
    """A washout/observation window with a declared residual-carryover order.

    ``carryover_order`` names additional observation steps, immediately
    after the declared washout, still assumed contaminated by the prior
    period and therefore discarded before the retained window used for
    inference. Retained steps satisfy ``step >= washout_steps +
    carryover_order``; the retained window's length is ``observation_steps -
    carryover_order``. ``carryover_order=0`` is the plain
    no-residual-carryover-after-washout case. Declaring an order is an
    assumption, not something a balance diagnostic can prove true.
    """

    washout_steps: int = Field(ge=0)
    observation_steps: int = Field(ge=1)
    carryover_order: int = 0

    @field_validator("washout_steps", "observation_steps", mode="before")
    @classmethod
    def _validate_steps(cls, v: object) -> object:
        return _reject_bool(v)

    @field_validator("carryover_order", mode="before")
    @classmethod
    def _validate_carryover_order_type(cls, v: object) -> object:
        if isinstance(v, bool):
            _raise("definition.assignment.reject_bool")
        if not isinstance(v, int):
            _raise("definition.assignment.carryover_order_type")
        return v

    @model_validator(mode="after")
    def _validate_carryover_order_range(self) -> SwitchbackWindow:
        # This bound is load-bearing, not cosmetic: it is what makes the
        # retained window (``observation_steps - carryover_order``)
        # nonempty for every constructible window.
        if not (0 <= self.carryover_order < self.observation_steps):
            _raise(
                "definition.assignment.carryover_order_range",
                carryover_order=self.carryover_order,
                observation_steps=self.observation_steps,
            )
        return self


class SwitchbackAssignment(_AssignmentModel):
    kind: Literal["switchback"] = "switchback"
    periods_per_cycle: Literal[2] = 2
    sequence: AssignmentSequence
    window: SwitchbackWindow


Assignment = Annotated[ParallelAssignment | SwitchbackAssignment, Field(discriminator="kind")]


@dataclass(frozen=True, slots=True)
class AssignmentSpec:
    """Portable replay inputs for an assignment ledger."""

    salt: str
    bucket_count: int
    allocation: tuple[tuple[str, tuple[int, ...]], ...]
    hash_function: str = "sha256"
    encoding: str = "utf-8"
    eligible_units: tuple[str, ...] | None = None
    schema_version: int = 1
    experiment: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            refuse_assignment_invalid("unsupported_schema_version", details=self.schema_version)
        if self.bucket_count < 1 or isinstance(self.bucket_count, bool):
            refuse_assignment_invalid("invalid_bucket_count", details=self.bucket_count)
        if self.hash_function != "sha256" or self.encoding != "utf-8":
            refuse_assignment_invalid("unsupported_hash_encoding")
        if not self.salt:
            refuse_assignment_invalid("missing_replay_salt")
        buckets = [bucket for _, selected in self.allocation for bucket in selected]
        if not self.allocation or len({arm for arm, _ in self.allocation}) != len(self.allocation):
            refuse_assignment_invalid("invalid_allocation_arms")
        if sorted(buckets) != list(range(self.bucket_count)):
            refuse_assignment_invalid("invalid_allocation_bucket_partition")
        if self.eligible_units is not None and len(set(self.eligible_units)) != len(
            self.eligible_units
        ):
            refuse_assignment_invalid("duplicate_eligible_unit")

    def to_json(self) -> str:
        """Encode a stable, versioned representation."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "experiment": self.experiment,
                "salt": self.salt,
                "bucket_count": self.bucket_count,
                "allocation": [[arm, list(buckets)] for arm, buckets in self.allocation],
                "hash_function": self.hash_function,
                "encoding": self.encoding,
                "eligible_units": self.eligible_units,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> AssignmentSpec:
        """Decode only the current schema and reject unknown fields."""
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            refuse_assignment_invalid("invalid_schema_json")
        if not isinstance(data, dict):
            refuse_assignment_invalid("invalid_schema_document")
        if data.get("schema_version") != 1:
            refuse_assignment_invalid(
                "unsupported_schema_version", details=data.get("schema_version")
            )
        allowed = {
            "schema_version",
            "experiment",
            "salt",
            "bucket_count",
            "allocation",
            "hash_function",
            "encoding",
            "eligible_units",
        }
        unknown = set(data) - allowed
        design_fields = unknown & {
            "randomization_law",
            "probabilities",
            "independence_grain",
            "carryover_order",
            "schedule",
        }
        if design_fields:
            refuse_assignment_invalid(
                "design_declaration_conflict",
                details=sorted(str(field) for field in design_fields),
            )
        if unknown:
            refuse_assignment_invalid(
                "unknown_schema_fields", details=sorted(str(field) for field in unknown)
            )
        try:
            data["allocation"] = tuple((arm, tuple(buckets)) for arm, buckets in data["allocation"])
            if data.get("eligible_units") is not None:
                data["eligible_units"] = tuple(data["eligible_units"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as error:
            refuse_assignment_invalid("invalid_schema_document", details=str(error))
        raise AssertionError("unreachable")

    def _bucket(self, unit: str) -> int:
        if not isinstance(unit, str) or not unit:
            refuse_assignment_invalid("invalid_unit_identity", details=unit)
        raw = json.dumps([self.salt, unit], ensure_ascii=False, separators=(",", ":")).encode(
            self.encoding
        )
        digest = hashlib.new(self.hash_function, raw).digest()
        return int.from_bytes(digest[:8], "big") % self.bucket_count

    def replay(
        self, units: Iterable[str], *, allow_eligible_subset: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Recreate assignments for the full eligible roster or an explicit subset."""
        selected = tuple(units)
        if len(set(selected)) != len(selected):
            refuse_assignment_invalid("duplicate_replay_unit")
        if self.eligible_units is not None:
            eligible = set(self.eligible_units)
            selected_set = set(selected)
            valid = (
                selected_set.issubset(eligible)
                if allow_eligible_subset
                else selected_set == eligible
            )
            if not valid:
                refuse_assignment_invalid("replay_eligibility_mismatch")
        bucket_to_arm = {bucket: arm for arm, buckets in self.allocation for bucket in buckets}
        return tuple(
            {
                "unit": unit,
                "experiment": self.experiment,
                "arm": bucket_to_arm[self._bucket(unit)],
                "period": None,
                "version": 1,
                "layer": None,
                "allocated_at": None,
                "exposed_at": None,
                "triggered_at": None,
            }
            for unit in selected
        )

    def verify(
        self, ledger: Iterable[Mapping[str, Any]], sample: Iterable[str] | None = None
    ) -> str:
        """Check replayed assignments, distinguishing a sample from full verification."""
        rows = tuple(ledger)
        try:
            identities = []
            for row in rows:
                unit = row["unit"]
                experiment = row["experiment"]
                version = row["version"]
                period = row["period"]
                arm = row["arm"]
                if (
                    not isinstance(unit, str)
                    or not isinstance(experiment, str)
                    or isinstance(version, bool)
                    or not isinstance(version, int)
                    or (period is not None and not isinstance(period, str))
                    or not isinstance(arm, str)
                ):
                    refuse_assignment_replay_mismatch(details="invalid ledger identity types")
                identities.append(((unit, experiment, version, period), arm))
            by_unit = {identity[0][0]: identity for identity in identities}
        except (KeyError, TypeError, ValueError) as error:
            refuse_assignment_replay_mismatch(details=f"invalid ledger identity: {error}")
        if len(by_unit) != len(rows):
            refuse_assignment_replay_mismatch(details="duplicate ledger unit identities")
        if (
            sample is None
            and self.eligible_units is not None
            and set(by_unit) != set(self.eligible_units)
        ):
            refuse_assignment_replay_mismatch(details="ledger does not match declared eligibility")
        units = tuple(sample) if sample is not None else tuple(sorted(by_unit))
        if any(unit not in by_unit for unit in units):
            refuse_assignment_replay_mismatch(details="sample contains an unmatched identity")
        for row in self.replay(units, allow_eligible_subset=sample is not None):
            actual_identity, actual_arm = by_unit[row["unit"]]
            expected_identity = (
                row["unit"],
                row["experiment"],
                row["version"],
                row["period"],
            )
            if actual_identity != expected_identity or actual_arm != row["arm"]:
                refuse_assignment_replay_mismatch(
                    details={
                        "unit": row["unit"],
                        "expected": expected_identity,
                        "actual": actual_identity,
                    }
                )
        return "sampled_check" if sample is not None else "verified_full"


__all__ = [
    "AssignmentSpec",
    "Assignment",
    "IndependentBernoulliOrder",
    "ParallelAssignment",
    "SharedScheduleOrder",
    "SwitchbackAssignment",
    "SwitchbackWindow",
]
