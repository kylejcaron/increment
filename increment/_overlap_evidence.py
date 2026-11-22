"""Physical inputs and bounded storage for overlap evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite, sqrt, ulp
from types import MappingProxyType
from typing import (
    Annotated,
    Literal,
    LiteralString,
    NamedTuple,
    NoReturn,
    Self,
    TypedDict,
    Unpack,
    cast,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PlainSerializer,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic.config import ExtraValues
from pydantic_core import InitErrorDetails, PydanticCustomError

from increment._canonical import canonical_json_bytes
from increment._overlap_refusals import (
    EVIDENCE_INVALID,
    FORMAT_INVALID,
    NUMERIC_RANGE,
    PLAN_INVALID,
)
from increment.errors import CodedError, CodedModel, refuse

CellId = Literal["00", "10", "01", "11"]
FactorId = Literal["A", "B"]
FeedRole = Literal["roster", "decisions", "deliveries", "outcomes"]
DecisionStatus = Literal["assigned", "held_out", "unallocated", "not_enrolled"]
GateDisposition = Literal[DecisionStatus, "other_selected"]
DeliveryCellId = Literal[CellId, "outside_target"]
OverlapIngress = Literal["frame", "warehouse"]

CELL_IDS: tuple[CellId, ...] = ("00", "10", "01", "11")
FACTOR_IDS: tuple[FactorId, ...] = ("A", "B")
FEED_ROLES: tuple[FeedRole, ...] = ("roster", "decisions", "deliveries", "outcomes")
DECISION_STATUSES: tuple[DecisionStatus, ...] = (
    "assigned",
    "held_out",
    "unallocated",
    "not_enrolled",
)
GATE_DISPOSITIONS: tuple[GateDisposition, ...] = (*DECISION_STATUSES, "other_selected")
GATE_KEYS = tuple((a, b) for a in GATE_DISPOSITIONS for b in GATE_DISPOSITIONS)
DELIVERY_KEYS = tuple(
    (cell, factor) for cell in (*CELL_IDS, "outside_target") for factor in FACTOR_IDS
)
OVERLAP_FORMAT = 1
MAX_EXACT_INTEGER = 2**53 - 1  # The canonical JSON integer range.


class OverlapColumn(NamedTuple):
    """One physical column, independent of dataframe or warehouse dtype names."""

    name: str
    kind: Literal["string", "timestamp", "boolean", "integer", "float"]
    nullable: bool = False


class OverlapRelation(NamedTuple):
    """Columns and unique key, with keys interpreted within a study."""

    key: tuple[str, ...]
    columns: tuple[OverlapColumn, ...]


_STUDY = OverlapColumn("study_id", "string")
_UNIT = OverlapColumn("unit_id", "string")
_TARGET = OverlapColumn("target", "string")
_REVISION = OverlapColumn("revision", "string")

OVERLAP_RELATIONS: Mapping[FeedRole, OverlapRelation] = MappingProxyType(
    {
        "roster": OverlapRelation(
            ("unit_id",),
            (
                _STUDY,
                _UNIT,
                OverlapColumn("entry_at", "timestamp"),
                OverlapColumn("eligible_a", "boolean"),
                OverlapColumn("eligible_b", "boolean"),
                OverlapColumn("eligibility_at_a", "timestamp"),
                OverlapColumn("eligibility_at_b", "timestamp"),
            ),
        ),
        "decisions": OverlapRelation(
            ("unit_id", "target"),
            (
                _STUDY,
                _UNIT,
                _TARGET,
                OverlapColumn("decided_at", "timestamp"),
                _REVISION,
                OverlapColumn("archive_sha256", "string"),
                OverlapColumn("selected_experiment", "string", True),
                OverlapColumn("status", "string"),
                OverlapColumn("arm", "string", True),
                OverlapColumn("assigned_weight", "integer", True),
            ),
        ),
        "deliveries": OverlapRelation(
            ("event_id",),
            (
                _STUDY,
                OverlapColumn("event_id", "string"),
                _UNIT,
                _TARGET,
                OverlapColumn("experiment", "string"),
                _REVISION,
                OverlapColumn("delivered_arm", "string"),
                OverlapColumn("delivered_at", "timestamp"),
            ),
        ),
        "outcomes": OverlapRelation(
            ("unit_id", "metric"),
            (
                _STUDY,
                _UNIT,
                OverlapColumn("metric", "string"),
                OverlapColumn("definition_sha256", "string"),
                OverlapColumn("window_start", "timestamp"),
                OverlapColumn("window_end", "timestamp"),
                OverlapColumn("value", "float", True),
                OverlapColumn("finalized_at", "timestamp", True),
            ),
        ),
    }
)
# Gate fields must be present for preassigned rosters; only both-eligible rows require values.
PREASSIGNED_GATE_COLUMNS = (
    OverlapColumn("gate_a", "boolean", True),
    OverlapColumn("gate_b", "boolean", True),
    OverlapColumn("gate_at_a", "timestamp", True),
    OverlapColumn("gate_at_b", "timestamp", True),
)

_SURROGATE = re.compile(r"[\ud800-\udfff]")
_INSTANT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


def _scalar(value: str) -> str:
    if _SURROGATE.search(value):
        raise PydanticCustomError("overlap_identity", "Expected Unicode scalar values")
    return value


def _ordered(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        raise PydanticCustomError("overlap_ordered", "Expected an ordered sequence")
    return value


def _utc_microsecond(value: object) -> datetime:
    try:
        if isinstance(value, str) and _INSTANT.fullmatch(value):
            value = datetime.fromisoformat(value)
        if (
            isinstance(value, datetime)
            and getattr(value, "nanosecond", 0) == 0
            and value.utcoffset() is not None
        ):
            utc = value.astimezone(UTC)
            return datetime(
                utc.year,
                utc.month,
                utc.day,
                utc.hour,
                utc.minute,
                utc.second,
                utc.microsecond,
                tzinfo=UTC,
            )
    except (TypeError, ValueError, OverflowError):
        pass
    raise PydanticCustomError("overlap_timestamp", "Expected an aware microsecond timestamp")


def _instant_json(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").removesuffix("+00:00") + "Z"


Identity = Annotated[str, StringConstraints(strict=True, min_length=1), AfterValidator(_scalar)]
Sha256 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
Count = Annotated[StrictInt, Field(ge=0, le=MAX_EXACT_INTEGER)]
Instant = Annotated[
    datetime, BeforeValidator(_utc_microsecond), PlainSerializer(_instant_json, when_used="json")
]
type Ordered[T] = Annotated[T, BeforeValidator(_ordered)]


class _JSONValidationOptions(TypedDict, total=False):
    strict: bool | None
    extra: ExtraValues | None
    context: object
    by_alias: bool | None
    by_name: bool | None


class OverlapModel(CodedModel, BaseModel):
    """Frozen values; validation exposes fields, not raw inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(PLAN_INVALID, field=field, reason=reason)

    def _validate_overlap(self) -> Self:
        return self

    @model_validator(mode="wrap")
    @classmethod
    def _checked(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        try:
            return handler(value)._validate_overlap()
        except ValidationError as exc:
            details = exc.errors(include_url=False, include_input=False)
            # Preserve the established priority of a coded child over an uncoded field.
            detail = next(
                (
                    item
                    for item in details
                    if isinstance(item.get("ctx", {}).get("error"), CodedError)
                ),
                details[0],
            )
            failure = detail.get("ctx", {}).get("error")
            kind = detail["type"]
            loc = detail["loc"]
        except CodedError as exc:
            failure, kind, loc = exc, "value_error", ()

        if not isinstance(failure, CodedError):
            field = loc[0] if loc and loc[0] in cls.model_fields else "model"
            reason = {"missing": "missing_field", "extra_forbidden": "unexpected_field"}.get(
                kind, "invalid_value"
            )
            try:
                cls._invalid_field(str(field), reason)
            except CodedError as exc:
                failure = exc
        if kind in {"extra_forbidden", "invalid_key"} and loc:
            loc = (*loc[:-1], "<unknown field>")
        # Only the schema path and kind survive; even a missing field may carry a whole record.
        redacted = InitErrorDetails(
            type=PydanticCustomError(
                cast(LiteralString, kind), "Invalid overlap input", {"error": failure}
            ),
            loc=loc,
            input=None,
        )
        raise ValidationError.from_exception_data(cls.__name__, [redacted], hide_input=True)

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **options: Unpack[_JSONValidationOptions]
    ) -> Self:
        # The native parser allows duplicate keys; check before retaining its JSON-mode validation.
        _decode_overlap_json(json_data)
        try:
            return super().model_validate_json(json_data, **options)
        except ValidationError:
            pass
        refuse(FORMAT_INVALID, field="payload", reason="invalid_json")

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: object) -> Self:
        return cls.model_validate(values)

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        data = {field: getattr(self, field) for field in type(self).model_fields}
        return type(self).model_validate({**data, **update})


class OverlapDeliveryCount(OverlapModel):
    """Per-cell and per-factor delivery totals without unit identifiers.

    Outside-target records count only unauthorized deliveries, never ordinary
    delivery to a unit legitimately assigned to one focal experiment alone.
    """

    cell: DeliveryCellId
    factor: FactorId
    n_without_delivery: Count
    n_matching: Count
    n_inconsistent: Count

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(EVIDENCE_INVALID, role="deliveries", reason=reason, count=1)


class _CellMoments[Key: str](OverlapModel):
    """Shared sufficient statistics for a complete assignment cell."""

    metric: Identity
    cell: Key
    n_assigned: Count
    n_observed: Count
    n_missing: Count
    n_pending: Count
    n_success: Count | None
    anchor: StrictFloat | None
    mean_delta: StrictFloat | None
    m2: StrictFloat | None

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(EVIDENCE_INVALID, role="cells", reason=reason, count=1)

    def _validate_overlap(self) -> Self:
        if self.n_observed + self.n_missing + self.n_pending != self.n_assigned:
            self._invalid_field("n_assigned", "count_mismatch")
        if self.n_success is not None and self.n_success > self.n_observed:
            self._invalid_field("n_success", "invalid_success_count")
        anchor, delta, m2 = self.anchor, self.mean_delta, self.m2
        if self.n_observed == 0 or self.n_observed != self.n_assigned:
            if anchor is not None or delta is not None or m2 is not None:
                self._invalid_field("anchor", "unusable_moments")
            return self
        if anchor is None or delta is None or m2 is None:
            self._invalid_field("anchor", "missing_moments")
        for quantity, value in (
            ("anchor", anchor),
            ("mean_delta", delta),
            ("m2", m2),
            ("mean", anchor + delta),
        ):
            if not isfinite(value):
                refuse(NUMERIC_RANGE, metric="unverified", quantity=quantity)
        if delta < 0 or m2 < 0 or ((delta == 0) != (m2 == 0)):
            self._invalid_field("m2", "inconsistent_moments")
        if self.n_observed == 1 and (delta != 0 or m2 != 0):
            self._invalid_field("m2", "singleton_moments")
        if delta > 0 and not _feasible_centered_ss(self.n_observed, delta, m2):
            self._invalid_field("m2", "inconsistent_moments")
        return self


class OverlapCell(_CellMoments[CellId]):
    """Observations and centered moments for a two-experiment assignment cell."""


def _feasible_centered_ss(n: int, delta: float, m2: float) -> bool:
    # The minimum anchor forces one zero offset. At a fixed mean residual,
    # the other offsets are either equal (min SS) or all but one zero (max SS).
    ratio = sqrt(m2) / delta
    # Compare in square-root scale so neither delta² nor the SS bounds overflow.
    # Allow rounding to the nearest SS even when its half ULP is subnormal.
    half_ulp_relative = 0.5 * (ulp(m2) / m2)
    lower = sqrt((n / (n - 1)) / (1 + half_ulp_relative))
    upper = sqrt((n * (n - 1)) / (1 - half_ulp_relative))
    # Mean and centered SS are independently computed by the reducers.
    slack = 1e-12
    return lower * (1 - slack) <= ratio <= upper * (1 + slack)


def overlap_binary_moments(n_observed: int, n_success: int) -> tuple[float, float, float]:
    """Return the minimum anchor, mean residual, and M2 for a complete binary cell."""
    if n_observed <= 0 or not 0 <= n_success <= n_observed:
        refuse(EVIDENCE_INVALID, role="cells", reason="invalid_success_count", count=1)
    failure = n_observed - n_success
    if failure == 0:
        return 1.0, 0.0, 0.0
    return 0.0, n_success / n_observed, n_success * failure / n_observed


def overlap_digest(
    kind: Literal["law", "plan", "family", "evidence", "result"], value: object
) -> str:
    """Hash canonical bytes under the declared overlap document kind."""
    digest = sha256(f"increment.overlap.v1\0{kind}\0".encode("ascii"))
    try:
        digest.update(canonical_json_bytes(value))
    except (ValueError, RecursionError):
        pass
    else:
        return digest.hexdigest()
    refuse(FORMAT_INVALID, field=kind, reason="not_canonicalizable")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            refuse(FORMAT_INVALID, field="payload", reason="duplicate_key")
        result[key] = item
    return result


def _nonfinite(_: str) -> NoReturn:
    refuse(FORMAT_INVALID, field="payload", reason="nonfinite_number")


def _decode_overlap_json(payload: str | bytes | bytearray) -> object:
    if not isinstance(payload, (str, bytes, bytearray)):
        refuse(FORMAT_INVALID, field="payload", reason="expected_json_text")
    try:
        text = payload if isinstance(payload, str) else payload.decode("utf-8")
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except CodedError:
        raise
    except (ValueError, RecursionError):
        pass
    refuse(FORMAT_INVALID, field="payload", reason="invalid_json")


def overlap_wire_payload(payload: str) -> dict[str, object]:
    """Parse a format-1 overlap object without losing duplicate keys."""
    if not isinstance(payload, str):
        refuse(FORMAT_INVALID, field="payload", reason="expected_json_text")
    data = _decode_overlap_json(payload)
    if not isinstance(data, dict):
        refuse(FORMAT_INVALID, field="payload", reason="expected_object")
    version = data.get("format")
    if type(version) is not int or version != OVERLAP_FORMAT:
        refuse(FORMAT_INVALID, field="format", reason="unsupported_version")
    return cast("dict[str, object]", data)
