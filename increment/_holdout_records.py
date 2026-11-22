"""Records and bindings for a registered persistent holdout comparison."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from operator import attrgetter
from typing import (
    Annotated,
    Literal,
    LiteralString,
    NamedTuple,
    NoReturn,
    NotRequired,
    Self,
    TypedDict,
    Unpack,
    cast,
)

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    ModelWrapValidatorHandler,
    SerializerFunctionWrapHandler,
    StrictBool,
    StringConstraints,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.config import ExtraValues
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError

from increment._immutable import _FrozenMapping
from increment.assignment import AllocationEvaluator, compile_allocation
from increment.errors import (
    MODEL_FIELD_REFUSALS,
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    refuse,
)

WINDOW_NOT_REGISTERED = RefusalSpec(
    "holdout.window_not_registered",
    InvalidRequestError,
    template=(
        "Study {study_id} did not originally register these terms for window {window_id}: "
        "{fields}. Use the original terms or describe the observed records without inference."
    ),
)
REGISTRATION_MISMATCH = RefusalSpec(
    "holdout.registration_mismatch",
    InvalidRequestError,
    template="The {field} evidence does not match holdout registration {study_id}.",
)


_SUBMICROSECOND = re.compile(r"[.,][0-9]{6}[0-9]*[1-9]")


def _timestamp(value: object) -> datetime:
    if isinstance(value, str):
        if _SUBMICROSECOND.search(value):
            raise PydanticCustomError(
                "datetime_type", "expected a timestamp representable to the microsecond"
            )
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise PydanticCustomError("datetime_type", "expected an aware timestamp") from None
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise PydanticCustomError("datetime_type", "expected an aware timestamp")
    try:
        return value.astimezone(UTC)
    except OverflowError:
        raise PydanticCustomError("datetime_type", "timestamp does not fit in UTC") from None


def _reference(value: object) -> object:
    if isinstance(value, str) and value == "latest":
        raise PydanticCustomError("reference_type", "expected a pinned source reference")
    return value


def _v1_protocol(value: object) -> object:
    if type(value) is not int:
        raise PydanticCustomError("int_type", "expected an integer protocol version")
    return value


_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _unicode_scalar(value: object) -> object:
    if isinstance(value, str) and _SURROGATE.search(value):
        raise PydanticCustomError("string_type", "expected a string of Unicode scalar values")
    return value


ID = Annotated[str, StringConstraints(strict=True, min_length=1), BeforeValidator(_unicode_scalar)]
Ref = Annotated[ID, BeforeValidator(_reference)]
Timestamp = Annotated[datetime, BeforeValidator(_timestamp)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Probability = Annotated[Finite, Field(ge=0, le=1)]
Count = Annotated[int, Field(strict=True, ge=0)]
BasisPoints = Annotated[int, Field(strict=True, ge=0, le=10_000)]
Group = Literal["held_out", "eligible"]
Feed = Literal["decision", "delivery"]
MembershipSource = Literal["sdk", "external", "reduced"]
LookRequest = Literal["describe", "fixed_horizon", "sequential"]

# At these midpoints, converting either arm's exact decimal probability to
# float rounds to 1. Keep the comparisons independent of Decimal context.
P0 = Annotated[
    Decimal,
    Field(
        gt=Decimal.from_float(2.0**-54),
        lt=Decimal("0.999999999999999944488848768742172978818416595458984375"),
        allow_inf_nan=False,
    ),
]


class _HoldoutRecord(CodedModel, BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class _JSONValidationOptions(TypedDict, total=False):
    strict: bool | None
    extra: ExtraValues | None
    context: object
    by_alias: bool | None
    by_name: bool | None


class _PrivateEvidence(_HoldoutRecord):
    """Keep untrusted observation values out of schema diagnostics."""

    model_config = ConfigDict(hide_input_in_errors=True)

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **options: Unpack[_JSONValidationOptions]
    ) -> Self:
        try:
            return super().model_validate_json(json_data, **options)
        except ValidationError as error:
            # The parser runs before any model validator and reports the raw input.
            details = error.errors(include_input=False, include_context=False, include_url=False)
            if len(details) != 1 or details[0]["loc"]:
                raise
            kind = details[0]["type"]
            if kind == "json_invalid":
                detail = InitErrorDetails(
                    type=kind, loc=(), input=None, ctx={"error": "malformed or truncated input"}
                )
            elif kind == "json_type":
                detail = InitErrorDetails(type=kind, loc=(), input=None)
            else:
                raise
            redacted = ValidationError.from_exception_data(error.title, [detail], hide_input=True)
        raise redacted

    @model_validator(mode="wrap")
    @classmethod
    def _protect_evidence(
        cls, data: object, validate: ModelWrapValidatorHandler[_PrivateEvidence]
    ) -> _PrivateEvidence:
        try:
            return validate(data)
        except ValidationError as error:
            # Missing fields can carry an entire enclosing record as their input.
            redacted = ValidationError.from_exception_data(
                error.title,
                [_schema_detail(detail) for detail in error.errors(include_url=False)],
                hide_input=True,
            )
        raise redacted


_UNKNOWN_FIELD = "<unknown field>"


def _private_location(loc: tuple[str | int, ...], *, extra: bool) -> tuple[str | int, ...]:
    safe_loc = tuple(
        "<key>" if index and loc[index - 1] == "context" else part for index, part in enumerate(loc)
    )
    return (*safe_loc[:-1], _UNKNOWN_FIELD) if extra and safe_loc else safe_loc


def _schema_detail(detail: ErrorDetails) -> InitErrorDetails:
    kind = detail["type"]
    loc = _private_location(detail["loc"], extra=kind == "extra_forbidden")
    context = detail.get("ctx", {})
    if "error" in context:
        embedded = context["error"]
        if isinstance(embedded, InvalidRequestError):
            safe_error = _private_coded_error(embedded, loc)
            # The coded error now owns this path; a parent supplies only its own prefix.
            loc = ()
        else:
            safe_error = ValueError("Invalid holdout evidence")
        return InitErrorDetails(type=kind, loc=loc, input=None, ctx={"error": safe_error})
    error_type = PydanticCustomError(
        cast(LiteralString, kind), "{message}", {**context, "message": detail["msg"]}
    )
    return InitErrorDetails(type=error_type, loc=loc, input=None)


def _private_coded_error(
    original: InvalidRequestError, prefix: tuple[str | int, ...]
) -> ValueError:
    # Nested CodedModel validation may already have translated a field failure.
    spec = MODEL_FIELD_REFUSALS.get(original.code)
    if spec is None:
        return InvalidRequestError(
            "Invalid holdout evidence", code=original.code, context={"loc": prefix}
        )
    original_loc = cast("tuple[str | int, ...]", original.context["loc"])
    extra = original.code == "model.field.unknown"
    loc = (*prefix, *_private_location(original_loc, extra=extra))
    safe_context = {
        "model": original.context["model"],
        "field": _UNKNOWN_FIELD if extra else original.context["field"],
        "loc": loc,
        "value": None,
        "constraint": original.context["constraint"],
    }
    return spec.error_type(spec.render(**safe_context), code=spec.code, context=safe_context)


class Terms(_HoldoutRecord):
    """One outcome, cohort freeze, observation window and final data cut."""

    population_id: ID
    population_version: ID
    outcome_id: ID
    outcome_version: ID
    outcome_unit: ID
    freeze_at: Timestamp
    start: Timestamp
    end: Timestamp
    final_as_of: Timestamp


class Component(_HoldoutRecord):
    """A launch and its decision target in the registered scope."""

    launch_id: ID
    target: ID
    experiment_key: ID | None = None


class HoldoutStudy(_HoldoutRecord):
    """The original holdout registration, before units entered."""

    study_id: ID
    registration_ref: Ref
    registered_at: Timestamp
    population_id: ID
    population_version: ID
    eligibility_rule_ref: Ref
    common_entry_ref: Ref
    entry_basis: Literal["pretreatment_first_entry", "post_assignment_filter", "unknown"]
    assignment_unit: ID
    law_id: ID
    law_ref: Ref
    allocation_scheme: ID
    no_interference: StrictBool | None = None
    design_ref: Ref | None = None
    holdout_key: ID
    holdout_probability: P0
    baseline_version: ID
    policy_id: ID
    policy_version: ID
    scope_kind: Literal["feature", "global"]
    components: Annotated[tuple[Component, ...], Field(min_length=1)]
    registered_window: Terms | None = None

    @model_validator(mode="after")
    def _unique_scope(self) -> HoldoutStudy:
        launch_ids = {component.launch_id for component in self.components}
        if len(launch_ids) != len(self.components) or (
            self.scope_kind == "feature" and len(self.components) != 1
        ):
            _mismatch(self, "components")
        return self

    @field_serializer("components", mode="wrap", when_used="json")
    def _ordered_components(
        self,
        value: tuple[Component, ...],
        serialize: SerializerFunctionWrapHandler,
        info: FieldSerializationInfo,
    ) -> object:
        return _serialize_rows(value, serialize, info, "launch_id", "target")


class HoldoutWindow(_HoldoutRecord):
    """A separately recorded confirmation of the original window terms."""

    study_id: ID
    registration_ref: Ref
    window_id: ID
    locked_at: Timestamp
    lock_ref: Ref
    terms: Terms


class Entry(_PrivateEvidence):
    """A common-entry census observation independent of assignment logging."""

    entry_id: ID
    unit_id: ID
    entered_at: Timestamp
    eligible: StrictBool
    rule_ref: Ref
    available_at: Timestamp


class SDKMembership(_HoldoutRecord):
    """The unchanged v1 membership payload stored as an immutable record."""

    protocol: Annotated[Literal[1], BeforeValidator(_v1_protocol)]
    revision: ID
    holdout: ID
    group: Group
    group_bps: BasisPoints


class ExternalMembership(_HoldoutRecord):
    """A contemporaneous assignment from an external randomization system."""

    holdout_key: ID
    revision: ID
    group: Group
    holdout_probability: P0
    assignment_ref: Ref


class MembershipEvent(_PrivateEvidence):
    """A membership payload tied to a specific common entry and assignment law."""

    event_id: ID
    entry_id: ID
    unit_id: ID
    assigned_at: Timestamp
    available_at: Timestamp
    law_id: ID
    payload: SDKMembership | ExternalMembership

    @field_validator("payload", mode="before")
    @classmethod
    def _payload(cls, value: object) -> object:
        # Select the wire shape before validation so a malformed SDK proof
        # cannot be interpreted as a malformed external proof, or vice versa.
        if isinstance(value, Mapping):
            model = (
                SDKMembership
                if any(k in value for k in ("protocol", "holdout", "group_bps"))
                else ExternalMembership
            )
            return model.model_validate(value)
        if isinstance(value, (SDKMembership, ExternalMembership)):
            return type(value).model_validate(value)
        raise PydanticCustomError("membership_type", "expected an SDK or external record")


class ExternalPolicy(_HoldoutRecord):
    """The holdout scope and law archived by the external assignment system."""

    holdout_key: ID
    targets: Annotated[tuple[ID, ...], Field(min_length=1)]
    assignment_unit: ID
    law_id: ID
    holdout_probability: P0

    @field_validator("targets")
    @classmethod
    def _unique_targets(cls, targets: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(targets)) != len(targets):
            raise PydanticCustomError("targets_type", "expected distinct archived targets")
        return tuple(sorted(targets))


class ConfigRevision(_HoldoutRecord):
    """An archived allocation configuration and its effective interval."""

    revision: ID
    archive_ref: Ref
    effective_from: Timestamp
    effective_to: Timestamp
    sdk_json: ID | None = None
    external_policy: ExternalPolicy | None = None

    @model_validator(mode="after")
    def _one_archive(self) -> ConfigRevision:
        if (self.sdk_json is None) == (self.external_policy is None):
            _invalid_field("ConfigRevision", "external_policy", "exactly one source archive")
        return self


class Alias(_PrivateEvidence):
    """A source-attested successor for an assignment-unit identity."""

    old_id: ID
    new_id: ID
    effective_at: Timestamp
    available_at: Timestamp
    identity_ref: Ref


class Phase(_HoldoutRecord):
    """An actual rollout version and offer probability over an interval."""

    launch_id: ID
    target: ID
    feature_version: ID
    rollout_revision: ID
    config_revision: ID
    from_at: Timestamp
    to_at: Timestamp
    eligible_offer_probability: Probability
    baseline_version: ID
    actual_ref: Ref
    available_at: Timestamp


class Coverage(_HoldoutRecord):
    """A logging-coverage attestation for one feed, launch and original group."""

    kind: Feed
    group: Group
    launch_id: ID
    from_at: Timestamp
    to_at: Timestamp
    source_ref: Ref
    available_at: Timestamp


class DecisionEvent(_PrivateEvidence):
    """An assignment decision, not proof of delivery."""

    event_id: ID
    unit_id: ID
    launch_id: ID
    target: ID
    config_revision: ID
    at: Timestamp
    available_at: Timestamp
    holdout_blocked: StrictBool
    nonbaseline_assigned: StrictBool


class Delivery(_PrivateEvidence):
    """An actual delivery, separate from the assignment decision."""

    event_id: ID
    unit_id: ID
    launch_id: ID
    target: ID
    feature_version: ID
    rollout_revision: ID
    at: Timestamp
    available_at: Timestamp
    nonbaseline: StrictBool


class Outcome(_PrivateEvidence):
    """A whole-window value or an explicit observation failure for one unit."""

    unit_id: ID
    window_id: ID
    outcome_version: ID
    start: Timestamp
    end: Timestamp
    observed_through: Timestamp
    available_at: Timestamp
    state: Literal["measured", "censored", "missing"]
    value: Finite | None = None
    reason: ID | None = None

    @model_validator(mode="after")
    def _value_matches_state(self) -> Outcome:
        if self.state == "measured":
            if self.value is None:
                _invalid_field("Outcome", "value", "a finite measurement")
        elif self.value is not None:
            _invalid_field("Outcome", "value", "null for an unmeasured outcome")
        elif self.reason is None:
            _invalid_field("Outcome", "reason", "a reason for an unmeasured outcome")
        return self


class HoldoutSnapshot(_PrivateEvidence):
    """An as-of evidence export explicitly bound to one registration and window."""

    study_id: ID
    registration_ref: Ref
    window_id: ID
    snapshot_ref: Ref
    as_of: Timestamp
    membership_source: MembershipSource
    entry_source_ref: Ref | None = None
    entry_complete_through: Timestamp | None = None
    history_ref: Ref | None = None
    outcome_source_ref: Ref | None = None
    entries: tuple[Entry, ...] = ()
    memberships: tuple[MembershipEvent, ...] = ()
    configs: tuple[ConfigRevision, ...] = ()
    aliases: tuple[Alias, ...] = ()
    phases: tuple[Phase, ...] = ()
    coverage: tuple[Coverage, ...] = ()
    decisions: tuple[DecisionEvent, ...] = ()
    deliveries: tuple[Delivery, ...] = ()
    outcomes: tuple[Outcome, ...] = ()

    @field_serializer(
        "entries",
        "memberships",
        "configs",
        "aliases",
        "phases",
        "coverage",
        "decisions",
        "deliveries",
        "outcomes",
        mode="wrap",
        when_used="json",
    )
    def _ordered_rows(
        self,
        rows: tuple[_HoldoutRecord, ...],
        serialize: SerializerFunctionWrapHandler,
        info: FieldSerializationInfo,
    ) -> object:
        return _serialize_rows(rows, serialize, info, *_ROW_KEYS[info.field_name])


_ROW_KEYS = {
    "entries": ("entry_id",),
    "memberships": ("event_id",),
    "configs": ("revision", "effective_from"),
    "aliases": ("old_id", "effective_at", "new_id"),
    "phases": ("launch_id", "from_at", "to_at", "rollout_revision"),
    "coverage": ("launch_id", "group", "kind", "from_at", "to_at", "source_ref"),
    "decisions": ("event_id",),
    "deliveries": ("event_id",),
    "outcomes": ("unit_id", "window_id", "start", "end"),
}


def _serialize_rows(
    rows: tuple[_HoldoutRecord, ...],
    serialize: SerializerFunctionWrapHandler,
    info: FieldSerializationInfo,
    *keys: str,
) -> object:
    if info.include is None and info.exclude is None:
        rows = tuple(sorted(rows, key=attrgetter(*keys)))
    return serialize(rows)


def _invalid_field(model: str, field: str, constraint: str) -> NoReturn:
    spec = MODEL_FIELD_REFUSALS["model.field.type"]
    context = {
        "model": model,
        "field": field,
        "loc": (field,),
        "value": None,
        "constraint": constraint,
    }
    error = spec.error_type(spec.render(**context), code=spec.code, context=context)
    # An after-validator can run outside the privacy wrap. Supply the cause
    # explicitly so Pydantic cannot attach the entire invalid record as input.
    raise ValidationError.from_exception_data(
        model,
        [InitErrorDetails(type="value_error", loc=(), input=None, ctx={"error": error})],
        hide_input=True,
    )


def _mismatch(study: HoldoutStudy, field: str) -> NoReturn:
    refuse(REGISTRATION_MISMATCH, study_id=study.study_id, field=field)


@dataclass(frozen=True, slots=True)
class RegistrationFinding:
    """An unregistered comparison that can still be described without inference."""

    study_id: str
    window_id: str
    fields: tuple[str, ...]

    @property
    def code(self) -> str:
        return WINDOW_NOT_REGISTERED.code

    @property
    def context(self) -> Mapping[str, object]:
        return _FrozenMapping(
            {"study_id": self.study_id, "window_id": self.window_id, "fields": self.fields}
        )

    def raise_error(self) -> NoReturn:
        """Refuse inference or planning for the unregistered comparison."""
        refuse(WINDOW_NOT_REGISTERED, **self.context)


def _check_header(study: HoldoutStudy, record: HoldoutWindow | HoldoutSnapshot) -> None:
    if record.study_id != study.study_id:
        _mismatch(study, "study_id")
    if record.registration_ref != study.registration_ref:
        _mismatch(study, "registration_ref")


def _check_population(study: HoldoutStudy, terms: Terms) -> None:
    if terms.population_id != study.population_id:
        _mismatch(study, "population_id")
    if terms.population_version != study.population_version:
        _mismatch(study, "population_version")


def _check_lock_clock(study: HoldoutStudy, window: HoldoutWindow) -> None:
    if not study.registered_at <= window.locked_at < window.terms.freeze_at:
        _mismatch(study, "locked_at")


def _check_window_clock(study: HoldoutStudy, terms: Terms) -> None:
    if terms.freeze_at > terms.start:
        _mismatch(study, "freeze_at")
    if terms.start >= terms.end:
        _mismatch(study, "end")
    if terms.end > terms.final_as_of:
        _mismatch(study, "final_as_of")


def check_registration(study: HoldoutStudy, window: HoldoutWindow) -> RegistrationFinding | None:
    """Bind the lock to independently registered terms, without moving assignment dates."""
    _check_header(study, window)
    _check_population(study, window.terms)
    _check_lock_clock(study, window)
    original = study.registered_window
    if original is None:
        _check_window_clock(study, window.terms)
        return RegistrationFinding(study.study_id, window.window_id, ("registered_window",))
    _check_population(study, original)
    _check_window_clock(study, original)
    fields = tuple(
        field
        for field in sorted(Terms.model_fields)
        if getattr(original, field) != getattr(window.terms, field)
    )
    if fields:
        return RegistrationFinding(study.study_id, window.window_id, fields)
    return None


def require_registration(study: HoldoutStudy, window: HoldoutWindow) -> None:
    """Require the original comparison for planning or inference."""
    finding = check_registration(study, window)
    if finding is not None:
        finding.raise_error()


def check_snapshot_binding(
    study: HoldoutStudy, window: HoldoutWindow, snapshot: HoldoutSnapshot
) -> None:
    """Check the evidence header without traversing its rows."""
    _check_header(study, window)
    _check_header(study, snapshot)
    if snapshot.window_id != window.window_id:
        _mismatch(study, "window_id")


@dataclass(frozen=True, slots=True)
class ConfigIndex:
    """Immutable activation intervals by archived revision, ordered by effective time."""

    activations: _FrozenMapping[str, tuple[ConfigRevision, ...]]

    def at(self, revision: str, at: datetime) -> ConfigRevision | None:
        """Find the named revision at its actual event time, without filling gaps."""
        spans = self.activations.get(revision, ())
        index = bisect_right(spans, at, key=attrgetter("effective_from")) - 1
        if index >= 0 and at < spans[index].effective_to:
            return spans[index]
        return None


def index_configs(study: HoldoutStudy, snapshot: HoldoutSnapshot) -> ConfigIndex:
    """Index each revision's nonoverlapping activations without changing its archive."""
    by_revision: dict[str, list[ConfigRevision]] = {}
    by_reference: dict[str, ConfigRevision] = {}
    for config in snapshot.configs:
        spans = by_revision.setdefault(config.revision, [])
        if spans and not _same_archive(spans[0], config):
            _mismatch(study, "configs")
        if not _same_archive(by_reference.setdefault(config.archive_ref, config), config):
            _mismatch(study, "configs")
        if config.effective_from >= config.effective_to:
            _mismatch(study, "effective_to")
        spans.append(config)
    return ConfigIndex(
        _FrozenMapping(
            {
                revision: _disjoint_spans(study, spans)
                for revision, spans in sorted(by_revision.items())
            }
        )
    )


def _same_archive(first: ConfigRevision, second: ConfigRevision) -> bool:
    return (
        first.revision == second.revision
        and first.archive_ref == second.archive_ref
        and first.sdk_json == second.sdk_json
        and first.external_policy == second.external_policy
    )


def _disjoint_spans(study: HoldoutStudy, spans: list[ConfigRevision]) -> tuple[ConfigRevision, ...]:
    ordered: list[ConfigRevision] = []
    for span in sorted(spans, key=attrgetter("effective_from", "effective_to")):
        if ordered and span.effective_from < ordered[-1].effective_to:
            previous = ordered[-1]
            if (
                span.effective_from == previous.effective_from
                and span.effective_to == previous.effective_to
            ):
                continue
            _mismatch(study, "configs")
        ordered.append(span)
    return tuple(ordered)


class _SDKHoldout(TypedDict):
    key: str
    held_out_bps: int | float
    targets: list[str]


class _SDKRange(TypedDict):
    experiment: str


class _SDKLayer(TypedDict):
    key: str
    ranges: list[_SDKRange]


class _SDKConfig(TypedDict):
    revision: str
    holdouts: NotRequired[list[_SDKHoldout]]
    layers: list[_SDKLayer]


def _unique_json_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate object members")
    return result


def _compiled_archive(config: ConfigRevision) -> tuple[AllocationEvaluator, _SDKConfig]:
    raw: object = None
    if config.sdk_json is not None:
        try:
            raw = json.loads(config.sdk_json, object_pairs_hook=_unique_json_object)
        except (ValueError, RecursionError):
            pass
    evaluator = compile_allocation(raw)
    return evaluator, cast("_SDKConfig", raw)


def parse_sdk_archive(config: ConfigRevision) -> AllocationEvaluator:
    """Compile the actual archived SDK JSON without reconstructing any records."""
    evaluator, _ = _compiled_archive(config)
    return evaluator


def check_config_binding(
    study: HoldoutStudy, snapshot: HoldoutSnapshot, config: ConfigRevision
) -> AllocationEvaluator | ExternalPolicy:
    """Validate an archive against the registered law, key, scope and rate."""
    if config.effective_from >= config.effective_to:
        _mismatch(study, "effective_to")
    if snapshot.membership_source == "sdk" and config.sdk_json is not None:
        return _check_sdk_binding(study, config)
    if snapshot.membership_source == "external" and config.external_policy is not None:
        policy = config.external_policy
        if policy.assignment_unit != study.assignment_unit:
            _mismatch(study, "assignment_unit")
        if policy.law_id != study.law_id:
            _mismatch(study, "law_id")
        _check_policy(study, policy.holdout_key, policy.targets, policy.holdout_probability)
        return policy
    _mismatch(study, "membership_source")


def _check_sdk_binding(study: HoldoutStudy, config: ConfigRevision) -> AllocationEvaluator:
    evaluator, parsed = _compiled_archive(config)
    if parsed["revision"] != config.revision:
        _mismatch(study, "revision")
    for holdout in parsed.get("holdouts", []):
        if holdout["key"] == study.holdout_key:
            probability = Decimal(f"{int(holdout['held_out_bps'])}e-4")
            _check_policy(study, holdout["key"], holdout["targets"], probability)
            _check_sdk_components(study, parsed["layers"])
            return evaluator
    _mismatch(study, "holdout_key")


def _check_sdk_components(study: HoldoutStudy, layers: list[_SDKLayer]) -> None:
    archived = {layer["key"]: {span["experiment"] for span in layer["ranges"]} for layer in layers}
    registered: dict[str, set[str | None]] = {}
    for component in study.components:
        registered.setdefault(component.target, set()).add(component.experiment_key)
    for target, experiments in registered.items():
        if target in archived:
            if experiments != archived[target]:
                _mismatch(study, "components")
        elif not experiments <= {None, target}:
            _mismatch(study, "components")


def _check_policy(
    study: HoldoutStudy, key: str, targets: tuple[str, ...] | list[str], probability: Decimal
) -> None:
    if key != study.holdout_key:
        _mismatch(study, "holdout_key")
    if probability != study.holdout_probability:
        _mismatch(study, "holdout_probability")
    if set(targets) != {component.target for component in study.components}:
        _mismatch(study, "components")


ReasonScalar = str | int | Finite | bool | Timestamp | None
ReasonValue = ReasonScalar | tuple[ReasonScalar, ...]


class HoldoutReason(_HoldoutRecord):
    """A stable reason and its associated evidence context."""

    code: ID
    context: Mapping[str, ReasonValue] = Field(default_factory=dict, validate_default=True)

    @field_validator("context")
    @classmethod
    def _freeze_context(cls, value: Mapping[str, ReasonValue]) -> Mapping[str, ReasonValue]:
        return _FrozenMapping(dict(sorted(value.items())))

    @field_serializer("context")
    def _context_dict(self, value: Mapping[str, ReasonValue]) -> dict[str, ReasonValue]:
        return dict(value)


class RosterMember(NamedTuple):
    """The original assignment for an included common entrant."""

    original_id: ID
    entered_at: Timestamp
    assigned_at: Timestamp
    group: Group
    assignment_revision: ID


class IdentitySpan(_PrivateEvidence):
    """The interval over which an observed ID identifies one original unit."""

    unit_id: ID
    original_id: ID
    from_at: Timestamp
    to_at: Timestamp | None = None


class ArmCounts(_HoldoutRecord):
    """Assigned units by outcome observation state."""

    assigned: Count = 0
    measured: Count = 0
    pending: Count = 0
    censored: Count = 0
    missing: Count = 0


class RosterEvidence(_PrivateEvidence):
    """The original roster and its certified outcome evidence."""

    members: tuple[RosterMember, ...]
    identity: tuple[IdentitySpan, ...]
    values: tuple[tuple[ID, Finite], ...]
    held_out: ArmCounts
    eligible: ArmCounts
    n_eligible_entries: Count
    n_ineligible_entries: Count
    n_unclassified: Count
    n_after_freeze: Count
    reasons: tuple[HoldoutReason, ...] = ()


class RolloutFinding(_HoldoutRecord):
    """Recorded units and violations for an original group, launch and feed."""

    launch_id: ID
    group: Group
    kind: Feed
    n_assigned: Count
    n_units_recorded: Count
    n_nonbaseline: Count
    n_violating: Count
    coverage_status: Literal["covered", "partial", "unknown"]
    finding: Literal["recorded_violation", "none_recorded", "unknown"]


class UnmatchedRolloutEvent(_HoldoutRecord):
    """An event that could not be assigned to an applicable rollout interval."""

    kind: Feed
    event_id: ID
    reason: ID


class RolloutAudit(_HoldoutRecord):
    """Actual implementation history and separate decision and delivery findings."""

    phases: tuple[Phase, ...]
    history_ref: Ref | None
    history_digest: ID | None
    findings: tuple[RolloutFinding, ...]
    unmatched_events: tuple[UnmatchedRolloutEvent, ...] = ()
    reasons: tuple[HoldoutReason, ...] = ()


class HoldoutPrecision(_HoldoutRecord):
    """Conditional precision for the projected original cohort at maturity."""

    window_id: ID
    start: Timestamp
    end: Timestamp
    assignment_unit: ID
    outcome_unit: ID
    p0: P0
    projected_assignments: Count
    projected_by_freeze: Count
    n_assigned_held_out: Count
    n_assigned_eligible: Count
    n_mature_held_out: Count
    n_mature_eligible: Count
    variance_held_out: Nonnegative
    variance_eligible: Nonnegative
    effect: Finite
    alpha: Probability = 0.05
    standard_error: Nonnegative | None
    df: Nonnegative | None
    interval_width: Nonnegative | None
    power: Probability | None
    basis: Literal["conditional_welch_approximation"] = "conditional_welch_approximation"
    reason_code: ID | None = None
    reason: Literal["arm_too_small", "zero_variance"] | None = None
