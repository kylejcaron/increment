"""The unit-day artifact contract: manifest, extension catalog, digests.

Moved out of `increment.semantics.models` (S4); `models.py` re-exports
every public name here for one release (D6), deleted next release.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Annotated, Any, ClassVar, Literal, cast
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from increment._canonical import canonical_json_bytes
from increment.errors import InvalidRequestError, raiser, refusals
from increment.semantics.models import (
    _Base,
    _definition_refusal,
    _ExplicitOnlyFields,
    _validate_day_boundary,
)

# ---------------------------------------------------------------------------
# Unit-day artifact contract: snapshot and provenance models. The query layer
# owns relation reads and writes and consumes this vocabulary.

_ARTIFACT_DIGEST_ROOT = b"increment.unit-day-artifact\x00v1\x00"
_ARTIFACT_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_ARTIFACT_CATALOG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$-]*$")
_ARTIFACT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_REFUSALS = refusals(
    InvalidRequestError,
    {
        "artifact.request.trigger_metric_scope": (
            "trigger_measure_stats must name exactly one metric"
        ),
    },
)
_raise_request = raiser(_REQUEST_REFUSALS)


RelationRole = Literal[
    "exposures",
    "measure_stats",
    "breakout_dimension",
    "factor_dimension",
    "cluster_identity",
    "cuped_preperiod",
    "assignment_counts",
    "trigger_population",
    "trigger_measure_stats",
    "encouragement_uptake",
    "site_volume",
    "unit_covariate",
    "unit_covariate_level",
]

_EXTENSION_KIND_RANK = {
    "breakout_dimension": 0,
    "factor_dimension": 1,
    "cluster_identity": 2,
    "cuped_preperiod": 3,
    "assignment_counts": 4,
    "trigger_population": 5,
    "trigger_measure_stats": 6,
    "encouragement_uptake": 7,
    "site_volume": 8,
    "unit_covariate": 9,
    "unit_covariate_level": 10,
}


def _artifact_scalar(value: str, field_name: str) -> str:
    """Validate an exact, non-empty UTF-8 scalar used in artifact identity."""
    if not isinstance(value, str) or not value:
        _definition_refusal(
            "definition.non_empty_string",
            f"{field_name} must be a non-empty string",
            field_name=field_name,
        )
    if value != value.strip():
        _definition_refusal(
            "definition.surrounding_whitespace",
            f"{field_name} must not have surrounding whitespace",
            field_name=field_name,
        )
    if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
        _definition_refusal(
            "definition.contain_control_characters",
            f"{field_name} must not contain control characters",
            field_name=field_name,
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _definition_refusal(
            "definition.utf",
            f"{field_name} must be valid UTF-8",
            field_name=field_name,
        )
    return value


def _artifact_identifier(value: str, field_name: str) -> str:
    value = _artifact_scalar(value, field_name)
    pattern = _ARTIFACT_CATALOG_RE if field_name == "catalog" else _ARTIFACT_IDENTIFIER_RE
    if not pattern.fullmatch(value):
        _definition_refusal(
            "definition.single_sql_identifier",
            f"{field_name} must be a single SQL identifier "
            "(letters, digits, '_', '$', '-' in catalogs; no quoting or separators)",
            field_name=field_name,
        )
    return value


def _artifact_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_SHA256_RE.fullmatch(value):
        _definition_refusal(
            "definition.lowercase_64_hex",
            f"{field_name} must be lowercase 64-hex SHA-256",
            field_name=field_name,
        )
    return value


def _artifact_json_value(value: Any) -> Any:
    """Map model values to the deterministic JSON scalar vocabulary."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            _definition_refusal(
                "definition.artifact_datetimes_timezone",
                "artifact datetimes must be timezone-aware",
            )
        normalized = value.astimezone(UTC)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _artifact_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_artifact_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            _definition_refusal(
                "definition.artifact_json_values",
                "artifact JSON values must be finite",
            )
        return 0.0 if value == 0.0 else value
    return value


def _artifact_canonical_json_bytes(value: Any) -> bytes:
    """Use the format-1 RFC8785-compatible serializer for every hash."""
    return canonical_json_bytes(value)


def _artifact_digest(domain: bytes, payload: bytes) -> str:
    return hashlib.sha256(domain + payload).hexdigest()


def _artifact_context_digest(canonical_json: str) -> str:
    return _artifact_digest(
        _ARTIFACT_DIGEST_ROOT + b"context\x00",
        canonical_json.encode("utf-8"),
    )


def _artifact_manifest_body(value: UnitDayArtifactManifest) -> bytes:
    return _artifact_canonical_json_bytes(
        value.model_dump(mode="python", by_alias=True, exclude={"manifest_sha256"})
    )


def _artifact_manifest_digest(value: UnitDayArtifactManifest) -> str:
    return _artifact_digest(
        _ARTIFACT_DIGEST_ROOT + b"manifest\x00",
        _artifact_manifest_body(value),
    )


def _validate_json_payload(value: str, field_name: str) -> str:
    value = _artifact_scalar(value, field_name)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        _definition_refusal(
            "definition.json",
            f"{field_name} must be valid JSON",
            field_name=field_name,
        )
    if not isinstance(parsed, dict):
        _definition_refusal(
            "definition.encode_json_object",
            f"{field_name} must encode a JSON object",
            field_name=field_name,
        )
    if _artifact_canonical_json_bytes(parsed).decode("utf-8") != value:
        _definition_refusal(
            "definition.already_canonical_json",
            f"{field_name} must already be canonical JSON",
            field_name=field_name,
        )
    return value


class _ArtifactBase(_Base):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )


class RelationLocator(_ArtifactBase):
    """A namespace-qualified relation name; never a backend SQL fragment."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    catalog: str | None = Field(default=None, max_length=255)
    schema_name: str | None = Field(default=None, alias="schema", max_length=255)
    name: str = Field(min_length=1, max_length=255)

    @field_validator("catalog", "schema_name")
    @classmethod
    def _namespace_identifier(cls, value: str | None, info) -> str | None:
        if value is not None:
            return _artifact_identifier(value, info.field_name)
        return value

    @field_validator("name")
    @classmethod
    def _relation_identifier(cls, value: str) -> str:
        return _artifact_identifier(value, "name")


class UnitDayArtifactRef(_ArtifactBase):
    """Caller-pinned trust root for one immutable artifact generation."""

    artifact_id: UUID
    generation_id: UUID
    manifest: RelationLocator
    manifest_sha256: str

    _validate_manifest_sha = field_validator("manifest_sha256")(
        lambda value: _artifact_sha256(value, "manifest_sha256")
    )


class ArtifactRelationRef(_ArtifactBase):
    """A relation locator bound to one artifact generation and semantic role."""

    digest_format: Literal[1, 2, 3] = 2
    artifact_id: UUID
    generation_id: UUID
    role: RelationRole
    relation: RelationLocator
    schema_sha256: str
    content_sha256: str
    row_count: int = Field(ge=0)
    primary_key: tuple[str, ...]

    _validate_schema_sha = field_validator("schema_sha256")(
        lambda value: _artifact_sha256(value, "schema_sha256")
    )
    _validate_content_sha = field_validator("content_sha256")(
        lambda value: _artifact_sha256(value, "content_sha256")
    )

    @field_validator("row_count")
    @classmethod
    def _strict_row_count(cls, value: int) -> int:
        if isinstance(value, bool):
            _definition_refusal(
                "definition.artifact_relation.row_count_integer",
                "row_count must be an integer",
            )
        return value

    @field_validator("primary_key")
    @classmethod
    def _strict_primary_key(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            _definition_refusal(
                "definition.artifact_relation.primary_key_contain",
                "primary_key must contain at least one identifier",
            )
        normalized = tuple(_artifact_identifier(item, "primary_key member") for item in value)
        if len(set(normalized)) != len(normalized):
            _definition_refusal(
                "definition.artifact_relation.primary_key_members",
                "primary_key members must be unique",
            )
        return normalized


_ARTIFACT_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "exposures": ("experiment_id", "unit_id"),
    "measure_stats": ("experiment_id", "unit_id", "ds", "measure_key"),
    "breakout_dimension": ("experiment_id", "unit_id"),
    "factor_dimension": ("experiment_id", "unit_id"),
    "cluster_identity": ("experiment_id", "unit_id"),
    "cuped_preperiod": ("experiment_id", "unit_id"),
    "assignment_counts": ("experiment_id", "population", "group_id"),
    "trigger_population": ("experiment_id", "unit_id"),
    "trigger_measure_stats": ("experiment_id", "unit_id", "ds", "measure_key"),
    "encouragement_uptake": ("experiment_id", "unit_id"),
    "site_volume": ("experiment_id", "ds", "measure_key"),
    "unit_covariate": ("experiment_id", "unit_id"),
    "unit_covariate_level": ("experiment_id", "unit_id"),
}


def _check_relation_binding(
    relation: ArtifactRelationRef,
    *,
    artifact_id: UUID,
    generation_id: UUID,
    role: str,
) -> None:
    if relation.artifact_id != artifact_id or relation.generation_id != generation_id:
        _definition_refusal(
            "definition.relation_bind_manifest",
            f"{role} relation must bind to the manifest artifact generation",
            role=role,
        )
    if relation.role != role:
        _definition_refusal(
            "definition.relation_role",
            f"{role} relation has role {relation.role!r}",
            expected_role=role,
            relation_role=relation.role,
        )
    expected_key = _ARTIFACT_PRIMARY_KEYS.get(role)
    if expected_key is not None and relation.primary_key != expected_key:
        _definition_refusal(
            "definition.relation_primary_key",
            f"{role} relation primary_key must be {expected_key!r}",
            role=role,
            expected_key=expected_key,
        )


class BaseRelations(_ArtifactBase):
    """Required base artifact relations bound to one artifact generation.

    Construct it with exposure and measure-stat relation references, usually
    through the artifact publication loader.
    """

    exposures: ArtifactRelationRef
    measure_stats: ArtifactRelationRef

    @model_validator(mode="after")
    def _validate_base_roles(self) -> BaseRelations:
        if self.exposures.role != "exposures":
            _definition_refusal(
                "definition.base_relations.exposures_relation_use",
                "exposures relation must use role 'exposures'",
            )
        if self.measure_stats.role != "measure_stats":
            _definition_refusal(
                "definition.base_relations.measure_stats_relation",
                "measure_stats relation must use role 'measure_stats'",
            )
        if self.exposures.artifact_id != self.measure_stats.artifact_id:
            _definition_refusal(
                "definition.base_relations.share_artifact_id",
                "base relations must share artifact_id",
            )
        if self.exposures.generation_id != self.measure_stats.generation_id:
            _definition_refusal(
                "definition.base_relations.share_generation_id",
                "base relations must share generation_id",
            )
        if self.exposures.primary_key != ("experiment_id", "unit_id"):
            _definition_refusal(
                "definition.base_relations.exposures_primary_key",
                "exposures primary_key is fixed to (experiment_id, unit_id)",
            )
        if self.measure_stats.primary_key != (
            "experiment_id",
            "unit_id",
            "ds",
            "measure_key",
        ):
            _definition_refusal(
                "definition.base_relations.measure_stats_primary",
                "measure_stats primary_key is fixed to (experiment_id, unit_id, ds, measure_key)",
            )
        return self


class Freshness(_ArtifactBase):
    loaded_through: date
    declared_complete: bool = False


class MeasureManifest(_ExplicitOnlyFields, _ArtifactBase):
    _explicit_only_fields: ClassVar[frozenset[str]] = frozenset({"event_horizon"})

    measure_key: str = Field(min_length=1)
    source_provenance_sha256: str
    freshness: Freshness
    #: Filtered event horizon over this measure's metric_events streams, with NULL
    #: values dropped and ratio denominators included; native uses it as spine edge.
    #: `NO_DATA_SIGNAL` or an explicit None marks a confirmed eventless measure.
    #: An omitted key uses freshness and stays unserialized, so older digests verify.
    event_horizon: date | None = None

    @field_validator("measure_key")
    @classmethod
    def _measure_key(cls, value: str) -> str:
        return _artifact_scalar(value, "measure_key")

    _validate_provenance_sha = field_validator("source_provenance_sha256")(
        lambda value: _artifact_sha256(value, "source_provenance_sha256")
    )


class SimpleMetricMeasure(_ArtifactBase):
    kind: Literal["simple"] = "simple"
    metric_name: str
    measure_key: str

    @field_validator("metric_name", "measure_key")
    @classmethod
    def _binding_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class RatioMetricMeasure(_ArtifactBase):
    kind: Literal["ratio"] = "ratio"
    metric_name: str
    numerator_measure_key: str
    denominator_measure_key: str
    allow_same_measure: bool = False

    @field_validator("metric_name", "numerator_measure_key", "denominator_measure_key")
    @classmethod
    def _binding_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)

    @model_validator(mode="after")
    def _different_ratio_measures(self) -> RatioMetricMeasure:
        if (
            self.numerator_measure_key == self.denominator_measure_key
            and not self.allow_same_measure
        ):
            _definition_refusal(
                "definition.ratio_metric.numerator_denominator_differ",
                "ratio numerator and denominator must differ unless allow_same_measure=True",
            )
        return self


MetricMeasure = Annotated[
    SimpleMetricMeasure | RatioMetricMeasure,
    Field(discriminator="kind"),
]


class ArtifactContext(_ArtifactBase):
    # Format 1 embedded raw source SQL. It stays representable only so admission can
    # refuse it by name; new contexts are always format 2.
    context_format: Literal[1, 2] = 2
    canonical_json: str
    sha256: str

    _validate_context_sha = field_validator("sha256")(
        lambda value: _artifact_sha256(value, "sha256")
    )

    @field_validator("canonical_json")
    @classmethod
    def _canonical_context(cls, value: str) -> str:
        return _validate_json_payload(value, "canonical_json")

    @model_validator(mode="after")
    def _context_digest_matches(self) -> ArtifactContext:
        expected = _artifact_context_digest(self.canonical_json)
        if self.sha256 != expected:
            _definition_refusal(
                "definition.artifact.context_sha256_does",
                "context sha256 does not match canonical_json",
            )
        if self.context_format == 2 and json.loads(self.canonical_json).get("context_format") != 2:
            _definition_refusal(
                "definition.artifact.context_format_disagrees",
                "context canonical_json must carry the same context_format as the model",
            )
        return self


class ExtensionRefBase(_ArtifactBase):
    extension_version: Literal[1, 2, 3] = 1
    relation: ArtifactRelationRef
    definition_sha256: str
    source_provenance_sha256: str

    @model_validator(mode="after")
    def _supported_extension_version(self):
        kind = getattr(self, "kind", None)
        if self.extension_version == 2 and kind != "encouragement_uptake":
            _definition_refusal(
                "definition.artifact_extension.unsupported_version",
                "extension version 2 is reserved for timestamp-bearing encouragement uptake",
            )
        if self.extension_version == 3 and kind != "trigger_population":
            _definition_refusal(
                "definition.artifact_extension.unsupported_version",
                "extension version 3 is reserved for trigger population anchors",
            )
        return self

    _validate_definition_sha = field_validator("definition_sha256")(
        lambda value: _artifact_sha256(value, "definition_sha256")
    )
    _validate_source_sha = field_validator("source_provenance_sha256")(
        lambda value: _artifact_sha256(value, "source_provenance_sha256")
    )


class BreakoutDimensionExtension(ExtensionRefBase):
    kind: Literal["breakout_dimension"] = "breakout_dimension"
    dimension_name: str
    source_name: str
    as_of: Literal["pre_exposure"] = "pre_exposure"

    @field_validator("dimension_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class FactorDimensionExtension(ExtensionRefBase):
    kind: Literal["factor_dimension"] = "factor_dimension"
    factor_name: str
    as_of: Literal["pre_exposure"] = "pre_exposure"
    source_name: str

    @field_validator("factor_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class ClusterIdentityExtension(ExtensionRefBase):
    kind: Literal["cluster_identity"] = "cluster_identity"
    cluster_name: str

    @field_validator("cluster_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "cluster_name")


class CupedPreperiodExtension(ExtensionRefBase):
    kind: Literal["cuped_preperiod"] = "cuped_preperiod"
    metric_name: str
    measure_key: str
    aggregation: Literal["sum", "count", "avg_event"]
    window_start_days: int
    window_end_days: int

    @field_validator("metric_name", "measure_key")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)

    @field_validator("window_start_days", "window_end_days")
    @classmethod
    def _strict_window_day(cls, value: int) -> int:
        if isinstance(value, bool):
            _definition_refusal(
                "definition.cuped_preperiod.window_bounds_integers",
                "window bounds must be integers",
            )
        return value

    @model_validator(mode="after")
    def _valid_window(self) -> CupedPreperiodExtension:
        if self.window_start_days >= self.window_end_days or self.window_end_days > 0:
            _definition_refusal(
                "definition.cuped_preperiod.window_satisfy_window",
                "cuped window must satisfy window_start_days < window_end_days <= 0",
            )
        return self


def _normalize_populations(
    populations: tuple[Literal["assigned", "triggered"], ...],
) -> tuple[Literal["assigned", "triggered"], ...]:
    if len(set(populations)) != len(populations):
        _definition_refusal(
            "definition.assignment_populations_contain",
            "assignment populations must not contain duplicates",
        )
    values = set(populations)
    if values not in ({"assigned"}, {"assigned", "triggered"}):
        _definition_refusal(
            "definition.assignment_populations_include",
            "assignment populations must include assigned",
        )
    return ("assigned", "triggered") if "triggered" in values else ("assigned",)


class AssignmentCountsExtension(ExtensionRefBase):
    kind: Literal["assignment_counts"] = "assignment_counts"
    populations: tuple[Literal["assigned", "triggered"], ...]
    mixed_unit_count: int = Field(ge=0, strict=True)
    unassigned_unit_count: int = Field(ge=0, strict=True)
    assignment_fingerprint_sha256: str

    _validate_fingerprint_sha = field_validator("assignment_fingerprint_sha256")(
        lambda value: _artifact_sha256(value, "assignment_fingerprint_sha256")
    )

    @field_validator("populations")
    @classmethod
    def _canonical_populations(
        cls, value: tuple[Literal["assigned", "triggered"], ...]
    ) -> tuple[Literal["assigned", "triggered"], ...]:
        return _normalize_populations(value)


class TriggerPopulationExtension(ExtensionRefBase):
    extension_version: Literal[3] = 3
    kind: Literal["trigger_population"] = "trigger_population"
    trigger_name: str
    observation_cutoff_ts: datetime
    complete_through_ts: datetime | None

    @field_validator("trigger_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "trigger_name")

    @field_validator("observation_cutoff_ts", "complete_through_ts")
    @classmethod
    def _utc_instant(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            _definition_refusal(
                "definition.artifact.trigger_timestamp_timezone",
                "trigger evidence timestamps must be timezone-aware",
            )
        return value.astimezone(UTC).replace(tzinfo=UTC)


class TriggerMeasureStatsExtension(ExtensionRefBase):
    kind: Literal["trigger_measure_stats"] = "trigger_measure_stats"
    trigger_name: str
    metric_names: tuple[str, ...]
    trigger_extension_version: Literal[3] = 3
    trigger_population_content_sha256: str
    observation_cutoff_ts: datetime
    complete_through_ts: datetime | None

    @field_validator("trigger_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "trigger_name")

    @field_validator("metric_names")
    @classmethod
    def _singleton_metric_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 1 or value[0] != _artifact_scalar(value[0], "metric_name"):
            _definition_refusal(
                "definition.artifact.trigger_metric_scope",
                "trigger_measure_stats must name exactly one metric",
            )
        return value

    _validate_trigger_population_sha = field_validator("trigger_population_content_sha256")(
        lambda value: _artifact_sha256(value, "trigger_population_content_sha256")
    )

    @field_validator("observation_cutoff_ts", "complete_through_ts")
    @classmethod
    def _utc_instant(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            _definition_refusal(
                "definition.artifact.trigger_timestamp_timezone",
                "trigger evidence timestamps must be timezone-aware",
            )
        return value.astimezone(UTC).replace(tzinfo=UTC)


class EncouragementUptakeExtension(_ExplicitOnlyFields, ExtensionRefBase):
    _explicit_only_fields: ClassVar[frozenset[str]] = frozenset({"observation_edge"})

    kind: Literal["encouragement_uptake"] = "encouragement_uptake"
    uptake_name: str
    window_days: int | None
    one_sided: bool
    #: Enrollment/uptake coverage before uptake-window filtering; independent of outcomes.
    #: Omitted on older artifacts to preserve their serialized manifest digests.
    observation_edge: date | None = None

    @field_validator("uptake_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "uptake_name")

    @field_validator("window_days")
    @classmethod
    def _strict_window(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if isinstance(value, bool) or value < 1:
            _definition_refusal(
                "definition.encouragement_uptake.window_days_integer",
                "window_days must be an integer >= 1",
            )
        return value


class SiteVolumeExtension(ExtensionRefBase):
    kind: Literal["site_volume"] = "site_volume"
    measure_keys: tuple[str, ...]
    first_ds: date
    last_ds: date
    freshness: Freshness

    @field_validator("measure_keys")
    @classmethod
    def _canonical_measure_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            _definition_refusal(
                "definition.site_volume.measure_keys_empty",
                "measure_keys must not be empty",
            )
        values = tuple(_artifact_scalar(item, "measure_keys member") for item in value)
        if len(set(values)) != len(values):
            _definition_refusal(
                "definition.site_volume.measure_keys_unique",
                "measure_keys must be unique",
            )
        return tuple(sorted(values, key=lambda item: item.encode("utf-8")))

    @model_validator(mode="after")
    def _valid_date_range(self) -> SiteVolumeExtension:
        if self.first_ds > self.last_ds:
            _definition_refusal(
                "definition.site_volume.first_ds_last",
                "site volume first_ds must be <= last_ds",
            )
        return self


class UnitCovariateExtension(ExtensionRefBase):
    """A declared numeric (int/float/bool) adjustment covariate, served as
    the nullable ``value`` of the frozen ``unit_covariate`` relation."""

    kind: Literal["unit_covariate"] = "unit_covariate"
    covariate_name: str
    source_name: str
    as_of: Literal["pre_exposure", "static"]

    @field_validator("covariate_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class UnitCovariateLevelExtension(ExtensionRefBase):
    """A declared categorical (string) adjustment covariate, served as the
    nullable ``level`` of its own ``unit_covariate_level`` relation so the
    numeric relation's wire schema and digests never change. A NULL level
    is a missing covariate value; the label is never coalesced."""

    kind: Literal["unit_covariate_level"] = "unit_covariate_level"
    covariate_name: str
    source_name: str
    as_of: Literal["pre_exposure", "static"]

    @field_validator("covariate_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


ArtifactExtensionRef = Annotated[
    BreakoutDimensionExtension
    | FactorDimensionExtension
    | ClusterIdentityExtension
    | CupedPreperiodExtension
    | AssignmentCountsExtension
    | TriggerPopulationExtension
    | TriggerMeasureStatsExtension
    | EncouragementUptakeExtension
    | SiteVolumeExtension
    | UnitCovariateExtension
    | UnitCovariateLevelExtension,
    Field(discriminator="kind"),
]


class ExtensionRequestBase(_ArtifactBase):
    pass


class BreakoutDimensionRequest(ExtensionRequestBase):
    kind: Literal["breakout_dimension"] = "breakout_dimension"
    property_name: str
    source_name: str

    @field_validator("property_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class FactorDimensionRequest(ExtensionRequestBase):
    kind: Literal["factor_dimension"] = "factor_dimension"
    property_name: str
    source_name: str

    @field_validator("property_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class ClusterIdentityRequest(ExtensionRequestBase):
    kind: Literal["cluster_identity"] = "cluster_identity"
    cluster_name: str

    @field_validator("cluster_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "cluster_name")


class CupedPreperiodRequest(ExtensionRequestBase):
    kind: Literal["cuped_preperiod"] = "cuped_preperiod"
    metric_name: str

    @field_validator("metric_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "metric_name")


class AssignmentCountsRequest(ExtensionRequestBase):
    kind: Literal["assignment_counts"] = "assignment_counts"
    populations: tuple[Literal["assigned", "triggered"], ...] = ("assigned",)

    @field_validator("populations")
    @classmethod
    def _canonical_populations(
        cls, value: tuple[Literal["assigned", "triggered"], ...]
    ) -> tuple[Literal["assigned", "triggered"], ...]:
        return _normalize_populations(value)


class TriggerPopulationRequest(ExtensionRequestBase):
    kind: Literal["trigger_population"] = "trigger_population"
    trigger_name: str

    @field_validator("trigger_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "trigger_name")


class TriggerMeasureStatsRequest(ExtensionRequestBase):
    kind: Literal["trigger_measure_stats"] = "trigger_measure_stats"
    trigger_name: str
    metric_names: tuple[str, ...]

    @field_validator("trigger_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "trigger_name")

    @field_validator("metric_names")
    @classmethod
    def _singleton_metric_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 1:
            _raise_request("artifact.request.trigger_metric_scope")
        return (_artifact_scalar(value[0], "metric_name"),)


class EncouragementUptakeRequest(ExtensionRequestBase):
    kind: Literal["encouragement_uptake"] = "encouragement_uptake"
    uptake_name: str

    @field_validator("uptake_name")
    @classmethod
    def _identity_scalar(cls, value: str) -> str:
        return _artifact_scalar(value, "uptake_name")


class SiteVolumeRequest(ExtensionRequestBase):
    kind: Literal["site_volume"] = "site_volume"
    metric_names: tuple[str, ...]

    @field_validator("metric_names")
    @classmethod
    def _canonical_metric_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            _definition_refusal(
                "definition.site_volume.metric_names_empty",
                "metric_names must not be empty",
            )
        values = tuple(_artifact_scalar(item, "metric_names member") for item in value)
        if len(set(values)) != len(values):
            _definition_refusal(
                "definition.site_volume.metric_names_unique",
                "metric_names must be unique",
            )
        return tuple(sorted(values, key=lambda item: item.encode("utf-8")))


class UnitCovariateRequest(ExtensionRequestBase):
    kind: Literal["unit_covariate"] = "unit_covariate"
    property_name: str
    source_name: str

    @field_validator("property_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


class UnitCovariateLevelRequest(ExtensionRequestBase):
    kind: Literal["unit_covariate_level"] = "unit_covariate_level"
    property_name: str
    source_name: str

    @field_validator("property_name", "source_name")
    @classmethod
    def _identity_scalar(cls, value: str, info) -> str:
        return _artifact_scalar(value, info.field_name)


ArtifactExtensionRequest = Annotated[
    BreakoutDimensionRequest
    | FactorDimensionRequest
    | ClusterIdentityRequest
    | CupedPreperiodRequest
    | AssignmentCountsRequest
    | TriggerPopulationRequest
    | TriggerMeasureStatsRequest
    | EncouragementUptakeRequest
    | SiteVolumeRequest
    | UnitCovariateRequest
    | UnitCovariateLevelRequest,
    Field(discriminator="kind"),
]


def _extension_definition_digest(kind: str, canonical_json: str, domain: str) -> str:
    return _artifact_digest(
        _ARTIFACT_DIGEST_ROOT + domain.encode("ascii") + b"\x00" + kind.encode("utf-8") + b"\x00",
        canonical_json.encode("utf-8"),
    )


class ArtifactExtensionCatalogEntry(_ArtifactBase):
    request: ArtifactExtensionRequest
    canonical_definition_json: str
    canonical_source_recipe_json: str
    definition_sha256: str
    source_provenance_sha256: str

    _validate_definition_sha = field_validator("definition_sha256")(
        lambda value: _artifact_sha256(value, "definition_sha256")
    )
    _validate_source_sha = field_validator("source_provenance_sha256")(
        lambda value: _artifact_sha256(value, "source_provenance_sha256")
    )

    @field_validator("canonical_definition_json", "canonical_source_recipe_json")
    @classmethod
    def _canonical_recipe(cls, value: str, info) -> str:
        return _validate_json_payload(value, info.field_name)

    @model_validator(mode="after")
    def _catalog_hashes(self) -> ArtifactExtensionCatalogEntry:
        kind = self.request.kind
        expected_definition = _extension_definition_digest(
            kind, self.canonical_definition_json, "extension-definition"
        )
        expected_source = _extension_definition_digest(
            kind, self.canonical_source_recipe_json, "extension-source"
        )
        if self.definition_sha256 != expected_definition:
            _definition_refusal(
                "definition.artifact_extension.definition_sha256_does",
                "definition_sha256 does not match canonical definition JSON",
            )
        if self.source_provenance_sha256 != expected_source:
            _definition_refusal(
                "definition.artifact_extension.source_provenance_sha256",
                "source_provenance_sha256 does not match canonical source recipe JSON",
            )
        return self


def _extension_relation(ref: ExtensionRefBase, *, artifact_id: UUID, generation_id: UUID) -> None:
    _check_relation_binding(
        ref.relation,
        artifact_id=artifact_id,
        generation_id=generation_id,
        role=cast("RelationRole", ref.model_dump(mode="python")["kind"]),
    )


class UnitDayArtifactManifest(_ArtifactBase):
    artifact_format: Literal[1] = 1
    artifact_id: UUID
    generation_id: UUID
    experiment_id: str = Field(min_length=1)
    created_at: datetime
    day_boundary: str
    first_ds: date
    last_ds: date
    base: BaseRelations
    measures: tuple[MeasureManifest, ...]
    metric_measures: tuple[MetricMeasure, ...]
    context: ArtifactContext
    extensions: tuple[ArtifactExtensionRef, ...] = ()
    manifest_sha256: str

    _validate_manifest_sha = field_validator("manifest_sha256")(
        lambda value: _artifact_sha256(value, "manifest_sha256")
    )

    @field_validator("experiment_id")
    @classmethod
    def _experiment_identifier(cls, value: str) -> str:
        return _artifact_scalar(value, "experiment_id")

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            _definition_refusal(
                "definition.unit_day.created_at_timezone",
                "created_at must be timezone-aware",
            )
        return value.astimezone(UTC)

    @field_validator("day_boundary")
    @classmethod
    def _manifest_day_boundary(cls, value: str) -> str:
        return _validate_day_boundary(value)

    @model_validator(mode="after")
    def _validate_manifest(self) -> UnitDayArtifactManifest:
        _validate_manifest_base(self)
        declared_measures = _validate_manifest_measures(self)
        _validate_manifest_extension_catalog(self)
        _validate_manifest_trigger_measure_stats(self, declared_measures)
        _validate_manifest_extension_order(self)
        _validate_manifest_digest(self)
        return self


def _validate_manifest_base(manifest: UnitDayArtifactManifest) -> None:
    if manifest.first_ds > manifest.last_ds:
        _definition_refusal(
            "definition.unit_day.manifest_first_ds",
            "manifest first_ds must be <= last_ds",
        )
    for relation, role in (
        (manifest.base.exposures, "exposures"),
        (manifest.base.measure_stats, "measure_stats"),
    ):
        _check_relation_binding(
            relation,
            artifact_id=manifest.artifact_id,
            generation_id=manifest.generation_id,
            role=role,
        )


def _validate_manifest_measures(manifest: UnitDayArtifactManifest) -> set[str]:
    measure_keys = tuple(measure.measure_key for measure in manifest.measures)
    if len(set(measure_keys)) != len(measure_keys):
        _definition_refusal(
            "definition.unit_day.manifest_measure_keys",
            "manifest measure keys must be unique",
        )
    if measure_keys != tuple(sorted(measure_keys, key=lambda value: value.encode("utf-8"))):
        _definition_refusal(
            "definition.unit_day.manifest_measures_use",
            "manifest measures must use canonical UTF-8 key order",
        )
    declared_measures = set(measure_keys)
    _validate_manifest_metric_bindings(manifest, declared_measures)
    return declared_measures


def _validate_manifest_metric_bindings(
    manifest: UnitDayArtifactManifest, declared_measures: set[str]
) -> None:
    binding_keys: set[tuple[str, str]] = set()
    metric_names: set[str] = set()
    binding_order: list[tuple[str, str]] = []
    for binding in manifest.metric_measures:
        key = (binding.metric_name, binding.kind)
        binding_order.append(key)
        if key in binding_keys:
            _definition_refusal(
                "definition.unit_day.manifest_metric_bindings",
                "manifest metric bindings must be unique",
            )
        binding_keys.add(key)
        if binding.metric_name in metric_names:
            _definition_refusal(
                "definition.unit_day.manifest_metric_names",
                "manifest metric names must be unique",
            )
        metric_names.add(binding.metric_name)
        if isinstance(binding, SimpleMetricMeasure):
            if binding.measure_key not in declared_measures:
                _definition_refusal(
                    "definition.unit_day.metric_references_undeclared",
                    f"metric {binding.metric_name!r} references undeclared measure "
                    f"{binding.measure_key!r}",
                    metric_name=binding.metric_name,
                    measure_key=binding.measure_key,
                )
        elif (
            binding.numerator_measure_key not in declared_measures
            or binding.denominator_measure_key not in declared_measures
        ):
            _definition_refusal(
                "definition.unit_day.ratio_metric_references",
                f"ratio metric {binding.metric_name!r} references undeclared measure",
                metric_name=binding.metric_name,
            )
    if binding_order != sorted(
        binding_order, key=lambda value: (value[0].encode("utf-8"), value[1].encode("utf-8"))
    ):
        _definition_refusal(
            "definition.unit_day.manifest_metric_bindings_canonical_order",
            "manifest metric bindings must use canonical UTF-8 order",
        )


def _validate_manifest_extension_catalog(manifest: UnitDayArtifactManifest) -> None:
    extension_keys: set[tuple[str, str]] = set()
    for extension in manifest.extensions:
        _extension_relation(
            extension,
            artifact_id=manifest.artifact_id,
            generation_id=manifest.generation_id,
        )
        extension_key = (extension.kind, _extension_identity(extension))
        if extension_key in extension_keys:
            _definition_refusal(
                "definition.unit_day.manifest_extensions_contain",
                "manifest extensions must not contain duplicates",
            )
        extension_keys.add(extension_key)
        matched_request = _extension_matches_catalog(extension, manifest.context)
        if matched_request is None:
            _definition_refusal(
                "definition.unit_day.manifest_extension_absent",
                "manifest extension is absent from the caller-trusted catalog",
            )
        if isinstance(extension, SiteVolumeExtension):
            _validate_manifest_site_volume(extension, matched_request, manifest)


def _validate_manifest_site_volume(
    extension: SiteVolumeExtension,
    matched_request: Mapping[str, Any],
    manifest: UnitDayArtifactManifest,
) -> None:
    declared_keys = {measure.measure_key for measure in manifest.measures}
    bindings_by_metric: dict[str, tuple[str, ...]] = {}
    for binding in manifest.metric_measures:
        if isinstance(binding, SimpleMetricMeasure):
            bindings_by_metric[binding.metric_name] = (binding.measure_key,)
        else:
            bindings_by_metric[binding.metric_name] = (
                binding.numerator_measure_key,
                binding.denominator_measure_key,
            )
    requested = tuple(matched_request.get("metric_names", ()))
    missing = [name for name in requested if name not in bindings_by_metric]
    if missing:
        _definition_refusal(
            "definition.unit_day.site_volume_request",
            "site volume request names metrics without manifest measure bindings",
        )
    required = {key for name in requested for key in bindings_by_metric[name]}
    if set(extension.measure_keys) != required:
        _definition_refusal(
            "definition.unit_day.site_volume_keys",
            "site volume keys must equal the requested metrics' measure bindings",
        )
    if not required <= declared_keys:
        _definition_refusal(
            "definition.unit_day.site_volume_keys_absent_from_manifest",
            "site volume keys are absent from manifest measures",
        )


def _validate_manifest_trigger_measure_stats(
    manifest: UnitDayArtifactManifest, declared_measures: set[str]
) -> None:
    trigger_populations = {
        extension.trigger_name: extension
        for extension in manifest.extensions
        if isinstance(extension, TriggerPopulationExtension)
    }
    for extension in manifest.extensions:
        if not isinstance(extension, TriggerMeasureStatsExtension):
            continue
        trigger = trigger_populations.get(extension.trigger_name)
        if (
            trigger is None
            or extension.trigger_population_content_sha256 != trigger.relation.content_sha256
            or extension.observation_cutoff_ts != trigger.observation_cutoff_ts
        ):
            _definition_refusal(
                "definition.unit_day.trigger_measure_anchor",
                "trigger-measure evidence must bind the matching v3 trigger population",
            )
        binding = next(
            (
                item
                for item in manifest.metric_measures
                if item.metric_name == extension.metric_names[0]
            ),
            None,
        )
        if binding is None:
            _definition_refusal(
                "definition.unit_day.trigger_measure_metric",
                "trigger-measure request names an unbound metric",
            )
        measure_keys = (
            {binding.measure_key}
            if isinstance(binding, SimpleMetricMeasure)
            else {binding.numerator_measure_key, binding.denominator_measure_key}
        )
        if not measure_keys <= declared_measures:
            _definition_refusal(
                "definition.unit_day.trigger_measure_keys",
                "trigger-measure keys must be declared in the manifest",
            )


def _validate_manifest_extension_order(manifest: UnitDayArtifactManifest) -> None:
    extension_order = [_extension_sort_key(extension) for extension in manifest.extensions]
    if extension_order != sorted(extension_order):
        _definition_refusal(
            "definition.unit_day.manifest_extensions_use",
            "manifest extensions must use canonical role/version/locator order",
        )


def _validate_manifest_digest(manifest: UnitDayArtifactManifest) -> None:
    expected = _artifact_manifest_digest(manifest)
    if manifest.manifest_sha256 != expected:
        _definition_refusal(
            "definition.unit_day.manifest_sha256_does",
            "manifest_sha256 does not match canonical manifest body",
        )


def _extension_identity(extension: ArtifactExtensionRef) -> str:
    if isinstance(extension, BreakoutDimensionExtension):
        return f"{extension.dimension_name}\x00{extension.source_name}"
    if isinstance(extension, FactorDimensionExtension):
        return f"{extension.factor_name}\x00{extension.source_name}"
    if isinstance(extension, ClusterIdentityExtension):
        return extension.cluster_name
    if isinstance(extension, CupedPreperiodExtension):
        return (
            f"{extension.metric_name}\x00{extension.measure_key}\x00"
            f"{extension.window_start_days}\x00{extension.window_end_days}"
        )
    if isinstance(extension, AssignmentCountsExtension):
        return ",".join(extension.populations)
    if isinstance(extension, TriggerPopulationExtension):
        return extension.trigger_name
    if isinstance(extension, TriggerMeasureStatsExtension):
        return extension.trigger_name + "\x00" + "\x00".join(extension.metric_names)
    if isinstance(extension, EncouragementUptakeExtension):
        return f"{extension.uptake_name}\x00{extension.window_days}"
    if isinstance(extension, UnitCovariateExtension | UnitCovariateLevelExtension):
        return f"{extension.covariate_name}\x00{extension.source_name}"
    return "\x00".join(extension.measure_keys)


def _extension_sort_key(
    extension: ArtifactExtensionRef,
) -> tuple[bytes, int, bytes, bytes]:
    locator = extension.relation.relation
    locator_key = "\x00".join(
        item or "" for item in (locator.catalog, locator.schema_name, locator.name)
    ).encode("utf-8")
    return (
        extension.kind.encode("utf-8"),
        extension.extension_version,
        locator_key,
        _extension_identity(extension).encode("utf-8"),
    )


def _catalog_request_identity(request: Mapping[str, Any]) -> str:
    kind = request.get("kind")
    if kind in {"breakout_dimension", "factor_dimension", "unit_covariate", "unit_covariate_level"}:
        return f"{request.get('property_name')}\x00{request.get('source_name')}"
    if kind in {"cluster_identity", "trigger_population", "encouragement_uptake"}:
        return str(
            request.get("cluster_name") or request.get("trigger_name") or request.get("uptake_name")
        )
    if kind == "trigger_measure_stats":
        return (
            str(request.get("trigger_name")) + "\x00" + "\x00".join(request.get("metric_names", ()))
        )
    if kind == "cuped_preperiod":
        return str(request.get("metric_name"))
    if kind == "assignment_counts":
        return ",".join(request.get("populations", ()))
    if kind == "site_volume":
        return "\x00".join(request.get("metric_names", ()))
    return ""


def _extension_matches_catalog(
    extension: ArtifactExtensionRef, context: ArtifactContext
) -> Mapping[str, Any] | None:
    entries = _extension_catalog_entries(context)
    if entries is None:
        return None
    for entry in entries:
        request = entry.get("request", {})
        if request.get("kind") != extension.kind:
            continue
        if entry.get("definition_sha256") != extension.definition_sha256:
            continue
        if entry.get("source_provenance_sha256") != extension.source_provenance_sha256:
            continue
        try:
            definition = json.loads(entry["canonical_definition_json"])
        except (KeyError, TypeError, ValueError):
            continue
        if _extension_matches_definition(extension, request, definition):
            return request
    return None


def _extension_catalog_entries(
    context: ArtifactContext,
) -> list[dict[str, Any]] | None:
    try:
        raw_entries = json.loads(context.canonical_json)["extension_catalog"]
        entries: list[dict[str, Any]] = []
        previous: tuple[int, bytes] | None = None
        for raw in raw_entries:
            entry = ArtifactExtensionCatalogEntry.model_validate(raw)
            request_json = _artifact_canonical_json_bytes(entry.request).decode("utf-8")
            key = (_EXTENSION_KIND_RANK[entry.request.kind], request_json.encode("utf-8"))
            if previous is not None and key <= previous:
                return None
            previous = key
            entries.append(entry.model_dump(mode="json"))
    except (KeyError, TypeError, ValueError):
        return None
    return entries


def _extension_matches_definition(
    extension: ArtifactExtensionRef,
    request: Mapping[str, Any],
    definition: Any,
) -> bool:
    match extension:
        case UnitCovariateExtension() | UnitCovariateLevelExtension():
            return _catalog_request_identity(request) == _extension_identity(extension)
        case BreakoutDimensionExtension() | FactorDimensionExtension():
            return _matches_dimension_extension(extension, request)
        case ClusterIdentityExtension():
            return _matches_cluster_extension(extension, request)
        case TriggerPopulationExtension():
            return _matches_trigger_extension(extension, request)
        case CupedPreperiodExtension():
            return _matches_cuped_extension(extension, request, definition)
        case AssignmentCountsExtension():
            return _matches_assignment_extension(extension, request)
        case EncouragementUptakeExtension():
            return _matches_encouragement_extension(extension, request, definition)
        case SiteVolumeExtension():
            return _matches_site_volume_extension(extension, request, definition)
        case _:
            return True


def _matches_dimension_extension(
    extension: BreakoutDimensionExtension | FactorDimensionExtension,
    request: Mapping[str, Any],
) -> bool:
    return _catalog_request_identity(request) == _extension_identity(extension)


def _matches_cluster_extension(
    extension: ClusterIdentityExtension,
    request: Mapping[str, Any],
) -> bool:
    return request.get("cluster_name") == extension.cluster_name


def _matches_trigger_extension(
    extension: TriggerPopulationExtension,
    request: Mapping[str, Any],
) -> bool:
    return request.get("trigger_name") == extension.trigger_name


def _matches_assignment_extension(
    extension: AssignmentCountsExtension,
    request: Mapping[str, Any],
) -> bool:
    return tuple(request.get("populations", ())) == extension.populations


def _matches_cuped_extension(
    extension: CupedPreperiodExtension,
    request: Mapping[str, Any],
    definition: Any,
) -> bool:
    if request.get("metric_name") != extension.metric_name:
        return False
    required = ("measure_key", "aggregation", "window_start_days", "window_end_days")
    if any(field not in definition for field in required):
        return False
    return all(getattr(extension, field) == definition[field] for field in required)


def _matches_encouragement_extension(
    extension: EncouragementUptakeExtension,
    request: Mapping[str, Any],
    definition: Any,
) -> bool:
    if request.get("uptake_name") != extension.uptake_name:
        return False
    required = ("window_days", "one_sided")
    if any(field not in definition for field in required):
        return False
    return (
        definition["window_days"] == extension.window_days
        and definition["one_sided"] == extension.one_sided
    )


def _matches_site_volume_extension(
    extension: SiteVolumeExtension,
    request: Mapping[str, Any],
    definition: Any,
) -> bool:
    expected_keys = definition.get("measure_keys", request.get("metric_names", ()))
    if tuple(expected_keys) != extension.measure_keys:
        return False
    freshness = definition.get("freshness")
    if not isinstance(freshness, Mapping):
        return False
    return (
        definition.get("first_ds") == extension.first_ds.isoformat()
        and definition.get("last_ds") == extension.last_ds.isoformat()
        and freshness.get("loaded_through") == extension.freshness.loaded_through.isoformat()
        and freshness.get("declared_complete") == extension.freshness.declared_complete
    )
