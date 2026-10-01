"""Immutable unit-day artifact lifecycle handles and pure context helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable
from uuid import UUID

from increment._canonical import canonical_json_loads
from increment._canonical import canonical_model_value as _dump

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from narwhals.typing import IntoDataFrame

    from increment.semantics.artifact import (
        ArtifactContext,
        ArtifactRelationRef,
        RelationLocator,
        UnitDayArtifactManifest,
        UnitDayArtifactRef,
    )
from increment.errors import (
    CodedError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)
from increment.query.artifact_digest import canonical_json
from increment.semantics.artifact import RelationRole
from increment.semantics.design import Encouragement

_BINARY64_U = 2.0**-53
_BINARY64_ETA = 2.0**-1074
_BINARY64_F = 1.0 + 8.0 * _BINARY64_U
_BINARY64_MAX = float.fromhex("0x1.fffffffffffffp+1023")


def _scalar_choice(condition: Any, yes: Any, no: Any) -> Any:
    return yes if condition else no


def _aggregate_mean_error_bound(
    n_events: Any,
    magnitude: Any,
    *,
    choose: Callable[[Any, Any, Any], Any] = _scalar_choice,
) -> Any:
    """Outward binary64 mean-error bound for counts from 1 through 2**53."""
    # prose: allow-long Derivation of the shared floating-point enclosure.
    # D=1-(n-1)u is exact. Summation and division give beta*M+(1/D+1)*eta,
    # where beta=n*u/D=(1+u)*gamma+u and gamma=(n-1)u/D.
    # Rounded half-scale comparisons require C=(beta+u)/(1-u).
    # F*(1-u)**6 >= 1 covers six normal roundings in evaluating the bound;
    # 4*eta*(1/D+3) also covers subnormal evaluation and comparison errors.
    denominator = 1.0 - (n_events - 1) * _BINARY64_U
    coefficient = (n_events * _BINARY64_U / denominator + _BINARY64_U) / (1.0 - _BINARY64_U)
    large_coefficient = coefficient > 1.0
    small = magnitude < _BINARY64_ETA / choose(large_coefficient, 1.0, coefficient)
    overflow = large_coefficient & (
        magnitude >= _BINARY64_MAX / choose(large_coefficient, coefficient, 1.0)
    )
    product = coefficient * choose(small | overflow, 0.0, magnitude)
    product = choose(small, _BINARY64_ETA, product)
    overflow = overflow | (product >= _BINARY64_MAX / _BINARY64_F)
    bound = choose(overflow, 0.0, product) * _BINARY64_F + (
        4.0 * _BINARY64_ETA * (1.0 / denominator + 3.0)
    )
    return choose(overflow, math.inf, bound)


def _aggregate_half(value: Any, choose: Callable[[Any, Any, Any], Any]) -> Any:
    """Halve without asking a SQL backend to underflow a nonzero operand."""
    tiny = (value > -2.0 * _BINARY64_ETA) & (value < 2.0 * _BINARY64_ETA)
    return choose(tiny, 0.0, value) / 2.0


def _aggregate_sum_within_extrema(
    total: Any,
    n_events: Any,
    minimum: Any,
    maximum: Any,
    *,
    choose: Callable[[Any, Any, Any], Any] = _scalar_choice,
) -> Any:
    """Check finite statistics with identical scalar and SQL arithmetic."""
    valid_count = (n_events >= 1) & (n_events <= 2**53)
    n = choose(valid_count, n_events, 1)
    magnitude = choose(maximum > -minimum, maximum, -minimum)
    bound = _aggregate_mean_error_bound(n, magnitude, choose=choose)
    underflow_threshold = n * _BINARY64_ETA
    tiny_sum = (total > -underflow_threshold) & (total < underflow_threshold)
    mean = choose(tiny_sum, 0.0, total) / n
    half_mean = _aggregate_half(mean, choose)
    half_bound = _aggregate_half(bound, choose)
    enclosed = (half_mean >= _aggregate_half(minimum, choose) - half_bound) & (
        half_mean <= _aggregate_half(maximum, choose) + half_bound
    )
    exact_singleton = (total == minimum) & (total == maximum)
    # Same-sign additions cannot fall below their largest summand's magnitude.
    monotone_sum = ((minimum < 0.0) | (total >= maximum)) & ((maximum > 0.0) | (total <= minimum))
    return (
        valid_count
        & (minimum <= maximum)
        & monotone_sum
        & choose(n == 1, exact_singleton, choose(magnitude == 0.0, total == 0.0, enclosed))
    )


def _aggregate_sum_within_extrema_sql(total: Any, n_events: Any, minimum: Any, maximum: Any) -> Any:
    """Adapt nullable SQL counts to the shared finite-statistic predicate."""
    import ibis

    count: Any = ibis.coalesce(n_events, 0)
    valid_count = (count >= 1) & (count <= 2**53)
    # Validate integers before conversion; PostgreSQL must evaluate in binary64.
    floating_count = ibis.ifelse(valid_count, count, 1).cast("float64")
    return valid_count & _aggregate_sum_within_extrema(
        total, floating_count, minimum, maximum, choose=ibis.ifelse
    )


ARTIFACT_DOMAIN_ROOT = b"increment.unit-day-artifact\x00v1\x00"
_EXTENSION_KIND_RANK: Mapping[str, int] = MappingProxyType(
    {
        "breakout_dimension": 0,
        "factor_dimension": 1,
        "cluster_identity": 2,
        "cuped_preperiod": 3,
        "assignment_counts": 4,
        "trigger_population": 5,
        "encouragement_uptake": 6,
        "site_volume": 7,
        "unit_covariate": 8,
        "unit_covariate_level": 9,
    }
)


class ArtifactContractError(InvalidRequestError):
    """Backward-compatible name for a shared coded artifact refusal."""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        context: Mapping[str, object] | None = None,
        **legacy_context: object,
    ) -> None:
        if message in _REFUSAL_CODES and code is not None:
            message, code = code, message
        if code is None:
            _raise("artifact.artifact_contract.refusal_code")
        # Both spellings of context survive; an explicit mapping wins a key clash.
        super().__init__(message, code=code, context={**legacy_context, **(context or {})})


_REFUSAL_CODES = (
    "artifact.format.unsupported",
    "artifact.manifest.invalid",
    "artifact.identifier.unsafe",
    "artifact.context.mismatch",
    "artifact.refresh.invalid_ref",
    "artifact.refresh.context_mismatch",
    "artifact.relation.schema_mismatch",
    "artifact.relation.digest_mismatch",
    "artifact.snapshot.mixed",
    "artifact.extension.missing",
    "artifact.extension.invalid",
    "artifact.evidence.unavailable",
    "artifact.metric.binding_mismatch",
    "artifact.generation.dropped",
    "artifact.manifest.unreferenced_relation",
    "artifact_contract.encouragement_uptake_conflict",
)

_REFUSALS: dict[str, RefusalSpec] = {}
REFUSALS: Mapping[str, RefusalSpec] = MappingProxyType(_REFUSALS)

# The loop's codes carry a free-text message, so they stay render_fn-shaped.
for _artifact_code in _REFUSAL_CODES:
    _REFUSALS[_artifact_code] = RefusalSpec(
        _artifact_code, ArtifactContractError, lambda *, message, **_: message
    )
del _artifact_code

_REFUSALS.update(
    refusals(
        InvalidRequestError,
        {
            "artifact.artifact_contract.refusal_code": "artifact refusal requires code",
            "artifact.on_mixed_assignment": "on_mixed_assignment must be 'error', 'warn', or 'exclude'",
            "artifact.coverage_dates_date": "coverage dates must be date values",
            "artifact.first_ds_last": "first_ds must be <= last_ds",
        },
    )
)
_raise = raiser(_REFUSALS)


@runtime_checkable
class ArtifactSnapshot(Protocol):
    """Immutable, generation-scoped read handle."""

    @property
    def artifact_id(self) -> UUID: ...
    @property
    def generation_id(self) -> UUID: ...
    def read_manifest(
        self, locator: RelationLocator, *, expected_sha256: str
    ) -> UnitDayArtifactManifest: ...
    def verify_relation(
        self, ref: ArtifactRelationRef, *, expected_role: RelationRole
    ) -> ir.Table: ...
    def execute(self, expression: ir.Table) -> IntoDataFrame: ...
    def batches(self, expression: ir.Table): ...


@runtime_checkable
class ArtifactPublication(Protocol):
    """Isolated create-only generation handle."""

    @property
    def artifact_id(self) -> UUID: ...
    @property
    def generation_id(self) -> UUID: ...
    def write_relation(self, role: RelationRole, table: ir.Table) -> ArtifactRelationRef: ...
    def publish_manifest(self, manifest: UnitDayArtifactManifest) -> UnitDayArtifactRef: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Namespace-scoped publication/snapshot lifecycle contract."""

    def validate_locator(
        self,
        locator: RelationLocator,
        *,
        artifact_id: UUID,
        generation_id: UUID,
        role: RelationRole | Literal["manifest"],
    ) -> None: ...
    def begin_publication(
        self,
        *,
        expected_context: ArtifactContext,
        refresh_of: UnitDayArtifactRef | None = None,
    ) -> AbstractContextManager[ArtifactPublication]: ...
    def open_snapshot(
        self, ref: UnitDayArtifactRef
    ) -> AbstractContextManager[ArtifactSnapshot]: ...
    def drop_generation(self, artifact_id: UUID, generation_id: UUID) -> None:
        """Durably invalidate a generation, then erase its relations.

        A committed tombstone is the invalidation point: every store sharing
        the namespace hides and refuses the generation from then on, even if
        physical erasure below fails and must be retried by a fresh caller.
        The manifest-index row is kept afterward as a cleanup receipt, not
        erased. Data already copied out before invalidation cannot be
        revoked. A never-published pair returns immediately with no effect.
        An already-dropped pair is not a true no-op: it revalidates the
        retained receipt and retries erasing every named relation.
        """
        ...


@dataclass(frozen=True, slots=True)
class ImmutableArtifactSnapshot:
    """Frozen callback adapter; the store supplies all I/O callbacks."""

    artifact_id: UUID
    generation_id: UUID
    _read_manifest: Callable[[RelationLocator, str], UnitDayArtifactManifest]
    _verify_relation: Callable[[ArtifactRelationRef, RelationRole], ir.Table]
    _execute: Callable[[ir.Table], IntoDataFrame]
    _batches: Callable[[ir.Table], object] | None = None

    def read_manifest(
        self, locator: RelationLocator, *, expected_sha256: str
    ) -> UnitDayArtifactManifest:
        return self._read_manifest(locator, expected_sha256)

    def verify_relation(self, ref: ArtifactRelationRef, *, expected_role: RelationRole) -> ir.Table:
        return self._verify_relation(ref, expected_role)

    def execute(self, expression: ir.Table) -> IntoDataFrame:
        return self._execute(expression)

    def batches(self, expression: ir.Table):
        if self._batches is None:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported", "artifact store has no bounded sequential record reader"
            )
        return self._batches(expression)


@dataclass(frozen=True, slots=True)
class ImmutableArtifactPublication:
    """Frozen callback adapter for an isolated create-only generation."""

    artifact_id: UUID
    generation_id: UUID
    _write_relation: Callable[[RelationRole, ir.Table], ArtifactRelationRef]
    _publish_manifest: Callable[[UnitDayArtifactManifest], UnitDayArtifactRef]

    def write_relation(self, role: RelationRole, table: ir.Table) -> ArtifactRelationRef:
        return self._write_relation(role, table)

    def publish_manifest(self, manifest: UnitDayArtifactManifest) -> UnitDayArtifactRef:
        return self._publish_manifest(manifest)


def _model(name: str) -> type[Any]:
    from increment.semantics import models

    try:
        return getattr(models, name)
    except AttributeError as exc:  # pragma: no cover
        raise ImportError(f"increment.semantics.models does not export {name}") from exc


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(_dump(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "artifact.context.mismatch", "context contains a non-canonical value"
        ) from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: str, *, code: str = "artifact.identifier.unsafe") -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ArtifactContractError(code, "SHA-256 values must be lowercase 64-hex", value=value)
    return value


_SUPPORTED_CONTEXT_FORMAT = 2
_WINDOW_DAY_EDGES = frozenset({"start", "end", "observation_horizon"})


def validate_artifact_context(context: Any) -> Any:
    """Validate context format, canonical JSON, and caller-visible hash."""
    received = getattr(context, "context_format", None)
    if received != _SUPPORTED_CONTEXT_FORMAT:
        raise ArtifactContractError(
            "artifact.format.unsupported",
            "unsupported artifact context format; republish from trusted definitions",
            received=received,
            supported=_SUPPORTED_CONTEXT_FORMAT,
            route="republish the artifact from trusted definitions",
        )
    payload = getattr(context, "canonical_json", None)
    supplied = _require_sha(getattr(context, "sha256", ""), code="artifact.context.mismatch")
    if not isinstance(payload, str):
        raise ArtifactContractError(
            "artifact.context.mismatch", "context canonical_json is not text"
        )
    try:
        parsed = canonical_json_loads(payload)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("artifact.context.mismatch", "context JSON is invalid") from exc
    if _canonical_json(parsed) != payload:
        raise ArtifactContractError("artifact.context.mismatch", "context JSON is not canonical")
    if not isinstance(parsed, dict) or parsed.get("context_format") != _SUPPORTED_CONTEXT_FORMAT:
        raise ArtifactContractError(
            "artifact.context.mismatch", "context JSON disagrees with the model context_format"
        )
    expected = _sha256(ARTIFACT_DOMAIN_ROOT + b"context\x00" + payload.encode())
    if expected != supplied:
        raise ArtifactContractError(
            "artifact.context.mismatch", "context hash does not match canonical JSON"
        )
    _validate_window_days(parsed)
    return context


def _validate_window_days(parsed: Any) -> None:
    """Require the stored window days: present, ISO dates, ``start`` <= ``end`` <= horizon.

    The horizon is ``observation_end`` when declared, else ``end``, and ``observation_end``
    requires ``end``: the horizon is absent exactly when ``end`` is."""
    if "window_days" not in parsed:
        raise ArtifactContractError(
            "artifact.context.mismatch",
            "context predates bound window days; republish from trusted definitions",
            missing="window_days",
            route="republish from trusted definitions",
        )
    days = parsed["window_days"]
    if not isinstance(days, dict) or set(days) != _WINDOW_DAY_EDGES:
        raise _malformed_window_days()
    start, end = _window_day(days["start"]), _window_day(days["end"])
    horizon = _window_day(days["observation_horizon"])
    if start is None or (end is None) != (horizon is None):
        raise _malformed_window_days()
    if end is not None and horizon is not None and not start <= end <= horizon:
        raise _malformed_window_days()


def _malformed_window_days() -> ArtifactContractError:
    return ArtifactContractError(
        "artifact.context.mismatch", "context window_days is malformed", invalid="window_days"
    )


def _window_day(raw: object) -> date | None:
    """An ISO calendar date, or None for an absent edge; anything else is malformed."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            raise _malformed_window_days() from None
        if parsed.isoformat() == raw:
            return parsed
    raise _malformed_window_days()


def validate_trusted_ref(
    ref: Any,
    *,
    expected_context: Any | None = None,
    artifact_id: UUID | None = None,
    generation_id: UUID | None = None,
) -> Any:
    """Validate a caller-pinned manifest reference before backend access."""
    if not isinstance(getattr(ref, "artifact_id", None), UUID) or not isinstance(
        getattr(ref, "generation_id", None), UUID
    ):
        raise ArtifactContractError(
            "artifact.refresh.invalid_ref", "artifact and generation IDs are required"
        )
    _require_sha(getattr(ref, "manifest_sha256", ""))
    locator = getattr(ref, "manifest", None)
    name = getattr(locator, "name", None)
    if (
        not isinstance(name, str)
        or not name
        or name.strip() != name
        or any(c in name for c in "\x00\n\r")
    ):
        raise ArtifactContractError("artifact.identifier.unsafe", "manifest locator name is unsafe")
    if artifact_id is not None and ref.artifact_id != artifact_id:
        raise ArtifactContractError("artifact.refresh.invalid_ref", "artifact ID does not match")
    if generation_id is not None and ref.generation_id != generation_id:
        raise ArtifactContractError("artifact.refresh.invalid_ref", "generation ID does not match")
    if expected_context is not None:
        validate_artifact_context(expected_context)
    return ref


def validate_snapshot_identity(snapshot: ArtifactSnapshot, ref: Any) -> None:
    if snapshot.artifact_id != ref.artifact_id or snapshot.generation_id != ref.generation_id:
        raise ArtifactContractError(
            "artifact.snapshot.mixed", "snapshot identity differs from pinned reference"
        )


@contextmanager
def open_trusted_manifest_snapshot(
    store: ArtifactStore, ref: Any, *, expected_context: Any | None = None
) -> Iterator[tuple[ArtifactSnapshot, Any]]:
    """Open one snapshot only after reading and verifying the pinned manifest.

    The stored context is validated before it is compared with the caller's pin, so an
    artifact from an older context contract always refuses by that name.
    """
    validate_trusted_ref(ref, expected_context=expected_context)
    store.validate_locator(
        ref.manifest, artifact_id=ref.artifact_id, generation_id=ref.generation_id, role="manifest"
    )
    with store.open_snapshot(ref) as snapshot:
        validate_snapshot_identity(snapshot, ref)
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        if getattr(manifest, "manifest_sha256", None) != ref.manifest_sha256:
            raise ArtifactContractError(
                "artifact.refresh.invalid_ref",
                "stored manifest digest differs from pinned reference",
            )
        if (
            getattr(manifest, "artifact_id", None) != ref.artifact_id
            or getattr(manifest, "generation_id", None) != ref.generation_id
        ):
            raise ArtifactContractError(
                "artifact.snapshot.mixed", "stored manifest identity differs from pinned reference"
            )
        validate_artifact_context(getattr(manifest, "context", None))
        if expected_context is not None:
            if getattr(getattr(manifest, "context", None), "sha256", None) != getattr(
                expected_context, "sha256", None
            ):
                raise ArtifactContractError(
                    "artifact.refresh.context_mismatch",
                    "stored manifest context differs from caller-pinned context",
                )
        yield snapshot, manifest


@contextmanager
def open_trusted_snapshot(
    store: ArtifactStore, ref: Any, *, expected_context: Any | None = None
) -> Iterator[ArtifactSnapshot]:
    """Open one snapshot only after reading and verifying the pinned manifest."""
    with open_trusted_manifest_snapshot(store, ref, expected_context=expected_context) as opened:
        yield opened[0]


def _request_key(request: Any) -> tuple[int, bytes]:
    kind = getattr(request, "kind", "")
    if kind not in _EXTENSION_KIND_RANK:
        raise ArtifactContractError(
            "artifact.extension.invalid", f"unknown extension kind {kind!r}"
        )
    return _EXTENSION_KIND_RANK[kind], _canonical_json(request).encode()


def _extension_hash(domain: str, request: Any, canonical: str) -> str:
    return _sha256(
        ARTIFACT_DOMAIN_ROOT
        + domain.encode()
        + b"\x00"
        + str(request.kind).encode()
        + b"\x00"
        + canonical.encode()
    )


def _require_hashed_recipe(recipe: Any) -> None:
    """Refuse a source recipe that is anything but the tagged digest object."""
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"source_recipe_format", "recipe_sha256"}
        or recipe["source_recipe_format"] != 2
        or isinstance(recipe["source_recipe_format"], bool)
    ):
        raise ArtifactContractError(
            "artifact.extension.invalid", "extension source recipe must be a hash-only object"
        )
    _require_sha(recipe["recipe_sha256"], code="artifact.extension.invalid")


def artifact_source_mapping(context: Any) -> dict[str, Any]:
    """Return the validated digest envelope a context stores for its native source recipe."""
    validate_artifact_context(context)
    mapping = json.loads(context.canonical_json).get("source_mapping")
    if (
        not isinstance(mapping, dict)
        or set(mapping) != {"source_mapping_format", "recipe_sha256"}
        or mapping["source_mapping_format"] != 2
        or isinstance(mapping["source_mapping_format"], bool)
    ):
        raise ArtifactContractError(
            "artifact.context.mismatch", "context source_mapping must be a format-2 digest"
        )
    _require_sha(mapping["recipe_sha256"], code="artifact.context.mismatch")
    return mapping


def unit_day_artifact_extension_catalog(context: Any) -> tuple[Any, ...]:
    """Validate and expose the immutable extension catalog in a context."""
    validate_artifact_context(context)
    try:
        raw_entries = json.loads(context.canonical_json)["extension_catalog"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "artifact.extension.invalid", "context lacks extension_catalog"
        ) from exc
    if not isinstance(raw_entries, list):
        raise ArtifactContractError(
            "artifact.extension.invalid", "extension_catalog must be an array"
        )
    model = _model("ArtifactExtensionCatalogEntry")
    result: list[Any] = []
    seen: set[bytes] = set()
    previous: tuple[int, bytes] | None = None
    for raw in raw_entries:
        try:
            entry = model.model_validate(raw)
            key = _request_key(entry.request)
            if key[1] in seen:
                raise ArtifactContractError(
                    "artifact.extension.invalid", "duplicate extension request"
                )
            if previous is not None and key < previous:
                raise ArtifactContractError(
                    "artifact.extension.invalid", "extension catalog is not canonical order"
                )
            seen.add(key[1])
            previous = key
            for field in ("canonical_definition_json", "canonical_source_recipe_json"):
                text = getattr(entry, field)
                if _canonical_json(canonical_json_loads(text)) != text:
                    raise ArtifactContractError(
                        "artifact.extension.invalid", f"{field} is not canonical"
                    )
            _require_hashed_recipe(canonical_json_loads(entry.canonical_source_recipe_json))
            if entry.definition_sha256 != _extension_hash(
                "extension-definition", entry.request, entry.canonical_definition_json
            ):
                raise ArtifactContractError(
                    "artifact.extension.invalid", "extension definition hash mismatch"
                )
            if entry.source_provenance_sha256 != _extension_hash(
                "extension-source", entry.request, entry.canonical_source_recipe_json
            ):
                raise ArtifactContractError(
                    "artifact.extension.invalid", "extension source hash mismatch"
                )
        except ArtifactContractError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactContractError(
                "artifact.extension.invalid", "invalid extension catalog entry"
            ) from exc
        result.append(entry)
    return tuple(result)


def _source_for(
    definitions: Any, property_name: str, requested: str | None, *, unit: str | None = None
) -> Any:
    if requested:
        source = next((s for s in definitions.fact_sources if s.name == requested), None)
        if source is None:
            raise ArtifactContractError(
                "artifact.context.mismatch", f"unknown source {requested!r}"
            )
        return source
    matches = [
        s
        for s in definitions.fact_sources
        if (unit is None or unit in s.entities)
        and any(p.name == property_name for p in definitions.properties_of(s))
    ]
    if len(matches) != 1:
        raise ArtifactContractError(
            "artifact.context.mismatch", f"property {property_name!r} has ambiguous source"
        )
    return matches[0]


def _fact_source_recipe(definitions: Any, fact_name: str, unit: str) -> Any:
    matches = [
        source
        for source in definitions.fact_sources
        if unit in source.entities and any(fact.name == fact_name for fact in source.facts)
    ]
    if len(matches) != 1:
        raise ArtifactContractError(
            "artifact.extension.invalid",
            f"fact {fact_name!r} has {len(matches)} unit-grain source matches",
        )
    return _dump(matches[0])


def _exposure_recipe(definitions: Any, experiment: Any, exposure_name: str | None = None) -> Any:
    selected_name = exposure_name or experiment.exposure
    exposure = next((item for item in definitions.exposures if item.name == selected_name), None)
    if exposure is None:
        raise ArtifactContractError(
            "artifact.extension.invalid", f"unknown exposure {selected_name!r}"
        )
    recipe: dict[str, Any] = {"exposure": _dump(exposure)}
    if exposure.fact is not None:
        recipe["fact_source"] = _fact_source_recipe(definitions, exposure.fact, experiment.unit)
    return recipe


def _measure_recipe_key(measure: Any) -> str:
    fact = getattr(measure, "fact", None)
    aggregation = getattr(measure, "aggregation", "count")
    filters = _dump(getattr(measure, "filters", ()))
    if not isinstance(fact, str):
        raise ArtifactContractError(
            "artifact.extension.invalid", "measure recipe requires a fact name"
        )
    identity = _canonical_json({"fact": fact, "aggregation": aggregation, "filters": filters})
    # Hash the fact/aggregation identity so every key emitted from a valid
    # definitions model satisfies the extension scalar contract.
    return f"measure_{_sha256(identity.encode('utf-8'))[:24]}"


def _hashed_source_recipe(request: Any, source_recipe: Any) -> dict[str, Any]:
    """Represent a source recipe by a tagged digest; the recipe itself may hold source SQL."""
    digest = _sha256(
        ARTIFACT_DOMAIN_ROOT
        + b"source-recipe\x00"
        + str(request.kind).encode()
        + b"\x00"
        + _canonical_json(source_recipe).encode()
    )
    return {"source_recipe_format": 2, "recipe_sha256": digest}


def _catalog_entry(request: Any, definition: Any, source_recipe: Any) -> Any:
    definition_json = _canonical_json(definition)
    source_json = _canonical_json(_hashed_source_recipe(request, source_recipe))
    return _model("ArtifactExtensionCatalogEntry").model_validate(
        {
            "request": _dump(request),
            "canonical_definition_json": definition_json,
            "canonical_source_recipe_json": source_json,
            "definition_sha256": _extension_hash("extension-definition", request, definition_json),
            "source_provenance_sha256": _extension_hash("extension-source", request, source_json),
        }
    )


@dataclass(frozen=True, slots=True)
class _ContextResolution:
    """Validated experiment identity and stable metric ordering."""

    experiment: Any
    metric_names: tuple[str, ...]


def _resolve_context_inputs(
    experiment_name: str,
    definitions: Any,
    on_mixed_assignment: str,
) -> _ContextResolution:
    if isinstance(definitions, (str, Path)):
        raise ArtifactContractError(
            "artifact.context.mismatch", "definitions must be an already-loaded Definitions model"
        )
    if on_mixed_assignment not in {"error", "warn", "exclude"}:
        _raise("artifact.on_mixed_assignment")
    experiment = definitions.experiment(experiment_name)
    if experiment is None:
        raise ArtifactContractError(
            "artifact.context.mismatch", f"unknown experiment {experiment_name!r}"
        )
    return _ContextResolution(
        experiment,
        tuple(sorted(experiment.metric_names, key=lambda value: value.encode("utf-8"))),
    )


def _selected_metric(definitions: Any, name: str) -> Any:
    metric = definitions.metric(name)
    if metric is None:
        raise ArtifactContractError(
            "artifact.context.mismatch", f"experiment metric {name!r} has no definition"
        )
    return metric


def _catalog_add(entries: list[Any], request: Any, definition: Any, source_recipe: Any) -> None:
    entries.append(_catalog_entry(request, definition, source_recipe))


def _dimension_catalog_entries(definitions: Any, experiment: Any) -> list[Any]:
    entries: list[Any] = []
    for items, kind, model_name in (
        (
            getattr(experiment, "breakouts", ()) or (),
            "breakout_dimension",
            "BreakoutDimensionRequest",
        ),
        (getattr(experiment, "factors", ()) or (), "factor_dimension", "FactorDimensionRequest"),
    ):
        for item in items:
            source = _source_for(
                definitions, item.property, getattr(item, "source", None), unit=experiment.unit
            )
            request = _model(model_name).model_validate(
                {"kind": kind, "property_name": item.property, "source_name": source.name}
            )
            prop = next(p for p in definitions.properties_of(source) if p.name == item.property)
            _catalog_add(
                entries,
                request,
                {"kind": kind, "property": _dump(item), "property_definition": _dump(prop)},
                {
                    "source": _dump(source),
                    "property": _dump(prop),
                    "dim_sources": _dump(definitions.dim_sources),
                },
            )
    return entries


def _covariate_catalog_entries(definitions: Any, experiment: Any) -> list[Any]:
    """One entry per covariate an observational design declares; an artifact
    can publish exactly these, never an ad hoc name. A numeric property is
    offered as `unit_covariate`, a string property as `unit_covariate_level`:
    the declared dtype picks the wire relation that can carry its values."""
    entries: list[Any] = []
    design = getattr(experiment, "design", None)
    if design is None or design.mechanism != "observational":
        return entries
    for covariate in design.covariates:
        source = _source_for(
            definitions, covariate.property, covariate.source, unit=experiment.unit
        )
        prop = next(p for p in definitions.properties_of(source) if p.name == covariate.property)
        kind, model_name = (
            ("unit_covariate_level", "UnitCovariateLevelRequest")
            if prop.dtype == "string"
            else ("unit_covariate", "UnitCovariateRequest")
        )
        request = _model(model_name).model_validate(
            {
                "kind": kind,
                "property_name": covariate.property,
                "source_name": source.name,
            }
        )
        _catalog_add(
            entries,
            request,
            {
                "kind": kind,
                "property": _dump(covariate),
                "property_definition": _dump(prop),
            },
            {
                "source": _dump(source),
                "property": _dump(prop),
                "dim_sources": _dump(definitions.dim_sources),
            },
        )
    return entries


def _identity_catalog_entries(definitions: Any, experiment: Any) -> list[Any]:
    entries: list[Any] = []
    if experiment.cluster is not None:
        request = _model("ClusterIdentityRequest").model_validate(
            {"kind": "cluster_identity", "cluster_name": experiment.cluster}
        )
        _catalog_add(
            entries,
            request,
            {"kind": "cluster_identity", "cluster_name": experiment.cluster},
            {
                "experiment": _dump(experiment),
                "exposure": _exposure_recipe(definitions, experiment),
                "definitions_fact_sources": _dump(definitions.fact_sources),
                "cluster": experiment.cluster,
            },
        )
    if experiment.trigger is not None:
        request = _model("TriggerPopulationRequest").model_validate(
            {"kind": "trigger_population", "trigger_name": experiment.trigger}
        )
        _catalog_add(
            entries,
            request,
            {"kind": "trigger_population", "trigger_name": experiment.trigger},
            {
                "experiment": _dump(experiment),
                "exposure": _exposure_recipe(definitions, experiment, experiment.trigger),
                "definitions_fact_sources": _dump(definitions.fact_sources),
                "trigger": experiment.trigger,
            },
        )
    return entries


def _cuped_catalog_entries(
    definitions: Any, experiment: Any, metric_names: tuple[str, ...]
) -> list[Any]:
    entries: list[Any] = []
    for metric_name in metric_names:
        binding = experiment.bindings.get(metric_name)
        if binding is None or not binding.wants_cuped:
            continue
        request = _model("CupedPreperiodRequest").model_validate(
            {"kind": "cuped_preperiod", "metric_name": metric_name}
        )
        metric = next(
            (candidate for candidate in definitions.metrics if candidate.name == metric_name), None
        )
        # A ratio metric's covariate is its numerator's pre-period total.
        measure = getattr(metric, "numerator", metric)
        if metric is None or metric.type == "quantile" or not hasattr(measure, "fact"):
            raise ArtifactContractError(
                "artifact.extension.invalid",
                f"CUPED binding for {metric_name!r} has no supported measure recipe",
            )
        aggregation = getattr(measure, "aggregation", "count")
        if aggregation not in {"sum", "count", "avg_event"}:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                f"CUPED aggregation {aggregation!r} is not representable",
            )
        _catalog_add(
            entries,
            request,
            {
                "kind": "cuped_preperiod",
                "metric_name": metric_name,
                "measure_key": measure.fact,
                "aggregation": aggregation,
                "window_start_days": -experiment.n_pre_periods,
                "window_end_days": 0,
                "metric_recipe": _dump(metric),
            },
            {
                "experiment": experiment.name,
                "metric": _dump(metric),
                "binding": _dump(binding),
                "fact_source": _fact_source_recipe(definitions, measure.fact, experiment.unit),
            },
        )
    return entries


def _assignment_catalog_entry(definitions: Any, experiment: Any, on_mixed_assignment: str) -> Any:
    populations = ("assigned", "triggered") if experiment.trigger is not None else ("assigned",)
    request = _model("AssignmentCountsRequest").model_validate(
        {"kind": "assignment_counts", "populations": populations}
    )
    entries: list[Any] = []
    _catalog_add(
        entries,
        request,
        {"kind": "assignment_counts", "populations": list(populations)},
        {
            "experiment": _dump(experiment),
            "exposure": _exposure_recipe(definitions, experiment),
            "plan": _dump(experiment.plan),
            "on_mixed_assignment": on_mixed_assignment,
            "assignment": _dump(experiment),
        },
    )
    return entries[0]


def _site_volume_catalog_entries(
    definitions: Any,
    experiment: Any,
    metric_names: tuple[str, ...],
    coverage: Mapping[str, Any] | None,
) -> list[Any]:
    if coverage is None or not metric_names:
        return []
    measure_keys: set[str] = set()
    metric_recipes: list[Any] = []
    for metric_name in metric_names:
        metric = next(
            (candidate for candidate in definitions.metrics if candidate.name == metric_name), None
        )
        if metric is None:
            raise ArtifactContractError(
                "artifact.extension.invalid", f"unknown metric {metric_name!r}"
            )
        metric_type = getattr(metric, "type", None)
        if metric_type in {"retention", "quantile"}:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                f"site volume does not support {metric_type} metric {metric_name!r}",
            )
        if metric_type == "ratio":
            parts = (metric.numerator, metric.denominator)
            measure_keys.update(_measure_recipe_key(part) for part in parts)
            metric_recipes.append(
                {
                    "metric": _dump(metric),
                    "parts": [
                        {
                            "measure": _dump(part),
                            "fact_source": _fact_source_recipe(
                                definitions, part.fact, experiment.unit
                            ),
                        }
                        for part in parts
                    ],
                }
            )
        elif hasattr(metric, "fact"):
            measure_keys.add(_measure_recipe_key(metric))
            metric_recipes.append(
                {
                    "metric": _dump(metric),
                    "fact_source": _fact_source_recipe(definitions, metric.fact, experiment.unit),
                }
            )
        else:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                f"metric {metric_name!r} has no materializable measure recipe",
            )
    try:
        first_ds = coverage["first_ds"]
        last_ds = coverage["last_ds"]
        freshness = coverage["freshness"]
    except (KeyError, TypeError) as exc:
        raise ArtifactContractError(
            "artifact.extension.invalid",
            "site volume coverage requires first_ds, last_ds, and freshness",
        ) from exc
    try:
        if isinstance(first_ds, str):
            first_ds = date.fromisoformat(first_ds)
        if isinstance(last_ds, str):
            last_ds = date.fromisoformat(last_ds)
        freshness = _model("Freshness").model_validate(freshness)
        if (
            not isinstance(first_ds, date)
            or isinstance(first_ds, datetime)
            or not isinstance(last_ds, date)
            or isinstance(last_ds, datetime)
        ):
            _raise("artifact.coverage_dates_date")
        if first_ds > last_ds:
            _raise("artifact.first_ds_last")
    except CodedError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "artifact.extension.invalid", "invalid typed site volume coverage"
        ) from exc
    request = _model("SiteVolumeRequest").model_validate(
        {"kind": "site_volume", "metric_names": metric_names}
    )
    keys = tuple(sorted(measure_keys, key=lambda value: value.encode("utf-8")))
    _catalog_add(
        entries := [],
        request,
        {
            "kind": "site_volume",
            "metric_names": list(metric_names),
            "measure_keys": list(keys),
            "first_ds": first_ds,
            "last_ds": last_ds,
            "freshness": freshness,
        },
        {
            "experiment": experiment.name,
            "metrics": list(metric_names),
            "metric_recipes": metric_recipes,
            "measure_keys": list(keys),
        },
    )
    return entries


def _encouragement_catalog_entries(
    definitions: Any, experiment: Any, encouragement_uptake: Any | None
) -> list[Any]:
    uptake_spec = encouragement_uptake or getattr(experiment, "uptake", None)
    if uptake_spec is None:
        return []
    design = uptake_spec
    uptake = getattr(design, "uptake", design)
    uptake_name = getattr(uptake, "fact", None)
    if not isinstance(uptake_name, str):
        raise ArtifactContractError(
            "artifact.extension.invalid",
            "encouragement uptake requires a typed design with UptakeSpec",
        )
    window_days = getattr(uptake, "window_days", None)
    one_sided = getattr(design, "one_sided", None)
    if not isinstance(one_sided, bool):
        raise ArtifactContractError(
            "artifact.extension.invalid",
            "encouragement uptake requires a typed Encouragement design",
        )
    request = _model("EncouragementUptakeRequest").model_validate(
        {"kind": "encouragement_uptake", "uptake_name": uptake_name}
    )
    entries: list[Any] = []
    _catalog_add(
        entries,
        request,
        {
            "kind": "encouragement_uptake",
            "uptake_name": uptake_name,
            "window_days": window_days,
            "one_sided": one_sided,
        },
        {
            "experiment": experiment.name,
            "uptake": _dump(design),
            "fact_source": _fact_source_recipe(definitions, uptake_name, experiment.unit),
        },
    )
    return entries


def _compile_extension_catalog(
    definitions: Any,
    resolution: _ContextResolution,
    *,
    on_mixed_assignment: str,
    encouragement_uptake: Any | None,
    site_volume_coverage: Mapping[str, Any] | None,
) -> list[Any]:
    experiment = resolution.experiment
    entries = _dimension_catalog_entries(definitions, experiment)
    entries.extend(_identity_catalog_entries(definitions, experiment))
    entries.extend(_cuped_catalog_entries(definitions, experiment, resolution.metric_names))
    entries.append(_assignment_catalog_entry(definitions, experiment, on_mixed_assignment))
    entries.extend(
        _site_volume_catalog_entries(
            definitions, experiment, resolution.metric_names, site_volume_coverage
        )
    )
    entries.extend(_encouragement_catalog_entries(definitions, experiment, encouragement_uptake))
    entries.extend(_covariate_catalog_entries(definitions, experiment))
    return entries


def compile_unit_day_artifact_context(
    experiment_name: str,
    definitions: Any,
    *,
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    encouragement_uptake: Encouragement | None = None,
    site_volume_coverage: Mapping[str, Any] | None = None,
) -> Any:
    """Compile a deterministic, non-executable context from an already-loaded Definitions model.

    A naive window edge is wall-clock time at its experiment's ``day_boundary`` and binds the
    equivalent UTC instant; any other naive datetime is read as UTC.
    Source SQL never enters the context: source recipes are represented by tagged hashes.
    """
    return _compile_context(
        experiment_name,
        definitions,
        on_mixed_assignment=on_mixed_assignment,
        encouragement_uptake=encouragement_uptake,
        site_volume_coverage=site_volume_coverage,
    )


def _utcify(value: Any) -> Any:
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    if isinstance(value, dict):
        return {key: _utcify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_utcify(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_utcify(item) for item in value)
    return value


def _has_naive_datetime(value: Any) -> bool:
    if isinstance(value, datetime):
        return value.tzinfo is None
    if isinstance(value, dict):
        return any(_has_naive_datetime(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_naive_datetime(item) for item in value)
    return False


def _boundary_normalized(definitions: Any) -> Any:
    """*definitions* with no naive datetime: a naive window edge becomes the UTC instant at
    which its experiment's day boundary reads that wall-clock time (so its local day is the
    declared one); every other naive datetime is labelled UTC. Aware input is returned as is."""
    dumped = definitions.model_dump(mode="python")
    if not _has_naive_datetime(dumped):
        return definitions
    for declared, entry in zip(definitions.experiments, dumped["experiments"], strict=True):
        offset = declared.day_boundary_offset
        for edge in ("start", "end", "observation_end"):
            value = entry.get(edge)
            if isinstance(value, datetime) and value.tzinfo is None:
                entry[edge] = (value - offset).replace(tzinfo=UTC)
    return type(definitions).model_validate(_utcify(dumped))


def _compile_context(
    experiment_name: str,
    definitions: Any,
    *,
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    encouragement_uptake: Encouragement | None = None,
    site_volume_coverage: Mapping[str, Any] | None = None,
) -> Any:
    """Compile a context; the source mapping and window days come from the caller's original
    *definitions*, everything else from their boundary-normalized copy."""
    from increment.semantics.models import window_days
    from increment.sequential_source import native_observation_mapping

    resolution = _resolve_context_inputs(experiment_name, definitions, on_mixed_assignment)
    identity = definitions
    definitions = _boundary_normalized(identity)
    if definitions is not identity:
        resolution = _resolve_context_inputs(experiment_name, definitions, on_mixed_assignment)
    experiment = resolution.experiment
    declared_design = experiment.resolved_design()
    if isinstance(declared_design, Encouragement):
        if encouragement_uptake is not None and encouragement_uptake != declared_design:
            raise ArtifactContractError(
                "artifact_contract.encouragement_uptake_conflict",
                "encouragement_uptake= disagrees with the experiment's declared design",
            )
        encouragement_uptake = declared_design
    # Native registration hashes the caller's original input, so the design-carrying copy
    # built below must never feed it.
    identity_experiment = identity.experiment(experiment_name)
    if identity_experiment is None:
        raise ArtifactContractError(
            "artifact.context.mismatch", f"unknown experiment {experiment_name!r}"
        )
    source_mapping = native_observation_mapping(
        identity, identity_experiment, on_mixed_assignment=on_mixed_assignment
    )
    metrics = tuple(_selected_metric(definitions, name) for name in resolution.metric_names)
    if (
        experiment.plan.inference is not None
        and experiment.plan.inference.kind in ("asymptotic_mean", "always_valid")
        and experiment.plan.inference.registration is None
    ):
        from increment.plan import bind_automatic_sequential_plan

        plan = bind_automatic_sequential_plan(
            experiment.plan,
            metrics,
            design=encouragement_uptake or declared_design,
            source_id=experiment.name,
            source_mapping=source_mapping,
            pre_period_covariate=experiment.n_pre_periods > 0,
        )
        experiment = type(experiment).model_validate(experiment.model_copy(update={"plan": plan}))
        resolution = _ContextResolution(experiment, resolution.metric_names)
    entries = _compile_extension_catalog(
        definitions,
        resolution,
        on_mixed_assignment=on_mixed_assignment,
        encouragement_uptake=encouragement_uptake,
        site_volume_coverage=site_volume_coverage,
    )
    entries.sort(key=lambda entry: _request_key(entry.request))
    # Experiment.design holds only a YAML-shaped declaration, so the effective design
    # (control arm, allocation, tuning) is projected into the dump, never into the model.
    experiment_payload = cast("dict[str, Any]", _dump(resolution.experiment))
    if encouragement_uptake is not None:
        experiment_payload["design"] = _dump(encouragement_uptake)
    payload = {
        "context_format": 2,
        "experiment_name": resolution.experiment.name,
        "experiment": experiment_payload,
        "definitions": {
            "dialect": definitions.dialect,
            "day_boundary": definitions.day_boundary,
            "metrics": [_dump(metric) for metric in metrics],
        },
        "source_mapping": source_mapping,
        "window_days": {
            edge: None if day is None else day.isoformat()
            for edge, day in window_days(identity_experiment).items()
        },
        "on_mixed_assignment": on_mixed_assignment,
        "extension_catalog": [_dump(entry) for entry in entries],
    }
    canonical = _canonical_json(payload)
    return _model("ArtifactContext").model_validate(
        {
            "context_format": 2,
            "canonical_json": canonical,
            "sha256": _sha256(ARTIFACT_DOMAIN_ROOT + b"context\x00" + canonical.encode()),
        }
    )


__all__ = [
    "ARTIFACT_DOMAIN_ROOT",
    "ArtifactContractError",
    "ArtifactSnapshot",
    "ArtifactPublication",
    "ArtifactStore",
    "ImmutableArtifactSnapshot",
    "ImmutableArtifactPublication",
    "RefusalSpec",
    "REFUSALS",
    "artifact_source_mapping",
    "compile_unit_day_artifact_context",
    "open_trusted_manifest_snapshot",
    "open_trusted_snapshot",
    "unit_day_artifact_extension_catalog",
    "validate_artifact_context",
    "validate_snapshot_identity",
    "validate_trusted_ref",
]
