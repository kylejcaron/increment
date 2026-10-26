"""Portable CUPED evidence envelopes carried by fixed moments format 13.

Each envelope row carries ``record_kind="cuped_evidence"`` and one canonical
JSON payload (version 1) for one metric, normalized method, population and
breakout coordinate. ``scalar_moments`` payloads hold the method's centered
per-arm moment rows; ``categorical_cells`` payloads hold joint categorical
evidence. Method identity excludes only the display label, so equivalent
requests share one envelope.
"""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import (
    Any,
    Literal,
    NoReturn,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from increment._canonical import CanonicalJSONError, canonical_json_bytes
from increment.errors import CodedError, RefusalSpec, WireFormatError, refuse
from increment.estimation.cuped.types import (
    CategoricalCells,
    FitState,
    InferenceSummary,
    MulticovariateCupedFit,
)

CUPED_EVIDENCE_FORMAT = 13
CUPED_EVIDENCE_RECORD = "cuped_evidence"
CUPED_EVIDENCE_FIELD = "cuped_evidence"
PAYLOAD_VERSION = 1
EvidenceKind = Literal["scalar_moments", "categorical_cells", "fitted_multicovariate"]
_KINDS: tuple[EvidenceKind, ...] = (
    "scalar_moments",
    "categorical_cells",
    "fitted_multicovariate",
)
_RESERVED_KINDS: tuple[str, ...] = ()
_PAYLOAD_FIELDS = frozenset(
    {
        "version",
        "kind",
        "metric",
        "method",
        "population",
        "trigger_name",
        "breakout",
        "source_sha256",
        "evidence",
    }
)
_POPULATIONS = ("assigned", "triggered")
_COUNT_FIELDS = frozenset({"n", "successes", "winsor_n", "winsor_n_lower", "winsor_n_upper"})
_TEXT_FIELDS = frozenset(
    {"experiment_id", "metric", "group_id", "x_role", "covariate_name", "missing"}
)
_CELL_FLOAT_FIELDS = (
    "outcome_references",
    "centered_first_moments",
    "symmetric_cross_moments",
)
_CELL_FIELDS = frozenset(field.name for field in fields(CategoricalCells)) - {"identity"}

_INVALID = RefusalSpec(
    "moments.cuped_evidence.invalid",
    WireFormatError,
    template="invalid portable CUPED evidence: {reason}",
    keys=frozenset({"reason"}),
)
_DUPLICATE = RefusalSpec(
    "moments.cuped_evidence.duplicate",
    WireFormatError,
    template=(
        "moments cube carries more than one CUPED evidence envelope for metric {metric!r}, "
        "population {population!r} and method {method}"
    ),
    keys=frozenset({"metric", "population", "method"}),
)
_UNSUPPORTED = RefusalSpec(
    "moments.cuped_evidence.unsupported",
    WireFormatError,
    template=(
        "CUPED evidence {field} {received!r} is not readable here; supported: {supported!r}. "
        "Upgrade increment, or re-export the cube from its original source."
    ),
    keys=frozenset({"field", "received", "supported"}),
)


def invalid_evidence(reason: str) -> NoReturn:
    refuse(_INVALID, reason=reason)


def duplicate_evidence(metric: str, population: str, method_key: str) -> NoReturn:
    refuse(_DUPLICATE, metric=metric, population=population, method=method_key)


@dataclass(frozen=True, slots=True)
class CupedEvidence:
    """One validated envelope carrying method-scoped sufficient evidence."""

    kind: EvidenceKind
    metric: str
    method_key: str
    population: str
    source_sha256: str
    payload_sha256: str
    trigger_name: str | None = None
    rows: tuple[Mapping[str, object], ...] = ()
    cells: CategoricalCells | None = None
    fit_state: Any | None = None
    inference_summary: Any | None = None

    def scalar_rows(self) -> list[dict[str, object]]:
        return [
            {key: value for key, value in row.items() if key not in {"covariate_name", "missing"}}
            for row in self.rows
        ]


def method_key(method: Any) -> str:
    """Canonical identity of a request, excluding only its display label."""
    return canonical_json_bytes(method.model_dump(mode="json", exclude={"name"})).decode()


def source_sha256(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def categorical_identity(
    metric: str, key: str, population: str, trigger_name: str | None = None
) -> tuple[object, ...]:
    """Evidence identity shared by partitions of one request and trigger."""
    return ("cuped_evidence", metric, population, trigger_name, key)


def categorical_request(method: Any) -> bool:
    covariates = method.covariates or ()
    return (
        method.variance_reduction == "cuped_multicovariate"
        and bool(covariates)
        and all(item.kind == "categorical" for item in covariates)
    )


def _payload(
    kind: EvidenceKind,
    metric: str,
    method: Any,
    *,
    population: str,
    trigger_name: str | None,
    source: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str:
    body = {
        "version": PAYLOAD_VERSION,
        "kind": kind,
        "metric": metric,
        "method": json.loads(method_key(method)),
        "population": population,
        "trigger_name": trigger_name,
        "breakout": None,
        "source_sha256": source_sha256(source),
        "evidence": evidence,
    }
    try:
        return canonical_json_bytes(body).decode()
    except CanonicalJSONError as exc:
        invalid_evidence(f"{kind} evidence for metric {metric!r} is not finite JSON: {exc}")


def scalar_payload(
    metric: str,
    method: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    population: str,
    trigger_name: str | None = None,
    source: Mapping[str, object],
):
    declaration = tuple(method.covariates or ())
    metadata = (
        {"covariate_name": declaration[0].name, "missing": declaration[0].missing}
        if len(declaration) == 1
        else {}
    )
    return _payload(
        "scalar_moments",
        metric,
        method,
        population=population,
        trigger_name=trigger_name,
        source=source,
        evidence={"rows": [{**dict(row), **metadata} for row in rows]},
    )


def categorical_payload(
    metric: str,
    method: Any,
    cells: CategoricalCells,
    *,
    population: str,
    trigger_name: str | None = None,
    source: Mapping[str, object],
) -> str:
    evidence = {name: getattr(cells, name) for name in sorted(_CELL_FIELDS)}
    return _payload(
        "categorical_cells",
        metric,
        method,
        population=population,
        trigger_name=trigger_name,
        source=source,
        evidence=evidence,
    )


def _wire_value(value: object) -> object:
    from enum import Enum

    from increment.estimation.cuped.encoding import categorical_missing_wire_value

    if (missing_category := categorical_missing_wire_value(value)) is not None:
        return missing_category
    if isinstance(value, Enum):
        invalid_evidence(f"fitted evidence cannot encode {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, (tuple, list)):
        return [_wire_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _wire_value(item) for key, item in value.items()}
    return value


_IDENTITY_MAPPING_KEYS = frozenset(
    {"artifacts", "source_snapshot_evidence", "complete_through_by_feed"}
)


def _restore_identity_mappings(value: object, *, mapping_sequence: bool = False) -> object:
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        if (
            items
            and mapping_sequence
            and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in items
            )
        ):
            return {
                key: _restore_identity_mappings(
                    item, mapping_sequence=key in _IDENTITY_MAPPING_KEYS
                )
                for key, item in cast(tuple[tuple[str, Any], ...], items)
            }
        return tuple(
            _restore_identity_mappings(item, mapping_sequence=mapping_sequence) for item in items
        )
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {
            key: _restore_identity_mappings(item, mapping_sequence=key in _IDENTITY_MAPPING_KEYS)
            for key, item in mapping.items()
        }
    return value


def _wire_fit_identity(identity: Sequence[object]) -> object:
    if len(identity) <= 3:
        return _wire_value(identity)
    source = identity[3]
    if isinstance(source, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in source
    ):
        identity = (
            *identity[:3],
            dict(cast(tuple[tuple[str, object], ...], source)),
            *identity[4:],
        )
    return _wire_value(identity)


def fitted_multicovariate_payload(
    metric: str,
    method: Any,
    fit_state: Any,
    inference_summary: Any,
    *,
    population: str,
    trigger_name: str | None = None,
    source: Mapping[str, object],
) -> str:
    from dataclasses import asdict

    state_value = asdict(fit_state)
    state_value["identity"] = _wire_fit_identity(fit_state.identity)
    summary_value = asdict(inference_summary)
    for field in ("fit_identity", "projection_identity", "evidence_identity"):
        summary_value[field] = _wire_fit_identity(getattr(inference_summary, field))
    summary_value["projection_specs"] = [
        {
            **asdict(spec),
            "identity": _wire_fit_identity(spec.identity),
        }
        for spec in inference_summary.projection_specs
    ]
    return _payload(
        "fitted_multicovariate",
        metric,
        method,
        population=population,
        trigger_name=trigger_name,
        source=source,
        evidence={
            "fit_state": _wire_value(state_value),
            "inference_summary": _wire_value(summary_value),
        },
    )


def _restore_wire_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_restore_wire_value(item) for item in value)
    if isinstance(value, dict):
        wire_value = cast(dict[str, object], value)
        if set(wire_value) == {"$categorical_missing"}:
            from increment.estimation.cuped.encoding import MISSING_CATEGORY

            if wire_value["$categorical_missing"] != MISSING_CATEGORY.value:
                invalid_evidence("fitted evidence contains an unknown categorical missing level")
            return MISSING_CATEGORY
        if set(wire_value) == {"$float"}:
            tag = wire_value["$float"]
            if tag not in ("nan", "inf", "-inf"):
                invalid_evidence("fitted evidence contains an invalid float tag")
            return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[cast(str, tag)]
        return {key: _restore_wire_value(item) for key, item in wire_value.items()}
    return value


def _validate_fitted_union(args: tuple[Any, ...], value: object, path: str) -> None:
    if value is None and type(None) in args:
        return
    for option in args:
        if option is type(None):
            continue
        try:
            _validate_fitted_wire_value(option, value, path)
            return
        except CodedError:
            pass
    invalid_evidence(f"fitted evidence {path} has an invalid type")


def _validate_fitted_tuple(args: tuple[Any, ...], value: object, path: str) -> None:
    if not isinstance(value, list):
        invalid_evidence(f"fitted evidence {path} must be an array")
    items = cast(list[object], value)
    if args and args[-1] is Ellipsis:
        for index, item in enumerate(items):
            _validate_fitted_wire_value(args[0], item, f"{path}[{index}]")
        return
    if len(items) != len(args):
        invalid_evidence(f"fitted evidence {path} has an invalid length")
    for index, (item_type, item) in enumerate(zip(args, items, strict=True)):
        _validate_fitted_wire_value(item_type, item, f"{path}[{index}]")


def _validate_fitted_float(value: object, path: str) -> None:
    if isinstance(value, dict) and set(value) == {"$float"}:
        tag = cast(dict[str, object], value)["$float"]
        if tag not in ("nan", "inf", "-inf"):
            invalid_evidence(f"fitted evidence {path} has an invalid float tag")
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        invalid_evidence(f"fitted evidence {path} must be numeric")


def _validate_fitted_wire_value(annotation: Any, value: object, path: str) -> None:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation in (Any, object):
        return
    if origin is Union or origin is types.UnionType:
        _validate_fitted_union(args, value, path)
        return
    if origin is tuple:
        _validate_fitted_tuple(args, value, path)
        return
    if annotation is float:
        _validate_fitted_float(value, path)
        return
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            invalid_evidence(f"fitted evidence {path} must be an integer")
        return
    if annotation is bool:
        if not isinstance(value, bool):
            invalid_evidence(f"fitted evidence {path} must be boolean")
        return
    if annotation is str:
        if not isinstance(value, str):
            invalid_evidence(f"fitted evidence {path} must be text")
        return
    if annotation is type(None):
        if value is not None:
            invalid_evidence(f"fitted evidence {path} must be null")
        return
    if isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
        _validate_fitted_wire_dataclass(annotation, value, path)


def _validate_fitted_wire_dataclass(cls: Any, value: object, path: str) -> None:
    if not isinstance(value, dict):
        invalid_evidence(f"fitted evidence {path} must be an object")
    wire_value = cast(dict[str, object], value)
    declared_fields = {item.name for item in fields(cls)}
    if set(wire_value) != declared_fields:
        invalid_evidence(f"fitted evidence {path} fields do not match its schema")
    hints = get_type_hints(cls)
    for name, annotation in hints.items():
        _validate_fitted_wire_value(annotation, wire_value[name], f"{path}.{name}")


def _fitted_dataclass(cls: Any, value: object, *, nested: Mapping[str, Any] | None = None) -> Any:
    _validate_fitted_wire_dataclass(cls, value, cls.__name__)
    wire_value = cast(dict[str, object], value)
    restored = {key: _restore_wire_value(item) for key, item in wire_value.items()}
    for key, item_cls in (nested or {}).items():
        nested_value = wire_value[key]
        if isinstance(nested_value, list):
            restored[key] = tuple(_fitted_dataclass(item_cls, item) for item in nested_value)
        else:
            restored[key] = _fitted_dataclass(item_cls, nested_value)
    try:
        return cls(**restored)
    except (TypeError, ValueError, CodedError) as exc:
        invalid_evidence(f"fitted evidence {cls.__name__} is inconsistent: {exc}")


def _decode_fitted_evidence(
    evidence: Mapping[str, object],
    declarations: list[dict[str, object]],
    metric: str,
    method_key_value: str,
    population: str,
    trigger_name: str | None,
    source: str,
    digest: str,
) -> CupedEvidence:
    from increment.estimation.cuped.encoding import MulticovariateCupedLayout
    from increment.estimation.cuped.types import FitState, InferenceSummary, ProjectionSpec

    if (
        set(evidence) != {"fit_state", "inference_summary"}
        or not declarations
        or any(item["kind"] not in ("continuous", "categorical") for item in declarations)
    ):
        invalid_evidence("fitted evidence must match a declared multicovariate method")
    state_value = evidence["fit_state"]
    if not isinstance(state_value, dict):
        invalid_evidence("fitted evidence FitState must be an object")
    state_value = dict(state_value)
    state = _fitted_dataclass(FitState, state_value, nested={"layout": MulticovariateCupedLayout})
    summary_value = evidence["inference_summary"]
    if not isinstance(summary_value, dict):
        invalid_evidence("fitted evidence InferenceSummary must be an object")
    summary = _fitted_dataclass(
        InferenceSummary, dict(summary_value), nested={"projection_specs": ProjectionSpec}
    )
    refusal_arms = tuple(arm for arm, _ in state.refusal_contexts)
    if refusal_arms != tuple(state.withheld_arms) or any(
        not isinstance(dict(context).get("code"), str) for _, context in state.refusal_contexts
    ):
        invalid_evidence("fitted evidence withheld-arm refusals are inconsistent")
    identity = tuple(state.identity)
    if len(identity) < 11 or not isinstance(identity[6], tuple):
        invalid_evidence("fitted evidence identity has an invalid shape")
    embedded_source = _restore_identity_mappings(identity[3], mapping_sequence=True)
    if identity[0] != "warehouse-cuped-fit-v1":
        invalid_evidence("fitted evidence identity has an unknown schema")
    if identity[1] != metric or identity[2] != population:
        invalid_evidence("fitted evidence metric or population differs from its envelope")
    if identity[4] != method_key_value or identity[5] != state.control:
        invalid_evidence("fitted evidence method or control differs from its fit state")
    if tuple(identity[7:11]) != (None, None, None, False):
        invalid_evidence("fitted evidence breakout scope differs from its envelope")
    if not isinstance(embedded_source, Mapping):
        invalid_evidence("fitted evidence source is not a mapping")
    embedded_digest = source_sha256(cast(Mapping[str, object], embedded_source))
    if embedded_digest != source:
        invalid_evidence(
            f"fitted evidence source digest {embedded_digest} differs from envelope {source}"
        )
    declarations_names = tuple(item["name"] for item in declarations)
    declarations_kinds = tuple(item["kind"] for item in declarations)
    identity_arms = tuple(identity[6])
    eligible_arms = tuple(state.eligible_arms)
    withheld_arms = tuple(state.withheld_arms)
    if (
        state.layout.names != declarations_names
        or state.layout.kinds != declarations_kinds
        or not (
            len(state.layout.names)
            == len(state.layout.kinds)
            == len(state.layout.levels)
            == len(state.layout.references)
        )
        or len(state.anchor) != state.p
        or len(state.theta) != len(state.slopes) * state.p
        or len(state.layout.encoded_columns) != state.p
        or len(set(identity_arms)) != len(identity_arms)
        or set(identity_arms) != set(eligible_arms) | set(withheld_arms)
        or set(eligible_arms) & set(withheld_arms)
        or tuple(arm for arm in identity_arms if arm in set(eligible_arms)) != eligible_arms
        or tuple(arm for arm in identity_arms if arm in set(withheld_arms)) != withheld_arms
    ):
        invalid_evidence("fitted evidence layout or arm roster differs from its method")
    if summary.fit_identity != state.identity or summary.evidence_identity != state.identity:
        invalid_evidence("fitted evidence summary is bound to a different fit")
    if summary.projection_identity != state.identity + ("all-pairs-v1",):
        invalid_evidence("fitted evidence projection identity differs from its fit")
    if any(
        spec.identity != state.identity + (spec.kind, spec.arm, spec.control, spec.identity[-1])
        for spec in summary.projection_specs
    ):
        invalid_evidence("fitted evidence projection specification has a different identity")
    if len(summary.projection_specs) != len(summary.scaled_projected_score_squares):
        invalid_evidence(
            "fitted evidence projection reduction count differs from its specifications"
        )
    return CupedEvidence(
        "fitted_multicovariate",
        metric,
        method_key_value,
        population,
        source,
        digest,
        trigger_name=trigger_name,
        fit_state=state,
        inference_summary=summary,
    )


def finalize_fitted_multicovariate(fit_state: Any, inference_summary: Any):
    from increment.estimation.cuped.inference import finalize_fit

    return finalize_fit(fit_state, inference_summary)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            invalid_evidence(f"payload repeats key {key!r}")
        out[key] = value
    return out


def _floats(value: object, field: str) -> object:
    if isinstance(value, list):
        return [_floats(item, field) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        invalid_evidence(f"categorical field {field!r} must hold numbers")
    return float(value)


def _scalar_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        invalid_evidence("scalar evidence rows must be objects")
    typed: dict[str, object] = {}
    for key, value in cast("dict[str, object]", row).items():
        if value is None:
            typed[key] = None
        elif key in _TEXT_FIELDS:
            if not isinstance(value, str):
                invalid_evidence(f"scalar evidence field {key!r} must be text")
            typed[key] = value
        elif key in _COUNT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int):
                invalid_evidence(f"scalar evidence count {key!r} must be an integer")
            typed[key] = value
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid_evidence(f"scalar evidence moment {key!r} must be a number")
        else:
            typed[key] = float(value)
    return typed


def _cells(evidence: Mapping[str, object], identity: tuple[object, ...]) -> CategoricalCells:
    if set(evidence) != _CELL_FIELDS:
        invalid_evidence(
            f"categorical evidence fields must be exactly {sorted(_CELL_FIELDS)}; "
            f"got {sorted(evidence)}"
        )
    values: dict[str, Any] = {
        name: _floats(evidence[name], name) if name in _CELL_FLOAT_FIELDS else evidence[name]
        for name in _CELL_FIELDS
    }
    try:
        return CategoricalCells(identity=identity, **values)
    except CodedError as exc:
        invalid_evidence(f"categorical evidence is inconsistent: {exc.context.get('reason', exc)}")
    except (TypeError, ValueError) as exc:
        invalid_evidence(f"categorical evidence is malformed: {exc}")


def _method_identity(method: object, kind: EvidenceKind) -> tuple[str, list[dict[str, object]]]:
    from increment.declarations import Method

    if not isinstance(method, dict):
        invalid_evidence("payload method must be a CUPED method identity")
    method_dict = cast("dict[str, object]", method)
    if method_dict.get("variance_reduction") not in ("cuped", "cuped_multicovariate"):
        invalid_evidence("payload method must be a CUPED method identity")
    try:
        normalized = Method.model_validate({"name": "wire", **method_dict}).model_dump(
            mode="json", exclude={"name"}
        )
    except (TypeError, ValueError):
        invalid_evidence("payload method is not a normalized method identity")
    if canonical_json_bytes(normalized) != canonical_json_bytes(method_dict):
        invalid_evidence("payload method is not a full normalized label-free identity")
    raw_declarations = method_dict.get("covariates")
    if raw_declarations is None:
        declarations: list[dict[str, object]] = []
    elif isinstance(raw_declarations, list):
        declarations = cast("list[dict[str, object]]", raw_declarations)
    else:
        invalid_evidence("payload method covariates must be a normalized list")
    expected = "cuped" if kind == "scalar_moments" else "cuped_multicovariate"
    if method_dict["variance_reduction"] != expected:
        invalid_evidence(f"{kind} evidence requires a {expected!r} method")
    return canonical_json_bytes(method_dict).decode(), declarations


def _decode_scalar_evidence(
    evidence: Mapping[str, object],
    body: Mapping[str, object],
    declarations: list[dict[str, object]],
    metric: str,
    method_key_value: str,
    population: str,
    trigger_name: str | None,
    source: str,
) -> CupedEvidence:
    if (
        set(evidence) != {"rows"}
        or not isinstance(evidence["rows"], list)
        or len(declarations) > 1
        or (declarations and declarations[0]["kind"] != "continuous")
    ):
        invalid_evidence("scalar evidence must match at most one continuous covariate")
    rows = tuple(_scalar_row(row) for row in evidence["rows"])
    declaration = declarations[0] if declarations else None
    if not rows or any(
        row.get("metric") != metric
        or (
            declaration is not None
            and (
                row.get("covariate_name") != declaration["name"]
                or row.get("missing") != declaration["missing"]
            )
        )
        for row in rows
    ):
        invalid_evidence("scalar evidence rows do not match the declared method")
    digest_body = {
        **body,
        "evidence": {
            "rows": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("covariate_name", "missing")
                }
                for row in rows
            ]
        },
    }
    digest = hashlib.sha256(canonical_json_bytes(digest_body)).hexdigest()
    return CupedEvidence(
        "scalar_moments",
        metric,
        method_key_value,
        population,
        source,
        digest,
        trigger_name=trigger_name,
        rows=rows,
    )


def _decode_categorical_evidence(
    evidence: Mapping[str, object],
    declarations: list[dict[str, object]],
    metric: str,
    method_key_value: str,
    population: str,
    trigger_name: str | None,
    source: str,
    digest: str,
) -> CupedEvidence:
    if not declarations or any(item["kind"] != "categorical" for item in declarations):
        invalid_evidence("categorical evidence method must declare categorical covariates only")
    cells = _cells(
        evidence, categorical_identity(metric, method_key_value, population, trigger_name)
    )
    names = tuple(item["name"] for item in declarations)
    missing = tuple(item["missing"] for item in declarations)
    if cells.covariate_names != names:
        invalid_evidence("categorical evidence covariate names or order differ from the method")
    if tuple(metadata[0] for metadata in cells.null_metadata) != missing:
        invalid_evidence("categorical evidence missing policies differ from the method")
    return CupedEvidence(
        "categorical_cells",
        metric,
        method_key_value,
        population,
        source,
        digest,
        trigger_name=trigger_name,
        cells=cells,
    )


def parse_cuped_evidence(payload: object) -> CupedEvidence:
    """Validate one envelope payload's grammar and decode its evidence."""
    if not isinstance(payload, str):
        invalid_evidence("payload must be a canonical JSON string")
    try:
        body = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError:
        invalid_evidence("payload is not JSON")
    if not isinstance(body, dict):
        invalid_evidence("payload must be a JSON object")
    _normalize_float_fields(body)
    try:
        canonical = canonical_json_bytes(body).decode()
    except CanonicalJSONError as exc:
        invalid_evidence(f"payload is not canonical JSON: {exc}")
    if canonical != payload:
        invalid_evidence("payload bytes are not canonical JSON")
    version = body.get("version")
    if isinstance(version, bool) or version != PAYLOAD_VERSION:
        refuse(_UNSUPPORTED, field="version", received=version, supported=(PAYLOAD_VERSION,))
    if set(body) != _PAYLOAD_FIELDS:
        invalid_evidence(
            f"payload fields must be exactly {sorted(_PAYLOAD_FIELDS)}; got {sorted(body)}"
        )
    kind = body["kind"]
    if kind not in _KINDS:
        refuse(_UNSUPPORTED, field="kind", received=kind, supported=_KINDS)
    metric, population, source = body["metric"], body["population"], body["source_sha256"]
    if not isinstance(metric, str) or not metric:
        invalid_evidence("payload metric must be a non-empty name")
    if population not in _POPULATIONS:
        invalid_evidence(f"payload population must be one of {_POPULATIONS}; got {population!r}")
    trigger_name = body["trigger_name"]
    if trigger_name is not None and (not isinstance(trigger_name, str) or not trigger_name):
        invalid_evidence("payload trigger_name must be a non-empty name or null")
    if population == "triggered" and trigger_name is None:
        invalid_evidence("triggered evidence must preserve its trigger identity")
    if body["breakout"] is not None:
        invalid_evidence(
            "version-1 payloads describe whole-population evidence; breakout must be null"
        )
    if (
        not isinstance(source, str)
        or len(source) != 64
        or not set(source) <= set("0123456789abcdef")
    ):
        invalid_evidence("payload source_sha256 must be a lowercase SHA-256 digest")
    key, declarations = _method_identity(body["method"], kind)
    evidence = body["evidence"]
    if not isinstance(evidence, dict):
        invalid_evidence("payload evidence must be an object")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if kind == "scalar_moments":
        return _decode_scalar_evidence(
            evidence, body, declarations, metric, key, population, trigger_name, source
        )
    if kind == "fitted_multicovariate":
        return _decode_fitted_evidence(
            evidence, declarations, metric, key, population, trigger_name, source, digest
        )
    return _decode_categorical_evidence(
        evidence, declarations, metric, key, population, trigger_name, source, digest
    )


def _normalize_float_fields(body: dict[str, object]) -> None:
    """Restore schema-declared floats before canonical JSON validation."""
    evidence = body.get("evidence")
    if not isinstance(evidence, dict):
        return
    evidence = cast("dict[str, object]", evidence)
    kind = body.get("kind")
    if kind == "scalar_moments":
        _normalize_scalar_float_fields(evidence)
    elif kind == "categorical_cells":
        _normalize_fields(evidence, _CELL_FLOAT_FIELDS)
    elif kind == "fitted_multicovariate":
        _normalize_fitted_float_fields(evidence)


def _normalize_scalar_float_fields(evidence: dict[str, object]) -> None:
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        return
    for item in rows:
        if not isinstance(item, dict):
            continue
        row = cast("dict[str, object]", item)
        for key, value in tuple(row.items()):
            if (
                key not in _COUNT_FIELDS
                and key not in _TEXT_FIELDS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                row[key] = float(value)


def _normalize_float_tree(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_float_tree(item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _normalize_fields(mapping: dict[str, object], fields: Sequence[str]) -> None:
    for field in fields:
        if field in mapping:
            mapping[field] = _normalize_float_tree(mapping[field])


def _normalize_fitted_float_fields(evidence: dict[str, object]) -> None:
    fit_state = evidence.get("fit_state")
    if isinstance(fit_state, dict):
        _normalize_fields(
            cast("dict[str, object]", fit_state),
            (
                "offsets",
                "scales",
                "centered_means",
                "anchor",
                "triangular_factor",
                "transformed_rhs",
                "slopes",
                "adjusted_means",
                "theta",
            ),
        )
    summary = evidence.get("inference_summary")
    if not isinstance(summary, dict):
        return
    summary = cast("dict[str, object]", summary)
    _normalize_fields(
        summary,
        (
            "max_leverage",
            "sum_leverage",
            "adjusted_component_covariance",
            "s2",
            "s4",
            "g",
            "g2",
            "u",
        ),
    )
    scores = summary.get("scaled_projected_score_squares")
    if isinstance(scores, list):
        for item in scores:
            if isinstance(item, list) and len(item) == 7:
                score = cast("list[object]", item)
                score[4:] = [_normalize_float_tree(value) for value in score[4:]]
    specs = summary.get("projection_specs")
    if isinstance(specs, list):
        for item in specs:
            if isinstance(item, dict):
                _normalize_fields(cast("dict[str, object]", item), ("jacobian", "scale", "divisor"))


def local_categorical_cells(source: Any, metric: Any, method: Any) -> CategoricalCells:
    """Reduce a local per-unit source to joint categorical evidence for one request."""
    import narwhals as nw
    import numpy as np

    from increment._evidence_types import unit_frame_covariate_columns
    from increment.estimation.cuped.cells import categorical_cells

    declarations = tuple(method.covariates or ())
    names = tuple(item.name for item in declarations)
    columns = unit_frame_covariate_columns(names)
    frame = nw.from_native(source.unit_frame(metric, covariates=names), eager_only=True)
    order_columns = ["group_id", *columns.values(), "y"]
    if getattr(metric, "type", None) == "ratio":
        order_columns.append("y_den")
    frame = frame.sort(order_columns)
    control = str(source.context.design.control_group)
    groups = [str(group) for group in frame.get_column("group_id").to_list()]
    arm_order = (control, *sorted(set(groups) - {control}))
    ratio = getattr(metric, "type", None) == "ratio"
    y = np.asarray(frame.get_column("y").to_list(), dtype=float)
    population, trigger_name = _source_coordinates(source)
    return categorical_cells(
        None if ratio else y,
        np.asarray(groups, dtype=object),
        identity=categorical_identity(
            metric.name,
            method_key(method),
            population,
            trigger_name,
        ),
        control=control,
        arm_order=arm_order,
        covariates={
            item.name: np.asarray(frame.get_column(columns[item.name]).to_list(), dtype=object)
            for item in declarations
        },
        missing={item.name: item.missing for item in declarations},
        numerator=y if ratio else None,
        denominator=(
            np.asarray(frame.get_column("y_den").to_list(), dtype=float) if ratio else None
        ),
    )


def _source_coordinates(source: Any) -> tuple[str, str | None]:
    population = source.population
    trigger_name = source.trigger_name
    if population not in _POPULATIONS:
        invalid_evidence("source population cannot be preserved in portable CUPED evidence")
    if trigger_name is not None and (not isinstance(trigger_name, str) or not trigger_name):
        invalid_evidence("source trigger identity cannot be preserved in portable CUPED evidence")
    if population == "triggered" and trigger_name is None:
        invalid_evidence("triggered evidence requires a preserved trigger identity")
    return population, trigger_name


PortableRequest = tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]


def portable_requests(
    configs: Sequence[Any], *, design: Any, categorical: bool
) -> tuple[PortableRequest, ...] | None:
    """Resolve method-scoped scalar, cell and compact fitted evidence requests."""
    from increment._analysis_config import effective_methods

    requests = []
    for config in configs:
        methods = effective_methods(config, design=design)
        scalar = {
            method_key(method): method for method in methods if method.variance_reduction == "cuped"
        }
        carried = {
            method_key(method): method
            for method in methods
            if categorical and categorical_request(method)
        }
        fitted = {
            method_key(method): method
            for method in methods
            if method.variance_reduction == "cuped_multicovariate"
            and not categorical_request(method)
        }
        requests.append(
            (config.metric, tuple(scalar.values()), tuple(carried.values()), tuple(fitted.values()))
        )
    if not any(len(scalar) > 1 or carried or fitted for _, scalar, carried, fitted in requests):
        return None
    return tuple(requests)


def _local_multicovariate_evidence(
    source: Any,
    metric: Any,
    method: Any,
    *,
    control: str,
    arm_order: tuple[str, ...],
    population: str,
) -> tuple[tuple[object, ...], tuple[FitState, InferenceSummary, MulticovariateCupedFit]]:
    import math
    from dataclasses import replace

    import narwhals as nw
    import numpy as np

    from increment._evidence_types import unit_frame_covariate_columns
    from increment._source_identity import source_identity
    from increment.estimation.cuped.multicovariate import fit_cuped_multicovariate

    declarations = tuple(method.covariates or ())
    names = tuple(item.name for item in declarations)
    frame = nw.from_native(source.unit_frame(metric, covariates=names), eager_only=True)
    columns = unit_frame_covariate_columns(names)
    ratio = metric.type == "ratio"
    groups = tuple(str(group) for group in frame["group_id"].to_list())
    numerator_values = np.asarray(frame["y"].to_list(), dtype=float)
    denominator_values = np.asarray(frame["y_den"].to_list(), dtype=float) if ratio else None
    withheld_by_arm = {}
    if ratio:
        from increment.errors import CodedError
        from increment.estimation.variance import require_positive_denominator_mean

        assert denominator_values is not None
        for arm in arm_order:
            values = [
                float(value)
                for value, group in zip(denominator_values, groups, strict=True)
                if group == arm
            ]
            if not values:
                continue
            mean = math.fsum(values) / len(values)
            try:
                require_positive_denominator_mean(metric=metric.name, group_id=str(arm), d_bar=mean)
            except CodedError as exc:
                withheld_by_arm[str(arm)] = exc
    admitted = tuple(arm for arm in arm_order if str(arm) not in withheld_by_arm)
    if ratio and withheld_by_arm and (control in withheld_by_arm or len(admitted) < 2):
        raise next(withheld_by_arm[arm] for arm in arm_order if str(arm) in withheld_by_arm)
    selected = np.asarray([group in admitted for group in groups], dtype=bool)
    groups = tuple(group for group, keep in zip(groups, selected, strict=True) if keep)
    state, summary, fit = fit_cuped_multicovariate(
        None if ratio else numerator_values[selected],
        groups,
        control=control,
        arm_order=admitted,
        covariates={
            item.name: np.asarray(frame[columns[item.name]].to_list(), dtype=object)[selected]
            for item in declarations
        },
        kinds={item.name: item.kind for item in declarations},
        missing={item.name: item.missing for item in declarations},
        numerator=numerator_values[selected] if ratio else None,
        denominator=denominator_values[selected]
        if ratio and denominator_values is not None
        else None,
        return_components=True,
        _portable_coordinates=True,
    )
    if withheld_by_arm:
        state = replace(
            state,
            withheld_arms=tuple(arm for arm in arm_order if str(arm) in withheld_by_arm),
            refusal_contexts=tuple(
                (
                    arm,
                    tuple(
                        sorted(
                            {
                                "code": error.code,
                                **dict(error.context),
                            }.items()
                        )
                    ),
                )
                for arm, error in withheld_by_arm.items()
            ),
        )
    identity = (
        "warehouse-cuped-fit-v1",
        metric.name,
        population,
        source_identity(source),
        method_key(method),
        control,
        arm_order,
        None,
        None,
        None,
        False,
    )
    previous_identity = tuple(state.identity)
    state = replace(state, identity=identity)
    identity = state.identity
    summary = replace(
        summary,
        fit_identity=identity,
        evidence_identity=identity,
        projection_identity=identity + ("all-pairs-v1",),
        projection_specs=tuple(
            replace(spec, identity=identity + tuple(spec.identity[len(previous_identity) :]))
            for spec in summary.projection_specs
        ),
    )
    return identity, (state, summary, fit)


def portable_cuped_rows(
    source: Any,
    requests: Sequence[PortableRequest],
    *,
    categorical_cells: Callable[[Any, Any], CategoricalCells] | None,
    identity: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Ordinary outcome rows plus portable scalar, cell and fitted evidence."""
    from increment._source_identity import source_identity
    from increment.declarations import Method
    from increment.errors import CodedError

    if identity is None:
        identity = source_identity(source)
    population, trigger_name = _source_coordinates(source)
    rows: list[dict[str, object]] = []
    for metric, scalar, categorical, fitted in requests:
        cell_evidence: list[tuple[Any, CategoricalCells]] = []
        fitted_methods = list(fitted)
        for method in categorical:
            if categorical_cells is None:
                raise AssertionError("categorical requests need a joint-cell producer")
            try:
                evidence = categorical_cells(metric, method)
            except CodedError as exc:
                if exc.code != "query.cuped.general_qr_required" or exc.context.get(
                    "reason"
                ) not in {
                    "cell_count",
                    "compression",
                    "component_contributors",
                    "component_scatter",
                    "component_range",
                    "encoded_bytes",
                }:
                    raise
                fitted_methods.append(method)
            else:
                cell_evidence.append((method, evidence))
        ordinary_rows = [
            dict(row) for row in source.moments_for_method(metric, Method.unadjusted())
        ]
        rows.extend(ordinary_rows)
        for method in scalar:
            family = source.moments_for_method(metric, method)
            rows.append(
                {
                    "record_kind": CUPED_EVIDENCE_RECORD,
                    CUPED_EVIDENCE_FIELD: scalar_payload(
                        metric.name,
                        method,
                        family,
                        population=population,
                        trigger_name=trigger_name,
                        source=identity,
                    ),
                }
            )
        for method, evidence in cell_evidence:
            rows.append(
                {
                    "record_kind": CUPED_EVIDENCE_RECORD,
                    CUPED_EVIDENCE_FIELD: categorical_payload(
                        metric.name,
                        method,
                        evidence,
                        population=population,
                        trigger_name=trigger_name,
                        source=identity,
                    ),
                }
            )
        for method in fitted_methods:
            arms = tuple(
                str(row["group_id"]) for row in ordinary_rows if row.get("group_id") is not None
            )
            fit_source = source
            control = str(fit_source.context.design.control_group)
            arm_order = (control, *sorted(set(arms) - {control}))
            fit_evidence = getattr(fit_source, "fit_cuped_multicovariate_evidence", None)
            if fit_evidence is not None:
                _fit_identity, compact = fit_evidence(
                    metric,
                    method,
                    control=control,
                    arm_order=arm_order,
                    population=population,
                )
            elif getattr(fit_source, "fit_cuped_multicovariate", None) is not None:
                _fit_identity, compact = fit_source.fit_cuped_multicovariate(
                    metric,
                    method,
                    control=control,
                    arm_order=arm_order,
                    population=population,
                    return_components=True,
                )
            else:
                _fit_identity, compact = _local_multicovariate_evidence(
                    fit_source,
                    metric,
                    method,
                    control=control,
                    arm_order=arm_order,
                    population=population,
                )
            if isinstance(compact, CodedError):
                raise compact
            carrier = compact
            compact = getattr(carrier, "fit", carrier)
            if not isinstance(compact, tuple) or len(compact) != 3:
                invalid_evidence("warehouse did not return compact fitted evidence")
            fit_state, summary, _finalized = compact
            if tuple(fit_state.identity[: len(_fit_identity)]) != tuple(_fit_identity):
                invalid_evidence("warehouse fitted evidence identity differs from its request")
            withheld = tuple(getattr(carrier, "withheld", ()))
            if withheld:
                from dataclasses import replace

                withheld_by_arm = dict(withheld)
                roster = arm_order
                fit_state = replace(
                    fit_state,
                    withheld_arms=tuple(arm for arm in roster if arm in withheld_by_arm),
                    refusal_contexts=tuple(
                        (
                            arm,
                            tuple(
                                sorted(
                                    {
                                        "code": error.code,
                                        **dict(error.context),
                                    }.items()
                                )
                            ),
                        )
                        for arm, error in withheld
                    ),
                )
            rows.append(
                {
                    "record_kind": CUPED_EVIDENCE_RECORD,
                    CUPED_EVIDENCE_FIELD: fitted_multicovariate_payload(
                        metric.name,
                        method,
                        fit_state,
                        summary,
                        population=population,
                        trigger_name=trigger_name,
                        source=identity,
                    ),
                }
            )
    return rows


__all__ = [
    "CUPED_EVIDENCE_FIELD",
    "CUPED_EVIDENCE_FORMAT",
    "CUPED_EVIDENCE_RECORD",
    "CupedEvidence",
    "categorical_request",
    "duplicate_evidence",
    "invalid_evidence",
    "local_categorical_cells",
    "method_key",
    "parse_cuped_evidence",
    "portable_cuped_rows",
    "portable_requests",
    "source_sha256",
]
