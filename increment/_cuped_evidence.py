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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, NoReturn, cast

from increment._canonical import CanonicalJSONError, canonical_json_bytes
from increment.errors import CodedError, RefusalSpec, WireFormatError, refuse
from increment.estimation.cuped.types import CategoricalCells

CUPED_EVIDENCE_FORMAT = 13
CUPED_EVIDENCE_RECORD = "cuped_evidence"
CUPED_EVIDENCE_FIELD = "cuped_evidence"
PAYLOAD_VERSION = 1
EvidenceKind = Literal["scalar_moments", "categorical_cells"]
_KINDS: tuple[EvidenceKind, ...] = ("scalar_moments", "categorical_cells")
_RESERVED_KINDS = ("fitted_multicovariate",)
_PAYLOAD_FIELDS = frozenset(
    {"version", "kind", "metric", "method", "population", "breakout", "source_sha256", "evidence"}
)
_POPULATIONS = ("assigned", "triggered")
_COUNT_FIELDS = frozenset({"n", "successes", "winsor_n", "winsor_n_lower", "winsor_n_upper"})
_TEXT_FIELDS = frozenset({"experiment_id", "metric", "group_id", "x_role", "covariate_name"})
_CELL_FLOAT_FIELDS = (
    "outcome_references",
    "centered_first_moments",
    "symmetric_cross_moments",
    "outcome_magnitudes",
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
    """One validated envelope: method-scoped scalar rows or joint categorical cells."""

    kind: EvidenceKind
    metric: str
    method_key: str
    population: str
    source_sha256: str
    payload_sha256: str
    rows: tuple[Mapping[str, object], ...] = ()
    cells: CategoricalCells | None = None

    def scalar_rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.rows]


def method_key(method: Any) -> str:
    """Canonical identity of a request, excluding only its display label."""
    return canonical_json_bytes(method.model_dump(mode="json", exclude={"name"})).decode()


def source_sha256(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def categorical_identity(metric: str, key: str, population: str) -> tuple[object, ...]:
    """Evidence identity shared by partitions of one request, independent of the source."""
    return ("cuped_evidence", metric, population, None, key)


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
    source: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str:
    body = {
        "version": PAYLOAD_VERSION,
        "kind": kind,
        "metric": metric,
        "method": json.loads(method_key(method)),
        "population": population,
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
    source: Mapping[str, object],
) -> str:
    return _payload(
        "scalar_moments",
        metric,
        method,
        population=population,
        source=source,
        evidence={"rows": [dict(row) for row in rows]},
    )


def categorical_payload(
    metric: str,
    method: Any,
    cells: CategoricalCells,
    *,
    population: str,
    source: Mapping[str, object],
) -> str:
    evidence = {name: getattr(cells, name) for name in sorted(_CELL_FIELDS)}
    return _payload(
        "categorical_cells",
        metric,
        method,
        population=population,
        source=source,
        evidence=evidence,
    )


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
    method = body["method"]
    if not isinstance(method, dict) or method.get("variance_reduction") not in (
        "cuped",
        "cuped_multicovariate",
    ):
        invalid_evidence("payload method must be a CUPED method identity")
    expected = "cuped" if kind == "scalar_moments" else "cuped_multicovariate"
    if method["variance_reduction"] != expected:
        invalid_evidence(f"{kind} evidence requires a {expected!r} method")
    key = canonical_json_bytes(method).decode()
    evidence = body["evidence"]
    if not isinstance(evidence, dict):
        invalid_evidence("payload evidence must be an object")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if kind == "scalar_moments":
        if set(evidence) != {"rows"} or not isinstance(evidence["rows"], list):
            invalid_evidence("scalar evidence must hold exactly one rows list")
        rows = tuple(_scalar_row(row) for row in evidence["rows"])
        if not rows or any(row.get("metric") != metric for row in rows):
            invalid_evidence(f"scalar evidence rows must belong to metric {metric!r}")
        return CupedEvidence(kind, metric, key, population, source, digest, rows=rows)
    cells = _cells(evidence, categorical_identity(metric, key, population))
    return CupedEvidence(
        cast("EvidenceKind", kind), metric, key, population, source, digest, cells=cells
    )


def local_categorical_cells(source: Any, metric: Any, method: Any) -> CategoricalCells:
    """Reduce a local per-unit source to joint categorical evidence for one request."""
    import narwhals as nw
    import numpy as np

    from increment._source_types import unit_frame_covariate_columns
    from increment.estimation.cuped.cells import categorical_cells

    declarations = tuple(method.covariates or ())
    names = tuple(item.name for item in declarations)
    columns = unit_frame_covariate_columns(names)
    frame = nw.from_native(source.unit_frame(metric, covariates=names), eager_only=True)
    control = str(source.context.design.control_group)
    groups = [str(group) for group in frame.get_column("group_id").to_list()]
    arm_order = tuple(dict.fromkeys((control, *groups)))
    ratio = getattr(metric, "type", None) == "ratio"
    y = np.asarray(frame.get_column("y").to_list(), dtype=float)
    return categorical_cells(
        None if ratio else y,
        np.asarray(groups, dtype=object),
        identity=categorical_identity(metric.name, method_key(method), _population(source)),
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


def _population(source: Any) -> str:
    return str(getattr(source, "population", "assigned"))


PortableRequest = tuple[Any, tuple[Any, ...], tuple[Any, ...]]


def portable_requests(
    configs: Sequence[Any], *, design: Any, categorical: bool
) -> tuple[PortableRequest, ...] | None:
    """Each metric's distinct scalar and carried categorical requests, or ``None`` for format 12.

    Format 13 is needed only when a metric resolves to more than one scalar
    family or to a carried categorical-only multicovariate request. Requests
    differing only by label share one entry. Without *categorical*, categorical
    requests stay with a source that serves them directly. Requests with
    continuous mixed covariates have no compact evidence in this version and
    are omitted, as in format 12.
    """
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
        requests.append((config.metric, tuple(scalar.values()), tuple(carried.values())))
    if not any(len(scalar) > 1 or carried for _, scalar, carried in requests):
        return None
    return tuple(requests)


def portable_cuped_rows(
    source: Any,
    requests: Sequence[PortableRequest],
    *,
    categorical_cells: Callable[[Any, Any], CategoricalCells] | None,
    identity: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Ordinary outcome rows plus one envelope per scalar family and categorical request.

    Ordinary rows carry outcome moments only. Categorical evidence is reduced
    before any moment query, so an ineligible request refuses first.
    """
    from increment._source_identity import source_identity
    from increment.estimation.engine import Method

    if identity is None:
        identity = source_identity(source)
    population = _population(source)
    cells: list[list[CategoricalCells]] = []
    for metric, _, categorical in requests:
        if not categorical:
            cells.append([])
            continue
        if categorical_cells is None:
            raise AssertionError("categorical requests need a joint-cell producer")
        cells.append([categorical_cells(metric, method) for method in categorical])
    rows: list[dict[str, object]] = []
    for (metric, scalar, categorical), produced in zip(requests, cells, strict=True):
        rows.extend(dict(row) for row in source.moments_for_method(metric, Method.unadjusted()))
        for method in scalar:
            family = source.moments_for_method(metric, method)
            rows.append(
                {
                    "record_kind": CUPED_EVIDENCE_RECORD,
                    CUPED_EVIDENCE_FIELD: scalar_payload(
                        metric.name, method, family, population=population, source=identity
                    ),
                }
            )
        for method, evidence in zip(categorical, produced, strict=True):
            rows.append(
                {
                    "record_kind": CUPED_EVIDENCE_RECORD,
                    CUPED_EVIDENCE_FIELD: categorical_payload(
                        metric.name, method, evidence, population=population, source=identity
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
