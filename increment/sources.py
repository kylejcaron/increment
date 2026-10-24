"""Seam between data sources and readouts.

Must not import ibis or narwhals, so one readout implementation works
against a warehouse table, an in-memory frame, or a moments cube through
the shared `MomentSource` protocol.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from increment._canonical import canonical_json_bytes
from increment._cuped_evidence import (
    CUPED_EVIDENCE_FIELD,
    CUPED_EVIDENCE_FORMAT,
    CUPED_EVIDENCE_RECORD,
    CupedEvidence,
    duplicate_evidence,
    invalid_evidence,
    local_categorical_cells,
    method_key,
    parse_cuped_evidence,
    portable_cuped_rows,
    portable_requests,
    source_sha256,
)
from increment._immutable import _FrozenMapping
from increment._labels import (
    MIXED_ASSIGNMENT_LABEL,
    UNASSIGNED_LABEL,
    boolean_group_count_aliases,
)
from increment._moment_plan import X_SLOT_ROLES
from increment._source_identity import (
    source_identities_equal,
    source_identity,
    source_identity_is_canonical,
)
from increment._source_types import (
    ComplianceArm,
    ComplianceSummary,
    CUPEDEvidenceExportSource,
    EvidenceCapabilities,
    Grain,
    MomentSource,
    ScalarEvidenceDeclaration,
    SourceContext,
    invalid_compliance_state,
)
from increment.errors import (
    CapabilityError as _CapabilityError,
)
from increment.errors import (
    CodedError as _CodedError,
)
from increment.errors import (
    RefusalSpec as _RefusalSpec,
)
from increment.errors import (
    WireFormatError as _WireFormatError,
)
from increment.errors import (
    refuse as _refuse,
)
from increment.sequential_source import SequentialSourceMixin


def _health_snapshot_id(source: Any, cutoff: str | None, health_evidence: object) -> str:
    """Fingerprint health evidence together with its source identity and cutoff."""
    from hashlib import sha256
    from math import isfinite

    def hash_safe(value: object) -> object:
        if isinstance(value, float) and not isfinite(value):
            marker = "nan" if value != value else ("inf" if value > 0 else "-inf")
            return {"$nonfinite_float": marker}
        if isinstance(value, Mapping):
            return {str(key): hash_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [hash_safe(item) for item in value]
        return value

    fingerprint = hash_safe(
        {
            "source": source_identity(source),
            "cutoff": cutoff,
            "health_evidence": health_evidence,
        }
    )
    return "sha256:" + sha256(canonical_json_bytes(fingerprint)).hexdigest()


_WINSOR_METADATA_MISSING = _RefusalSpec(
    "moments.winsorization.metadata_missing",
    _CapabilityError,
    template="winsorization metadata is not declared for metric {metric!r}; field {field!r} has value {value!r}",
)
_WINSOR_METADATA_CONFLICT = _RefusalSpec(
    "moments.winsorization.metadata_conflict",
    _CapabilityError,
    template=(
        "winsorization metadata for metric {metric!r} conflicts on {side!r} bound: "
        "expected percentile {expected_percentile!r}, expected bound {expected_bound!r}, "
        "received percentile {actual_percentile!r}, bound {actual_bound!r}, count {actual_n!r}"
    ),
)
_WINSOR_METADATA_SHAPE = _RefusalSpec(
    "moments.winsorization.metadata_shape",
    _CapabilityError,
    template="winsorization metadata for metric {metric!r} has invalid field {field!r} value {value!r}",
)

SourceOperation = Literal[
    "allocation_history",
    "readout_snapshot",
    "dashboard_group_data",
    "materialize",
    "triggered_counts",
    "triggered_source",
    "sitewide_evidence",
    "unit_counts",
    "export_moments",
    "moments_source",
    "panel_sql",
    "summary_sql",
    "breakout_summaries",
    "factor_summaries",
    "breakout_source",
    "breakout_sources",
    "day_source",
    "exploratory_source",
    "artifact_experiment",
]


def require_operation[T](
    source: MomentSource,
    operation: SourceOperation,
    protocol: type[T],
) -> T:
    """Return *source* after checking a declared operation is callable."""
    if operation not in getattr(source, "operations", frozenset()) or not isinstance(
        source, protocol
    ):
        from increment._evidence_refusals import refuse_operation

        refuse_operation(
            source,
            operation,
            tuple(sorted(getattr(source, "operations", ()))),
            route="use a source that implements the requested operation",
        )
    return cast("T", source)


if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import AnalysisPlan, ExperimentMetric, Metric

_MOMENTS_COUNTS = _RefusalSpec(
    "source.moments.assignment_counts",
    _CapabilityError,
    template="trustworthy source-level assignment counts; use Analysis.export() from a source or artifact that carries assignment-count evidence, or supply a cube whose rows carry the assignment_counts field.",
)


def refuse_cluster_grain_transport(
    *,
    operation: str,
    source: str,
    cluster: str,
    design: object | None,
    route_forward: str,
) -> NoReturn:
    from increment._evidence_refusals import CLUSTER_UNAVAILABLE
    from increment.errors import refuse

    refuse(
        CLUSTER_UNAVAILABLE,
        constructor=source,
        requested=cluster,
        offered=(),
        route=route_forward,
    )


SOURCE_QUANTILE_NO_MOMENTS = _RefusalSpec(
    "source.frame.quantile_no_moments",
    _CapabilityError,
    template="quantile metric {metric!r} has no moment representation; it is served through unit_frame. {route}",
)


def refuse_quantile_moments_export(
    metrics: Sequence[Any],
    *,
    design: object | None,
    observational_refusal: Callable[[Any], object],
) -> None:
    """Refuse exporting a quantile: a moments cube holds additive moments, never the per-unit
    values an order statistic needs, so a quantile exported there would carry mean moments
    under the quantile's name. Runs on catalog metadata alone, before any count or moment is
    read or a file written.

    Under an observational design the estimator refusal (``readout.observational.quantile``)
    takes precedence, because no estimator could use the cube. It lives below this module, so
    the caller supplies it as ``observational_refusal``; that callable owns any source
    authentication and raises for the first quantile in catalog order.
    """
    quantile = next(
        (metric for metric in metrics if getattr(metric, "type", None) == "quantile"), None
    )
    if quantile is None:
        return
    if getattr(design, "mechanism", None) == "observational":
        observational_refusal(quantile)
        return
    _refuse(
        SOURCE_QUANTILE_NO_MOMENTS,
        metric=quantile.name,
        route="estimate the quantile with run(), or export a cube of the non-quantile metrics",
    )


_BREAKOUT_COUNTS = _RefusalSpec(
    "source.breakout.assignment_counts",
    _CapabilityError,
    template="unit_counts() is not supported on a breakout-scoped moments source -- its rows are per (group_id, dimension_value), so a per-group count would silently report one segment's n as the whole arm's. Use DefinitionsMomentSource.moments_source() for total-grain counts.",
)
_BREAKOUT_COMPLIANCE = _RefusalSpec(
    "source.breakout.compliance_summary",
    _CapabilityError,
    template="compliance_summary() is not supported on a breakout-scoped moments source -- design-level compliance is a whole-population cohort, never conditioned on a breakout dimension. Read it off the unscoped MomentSource.",
)


# Fixed and sequential formats are separate sequences. Fixed format 12 carries
# ``cden3`` and format 11 added source identity; format 13 adds method-scoped
# CUPED evidence only when an export needs it. Formats 10 and 11 stay readable
# with the denominator-skew check unavailable.
MOMENTS_FORMAT = 12
SEQUENTIAL_MOMENTS_FORMAT = 10
_PREVIOUS_FIXED_MOMENTS_FORMATS = frozenset({10, 11})
_PREVIOUS_SEQUENTIAL_MOMENTS_FORMAT = 9
_CURRENT_FIXED_MOMENTS_FORMATS = frozenset({MOMENTS_FORMAT, CUPED_EVIDENCE_FORMAT})
_SUPPORTED_FIXED_MOMENTS_FORMATS = _PREVIOUS_FIXED_MOMENTS_FORMATS | _CURRENT_FIXED_MOMENTS_FORMATS
_SUPPORTED_SEQUENTIAL_MOMENTS_FORMATS = frozenset(
    {_PREVIOUS_SEQUENTIAL_MOMENTS_FORMAT, SEQUENTIAL_MOMENTS_FORMAT}
)
_SUPPORTED_MOMENTS_FORMATS = (
    _SUPPORTED_FIXED_MOMENTS_FORMATS | _SUPPORTED_SEQUENTIAL_MOMENTS_FORMATS
)
SOURCE_IDENTITY_FIELD = "source_identity"


def _legacy_moments_source_identity(version: int, study_id: str) -> dict[str, object] | None:
    if version not in _PREVIOUS_FIXED_MOMENTS_FORMATS | {_PREVIOUS_SEQUENTIAL_MOMENTS_FORMAT}:
        return None
    return {
        "legacy_moments_format": version,
        "study_id": study_id,
        "artifacts": [],
    }


ASSIGNMENT_COUNTS_FIELD = "assignment_counts"
DECISION_PLAN_FIELD = "decision_plan"
COMPLIANCE_SUMMARY_FIELD = "compliance_summary"
SEQUENTIAL_SNAPSHOT_FIELD = "sequential_snapshot"
TRIGGER_NAME_FIELD = "trigger_name"
WINSORIZATION_MOMENT_FIELDS = (
    "winsor_lower_percentile",
    "winsor_upper_percentile",
    "winsor_lower_bound",
    "winsor_upper_bound",
    "winsor_n",
    "winsor_n_lower",
    "winsor_n_upper",
)


# Censoring above this fraction of enrolled units counts as "material";
# unit_totals and frame's windowed panel share the threshold.
CENSOR_WARN_FRACTION = 0.10
_MOMENTS_PLAN_INVALID = _RefusalSpec(
    "moments.plan.invalid",
    _WireFormatError,
    template="moments cube has invalid {field!r}: {value!r}; {constraint}",
)
_MOMENTS_PLAN_CONFLICT = _RefusalSpec(
    "moments.plan.conflict",
    _WireFormatError,
    template="moments cube has conflicting {field!r}: {value!r}; {constraint}",
)
_MOMENTS_DUPLICATE_KEY = _RefusalSpec(
    "moments.duplicate_key", _WireFormatError, template="duplicate key {key!r}"
)
_MOMENTS_WINSORIZATION_FIELDS_MISSING = _RefusalSpec(
    "moments.v7_rows_missing_winsorization_fields",
    _WireFormatError,
    template="moments rows are missing canonical winsorization metadata fields: {missing}",
)
_MOMENTS_COUNT_FIELD_MISSING = _RefusalSpec(
    "moments.count_field_missing",
    _WireFormatError,
    template="current moments rows require {field!r}; re-export from the original data",
)
_MOMENTS_DENOMINATOR_THIRD_MOMENT_MISSING = _RefusalSpec(
    "moments.denominator_third_moment_missing",
    _WireFormatError,
    template="moments_format {version} denominator rows require the 'cden3' field; row (metric={metric!r}, group_id={group_id!r}) omits that field. Re-export from the original data, or keep the file at the format that wrote it.",
)
_MOMENTS_COUNT_NOT_INTEGER = _RefusalSpec(
    "moments.count_not_integer",
    _WireFormatError,
    template="moments field {field!r} must retain an integer, not {value!r}; re-export from the original data",
)
_MOMENTS_COUNT_OUT_OF_RANGE = _RefusalSpec(
    "moments.count_out_of_range",
    _WireFormatError,
    template="moments field {field!r} has invalid count {value!r} for n={n}; require n >= 1 and 0 <= successes <= n when successes is present",
)
_READOUT_METRIC_QUANTILE_GRAIN = _RefusalSpec(
    "readout.metric.quantile_grain",
    _CapabilityError,
    lambda *, metric, grain: (
        f"quantile metric {metric!r} cannot be estimated at {grain!r} grain -- "
        + (
            "quantiles do not decompose into per-day moments"
            if grain in ("daily", "asof")
            else "quantiles do not decompose into moment rows"
        )
    ),
)

_MOMENTS_FORMAT_LEGACY = _RefusalSpec(
    "moments.format.unsupported_legacy",
    _WireFormatError,
    template="moments_format {received!r} is a legacy format; this reader requires format {required}. Re-export the cube from the current Analysis.",
)
_MOMENTS_FORMAT_FUTURE = _RefusalSpec(
    "moments.format.unsupported_future",
    _WireFormatError,
    template="moments_format {received!r} is newer than the required format {required}. Upgrade increment to read this cube.",
)
_MOMENTS_FORMAT_INVALID = _RefusalSpec(
    "moments.format.invalid",
    _WireFormatError,
    template="moments_format must be an exact integer version; got {received!r}; required format is {required}.",
)
_MOMENTS_FORMAT_MIXED = _RefusalSpec(
    "moments.format.mixed",
    _WireFormatError,
    template="moments cube mixes wire formats {seen!r}; one cube must use format {required}.",
)

_MOMENTS_DUPLICATE_ROWS = _RefusalSpec(
    "moments.rows.duplicate",
    _WireFormatError,
    template="moments cube repeats (metric, group_id)={key!r}; one cube is one experiment's analysis with exactly one row per arm.",
)

_MOMENTS_EXPERIMENT_CONFLICT = _RefusalSpec(
    "moments.rows.experiment_conflict",
    _WireFormatError,
    template="one moments cube cannot combine experiment identities {experiments!r}",
)


def _validate_moment_experiment_identity(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Refuse a cube whose rows carry more than one experiment identity.

    A duplicate (metric, group_id) key does not imply mixed experiments --
    disjoint arm names across two experiments would otherwise estimate lift
    from unrelated cohorts. Homogeneous anonymous cubes (every row missing
    or null experiment_id) are fine; mixing anonymous and named identities,
    or two distinct named ones, is not.
    """
    identities = {
        None if row.get("experiment_id") is None else str(row["experiment_id"]) for row in rows
    }
    if len(identities) > 1:
        experiments = tuple(sorted(identities, key=lambda value: (value is not None, value or "")))
        _refuse(_MOMENTS_EXPERIMENT_CONFLICT, experiments=experiments)


def _parse_moments_format(stamp: object, *, classify: bool = True) -> int:
    """Accept fixed-horizon supported versions; checkpoints have their own gate."""
    if isinstance(stamp, bool) or isinstance(stamp, Fraction):
        _refuse(_MOMENTS_FORMAT_INVALID, received=stamp, required=MOMENTS_FORMAT)
    if isinstance(stamp, Integral):
        version = int(stamp)
    elif isinstance(stamp, Real):
        value = float(stamp)
        if not math.isfinite(value) or not value.is_integer():
            _refuse(_MOMENTS_FORMAT_INVALID, received=stamp, required=MOMENTS_FORMAT)
        version = int(value)
    elif isinstance(stamp, str) and stamp.isdecimal():
        version = int(stamp)
    else:
        _refuse(_MOMENTS_FORMAT_INVALID, received=stamp, required=MOMENTS_FORMAT)
    if classify and version not in _SUPPORTED_MOMENTS_FORMATS:
        if version < min(_SUPPORTED_MOMENTS_FORMATS):
            _refuse(_MOMENTS_FORMAT_LEGACY, received=version, required=MOMENTS_FORMAT)
        if version > max(_SUPPORTED_MOMENTS_FORMATS):
            _refuse(_MOMENTS_FORMAT_FUTURE, received=version, required=MOMENTS_FORMAT)
    return version


def _refuse_unsupported_fixed_format(version: int) -> NoReturn:
    """A fixed-horizon row stamped outside the supported fixed formats: a
    later one is future, anything else (a sequential-only stamp on a fixed
    row) is invalid; stamps below every supported format were already
    classified legacy."""
    if version > max(_SUPPORTED_FIXED_MOMENTS_FORMATS):
        _refuse(_MOMENTS_FORMAT_FUTURE, received=version, required=MOMENTS_FORMAT)
    _refuse(_MOMENTS_FORMAT_INVALID, received=version, required=MOMENTS_FORMAT)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``object_pairs_hook`` that refuses a JSON object with a repeated key.

    Plain ``json.loads`` keeps the last of duplicate keys, silently
    discarding earlier values -- unacceptable for a wire payload that
    feeds trustworthy assignment counts.
    """
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            _refuse(_MOMENTS_DUPLICATE_KEY, key=key)
        seen[key] = value
    return seen


def _parse_assignment_counts(payload: object) -> dict[str, int] | None:
    """Decode a source-level assignment-count payload."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        except _CodedError:
            raise
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(payload, Mapping):
        return None
    counts: dict[str, int] = {}
    for group, count in payload.items():
        if (
            not isinstance(group, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            return None
        counts[group] = count
    return counts


def _parse_compliance_payload(payload: object) -> dict[str, object]:
    """Decode current state; malformed payloads are never treated as legacy."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        except _CodedError:
            raise
        except (json.JSONDecodeError, ValueError):
            invalid_compliance_state("invalid JSON or duplicate compliance keys")
    if not isinstance(payload, Mapping):
        invalid_compliance_state("compliance payload must be an object")
    return dict(cast("Mapping[str, object]", payload))


def _strip_sequential_envelope(stripped: dict[str, object], *, version: int, n_rows: int) -> bool:
    from increment.sequential_state import sequential_refuse, snapshot_from_json

    checkpoint = stripped.pop(SEQUENTIAL_SNAPSHOT_FIELD, None)
    kind = stripped.pop("record_kind", None)
    if kind != "sequential_checkpoint":
        if checkpoint is not None or kind is not None:
            sequential_refuse("source.invalid", "checkpoint state requires a typed envelope")
        return False
    if not isinstance(checkpoint, str):
        sequential_refuse("source.invalid", "checkpoint envelope requires its snapshot payload")
    if version not in _SUPPORTED_SEQUENTIAL_MOMENTS_FORMATS:
        sequential_refuse(
            "continuation.legacy", "checkpoint envelopes require a supported sequential format"
        )
    if n_rows != 1:
        sequential_refuse("source.invalid", "checkpoint envelopes cannot mix with moment rows")
    snapshot = snapshot_from_json(checkpoint)
    if stripped != {"experiment_id": snapshot.registration.source_id}:
        sequential_refuse("source.invalid", "checkpoint envelope source or fields disagree")
    return True


def _validate_moments_plan_format_pair(plans: set[str], *, sequential_checkpoint: bool) -> None:
    """Reject a sequential/fixed plan before compiled-plan semantic decoding."""
    if not plans:
        return
    from increment.sequential_state import sequential_refuse

    for payload in plans:
        try:
            wire = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(wire, Mapping):
            continue
        inference = wire.get("inference")
        kind = inference.get("kind") if isinstance(inference, Mapping) else None
        if not sequential_checkpoint and kind in ("always_valid", "asymptotic_mean"):
            sequential_refuse(
                "continuation.legacy",
                "legacy sequential plans cannot replay without the current checkpoint envelope",
            )
        if sequential_checkpoint and kind == "fixed":
            sequential_refuse(
                "source.invalid",
                "sequential checkpoint format requires a sequential checkpoint plan",
            )


def _cube_assignment_counts(
    rows: Sequence[Mapping[str, object]], *, boolean_count_aliases: Mapping[str, str]
) -> dict[str, int] | None:
    candidate = None
    for row in rows:
        decoded = _parse_assignment_counts(row.get(ASSIGNMENT_COUNTS_FIELD))
        if decoded is not None and boolean_count_aliases:
            normalized: dict[str, int] = {}
            for group, count in decoded.items():
                identity = boolean_count_aliases.get(group, group)
                if identity in normalized:
                    return None
                normalized[identity] = count
            decoded = normalized
        if decoded is None or (candidate is not None and decoded != candidate):
            return None
        candidate = decoded
    return candidate


def _validated_cube_assignment_counts(
    rows: Sequence[Mapping[str, object]],
    *,
    boolean_count_aliases: Mapping[str, str],
    design_summary: bool,
) -> dict[str, int] | None:
    counts = _cube_assignment_counts(rows, boolean_count_aliases=boolean_count_aliases)
    if design_summary and counts is None:
        invalid_compliance_state(
            "design_summary requires valid assignment counts; re-export the complete source"
        )
    return counts


def _cube_compliance_state(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    present = [row for row in rows if COMPLIANCE_SUMMARY_FIELD in row]
    if not present:
        return None
    if len(present) != len(rows):
        invalid_compliance_state("compliance state is missing from some cube rows")
    candidate = _parse_compliance_payload(present[0][COMPLIANCE_SUMMARY_FIELD])
    for row in present[1:]:
        if _parse_compliance_payload(row[COMPLIANCE_SUMMARY_FIELD]) != candidate:
            invalid_compliance_state("cube rows disagree on compliance state")
    return candidate


def _validate_moment_counts(row: Mapping[str, object]) -> None:
    """Count columns never pass through floating-point serialization."""
    for field in ("n", "successes"):
        if field not in row:
            _refuse(_MOMENTS_COUNT_FIELD_MISSING, field=field)
        value = row[field]
        if field == "successes" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral):
            _refuse(_MOMENTS_COUNT_NOT_INTEGER, field=field, value=value)
        minimum = 1 if field == "n" else 0
        count = int(value)
        n = cast("int", row["n"])
        if count < minimum or (field == "successes" and count > n):
            _refuse(_MOMENTS_COUNT_OUT_OF_RANGE, field=field, value=count, n=int(n))


def _validate_moment_row(row: Mapping[str, object], version: int) -> None:
    """Count and current-format field checks for one ordinary moments row."""
    _validate_moment_counts(row)
    _validate_denominator_third_moment(row, version)


def _validate_denominator_third_moment(row: Mapping[str, object], version: int) -> None:
    """A current-format denominator row carries ``cden3``; the previous format
    predates it and loads with the ratio coverage check unavailable."""
    if version not in _CURRENT_FIXED_MOMENTS_FORMATS or _is_null(row.get("ref_den")):
        return
    if "cden3" not in row:
        _refuse(
            _MOMENTS_DENOMINATOR_THIRD_MOMENT_MISSING,
            version=version,
            metric=row.get("metric"),
            group_id=row.get("group_id"),
        )


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_source_identity(payload: object) -> dict[str, object] | None:
    if payload is None:
        return None
    if not isinstance(payload, str):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received=payload,
            required="a source identity JSON object",
        )
    try:
        identity = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except _CodedError:
        raise
    except (json.JSONDecodeError, ValueError):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="invalid source identity JSON",
            required="a source identity JSON object",
        )
    if not isinstance(identity, Mapping) or any(not isinstance(key, str) for key in identity):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received=identity,
            required="a source identity JSON object",
        )
    if not source_identity_is_canonical(identity):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="noncanonical source identity",
            required="a canonical JSON source identity",
        )
    return dict(cast("Mapping[str, object]", identity))


def _strip_moments_envelope_row(row, row_count, *, require_source_identity: bool):
    stripped = dict(row)
    if "moments_format" not in stripped:
        _refuse(_MOMENTS_FORMAT_LEGACY, received=None, required=MOMENTS_FORMAT)
    stamp = stripped.pop("moments_format")
    version = _parse_moments_format(stamp, classify=False)
    payload = stripped.pop(DECISION_PLAN_FIELD, None)
    if payload is not None and not isinstance(payload, str):
        _refuse(_MOMENTS_FORMAT_INVALID, received=payload, required=MOMENTS_FORMAT)
    stripped.pop(ASSIGNMENT_COUNTS_FIELD, None)
    stripped.pop(COMPLIANCE_SUMMARY_FIELD, None)
    source_identity = _parse_source_identity(stripped.pop(SOURCE_IDENTITY_FIELD, None))
    has_trigger = TRIGGER_NAME_FIELD in stripped
    trigger_name = stripped.pop(TRIGGER_NAME_FIELD, None)
    if (
        has_trigger
        and trigger_name is not None
        and (not isinstance(trigger_name, str) or not trigger_name)
    ):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received=trigger_name,
            required="a non-empty declared trigger name or null",
        )
    envelope_identity = None
    evidence_payload = None
    kind = stripped.get("record_kind")
    sequential_checkpoint = kind == "sequential_checkpoint"
    if kind == "design_summary":
        if version not in _SUPPORTED_FIXED_MOMENTS_FORMATS:
            _refuse_unsupported_fixed_format(version)
        if (
            version in _CURRENT_FIXED_MOMENTS_FORMATS
            and require_source_identity
            and source_identity is None
        ):
            _refuse(
                _MOMENTS_FORMAT_INVALID,
                received=None,
                required="source identity in current moments format",
            )
        if row_count != 1:
            invalid_compliance_state("design_summary requires exactly one complete envelope")
        identity = stripped.pop("experiment_id", None)
        if not isinstance(identity, str) or not identity:
            invalid_compliance_state("design_summary requires its exported experiment identity")
        envelope_identity = identity
        if set(stripped) != {"record_kind"}:
            invalid_compliance_state("design_summary cannot contain outcome or checkpoint fields")
        if row.get(ASSIGNMENT_COUNTS_FIELD) is None or row.get(COMPLIANCE_SUMMARY_FIELD) is None:
            invalid_compliance_state(
                "design_summary requires assignment counts and compliance state; "
                "re-export the complete Encouragement source"
            )
        stripped.pop("record_kind")
        output_row = None
    elif kind == CUPED_EVIDENCE_RECORD:
        stripped.pop("record_kind")
        evidence_payload = stripped.pop(CUPED_EVIDENCE_FIELD, None)
        if version != CUPED_EVIDENCE_FORMAT:
            invalid_evidence(
                f"evidence envelopes require moments format {CUPED_EVIDENCE_FORMAT}; got {version}"
            )
        if require_source_identity and source_identity is None:
            invalid_evidence("evidence envelopes require the cube's source identity")
        carried = sorted(key for key, value in stripped.items() if not _is_null(value))
        if carried:
            invalid_evidence(f"envelope rows carry only transport fields; found {carried}")
        output_row = None
    elif _strip_sequential_envelope(stripped, version=version, n_rows=row_count):
        if (
            version == SEQUENTIAL_MOMENTS_FORMAT
            and require_source_identity
            and source_identity is None
        ):
            _refuse(
                _MOMENTS_FORMAT_INVALID,
                received=None,
                required="source identity in current sequential moments format",
            )
        output_row = None
    else:
        if (
            version in _CURRENT_FIXED_MOMENTS_FORMATS
            and require_source_identity
            and source_identity is None
        ):
            _refuse(
                _MOMENTS_FORMAT_INVALID,
                received=None,
                required="source identity in current moments format",
            )
        if not _is_null(stripped.pop(CUPED_EVIDENCE_FIELD, None)):
            invalid_evidence("ordinary outcome rows cannot carry an evidence payload")
        output_row = stripped
    return (
        version,
        payload,
        has_trigger,
        trigger_name,
        envelope_identity,
        output_row,
        source_identity,
        sequential_checkpoint,
        evidence_payload,
    )


def _canonicalize_boolean_moment_group_ids(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], dict[str, str]]:
    raw_group_ids = tuple(row.get("group_id") for row in rows if row.get("group_id") is not None)
    count_labels = tuple(
        label
        for row in rows
        if (counts := _parse_assignment_counts(row.get(ASSIGNMENT_COUNTS_FIELD))) is not None
        for label in counts
    )
    boolean_count_aliases = boolean_group_count_aliases(raw_group_ids, count_labels)
    normalized = [
        {**row, "group_id": "true" if row.get("group_id") else "false"}
        if isinstance(row.get("group_id"), bool)
        else row
        for row in rows
    ]
    return normalized, boolean_count_aliases


def _check_moments_format(
    rows: Sequence[Mapping[str, object]],
    *,
    require_plan: bool = True,
    require_source_identity: bool = True,
) -> tuple[
    int,
    list[dict[str, object]],
    str | None,
    dict[str, int] | None,
    dict[str, object] | None,
    str | None,
    str | None,
    dict[str, object] | None,
    tuple[CupedEvidence, ...],
]:
    """Validate and strip moments format, plan, count, and source envelopes."""
    rows, boolean_count_aliases = _canonicalize_boolean_moment_group_ids(rows)
    out: list[dict[str, object]] = []
    versions: set[int] = set()
    plans: set[str] = set()
    envelope_identity: str | None = None
    source_identities: list[dict[str, object]] = []
    sequential_rows: set[bool] = set()
    trigger_values: list[object] = []
    trigger_rows = 0
    unplanned = 0
    planned_rows = 0
    evidence_payloads: list[object] = []
    for row in rows:
        (
            version,
            payload,
            has_trigger,
            trigger_name,
            row_identity,
            output_row,
            source_identity,
            sequential_checkpoint,
            evidence_payload,
        ) = _strip_moments_envelope_row(
            row, len(rows), require_source_identity=require_source_identity
        )
        if evidence_payload is not None or row.get("record_kind") == CUPED_EVIDENCE_RECORD:
            evidence_payloads.append(evidence_payload)
        versions.add(version)
        sequential_rows.add(sequential_checkpoint)
        if source_identity is not None:
            source_identities.append(source_identity)
        if payload is None:
            unplanned += 1
        else:
            planned_rows += 1
            if not isinstance(payload, str):
                _refuse(_MOMENTS_FORMAT_INVALID, received=payload, required=MOMENTS_FORMAT)
            plans.add(payload)
        if has_trigger:
            trigger_rows += 1
            trigger_values.append(trigger_name)
        if row_identity is not None:
            envelope_identity = row_identity
        if output_row is not None:
            out.append(output_row)
    if len(versions) > 1:
        _refuse(
            _MOMENTS_FORMAT_MIXED,
            seen=sorted(versions),
            required=MOMENTS_FORMAT,
        )
    if len(sequential_rows) > 1:
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="mixed fixed and sequential records",
            required="one moments source kind",
        )
    cube_identity = _cube_source_identity(source_identities, len(rows))
    version = next(iter(versions), MOMENTS_FORMAT)
    _parse_moments_format(version)
    sequential_checkpoint = next(iter(sequential_rows), False)
    if not sequential_checkpoint and version not in _SUPPORTED_FIXED_MOMENTS_FORMATS:
        _refuse_unsupported_fixed_format(version)
    _validate_moments_plan_format_pair(plans, sequential_checkpoint=sequential_checkpoint)
    _validate_cube_plan_payload(
        plans,
        unplanned,
        planned_rows,
        require_plan=require_plan or envelope_identity is not None,
        has_rows=bool(rows),
    )
    _validate_moment_experiment_identity(out)
    seen_keys: set[tuple[str, str]] = set()
    for row in out:
        _validate_moment_row(row, version)
        key = (str(row.get("metric")), str(row.get("group_id")))
        if key in seen_keys:
            _refuse(_MOMENTS_DUPLICATE_ROWS, key=key)
        seen_keys.add(key)
    evidence = _cube_evidence(evidence_payloads, version, cube_identity, out)
    assignment_counts = _validated_cube_assignment_counts(
        rows,
        boolean_count_aliases=boolean_count_aliases,
        design_summary=envelope_identity is not None,
    )
    if trigger_rows not in {0, len(rows)}:
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="trigger name missing from some rows",
            required="one consistent trigger declaration across the moments cube",
        )
    if len(set(trigger_values)) > 1:
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="conflicting trigger names",
            required="one consistent trigger declaration across the moments cube",
        )
    trigger_name = cast("str | None", trigger_values[0]) if trigger_values else None
    return (
        version,
        out,
        next(iter(plans), None),
        assignment_counts,
        _cube_compliance_state(rows),
        envelope_identity,
        trigger_name,
        cube_identity,
        evidence,
    )


def _cube_source_identity(
    identities: Sequence[dict[str, object]], n_rows: int
) -> dict[str, object] | None:
    """The one source identity every row carries, or ``None`` when no row carries one."""
    if identities and len(identities) != n_rows:
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="source identity missing from some rows",
            required="one consistent source identity across the moments cube",
        )
    if identities and any(
        not source_identities_equal(item, identities[0]) for item in identities[1:]
    ):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received="conflicting source identities",
            required="one consistent source identity across the moments cube",
        )
    return identities[0] if identities else None


def _cube_evidence(
    payloads: Sequence[object],
    version: int,
    identity: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
) -> tuple[CupedEvidence, ...]:
    """Decode envelopes bound to this cube's source, metrics and population keys."""
    if version == CUPED_EVIDENCE_FORMAT and not payloads:
        invalid_evidence(
            f"moments format {CUPED_EVIDENCE_FORMAT} requires CUPED evidence envelopes"
        )
    if not payloads:
        return ()
    if identity is None:
        invalid_evidence("evidence envelopes require the cube's source identity")
    digest = source_sha256(identity)
    metrics = {str(row.get("metric")) for row in rows}
    seen: set[tuple[str, str, str]] = set()
    decoded: list[CupedEvidence] = []
    for payload in payloads:
        item = parse_cuped_evidence(payload)
        if item.source_sha256 != digest:
            invalid_evidence("envelope source digest differs from the cube's source identity")
        if item.metric not in metrics:
            invalid_evidence(f"envelope metric {item.metric!r} has no ordinary outcome rows")
        key = (item.metric, item.population, item.method_key)
        if key in seen:
            duplicate_evidence(*key)
        seen.add(key)
        ordinary_rows = [row for row in rows if str(row.get("metric")) == item.metric]
        ordinary_counts = {str(row.get("group_id")): row.get("n") for row in ordinary_rows}
        if item.kind == "scalar_moments":
            evidence_arms = {str(row.get("group_id")) for row in item.rows}
            evidence_counts = {str(row.get("group_id")): row.get("n") for row in item.rows}
        elif item.kind == "categorical_cells":
            if item.cells is None:
                invalid_evidence("categorical envelope has no decoded cells")
            evidence_arms = {str(arm) for arm in item.cells.arms}
            evidence_counts = {
                str(arm): sum(item.cells.counts[index]) for index, arm in enumerate(item.cells.arms)
            }
        else:
            if item.fit_state is None:
                invalid_evidence("fitted envelope has no decoded fit state")
            evidence_arms = {
                str(arm) for arm in (*item.fit_state.eligible_arms, *item.fit_state.withheld_arms)
            }
            if len(evidence_arms) != len(
                item.fit_state.eligible_arms + item.fit_state.withheld_arms
            ):
                invalid_evidence("fitted evidence contains duplicate arms")
            # Withheld arms carry no fitted count; only fitted arms are compared.
            evidence_counts = {
                str(arm): count
                for arm, count in zip(
                    item.fit_state.eligible_arms, item.fit_state.counts, strict=True
                )
            }
        if evidence_arms != set(ordinary_counts):
            invalid_evidence(
                f"envelope arm roster differs from ordinary rows for metric {item.metric!r}"
            )
        if evidence_counts != {arm: ordinary_counts[arm] for arm in evidence_counts}:
            invalid_evidence(
                f"envelope arm counts differ from ordinary rows for metric {item.metric!r}"
            )
        _validate_moment_experiment_identity([*rows, *item.rows])
        arms: set[str] = set()
        for row in item.rows:
            _validate_moment_row(row, version)
            group = str(row.get("group_id"))
            if group in arms:
                _refuse(_MOMENTS_DUPLICATE_ROWS, key=(item.metric, group))
            arms.add(group)
        decoded.append(item)
    return tuple(decoded)


def _validate_cube_plan_payload(
    plans: set[str],
    unplanned: int,
    planned_rows: int,
    *,
    require_plan: bool,
    has_rows: bool,
) -> None:
    if not require_plan:
        return
    if len(plans) > 1:
        _refuse(
            _MOMENTS_PLAN_CONFLICT,
            field=DECISION_PLAN_FIELD,
            value=tuple(sorted(plans)),
            constraint="one cube must use one compiled plan",
        )
    if plans and unplanned:
        _refuse(
            _MOMENTS_PLAN_CONFLICT,
            field=DECISION_PLAN_FIELD,
            value={"planned_rows": planned_rows, "unplanned_rows": unplanned},
            constraint="a compiled plan covers the whole cube or none of it",
        )
    if has_rows and not plans:
        _refuse(
            _MOMENTS_PLAN_INVALID,
            field=DECISION_PLAN_FIELD,
            value=None,
            constraint="fixed-horizon moments cubes require the decision plan on every row",
        )


def _validate_winsorization_metadata(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[object],
) -> None:
    """Validate format-3 winsorization metadata against selected metric declarations."""
    by_metric = {
        getattr(metric, "name", None): getattr(metric, "winsorization", None) for metric in metrics
    }
    for row in rows:
        metric_name = row.get("metric")
        if metric_name not in by_metric:
            continue
        config = by_metric[metric_name]
        lower_fields = (
            row.get("winsor_lower_percentile"),
            row.get("winsor_lower_bound"),
            row.get("winsor_n_lower"),
        )
        upper_fields = (
            row.get("winsor_upper_percentile"),
            row.get("winsor_upper_bound"),
            row.get("winsor_n_upper"),
        )
        total_n = row.get("winsor_n")
        if config is None:
            if any(value not in (None, 0) for value in (*lower_fields, *upper_fields, total_n)):
                fields = (
                    ("winsor_lower_percentile", lower_fields[0]),
                    ("winsor_lower_bound", lower_fields[1]),
                    ("winsor_n_lower", lower_fields[2]),
                    ("winsor_upper_percentile", upper_fields[0]),
                    ("winsor_upper_bound", upper_fields[1]),
                    ("winsor_n_upper", upper_fields[2]),
                    ("winsor_n", total_n),
                )
                field, value = next((name, item) for name, item in fields if item not in (None, 0))
                _refuse(
                    _WINSOR_METADATA_MISSING,
                    metric=metric_name,
                    field=field,
                    value=value,
                )
            continue

        expected = (
            ("lower", lower_fields, config.lower_percentile, config.lower_value),
            ("upper", upper_fields, config.upper_percentile, config.upper_value),
        )
        for side, fields, expected_percentile, expected_bound in expected:
            actual_percentile, actual_bound, actual_n = fields
            if expected_percentile is not None:
                bound_matches = (
                    actual_percentile == expected_percentile
                    and isinstance(actual_bound, int | float)
                    and not isinstance(actual_bound, bool)
                    and math.isfinite(actual_bound)
                )
            else:
                bound_matches = actual_percentile is None and actual_bound == expected_bound
            if (
                not bound_matches
                or not isinstance(actual_n, int)
                or isinstance(actual_n, bool)
                or actual_n < 0
            ):
                _refuse(
                    _WINSOR_METADATA_CONFLICT,
                    metric=metric_name,
                    side=side,
                    expected_percentile=expected_percentile,
                    expected_bound=expected_bound,
                    actual_percentile=actual_percentile,
                    actual_bound=actual_bound,
                    actual_n=actual_n,
                )
        if not isinstance(total_n, int) or isinstance(total_n, bool) or total_n < 0:
            _refuse(
                _WINSOR_METADATA_SHAPE,
                metric=metric_name,
                field="winsor_n",
                value=total_n,
            )


def validate_readout_source(request: ReadoutRequest) -> None:
    """Validate static source/grain shape before a readout queries moments."""
    grain = request.grain
    capabilities = request.capabilities
    checkpoint_asof = (
        request.view == "asof" and getattr(request.plan.inference, "registration", None) is not None
    )
    if grain not in capabilities and not checkpoint_asof:
        from increment._evidence_refusals import GRAIN_UNAVAILABLE
        from increment._evidence_refusals import refuse as _refuse_evidence

        _refuse_evidence(
            GRAIN_UNAVAILABLE,
            constructor=request.source_name,
            requested=grain,
            offered=tuple(sorted(capabilities)),
            route="use a source that retains the requested grain",
        )
    by = request.by
    source_breakouts = tuple(getattr(request, "source_breakouts", ()))

    if by:
        missing = [name for name in by if name not in source_breakouts]
        if missing:
            from increment._evidence_refusals import BREAKOUT_UNAVAILABLE
            from increment._evidence_refusals import refuse as _refuse_evidence

            _refuse_evidence(
                BREAKOUT_UNAVAILABLE,
                constructor=request.source_name,
                requested=tuple(by),
                offered=source_breakouts,
                route="use a source with the requested breakout dimensions",
            )

    declared_metrics = tuple(getattr(request, "metrics", ()))
    if any(getattr(metric, "type", None) == "quantile" for metric in declared_metrics):
        if grain != "total":
            metric = next(
                metric for metric in declared_metrics if getattr(metric, "type", None) == "quantile"
            )
            _refuse(
                _READOUT_METRIC_QUANTILE_GRAIN,
                metric=getattr(metric, "name", "<unknown>"),
                grain=grain,
            )


def _configs_from_compiled_plan(
    metrics: tuple[Metric, ...],
    plan: CompiledDecisionPlan,
    fallback: tuple[ResolvedMetricConfig, ...],
) -> tuple[ResolvedMetricConfig, ...]:
    """Restore wire defaults without replacing resolved method declarations."""
    from increment._analysis_config import ResolvedMetricConfig

    fallback_by_name = {config.metric.name: config for config in fallback}
    out: list[ResolvedMetricConfig] = []
    for metric in metrics:
        config = fallback_by_name[metric.name]
        procedure = plan.procedures.get(metric.name)
        if (
            not config.decision_defaulted
            or config.sensitivity_methods
            or config.source_methods
            or config.methods_explicitly_empty
            or not hasattr(procedure, "decision_method")
        ):
            out.append(config)
            continue
        runtime_procedure = cast("Any", procedure)
        out.append(
            ResolvedMetricConfig(
                metric=metric,
                decision_method=runtime_procedure.decision_method,
                sensitivity_methods=runtime_procedure.sensitivity_methods,
                prior=(
                    runtime_procedure.prior
                    if runtime_procedure.prior is not None
                    else fallback_by_name[metric.name].prior
                ),
                prior_is_global=runtime_procedure.prior_is_global
                if runtime_procedure.prior is not None
                else fallback_by_name[metric.name].prior_is_global,
                covariates=fallback_by_name[metric.name].covariates,
                source_methods=(
                    runtime_procedure.decision_method,
                    *runtime_procedure.sensitivity_methods,
                ),
                covariate_available=bool(fallback_by_name[metric.name].covariates),
                methods_explicitly_empty=runtime_procedure.methods_explicitly_empty,
            )
        )
    return tuple(out)


def _validated_compliance_payload(
    payload: dict[str, object] | None,
    *,
    study_id: str,
    design: Randomized | Encouragement | Observational | None,
    assignment_counts: dict[str, int] | None,
    rows: Sequence[Mapping[str, object]],
) -> ComplianceSummary | None:
    if payload is None:
        return None
    from increment._source_types import validate_compliance_design_match
    from increment.semantics.design import Encouragement

    compliance = ComplianceSummary.from_wire(payload, study_id=study_id)
    if not isinstance(design, Encouragement):
        invalid_compliance_state(
            "an encouragement compliance payload requires its declared Encouragement design"
        )
    if compliance.as_of is not None:
        invalid_compliance_state("total cube cannot carry as-of compliance")
    validate_compliance_design_match(compliance, design)
    if assignment_counts is not None:
        real_counts = {
            group: count
            for group, count in assignment_counts.items()
            if group not in {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL}
        }
        compliance_counts = {arm.group_id: arm.n_units for arm in compliance.arms}
        if real_counts != compliance_counts:
            invalid_compliance_state("assignment counts differ from compliance unit counts")
    clustered = compliance.cluster is not None
    for row in rows:
        if (row.get("x_role") == X_SLOT_ROLES["uptake"]) != clustered:
            invalid_compliance_state("moment grain differs from compliance cluster identity")
        arm = compliance.arm(str(row.get("group_id")))
        if arm is None:
            invalid_compliance_state("moment arm is absent from compliance cohort")
        if row.get("experiment_id") != compliance.study_id:
            invalid_compliance_state("moment study differs from compliance cohort")
    return compliance


def _aggregate_configs(
    metrics: tuple[Metric, ...],
    plan: CompiledDecisionPlan,
    configs: Sequence[ResolvedMetricConfig] | None,
    *,
    bindings: Mapping[str, Any] | None = None,
    design: Randomized | Encouragement | Observational | None,
    route: str,
    representation: Literal["moments", "breakout"],
    portable_metrics: frozenset[str] = frozenset(),
) -> tuple[ResolvedMetricConfig, ...]:
    from increment._analysis_config import resolve_configs, validate_scalar_route

    base = (
        tuple(configs)
        if configs is not None
        else resolve_configs(metrics, bindings=bindings, specs=None, methods=None, prior=None)
    )
    resolved = _configs_from_compiled_plan(metrics, plan, base)
    # Metrics whose scalar families travel in method-scoped evidence are not
    # limited to one family by the ordinary rows.
    validate_scalar_route(
        tuple(config for config in resolved if config.metric.name not in portable_metrics),
        route=route,
        representation=representation,
        design=design,
    )
    return resolved


def _portable_population_and_trigger(
    evidence, source: CUPEDEvidenceExportSource | None, trusted_trigger_name
):
    populations = {item.population for item in evidence}
    triggers = {item.trigger_name for item in evidence}
    if len(populations) > 1 or len(triggers) > 1:
        invalid_evidence("CUPED envelopes disagree on population or trigger identity")
    source_population = source.population if source is not None else None
    source_trigger = source.trigger_name if source is not None else None
    if (
        populations
        and source_population is not None
        and next(iter(populations)) != source_population
    ):
        invalid_evidence("CUPED envelope population differs from its source")
    trigger_name = next(iter(triggers), source_trigger)
    if trigger_name is not None and source_trigger is not None and trigger_name != source_trigger:
        invalid_evidence("CUPED envelope trigger identity differs from its source")
    if trigger_name is not None and trusted_trigger_name is not None:
        if trigger_name != trusted_trigger_name:
            invalid_evidence("CUPED envelope trigger identity differs from its source")
    population = next(iter(populations), source_population or "assigned")
    if population == "triggered" and trigger_name is None:
        invalid_evidence("triggered evidence requires a preserved trigger identity")
    return population, trigger_name


def _moments_trigger_name(
    envelope_identity, study_id, embedded_trigger_name, trusted_trigger_name, evidence_trigger_name
):
    if envelope_identity is not None and envelope_identity != study_id:
        invalid_compliance_state("design summary experiment identity differs from requested study")
    if (
        embedded_trigger_name is not None
        and trusted_trigger_name is not None
        and embedded_trigger_name != trusted_trigger_name
    ):
        _refuse(
            _MOMENTS_FORMAT_INVALID,
            received=embedded_trigger_name,
            required="the trusted source trigger name",
        )
    if (
        evidence_trigger_name is not None
        and embedded_trigger_name is not None
        and evidence_trigger_name != embedded_trigger_name
    ):
        invalid_evidence("CUPED envelope trigger identity differs from moments metadata")
    return embedded_trigger_name or trusted_trigger_name or evidence_trigger_name


def _moments_source_identity(source_record, requested_record, version, study_id):
    if source_record is not None:
        return source_record
    if requested_record is not None:
        return requested_record
    return _legacy_moments_source_identity(version, study_id)


class MomentsSource(SequentialSourceMixin):
    """`MomentSource` over pre-computed `group_summary` rows: a moments cube
    (e.g. exported to parquet) rehydrated with no warehouse or frame, just
    centered moments.

    Centered moment rows and complete fixed-horizon design summaries carry a
    mandatory format stamp so an old additive-sum cube cannot be misread.
    """

    @staticmethod
    def design_summary_design(
        rows: Sequence[Mapping[str, object]], *, control_group: str
    ) -> Encouragement | None:
        """Decode a complete design-summary envelope into its declared design."""
        if not rows or rows[0].get("record_kind") != "design_summary":
            return None
        row = rows[0]
        study_id = row.get("experiment_id")
        if not isinstance(study_id, str) or not study_id:
            invalid_compliance_state("design_summary requires its exported experiment identity")
        compliance = ComplianceSummary.from_wire(
            _parse_compliance_payload(row.get(COMPLIANCE_SUMMARY_FIELD)),
            study_id=study_id,
        )
        from increment.semantics.design import Encouragement, UptakeSpec

        return Encouragement(
            control_group=control_group,
            uptake=UptakeSpec(fact=compliance.cohort, window_days=compliance.window_days),
            one_sided=compliance.one_sided,
        )

    capabilities: frozenset[Grain] = frozenset({"total"})
    operations: frozenset[SourceOperation] = frozenset({"unit_counts"})
    shape: Literal["unit_summary", "unit_panel"] | None = None
    breakouts: tuple[str, ...] = ()

    @property
    def evidence(self) -> EvidenceCapabilities:
        from increment._source_types import declared_evidence

        backing = declared_evidence(self._source) if self._source is not None else None
        retained_fit = any(
            item.kind in {"fitted_multicovariate", "categorical_cells"}
            for item in self._evidence.values()
        )
        count_grains: set[Literal["unit", "cluster", "assignment", "triggered"]] = set()
        if self._assignment_counts is not None:
            count_grains.add("unit")
        if self._compliance is not None and self._compliance.cluster is not None:
            count_grains.add("cluster")
        return EvidenceCapabilities(
            kind="moments",
            grains=self.capabilities,
            breakouts=self.breakouts,
            shape=self.shape,
            operations=self.operations,
            unit_rows=backing.unit_rows if backing is not None else False,
            raw_outcomes=backing.raw_outcomes if backing is not None else False,
            method_fit="retained" if retained_fit else "none",
            compliance_axis="total" if self._compliance is not None else "none",
            count_grains=frozenset(count_grains),
            populations=(
                frozenset({"triggered"})
                if self.population == "triggered"
                else frozenset({"assigned"})
            ),
            sequential_capture=(
                "replay" if getattr(self, "_sequential_snapshot", None) is not None else "none"
            ),
            scalar=ScalarEvidenceDeclaration(
                bindings=None,
                moments_only=self.scalar_moments_only,
                recipe_fixed=self.scalar_recipe_fixed,
                recipe_configs=self.scalar_recipe_configs,
                anonymous_metrics=self.scalar_anonymous_metrics,
                method_evidence=True,
                breakout_evidence=False,
                covariate_grains=None,
            ),
            available_covariates=None,
        )

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        metrics: Sequence[object],
        study_id: str,
        bindings_by_name: Mapping[str, ExperimentMetric] | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | CompiledDecisionPlan | None = None,
        path: Literal["warehouse", "frame"] = "frame",
        _trusted_trigger_name: str | None = None,
        configs: Sequence[ResolvedMetricConfig] | None = None,
        _source_snapshot_evidence: Any | None = None,
        _source: Any | None = None,
        _source_identity_record: Mapping[str, object] | None = None,
    ) -> None:
        self._source_snapshot_evidence = _source_snapshot_evidence
        self._source = _source
        (
            _version,
            stripped,
            plan_payload,
            assignment_counts,
            compliance_payload,
            envelope_identity,
            embedded_trigger_name,
            source_identity_record,
            evidence,
        ) = _check_moments_format(
            rows,
            require_plan=plan is None,
            require_source_identity=_source_identity_record is None,
        )
        self._population, self._trigger_name = _portable_population_and_trigger(
            evidence,
            None if _source is None else cast(CUPEDEvidenceExportSource, _source),
            _trusted_trigger_name,
        )
        evidence_trigger_name = self._trigger_name
        self._bind_evidence(evidence)
        self._source_identity_record = _moments_source_identity(
            source_identity_record, _source_identity_record, _version, study_id
        )
        trigger_name = _moments_trigger_name(
            envelope_identity,
            study_id,
            embedded_trigger_name,
            _trusted_trigger_name,
            evidence_trigger_name,
        )
        self._rows = stripped
        self._assignment_counts = assignment_counts
        self._compliance = _validated_compliance_payload(
            compliance_payload,
            study_id=study_id,
            design=design,
            assignment_counts=assignment_counts,
            rows=stripped,
        )
        missing = sorted(
            {field for row in stripped for field in WINSORIZATION_MOMENT_FIELDS if field not in row}
        )
        if missing:
            _refuse(_MOMENTS_WINSORIZATION_FIELDS_MISSING, missing=missing)
        _validate_winsorization_metadata(stripped, metrics)
        from increment.decision import CompiledDecisionPlan
        from increment.plan import (
            compile_decision_plan,
            refuse_observational_relative_margin,
            validate_compiled_encouragement_plan,
        )
        from increment.sequential_source import validate_sequential_plan

        metric_catalog = tuple(cast("Metric", metric) for metric in metrics)
        stored_plan: CompiledDecisionPlan | None = None
        if plan_payload is not None:
            from increment.decision_wire import compiled_plan_from_json

            stored_plan = compiled_plan_from_json(
                plan_payload,
                metric_types={metric.name: metric.type for metric in metric_catalog},
            )
        if plan is None and stored_plan is not None:
            compiled = stored_plan
            missing = [
                metric.name for metric in metric_catalog if metric.name not in compiled.procedures
            ]
            if missing:
                _refuse(
                    _MOMENTS_PLAN_INVALID,
                    field=DECISION_PLAN_FIELD,
                    value=tuple(sorted(missing)),
                    constraint="the embedded decision plan must cover every supplied metric",
                )
        elif isinstance(plan, CompiledDecisionPlan):
            compiled = plan
        else:
            compiled = compile_decision_plan(
                cast("AnalysisPlan | None", plan),
                metric_catalog,
                path=path,
                design=design,
                configs=configs,
            )
        if envelope_identity is not None:
            from increment.estimation.decision_types import FixedInference

            if metric_catalog or any(
                candidate is None
                or candidate.procedures
                or not isinstance(candidate.inference, FixedInference)
                for candidate in (stored_plan, compiled)
            ):
                invalid_compliance_state(
                    "design_summary requires an empty metric catalog and fixed-horizon plans; "
                    "export outcome rows or a sequential checkpoint for other analyses"
                )
        refuse_observational_relative_margin(design, compiled)

        self._legacy_prior_exclusions = frozenset()
        if plan is None and stored_plan is not None and plan_payload is not None:
            from increment.decision_wire import _legacy_prior_exclusion_metrics

            self._legacy_prior_exclusions = _legacy_prior_exclusion_metrics(plan_payload)

        base_configs = _aggregate_configs(
            metric_catalog,
            compiled,
            configs,
            bindings=bindings_by_name,
            route="MomentsSource",
            representation="moments",
            design=design,
            portable_metrics=self._scalar_evidence_metrics,
        )
        self._require_method_evidence(base_configs, design)
        from increment._analysis_config import (
            effective_methods,
            resolve_configs,
            validate_scalar_route,
        )

        self._scalar_recipe_configs: tuple[ResolvedMetricConfig, ...] = ()
        if stored_plan is not None:
            recorded_configs = _configs_from_compiled_plan(
                metric_catalog,
                stored_plan,
                resolve_configs(
                    metric_catalog, bindings=None, specs=None, methods=None, prior=None
                ),
            )
            self._scalar_recipe_configs = tuple(
                config
                for config in recorded_configs
                if any(
                    method.variance_reduction == "cuped"
                    for method in effective_methods(config, design=design)
                )
            )
        named_metrics = {
            str(row["metric"]) for row in self._rows if row.get("covariate_name") is not None
        } | self._scalar_evidence_metrics
        self._scalar_anonymous_metrics = frozenset(
            metric.name for metric in metric_catalog if metric.name not in named_metrics
        )
        validate_compiled_encouragement_plan(compiled, design)
        validate_sequential_plan(compiled, metric_catalog, design)
        self._context = SourceContext(
            study_id=study_id,
            design=design,
            plan=compiled,
            metrics=metric_catalog,
            configs=base_configs,
            cluster=self._compliance.cluster if self._compliance is not None else None,
            trigger_name=trigger_name,
        )
        if self.scalar_recipe_fixed:
            recorded = {config.metric.name for config in self.scalar_recipe_configs}
            validate_scalar_route(
                tuple(config for config in base_configs if config.metric.name in recorded),
                route="MomentsSource",
                representation="prepared",
                design=design,
                retained_configs=self.scalar_recipe_configs,
                anonymous_metrics=self.scalar_anonymous_metrics,
            )
        self._restore_checkpoint(rows, stored_plan)

    def _bind_evidence(self, evidence: tuple[CupedEvidence, ...]) -> None:
        """Index method-scoped evidence; every envelope belongs to this cube's population."""
        if any(item.population != self.population for item in evidence):
            invalid_evidence(f"evidence population differs from this {self.population} cube")
        self._evidence = {(item.metric, item.method_key): item for item in evidence}
        self._scalar_evidence_metrics = frozenset(
            item.metric for item in evidence if item.kind == "scalar_moments"
        )

    def _require_method_evidence(
        self,
        configs: Sequence[ResolvedMetricConfig],
        design: Randomized | Encouragement | Observational | None,
    ) -> None:
        """Refuse an effective request whose evidence this cube does not carry."""
        from increment._analysis_config import (
            effective_methods,
            refuse_cuped_linear_moments,
            refuse_missing_scalar_evidence,
        )

        for config in configs:
            name = config.metric.name
            for method in effective_methods(config, design=design):
                evidence = self._evidence.get((name, method_key(method)))
                if method.variance_reduction == "cuped_multicovariate":
                    if self._source is None and (
                        evidence is None
                        or evidence.kind not in ("categorical_cells", "fitted_multicovariate")
                    ):
                        refuse_cuped_linear_moments(name)
                elif (
                    method.variance_reduction == "cuped"
                    and name in self._scalar_evidence_metrics
                    and evidence is None
                ):
                    refuse_missing_scalar_evidence("MomentsSource", method.name)

    def categorical_evidence(self, metric, method) -> CupedEvidence | None:
        """The joint categorical evidence retained for this request, if any."""
        evidence = self._evidence.get((metric.name, method_key(method)))
        return evidence if evidence is not None and evidence.kind == "categorical_cells" else None

    def fit_cuped_multicovariate(
        self,
        metric,
        method,
        *,
        control: str,
        arm_order: tuple[str, ...],
        population: Literal["assigned", "triggered"] = "assigned",
        return_components: bool = False,
        **_scope,
    ):
        evidence = self._evidence.get((metric.name, method_key(method)))
        if evidence is None or evidence.kind != "fitted_multicovariate":
            from increment._analysis_config import refuse_cuped_linear_moments

            refuse_cuped_linear_moments(metric.name)
        if evidence.population != population:
            invalid_evidence("fitted evidence population differs from the requested fit")
        state, summary = evidence.fit_state, evidence.inference_summary
        if state is None or summary is None:
            invalid_evidence("fitted evidence FitState or inference summary is missing")
        if any(value not in (None, False) for value in _scope.values()):
            invalid_evidence("fitted evidence cannot change its breakout fit scope")
        if control != state.control or tuple(arm_order) != tuple(state.identity[6]):
            invalid_evidence("fitted evidence arm or control scope differs from the requested fit")
        from increment._cuped_evidence import (
            finalize_fitted_multicovariate,
            warehouse_cuped_fit_evidence,
        )

        withheld = []
        for arm, encoded_context in state.refusal_contexts:
            context = dict(encoded_context)
            code = context.pop("code")
            withheld.append(
                (
                    arm,
                    _CodedError(
                        f"fitted CUPED evidence withheld arm {arm!r}", code=code, context=context
                    ),
                )
            )
        fit = finalize_fitted_multicovariate(state, summary)
        result = (state, summary, fit) if return_components else fit
        return state.identity, warehouse_cuped_fit_evidence(
            result, tuple(arm_order), tuple(withheld)
        )

    def _restore_checkpoint(
        self,
        rows: Sequence[Mapping[str, object]],
        stored_plan: CompiledDecisionPlan | None,
    ) -> None:
        compiled = self.context.plan
        base_configs = self.context.configs
        design = self.context.design

        if getattr(compiled.inference, "registration", None) is not None:
            from increment._analysis_config import validate_scalar_route

            validate_scalar_route(
                base_configs,
                route="sequential.checkpoint_adoption",
                representation="registered",
                registered_plan=stored_plan if stored_plan is not None else compiled,
                design=design,
            )

        from increment.sequential_state import sequential_refuse, snapshot_from_json

        if getattr(compiled.inference, "registration", None) is not None and any(
            row.get("moments_format") not in _SUPPORTED_SEQUENTIAL_MOMENTS_FORMATS for row in rows
        ):
            sequential_refuse(
                "continuation.legacy",
                "legacy moments cannot resume registered sequential inference",
            )
        snapshot_payloads = [row.get(SEQUENTIAL_SNAPSHOT_FIELD) for row in rows]
        if any(payload is not None for payload in snapshot_payloads):
            if len(snapshot_payloads) != 1 or not isinstance(snapshot_payloads[0], str):
                sequential_refuse("source.invalid", "checkpoint requires one exact envelope")
            snapshot = snapshot_from_json(snapshot_payloads[0])
            self.adopt_sequential_snapshot(snapshot)
        elif getattr(compiled.inference, "registration", None) is not None:
            sequential_refuse(
                "continuation.legacy", "sequential moments require the current exact checkpoint"
            )

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def population(self) -> Literal["assigned", "triggered"]:
        return self._population

    @property
    def trigger_name(self) -> str | None:
        return self.context.trigger_name

    @contextmanager
    def readout_snapshot(
        self, *, metrics, population, uptake_facts=(), include_breakouts=False, configs=None
    ):
        snapshot = getattr(self._source, "readout_snapshot", None)
        if snapshot is None:
            yield self
            return
        with snapshot(
            metrics=metrics,
            population=self.population,
            uptake_facts=uptake_facts,
            include_breakouts=include_breakouts,
            configs=configs,
        ) as pinned:
            yield pinned

    def moments(
        self,
        metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, object]]:
        from increment._evidence_refusals import require_evidence

        require_evidence(self, grain=grain, by=by)
        return [dict(r) for r in self._rows if r["metric"] == metric.name]

    @property
    def scalar_moments_only(self) -> bool:
        """Ordinary rows alone carry the scalar input unless method-scoped evidence is present."""
        return not self._scalar_evidence_metrics

    @property
    def scalar_recipe_fixed(self) -> bool:
        return (
            bool(self._scalar_recipe_configs)
            and getattr(self.context.plan.inference, "registration", None) is None
        )

    @property
    def scalar_recipe_configs(self) -> tuple[ResolvedMetricConfig, ...]:
        """Scalar preparations recorded with the cube, independent of requested defaults."""
        return self._scalar_recipe_configs

    @property
    def scalar_anonymous_metrics(self) -> frozenset[str]:
        """Metrics without recorded input-name provenance, even when preparation is known."""
        return self._scalar_anonymous_metrics

    def moments_for_method(
        self,
        metric,
        method,
        *,
        by: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        """Select a scalar family by its method evidence, or a named legacy x family."""
        if method.variance_reduction == "cuped" and metric.name in self._scalar_evidence_metrics:
            if by:
                from increment._evidence_refusals import require_evidence

                require_evidence(self, by=tuple(by))
            evidence = self._evidence.get((metric.name, method_key(method)))
            if evidence is None:
                from increment._analysis_config import refuse_missing_scalar_evidence

                refuse_missing_scalar_evidence("MomentsSource", method.name)
            return evidence.scalar_rows()
        rows = self.moments(metric, by=by)
        covariates = method.covariates or ()
        if not covariates:
            return rows
        covariate_name = covariates[0].name
        named = [row for row in rows if row.get("covariate_name") is not None]
        if not named:
            return rows
        selected = [row for row in named if row.get("covariate_name") == covariate_name]
        if not selected:
            from increment._analysis_config import refuse_unidentified_scalar_moments

            refuse_unidentified_scalar_moments(metric.name)
        return selected

    def unit_frame(
        self,
        metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> Any:
        if self._source is not None and hasattr(self._source, "unit_frame"):
            return self._source.unit_frame(
                metric, covariates=covariates, outcome_stage=outcome_stage
            )
        from increment._evidence_refusals import refuse_unit_rows, require_evidence

        if outcome_stage == "raw":
            require_evidence(self, raw_outcomes=True)
        if covariates:
            require_evidence(self, covariates=covariates)
        require_evidence(self, unit_rows=True)
        refuse_unit_rows(
            self,
            offered=False,
            route="use a source that retains unit-level rows",
        )

    def unit_counts(self) -> dict[str, int]:
        """Return source-level enrolled counts carried by supported cubes."""
        if self._assignment_counts is None:
            _refuse(_MOMENTS_COUNTS)
        return dict(self._assignment_counts)

    def cluster_counts(self) -> dict[str, int]:
        if self._compliance is not None and self._compliance.cluster is not None:
            return {
                arm.group_id: arm.n_clusters
                for arm in self._compliance.arms
                if arm.n_clusters is not None
            }
        from increment._evidence_refusals import refuse_cluster, require_evidence

        require_evidence(self, cluster=True)
        refuse_cluster(
            self,
            offered=(),
            route="use a source that retains cluster identities or cluster counts",
        )

    def compliance_dates(self) -> Sequence[object]:
        from increment._evidence_refusals import refuse_grain

        refuse_grain(
            self, "asof", self.capabilities, route="use a source with as-of compliance dates"
        )

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        """Design-level uptake state carried by this cube's own
        ``compliance_summary`` wire payload (stamped by ``export_moments``/
        ``moments_source()`` alongside ``assignment_counts``/
        ``decision_plan``) -- never any one metric's moments."""
        if completed_windows_only:
            from increment._source_types import validate_compliance_completion

            validate_compliance_completion(design, as_of=as_of)
        if as_of is not None:
            from increment._evidence_refusals import require_evidence

            require_evidence(self, grain="asof")
        if self._compliance is None:
            from increment._source_types import raise_legacy_compliance_state

            raise_legacy_compliance_state(
                study_id=self._context.study_id,
                reason="this cube predates the compliance_summary payload",
            )
        from increment._source_types import validate_compliance_design_match

        validate_compliance_design_match(self._compliance, design)
        return self._compliance

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        from increment._evidence_refusals import refuse_operation

        refuse_operation(
            self,
            "sql",
            tuple(sorted(self.operations)),
            route="moments sources carry precomputed rows, not a query",
        )

    def close(self) -> None:
        pass


def _design_summary_row(
    study_id: str,
    plan: str,
    counts: str,
    compliance: str | None,
    trigger_name: str | None,
    source_identity: str,
) -> dict[str, object]:
    """Encode a complete fixed-horizon, outcome-free Encouragement source."""
    if compliance is None:
        invalid_compliance_state(
            "a metric-free export requires the declared Encouragement uptake state"
        )
    row = {
        "record_kind": "design_summary",
        "experiment_id": study_id,
        "moments_format": MOMENTS_FORMAT,
        DECISION_PLAN_FIELD: plan,
        ASSIGNMENT_COUNTS_FIELD: counts,
        COMPLIANCE_SUMMARY_FIELD: compliance,
        SOURCE_IDENTITY_FIELD: source_identity,
    }
    if trigger_name is not None:
        row[TRIGGER_NAME_FIELD] = trigger_name
    return row


def export_source_moments(
    source: MomentSource,
    path: str | Path,
    *,
    observational_refusal: Callable[[Any], object],
) -> None:
    """Export total moments with immutable source identity and compliance state.

    A registered sequential plan exports its finalized checkpoint, which holds unit-record
    proofs and the declaration, never moments rows, so an unmodeled catalog quantile is not
    refused there. Every moments export refuses a catalog quantile before reading evidence
    (see :func:`refuse_quantile_moments_export` for ``observational_refusal``).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment._source_identity import source_identity
    from increment.decision_wire import compiled_plan_to_json
    from increment.semantics.design import Encouragement

    context = source.context
    identity_payload = json.dumps(source_identity(source), sort_keys=True, separators=(",", ":"))
    plan_payload = compiled_plan_to_json(
        context.plan,
        legacy_prior_exclusions=getattr(source, "_legacy_prior_exclusions", frozenset()),
    )
    if getattr(context.plan.inference, "registration", None) is not None:
        from increment.sequential_source import source_snapshot

        snapshot = source_snapshot(source)
        row = {
            "record_kind": "sequential_checkpoint",
            "experiment_id": context.study_id,
            "moments_format": SEQUENTIAL_MOMENTS_FORMAT,
            DECISION_PLAN_FIELD: plan_payload,
            SEQUENTIAL_SNAPSHOT_FIELD: snapshot.model_dump_json(),
            SOURCE_IDENTITY_FIELD: identity_payload,
        }
        if snapshot.assignment_counts is not None:
            accounting_labels = {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL}
            row[ASSIGNMENT_COUNTS_FIELD] = json.dumps(
                {
                    group: count
                    for group, count in snapshot.assignment_counts.items()
                    if group not in accounting_labels
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        rows = [row]
        if context.trigger_name is not None:
            rows[0][TRIGGER_NAME_FIELD] = context.trigger_name
        table = pa.Table.from_pylist(rows).replace_schema_metadata(
            {b"increment.moments_format": str(SEQUENTIAL_MOMENTS_FORMAT).encode()}
        )
        pq.write_table(table, str(path))
        return
    if context.cluster is not None and not isinstance(context.design, Encouragement):
        refuse_cluster_grain_transport(
            operation="export_moments",
            source="frame",
            cluster=context.cluster,
            design=context.design,
            route_forward=(
                "analyse the clustered source directly; direct clustered inference "
                "preserves cluster-grain degrees of freedom and counts"
            ),
        )
    refuse_quantile_moments_export(
        context.metrics, design=context.design, observational_refusal=observational_refusal
    )
    compliance = (
        source.compliance_summary(context.design)
        if isinstance(context.design, Encouragement)
        else None
    )
    plan = plan_payload
    counts = json.dumps(source.unit_counts(), sort_keys=True, separators=(",", ":"))
    payload = (
        json.dumps(compliance.to_wire(), sort_keys=True, separators=(",", ":"))
        if compliance is not None
        else None
    )
    requests = portable_requests(context.configs, design=context.design, categorical=True)
    portable = (
        None
        if requests is None
        else portable_cuped_rows(
            cast(CUPEDEvidenceExportSource, source),
            requests,
            categorical_cells=_categorical_cells_for(source),
        )
    )
    version = MOMENTS_FORMAT if portable is None else CUPED_EVIDENCE_FORMAT
    raw_rows = (
        portable
        if portable is not None
        else [
            dict(cast("Mapping[str, object]", raw))
            for metric in context.metrics
            for raw in source.moments(metric)
        ]
    )
    if not raw_rows and not context.metrics:
        rows = [
            _design_summary_row(
                context.study_id, plan, counts, payload, context.trigger_name, identity_payload
            )
        ]
    else:
        rows = stamp_fixed_rows(
            raw_rows,
            version=version,
            plan=plan,
            identity=identity_payload,
            counts=counts,
            compliance=payload,
            trigger_name=context.trigger_name,
        )
    write_moments_table(path, rows, version)


def stamp_fixed_rows(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    version: int,
    identity: str,
    counts: str | None,
    plan: str | None = None,
    compliance: str | None = None,
    trigger_name: str | None = None,
) -> list[dict[str, object]]:
    """Fixed-horizon transport fields on every outcome and envelope row."""
    stamps: dict[str, object] = {"moments_format": version}
    if plan is not None:
        stamps[DECISION_PLAN_FIELD] = plan
    if counts is not None:
        stamps[ASSIGNMENT_COUNTS_FIELD] = counts
    stamps[SOURCE_IDENTITY_FIELD] = identity
    if compliance is not None:
        stamps[COMPLIANCE_SUMMARY_FIELD] = compliance
    if trigger_name is not None:
        stamps[TRIGGER_NAME_FIELD] = trigger_name
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        if row.get("record_kind") != CUPED_EVIDENCE_RECORD:
            _validate_moment_counts(row)
        row.update(stamps)
        rows.append(row)
    return rows


def write_moments_table(
    path: str | Path, rows: Sequence[Mapping[str, object]], version: int
) -> None:
    """Write rows over the union of their columns; envelope and outcome rows differ."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = list(dict.fromkeys(key for row in rows for key in row))
    table = pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows])
    table = table.replace_schema_metadata({b"increment.moments_format": str(version).encode()})
    if "successes" in table.column_names:
        index = table.schema.get_field_index("successes")
        table = table.set_column(index, "successes", table["successes"].cast(pa.int64()))
    pq.write_table(table, str(path))


def cuped_evidence_items(
    rows: Sequence[Mapping[str, object]], *, require_plan: bool
) -> dict[str, dict[str, CupedEvidence]]:
    """Validated evidence by metric and method, retaining decoded cells for admission checks."""
    evidence = _check_moments_format(rows, require_plan=require_plan)[-1]
    items: dict[str, dict[str, CupedEvidence]] = {}
    for item in evidence:
        items.setdefault(item.metric, {})[item.method_key] = item
    return items


def _categorical_cells_for(source: Any) -> Callable[[Any, Any], Any]:
    """Retained portable evidence for a reloaded cube; otherwise a local reduction."""

    def cells(metric: Any, method: Any) -> Any:
        retained = getattr(source, "categorical_evidence", None)
        if retained is None:
            return local_categorical_cells(source, metric, method)
        evidence = retained(metric, method)
        if evidence is None or evidence.cells is None:
            from increment._analysis_config import refuse_cuped_linear_moments

            refuse_cuped_linear_moments(metric.name)
        return evidence.cells

    return cells


class BreakoutMomentsSource(SequentialSourceMixin):
    """`MomentSource` over one declared `Breakout`'s totals-grain moments:
    the breakout-scoped counterpart to `MomentsSource`.

    Scoped to exactly one breakout so a `dimension` name shared by two
    `Breakout`s resolving to different fact sources never collides. Rows
    reuse `breakout_summaries()`'s CUPED-preserving construction, but
    aren't round-tripped through any wire format, so this source carries
    no `moments_format` stamp and no winsorization validation.
    """

    capabilities: frozenset[Grain] = frozenset({"total"})
    operations: frozenset[SourceOperation] = frozenset({"unit_counts"})
    shape: Literal["unit_summary", "unit_panel"] | None = None

    @property
    def evidence(self) -> EvidenceCapabilities:
        from increment._source_types import declared_evidence

        backing = declared_evidence(self._source) if self._source is not None else None
        return EvidenceCapabilities(
            kind="breakout",
            grains=self.capabilities,
            breakouts=self.breakouts,
            shape=self.shape,
            operations=self.operations,
            unit_rows=backing.unit_rows if backing is not None else False,
            raw_outcomes=backing.raw_outcomes if backing is not None else False,
            method_fit="none",
            compliance_axis="none",
            count_grains=frozenset(),
            populations=(
                frozenset({"triggered"})
                if self.population == "triggered"
                else frozenset({"assigned"})
            ),
            sequential_capture="none",
            scalar=ScalarEvidenceDeclaration(
                bindings=None,
                moments_only=self.scalar_moments_only,
                recipe_fixed=self.scalar_recipe_fixed,
                recipe_configs=self.scalar_recipe_configs,
                anonymous_metrics=self.scalar_anonymous_metrics,
                method_evidence=False,
                breakout_evidence=True,
                covariate_grains=None,
            ),
            available_covariates=None,
        )

    @property
    def scalar_moments_only(self) -> bool:
        return True

    @property
    def scalar_recipe_fixed(self) -> bool:
        return True

    @property
    def scalar_recipe_configs(self) -> tuple[ResolvedMetricConfig, ...]:
        return self.context.configs

    @property
    def scalar_anonymous_metrics(self) -> frozenset[str]:
        return frozenset()

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        dimension: str,
        metrics: Sequence[object],
        study_id: str,
        source_name: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | CompiledDecisionPlan | None = None,
        configs: Sequence[ResolvedMetricConfig] | None = None,
        _source: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        scalar_refusals: Mapping[str, Sequence[tuple[str, str, _RefusalSpec, Mapping[str, object]]]]
        | None = None,
    ) -> None:
        self._source = _source
        self._dimension = dimension
        self._source_name = source_name
        self._population = population
        metric_catalog = tuple(cast("Metric", metric) for metric in metrics)
        from increment.decision import CompiledDecisionPlan
        from increment.plan import compile_decision_plan, validate_compiled_encouragement_plan

        compiled = (
            plan
            if isinstance(plan, CompiledDecisionPlan)
            else compile_decision_plan(
                cast("AnalysisPlan | None", plan),
                metric_catalog,
                path="warehouse",
                design=design,
                configs=configs,
            )
        )
        base_configs = _aggregate_configs(
            metric_catalog,
            compiled,
            configs,
            route="BreakoutMomentsSource",
            representation="breakout",
            design=design,
        )
        self._rows = [dict(row) for row in rows]
        self._scalar_refusals = _FrozenMapping(
            {
                metric: tuple(
                    (dimension, value, spec, _FrozenMapping(context))
                    for dimension, value, spec, context in failures
                )
                for metric, failures in (scalar_refusals or {}).items()
            }
        )
        validate_compiled_encouragement_plan(compiled, design)
        self._context = SourceContext(
            study_id=study_id,
            design=design,
            plan=compiled,
            metrics=metric_catalog,
            configs=base_configs,
            cluster=None,
        )

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def breakouts(self) -> tuple[str, ...]:
        return (self._dimension,)

    @property
    def source_name(self) -> str | None:
        """Resolved fact-source identity for this scoped breakout view."""
        return self._source_name

    def moments(
        self,
        metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, object]]:
        if grain != "total":
            from increment._evidence_refusals import require_evidence

            require_evidence(self, grain=grain)
        if tuple(by) != (self._dimension,):
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self,
                tuple(by),
                self.breakouts,
                route="request exactly the breakout dimension this source retains",
            )
        return [dict(r) for r in self._rows if r["metric"] == metric.name]

    def breakout_moments_for_method(self, metric, method, *, by):
        """Select an input already represented by this prepared breakout cube."""
        from increment._analysis_config import overlay_configs, validate_scalar_route
        from increment._source_types import ScalarBreakoutRows

        validate_scalar_route(
            overlay_configs((metric,), self.context.configs, methods=[method], prior=None),
            route="BreakoutMomentsSource.breakout",
            representation="prepared",
            design=self.context.design,
            retained_configs=self.context.configs,
        )
        return ScalarBreakoutRows(
            tuple(self.moments(metric, by=by)),
            self._scalar_refusals.get(metric.name, ())
            if method.variance_reduction == "cuped"
            else (),
        )

    def unit_frame(
        self,
        metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> Any:
        if self._source is not None:
            from increment.semantics.models import Breakout

            return self._source.breakout_unit_frame(
                metric,
                breakout=Breakout(property=self._dimension, source=self._source_name),
                covariates=covariates,
                population=self._population,
                outcome_stage=outcome_stage,
            )
        from increment._evidence_refusals import refuse_unit_rows, require_evidence

        if outcome_stage == "raw":
            require_evidence(self, raw_outcomes=True)
        if covariates:
            require_evidence(self, covariates=covariates)
        require_evidence(self, unit_rows=True)
        refuse_unit_rows(
            self,
            offered=False,
            route="use a source that retains unit-level rows",
        )

    def unit_counts(self) -> dict[str, int]:
        _refuse(_BREAKOUT_COUNTS)

    def cluster_counts(self) -> dict[str, int]:
        from increment._evidence_refusals import refuse_cluster, require_evidence

        require_evidence(self, cluster=True)
        refuse_cluster(
            self,
            offered=(),
            route="use a source that retains cluster identities or cluster counts",
        )

    def compliance_dates(self) -> Sequence[object]:
        _refuse(_BREAKOUT_COMPLIANCE)

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        _refuse(_BREAKOUT_COMPLIANCE)

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        from increment._evidence_refusals import refuse_operation

        refuse_operation(
            self,
            "sql",
            tuple(sorted(self.operations)),
            route="breakout moments sources carry precomputed rows, not a query",
        )

    def close(self) -> None:
        pass


__all__ = [
    "ASSIGNMENT_COUNTS_FIELD",
    "BreakoutMomentsSource",
    "CENSOR_WARN_FRACTION",
    "COMPLIANCE_SUMMARY_FIELD",
    "ComplianceArm",
    "ComplianceSummary",
    "DECISION_PLAN_FIELD",
    "Grain",
    "MIXED_ASSIGNMENT_LABEL",
    "MOMENTS_FORMAT",
    "MomentSource",
    "MomentsSource",
    "SOURCE_QUANTILE_NO_MOMENTS",
    "SourceContext",
    "SourceOperation",
    "UNASSIGNED_LABEL",
    "WINSORIZATION_MOMENT_FIELDS",
    "require_operation",
    "refuse_cluster_grain_transport",
    "refuse_quantile_moments_export",
    "validate_readout_source",
]
