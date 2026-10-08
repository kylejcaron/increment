"""Seam between data sources and readouts.

Must not import ibis or narwhals, so one readout implementation works
against a warehouse table, an in-memory frame, or a moments cube through
the shared `MomentSource` protocol.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL
from increment._moment_plan import X_SLOT_ROLES
from increment._source_types import (
    ComplianceArm,
    ComplianceSummary,
    Grain,
    MomentSource,
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

SOURCE_OPERATION_UNSUPPORTED = _RefusalSpec(
    "source.operation.unsupported",
    _CapabilityError,
    template="source {source!r} does not support operation {operation!r}; the requested operation is unavailable for this source.",
)


def require_operation[T](
    source: MomentSource,
    operation: SourceOperation,
    protocol: type[T],
) -> T:
    """Return *source* after checking a declared operation is callable."""
    if operation not in getattr(source, "operations", frozenset()) or not isinstance(
        source, protocol
    ):
        _refuse(
            SOURCE_OPERATION_UNSUPPORTED,
            operation=operation,
            source=type(source).__name__,
        )
    return cast("T", source)


if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import AnalysisPlan, ExperimentMetric, Metric

_MOMENTS_GRAIN = _RefusalSpec(
    "source.moments.grain",
    _CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_MOMENTS_BREAKOUT = _RefusalSpec(
    "source.moments.breakout",
    _CapabilityError,
    template="MomentsSource does not support breakout dimensions -- rows are precomputed at construction, over (experiment_id, metric, group_id) only.",
)
_MOMENTS_UNIT_GRAIN = _RefusalSpec(
    "source.moments.unit_grain",
    _CapabilityError,
    template="{method!r} needs unit-grain rows, which this source does not retain -- only aggregated moments are available here. {method}() unavailable; use a frame-backed source (from_unit_summary / from_unit_panel) for unit-grain estimators.",
)
_MOMENTS_COVARIATE_UNAVAILABLE = _RefusalSpec(
    "source.moments.covariate_unavailable",
    _CapabilityError,
    lambda *, covariates: (
        f"unit_frame: covariates {list(covariates)!r} are not available from a "
        "moments-backed source -- a moments cube holds no per-unit rows to attach "
        "covariates or reconstruct observational weight diagnostics from. "
        "IPTW/AIPW diagnostics need per-arm weight sums, squared-weight sums, "
        "maximum weight, positive-weight counts, and the weight definition and "
        "unit/cluster grain (with cluster-total summaries at cluster grain). "
        "Use the original per-unit data with Analysis.from_definitions, "
        "from_unit_day_artifact (covariates the experiment declares), "
        "from_unit_summary, or from_unit_panel instead."
    ),
)
_MOMENTS_COUNTS = _RefusalSpec(
    "source.moments.assignment_counts",
    _CapabilityError,
    template="trustworthy source-level assignment counts; use Analysis.export() from a source or artifact that carries assignment-count evidence, or supply a cube whose rows carry the assignment_counts field.",
)
_MOMENTS_CLUSTER_GRAIN = _RefusalSpec(
    "source.moments.cluster_grain",
    _CapabilityError,
    lambda *, operation, source, cluster, design, mechanism, route_forward: (
        f"cluster_counts is unavailable on {source}: these moments retain no "
        f"cluster identity or cluster-count evidence. {route_forward}"
        if operation == "cluster_counts"
        else (
            f"{operation} is unavailable on {source}: clustered {design}/{mechanism} "
            f"source {cluster!r} cannot be transported through the moments adapter "
            f"because the wire format has no cluster marker. {route_forward}"
        )
    ),
)


def refuse_cluster_grain_transport(
    *,
    operation: str,
    source: str,
    cluster: str,
    design: object | None,
    route_forward: str,
) -> NoReturn:
    """Refuse ordinary clustered moments transport before destination mutation."""
    mechanism = getattr(design, "mechanism", design if isinstance(design, str) else None)
    _refuse(
        _MOMENTS_CLUSTER_GRAIN,
        operation=operation,
        source=source,
        cluster=cluster,
        design=mechanism,
        mechanism=mechanism,
        route_forward=route_forward,
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


_MOMENTS_SQL = _RefusalSpec(
    "source.moments.sql",
    _CapabilityError,
    template="sql() is not supported on a moments-backed source -- there is no query behind these rows, only precomputed arithmetic.",
)
_BREAKOUT_GRAIN = _RefusalSpec(
    "source.breakout.grain",
    _CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_BREAKOUT_DIMENSION = _RefusalSpec(
    "source.breakout.dimension",
    _CapabilityError,
    lambda *, dimension, requested: (
        f"BreakoutMomentsSource is scoped to by=[{dimension!r}] only -- got "
        f"by={list(requested)!r}. Construct a source scoped to the requested "
        "dimension instead."
    ),
)
_BREAKOUT_UNIT_GRAIN = _RefusalSpec(
    "source.breakout.unit_grain",
    _CapabilityError,
    template="{method!r} needs unit-grain rows, which this source does not retain -- only aggregated moments are available here. {method}() unavailable; use a frame-backed source (from_unit_summary / from_unit_panel) for unit-grain estimators.",
)
_BREAKOUT_COUNTS = _RefusalSpec(
    "source.breakout.assignment_counts",
    _CapabilityError,
    template="unit_counts() is not supported on a breakout-scoped moments source -- its rows are per (group_id, dimension_value), so a per-group count would silently report one segment's n as the whole arm's. Use DefinitionsMomentSource.moments_source() for total-grain counts.",
)
_BREAKOUT_CLUSTER_GRAIN = _RefusalSpec(
    "source.breakout.cluster_grain",
    _CapabilityError,
    template="cluster_counts() is unavailable on {source}: {because}. The randomization-grain count needs a declared cluster column; build the source with from_unit_summary(..., cluster=...) or analyse from definitions declaring Experiment.cluster.",
)
_BREAKOUT_SQL = _RefusalSpec(
    "source.breakout.sql",
    _CapabilityError,
    template="sql() is not supported on BreakoutMomentsSource -- there is no query behind these rows, only precomputed arithmetic.",
)
_BREAKOUT_COMPLIANCE = _RefusalSpec(
    "source.breakout.compliance_summary",
    _CapabilityError,
    template="compliance_summary() is not supported on a breakout-scoped moments source -- design-level compliance is a whole-population cohort, never conditioned on a breakout dimension. Read it off the unscoped MomentSource.",
)


# Fixed-horizon wire-format version stamped on exported moments cubes; the column
# survives parquet -> to_pylist round-trips. Registered sequential checkpoints
# use the separate sequential format below.
MOMENTS_FORMAT = 10
SEQUENTIAL_MOMENTS_FORMAT = 9
_SUPPORTED_MOMENTS_FORMATS = frozenset({MOMENTS_FORMAT, SEQUENTIAL_MOMENTS_FORMAT})

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
_READOUT_SOURCE_GRAIN = _RefusalSpec(
    "readout.source.grain",
    _CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_READOUT_SOURCE_DIMENSION = _RefusalSpec(
    "readout.source.dimension",
    _CapabilityError,
    lambda *, dimension, declared, source: (
        f"{source} does not support breakout dimension {dimension!r}; declared "
        f"breakouts: {list(declared)!r}"
    ),
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
        if version == SEQUENTIAL_MOMENTS_FORMAT:
            sequential_refuse(
                "source.invalid",
                f"format {SEQUENTIAL_MOMENTS_FORMAT} moments require exactly one typed sequential checkpoint envelope",
            )
        return False
    if version != SEQUENTIAL_MOMENTS_FORMAT or not isinstance(checkpoint, str):
        sequential_refuse(
            "continuation.legacy", "checkpoint envelopes require the current exact format"
        )
    if n_rows != 1:
        sequential_refuse("source.invalid", "checkpoint envelopes cannot mix with moment rows")
    snapshot = snapshot_from_json(checkpoint)
    if stripped != {"experiment_id": snapshot.registration.source_id}:
        sequential_refuse("source.invalid", "checkpoint envelope source or fields disagree")
    return True


def _validate_moments_plan_format_pair(version: int, plans: set[str]) -> None:
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
        if version != SEQUENTIAL_MOMENTS_FORMAT and kind in ("always_valid", "asymptotic_mean"):
            sequential_refuse(
                "continuation.legacy",
                "legacy sequential plans cannot replay without the current checkpoint envelope",
            )
        if version == SEQUENTIAL_MOMENTS_FORMAT and kind == "fixed":
            sequential_refuse(
                "source.invalid",
                f"format {SEQUENTIAL_MOMENTS_FORMAT} moments require a sequential checkpoint plan",
            )


def _cube_assignment_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int] | None:
    candidate = None
    for row in rows:
        decoded = _parse_assignment_counts(row.get(ASSIGNMENT_COUNTS_FIELD))
        if decoded is None or (candidate is not None and decoded != candidate):
            return None
        candidate = decoded
    return candidate


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


def _strip_moments_envelope_row(row, row_count):
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
    kind = stripped.get("record_kind")
    if kind == "design_summary":
        if version != MOMENTS_FORMAT:
            _refuse(_MOMENTS_FORMAT_INVALID, received=version, required=MOMENTS_FORMAT)
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
    elif _strip_sequential_envelope(stripped, version=version, n_rows=row_count):
        output_row = None
    else:
        output_row = stripped
    return version, payload, has_trigger, trigger_name, envelope_identity, output_row


def _check_moments_format(
    rows: Sequence[Mapping[str, object]],
    *,
    require_plan: bool = True,
) -> tuple[
    int,
    list[dict[str, object]],
    str | None,
    dict[str, int] | None,
    dict[str, object] | None,
    str | None,
    str | None,
]:
    """Validate and strip current format, plan, count and compliance envelopes."""
    out: list[dict[str, object]] = []
    versions: set[int] = set()
    plans: set[str] = set()
    envelope_identity: str | None = None
    trigger_values: list[object] = []
    trigger_rows = 0
    unplanned = 0
    planned_rows = 0
    for row in rows:
        version, payload, has_trigger, trigger_name, row_identity, output_row = (
            _strip_moments_envelope_row(row, len(rows))
        )
        versions.add(version)
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
    version = next(iter(versions), MOMENTS_FORMAT)
    _parse_moments_format(version)
    _validate_moments_plan_format_pair(version, plans)
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
        _validate_moment_counts(row)
        key = (str(row.get("metric")), str(row.get("group_id")))
        if key in seen_keys:
            _refuse(_MOMENTS_DUPLICATE_ROWS, key=key)
        seen_keys.add(key)
    assignment_counts = _cube_assignment_counts(rows)
    if envelope_identity is not None and assignment_counts is None:
        invalid_compliance_state(
            "design_summary requires valid assignment counts; re-export the complete source"
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
    )


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
        _refuse(_READOUT_SOURCE_GRAIN, grain=grain, offered=capabilities)
    by = request.by
    source_breakouts = tuple(getattr(request, "source_breakouts", ()))

    if by:
        missing = [name for name in by if name not in source_breakouts]
        if missing:
            _refuse(
                _READOUT_SOURCE_DIMENSION,
                dimension=missing[0],
                declared=source_breakouts,
                source=getattr(request, "source_name", "source"),
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
    """Use wire-carried runtime-safe methods and priors for rehydrated cubes."""
    from increment._analysis_config import ResolvedMetricConfig

    fallback_by_name = {config.metric.name: config for config in fallback}
    out: list[ResolvedMetricConfig] = []
    for metric in metrics:
        procedure = plan.procedures.get(metric.name)
        if not hasattr(procedure, "decision_method"):
            out.append(fallback_by_name[metric.name])
            continue
        runtime_procedure = cast("Any", procedure)
        out.append(
            ResolvedMetricConfig(
                metric=metric,
                decision_method=runtime_procedure.decision_method,
                sensitivity_methods=runtime_procedure.sensitivity_methods,
                prior=runtime_procedure.prior,
                prior_is_global=runtime_procedure.prior_is_global,
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


class MomentsSource(SequentialSourceMixin):
    """`MomentSource` over pre-computed `group_summary` rows: a moments cube
    (e.g. exported to parquet) rehydrated with no warehouse or frame, just
    centered moments.

    Centered moment rows and complete fixed-horizon design summaries carry a
    mandatory format stamp so an old additive-sum cube cannot be misread.
    """

    capabilities: frozenset[Grain] = frozenset({"total"})
    operations: frozenset[SourceOperation] = frozenset()
    shape: Literal["unit_summary", "unit_panel"] | None = None
    breakouts: tuple[str, ...] = ()

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
    ) -> None:
        (
            _version,
            stripped,
            plan_payload,
            assignment_counts,
            compliance_payload,
            envelope_identity,
            embedded_trigger_name,
        ) = _check_moments_format(rows, require_plan=plan is None)
        if envelope_identity is not None and envelope_identity != study_id:
            invalid_compliance_state(
                "design summary experiment identity differs from requested study"
            )
        if (
            embedded_trigger_name is not None
            and _trusted_trigger_name is not None
            and embedded_trigger_name != _trusted_trigger_name
        ):
            _refuse(
                _MOMENTS_FORMAT_INVALID,
                received=embedded_trigger_name,
                required="the trusted source trigger name",
            )
        trigger_name = embedded_trigger_name or _trusted_trigger_name
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
        if plan_payload is not None and (plan is None or envelope_identity is not None):
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

        from increment._analysis_config import resolve_configs

        base_configs = resolve_configs(
            metric_catalog,
            bindings=bindings_by_name,
            specs=None,
            methods=None,
            prior=None,
        )
        base_configs = _configs_from_compiled_plan(metric_catalog, compiled, base_configs)
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

        from increment.sequential_state import sequential_refuse, snapshot_from_json

        if getattr(compiled.inference, "registration", None) is not None and any(
            row.get("moments_format") != SEQUENTIAL_MOMENTS_FORMAT for row in rows
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
            _refuse(_MOMENTS_GRAIN, grain=grain, offered=self.capabilities)
        if by:
            _refuse(_MOMENTS_BREAKOUT)
        return [dict(r) for r in self._rows if r["metric"] == metric.name]

    def unit_frame(
        self,
        metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> NoReturn:
        if outcome_stage == "raw":
            from increment.winsor import winsor_refuse

            winsor_refuse(
                "raw_state_required", "This source does not retain exact pre-winsor unit outcomes."
            )
        if covariates:
            _refuse(_MOMENTS_COVARIATE_UNAVAILABLE, covariates=tuple(covariates))
        _refuse(_MOMENTS_UNIT_GRAIN, method="unit_frame")

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
        _refuse(
            _MOMENTS_CLUSTER_GRAIN,
            operation="cluster_counts",
            source="moments",
            cluster=self._context.cluster,
            design=getattr(self._context.design, "mechanism", None),
            mechanism=getattr(self._context.design, "mechanism", None),
            route_forward=(
                "build the source with from_unit_summary(..., cluster=...) or "
                "analyse from definitions declaring Experiment.cluster"
            ),
        )

    def compliance_dates(self) -> Sequence[object]:
        _refuse(_MOMENTS_GRAIN, grain="asof", offered=self.capabilities)

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
            _refuse(_MOMENTS_GRAIN, grain="asof", offered=self.capabilities)
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
        _refuse(_MOMENTS_SQL)

    def close(self) -> None:
        pass


def _design_summary_row(
    study_id: str,
    plan: str,
    counts: str,
    compliance: str | None,
    trigger_name: str | None,
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
    """Export total moments and immutable source-level compliance identity.

    A registered sequential plan exports its finalized checkpoint, which holds unit-record
    proofs and the declaration, never moments rows, so an unmodeled catalog quantile is not
    refused there. Every moments export refuses a catalog quantile before reading evidence
    (see :func:`refuse_quantile_moments_export` for ``observational_refusal``).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment.decision_wire import compiled_plan_to_json
    from increment.semantics.design import Encouragement

    context = source.context
    if getattr(context.plan.inference, "registration", None) is not None:
        from collections import Counter

        from increment.sequential_source import source_snapshot

        snapshot = source_snapshot(source)
        rows = [
            {
                "record_kind": "sequential_checkpoint",
                "experiment_id": context.study_id,
                "moments_format": SEQUENTIAL_MOMENTS_FORMAT,
                DECISION_PLAN_FIELD: compiled_plan_to_json(context.plan),
                SEQUENTIAL_SNAPSHOT_FIELD: snapshot.model_dump_json(),
                ASSIGNMENT_COUNTS_FIELD: json.dumps(
                    dict(Counter(r.group_id for r in snapshot.records)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ]
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
    plan = compiled_plan_to_json(context.plan)
    counts = json.dumps(source.unit_counts(), sort_keys=True, separators=(",", ":"))
    payload = (
        json.dumps(compliance.to_wire(), sort_keys=True, separators=(",", ":"))
        if compliance is not None
        else None
    )
    rows = []
    for metric in context.metrics:
        for raw in source.moments(metric):
            row = dict(cast("Mapping[str, object]", raw))
            _validate_moment_counts(row)
            row.update(
                {
                    "moments_format": MOMENTS_FORMAT,
                    DECISION_PLAN_FIELD: plan,
                    ASSIGNMENT_COUNTS_FIELD: counts,
                }
            )
            if payload is not None:
                row[COMPLIANCE_SUMMARY_FIELD] = payload
            if context.trigger_name is not None:
                row[TRIGGER_NAME_FIELD] = context.trigger_name
            rows.append(row)
    if not rows and not context.metrics:
        rows = [_design_summary_row(context.study_id, plan, counts, payload, context.trigger_name)]
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"increment.moments_format": str(MOMENTS_FORMAT).encode()}
    )
    if "successes" in table.column_names:
        index = table.schema.get_field_index("successes")
        table = table.set_column(index, "successes", table["successes"].cast(pa.int64()))
    pq.write_table(table, str(path))


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
    operations: frozenset[SourceOperation] = frozenset()
    shape: Literal["unit_summary", "unit_panel"] | None = None

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
    ) -> None:
        self._rows = [dict(r) for r in rows]
        self._dimension = dimension
        self._source_name = source_name
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
        from increment._analysis_config import resolve_configs

        base_configs = (
            tuple(configs)
            if configs is not None
            else resolve_configs(
                metric_catalog,
                bindings=None,
                specs=None,
                methods=None,
                prior=None,
            )
        )
        base_configs = _configs_from_compiled_plan(metric_catalog, compiled, base_configs)
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
            _refuse(_BREAKOUT_GRAIN, grain=grain, offered=self.capabilities)
        if list(by) != [self._dimension]:
            _refuse(
                _BREAKOUT_DIMENSION,
                dimension=self._dimension,
                requested=by,
            )
        return [dict(r) for r in self._rows if r["metric"] == metric.name]

    def unit_frame(
        self,
        metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> NoReturn:
        if outcome_stage == "raw":
            from increment.winsor import winsor_refuse

            winsor_refuse(
                "raw_state_required", "This source does not retain exact pre-winsor unit outcomes."
            )
        _refuse(_BREAKOUT_UNIT_GRAIN, method="unit_frame")

    def unit_counts(self) -> dict[str, int]:
        _refuse(_BREAKOUT_COUNTS)

    def cluster_counts(self) -> dict[str, int]:
        _refuse(
            _BREAKOUT_CLUSTER_GRAIN,
            source="a breakout-scoped moments source",
            because=(
                "breakout rows carry no cluster-robust variance -- "
                "group_summary()/breakout_summaries() never pass cluster= "
                "for a breakout, so a clustered experiment's breakout path "
                "is non-cluster-robust today"
            ),
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
        _refuse(_BREAKOUT_SQL)

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
