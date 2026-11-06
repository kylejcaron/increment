from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from increment._evidence_refusals import OPERATION_UNAVAILABLE
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    _safe_error_value,
)
from increment.errors import refuse as _refuse
from increment.semantics.models import Metric

if TYPE_CHECKING:
    from increment.semantics.models import Winsorization


_NATIVE_COVARIATE_UNRESOLVED = RefusalSpec(
    "source.native.covariate_unresolved",
    CapabilityError,
    lambda *, covariate, unit, source: (
        f"unit_frame: covariate {covariate!r} could not be resolved to a property "
        + (f"on fact source {source!r}" if source else "on any fact source")
        + f" with unit {unit!r} as an entity -- declare it as a Property on the "
        "fact source that carries it."
    ),
)
_NATIVE_COVARIATE_AMBIGUOUS = RefusalSpec(
    "source.native.covariate_ambiguous",
    CapabilityError,
    lambda *, covariate, candidates: (
        f"unit_frame: covariate {covariate!r} matches {len(candidates)} fact "
        f"sources ({', '.join(candidates)}) -- rename the property on one source, "
        "or declare the covariate under the experiment's observational design "
        "with 'source: <name>' to disambiguate."
    ),
)
_NATIVE_COVARIATE_AS_OF = RefusalSpec(
    "source.native.covariate_as_of",
    CapabilityError,
    template="unit_frame: covariate {covariate!r} (source {source!r}) has as_of={as_of!r} -- conditioning on a value measured at/after exposure biases the adjustment. Declare as_of='pre_exposure' or 'static' on the property.",
)
_NATIVE_COVARIATE_DTYPE = RefusalSpec(
    "source.native.covariate_dtype",
    CapabilityError,
    template="unit_frame: covariate {covariate!r} (source {source!r}) has dtype {dtype!r}; covariate adjustment needs a numeric (int/float/bool) or categorical (string) column -- a date carries no adjustment meaning. Derive a numeric or categorical pre-exposure property.",
)
_NATIVE_COVARIATE_KIND = RefusalSpec(
    "source.native.covariate_kind_mismatch",
    CapabilityError,
    template="unit_frame: covariate {covariate!r} (source {source!r}) has property type {dtype!r}, incompatible with declared kind {kind!r}.",
)


_NATIVE_OPERATION = OPERATION_UNAVAILABLE
_NATIVE_TRIGGER_EVIDENCE_REQUIRED = RefusalSpec(
    "source.native.trigger_evidence_required",
    CapabilityError,
    template=(
        "{operation} requires explicit SourceSnapshotEvidence from the upstream "
        "source; provide its event-time observation_cutoff_ts. Missing per-feed "
        "completeness certification remains unknown. {route}"
    ),
)


_NATIVE_COMPLIANCE_DESIGN_MISMATCH = RefusalSpec(
    "source.native.compliance_design_mismatch",
    InvalidRequestError,
    template=(
        "compliance_summary design mismatch: expected source design {expected!r}, "
        "received {received!r}; rebuild the source for the requested design"
    ),
)
_NATIVE_EXPERIMENT_KEY_UNMATCHED = RefusalSpec(
    "source.native.experiment_key_unmatched",
    InvalidRequestError,
    lambda *, experiment, observed_keys: (
        f"no exposure row carries experiment_id {experiment!r}, so no unit enrolled; the "
        f"rows carry {list(observed_keys)!r}. Name the experiment after the key its exposure "
        f"rows carry, or map that key to {experiment!r} in the exposure's source"
    ),
)
_NATIVE_FACT_NO_ROWS = WarningSpec(
    "source.native.fact_no_rows",
    IncrementWarning,
    lambda *, fact, fact_source, observed_events: (
        f"fact {fact!r} has no loaded rows in fact source {fact_source!r}, so every unit "
        f"scores zero on it; the source's event labels include {list(observed_events)!r}. "
        "Check the fact name against those labels and that the source is loaded"
    ),
)
_NATIVE_FACT_NULL_VALUES = RefusalSpec(
    "source.native.fact_null_values",
    InvalidRequestError,
    lambda *, fact, column, null_rows: (
        f"fact {fact!r} has {null_rows} row(s) with a NULL {column!r} among the rows this "
        "analysis reads, and a NULL value has no declared meaning. Fill those values, or "
        "declare the fact's missing: as 'skip' (drop those rows from every metric on the "
        "fact, counts included) or 'zero' (read them as 0)"
    ),
)


def _bounded_context_value(value: object, *, depth: int = 0) -> object:
    """Bound scalar values while preserving nested request context."""
    if depth >= 5:
        return _safe_error_value(value)
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_context_value(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_bounded_context_value(item, depth=depth + 1) for item in value)
    return _safe_error_value(value)


def _refuse_operation(
    *,
    operation: str,
    request: Mapping[str, object],
    offered: object,
    route: str,
) -> NoReturn:
    _refuse(
        _NATIVE_OPERATION,
        constructor="native source",
        requested=operation,
        request=_bounded_context_value(request),
        offered=_bounded_context_value(offered),
        route=route,
    )


def _percentile_winsorization(metric: Metric) -> Winsorization | None:
    """The metric's winsorization when any declared bound is a percentile."""
    config = getattr(metric, "winsorization", None)
    return config if config is not None and config.has_percentile else None
