from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, _safe_error_value
from increment.errors import refuse as _refuse
from increment.semantics.models import Metric

if TYPE_CHECKING:
    from increment.semantics.models import Winsorization

    _ContextScalar = str | int | float | bool | None
    _ContextValue = _ContextScalar | tuple[_ContextScalar, ...] | Mapping[str, _ContextScalar]


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


_NATIVE_WAREHOUSE_CLUSTER = RefusalSpec(
    "source.native.warehouse_cluster_grain",
    CapabilityError,
    template="cluster_counts() is unavailable on {source}: {because}. The randomization-grain count needs a declared cluster column; build the source with from_unit_summary(..., cluster=...) or analyse from definitions declaring Experiment.cluster.",
)
_NATIVE_SQL_GRAIN = RefusalSpec(
    "source.native_sql_grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_NATIVE_OPERATION = RefusalSpec(
    "source.native.operation",
    CapabilityError,
    template=(
        "native source cannot perform {operation!r} for {request!r}; "
        "available capability is {offered!r}. {route}"
    ),
)
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


def _bounded_context_value(value: object) -> object:
    """One bounded, pickle-safe context value; containers one level deep."""
    if isinstance(value, Mapping):
        return {str(key): _safe_error_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_safe_error_value(item) for item in value)
    return _safe_error_value(value)


def _refuse_operation(
    *,
    operation: str,
    request: Mapping[str, _ContextValue],
    offered: _ContextValue,
    route: str,
) -> NoReturn:
    """Refuse one native operation with its actual request and served capability."""
    _refuse(
        _NATIVE_OPERATION,
        operation=operation,
        request={key: _bounded_context_value(value) for key, value in request.items()},
        offered=_bounded_context_value(offered),
        route=route,
    )


def _winsorization_bounds(winsorization: Winsorization) -> dict[str, float | None]:
    """The declared winsorization bounds as plain context values."""
    return {
        "lower_percentile": winsorization.lower_percentile,
        "upper_percentile": winsorization.upper_percentile,
        "lower_value": winsorization.lower_value,
        "upper_value": winsorization.upper_value,
    }


def _percentile_winsorization(metric: Metric) -> Winsorization | None:
    """The metric's winsorization when any declared bound is a percentile."""
    config = getattr(metric, "winsorization", None)
    return config if config is not None and config.has_percentile else None
