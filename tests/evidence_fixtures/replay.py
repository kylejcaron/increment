from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Literal, cast

from increment._evidence_types import MetricEvidenceRequest, warehouse_cuped_fit_evidence
from increment.declarations import Method
from increment.errors import CodedError
from increment.estimation.engine import estimate_cuped_linear_cells, estimate_lift
from increment.evidence import (
    CountsRequest,
    MethodFitRequest,
    MomentsRequest,
    UnitRowsRequest,
)
from increment.semantics.models import Metric
from tests.evidence_fixtures.transcript import Transcript, _plain, _validate_record
from tests.parity_harness.comparison import TOLERANCE


@dataclass(frozen=True, slots=True)
class ScalarReplayConfig:
    control_group: str
    methods: tuple[Method, ...] | None = None
    prior: Any = None
    alpha: float | None = None
    alternative: str | None = None
    inference: Any = None
    null_lift: float | None = None
    null_abs: float | None = None
    preferred_direction: Literal["increase", "decrease", "neutral"] | None = None
    cluster: str | None = None
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None
    summary_population: Literal["assigned", "triggered"] | None = None


def estimate_scalar_transcript(
    transcript: Transcript,
    metrics: Sequence[Metric],
    config: ScalarReplayConfig,
) -> Any:
    """Run the production estimator over frozen aggregate rows in a transcript.

    Metrics and every estimator option are caller-supplied declarations. The
    adapter reads only plain transcript records; it never accepts or resolves a
    source, provider, or acquisition callback.
    """
    if not metrics:
        raise ValueError("at least one declared metric is required")

    metric_names = {metric.name for metric in metrics}
    if len(metric_names) != len(metrics):
        raise ValueError("metric declarations must have unique names")

    summary_rows: list[Mapping[str, Any]] = []
    found: set[str] = set()
    for record in transcript.records:
        if record["request_type"] != "MomentsRequest":
            continue
        request_metric = record["request"].get("metric")
        if not isinstance(request_metric, Mapping):
            continue
        name = request_metric.get("name")
        if name not in metric_names:
            continue
        if name in found:
            raise ValueError(f"duplicate moments answer for declared metric {name!r}")
        answer = record["answer"]
        if not isinstance(answer, list) or not all(isinstance(row, Mapping) for row in answer):
            raise TypeError(f"moments answer for {name!r} must be a list of plain rows")
        summary_rows.extend(answer)
        found.add(name)

    missing = metric_names - found
    if missing:
        raise ValueError(f"missing moments answers for declared metrics: {sorted(missing)}")

    return estimate_lift(
        metrics,
        summary_rows,
        control_group=config.control_group,
        methods=None if config.methods is None else list(config.methods),
        prior=config.prior,
        alpha=config.alpha,
        alternative=config.alternative,
        inference=config.inference,
        null_lift=config.null_lift,
        null_abs=config.null_abs,
        preferred_direction=config.preferred_direction,
        cluster=config.cluster,
        method_roles=config.method_roles,
        summary_population=config.summary_population,
    )


def estimate_fitted_cuped_transcript(
    transcript: Transcript,
    request: MethodFitRequest,
    *,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    prior: Any = None,
) -> Any:
    """Project a retained or warehouse fit through its pure estimator."""
    from increment.estimation.engine import estimate_cuped_linear_fit

    typed_envelope = _fitted_cuped_envelope(_method_fit_answer(transcript, request), request)
    return estimate_cuped_linear_fit(
        request.metric,
        typed_envelope,
        arm_order=request.arm_order,
        control_group=request.control,
        method=request.method,
        method_role=method_role,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
    )


def _fitted_cuped_envelope(answer: Mapping[str, Any], request: MethodFitRequest) -> Any:
    from increment._cuped_evidence import _fitted_dataclass
    from increment.estimation.cuped.encoding import MulticovariateCupedLayout
    from increment.estimation.cuped.types import MulticovariateCupedFit

    if answer["origin"] not in {"retained_fitted", "warehouse"}:
        raise ValueError("fitted CUPED requires retained_fitted or warehouse evidence")
    envelope = answer["envelope"]
    if not isinstance(envelope, Mapping) or envelope.keys() != {"arm_order", "fit", "withheld"}:
        raise ValueError("fitted CUPED envelope is incomplete or unexpected")
    if tuple(envelope["arm_order"]) != request.arm_order:
        raise ValueError("fitted CUPED envelope arms differ from the request")
    withheld = tuple((arm, _coded_error(error)) for arm, error in envelope["withheld"])
    withheld_arms = tuple(arm for arm, _ in withheld)
    if len(set(withheld_arms)) != len(withheld_arms) or any(
        arm not in request.arm_order for arm in withheld_arms
    ):
        raise ValueError("fitted CUPED withheld arms differ from the request")
    admitted = tuple(arm for arm in request.arm_order if arm not in withheld_arms)
    _validate_fit_identity(answer["fit_identity"], request, admitted)
    fit_data = envelope["fit"]
    if fit_data is None:
        if not withheld or (request.control in admitted and len(admitted) >= 2):
            raise ValueError("fitted CUPED envelope is missing an admitted fit")
        fit = None
    elif isinstance(fit_data, Mapping) and fit_data.keys() == {"code", "message", "context"}:
        fit = _coded_error(fit_data)
    else:
        fit = _fitted_dataclass(
            MulticovariateCupedFit,
            fit_data,
            nested={"layout": MulticovariateCupedLayout},
        )
        covariates = request.method.covariates or ()
        if (
            fit.control != request.control
            or tuple(fit.arm_order) != admitted
            or fit.layout.names != tuple(item.name for item in covariates)
            or fit.layout.kinds != tuple(item.kind for item in covariates)
            or fit.is_ratio != (request.metric.type == "ratio")
        ):
            raise ValueError("fitted CUPED fit declarations differ from the request")
    return warehouse_cuped_fit_evidence(fit, request.arm_order, withheld)


def _coded_error(value: Mapping[str, Any]) -> CodedError:
    if (
        not isinstance(value, Mapping)
        or value.keys() != {"code", "message", "context"}
        or not isinstance(value["code"], str)
        or not isinstance(value["message"], str)
        or not isinstance(value["context"], Mapping)
    ):
        raise ValueError("coded failure fields are incomplete or unexpected")
    return CodedError(value["message"], code=value["code"], context=value["context"])


def _method_fit_answer(transcript: Transcript, request: MethodFitRequest) -> Mapping[str, Any]:
    answer = transcript.answer_for(request)
    if answer["failure"] is not None:
        raise _coded_error(answer["failure"])
    return answer


def _validate_fit_identity(
    identity: Sequence[Any], request: MethodFitRequest, admitted: tuple[str, ...]
) -> None:
    from increment._cuped_evidence import method_key

    cell = request.cell
    scope = (
        (None, None, None, False)
        if cell is None
        else (cell.breakout.property, cell.breakout.source, cell.value, cell.is_null)
    )
    if (
        len(identity) not in (11, 13)
        or tuple(identity[:3])
        != ("warehouse-cuped-fit-v1", request.metric.name, request.population)
        or identity[4] != method_key(request.method)
        or identity[5] != request.control
        or tuple(identity[6]) != request.arm_order
        or _plain(identity[7:11]) != _plain(scope)
        or (len(identity) == 13 and _plain(identity[11:]) != _plain(("admitted_arms", admitted)))
    ):
        raise ValueError("fitted CUPED identity differs from the request")


_ASSEMBLY_ONLY_FIELDS = frozenset(
    {
        "source_snapshot_id",
        "family_axes",
        "family_guarantee",
        "family_id",
        "family_nominal_alpha",
        "family_procedure",
        "family_q",
        "family_size",
        "family_threshold",
        "ds",
        "metric_type",
        "multiplicity_status",
        "decision_scope_complete",
        "decision_scope_reason_code",
        "decision_scope_reason_context",
        "role",
        "discovery",
    }
)

_ESTIMATOR_FIELDS = frozenset(
    {
        "metric",
        "group_id",
        "method",
        "method_role",
        "inference",
        "alternative",
        "sampling_available",
        "sampling_reason_code",
        "sampling_reason_context",
        "posterior_available",
        "posterior_components",
        "posterior_reason_code",
        "posterior_reason_context",
        "posterior_model",
        "posterior_scale",
        "posterior_estimate",
        "posterior_lb",
        "posterior_ub",
        "posterior_level",
        "posterior_alpha",
        "posterior_latent_mean",
        "posterior_latent_sd",
        "posterior_prob_favorable",
        "posterior_prob_favorable_reason_code",
        "absolute_error_code",
        "absolute_error_context",
        "relative_error_code",
        "relative_error_context",
        "failure_code",
        "failure_context",
        "weight_diagnostics_available",
        "weight_diagnostics_reason_code",
        "weight_diagnostics_reason_context",
        "weight_definition",
        "weight_grain",
        "control_weight_ess",
        "treatment_weight_ess",
        "control_weight_max_share",
        "treatment_weight_max_share",
        "control_weight_n",
        "treatment_weight_n",
        "n_control",
        "n_treat",
        "mean_control",
        "mean_treat",
        "null_lift",
        "preferred_direction",
        "lift",
        "estimand",
        "analysis_population",
        "value_scale",
        "note",
        "abs_diff",
        "abs_se",
        "null_abs",
        "abs_lb",
        "abs_ub",
        "abs_reference_kind",
        "abs_reference_df",
        "abs_alpha",
        "allocation_alpha",
        "winsor_lower_percentile",
        "winsor_upper_percentile",
        "winsor_lower_bound",
        "winsor_upper_bound",
        "winsor_control_n",
        "winsor_control_n_lower",
        "winsor_control_n_upper",
        "winsor_treatment_n",
        "winsor_treatment_n_lower",
        "winsor_treatment_n_upper",
        "n_clusters",
        "dof",
        "reference_kind",
        "confidence_set",
        "independent_mean_reference",
        "reference_df",
        "population",
        "relative_confidence_set",
        "relative_unavailable_reason",
        "quantile_p_value",
        "sequential_result",
        "binomial_set",
        "delta_evidence",
        "scale",
        "prior_spec",
        "prior_shrunk",
    }
)


def estimate_categorical_cuped_transcript(
    transcript: Transcript,
    request: MethodFitRequest,
    *,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    prior: Any = None,
) -> Any:
    """Project the categorical cells belonging to one complete fit request."""
    from increment._cuped_evidence import method_key
    from increment.estimation.cuped.types import CategoricalCells

    answer = _method_fit_answer(transcript, request)
    if answer["origin"] != "retained_categorical" or not isinstance(answer["cells"], Mapping):
        raise ValueError("categorical CUPED requires retained_categorical cells")
    cells = CategoricalCells(**answer["cells"])
    covariates = request.method.covariates or ()
    if (
        len(cells.identity) != 5
        or cells.identity[:3] != ("cuped_evidence", request.metric.name, request.population)
        or cells.identity[4] != method_key(request.method)
        or cells.arms != request.arm_order
        or request.control not in cells.arms
        or cells.covariate_names != tuple(item.name for item in covariates)
        or any(item.kind != "categorical" for item in covariates)
        or tuple(item[0] for item in cells.null_metadata)
        != tuple(item.missing for item in covariates)
    ):
        raise ValueError("categorical CUPED cell declarations differ from the request")
    return estimate_cuped_linear_cells(
        request.metric,
        cells,
        control_group=request.control,
        method=request.method,
        method_role=method_role,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
    )


def estimate_unit_cuped_transcript(
    transcript: Transcript,
    request: MetricEvidenceRequest,
    *,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    prior: Any = None,
) -> Any:
    """Fit the declared method on materialized metric-level unit evidence."""
    import pandas as pd

    from increment.estimation.engine import estimate_cuped_linear

    answer = transcript.answer_for(request)
    if _plain(method) not in [_plain(item) for item in request.methods]:
        raise ValueError("CUPED method differs from the metric evidence request")
    covariates = method.covariates or request.covariates
    if not covariates or any(
        _plain(item) not in [_plain(declared) for declared in request.covariates]
        for item in covariates
    ):
        raise ValueError("CUPED covariates differ from the metric evidence request")
    arm_order = (
        request.control_group,
        *sorted(set(answer["observed_arms"]) - {request.control_group}),
    )
    expected_fit_request = MethodFitRequest(
        request.metric, method, request.control_group, arm_order, request.population
    )
    for fit_request, fit_answer in answer["method_fit_answers"]:
        if fit_request["method"]["name"] == method.name:
            if fit_request != _plain(expected_fit_request):
                raise ValueError("CUPED fit identity differs from the metric evidence request")
            if fit_answer["failure"] is not None:
                raise _coded_error(fit_answer["failure"])
            if fit_answer["origin"] != "unavailable":
                raise ValueError("CUPED metric evidence already contains a fit")
    frame_data = answer["unit_frame"]
    if not isinstance(frame_data, Mapping) or frame_data.keys() != {"columns", "schema", "rows"}:
        raise ValueError("CUPED unit frame is missing or incomplete")
    if set(frame_data["schema"]) != set(frame_data["columns"]):
        raise ValueError("CUPED unit frame schema differs from its columns")
    frame = pd.DataFrame(frame_data["rows"], columns=frame_data["columns"])
    columns = dict(answer["covariate_columns"])
    if (
        len(columns) != len(answer["covariate_columns"])
        or set(columns) != {item.name for item in request.covariates}
        or not set(columns.values()) <= set(frame.columns)
        or answer["control_group"] != request.control_group
        or set(frame["group_id"]) != set(answer["observed_arms"])
        or request.control_group not in answer["observed_arms"]
        or answer["n_treatment_arms"] != len(set(answer["observed_arms"]) - {request.control_group})
    ):
        raise ValueError("CUPED unit evidence declarations differ from the request")
    return estimate_cuped_linear(
        request.metric,
        frame,
        control_group=request.control_group,
        method=method,
        method_role=method_role,
        covariates=covariates,
        covariate_columns=columns,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
    )


_REQUIRED_ESTIMATOR_FIELDS = frozenset(
    {
        "metric",
        "group_id",
        "method",
        "method_role",
        "inference",
        "alternative",
        "sampling_available",
        "lift",
        "abs_diff",
        "abs_se",
        "abs_lb",
        "abs_ub",
        "reference_kind",
    }
)


def assert_estimator_matches_old_row(
    result: Any,
    old_row: Mapping[str, Any],
    *,
    consumer: Literal["scalar", "cuped"] = "scalar",
) -> None:
    """Compare all serialized estimator-owned fields to an unchanged frozen row.

    The CUPED fit consumer itself leaves ``sampling_available`` unset; its
    public row assembly stamps that status after the pure estimate is returned.
    """
    actual = result.model_dump(mode="python")
    excluded = _ASSEMBLY_ONLY_FIELDS
    if consumer == "cuped":
        excluded |= {"sampling_available"}
    model_fields = set(type(result).model_fields)
    unclassified = model_fields - _ASSEMBLY_ONLY_FIELDS - _ESTIMATOR_FIELDS
    if unclassified:
        raise AssertionError(f"result fields lack an ownership mapping: {sorted(unclassified)}")
    required = (_ESTIMATOR_FIELDS - excluded) & old_row.keys()
    missing = required - actual.keys()
    if missing:
        raise AssertionError(f"estimator result omitted frozen owned fields: {sorted(missing)}")
    required_core = _REQUIRED_ESTIMATOR_FIELDS - excluded
    absent_oracle = required_core - old_row.keys()
    absent_actual = required_core - actual.keys()
    if absent_oracle or absent_actual:
        raise AssertionError(
            "estimator projection is incomplete: "
            f"old_row_missing={sorted(absent_oracle)}, actual_missing={sorted(absent_actual)}"
        )
    projection = {key: actual[key] for key in required}
    if not projection:
        raise AssertionError("frozen row has no estimator-owned fields in common")
    for key, value in projection.items():
        _assert_projected_value(key, value, old_row[key])


def _assert_projected_value(path: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"{path} fields differ: actual={sorted(actual)}, expected={sorted(expected)}"
            )
        for key, value in actual.items():
            _assert_projected_value(f"{path}.{key}", value, expected[key])
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(
                f"{path} length differs: actual={len(actual)}, expected={len(expected)}"
            )
        for index, value in enumerate(actual):
            _assert_projected_value(f"{path}[{index}]", value, expected[index])
        return
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise AssertionError(f"{path} type differs: actual={actual!r}, expected={expected!r}")
        same = abs(actual - expected) <= TOLERANCE * max(1.0, abs(expected))
        if abs(actual) > TOLERANCE or abs(expected) > TOLERANCE:
            same = (
                same
                and (actual == 0.0) == (expected == 0.0)
                and math.copysign(1, actual) == math.copysign(1, expected)
            )
        if not same:
            raise AssertionError(f"{path} differs: actual={actual!r}, expected={expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path} differs: actual={actual!r}, expected={expected!r}")


_RAW_MOMENT_SCENARIOS = frozenset(
    {
        "audit-ratio-neighboring-means",
        "audit-cuped-covariate-rescaling",
        "exact_binomial_beyond_the_former_arm_ceiling",
    }
)
_RAW_CENTERED_OPERANDS = {
    "cy1": ("y",),
    "cy2": ("y", "y"),
    "cx1": ("x",),
    "cx2": ("x", "x"),
    "cxy": ("x", "y"),
    "cden1": ("den",),
    "cden2": ("den", "den"),
    "cden3": ("den", "den", "den"),
    "cyden": ("y", "den"),
    "cxden": ("x", "den"),
}
_RAW_NULL_FIELDS = frozenset(
    {
        "sum_d",
        "cyd",
        "cy2d",
        "cxd",
        "winsor_lower_percentile",
        "winsor_upper_percentile",
        "winsor_lower_bound",
        "winsor_upper_bound",
        "winsor_n",
        "winsor_n_lower",
        "winsor_n_upper",
    }
)
_RAW_MOMENT_ROW_FIELDS = (
    _RAW_NULL_FIELDS
    | _RAW_CENTERED_OPERANDS.keys()
    | {"experiment_id", "group_id", "metric", "n", "successes", "x_role"}
    | {"ref_y", "ref_x", "ref_den"}
)


def _raw_moment_population(
    scenario_id: str,
) -> dict[str, list[tuple[dict[str, float], int]]]:
    from tests.parity_harness import cases

    if scenario_id == "exact_binomial_beyond_the_former_arm_ceiling":
        counts = {
            "control": (cases._CEILING_CONTROL_UNITS, cases._CEILING_CONTROL_CONVERSIONS),
            "treatment": (cases._CEILING_TREATMENT_UNITS, cases._CEILING_TREATMENT_CONVERSIONS),
        }
        return {
            arm: [({"y": 1.0, "converted": 1.0}, k), ({"y": 0.0, "converted": 0.0}, n - k)]
            for arm, (n, k) in counts.items()
        }

    events = (
        cases._neighboring_ratio_event_rows()
        if scenario_id == "audit-ratio-neighboring-means"
        else cases._covariate_rescaled_event_rows()
    )
    assignments: dict[str, Mapping[str, Any]] = {}
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        unit = event["user_id"]
        by_unit[unit].append(event)
        if event["event"] == "exposure":
            if unit in assignments:
                raise ValueError("raw moment declaration has duplicate assignments")
            assignments[unit] = event
    population: dict[str, list[tuple[dict[str, float], int]]] = {
        "control": [],
        "treatment": [],
    }
    for unit, assignment in assignments.items():
        start = assignment["event_at"]
        stop = start + dt.timedelta(days=1)
        purchases = [
            event["revenue"]
            for event in by_unit[unit]
            if event["event"] == "purchase" and start <= event["event_at"] < stop
        ]
        pre = [
            event["revenue"]
            for event in by_unit[unit]
            if event["event"] == "purchase"
            and start - dt.timedelta(days=7) <= event["event_at"] < start
        ]
        if len(purchases) > 1 or len(pre) != 1 or assignment["group_id"] not in population:
            raise ValueError("raw moment event declaration has an unsupported unit shape")
        y = math.fsum(purchases)
        converted = float(bool(purchases))
        if converted != float(y > 0):
            raise ValueError("raw conversion event and summary projections disagree")
        den = sum(
            event["event"] == "session_end" and start <= event["event_at"] < stop
            for event in by_unit[unit]
        )
        population[assignment["group_id"]].append(
            ({"y": y, "x": math.fsum(pre), "den": float(den), "converted": converted}, 1)
        )
    if any(not units for units in population.values()) or by_unit.keys() != assignments.keys():
        raise ValueError("raw moment declaration has incomplete assignments")
    return population


def _raw_moment_sum(terms: Sequence[tuple[float, int]]) -> float:
    if all(count == 1 for _, count in terms):
        return math.fsum(term for term, _ in terms)
    # Multiply exact binary64 values, not rounded count * term products.
    return float(sum((Fraction.from_float(term) * count for term, count in terms), Fraction()))


def _raw_moment_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _validate_raw_moment_rows(
    rows: Any,
    *,
    path: str,
    metric: str,
    experiment: str,
    covariate: bool,
    population: Mapping[str, Sequence[tuple[Mapping[str, float], int]]],
) -> None:
    if not isinstance(rows, (list, tuple)) or len(rows) != len(population):
        raise ValueError(f"{path} must contain exactly one row per declared arm")
    seen: set[str] = set()
    present = {"ref_y", "cy1", "cy2"}
    if covariate:
        present |= {"ref_x", "cx1", "cx2", "cxy"}
    if metric in {"rps", "rps_cuped"}:
        present |= {"ref_den", "cden1", "cden2", "cden3", "cyden"}
        if covariate:
            present.add("cxden")
    for index, row in enumerate(rows):
        row_path = f"{path}[{index}]"
        if not isinstance(row, Mapping) or row.keys() != _RAW_MOMENT_ROW_FIELDS:
            raise ValueError(f"{row_path} moment fields are incomplete or unexpected")
        arm = row["group_id"]
        if not isinstance(arm, str) or arm not in population or arm in seen:
            raise ValueError(f"{row_path} arm is undeclared or duplicated")
        seen.add(arm)
        if row["metric"] != metric or row["experiment_id"] != experiment:
            raise ValueError(f"{row_path} metric or experiment differs from its request")
        units = population[arm]
        n = sum(count for _, count in units)
        successes = sum(int(unit["converted"]) * count for unit, count in units)
        if type(row["n"]) is not int or row["n"] != n:
            raise ValueError(f"{row_path}.n differs from the declared exact count")
        binary = metric in {"purchase_rate", "conversion"}
        if (binary and (type(row["successes"]) is not int or row["successes"] != successes)) or (
            not binary and row["successes"] is not None
        ):
            raise ValueError(f"{row_path}.successes differs from the declared exact count")
        if row["x_role"] != ("covariate" if covariate else None):
            raise ValueError(f"{row_path}.x_role differs from the declared role")
        numeric_fields = _RAW_CENTERED_OPERANDS.keys() | {"ref_y", "ref_x", "ref_den"}
        for field in _RAW_NULL_FIELDS | (numeric_fields - present):
            if row[field] is not None:
                raise ValueError(f"{row_path}.{field} must be null for this declaration")
        for field in present:
            _raw_moment_number(row[field], f"{row_path}.{field}")
        for variable in ("y", "x", "den"):
            field = f"ref_{variable}"
            if field not in present:
                continue
            key = "converted" if variable == "y" and binary else variable
            expected = _raw_moment_sum([(unit[key], count) for unit, count in units]) / n
            if not math.isclose(row[field], expected, rel_tol=1e-9, abs_tol=0.0):
                raise ValueError(f"{row_path}.{field} differs from the independent raw mean")
        for field, operands in _RAW_CENTERED_OPERANDS.items():
            if field not in present:
                continue
            terms = []
            for unit, count in units:
                term = 1.0
                for variable in operands:
                    key = "converted" if variable == "y" and binary else variable
                    term *= unit[key] - row[f"ref_{variable}"]
                terms.append((term, count))
            expected = _raw_moment_sum(terms)
            scale = _raw_moment_sum([(abs(term), count) for term, count in terms])
            error = abs(row[field] - expected)
            bound = 1e-9 * scale
            if not math.isfinite(expected) or not math.isfinite(scale) or error > bound:
                raise ValueError(
                    f"{row_path}.{field} violates the raw-term bound: "
                    f"actual={row[field]!r}, expected={expected!r}, bound={bound!r}"
                )


def _raw_moment_record(record: Any, path: str) -> tuple[str, Mapping[str, Any], Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{path} must be a plain mapping")
    plain = cast(Mapping[str, Any], record)
    try:
        _validate_record(dict(plain))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid transcript schema: {exc}") from exc
    request_class = {
        "MetricEvidenceRequest": MetricEvidenceRequest,
        "MomentsRequest": MomentsRequest,
        "CountsRequest": CountsRequest,
        "MethodFitRequest": MethodFitRequest,
        "UnitRowsRequest": UnitRowsRequest,
    }.get(plain["request_type"])
    request = plain["request"]
    if (
        request_class is None
        or not isinstance(request, Mapping)
        or request.keys() != {field.name for field in fields(request_class)}
        or request["population"] != "assigned"
    ):
        raise ValueError(f"{path} has an unsupported raw moment request")
    return plain["request_type"], request, plain["answer"]


def validate_raw_moment_records(scenario_id: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Validate three declared raw populations at each producer's own references.

    Finite binary64 residual products are formed left to right. Each centered
    field must differ from their independently summed raw terms by at most
    ``1e-9 * sum(abs(terms))``, with no absolute floor. References use relative
    ``1e-9`` and absolute zero; counts, roles, null masks and row schemas are exact.
    Repeated binary values use exact rational multiplicities before rounding,
    avoiding both a large unit expansion and rounded ``count * term`` products.

    Only ordinary metric rows, scalar CUPED method rows and portable total
    moment rows are admitted. Ancillary fit and unit records retain their
    transcript schema checks, without receiving centered-field authority.
    This function reads original event declarations or binary counts, never
    acquired sources or captured centered answers. Other transcript fields
    remain the caller's comparison responsibility.
    """
    if not isinstance(scenario_id, str) or scenario_id not in _RAW_MOMENT_SCENARIOS:
        raise ValueError(f"unsupported raw moment scenario: {scenario_id!r}")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes, bytearray))
        or not records
    ):
        raise ValueError("raw moment records must be a nonempty plain sequence")
    metric_types = (
        {"conversion": "conversion"}
        if scenario_id == "exact_binomial_beyond_the_former_arm_ceiling"
        else {
            "revenue": "mean",
            "purchase_rate": "conversion",
            "rps": "ratio",
            "revenue_cuped": "mean",
            "rps_cuped": "ratio",
        }
    )
    population = _raw_moment_population(scenario_id)
    seen_metrics: set[str] = set()
    counts_seen = False
    row_kinds: set[str] = set()
    for index, record in enumerate(records):
        path = f"records[{index}]"
        kind, request, answer = _raw_moment_record(record, path)
        if kind in {"MethodFitRequest", "UnitRowsRequest"}:
            if scenario_id != "audit-cuped-covariate-rescaling":
                raise ValueError(f"{path} ancillary record is not declared for this scenario")
            continue
        if kind == "CountsRequest":
            counts = {arm: sum(count for _, count in units) for arm, units in population.items()}
            if (
                counts_seen
                or request["kind"] != "unit"
                or answer["population"] != "assigned"
                or answer["grain"] != "unit"
                or answer["unit_counts"] != {}
                or not isinstance(answer["counts"], Mapping)
                or answer["counts"] != counts
                or any(type(value) is not int for value in answer["counts"].values())
            ):
                raise ValueError(f"{path} arm counts differ from the exact declaration")
            counts_seen = True
            continue
        row_kinds.add(kind)
        declaration = request["metric"]
        if not isinstance(declaration, Mapping):
            raise ValueError(f"{path} metric declaration must be a mapping")
        metric = declaration.get("name")
        if (
            not isinstance(metric, str)
            or metric not in metric_types
            or metric in seen_metrics
            or declaration.get("type") != metric_types[metric]
            or request["by"] not in ([], ())
            or declaration.get("entity") not in {"user_id", "__frame__"}
        ):
            raise ValueError(f"{path} metric identity or scope differs from the declaration")
        seen_metrics.add(metric)
        experiment = "frame" if declaration["entity"] == "__frame__" else "exp"
        cuped = metric in {"revenue_cuped", "rps_cuped"}
        if kind == "MomentsRequest":
            if (
                request["grain"] != "total"
                or request["method"] is not None
                or request["experiment"] is not None
                or request["include_covariate"] is not False
                or request["completed_windows_only"] is not False
            ):
                raise ValueError(f"{path} portable moment scope is unsupported")
            collections = [(f"{path}.answer", answer, cuped)]
        else:
            if (
                request["control_group"] != "control"
                or answer["control_group"] != "control"
                or answer["cluster"] is not None
                or answer["observed_arms"] != ["control", "treatment"]
                or type(answer["n_treatment_arms"]) is not int
                or answer["n_treatment_arms"] != 1
            ):
                raise ValueError(f"{path} metric arm declarations are unsupported")
            collections = [(f"{path}.answer.rows", answer["rows"], False)]
            methods = answer["method_rows"]
            if not isinstance(methods, (list, tuple)) or len(methods) != int(cuped):
                raise ValueError(f"{path} method row roster differs from the declaration")
            if cuped:
                pair = methods[0]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] != "cuped":
                    raise ValueError(f"{path} method row identity differs from the declaration")
                collections.append((f"{path}.answer.method_rows[0][1]", pair[1], True))
        for row_path, rows, covariate in collections:
            _validate_raw_moment_rows(
                rows,
                path=row_path,
                metric=metric,
                experiment=experiment,
                covariate=covariate,
                population=population,
            )
    if seen_metrics != metric_types.keys() or not counts_seen or len(row_kinds) != 1:
        raise ValueError("raw moment metric/count roster is incomplete or mixed")
