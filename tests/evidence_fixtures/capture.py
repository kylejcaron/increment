from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from functools import partial, wraps
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from increment import Experiment, Scope, run
from increment.errors import CodedError
from increment.evidence import DayAxisRows, MetricEvidence
from increment.kernel.contracts import contract_hash, method_estimator_id
from increment.kernel.resolver import Resolver

# Each entry is an acquisition boundary, not a reducer or estimator registry.
PRODUCERS = (
    ("increment.readouts._metric_rows", "load_metric_evidence"),
    ("increment.readouts._metric_rows", "_load_breakout_metric_rows"),
    ("increment.readouts._asof", "_asof_metric_rows"),
    ("increment.readouts._evidence", "load_day_axis_rows"),
    ("increment.readouts._evidence", "load_counts"),
    ("increment.readouts._evidence", "load_assignment"),
    ("increment.readouts._evidence", "load_method_fit"),
    ("increment.readouts._evidence", "load_breakout_evidence"),
    ("increment.readouts._evidence", "load_breakout_method_moments"),
    ("increment.readouts._evidence", "load_compliance"),
    ("increment.readouts._evidence", "load_compliance_series"),
    ("increment.readouts._evidence", "load_sequential_capture"),
    ("increment.readouts._evidence", "current_source_snapshot"),
    ("increment._evidence_loaders", "load_moments"),
    ("increment._evidence_loaders", "load_unit_rows"),
    ("increment._evidence_loaders", "load_contrast_stats"),
    ("increment.estimation.cuped.prepare", "prepare_multicovariate_fit"),
)


@dataclass(frozen=True, slots=True)
class RequestCoordinates:
    population: str
    grain: str
    day: str | None = None
    segment: tuple[tuple[str, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class CaptureKey:
    estimator_id: str
    method_identity: str
    metric: str
    scope_hash: str
    request_coordinates: RequestCoordinates
    arm: str


def method_identity(method: Any) -> str:
    """Hash the complete declaration, including its name and covariates."""
    return contract_hash("evidence_fixture_method", method.model_dump(mode="json"))


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, CodedError):
        return {"code": value.code, "context": _json(value.context)}
    if isinstance(value, BaseModel):
        return _json(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return {item.name: _json(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_json(item) for item in value), key=_encoded)
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if hasattr(value, "to_dicts"):
        return _json(value.to_dicts())
    if hasattr(value, "to_pylist"):
        return _json(value.to_pylist())
    raise TypeError(f"No evidence fixture encoding for {type(value).__qualname__}")


def _encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(init=False)
class EvidenceRecord:
    entries: Mapping[CaptureKey, Any]

    def __init__(self, entries: Iterable[tuple[CaptureKey, Any]]) -> None:
        captured = {}
        for key, value in entries:
            assert key not in captured, f"Duplicate capture key: {key}"
            captured[key] = _json(value)
        self.entries = MappingProxyType(captured)

    def to_payload(self) -> dict:
        records = [{"key": _json(key), "evidence": value} for key, value in self.entries.items()]
        return {
            "fixture_version": 1,
            "records": sorted(records, key=lambda row: _encoded(row["key"])),
        }


def assert_same_evidence(a: EvidenceRecord, b: EvidenceRecord, *, rtol: float = 1e-12) -> None:
    assert a.entries and b.entries, "Empty evidence is not a parity comparison"
    assert a.entries.keys() == b.entries.keys(), (
        a.entries.keys() - b.entries.keys(),
        b.entries.keys() - a.entries.keys(),
    )

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, float) or isinstance(right, float):
            assert isinstance(left, (int, float)) and not isinstance(left, bool), path
            assert isinstance(right, (int, float)) and not isinstance(right, bool), path
            assert math.isclose(left, right, rel_tol=rtol, abs_tol=rtol), (path, left, right)
        elif isinstance(left, dict):
            assert isinstance(right, dict) and left.keys() == right.keys(), path
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            assert isinstance(right, list) and len(left) == len(right), path
            for index, (x, y) in enumerate(zip(left, right, strict=True)):
                compare(x, y, f"{path}[{index}]")
        else:
            assert type(left) is type(right) and left == right, (path, left, right)

    for key in a.entries:
        compare(a.entries[key], b.entries[key], str(key))


# The versioned fixture vocabulary is independent of the live reducer inventory.
_MOMENTS = (
    "n",
    "ref_y",
    "cy1",
    "cy2",
    "ref_x",
    "cx1",
    "cx2",
    "cxy",
    "ref_den",
    "cden1",
    "cden2",
    "cden3",
    "cyden",
    "cxden",
    "sum_d",
    "cyd",
    "cy2d",
    "cxd",
    "successes",
    "x_role",
    "winsor_lower_percentile",
    "winsor_upper_percentile",
    "winsor_lower_bound",
    "winsor_upper_bound",
    "winsor_n",
    "winsor_n_lower",
    "winsor_n_upper",
)


def _capture_key(scope, request, values, coordinates, estimator, method, metric, arm, row=None):
    row = row or {}
    by = getattr(request, "by", ()) or values.get("by", ())
    dimension = getattr(request, "dimension", values.get("dimension"))
    breakout = getattr(request, "breakout", None)
    if not by and breakout is not None:
        by = (breakout.property,)
    if dimension:
        by = (*by, dimension)
    source_name = values.get("source_name") or getattr(breakout, "source", None)
    segment = tuple(
        (
            f"{source_name}:{column}" if source_name else str(column),
            None if row.get(column) is None else str(row[column]),
        )
        for column in by
    )
    cell = getattr(request, "cell", values.get("cell"))
    if cell is not None:
        label = cell.breakout.property
        if cell.breakout.source is not None:
            label = f"{cell.breakout.source}:{label}"
        segment = ((label, None if cell.is_null else str(cell.value)),)
    day = row.get("ds", values.get("date", values.get("date_value")))
    coords = replace(coordinates, day=None if day is None else str(day), segment=segment)
    return CaptureKey(
        estimator,
        "" if method is None else method_identity(method),
        metric,
        scope.hash(),
        coords,
        arm,
    )


@dataclass
class _Capture:
    scopes: Sequence[Scope]
    expected: frozenset[CaptureKey]
    entries: list[tuple[CaptureKey, Any]] = field(default_factory=list)
    active: list[str] = field(default_factory=list)

    def add(self, key: CaptureKey, value: Any) -> None:
        assert all(previous != key for previous, _ in self.entries), f"Duplicate capture key: {key}"
        assert key in self.expected, {"missing": frozenset(), "unexpected": frozenset({key})}
        self.entries.append((key, value))

    def context(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        # Some assembly producers receive rows rather than a request. Their
        # immediate callers retain the metric/date and execute_views retains scope.
        values = dict(arguments)
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            while frame is not None:
                if frame.f_globals.get("__name__", "").startswith("increment."):
                    for name in ("scope", "metric", "date", "date_value", "methods", "request"):
                        if name not in values and name in frame.f_locals:
                            candidate = frame.f_locals[name]
                            if name != "scope" or isinstance(candidate, Scope):
                                values[name] = candidate
                frame = frame.f_back
        finally:
            del frame
        return values

    def record(self, name: str, result: Any, arguments: Mapping[str, Any]) -> Any:
        values = self.context(arguments)
        scope = values.get("scope")
        if not isinstance(scope, Scope):
            if len(self.scopes) > 1:
                for selected in self.scopes:
                    result = self.record(name, result, {**values, "scope": selected})
                return result
            assert self.scopes, f"Acquisition {name} has no selected scope"
            scope = self.scopes[0]
        request = values.get("request")
        population = getattr(request, "population", values.get("population", scope.population))
        grain = getattr(request, "grain", scope.grain)
        coordinates = RequestCoordinates(population, grain)
        metric = getattr(request, "metric", values.get("metric"))
        methods = getattr(request, "methods", values.get("methods", ()))
        method = getattr(request, "method", values.get("method"))
        if method is not None:
            methods = (method,)

        key = partial(_capture_key, scope, request, values, coordinates)

        if isinstance(result, MetricEvidence):
            start = len(self.entries)
            self.metric_records(name, result, metric, methods, key)
            self.check_metric_roster(self.entries[start:], metric, methods, key)
            return result
        if isinstance(result, DayAxisRows):

            def slices():
                for item in result.slices:
                    for row in item.rows:
                        metric_name = str(row.get("metric", item.metrics[0].name))
                        self.add(
                            key("day_axis", None, metric_name, str(row["group_id"]), row),
                            {
                                "moments": {field: row.get(field) for field in _MOMENTS},
                                "observation_edges": item.observation_edges,
                            },
                        )
                    yield item

            return replace(result, slices=slices())
        if name == "load_counts":
            assert request is not None
            kind = request.kind
            if kind in {"unit", "assignment"} and values["src"].context.cluster is None:
                kind = "arm_roster"
            for arm, count in result.counts.items():
                self.add(
                    key("counts." + kind, None, "", str(arm)),
                    {
                        "count": count,
                        "unit_count": result.unit_counts.get(arm, count),
                        "population": result.population,
                        "grain": "unit" if kind == "arm_roster" else result.grain,
                    },
                )
            return result
        if name in {"load_moments", "load_breakout_method_moments"}:
            assert metric is not None
            rows = result.rows if hasattr(result, "rows") else result
            for row in rows:
                self.add(
                    key(name, method, metric.name, str(row["group_id"]), row),
                    {field: row.get(field) for field in _MOMENTS},
                )
            for dimension, value, refusal, context in getattr(result, "refusals", ()):
                self.add(
                    key("refusal." + name, method, metric.name, "", {dimension: value}),
                    {"code": refusal.code, "context": context},
                )
            return result
        if name == "load_breakout_evidence":
            # Slices carry runtime source views; their numerical answers are
            # captured by the nested metric/day-axis producer, not serialized.
            if hasattr(result, "slices"):
                self.add(key(name, None, "", ""), [item.identity for item in result.slices])
            else:
                assert metric is not None
                import narwhals as nw

                frame = nw.from_native(result.frame, eager_only=True).to_dicts()
                self.add(key(name, None, metric.name, ""), sorted(frame, key=_encoded))
            return result
        if name == "load_unit_rows":
            import narwhals as nw

            result_payload = nw.from_native(result, eager_only=True).to_dicts()
        else:
            result_payload = result
        self.add(key(name, method, "" if metric is None else metric.name, ""), result_payload)
        return result

    def check_metric_roster(self, entries, metric, methods, key):
        actual = frozenset(item[0] for item in entries)
        estimator_ids = {method_estimator_id(method) or method.name for method in methods}
        estimator_ids |= {
            prefix + item for prefix in ("units.", "refusal.") for item in tuple(estimator_ids)
        }
        identities = {method_identity(method) for method in methods}
        coordinate = key("", None, metric.name, "").request_coordinates
        days = {item.request_coordinates.day for item in actual} or {coordinate.day}
        expected = frozenset(
            item
            for item in self.expected
            if item.metric == metric.name
            and item.method_identity in identities
            and item.estimator_id in estimator_ids
            and item.scope_hash == key("", None, metric.name, "").scope_hash
            and item.request_coordinates.population == coordinate.population
            and item.request_coordinates.grain == coordinate.grain
            and item.request_coordinates.day in days
        )
        assert actual == expected, {"missing": expected - actual, "unexpected": actual - expected}

    def metric_records(self, name, result, metric, methods, key):
        assert metric is not None and methods, f"Missing metric/method declaration at {name}"
        for method in methods:
            refused = any(label == method.name for label, _spec, _ctx in result.method_refusals)
            rows = (
                ()
                if refused
                else next(
                    (rows for label, rows in result.method_rows if label == method.name),
                    result.rows,
                )
            )
            for row in rows:
                payload = {"moments": {field: row.get(field) for field in _MOMENTS}}
                fits = [
                    answer
                    for fit_request, answer in result.method_fit_answers
                    if fit_request.method == method
                ]
                if fits:
                    payload["method_fits"] = fits
                for field_name in ("_winsor_raw_state", "_winsor_references"):
                    if field_name in row:
                        payload[field_name] = row[field_name]
                self.add(
                    key(
                        method_estimator_id(method) or method.name,
                        method,
                        metric.name,
                        str(row["group_id"]),
                        row,
                    ),
                    payload,
                )
            if result.unit_frame is not None:
                import narwhals as nw

                units = nw.from_native(result.unit_frame, eager_only=True).to_dicts()
                for arm in sorted(result.observed_arms):
                    selected = [row for row in units if str(row["group_id"]) == arm]
                    self.add(
                        key(
                            "units." + (method_estimator_id(method) or method.name),
                            method,
                            metric.name,
                            arm,
                        ),
                        sorted(selected, key=_encoded),
                    )
            for label, refusal, context in result.method_refusals:
                if label == method.name:
                    self.add(
                        key(
                            "refusal." + (method_estimator_id(method) or method.name),
                            method,
                            metric.name,
                            "",
                        ),
                        {"code": refusal.code, "context": context},
                    )
            for label, dimension, value, refusal, context in result.method_cell_refusals:
                if label == method.name:
                    self.add(
                        key(
                            "refusal." + (method_estimator_id(method) or method.name),
                            method,
                            metric.name,
                            "",
                            {dimension: value},
                        ),
                        {"code": refusal.code, "context": context},
                    )

    def wrapper(self, function):
        signature = inspect.signature(function)
        name = function.__name__

        @wraps(function)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            nested = bool(self.active)
            self.active.append(name)
            try:
                result = function(*args, **kwargs)
                # A terminal envelope already contains nested rows and fit answers.
                return result if nested else self.record(name, result, arguments)
            finally:
                self.active.pop()

        return wrapped


@contextmanager
def _wrapped_producers(capture: _Capture):
    originals = [getattr(importlib.import_module(module), name) for module, name in PRODUCERS]
    replacements = {original: capture.wrapper(original) for original in originals}

    def substitute(mapping):
        for name, value in tuple(mapping.items()):
            if inspect.isfunction(value) and value in replacements:
                mapping[name] = replacements[value]

    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith("increment."):
            substitute(vars(module))
    try:
        yield
    finally:
        replacements = {wrapped: original for original, wrapped in replacements.items()}
        # Also restore aliases imported lazily while the public run was active.
        for module in tuple(sys.modules.values()):
            if module is not None and getattr(module, "__name__", "").startswith("increment."):
                substitute(vars(module))


def capture_acquired_evidence(
    experiment: Experiment,
    resolver: Resolver,
    *,
    views: Sequence[Scope] | None = None,
    expected: frozenset[CaptureKey],
) -> EvidenceRecord:
    assert expected, "An evidence capture requires a nonempty expected roster"
    selected = experiment.scopes if views is None else views
    capture = _Capture(selected, expected)
    with _wrapped_producers(capture):
        run(experiment, resolver, views=views)
    record = EvidenceRecord(capture.entries)
    actual = frozenset(record.entries)
    assert actual == expected, {"missing": expected - actual, "unexpected": actual - expected}
    return record
