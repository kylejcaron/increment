"""Capture stable readout identities, portable cubes, and a sequential checkpoint."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from itertools import count
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import ibis
import pandas as pd
import pyarrow.parquet as pq

from examples._seed import seed_event_log
from increment import Method, MetricSpec, builders, workflows
from increment._binding import binding_for
from increment.semantics.models import AnalysisPlan
from scripts.capture_parity_baseline import (
    DEFINITIONS,
    EXPERIMENT,
    _arm_rows,
    _day,
    _lift_value,
    _value_of,
)
from tests.parity_harness import dataset as ds
from tests.parity_harness.cases import (
    Bound,
    _categorical_cuped_moments_case,
    _close_parity_binding,
    _mixed_cuped_multicovariate_case,
    _sequential_asymptotic_mean_case,
    sequential_as_of,
)
from tests.parity_harness.views import read

_DUCKDB_CONNECT = ibis.duckdb.connect


def _connect_single_thread(**kwargs):
    kwargs["threads"] = 1
    return _DUCKDB_CONNECT(**kwargs)


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if isoformat is not None else str(value)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n"
    ).encode()


def _stable_capture_value(value: Any) -> Any:
    """Normalize floating-point noise to precision within the oracle tolerance."""
    if isinstance(value, float):
        return value if value == 0.0 else float(format(value, ".10g"))
    if isinstance(value, Mapping):
        return {key: _stable_capture_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_capture_value(item) for item in value]
    return value


def _canonical_json_bytes(value: Any, *, normalize: bool = True) -> bytes:
    canonical = _stable_capture_value(value) if normalize else value
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _payload_reference(
    payloads: dict[str, Any],
    value: Any,
    *,
    exact: bool = False,
    exact_payload_hashes: set[str] | None = None,
) -> dict[str, Any]:
    payload = value
    reference: dict[str, Any] = {}
    if isinstance(value, dict) and "provenance" in value and "identity" in value:
        payload = {key: item for key, item in value.items() if key != "provenance"}
        reference["provenance"] = value["provenance"]
    digest = hashlib.sha256(_canonical_json_bytes(payload, normalize=not exact)).hexdigest()
    payloads.setdefault(digest, payload)
    if exact and exact_payload_hashes is not None:
        exact_payload_hashes.add(digest)
    return {"payload_hash": digest, **reference}


def load_operations_capture(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    payloads = data.pop("payloads")
    for digest, payload in payloads.items():
        stable_digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        exact_digest = hashlib.sha256(_canonical_json_bytes(payload, normalize=False)).hexdigest()
        if digest not in {stable_digest, exact_digest}:
            raise ValueError(f"content-addressed payload hash mismatch: {digest}")

    def expand(value: Any) -> Any:
        if isinstance(value, dict):
            if "payload_hash" in value:
                raw_payload = payloads[value["payload_hash"]]
                payload = dict(raw_payload) if isinstance(raw_payload, dict) else raw_payload
                if "provenance" in value:
                    assert isinstance(payload, dict)
                    payload["provenance"] = value["provenance"]
                extras = {
                    key: expand(item)
                    for key, item in value.items()
                    if key not in {"payload_hash", "provenance"}
                }
                if "operations" in extras:
                    return {"result_cell": payload, **extras}
                if "status" in extras:
                    return {"status": extras["status"], "result_cell": payload}
                return payload
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    data["scenarios"] = expand(data["scenarios"])
    return data


def _population_identity(bound: Bound, case: Any, cells: Sequence[Any]) -> dict[str, Any]:
    import narwhals as nw

    from increment._execute import readout_source
    from increment.cate import _cate_source
    from increment.errors import CodedError, InvalidRequestError

    binding = binding_for(*bound)
    populations = {cell.identity.key.analysis_population for cell in cells} or {case.population}
    by_population: dict[str, dict[str, set[str]]] = {}
    for population in sorted(populations):
        try:
            source = readout_source(binding, population=population)
            if population == "triggered":
                source = cast(Any, _cate_source(binding, "capture population identity"))
        except (CodedError, InvalidRequestError):
            continue
        if not hasattr(source, "unit_frame"):
            continue
        cell_metrics = {cell.identity.key.metric for cell in cells}
        if not cell_metrics:
            cell_metrics = {config.metric.name for config in source.context.configs}
        by_arm: dict[str, set[str]] = {}
        for config in source.context.configs:
            metric = config.metric
            if metric.name not in cell_metrics:
                continue
            try:
                frame = nw.from_native(source.unit_frame(metric), eager_only=True)
                for row in frame.select("unit_id", "group_id").iter_rows(named=True):
                    by_arm.setdefault(str(row["group_id"]), set()).add(str(row["unit_id"]))
            except (CodedError, InvalidRequestError, KeyError, ValueError, TypeError):
                continue
        if by_arm:
            by_population[population] = by_arm
    if not by_population:
        return {"status": "unavailable", "reason": "source retains aggregate moments only"}
    return population_identity_from_ids(
        {
            population: {arm: tuple(unit_ids) for arm, unit_ids in by_arm.items()}
            for population, by_arm in by_population.items()
        }
    )


def population_digest(unit_ids: Iterable[object], *, population: str, arm: str) -> tuple[str, int]:
    """Hash canonical sorted identity ids without delimiter ambiguity."""
    del population, arm
    normalized = sorted(str(unit_id) for unit_id in unit_ids)
    payload = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(normalized)


def population_identity_from_ids(
    units: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        population: {
            arm: {
                "digest": population_digest(ids, population=population, arm=arm)[0],
                "unit_count": len(ids),
            }
            for arm, ids in sorted(arms.items())
        }
        for population, arms in sorted(units.items())
    }


def iter_capture_keys() -> tuple[tuple[str, str], ...]:
    """Enumerate parity cases and refusal owners at their declared ingresses."""
    from tests.capability_contracts import CONSTRUCTORS
    from tests.parity_harness.cases import PARITY_CASES
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS

    return (
        *((case.id, ingress) for case in PARITY_CASES for ingress in CONSTRUCTORS),
        *((code, "refusal") for code in REFUSAL_SCENARIOS),
    )


def _identity(results, row) -> dict[str, Any]:
    source = results.source
    components = source.get("components", []) if isinstance(source, dict) else []
    families = getattr(getattr(results, "metadata", None), "scope", None)
    decision_id = None
    if families is not None:
        decision_id = next(
            (
                family.decision_snapshot_id
                for family in families.families
                if row.source_snapshot_id == family.source_snapshot_id
                and family.decision_snapshot_id is not None
            ),
            None,
        )
    return {
        "source_snapshot_id": row.source_snapshot_id,
        "evidence_components": components,
        "decision_evidence_components": components if decision_id is not None else [],
    }


def _rows(results) -> list[dict[str, Any]]:
    rows = []
    identity_keys = (
        "metric",
        "group_id",
        "method",
        "estimand",
        "ds",
        "dimension",
        "dimension_value",
    )
    for row in results:
        dumped = row.model_dump(mode="json")
        item = {
            "row": {key: dumped[key] for key in identity_keys if key in dumped},
            **_identity(results, row),
        }
        rows.append(item)
    return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))


def _frame_rows(results) -> list[dict[str, Any]]:
    """Serialize full result models alongside their presentation-derived fields."""
    frame = results.to_frame()
    rendered = json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))
    rows = []
    for result, projection in zip(results, rendered, strict=True):
        row = result.model_dump(mode="json")
        row.update(projection)
        advisories = row.get("advisories")
        if isinstance(advisories, str):
            row["advisories"] = [
                {key: notice.get(key) for key in ("code", "context", "severity")}
                for notice in json.loads(advisories)
            ]
        row.pop("failure_message", None)
        # These capture/result envelope fields postdate the frozen replay fixture;
        # all pre-existing result fields remain compared below.
        for key in ("binomial_set", "coherence", "effective_modes", "provider_owned"):
            row.pop(key, None)
        rows.append(row)
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _views(bound: Bound) -> tuple[dict[str, Any], dict[str, Any]]:
    unadjusted = Method(name="unadjusted")
    results = {
        "run": read(bound, decision_method=unadjusted),
        "run_daily": read(bound, "daily"),
        "run_asof": read(bound, "asof"),
        "run_breakout": read(bound, "breakout", decision_method=unadjusted),
        "run_daily_lift": read(bound, "daily_lift", decision_method=unadjusted),
        "run_asof_lift": read(bound, "asof_lift", decision_method=unadjusted),
    }
    values = {
        "run": {
            f"{row.metric}|{row.group_id}": _lift_value(row) for row in _arm_rows(results["run"])
        },
        "run_daily": {
            f"{row.metric}|{row.group_id}|{_day(row.ds)}": _value_of(row)
            for row in results["run_daily"]
        },
        "run_asof": {
            f"{row.metric}|{row.group_id}|{_day(row.ds)}": _value_of(row)
            for row in results["run_asof"]
        },
        "run_breakout": {
            f"{row.metric}|{row.group_id}|{row.dimension}={row.dimension_value}": _lift_value(row)
            for row in results["run_breakout"]
        },
        "run_daily_lift": {
            f"{row.metric}|{row.group_id}|{_day(row.ds)}|{row.estimand}": _lift_value(row)
            for row in results["run_daily_lift"]
        },
        "run_asof_lift": {
            f"{row.metric}|{row.group_id}|{_day(row.ds)}|{row.estimand}": _lift_value(row)
            for row in results["run_asof_lift"]
        },
    }
    identities = {name: _rows(result) for name, result in results.items()}
    return values, identities


def _stable_export_value(value: Any) -> Any:
    if isinstance(value, float):
        if abs(value) < 1e-10:
            return 0
        return int(value) if value.is_integer() else float(format(value, ".9g"))
    if isinstance(value, dict):
        return {
            key: (
                sorted(_stable_export_value(item), key=str)
                if key == "denominator_support"
                else _stable_export_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_stable_export_value(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            return json.dumps(
                _stable_export_value(json.loads(value)), sort_keys=True, separators=(",", ":")
            )
        except json.JSONDecodeError:
            return value
    return value


def _export(bound: Bound, path: Path) -> list[dict[str, Any]]:
    workflows.export_moments(*bound, path)
    rows = pq.read_table(path).to_pylist()
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _metric_specs(bound: Bound) -> list[MetricSpec]:
    specs = []
    for config in binding_for(*bound).source.context.configs:
        metric = config.metric
        values = {
            "name": metric.name,
            "type": metric.type,
            "decision_method": config.decision_method,
            "sensitivity_methods": config.sensitivity_methods,
        }
        if metric.type == "ratio":
            values["numerator"] = metric.numerator.fact
            values["denominator"] = metric.denominator.fact
        winsorization = getattr(metric, "winsorization", None)
        if winsorization is not None:
            values["winsorization"] = winsorization
        preferred_direction = getattr(metric, "preferred_direction", None)
        if preferred_direction is not None:
            values["preferred_direction"] = preferred_direction
        threshold_days = getattr(metric, "threshold_days", None)
        if threshold_days is not None:
            values["threshold_days"] = threshold_days
        specs.append(MetricSpec.model_validate(values))
    return specs


def replay_export(template: Bound, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay a format-12/13 cube using the source experiment's configuration."""
    replay = builders.from_moments(
        payload,
        metrics=_metric_specs(template),
        design=binding_for(*template).source.context.design,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _frame_rows(read(replay))
    finally:
        replay[1].close()


def _replay_rows(template: Bound, cube_path: Path) -> list[dict[str, Any]]:
    return replay_export(template, pq.read_table(cube_path).to_pylist())


@contextmanager
def _single_thread_duckdb():
    original = ibis.duckdb.connect
    ibis.duckdb.connect = _connect_single_thread
    try:
        yield
    finally:
        ibis.duckdb.connect = original


def build_pairs() -> dict[str, Bound]:
    import shutil

    from tests.parity_harness.cases import _track_cleanup

    # The definitions builder records its catalog beside the definitions, so it reads a copy
    # that lives as long as the baseline pair rather than the checked-in examples.
    directory = tempfile.TemporaryDirectory(prefix="increment-identity-")
    definitions = Path(directory.name) / "definitions"
    shutil.copytree(DEFINITIONS, definitions)
    with _single_thread_duckdb():
        con = ibis.duckdb.connect(threads=1)
        seed_event_log(con)
        try:
            baseline = builders.from_definitions(definitions, con, experiment_name=EXPERIMENT)
        except BaseException:
            directory.cleanup()
            raise
        _track_cleanup(baseline, directory.cleanup)
        mixed = _mixed_cuped_multicovariate_case()
        categorical = _categorical_cuped_moments_case()
        return {
            "parity_baseline": baseline,
            "mixed_local": mixed.build["from_unit_summary"](),
            "mixed_warehouse": mixed.build["from_definitions"](),
            "mixed_fitted_replay": mixed.build["from_moments"](),
            "mixed_categorical_replay": categorical.build["from_moments"](),
        }


def _aggregate_evidence(
    payloads: dict[str, Any],
    rows: Sequence[Mapping[str, object]],
    *,
    exact_payload_hashes: set[str] | None = None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    versions = {row.get("moments_format") for row in rows}
    if len(versions) != 1:
        raise ValueError(f"aggregate moments have inconsistent format versions: {versions}")
    from increment.sources import source_sha256

    for row in rows:
        evidence = row.get("cuped_evidence")
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        supplied_digest = evidence.get("source_sha256") if isinstance(evidence, Mapping) else None
        if supplied_digest is None:
            continue
        source_identity = row.get("source_identity")
        identity = (
            json.loads(source_identity) if isinstance(source_identity, str) else source_identity
        )
        if (
            not isinstance(identity, Mapping)
            or source_sha256(cast(Mapping[str, object], identity)) != supplied_digest
        ):
            raise ValueError("aggregate evidence source digest mismatch")
    return {
        "status": "available",
        "moments_format": next(iter(versions)),
        "moments": _payload_reference(
            payloads,
            [dict(row) for row in rows],
            exact=True,
            exact_payload_hashes=exact_payload_hashes,
        ),
    }


def _capture_operations(cutover: str) -> dict[str, Any]:  # noqa: PLR0915
    from increment.errors import CodedError, InvalidRequestError
    from increment.result_envelope import AnalysisEnvelope, relevel
    from tests.parity_harness.cases import PARITY_CASES, _capture_kwargs
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS, Unreachable
    from tests.parity_harness.runner import _read

    cases = {case.id: case for case in PARITY_CASES}
    scenarios: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    exact_payload_hashes: set[str] = set()
    for scenario_id, ingress in iter_capture_keys():
        record = scenarios.setdefault(scenario_id, {})
        if ingress == "refusal":
            scenario = REFUSAL_SCENARIOS[scenario_id]
            if isinstance(scenario, Unreachable):
                record[ingress] = {"kind": "absent", "reason": scenario.reason}
                continue
            try:
                scenario()
            except CodedError as error:
                record[ingress] = {
                    "kind": "refusal",
                    "code": error.code,
                    "stage": "run",
                    "context": dict(error.context),
                }
            else:
                raise AssertionError(f"refusal scenario did not refuse: {scenario_id}")
            continue
        case = cases[scenario_id]
        if ingress not in case.build:
            record[ingress] = {
                "kind": "absent",
                "reason": case.waive.get(ingress, "constructor ingress is not declared"),
            }
            continue
        aggregate_rows: list[dict[str, object]] = []
        try:
            if ingress == "from_moments":
                original_from_moments = builders.from_moments

                def capture_moments(
                    rows: Sequence[Mapping[str, object]],
                    _aggregate_rows=aggregate_rows,
                    _original=original_from_moments,
                    **kwargs,
                ) -> Bound:
                    _aggregate_rows.extend(copy.deepcopy(dict(row)) for row in rows)
                    return _original(rows, **kwargs)

                with patch.object(builders, "from_moments", capture_moments):
                    bound = case.build[ingress]()
            else:
                bound = case.build[ingress]()
        except CodedError as error:
            record[ingress] = {
                "kind": "refusal",
                "code": error.code,
                "stage": "construct",
                "context": dict(error.context),
            }
            if aggregate := _aggregate_evidence(
                payloads, aggregate_rows, exact_payload_hashes=exact_payload_hashes
            ):
                record[ingress]["aggregate_evidence"] = aggregate
            continue
        try:
            if case.sequential:
                workflows.capture_sequential(*bound, finalized=True, **_capture_kwargs(bound))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = _read(bound, ingress, case)
            metadata_value = getattr(results, "metadata", None)
            daily_values = []
            cells = []
            for row in results:
                if hasattr(row, "as_cells"):
                    # A fixed stamp keeps the capture byte-stable; provenance is
                    # recorded, never compared.
                    cells.extend(
                        row.as_cells(
                            metadata_value,
                            created_at=datetime(2026, 1, 1, tzinfo=UTC),
                        )
                    )
                else:
                    daily_values.append(row.model_dump(mode="json"))
            population_identity = _population_identity(bound, case, cells)
        except CodedError as error:
            record[ingress] = {
                "kind": "refusal",
                "code": error.code,
                "stage": "run",
                "context": dict(error.context),
            }
            if aggregate := _aggregate_evidence(
                payloads, aggregate_rows, exact_payload_hashes=exact_payload_hashes
            ):
                record[ingress]["aggregate_evidence"] = aggregate
            continue
        finally:
            _close_parity_binding(bound)
        by_hash: dict[str, Any] = {}
        for cell in cells:
            key = cell.identity.hash()
            if key in by_hash:
                raise ValueError(f"duplicate cell identity hash {key} for {scenario_id}/{ingress}")
            by_hash[key] = _payload_reference(
                payloads,
                cell.to_payload(),
                exact=True,
                exact_payload_hashes=exact_payload_hashes,
            )
            by_hash[key] = {
                **by_hash[key],
                "operations": {
                    "relevel": {
                        str(alpha): _operation_result(
                            payloads,
                            lambda cell=cell, alpha=alpha: relevel(cell, alpha=alpha),
                            exact_payload_hashes,
                        )
                        for alpha in (0.01, 0.05)
                    },
                },
            }
        family_selection = {"status": "unavailable"}
        if cells:
            try:
                from increment.estimation.family import select_exploratory_family

                selected = select_exploratory_family(list(results), q=0.05)
                family_selection = {
                    "status": "available",
                    "q": 0.05,
                    "results": [row.model_dump(mode="json") for row in selected],
                }
            except (CodedError, InvalidRequestError) as error:
                family_selection = {
                    "status": "refusal",
                    "code": error.code,
                    "context": dict(error.context),
                }
        record[ingress] = {
            "kind": "cells",
            "status": "captured",
            "cells": by_hash,
            "daily_values": _payload_reference(payloads, daily_values) if daily_values else [],
            "family_selection": _payload_reference(payloads, family_selection),
            "metadata": _payload_reference(
                payloads,
                AnalysisEnvelope(tuple(cells), getattr(results, "metadata", None)).to_payload()[
                    "metadata"
                ],
            ),
            "population_identity": population_identity,
        }
        if aggregate := _aggregate_evidence(
            payloads, aggregate_rows, exact_payload_hashes=exact_payload_hashes
        ):
            record[ingress]["aggregate_evidence"] = aggregate
    return {
        "schema_version": 1,
        "cutover": cutover,
        "scenarios": scenarios,
        "payloads": payloads,
        "_exact_payload_hashes": sorted(exact_payload_hashes),
    }


def _operation_result(
    payloads: dict[str, Any], operation, exact_payload_hashes: set[str]
) -> dict[str, Any]:
    from increment.errors import CodedError

    try:
        cell = operation()
    except CodedError as error:
        return {"status": "refusal", "code": error.code, "context": dict(error.context)}
    return {
        "status": "available",
        **_payload_reference(
            payloads,
            cell.to_payload(),
            exact=True,
            exact_payload_hashes=exact_payload_hashes,
        ),
    }


@contextmanager
def _deterministic_capture_runtime() -> Iterator[None]:
    counter = count(1)
    fixed_stamp = datetime(2026, 1, 1, tzinfo=UTC)
    with (
        patch("uuid.uuid4", side_effect=lambda: UUID(int=next(counter), version=4)),
        patch("increment.query.artifact_publish._utc_now", return_value=fixed_stamp),
    ):
        yield


def capture(out: Path, *, cutover: str | None = None) -> None:
    if cutover is not None:
        target = out / cutover / "operations.json.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        with _single_thread_duckdb(), _deterministic_capture_runtime():
            operations = _capture_operations(cutover)
        compressed = BytesIO()
        exact_hashes = operations.pop("_exact_payload_hashes", ())
        normalized = _stable_capture_value(operations)
        for digest in exact_hashes:
            normalized["payloads"][digest] = operations["payloads"][digest]
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
            stream.write(_json_bytes(normalized))
        target.write_bytes(compressed.getvalue())
        return
    out.mkdir(parents=True, exist_ok=True)
    exports = out / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "mixed_fitted_replay.parquet").unlink(missing_ok=True)
    readouts: dict[str, Any] = {}
    pairs = build_pairs()
    base = pairs.pop("parity_baseline")
    try:
        views, identities = _views(base)
        readouts["parity_baseline"] = {"views": views, "rows": identities}
        _export(base, exports / "parity_baseline.parquet")
        for name, pair in pairs.items():
            readouts[name] = {"rows": {"run": _rows(read(pair))}}
            if name == "mixed_warehouse":
                _export(pair, exports / f"{name}.parquet")
        # The retained fitted replay uses the warehouse experiment's format-13 cube.
        with _single_thread_duckdb():
            categorical_export = _categorical_cuped_moments_case().build["from_definitions"]()
        try:
            _export(categorical_export, exports / "mixed_categorical_replay.parquet")
        finally:
            _close_parity_binding(categorical_export)
        replay_cases = {
            "parity_baseline": base,
            "mixed_warehouse": pairs["mixed_warehouse"],
            "mixed_fitted_replay": pairs["mixed_fitted_replay"],
            "mixed_categorical_replay": pairs["mixed_categorical_replay"],
        }
        for name, template in replay_cases.items():
            cube_name = "mixed_warehouse" if name == "mixed_fitted_replay" else name
            readouts[name]["replay_rows"] = _replay_rows(template, exports / f"{cube_name}.parquet")
        _write_checkpoint(out / "checkpoint.json")
        (out / "readouts.json").write_bytes(_json_bytes(readouts))
    finally:
        for pair in pairs.values():
            _close_parity_binding(pair)
        _close_parity_binding(base)


def _write_checkpoint(path: Path) -> None:
    case = _sequential_asymptotic_mean_case()
    bound = case.build["from_unit_panel"]()
    try:
        with _single_thread_duckdb():
            con = ds.duckdb_connection()
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()
        as_of = sequential_as_of(bound)
        assert as_of is not None
        design = binding_for(*bound).source.context.design
        before = panel.loc[panel["date"] <= as_of]
        after = panel.loc[panel["date"] > as_of].to_dict(orient="records")
        later_date = max(row["date"] for row in after)
        after.extend(
            [
                {
                    "user_id": "later_control",
                    "variant": "control",
                    "exposure_date": later_date,
                    "date": later_date,
                    "revenue": 4.0,
                    "sessions": 1,
                },
                {
                    "user_id": "later_treatment",
                    "variant": "treatment",
                    "exposure_date": later_date,
                    "date": later_date,
                    "revenue": 7.0,
                    "sessions": 1,
                },
            ]
        )
        plan = AnalysisPlan(
            primary="revenue",
            inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
        )
        prefix = builders.from_frame(
            before,
            shape="panel",
            unit="user_id",
            group="variant",
            date="date",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            design=design,
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        parent_snapshot = workflows.capture_sequential(*prefix, finalized=True, as_of=as_of)
        with tempfile.TemporaryDirectory() as directory:
            cube_path = Path(directory) / "checkpoint.parquet"
            workflows.export_moments(*prefix, cube_path)
            checkpoint_rows = pq.read_table(cube_path).to_pylist()
        later_rows = sorted(after, key=lambda row: (str(row["user_id"]), str(row["date"])))
        continued_panel = pd.concat([before, pd.DataFrame(later_rows)], ignore_index=True)
        continued = builders.from_frame(
            continued_panel,
            shape="panel",
            unit="user_id",
            group="variant",
            date="date",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            design=design,
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        try:
            continued_as_of = max(row["date"] for row in later_rows)
            continued_snapshot = workflows.capture_sequential(
                *continued, finalized=True, previous=parent_snapshot, as_of=continued_as_of
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                continuation_rows = _frame_rows(read(continued, "asof"))
            continuation = {
                "registration_id": continued_snapshot.registration_id,
                "prefix_id": continued_snapshot.prefix_id,
                "parent_id": continued_snapshot.parent_id,
                "ancestors": [
                    item.model_dump(mode="json") for item in continued_snapshot.ancestors
                ],
                "records": [item.model_dump(mode="json") for item in continued_snapshot.records],
                "states": [item.model_dump(mode="json") for item in continued_snapshot.states],
                "as_of": continued_as_of.isoformat(),
                "rows": continuation_rows,
            }
        finally:
            continued[1].close()
        payload = {
            "checkpoint": sorted(
                (_stable_export_value(row) for row in checkpoint_rows),
                key=lambda row: json.dumps(row, sort_keys=True, default=str),
            ),
            "later_rows": later_rows,
            "as_of": as_of.isoformat(),
            "continuation": continuation,
        }
        path.write_bytes(_json_bytes(payload))
        prefix[1].close()
    finally:
        _close_parity_binding(bound)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutover")
    args = parser.parse_args()
    capture(args.out, cutover=args.cutover)


if __name__ == "__main__":
    main()
