"""Capture stable readout identities, portable cubes, and a sequential checkpoint."""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ibis
import pandas as pd
import pyarrow.parquet as pq

from examples._seed import seed_event_log
from increment import Analysis, Method, MetricSpec
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
    _categorical_cuped_moments_case,
    _mixed_cuped_multicovariate_case,
    _sequential_asymptotic_mean_case,
)

_DUCKDB_CONNECT = ibis.duckdb.connect


def _connect_single_thread(**kwargs):
    kwargs["threads"] = 1
    return _DUCKDB_CONNECT(**kwargs)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    ).encode()


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
    frame = results.to_frame()
    rows = json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))
    for row in rows:
        advisories = row.get("advisories")
        if isinstance(advisories, str):
            row["advisories"] = [
                {key: notice.get(key) for key in ("code", "context", "severity")}
                for notice in json.loads(advisories)
            ]
        row.pop("failure_message", None)
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _views(analysis: Analysis) -> tuple[dict[str, Any], dict[str, Any]]:
    results = {
        "run": analysis.run(decision_method=Method(name="unadjusted")),
        "run_daily": analysis.run_daily(),
        "run_asof": analysis.run_asof(),
        "run_breakout": analysis.run_breakout(decision_method=Method(name="unadjusted")),
        "run_daily_lift": analysis.run_daily_lift(decision_method=Method(name="unadjusted")),
        "run_asof_lift": analysis.run_asof_lift(decision_method=Method(name="unadjusted")),
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


def _export(analysis: Analysis, path: Path) -> list[dict[str, Any]]:
    analysis.export(path)
    rows = pq.read_table(path).to_pylist()
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _metric_specs(analysis: Analysis) -> list[MetricSpec]:
    specs = []
    for config in analysis._src.context.configs:
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


def replay_export(template: Analysis, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay a format-12/13 cube using the source analysis configuration."""
    replay = Analysis.from_moments(
        payload,
        metrics=_metric_specs(template),
        design=template._src.context.design,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _frame_rows(replay.run())
    finally:
        replay.close()


def _replay_rows(template: Analysis, cube_path: Path) -> list[dict[str, Any]]:
    return replay_export(template, pq.read_table(cube_path).to_pylist())


@contextmanager
def _single_thread_duckdb():
    original = ibis.duckdb.connect
    ibis.duckdb.connect = _connect_single_thread
    try:
        yield
    finally:
        ibis.duckdb.connect = original


def build_analyses() -> dict[str, Analysis]:
    with _single_thread_duckdb():
        con = ibis.duckdb.connect(threads=1)
        seed_event_log(con)
        baseline = Analysis(EXPERIMENT, DEFINITIONS, con)
        mixed = _mixed_cuped_multicovariate_case()
        categorical = _categorical_cuped_moments_case()
        return {
            "parity_baseline": baseline,
            "mixed_local": mixed.build["from_unit_summary"](),
            "mixed_warehouse": mixed.build["from_definitions"](),
            "mixed_fitted_replay": mixed.build["from_moments"](),
            "mixed_categorical_replay": categorical.build["from_moments"](),
        }


def capture(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    exports = out / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "mixed_fitted_replay.parquet").unlink(missing_ok=True)
    readouts: dict[str, Any] = {}
    analyses = build_analyses()
    try:
        base = analyses.pop("parity_baseline")
        views, identities = _views(base)
        readouts["parity_baseline"] = {"views": views, "rows": identities}
        _export(base, exports / "parity_baseline.parquet")
        for name, analysis in analyses.items():
            readouts[name] = {"rows": {"run": _rows(analysis.run())}}
            if name == "mixed_warehouse":
                _export(analysis, exports / f"{name}.parquet")
        # The retained fitted replay uses the warehouse analysis's format-13 cube.
        with _single_thread_duckdb():
            categorical_export = _categorical_cuped_moments_case().build["from_definitions"]()
        try:
            _export(categorical_export, exports / "mixed_categorical_replay.parquet")
        finally:
            categorical_export.close()
        replay_cases = {
            "parity_baseline": base,
            "mixed_warehouse": analyses["mixed_warehouse"],
            "mixed_fitted_replay": analyses["mixed_fitted_replay"],
            "mixed_categorical_replay": analyses["mixed_categorical_replay"],
        }
        for name, template in replay_cases.items():
            cube_name = "mixed_warehouse" if name == "mixed_fitted_replay" else name
            readouts[name]["replay_rows"] = _replay_rows(template, exports / f"{cube_name}.parquet")
        _write_checkpoint(out / "checkpoint.json")
        (out / "readouts.json").write_bytes(_json_bytes(readouts))
    finally:
        for analysis in analyses.values():
            close = getattr(analysis, "close", None)
            if close is not None:
                close()
        base.close()


def _write_checkpoint(path: Path) -> None:
    case = _sequential_asymptotic_mean_case()
    analysis = case.build["from_unit_panel"]()
    try:
        with _single_thread_duckdb():
            con = ds.duckdb_connection()
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()
        as_of = analysis._sequential_as_of  # ty: ignore[unresolved-attribute]
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
        prefix = Analysis.from_unit_panel(
            before,
            unit="user_id",
            group="variant",
            date="date",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            design=analysis._src.context.design,
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        parent_snapshot = prefix.capture_sequential(finalized=True, as_of=as_of)
        with tempfile.TemporaryDirectory() as directory:
            cube_path = Path(directory) / "checkpoint.parquet"
            prefix.export(cube_path)
            checkpoint_rows = pq.read_table(cube_path).to_pylist()
        later_rows = sorted(after, key=lambda row: (str(row["user_id"]), str(row["date"])))
        continued_panel = pd.concat([before, pd.DataFrame(later_rows)], ignore_index=True)
        continued = Analysis.from_unit_panel(
            continued_panel,
            unit="user_id",
            group="variant",
            date="date",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            design=analysis._src.context.design,
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )
        try:
            continued_as_of = max(row["date"] for row in later_rows)
            continued_snapshot = continued.capture_sequential(
                finalized=True, previous=parent_snapshot, as_of=continued_as_of
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                continuation_rows = _frame_rows(continued.run_asof())
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
            continued.close()
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
        prefix.close()
    finally:
        analysis.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    capture(args.out)


if __name__ == "__main__":
    main()
