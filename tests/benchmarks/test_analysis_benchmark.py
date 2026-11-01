from __future__ import annotations

from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from increment import (
    AnalysisPlan,
    InferenceSpec,
    Method,
    MetricSpec,
    Scope,
    from_definitions,
    from_frame,
    from_unit_day_artifact,
)
from increment.artifacts import artifact_context, publish_unit_day_artifact
from increment.semantics.design import Randomized
from tests.analysis_factory import view_readout

# Generated frames and seeded DuckDB event logs are prepared before benchmarking.
_CASES = (
    ("dataframe-mean-10000", "dataframe", "mean", 10_000),
    ("dataframe-mean-10000000", "dataframe", "mean", 10_000_000),
    ("dataframe-ratio-10000", "dataframe", "ratio", 10_000),
    ("dataframe-sequential-10000", "dataframe", "sequential", 10_000),
    ("definitions-mean", "definitions", "mean", 0),
    ("definitions-conversion", "definitions", "conversion", 0),
    ("definitions-retention", "definitions", "retention", 0),
    ("definitions-all-metrics", "definitions", "all", 0),
    ("definitions-breakout", "definitions", "breakout", 0),
    ("definitions-daily", "definitions", "daily", 0),
    ("definitions-asof", "definitions", "asof", 0),
    ("definitions-daily-lift", "definitions", "daily_lift", 0),
    ("definitions-asof-lift", "definitions", "asof_lift", 0),
    ("definitions-cuped", "definitions", "cuped", 0),
    ("definitions-windowed-conversion", "definitions", "windowed_conversion", 0),
    ("definitions-windowed-mean", "definitions", "windowed_mean", 0),
    ("definitions-breakout-lift", "definitions", "breakout_lift", 0),
    ("definitions-daily-conversion", "definitions", "daily_conversion", 0),
    ("definitions-daily-mean", "definitions", "daily_mean", 0),
    ("definitions-asof-conversion", "definitions", "asof_conversion", 0),
    ("artifact-readout", "artifact", "run", 0),
    ("artifact-daily", "artifact", "daily", 0),
    ("definitions-cuped-readout", "definitions", "cuped_readout", 0),
    ("definitions-mean-readout", "definitions", "mean_readout", 0),
)


def _dataframe_analysis(
    rows: int, kind: str, *, frame: pd.DataFrame | None = None
) -> Callable[[], object]:
    if frame is None:
        index = np.arange(rows, dtype=np.int64)
        frame = pd.DataFrame(
            {
                "unit": index,
                "arm": np.where(index % 2, "treatment", "control"),
                "outcome": (index % 17).astype(np.float64) + 1.0,
                "denominator": (index % 5).astype(np.float64) + 1.0,
                "exposure_day": index,
            }
        )
    if kind == "ratio":
        metrics = [
            MetricSpec(
                name="revenue_per_session",
                type="ratio",
                numerator="outcome",
                denominator="denominator",
            )
        ]
        plan = None
    else:
        metrics = {"outcome": "mean"}
        plan = (
            AnalysisPlan(
                primary="outcome",
                inference=InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=rows),
            )
            if kind == "sequential"
            else None
        )

    def execute() -> object:
        experiment, resolver = from_frame(
            frame,
            shape="summary",
            unit="unit",
            group="arm",
            metrics=metrics,
            plan=plan,
            exposure_date="exposure_day" if kind == "sequential" else None,
            design=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
        )
        with resolver:
            return view_readout(experiment, resolver)

    return execute


def _population_identity_hashing() -> Callable[[], object]:
    from scripts.capture_evidence_identity import population_digest

    ids = np.arange(10_000_000, dtype=np.int64)

    def execute() -> tuple[tuple[str, int], tuple[str, int]]:
        return (
            population_digest(ids[::2], population="assigned", arm="control"),
            population_digest(ids[1::2], population="assigned", arm="treatment"),
        )

    return execute


def test_dataframe_benchmark_callable_recomputes_from_mutated_frame() -> None:
    frame = pd.DataFrame(
        {
            "unit": list(range(8)),
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "outcome": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
            "denominator": [1.0] * 8,
            "exposure_day": list(range(8)),
        }
    )
    execute = _dataframe_analysis(len(frame), "mean", frame=frame)
    first = cast(Any, execute())[0].require_lift().value
    frame.loc[frame["arm"] == "treatment", "outcome"] *= 2
    second = cast(Any, execute())[0].require_lift().value
    assert first != second


def _readout(experiment: Any, resolver: Any, operation: str) -> Callable[[], object]:
    match operation:
        case "mean":
            return lambda: view_readout(experiment, resolver, metrics=["avg_session_duration"])
        case "conversion":
            return lambda: view_readout(experiment, resolver, metrics=["purchase_rate"])
        case "retention":
            return lambda: view_readout(experiment, resolver, metrics=["d7_retention"])
        case "all" | "run":
            return lambda: view_readout(experiment, resolver)
        case "breakout" | "breakout_lift":
            return lambda: view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
        case "daily" | "daily_conversion" | "daily_mean":
            metric = {
                "daily_conversion": ["purchase_rate"],
                "daily_mean": ["avg_session_duration"],
            }.get(operation)
            return lambda: view_readout(
                experiment,
                resolver,
                Scope(view="daily", grain="daily", values=True),
                metrics=metric,
            )
        case "asof" | "asof_conversion":
            metric = ["purchase_rate"] if operation == "asof_conversion" else None
            return lambda: view_readout(
                experiment, resolver, Scope(view="asof", grain="asof", values=True), metrics=metric
            )
        case "daily_lift":
            return lambda: view_readout(experiment, resolver, Scope(view="daily", grain="daily"))
        case "asof_lift":
            return lambda: view_readout(experiment, resolver, Scope(view="asof", grain="asof"))
        case "cuped" | "cuped_readout":
            return lambda: view_readout(experiment, resolver, sensitivity_methods=(Method.cuped(),))
        case "windowed_conversion":
            return lambda: view_readout(experiment, resolver, metrics=["purchase_rate"])
        case "windowed_mean" | "mean_readout":
            return lambda: view_readout(experiment, resolver, metrics=["avg_session_duration"])
        case other:
            raise AssertionError(other)


def _prepare(
    case: str,
    route: str,
    operation: str,
    rows: int,
    seeded_con,
    seeded_pre_period_con,
    definitions: Path,
):
    if route == "dataframe":
        return _dataframe_analysis(rows, operation)
    con = seeded_pre_period_con if operation in ("cuped", "cuped_readout") else seeded_con
    artifact_counter = count()

    def execute() -> object:
        experiment, resolver = cast(
            "tuple[Any, Any]",
            from_definitions(definitions, con, experiment_name="new_onboarding_v2"),
        )
        with resolver:
            if route != "artifact":
                return _readout(experiment, resolver, operation)()
            from increment.query.session import WarehouseArtifactStore

            store = WarehouseArtifactStore(
                con,
                schema_name=f"bench_{case.replace('-', '_')}_{next(artifact_counter)}",
            )
            ref = publish_unit_day_artifact(experiment, resolver, store)
            artifact_experiment, artifact_resolver = from_unit_day_artifact(
                store, ref, expected_context=artifact_context(experiment, resolver)
            )
            with artifact_resolver:
                return _readout(artifact_experiment, artifact_resolver, operation)()

    return execute


@pytest.mark.filterwarnings(
    "ignore:metrics .* have available CUPED covariates that the default decision method does not use:"
    "increment.errors.IncrementWarning"
)
@pytest.mark.integration
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("case", "route", "operation", "rows"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_analysis_end_to_end_benchmark(
    benchmark,
    case: str,
    route: str,
    operation: str,
    rows: int,
    seeded_con,
    seeded_pre_period_con,
    example_definitions: Path,
) -> None:
    execute = _prepare(
        case, route, operation, rows, seeded_con, seeded_pre_period_con, example_definitions
    )
    result = benchmark(execute)
    assert result


@pytest.mark.integration
@pytest.mark.benchmark
def test_population_identity_hashing_benchmark(benchmark) -> None:
    result = benchmark(_population_identity_hashing())
    assert result[0][1] == result[1][1] == 5_000_000
