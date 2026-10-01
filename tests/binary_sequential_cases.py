"""One per-unit binary dataset read through the dataframe oracle and every other path.

``unit_rows``, ``frame_analysis`` and ``axis_run`` take the same axis keywords,
each naming one method axis the way a caller would request it: ``cluster``
(the randomization-grain column), ``cuped`` (a pre-period covariate column and
a CUPED decision method), ``pre_period`` (a coefficient and centre fixed before
the experiment, on ``InferenceSpec.adjustments``), ``winsor_percentile`` (a
percentile clip), ``metric_type`` (the primary's type), ``secondaries`` (a count
of secondary conversion metrics), ``breakout`` (a segment column read through
``run_breakout``) and ``switchback`` (the switchback panel constructor).

``check_path_parity`` is the one path-parity assertion for both automatic
binary routes: the unit panel, the definitions reader, the unit-day artifact
reader and a replayed portable checkpoint must retain the unit-summary
oracle's state, registration and interval byte for byte. The DuckDB suites
and the live-backend gate call it on the same per-unit data.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from increment import Analysis, Method, MetricSpec
from increment._metric_specs import MetricSpecType
from increment.query.session import WarehouseArtifactStore
from increment.semantics.artifact import ArtifactContext, UnitDayArtifactRef
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, InferenceSpec, Winsorization
from increment.semantics.sequential import PredeclaredAdjustment

DESIGN = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
AS_OF = dt.date(2025, 1, 14)
_EXPOSURE = dt.datetime(2025, 1, 10, 8, tzinfo=dt.UTC)
_PURCHASE = dt.datetime(2025, 1, 11, 9, tzinfo=dt.UTC)


class Route(NamedTuple):
    """One automatic binary route: its frame declaration, its YAML plan and its law."""

    inference: InferenceSpec
    plan_yaml: str
    law: str


ROUTES: dict[str, Route] = {
    "asymptotic_mean": Route(
        InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=400),
        "      primary: purchase\n"
        "      inference: {kind: asymptotic_mean, expected_decision_sample_size: 400}\n",
        "scalar_mean",
    ),
    "always_valid": Route(
        InferenceSpec(kind="always_valid", baseline_rate=Fraction(3, 10)),
        "      primary: purchase\n      inference: {kind: always_valid, baseline_rate: 0.3}\n",
        "bernoulli",
    ),
}
AXES = frozenset(
    {
        "cluster",
        "cuped",
        "pre_period",
        "winsor_percentile",
        "metric_type",
        "secondaries",
        "breakout",
        "segments",
        "switchback",
    }
)


def _check_axes(axes: dict[str, object]) -> None:
    unknown = sorted(set(axes) - AXES)
    if unknown:
        raise TypeError(f"unknown method axes {unknown}; expected a subset of {sorted(AXES)}")


def _secondary_names(axes: dict[str, object]) -> list[str]:
    return [f"secondary_{k}" for k in range(cast("int", axes.get("secondaries", 0)))]


def unit_rows(seed: int = 11, n: int = 200, **axes) -> list[dict]:
    """Per-unit rows at control rate 0.30 and treatment rate 0.42, plus each axis's columns."""
    _check_axes(axes)
    rng = np.random.default_rng(seed)
    secondaries = _secondary_names(axes)
    rows = []
    for variant, rate in (("control", 0.30), ("treatment", 0.42)):
        outcomes = rng.random(n) < rate
        for i in range(n):
            row = {
                "user_id": f"{variant}-{i:04d}",
                "variant": variant,
                "purchase": int(outcomes[i]),
                "enrollment": _EXPOSURE.date(),
            }
            if axes.get("cluster"):
                # Two units per household, every household inside one arm.
                row[str(axes["cluster"])] = f"{variant}-{i // 2:04d}"
            if axes.get("cuped"):
                row["pre_purchase"] = int(rng.random() < (0.6 if outcomes[i] else 0.2))
            if axes.get("breakout"):
                row[str(axes["breakout"])] = "a" if i % 2 else "b"
            for name in secondaries:
                row[name] = int(rng.random() < rate)
            rows.append(row)
    return rows


def switchback_rows(seed: int = 11, units: int = 8, cycles: int = 4) -> list[dict]:
    """A switchback panel: two periods per cycle, two steps per period, one 0/1 outcome per step."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(units):
        for cycle in range(cycles):
            first = "control" if rng.random() < 0.5 else "treatment"
            order = (first, "treatment" if first == "control" else "control")
            for period, group in enumerate(order):
                for step in range(2):
                    rate = 0.42 if group == "treatment" else 0.30
                    rows.append(
                        {
                            "unit": f"u{unit:02d}",
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "purchase": int(rng.random() < rate),
                        }
                    )
    return rows


def event_rows(units: list[dict]) -> list[dict]:
    def event(u, ts, name, exposure=False):
        return {
            "user_id": u["user_id"],
            "ts": ts,
            "event": name,
            "experiment_id": "exp" if exposure else None,
            "group_id": u["variant"] if exposure else None,
        }

    events = []
    for u in units:
        events.append(event(u, _EXPOSURE, "enrolled", exposure=True))
        if u["purchase"]:
            events.append(event(u, _PURCHASE, "buy"))
    return events


def definitions_yaml(dialect: str, source: str, *, plan: str) -> str:
    return f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {source}
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: enrolled, column: null}}
      - {{name: buy, column: null}}
exposures:
  - {{name: enrollment, fact: enrolled}}
metrics:
  - name: purchase
    type: conversion
    entity: user_id
    fact: buy
    window_days: 2
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-10T00:00:00
    end: 2025-01-12T00:00:00
    control_group: control
    allocation: {{control: 0.5, treatment: 0.5}}
    plan:
{plan}
"""


def _primary_spec(axes: dict[str, object]) -> MetricSpec:
    metric_type = cast("MetricSpecType", axes.get("metric_type") or "conversion")
    percentile = axes.get("winsor_percentile")
    return MetricSpec(
        name="purchase",
        type=metric_type,
        quantile=0.5 if metric_type == "quantile" else None,
        covariate="pre_purchase" if axes.get("cuped") else None,
        decision_method=(
            Method(name="cuped", variance_reduction="cuped") if axes.get("cuped") else None
        ),
        winsorization=(
            Winsorization(upper_percentile=float(cast("float", percentile)))
            if percentile is not None
            else None
        ),
    )


def analysis_plan(inference: InferenceSpec, **axes) -> AnalysisPlan:
    """The plan each axis requests: secondaries, a pre-period adjustment and the
    predeclared segment family (``segments``: the levels of the ``breakout``
    column, fixed before any outcome is read) sit on the plan."""
    _check_axes(axes)
    if axes.get("pre_period"):
        adjustment = PredeclaredAdjustment(coefficient=Fraction(1, 2), center=Fraction(1, 5))
        inference = InferenceSpec.model_validate(
            {**inference.model_dump(), "adjustments": {"purchase": adjustment}}
        )
    if axes.get("segments"):
        inference = InferenceSpec.model_validate(
            {
                **inference.model_dump(),
                "segments": {str(axes["breakout"]): tuple(cast("tuple", axes["segments"]))},
            }
        )
    return AnalysisPlan(primary="purchase", secondaries=_secondary_names(axes), inference=inference)


def frame_analysis(units: list[dict], inference: InferenceSpec, **axes) -> Analysis:
    _check_axes(axes)
    return Analysis.from_unit_summary(
        pa.Table.from_pylist(units),
        unit="user_id",
        group="variant",
        metrics=[
            _primary_spec(axes),
            *(MetricSpec(name=n, type="conversion") for n in _secondary_names(axes)),
        ],
        design=DESIGN,
        plan=analysis_plan(inference, **axes),
        experiment_id="exp",
        exposure_date="enrollment",
        cluster=cast("str | None", axes.get("cluster")),
    )


def switchback_analysis(inference: InferenceSpec) -> Analysis:
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    return Analysis.from_switchback_panel(
        pa.Table.from_pylist(switchback_rows()),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=[MetricSpec(name="purchase", type="conversion")],
        identification=DESIGN,
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        experiment_id="exp",
        plan=AnalysisPlan(primary="purchase", inference=inference),
        contrast_references={"purchase": UnitCycleTApproximation()},
    )


def axis_run(inference: InferenceSpec, *, seed: int = 7, **axes) -> Callable[[], list]:
    """Construction and readout for one axis, deferred so a refusal at either step is caught."""
    _check_axes(axes)
    if axes.get("switchback"):
        return lambda: list(switchback_analysis(inference).run())

    def run():
        analysis = frame_analysis(unit_rows(seed=seed, **axes), inference, **axes)
        if axes.get("breakout"):
            return list(analysis.run_breakout())
        return list(analysis.run())

    return run


def _definitions_file(directory: str | Path, dialect: str, source: str, plan_yaml: str) -> Path:
    """Write one route's definitions YAML and return its path."""
    path = Path(directory) / "defs.yaml"
    path.write_text(definitions_yaml(dialect, source, plan=plan_yaml))
    return path


def warehouse_analysis(con, dialect: str, source: str, plan_yaml: str) -> Analysis:
    with tempfile.TemporaryDirectory() as td:
        path = _definitions_file(td, dialect, source, plan_yaml)
        return Analysis.from_definitions("exp", path, con, store="none")


def assert_same_interval(a, b) -> None:
    la, lb = a.require_lift(), b.require_lift()
    assert la.value == lb.value, (la.value, lb.value)
    assert la.lb == lb.lb and la.ub == lb.ub, (la, lb)


def panel_rows(units: list[dict]) -> list[dict]:
    """The same units as one row per unit per day: the exposure day, then the purchase day."""
    return [
        {
            "user_id": u["user_id"],
            "variant": u["variant"],
            "ds": day,
            "exposed_on": _EXPOSURE.date(),
            "purchase": value,
        }
        for u in units
        for day, value in ((_EXPOSURE.date(), 0), (_PURCHASE.date(), u["purchase"]))
    ]


def panel_analysis(units: list[dict], inference: InferenceSpec) -> Analysis:
    """The unit-panel reading under the definitions' two-day window, complete by ``AS_OF``."""
    return Analysis.from_unit_panel(
        pa.Table.from_pylist(panel_rows(units)),
        unit="user_id",
        group="variant",
        date="ds",
        exposure_date="exposed_on",
        metrics=[MetricSpec(name="purchase", type="conversion", window_days=2)],
        design=DESIGN,
        plan=AnalysisPlan(primary="purchase", inference=inference),
        experiment_id="exp",
        observation_end=AS_OF,
    )


def expected_artifact_context(dialect: str, source: str, plan_yaml: str) -> ArtifactContext:
    """The context a caller of the artifact reader compiles for one route's definitions."""
    from increment.query.artifact_publish import artifact_context
    from increment.semantics.loader import load

    with tempfile.TemporaryDirectory() as td:
        definitions = load(str(_definitions_file(td, dialect, source, plan_yaml)))
    experiment = definitions.experiment("exp")
    assert experiment is not None
    return artifact_context(definitions, experiment, "error")


def adopt(
    analysis: Analysis, con, expected_context: ArtifactContext
) -> tuple[Analysis, WarehouseArtifactStore, UnitDayArtifactRef]:
    """Publish a definitions analysis as a unit-day artifact and reopen it through that reader."""
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = analysis.publish_unit_day_artifact(store)
    analysis.close()
    return (
        Analysis.from_unit_day_artifact(store, ref, expected_context=expected_context),
        store,
        ref,
    )


def portable_replay(analysis: Analysis) -> Analysis:
    """The exported checkpoint of *analysis* reopened through ``from_moments``."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "checkpoint.parquet"
        analysis.export(path)
        rows = pq.read_table(path).to_pylist()
    return Analysis.from_moments(
        rows, metrics=[MetricSpec(name="purchase", type="conversion")], control="control"
    )


def check_path_parity(con, dialect: str, source: str, kind: str) -> None:
    """Every path retains the unit-summary oracle's state, registration and interval."""
    route = ROUTES[kind]
    units = unit_rows()
    frame = frame_analysis(units, route.inference)
    oracle_snapshot = frame.capture_sequential(finalized=True)
    oracle = frame.run()[0]
    assert oracle_snapshot.registration.models[0].law == route.law

    def assert_same_state(analysis: Analysis):
        snapshot = analysis.capture_sequential(finalized=True, as_of=AS_OF)
        assert snapshot.registration.models == oracle_snapshot.registration.models
        assert snapshot.states == oracle_snapshot.states
        assert_same_interval(oracle, analysis.run()[0])
        return snapshot

    assert_same_state(panel_analysis(units, route.inference))
    definitions = warehouse_analysis(con, dialect, source, route.plan_yaml)
    definitions_snapshot = assert_same_state(definitions)
    replay = portable_replay(definitions)
    assert replay.sequential_snapshot() == definitions_snapshot
    assert_same_interval(oracle, replay.run()[0])
    artifact, store, ref = adopt(
        definitions, con, expected_artifact_context(dialect, source, route.plan_yaml)
    )
    try:
        assert assert_same_state(artifact).prefix_id == definitions_snapshot.prefix_id
    finally:
        artifact.close()
        store.drop_generation(ref.artifact_id, ref.generation_id)
