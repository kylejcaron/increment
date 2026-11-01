"""Every scoped result row states the declared type of its metric.

``metric_type`` is the user's declared ``Metric.type``; it is stamped wherever a
readout stamps row provenance, survives every serialization round trip, and is
``None`` only when the type is genuinely unknown.
"""

from __future__ import annotations

import json
import pickle

import narwhals as nw
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from increment import AnalysisPlan, MetricSpec, Scope, from_frame, from_switchback_frame
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.engine import estimate_lift
from increment.estimation.readout_types import ReadoutResults
from increment.estimation.results import LiftEstimate
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    IdentificationGate,
    Observational,
    Randomized,
    UptakeSpec,
)
from increment.semantics.models import InferenceSpec, MeanMetric
from tests.analysis_factory import lift_rows, make_binding, view_readout
from tests.parity_harness import dataset as ds

DECLARED = {"revenue": "mean", "purchase_rate": "conversion", "rps": "ratio"}
FRAME_METRICS = [
    MetricSpec(name="revenue", type="mean", missing="zero", preferred_direction="increase"),
    MetricSpec(name="purchase_rate", type="conversion", value_column="converted"),
    MetricSpec(
        name="rps", type="ratio", numerator="revenue", denominator="sessions", missing="zero"
    ),
]


def _summary_frame() -> pa.Table:
    con = ds.duckdb_connection()
    try:
        return ds.unit_summary_frame(con)
    finally:
        con.disconnect()


def _declared_types(rows) -> dict[str, set[str | None]]:
    """Each metric's stamped types, excluding the design-level compliance row."""
    out: dict[str, set[str | None]] = {}
    for row in rows:
        if getattr(row, "estimand", None) == "compliance":
            continue
        out.setdefault(row.metric, set()).add(row.metric_type)
    return out


def _assert_stamped(rows, declared) -> None:
    assert rows, "the readout produced no rows"
    stamped = _declared_types(rows)
    assert set(stamped) == set(declared)
    assert stamped == {name: {kind} for name, kind in declared.items()}


@pytest.mark.integration
def test_run_rows_carry_declared_metric_type():
    experiment, resolver = from_frame(
        _summary_frame(),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=FRAME_METRICS,
    )
    try:
        rows = lift_rows(view_readout(experiment, resolver))
    finally:
        resolver.close()
    _assert_stamped(rows, DECLARED)


def _confounded_conversion_table(n: int = 400, seed: int = 11) -> pa.Table:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    d = rng.binomial(1, propensity)
    rate = 1.0 / (1.0 + np.exp(-(-0.5 + 0.4 * x1 + 0.3 * d)))
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C").tolist(),
            "converted": rng.binomial(1, rate).astype(int).tolist(),
            "x1": x1,
            "x2": x2,
        }
    )


@pytest.mark.integration
def test_observational_and_breakout_rows_carry_metric_type():
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x1", "x2")),
        gate=IdentificationGate(overlap="trim"),
    )
    observational_experiment, observational_resolver = from_frame(
        _confounded_conversion_table(),
        shape="summary",
        unit="user_id",
        group="variant",
        metrics={"converted": "conversion"},
        design=design,
    )
    try:
        rows = lift_rows(view_readout(observational_experiment, observational_resolver))
    finally:
        observational_resolver.close()
    _assert_stamped(rows, {"converted": "conversion"})

    panel_experiment, panel_resolver = from_frame(
        ds.unit_panel_frame(_summary_frame()),
        shape="panel",
        unit="user_id",
        group="variant",
        date="date",
        control="control",
        metrics=[MetricSpec(name="purchase_rate", type="conversion", value_column="converted")],
        breakouts=("store",),
    )
    try:
        breakout = view_readout(
            panel_experiment, panel_resolver, Scope(view="breakout", grain="total")
        )
    finally:
        panel_resolver.close()
    assert {row.dimension for row in breakout} == {"store"}
    _assert_stamped(breakout, {"purchase_rate": "conversion"})


@pytest.mark.slow
@pytest.mark.integration
def test_metric_type_is_identical_across_ingresses():
    from tests.parity_harness.cases import PARITY_CASES
    from tests.parity_harness.runner import run_case

    case = next(
        case for case in PARITY_CASES if case.id == "mean_ratio_conversion_cuped_fixed_horizon"
    )
    declared = {**DECLARED, "revenue_cuped": "mean"}
    result = run_case(case)
    assert len(result.rows) >= 2
    by_constructor = {}
    for constructor, cells in result.rows.items():
        stamped = {}
        for key, by_method in cells.items():
            for payload in by_method.values():
                assert payload["metric_type"] == declared[key[0]], (constructor, key)
                stamped[key] = payload["metric_type"]
        by_constructor[constructor] = stamped
    reference = next(iter(by_constructor.values()))
    for constructor, stamped in by_constructor.items():
        assert stamped == reference, constructor


@pytest.mark.integration
def test_metric_type_round_trips():
    experiment, resolver = from_frame(
        _summary_frame(),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=FRAME_METRICS,
    )
    try:
        results = lift_rows(view_readout(experiment, resolver))
    finally:
        resolver.close()
    row = next(row for row in results if row.metric == "rps")
    assert row.metric_type == "ratio"
    assert LiftEstimate.model_validate(row.model_dump()).metric_type == "ratio"
    assert LiftEstimate.model_validate_json(row.model_dump_json()).metric_type == "ratio"
    assert pickle.loads(pickle.dumps(row)).metric_type == "ratio"

    payload = results.model_dump_json()
    restored = ReadoutResults.model_validate_json(payload)
    assert [item.metric_type for item in restored] == [item.metric_type for item in results]

    legacy = json.loads(payload)
    for item in legacy["rows"]:
        del item["metric_type"]
    reloaded = ReadoutResults.model_validate_json(json.dumps(legacy))
    assert {item.metric_type for item in reloaded} == {None}
    assert pickle.loads(pickle.dumps(reloaded))[0].metric_type is None


def test_low_level_rows_leave_metric_type_unknown():
    def arm(group_id, mean):
        n = 400
        total = n * mean
        return centered_row_from_raw_sums(
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": group_id,
                "n": n,
                "sum_y": total,
                "sum_y2": 4.0 * (n - 1) + total**2 / n,
            }
        )

    (row,) = estimate_lift(
        [MeanMetric(name="m", entity="u", fact="m")],
        [arm("control", 10.0), arm("treatment", 11.0)],
        control_group="control",
    ).results
    assert row.metric_type is None


def test_encouragement_rows_carry_metric_type_and_compliance_stays_unknown():
    rows = [
        ("u01", "control", 10.0, 0),
        ("u02", "control", 12.0, 0),
        ("u03", "control", 9.0, 0),
        ("u04", "control", 20.0, 1),
        ("u05", "treatment", 25.0, 1),
        ("u06", "treatment", 30.0, 1),
        ("u07", "treatment", 28.0, 1),
        ("u08", "treatment", 15.0, 0),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "revenue", "clicked"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    experiment, resolver = from_frame(
        table,
        shape="summary",
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                justification="assignment only moves revenue via uptake"
            ),
            min_first_stage_z=0.5,
        ),
    )
    try:
        results = lift_rows(view_readout(experiment, resolver))
    finally:
        resolver.close()
    compliance = [row for row in results if row.estimand == "compliance"]
    assert compliance and {row.metric_type for row in compliance} == {None}
    assert {row.estimand for row in results} >= {"itt", "late"}
    _assert_stamped(results, {"revenue": "mean"})


def test_sequential_rows_carry_metric_type():
    frame = pd.DataFrame(
        [
            {"unit": f"{i:04d}-{arm}", "arm": arm, "outcome": value, "exposure": i}
            for i in range(40)
            for arm, value in (("control", 1.0 + (i % 5) * 0.1), ("treatment", 1.4 + (i % 7) * 0.1))
        ]
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        metrics=[MetricSpec(name="outcome", type="mean")],
        experiment_id="experiment",
        plan=AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean")),
        exposure_date="exposure",
    )
    try:
        rows = lift_rows(view_readout(experiment, resolver))
    finally:
        resolver.close()
    assert {row.inference for row in rows} == {"asymptotic_mean"}
    _assert_stamped(rows, {"outcome": "mean"})


@pytest.mark.integration
def test_daily_lift_rows_carry_metric_type():
    experiment, resolver = from_frame(
        ds.unit_panel_frame(_summary_frame()),
        shape="panel",
        unit="user_id",
        group="variant",
        date="date",
        control="control",
        metrics=FRAME_METRICS,
    )
    try:
        rows = view_readout(experiment, resolver, Scope(view="daily", grain="daily"))
    finally:
        resolver.close()
    _assert_stamped(rows, DECLARED)


@pytest.mark.integration
def test_daily_and_asof_value_rows_carry_metric_type():
    from increment.breakout.estimates import DailyMetricValue

    experiment, resolver = from_frame(
        ds.unit_panel_frame(_summary_frame()),
        shape="panel",
        unit="user_id",
        group="variant",
        date="date",
        control="control",
        metrics=FRAME_METRICS,
    )
    try:
        daily = view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True))
        asof = view_readout(experiment, resolver, Scope(view="asof", grain="asof", values=True))
    finally:
        resolver.close()
    for values in (daily, asof):
        assert values and all(isinstance(row, DailyMetricValue) for row in values)
        assert all(row.source_snapshot_id is not None for row in values)
        _assert_stamped(values, DECLARED)
    restored = ReadoutResults.model_validate_json(daily.model_dump_json())
    assert [row.metric_type for row in restored] == [row.metric_type for row in daily]
    frame = nw.from_native(asof.to_frame(backend="polars"), eager_only=True)
    assert set(frame["metric_type"].to_list()) == set(DECLARED.values())


def test_switchback_contrast_rows_carry_metric_type():
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    panel = []
    for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control"))):
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    panel.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": float(cycle + period + step + (group == "treatment")),
                        }
                    )
    experiment, resolver = from_switchback_frame(
        pd.DataFrame(panel),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    try:
        results = view_readout(experiment, resolver)
    finally:
        resolver.close()
    assert [row.metric_type for row in results] == ["mean"]
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert [row.metric_type for row in restored] == ["mean"]
    frame = nw.from_native(results.to_frame(backend="polars"), eager_only=True)
    assert frame["metric_type"].to_list() == ["mean"]


@pytest.mark.integration
def test_exploratory_metric_rows_carry_metric_type():
    import warnings

    from increment.errors import IncrementWarning
    from increment.semantics.models import Definitions

    con = ds.duckdb_connection()
    try:
        definitions = Definitions.model_validate(
            ds.definitions_dict(plan=AnalysisPlan(primary="purchase_rate"))
        )
        analysis_experiment, analysis_resolver = make_binding(con, definitions, experiment="exp")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with pytest.warns(IncrementWarning):
                    rows = lift_rows(
                        view_readout(
                            analysis_experiment,
                            analysis_resolver,
                            exploratory_metrics=["rps", "revenue"],
                        )
                    )
        finally:
            analysis_resolver.close()
    finally:
        con.disconnect()
    assert {row.role for row in rows if row.metric in ("rps", "revenue")} == {"exploratory"}
    _assert_stamped(rows, DECLARED)


@pytest.mark.integration
def test_frame_export_carries_metric_type_beside_source_snapshot_id():
    from increment.tables import estimates_to_readout

    experiment, resolver = from_frame(
        _summary_frame(),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=FRAME_METRICS,
    )
    try:
        results = lift_rows(view_readout(experiment, resolver))
    finally:
        resolver.close()
    frame = nw.from_native(results.to_frame(backend="polars"), eager_only=True)
    assert "source_snapshot_id" in frame.columns
    assert dict(zip(frame["metric"].to_list(), frame["metric_type"].to_list(), strict=True)) == {
        row.metric: row.metric_type for row in results
    }
    readout = estimates_to_readout(results)
    assert {entry["metric"]: entry["metric_type"] for entry in readout} == {
        row.metric: row.metric_type for row in results
    }
