"""One per-unit dataset read through the dataframe oracle and both warehouse readers.

Every per-unit value is a small dyadic rational so each SQL/Arrow sum is exact
and the warehouse paths can agree with the dataframe path bit for bit. The
warehouse covariate is the metric's pre-period total (a ratio's numerator);
the frame carries the same total as an explicit ``pre`` column.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from increment import Analysis, Definitions, Method, MetricSpec
from increment.query.artifact_contract import unit_day_artifact_extension_catalog
from increment.query.artifact_publish import artifact_context
from increment.query.session import WarehouseArtifactStore
from increment.semantics.design import Randomized
from increment.semantics.loader import load
from increment.semantics.models import AnalysisPlan, InferenceSpec

CUPED = Method(name="cuped", variance_reduction="cuped")
UNADJUSTED = Method(name="unadjusted")
DESIGN = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
AS_OF = dt.date(2025, 1, 14)
_EXPOSURE = dt.datetime(2025, 1, 10, 8, tzinfo=dt.UTC)
_POST = dt.datetime(2025, 1, 11, 9, tzinfo=dt.UTC)
_PRE = dt.datetime(2025, 1, 7, 9, tzinfo=dt.UTC)  # inside the 7-day lookback


def unit_rows(seed: int = 7, n: int = 200) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 1.5, n)
    rows = []
    for variant, lift in (("control", 1.0), ("treatment", 1.05)):
        pre = rng.permutation(base)
        orders = np.clip(np.rint(6.0 + 0.5 * (pre - 5.0) + rng.normal(0.0, 1.0, n)), 1, None)
        revenue = orders * (2.0 + 0.4 * (pre - 5.0)) * lift + rng.normal(0.0, 0.5, n)
        for i in range(n):
            rows.append(
                {
                    "user_id": f"{variant}-{i:04d}",
                    "variant": variant,
                    "revenue": float(np.rint(max(revenue[i], 0.25) * 4) / 4),
                    "orders": float(int(orders[i])),
                    "pre": float(np.rint(max(pre[i], 0.0) * 4) / 4),
                }
            )
    return rows


def event_rows(units: list[dict]) -> list[dict]:
    def event(u, ts, name, value=None, exposure=False):
        return {
            "user_id": u["user_id"],
            "ts": ts,
            "event": name,
            "experiment_id": "exp" if exposure else None,
            "group_id": u["variant"] if exposure else None,
            "value": value,
        }

    events = []
    for u in units:
        events.append(event(u, _EXPOSURE, "enrolled", exposure=True))
        events.append(event(u, _PRE, "rev", u["pre"]))
        events.append(event(u, _POST, "rev", u["revenue"]))
        events.extend(
            event(u, _POST + dt.timedelta(minutes=j), "ord") for j in range(int(u["orders"]))
        )
    return events


def definitions_yaml(dialect: str, source: str, *, n_pre_periods: int, plan: str) -> str:
    return f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {source}
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: enrolled, column: null}}
      - {{name: rev, column: value}}
      - {{name: ord, column: null}}
exposures:
  - {{name: enrollment, fact: enrolled}}
metrics:
  - name: rpo
    type: ratio
    entity: user_id
    numerator: {{fact: rev, aggregation: sum, window_days: 2}}
    denominator: {{fact: ord, aggregation: count, window_days: 2}}
  - name: revenue
    type: mean
    entity: user_id
    fact: rev
    aggregation: sum
    window_days: 2
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-10T00:00:00
    end: 2025-01-12T00:00:00
    control_group: control
    allocation: {{control: 0.5, treatment: 0.5}}
    n_pre_periods: {n_pre_periods}
    plan:
{plan}
"""


def _spec(metric: str, *, cuped: bool) -> MetricSpec:
    spec = (
        MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders")
        if metric == "rpo"
        else MetricSpec(name="revenue", type="mean")
    )
    if cuped:
        spec = spec.model_copy(update={"covariate": "pre", "decision_method": CUPED})
    return spec


def frame_fixed(units, metric: str):
    """``(unadjusted, cuped)`` fixed-horizon rows from the unit-summary frame."""
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(units),
        unit="user_id",
        group="variant",
        metrics=[_spec(metric, cuped=True).model_copy(update={"decision_method": None})],
        design=DESIGN,
    )
    return analysis.run(decision_method=UNADJUSTED)[0], analysis.run(decision_method=CUPED)[0]


def frame_sequential(units, metric: str, *, cuped: bool):
    """``(snapshot, row)`` from the unit-summary frame under asymptotic_mean."""
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist([{**u, "enrolled": _EXPOSURE.date()} for u in units]),
        unit="user_id",
        group="variant",
        metrics=[_spec(metric, cuped=cuped)],
        design=DESIGN,
        plan=AnalysisPlan(
            primary=metric,
            inference=InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=400),
        ),
        experiment_id="exp",
        exposure_date="enrolled",
    )
    return analysis.capture_sequential(finalized=True), analysis.run()[0]


def _adopt(analysis: Analysis, definitions: Definitions, con) -> Analysis:
    """Publish ``analysis``'s unit-day artifact and reopen it through the public reader."""
    context = artifact_context(definitions, analysis.experiment, "error")
    requests = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind == "cuped_preperiod"
    ]
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = analysis.publish_unit_day_artifact(store, extensions=requests)
    analysis.close()
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context)


def _open(con, dialect: str, source: str, plan: str, *, artifact: bool) -> Analysis:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "defs.yaml"
        path.write_text(definitions_yaml(dialect, source, n_pre_periods=7, plan=plan))
        definitions = load(path)
        analysis = Analysis.from_definitions("exp", path, con, store="none")
    return _adopt(analysis, definitions, con) if artifact else analysis


def warehouse_fixed(con, dialect: str, source: str, metric: str, *, artifact: bool = False):
    """``(unadjusted, cuped)`` fixed-horizon rows from a warehouse reader."""
    plan = (
        f"      secondaries:\n        - metric: {metric}\n"
        "          decision_method: {name: unadjusted}\n"
        "          sensitivity_methods: [{name: cuped, variance_reduction: cuped}]\n"
    )
    analysis = _open(con, dialect, source, plan, artifact=artifact)
    rows = {row.method: row for row in analysis.run()}
    analysis.close()
    return rows["unadjusted"], rows["cuped"]


def warehouse_sequential(
    con, dialect: str, source: str, metric: str, *, cuped: bool, artifact: bool = False
):
    """``(snapshot, row)`` from a warehouse reader under asymptotic_mean."""
    binding = (
        f"      primary:\n        metric: {metric}\n"
        "        decision_method: {name: cuped, variance_reduction: cuped}\n"
        if cuped
        else f"      primary: {metric}\n"
    )
    plan = (
        binding + "      inference: {kind: asymptotic_mean, expected_decision_sample_size: 400}\n"
    )
    analysis = _open(con, dialect, source, plan, artifact=artifact)
    snapshot = analysis.capture_sequential(finalized=True, as_of=AS_OF)
    row = analysis.run()[0]
    analysis.close()
    return snapshot, row


def assert_same_interval(a, b, *, tolerance: float = 1e-12) -> None:
    la, lb = a.require_lift(), b.require_lift()
    assert abs(la.value - lb.value) <= tolerance, (la.value, lb.value)
    assert abs(la.lb - lb.lb) <= tolerance and abs(la.ub - lb.ub) <= tolerance, (la, lb)


def check_fixed_horizon_parity(con, dialect: str, source: str, tmp_path=None) -> dict[str, tuple]:
    """Ratio CUPED: dataframe oracle, definitions, artifact and portable moments agree."""
    units = unit_rows()
    frame_plain, frame_adj = frame_fixed(units, "rpo")
    intervals = {"frame": (frame_plain, frame_adj)}
    for artifact in (False, True):
        plain, adjusted = warehouse_fixed(con, dialect, source, "rpo", artifact=artifact)
        assert_same_interval(frame_plain, plain)
        assert_same_interval(frame_adj, adjusted)
        assert adjusted.reference_df == pytest.approx(frame_adj.reference_df)
        assert adjusted.abs_se == pytest.approx(frame_adj.abs_se, rel=1e-12)
        intervals["artifact" if artifact else dialect] = (plain, adjusted)
    if tmp_path is not None:
        plain, adjusted = portable_fixed(con, dialect, source, "rpo", tmp_path)
        assert_same_interval(frame_plain, plain)
        assert_same_interval(frame_adj, adjusted)
        assert adjusted.abs_se == pytest.approx(frame_adj.abs_se, rel=1e-12)
        intervals["portable"] = (plain, adjusted)
    return intervals


def portable_fixed(con, dialect: str, source: str, metric: str, tmp_path):
    """``(unadjusted, cuped)`` from an exported moments cube replayed through from_moments."""
    import pyarrow.parquet as pq

    from increment import Analysis, Method
    from increment.frame import MetricSpec

    plan = (
        f"      secondaries:\n        - metric: {metric}\n"
        "          decision_method: {name: unadjusted}\n"
        "          sensitivity_methods: [{name: cuped, variance_reduction: cuped}]\n"
    )
    analysis = _open(con, dialect, source, plan, artifact=False)
    path = tmp_path / "moments.parquet"
    analysis.export(path)
    analysis.close()
    rows = pq.read_table(path).to_pylist()
    replay = Analysis.from_moments(
        rows,
        metrics=[MetricSpec(name=metric, type="ratio", numerator="revenue", denominator="orders")],
        control="control",
    )
    plain = [r for r in replay.run() if r.method == "unadjusted"][0]
    adjusted = [
        r
        for r in replay.run(sensitivity_methods=[Method(name="cuped", variance_reduction="cuped")])
        if r.method == "cuped"
    ][0]
    return plain, adjusted


def check_sequential_parity(con, dialect: str, source: str) -> dict[str, tuple]:
    """Adjusted laws: retained states and intervals agree across every reader."""
    units = unit_rows()
    widths: dict[str, tuple] = {}
    for metric in ("revenue", "rpo"):
        frame_plain_snap, frame_plain = frame_sequential(units, metric, cuped=False)
        frame_adj_snap, frame_adj = frame_sequential(units, metric, cuped=True)
        plain_snap, plain = warehouse_sequential(con, dialect, source, metric, cuped=False)
        adj_snap, adjusted = warehouse_sequential(con, dialect, source, metric, cuped=True)
        art_snap, art_adjusted = warehouse_sequential(
            con, dialect, source, metric, cuped=True, artifact=True
        )
        expected_law = "adjusted_ratio_mean" if metric == "rpo" else "adjusted_mean"
        for snapshot in (frame_adj_snap, adj_snap, art_snap):
            assert snapshot.registration.models[0].law == expected_law
        assert frame_plain_snap.states == plain_snap.states
        assert frame_adj_snap.states == adj_snap.states == art_snap.states
        assert adj_snap.prefix_id == art_snap.prefix_id
        assert_same_interval(frame_plain, plain, tolerance=0.0)
        assert_same_interval(frame_adj, adjusted, tolerance=0.0)
        assert_same_interval(frame_adj, art_adjusted, tolerance=0.0)
        widths[metric] = (plain, adjusted)
    return widths


def check_day_axis_parity(con, dialect: str, source: str) -> None:
    """Day-axis ratio CUPED: the final as-of equals the total, and both readers agree."""
    from increment import Method
    from increment.results import LiftEstimate

    cuped = Method(name="cuped", variance_reduction="cuped")
    plan = (
        "      secondaries:\n        - metric: rpo\n"
        "          decision_method: {name: unadjusted}\n"
        "          sensitivity_methods: [{name: cuped, variance_reduction: cuped}]\n"
    )
    definitions = _open(con, dialect, source, plan, artifact=False)
    artifact = _open(con, dialect, source, plan, artifact=True)
    try:
        total_row = [r for r in definitions.run() if r.method == "cuped"][0]
        assert isinstance(total_row, LiftEstimate)
        total = total_row.require_lift()

        def rows(analysis, fn):
            return {
                (r.group_id, r.ds): r
                for r in fn(metrics=["rpo"], sensitivity_methods=[cuped])
                if r.method == "cuped"
            }

        asof = rows(definitions, definitions.run_asof_lift)
        last = max(ds for _, ds in asof)
        final = asof[("treatment", last)].require_lift()
        # Cumulative through the last day is the whole experiment.
        assert final.value == pytest.approx(total.value, abs=1e-12)
        assert final.lb == pytest.approx(total.lb, abs=1e-12)
        assert final.ub == pytest.approx(total.ub, abs=1e-12)
        for fn_name in ("run_asof_lift", "run_daily_lift"):
            left = rows(definitions, getattr(definitions, fn_name))
            right = rows(artifact, getattr(artifact, fn_name))
            assert set(left) == set(right)
            available = 0
            for key in left:
                a, b = left[key].lift, right[key].lift
                assert (a is None) == (b is None), key
                if a is None:
                    continue
                available += 1
                assert a.value == pytest.approx(b.value, abs=1e-12), key
                assert a.lb == pytest.approx(b.lb, abs=1e-12), key
                assert a.ub == pytest.approx(b.ub, abs=1e-12), key
            assert available >= 1, fn_name
    finally:
        definitions.close()
        artifact.close()
