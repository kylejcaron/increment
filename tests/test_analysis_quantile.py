"""Quantile metrics on the definitions (warehouse) path.

A quantile is not additive across units, so it cannot ride the moments
transport. These tests pin the per-unit route: the same pre-collapse rows
`group_summary` would have consumed, handed to the order-statistic
estimator instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import increment as inc
from tests.analysis_factory import lift_rows, make_analysis_like
from tests.sequential_cases import registration


def _defs_yaml() -> str:
    return """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: latency
        column: value
      - name: enrolled
        column: null
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
  - name: mean_latency
    type: mean
    entity: user_id
    fact: latency
    window_days: 7
    aggregation: sum
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2024-01-01T00:00:00
    end: 2024-01-08T00:00:00
    control_group: C
    plan: {secondaries: [p90_latency, mean_latency]}
"""


def _defs_yaml_with_breakout() -> str:
    """Same experiment, plus a static property and a declared breakout/
    factor on it - for pinning the day-axis/breakout/factor refusals."""
    return """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: latency
        column: value
      - name: enrolled
        column: null
    properties:
      - name: country
        column: country_code
        dtype: string
        as_of: static
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
  - name: mean_latency
    type: mean
    entity: user_id
    fact: latency
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2024-01-01T00:00:00
    end: 2024-01-08T00:00:00
    control_group: C
    plan: {secondaries: [p90_latency, mean_latency]}
    breakouts: [{property: country}]
    factors: [{property: country}]
"""


def _events_table(n_per_arm: int = 400, seed: int = 7, shift: float = 1.25):
    """Two rows per unit: one `enrolled` occurrence (marks exposure) and
    one `latency` value. Treatment is a multiplicative shift, so the
    log-scale quantile lift has a known sign. `experiment_id`/`group_id`
    are what the fact-based exposure reads to assign units to arms;
    stamping them on every row is redundant but harmless (see
    examples/_seed.py)."""
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    control = rng.lognormal(mean=0.0, sigma=0.6, size=n_per_arm)
    treatment = rng.lognormal(mean=0.0, sigma=0.6, size=n_per_arm) * shift
    users = [f"c{i}" for i in range(n_per_arm)] + [f"t{i}" for i in range(n_per_arm)]
    groups = ["C"] * n_per_arm + ["T"] * n_per_arm
    values = np.concatenate([control, treatment])
    enroll_ts = np.datetime64("2024-01-02T00:00:00")
    metric_ts = np.datetime64("2024-01-08T01:00:00")

    rows = []
    for user, group, value in zip(users, groups, values, strict=True):
        common = {"user_id": user, "experiment_id": "exp", "group_id": group}
        rows.append({**common, "ts": enroll_ts, "event": "enrolled", "value": None})
        rows.append({**common, "ts": metric_ts, "event": "latency", "value": float(value)})
    return pa.Table.from_pylist(rows)


def _analysis(tmp_path, table):
    import ibis

    con = ibis.duckdb.connect()
    con.create_table("events", table)
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(_defs_yaml())
    return inc.Analysis.from_definitions("exp", defs_path, con)


def _analysis_with_breakout(tmp_path, table):
    """Same as `_analysis`, loaded from `_defs_yaml_with_breakout` - for
    tests that need a declared breakout/factor on the experiment. `table`
    must carry a `country_code` column."""
    import ibis

    con = ibis.duckdb.connect()
    con.create_table("events", table)
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(_defs_yaml_with_breakout())
    return inc.Analysis.from_definitions("exp", defs_path, con)


def _analysis_from_defs(defs_path, con):
    return inc.Analysis.from_definitions("exp", defs_path, con)


def test_quantile_metric_estimates_on_the_definitions_path(tmp_path):
    """The load-time refusal is gone and the metric produces a real,
    ordered interval with a positive lift for a positive shift."""
    an = _analysis(tmp_path, _events_table())
    rows = {r.metric: r for r in lift_rows(an.run())}
    assert set(rows) == {"p90_latency", "mean_latency"}
    q = rows["p90_latency"]
    lift = q.require_lift()
    assert lift.lb is not None and lift.ub is not None and lift.value is not None
    assert lift.lb < lift.value < lift.ub
    assert lift.value > 0.0


def test_quantile_estimator_refuses_cuped_before_reading_unit_rows():
    """Request-shape refusals fire before the estimator touches the source."""
    from increment.errors import UnsupportedRequestError
    from increment.estimation.engine import Method
    from increment.estimation.quantile import estimate_quantile_lift

    class _NeverRead:
        def unit_frame(self, metric):
            raise AssertionError("unit_frame must not be read before validation")

    metric = SimpleNamespace(name="p90", quantile=0.9)
    with pytest.raises(UnsupportedRequestError) as exc_info:
        estimate_quantile_lift(
            _NeverRead(),
            metric,
            "C",
            methods=[Method(name="unadjusted", variance_reduction="cuped")],
        )
    assert exc_info.value.code == "arm.metric.quantile_cuped"


def _events_with_null_assignment(table, *, mixed: bool = False):
    import pyarrow as pa

    rows = table.to_pylist()
    for row in rows:
        if row["user_id"] == "c0" and row["event"] == "enrolled":
            original = dict(row)
            row["group_id"] = None
            if mixed:
                rows.extend([{**original}, {**original, "group_id": "T"}])
            break
    return pa.Table.from_pylist(rows, schema=table.schema)


@pytest.mark.parametrize("mixed", [False, True], ids=["null-only", "mixed-plus-null"])
def test_quantile_default_assignment_policy_rejects_null_unit(tmp_path, mixed):
    from increment.errors import InvalidRequestError

    an = _analysis(tmp_path, _events_with_null_assignment(_events_table(), mixed=mixed))
    with pytest.raises(InvalidRequestError) as raised:
        lift_rows(an.run(metrics=["p90_latency"]))
    assert raised.value.code == (
        "query.integrity.mixed_assignment_units"
        if mixed
        else "query.integrity.unassigned_assignment_units"
    )


def test_sparse_metric_refuses_rather_than_reporting_a_zero_quantile(tmp_path):
    """The spine zero-fills units with no events, so a sparse metric's p90
    is 0 and has no log. Refuse by name - a silent 0 would read as a real
    baseline."""
    import pyarrow as pa
    import pyarrow.compute as pc

    table = _events_table(n_per_arm=200)
    # Keep every unit's exposure (spine still enrolls all of them), drop
    # most latency rows: the dropped units land at y=0.
    enrolled = table.filter(pc.field("event") == "enrolled")
    latency = table.filter(pc.field("event") == "latency")
    keep = [i for i, u in enumerate(latency["user_id"].to_pylist()) if u.endswith("0")]
    sparse_latency = latency.take(keep)
    combined = pa.concat_tables([enrolled, sparse_latency])

    from increment.errors import InvalidRequestError

    an = _analysis(tmp_path, combined)
    with pytest.raises(InvalidRequestError) as raised:
        lift_rows(an.run())
    assert raised.value.code == "estimation.quantile.estimate_quantile.metric_arm"
    assert raised.value.context["metric"] == "p90_latency"


def test_warehouse_quantile_matches_the_frame_path(tmp_path):
    """Same units, same quantile, two substrates. The order statistic is
    computed from identical per-unit values, so this is tighter than the
    moments-path parity bound: no float summation order differs. Both
    sides declare the SAME two secondaries (`p90_latency`, `mean_latency`)
    so the family sizes are structurally equal (not just coincidentally
    equal because both fixtures' shifts happen to select everything) --
    both independently run the same BH/FCR family selection over an
    identical m=2 family and land on the same FCR level."""
    import pyarrow.compute as pc

    table = _events_table()
    warehouse = {r.metric: r for r in lift_rows(_analysis(tmp_path, table).run())}["p90_latency"]

    latency_rows = table.filter(pc.field("event") == "latency")
    per_unit = latency_rows.select(["user_id", "group_id", "value"]).rename_columns(
        ["user_id", "group_id", "latency"]
    )
    frame_an = inc.Analysis.from_unit_summary(
        per_unit,
        unit="user_id",
        group="group_id",
        control="C",
        metrics=[
            inc.MetricSpec(
                name="p90_latency", value_column="latency", type="quantile", quantile=0.9
            ),
            inc.MetricSpec(name="mean_latency", value_column="latency", type="mean"),
        ],
        plan=inc.AnalysisPlan(secondaries=["p90_latency", "mean_latency"]),
    )
    frame = {r.metric: r for r in lift_rows(frame_an.run())}["p90_latency"]

    assert warehouse.discovery is True and warehouse.discovery == frame.discovery
    assert warehouse.require_lift().level == pytest.approx(frame.require_lift().level, rel=1e-12)
    assert warehouse.require_lift().value == pytest.approx(frame.require_lift().value, rel=1e-12)
    assert warehouse.require_lift().lb == pytest.approx(frame.require_lift().lb, rel=1e-12)
    assert warehouse.require_lift().ub == pytest.approx(frame.require_lift().ub, rel=1e-12)


@pytest.mark.parametrize(
    "method_name",
    ["panel_sql", "summary_sql", "run_daily", "run_asof", "run_daily_lift", "run_asof_lift"],
)
def test_moments_only_surfaces_refuse_a_quantile_metric(tmp_path, method_name):
    """No summary SQL, day-axis, or as-of moments exist for an order
    statistic. Refuse by name rather than emit a dict/rows that silently
    treat the quantile's per-unit column as an additive sum."""
    from increment.errors import CapabilityError

    sql_methods = {"panel_sql", "summary_sql"}
    an = _analysis(tmp_path, _events_table(n_per_arm=50))
    with pytest.raises(CapabilityError) as raised:
        getattr(an, method_name)()
    expected = "source.native.operation" if method_name in sql_methods else "breakout.quantile"
    assert raised.value.code == expected


def test_run_breakout_refuses_a_quantile_metric(tmp_path):
    """A quantile does not decompose over segment moments - run_breakout()
    must refuse rather than silently sum the per-unit column into a
    mean-shaped breakout row."""
    import pyarrow as pa

    from increment.errors import CapabilityError

    table = _events_table(n_per_arm=50).append_column(
        "country_code", pa.array(["US"] * (2 * 2 * 50))
    )
    an = _analysis_with_breakout(tmp_path, table)
    with pytest.raises(CapabilityError) as raised:
        an.run_breakout()
    assert raised.value.code == "readout.metric.quantile_breakout"


def test_factor_summaries_refuses_a_quantile_metric(tmp_path):
    """A quantile has no whole-window moments to absorb a factor over."""
    import pyarrow as pa

    from increment.errors import CapabilityError

    table = _events_table(n_per_arm=50).append_column(
        "country_code", pa.array(["US"] * (2 * 2 * 50))
    )
    an = _analysis_with_breakout(tmp_path, table)
    with pytest.raises(CapabilityError) as raised:
        an.factor_summaries()
    assert raised.value.code == "breakout.quantile"


def test_warehouse_quantile_refuses_call_time_inference(tmp_path):
    """Quantile readouts now route through the shared readout pipeline
    (Task 3), which reads its epistemic policy exclusively from the
    declared AnalysisPlan - call-time inference= is refused
    unconditionally, declared plan or not, matching every other
    non-Encouragement design."""
    from increment.plan import compile_decision_plan

    an = _analysis(tmp_path, _events_table(n_per_arm=50))
    an = make_analysis_like(an, plan=compile_decision_plan(None, an.metrics, path="warehouse"))
    assert lift_rows(an.run())
    with pytest.raises(TypeError):
        lift_rows(an.run(inference=inc.AlwaysValid(registration=registration())))  # ty: ignore[unknown-argument]


def test_native_encouragement_refuses_a_quantile_metric(tmp_path):
    """No native (Analysis.from_definitions) call site populates an
    Encouragement design today (this guard is currently unreachable in
    practice), but if one ever does, a quantile metric must be refused
    here rather than silently pushed through estimate_encouragement on
    its mean-based group_summary moments. Manual `_design` assignment
    mirrors the established native-encouragement test fixture pattern
    (tests/test_analysis.py::_analysis_with_asof_encouragement_events)."""
    from increment.errors import CapabilityError
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    an = _analysis(tmp_path, _events_table(n_per_arm=50))
    an = make_analysis_like(
        an,
        design=Encouragement(
            control_group="C",
            uptake=UptakeSpec(fact="enrolled"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="test-only encouragement fixture"
            ),
        ),
    )
    with pytest.raises(CapabilityError) as raised:
        lift_rows(an.run())
    assert raised.value.code == "breakout.quantile"


def test_quantile_primary_splits_alpha_by_treatment_arm_count(tmp_path):
    """2 primaries (one quantile) x 2 treatment arms: cell_alpha =
    (plan.alpha/2)/2 = 0.0125 -> level 0.9875 on both primaries' rows -
    pins the quantile branch of arm-counting, which reads `unit_rows`'
    distinct `group_id` (minus control) since a quantile metric has no
    `combined` moments rows to count arms from. Two treatment arms (not
    one) so the result discriminates a correct count from the n_arms=0
    degeneracy (`cell_alpha = alpha_share/n_arms if n_arms else
    alpha_share` gives the same 0.975 for n_arms in {0, 1})."""
    import ibis
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(11)
    n = 50
    control = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    treatment_a = rng.lognormal(mean=0.0, sigma=0.6, size=n) * 1.25
    treatment_b = rng.lognormal(mean=0.0, sigma=0.6, size=n) * 1.4
    users = [f"c{i}" for i in range(n)] + [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
    groups = ["C"] * n + ["A"] * n + ["B"] * n
    values = np.concatenate([control, treatment_a, treatment_b])
    enroll_ts = np.datetime64("2024-01-02T00:00:00")
    metric_ts = np.datetime64("2024-01-08T01:00:00")
    rows = []
    for user, group, value in zip(users, groups, values, strict=True):
        common = {"user_id": user, "experiment_id": "exp", "group_id": group}
        rows.append({**common, "ts": enroll_ts, "event": "enrolled", "value": None})
        rows.append({**common, "ts": metric_ts, "event": "latency", "value": float(value)})
    table = pa.Table.from_pylist(rows)

    con = ibis.duckdb.connect()
    con.create_table("events", table)
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(
        _defs_yaml().replace(
            "plan: {secondaries: [p90_latency, mean_latency]}",
            "plan: {primary: [p90_latency, mean_latency]}",
        )
    )
    an = _analysis_from_defs(defs_path, con)
    results = lift_rows(an.run())
    assert {r.group_id for r in results} == {"A", "B"}
    for r in results:
        assert r.role == "primary"
        assert r.require_lift().level == pytest.approx(0.9875)
    rows_by_metric = {r.metric: r for r in results}
    assert rows_by_metric["p90_latency"].role == "primary"
    assert rows_by_metric["mean_latency"].role == "primary"


def test_quantile_primary_casts_int_group_id_before_arm_counting(tmp_path):
    """Same regression `test_execute_path_casts_int_group_id_before_arm_counting`
    pins for the mean-metric (`combined`-based) arm count, but for the
    quantile branch specifically: it reads a different source
    (`unit_rows[metric.name]["group_id"]`), so an int64 `group_id` column
    could in principle type-handle differently there. A single quantile
    primary with one treatment arm: an uncast subtraction leaves the
    control arm's int id in the set, over-counts n_arms to 2, and
    under-splits alpha to level 0.975 instead of the correct 0.95."""
    import ibis
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(3)
    n = 50
    control = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    treatment = rng.lognormal(mean=0.0, sigma=0.6, size=n) * 1.25
    users = [f"c{i}" for i in range(n)] + [f"t{i}" for i in range(n)]
    group_ids = [0] * n + [1] * n
    values = np.concatenate([control, treatment])
    enroll_ts = np.datetime64("2024-01-02T00:00:00", "us")
    metric_ts = np.datetime64("2024-01-02T01:00:00", "us")
    rows = []
    for user, group_id, value in zip(users, group_ids, values, strict=True):
        common = {"user_id": user, "experiment_id": "exp", "group_id": group_id}
        rows.append({**common, "ts": enroll_ts, "event": "enrolled", "value": None})
        rows.append({**common, "ts": metric_ts, "event": "latency", "value": float(value)})
    schema = pa.schema(
        [
            ("user_id", pa.string()),
            ("experiment_id", pa.string()),
            ("group_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("event", pa.string()),
            ("value", pa.float64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    con = ibis.duckdb.connect()
    con.create_table("events", table)
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(
        _defs_yaml()
        .replace("plan: {secondaries: [p90_latency, mean_latency]}", "plan: {primary: p90_latency}")
        .replace("control_group: C", "control_group: '0'")
    )
    an = _analysis_from_defs(defs_path, con)
    result = {r.metric: r for r in lift_rows(an.run())}["p90_latency"]
    assert result.role == "primary"
    assert result.require_lift().level == pytest.approx(0.95)


@pytest.mark.filterwarnings(
    "ignore:.*quantile metric.*fixed-horizon order-statistic bracket:UserWarning"
)
def test_declared_plan_always_valid_inference_reaches_quantile_readout(tmp_path):
    from increment.errors import CapabilityError
    from tests.sequential_cases import registered_native

    original = _analysis(tmp_path, _events_table(n_per_arm=50))
    with pytest.raises(CapabilityError) as raised:
        registered_native(
            original,
            metrics=[m for m in original.metrics if m.type == "quantile"],
        )
    assert raised.value.code == "sequential.route.unsupported"


def test_quantile_family_discovery_is_independent_of_display_alpha():
    """Quantile-family evidence comes from the stored inversion, not its bracket."""
    import pyarrow as pa

    from increment.decision import PValueEvidence
    from increment.estimation.quantile import estimate_quantile_lift_computation
    from increment.semantics.models import QuantileMetric

    values = np.arange(1.0, 101.0)
    rows = pa.table(
        {
            "unit_id": list(range(200)),
            "group_id": ["C"] * 100 + ["T"] * 100,
            "y": np.concatenate([values, values + 10.0]),
        }
    )
    metric = QuantileMetric(name="median", entity="unit", fact="outcome", quantile=0.5)

    computations = [
        estimate_quantile_lift_computation(None, metric, "C", unit_rows=rows, alpha=display_alpha)
        for display_alpha in (0.05, 0.005)
    ]
    for computation in computations:
        result = computation.results[0]
        evidence = next(iter(computation.evidence.values()))
        assert isinstance(evidence, PValueEvidence)
        assert result.p_value() == pytest.approx(0.19334790449564246)
        assert evidence.p_value == pytest.approx(0.19334790449564246)
        assert evidence.reference == "quantile_inversion"

    for alpha in (0.05, 0.005):
        for q, discovery in ((0.19, False), (0.20, True)):
            analysis = inc.Analysis.from_unit_summary(
                rows,
                unit="unit_id",
                group="group_id",
                control="C",
                metrics=[
                    inc.MetricSpec(name="median", type="quantile", value_column="y", quantile=0.5)
                ],
                plan=inc.AnalysisPlan(secondaries=["median"], alpha=alpha, q=q),
            )
            result = lift_rows(analysis.run())[0]
            assert result.p_value() == pytest.approx(0.19334790449564246)
            assert result.discovery is discovery


def test_quantile_joint_metric_arm_family_uses_inversion_at_every_display_alpha():
    import pyarrow as pa

    baseline = np.arange(1.0, 101.0)
    frame = pa.table(
        {
            "unit": list(range(300)),
            "arm": ["C"] * 100 + ["T1"] * 100 + ["T2"] * 100,
            "y": np.concatenate([baseline, baseline + 10, baseline + 50]),
            "z": np.concatenate([baseline, baseline, baseline + 30]),
        }
    )
    for alpha in (0.05, 0.005):
        analysis = inc.Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="C",
            metrics=[
                inc.MetricSpec(name=name, type="quantile", value_column=column, quantile=0.5)
                for name, column in (("median", "y"), ("other", "z"))
            ],
            plan=inc.AnalysisPlan(secondaries=["median", "other"], alpha=alpha, q=0.26),
        )
        rows = lift_rows(analysis.run())
        assert {(row.metric, row.group_id) for row in rows} == {
            ("median", "T1"),
            ("median", "T2"),
            ("other", "T1"),
            ("other", "T2"),
        }
        assert {(row.metric, row.group_id) for row in rows if row.discovery} == {
            ("median", "T1"),
            ("median", "T2"),
            ("other", "T2"),
        }
        borderline = next(row for row in rows if (row.metric, row.group_id) == ("median", "T1"))
        assert borderline.p_value() == pytest.approx(0.19334790449564246)
        assert all(row.family_axes == ("metric", "arm") for row in rows)


def test_equal_quantiles_are_not_selected_at_the_largest_family_threshold():
    import math

    import pyarrow as pa

    analysis = inc.Analysis.from_unit_summary(
        pa.table(
            {
                "unit": list(range(200)),
                "arm": ["C"] * 100 + ["T"] * 100,
                "outcome": np.tile(np.arange(1.0, 101.0), 2),
            }
        ),
        unit="unit",
        group="arm",
        control="C",
        metrics=[
            inc.MetricSpec(name="median", type="quantile", value_column="outcome", quantile=0.5)
        ],
        plan=inc.AnalysisPlan(secondaries=["median"], q=math.nextafter(1.0, 0.0)),
    )
    row = lift_rows(analysis.run())[0]
    assert row.lift is not None
    assert row.lift.lb is not None and row.lift.ub is not None
    assert row.lift.lb < 0.0 < row.lift.ub
    assert row.p_value() == 1.0
    assert row.discovery is False
