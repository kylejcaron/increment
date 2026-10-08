"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from typing import Any, cast

import narwhals as nw
import pyarrow as pa
import pytest

from increment import readouts
from increment.errors import InvalidRequestError
from increment.estimation.diagnostics import NotApplicable, SRMResult
from increment.estimation.engine import Method
from increment.frame import (
    FramePanelSource,
    FrameTotalsSource,
    MetricSpec,
    from_unit_panel,
    from_unit_summary,
)
from increment.semantics.design import Randomized
from increment.semantics.models import Metric


@pytest.mark.parametrize(
    "module_name",
    (
        "increment.frame",
        "increment._frame_validation",
        "increment._frame_moments",
        "increment._frame_panel",
    ),
)
def test_frame_modules_keep_query_free_import_boundary(module_name: str) -> None:
    code = (
        f"import sys; import {module_name}; "
        "assert 'ibis' not in sys.modules; "
        "assert not any(name == 'increment.query' or name.startswith('increment.query.') "
        "for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


_PLUGIN_TOL = 1e-6


_DESIGN = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
)

_CANCELLING_REFS = {"cy1": "ref_y", "cx1": "ref_x", "cden1": "ref_den"}


def _assert_moments_agree(got: Any, exp: Any, cols: tuple[str, ...]) -> None:
    """Assert two centered-moment rows agree on *cols* at rel=1e-9.

    Cancelling first moments get an absolute tolerance scaled to n*|ref|,
    which pins the recovered raw sum n*ref + c1 at that same 1e-9.
    """
    for col in cols:
        if col in _CANCELLING_REFS:
            scale = 1e-9 * max(1.0, exp["n"] * abs(exp[_CANCELLING_REFS[col]]))
            assert got[col] == pytest.approx(exp[col], abs=scale), col
        else:
            assert got[col] == pytest.approx(exp[col], rel=1e-9), col


def _sum_y(row: Any) -> float:
    """Recover sum(y) from a moments row: n*ref_y + cy1, exact and
    cancellation-free."""
    return row["n"] * row["ref_y"] + row["cy1"]


def _sum_x(row: Any) -> float:
    """Recover sum(x) from a moments row: n*ref_x + cx1."""
    return row["n"] * row["ref_x"] + row["cx1"]


def _sum_den(row: Any) -> float:
    """Recover sum(den) from a moments row: n*ref_den + cden1."""
    return row["n"] * row["ref_den"] + row["cden1"]


def _sum_yd(row: Any) -> float:
    """Recover sum(d*y) from a moments row: cyd + sum_d*ref_y."""
    return row["cyd"] + row["sum_d"] * row["ref_y"]


def _sum_y2d(row: Any) -> float:
    """Recover sum(d*y**2): cy2d + 2*ref_y*cyd + sum_d*ref_y**2."""
    return row["cy2d"] + 2.0 * row["ref_y"] * row["cyd"] + row["sum_d"] * row["ref_y"] ** 2


_ROWS = [
    # `pre_balanced` adds a zero-sum within-arm perturbation: arm means stay
    # balanced, but the covariate is no longer affine in `revenue`, whose exact
    # within-arm collinearity would leave a refused zero-variance residual.
    # unit,  variant,     revenue, converted, pre_revenue, orders, pre_balanced
    ("u01", "control", 22.40, 1.0, 9.10, 3.0, 18.3333),
    ("u02", "control", 10.00, 0.0, 3.75, 1.0, 1.9333),
    ("u03", "control", 41.05, 1.0, 28.60, 5.0, 35.9833),
    ("u04", "control", 14.20, 1.0, 5.05, 2.0, 7.1333),
    ("u05", "control", 10.00, 0.0, 0.00, 1.0, 5.4333),
    ("u06", "control", 28.75, 1.0, 15.20, 4.0, 21.1833),
    ("u07", "treatment", 32.10, 1.0, 8.90, 4.0, 17.8000),
    ("u08", "treatment", 15.60, 1.0, 4.10, 2.0, 4.3000),
    ("u09", "treatment", 51.30, 1.0, 30.15, 7.0, 40.5000),
    ("u10", "treatment", 10.00, 0.0, 2.20, 1.0, -4.8000),
    ("u11", "treatment", 37.85, 1.0, 16.40, 5.0, 24.0500),
    ("u12", "treatment", 19.95, 1.0, 6.75, 3.0, 8.1500),
]


_COLUMNS = [
    "user_id",
    "variant",
    "revenue",
    "converted",
    "pre_revenue",
    "orders",
    "pre_balanced",
]


def _arrow_table() -> pa.Table:
    cols = list(zip(*_ROWS, strict=True))
    return pa.table(dict(zip(_COLUMNS, cols, strict=True)))


@pytest.fixture
def arrow_frame() -> pa.Table:
    return _arrow_table()


def _backend_frames() -> list[tuple[str, Any]]:
    """One frame per installed backend, for cross-backend equivalence."""
    frames: list[tuple[str, Any]] = [("pyarrow", _arrow_table())]
    if importlib.util.find_spec("pandas") is not None:
        frames.append(("pandas", _arrow_table().to_pandas()))
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.append(("polars", pl.from_arrow(_arrow_table())))
    return frames


def test_moments_match_group_summary(arrow_frame: pa.Table) -> None:
    """The narwhals moments equal builders.group_summary, every column.

    This is the whole justification for reimplementing the arithmetic: it is
    a reference mean plus centered sums per family, with no decision encoded,
    and the equivalence is directly assertable.  If this drifts, the frame
    path is wrong.
    """
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")
    from increment.query.builders import group_summary

    # Reshape the fixture into the UNIT_TOTALS column set the builder expects.
    unit_totals_rows = {
        "unit_id": [r[0] for r in _ROWS],
        "experiment_id": ["frame"] * len(_ROWS),
        "group_id": [r[1] for r in _ROWS],
        "metric": ["revenue"] * len(_ROWS),
        "y": [r[2] for r in _ROWS],
        "x": [r[4] for r in _ROWS],
        "y_den": [r[5] for r in _ROWS],
    }
    con = ibis.duckdb.connect()
    totals = ibis.memtable(unit_totals_rows)
    expected = con.to_pyarrow(group_summary(totals)).to_pylist()

    analysis = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                covariate="pre_revenue",
            )
        ],
    )
    actual = {r["group_id"]: r for r in analysis.raw_moments}

    assert len(expected) == 2
    for exp in expected:
        got = actual[exp["group_id"]]
        assert got["n"] == exp["n"]
        _assert_moments_agree(
            got,
            exp,
            (
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
                "cyden",
            ),
        )


def test_parity_uptake_moments_match_group_summary() -> None:
    """sum_d/cyd/cy2d parity between builders.group_summary (the
    ibis-backed warehouse path) and from_unit_summary's narwhals moments
    (the frame path) - the uptake-moment counterpart to
    ``test_moments_match_group_summary`` above, for the ``Encouragement``
    design's uptake moments.

    ``tests/test_parity.py`` is a different, unrelated regression gate (a
    JSON-baseline fixture proving a future spine refactor doesn't move
    numbers) - not this notebook/plan's warehouse-frame moment parity, so
    that coverage lives here instead, alongside its sibling gate.
    """
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")
    from increment.query.builders import group_summary

    # unit, variant, revenue, clicked - non-degenerate compliance in both arms
    # so sum_d/cyd/cy2d are all nonzero; u03 taking up leaves the remaining control takers straddling the arm reference exactly, so cyd cancels to 0 there.
    rows = [
        ("u01", "control", 5.0, 1),
        ("u02", "control", 6.0, 0),
        ("u03", "control", 7.0, 1),
        ("u04", "control", 8.0, 1),
        ("u05", "treatment", 1.0, 0),
        ("u06", "treatment", 2.0, 1),
        ("u07", "treatment", 3.0, 1),
        ("u08", "treatment", 4.0, 0),
        ("u09", "treatment", 9.5, 1),
        ("u10", "treatment", 0.0, 0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))

    unit_totals_rows = {
        "unit_id": [r[0] for r in rows],
        "experiment_id": ["frame"] * len(rows),
        "group_id": [r[1] for r in rows],
        "metric": ["revenue"] * len(rows),
        "y": [r[2] for r in rows],
        "x": [r[2] for r in rows],  # unused by this assertion, must be real floats
        "y_den": [1.0] * len(rows),  # unused by this assertion, must be real floats
        "d": [float(r[3]) for r in rows],
    }
    con = ibis.duckdb.connect()
    totals = ibis.memtable(unit_totals_rows)
    expected = {r["group_id"]: r for r in con.to_pyarrow(group_summary(totals)).to_pylist()}

    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
    )
    actual = {r["group_id"]: r for r in src.raw_moments}

    assert set(expected) == set(actual)
    for group_id, exp in expected.items():
        got = actual[group_id]
        _assert_moments_agree(got, exp, ("sum_d", "cyd", "cy2d"))


def test_frame_module_does_not_import_ibis() -> None:
    """`increment.frame` must work on a bare install with no backend."""
    import subprocess
    import sys

    code = (
        "import sys; import increment.frame; "
        "assert 'ibis' not in sys.modules, sorted(m for m in sys.modules if 'ibis' in m)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_mean_lift_matches_hand_computed_ratio_of_means(arrow_frame: pa.Table) -> None:
    control = [r[2] for r in _ROWS if r[1] == "control"]
    treatment = [r[2] for r in _ROWS if r[1] == "treatment"]
    expected_lift = (sum(treatment) / len(treatment)) / (sum(control) / len(control)) - 1.0

    results = readouts.run(
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=_DESIGN,
        )
    )

    assert len(results) == 1
    assert results[0].require_lift().value == pytest.approx(expected_lift, rel=_PLUGIN_TOL)


def test_ratio_matches_hand_computed_ratio_of_sums(arrow_frame: pa.Table) -> None:
    def ratio(arm: str) -> float:
        num = sum(r[2] for r in _ROWS if r[1] == arm)
        den = sum(r[5] for r in _ROWS if r[1] == arm)
        return num / den

    expected_lift = ratio("treatment") / ratio("control") - 1.0

    results = readouts.run(
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")
            ],
            design=_DESIGN,
        )
    )

    assert len(results) == 1
    assert results[0].require_lift().value == pytest.approx(expected_lift, rel=_PLUGIN_TOL)


def test_identical_estimates_across_backends() -> None:
    """pandas, polars and pyarrow must agree exactly."""
    frames = _backend_frames()
    assert len(frames) >= 2, f"need >=2 backends to compare, got {[n for n, _ in frames]}"

    values: dict[str, float] = {}
    for name, frame in frames:
        results = readouts.run(
            from_unit_summary(
                frame,
                unit="user_id",
                group="variant",
                control="control",
                metrics={"revenue": "mean", "converted": "conversion"},
                design=_DESIGN,
            )
        )
        values[name] = sum(r.require_lift().value for r in results)

    reference = next(iter(values.values()))
    for name, value in values.items():
        assert value == pytest.approx(reference, rel=1e-12), name


def test_value_column_reads_aliased_column(arrow_frame: pa.Table) -> None:
    """`value_column` decouples the metric name from the source column."""
    aliased = readouts.run(
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="rev", type="mean", value_column="revenue")],
            design=_DESIGN,
        )
    )
    canonical = readouts.run(
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=_DESIGN,
        )
    )

    assert len(aliased) == len(canonical) == 1
    assert aliased[0].require_lift().value == pytest.approx(
        canonical[0].require_lift().value, rel=1e-12
    )


def test_multi_arm_produces_one_estimate_per_non_control_arm() -> None:
    """More than two groups: one LiftEstimate per non-control arm, none for control itself."""
    rows = [
        ("u1", "control", 10.0),
        ("u2", "control", 12.0),
        ("u3", "control", 8.0),
        ("u4", "treatment_a", 20.0),
        ("u5", "treatment_a", 18.0),
        ("u6", "treatment_a", 22.0),
        ("u7", "treatment_b", 5.0),
        ("u8", "treatment_b", 7.0),
        ("u9", "treatment_b", 6.0),
    ]
    cols = list(zip(*rows, strict=True))
    frame = pa.table(dict(zip(["user_id", "variant", "revenue"], cols, strict=True)))

    results = readouts.run(
        from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=Randomized(
                control_group="control",
                allocation={"control": 1 / 3, "treatment_a": 1 / 3, "treatment_b": 1 / 3},
            ),
        )
    )

    groups = {r.group_id for r in results}
    assert groups == {"treatment_a", "treatment_b"}
    assert len(results) == 2


@pytest.mark.parametrize(
    ("undeclared_arm", "expected_undeclared"),
    [("treatment_a", ("treatment_a",)), ("None", ("None",))],
)
def test_declared_allocation_refuses_observed_undeclared_arms(
    undeclared_arm: str, expected_undeclared: tuple[str, ...]
) -> None:
    rows: list[tuple[str, str | None, float]] = [
        ("u1", "control", 10.0),
        ("u2", "treatment", 12.0),
        ("u3", undeclared_arm, 8.0),
        ("u4", None, 9.0),
    ]
    cols = list(zip(*rows, strict=True))
    frame = pa.table(dict(zip(["user_id", "variant", "revenue"], cols, strict=True)))
    source = from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        on_unassigned="exclude",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
    )

    with pytest.raises(InvalidRequestError) as raised:
        readouts.run(source)

    assert raised.value.code == "readout.roster.undeclared_observed_arms"
    assert raised.value.context == {
        "declared_arms": ("control", "treatment"),
        "undeclared_arms": expected_undeclared,
        "analysis_population": "assigned",
    }


def test_empty_metrics_mapping_raises() -> None:
    """An empty terse mapping must not silently construct a zero-metric analysis.

    Regression: the emptiness guard originally lived only on the MetricSpec
    list branch of `coerce_metrics`; `metrics={}` skipped it entirely and
    surfaced a confusing "control_group not found" error out of `run()`
    instead of a clear message naming the real problem.
    """
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _arrow_table(), unit="user_id", group="variant", control="control", metrics={}
        )
    assert exc_info.value.code == "frame.metrics_empty_supply"
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _arrow_table(), unit="user_id", group="variant", control="control", metrics=[]
        )
    assert exc_info.value.code == "frame.metrics_empty_supply"


def _cuped_pair(frame: pa.Table, covariate: str):
    analysis = from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", covariate=covariate)],
        design=_DESIGN,
    )
    plain = readouts.run(analysis)[0]
    adjusted = readouts.run(
        analysis, decision_method=Method(name="cuped", variance_reduction="cuped")
    )[0]
    return plain, adjusted


def test_cuped_narrows_interval_without_moving_the_point(arrow_frame: pa.Table) -> None:
    """With arms balanced on the covariate, CUPED buys precision and nothing else.

    `pre_balanced` has identical arm means by construction, so there is no
    pre-period imbalance to correct: any movement in the point estimate would
    be the adjustment introducing bias.
    """
    plain, adjusted = _cuped_pair(arrow_frame, "pre_balanced")

    plain_width = plain.require_lift().ub - plain.require_lift().lb
    adjusted_width = adjusted.require_lift().ub - adjusted.require_lift().lb

    assert adjusted_width < plain_width
    assert adjusted.require_lift().value == pytest.approx(plain.require_lift().value, rel=0.05)


def test_cuped_corrects_pre_period_imbalance(arrow_frame: pa.Table) -> None:
    """With treatment ahead pre-experiment, CUPED must pull the lift down.

    Control mean `pre_revenue` is 10.28 against treatment's 11.42, so part of
    the naive lift is pre-existing difference rather than treatment effect.
    An adjustment that left the estimate alone here would not be doing its job.
    """
    plain, adjusted = _cuped_pair(arrow_frame, "pre_revenue")

    assert (
        adjusted.require_lift().ub - adjusted.require_lift().lb
        < plain.require_lift().ub - plain.require_lift().lb
    )
    assert adjusted.require_lift().value < plain.require_lift().value


def _ratio_cuped_table(n: int, seed: int, *, predictive: bool) -> pa.Table:
    """Two arms carrying the SAME multiset of covariate values, so the arms
    are exactly balanced on the covariate and no pre-period imbalance is
    available to correct: any movement in the point estimate would be the
    adjustment introducing bias rather than removing it."""
    import numpy as np

    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 1.5, n)
    columns: dict[str, list[Any]] = {
        name: [] for name in ("user_id", "variant", "revenue", "orders", "pre")
    }
    for variant, lift in (("control", 1.0), ("treatment", 1.05)):
        pre = rng.permutation(base)
        if predictive:
            orders = np.clip(6.0 + 0.5 * (pre - 5.0) + rng.normal(0.0, 1.0, n), 0.5, None)
            revenue = orders * (2.0 + 0.4 * (pre - 5.0)) * lift + rng.normal(0.0, 0.5, n)
        else:
            orders = np.clip(rng.normal(6.0, 1.0, n), 0.5, None)
            revenue = orders * rng.normal(2.0, 0.6, n) * lift
        for i in range(n):
            columns["user_id"].append(f"{variant}-{i}")
            columns["variant"].append(variant)
            columns["revenue"].append(float(revenue[i]))
            columns["orders"].append(float(orders[i]))
            columns["pre"].append(float(pre[i]))
    return pa.table(columns)


def _ratio_cuped_pair(table: pa.Table):
    analysis = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="rpo",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                covariate="pre",
            )
        ],
        design=_DESIGN,
    )
    plain = readouts.run(analysis, decision_method=Method(name="unadjusted"))[0]
    adjusted = readouts.run(
        analysis, decision_method=Method(name="cuped", variance_reduction="cuped")
    )[0]
    return plain, adjusted


def test_ratio_cuped_narrows_the_interval_without_moving_the_point() -> None:
    """The dataframe path carries a ratio metric's covariate moments -- both
    components' cross moments with it -- through to a CUPED-adjusted ratio
    read on the same interval an unadjusted ratio uses."""
    plain, adjusted = _ratio_cuped_pair(_ratio_cuped_table(1500, 909, predictive=True))

    plain_lift, adjusted_lift = plain.require_lift(), adjusted.require_lift()
    plain_width = plain_lift.ub - plain_lift.lb
    adjusted_width = adjusted_lift.ub - adjusted_lift.lb

    assert adjusted_width < 0.75 * plain_width
    assert adjusted.abs_se is not None and plain.abs_se is not None
    assert adjusted.abs_se < 0.75 * plain.abs_se
    assert adjusted_lift.value == pytest.approx(plain_lift.value, rel=1e-9)
    # The adjustment keeps the Welch-Satterthwaite reference the unadjusted
    # ratio earns, on both scales.
    assert adjusted.reference_kind == "t"
    assert adjusted.reference_df == pytest.approx(plain.reference_df, rel=0.01)
    assert adjusted.abs_reference_kind == "t"


def test_ratio_cuped_with_an_unrelated_covariate_changes_nothing_material() -> None:
    """A covariate that predicts neither component buys no precision and must
    not move the estimate: the adjusted ratio is consistent for the same
    estimand as the unadjusted one."""
    plain, adjusted = _ratio_cuped_pair(_ratio_cuped_table(1500, 909, predictive=False))

    assert adjusted.require_lift().value == pytest.approx(plain.require_lift().value, rel=1e-9)
    assert adjusted.abs_se == pytest.approx(plain.abs_se, rel=0.02)


def test_metric_spec_roles_drive_run_dispatch(arrow_frame: pa.Table) -> None:
    """Declared role fields drive ``run()`` per metric.

    A sibling metric with no override keeps the design default, and neither
    metric's methods leak onto the other.
    """
    analysis = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                covariate="pre_revenue",
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
            ),
            MetricSpec(name="orders", type="ratio", numerator="revenue", denominator="orders"),
        ],
        design=_DESIGN,
    )
    results = readouts.run(analysis)
    by_metric = {(r.metric, r.method) for r in results}
    assert ("revenue", "cuped") in by_metric
    assert ("revenue", "unadjusted") in by_metric
    assert ("orders", "unadjusted") in by_metric
    assert ("orders", "cuped") not in by_metric


def test_run_call_wide_methods_override_declared_per_metric(arrow_frame: pa.Table) -> None:
    """A call-wide ``methods=`` wins over declared role fields; the declared
    CUPED never runs when the caller narrows to unadjusted."""
    analysis = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                covariate="pre_revenue",
                decision_method=Method(name="cuped", variance_reduction="cuped"),
            )
        ],
        design=_DESIGN,
    )
    results = readouts.run(analysis, decision_method=Method(name="unadjusted"))
    assert {(r.metric, r.method) for r in results} == {("revenue", "unadjusted")}


def test_no_covariate_yields_none_not_zero(arrow_frame: pa.Table) -> None:
    """Regression: an unpopulated slot must be None, never an aggregated 0.0.

    narwhals `sum()` over an all-null column returns 0.0 on every backend,
    which would slip past the `is None` guards at cuped.py:50 and
    variance.py:92 and produce a misleading downstream failure.
    """
    moments = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    ).raw_moments

    for row in moments:
        assert row["ref_x"] is None
        assert row["cx1"] is None
        assert row["cx2"] is None
        assert row["cxy"] is None
        assert row["ref_den"] is None
        assert row["cden1"] is None
        assert row["cden2"] is None
        assert row["cyden"] is None
        # ...while the populated slots are real numbers.
        assert isinstance(row["ref_y"], float)
        assert not math.isnan(row["ref_y"])
        assert isinstance(row["cy2"], float)
        assert not math.isnan(row["cy2"])


def test_duplicate_units_are_refused() -> None:
    rows = [*_ROWS, ("u01", "control", 5.00, 1.0, 9.10, 2.0, 16.3333)]
    cols = list(zip(*rows, strict=True))
    frame = pa.table(dict(zip(_COLUMNS, cols, strict=True)))

    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )

    assert exc.value.code == "source.frame.duplicate_units"


def _backend_table(rows: list[tuple[Any, ...]], backend: str) -> Any:
    table = pa.table(dict(zip(_COLUMNS, zip(*rows, strict=True), strict=True)))
    if backend == "pandas":
        return table.to_pandas()
    if backend == "polars":
        import polars as pl

        return pl.from_arrow(table)
    return table


_DUPLICATE_U01 = ("u01", "control", 5.00, 1.0, 9.10, 2.0, 16.3333)


@pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
def test_reserved_group_label_refuses_before_duplicate_units(backend: str) -> None:
    """Group identity is validated before the unit scan: a reserved arm label
    wins over a duplicated unit in the same frame."""
    rows = [*_ROWS, _DUPLICATE_U01, ("u13", "(unassigned)", 8.00, 0.0, 1.00, 1.0, 2.0)]
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            _backend_table(rows, backend),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "frame.validation.group_label_reserved"


@pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
def test_null_unit_refuses_before_duplicate_units(backend: str) -> None:
    rows = [*_ROWS, _DUPLICATE_U01, (None, "treatment", 8.00, 0.0, 1.00, 1.0, 2.0)]
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            _backend_table(rows, backend),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "source.frame.null_identity"


@pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
def test_duplicate_units_refuse_before_a_missing_metric_value(backend: str) -> None:
    rows = [*_ROWS, _DUPLICATE_U01, ("u13", "treatment", None, 0.0, 1.00, 1.0, 2.0)]
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            _backend_table(rows, backend),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="error")],
        )
    assert exc.value.code == "source.frame.duplicate_units"


def test_missing_columns_are_all_reported_at_once(arrow_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            arrow_frame,
            unit="nope_unit",
            group="variant",
            control="control",
            metrics={"nope_metric": "mean"},
        )

    assert exc.value.code == "source.frame.metric_missing"
    missing = tuple(cast("Any", exc.value.context["missing"]))
    assert len(missing) == 2
    assert any("nope_unit" in entry for entry in missing)
    assert any("nope_metric" in entry for entry in missing)


def test_absent_control_is_refused(arrow_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="holdout",
            metrics={"revenue": "mean"},
        )

    assert exc.value.code == "source.frame.control_missing"


def test_cuped_without_covariate_raises_naming_the_metric(arrow_frame: pa.Table) -> None:
    analysis = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=_DESIGN,
    )

    with pytest.raises(InvalidRequestError) as exc:
        readouts.run(analysis, decision_method=Method(name="cuped", variance_reduction="cuped"))

    assert exc.value.code == "estimation.cuped.arm_no_covariate"
    assert exc.value.context["arm_metric"] == "revenue"


def test_panel_cuped_refuses_without_covariate(panel_frame: pa.Table) -> None:
    """The CUPED-without-covariate guard is estimation-internal (cuped_adjust), not
    panel-side: from_unit_panel's own construction-time guard only rejects a
    covariate= metric outright (_reject_covariates), so every panel metric always
    lacks a covariate by construction. Requesting cuped on a MEAN metric here still
    reaches cuped_adjust's "no covariate data" check for the one treatment arm
    present, which is unconditional in that sense - but the guard is no longer
    panel-specific code, and it is conditional on a treatment arm actually being
    reached (see test_panel_ratio_cuped_raises_naming_from_unit_summary and
    test_panel_control_only_cuped_returns_empty_without_raising for the two edge
    cases this test does not cover).
    """
    analysis = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        design=_DESIGN,
    )

    with pytest.raises(InvalidRequestError) as exc:
        readouts.run(analysis, decision_method=Method(name="cuped", variance_reduction="cuped"))

    assert exc.value.code == "estimation.cuped.arm_no_covariate"
    assert exc.value.context["arm_metric"] == "revenue"


def test_panel_ratio_cuped_refuses_without_covariate(panel_frame: pa.Table) -> None:
    """A panel RATIO metric refuses for the same reason a panel mean metric
    does -- no covariate was materialised. A ratio metric is no longer
    refused for being a ratio, so the refusal is the missing covariate,
    not the type.
    """
    analysis = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")],
        design=_DESIGN,
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.run(analysis, decision_method=Method(name="cuped", variance_reduction="cuped"))

    assert exc_info.value.code == "estimation.cuped.arm_no_covariate"
    assert exc_info.value.context["arm_metric"] == "aov"


def test_panel_control_only_run_refuses_instead_of_returning_empty() -> None:
    """A control-only panel has no treatment arm to compare against - run()
    refuses rather than silently reporting no rows."""
    control_only_rows = [r for r in _PANEL_ROWS if r[1] == "control"]
    analysis = from_unit_panel(
        _panel_table(control_only_rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        design=_DESIGN,
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.run(analysis, decision_method=Method(name="cuped", variance_reduction="cuped"))
    assert exc_info.value.code == "readout.arms.no_treatment"
    assert exc_info.value.context["observed_arms"] == ("control",)


def test_retention_type_names_the_supported_alternative() -> None:
    """The terse retention form refuses, pointing at the explicit MetricSpec
    form - it cannot carry the required threshold_days.
    """
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "retention"},
        )

    assert exc.value.code == "frame.metric_type_retention"
    assert exc.value.context["name"] == "revenue"


def test_ratio_without_denominator_raises() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="aov", type="ratio", numerator="revenue")
    assert exc_info.value.code == "frame.metric.type_ratio_but"
    assert exc_info.value.context["missing"] == "denominator"
    assert exc_info.value.context["name"] == "aov"


def test_summary_srm_coerces_non_string_group_keys() -> None:
    """Integer-labeled arms retain their counts in SRM."""
    rows = [
        ("u1", 0, 10.0),
        ("u2", 0, 12.0),
        ("u3", 0, 8.0),
        ("u4", 1, 20.0),
        ("u5", 1, 18.0),
        ("u6", 1, 22.0),
    ]
    cols = list(zip(*rows, strict=True))
    frame = pa.table(dict(zip(["user_id", "variant", "revenue"], cols, strict=True)))

    result = readouts.srm(
        from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="0",
            metrics={"revenue": "mean"},
            design=Randomized(
                control_group="0",
                allocation={"0": 0.5, "1": 0.5},
                allocation_scheme="independent",
            ),
        ),
        expected={"0": 0.5, "1": 0.5},
    )

    assert isinstance(result, SRMResult)
    assert result.observed == {"0": 3, "1": 3}
    assert result.is_srm is False


def test_srm_zero_fills_declared_missing_arm_for_always_valid_prefix() -> None:
    """A cumulative prefix may legitimately contain no treatment assignments yet."""
    frame = pa.table(
        {
            "user_id": [f"u{i}" for i in range(14)],
            "variant": ["control"] * 14,
            "revenue": [1.0] * 14,
        }
    )

    result = readouts.srm(
        from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=Randomized(
                control_group="control",
                allocation={"control": 0.5, "treatment": 0.5},
                allocation_scheme="independent",
            ),
        ),
    )
    assert isinstance(result, SRMResult)
    assert result.inference == "always_valid"
    assert result.observed == {"control": 14, "treatment": 0}
    assert result.is_srm is True


def test_srm_always_valid_missing_support_refuses_before_reading_source_counts() -> None:
    """Support is a precondition, not an error discovered by reading the source."""

    from types import SimpleNamespace

    class NeverReadSource:
        def __init__(self):
            self.context = SimpleNamespace(
                design=Randomized(
                    control_group="control",
                    allocation_scheme="independent",
                ),
                cluster=None,
            )

        def unit_counts(self) -> dict[str, int]:
            raise AssertionError("srm() read unit counts before validating support")

    with pytest.raises(InvalidRequestError) as exc:
        readouts.srm(cast("Any", NeverReadSource()))
    assert exc.value.code == "estimation.diagnostics.always_srm_predeclared"


def test_srm_without_assignment_scheme_is_not_applicable() -> None:
    result = readouts.srm(
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        ),
        expected={"control": 0.5, "treatment": 0.5},
    )
    assert isinstance(result, NotApplicable)
    assert result.check == "srm"
    assert result.reason.startswith("integrity.allocation_scheme_missing")


def test_srm_rejects_accounting_label_support_before_reading_source_counts() -> None:
    from types import SimpleNamespace

    class NeverReadSource:
        def __init__(self):
            self.context = SimpleNamespace(
                design=Randomized(
                    control_group="control",
                    allocation={
                        "control": 0.5,
                        "treatment": 0.5,
                        "(unassigned)": 0.1,
                    },
                    allocation_scheme="independent",
                ),
                cluster=None,
            )

        def unit_counts(self) -> dict[str, int]:
            raise AssertionError("srm() read unit counts before validating support")

    with pytest.raises(InvalidRequestError) as exc:
        readouts.srm(cast("Any", NeverReadSource()))
    assert exc.value.code == "estimation.diagnostics.expected_allocation_contain"
    assert "(unassigned)" in cast("Any", exc.value.context["accounting"])


def test_srm_zero_fill_preserves_unexpected_observed_unit_arm_for_strict_mismatch() -> None:
    """A declared-but-absent arm is filled without erasing an observed stray arm."""
    frame = pa.table(
        {
            "user_id": [*(f"c{i}" for i in range(14)), "h0"],
            "variant": ["control"] * 14 + ["holdout"],
            "revenue": [1.0] * 15,
        }
    )
    source = from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
    )

    with pytest.raises(InvalidRequestError) as exc:
        readouts.srm(source)
    assert exc.value.code == "estimation.diagnostics.expected_keys_do"
    assert "holdout" in cast("Any", exc.value.context["counts_keys"])


def test_srm_always_valid_requires_declared_allocation_but_fixed_keeps_equal_split() -> None:
    """Only fixed-look Pearson inference may infer its support from observations."""
    source = from_unit_summary(
        _arrow_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=Randomized(
            control_group="control",
            allocation_scheme="independent",
        ),
    )

    with pytest.raises(InvalidRequestError) as exc:
        readouts.srm(source)

    assert exc.value.code == "estimation.diagnostics.always_srm_predeclared"

    fixed = readouts.srm(source, inference="fixed")
    assert isinstance(fixed, SRMResult)
    assert fixed.observed == {"control": 6, "treatment": 6}
    assert fixed.log_e_value is None


def _revenue_spec() -> MetricSpec:
    return MetricSpec(name="revenue", type="mean")


def _metric(src: FrameTotalsSource | FramePanelSource, name: str) -> Metric:
    """Typed lookup into ``src.metrics`` - the protocol types it
    ``Sequence[object]`` (zero-dependency seam), so tests that need
    ``.name`` narrow it back explicitly rather than fighting ty at every
    call site."""
    return next(m for m in src.context.metrics if m.name == name)


def test_frame_totals_source_retains_unit_rows(arrow_frame: pa.Table) -> None:
    """I1 on the frame path: per-unit rows must survive construction."""
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )
    metric = _metric(src, "revenue")
    units = nw.from_native(src.unit_frame(metric), eager_only=True)
    assert units.shape[0] == arrow_frame.shape[0], "one row per unit must survive"
    assert {"unit_id", "group_id", "y"} <= set(units.columns)
    assert sorted(units["y"].to_list()) == sorted(r[2] for r in _ROWS)


def test_unit_frame_includes_requested_covariates(arrow_frame: pa.Table) -> None:
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )
    metric = _metric(src, "revenue")
    units = nw.from_native(src.unit_frame(metric, covariates=["pre_revenue"]), eager_only=True)
    assert "pre_revenue" in units.columns
    assert sorted(units["pre_revenue"].to_list()) == sorted(r[4] for r in _ROWS)


def test_unit_frame_checks_identity_after_metric_missingness():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "unit": pd.Series([1, "1", 2, "other"], dtype=object),
            "arm": ["control", "control", "treatment", "treatment"],
            "value": [None, 10.0, 20.0, 25.0],
        }
    )
    with pytest.warns(UserWarning):
        source = from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(name="dropped", value_column="value", missing="drop"),
                MetricSpec(name="filled", value_column="value", missing="zero"),
            ],
        )
    selected = nw.from_native(source.unit_frame(_metric(source, "dropped")))
    assert selected["unit_id"].to_list() == ["1", "2", "other"]
    with pytest.raises(InvalidRequestError) as caught:
        source.unit_frame(_metric(source, "filled"))
    assert caught.value.code == "estimation.crossfit.identity_collision"
    assert caught.value.context["canonical"] == "1"


def test_unit_frame_keeps_noncolliding_mixed_identities_distinct():
    import numpy as np
    import pandas as pd

    from increment.estimation.crossfit import fold_assignments

    frame = pd.DataFrame(
        {
            "unit": pd.Series([1, "two", 3, "four", 5, "six"], dtype=object),
            "arm": ["control"] * 3 + ["treatment"] * 3,
            "value": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )
    source = from_unit_summary(
        frame, unit="unit", group="arm", control="control", metrics=[MetricSpec(name="value")]
    )
    units = nw.from_native(source.unit_frame(_metric(source, "value")))
    ids = units["unit_id"].to_numpy()
    expected = np.array(["1", "two", "3", "four", "5", "six"])
    assert ids.tolist() == expected.tolist()
    np.testing.assert_array_equal(
        fold_assignments(ids, n_folds=2, seed=11),
        fold_assignments(expected, n_folds=2, seed=11),
    )


def test_unit_frame_unknown_covariate_raises_naming_it(arrow_frame: pa.Table) -> None:
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )
    metric = _metric(src, "revenue")
    with pytest.raises(InvalidRequestError) as exc_info:
        src.unit_frame(metric, covariates=["nope_col"])
    assert exc_info.value.code == "frame.frame_totals.unit_covariate_column"
    assert exc_info.value.context["missing"] == ("nope_col",)


def test_unit_frame_unknown_metric_raises() -> None:
    src = from_unit_summary(
        _arrow_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )

    class _Fake:
        name = "not_declared"

    with pytest.raises(InvalidRequestError) as exc_info:
        src.unit_frame(_Fake())  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "frame.frame_totals.metric_was_declared"
    assert exc_info.value.context["name"] == "not_declared"


def test_frame_totals_source_snapshots_moments_at_construction() -> None:
    """Caller mutations cannot alter subsequent readouts."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(8)],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "revenue": [10.0, 11.0, 9.0, 10.0, 11.0, 12.0, 10.0, 11.0],
        }
    )
    base = from_unit_summary(
        df, unit="unit", group="arm", control="control", metrics={"revenue": "mean"}
    )
    owned_moments = base.raw_moments
    source = FrameTotalsSource(
        frame=nw.from_native(df, eager_only=True),
        unit="unit",
        group="arm",
        moments=owned_moments,
        metrics=[MetricSpec(name="revenue")],
        control="control",
        experiment_id="alias-demo",
        design=base.design,
        plan=base.plan,
        synthesised_metrics=base.context.metrics,
    )
    before = readouts.run(source)[0].require_lift().value
    next(r for r in owned_moments if r["group_id"] == "treatment")["ref_y"] += 10.0
    after = readouts.run(source)[0].require_lift().value
    assert after == before
    owned_moments.clear()
    returned = source.raw_moments
    returned[0]["ref_y"] += 100.0
    returned.clear()
    assert readouts.run(source)[0].require_lift().value == before


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("undeclared_metric", "frame.frame_totals.moments_metric_undeclared"),
        ("missing_control", "frame.frame_totals.control_group_absent"),
    ],
)
def test_frame_totals_source_rejects_inconsistent_moments(mutation, code) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {"unit": ["u1", "u2"], "arm": ["control", "treatment"], "revenue": [1.0, 2.0]}
    )
    base = from_unit_summary(
        df, unit="unit", group="arm", control="control", metrics={"revenue": "mean"}
    )
    moments = base.raw_moments
    if mutation == "undeclared_metric":
        moments[0] = {**moments[0], "metric": "not_declared"}
    else:
        moments = [row for row in moments if row["group_id"] != "control"]
    with pytest.raises(InvalidRequestError) as exc:
        FrameTotalsSource(
            frame=nw.from_native(df, eager_only=True),
            unit="unit",
            group="arm",
            moments=moments,
            metrics=[MetricSpec(name="revenue")],
            control="control",
            experiment_id="undeclared-demo",
            design=base.design,
            plan=base.plan,
            synthesised_metrics=base.context.metrics,
        )
    assert exc.value.code == code


def test_unit_frame_ratio_metric_reads_numerator_as_y(arrow_frame: pa.Table) -> None:
    """A ratio metric's y_column is the numerator - unasserted before this test."""
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")],
    )
    metric = _metric(src, "aov")
    units = nw.from_native(src.unit_frame(metric), eager_only=True)
    assert sorted(units["y"].to_list()) == sorted(r[2] for r in _ROWS), "y must read the numerator"


def test_unit_frame_ratio_metric_also_serves_y_den(arrow_frame: pa.Table) -> None:
    """A ratio metric's unit_frame additionally carries y_den (the
    denominator) - y alone is the numerator, never the ratio itself."""
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")],
    )
    metric = _metric(src, "aov")
    units = nw.from_native(src.unit_frame(metric), eager_only=True)
    assert "y_den" in units.columns
    assert sorted(units["y_den"].to_list()) == sorted(r[5] for r in _ROWS)


def test_unit_frame_mean_metric_has_no_y_den(arrow_frame: pa.Table) -> None:
    """y_den is additive: a non-ratio metric's unit_frame stays exactly
    unit_id/group_id/y (plus requested covariates)."""
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )
    metric = _metric(src, "revenue")
    units = nw.from_native(src.unit_frame(metric), eager_only=True)
    assert "y_den" not in units.columns


def test_unit_frame_conversion_metric_reads_the_01_column_as_y(arrow_frame: pa.Table) -> None:
    """A conversion metric's y_column is its own 0/1 column - unasserted before this test."""
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    metric = _metric(src, "converted")
    units = nw.from_native(src.unit_frame(metric), eager_only=True)
    assert sorted(units["y"].to_list()) == sorted(r[3] for r in _ROWS)


def test_unit_frame_group_id_matches_moments_dtype() -> None:
    """Canonical arm keys align between unit data and moments."""
    rows = [
        ("u1", 0, 10.0),
        ("u2", 0, 12.0),
        ("u3", 1, 20.0),
        ("u4", 1, 18.0),
    ]
    cols = list(zip(*rows, strict=True))
    int_group_frame = pa.table(dict(zip(["user_id", "variant", "revenue"], cols, strict=True)))

    src = from_unit_summary(
        int_group_frame,
        unit="user_id",
        group="variant",
        control="0",
        metrics=[_revenue_spec()],
    )
    metric = _metric(src, "revenue")
    units = nw.from_native(src.unit_frame(metric), eager_only=True)
    moment_group_ids = {r["group_id"] for r in src.raw_moments}
    unit_frame_group_ids = set(units["group_id"].to_list())
    assert unit_frame_group_ids == moment_group_ids == {"0", "1"}


def test_public_exports_resolve() -> None:
    import increment as inc

    assert inc.MetricSpec is MetricSpec


_PANEL_ROWS = [
    # unit, variant,     day,   revenue, orders
    ("u1", "control", "d1", 4.0, 2.0),
    ("u1", "control", "d2", 8.0, 3.0),
    ("u1", "control", "d3", 0.0, 0.0),
    ("u2", "control", "d1", 0.0, 0.0),
    ("u2", "control", "d2", 3.0, 1.0),
    ("u2", "control", "d3", 6.0, 2.0),
    ("u3", "control", "d1", 2.0, 1.0),
    ("u3", "control", "d2", 2.0, 1.0),
    ("u3", "control", "d3", 2.0, 1.0),
    ("u4", "control", "d1", 10.0, 4.0),
    ("u4", "control", "d2", 0.0, 0.0),
    ("u4", "control", "d3", 5.0, 2.0),
    ("u5", "treatment", "d1", 6.0, 3.0),
    ("u5", "treatment", "d2", 6.0, 3.0),
    ("u5", "treatment", "d3", 6.0, 3.0),
    ("u6", "treatment", "d1", 12.0, 5.0),
    ("u6", "treatment", "d2", 0.0, 0.0),
    ("u6", "treatment", "d3", 3.0, 1.0),
    ("u7", "treatment", "d1", 4.0, 2.0),
    ("u7", "treatment", "d2", 9.0, 4.0),
    ("u7", "treatment", "d3", 4.0, 2.0),
    ("u8", "treatment", "d1", 0.0, 0.0),
    ("u8", "treatment", "d2", 5.0, 2.0),
    ("u8", "treatment", "d3", 8.0, 3.0),
]


_PANEL_COLUMNS = ["user_id", "variant", "day", "revenue", "orders"]


def _panel_table(rows=None) -> pa.Table:
    rows = _PANEL_ROWS if rows is None else rows
    cols = list(zip(*rows, strict=True))
    return pa.table(dict(zip(_PANEL_COLUMNS, cols, strict=True)))


@pytest.fixture
def panel_frame() -> pa.Table:
    return _panel_table()


class TestDegenerateRatioIsRefusedWhereColumnsAreRead:
    """Both sides summing one frame column gives a ratio identically 1. The
    check belongs where the names are read as columns: on the from_moments path
    they are labels on precomputed moments and may legitimately coincide (a
    semantic ratio over one fact with two aggregations)."""

    @staticmethod
    def _frame():
        return pa.table(
            {
                "unit_id": ["u1", "u2", "u3", "u4"],
                "variant": ["control", "control", "treatment", "treatment"],
                "amount": [1.0, 2.0, 3.0, 4.0],
                "orders": [1.0, 1.0, 2.0, 2.0],
            }
        )

    def test_the_same_column_on_both_sides_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            from_unit_summary(
                self._frame(),
                unit="unit_id",
                group="variant",
                control="control",
                metrics=[
                    MetricSpec(name="r", type="ratio", numerator="amount", denominator="amount")
                ],
            )
        assert exc_info.value.code == "source.frame.ratio_same_column"

    def test_different_columns_are_accepted(self):
        source = from_unit_summary(
            self._frame(),
            unit="unit_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="r", type="ratio", numerator="amount", denominator="orders")],
        )
        assert source.unit_counts() == {"control": 2, "treatment": 2}

    def test_the_spec_itself_still_accepts_coinciding_names(self):
        # from_moments carries these as labels, not columns.
        spec = MetricSpec(name="r", type="ratio", numerator="order", denominator="order")
        assert spec.numerator == spec.denominator
