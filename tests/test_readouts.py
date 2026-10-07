"""Tests for the one readout set (`increment.readouts`) over `MomentSource`.

The load-bearing test is `test_same_readout_serves_both_substrates`: the
whole point of the protocol is that one readout implementation, run against
two different `MomentSource` backends built from the same logical data,
reports the same number.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import ibis
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from increment.errors import (
    CapabilityError,
    DefinitionError,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.diagnostics import NotApplicable, SRMResult
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.frame import MetricSpec
from increment.semantics.design import AdjustmentSet, Observational, Randomized
from increment.semantics.models import AnalysisPlan, MeanMetric
from tests.test_sequential_public_sources import gaussian_plan

# 6 units, two arms, deliberately unbalanced revenue so a transposed group
# would show up; spread stays moderate so infer_lift's delta-method SE guard (>= 0.5) doesn't refuse the readout.
_ROWS = [
    # unit,  variant,     revenue
    ("u1", "control", 10.0),
    ("u2", "control", 20.0),
    ("u3", "control", 15.0),
    ("u4", "treatment", 30.0),
    ("u5", "treatment", 40.0),
    ("u6", "treatment", 25.0),
]
_COLUMNS = ["user_id", "variant", "revenue"]


def _unit_summary_table() -> pa.Table:
    cols = list(zip(*_ROWS, strict=True))
    return pa.table(dict(zip(_COLUMNS, cols, strict=True)))


@pytest.fixture
def unit_summary_frame() -> pa.Table:
    return _unit_summary_table()


@pytest.fixture(scope="session")
def con():
    return ibis.duckdb.connect()


def _revenue_spec() -> MetricSpec:
    return MetricSpec(name="revenue", type="mean")


def test_same_readout_serves_both_substrates(con, unit_summary_frame):
    """The whole point: one readout implementation, identical results to
    1e-12 relative tolerance across substrates (float summation order
    differs, so byte-identity is not the contract)."""
    from increment import readouts
    from increment.frame import FrameTotalsSource
    from increment.query.source import SqlPanelSource

    design = Randomized(control_group="control")
    frame_src = FrameTotalsSource.from_frame(
        unit_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
        design=design,
    )
    sql_src = SqlPanelSource.from_frame_via_memtable(
        con,
        unit_summary_frame,
        unit="user_id",
        group="variant",
        metrics=[_revenue_spec()],
        design=design,
    )

    a = readouts.run(frame_src)
    b = readouts.run(sql_src)
    assert len(a) == len(b) == 1
    assert a[0].require_lift().value == pytest.approx(b[0].require_lift().value, rel=1e-12)
    assert a[0].require_lift().lb == pytest.approx(b[0].require_lift().lb, rel=1e-12)
    assert a[0].require_lift().ub == pytest.approx(b[0].require_lift().ub, rel=1e-12)


@pytest.mark.slow
def test_memtable_missingness_keeps_enrollment_and_absent_denominator(con):
    from increment.frame import FrameTotalsSource
    from increment.query.source import SqlPanelSource

    frame = pa.table(
        {
            "unit": range(8),
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, None, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    metrics = [
        MetricSpec(name="dropped", type="conversion", value_column="value", missing="drop"),
        MetricSpec(name="filled", type="conversion", value_column="value", missing="zero"),
    ]
    with pytest.warns(UserWarning):
        native = FrameTotalsSource.from_frame(
            frame, unit="unit", group="arm", control="control", metrics=metrics
        )
    for declared in (metrics, metrics[::-1]):
        with pytest.warns(UserWarning):
            sql = SqlPanelSource.from_frame_via_memtable(
                con, frame, unit="unit", group="arm", metrics=declared
            )
        assert sql.unit_counts() == native.unit_counts() == {"control": 4, "treatment": 4}
        for metric in native.context.metrics:
            oracle = {row["group_id"]: row for row in native.moments(metric)}
            for row in sql.moments(metric):
                expected = oracle[row["group_id"]]
                for field in ("n", "ref_y", "cy1", "cy2"):
                    assert row[field] == pytest.approx(expected[field])
                assert row["ref_den"] is None


def test_memtable_source_refuses_the_same_frame_from_unit_summary_refuses(con):
    """Reproduces the issue's own probe: a duplicated unit, a null group, and
    a NaN outcome. from_unit_summary refuses this frame outright (null group
    is unassigned by default); the memtable source must refuse identically,
    not silently admit a wrong n and a wrong second moment."""
    from increment.frame import FrameTotalsSource
    from increment.query.source import SqlPanelSource

    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],  # u1 duplicated
            "variant": ["control", "control", None, "treatment"],  # u2's group is null
            "revenue": [10.0, 10.0, 20.0, np.nan],  # u3's outcome is NaN
        }
    )
    design = Randomized(control_group="control")

    with pytest.raises(InvalidRequestError) as frame_error:
        FrameTotalsSource.from_frame(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[_revenue_spec()],
            design=design,
        )

    with pytest.raises(InvalidRequestError) as sql_error:
        SqlPanelSource.from_frame_via_memtable(
            con,
            frame,
            unit="user_id",
            group="variant",
            metrics=[_revenue_spec()],
            design=design,
        )

    assert sql_error.value.code == frame_error.value.code == "source.frame.unassigned"


def test_memtable_source_on_unassigned_exclude_surfaces_the_same_unit_counts_as_the_oracle(con):
    """Regression: the unassigned refusal's suggested route forward
    ("Pass on_unassigned='exclude' to from_frame_via_memtable") must name a
    parameter that actually exists, actually excludes the row, and surfaces
    the excluded count the same way the dataframe oracle does --
    FrameTotalsSource.unit_counts() reports it under UNASSIGNED_LABEL
    beside the SRM check, not silently dropped."""
    from increment.frame import FrameTotalsSource
    from increment.query.source import SqlPanelSource

    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "variant": ["control", None, "treatment"],  # u2's group is null
            "revenue": [10.0, 20.0, 30.0],
        }
    )
    design = Randomized(control_group="control")
    frame_src = FrameTotalsSource.from_frame(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
        design=design,
        on_unassigned="exclude",
    )
    sql_src = SqlPanelSource.from_frame_via_memtable(
        con,
        frame,
        unit="user_id",
        group="variant",
        metrics=[_revenue_spec()],
        design=design,
        on_unassigned="exclude",
    )
    assert sql_src.unit_counts() == frame_src.unit_counts()


def test_memtable_source_refuses_a_frame_missing_the_declared_control_arm(con):
    """Regression: design carries control_group even though this
    constructor has no `control` parameter of its own; a frame missing that
    arm must be refused, not silently admitted with a phantom-only SRM."""
    from increment.query.source import SqlPanelSource

    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "variant": ["treatment_a", "treatment_b"],  # no "control" row at all
            "revenue": [10.0, 20.0],
        }
    )
    design = Randomized(control_group="control")
    with pytest.raises(InvalidRequestError):
        SqlPanelSource.from_frame_via_memtable(
            con,
            frame,
            unit="user_id",
            group="variant",
            metrics=[_revenue_spec()],
            design=design,
        )


def test_run_plain_cuped_methods_have_unique_to_frame_rows():
    from increment import Analysis

    frame = pa.table(
        {
            "user_id": ["c1", "c2", "c3", "t1", "t2", "t3"],
            "variant": ["control"] * 3 + ["treatment"] * 3,
            "revenue": [10.0, 20.0, 15.0, 30.0, 40.0, 25.0],
            "pre_revenue": [9.0, 19.0, 14.0, 28.0, 39.0, 23.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", covariate="pre_revenue")],
    )
    results = analysis.run(
        decision_method=Method(name="plain"),
        sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
    )

    rows = cast(pd.DataFrame, results.to_frame())
    assert len(rows) == 2
    assert set(rows["method"]) == {"plain", "cuped"}
    assert rows["method"].is_unique


def test_run_explicit_empty_methods_preserves_no_rows(unit_summary_frame):
    from increment import readouts
    from increment.frame import FrameTotalsSource

    source = FrameTotalsSource.from_frame(
        unit_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
        design=Randomized(control_group="control"),
    )

    assert readouts.run(source, sensitivity_methods=())
    default_rows = readouts.run(source)
    assert [row.method for row in default_rows] == ["unadjusted"]


def test_sql_panel_source_resolves_plan_on_the_frame_path_like_from_unit_summary(
    con, unit_summary_frame
):
    """SqlPanelSource's frame adapter resolves a declared plan the same way
    FrameTotalsSource (from_unit_summary) does: a metric declared alongside
    an empty (no primary/guardrail) AnalysisPlan defaults to role="secondary"
    and is BH-eligible on the frame path."""
    from increment.query.source import SqlPanelSource
    from increment.semantics.models import AnalysisPlan

    design = Randomized(control_group="control")
    src = SqlPanelSource.from_frame_via_memtable(
        con,
        unit_summary_frame,
        unit="user_id",
        group="variant",
        metrics=[_revenue_spec()],
        design=design,
        plan=AnalysisPlan(),
    )

    assert src.context.plan.declared is True
    assert src.context.plan.procedures["revenue"].role == "secondary"
    assert src.context.plan.procedures["revenue"].family.member is True  # ty: ignore[unresolved-attribute]


def test_sql_source_applies_pooled_winsorization(con):
    from increment.query.source import SqlPanelSource

    table = pa.table(
        {
            "user_id": ["c1", "c2", "t1", "t2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 2.0, 100.0, 200.0],
        }
    )
    spec = MetricSpec(
        name="revenue",
        type="mean",
        winsorization={"upper_percentile": 0.75},
    )
    source = SqlPanelSource.from_frame_via_memtable(
        con,
        table,
        unit="user_id",
        group="variant",
        metrics=[spec],
    )

    metric = cast(MeanMetric, source.context.metrics[0])
    rows = {row["group_id"]: row for row in source.moments(metric)}
    assert rows["control"]["ref_y"] == pytest.approx(1.5)
    assert rows["treatment"]["ref_y"] == pytest.approx(112.5)
    assert rows["treatment"]["winsor_upper_bound"] == pytest.approx(125.0)
    assert rows["treatment"]["winsor_n_upper"] == 1


def _winsorized_totals_source(winsorization: dict[str, object], *, design=None, plan=None):
    from increment.frame import FrameTotalsSource

    return FrameTotalsSource.from_frame(
        _unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", winsorization=winsorization)],
        design=design,
        plan=plan,
    )


def test_run_duplicate_method_names_refuse_before_source_access(monkeypatch):
    from increment import readouts

    source = _winsorized_totals_source(
        {"upper_value": 100.0},
        design=Randomized(control_group="control"),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("source moments accessed before validation")

    monkeypatch.setattr(source, "moments", fail)
    with pytest.raises(InvalidRequestError) as raised:
        readouts.run(
            source,
            decision_method=Method(name="same"),
            sensitivity_methods=(Method(name="same", variance_reduction="cuped"),),
        )
    assert raised.value.code == "estimation.engine.method_names_unique"
    assert raised.value.context["duplicates"] == ("same",)


def test_run_mixed_prior_method_scales_refuse_before_source_access(monkeypatch):
    from increment import readouts

    source = _winsorized_totals_source(
        {"upper_value": 100.0},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("revenue",)),
        ),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("source moments accessed before validation")

    monkeypatch.setattr(source, "moments", fail)
    with pytest.raises(InvalidRequestError) as raised:
        readouts.run(
            source,
            decision_method=Method(name="unadjusted"),
            sensitivity_methods=(Method(name="iptw"),),
            prior=Normal(mu=0.0, sigma=0.05),
        )
    assert raised.value.code == "estimation.adjust.prior.method_scale"


def test_run_refuses_a_single_arm_source_instead_of_returning_empty_rows() -> None:
    from increment import readouts
    from increment.frame import from_unit_summary

    frame = pd.DataFrame(
        {"unit_id": ["u1", "u2", "u3"], "variant": ["a", "a", "a"], "rev": [1.0, 2.0, 3.0]}
    )
    src = from_unit_summary(
        frame, unit="unit_id", group="variant", control="a", metrics={"rev": "mean"}
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.run(src)
    assert exc_info.value.code == "readout.arms.no_treatment"
    assert exc_info.value.context["observed_arms"] == ("a",)

    assert readouts.srm(src, expected={"a": 0.5, "b": 0.5})


def _two_metric_rows(*, second_metric_has_treatment: bool) -> tuple[list[dict], Any, Any]:
    """A moments cube with two metrics sharing one decision plan: ``a`` is
    always control-only, ``b`` optionally carries a treatment arm too."""
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    metric_a = MeanMetric(name="a", entity="user_id", fact="a")
    metric_b = MeanMetric(name="b", entity="user_id", fact="b")
    payload = compiled_plan_to_json(compile_decision_plan(None, [metric_a, metric_b]))

    def _row(metric_name: str, group_id: str, sum_y: float) -> dict:
        return {
            **centered_row_from_raw_sums(
                {
                    "experiment_id": "e",
                    "metric": metric_name,
                    "group_id": group_id,
                    "n": 100,
                    "sum_y": sum_y,
                    "sum_y2": sum_y + 50.0,
                }
            ),
            "moments_format": 7,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "decision_plan": payload,
        }

    rows = [_row("a", "control", 100.0), _row("b", "control", 100.0)]
    if second_metric_has_treatment:
        rows.append(_row("b", "treatment", 102.0))
    return rows, metric_a, metric_b


def test_run_unions_observed_arms_across_selected_metrics_before_refusing():
    """Metric ``a`` alone has no treatment arm, but ``b`` does - the gate
    must union arms across every selected metric, not just the first, so
    ``b``'s row is still reported instead of a spurious refusal."""
    from increment import readouts
    from increment.sources import MomentsSource

    design = Randomized(control_group="control")
    rows, metric_a, metric_b = _two_metric_rows(second_metric_has_treatment=True)
    src = MomentsSource(rows, metrics=[metric_a, metric_b], study_id="e", design=design)
    results = readouts.run(src)
    assert {r.metric for r in results} == {"b"}


def test_run_refuses_when_no_selected_metric_has_a_treatment_arm():
    """The converse: neither metric carries a non-control arm, so the
    union is exactly {control} and the readout still refuses."""
    from increment import readouts
    from increment.sources import MomentsSource

    design = Randomized(control_group="control")
    rows, metric_a, metric_b = _two_metric_rows(second_metric_has_treatment=False)
    src = MomentsSource(rows, metrics=[metric_a, metric_b], study_id="e", design=design)
    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.run(src)
    assert exc_info.value.code == "readout.arms.no_treatment"
    assert exc_info.value.context["observed_arms"] == ("control",)


def test_control_only_percentile_metric_uses_the_aggregate_arm_gate():
    from increment import readouts
    from increment.frame import from_unit_summary
    from increment.semantics.models import Winsorization

    with pytest.warns(UserWarning):
        source = from_unit_summary(
            pa.table(
                {
                    "u": list(range(6)),
                    "g": ["C"] * 3 + ["T"] * 3,
                    "winsor": [1.0, 2.0, 3.0, None, None, None],
                    "plain": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                }
            ),
            unit="u",
            group="g",
            control="C",
            metrics=[
                MetricSpec(
                    name="winsor",
                    missing="drop",
                    winsorization=Winsorization(upper_percentile=0.95),
                ),
                MetricSpec(name="plain"),
            ],
        )
    (result,) = readouts.run(source)
    assert result.metric == "plain"
    assert result.require_lift().value == pytest.approx(1.0)
    with pytest.raises(InvalidRequestError) as refused:
        readouts.run(source, metrics=("winsor",))
    assert refused.value.code == "readout.arms.no_treatment"
    assert refused.value.context["observed_arms"] == ("C",)


def test_percentile_winsorization_refuses_sequential_run():
    from increment.frame import FrameTotalsSource, synthesise_metric
    from tests.sequential_cases import declared_plan

    spec = MetricSpec(name="revenue", winsorization={"upper_percentile": 0.99})
    design = Randomized(control_group="control")
    plan = declared_plan(
        [synthesise_metric(spec)],
        source_id="frame",
        design=design,
        transformations=[spec],
    )

    from tests.sequential_cases import UnreadFrame

    with pytest.raises(CapabilityError) as raised:
        FrameTotalsSource.from_frame(
            UnreadFrame(),
            unit="unit",
            group="variant",
            control="control",
            metrics=[spec],
            plan=plan,
        )
    assert raised.value.code == "sequential.transform.unpredictable"


def test_percentile_winsorization_requires_support_for_fixed_horizon_run():
    from increment import readouts

    source = _winsorized_totals_source(
        {"upper_percentile": 0.99, "inference": {"method": "joint-rank-projection-v1"}}
    )
    with pytest.raises(CapabilityError) as exc_info:
        readouts.run(source)
    assert exc_info.value.code == "estimation.winsor.support_required"


def test_fixed_winsorization_refuses_without_clipped_sampling_proof():
    from increment.frame import FrameTotalsSource, synthesise_metric
    from tests.sequential_cases import declared_plan

    spec = MetricSpec(name="revenue", winsorization={"upper_value": 100.0})
    design = Randomized(control_group="control")
    plan = declared_plan(
        [synthesise_metric(spec)],
        source_id="frame",
        design=design,
        transformations=[spec],
    )

    from tests.sequential_cases import UnreadFrame

    with pytest.raises(CapabilityError) as raised:
        FrameTotalsSource.from_frame(
            UnreadFrame(),
            unit="unit",
            group="variant",
            control="control",
            metrics=[spec],
            plan=plan,
        )
    assert raised.value.code == "sequential.route.unsupported"


def test_percentile_winsorization_refuses_observational_run():
    from increment import readouts

    source = _winsorized_totals_source(
        {"upper_percentile": 0.99},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("revenue",))
        ),
    )
    with pytest.raises(CapabilityError) as exc_info:
        readouts.run(source, sensitivity_methods=())
    assert exc_info.value.code == "readout.metric.percentile_winsorization"


def test_daily_allows_unwinsorized_panel_values():
    from increment import readouts
    from increment.frame import from_unit_panel

    source = from_unit_panel(
        pa.table(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "variant": ["control", "control", "treatment", "treatment"],
                "day": [1, 2, 1, 2],
                "revenue": [10.0, 12.0, 11.0, 15.0],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )

    assert readouts.daily(source)


def test_daily_refuses_fixed_winsorized_panel_values():
    from increment import readouts
    from increment.frame import from_unit_panel

    source = from_unit_panel(
        pa.table(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "variant": ["control", "control", "treatment", "treatment"],
                "day": [1, 2, 1, 2],
                "revenue": [10.0, 12.0, 11.0, 15.0],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_value": 100.0})],
    )

    with pytest.raises(CapabilityError) as raised:
        readouts.daily(source)
    assert raised.value.code == "readout.metric.daily_winsorization"


def test_winsorization_refuses_direct_daily_readout():
    from increment import readouts

    source = _winsorized_totals_source({"upper_value": 100.0})
    with pytest.raises(CapabilityError) as exc_info:
        readouts.daily(source)
    assert exc_info.value.code == "readout.metric.daily_winsorization"


def test_daily_skips_inferential_plan_validation():
    from increment import readouts
    from increment.frame import from_unit_panel

    source = from_unit_panel(
        pa.table(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "variant": ["control", "control", "treatment", "treatment"],
                "day": [1, 2, 1, 2],
                "revenue": [0, 0, 1, 1],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="conversion")],
        plan=gaussian_plan(
            [MetricSpec(name="revenue", type="conversion")],
            law="bernoulli",
            date="day",
            exposure_date=None,
        ),
    )

    assert readouts.daily(source)


def test_daily_refuses_on_a_totals_source(unit_summary_frame):
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        unit_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )
    with pytest.raises(CapabilityError) as exc_info:
        readouts.daily(src)
    assert exc_info.value.code == "readout.source.grain"


def test_srm_not_applicable_on_a_non_randomized_design(unit_summary_frame):
    """srm's control_group concept is meaningless outside a randomized
    design - a non-randomized comparison returns NotApplicable rather
    than raising, since it is a real, expected design shape, not caller
    error."""
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        unit_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
        design=Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x",))),
    )
    result = readouts.srm(src)
    assert isinstance(result, NotApplicable)
    assert result.check == "srm"
    assert "not randomized" in result.reason


def test_srm_reports_the_observed_group_counts(unit_summary_frame):
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        unit_summary_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
    )
    result = readouts.srm(src)
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 3, "treatment": 3}


def test_cluster_counts_refuses_on_a_moments_source():
    """The wire format carries no cluster marker (which is why export()
    refuses a clustered experiment), so this source must refuse by name
    rather than hand back unit counts under a cluster-grain name."""
    import json

    from increment.semantics.models import MeanMetric
    from increment.sources import ASSIGNMENT_COUNTS_FIELD, MomentsSource

    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    assignment_counts = json.dumps({"control": 100, "treatment": 100})
    rows = [
        {
            **raw_row,
            "moments_format": 7,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            ASSIGNMENT_COUNTS_FIELD: assignment_counts,
        }
        for raw_row in _rows_for("revenue")
    ]
    src = MomentsSource(rows, metrics=[metric], study_id="e")
    assert src.unit_counts() == {"control": 100, "treatment": 100}
    with pytest.raises(CapabilityError) as exc_info:
        src.cluster_counts()
    assert exc_info.value.code == "source.moments.cluster_grain"


def test_cluster_counts_refuses_on_a_sql_summary_source(con, unit_summary_frame):
    from increment.query.source import SqlPanelSource

    src = SqlPanelSource.from_frame_via_memtable(
        con,
        unit_summary_frame,
        unit="user_id",
        group="variant",
        metrics=[_revenue_spec()],
    )
    with pytest.raises(CapabilityError) as exc_info:
        src.cluster_counts()
    assert exc_info.value.code == "source.sql.cluster_grain"


def test_breakout_preserves_breakout_estimates_type():
    """breakout() is annotated -> BreakoutEstimates, and both its return
    statements (increment/readouts/_breakout.py) construct one - this test
    pins that the dimensioned path never silently degrades to a plain list."""
    from increment import readouts
    from increment.breakout.estimates import BreakoutEstimates
    from increment.semantics.models import MeanMetric
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(name="revenue", entity="user_id", fact="orders", aggregation="sum")
    rows = [
        {
            "experiment_id": "s",
            "metric": "revenue",
            "group_id": gid,
            "plan": plan,
            "n": n,
            "sum_y": sum_y,
            "sum_y2": sum_y2,
        }
        for plan, gid, n, sum_y, sum_y2 in [
            ("basic", "control", 50, 500.0, 5100.0),
            ("basic", "treatment", 50, 600.0, 7300.0),
            ("pro", "control", 50, 800.0, 13000.0),
            ("pro", "treatment", 50, 1000.0, 20500.0),
        ]
    ]
    design = Randomized(control_group="control")
    src = FakeMomentSource(
        [centered_row_from_raw_sums(r) for r in rows],
        metrics=[metric],
        capabilities={"total"},
        design=design,
        breakouts=("plan",),
    )

    out = readouts.breakout(src, "plan")
    assert isinstance(out, BreakoutEstimates)


def _quantile_unit_frame(n: int = 200) -> pd.DataFrame:
    """A larger per-unit frame - big enough that the q=0.9 order-stat
    interval does not collapse (see log_quantile_se's 'too small' guard)."""
    rng = np.random.default_rng(23)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "revenue": rng.lognormal(1.0, 0.5, 2 * n),
            "lat": rng.lognormal(1.0, 0.5, 2 * n),
        }
    )


def test_run_routes_quantile_through_unit_frame():
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="lat", type="quantile", quantile=0.9),
        ],
    )
    results = readouts.run(src)
    assert {r.metric for r in results} == {"revenue", "lat"}
    assert {r.method for r in results if r.metric == "lat"} == {"unadjusted"}


def test_run_quantile_secondary_control_only_source_refuses_no_treatment():
    from increment import readouts
    from increment.errors import InvalidRequestError
    from increment.frame import FrameTotalsSource

    frame = _quantile_unit_frame().assign(variant="control")
    src = FrameTotalsSource.from_frame(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="lat", type="quantile", quantile=0.9),
        ],
        plan=AnalysisPlan(primary="revenue", secondaries=["lat"]),
    )
    with pytest.raises(InvalidRequestError) as raised:
        readouts.run(src, sensitivity_methods=())
    assert raised.value.code == "readout.arms.no_treatment"
    assert raised.value.context["observed_arms"] == ("control",)


def test_run_quantile_primary_preserves_explicit_empty_methods():
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="lat", type="quantile", quantile=0.9)],
        plan=AnalysisPlan(primary="lat"),
    )

    assert readouts.run(src, sensitivity_methods=())


def test_run_quantile_secondary_nominal_preserves_explicit_empty_methods():
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="lat", type="quantile", quantile=0.9),
        ],
        plan=AnalysisPlan(primary="revenue", secondaries=["lat"]),
    )

    assert readouts.run(src, sensitivity_methods=())


def test_run_quantile_secondary_fcr_preserves_explicit_empty_methods():
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="lat", type="quantile", quantile=0.9),
        ],
        plan=AnalysisPlan(primary="revenue", secondaries=["lat"]),
    )

    assert readouts.run(src, sensitivity_methods=())


def test_run_quantile_with_by_refused():
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="lat", type="quantile", quantile=0.9)],
    )
    with pytest.raises(CapabilityError) as raised:
        readouts.run(src, by=["country"])

    error = raised.value
    assert error.code == "readout.metric.quantile_breakout"
    assert error.context == {"metric": "lat"}


def test_run_quantile_with_alternative_refused():
    from increment import readouts
    from increment.frame import FrameTotalsSource
    from increment.semantics.models import AnalysisPlan

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="lat", type="quantile", quantile=0.9)],
        plan=AnalysisPlan(alternative="greater"),
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        readouts.run(src)

    error = raised.value
    assert error.code == "readout.metric.quantile_alternative"
    assert error.context == {"metric": "lat"}


def test_quantile_cuped_refusal_is_coded() -> None:
    """A quantile metric cannot carry CUPED: run() refuses by code before any
    moments are read, naming the metric it refused."""
    from increment import readouts
    from increment.frame import FrameTotalsSource

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="lat", type="quantile", quantile=0.9)],
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        readouts.run(
            src,
            decision_method=Method(name="unadjusted"),
            sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
        )
    assert raised.value.code == "arm.metric.quantile_cuped"
    assert raised.value.context == {"metric": "lat"}


@pytest.mark.filterwarnings(
    "ignore:.*quantile metric.*fixed-horizon order-statistic bracket:UserWarning"
)
def test_run_quantile_refuses_always_valid_before_frame_access():
    from increment.frame import FrameTotalsSource, synthesise_metric
    from tests.sequential_cases import declared_plan

    spec = MetricSpec(name="lat", type="quantile", quantile=0.9)
    design = Randomized(control_group="control")
    plan = declared_plan(
        [synthesise_metric(spec)],
        source_id="frame",
        design=design,
        transformations=[spec],
    )

    from tests.sequential_cases import UnreadFrame

    with pytest.raises(CapabilityError) as raised:
        FrameTotalsSource.from_frame(
            UnreadFrame(),
            unit="unit",
            group="variant",
            control="control",
            metrics=[spec],
            plan=plan,
        )
    assert raised.value.code == "sequential.route.unsupported"


def test_run_quantile_with_null_lifts_refused():
    """A guardrail margin (the plan-based shifted-null mechanism)
    isolates the same quantile-specific one-sided refusal the old raw
    null_lifts= escape hatch used to."""
    from increment import readouts
    from increment.frame import FrameTotalsSource
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="lat", type="quantile", quantile=0.9, preferred_direction="decrease")
        ],
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="lat", margin=0.01)]),
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        readouts.run(src)
    assert raised.value.code == "readout.metric.quantile_alternative"


def test_run_margins_on_a_synthesized_metric_raises(unit_summary_frame):
    """The frame entry point's synthesized metrics never have an explicitly
    declared preferred_direction (documented limitation) - a plan-bound
    margin must raise at plan-resolution time (construction), not silently
    guess the adverse side."""
    from increment.frame import FrameTotalsSource
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    with pytest.raises(DefinitionError) as raised:
        FrameTotalsSource.from_frame(
            unit_summary_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[_revenue_spec()],
            plan=AnalysisPlan(primary=ExperimentMetric(metric="revenue", margin=0.01)),
        )
    assert raised.value.code == "definition.metric_margin_requires_direction"


def _rows_for(metric_name: str) -> list[dict]:
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    payload = compiled_plan_to_json(
        compile_decision_plan(
            None,
            [MeanMetric(name=metric_name, entity="user_id", fact=metric_name)],
        )
    )
    return [
        {
            **centered_row_from_raw_sums(
                {
                    "experiment_id": "e",
                    "metric": metric_name,
                    "group_id": "control",
                    "n": 100,
                    "sum_y": 100.0,
                    "sum_y2": 150.0,
                }
            ),
            "moments_format": 7,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "decision_plan": payload,
        },
        {
            **centered_row_from_raw_sums(
                {
                    "experiment_id": "e",
                    "metric": metric_name,
                    "group_id": "treatment",
                    "n": 100,
                    "sum_y": 102.0,
                    "sum_y2": 160.0,
                }
            ),
            "moments_format": 7,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "decision_plan": payload,
        },
    ]


def test_run_stamps_none_preferred_direction_when_metric_never_declared_one():
    """The bug: a warehouse metric that never sets preferred_direction must
    not silently inherit MetricIdentity's "increase" default onto the
    readout - preferred_direction stays None so prob_favorable() correctly
    refuses instead of assuming an "increase" direction."""
    from increment import readouts
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    design = Randomized(control_group="control")
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    assert "preferred_direction" not in metric.model_fields_set  # never declared

    (est,) = readouts.run(
        MomentsSource(_rows_for("revenue"), metrics=[metric], study_id="e", design=design)
    )
    assert est.preferred_direction is None
    with pytest.raises(InvalidRequestError) as raised:
        est.prob_favorable()
    assert raised.value.code == "estimation.results.lift.liftestimate_prob_favorable"


def test_run_declared_margin_flips_stat_sig():
    """A metric declaring margin=0.5 (an enormous tolerance relative to
    this ~2% observed lift) is tested against H0: lift <= -50%, one-sided
    greater - trivially non-inferior, unlike the two-sided-vs-0 default
    which is not significant at this small a difference."""
    from increment import readouts
    from increment.plan import compile_decision_plan
    from increment.sources import MomentsSource

    design = Randomized(control_group="control")
    plain_metric = MeanMetric(
        name="revenue", entity="user_id", fact="revenue", aggregation="avg_event"
    )
    guardrail = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="avg_event",
        preferred_direction="increase",
        margin=0.5,
    )

    plain_src = MomentsSource(
        _rows_for("revenue"), metrics=[plain_metric], study_id="e", design=design
    )
    plain = readouts.run(plain_src)
    assert plain[0].null_lift == 0.0
    assert plain[0].alternative == "two-sided"

    guardrail_src = MomentsSource(
        _rows_for("revenue"),
        metrics=[guardrail],
        study_id="e",
        design=design,
        plan=compile_decision_plan(None, [guardrail]),
    )
    guardrailed = readouts.run(guardrail_src)
    assert guardrailed[0].null_lift == pytest.approx(-0.5)
    assert guardrailed[0].alternative == "greater"
    assert guardrailed[0].preferred_direction == "increase"
    # Non-inferior against a -50% tolerance is essentially certain here.
    assert guardrailed[0].prob_favorable() > 0.999
    lift = guardrailed[0].require_lift()
    assert lift.lb is not None and lift.lb > -0.5


def test_run_declared_margin_abs_reaches_the_estimate():
    """The declared margin_abs travels run -> estimate_lift -> infer_lift:
    LiftEstimate.null_abs is stamped, the additive endpoints are computed,
    and the readout decision layer decides on the abs interval - the
    wiring the interim refusal guarded until it existed."""
    from increment import readouts
    from increment.semantics.models import MeanMetric
    from increment.tables import estimates_to_readout
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="increase",
        margin_abs=0.5,
    )
    rows = [
        {
            "experiment_id": "s",
            "metric": "revenue",
            "group_id": "control",
            "n": 50,
            "sum_y": 500.0,
            "sum_y2": 5100.0,
        },
        {
            "experiment_id": "s",
            "metric": "revenue",
            "group_id": "treatment",
            "n": 50,
            "sum_y": 600.0,
            "sum_y2": 7300.0,
        },
    ]
    design = Randomized(control_group="control")
    src = FakeMomentSource(
        [centered_row_from_raw_sums(r) for r in rows],
        metrics=[metric],
        capabilities={"total"},
        design=design,
    )

    est = readouts.run(src)[0]
    assert est.null_abs == pytest.approx(-0.5)
    assert est.null_lift == 0.0
    assert est.alternative == "greater"
    assert est.abs_lb is not None and est.abs_ub is not None
    row = estimates_to_readout([est])[0]
    assert row["null_abs"] == pytest.approx(-0.5)
    # The decision is the abs interval vs null_abs, nothing else.
    assert row["stat_sig"] is (est.abs_lb > -0.5)
    assert row["stat_sig"] is True


def test_welch_absolute_margin_decides_on_its_own_t_reference():
    """A Welch-referenced row with a declared absolute margin decides on the
    additive sidecar's OWN Welch reference. The margin test and the margin
    interval must be cut from the same distribution, so both are t at
    ``abs_reference_df`` -- a Normal additive tail beside a t relative one
    would understate exactly the small-sample uncertainty the Welch reference
    was introduced to carry."""
    from scipy.stats import norm
    from scipy.stats import t as t_dist

    from increment import readouts
    from increment.semantics.models import MeanMetric
    from increment.tables import estimates_to_readout
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="orders",
        preferred_direction="increase",
        margin_abs=0.5,
    )
    rows = [
        {
            "experiment_id": "s",
            "metric": "revenue",
            "group_id": group,
            "n": 50,
            "ref_y": mean,
            "cy1": 0.0,
            "cy2": 100.0,
        }
        for group, mean in (("control", 10.0), ("treatment", 12.0))
    ]
    src = FakeMomentSource(
        rows,
        metrics=[metric],
        capabilities={"total"},
        design=Randomized(control_group="control"),
    )

    (est,) = readouts.run(src)

    assert est.reference_kind == "t" and est.reference_df is not None
    assert est.abs_reference_kind == "t"
    assert est.abs_reference_df == pytest.approx(98.0)
    assert est.null_abs == pytest.approx(-0.5)
    assert est.alternative == "greater"
    assert est.abs_diff is not None and est.abs_se is not None
    # alternative='greater' doubles alpha internally, so the stored endpoints
    # sit at the t_98 quantile for alpha_eff=0.10, not the Normal one.
    crit = t_dist.isf(0.05, est.abs_reference_df)
    assert crit > norm.isf(0.05)
    assert est.abs_lb == pytest.approx(est.abs_diff - crit * est.abs_se)
    assert est.abs_ub == pytest.approx(est.abs_diff + crit * est.abs_se)
    # The decision is the additive interval against the declared margin, and
    # it completes: the Welch row is no longer withheld from the margin test.
    assert est.abs_lb is not None
    readout = estimates_to_readout([est])[0]
    assert readout["stat_sig"] is (est.abs_lb > -0.5)
    assert readout["stat_sig"] is True


def test_run_declared_margin_abs_flips_stat_sig():
    """Absolute sibling of test_run_declared_margin_flips_stat_sig: the
    same tiny ~2% observed lift is not significant two-sided-vs-0, but a
    declared margin_abs=0.5 (an enormous tolerance in these units) is
    tested against H0: abs_diff <= -0.5, one-sided greater - trivially
    non-inferior on the additive scale."""
    from increment import readouts
    from increment.plan import compile_decision_plan
    from increment.sources import MomentsSource
    from increment.tables import estimates_to_readout

    design = Randomized(control_group="control")
    plain = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    guardrail = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="avg_event",
        preferred_direction="increase",
        margin_abs=0.5,
    )

    plain_src = MomentsSource(_rows_for("revenue"), metrics=[plain], study_id="e", design=design)
    plain_est = readouts.run(plain_src)[0]
    assert plain_est.null_abs is None
    assert estimates_to_readout([plain_est])[0]["stat_sig"] is False

    guardrail_src = MomentsSource(
        _rows_for("revenue"),
        metrics=[guardrail],
        study_id="e",
        design=design,
        plan=compile_decision_plan(None, [guardrail]),
    )
    est = readouts.run(guardrail_src)[0]
    assert est.null_abs == pytest.approx(-0.5)
    assert est.null_lift == 0.0
    assert est.alternative == "greater"
    assert est.preferred_direction == "increase"
    assert est.abs_lb is not None and est.abs_lb > -0.5
    assert estimates_to_readout([est])[0]["stat_sig"] is True


def _guardrail_rows(metric_name: str, *, direction: str) -> list[dict]:
    """n=20000/arm, control mean 1.0, sd 0.2, ~2% adverse move."""
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan

    adverse_mean = 0.98 if direction == "increase" else 1.02
    n = 20_000
    metric = _directed_mean_metric(direction)
    payload = compiled_plan_to_json(compile_decision_plan(None, [metric]))
    rows = []
    for group_id, mean in (("control", 1.0), ("treatment", adverse_mean)):
        rows.append(
            {
                **centered_row_from_raw_sums(
                    {
                        "experiment_id": "e",
                        "metric": metric_name,
                        "group_id": group_id,
                        "n": n,
                        "sum_y": n * mean,
                        "sum_y2": n * (mean**2 + 0.04),
                    }
                ),
                "moments_format": 7,
                "winsor_lower_percentile": None,
                "winsor_upper_percentile": None,
                "winsor_lower_bound": None,
                "winsor_upper_bound": None,
                "winsor_n": None,
                "winsor_n_lower": None,
                "winsor_n_upper": None,
                "decision_plan": payload,
            }
        )
    return rows


def _directed_mean_metric(direction: Any, **over):
    from increment.semantics.models import MeanMetric

    return MeanMetric(
        name="revenue",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction=direction,
        **over,
    )


@pytest.mark.parametrize(
    ("direction", "margin", "expected_sig"),
    [
        ("increase", 0.01, False),
        ("increase", 0.05, True),
        ("decrease", 0.01, False),
        ("decrease", 0.05, True),
    ],
)
def test_asof_lift_declared_margin_coherent_with_run(direction, margin, expected_sig):
    """The as-of view of a declared-margin metric tests the SAME shifted
    null as the whole-window view: same null_lift, same tail, same
    decision. The ~2% adverse move at n=20000/arm is decisively
    significant two-sided-vs-0, so a silent fallback to the zero null
    flips the margin=0.01 cells from a failed guardrail read to a
    'significant' one."""
    from datetime import date

    from increment import readouts
    from increment.plan import compile_decision_plan
    from increment.sources import MomentsSource
    from increment.tables import estimates_to_readout
    from tests.test_readouts_encouragement import FakeMomentSource

    design = Randomized(control_group="control")
    metric = _directed_mean_metric(direction, margin=margin)
    rows = _guardrail_rows("revenue", direction=direction)
    (headline,) = readouts.run(
        MomentsSource(
            rows,
            metrics=[metric],
            study_id="e",
            design=design,
            plan=compile_decision_plan(None, [metric]),
        )
    )

    asof_rows = [dict(r, ds=date(2025, 1, 31)) for r in rows]
    (trend,) = readouts.asof_lift(
        FakeMomentSource(
            asof_rows,
            metrics=[metric],
            capabilities={"asof"},
            design=design,
        )
    )

    expected_null = -margin if direction == "increase" else margin
    expected_alt = "greater" if direction == "increase" else "less"
    for est in (headline, trend):
        assert est.null_lift == pytest.approx(expected_null)
        assert est.alternative == expected_alt
        assert est.preferred_direction == direction
        assert estimates_to_readout([est])[0]["stat_sig"] is expected_sig
    # The shifted null is decision metadata only - both views share the
    # data, so the intervals must be identical too.
    assert trend.require_lift().value == pytest.approx(headline.require_lift().value, rel=1e-12)
    assert trend.require_lift().lb == pytest.approx(headline.require_lift().lb, rel=1e-12)

    # The two-sided-vs-0 read of the same data IS significant - the cell
    # where a silently dropped margin actively misleads.
    plain = _directed_mean_metric(direction)
    (vs_zero,) = readouts.run(MomentsSource(rows, metrics=[plain], study_id="e", design=design))
    assert estimates_to_readout([vs_zero])[0]["stat_sig"] is True


def test_asof_lift_stamps_none_preferred_direction_when_metric_never_declared_one():
    """The as-of view's stamping site (a separate call from run()'s) must
    apply the same fix: an undeclared metric leaves preferred_direction
    None instead of inheriting MetricIdentity's "increase" default."""
    from datetime import date

    from increment import readouts
    from tests.test_readouts_encouragement import FakeMomentSource

    design = Randomized(control_group="control")
    metric = MeanMetric(name="revenue", entity="user_id", fact="orders", aggregation="sum")
    rows = [dict(r, ds=date(2025, 1, 31)) for r in _guardrail_rows("revenue", direction="increase")]
    src = FakeMomentSource(rows, metrics=[metric], capabilities={"asof"}, design=design)

    (est,) = readouts.asof_lift(src)
    assert est.preferred_direction is None
    with pytest.raises(InvalidRequestError) as raised:
        est.prob_favorable()
    assert raised.value.code == "estimation.results.lift.liftestimate_prob_favorable"


def test_asof_lift_refuses_segmentation_with_a_stable_code():
    """A segmented as-of series has no segment identity on its row type, so
    every segment for a date would collapse into one bucket and estimate
    against the wrong segment's control. Refuse rather than emit a silently
    wrong number; the supported per-segment as-of path is
    Analysis.run_asof_lift(dimension=...)."""
    from datetime import date

    from increment import readouts
    from increment.errors import CodedError
    from tests.test_readouts_encouragement import FakeMomentSource

    design = Randomized(control_group="control")
    metric = MeanMetric(name="revenue", entity="user_id", fact="orders", aggregation="sum")
    rows = [dict(r, ds=date(2025, 1, 31)) for r in _guardrail_rows("revenue", direction="increase")]
    src = FakeMomentSource(rows, metrics=[metric], capabilities={"asof"}, design=design)

    with pytest.raises(CodedError) as raised:
        readouts.asof_lift(src, by=["country"])
    assert raised.value.code == "readout.asof.segment_unsupported"


@pytest.mark.parametrize("declaration", [{"margin": 0.01}, {"margin_abs": 0.5}])
@pytest.mark.parametrize(
    ("site", "mechanism"),
    [("metric", "randomized"), ("plan", "randomized"), ("plan", "encouragement")],
)
def test_breakout_refuses_a_declared_margin_metric(declaration, site, mechanism):
    """Per-segment rows are tested two-sided vs 0 - running a guardrail
    metric through breakout() would silently change stat_sig's estimand
    per segment. Refused loudly until per-segment shifted nulls exist. A
    plan-bound ExperimentMetric margin resolves to the same shifted null as
    the metric's own declaration, so it is refused identically."""
    from increment import readouts
    from increment.semantics.models import ExperimentMetric
    from tests.test_readouts_encouragement import FakeMomentSource, _design

    metric = _directed_mean_metric("decrease", **(declaration if site == "metric" else {}))
    plan = (
        AnalysisPlan(secondaries=[ExperimentMetric(metric="revenue", **declaration)])
        if site == "plan"
        else None
    )
    src = FakeMomentSource(
        [],
        metrics=[metric],
        capabilities={"total"},
        design=Randomized(control_group="control") if mechanism == "randomized" else _design(),
        plan=plan,
        breakouts=("country",),
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        readouts.breakout(src, "country", correction="none")
    assert raised.value.code == "readout.margin.breakout"
    assert raised.value.context["names"] == ("revenue",)


def test_run_quantile_secondary_refuses_breakout_dimensions(monkeypatch):
    # Guards the secondary-path _refuse_unsupported_quantile call: a
    # quantile secondary must hit the same structural refusals as a primary.
    from increment import readouts
    from increment.errors import CapabilityError
    from increment.frame import FrameTotalsSource
    from increment.readouts import _requests

    src = FrameTotalsSource.from_frame(
        _quantile_unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="lat", type="quantile", quantile=0.9),
        ],
        plan=AnalysisPlan(primary="revenue", secondaries=["lat"]),
    )
    # Bypass the request preflight so this regression proves the secondary
    # dispatch site itself still invokes the quantile refusal. Narrowed to
    # the quantile secondary alone: `revenue` (non-quantile primary) would
    # otherwise trip its own, unrelated by= refusal first.
    monkeypatch.setattr(_requests, "validate_request", lambda request: None)
    original_moments = src.moments
    monkeypatch.setattr(src, "moments", lambda metric, **kwargs: original_moments(metric))
    with pytest.raises(CapabilityError) as raised:
        readouts.run(src, by=("region",), metrics=["lat"], sensitivity_methods=())
    assert raised.value.code == "readout.metric.quantile_breakout"


def _daily_panel(n: int = 40) -> pd.DataFrame:
    """A per-unit-per-day panel: one row per unit per day, three days each."""
    rng = np.random.default_rng(0)
    base = dt.date(2025, 1, 1)
    return pd.DataFrame(
        [
            {
                "unit_id": f"u{i}",
                "group_id": "control" if i < n else "treatment",
                "ds": base + dt.timedelta(days=day),
                "revenue": float(rng.lognormal(1.0, 0.3)),
            }
            for i in range(2 * n)
            for day in range(3)
        ]
    )


class TestDailyInferenceRefusalIsCoded:
    """The disjoint-slice condition has one registered code, so the facade must
    not raise an unstructured exception for it."""

    def test_the_facade_validator_raises_the_registered_refusal(self):
        from increment import Analysis
        from increment.errors import CodedError
        from increment.frame import synthesise_metric
        from increment.sequential_source import frame_observation_mapping
        from tests.sequential_cases import declared_plan

        spec = MetricSpec(name="revenue", type="mean")
        design = Randomized(control_group="control")
        plan = declared_plan(
            [synthesise_metric(spec)],
            source_id="frame",
            design=design,
            transformations=[spec],
            source_mapping=frame_observation_mapping(unit="unit_id", group="group_id", date="ds"),
            public_mean=True,
        )
        analysis = Analysis.from_unit_panel(
            _daily_panel(),
            unit="unit_id",
            group="group_id",
            date="ds",
            control="control",
            metrics=[spec],
            plan=plan,
        )

        with pytest.raises(CodedError) as exc:
            analysis.run_daily_lift()
        assert exc.value.code == "readout.inference.disjoint_slices"
        assert exc.value.context["inference"] == "AsymptoticMean"

    def test_no_inference_is_accepted(self):
        from increment import Analysis

        analysis = Analysis.from_unit_panel(
            _daily_panel(),
            unit="unit_id",
            group="group_id",
            date="ds",
            control="control",
            metrics=[MetricSpec(name="revenue", type="mean")],
            plan=AnalysisPlan(primary="revenue"),
        )

        assert analysis.run_daily_lift()


def test_readout_cells_refuse_duplicate_metric_group_rows():
    """A readout rejects duplicate arm rows exactly like estimate_lift's own
    ingress; keying on group_id alone silently keeps the last row read."""
    from increment import readouts
    from tests.test_readouts_encouragement import FakeMomentSource

    def _row(group_id, mean, n=1000, var=0.25):
        return {
            "experiment_id": "e1",
            "metric": "rev",
            "group_id": group_id,
            "n": n,
            "ref_y": 0.0,
            "cy1": n * mean,
            "cy2": n * (mean * mean + var),
        }

    def _source(summary):
        return FakeMomentSource(
            summary,
            metrics=[MeanMetric(name="rev", entity="u", fact="rev")],
            capabilities={"total"},
            design=Randomized(control_group="control"),
        )

    summary = [_row("control", 1.0), _row("treatment", 1.1), _row("control", 1.05)]
    with pytest.raises(InvalidRequestError) as forward:
        readouts.run(_source(summary))
    with pytest.raises(InvalidRequestError) as backward:
        readouts.run(_source(list(reversed(summary))))
    assert forward.value.code == backward.value.code == "estimation.engine.arm.duplicate_rows"


def test_asof_lift_refuses_a_declared_cluster():
    """Day-axis views have no cluster-robust variance; the public readout must
    refuse a clustered source like the facade does instead of silently
    reporting unit-grain uncertainty."""
    from increment import readouts
    from increment.errors import CodedError
    from increment.frame import FrameTotalsSource

    frame = pd.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(6)],
            "group_id": ["control"] * 3 + ["treatment"] * 3,
            "store": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "revenue": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    source = FrameTotalsSource.from_frame(
        frame,
        unit="unit_id",
        group="group_id",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean")],
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        cluster="store",
    )

    with pytest.raises(CodedError) as refusal:
        readouts.asof_lift(source)
    assert refusal.value.code == "readout.asof.cluster"


def test_resolve_window_days_refuses_a_non_metric_argument():
    """Every real Metric subtype is handled; a non-Metric argument is the
    only way to reach the fallback."""
    from increment._window import resolve_window_days
    from increment.errors import UnsupportedRequestError

    with pytest.raises(UnsupportedRequestError) as exc_info:
        resolve_window_days(object())  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "facade.window.resolve_window_days"
    assert exc_info.value.context["metric_type"] == "object"


def test_overlay_configs_refuses_a_metric_missing_from_the_base_catalog():
    from increment._analysis_config import overlay_configs
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="rev", entity="u", fact="rev")
    with pytest.raises(InvalidRequestError) as exc_info:
        overlay_configs([metric], (), methods=None, prior=None)
    assert exc_info.value.code == "facade.analysis_config.metric_no_resolved"
    assert exc_info.value.context["metric_name"] == "rev"


def test_contrast_handler_refuses_a_metric_with_no_procedure():
    from increment._evidence_dispatch import ContrastHandler
    from increment.semantics.models import MeanMetric

    class _FakeContrastSource:
        def contrast_stats(self, metric):
            raise AssertionError("must not be reached")

    class _FakeContrastRequest:
        metrics = (MeanMetric(name="rev", entity="u", fact="rev"),)
        procedures: dict = {}

    with pytest.raises(InvalidRequestError) as exc_info:
        ContrastHandler().run(_FakeContrastSource(), _FakeContrastRequest())  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "facade.evidence_dispatch.contrast_handler.readout_request_no"
    assert exc_info.value.context["metric_name"] == "rev"


def test_readout_request_from_source_refuses_a_metrics_configs_arity_mismatch():
    from increment._readout_request import ReadoutRequest
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="rev", entity="u", fact="rev")

    with pytest.raises(InvalidRequestError) as exc_info:
        ReadoutRequest.from_source(
            cast("Any", object()),
            metrics=[metric],
            configs=[],
            view="run",
            grain="total",
        )
    assert exc_info.value.code == "readout.request.readoutrequest_requires_one_resolved"


@pytest.mark.parametrize(
    "code,support_type",
    [("compatibility.support_unsupported", str)],
)
def test_refuse_unsupported_rejects_a_non_unsupported_argument(code, support_type):
    from increment.compatibility import refuse_unsupported

    with pytest.raises(InvalidRequestError) as exc_info:
        refuse_unsupported(cast("Any", "not-unsupported"))
    assert exc_info.value.code == code
    assert exc_info.value.context["support_type"] == support_type.__name__


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"numerator": 1.0, "multiplier": 1, "denominator": 0}, "compatibility.denominator"),
        ({"numerator": 1.0, "multiplier": 0, "denominator": 1}, "compatibility.multiplier"),
    ],
)
def test_conservative_ratio_refuses_a_sub_unit_denominator_or_multiplier(kwargs, code):
    from increment.compatibility import _conservative_ratio

    with pytest.raises(InvalidRequestError) as exc_info:
        _conservative_ratio(**kwargs)
    assert exc_info.value.code == code


@pytest.mark.parametrize("other_adjustment", ["prior", "cuped"])
@pytest.mark.parametrize("adjust_winsor", [False, True])
def test_winsor_validation_scopes_adjustments_to_selected_metrics(other_adjustment, adjust_winsor):
    """A percentile-winsorized metric refuses the adjustments it cannot carry,
    while the same adjustment on the plain metric of the same readout is fine."""
    from increment import readouts
    from increment.frame import from_unit_summary
    from increment.semantics.models import Winsorization

    frame = pa.table(
        {
            "u": list(range(16)),
            "g": ["C"] * 8 + ["T"] * 8,
            "winsor": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0] * 2,
            "plain": [10.0, 12.0, 11.0, 14.0, 13.0, 16.0, 15.0, 19.0] * 2,
            "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0] * 2,
        }
    )
    specs = []
    for name in ("winsor", "plain"):
        spec: dict[str, Any] = {}
        if name == "winsor":
            spec["winsorization"] = Winsorization(upper_percentile=0.95)
        if (name == "winsor") is adjust_winsor:
            if other_adjustment == "cuped":
                spec["covariate"] = "x"
                spec["decision_method"] = Method(name="cuped", variance_reduction="cuped")
            else:
                spec["prior"] = Normal(mu=0, sigma=0.1)
        specs.append(MetricSpec(name=name, **spec))
    source = from_unit_summary(frame, unit="u", group="g", control="C", metrics=specs)

    if adjust_winsor:
        with pytest.raises(CapabilityError) as error:
            readouts.run(source)
        assert error.value.code == "readout.metric.percentile_winsorization"
    else:
        assert {row.metric for row in readouts.run(source)} == {"winsor", "plain"}


@pytest.mark.slow
@pytest.mark.parametrize("other_adjustment", ["prior", "cuped"])
def test_mixed_readout_emits_winsor_and_adjusted_other_metric(other_adjustment):
    from increment import readouts
    from increment.frame import from_unit_summary
    from increment.semantics.models import Winsorization

    source = from_unit_summary(
        pa.table(
            {
                "u": list(range(16)),
                "g": ["C"] * 8 + ["T"] * 8,
                "winsor": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0] * 2,
                "plain": [10.0, 12.0, 11.0, 14.0, 13.0, 16.0, 15.0, 19.0] * 2,
                "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0] * 2,
            }
        ),
        unit="u",
        group="g",
        control="C",
        metrics=[
            MetricSpec(name="winsor", winsorization=Winsorization(upper_percentile=0.95)),
            MetricSpec(
                name="plain",
                covariate="x" if other_adjustment == "cuped" else None,
                decision_method=Method(name="cuped", variance_reduction="cuped")
                if other_adjustment == "cuped"
                else None,
                prior=Normal(mu=0, sigma=0.1) if other_adjustment == "prior" else None,
            ),
        ],
    )
    rows = {row.metric: row for row in readouts.run(source)}
    assert set(rows) == {"winsor", "plain"}
    assert rows["winsor"].confidence_set is not None
    assert rows["plain"].method == ("cuped" if other_adjustment == "cuped" else "unadjusted")
    assert rows["plain"].prior_shrunk == (other_adjustment == "prior")


@pytest.mark.slow
@pytest.mark.parametrize("population", ["assigned", "triggered"])
def test_mixed_native_readout_pins_selected_metrics_before_source_mutation(population):
    """A readout taken through a snapshot pinned before an upstream mutation
    reports the pinned numbers; the mutation only shows up on a fresh read."""
    from increment import readouts
    from increment.semantics.models import AnalysisPlan, Winsorization
    from tests.analysis_factory import _native_source, make_analysis
    from tests.test_sequential_public_sources import _native_fixture

    connection, definitions, original = _native_fixture("scalar_mean")
    analysis = None
    try:
        outcome = definitions.metrics[0].model_copy(update={"window_days": 1})
        metric_source = next(
            source
            for source in definitions.fact_sources
            if any(fact.name == "outcome_event" for fact in source.facts)
        )
        missing_source = metric_source.model_copy(
            update={
                "name": "unused_missing",
                "sql": "SELECT * FROM missing_metric_table",
                "facts": (metric_source.facts[0].model_copy(update={"name": "missing_outcome"}),),
            }
        )
        exposure = definitions.exposures[0]
        trigger = exposure.model_copy(
            update={
                "name": "triggered_exposure",
                "sql": "SELECT unit_id, ts, group_id FROM enrolled WHERE CAST(SUBSTR(unit_id, 1, 8) AS INTEGER) < 48",
            }
        )
        metrics = (
            outcome.model_copy(
                update={"name": "winsor", "winsorization": Winsorization(upper_percentile=0.95)}
            ),
            outcome.model_copy(update={"name": "plain"}),
            outcome.model_copy(update={"name": "unused", "fact": "missing_outcome"}),
        )
        experiment = definitions.experiments[0].model_copy(
            update={"plan": AnalysisPlan(primary="plain"), "trigger": trigger.name}
        )
        definitions = definitions.model_copy(
            update={
                "metrics": metrics,
                "fact_sources": (*definitions.fact_sources, missing_source),
                "exposures": (*definitions.exposures, trigger),
                "experiments": (experiment,),
            }
        )
        analysis = make_analysis(
            connection, definitions, experiment=experiment, metrics=list(metrics)
        )
        source = _native_source(analysis)
        view = source.triggered_source() if population == "triggered" else source
        raw = source.unit_frame(metrics[0], population=population, outcome_stage="raw")
        assert raw.num_rows == (96 if population == "triggered" else 192)
        selected = ("winsor", "plain")
        expected = {
            row.metric: row for row in readouts.run(view, metrics=selected, _population=population)
        }
        selected_metrics = [metric for metric in source.context.metrics if metric.name in selected]
        with source.readout_snapshot(metrics=selected_metrics, population=population) as pinned:
            connection.raw_sql(
                "UPDATE events SET value = value * 10 WHERE unit_id LIKE '%-treatment'"
            )
            actual = {
                row.metric: row
                for row in readouts.run(pinned, metrics=selected, _population=population)
            }

        assert set(actual) == {"winsor", "plain"}
        for name in selected:
            assert actual[name].require_lift() == expected[name].require_lift()
        assert actual["plain"].require_lift().value == pytest.approx(3.8)
        fresh = {
            row.metric: row for row in readouts.run(view, metrics=selected, _population=population)
        }
        assert fresh["plain"].require_lift().value == pytest.approx(47.0)
    finally:
        if analysis is not None:
            analysis.close()
        original.close()


def test_run_declared_secondary_family_survives_a_degenerate_cell():
    """Repro at the public layer: an AnalysisPlan puts every non-primary,
    non-guardrail metric into the BH secondary family by default. A
    degenerate secondary (refunds, all-zero treatment) must not abort
    the family -- revenue (primary) and converted (the other
    secondary) must still return, AND refunds itself must return a
    real row with its additive result."""
    from increment.analysis import Analysis
    from increment.estimation.results import LiftEstimate

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "revenue": np.concatenate([rng.normal(10, 2, n), rng.normal(11, 2, n)]),
            "refunds": np.concatenate([rng.normal(1, 0.2, n), np.zeros(n)]),
            "converted": rng.binomial(1, 0.3, 2 * n),
        }
    )
    metrics = [
        MetricSpec(name="revenue", type="mean"),
        MetricSpec(name="refunds", type="mean"),
        MetricSpec(name="converted", type="conversion"),
    ]
    plan = AnalysisPlan(primary=["revenue"], secondaries=["refunds", "converted"])
    analysis = Analysis.from_unit_summary(
        df, unit="unit_id", group="group_id", control="control", metrics=metrics, plan=plan
    )
    results = analysis.run()
    assert {r.metric for r in results} == {"revenue", "refunds", "converted"}
    refunds_row = cast("LiftEstimate", next(r for r in results if r.metric == "refunds"))
    assert refunds_row.lift is None
    assert refunds_row.relative_unavailable_reason == "nonpositive_arm_mean"
    assert refunds_row.abs_diff is not None
