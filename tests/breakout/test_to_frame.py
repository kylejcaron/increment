"""to_frame serialization of non-str Sequence cells."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import pandas as pd
import pytest

from increment.breakout.estimates import run_breakout
from tests.breakout.test_estimates import _make_arm_row, _mean_metric


def _bh_rows():
    return [
        _make_arm_row(100, 10.0, 4.0, country="US", metric="rev", group_id="control"),
        _make_arm_row(100, 8.0, 4.0, country="US", metric="rev", group_id="treatment"),
        _make_arm_row(100, 10.0, 4.0, country="CA", metric="rev", group_id="control"),
        _make_arm_row(100, 11.0, 4.0, country="CA", metric="rev", group_id="treatment"),
    ]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_serializes_family_axes_tuple_on_every_backend(backend):
    pytest.importorskip(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_breakout(
            _bh_rows(),
            [_mean_metric("rev")],
            control_group="control",
            dimension="country",
            correction="bh",
        )
    frame: Any = out.to_frame(backend=backend)
    if backend == "pandas":
        values = frame["family_axes"].tolist()
    elif backend == "polars":
        values = frame["family_axes"].to_list()
    else:
        values = frame.column("family_axes").to_pylist()
    assert values[0] == "('metric', 'arm', 'segment')"


def _directional_bh_rows():
    return [
        _make_arm_row(500, 10.0, 4.0, country="US", metric="rev", group_id="control"),
        _make_arm_row(500, 12.0, 4.0, country="US", metric="rev", group_id="treatment"),
        _make_arm_row(500, 10.0, 4.0, country="CA", metric="rev", group_id="control"),
        _make_arm_row(500, 11.5, 4.0, country="CA", metric="rev", group_id="treatment"),
    ]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_open_side_column_on_directional_bh_breakout(backend):
    """``correction="bh"`` + ``alternative="greater"`` opens the far bound
    on every selected cell -- the flattened frame must carry an explicit
    ``open_side`` column so a consumer can tell a genuinely unbounded
    endpoint from a closed one without inspecting ``lb``/``ub`` nulls
    alone, on every supported backend."""
    pytest.importorskip(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_breakout(
            _directional_bh_rows(),
            [_mean_metric("rev")],
            control_group="control",
            dimension="country",
            correction="bh",
            alternative="greater",
            q=0.5,
        )
    frame: Any = out.to_frame(backend=backend)
    if backend == "pandas":
        rows = frame[["dimension_value", "lift", "lb", "ub", "open_side"]].to_dict("records")
    elif backend == "polars":
        rows = frame.select(["dimension_value", "lift", "lb", "ub", "open_side"]).to_dicts()
    else:
        rows = frame.select(["dimension_value", "lift", "lb", "ub", "open_side"]).to_pylist()
    by_segment = {r["dimension_value"]: r for r in rows}
    assert set(by_segment) == {"US", "CA"}
    for segment in ("US", "CA"):
        row = by_segment[segment]
        assert row["lift"] is not None
        assert row["lb"] is not None
        assert row["ub"] is None or (isinstance(row["ub"], float) and math.isnan(row["ub"]))
        assert row["open_side"] == "upper"


def test_to_frame_open_side_distinguishes_open_from_closed_and_unavailable():
    """Direct construction (not run_breakout) pinning the three states
    ``open_side`` must tell apart: a closed interval (both bounds finite,
    ``open_side=None``), a genuinely unbounded one-sided interval
    (``open_side`` set, exactly one bound ``None``), and an unavailable
    row (``lift=None``, every flattened column including ``open_side``
    is null)."""
    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.estimation.results import Estimate

    closed = BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="US",
        source=None,
        lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
    )
    unavailable = BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="CA",
        source=None,
        lift=None,
        excluded="zero_variance",
    )
    open_row = BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="MX",
        source=None,
        lift=Estimate(value=0.1, lb=0.05, ub=None, level=0.95, alpha=0.05, open_side="upper"),
    )
    frame = cast(
        pd.DataFrame, BreakoutEstimates([closed, unavailable, open_row]).to_frame(backend="pandas")
    )
    by_segment = frame.set_index("dimension_value")
    assert by_segment.loc["US", "lb"] == pytest.approx(0.05)
    assert by_segment.loc["US", "ub"] == pytest.approx(0.15)
    assert pd.isna(by_segment.loc["US", "open_side"])

    assert pd.isna(by_segment.loc["CA", "lift"])
    assert pd.isna(by_segment.loc["CA", "lb"])
    assert pd.isna(by_segment.loc["CA", "ub"])
    assert pd.isna(by_segment.loc["CA", "open_side"])

    assert by_segment.loc["MX", "lb"] == pytest.approx(0.05)
    assert pd.isna(by_segment.loc["MX", "ub"])
    assert by_segment.loc["MX", "open_side"] == "upper"


def test_to_frame_summary_refuses_daily_metric_value():
    from increment.breakout.estimates import DailyMetricValue, DailyMetricValues
    from increment.errors import InvalidRequestError
    from increment.estimation.results import Estimate

    result = DailyMetricValues(
        [
            DailyMetricValue(
                metric="rev",
                group_id="treatment",
                ds="2025-01-01",
                value=Estimate(value=1.0),
                n=10,
            )
        ]
    )
    with pytest.raises(InvalidRequestError) as raised:
        result.to_frame(columns="summary")
    assert raised.value.code == "results.frame.summary_unavailable"


def test_to_frame_columns_validates_and_returns_canonical_order():
    from increment.breakout.estimates import LiftEstimates
    from increment.errors import InvalidRequestError
    from increment.estimation.results import Estimate, LiftEstimate

    results = LiftEstimates(
        [
            LiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            )
        ]
    )
    assert list(results.to_frame(columns=["ub", "metric", "lift"]).columns) == [
        "metric",
        "lift",
        "ub",
        "view_partial",
    ]
    with pytest.raises(InvalidRequestError) as raised:
        results.to_frame(columns=["metric", "not_a_column"])
    assert raised.value.code == "results.frame.unknown_columns"
    assert raised.value.context["unknown"] == ("not_a_column",)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_to_frame_refuses_empty_columns_without_dropping_rows(backend):
    pytest.importorskip(backend)
    from increment.breakout.estimates import LiftEstimates
    from increment.errors import InvalidRequestError
    from increment.estimation.results import Estimate, LiftEstimate

    results = LiftEstimates(
        [
            LiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1),
            )
        ]
    )
    with pytest.raises(InvalidRequestError) as raised:
        results.to_frame(backend=backend, columns=[])
    assert raised.value.code == "results.frame.empty_columns"


def test_to_frame_summary_presets_match_full_frame_subsets():
    from datetime import date

    from increment.breakout.estimates import (
        BreakoutEstimate,
        BreakoutEstimates,
        DailyLiftEstimate,
        DailyLiftEstimates,
        LiftEstimates,
    )
    from increment.estimation.results import Estimate, LiftEstimate

    lift = Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95)
    collections = [
        (
            LiftEstimates(
                [
                    LiftEstimate(
                        metric="rev",
                        group_id="treatment",
                        method="unadjusted",
                        method_role="decision",
                        lift=lift,
                    )
                ]
            ),
            (
                "metric",
                "group_id",
                "method",
                "role",
                "analysis_population",
                "lift",
                "lb",
                "ub",
                "set_lower",
                "set_upper",
                "set_level",
                "set_numerical_qualification",
                "discovery",
                "failure_code",
                "estimand",
                "value_scale",
                "decision_scope_complete",
                "multiplicity_status",
                "relative_unavailable_reason",
                "view_partial",
                "posterior_prob_favorable_reason_code",
            ),
        ),
        (
            BreakoutEstimates(
                [
                    BreakoutEstimate(
                        metric="rev",
                        group_id="treatment",
                        method="unadjusted",
                        method_role="decision",
                        dimension="country",
                        dimension_value="US",
                        lift=lift,
                    )
                ]
            ),
            (
                "metric",
                "group_id",
                "method",
                "role",
                "analysis_population",
                "dimension",
                "dimension_value",
                "lift",
                "lb",
                "ub",
                "set_lower",
                "set_upper",
                "set_level",
                "set_numerical_qualification",
                "discovery",
                "failure_code",
                "estimand",
                "value_scale",
                "decision_scope_complete",
                "multiplicity_status",
                "relative_unavailable_reason",
                "posterior_prob_favorable_reason_code",
                "excluded",
                "view_partial",
            ),
        ),
        (
            DailyLiftEstimates(
                [
                    DailyLiftEstimate(
                        metric="rev",
                        group_id="treatment",
                        method="unadjusted",
                        method_role="decision",
                        ds=date(2025, 1, 1),
                        lift=lift,
                    )
                ]
            ),
            (
                "metric",
                "group_id",
                "method",
                "role",
                "analysis_population",
                "ds",
                "lift",
                "lb",
                "ub",
                "set_lower",
                "set_upper",
                "set_level",
                "set_numerical_qualification",
                "discovery",
                "failure_code",
                "dimension",
                "dimension_value",
                "estimand",
                "value_scale",
                "decision_scope_complete",
                "multiplicity_status",
                "unavailable",
                "relative_unavailable_reason",
                "posterior_prob_favorable_reason_code",
                "view_partial",
            ),
        ),
    ]

    required_decision_columns = {
        "stat_sig",
        "p_value",
        "p_value_reason_code",
        "n_control",
        "n_treat",
        "mean_control",
        "mean_treat",
    }
    for results, expected_columns in collections:
        full = cast(pd.DataFrame, results.to_frame(backend="pandas"))
        summary = cast(pd.DataFrame, results.to_frame(backend="pandas", columns="summary"))
        assert set(summary.columns) == set(expected_columns) | required_decision_columns
        pd.testing.assert_frame_equal(summary, full.loc[:, summary.columns])


def test_encouragement_summary_retains_estimand_and_scale_identity():
    import pandas as pd

    from increment import Analysis
    from increment.breakout.estimates import LiftEstimates
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    df = pd.DataFrame(
        [
            ("u01", "control", 10.0, 0),
            ("u02", "control", 12.0, 0),
            ("u03", "control", 9.0, 0),
            ("u04", "control", 20.0, 1),
            ("u05", "treatment", 25.0, 1),
            ("u06", "treatment", 30.0, 1),
            ("u07", "treatment", 28.0, 1),
            ("u08", "treatment", 15.0, 0),
        ],
        columns=["user_id", "variant", "revenue", "clicked"],
    )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        min_first_stage_z=0.5,
    )
    results = cast(
        LiftEstimates,
        Analysis.from_unit_summary(
            df,
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=design,
            uptake="clicked",
        ).run(),
    )

    frame = cast(pd.DataFrame, results.to_frame(columns="summary"))
    assert {"estimand", "value_scale"} <= set(frame.columns)
    assert set(frame["estimand"]) == {"itt", "compliance", "late"}
    identity = [
        "metric",
        "group_id",
        "method",
        "analysis_population",
        "estimand",
        "value_scale",
    ]
    assert frame.duplicated(identity).sum() == 0


def test_dimensioned_daily_summary_retains_segment_identity_and_unavailability():
    from datetime import date

    from increment.breakout.estimates import DailyLiftEstimate, DailyLiftEstimates, run_daily_lift
    from tests.breakout.test_daily import _make_daily_row, _mean_metric

    day = date(2025, 1, 1)
    rows = [
        _make_daily_row(200, 10.0, 4.0, ds=day, group_id="control"),
        _make_daily_row(200, 12.0, 4.0, ds=day, group_id="treatment"),
        _make_daily_row(200, 20.0, 4.0, ds=day, group_id="control"),
        _make_daily_row(200, 19.0, 4.0, ds=day, group_id="treatment"),
    ]
    for row in rows[:2]:
        row["country"] = "US"
    for row in rows[2:]:
        row["country"] = "CA"
    results = run_daily_lift(rows, [_mean_metric()], control_group="control", dimension="country")

    frame = cast(pd.DataFrame, results.to_frame(columns="summary"))
    assert {
        "dimension",
        "dimension_value",
        "estimand",
        "value_scale",
        "unavailable",
    } <= set(frame.columns)
    assert set(frame["dimension_value"]) == {"US", "CA"}
    identity = [
        "metric",
        "group_id",
        "method",
        "analysis_population",
        "ds",
        "dimension",
        "dimension_value",
        "estimand",
        "value_scale",
    ]
    assert frame.duplicated(identity).sum() == 0

    unavailable = DailyLiftEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        ds=day,
        lift=None,
        unavailable="few_units",
        dimension="country",
        dimension_value="MX",
    )
    unavailable_frame = cast(
        pd.DataFrame, DailyLiftEstimates([unavailable]).to_frame(columns="summary")
    )
    assert unavailable_frame.loc[0, "unavailable"] == "few_units"


def test_selected_views_preserve_partial_and_scope_disclosures():
    import warnings

    from tests.test_readout_public_scope import _analysis, _frame

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = _analysis(_frame()).run().filter(lambda row: row.metric != "guard")

    summary = cast(pd.DataFrame, filtered.to_frame(columns="summary"))
    custom = cast(
        pd.DataFrame,
        filtered.to_frame(columns=["metric", "decision_scope_complete", "multiplicity_status"]),
    )
    for frame in (summary, custom):
        assert frame["view_partial"].all()
        assert not frame["decision_scope_complete"].any()
        assert set(frame["multiplicity_status"]) == {"declared_plan"}


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_projects_decision_statistics(backend):
    pytest.importorskip(backend)
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import Estimate, LiftEstimate

    row = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(
            value=0.1,
            lb=0.02,
            ub=0.18,
            level=0.95,
            alpha=0.05,
            log_mean=0.1,
            log_se=0.04,
        ),
        alpha=0.05,
        sampling_available=True,
        reference_kind="normal",
    )
    frame: Any = LiftEstimates([row]).to_frame(backend=backend)

    if backend == "pandas":
        projected = frame.iloc[0]
    elif backend == "polars":
        projected = frame.row(0, named=True)
    else:
        projected = frame.to_pylist()[0]
    assert projected["stat_sig"] == row.stat_sig()
    assert projected["p_value"] == pytest.approx(row.p_value())
    assert pd.isna(projected["p_value_reason_code"])


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_keeps_unavailable_stat_sig_missing(backend):
    pytest.importorskip(backend)
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import Estimate, LiftEstimate

    valid = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(
            value=0.1,
            lb=0.02,
            ub=0.18,
            level=0.95,
            alpha=0.05,
            log_mean=0.1,
            log_se=0.04,
        ),
        sampling_available=True,
        reference_kind="normal",
    )
    failed = valid.model_copy(update={"group_id": "failed", "sampling_available": False})
    frame: Any = LiftEstimates([failed, valid]).to_frame(backend=backend)

    if backend == "pandas":
        values = frame["stat_sig"].tolist()
        assert str(frame["stat_sig"].dtype) == "boolean"
    elif backend == "polars":
        values = frame["stat_sig"].to_list()
    else:
        values = frame.column("stat_sig").to_pylist()
    assert values[0] is None or pd.isna(values[0])
    assert values[1] is True


def test_to_frame_projects_p_value_when_sampling_availability_is_unrecorded():
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import Estimate, LiftEstimate

    row = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(
            value=0.1,
            lb=0.02,
            ub=0.18,
            level=0.95,
            alpha=0.05,
            log_mean=0.1,
            log_se=0.04,
        ),
        alpha=0.05,
        sampling_available=None,
        reference_kind="normal",
    )
    assert row.p_value() is not None
    frame: Any = LiftEstimates([row]).to_frame(backend="pandas")
    assert frame.loc[0, "p_value"] == pytest.approx(row.p_value())
