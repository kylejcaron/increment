"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Literal, cast

import pytest

from increment import Analysis
from increment import readouts as readout_functions
from increment.breakout.estimates import DailyLiftEstimates, DailyMetricValues
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.engine import Method
from increment.frame import MetricSpec
from increment.plan import compile_decision_plan
from increment.semantics.design import AdjustmentSet, Observational
from increment.semantics.models import AnalysisPlan, CupedCovariate, MeanMetric
from increment.sources import MomentSource
from tests.analysis_factory import lift_rows
from tests.sequential_cases import registered_spec
from tests.warning_codes import warning_codes


@pytest.fixture
def unit_summary_analysis():
    """A runnable `from_unit_summary` (seam-family) Analysis - same shape
    as ``test_analysis_from_unit_summary_runs``' inline DataFrame, factored
    out for tests that need to `run()` it more than once."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    return Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )


def test_declared_duplicate_methods_refuse_during_configuration_resolution():
    import pandas as pd

    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            pd.DataFrame(
                {
                    "user_id": ["u1", "u2", "u3", "u4"],
                    "variant": ["treatment", "treatment", "control", "control"],
                    "revenue": [25.0, 35.0, 22.5, 13.0],
                    "pre_revenue": [24.0, 34.0, 21.0, 12.0],
                }
            ),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method(name="same"),
                    sensitivity_methods=(
                        Method.cuped(
                            name="same",
                            covariates=CupedCovariate(name="pre_revenue", kind="continuous"),
                        ),
                    ),
                )
            ],
        )
    assert raised.value.code == "estimation.engine.method_names_unique"


def test_declared_method_names_are_scoped_per_metric():
    import pandas as pd

    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3", "u4"],
                "variant": ["treatment", "treatment", "control", "control"],
                "revenue": [25.0, 35.0, 22.5, 13.0],
                "orders": [2.0, 3.0, 1.0, 2.0],
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", decision_method=Method(name="unadjusted")),
            MetricSpec(name="orders", decision_method=Method(name="unadjusted")),
        ],
    )

    results = lift_rows(analysis.run())
    assert {row.metric for row in results} == {"revenue", "orders"}
    assert {row.method for row in results} == {"unadjusted"}


def _panel_with_breakout_rows() -> list[dict[str, object]]:
    return [
        {
            "user_id": "u1",
            "variant": "control",
            "day": "2026-01-01",
            "country": "US",
            "revenue": 1.0,
        },
        {
            "user_id": "u1",
            "variant": "control",
            "day": "2026-01-02",
            "country": "US",
            "revenue": 2.0,
        },
        {
            "user_id": "u2",
            "variant": "treatment",
            "day": "2026-01-01",
            "country": "CA",
            "revenue": 3.0,
        },
        {
            "user_id": "u2",
            "variant": "treatment",
            "day": "2026-01-02",
            "country": "CA",
            "revenue": 5.0,
        },
    ]


def test_analysis_from_unit_panel_forwards_breakouts():
    import pandas as pd

    analysis = Analysis.from_unit_panel(
        pd.DataFrame(_panel_with_breakout_rows()),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )
    results = analysis.run_daily(dimension="country")
    assert results
    assert {row.dimension for row in results} == {"country"}


def _summary_breakout_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "user_id": f"{segment}-{group}-{index}",
                "variant": group,
                "segment": segment,
                "value": value,
            }
            for segment, values in (
                ("A", {"control": (1.0, 2.0), "treatment": (2.0, 4.0)}),
                ("B", {"control": (4.0, 5.0), "treatment": (5.0, 7.0)}),
            )
            for group, observations in values.items()
            for index, value in enumerate(observations)
        ]
    )


def test_unit_summary_breakouts_retain_total_only_identity_and_other_reads(tmp_path):
    baseline = Analysis.from_unit_summary(
        _summary_breakout_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"value": "mean"},
    )
    analysis = Analysis.from_unit_summary(
        _summary_breakout_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"value": "mean"},
        breakouts=("segment",),
    )

    assert analysis.unit_counts() == baseline.unit_counts() == {"control": 4, "treatment": 4}
    assert analysis.srm(expected={"control": 0.5, "treatment": 0.5}, inference="fixed") == (
        baseline.srm(expected={"control": 0.5, "treatment": 0.5}, inference="fixed")
    )
    assert {row.dimension_value for row in analysis.run_breakout()} == {"A", "B"}
    for method in (analysis.run_daily, analysis.run_asof):
        with pytest.raises(CapabilityError) as raised:
            method()
        assert raised.value.code == "evidence.grain_unavailable"

    original_counts = analysis.unit_counts()
    original_srm = analysis.srm(expected={"control": 0.5, "treatment": 0.5}, inference="fixed")
    original_breakout = analysis.run_breakout()
    export_path = tmp_path / "summary-moments.parquet"
    analysis.export(export_path)
    import pyarrow.parquet as pq

    rows = pq.read_table(export_path).to_pylist()
    assert {row["experiment_id"] for row in rows} == {"frame"}
    reopened = Analysis.from_moments(
        rows,
        control="control",
        metrics=[MetricSpec(name="value", type="mean")],
    )
    assert reopened.unit_counts() == original_counts
    assert analysis.unit_counts() == original_counts
    assert (
        analysis.srm(expected={"control": 0.5, "treatment": 0.5}, inference="fixed") == original_srm
    )
    assert analysis.run_breakout() == original_breakout


def test_unit_summary_breakout_uses_exposure_excluded_population():
    frame = _summary_breakout_frame()
    frame.loc[len(frame)] = {
        "user_id": "A-control-extra",
        "variant": "control",
        "segment": "A",
        "value": 3.0,
    }
    frame["exposure"] = [date(2026, 1, 1)] * len(frame)
    frame.loc[0, "exposure"] = None
    analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"value": "mean"},
        exposure_date="exposure",
        on_unassigned="exclude",
        breakouts=("segment",),
    )
    results = analysis.run_breakout()
    expected = Analysis.from_unit_summary(
        frame.dropna(subset=["exposure"]),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"value": "mean"},
        exposure_date="exposure",
        breakouts=("segment",),
    ).run_breakout()

    by_segment = {row.dimension_value: row for row in results}
    expected_by_segment = {row.dimension_value: row for row in expected}
    assert set(by_segment) == set(expected_by_segment)
    for segment, row in by_segment.items():
        actual_lift = row.require_lift()
        expected_lift = expected_by_segment[segment].require_lift()
        assert actual_lift.value == pytest.approx(expected_lift.value, abs=1e-12)
        assert actual_lift.lb == pytest.approx(expected_lift.lb, abs=1e-12)
        assert actual_lift.ub == pytest.approx(expected_lift.ub, abs=1e-12)


def test_unit_summary_breakout_missing_drop_warns_once_per_metric():

    frame = _summary_breakout_frame()
    frame.loc[len(frame)] = {
        "user_id": "A-control-extra",
        "variant": "control",
        "segment": "A",
        "value": 3.0,
        "covariate": 1.0,
    }
    frame["covariate"] = [1.0] * len(frame)
    frame.loc[0, "value"] = None
    frame.loc[1, "covariate"] = None
    with pytest.warns(IncrementWarning) as recorded:
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="value",
                    type="mean",
                    missing="drop",
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(
                            name="covariate", kind="continuous", missing="impute"
                        )
                    ),
                )
            ],
            breakouts=("segment",),
        )
        results = analysis.run_breakout(
            decision_method=Method(name="unadjusted"),
            sensitivity_methods=(),
        )

    codes = warning_codes(recorded)
    assert codes.count("frame.validation.metric_missing_drop") == 1
    assert codes.count("frame.validation.metric_covariate_missing_imputed") == 1
    assert results
    treatment_a = next(
        row
        for row in results
        if row.metric == "value" and row.dimension_value == "A" and row.group_id == "treatment"
    )
    assert treatment_a.require_lift().value == pytest.approx(0.2)


def test_unit_summary_breakout_preserves_multiple_empty_segments_as_unavailable():
    import pandas as pd

    frame = _summary_breakout_frame()
    empty = frame[frame["segment"] == "A"].copy()
    empty["user_id"] = "empty-" + empty["user_id"]
    empty["segment"] = "C"
    valid = frame[frame["segment"] == "B"].copy()
    valid["user_id"] = "valid-" + valid["user_id"]
    valid["segment"] = "D"
    frame = pd.concat([frame, empty, valid], ignore_index=True)
    frame.loc[frame["segment"].isin(("A", "C")), "value"] = None
    with pytest.warns(IncrementWarning):
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="value", type="mean", missing="drop")],
            breakouts=("segment",),
        )

    with pytest.warns(IncrementWarning):
        rows = analysis.run_breakout()
    identities = [
        (row.dimension_value, row.metric, row.group_id, row.method, row.estimand, row.value_scale)
        for row in rows
    ]
    assert len(identities) == len(set(identities))
    for value in ("A", "C"):
        (unavailable,) = [row for row in rows if row.dimension_value == value]
        assert unavailable.excluded == "missing_metric_observations"
        assert unavailable.failure_code == "readout.cell.missing_metric_observations"
        assert unavailable.sampling_available is False
        assert unavailable.sampling_reason_code == "readout.cell.missing_metric_observations"
    assert {row.dimension_value for row in rows if row.excluded is None} == {"B", "D"}


def test_encouragement_breakout_deduplicates_multiple_empty_segments():
    import pandas as pd

    from increment.estimation.inference import Normal
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    frame = pd.DataFrame(
        [
            {
                "unit": f"{segment}-{group}-{index}",
                "arm": (
                    "control"
                    if group == "control"
                    else "treatment_2"
                    if segment in {"A", "C"}
                    else "treatment_1"
                ),
                "segment": segment,
                "y": None if segment in {"A", "C"} else 10.0 + (group != "control") + index,
                "clicked": float(group != "control" and index < 2),
            }
            for segment in ("A", "B", "C", "D")
            for group in ("control", "treatment")
            for index in range(4)
        ]
    )
    with pytest.warns(IncrementWarning):
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            metrics=[MetricSpec(name="y", type="mean", missing="drop")],
            design=Encouragement(
                control_group="control",
                uptake=UptakeSpec(fact="clicked"),
                exclusion_restriction=ExclusionRestriction(
                    justification="assignment affects y through uptake"
                ),
                min_first_stage_z=0.001,
            ),
            breakouts=("segment",),
        )

    rows = analysis.run_breakout(prior=Normal(mu=0.0, sigma=0.1))
    for value in ("A", "C"):
        segment_rows = [row for row in rows if row.dimension_value == value]
        assert segment_rows
        for unavailable in segment_rows:
            assert unavailable.posterior_available is False
            assert unavailable.posterior_reason_code == ("readout.cell.missing_metric_observations")
            assert unavailable.posterior_reason_context is not None
            assert unavailable.posterior_reason_context["reason"] == (
                "no usable metric observations"
            )
            assert unavailable.n_treat == 0.0
            assert unavailable.n_control == 0.0
            assert unavailable.low_reliability is False
            assert unavailable.delta_evidence is None
            assert (
                type(unavailable).model_validate_json(unavailable.model_dump_json()) == unavailable
            )
    identities = [
        (row.dimension_value, row.metric, row.group_id, row.estimand, row.value_scale, row.method)
        for row in rows
    ]
    assert len(identities) == len(set(identities))
    for value in ("A", "C"):
        segment_rows = [row for row in rows if row.dimension_value == value]
        assert any(row.group_id == "treatment_2" for row in segment_rows)
        assert all(row.excluded == "missing_metric_observations" for row in segment_rows)
        assert all(
            row.failure_code == "readout.cell.missing_metric_observations" for row in segment_rows
        )
    for value in ("B", "D"):
        assert any(row.dimension_value == value and row.excluded is None for row in rows)


@pytest.fixture
def encouragement_summary_breakout_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "unit": f"{segment}-{group}-{index}",
                "arm": group,
                "segment": segment,
                "y": 10.0 + (group != "control") + index,
                "x": float(index % 2),
                "clicked": float(group != "control" and index < 2),
            }
            for segment in ("A", "B")
            for group in ("control", "treatment_1", "treatment_2")
            for index in range(4)
        ]
    )


def _encouragement_summary_breakout_analysis(frame, metrics):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=metrics,
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            one_sided=True,
            exclusion_restriction=ExclusionRestriction(
                justification="assignment affects y through uptake"
            ),
            min_first_stage_z=0.001,
        ),
        breakouts=("segment",),
    )


def _encouragement_breakout_identity(row):
    return row.estimand, row.value_scale, row.method, row.method_role


@pytest.mark.parametrize("empty_arm", ["control", "treatment_2"])
def test_encouragement_summary_breakout_partitions_empty_arms(
    encouragement_summary_breakout_frame, empty_arm
):
    frame = encouragement_summary_breakout_frame
    frame.loc[(frame["segment"] == "A") & (frame["arm"] == empty_arm), "y"] = None
    with pytest.warns(IncrementWarning):
        analysis = _encouragement_summary_breakout_analysis(
            frame, [MetricSpec(name="y", missing="drop")]
        )
    rows = analysis.run_breakout()
    for group in ("treatment_1", "treatment_2"):
        actual = [row for row in rows if row.dimension_value == "A" and row.group_id == group]
        valid = [row for row in rows if row.dimension_value == "B" and row.group_id == group]
        assert actual and valid
        assert set(map(_encouragement_breakout_identity, actual)) == set(
            map(_encouragement_breakout_identity, valid)
        )
        assert all(row.lift is not None and row.failure_code is None for row in valid)
        if empty_arm in ("control", group):
            assert all(row.excluded == "missing_metric_observations" for row in actual)
            assert all(
                row.failure_code == "readout.cell.missing_metric_observations" for row in actual
            )
            assert all(row.lift is None and row.sampling_available is False for row in actual)
            if empty_arm == "control":
                assert all(row.n_treat == 4 and row.n_control == 0 for row in actual)
            else:
                assert all(row.n_treat == 0 and row.n_control == 4 for row in actual)
        else:
            assert all(row.lift is not None and row.failure_code is None for row in actual)
            assert all(row.n_treat == row.n_control == 4 for row in actual)


@pytest.mark.parametrize("include_valid_metric", [False, True])
@pytest.mark.integration
def test_encouragement_summary_breakout_keeps_metric_empty_in_every_segment(
    encouragement_summary_breakout_frame, include_valid_metric
):
    frame = encouragement_summary_breakout_frame
    frame["z"] = frame["y"] + 1.0
    metrics = [MetricSpec(name="y", missing="drop")]
    if include_valid_metric:
        metrics.append(MetricSpec(name="z"))
    baseline = _encouragement_summary_breakout_analysis(frame, metrics).run_breakout()
    frame["y"] = None
    with pytest.warns(IncrementWarning):
        analysis = _encouragement_summary_breakout_analysis(frame, metrics)
    rows = analysis.run_breakout()

    def identity(row):
        return (
            row.metric,
            row.dimension_value,
            row.group_id,
            *_encouragement_breakout_identity(row),
        )

    # Absolute compliance stays in the roster; relative compliance needs observed
    # control uptake, so an entirely unavailable metric does not invent that scale.
    expected = {
        key
        for key in map(identity, baseline)
        if not (key[0] == "y" and key[3] == "compliance" and key[4] != "absolute")
    }
    assert set(map(identity, rows)) == expected
    missing = [row for row in rows if row.metric == "y"]
    assert missing
    assert all(row.failure_code == "readout.cell.missing_metric_observations" for row in missing)
    assert all(row.excluded == "missing_metric_observations" for row in missing)
    assert all(row.n_treat == row.n_control == 0 for row in missing)
    assert all(row.low_reliability is False and row.delta_evidence is None for row in missing)
    if include_valid_metric:
        assert all(row.lift is not None for row in rows if row.metric == "z")


def test_encouragement_summary_breakout_empty_identities_match_requested_methods(
    encouragement_summary_breakout_frame,
):
    frame = encouragement_summary_breakout_frame
    frame.loc[frame["segment"] == "A", "y"] = None
    with pytest.warns(IncrementWarning):
        analysis = _encouragement_summary_breakout_analysis(
            frame,
            [
                MetricSpec(
                    name="y",
                    missing="drop",
                    sensitivity_methods=(
                        Method.cuped(covariates=CupedCovariate(name="x", kind="continuous")),
                    ),
                )
            ],
        )
    sensitivity = [Method.cuped(covariates=CupedCovariate(name="x", kind="continuous"))]
    rows = analysis.run_breakout(sensitivity_methods=sensitivity)
    for group in ("treatment_1", "treatment_2"):
        missing = [row for row in rows if row.dimension_value == "A" and row.group_id == group]
        valid = [row for row in rows if row.dimension_value == "B" and row.group_id == group]
        assert missing and valid
        identities = set(map(_encouragement_breakout_identity, missing))
        assert len(missing) == len(identities)
        assert identities == set(map(_encouragement_breakout_identity, valid))
        assert all(
            row.failure_code == "readout.cell.missing_metric_observations" for row in missing
        )
        assert all(row.lift is not None and row.failure_code is None for row in valid)
        assert all(row.value_scale == "absolute" for row in valid if row.estimand == "late")
    assert {row.estimand for row in rows} == {"itt", "compliance", "late"}


def test_unit_summary_breakout_keeps_metric_missing_in_every_segment():

    from increment import MetricSpec

    frame = _summary_breakout_frame()
    frame["missing_metric"] = None
    with pytest.warns(IncrementWarning):
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="value", type="mean"),
                MetricSpec(
                    name="missing", type="mean", value_column="missing_metric", missing="drop"
                ),
            ],
            breakouts=("segment",),
        )
    with pytest.warns(IncrementWarning):
        rows = analysis.run_breakout()
    missing_rows = [row for row in rows if row.metric == "missing"]
    assert {(row.dimension_value, row.group_id) for row in missing_rows} == {
        (segment, "treatment") for segment in ("A", "B")
    }
    assert all(row.excluded == "missing_metric_observations" for row in missing_rows)


def test_unit_summary_breakout_remaps_resolved_default_uptake():
    import pandas as pd

    from increment.frame import from_unit_summary
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(8)],
            "arm": ["control", "control", "treatment", "treatment"] * 2,
            "segment": ["A"] * 4 + ["B"] * 4,
            "y": [2.0, 3.0, 4.0, 5.0, 3.0, 5.0, 6.0, 8.0],
            "ds": [0, 0, 1, 1] * 2,
        }
    )
    source = from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics={"y": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="ds"),
            exclusion_restriction=ExclusionRestriction(
                justification="assignment only moves y through uptake"
            ),
        ),
        breakouts=("segment",),
    )

    assert readout_functions.breakout(source, "segment")


def test_unit_summary_breakout_preserves_unit_column_used_as_covariate():
    import pandas as pd

    from increment.frame import from_unit_summary

    frame = pd.DataFrame(
        {
            "ds": [101, 102, 103, 104, 105, 106, 107, 108],
            "arm": ["control", "control", "treatment", "treatment"] * 2,
            "segment": ["A"] * 4 + ["B"] * 4,
            "y": [2.0, 3.0, 4.0, 5.0, 3.0, 5.0, 6.0, 8.0],
        }
    )
    source = from_unit_summary(
        frame,
        unit="ds",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="y",
                type="mean",
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="ds", kind="continuous")
                ),
            )
        ],
        breakouts=("segment",),
    )

    assert readout_functions.breakout(source, "segment")


def test_unit_summary_breakouts_apply_drop_and_zero_per_metric():
    frame = _summary_breakout_frame()
    for index in range(8):
        for group, value in (
            ("control", 1.0 if index % 2 == 0 else 2.0),
            ("treatment", 2.0 if index % 2 == 0 else 4.0),
        ):
            frame.loc[len(frame)] = {
                "user_id": f"A-{group}-extra-{index}",
                "variant": group,
                "segment": "A",
                "value": value,
            }
    frame.loc[0, "value"] = None
    with pytest.warns(IncrementWarning):
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="drop", type="mean", value_column="value", missing="drop"),
                MetricSpec(name="zero", type="mean", value_column="value", missing="zero"),
            ],
            breakouts=("segment",),
        )

    results = analysis.run_breakout()
    treatment_by_metric = {
        metric_name: next(
            row
            for row in results
            if row.metric == metric_name
            and row.dimension_value == "A"
            and row.group_id == "treatment"
        )
        for metric_name in ("drop", "zero")
    }
    assert treatment_by_metric["drop"].require_lift().value == pytest.approx(13 / 14)
    assert treatment_by_metric["zero"].require_lift().value == pytest.approx(8 / 7)


def test_unit_summary_breakout_is_available_through_readout_api():
    analysis = Analysis.from_unit_summary(
        _summary_breakout_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"value": "mean"},
        breakouts=("segment",),
    )

    direct = analysis.run_breakout()

    assert {(row.metric, row.dimension_value) for row in direct} == {
        ("value", "A"),
        ("value", "B"),
    }
    assert direct == analysis.run_breakout()


def test_unit_summary_breakouts_allow_metric_named_ds():
    frame = _summary_breakout_frame().rename(columns={"value": "ds"})
    analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="ds", type="mean")],
        breakouts=("segment",),
    )

    assert analysis.run_breakout()


def test_unit_summary_breakouts_refuse_cluster_with_summary_capability_code():
    frame = _summary_breakout_frame()
    frame["cluster"] = [f"cluster-{i // 2}" for i in range(len(frame))]
    with pytest.raises(CapabilityError) as raised:
        Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"value": "mean"},
            cluster="cluster",
            breakouts=("segment",),
        )
    assert raised.value.code == "source.frame.cluster_capability"


@pytest.mark.parametrize("control", [True, "True", "true"])
def test_unit_summary_boolean_control_breakouts_keep_canonical_arm(control):
    import pandas as pd

    from increment.frame import from_unit_summary

    source = from_unit_summary(
        pd.DataFrame(
            {
                "unit": [f"u{i}" for i in range(8)],
                "arm": [True, True, False, False] * 2,
                "segment": ["A"] * 4 + ["B"] * 4,
                "y": [10.0, 11.0, 13.0, 14.0] * 2,
            }
        ),
        unit="unit",
        group="arm",
        control=control,
        metrics={"y": "mean"},
        breakouts=["segment"],
    )

    rows = readout_functions.breakout(source, "segment")
    assert {row.dimension_value for row in rows} == {"A", "B"}


def test_from_unit_panel_run_daily_returns_daily_metric_values():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "variant": ["treatment", "treatment", "control", "control"],
            "day": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "revenue": [10.0, 20.0, 5.0, 7.0],
        }
    )
    a = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    results = a.run_daily()
    assert isinstance(results, DailyMetricValues)
    assert len(results) > 0
    day1_treatment = next(
        r for r in results if r.group_id == "treatment" and r.ds.isoformat() == "2026-01-01"
    )
    # One unit ("u1") on day 1 - n=1 is unestimable (no ddof=1 variance),
    # so this must come back unavailable, not raise and not silently drop the row.
    assert day1_treatment.n == 1
    assert day1_treatment.value is None
    assert day1_treatment.unavailable == "few_units"


def test_day_axis_snapshot_identity_tracks_binary_input_counts():
    import pandas as pd

    rows = [
        {"user_id": unit, "variant": arm, "day": "2026-01-01", "converted": value}
        for arm, values in (
            ("control", (0, 0, 0, 0)),
            ("treatment", (0, 0, 0, 0)),
        )
        for index, value in enumerate(values)
        for unit in (f"{arm}-{index}",)
    ]
    changed_rows = [dict(row) for row in rows]
    changed_rows[-1]["converted"] = 1

    def analyze(frame):
        return Analysis.from_unit_panel(
            pd.DataFrame(frame),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"converted": "conversion"},
            experiment_id="same-study",
        )

    baseline, changed = analyze(rows), analyze(changed_rows)
    try:
        baseline_rows, changed_results = baseline.run_daily(), changed.run_daily()
        assert {row.n for row in baseline_rows} == {row.n for row in changed_results}
        assert {row.source_snapshot_id for row in baseline_rows}.isdisjoint(
            {row.source_snapshot_id for row in changed_results}
        )
    finally:
        baseline.close()
        changed.close()


@pytest.fixture
def panel_breakout_analysis():
    import pandas as pd

    rows = []
    units = [
        ("c_us_1", "control", "US", 2.0),
        ("c_us_2", "control", "US", 4.0),
        ("t_us_1", "treatment", "US", 5.0),
        ("t_us_2", "treatment", "US", 8.0),
        ("c_ca_1", "control", "CA", 6.0),
        ("c_ca_2", "control", "CA", 9.0),
        ("t_ca_1", "treatment", "CA", 3.0),
        ("t_ca_2", "treatment", "CA", 5.0),
    ]
    for unit, variant, country, base in units:
        rows.append(
            {
                "user_id": unit,
                "variant": variant,
                "day": "2026-01-01",
                "country": country,
                "plan": "all",
                "revenue": base,
                "orders": base / 2.0,
            }
        )
        rows.append(
            {
                "user_id": unit,
                "variant": variant,
                "day": "2026-01-02",
                "country": country,
                "plan": "all",
                "revenue": base + (1.0 if unit.endswith("1") else 2.0),
                "orders": base / 2.0 + 1.0,
            }
        )
    return Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean", "orders": "mean"},
        breakouts=["country", "plan"],
    )


def test_from_unit_panel_daily_lift_refuses_sequential_before_moments():
    from increment.frame import synthesise_metric
    from increment.semantics.design import Randomized
    from increment.sequential_source import frame_observation_mapping
    from tests.sequential_cases import declared_plan

    specs = [MetricSpec(name="revenue", type="conversion")]
    plan = declared_plan(
        [synthesise_metric(s) for s in specs],
        source_id="frame",
        design=Randomized(control_group="control"),
        transformations=specs,
        source_mapping=frame_observation_mapping(unit="user_id", group="variant", date="day"),
    )
    import pandas as pd

    rows = [
        {"user_id": f"c{i}", "variant": "control", "day": "2026-01-01", "revenue": float(i % 2)}
        for i in range(2)
    ] + [
        {
            "user_id": f"t{i}",
            "variant": "treatment",
            "day": "2026-01-01",
            "revenue": float((i + 1) % 2),
        }
        for i in range(2)
    ]
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "conversion"},
        plan=plan,
    )

    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run_daily_lift()
    assert raised.value.code == "readout.inference.disjoint_slices"


def test_from_unit_panel_daily_lift_refuses_cuped_without_covariate(panel_breakout_analysis):
    with pytest.raises(InvalidRequestError) as raised:
        panel_breakout_analysis.run_daily_lift(
            decision_method=Method.cuped(
                covariates=CupedCovariate(name="revenue", kind="continuous")
            ),
            metrics=["revenue"],
        )
    assert raised.value.code == "estimation.cuped.arm_no_covariate"


def test_from_unit_panel_quantile_cuped_refusal_is_preserved():
    import pandas as pd

    rows = [
        {"user_id": "c1", "variant": "control", "day": "2026-01-01", "revenue": 1.0},
        {"user_id": "c2", "variant": "control", "day": "2026-01-01", "revenue": 2.0},
        {"user_id": "t1", "variant": "treatment", "day": "2026-01-01", "revenue": 3.0},
        {"user_id": "t2", "variant": "treatment", "day": "2026-01-01", "revenue": 4.0},
    ]
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_panel(
            pd.DataFrame(rows),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="quantile",
                    quantile=0.5,
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(name="revenue", kind="continuous")
                    ),
                )
            ],
        )
    assert raised.value.code == "frame.metric.cuped_does_apply"


def test_from_unit_summary_clustered_cuped_refusal_is_preserved():
    import pandas as pd

    rows = [
        {
            "user_id": "c1",
            "variant": "control",
            "store": "s1",
            "day": "2026-01-01",
            "revenue": 1.0,
            "pre_revenue": 0.5,
        },
        {
            "user_id": "c2",
            "variant": "control",
            "store": "s1",
            "day": "2026-01-01",
            "revenue": 2.0,
            "pre_revenue": 1.0,
        },
        {
            "user_id": "t1",
            "variant": "treatment",
            "store": "s2",
            "day": "2026-01-01",
            "revenue": 3.0,
            "pre_revenue": 2.0,
        },
        {
            "user_id": "t2",
            "variant": "treatment",
            "store": "s2",
            "day": "2026-01-01",
            "revenue": 4.0,
            "pre_revenue": 2.5,
        },
    ]
    with pytest.raises(CapabilityError) as raised:
        Analysis.from_unit_summary(
            pd.DataFrame(rows),
            unit="user_id",
            group="variant",
            control="control",
            cluster="store",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(name="pre_revenue", kind="continuous")
                    ),
                )
            ],
        )
    assert raised.value.code == "source.frame.cluster_capability"


def _configured_panel_breakout_analysis(correction: Literal["none", "bonferroni", "bh"]):
    from increment.estimation import Normal

    return _panel_breakout_analysis_with_prior(correction, Normal(mu=0.0, sigma=0.01))


def _panel_breakout_analysis_with_prior(correction: Literal["none", "bonferroni", "bh"], prior):
    import pandas as pd

    from increment.semantics.models import MultiplicitySpec

    metric = MetricSpec(
        name="revenue",
        decision_method=Method(name="declared"),
        prior=prior,
    )
    return Analysis.from_unit_panel(
        pd.DataFrame(_panel_with_breakout_rows_for_config()),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["country"],
        plan=AnalysisPlan(
            view_multiplicity=MultiplicitySpec(correction=correction),
        ),
    )


def _panel_with_breakout_rows_for_config() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    units = [
        ("c_us_1", "control", "US", 2.0),
        ("c_us_2", "control", "US", 4.0),
        ("t_us_1", "treatment", "US", 5.0),
        ("t_us_2", "treatment", "US", 8.0),
        ("c_ca_1", "control", "CA", 6.0),
        ("c_ca_2", "control", "CA", 9.0),
        ("t_ca_1", "treatment", "CA", 3.0),
        ("t_ca_2", "treatment", "CA", 5.0),
    ]
    for unit, variant, country, base in units:
        rows.append(
            {
                "user_id": unit,
                "variant": variant,
                "day": "2026-01-01",
                "country": country,
                "revenue": base,
            }
        )
        rows.append(
            {
                "user_id": unit,
                "variant": variant,
                "day": "2026-01-02",
                "country": country,
                "revenue": base + (1.0 if unit.endswith("1") else 2.0),
            }
        )
    return rows


@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
def test_panel_breakout_uses_callwide_methods_not_declared_methods(correction):
    analysis = _configured_panel_breakout_analysis(correction)

    results = analysis.run_breakout()
    assert results
    assert {row.method for row in results} == {"declared"}

    explicit = analysis.run_breakout(decision_method=Method(name="call-wide"))
    assert explicit
    assert {row.method for row in explicit} == {"call-wide"}


@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
def test_panel_breakout_keeps_declared_prior_separate_from_sampling(correction):
    from increment.estimation import Normal

    declared_prior = Normal(mu=0.0, sigma=0.01)
    with_declared_prior = _panel_breakout_analysis_with_prior(correction, declared_prior)
    without_declared_prior = _panel_breakout_analysis_with_prior(correction, None)

    declared_results = with_declared_prior.run_breakout()
    undeclared_results = without_declared_prior.run_breakout()
    for declared, undeclared in zip(declared_results, undeclared_results, strict=True):
        assert declared.require_lift().value == pytest.approx(undeclared.require_lift().value)
        assert declared.posterior_estimate is not None
        assert undeclared.posterior_estimate is None

    override_results = without_declared_prior.run_breakout(prior=declared_prior)
    assert [row.require_lift().value for row in override_results] == pytest.approx(
        [row.require_lift().value for row in declared_results]
    )
    assert [row.posterior_estimate for row in override_results] == pytest.approx(
        [row.posterior_estimate for row in declared_results]
    )


def test_from_unit_panel_run_daily_lift_default(panel_breakout_analysis):
    results = panel_breakout_analysis.run_daily_lift()
    assert isinstance(results, DailyLiftEstimates)
    assert {(row.ds.isoformat(), row.group_id) for row in results} == {
        ("2026-01-01", "treatment"),
        ("2026-01-02", "treatment"),
    }
    assert all(row.dimension is None and row.source is None for row in results)


@pytest.mark.parametrize("method", ["run_daily", "run_asof"])
def test_from_unit_panel_day_axis_winsorization_refusal_is_coded(method):
    import pandas as pd

    analysis = Analysis.from_unit_panel(
        pd.DataFrame(_panel_with_breakout_rows_for_config()),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_value": 100.0})],
    )

    with pytest.raises(CapabilityError) as raised:
        getattr(analysis, method)()
    assert raised.value.code == "evidence.operation_unavailable"
    assert raised.value.context["requested"] == ("revenue",)


def test_from_unit_panel_daily_readouts_by_dimension(panel_breakout_analysis):
    values = panel_breakout_analysis.run_daily(dimension="country")
    lifts = panel_breakout_analysis.run_daily_lift(dimension="country")
    assert {row.dimension_value for row in values} == {"US", "CA"}
    assert {row.dimension_value for row in lifts} == {"US", "CA"}
    assert all(row.dimension == "country" and row.source is None for row in [*values, *lifts])


def test_from_unit_panel_run_asof_lift_rejects_policy_overrides(panel_breakout_analysis):
    with pytest.raises(TypeError):
        panel_breakout_analysis.run_asof_lift(
            dimension="country",
            inference=registered_spec(),
        )
    with pytest.raises(TypeError):
        panel_breakout_analysis.run_asof_lift(alpha=0.02)
    with pytest.raises(TypeError):
        panel_breakout_analysis.run_asof_lift(correction="bonferroni")


def test_from_unit_panel_run_asof_values(panel_breakout_analysis):
    values = panel_breakout_analysis.run_asof(dimension="country")
    assert isinstance(values, DailyMetricValues)
    assert {row.dimension_value for row in values} == {"US", "CA"}
    assert all(row.ds_basis == "calendar" and row.source is None for row in values)


def test_from_unit_panel_run_asof_lift(panel_breakout_analysis):
    lifts = panel_breakout_analysis.run_asof_lift(dimension="country")
    assert isinstance(lifts, DailyLiftEstimates)
    assert {row.dimension_value for row in lifts} == {"US", "CA"}
    assert all(row.ds_basis == "calendar" and row.source is None for row in lifts)


def test_from_unit_panel_run_asof_lift_rejects_inference_override(panel_breakout_analysis):
    with pytest.raises(TypeError):
        panel_breakout_analysis.run_asof_lift(inference="always_valid")


def test_from_unit_panel_asof_validates_dimensions_and_filters_metrics(
    panel_breakout_analysis,
):
    for method in (panel_breakout_analysis.run_asof, panel_breakout_analysis.run_asof_lift):
        with pytest.raises(InvalidRequestError) as raised:
            method(dimension="unknown")
        assert raised.value.code == "facade.analysis.unknown_dimension"
        filtered = method(metrics=["revenue"])
        assert filtered != []
        assert {row.metric for row in filtered} == {"revenue"}


@pytest.fixture
def bounded_retention_panel_analysis():
    import pandas as pd

    base = date(2026, 1, 1)
    rows = [
        {
            "user_id": unit,
            "variant": group,
            "day": base + timedelta(days=day),
            "exposed_on": base,
            "returned": float(day == 1 and unit.endswith("1")),
        }
        for unit, group in (
            ("c1", "control"),
            ("c2", "control"),
            ("t1", "treatment"),
            ("t2", "treatment"),
        )
        for day in range(3)
    ]
    return Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="returned",
                type="retention",
                value_column="returned",
                threshold_days=(0, 2),
            )
        ],
        exposure_date="exposed_on",
    )


def test_from_unit_panel_asof_completed_windows_only_emits_mature_retention_rows(
    bounded_retention_panel_analysis,
):
    monitoring = bounded_retention_panel_analysis.run_asof()
    decision = bounded_retention_panel_analysis.run_asof(completed_windows_only=True)
    monitoring_lift = bounded_retention_panel_analysis.run_asof_lift()
    decision_lift = bounded_retention_panel_analysis.run_asof_lift(completed_windows_only=True)

    assert min(row.ds for row in monitoring) == date(2026, 1, 1)
    assert min(row.ds for row in decision) == date(2026, 1, 3)
    assert min(row.ds for row in monitoring_lift) == date(2026, 1, 1)
    assert min(row.ds for row in decision_lift) == date(2026, 1, 3)


def test_from_unit_panel_run_breakout_returns_every_declaration(panel_breakout_analysis):
    default = panel_breakout_analysis.run_breakout()
    assert {row.dimension for row in default} == {"country", "plan"}
    assert all(row.source is None for row in default)


def test_from_unit_panel_dimension_must_be_declared(panel_breakout_analysis):
    with pytest.raises(InvalidRequestError) as raised:
        panel_breakout_analysis.run_daily(dimension="unknown")
    assert raised.value.code == "facade.analysis.unknown_dimension"


def test_from_unit_panel_run_daily_filters_by_name(panel_breakout_analysis):
    # sm8m repro: exclude siblings by constructor-declared name
    values = panel_breakout_analysis.run_daily(metrics=["revenue"])
    assert {row.metric for row in values} == {"revenue"}


def test_from_unit_panel_run_daily_lift_filters_by_name(panel_breakout_analysis):
    lifts = panel_breakout_analysis.run_daily_lift(metrics=["revenue"])
    assert lifts != []
    assert {row.metric for row in lifts} == {"revenue"}


def test_from_unit_panel_metrics_unknown_name_raises(panel_breakout_analysis):
    with pytest.raises(InvalidRequestError) as raised:
        panel_breakout_analysis.run_daily(metrics=["revnue"])
    assert raised.value.code == "facade.analysis_config.unknown_metric_declared"


def test_from_unit_panel_metrics_duplicate_name_raises(panel_breakout_analysis):
    with pytest.raises(InvalidRequestError) as raised:
        panel_breakout_analysis.run_daily(metrics=["revenue", "revenue"])
    assert raised.value.code == "facade.analysis_config.duplicate_metric_name"


def test_from_unit_panel_metrics_accepts_declared_metric_object(panel_breakout_analysis):
    metric = cast(MeanMetric, panel_breakout_analysis.metrics[0])
    values = panel_breakout_analysis.run_daily(metrics=[metric])
    assert {row.metric for row in values} == {metric.name}


def test_from_unit_panel_run_breakout_filters_by_name(panel_breakout_analysis):
    filtered = panel_breakout_analysis.run_breakout(metrics=["revenue"])
    assert filtered != []
    assert {row.metric for row in filtered} == {"revenue"}
    assert {row.dimension for row in filtered} == {"country", "plan"}


def test_from_unit_panel_filtered_margins_key_raises_before_moments(panel_breakout_analysis):
    """margins= is refused outright on the frame/moments seam path now
    (Analysis.run() has no execute() dispatch to forward it to) - a key
    naming a declared-but-FILTERED metric never even reaches the old
    per-key unknown-metric check; the refusal is broader, not narrower,
    than "never silently dropped"."""
    with pytest.raises(TypeError):
        lift_rows(panel_breakout_analysis.run(metrics=["revenue"], margins={"orders": 0.01}))


def test_from_unit_summary_run_filters_by_name(unit_summary_analysis):
    results = lift_rows(unit_summary_analysis.run(metrics=["revenue"]))
    assert {r.metric for r in results} == {"revenue"}


def test_from_unit_panel_run_daily_lift_forwards_methods(panel_breakout_analysis):
    rows = panel_breakout_analysis.run_daily_lift(sensitivity_methods=())
    assert rows and {row.method for row in rows} == {"unadjusted"}


def test_from_unit_summary_run_breakout_remains_unsupported(unit_summary_analysis) -> None:
    with pytest.raises(CapabilityError) as raised:
        unit_summary_analysis.run_breakout()
    assert raised.value.code == "evidence.breakout_unavailable"


def test_artifact_context_refuses_unit_summary_source_with_operation_context(
    unit_summary_analysis,
):
    with pytest.raises(CapabilityError) as raised:
        _ = unit_summary_analysis.artifact_context

    assert raised.value.code == "evidence.operation_unavailable"
    assert raised.value.context["requested"] == "materialize"


def test_from_unit_panel_encouragement_breakout_retains_estimands():
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = []
    for unit, variant, country, base in [
        ("c_us_1", "control", "US", 2.0),
        ("c_us_2", "control", "US", 4.0),
        ("t_us_1", "treatment", "US", 5.0),
        ("t_us_2", "treatment", "US", 8.0),
        ("c_ca_1", "control", "CA", 6.0),
        ("c_ca_2", "control", "CA", 9.0),
        ("t_ca_1", "treatment", "CA", 3.0),
        ("t_ca_2", "treatment", "CA", 5.0),
    ]:
        rows.append(
            {
                "user_id": unit,
                "variant": variant,
                "day": "2026-01-01",
                "country": country,
                "revenue": base,
                "clicked": float(variant == "treatment" and unit.endswith("1")),
            }
        )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            justification="assignment only moves revenue via uptake"
        ),
        min_first_stage_z=0.001,
    )
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=design,
        uptake="clicked",
        breakouts=["country"],
    )
    assert {row.estimand for row in analysis.run_breakout()} == {"itt", "compliance", "late"}
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift()
    assert raised.value.code == "facade.analysis.encouragement_daily_late"


@pytest.fixture
def encouragement_panel_analysis():
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = [
        {
            "user_id": unit,
            "variant": variant,
            "day": "2026-01-01",
            "revenue": revenue,
            "clicked": clicked,
        }
        for unit, variant, revenue, clicked in (
            ("c1", "control", 2.0, 0.0),
            ("c2", "control", 4.0, 0.0),
            ("t1", "treatment", 6.0, 1.0),
            ("t2", "treatment", 8.0, 0.0),
        )
    ]
    return Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                justification="assignment only moves revenue via uptake"
            ),
            min_first_stage_z=0.001,
        ),
        uptake="clicked",
    )


def test_compliance_asof_snapshot_identity_tracks_arm_counts():
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    def analysis(units_per_arm: int):
        rows = []
        for arm, variant in (("control", 0), ("treatment", 1)):
            for unit in range(units_per_arm):
                clicked = float(variant and unit % 2 == 0)
                for day in ("2026-01-01", "2026-01-02"):
                    rows.append(
                        {
                            "unit": f"{arm}-{unit}",
                            "variant": arm,
                            "day": day,
                            "revenue": float(unit),
                            "clicked": clicked,
                        }
                    )
        return Analysis.from_unit_panel(
            pd.DataFrame(rows),
            unit="unit",
            group="variant",
            date="day",
            metrics={"revenue": "mean"},
            design=Encouragement(
                control_group="control",
                uptake=UptakeSpec(fact="clicked"),
                exclusion_restriction=ExclusionRestriction(
                    justification="assignment only changes uptake"
                ),
                min_first_stage_z=0.001,
            ),
            uptake="clicked",
        )

    small = analysis(100)
    large = analysis(120)
    try:
        small_rows = list(small.run_asof_lift(estimands=("compliance",)))
        large_rows = list(large.run_asof_lift(estimands=("compliance",)))
        assert small_rows and large_rows
        assert all(row.n_control == 100 and row.n_treat == 100 for row in small_rows)
        assert all(row.n_control == 120 and row.n_treat == 120 for row in large_rows)
        assert {row.source_snapshot_id for row in small_rows}.isdisjoint(
            {row.source_snapshot_id for row in large_rows}
        )
    finally:
        small.close()
        large.close()


def test_from_unit_panel_run_asof_lift_encouragement_estimands(encouragement_panel_analysis):
    itt = encouragement_panel_analysis.run_asof_lift(estimands=("itt",))
    late = encouragement_panel_analysis.run_asof_lift(estimands=("late",))

    assert itt and {row.estimand for row in itt} == {"itt"}
    assert late and {row.estimand for row in late} == {"late"}


def test_asof_compliance_keeps_estimable_arm_when_another_arm_is_too_small():
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = []
    for group, count, uptake_count in (
        ("control", 8, 0),
        ("treatment_a", 8, 4),
        ("treatment_b", 1, 1),
    ):
        for index in range(count):
            rows.append(
                {
                    "unit": f"{group}-{index}",
                    "variant": group,
                    "day": date(2026, 1, 1),
                    "revenue": float(index),
                    "clicked": float(index < uptake_count),
                }
            )
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="unit",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                justification="assignment only affects uptake"
            ),
        ),
        uptake="clicked",
    )
    try:
        result = analysis.run_asof_lift(estimands=("compliance",))
    finally:
        analysis.close()

    by_group = {row.group_id: row for row in result}
    assert set(by_group) == {"treatment_a", "treatment_b"}
    assert by_group["treatment_a"].failure_code is None
    assert by_group["treatment_a"].require_lift().value is not None
    assert by_group["treatment_b"].failure_code == "readout.cell.missing_metric_observations"


def test_from_unit_panel_run_asof_late_honors_requested_methods(encouragement_panel_analysis):
    cleared = encouragement_panel_analysis.run_asof_lift(
        sensitivity_methods=(), estimands=("late",)
    )
    assert cleared and {row.method for row in cleared} == {"unadjusted"}

    results = encouragement_panel_analysis.run_asof_lift(
        decision_method=Method(name="panel-unadjusted"), estimands=("late",)
    )

    assert results
    assert {row.method for row in results} == {"panel-unadjusted"}


def test_from_unit_panel_run_asof_late_refuses_cuped(encouragement_panel_analysis):
    with pytest.raises(InvalidRequestError) as raised:
        encouragement_panel_analysis.run_asof_lift(
            decision_method=Method.cuped(),
            estimands=("late",),
        )
    assert raised.value.code == "estimation.cuped.arm_no_covariate"


def _make_windowed_encouragement_panel_analysis(
    *,
    control_uptake: bool = False,
    one_sided: bool = True,
    plan: AnalysisPlan | None = None,
    sequential: bool = False,
):
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    start = date(2026, 1, 1)
    rows = []
    for arm in ("control", "treatment"):
        for index in range(50):
            unit = f"{arm}-{index}"
            country = "US" if index % 2 == 0 else "CA"
            for offset in range(5):
                treatment_click = arm == "treatment" and (
                    (index == 0 and offset == 0) or (1 <= index < 31 and offset == 1)
                )
                clicked = treatment_click or (
                    control_uptake and arm == "control" and index == 0 and offset == 0
                )
                rows.append(
                    {
                        "user_id": unit,
                        "variant": arm,
                        "day": start + timedelta(days=offset),
                        "exposed_on": start,
                        "country": country,
                        "revenue": float(10 + index % 5 + (4 if arm == "treatment" else 0))
                        if offset == 0
                        else 0.0,
                        "converted": float(offset == 0),
                        "clicked": float(clicked),
                    }
                )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=2),
        exclusion_restriction=ExclusionRestriction(
            justification="assignment affects revenue only through uptake",
        ),
        one_sided=one_sided,
        min_first_stage_z=4.0,
    )
    metrics = [MetricSpec(name="revenue", type="mean", window_days=2)]
    if sequential:
        from increment.frame import synthesise_metric
        from increment.sequential_source import frame_observation_mapping
        from tests.sequential_cases import declared_plan

        # Sequential encouragement plans admit only Bernoulli or registered
        # scalar-mean observations (validate_sequential_plan), and scalar-mean
        # registration needs a fixed randomized design, so raw ITT over
        # Encouragement needs a Bernoulli outcome.
        specs = [MetricSpec(name="converted", type="conversion", window_days=2)]
        metrics = specs
        plan = declared_plan(
            [synthesise_metric(spec) for spec in specs],
            source_id="frame",
            design=design,
            transformations=specs,
            source_mapping=frame_observation_mapping(
                unit="user_id",
                group="variant",
                date="day",
                exposure_date="exposed_on",
                uptake="clicked",
            ),
        )
    return Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        metrics=metrics,
        design=design,
        uptake="clicked",
        exposure_date="exposed_on",
        breakouts=["country"],
        plan=plan,
    )


@pytest.fixture
def windowed_encouragement_panel_analysis():
    return _make_windowed_encouragement_panel_analysis()


def test_from_unit_panel_asof_encouragement_weak_then_strong(
    windowed_encouragement_panel_analysis,
):
    results = windowed_encouragement_panel_analysis.run_asof_lift()
    by_day = defaultdict(set)
    for row in results:
        by_day[row.ds].add(row.estimand)
    first, second = sorted(by_day)[:2]
    assert by_day[first] == {"itt", "compliance"}
    assert by_day[second] == {"itt", "compliance", "late"}
    compliance = [row for row in results if row.estimand == "compliance"]
    assert {row.metric for row in compliance} == {"uptake", "revenue_uptake"}
    # A weak first stage suppresses the late readout on the first day only:
    # the revenue_uptake compliance row stands in for it there and nowhere else.
    assert {(row.metric, row.ds) for row in compliance if row.metric == "revenue_uptake"} == {
        ("revenue_uptake", first)
    }
    late_pairs = {(row.metric, row.ds) for row in results if row.estimand == "late"}
    assert late_pairs == {("revenue", day) for day in sorted(by_day)[1:]}


def test_from_unit_panel_asof_encouragement_dimension_has_same_estimands(
    windowed_encouragement_panel_analysis,
):
    results = windowed_encouragement_panel_analysis.run_asof_lift(dimension="country")
    assert {row.dimension_value for row in results} == {"US", "CA"}
    assert {row.estimand for row in results} == {"itt", "compliance", "late"}


def test_from_unit_panel_encouragement_raw_itt_requires_finalized_checkpoint():
    analysis = _make_windowed_encouragement_panel_analysis(sequential=True)
    with pytest.raises(CapabilityError) as raised:
        analysis.run_asof_lift(estimands=("late",), completed_windows_only=True)
    assert raised.value.code == "sequential.route.unsupported"
    snapshot = analysis.capture_sequential(finalized=True, as_of=date(2026, 1, 15))
    results = analysis.run_asof_lift(estimands=("itt",), completed_windows_only=True)
    assert results and len(snapshot.records) == 100
    assert all(row.sequential_result is not None for row in results)
    assert all(row.n_control == row.n_treat == 50 for row in results)
    assert {row.ds for row in results} == {date(2026, 1, 15)}


def test_from_unit_panel_asof_one_sided_control_uptake_still_raises():
    analysis = _make_windowed_encouragement_panel_analysis(control_uptake=True)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof_lift()
    assert raised.value.code == "estimation.encouragement.one_sided_encouragement"


@pytest.mark.integration
def test_from_unit_panel_asof_two_sided_allows_control_uptake():
    analysis = _make_windowed_encouragement_panel_analysis(control_uptake=True, one_sided=False)
    results = analysis.run_asof_lift()
    assert results
    assert {row.estimand for row in results} >= {"itt", "compliance"}


def test_encouragement_always_valid_rejects_call_time_inference_override(
    encouragement_panel_analysis,
):
    with pytest.raises(TypeError):
        encouragement_panel_analysis.run_asof_lift(inference="always_valid")


def test_from_unit_panel_encouragement_breakout_skips_missing_control_segment():
    """A treatment-only segment warns and leaves valid encouragement estimates intact."""
    import pandas as pd

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = []
    for unit, variant, country, base in [
        ("c_us_1", "control", "US", 2.0),
        ("c_us_2", "control", "US", 4.0),
        ("t_us_1", "treatment", "US", 5.0),
        ("t_us_2", "treatment", "US", 8.0),
        ("t_zz_1", "treatment", "ZZ", 3.0),
        ("t_zz_2", "treatment", "ZZ", 6.0),
    ]:
        rows.append(
            {
                "user_id": unit,
                "variant": variant,
                "day": "2026-01-01",
                "country": country,
                "revenue": base,
                "clicked": float(variant == "treatment" and unit.endswith("1")),
            }
        )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            justification="assignment only moves revenue via uptake"
        ),
        min_first_stage_z=0.001,
    )
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=design,
        uptake="clicked",
        breakouts=["country"],
    )

    with pytest.warns(IncrementWarning) as rec:
        results = analysis.run_breakout()
    assert any(
        isinstance(w.message, IncrementWarning)
        and w.message.code == "readouts.breakout.segment_no_control_arm"
        and w.message.context["value"] == "ZZ"
        for w in rec
    )

    assert results
    assert {row.dimension_value for row in results} == {"US"}


def test_encouragement_breakout_all_control_free_segments_warn_and_skip():
    """Control-free encouragement segments warn/skip after dispatch validation."""
    from types import SimpleNamespace

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.sources import SourceContext

    rows = [{"group_id": "treatment", "country": country} for country in ("US", "CA")]
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            justification="assignment only moves revenue via uptake"
        ),
    )
    metrics = [MeanMetric(name="revenue", entity="user_id", fact="purchase")]
    from increment._analysis_config import resolve_configs

    context = SourceContext(
        study_id="test",
        design=design,
        plan=compile_decision_plan(None, metrics, path="warehouse"),
        metrics=tuple(metrics),
        configs=resolve_configs(
            metrics,
            bindings=None,
            specs=None,
            methods=None,
            prior=None,
        ),
        cluster=None,
    )
    src = cast(
        "MomentSource",
        SimpleNamespace(
            operations=frozenset(),
            context=context,
            breakouts=("country",),
            moments=lambda *_args, **_kwargs: rows,
        ),
    )

    with pytest.warns(IncrementWarning) as rec:
        assert readout_functions.breakout(src, "country", correction="none") == []
    assert "readouts.breakout.segment_no_control_arm" in warning_codes(rec)

    with pytest.raises(InvalidRequestError) as raised:
        readout_functions.breakout(src, "country", estimands=("invalid",), correction="none")
    assert raised.value.code == "readout.estimands.unknown"
    with pytest.raises(InvalidRequestError) as raised:
        readout_functions.breakout(
            src, "country", decision_method=Method(name="iptw"), correction="none"
        )
    assert raised.value.code == "estimation.engine.method_name_observational"


def test_from_unit_panel_observational_daily_only():
    import pandas as pd

    rows = [
        {
            "user_id": f"{variant}_{country}",
            "variant": variant,
            "day": "2026-01-01",
            "country": country,
            "pre": float(index),
            "revenue": float(index + 1),
        }
        for index, (variant, country) in enumerate(
            [("control", "US"), ("treatment", "US"), ("control", "CA"), ("treatment", "CA")]
        )
    ]
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("pre",))
        ),
        breakouts=["country"],
    )
    assert {row.dimension_value for row in analysis.run_daily(dimension="country")} == {"US", "CA"}
    assert analysis.run_asof()
    for method in (analysis.run_asof_lift, analysis.run_daily_lift):
        with pytest.raises(UnsupportedRequestError) as raised:
            method()
        assert raised.value.code == "facade.analysis.observational_day_axis"
    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run_breakout()
    assert raised.value.code == "readout.view.observational"


def test_from_unit_panel_sql_and_materialize_raise():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "variant": ["treatment", "control"],
            "day": ["2026-01-01", "2026-01-01"],
            "revenue": [10.0, 5.0],
        }
    )
    a = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError):
        a.panel_sql()
    with pytest.raises(CapabilityError):
        a.summary_sql()
    with pytest.raises(CapabilityError):
        a.materialize()
    a.close()


def test_from_unit_panel_forwards_exposure_date_for_windowed_metric():
    """A windowed metric works end-to-end through the facade: exposure_date
    forwards to increment.frame.from_unit_panel and run() estimates."""
    import pandas as pd

    rows = []
    for uid, variant, vals in [
        ("u1", "treatment", [10.0, 20.0, 30.0, 40.0]),
        ("u2", "treatment", [12.0, 18.0, 25.0, 33.0]),
        ("u3", "control", [8.0, 9.0, 10.0, 11.0]),
        ("u4", "control", [7.0, 8.0, 9.0, 10.0]),
    ]:
        for day, v in enumerate(vals):
            rows.append((uid, variant, f"2026-01-{day + 1:02d}", "2026-01-01", v))
    df = pd.DataFrame(rows, columns=["user_id", "variant", "day", "exposed_on", "revenue"])
    a = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
    )
    results = lift_rows(a.run())
    assert len(results) == 1
    assert results[0].metric == "revenue"
    assert math.isfinite((results[0]).require_lift().value)


def test_from_unit_panel_windowed_metric_without_exposure_date_raises():
    """The factory's pointed refusal surfaces through the facade."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "variant": ["treatment", "control"],
            "day": ["2026-01-01", "2026-01-01"],
            "revenue": [10.0, 5.0],
        }
    )
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=3)],
        )
    assert raised.value.code == "source.panel.exposure_date"


def test_from_unit_panel_forwards_observation_end_censoring():
    """observation_end forwards: units enrolled too late for their window to
    close by it are censored, and dropping >10% warns."""
    import pandas as pd

    base = date(2026, 1, 1)
    # u0-u7 enroll on day 0 (windows close exactly at observation_end); u8/u9 enroll on day 4, so their 3-day windows can't have closed by base+3, censoring both (20% of enrolled units -> warning).
    rows = []
    for i in range(8):
        group = "control" if i % 2 == 0 else "treatment"
        rows.append((f"u{i}", group, base, base, float(i + 1)))
    for i, group in [(8, "control"), (9, "treatment")]:
        late = base + timedelta(days=4)
        rows.append((f"u{i}", group, late, late, 100.0))
    df = pd.DataFrame(rows, columns=["user_id", "variant", "day", "exposed_on", "revenue"])
    a = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
        observation_end=base + timedelta(days=3),
    )
    with pytest.warns(IncrementWarning) as rec:
        results = lift_rows(a.run())
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert len(results) == 1
    assert math.isfinite((results[0]).require_lift().value)


def test_frame_functions_no_longer_exported():
    import increment

    # replaced by Analysis.from_unit_summary / Analysis.from_unit_panel; __getattr__ called directly to dodge ruff's B018/B009 autofix ping-pong on a bare "module.attr" expression.
    with pytest.raises(AttributeError):
        increment.__getattr__("from_unit_summary")
    with pytest.raises(AttributeError):
        increment.__getattr__("from_unit_panel")


def test_frame_panel_day_axis_refuses_an_undeclared_metric_object():
    """A call-time Metric object the panel never declared has no moments on
    the frame route; the day-axis views refuse it instead of returning []."""
    import datetime as dt

    import pandas as pd

    from increment.analysis import Analysis
    from increment.semantics.models import MeanMetric

    frame = pd.DataFrame(
        {
            "uid": [1, 1, 2, 2, 3, 3, 4, 4],
            "v": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "ds": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)] * 4,
            "exposure_date": [dt.date(2025, 1, 1)] * 8,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    analysis = Analysis.from_unit_panel(
        frame,
        unit="uid",
        group="v",
        control="a",
        date="ds",
        exposure_date="exposure_date",
        metrics={"y": "mean"},
    )
    undeclared = MeanMetric(name="zzz", entity="unit", fact="f")
    # select_metrics' unknown-metric refusal is a bare ValueError; a
    # typed code lands with the _analysis_config migration.
    for method in (analysis.run_daily, analysis.run_asof):
        with pytest.raises(ValueError):
            method(metrics=[undeclared])


@pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
def test_panel_daily_lift_accepts_midnight_iso_datetime_day_labels(backend):
    """Timestamp-shaped strings represent a calendar day only at midnight."""
    import pandas as pd
    import pyarrow as pa

    from increment.semantics.models import AnalysisPlan

    rows = [
        {
            "unit": f"{arm}-{unit}",
            "arm": arm,
            "day": "2025-01-01T00:00:00",
            "revenue": float(unit + (10 if arm == "treatment" else 1)),
        }
        for arm in ("control", "treatment")
        for unit in range(8)
    ]
    frame = pd.DataFrame(rows) if backend == "pandas" else pa.Table.from_pylist(rows)
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        control="control",
        date="day",
        metrics=[MetricSpec(name="revenue")],
        plan=AnalysisPlan(primary="revenue"),
    )

    result = analysis.run_daily_lift()
    assert result[0].ds == date(2025, 1, 1)
    from increment.breakout.estimates import DailyLiftEstimate

    serialized = result[0].model_dump_json()
    restored = DailyLiftEstimate.model_validate_json(serialized)
    assert restored.ds == result[0].ds


@pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
def test_panel_refuses_non_midnight_datetime_string_day_labels(backend):
    import pandas as pd
    import pyarrow as pa

    rows = [
        {"unit": f"{arm}-{unit}", "arm": arm, "day": "2025-01-01T12:00:00", "revenue": 1.0}
        for arm in ("control", "treatment")
        for unit in range(2)
    ]
    frame = pd.DataFrame(rows) if backend == "pandas" else pa.Table.from_pylist(rows)
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_panel(
            frame,
            unit="unit",
            group="arm",
            control="control",
            date="day",
            metrics=[MetricSpec(name="revenue")],
        )
    assert raised.value.code == "source.panel.day_grain_columns"
