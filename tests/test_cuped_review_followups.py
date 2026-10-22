from __future__ import annotations

import copy
import pickle
from typing import Any, Literal, cast

import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec, MultiplicitySpec
from increment.breakout.estimates import BreakoutEstimates
from increment.errors import IncrementWarning, UnsupportedRequestError
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.inference import Normal
from increment.estimation.readout_types import ReadoutResults
from increment.estimation.results import LiftEstimate
from increment.frame import CupedCovariate
from increment.semantics.models import ExperimentMetric

LINEAR = Method.cuped_multicovariate(
    name="linear", covariates=CupedCovariate(name="pre", kind="continuous")
)


def _unit_rows(*, panel: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ("control", "treatment"):
        for index in range(16):
            row: dict[str, Any] = {
                "unit": f"{group}-{index}",
                "group": group,
                "pre": float(index % 5),
                "revenue": float(index + (group == "treatment") * 3),
            }
            if panel:
                row["day"] = 1
            rows.append(row)
    return rows


def _run_with_covariate_name(name: str, *, panel: bool) -> LiftEstimate:
    rows = [{**row, name: row["pre"]} for row in _unit_rows(panel=panel)]
    if name != "pre":
        rows = [{key: value for key, value in row.items() if key != "pre"} for row in rows]
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name=name, kind="continuous")
        ),
    )
    table = pa.Table.from_pylist(rows)
    if panel:
        analysis = Analysis.from_unit_panel(
            table, unit="unit", group="group", date="day", control="control", metrics=[metric]
        )
    else:
        analysis = Analysis.from_unit_summary(
            table, unit="unit", group="group", control="control", metrics=[metric]
        )
    (row,) = cast(list[LiftEstimate], list(analysis.run()))
    return row


@pytest.mark.parametrize("panel", [False, True], ids=["from_unit_summary", "from_unit_panel"])
@pytest.mark.parametrize("name", ["y", "y_den", "unit_id", "group_id"])
def test_covariate_named_like_a_unit_frame_column_keeps_its_own_values(name, panel):
    baseline = _run_with_covariate_name("pre", panel=panel)
    renamed = _run_with_covariate_name(name, panel=panel)

    assert renamed.require_lift().value == pytest.approx(baseline.require_lift().value)
    assert renamed.require_lift().lb == pytest.approx(baseline.require_lift().lb)
    assert renamed.abs_diff == pytest.approx(baseline.abs_diff)
    assert renamed.abs_se == pytest.approx(baseline.abs_se)


def test_non_colliding_covariate_columns_are_served_under_their_own_names():
    from increment._source_types import unit_frame_covariate_columns

    assert unit_frame_covariate_columns(("pre", "region")) == {"pre": "pre", "region": "region"}
    mapped = unit_frame_covariate_columns(("y", "pre"))
    assert mapped["pre"] == "pre"
    assert mapped["y"] not in {"y", "pre", "unit_id", "group_id", "y_den"}


def _margin_rows(
    control_mean: float, treatment_mean: float, *, noise: tuple[float, float] = (1.0, 1.0)
) -> pa.Table:
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "pre": 1.0,
            "revenue": mean + (spread if index % 2 == 0 else -spread),
        }
        for group, mean, spread in (
            ("control", control_mean, noise[0]),
            ("treatment", treatment_mean, noise[1]),
        )
        for index in range(20)
    ]
    return pa.Table.from_pylist(rows)


def _margin_analysis(table: pa.Table, **methods: Any) -> Analysis:
    metric = MetricSpec(
        name="revenue",
        preferred_direction="increase",
        **(methods or {"decision_method": LINEAR}),
    )
    return Analysis.from_unit_summary(
        table,
        unit="unit",
        group="group",
        control="control",
        metrics=[metric],
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="revenue", margin_abs=0.5)]),
    )


@pytest.mark.parametrize(
    ("treatment_mean", "reason"),
    [
        # Bounds alone underflow at the nominal level: point and statistics kept.
        (0.02, None),
        # The relative point itself rounds to the excluded -1 floor.
        (1e-20, "relative_unrepresentable"),
    ],
)
def test_unrepresentable_relative_projection_keeps_absolute_margin_inference(
    treatment_mean: float, reason: str | None
):
    control_mean = 0.01 if reason is None else 1.0
    noise = (1.0, 1.0) if reason is None else (0.1, treatment_mean / 10.0)
    table = _margin_rows(control_mean, treatment_mean, noise=noise)
    results = _margin_analysis(table).run(prior=Normal(mu=0.0, sigma=0.1))
    (row,) = cast(list[LiftEstimate], list(results))

    assert row.failure_code is None
    assert row.relative_unavailable_reason == reason
    assert row.relative_error_code == "estimation.tails.unresolvable"
    if reason is None:
        lift = row.require_lift()
        assert lift.lb is None and lift.ub is None
        assert lift.log_mean is not None and lift.log_se is not None
    else:
        assert row.lift is None
    assert row.abs_diff == pytest.approx(treatment_mean - control_mean)
    assert row.abs_se is not None and row.abs_se > 0.0
    assert row.abs_lb is not None and row.abs_ub is not None
    assert row.abs_reference_kind == "t" and row.abs_reference_df is not None
    p_value = row.p_value()
    assert p_value is not None and 0.0 <= p_value <= 1.0
    assert row.posterior_available is False
    assert row.posterior_reason_code == "estimation.results.lift.t_posterior_sufficient_statistics"
    assert results.metadata is not None
    (record,) = results.metadata.cells
    assert record.failure is None
    assert record.sampling is not None and record.sampling.available is True
    assert isinstance(record.sampling.evidence, PValueEvidence)
    assert record.sampling.evidence.p_value == pytest.approx(p_value)
    assert record.posterior is not None and record.posterior.available is False
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    restored_row = restored[0]
    assert isinstance(restored_row, LiftEstimate)
    assert restored_row.p_value() == pytest.approx(p_value)


def test_projection_failure_row_states_posterior_unavailability():
    """A linear row that fails its relative projection with no absolute
    projection to keep still reports the declared prior's unavailability."""
    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(12)],
            "arm": ["c"] * 12 + ["t"] * 12,
            "num": [float(2 + i % 3) for i in range(12)] + [float(-2 - i % 3) for i in range(12)],
            "den": [1.0] * 24,
            "pre": [float(i % 4) for _ in ("c", "t") for i in range(12)],
        }
    )
    metric = MetricSpec(
        name="ratio",
        type="ratio",
        numerator="num",
        denominator="den",
        decision_method=LINEAR,
    )
    results = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    ).run(prior=Normal(mu=0.0, sigma=0.1))
    (row,) = cast(list[LiftEstimate], list(results))
    assert row.lift is None
    assert row.posterior_available is False
    assert row.posterior_reason_code == "estimation.results.lift.t_posterior_sufficient_statistics"
    assert row.posterior_reason_context is not None
    assert row.posterior_reason_context["group_id"] == "t"


def test_zero_se_sensitivity_row_with_absolute_margin_does_not_abort_the_readout():
    rows = [
        {"unit": f"{group}-{index}", "group": group, "pre": float(index % 3), "revenue": mean}
        for group, mean in (("control", 1.0), ("treatment", 2.0))
        for index in range(10)
    ]
    analysis = _margin_analysis(
        pa.Table.from_pylist(rows),
        decision_method=Method(name="plain"),
        sensitivity_methods=[LINEAR],
    )
    with pytest.warns(IncrementWarning):
        results = cast(list[LiftEstimate], list(analysis.run()))

    linear = next(row for row in results if row.method == "linear")
    assert linear.abs_diff == pytest.approx(1.0)
    assert linear.abs_se is None
    with pytest.raises(Exception) as raised:
        linear.p_value()
    assert getattr(raised.value, "code", None) == (
        "estimation.results.lift.p_value_null_abs_missing_abs_se"
    )
    import pandas as pd

    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import Estimate

    valid = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="plain",
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
    frame = cast(pd.DataFrame, LiftEstimates([linear, valid]).to_frame(backend="pandas"))
    linear_projection = frame.loc[frame["method"] == "linear"].iloc[0]
    plain_projection = frame.loc[frame["method"] == "plain"].iloc[0]
    assert pd.isna(linear_projection["p_value"])
    assert linear_projection["p_value_reason_code"] == (
        "estimation.results.lift.p_value_null_abs_missing_abs_se"
    )
    assert pd.notna(plain_projection["p_value"])


def _fcr_rows(*, segment: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = (("control", 5.0, 0.1), ("strong", 10.0, 0.1), ("weak", 0.02, 1.0))
    for seg in ("north", "south") if segment else (None,):
        for group, mean, noise in specs:
            for index in range(20):
                row: dict[str, Any] = {
                    "unit": f"{seg}-{group}-{index}",
                    "group": group,
                    "pre": float(index % 4),
                    "revenue": mean + (noise if index % 2 == 0 else -noise),
                    "primary": 5.0 + (group == "strong") + float(index % 3),
                }
                if segment:
                    row["segment"] = seg
                    row["day"] = 1
                rows.append(row)
    return rows


def test_fcr_reinversion_rebuilds_selected_cuped_arms_only_in_run():
    cov = (CupedCovariate(name="pre", kind="continuous"),)
    method = Method.cuped_multicovariate(name="linear", covariates=cov)
    metrics = [
        MetricSpec(name="primary", decision_method=method),
        MetricSpec(name="revenue", decision_method=method),
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(_fcr_rows(segment=False)),
        unit="unit",
        group="group",
        control="control",
        metrics=metrics,
        plan=AnalysisPlan(q=1e-5, primary="primary", secondaries=["revenue"]),
    )
    rows = {
        row.group_id: row
        for row in cast(list[LiftEstimate], list(analysis.run()))
        if row.metric == "revenue"
    }

    assert rows["strong"].discovery is True
    strong_alpha = rows["strong"].require_lift().alpha
    assert strong_alpha is not None and strong_alpha < 0.05
    assert rows["weak"].discovery is False
    assert rows["weak"].require_lift().alpha == pytest.approx(0.05)
    assert rows["weak"].failure_code is None


def _deep_rows(*, segment: bool) -> list[dict[str, Any]]:
    """A selected arm whose nominal relative bounds are representable while its
    FCR-level bounds are not: log_mean about -33 with log_se about 1."""
    import math

    rows: list[dict[str, Any]] = []
    deep = 5.0 * math.exp(-33.0)
    specs = (("control", 5.0, 0.1), ("deep", deep, 4.0 * deep), ("mild", 5.5, 0.1))
    for seg in ("north", "south") if segment else (None,):
        for group, mean, noise in specs:
            for index in range(20):
                row: dict[str, Any] = {
                    "unit": f"{seg}-{group}-{index}",
                    "group": group,
                    "pre": 1.0,
                    "revenue": mean + (noise if index % 2 == 0 else -noise),
                    "primary": 5.0 + (group == "mild") + float(index % 3),
                }
                if segment:
                    row["segment"] = seg
                    row["day"] = 1
                rows.append(row)
    return rows


def test_selected_cell_with_unrepresentable_fcr_bounds_keeps_its_selection_evidence():
    cov = (CupedCovariate(name="pre", kind="continuous"),)
    method = Method.cuped_multicovariate(name="linear", covariates=cov)
    metrics = [
        MetricSpec(name="primary", decision_method=method),
        MetricSpec(name="revenue", decision_method=method),
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(_deep_rows(segment=False)),
        unit="unit",
        group="group",
        control="control",
        metrics=metrics,
        plan=AnalysisPlan(q=1e-5, primary="primary", secondaries=["revenue"]),
    )
    results = analysis.run()
    deep = next(
        row
        for row in cast(list[LiftEstimate], list(results))
        if row.metric == "revenue" and row.group_id == "deep"
    )

    assert deep.discovery is True
    assert deep.failure_code is None
    assert deep.relative_error_code == "estimation.tails.unresolvable"
    lift = deep.require_lift()
    assert lift.lb is None and lift.ub is None
    assert lift.log_mean is not None and lift.log_se is not None
    p_value = deep.p_value()
    assert p_value is not None and p_value < 1e-5
    assert results.metadata is not None
    record = next(
        c
        for c in results.metadata.cells
        if c.cell.metric == "revenue" and c.cell.group_id == "deep"
    )
    assert record.sampling is not None and record.sampling.available is True
    assert isinstance(record.sampling.evidence, PValueEvidence)
    assert record.sampling.evidence.p_value == pytest.approx(p_value)
    # The withheld-interval row and the collection carrying it survive every
    # retained-evidence route.
    for restored in (
        pickle.loads(pickle.dumps(results)),
        copy.deepcopy(results),
        ReadoutResults.model_validate_json(results.model_dump_json()),
    ):
        assert list(restored) == list(results)
        assert restored.metadata == results.metadata
        restored_deep = next(
            row
            for row in cast(list[LiftEstimate], list(restored))
            if row.metric == "revenue" and row.group_id == "deep"
        )
        assert restored_deep.relative_error_code == "estimation.tails.unresolvable"
        assert restored_deep.p_value() == pytest.approx(p_value)


def test_selected_breakout_cell_with_unrepresentable_fcr_bounds_keeps_its_selection_evidence():
    metric = MetricSpec(name="revenue", decision_method=LINEAR)
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(_deep_rows(segment=True)),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="bh", q=1e-5)),
        breakouts=["segment"],
    )
    result = analysis.run_breakout()
    deep = next(row for row in result if row.dimension_value == "north" and row.group_id == "deep")

    assert deep.discovery is True
    assert deep.excluded is None and deep.failure_code is None
    assert deep.require_lift().lb is None
    assert deep.require_lift().log_se is not None
    assert result.metadata is not None
    record = next(
        c
        for c in result.metadata.cells
        if c.cell.dimension_value == "north" and c.cell.group_id == "deep"
    )
    assert record.sampling is not None and record.sampling.available is True
    for restored in (
        pickle.loads(pickle.dumps(result)),
        copy.deepcopy(result),
        ReadoutResults.model_validate_json(result.model_dump_json()),
    ):
        assert list(restored) == list(result)
        assert restored.metadata == result.metadata


def test_fcr_reinversion_rebuilds_selected_cuped_cells_only_in_breakout():
    metric = MetricSpec(name="revenue", decision_method=LINEAR)
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(_fcr_rows(segment=True)),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="bh", q=1e-5)),
        breakouts=["segment"],
    )
    rows = {(row.dimension_value, row.group_id): row for row in analysis.run_breakout()}

    assert all(rows[(seg, "strong")].discovery is True for seg in ("north", "south"))
    assert all(rows[(seg, "weak")].discovery is False for seg in ("north", "south"))
    assert all(rows[(seg, "weak")].lift is not None for seg in ("north", "south"))


def test_scalar_cuped_constant_covariate_is_localized_to_its_breakout_cell():
    rows = [
        {
            "unit": f"{seg}-{group}-{index}",
            "group": group,
            "day": 1,
            "segment": seg,
            "pre": float(index % 3),
            "scalar_pre": 1.0 if seg == "north" else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for seg in ("north", "south")
        for group in ("control", "treatment")
        for index in range(18)
    ]
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="continuous")
        ),
        sensitivity_methods=[
            Method.cuped(
                name="scalar",
                covariates=CupedCovariate(name="scalar_pre", kind="continuous"),
            )
        ],
    )
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    )
    result = {(row.dimension_value, row.method): row for row in analysis.run_breakout()}

    assert result[("north", "scalar")].lift is None
    assert result[("north", "scalar")].failure_code == "estimation.cuped.covariate_zero_variance"
    assert result[("north", "linear")].lift is not None
    assert result[("south", "scalar")].lift is not None
    assert result[("south", "linear")].lift is not None


def test_failed_breakout_cell_reports_requested_posterior_unavailability():
    from tests.test_cuped_breakout_review import _breakout_analysis

    with pytest.warns(IncrementWarning):
        result = _breakout_analysis(unsupported=True).run_breakout(prior=Normal(mu=0.0, sigma=0.1))
    failed = next(row for row in result if row.dimension_value == "unsupported")
    assert failed.failure_code == "estimation.cuped_linear.level_support"
    assert failed.posterior_available is False
    assert (
        failed.posterior_reason_code == "estimation.results.lift.t_posterior_sufficient_statistics"
    )
    assert result.metadata is not None
    record = next(c for c in result.metadata.cells if c.cell.dimension_value == "unsupported")
    assert record.posterior is not None and record.posterior.available is False
    assert record.posterior.reason_code == failed.posterior_reason_code
    restored = ReadoutResults.model_validate_json(result.model_dump_json())
    assert restored.metadata == result.metadata
    restored_failed = next(row for row in restored if row.dimension_value == "unsupported")
    assert restored_failed.posterior_available is False
    assert restored_failed.posterior_reason_code == failed.posterior_reason_code


def _declaration_analysis(missing: Literal["refuse", "impute-indicator"]) -> Analysis:
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "history": None if index % 4 == 0 else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for group in ("control", "treatment")
        for index in range(20)
    ]
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear",
            covariates=CupedCovariate(name="history", kind="continuous", missing=missing),
        ),
    )
    return Analysis.from_unit_summary(
        pa.Table.from_pylist(rows), unit="unit", group="group", control="control", metrics=[metric]
    )


def test_covariate_declaration_changes_the_readout_identity():
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "history": None if index % 4 == 0 else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for group in ("control", "treatment")
        for index in range(20)
    ]

    def run_with_policy(missing: Literal["refuse", "impute-indicator"]):
        analysis = Analysis.from_unit_summary(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="group",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped_multicovariate(
                        name="linear",
                        covariates=CupedCovariate(
                            name="history", kind="continuous", missing=missing
                        ),
                    ),
                )
            ],
        )
        return analysis.run()

    imputed = run_with_policy("impute-indicator")
    refused = run_with_policy("refuse")

    imputed_row, refused_row = imputed[0], refused[0]
    assert isinstance(imputed_row, LiftEstimate) and isinstance(refused_row, LiftEstimate)
    assert imputed_row.lift is not None
    assert refused_row.lift is None
    assert refused_row.failure_code == "estimation.cuped_linear.missing_refused"
    assert imputed.metadata is not None and refused.metadata is not None
    assert imputed.metadata.scope.snapshot_id != refused.metadata.scope.snapshot_id


def test_scalar_companion_evidence_changes_snapshot_without_changing_decision_estimate():
    def run(scale: float):
        rows = [
            {
                "unit": f"{group}-{index}",
                "group": group,
                "history_a": float(index % 5),
                "history_b": scale * float((index * 3) % 7),
                "revenue": float(index + (group == "treatment") * 3),
            }
            for group in ("control", "treatment")
            for index in range(30)
        ]
        return Analysis.from_unit_summary(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="group",
            control="control",
            plan=AnalysisPlan(primary="revenue"),
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped(
                        name="scalar_a",
                        covariates=CupedCovariate(name="history_a", kind="continuous"),
                    ),
                    sensitivity_methods=(
                        Method.cuped(
                            name="scalar_b",
                            covariates=CupedCovariate(name="history_b", kind="continuous"),
                        ),
                    ),
                )
            ],
        ).run()

    first, changed = run(1.0), run(2.0)
    first_a = next(
        row for row in first if isinstance(row, LiftEstimate) and row.method == "scalar_a"
    )
    changed_a = next(
        row for row in changed if isinstance(row, LiftEstimate) and row.method == "scalar_a"
    )
    assert first.metadata is not None and changed.metadata is not None
    assert first.metadata.scope.snapshot_id != changed.metadata.scope.snapshot_id
    assert first_a.require_lift().value == pytest.approx(changed_a.require_lift().value, rel=1e-10)
    assert first_a.require_lift().log_se == pytest.approx(
        changed_a.require_lift().log_se, rel=1e-10
    )
    assert first_a.family_id is not None and first_a.family_id == changed_a.family_id


def test_call_time_scalar_override_accepts_provenance_free_legacy_x():
    from tests.test_analysis_moments import _legacy_fixed10_rows

    rows = _legacy_fixed10_rows()
    for row in rows:
        row.update(
            ref_x=10.0,
            cx1=1.0 if row["group_id"] == "control" else 2.0,
            cx2=20.0,
            cxy=4.0 if row["group_id"] == "control" else 5.0,
            x_role="covariate",
        )
    analysis = Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")

    result = analysis.run(
        decision_method=Method.cuped(covariates=CupedCovariate(name="prior", kind="continuous"))
    )[0]

    assert isinstance(result, LiftEstimate)


def test_moments_provenance_must_cover_each_requested_scalar_family():
    from tests.test_analysis_moments import _legacy_fixed10_rows

    scalar_a = Method.cuped(
        name="scalar_a", covariates=CupedCovariate(name="history_a", kind="continuous")
    )
    scalar_b = Method.cuped(
        name="scalar_b", covariates=CupedCovariate(name="history_b", kind="continuous")
    )
    rows = _legacy_fixed10_rows(
        covariate_name="history_a",
        ref_x=10.0,
        cx1=1.0,
        cx2=20.0,
        cxy=4.0,
        x_role="covariate",
    )

    with pytest.raises(UnsupportedRequestError) as raised:
        Analysis.from_moments(
            rows,
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=scalar_a,
                    sensitivity_methods=(scalar_b,),
                )
            ],
            control="control",
        )

    assert raised.value.code == "facade.analysis_config.cuped_scalar_moments"


def test_scalar_sensitivity_refuse_fails_only_its_estimate():
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "history": None if index % 4 == 0 else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for group in ("control", "treatment")
        for index in range(20)
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(
                    Method.cuped(
                        name="scalar_sensitivity",
                        covariates=CupedCovariate(
                            name="history", kind="continuous", missing="refuse"
                        ),
                    ),
                ),
            )
        ],
    )
    results = {row.method: row for row in analysis.run() if isinstance(row, LiftEstimate)}
    assert results["unadjusted"].lift is not None
    assert results["scalar_sensitivity"].lift is None
    assert results["scalar_sensitivity"].failure_code == "frame.validation.metric_covariate_column"


def test_multicov_and_scalar_missing_refusals_remain_method_specific():
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "history": None if index % 4 == 0 else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for group in ("control", "treatment")
        for index in range(20)
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method.cuped_multicovariate(
                    name="mixed",
                    covariates=CupedCovariate(name="history", kind="continuous", missing="refuse"),
                ),
                sensitivity_methods=(
                    Method.cuped(
                        name="scalar",
                        covariates=CupedCovariate(
                            name="history", kind="continuous", missing="refuse"
                        ),
                    ),
                ),
            )
        ],
    )
    results = {row.method: row for row in analysis.run() if isinstance(row, LiftEstimate)}
    assert results["mixed"].lift is None
    assert results["mixed"].failure_code == "estimation.cuped_linear.missing_refused"
    assert results["scalar"].lift is None
    assert results["scalar"].failure_code == "frame.validation.metric_covariate_column"


def test_scalar_missing_refusal_is_rechecked_for_call_time_override():
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "history": None if index % 4 == 0 else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for group in ("control", "treatment")
        for index in range(20)
    ]
    with pytest.warns(IncrementWarning):
        analysis = Analysis.from_unit_summary(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="group",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped(
                        name="scalar",
                        covariates=CupedCovariate(
                            name="history", kind="continuous", missing="impute"
                        ),
                    ),
                )
            ],
        )
    override = Method.cuped(
        name="scalar_refuse",
        covariates=CupedCovariate(name="history", kind="continuous", missing="refuse"),
    )

    (row,) = analysis.run(decision_method=override)
    assert isinstance(row, LiftEstimate)
    assert row.lift is None
    assert row.method == override.name and row.method_role == "decision"
    assert row.failure_code == "frame.validation.metric_covariate_column"


def test_multicovariate_refuse_fails_only_its_estimate():
    imputed = _declaration_analysis("impute-indicator").run()
    imputed_row = imputed[0]
    assert isinstance(imputed_row, LiftEstimate)
    assert imputed_row.lift is not None

    refused = _declaration_analysis("refuse").run()
    refused_row = refused[0]
    assert isinstance(refused_row, LiftEstimate)
    assert refused_row.lift is None
    assert refused_row.failure_code == "estimation.cuped_linear.missing_refused"


def test_breakout_unit_covariate_values_enter_the_source_identity():
    from tests.test_cuped_breakout_review import _breakout_analysis

    def with_scaled_pre(scale: float) -> BreakoutEstimates:
        rows = [
            {
                "unit": f"{seg}-{group}-{index}",
                "group": group,
                "day": 1,
                "segment": seg,
                "pre": scale * float(index % 4),
                "revenue": float(index + (group == "treatment") * 3),
            }
            for seg in ("north", "south")
            for group in ("control", "treatment")
            for index in range(18)
        ]
        metric = MetricSpec(name="revenue", decision_method=LINEAR)
        return Analysis.from_unit_panel(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="group",
            date="day",
            control="control",
            metrics=[metric],
            breakouts=["segment"],
        ).run_breakout()

    first, second = with_scaled_pre(1.0), with_scaled_pre(1.0)
    assert first.metadata is not None and second.metadata is not None
    assert first.metadata.scope.snapshot_id == second.metadata.scope.snapshot_id
    assert _breakout_analysis().run_breakout().metadata is not None
    changed = with_scaled_pre(2.0)
    assert changed.metadata is not None
    assert changed.metadata.scope.snapshot_id != first.metadata.scope.snapshot_id


def test_ordinary_lane_breakout_failure_states_posterior_unavailability():
    rows = [
        {
            "unit": f"{seg}-{group}-{index}",
            "group": group,
            "day": 1,
            "segment": seg,
            "pre": float(index % 3),
            "scalar_pre": 1.0 if seg == "north" else float(index % 5),
            "revenue": float(index + (group == "treatment") * 3),
        }
        for seg in ("north", "south")
        for group in ("control", "treatment")
        for index in range(18)
    ]
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="continuous")
        ),
        sensitivity_methods=[
            Method.cuped(
                name="scalar",
                covariates=CupedCovariate(name="scalar_pre", kind="continuous"),
            )
        ],
    )
    result = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    ).run_breakout(prior=Normal(mu=0.0, sigma=0.1))
    failed = next(
        row for row in result if row.dimension_value == "north" and row.method == "scalar"
    )

    assert failed.failure_code == "estimation.cuped.covariate_zero_variance"
    assert failed.posterior_available is False
    assert failed.posterior_reason_code == failed.failure_code
    assert result.metadata is not None
    record = next(
        c
        for c in result.metadata.cells
        if c.cell.dimension_value == "north" and c.cell.method == "scalar"
    )
    assert record.posterior is not None and record.posterior.available is False
    assert record.posterior.reason_code == failed.failure_code


def _no_covariate_rows() -> list[dict[str, Any]]:
    return [
        {
            "unit": f"{g}-{i}",
            "group": g,
            "day": 1,
            "segment": ("north", "south")[i % 2],
            "revenue": float(i + (g == "treatment") * 3),
        }
        for g in ("control", "treatment")
        for i in range(16)
    ]


def test_no_covariate_metric_keeps_its_source_snapshot_identity():
    """Pinned at 1112c7d7: identities of metrics without covariate declarations are unchanged."""
    rows = _no_covariate_rows()
    run = Analysis.from_unit_summary(
        pa.Table.from_pylist(
            [{k: v for k, v in r.items() if k not in ("day", "segment")} for r in rows]
        ),
        unit="unit",
        group="group",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    ).run()
    breakout = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        breakouts=["segment"],
    ).run_breakout()

    assert run.metadata is not None and breakout.metadata is not None
    assert run.metadata.scope.snapshot_id == (
        "sha256:60b464aedd6d68490cecf4cc7820ab263123805ef319e601dc072f8f5fa21300"
    )
    assert breakout.metadata.scope.snapshot_id == (
        "sha256:f6e9391ebb457b8ac9efeaa8a0e831b77ec4bd3ec5b7febf5dd638b44e364b5d"
    )


def test_unscoped_export_preserves_unknown_legacy_sampling_availability():
    from tests.test_tables import _breakout_estimate

    estimate = _breakout_estimate(
        "revenue", "T", "unadjusted", "country", "US", 0.2, lb=0.1, ub=0.3
    ).model_copy(update={"sampling_available": None, "prior_shrunk": True})
    frame = cast(Any, BreakoutEstimates([estimate]).to_frame(backend="pandas"))
    assert frame["sampling_available"].isna().all()
    assert frame["sampling_available"].dtype == "boolean"

    known = estimate.model_copy(update={"sampling_available": False})
    exported = cast(Any, BreakoutEstimates([known]).to_frame(backend="pandas"))
    assert exported["sampling_available"].tolist() == [False]


def _observational_frame(covariate: str) -> pa.Table:
    import numpy as np

    rng = np.random.default_rng(7)
    n = 160
    x = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-x))).astype(float)
    y = 1.0 + 0.5 * x + 2.0 * d + rng.normal(scale=0.3, size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C").tolist(),
            "revenue": y,
            covariate: x,
        }
    )


@pytest.mark.parametrize("name", ["y", "y_den", "unit_id", "group_id"])
def test_observational_frame_covariate_named_like_a_unit_frame_column_is_unchanged(name):
    from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational

    def estimate(covariate: str):
        design = Observational(
            control_group="C",
            adjustment=AdjustmentSet(covariates=(covariate,)),
            gate=IdentificationGate(overlap="trim"),
        )
        results = Analysis.from_unit_summary(
            _observational_frame(covariate),
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=design,
        ).run(decision_method=Method(name="iptw"))
        (row,) = results
        return row

    baseline, renamed = estimate("pre"), estimate(name)
    assert renamed.require_lift().value == pytest.approx(baseline.require_lift().value)
    assert renamed.require_lift().lb == pytest.approx(baseline.require_lift().lb)


@pytest.mark.parametrize("name", ["y", "y_den", "unit_id", "group_id"])
def test_cate_frame_covariate_named_like_a_unit_frame_column_is_unchanged(name):
    from increment.cate import estimate_cate
    from increment.frame import from_unit_summary

    def estimate(covariate: str):
        source = from_unit_summary(
            _observational_frame(covariate).rename_columns(
                ["user_id", "variant", "revenue", covariate]
            ),
            unit="user_id",
            group="variant",
            control="C",
            metrics=[MetricSpec(name="revenue")],
        )
        return estimate_cate(source, "revenue", control="C", interact=[covariate])

    baseline, renamed = estimate("pre"), estimate(name)
    assert renamed.ate == pytest.approx(baseline.ate)
    assert renamed.se == pytest.approx(baseline.se)
    (base_effect,), (renamed_effect,) = baseline.interactions, renamed.interactions
    assert renamed_effect.lb == pytest.approx(base_effect.lb)
    assert renamed_effect.ub == pytest.approx(base_effect.ub)


def test_t_reintervalization_keeps_additive_projection_when_relative_critical_value_overflows():
    from tests.test_cuped_reintervalization import _t_row

    row = _t_row().model_copy(update={"reference_df": 1, "dof": 1})

    projected = row.reintervalize(1e-310)

    assert projected.relative_error_code is not None
    assert projected.require_lift().lb is None
    assert projected.require_lift().log_mean == row.require_lift().log_mean
    assert projected.abs_lb is not None and row.abs_lb is not None
    assert projected.abs_lb < row.abs_lb
    assert projected.p_value() == pytest.approx(row.p_value())
