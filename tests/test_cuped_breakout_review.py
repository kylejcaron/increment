from __future__ import annotations

from typing import Any, Literal

import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec, MultiplicitySpec
from increment.frame import CupedCovariate


def _linear_metric(
    *, metric_type: Literal["mean", "conversion", "ratio", "retention", "quantile"] = "mean"
) -> MetricSpec:
    return MetricSpec(
        name="revenue",
        type=metric_type,
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
        covariates=(CupedCovariate(name="pre", kind="categorical"),),
    )


def _breakout_analysis(*, unsupported: bool = False, null_segment: bool = False) -> Analysis:
    rows: list[dict[str, Any]] = []
    segments: tuple[str | None, ...] = (
        ("north", "south", "unsupported")
        if unsupported
        else (
            "north",
            "south",
        )
    )
    if null_segment:
        segments = ("north", None)
    for segment in segments:
        for group in ("control", "treatment"):
            for index in range(18):
                pre = (
                    "rare"
                    if segment == "unsupported" and group == "treatment" and index == 0
                    else ("common" if index < 9 else "other")
                )
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": pre,
                        "revenue": float(index + (group == "treatment") * (3 + index % 3)),
                    }
                )
    return Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[_linear_metric()],
        breakouts=["segment"],
    )


def test_public_breakout_localizes_failed_cuped_projection_and_keeps_siblings() -> None:
    from increment.errors import IncrementWarning

    analysis = _breakout_analysis(unsupported=True)
    with pytest.warns(IncrementWarning, match="estimation.cuped_linear.level_support"):
        rows = analysis.run_breakout()

    by_segment = {row.dimension_value: row for row in rows}
    assert set(by_segment) == {"north", "south", "unsupported"}
    assert by_segment["north"].lift is not None
    assert by_segment["south"].lift is not None
    assert by_segment["unsupported"].lift is None
    assert by_segment["unsupported"].excluded == "estimation_failed"


def test_public_breakout_sensitivity_failure_does_not_enter_decision_family() -> None:
    analysis = _breakout_analysis(unsupported=True)
    rows = analysis.run_breakout(
        decision_method=Method(name="plain"),
        sensitivity_methods=[Method(name="linear", variance_reduction="cuped_linear")],
    )

    unsupported = [row for row in rows if row.dimension_value == "unsupported"]
    assert {row.method for row in unsupported} == {"plain", "linear"}
    assert next(row for row in unsupported if row.method == "plain").lift is not None
    assert (
        next(row for row in unsupported if row.method == "linear").excluded == "estimation_failed"
    )
    assert all(
        row.family_axes == ("metric", "arm", "segment") for row in rows if row.method == "plain"
    )


def test_public_breakout_null_segment_uses_its_unit_rows() -> None:
    analysis = _breakout_analysis(null_segment=True)
    rows = analysis.run_breakout()

    null_rows = [row for row in rows if row.dimension_value == "__null__"]
    assert len(null_rows) == 1
    assert null_rows[0].lift is not None
    assert null_rows[0].n_control == null_rows[0].n_treat == 18


def test_public_breakout_keeps_all_conversion_methods_in_mixed_cuped_request() -> None:
    rows: list[dict[str, Any]] = []
    for segment in ("north", "south"):
        for group in ("control", "treatment"):
            for index in range(24):
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": float(index),
                        "region": "a" if index % 2 else "b",
                        "converted": int(index < (8 if group == "control" else 14)),
                    }
                )
    methods = [
        Method(name="plain"),
        Method(name="scalar", variance_reduction="cuped"),
        Method(name="linear", variance_reduction="cuped_linear"),
    ]
    metric = MetricSpec(
        name="conversion",
        covariate="pre",
        value_column="converted",
        decision_method=methods[0],
        covariates=(
            CupedCovariate(name="pre", kind="continuous"),
            CupedCovariate(name="region", kind="categorical"),
        ),
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

    result = analysis.run_breakout(
        decision_method=methods[0],
        sensitivity_methods=methods[1:],
    )
    assert {(row.dimension_value, row.method) for row in result} == {
        (segment, method.name) for segment in ("north", "south") for method in methods
    }
    assert all(row.lift is not None for row in result)


def test_definitions_and_artifact_breakouts_supply_cuped_unit_evidence(tmp_path) -> None:
    import yaml

    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.models import Definitions
    from tests.covariate_cases import _CATEGORICAL_DESIGN, categorical_defs_and_con

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    text = defs_path.read_text().replace(
        "    plan: {primary: revenue}",
        "    plan: {primary: {metric: revenue, "
        "decision_method: {name: linear, variance_reduction: cuped_linear}, "
        "covariates: [{name: region, kind: categorical, property: region, source: events}]}}\n"
        "    breakouts: [{property: region, source: events}]",
    )
    defs_path.write_text(text.replace(_CATEGORICAL_DESIGN, ""))
    native = Analysis.from_definitions("cat_test", defs_path, con)
    definitions = Definitions.model_validate(yaml.safe_load(defs_path.read_text()))
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    store = WarehouseArtifactStore(con, schema_name="breakout_review_artifacts")
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = native.publish_unit_day_artifact(store, extensions=extensions)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)

    native_rows = native.run_breakout()
    artifact_rows = restored.run_breakout()

    def identity(rows):
        return [
            (
                row.dimension_value,
                row.group_id,
                row.method,
                row.excluded,
                row.family_axes,
                row.family_procedure,
            )
            for row in rows
        ]

    assert identity(native_rows) == identity(artifact_rows)
    assert [row.require_lift().value for row in native_rows] == pytest.approx(
        [row.require_lift().value for row in artifact_rows]
    )
    assert native_rows.metadata is not None and artifact_rows.metadata is not None
    assert native_rows.metadata.scope is not None and artifact_rows.metadata.scope is not None
    assert native_rows.metadata.scope.cells == artifact_rows.metadata.scope.cells


def test_public_breakout_retains_relative_ratio_when_absolute_projection_overflows() -> None:
    rows: list[dict[str, Any]] = []
    for group in ("control", "treatment"):
        for index in range(18):
            rows.append(
                {
                    "unit": f"{group}-{index}",
                    "group": group,
                    "day": 1,
                    "segment": "all",
                    "pre": ("a", "b", "c")[index % 3],
                    "revenue": 1e150 * (1.0 + index / 1000.0 + (group == "treatment") / 100.0),
                    "orders": 1e-150,
                }
            )
    metric = MetricSpec(
        name="revenue_per_order",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
        covariates=(CupedCovariate(name="pre", kind="categorical"),),
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

    (row,) = analysis.run_breakout()

    assert row.lift is not None
    assert row.abs_diff is None and row.abs_se is None
    assert row.failure_code is not None
    assert row.failure_context is not None
    assert row.failure_context["projection"] == "absolute"


def test_public_breakout_localizes_nonpositive_adjusted_arm_with_positive_raw_mean() -> None:
    rows: list[dict[str, Any]] = []
    offsets = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    for segment in ("north", "south"):
        for group in ("control", "treatment"):
            x_center, y_center = (
                (0.0, 10.0)
                if group == "control"
                else ((0.0, 12.0) if segment == "north" else (100.0, 5.0))
            )
            for index in range(21):
                offset = offsets[index % len(offsets)]
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": x_center + offset,
                        "revenue": y_center + offset,
                    }
                )
    metric = MetricSpec(
        name="revenue",
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
        covariates=(CupedCovariate(name="pre", kind="continuous"),),
    )
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
    )

    with pytest.warns():
        result = analysis.run_breakout()

    by_cell = {(row.dimension_value, row.group_id): row for row in result}
    assert by_cell[("north", "treatment")].lift is not None
    assert by_cell[("south", "treatment")].lift is None
    assert by_cell[("south", "treatment")].excluded == "estimation_failed"
