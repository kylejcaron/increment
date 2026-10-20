from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from increment import Analysis, CupedCovariate, Method
from increment.errors import IncrementWarning
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.semantics.design import Randomized


def _mixed_methods(*, categorical_decision: bool):
    categorical = Method.cuped_multicovariate(
        name="category",
        covariates=(CupedCovariate(name="region", kind="categorical", missing="impute-indicator"),),
    )
    continuous = Method.cuped_multicovariate(
        name="continuous",
        covariates=(CupedCovariate(name="pre", kind="continuous", missing="impute-indicator"),),
    )
    return (categorical, continuous) if categorical_decision else (continuous, categorical)


def _rows(analysis):
    return {
        (row.metric, row.method): row for row in analysis.run() if isinstance(row, LiftEstimate)
    }


@pytest.mark.parametrize("categorical_decision", [True, False])
def test_mixed_categorical_and_fitted_methods_replay_per_method(tmp_path, categorical_decision):
    data = []
    for arm, shift in (("control", 0.0), ("treatment", 0.7)):
        for i in range(48):
            pre = float((i % 12) + shift)
            data.append(
                {
                    "unit": f"{arm}-{i}",
                    "arm": arm,
                    "outcome": 2.0 + 0.4 * pre + (i % 5) * 0.15 + (arm == "treatment"),
                    "pre": pre,
                    "region": ("north", "south", "west")[i % 3],
                }
            )
    decision, sensitivity = _mixed_methods(categorical_decision=categorical_decision)
    metric = MetricSpec(
        name="outcome",
        decision_method=decision,
        sensitivity_methods=(sensitivity,),
    )
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    source = Analysis.from_unit_summary(
        pa.Table.from_pylist(data), unit="unit", group="arm", metrics=[metric], design=design
    )
    try:
        expected = _rows(source)
        destination = tmp_path / "mixed-cuped.parquet"
        source.export(destination)
        restored = Analysis.from_moments(
            pq.read_table(destination).to_pylist(), metrics=[metric], design=design
        )
        try:
            actual = _rows(restored)
        finally:
            restored.close()
    finally:
        source.close()
    assert actual.keys() == expected.keys()
    for key, row in expected.items():
        for field in ("value", "log_se", "lb", "ub"):
            assert getattr(actual[key].require_lift(), field) == pytest.approx(
                getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
            )


@pytest.mark.parametrize("ingress", ["native", "artifact"])
@pytest.mark.integration
@pytest.mark.slow
def test_categorical_ratio_export_retains_withheld_arm_components(tmp_path, ingress):
    import json
    import warnings

    import ibis
    import yaml

    from increment.semantics.loader import load
    from increment.semantics.models import AnalysisPlan, CupedCovariateBinding
    from tests.test_cuped_portable_journey import _definitions, _events, _publish, _units

    ratio_method = Method.cuped_multicovariate(
        name="ratio_category",
        covariates=(
            CupedCovariateBinding(
                name="region", kind="categorical", property="region", source="events"
            ),
        ),
    )

    definitions = _definitions(tmp_path, fixed_caps=False)
    config = yaml.safe_load(definitions.read_text())
    config["experiments"][0]["allocation"] = {
        "control": 0.34,
        "treatment": 0.33,
        "invalid": 0.33,
    }
    config["experiments"][0]["plan"]["guardrails"][0]["decision_method"] = ratio_method.model_dump(
        mode="json", exclude_none=True
    )
    units = _units()
    for index, unit in enumerate(units[:40]):
        units.append(
            {
                **unit,
                "user_id": f"invalid-{index}",
                "variant": "invalid",
                "spend": 0.0,
            }
        )
    definitions.write_text(yaml.safe_dump(config))
    experiment = load(definitions).experiment("mixed_cuped")
    assert experiment is not None
    design = experiment.resolved_design()
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(_events(units)))
    source = Analysis.from_definitions("mixed_cuped", definitions, con, store="none")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.warns(IncrementWarning):
                expected = source.run(metrics=["revenue_per_spend"])
        if ingress == "artifact":
            source, _requests = _publish(con, source)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with pytest.warns(IncrementWarning):
                    expected = source.run(metrics=["revenue_per_spend"])
        destination = tmp_path / "withheld-ratio.parquet"
        source.export(destination)
        rows = pq.read_table(destination).to_pylist()
        fitted = [
            json.loads(row["cuped_evidence"])
            for row in rows
            if row.get("record_kind") == "cuped_evidence"
            and json.loads(row["cuped_evidence"]).get("metric") == "revenue_per_spend"
        ]
        assert len(fitted) == 1
        fit_state = fitted[0]["evidence"]["fit_state"]
        assert "invalid" in fit_state["withheld_arms"]
        assert any(
            arm == "invalid"
            and dict(context)["code"]
            == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
            for arm, context in fit_state["refusal_contexts"]
        )
        replay = Analysis.from_moments(
            rows,
            metrics=[
                MetricSpec(
                    name="revenue_per_spend",
                    type="ratio",
                    numerator="revenue",
                    denominator="spend",
                    decision_method=ratio_method,
                )
            ],
            design=design,
            plan=AnalysisPlan(primary="revenue_per_spend"),
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with pytest.warns(IncrementWarning):
                    actual = replay.run()
        finally:
            replay.close()
    finally:
        source.close()
        con.disconnect()
    assert [(row.metric, row.method) for row in actual] == [
        (row.metric, row.method) for row in expected
    ]


@pytest.mark.parametrize("ingress", ["from_unit_summary", "from_unit_panel"])
def test_frame_ratio_export_preserves_withheld_arm_reason(tmp_path, ingress):
    import datetime as dt
    import json

    from increment.errors import IncrementWarning

    rows = []
    for arm, denominator in (("control", 2.0), ("treatment", 2.5), ("invalid", 0.0)):
        for i in range(48):
            x = float(i % 11) + (arm == "treatment")
            rows.append(
                {
                    "unit": f"{arm}-{i}",
                    "arm": arm,
                    "num": 4.0 + 0.6 * x + i % 4,
                    "den": denominator,
                    "x": x,
                    "day": dt.date(2025, 1, 10),
                }
            )
    method = Method.cuped_multicovariate(
        name="linear",
        covariates=(CupedCovariate(name="x", kind="continuous", missing="impute-indicator"),),
    )
    metric = MetricSpec(
        name="ratio",
        type="ratio",
        numerator="num",
        denominator="den",
        decision_method=method,
    )
    design = Randomized(
        control_group="control",
        allocation={"control": 1 / 3, "treatment": 1 / 3, "invalid": 1 / 3},
    )
    if ingress == "from_unit_summary":
        source = Analysis.from_unit_summary(
            pa.Table.from_pylist(rows), unit="unit", group="arm", metrics=[metric], design=design
        )
    else:
        source = Analysis.from_unit_panel(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="arm",
            date="day",
            metrics=[metric],
            design=design,
        )
    try:
        with pytest.warns(IncrementWarning):
            expected = _rows(source)
        destination = tmp_path / f"{ingress}.parquet"
        source.export(destination)
        exported = pq.read_table(destination).to_pylist()
        fitted = [
            json.loads(row["cuped_evidence"])
            for row in exported
            if row.get("record_kind") == "cuped_evidence"
            and json.loads(row["cuped_evidence"]).get("metric") == "ratio"
        ]
        fit_state = fitted[0]["evidence"]["fit_state"]
        assert "invalid" in fit_state["withheld_arms"]
        assert any(
            arm == "invalid"
            and dict(context)["code"]
            == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
            for arm, context in fit_state["refusal_contexts"]
        )
        replay = Analysis.from_moments(exported, metrics=[metric], design=design)
        try:
            with pytest.warns(IncrementWarning):
                actual = _rows(replay)
        finally:
            replay.close()
    finally:
        source.close()
    assert actual.keys() == expected.keys()
    for key, row in expected.items():
        for field in ("value", "log_se", "lb", "ub"):
            assert getattr(actual[key].require_lift(), field) == pytest.approx(
                getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
            )


@pytest.mark.integration
@pytest.mark.slow
def test_breakout_snapshot_identity_tracks_aggregate_fit_evidence(tmp_path):
    import ibis
    import yaml

    from increment.semantics.models import CupedCovariateBinding
    from tests.test_cuped_portable_journey import _definitions, _events, _units

    definitions = _definitions(tmp_path, fixed_caps=False)
    config = yaml.safe_load(definitions.read_text())
    config["experiments"][0]["plan"]["primary"]["decision_method"] = Method.cuped_multicovariate(
        name="continuous",
        covariates=(
            CupedCovariateBinding(
                name="prior", kind="continuous", property="pre_revenue", source="events"
            ),
        ),
    ).model_dump(mode="json", exclude_none=True)
    definitions.write_text(yaml.safe_dump(config))
    units = _units()
    con = ibis.duckdb.connect()

    def capture(rows):
        if "events" in con.list_tables():
            con.drop_table("events")
        con.create_table("events", obj=pa.Table.from_pylist(_events(rows)))
        analysis = Analysis.from_definitions("mixed_cuped", definitions, con, store="none")
        try:
            estimates = analysis.run_breakout(metrics=["revenue"])
            return (
                {row.source_snapshot_id for row in estimates},
                {
                    (row.dimension_value, row.group_id): (
                        row.require_lift().value,
                        row.require_lift().log_se,
                    )
                    for row in estimates
                },
            )
        finally:
            analysis.close()

    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(units):
        groups.setdefault((row["variant"], row["region"]), []).append(index)
    permuted = [dict(row) for row in units]
    for indexes in groups.values():
        values = [permuted[index]["pre_revenue"] for index in indexes]
        for index, value in zip(indexes, reversed(values), strict=True):
            permuted[index]["pre_revenue"] = value
    try:
        original_ids, original_estimates = capture(units)
        repeated_ids, _repeated_estimates = capture(units)
        permuted_ids, permuted_estimates = capture(permuted)
    finally:
        con.disconnect()
    assert len(original_ids) == len(repeated_ids) == len(permuted_ids) == 1
    assert original_ids == repeated_ids
    assert original_estimates.keys() == permuted_estimates.keys()
    assert any(
        original_estimates[key] != pytest.approx(permuted_estimates[key])
        for key in original_estimates
    )
    assert original_ids.isdisjoint(permuted_ids)
