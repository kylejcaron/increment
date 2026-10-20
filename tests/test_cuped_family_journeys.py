from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec
from increment.estimation.readout_types import ReadoutResults
from increment.frame import CupedCovariate
from increment.semantics.models import ExperimentMetric


def _family_analysis(ingress: str, q: float):
    rng = np.random.default_rng(311)
    n_per_arm = 120
    arms = ["control"] * n_per_arm + ["treatment"] * n_per_arm
    treatment = np.repeat([0.0, 1.0], n_per_arm)
    history = rng.normal(size=2 * n_per_arm)
    history[::9] = np.nan
    region = rng.choice(["north", "south", "west"], size=2 * n_per_arm)
    noise = rng.normal(0.0, 0.25, size=(4, 2 * n_per_arm))
    imputed = np.nan_to_num(history)
    data = {
        "unit": [f"u{i}" for i in range(2 * n_per_arm)],
        "arm": arms,
        "history": history,
        "region": region,
        "primary_mean": 10.0 + treatment + 0.8 * imputed + noise[0],
        "secondary_mean": 8.0 + 1.4 * treatment + 0.7 * imputed + noise[1],
        "guard_num": 4.0 + 0.08 * treatment + 0.1 * imputed + noise[2],
        "guard_den": 2.0 + 0.05 * imputed + 0.03 * noise[3],
        "secondary_num": 3.0 + 0.1 * imputed + noise[3],
        "secondary_den": 1.5 + 0.03 * imputed + 0.02 * noise[2],
    }
    if ingress == "from_unit_panel":
        panel = {key: [] for key in (*data, "day")}
        for index in range(2 * n_per_arm):
            for day in ("2025-01-01", "2025-01-02"):
                for key, values in data.items():
                    panel[key].append(
                        float(values[index]) / 2
                        if key
                        in {
                            "primary_mean",
                            "secondary_mean",
                            "guard_num",
                            "guard_den",
                            "secondary_num",
                            "secondary_den",
                        }
                        else values[index]
                    )
                panel["day"].append(day)
        frame = pa.table(panel)
    else:
        frame = pa.table(data)

    covariates = (
        CupedCovariate(name="history", kind="continuous", missing="impute-indicator"),
        CupedCovariate(name="region", kind="categorical"),
    )
    method = Method.cuped_multicovariate(name="linear", covariates=covariates)
    metrics = [
        MetricSpec(name="primary_mean", decision_method=method),
        MetricSpec(name="secondary_mean", decision_method=method),
        MetricSpec(
            name="guardrail_ratio",
            type="ratio",
            numerator="guard_num",
            denominator="guard_den",
            preferred_direction="decrease",
            decision_method=method,
        ),
        MetricSpec(
            name="secondary_ratio",
            type="ratio",
            numerator="secondary_num",
            denominator="secondary_den",
            decision_method=method,
        ),
    ]
    plan = AnalysisPlan(
        q=q,
        primary="primary_mean",
        guardrails=[ExperimentMetric(metric="guardrail_ratio", margin_abs=0.02)],
        secondaries=["secondary_mean", "secondary_ratio"],
    )
    if ingress == "from_unit_panel":
        return Analysis.from_unit_panel(
            frame,
            unit="unit",
            group="arm",
            date="day",
            control="control",
            metrics=metrics,
            plan=plan,
        )
    return Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="control", metrics=metrics, plan=plan
    )


@pytest.mark.parametrize("ingress", ["from_unit_summary", "from_unit_panel"])
def test_mixed_cuped_family_and_json_replay(ingress):
    result = _family_analysis(ingress, 0.2).run()
    rows = {row.metric: row for row in result}
    assert set(rows) == {"primary_mean", "guardrail_ratio", "secondary_mean", "secondary_ratio"}
    assert rows["primary_mean"].role == "primary"
    assert rows["guardrail_ratio"].role == "guardrail"
    assert rows["guardrail_ratio"].null_abs == pytest.approx(0.02)
    secondaries = [rows["secondary_mean"], rows["secondary_ratio"]]
    assert all(row.family_procedure == "bh" and row.family_size == 2 for row in secondaries)
    assert all(row.family_axes == ("metric", "arm") for row in secondaries)
    assert result.metadata is not None
    assert any(
        advisory.code == "estimation.cuped_linear.fit"
        for record in result.metadata.cells
        for advisory in record.advisories
    )
    replay = ReadoutResults.model_validate_json(result.model_dump_json())
    assert list(replay) == list(result)
    assert replay.metadata == result.metadata


@pytest.mark.parametrize("ingress", ["from_unit_summary", "from_unit_panel"])
@pytest.mark.integration
def test_secondary_cuped_fcr_reintervalizes_without_refitting(ingress):
    q05 = {row.metric: row for row in _family_analysis(ingress, 0.05).run()}["secondary_mean"]
    q20 = {row.metric: row for row in _family_analysis(ingress, 0.2).run()}["secondary_mean"]
    assert q05.discovery is q20.discovery is True
    assert q05.require_lift().value == pytest.approx(q20.require_lift().value)
    assert q05.p_value() == pytest.approx(q20.p_value())
    assert q05.require_lift().lb != pytest.approx(q20.require_lift().lb)
    assert q05.require_lift().ub != pytest.approx(q20.require_lift().ub)
    assert q05.abs_lb != pytest.approx(q20.abs_lb)
    assert q05.abs_ub != pytest.approx(q20.abs_ub)
    assert q05.note == q20.note


def test_complete_cuped_yaml_guide_examples_execute(tmp_path):
    """Execute both experiment YAML examples in the guide against real DuckDB facts."""
    import datetime as dt
    import re
    import warnings

    import ibis
    import yaml

    from tests.parity_harness import matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    guide = (Path(__file__).parents[1] / "docs/guides/cuped.md").read_text()
    examples = [
        yaml.safe_load(block)
        for block in re.findall(r"```yaml\n(.*?)```", guide, flags=re.DOTALL)
        if "experiments:" in block
    ]
    assert len(examples) == 2
    units = matrix_data.linear_cuped_rows()
    for row in units:
        if row["variant"] == "control":
            row["variant"] = "C"
    events = mixed_cuped_event_rows(units)
    for event in events:
        event["ts"] += dt.timedelta(days=5)
        if event["group_id"] == "control":
            event["group_id"] = "C"
        if event["event"] == "enrolled":
            event["experiment_id"] = "new_onboarding_v2"

    connection = ibis.duckdb.connect()
    connection.create_table("cuped_yaml_events", obj=events)
    try:
        for index, example in enumerate(examples):
            document = yaml.safe_load(
                mixed_cuped_definitions_yaml("cuped_yaml_events", n_pre_periods=14)
            )
            document["exposures"][0]["name"] = "first_page_view"
            document["metrics"] = [
                {
                    **metric,
                    "name": {
                        "revenue": "avg_session_duration",
                        "signups": "purchase_rate",
                    }.get(metric["name"], metric["name"]),
                }
                for metric in document["metrics"]
                if metric["name"] != "rpo"
            ]
            experiment = example["experiments"][0]
            experiment["allocation"] = {"C": 0.5, "treatment": 0.5}
            document["experiments"] = [experiment]
            path = tmp_path / f"cuped-guide-{index}.yaml"
            path.write_text(yaml.safe_dump(document))
            analysis = Analysis.from_definitions("new_onboarding_v2", path, connection)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    rows = analysis.run()
                if index == 0:
                    assert [
                        getattr(warning.message, "code", None)
                        for warning in caught
                        if getattr(warning.message, "code", None) is not None
                    ] == ["analysis.cuped.covariate_unused"]
                assert {row.metric for row in rows} == {
                    "purchase_rate",
                    "avg_session_duration",
                }
                purchase_rows = [row for row in rows if row.metric == "purchase_rate"]
                if index == 0:
                    assert {row.method for row in purchase_rows} == {"unadjusted"}
                else:
                    assert {row.method for row in purchase_rows} == {"unadjusted", "cuped"}
            finally:
                analysis.close()
    finally:
        connection.disconnect()
