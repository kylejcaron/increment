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
from increment.semantics.models import AnalysisPlan, MeanMetric
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
                    covariate="pre_revenue",
                    decision_method=Method(name="same"),
                    sensitivity_methods=(Method(name="same", variance_reduction="cuped"),),
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
            decision_method=Method(name="cuped", variance_reduction="cuped"),
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
                    decision_method=Method(name="cuped", variance_reduction="cuped"),
                )
            ],
        )
    assert raised.value.code == "frame.metric.cuped_supported_metrics"


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
                    covariate="pre_revenue",
                    decision_method=Method(name="cuped", variance_reduction="cuped"),
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
    assert raised.value.code == "breakout.metric.daily_winsorization"
    assert raised.value.context["method"] == method
    assert raised.value.context["names"] == ("revenue",)


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
    assert raised.value.code == "facade.analysis.operation"


def test_artifact_context_refuses_unit_summary_source_with_operation_context(
    unit_summary_analysis,
):
    with pytest.raises(CapabilityError) as raised:
        _ = unit_summary_analysis.artifact_context

    assert raised.value.code == "facade.analysis.operation"
    assert raised.value.context["operation"] == "materialize"


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
            acknowledged=True, justification="assignment only moves revenue via uptake"
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
                acknowledged=True, justification="assignment only moves revenue via uptake"
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
                    acknowledged=True, justification="assignment only changes uptake"
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
                acknowledged=True, justification="assignment only affects uptake"
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
            decision_method=Method(name="cuped", variance_reduction="cuped"),
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
            acknowledged=True,
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
            acknowledged=True, justification="assignment only moves revenue via uptake"
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
            acknowledged=True, justification="assignment only moves revenue via uptake"
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
