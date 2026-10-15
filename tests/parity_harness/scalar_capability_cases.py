"""Lazy parity cases for scalar capability claims without an existing row case.

No scenarios or data are built until ``capability_cases`` is called by the
parity owner after this module has been imported.  Expected dispositions come
from ``tests.capability_consumers_scalar.PARITY_DECLARATIONS`` through the
merged declaration registry ``ParityCase`` consults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.parity_harness.cases import ParityCase


def _sequential_fixed_winsor_case() -> ParityCase:
    """A mean with a fixed clip under registered ``asymptotic_mean`` monitoring.

    ``dataset.event_rows`` gives 60 units per arm; 40 of them carry an
    in-window purchase of ``5 + (treatment) + (i % 5)`` and the other 20 retain
    the declared zero.  A clip at 8 turns the control purchases into
    ``8 x (5 + 6 + 7 + 8 + 8) = 272`` and the treatment purchases into
    ``8 x (6 + 7 + 8 + 8 + 8) = 296``, so every route must retain the exact
    scalar-mean states ``272/60 = 68/15`` and ``296/60 = 74/15``.
    """
    from datetime import date, timedelta
    from fractions import Fraction as F

    from increment import Analysis
    from increment.frame import MetricSpec
    from increment.plan import bind_automatic_sequential_plan
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, Definitions, Experiment
    from increment.sequential_source import native_observation_mapping
    from tests.analysis_factory import make_analysis
    from tests.parity_harness import dataset as ds
    from tests.parity_harness.cases import (
        ParityCase,
        _export_and_replay,
        _publish_and_adopt,
        _track_connection,
    )

    upper = 8.0
    rows = ds.event_rows()
    plan = AnalysisPlan(
        primary="revenue",
        inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
    )
    definitions_data = ds.definitions_dict(plan=plan, allocation={"control": 0.5, "treatment": 0.5})
    revenue = next(metric for metric in definitions_data["metrics"] if metric["name"] == "revenue")
    revenue["winsorization"] = {"upper_value": upper}
    definitions = Definitions.model_validate(definitions_data)
    experiment = definitions.experiment("exp")
    assert experiment is not None
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    warehouse_as_of = ds._EXPERIMENT_END - timedelta(days=1)
    # The panel window is [exposure, exposure + window_days); its capture closes the day after.
    panel_as_of = date(2025, 1, 11)

    def native_plan(experiment: Experiment = experiment) -> AnalysisPlan:
        bound = bind_automatic_sequential_plan(
            experiment.plan,
            [metric for metric in definitions.metrics if metric.name in experiment.metric_names],
            design=design,
            source_id=experiment.name,
            source_mapping=native_observation_mapping(
                definitions, experiment, on_mixed_assignment="error"
            ),
            pre_period_covariate=experiment.n_pre_periods > 0,
        )
        assert bound is not None
        return bound

    def mark(analysis: Analysis, as_of: date) -> Analysis:
        analysis._sequential_as_of = as_of  # ty: ignore[unresolved-attribute]
        return analysis

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, definitions, experiment="exp", plan=native_plan())
        return _track_connection(mark(native, warehouse_as_of), con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, definitions, experiment="exp", plan=native_plan())
        return mark(_publish_and_adopt(con, native), warehouse_as_of)

    def build_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.unit_summary_frame(con)
        con.disconnect()
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            design=design,
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    missing="zero",
                    winsorization={"upper_value": upper},
                )
            ],
            plan=plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()
        return mark(
            Analysis.from_unit_panel(
                panel,
                unit="user_id",
                group="variant",
                date="date",
                design=design,
                metrics=[
                    MetricSpec(
                        name="revenue",
                        type="mean",
                        missing="zero",
                        window_days=1,
                        winsorization={"upper_value": upper},
                    )
                ],
                plan=plan,
                experiment_id="exp",
                exposure_date="exposure_date",
            ),
            panel_as_of,
        )

    def build_moments() -> Analysis:
        return _export_and_replay(
            build_summary(),
            [
                MetricSpec(
                    name="revenue",
                    type="mean",
                    winsorization={"upper_value": upper},
                )
            ],
        )

    def probe(_ingress: str, analysis: Analysis) -> None:
        as_of = getattr(analysis, "_sequential_as_of", None)
        snapshot = analysis.capture_sequential(
            finalized=True, **({"as_of": as_of} if as_of is not None else {})
        )
        assert snapshot.registration.models[0].law == "scalar_mean"
        states = {state.group_id: state for state in snapshot.states}
        assert {group: state.n for group, state in states.items()} == {
            "control": 60,
            "treatment": 60,
        }
        assert {group: state.mean[0] for group, state in states.items()} == {
            "control": F(68, 15),
            "treatment": F(74, 15),
        }

    return ParityCase(
        id="scalar_sequential_fixed_threshold_winsor",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_summary,
            "from_unit_panel": build_panel,
            "from_moments": build_moments,
        },
        sequential=True,
        source_probe=probe,
        # Publishes/adopts an artifact and exports/replays moments on every run.
        slow=True,
    )


def _sequential_mean_cuped_case() -> ParityCase:
    """A mean CUPED decision under registered ``asymptotic_mean`` monitoring.

    The registered ``adjusted_mean`` law retains each arm's joint (Y, X)
    moments.  ``dataset.event_rows`` carries one pre-period purchase per unit
    inside the declared seven-day lookback, so the definitions, artifact and
    unit-summary routes read the same covariate and the exported checkpoint
    replays through ``from_moments``.  The unit panel's sequential capture
    needs a bounded window, and a windowed metric cannot carry a per-unit
    covariate, so that constructor refuses; the switchback plan gate refuses
    the CUPED decision override and the sequential inference before any frame
    is read.
    """
    from datetime import timedelta

    import polars as pl

    from increment import Analysis, CupedCovariate, Method
    from increment.frame import MetricSpec
    from increment.plan import bind_automatic_sequential_plan
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.models import (
        AnalysisPlan,
        CupedCovariateBinding,
        Definitions,
        Experiment,
        ExperimentMetric,
        Measure,
        MethodSpec,
    )
    from increment.sequential_source import native_observation_mapping
    from tests.analysis_factory import make_analysis
    from tests.parity_harness import dataset as ds
    from tests.parity_harness.cases import (
        ParityCase,
        _export_and_replay,
        _publish_and_adopt,
        _track_connection,
    )

    cuped = MethodSpec(
        name="cuped",
        variance_reduction="cuped",
        covariates=(
            CupedCovariateBinding(
                name="pre_revenue",
                kind="continuous",
                measure=Measure(fact="purchase", aggregation="sum"),
            ),
        ),
    )
    cuped_frame = Method.cuped(covariates=CupedCovariate(name="pre_revenue", kind="continuous"))
    rows = ds.event_rows()
    inference = {"kind": "asymptotic_mean", "expected_decision_sample_size": 100}
    # The warehouse plan binds the CUPED decision on the plan entry; a frame declares it
    # on its own MetricSpec and refuses a plan-level override (plan.frame.override).
    plan = AnalysisPlan(
        primary=ExperimentMetric(metric="revenue", decision_method=cuped), inference=inference
    )
    frame_plan = AnalysisPlan(primary="revenue", inference=inference)
    definitions = Definitions.model_validate(
        ds.definitions_dict(plan=plan, allocation={"control": 0.5, "treatment": 0.5})
    )
    experiment = definitions.experiment("exp")
    assert experiment is not None
    assert experiment.n_pre_periods > 0
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    warehouse_as_of = ds._EXPERIMENT_END - timedelta(days=1)

    def native_plan(experiment: Experiment = experiment) -> AnalysisPlan:
        bound = bind_automatic_sequential_plan(
            experiment.plan,
            [metric for metric in definitions.metrics if metric.name in experiment.metric_names],
            design=design,
            source_id=experiment.name,
            source_mapping=native_observation_mapping(
                definitions, experiment, on_mixed_assignment="error"
            ),
            pre_period_covariate=True,
        )
        assert bound is not None
        return bound

    def mark(analysis: Analysis) -> Analysis:
        analysis._sequential_as_of = warehouse_as_of  # ty: ignore[unresolved-attribute]
        return analysis

    def build_definitions() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, definitions, experiment="exp", plan=native_plan())
        return _track_connection(mark(native), con)

    def build_artifact() -> Analysis:
        con = ds.duckdb_connection(rows)
        native = make_analysis(con, definitions, experiment="exp", plan=native_plan())
        return mark(_publish_and_adopt(con, native))

    def summary_spec(window_days: int | None = None) -> MetricSpec:
        return MetricSpec(
            name="revenue",
            type="mean",
            missing="zero",
            decision_method=cuped_frame,
            window_days=window_days,
        )

    def build_summary() -> Analysis:
        con = ds.duckdb_connection(rows)
        frame = ds.unit_summary_frame(con)
        con.disconnect()
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            design=design,
            metrics=[summary_spec()],
            plan=frame_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_panel() -> Analysis:
        con = ds.duckdb_connection(rows)
        panel = ds.sequential_unit_panel_frame(con)
        con.disconnect()
        return Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            design=design,
            metrics=[summary_spec(window_days=1)],
            plan=frame_plan,
            experiment_id="exp",
            exposure_date="exposure_date",
        )

    def build_switchback() -> Analysis:
        schedule = pl.DataFrame(
            [
                {
                    "unit": unit,
                    "cycle": 0,
                    "period": period,
                    "step": 0,
                    "arm": "control" if period == 0 else "treatment",
                    "revenue": 5.0 + period + offset,
                }
                for unit, offset in (("a", 0.0), ("b", 1.0))
                for period in (0, 1)
            ]
        )
        return Analysis.from_switchback_panel(
            schedule,
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="arm",
            metrics={"revenue": "mean"},
            identification=design,
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=0, observation_steps=1),
            ),
            plan=plan,
        )

    def build_moments() -> Analysis:
        return _export_and_replay(build_summary(), [MetricSpec(name="revenue", type="mean")])

    def probe(_ingress: str, analysis: Analysis) -> None:
        as_of = getattr(analysis, "_sequential_as_of", None)
        snapshot = analysis.capture_sequential(
            finalized=True, **({"as_of": as_of} if as_of is not None else {})
        )
        assert snapshot.registration.models[0].law == "adjusted_mean"
        assert {state.group_id: (state.n, len(state.mean)) for state in snapshot.states} == {
            "control": (60, 2),
            "treatment": (60, 2),
        }
        (row,) = analysis.run()
        assert row.method == "cuped"
        assert row.inference == "asymptotic_mean"

    return ParityCase(
        id="scalar_sequential_mean_cuped_asymptotic",
        build={
            "from_definitions": build_definitions,
            "from_unit_day_artifact": build_artifact,
            "from_unit_summary": build_summary,
            "from_unit_panel": build_panel,
            "from_switchback_panel": build_switchback,
            "from_moments": build_moments,
        },
        sequential=True,
        source_probe=probe,
        # Publishes/adopts an artifact and exports/replays moments on every run.
        slow=True,
    )


def capability_cases() -> tuple[ParityCase, ...]:
    """Return real cases on explicit request; import alone remains metadata-only."""
    return (_sequential_fixed_winsor_case(), _sequential_mean_cuped_case())
