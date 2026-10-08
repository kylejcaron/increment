"""Readout dispatch + `Analysis.design=` plumbing for an
`Observational` design.

Exercises the seam from `Analysis.from_unit_summary(..., design=Observational(...))`
through `increment.readouts` into `increment.estimation.adjust.estimate_ate`
(real `LogisticPropensity`, not a stub) - and confirms the randomized path
stays byte-identical, `srm()`/`run_breakout()` refuse under an observational
design, and `control=`/`design=` are mutually exclusive on every seam
constructor.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pyarrow as pa
import pytest
from scipy.special import expit

from increment import AdjustmentSet, Analysis, IdentificationGate, Method, Observational
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.diagnostics import SRMResult
from increment.results import NotApplicable
from tests.analysis_factory import _moment_source, lift_rows, make_analysis_like
from tests.warning_codes import warning_codes, warning_context

if TYPE_CHECKING:
    from increment.sources import MomentSource


def _moment_fields() -> dict[str, object]:
    return {
        "moments_format": 10,
        "successes": None,
        "winsor_lower_percentile": None,
        "winsor_upper_percentile": None,
        "winsor_lower_bound": None,
        "winsor_upper_bound": None,
        "winsor_n": None,
        "winsor_n_lower": None,
        "winsor_n_upper": None,
    }


def _confounded_table(n: int, seed: int, effect: float = 0.2) -> pa.Table:
    # Same DGP tests/estimation/test_adjust.py's recovery tests use (repeated here: different test file).
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, expit(-0.3 + 1.2 * x1 - 0.8 * x2))
    y = 1.0 + 0.9 * x1 + 0.6 * x2 + effect * d + rng.normal(scale=0.5, size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C").tolist(),
            "revenue": y,
            "x1": x1,
            "x2": x2,
        }
    )


_OBS = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x1", "x2")))
# End-to-end runs gate with "trim": the DGP's true propensities fall outside
# [0.01, 0.99] for ~0.17% of units, so the default refuse-gate fires with near-certainty at realistic n, by design.
_OBS_TRIM = _OBS.model_copy(update={"gate": IdentificationGate(overlap="trim")})


def _obs_analysis(n=200, seed=11, design=None):
    return Analysis.from_unit_summary(
        _confounded_table(n, seed=seed),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=design or _OBS,
    )


def test_unadjusted_sensitivity_keeps_observational_design_default():
    analysis = _obs_analysis(n=400, seed=11, design=_OBS_TRIM)
    with pytest.warns(IncrementWarning):
        implicit = lift_rows(analysis.run(sensitivity_methods=[Method(name="unadjusted")]))
        explicit = lift_rows(
            analysis.run(
                decision_method=Method(name="iptw"),
                sensitivity_methods=[Method(name="unadjusted")],
            )
        )
    assert [(row.method, row.method_role) for row in implicit] == [
        ("iptw", "decision"),
        ("unadjusted", "sensitivity"),
    ]
    for actual, expected in zip(implicit, explicit, strict=True):
        assert actual.require_lift().value == pytest.approx(expected.require_lift().value)
        assert actual.require_lift().lb == pytest.approx(expected.require_lift().lb)
        assert actual.require_lift().ub == pytest.approx(expected.require_lift().ub)


@pytest.mark.parametrize("decision_name", [None, "unadjusted"])
def test_declared_and_call_time_methods_match_explicit_secondary_decision(decision_name):
    from contextlib import nullcontext

    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    table = _confounded_table(400, seed=11, effect=1.5)
    sensitivity = Method(name="dml")
    decision = None if decision_name is None else Method(name=decision_name)
    declared = Analysis.from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(name="revenue", decision_method=decision, sensitivity_methods=(sensitivity,))
        ],
        design=_OBS_TRIM,
        plan=AnalysisPlan(secondaries=["revenue"], q=0.01),
    )
    call_time = Analysis.from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=_OBS_TRIM,
        plan=AnalysisPlan(secondaries=["revenue"], q=0.01),
    )
    with pytest.warns(IncrementWarning) if decision is None else nullcontext():
        declared_rows = lift_rows(declared.run())
        call_rows = lift_rows(
            call_time.run(sensitivity_methods=[sensitivity])
            if decision is None
            else call_time.run(decision_method=decision, sensitivity_methods=[sensitivity])
        )
        oracle = lift_rows(
            call_time.run(
                decision_method=Method(name=decision_name or "iptw"),
                sensitivity_methods=[sensitivity],
            )
        )
        inherited = lift_rows(declared.run(decision_method=Method(name="unadjusted")))
        cleared = lift_rows(declared.run(sensitivity_methods=[]))
    expected_roles = [(decision_name or "iptw", "decision"), ("dml", "sensitivity")]
    assert oracle[0].family_threshold == pytest.approx(0.01)
    for rows in (declared_rows, call_rows):
        assert [(row.method, row.method_role) for row in rows] == expected_roles
        for actual, expected in zip(rows, oracle, strict=True):
            actual_lift, expected_lift = actual.require_lift(), expected.require_lift()
            assert (actual_lift.value, actual_lift.lb, actual_lift.ub) == pytest.approx(
                (expected_lift.value, expected_lift.lb, expected_lift.ub), rel=1e-12, abs=1e-12
            )
    assert [(row.method, row.method_role) for row in inherited] == [
        ("unadjusted", "decision"),
        ("dml", "sensitivity"),
    ]
    assert [(row.method, row.method_role) for row in cleared] == [
        (decision_name or "iptw", "decision")
    ]


def test_run_dispatches_clustered_dml_with_arm_atomic_folds():
    tbl = _confounded_table(200, seed=13)
    variants = tbl["variant"].to_pylist()
    seen = {"C": 0, "T": 0}
    geo_ids = []
    for variant in variants:
        geo_ids.append(f"{variant}-geo-{seen[variant] // 2}")
        seen[variant] += 1
    tbl = tbl.append_column("geo_id", pa.array(geo_ids))
    analysis = Analysis.from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        cluster="geo_id",
        design=_OBS_TRIM,
    )
    with pytest.warns(IncrementWarning) as rec:
        (estimate,) = lift_rows(analysis.run(decision_method=Method(name="dml")))
    assert "estimation.adjust_common.covariate_balance_advisory" in warning_codes(rec)
    assert estimate.method == "dml"
    assert estimate.n_clusters is not None and estimate.n_clusters >= 10


def test_clustered_observational_cuped_sensitivity_refuses_and_dropping_it_keeps_iptw():
    """A CUPED-configured sensitivity refuses under a declared cluster; the remedy keeps the
    causal iptw decision and removes only the CUPED method."""
    tbl = _confounded_table(200, seed=13)
    seen = {"C": 0, "T": 0}
    geo_ids = []
    for variant in tbl["variant"].to_pylist():
        geo_ids.append(f"{variant}-geo-{seen[variant] // 2}")
        seen[variant] += 1
    analysis = Analysis.from_unit_summary(
        tbl.append_column("geo_id", pa.array(geo_ids)),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        cluster="geo_id",
        design=_OBS_TRIM,
    )
    cuped_unadjusted = Method(name="unadjusted", variance_reduction="cuped")
    with pytest.raises(CapabilityError) as exc_info:
        analysis.run(decision_method=Method(name="iptw"), sensitivity_methods=[cuped_unadjusted])
    assert exc_info.value.code == "arm.adjustment.cluster_cuped"
    assert exc_info.value.context["cluster"] == "geo_id"
    with pytest.warns(IncrementWarning):
        (estimate,) = lift_rows(analysis.run(decision_method=Method(name="iptw")))
    assert (estimate.method, estimate.method_role) == ("iptw", "decision")


def test_run_refuses_a_control_only_observational_source():
    """The arm gate covers the observational branch too: a control-only
    source refuses with the stable code instead of an estimator error."""
    from increment.errors import InvalidRequestError

    table = _confounded_table(120, seed=5)
    control_only = table.set_column(
        table.schema.get_field_index("variant"),
        "variant",
        pa.array(["C"] * table.num_rows),
    )
    analysis = Analysis.from_unit_summary(
        control_only, unit="user_id", group="variant", metrics={"revenue": "mean"}, design=_OBS
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        analysis.run()
    assert exc_info.value.code == "readout.arms.no_treatment"
    assert exc_info.value.context["observed_arms"] == ("C",)


def test_run_dispatches_iptw_end_to_end():
    # Real LogisticPropensity, confounded frame; run() must return one
    # LiftEstimate with method == "iptw" and a finite ordered interval.
    (est,) = lift_rows(_obs_analysis(design=_OBS_TRIM).run())
    assert est.method == "iptw"
    lift = est.require_lift()
    assert lift.lb is not None
    assert lift.value is not None
    assert lift.ub is not None
    assert lift.lb < lift.value < lift.ub


def test_run_undeclared_metric_preferred_direction_is_none_under_observational():
    """Finding 1: `estimate_ate`'s own unconditional
    `preferred_direction=metric.preferred_direction` stamps (unadjusted
    branch and adjust_fn branch) bypassed the readouts' explicitness fix
    entirely for every Observational-design run -- an undeclared metric
    silently reported "increase" end-to-end through `Analysis.run()`."""
    (est,) = lift_rows(_obs_analysis(design=_OBS_TRIM).run())
    assert est.preferred_direction is None
    (unadj,) = lift_rows(
        _obs_analysis(design=_OBS_TRIM).run(decision_method=Method(name="unadjusted"))
    )
    assert unadj.preferred_direction is None


def test_control_and_design_mutually_exclusive():
    tbl = _confounded_table(20, seed=0)
    with pytest.raises(InvalidRequestError) as exc:
        Analysis.from_unit_summary(
            tbl,
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            control="C",
            design=_OBS,
        )
    assert exc.value.code == "query.fact_resolution.pass_exactly_one"
    with pytest.raises(InvalidRequestError) as exc:
        Analysis.from_unit_summary(
            tbl, unit="user_id", group="variant", metrics={"revenue": "mean"}
        )
    assert exc.value.code == "query.fact_resolution.pass_exactly_one"


def test_srm_not_applicable_under_observational():
    result = _obs_analysis().srm()
    assert isinstance(result, NotApplicable)
    assert result.check == "srm"
    assert "not randomized" in result.reason


def test_run_breakout_refused_under_observational():
    with pytest.raises(CapabilityError) as exc:
        _obs_analysis().run_breakout()
    assert exc.value.code == "facade.analysis.operation"


def test_run_by_segment_refused_under_observational():
    from increment import readouts as ro
    from increment.frame import from_unit_summary

    src = from_unit_summary(
        _confounded_table(200, seed=11),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        design=_OBS,
    )
    with pytest.raises(UnsupportedRequestError) as exc:
        ro.run(src, by=("x1",))
    assert exc.value.code == "readout.view.observational"


def test_run_sequential_inference_refused_under_observational():
    from increment.frame import MetricSpec, from_unit_summary, synthesise_metric
    from tests.sequential_cases import declared_plan

    specs = [MetricSpec(name="revenue")]
    plan = declared_plan(
        [synthesise_metric(s) for s in specs], source_id="frame", design=_OBS, transformations=specs
    )
    with pytest.raises(CapabilityError) as raised:
        from_unit_summary(
            _confounded_table(200, seed=11),
            unit="user_id",
            group="variant",
            control="C",
            metrics=specs,
            design=_OBS,
            plan=plan,
        )
    assert raised.value.code == "sequential.route.unsupported"


def test_run_one_sided_alternative_observational_doubles_alpha_and_labels():
    """Under an Observational design, alternative= now dispatches through
    estimate_ate -> infer_ate: the interval matches a two-sided run at
    2*alpha (the same identity infer_lift/infer_ate use for the randomized
    path), and the row is labeled with the requested direction."""
    from increment import readouts as ro
    from increment.frame import from_unit_summary
    from increment.semantics.models import AnalysisPlan

    table = _confounded_table(200, seed=11)
    src_two = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        design=_OBS_TRIM,
        plan=AnalysisPlan(alpha=0.10, primary="revenue"),
    )
    src_one = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        design=_OBS_TRIM,
        plan=AnalysisPlan(alpha=0.05, primary="revenue", alternative="greater"),
    )
    (two,) = ro.run(src_two)
    (one,) = ro.run(src_one)
    assert one.require_lift().lb == pytest.approx(two.require_lift().lb, rel=1e-9)
    assert one.require_lift().ub == pytest.approx(two.require_lift().ub, rel=1e-9)
    assert one.require_lift().level == pytest.approx(0.90)
    assert one.alternative == "greater"
    assert two.alternative == "two-sided"


def test_run_observational_declared_relative_margin_refused_at_construction():
    """A DECLARED relative margin (a plan-bound ExperimentMetric.margin)
    rides the relative axis, which the observational path does not build
    a shifted null for - this is a mechanism x capability check, so it
    now fires at construction (not sailing through to a first `run()`
    call, which would otherwise silently test two-sided vs 0 and change
    what the metric's stat_sig means vs a randomized read of the same
    guardrail). The refusal must be raised by the constructor call itself."""
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    with pytest.raises(UnsupportedRequestError) as exc:
        from_unit_summary(
            _confounded_table(20, seed=0),
            unit="user_id",
            group="variant",
            control="C",
            metrics=[MetricSpec(name="revenue", preferred_direction="decrease")],
            design=_OBS,
            plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="revenue", margin=0.02)]),
        )
    assert exc.value.code == "plan.observational.relative_margin"


def test_moments_source_observational_declared_relative_margin_refused_at_construction():
    """`MomentsSource` (reached via `Analysis.from_moments`) is a second
    `MomentSource`-implementing constructor that can carry
    `design=Observational(...)` alongside a declared plan; a cleanup that
    moved this refusal into `increment.frame._resolve_design_and_plan`
    once narrowed its coverage to that module's two frame constructors,
    silently dropping the declared margin here instead of raising. It
    must hit the same construction-time refusal `from_unit_summary` does
    above."""
    from increment.semantics.models import AnalysisPlan, ExperimentMetric, MeanMetric
    from increment.sources import MomentsSource

    guardrail = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="decrease",
    )
    rows = [
        {
            **_moment_fields(),
            "experiment_id": "e",
            "metric": "revenue",
            "group_id": "C",
            "n": 100,
            "sum_y": 100.0,
            "sum_y2": 150.0,
        },
        {
            **_moment_fields(),
            "experiment_id": "e",
            "metric": "revenue",
            "group_id": "T",
            "n": 100,
            "sum_y": 102.0,
            "sum_y2": 160.0,
        },
    ]
    with pytest.raises(UnsupportedRequestError) as exc:
        MomentsSource(
            rows,
            metrics=[guardrail],
            study_id="e",
            design=_OBS,
            plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="revenue", margin=0.02)]),
        )
    assert exc.value.code == "plan.observational.relative_margin"


def test_sql_panel_source_observational_declared_relative_margin_refused_at_construction():
    """SqlPanelSource's frame adapter must hit the same construction-time
    refusal as from_unit_summary/MomentsSource, not silently drop the
    declared margin."""
    import ibis

    from increment.frame import MetricSpec
    from increment.query.source import SqlPanelSource
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    con = ibis.duckdb.connect()
    with pytest.raises(UnsupportedRequestError) as exc:
        SqlPanelSource.from_frame_via_memtable(
            con,
            _confounded_table(20, seed=0),
            unit="user_id",
            group="variant",
            metrics=[MetricSpec(name="revenue", preferred_direction="decrease")],
            design=_OBS,
            plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="revenue", margin=0.02)]),
        )
    assert exc.value.code == "plan.observational.relative_margin"


def test_run_observational_declared_absolute_margin_is_no_longer_refused():
    """The counterpart: a declared Metric.margin_abs rides the absolute
    axis, which this path now supports - it must reach the readout
    pipeline instead of being refused at the margin gate. (This source
    carries no treatment moments, so the run stops at the arm gate; the
    point is that no margin-axis refusal fires.)"""
    from increment import readouts as ro
    from increment.semantics.models import MeanMetric
    from tests.test_readouts_encouragement import FakeMomentSource

    guardrail = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="decrease",
        margin_abs=0.5,
    )
    src = FakeMomentSource([], metrics=[guardrail], capabilities={"total"}, design=_OBS)
    with pytest.raises(InvalidRequestError) as exc:
        ro.run(src)
    assert exc.value.code == "readout.arms.no_treatment"


@pytest.mark.parametrize("declaration", [{}, {"margin_abs": 0.5}])
def test_asof_lift_observational_refusal_covers_declared_margin(declaration):
    """asof_lift refuses observational designs unconditionally at entry,
    so a declared guardrail can never silently degrade there - pinned
    both with and without a margin-declaring metric so the entry refusal
    is proven to fire before any margin handling could be skipped.

    A relative `margin` is deliberately absent from this space: an
    observational design refuses one while resolving the plan, so such a
    source cannot be constructed at all, let alone reach asof_lift (see
    test_run_observational_declared_relative_margin_refused_at_construction).
    The absolute axis is the one that rides here."""
    from increment import readouts as ro
    from increment.semantics.models import MeanMetric
    from tests.test_readouts_encouragement import FakeMomentSource

    guardrail = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="decrease",
        **declaration,
    )
    src = FakeMomentSource([], metrics=[guardrail], capabilities={"asof"}, design=_OBS)
    with pytest.raises(UnsupportedRequestError) as exc:
        ro.asof_lift(src)
    assert exc.value.code == "readout.view.observational"


def test_from_moments_observational_hits_capability_error():
    # A moments cube has no unit grain to attach the adjustment covariates to;
    # the refusal names the covariates, not only the missing unit grain.
    from increment.semantics.models import AnalysisPlan

    rows = [
        {
            **_moment_fields(),
            "experiment_id": "e",
            "metric": "revenue",
            "group_id": g,
            "n": 10,
            "sum_y": s,
            "sum_y2": q,
        }
        for g, s, q in (("T", 45.0, 240.0), ("C", 30.0, 110.0))
    ]
    an = Analysis.from_moments(rows, metrics={"revenue": "mean"}, design=_OBS, plan=AnalysisPlan())
    with pytest.raises(CapabilityError) as exc_info:
        lift_rows(an.run())
    assert exc_info.value.code == "source.moments.covariate_unavailable"


def test_from_unit_panel_observational_missing_covariate_column():
    # FramePanelSource now serves unit_frame for an unwindowed mean metric
    # (Analysis.from_unit_panel's own docstring): _OBS declares covariates
    # x1/x2, which this panel frame does not carry, so the refusal now
    # names the missing covariate columns rather than the whole path.
    tbl = pa.table(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "variant": ["T", "T", "T", "T", "C", "C", "C", "C"],
            "day": ["2025-01-01", "2025-01-02"] * 4,
            "revenue": [1.0, 2.0, 3.0, 4.0, 1.0, 1.5, 2.0, 2.5],
        }
    )
    an = Analysis.from_unit_panel(
        tbl,
        unit="user_id",
        group="variant",
        date="day",
        metrics={"revenue": "mean"},
        design=_OBS,
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        lift_rows(an.run())
    assert exc_info.value.code == "frame.frame_panel.unit_covariate_column"


@pytest.mark.parametrize("shape", ["summary", "panel"])
def test_observational_quantile_metric_refuses_on_every_frame_shape(shape):
    """A quantile metric has no observational estimator: both frame shapes refuse it by name
    instead of reporting an adjusted mean effect under the quantile metric's name."""
    from increment.frame import MetricSpec

    table = _confounded_table(200, seed=11)
    metrics = [MetricSpec(name="revenue", type="quantile", quantile=0.5)]
    if shape == "summary":
        analysis = Analysis.from_unit_summary(
            table, unit="user_id", group="variant", metrics=metrics, design=_OBS_TRIM
        )
    else:
        panel = table.append_column("day", pa.array(["2025-01-01"] * table.num_rows))
        analysis = Analysis.from_unit_panel(
            panel, unit="user_id", group="variant", date="day", metrics=metrics, design=_OBS_TRIM
        )
    with pytest.raises(UnsupportedRequestError) as exc_info:
        lift_rows(analysis.run())
    assert exc_info.value.code == "readout.observational.quantile"
    assert exc_info.value.context == {"metric": "revenue"}


def test_from_unit_panel_daily_ratio_ignores_observational_adjustment_capability():
    """Descriptive daily ratios remain available for observational sources."""
    from increment.frame import MetricSpec

    table = pa.table(
        {
            "user_id": ["c1", "c2", "t1", "t2"] * 2,
            "variant": ["C", "C", "T", "T"] * 2,
            "day": ["2025-01-01"] * 4 + ["2025-01-02"] * 4,
            "revenue": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            "sessions": [1.0] * 8,
            "x1": [0.0, 1.0, 0.5, 1.5] * 2,
        }
    )
    analysis = Analysis.from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        metrics=[
            MetricSpec(
                name="revenue_per_session",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
            )
        ],
        design=Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x1",))),
    )

    daily = analysis.run_daily()

    assert len(daily) == 4
    assert {row.metric for row in daily} == {"revenue_per_session"}
    expected = {
        ("2025-01-01", "C"): 1.5,
        ("2025-01-01", "T"): 3.5,
        ("2025-01-02", "C"): 3.0,
        ("2025-01-02", "T"): 7.0,
    }
    actual = {}
    for row in daily:
        assert row.value is not None
        actual[(row.ds.isoformat(), row.group_id)] = row.value.value
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-12)


def test_randomized_frame_path_unchanged():
    # control="C" sugar still builds Randomized and yields unadjusted estimates.
    an = Analysis.from_unit_summary(
        _confounded_table(200, seed=11),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    (est,) = lift_rows(an.run())
    assert est.method == "unadjusted"
    assert isinstance(an.srm(expected={"C": 0.5, "T": 0.5}), SRMResult)


# The absolute (additive) channel at the readout layer: margins_abs
# unblocking and the observational-only value_scale= selector.


def _near_zero_table(n=2000, seed=20260811):
    """E[Y(0)] = 0 under confounding - the regime where relative lift is
    unidentified and only the additive effect is estimable."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = (rng.random(n) < expit(0.8 * x)).astype(int)
    y = 0.5 * x + 0.3 * d + rng.normal(size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x": x,
        }
    )


def _obs_src(table=None, *, design=None, margin=None, margin_abs=None, preferred_direction=None):
    """Build a `revenue` metric with guardrail semantics baked in at
    construction (the compiled decision procedure resolves once, at
    construction, so a post-hoc `src.metrics` swap - the old approach here -
    on the frame path, so a margin travels through a plan-bound
    `ExperimentMetric` guardrail instead - the canonical mechanism for
    this now.
    """
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    spec_kwargs = (
        {} if preferred_direction is None else {"preferred_direction": preferred_direction}
    )
    plan = None
    if margin is not None or margin_abs is not None:
        plan = AnalysisPlan(
            guardrails=[ExperimentMetric(metric="revenue", margin=margin, margin_abs=margin_abs)]
        )
    return from_unit_summary(
        table if table is not None else _confounded_table(400, seed=3),
        unit="user_id",
        group="variant",
        control="C",
        metrics=[MetricSpec(name="revenue", **spec_kwargs)],
        design=design or _OBS_TRIM,
        plan=plan,
    )


_OBS_X = Observational(
    control_group="C",
    adjustment=AdjustmentSet(covariates=("x",)),
    gate=IdentificationGate(overlap="trim"),
)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_run_margins_abs_reaches_the_estimate_and_decides_on_the_additive_interval():
    """The absolute guardrail travels run -> estimate_ate -> contrast ->
    infer_ate: null_abs stamped, preferred_direction stamped from the
    declaration (without which prob_favorable raises on every
    observational row), the additive endpoints present, and the decision
    read off them rather than the relative interval."""
    from increment import readouts as ro

    src = _obs_src(preferred_direction="decrease", margin_abs=0.10)
    (est,) = ro.run(src)

    assert est.null_abs == pytest.approx(0.10)
    assert est.alternative == "less"  # decrease-preferred: adverse side is up
    assert est.preferred_direction == "decrease"
    assert est.abs_lb is not None and est.abs_ub is not None
    assert est.prob_favorable() is None
    assert est.stat_sig() == (est.abs_ub < 0.10)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_run_declared_margin_abs_resolves_like_the_randomized_branch():
    from increment import readouts as ro

    src = _obs_src(preferred_direction="increase", margin_abs=0.25)
    (est,) = ro.run(src)
    assert est.null_abs == pytest.approx(-0.25)
    assert est.alternative == "greater"


def test_run_margins_abs_key_must_name_a_declared_metric_under_observational():
    """The absolute axis rides now, so its anti-typo gate has to ride with
    it - a typo'd guardrail key would silently drop the guardrail it meant
    to set, refused at plan-resolution (construction) time."""
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            _confounded_table(400, seed=3),
            unit="user_id",
            group="variant",
            control="C",
            metrics=[MetricSpec(name="revenue", preferred_direction="decrease")],
            design=_OBS_TRIM,
            plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="revenu", margin_abs=0.10)]),
        )
    assert exc.value.code == "plan.metrics.unknown"


def test_run_margins_abs_also_reaches_an_unadjusted_observational_row():
    """`Method(name='unadjusted')` under an observational design runs the
    moments path, which populates the additive pair too - the resolved
    boundary must reach it rather than being dropped on the way."""
    from increment import readouts as ro
    from increment.estimation.engine import Method

    src = _obs_src(preferred_direction="decrease", margin_abs=0.10)
    (est,) = ro.run(src, decision_method=Method(name="unadjusted"))
    assert est.method == "unadjusted"
    assert est.null_abs == pytest.approx(0.10)
    assert est.preferred_direction == "decrease"
    assert est.abs_lb is not None and est.abs_ub is not None


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_absolute_margin_without_posterior_probability_stays_unavailable():
    """A sampling-only absolute guardrail has no posterior probability."""
    from increment import readouts as ro

    src = _obs_src(preferred_direction="decrease", margin_abs=0.10)
    (est,) = ro.run(src)
    degenerate = est.model_copy(update={"abs_se": None})
    assert degenerate.prob_favorable() is None


def test_run_value_scale_rescues_a_near_zero_metric_end_to_end():
    """At a near-zero control mean the relative scale is unidentified. The
    prior-free joint path represents that rather than refusing: the Fieller
    set comes back disconnected, so the row publishes no relative interval
    and records why, while the additive pair stays intact. `value_scale=`
    is what turns that additive effect into the reported number."""
    from increment import readouts as ro

    src = _obs_src(_near_zero_table(), design=_OBS_X)
    (unrescued,) = ro.run(src)
    assert unrescued.value_scale == "relative"
    assert unrescued.relative_confidence_set is not None
    assert unrescued.relative_confidence_set.geometry == "disconnected"
    unusable = unrescued.require_lift()
    assert unusable.lb is None and unusable.ub is None
    assert unrescued.abs_lb is not None and unrescued.abs_ub is not None

    (est,) = ro.run(src, value_scale={"revenue": "absolute"})
    assert est.value_scale == "absolute"
    assert est.require_lift().value == pytest.approx(0.3, abs=0.15)


def test_run_value_scale_key_naming_a_filtered_metric_raises():
    """`value_scale` rides the same `metrics=`-selected contract as
    `margins`/`null_lifts`/`margins_abs` - a key naming a declared but
    filtered-out metric must refuse, not silently go unread."""
    from increment.frame import from_unit_summary

    table = _confounded_table(200, seed=3).append_column(
        "second", pa.array(np.asarray(_confounded_table(200, seed=3)["revenue"].to_numpy()))
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean", "second": "mean"},
        design=_OBS_TRIM,
    )
    from increment import readouts as ro

    with pytest.raises(InvalidRequestError) as exc:
        ro.run(src, metrics=["revenue"], value_scale={"second": "absolute"})
    assert exc.value.code == "readout.value_scale.unknown_metric"


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_run_groups_observational_dispatch_by_declared_methods():
    """Two metrics with different declared `methods` split into separate
    `estimate_ate` groups under an observational design - one metric's
    adjustment choice must not silently apply to its sibling."""
    from increment.estimation.engine import Method
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(300, seed=7)
    table = table.append_column("signups", pa.array(np.asarray(table["revenue"].to_numpy()) * 0.5))
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="revenue"),  # no override - design default (iptw)
            MetricSpec(name="signups", decision_method=Method(name="unadjusted")),
        ],
        design=_OBS_TRIM,
    )
    from increment import readouts as ro

    by_metric_method = {(r.metric, r.method) for r in ro.run(src)}
    assert ("revenue", "iptw") in by_metric_method
    assert ("signups", "unadjusted") in by_metric_method
    assert ("signups", "iptw") not in by_metric_method
    assert ("revenue", "unadjusted") not in by_metric_method


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_run_observational_grouping_preserves_declaration_order():
    """A metric in the MIDDLE of the declared order that splits into its
    own group (a declared `methods` binding) must not move in the
    output - grouping is an internal dispatch detail, not a caller-
    visible reordering. Regression: grouped dispatch emitted `[a, c, b]`
    for declared `[a, b(bound), c]`."""
    from increment.estimation.engine import Method
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(300, seed=7)
    table = table.append_column("b", pa.array(np.asarray(table["revenue"].to_numpy()) * 2.0))
    table = table.append_column("c", pa.array(np.asarray(table["revenue"].to_numpy()) * 3.0))
    table = table.rename_columns(
        ["a" if name == "revenue" else name for name in table.column_names]
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="a"),
            MetricSpec(name="b", decision_method=Method(name="unadjusted")),
            MetricSpec(name="c"),
        ],
        design=_OBS_TRIM,
    )
    from increment import readouts as ro

    order = [r.metric for r in ro.run(src)]
    assert order == ["a", "b", "c"], f"declaration order violated: {order}"


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_run_absolute_metric_in_one_group_does_not_refuse_an_unadjusted_sibling_group():
    """A metric asking for `value_scale="absolute"` must not poison a
    SIBLING group whose only method is `unadjusted`. Grouping sends one
    `estimate_ate` call per resolved-methods group, and the
    unadjusted-cannot-be-absolute refusal compares `value_scale` against
    that call's own `methods` - so it has to ignore metrics the call
    never estimates. Regression: the unadjusted group raised on the
    absolute request belonging to the adjusted group."""
    from increment.estimation.engine import Method
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(300, seed=7)
    table = table.append_column(
        "adjusted_metric", pa.array(np.asarray(table["revenue"].to_numpy()) * 2.0)
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="revenue", decision_method=Method(name="unadjusted")),
            MetricSpec(name="adjusted_metric", decision_method=Method(name="iptw")),
        ],
        design=_OBS_TRIM,
    )
    from increment import readouts as ro

    rows = ro.run(src, value_scale={"adjusted_metric": "absolute"})
    by_metric = {r.metric: r for r in rows}
    assert by_metric["revenue"].value_scale == "relative"
    assert by_metric["revenue"].method == "unadjusted"
    assert by_metric["adjusted_metric"].value_scale == "absolute"
    assert by_metric["adjusted_metric"].method == "iptw"


def test_run_groups_observational_dispatch_by_declared_priors():
    """Two metrics with distinct declared `prior`s split into separate
    `estimate_ate` groups - each shrinks toward its OWN prior, proving
    `readouts.run`'s grouping forwards `config.prior` (not the call-wide
    scalar) per group."""
    from increment.estimation.engine import Method
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(300, seed=7)
    table = table.append_column("signups", pa.array(np.asarray(table["revenue"].to_numpy()) * 0.5))

    from increment import readouts as ro

    flat_src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[MetricSpec(name="revenue"), MetricSpec(name="signups")],
        design=_OBS_TRIM,
    )
    flat_by_metric = {
        r.metric: r for r in ro.run(flat_src, decision_method=Method(name="unadjusted"))
    }

    bound_src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="revenue", prior=Normal(mu=0.0, sigma=0.01)),
            MetricSpec(name="signups", prior=Normal(mu=0.0, sigma=0.02)),
        ],
        design=_OBS_TRIM,
    )
    shrunk_by_metric = {
        r.metric: r for r in ro.run(bound_src, decision_method=Method(name="unadjusted"))
    }

    for name in ("revenue", "signups"):
        assert abs(shrunk_by_metric[name].require_lift().value) < abs(
            flat_by_metric[name].require_lift().value
        ), f"{name}: declared prior did not shrink its own estimate"


def _two_metric_src(specs):
    from increment.frame import from_unit_summary

    table = _confounded_table(300, seed=7)
    table = table.append_column("signups", pa.array(np.asarray(table["revenue"].to_numpy()) * 0.5))
    return from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics=specs, design=_OBS_TRIM
    )


class _ConstantPropensity:
    """Minimal `Learner`-conforming stand-in: one intercept fitted at the
    treated fraction. The grouping shapes below need two factories that
    compare unequal by identity and predictions a real IPTW fit can consume."""

    def __init__(self) -> None:
        self._probability = 0.5

    def fit(self, X: np.ndarray, d: np.ndarray) -> None:
        self._probability = float(np.clip(np.mean(d), 0.05, 0.95))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._probability)


def _split_spec_shapes():
    """Three `MetricSpec` pairs: one that collapses to a single
    `estimate_ate` group, and two that split into one group per metric -
    by distinct method names, and by distinct-but-functionally-identical
    learner callables (a plain callable compares by object identity, so
    two separately-built factories never group)."""
    from increment.estimation.engine import Method
    from increment.frame import MetricSpec

    return {
        "one_group": [MetricSpec(name="revenue"), MetricSpec(name="signups")],
        "split_by_name": [
            MetricSpec(name="revenue", decision_method=Method(name="iptw")),
            MetricSpec(name="signups", decision_method=Method(name="dml")),
        ],
        "split_by_learner_identity": [
            MetricSpec(
                name="revenue",
                decision_method=Method(
                    name="iptw", propensity_learner=lambda: _ConstantPropensity()
                ),
            ),
            MetricSpec(
                name="signups",
                decision_method=Method(
                    name="iptw", propensity_learner=lambda: _ConstantPropensity()
                ),
            ),
        ],
    }


@pytest.mark.parametrize("shape", ["one_group", "split_by_name", "split_by_learner_identity"])
def test_shared_prior_scale_refusal_is_judged_across_method_groups(shape):
    """One scalar `prior=` spanning a relative metric and an absolute one is
    refused no matter how many `estimate_ate` groups the metrics land in.
    The judgment compares metrics against each other, so it must run over
    the whole selected set - per-group it sees one metric at a time and
    every fragment passes alone. Regression: per-metric `methods=` used
    to silently buy exemption from the guardrail."""
    from increment import readouts as ro
    from increment.estimation.inference import Normal

    with pytest.raises(InvalidRequestError) as exc:
        ro.run(
            _two_metric_src(_split_spec_shapes()[shape]),
            prior=Normal(mu=0.0, sigma=0.1),
            value_scale={"revenue": "relative", "signups": "absolute"},
        )
    assert exc.value.code == "estimation.adjust.prior_interpreted_call"


def test_global_prior_method_scales_refuse_across_metric_groups_including_default_iptw(
    monkeypatch,
):
    """A call-wide prior must use one method parameterization across all
    metric groups, including metrics that inherit estimate_ate's IPTW
    default. The refusal belongs before either group's moments query."""
    from increment import readouts as ro
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec

    source = _two_metric_src(
        [
            MetricSpec(name="revenue", decision_method=Method(name="unadjusted")),
            MetricSpec(name="signups"),
        ]
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("source moments accessed before global-prior validation")

    monkeypatch.setattr(source, "moments", fail)
    with pytest.raises(InvalidRequestError) as exc:
        ro.run(source, prior=Normal(mu=0.0, sigma=0.1))
    assert exc.value.code == "estimation.adjust.prior.method_scale"


def test_default_adjustment_ratio_refuses_during_prior_preflight():
    """A call-wide prior does not bypass a structurally unsupported ratio method."""
    from increment import readouts as ro
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(300, seed=7).append_column(
        "sessions",
        pa.array(np.ones(300)),
    )
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="revenue", decision_method=Method(name="unadjusted")),
            MetricSpec(
                name="rev_per_session",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
            ),
        ],
        design=_OBS_TRIM,
    )

    with pytest.raises(UnsupportedRequestError) as exc:
        ro.run(source, prior=Normal(mu=0.0, sigma=0.1))
    assert exc.value.code == "estimation.adjust_common.supported_ratio_metric"
    assert exc.value.context["metric"] == "rev_per_session"
    assert exc.value.context["method"] == "iptw"


def test_all_ratio_metrics_refuse_at_validation_with_capability_code():
    from increment import readouts as ro
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(80, seed=8).append_column("sessions", pa.array(np.ones(80)))
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(
                name="rev_per_session",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
            )
        ],
        design=_OBS_TRIM,
    )
    with pytest.raises(UnsupportedRequestError) as exc:
        ro.run(source)
    assert exc.value.code == "estimation.adjust_common.supported_ratio_metric"


def test_global_prior_counts_custom_ratio_adjustment(monkeypatch):
    """Custom ratio adjustments may emit linear-relative rows and must count."""
    from increment import readouts as ro
    from increment.estimation.adjust import ADJUSTMENTS
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec, from_unit_summary

    table = _confounded_table(300, seed=7).append_column(
        "sessions",
        pa.array(np.ones(300)),
    )
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="revenue", decision_method=Method(name="unadjusted")),
            MetricSpec(
                name="rev_per_session",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
                decision_method=Method(name="custom_ratio"),
            ),
        ],
        design=_OBS_TRIM,
    )
    monkeypatch.setitem(ADJUSTMENTS._entries, "custom_ratio", lambda *_args, **_kwargs: [])

    def fail(*_args, **_kwargs):
        raise AssertionError("source accessed before global-prior validation")

    monkeypatch.setattr(source, "moments", fail)
    with pytest.raises(InvalidRequestError) as exc:
        ro.run(source, prior=Normal(mu=0.0, sigma=0.1))
    assert exc.value.code == "estimation.adjust.prior.method_scale"


@pytest.mark.parametrize("shape", ["one_group", "split_by_name", "split_by_learner_identity"])
@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_shared_prior_absolute_warning_fires_once_across_method_groups(shape):
    """The incommensurable-additive-units warning counts metrics too, so it
    is emitted exactly once over the whole call - not once per group, and
    not zero times because each group holds a single metric."""
    from increment import readouts as ro
    from increment.estimation.inference import Normal

    with pytest.warns(IncrementWarning) as record:
        rows = ro.run(
            _two_metric_src(_split_spec_shapes()[shape]),
            prior=Normal(mu=0.0, sigma=0.1),
            value_scale={"revenue": "absolute", "signups": "absolute"},
        )
    assert {r.metric for r in rows} == {"revenue", "signups"}
    spans = warning_context(record, "estimation.adjust.prior_absolute_scale_spans_metrics")
    assert spans["n_metrics"] == 2


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_declared_per_metric_priors_escape_the_scale_uniformity_judgment():
    """Mixed scales are fine when each metric declares its OWN prior: the
    refusal exists for ONE scalar read in two unit systems, which cannot
    happen when no scalar spans the metrics. Guards the call-wide judgment
    against over-refusing what per-group `prior_shared=False` allowed."""
    from increment import readouts as ro
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec

    rows = ro.run(
        _two_metric_src(
            [
                MetricSpec(name="revenue", prior=Normal(mu=0.0, sigma=0.01)),
                MetricSpec(name="signups", prior=Normal(mu=0.0, sigma=0.02)),
            ]
        ),
        value_scale={"revenue": "relative", "signups": "absolute"},
    )
    by_metric = {r.metric: r.value_scale for r in rows}
    assert by_metric == {"revenue": "relative", "signups": "absolute"}


@pytest.mark.parametrize(
    ("prior_kind", "value_scale", "code"),
    [
        ("mixture", {"revenue": "relative", "signups": "absolute"}, "readout.observational.prior"),
        ("studentt", {"revenue": "relative", "signups": "absolute"}, "readout.observational.prior"),
        ("normal", {"revenue": "abs", "signups": "absolute"}, "readout.value_scale.invalid"),
    ],
)
def test_entry_refusals_outrank_the_shared_prior_scale_judgment(prior_kind, value_scale, code):
    """A compound-invalid call reports the refusal the caller must act on.

    The scale-uniformity judgment runs call-wide, ahead of the per-group
    dispatch, so it would otherwise shadow the entry checks that used to
    precede it: an unreconstructable prior type is refused whatever the
    scales are, and an unrecognized scale value would be miscounted as a
    distinct scale - reporting a relative/absolute split the request does
    not contain. Both must beat the mixed-scale refusal.
    """
    from increment import readouts as ro
    from increment.estimation.inference import MixturePrior, Normal, StudentTPrior

    priors = {
        "mixture": MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.01, 0.1)),
        "studentt": StudentTPrior(nu=5.0, scale=0.01),
        "normal": Normal(mu=0.0, sigma=0.1),
    }
    with pytest.raises(InvalidRequestError) as exc:
        ro.run(
            _two_metric_src(_split_spec_shapes()["one_group"]),
            prior=priors[prior_kind],
            value_scale=value_scale,
        )
    assert exc.value.code == code


def test_analysis_run_forwards_value_scale():
    """The mapping reaches the readout through the facade: the only
    difference between these two calls is the kwarg, so the scale the rows
    report is what proves it was forwarded rather than dropped."""
    an = Analysis.from_unit_summary(
        _near_zero_table(),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=_OBS_X,
    )
    (default,) = lift_rows(an.run())
    assert default.value_scale == "relative"
    (est,) = lift_rows(an.run(value_scale={"revenue": "absolute"}))
    assert est.value_scale == "absolute"
    assert est.require_lift().value == pytest.approx(0.3, abs=0.15)


def test_value_scale_refused_under_a_randomized_design():
    """The randomized path already reports the absolute axis on every row;
    a second, incompatible way to ask for it would be a silent no-op."""
    an = Analysis.from_unit_summary(
        _confounded_table(200, seed=11),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(UnsupportedRequestError) as exc:
        lift_rows(an.run(value_scale={"revenue": "absolute"}))
    assert exc.value.code == "readout.randomized.value_scale"


def test_value_scale_refused_under_an_encouragement_design():
    """Encouragement's LATE rows are already additive - the mapping never
    travels into that path, so it must be refused by name at run()."""
    from increment import readouts as ro
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import MeanMetric
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(name="revenue", entity="user_id", fact="orders", aggregation="sum")
    design = Encouragement(
        control_group="C",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    src = FakeMomentSource([], metrics=[metric], capabilities={"total"}, design=design)
    with pytest.raises(UnsupportedRequestError) as exc:
        ro.run(src, value_scale={"revenue": "absolute"})
    assert exc.value.code == "readout.encouragement.value_scale"


def test_value_scale_refused_on_the_moments_only_analysis_path():
    """`Analysis.run` also serves definitions/warehouse sources that never
    reach readouts.run - the selector must not be silently dropped there."""
    from increment.semantics.models import AnalysisPlan

    rows = [
        {
            **_moment_fields(),
            "experiment_id": "e",
            "metric": "revenue",
            "group_id": g,
            "n": 10,
            "sum_y": s,
            "sum_y2": q,
        }
        for g, s, q in (("T", 45.0, 240.0), ("C", 30.0, 110.0))
    ]
    an = Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="C", plan=AnalysisPlan())
    with pytest.raises(UnsupportedRequestError) as exc:
        lift_rows(an.run(value_scale={"revenue": "absolute"}))
    assert exc.value.code == "readout.randomized.value_scale"


def _observational_multi_arm_two_secondaries(seed=5):
    """Three arms (one control, two treatment), two secondaries, confounded
    assignment on a numeric covariate -- large enough that neither arm
    collapses to a zero-variance/zero-mean cell."""
    import polars as pl

    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(seed)
    n = 90
    arms = rng.choice(["control", "treat_a", "treat_b"], size=n)
    tenure = rng.normal(100, 15, size=n)
    revenue = 20.0 + 0.05 * tenure + (arms != "control") * 2.0 + rng.normal(0, 1, size=n)
    errors = 1.0 + 0.01 * tenure + (arms != "control") * 0.1 + rng.normal(0, 0.5, size=n)
    df = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": arms,
            "revenue": revenue,
            "errors": errors,
            "tenure": tenure,
        }
    )
    return Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean", "errors": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("tenure",)),
        ),
        plan=AnalysisPlan(primary="revenue", secondaries=["errors"]),
    )


def test_observational_primary_alpha_splits_across_treatment_arms():
    analysis = _observational_multi_arm_two_secondaries()
    rows = lift_rows(analysis.run())
    primary_rows = [r for r in rows if r.metric == "revenue"]
    assert len(primary_rows) == 2  # one row per non-control arm
    for row in primary_rows:
        # Bonferroni split across 2 non-control arms (1 primary metric):
        # nominal 0.05 / (1 primary * 2 arms) = 0.025 two-sided per arm.
        assert row.require_lift().level == pytest.approx(1.0 - 0.025, rel=1e-9)


def test_observational_secondary_family_selects_and_stamps_fcr():
    analysis = _observational_multi_arm_two_secondaries()
    rows = lift_rows(analysis.run())
    secondary_rows = [r for r in rows if r.metric == "errors"]
    assert secondary_rows  # at least the single-secondary family ran
    assert all(row.role == "secondary" for row in secondary_rows)
    # A single-secondary family's realized BH threshold degenerates to q
    # itself -- discovery must be a real bool (selected or not), never None,
    # and family_q must be populated whenever a decision-role row is present.
    decision_rows = [r for r in secondary_rows if r.method_role == "decision"]
    assert decision_rows
    for row in decision_rows:
        assert row.discovery is not None
        from increment.semantics.models import AnalysisPlan

        assert row.family_q == pytest.approx(AnalysisPlan().q)


def test_observational_secondary_family_refuses_on_failed_cell_not_silently_shrunk(monkeypatch):
    """A ratio secondary in a complete evidence family refuses before moments
    are loaded, using the ratio-adjustment refusal rather than a family error."""
    import polars as pl

    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(11)
    n = 60
    arms = rng.choice(["control", "treat"], size=n)
    tenure = rng.normal(100, 15, size=n)
    revenue = 20.0 + 0.05 * tenure + (arms == "treat") * 2.0 + rng.normal(0, 1, size=n)
    spend = 5.0 + 0.01 * tenure + rng.normal(0, 0.5, size=n)
    clicks = rng.poisson(3, size=n) + 1
    df = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": arms,
            "revenue": revenue,
            "spend": spend,
            "clicks": clicks,
            "tenure": tenure,
        }
    )
    analysis = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(name="revenue"),
            MetricSpec(
                name="spend_per_click", type="ratio", numerator="spend", denominator="clicks"
            ),
        ],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
        plan=AnalysisPlan(secondaries=["revenue", "spend_per_click"]),
    )
    from increment.frame import FrameTotalsSource

    def _no_read(*_args, **_kwargs):
        raise AssertionError("data was read before the request was validated")

    for name in ("moments", "unit_frame"):
        monkeypatch.setattr(FrameTotalsSource, name, _no_read)
    with pytest.raises(UnsupportedRequestError) as exc_info:
        lift_rows(analysis.run())
    error = exc_info.value
    assert error.code == "estimation.adjust_common.supported_ratio_metric"
    assert error.context["metric"] == "spend_per_click"
    assert error.context["method"] == "iptw"
    assert error.context["role"] == "secondary"
    assert error.context["family"] == "secondary"
    assert error.context["correction"] == "bh"


@pytest.mark.parametrize("method", ["iptw", "unadjusted"])
@pytest.mark.parametrize("reverse_metrics", [False, True])
def test_clearing_bound_ratio_prior_revalidates_observational_family(method, reverse_metrics):
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    tenure = np.tile(np.linspace(-1, 1, 30), 2)
    treated = np.repeat([0, 1], 30)
    frame = pa.table(
        {
            "user_id": range(60),
            "variant": np.where(treated, "T", "C"),
            "tenure": tenure,
            "revenue": 20 + 2 * tenure + 2 * treated,
            "spend": 5 + tenure / 2 + 3 * treated,
            "clicks": np.tile(np.arange(30) % 3 + 1, 2),
        }
    )

    def build(bound):
        metrics = [
            MetricSpec(name="revenue"),
            MetricSpec(
                name="spend_per_click",
                type="ratio",
                numerator="spend",
                denominator="clicks",
                prior=Normal(mu=0.0, sigma=0.1) if bound else None,
            ),
        ]
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            metrics=metrics[::-1] if reverse_metrics else metrics,
            design=Observational(
                control_group="C", adjustment=AdjustmentSet(covariates=("tenure",))
            ),
            plan=AnalysisPlan(primary="revenue", secondaries=["spend_per_click"]),
        )

    bound = build(True)
    prior_free = build(False) if method == "unadjusted" else None
    decision = Method(name=method)
    inherited_values = {}
    try:
        if method == "iptw":
            for kwargs in ({}, {"prior": None}):
                with pytest.raises(UnsupportedRequestError) as raised:
                    bound.run(decision_method=decision, **kwargs)
                assert raised.value.code == "estimation.adjust_common.supported_ratio_metric"
                assert raised.value.context["metric"] == "spend_per_click"
                assert raised.value.context["method"] == "iptw"
                assert raised.value.context["family"] == "secondary"
                assert raised.value.context["correction"] == "bh"
            return
        expected = (
            {
                row.metric: row
                for row in lift_rows(prior_free.run(decision_method=decision))
                if row.method_role == "decision" and row.sampling_available is True
            }
            if prior_free is not None
            else {}
        )
        if expected:
            assert expected["spend_per_click"].discovery is True
            assert expected["spend_per_click"].family_axes == ("metric", "arm")
            assert expected["spend_per_click"].require_lift().value == pytest.approx(0.6)
        for clear in (False, True, True, False):
            kwargs = {"prior": None} if clear else {}
            rows = lift_rows(bound.run(decision_method=decision, **kwargs))
            rows = [
                row
                for row in rows
                if row.method_role == "decision" and row.sampling_available is True
            ]
            assert {row.metric for row in rows} == {"revenue", "spend_per_click"}
            for row in rows:
                interval = row.require_lift()
                values = (interval.value, interval.lb, interval.ub)
                if clear:
                    oracle = expected[row.metric]
                    target = oracle.require_lift()
                    assert values == pytest.approx((target.value, target.lb, target.ub))
                    assert row.discovery == oracle.discovery
                    assert row.family_axes == oracle.family_axes
                else:
                    if row.metric in inherited_values:
                        assert values == pytest.approx(inherited_values[row.metric])
                    else:
                        inherited_values[row.metric] = values
                    if expected:
                        oracle = expected[row.metric]
                        assert row.discovery == oracle.discovery
                        assert row.family_axes == oracle.family_axes
                        assert row.family_size == oracle.family_size
                        if row.metric == "spend_per_click":
                            assert row.posterior_estimate is not None
    finally:
        bound.close()
        if prior_free is not None:
            prior_free.close()


def test_observational_secondary_ratio_refuses_early_on_definitions_path(seeded_defs, seeded_con):
    """Native definitions validation rejects before the warehouse readout."""
    from increment.semantics.models import AnalysisPlan

    base = Analysis.from_definitions("pricing_tier_test", seeded_defs, seeded_con, store="none")
    ratio = next(metric for metric in base.metrics if metric.name == "revenue_per_session")
    mean = next(metric for metric in base.metrics if metric.name == "avg_session_duration")
    analysis = make_analysis_like(
        base,
        metrics=[mean, ratio],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("revenue",))
        ),
        plan=AnalysisPlan(primary="avg_session_duration", secondaries=["revenue_per_session"]),
    )
    with pytest.raises(UnsupportedRequestError) as exc_info:
        analysis.run()
    error = exc_info.value
    assert error.code == "estimation.adjust_common.supported_ratio_metric"
    assert error.context["metric"] == "revenue_per_session"
    assert error.context["method"] == "iptw"


def test_all_ratio_secondary_family_refusal_names_the_family():
    """When every metric is an unsupported ratio, an in-family one still
    reports its family, the same context a mixed request gets."""
    import polars as pl

    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(12)
    n = 60
    df = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": rng.choice(["control", "treat"], size=n),
            "spend": rng.normal(5.0, 0.5, size=n),
            "revenue": rng.normal(20.0, 1.0, size=n),
            "clicks": rng.poisson(3, size=n) + 1,
            "tenure": rng.normal(100, 15, size=n),
        }
    )
    analysis = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(
                name="spend_per_click", type="ratio", numerator="spend", denominator="clicks"
            ),
            MetricSpec(
                name="revenue_per_click", type="ratio", numerator="revenue", denominator="clicks"
            ),
        ],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
        plan=AnalysisPlan(secondaries=["spend_per_click", "revenue_per_click"]),
    )
    with pytest.raises(UnsupportedRequestError) as exc_info:
        analysis.run()
    context = exc_info.value.context
    assert exc_info.value.code == "estimation.adjust_common.supported_ratio_metric"
    assert (context["role"], context["family"], context["correction"]) == (
        "secondary",
        "secondary",
        "bh",
    )


def test_observational_guardrail_keeps_full_alpha_outside_any_family():
    import polars as pl

    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(9)
    n = 60
    arms = rng.choice(["control", "treat"], size=n)
    tenure = rng.normal(100, 15, size=n)
    revenue = 20.0 + 0.05 * tenure + (arms == "treat") * 2.0 + rng.normal(0, 1, size=n)
    errors = 1.0 + 0.01 * tenure + rng.normal(0, 0.5, size=n)
    df = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": arms,
            "revenue": revenue,
            "errors": errors,
            "tenure": tenure,
        }
    )
    analysis = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(name="revenue"),
            MetricSpec(name="errors", preferred_direction="decrease"),
        ],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
        plan=AnalysisPlan(primary="revenue", guardrails=["errors"]),
    )
    rows = lift_rows(analysis.run())
    guardrail_row = next(r for r in rows if r.metric == "errors")
    assert guardrail_row.role == "guardrail"
    assert guardrail_row.alternative == "less"
    # One-sided at full nominal alpha=0.05 displays through the standard
    # alpha-doubling convention (AnalysisPlan's own docstring): reported
    # alpha=0.1, level=0.9 -- not the two-sided level=0.95 a naive read
    # of "full alpha" would suggest.
    assert guardrail_row.require_lift().level == pytest.approx(0.9, rel=1e-9)
    assert guardrail_row.require_lift().alpha == pytest.approx(0.1, rel=1e-9)
    assert guardrail_row.family_q is None
    assert guardrail_row.discovery is None


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_three_arm_aipw_reaches_the_common_population_from_definitions_and_frames(tmp_path):
    """The same three-arm per-unit data through the warehouse definitions path
    and the dataframe oracle: both rows reach the common-population ATEs
    (3, 4) over one control mean 4 (relative 0.75, 1.0), not pair-population
    reanalyses of each treatment."""
    import ibis

    from tests.covariate_cases import _defs_yaml, _unit_rows
    from tests.estimation.test_adjust import _CellMean, _three_arm_table

    table = _three_arm_table()
    labels = {"C": "control", "T1": "t1", "T2": "t2"}
    units = [
        (unit, labels[arm], x, y)
        for unit, arm, x, y in zip(
            table["user_id"].to_pylist(),
            table["variant"].to_pylist(),
            table["x"].to_pylist(),
            table["revenue"].to_pylist(),
            strict=True,
        )
    ]
    con = ibis.duckdb.connect()
    con.create_table(
        "tri_events",
        obj=[row for unit in units for row in _unit_rows("tri_obs", *unit)],
    )
    defs_path = tmp_path / "tri_defs.yaml"
    defs_path.write_text(_defs_yaml("tri_events", "tri_obs", observational=True))
    frame = pa.table(
        {
            "user_id": [unit for unit, *_ in units],
            "variant": [arm for _, arm, _, _ in units],
            "revenue": [y for *_, y in units],
            "tenure": [x for _, _, x, _ in units],
        }
    )
    method = Method(name="aipw", propensity_learner=_CellMean, outcome_learner=_CellMean, folds=5)
    warehouse = lift_rows(
        Analysis.from_definitions("tri_obs", defs_path, con).run(decision_method=method)
    )
    oracle = lift_rows(
        Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
            ),
        ).run(decision_method=method)
    )
    for rows in (warehouse, oracle):
        by_arm = {row.group_id: row for row in rows if row.metric == "revenue"}
        assert set(by_arm) == {"t1", "t2"}
        for arm, (tau, lift) in {"t1": (3.0, 0.75), "t2": (4.0, 1.0)}.items():
            row = by_arm[arm]
            assert row.require_lift().value == pytest.approx(lift, rel=1e-12)
            assert row.abs_diff == pytest.approx(tau, rel=1e-12)
            assert row.relative_confidence_set is not None
            assert row.relative_confidence_set.reference.c == pytest.approx(4.0, rel=1e-12)


# Categorical adjustment columns declared on the ordinary AdjustmentSet: the
# unit-summary and unit-panel constructors run every adjusted method on a
# raw string column and reproduce hand-built modal-reference dummies.


def _categorical_analysis(constructor: str, *, dummies: bool):
    import datetime as dt

    from tests.categorical_cases import DUMMY_COLUMNS, categorical_units, dummy_table, raw_table

    units = categorical_units(600, seed=3)
    table = dummy_table(units) if dummies else raw_table(units)
    covariates = ("spend", *DUMMY_COLUMNS) if dummies else ("spend", "region")
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=covariates),
        gate=IdentificationGate(overlap="trim"),
    )
    if constructor == "from_unit_summary":
        return Analysis.from_unit_summary(
            table, unit="user_id", group="variant", metrics={"revenue": "mean"}, design=design
        )
    panel = table.append_column("date", pa.array([dt.date(2025, 1, 10)] * table.num_rows))
    return Analysis.from_unit_panel(
        panel,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        design=design,
    )


@pytest.mark.parametrize("constructor", ["from_unit_summary", "from_unit_panel"])
@pytest.mark.parametrize("method", ["iptw", "aipw", "dml"])
def test_categorical_adjustment_readout_matches_dummy_oracle(constructor, method):
    import warnings

    from tests.categorical_cases import assert_rows_match

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oracle = lift_rows(
            _categorical_analysis(constructor, dummies=True).run(
                decision_method=Method(name=method)
            )
        )
        actual = lift_rows(
            _categorical_analysis(constructor, dummies=False).run(
                decision_method=Method(name=method)
            )
        )
    assert [row.method for row in actual] == [method]
    assert_rows_match([row.model_dump() for row in oracle], [row.model_dump() for row in actual])


# The same contract on the definitions-backed paths: a declared string
# property is a categorical covariate on the warehouse read and on the
# published artifact, with a NULL level carried as a missing value.


def _categorical_oracle_table(units: dict[str, dict], *, dummies: bool) -> pa.Table:
    """The per-unit truth as a frame: the raw ``region`` strings (None where
    missing) or the modal-reference dummies a careful user hand-builds."""
    from tests.covariate_cases import CATEGORICAL_LEVELS

    columns: dict[str, list] = {
        "user_id": list(units),
        "variant": [truth["variant"] for truth in units.values()],
        "revenue": [truth["revenue"] for truth in units.values()],
        "tenure": [truth["tenure"] for truth in units.values()],
    }
    if dummies:
        for level in CATEGORICAL_LEVELS[1:]:
            columns[f"region_{level}"] = [
                float(truth["region"] == level) for truth in units.values()
            ]
    else:
        columns["region"] = [truth["region"] for truth in units.values()]
    return pa.table(columns)


def _categorical_oracle(
    units: dict[str, dict],
    *,
    dummies: bool,
    missing: Literal["refuse", "impute-indicator", "pattern", "complete-case", "allow"] = "refuse",
):
    """The dataframe oracle under the same plan the definitions fixture
    declares (`revenue` primary), so every emitted field is comparable."""
    from increment.semantics.models import AnalysisPlan

    covariates = ("tenure", "region_west", "region_north") if dummies else ("tenure", "region")
    return Analysis.from_unit_summary(
        _categorical_oracle_table(units, dummies=dummies),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=covariates, missing=missing),
        ),
        plan=AnalysisPlan(primary="revenue"),
    )


def _adopted_categorical_artifact(defs_path, con) -> Analysis:
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics import load

    defs = load(defs_path)
    experiment = defs.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = Analysis.from_definitions("cat_test", defs_path, con).publish_unit_day_artifact(store)
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
@pytest.mark.parametrize("constructor", ["from_definitions", "from_unit_day_artifact"])
@pytest.mark.parametrize("method", ["iptw", "aipw", "dml"])
def test_categorical_adjustment_on_definitions_paths_matches_the_dummy_oracle(
    tmp_path, constructor, method
):
    """The warehouse read joins the declared string property as a string
    column and the artifact serves it from its level relation; both
    reproduce, for every adjusted method, the dataframe oracle a careful
    user builds from modal-reference dummies -- the same rows, estimates,
    intervals and diagnostics, with no encoding declared anywhere."""
    from tests.categorical_cases import assert_rows_match
    from tests.covariate_cases import categorical_defs_and_con
    from tests.parity_harness.runner import _normalize

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    analysis = (
        Analysis.from_definitions("cat_test", defs_path, con)
        if constructor == "from_definitions"
        else _adopted_categorical_artifact(defs_path, con)
    )
    oracle = _normalize(
        _categorical_oracle(units, dummies=True).run(decision_method=Method(name=method))
    )
    actual = _normalize(analysis.run(decision_method=Method(name=method)))
    assert {method_name for methods in actual.values() for method_name in methods} == {method}
    assert_rows_match(oracle, actual)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_categorical_null_level_in_warehouse_source_matches_frame_when_imputed(tmp_path):
    """The internal warehouse source accepts an imputation design and keeps
    the same full-cohort ATE as the frame. Public definitions constructors
    instead use the declared default missing-value refusal."""
    from increment.semantics import load
    from tests.analysis_factory import make_analysis
    from tests.categorical_cases import assert_rows_match
    from tests.covariate_cases import categorical_defs_and_con
    from tests.parity_harness.runner import _normalize

    null_units = tuple(f"u{i}" for i in range(12))
    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=null_units)
    design = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("tenure", "region"), missing="impute-indicator"),
    )
    warehouse = make_analysis(con, load(defs_path), experiment="cat_test", _design=design)
    oracle = _categorical_oracle(units, dummies=False, missing="impute-indicator")
    oracle_rows = lift_rows(oracle.run())
    warehouse_rows = lift_rows(warehouse.run())
    assert [row.estimand for row in warehouse_rows] == ["ate"]
    assert_rows_match(_normalize(oracle_rows), _normalize(warehouse_rows))


@pytest.mark.parametrize("constructor", ["from_definitions", "from_unit_day_artifact"])
def test_categorical_null_level_refuses_by_name_on_definitions_paths(tmp_path, constructor):
    """Under the default ``missing="refuse"`` policy a NULL level refuses by
    the identification code naming the column and the exact count of
    missing units -- on the warehouse read and on the published artifact
    alike, so neither wire turned the NULL into a level or dropped the unit."""
    from increment.errors import CodedError
    from tests.covariate_cases import categorical_defs_and_con

    null_units = ("u3", "u10", "u21")
    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=null_units)
    analysis = (
        Analysis.from_definitions("cat_test", defs_path, con)
        if constructor == "from_definitions"
        else _adopted_categorical_artifact(defs_path, con)
    )
    with pytest.raises(CodedError) as raised:
        analysis.run()
    assert raised.value.code == "adjust.identification.missing_covariates"
    assert raised.value.context["missing_covariates"] == (("region", len(null_units)),)
    assert raised.value.context["n"] == len(units)


def _scalar_mean_cube_rows():
    """Real scalar moments: a randomized unit-summary mean over the quantile's outcome column.

    The arms are ``"C"``/``"T"`` so the one cube is a valid source for both the
    observational design (``_OBS``, control group ``"C"``) and a randomized read."""
    import tempfile
    from pathlib import Path

    import pyarrow.parquet as pq

    from increment.frame import MetricSpec

    n = 80
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": ["C" if i % 2 == 0 else "T" for i in range(n)],
            "latency": [1.0 + (i % 7) * 0.3 + (0.2 if i % 2 else 0.0) for i in range(n)],
        }
    )
    producer = Analysis.from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(
                name="lat", type="mean", value_column="latency", preferred_direction="decrease"
            )
        ],
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moments.parquet"
            producer.export(path)
            return pq.read_table(path).to_pylist()
    finally:
        producer.close()


@pytest.mark.parametrize(
    ("design", "request_kind", "stage", "code"),
    [
        (_OBS, "two-sided", "run", "readout.observational.quantile"),
        (_OBS, "alternative", "run", "readout.metric.quantile_alternative"),
        (_OBS, "guardrail", "run", "readout.metric.quantile_alternative"),
        (_OBS, "margin_abs", "run", "readout.metric.quantile_alternative"),
        (_OBS, "margin", "construct", "plan.observational.relative_margin"),
        (None, "two-sided", "run", "source.moments.unit_grain"),
        (None, "alternative", "run", "readout.metric.quantile_alternative"),
        (None, "guardrail", "run", "readout.metric.quantile_alternative"),
        (None, "margin_abs", "run", "readout.metric.quantile_alternative"),
        (None, "margin", "run", "readout.metric.quantile_alternative"),
    ],
    ids=[
        "observational-two-sided",
        "observational-one-sided-alternative",
        "observational-marginless-guardrail",
        "observational-absolute-margin",
        "observational-relative-margin",
        "randomized-two-sided",
        "randomized-one-sided-alternative",
        "randomized-marginless-guardrail",
        "randomized-absolute-margin",
        "randomized-relative-margin",
    ],
)
def test_quantile_over_scalar_moments_refuses_in_the_documented_precedence(
    design, request_kind, stage, code
):
    """A quantile declared over a real scalar-moments cube is refused by whichever gate the
    request reaches first: the plan's relative-margin construction guard (observational), the
    engine's one-sided/shifted-null check (any one-sided request: a standalone alternative, a
    marginless guardrail's adverse tail, or a margin), the observational estimator seam, then
    the cube's missing unit grain."""
    from increment.errors import CodedError
    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    rows = _scalar_mean_cube_rows()
    plan = {
        "two-sided": None,
        "alternative": AnalysisPlan(alternative="greater", primary="lat"),
        "guardrail": AnalysisPlan(guardrails=[ExperimentMetric(metric="lat")]),
        "margin_abs": AnalysisPlan(guardrails=[ExperimentMetric(metric="lat", margin_abs=0.5)]),
        "margin": AnalysisPlan(guardrails=[ExperimentMetric(metric="lat", margin=0.02)]),
    }[request_kind]
    spec = [MetricSpec(name="lat", type="quantile", quantile=0.9, preferred_direction="decrease")]

    def construct():
        if design is not None:
            return Analysis.from_moments(rows, metrics=spec, design=design, plan=plan)
        return Analysis.from_moments(rows, metrics=spec, control="C", plan=plan)

    if stage == "construct":
        with pytest.raises(CodedError) as raised:
            construct()
    else:
        analysis = construct()
        with pytest.raises(CodedError) as raised:
            analysis.run()
    assert raised.value.code == code


class _DriftingMoments:
    """A source whose arm inventory changes after each metric's first moments read, as a
    warehouse does when a late arm lands between a readout's reads. Every other attribute is
    the late source's."""

    def __init__(self, early, late):
        self._late = late
        self._early_moments = early.moments
        self._late_moments = late.moments
        self.reads: dict[str, int] = {}

    def moments(self, metric, **options):
        self.reads[metric.name] = self.reads.get(metric.name, 0) + 1
        moments = self._early_moments if self.reads[metric.name] == 1 else self._late_moments
        return moments(metric, **options)

    def __getattr__(self, name):
        return getattr(self._late, name)


def _drifting_analysis(early, late):
    source = _DriftingMoments(_moment_source(early), _moment_source(late))
    # Capture both original reductions before substituting the drifting read behavior.
    cast("Any", _moment_source(late)).moments = source.moments
    return late, source


_UNITS_PER_ARM = 12_000
# Per-arm successes: with ``q = 0.1`` a conversion cell is dense (``auto`` routes it asymptotic)
# at its family level only above 2140 of each count for two hypotheses, 3660 for four.
_SUCCESSES = {"control": 4800, "T2": 3000}


def _conversion_family_analysis(arms, t1_successes):
    import polars as pl

    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    successes = {**_SUCCESSES, "T1": t1_successes}
    rng = np.random.default_rng(3)
    frames = []
    for arm in arms:
        y = np.r_[np.ones(successes[arm]), np.zeros(_UNITS_PER_ARM - successes[arm])]
        rng.shuffle(y)
        frames.append(
            pl.DataFrame(
                {
                    "user_id": [f"{arm}-{i}" for i in range(_UNITS_PER_ARM)],
                    "variant": arm,
                    "a": y,
                    "b": y[::-1].copy(),
                    "x": rng.normal(size=_UNITS_PER_ARM),
                }
            )
        )
    return Analysis.from_unit_summary(
        pl.concat(frames),
        unit="user_id",
        group="variant",
        metrics=[MetricSpec(name="a", type="conversion"), MetricSpec(name="b", type="conversion")],
        design=Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x",))),
        plan=AnalysisPlan(secondaries=["a", "b"]),
    )


@pytest.mark.parametrize("t1_successes", [4800, 3800], ids=["no_selection", "fcr_reinterval"])
def test_observational_conversion_family_routes_from_the_evidence_it_estimates(t1_successes):
    """A late arm between reads must not enter the estimates of a family sized without it: the
    routing level ``q / m``, the BH family and every returned row (including an FCR reinterval)
    come from the one moments read each metric was sized from."""
    from increment.estimation.conversion_route import dense_min_count, family_route_alpha

    early = _conversion_family_analysis(["control", "T1"], t1_successes)
    late = _conversion_family_analysis(["control", "T1", "T2"], t1_successes)
    drifting, source = _drifting_analysis(early, late)
    try:
        rows = list(lift_rows(drifting.run(decision_method=Method(name="unadjusted"))))
    finally:
        for analysis in (early, late):
            analysis.close()

    assert {(row.metric, row.group_id) for row in rows} == {("a", "T1"), ("b", "T1")}
    assert source.reads == {"a": 1, "b": 1}
    if t1_successes == 3800:
        assert all(row.discovery for row in rows)
    for row in rows:
        if row.discovery:
            continue
        # An unselected cell is decided at its family's smallest level ``q / m``; its route is
        # the one valid there.
        assert row.family_q is not None
        level = family_route_alpha(row.family_q, len(rows))
        assert level is not None
        dense = min(t1_successes, _UNITS_PER_ARM - t1_successes, 4800, 7200) >= dense_min_count(
            level / 2
        )
        assert row.reference_kind == ("t" if dense else "binomial")


def _mean_family_analysis(arms):
    import polars as pl

    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(17)
    per_arm = 150
    effects = {"control": 0.0, "T1": 2.0, "T2": 3.0}
    frames = []
    for arm in arms:
        x = rng.normal(size=per_arm)
        frames.append(
            pl.DataFrame(
                {
                    "user_id": [f"{arm}-{i}" for i in range(per_arm)],
                    "variant": arm,
                    "a": 1.0 + 0.5 * x + effects[arm] + rng.normal(scale=0.5, size=per_arm),
                    "b": 2.0 - 0.4 * x + effects[arm] + rng.normal(scale=0.5, size=per_arm),
                    "x": x,
                }
            )
        )
    return Analysis.from_unit_summary(
        pl.concat(frames),
        unit="user_id",
        group="variant",
        metrics={"a": "mean", "b": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("x",)),
            gate=IdentificationGate(overlap="trim"),
        ),
        plan=AnalysisPlan(secondaries=["a", "b"]),
    )


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
@pytest.mark.parametrize("method", ["unadjusted", "iptw", "aipw", "dml"])
def test_observational_adjusted_family_estimates_only_the_inventory_it_selected_from(method):
    early = _mean_family_analysis(["control", "T1"])
    late = _mean_family_analysis(["control", "T1", "T2"])
    drifting, source = _drifting_analysis(early, late)
    try:
        rows = list(lift_rows(drifting.run(decision_method=Method(name=method))))
    finally:
        for analysis in (early, late):
            analysis.close()

    assert {(row.metric, row.group_id) for row in rows} == {("a", "T1"), ("b", "T1")}
    assert source.reads == {"a": 1, "b": 1}
    # Both T1 cells are selected, so every row is the FCR reinterval of the held evidence.
    assert all(row.discovery for row in rows)


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_estimate_ate_reads_each_metrics_moments_once_across_methods():
    from increment.estimation.adjust import estimate_ate

    early = _mean_family_analysis(["control", "T1"])
    late = _mean_family_analysis(["control", "T1", "T2"])
    drifting, source = _drifting_analysis(early, late)
    try:
        context = _moment_source(late).context
        computation = estimate_ate(
            cast("MomentSource", source),
            cast("Observational", context.design),
            methods=[Method(name="unadjusted"), Method(name="iptw")],
            metrics=list(context.metrics),
        )
    finally:
        for analysis in (early, late):
            analysis.close()

    assert {(row.metric, row.method, row.group_id) for row in computation.results} == {
        (metric, method, "T1") for metric in ("a", "b") for method in ("unadjusted", "iptw")
    }
    assert source.reads == {"a": 1, "b": 1}


def _snapshot_units(tag: str, arm: str, n: int, seed: int) -> list[dict[str, object]]:
    from tests.covariate_cases import _unit_rows

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        tenure = float(rng.normal(100, 15))
        revenue = 20.0 + 0.05 * tenure + (2.0 if arm != "control" else 0.0) + float(rng.normal())
        rows.extend(_unit_rows("snap_obs", f"{tag}-{i}", arm, tenure, revenue))
    return rows


def _snapshot_analysis(tmp_path, name):
    import ibis

    from tests.covariate_cases import _defs_yaml

    con = ibis.duckdb.connect()
    rows = _snapshot_units("c", "control", 40, 1) + _snapshot_units("t", "treatment", 40, 2)
    con.create_table("snap_events", obj=rows)
    defs_path = tmp_path / f"{name}.yaml"
    defs_path.write_text(_defs_yaml("snap_events", "snap_obs", observational=True))
    return con, Analysis.from_definitions("snap_obs", defs_path, con)


@pytest.mark.parametrize("method", ["iptw", "unadjusted"])
def test_native_observational_readout_is_one_snapshot_when_arms_land_mid_readout(
    tmp_path, monkeypatch, method
):
    """A warehouse that gains a new arm after each moments read still yields exactly the rows,
    family stamps and intervals of the warehouse as it stood at the first read, while a later
    unpinned readout does see the arms: moments, family size and every estimate share one
    execution."""
    from increment.query.native_source import DefinitionsMomentSource

    clean_con, clean = _snapshot_analysis(tmp_path, "clean")
    live_con, live = _snapshot_analysis(tmp_path, "live")
    try:
        decision = Method(name=method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            expected = list(lift_rows(clean.run(decision_method=decision)))

            original = DefinitionsMomentSource.moments
            landed: list[str] = []

            def moments_then_land_an_arm(self, metric, **options):
                rows = original(self, metric, **options)
                arm = f"late{len(landed)}"
                landed.append(arm)
                live_con.insert("snap_events", _snapshot_units(arm, arm, 40, 10 + len(landed)))
                return rows

            monkeypatch.setattr(DefinitionsMomentSource, "moments", moments_then_land_an_arm)
            actual = list(lift_rows(live.run(decision_method=decision)))
            monkeypatch.setattr(DefinitionsMomentSource, "moments", original)
            fresh = list(lift_rows(live.run(decision_method=decision)))

        assert landed, "the warehouse never changed, so nothing was pinned against"
        assert {row.group_id for row in actual} == {"treatment"}
        assert {row.group_id for row in fresh} > {"treatment"}
        assert len(actual) == len(expected) == 1
        for got, want in zip(actual, expected, strict=True):
            # Two physical scans sum the same units in warehouse order: equal to rounding.
            for field in ("value", "lb", "ub"):
                assert getattr(got.require_lift(), field) == pytest.approx(
                    getattr(want.require_lift(), field), rel=1e-9
                )
            assert got.reference_kind == want.reference_kind
            assert (got.discovery, got.family_q, got.family_threshold) == (
                want.discovery,
                want.family_q,
                want.family_threshold,
            )
    finally:
        clean.close()
        live.close()
        clean_con.disconnect()
        live_con.disconnect()


def test_native_observational_empty_selection_touches_no_source(tmp_path, monkeypatch):
    """An empty metric selection owns no evidence: it neither opens a snapshot (a definitions
    capture writes TEMP tables) nor reads moments or unit frames, through the facade and
    through the readout entry points that skip the facade's early return. A nonempty selection
    still reads inside one source-owned snapshot."""
    from increment import readouts
    from increment.query.native_source import DefinitionsMomentSource

    con, analysis = _snapshot_analysis(tmp_path, "empty")
    touched: list[str] = []

    def spy(name):
        original = getattr(DefinitionsMomentSource, name)

        def wrapped(self, *args, **kwargs):
            touched.append(name)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(DefinitionsMomentSource, name, wrapped)

    try:
        for name in ("readout_snapshot", "_pinned_source_execution", "moments", "unit_frame"):
            spy(name)
        source = _moment_source(analysis)
        assert isinstance(source.context.design, Observational)

        assert list(analysis.run(metrics=[])) == []
        assert readouts.run(source, metrics=[]) == []
        assert readouts.arm_moments(source, metrics=[]) == []
        assert touched == []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            rows = list(lift_rows(analysis.run()))
        assert len(rows) == 1
        assert touched.count("readout_snapshot") == 1
        assert "moments" in touched
    finally:
        analysis.close()
        con.disconnect()
