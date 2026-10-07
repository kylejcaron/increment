"""Tests for `readouts.run`'s plan-based rewire: role stamping, per-arm
alpha splitting, and the secondary family's BH / e-BH / FCR machinery.

Every scenario is deterministic (seeded numpy data) and frame-backed via
`FrameTotalsSource`, except `test_prior_secondary_outside_family` and
`test_declared_plan_leaves_an_unnamed_metric_unassigned`: a prior-bound
plan entry is refused on the frame path (methods=/prior= overrides are
frame-path-only refusals - see the decision compiler), and an unnamed metric
defaults to `role="secondary"` there rather than `"unassigned"` - both
cases are built over `increment.sources.MomentsSource` with an explicit
`path="warehouse"` instead (`MomentsSource`'s own default is
`path="frame"`, matching `Analysis.from_moments`'s contract), the only
path that can express either.
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from increment import Analysis, readouts
from increment.compatibility import _conservative_ratio
from increment.decision import ArmHypothesisKey, FixedInference
from increment.errors import (
    CapabilityError,
    IncrementRuntimeWarning,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.family import bh_select, e_bh_select
from increment.estimation.inference import Normal
from increment.estimation.sequential import AlwaysValid
from increment.frame import FrameTotalsSource, MetricSpec
from increment.readouts import _passes
from increment.semantics.design import (
    Encouragement,
    ExclusionRestriction,
    Randomized,
    UptakeSpec,
)
from increment.semantics.models import (
    AnalysisPlan,
    ExperimentMetric,
    MeanMetric,
    Metric,
    NormalPriorSpec,
)
from increment.sources import MomentsSource
from tests.sequential_cases import registration
from tests.test_sequential_public_sources import gaussian_plan
from tests.warning_codes import warning_codes


def _normal_column(mean: float, sd: float, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def _cuped_family_table(*, uptake: bool = False):
    n = 600
    unit = np.linspace(-1.0, 1.0, n)
    baseline = np.tile(unit, 2)
    assigned = np.concatenate([np.zeros(n), np.ones(n)])
    columns = {
        "user_id": [f"u{i}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "baseline": baseline,
        # Reuse each noise vector in both arms so the null family members
        # are exactly null while retaining nonzero within-arm variance.
        "m_a": 10.0
        + 0.05 * assigned
        + 0.8 * baseline
        + 0.15 * np.tile(np.sin(np.arange(n) / 7.0), 2),
        "m_b": 10.0 + 0.8 * baseline + 0.15 * np.tile(np.cos(np.arange(n) / 11.0), 2),
        "m_c": 10.0 + 0.8 * baseline + 0.15 * np.tile(np.sin(np.arange(n) / 13.0), 2),
        "m_d": 10.0 + 0.8 * baseline + 0.15 * np.tile(np.cos(np.arange(n) / 17.0), 2),
    }
    if uptake:
        columns["clicked"] = assigned
    return pa.table(columns)


def _cuped_family_specs() -> list[MetricSpec]:
    cuped = Method(name="cuped", variance_reduction="cuped")
    unadjusted = Method(name="unadjusted")
    return [
        MetricSpec(
            name=name,
            covariate="baseline",
            decision_method=cuped,
            sensitivity_methods=(unadjusted,),
        )
        for name in ("m_a", "m_b", "m_c", "m_d")
    ]


def _cuped_family_source():
    return FrameTotalsSource.from_frame(
        _cuped_family_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=_cuped_family_specs(),
        plan=AnalysisPlan(secondaries=["m_a", "m_b", "m_c", "m_d"]),
    )


def _cuped_encouragement_source():
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="assignment only changes outcomes through uptake",
        ),
        min_first_stage_z=0.5,
    )
    return FrameTotalsSource.from_frame(
        _cuped_family_table(uptake=True),
        unit="user_id",
        group="variant",
        control="control",
        metrics=_cuped_family_specs(),
        uptake="clicked",
        design=design,
        plan=AnalysisPlan(secondaries=["m_a", "m_b", "m_c", "m_d"]),
    )


def _cuped_asof_source():
    from tests.test_readouts_encouragement import FakeMomentSource

    n = 600
    unit = np.linspace(-1.0, 1.0, n)
    baseline = np.tile(unit, 2)
    assigned = np.concatenate([np.zeros(n), np.ones(n)])
    noise = (
        np.tile(np.sin(np.arange(n) / 7.0), 2),
        np.tile(np.cos(np.arange(n) / 11.0), 2),
        np.tile(np.sin(np.arange(n) / 13.0), 2),
        np.tile(np.cos(np.arange(n) / 17.0), 2),
    )
    metrics = tuple(
        MeanMetric(name=name, entity="user_id", fact=name, aggregation="sum")
        for name in ("m_a", "m_b", "m_c", "m_d")
    )
    rows = []
    for ds in (date(2025, 1, 1), date(2025, 1, 2)):
        for group, offset in (("control", 0), ("treatment", n)):
            for index, name in enumerate(("m_a", "m_b", "m_c", "m_d")):
                values = 10.0 + 0.05 * assigned + 0.8 * baseline + 0.15 * noise[index]
                values = values[offset : offset + n]
                x_values = baseline[offset : offset + n]
                rows.append(
                    {
                        **centered_row_from_raw_sums(
                            {
                                "experiment_id": "asof",
                                "metric": name,
                                "group_id": group,
                                "n": n,
                                "sum_y": float(values.sum()),
                                "sum_y2": float((values**2).sum()),
                                "sum_x": float(x_values.sum()),
                                "sum_x2": float((x_values**2).sum()),
                                "sum_xy": float((x_values * values).sum()),
                            }
                        ),
                        "ds": ds,
                    }
                )
    plan = AnalysisPlan(
        secondaries=["m_a", "m_b", "m_c", "m_d"],
    )
    return FakeMomentSource(
        rows,
        metrics=metrics,
        capabilities={"asof"},
        design=Randomized(control_group="control"),
        plan=plan,
    )


def test_primary_interval_at_split_alpha_two_arms():
    """1 primary, 2 treatment arms: cell alpha = 0.05/2 -> level 0.975 on
    each row (alpha_share=plan.alpha/n_primaries=0.05/1, split again by
    n_arms=2 at estimation time)."""
    n = 800
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(3 * n)],
            "variant": ["control"] * n + ["treatment_a"] * n + ["treatment_b"] * n,
            "revenue": np.concatenate(
                [
                    _normal_column(10.0, 2.0, n, 21),
                    _normal_column(11.0, 2.0, n, 22),
                    _normal_column(10.5, 2.0, n, 23),
                ]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        plan=AnalysisPlan(primary="revenue"),
    )
    test = src.context.plan.procedures["revenue"]
    assert test.role == "primary"
    assert test.alpha == pytest.approx(0.05)

    results = readouts.run(src)
    assert {r.group_id for r in results} == {"treatment_a", "treatment_b"}
    for r in results:
        assert r.role == "primary"
        assert r.require_lift().level == pytest.approx(0.975)


def test_primary_interval_at_split_alpha_two_primaries_two_arms():
    """2 declared primaries, 2 treatment arms each: cell alpha must equal
    `_conservative_divide(plan.alpha, n_primaries * n_arms)` -- the split is
    across every confirmatory primary at once, then again by this metric's
    own arms, in a single division (0.05 / (2 primaries * 2 arms) = 0.0125,
    not 0.05/2 halved twice)."""
    n = 800
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(3 * n)],
            "variant": ["control"] * n + ["treatment_a"] * n + ["treatment_b"] * n,
            "revenue": np.concatenate(
                [
                    _normal_column(10.0, 2.0, n, 41),
                    _normal_column(11.0, 2.0, n, 42),
                    _normal_column(10.5, 2.0, n, 43),
                ]
            ),
            "clicks": np.concatenate(
                [
                    _normal_column(3.0, 1.0, n, 51),
                    _normal_column(3.5, 1.0, n, 52),
                    _normal_column(3.2, 1.0, n, 53),
                ]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue"), MetricSpec(name="clicks")],
        plan=AnalysisPlan(primary=["revenue", "clicks"]),
    )
    for name in ("revenue", "clicks"):
        test = src.context.plan.procedures[name]
        assert test.role == "primary"
        assert test.alpha == pytest.approx(0.025)

    results = readouts.run(src)
    primary_results = [r for r in results if r.role == "primary"]
    assert {(r.metric, r.group_id) for r in primary_results} == {
        ("revenue", "treatment_a"),
        ("revenue", "treatment_b"),
        ("clicks", "treatment_a"),
        ("clicks", "treatment_b"),
    }
    for r in primary_results:
        assert r.require_lift().level == pytest.approx(1 - 0.0125)


def _guarded_three_arm_source(*, all_degenerate: bool = False):
    n = 200
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(3 * n)],
            "variant": ["control"] * n + ["treatment_good"] * n + ["treatment_bad"] * n,
            "revenue": np.concatenate(
                [
                    np.full(n, 10.0) if all_degenerate else _normal_column(10.0, 2.0, n, 31),
                    np.full(n, 11.0) if all_degenerate else _normal_column(11.0, 2.0, n, 32),
                    np.full(n, 12.0),
                ]
            ),
        }
    )
    return FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        plan=AnalysisPlan(primary="revenue"),
    )


def test_run_retains_valid_whole_window_cells_after_guard():
    src = _guarded_three_arm_source()

    with pytest.warns(IncrementWarning) as rec:
        results = readouts.run(src)
    cell_refused = [
        w.message
        for w in rec
        if isinstance(w.message, IncrementWarning) and w.message.code == "readouts.run.cell_refused"
    ]
    assert any(m.context["group_id"] == "treatment_bad" for m in cell_refused)

    assert [(r.metric, r.group_id, r.method) for r in results] == [
        ("revenue", "treatment_good", "unadjusted")
    ]


def test_registered_raw_run_refuses_unproved_sensitivity_method():
    """An adjusted sensitivity row cannot inherit raw likelihood evidence."""
    from increment.errors import CapabilityError
    from tests.test_sequential_public_sources import gaussian_plan

    n = 200
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "exposure": list(range(2 * n)),
            "revenue": np.concatenate(
                [
                    np.random.default_rng(41).integers(0, 2, n),
                    np.random.default_rng(42).integers(0, 2, n),
                ]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        exposure_date="exposure",
        metrics=[MetricSpec(name="revenue", type="conversion")],
        plan=gaussian_plan(
            [MetricSpec(name="revenue", type="conversion")],
            law="bernoulli",
            exposure_date="exposure",
        ),
    )

    with pytest.raises(CapabilityError) as raised:
        readouts.run(
            src, decision_method=Method(name="unadjusted"), sensitivity_methods=(Method(name="m2"),)
        )
    assert raised.value.code == "sequential.route.unsupported"


def test_run_deduplicates_cluster_advisories_per_metric():
    """Independent methods still emit one small-cluster advisory."""
    n = 15
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "store": [f"s{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "revenue": np.concatenate(
                [
                    np.arange(n, dtype=float) + 10.0,
                    np.arange(n, dtype=float) + 11.0,
                ]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        plan=AnalysisPlan(primary="revenue"),
        cluster="store",
    )

    with pytest.warns(IncrementRuntimeWarning) as captured:
        results = readouts.run(
            src, decision_method=Method(name="unadjusted"), sensitivity_methods=(Method(name="m2"),)
        )

    small_k = [
        code for code in warning_codes(captured) if code == "estimation.engine.small_total_clusters"
    ]
    assert len(small_k) == 1
    assert {(result.group_id, result.method) for result in results} == {
        ("treatment", "unadjusted"),
        ("treatment", "m2"),
    }


def test_run_all_guarded_whole_window_cells_have_explicit_outcome():
    src = _guarded_three_arm_source(all_degenerate=True)

    with pytest.warns(IncrementWarning) as captured:
        with pytest.raises(UnsupportedRequestError) as exc_info:
            readouts.run(src)
        assert exc_info.value.code == "readout.estimate_lift_every"
    assert {
        w.message.context["group_id"]
        for w in captured
        if isinstance(w.message, IncrementWarning) and w.message.code == "readouts.run.cell_refused"
    } == {
        "treatment_bad",
        "treatment_good",
    }


def test_family_missing_control_reports_incomplete():
    src = _cuped_family_source()
    metric = cast("Metric", src.context.metrics[0])
    moments = src.moments

    def without_control(requested, **kwargs):
        rows = moments(requested, **kwargs)
        if requested.name == metric.name:
            return [row for row in rows if str(row["group_id"]) != "control"]
        return rows

    src.moments = without_control

    with pytest.raises(CapabilityError) as raised:
        readouts.run(src)
    assert raised.value.code == "family.evidence.incomplete"
    key = ArmHypothesisKey(metric.name, "treatment", "itt")
    assert key in raised.value.context["failed"]  # ty: ignore[unsupported-operator]


def test_run_replays_advisory_before_reraising_unrelated_whole_window_cell_error(monkeypatch):
    src = _guarded_three_arm_source()
    original = _passes.estimate_lift

    def raise_unrelated(*args, **kwargs):
        if any(str(row["group_id"]) == "treatment_bad" for row in kwargs["summary"]):
            warnings.warn("metric is open-ended (window_days=None)", UserWarning, stacklevel=2)
            raise ValueError("unexpected estimator failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(_passes, "estimate_lift", raise_unrelated)
    with pytest.warns(UserWarning, match="is open-ended"):
        with pytest.raises(ValueError, match="unexpected estimator failure"):
            readouts.run(src)


def test_run_replays_advisory_before_reraising_missing_control(monkeypatch):
    src = _guarded_three_arm_source()
    metric = cast("Metric", src.context.metrics[0])
    moments = src.moments

    def without_control(requested, **kwargs):
        rows = moments(requested, **kwargs)
        if requested.name == metric.name:
            return [row for row in rows if str(row["group_id"]) != "control"]
        return rows

    src.moments = without_control

    def raise_unrelated(*args, **kwargs):
        warnings.warn("metric is open-ended (window_days=None)", UserWarning, stacklevel=2)
        raise ValueError("unexpected estimator failure")

    monkeypatch.setattr(_passes, "estimate_lift", raise_unrelated)
    with pytest.warns(UserWarning, match="is open-ended"):
        with pytest.raises(ValueError, match="unexpected estimator failure"):
            readouts.run(src)


def test_family_selection_uses_explicit_cuped_decision_for_randomized_fcr():
    results = readouts.run(_cuped_family_source())
    cuped = [row for row in results if row.method == "cuped"]
    unadjusted = [row for row in results if row.method == "unadjusted"]

    # The declared CUPED decision row alone supplies the family verdict;
    # configured unadjusted sensitivity remains metadata-free.
    assert [(row.metric, row.discovery) for row in cuped] == [
        ("m_a", True),
        ("m_b", False),
        ("m_c", False),
        ("m_d", False),
    ]
    assert next(row for row in cuped if row.metric == "m_a").require_lift().level == pytest.approx(
        0.975
    )
    assert next(row for row in cuped if row.metric == "m_a").family_axes == ("metric", "arm")
    assert next(row for row in unadjusted if row.metric == "m_a").discovery is None


def test_family_selection_uses_explicit_cuped_decision_for_encouragement_fcr():
    results = readouts.run(_cuped_encouragement_source(), estimands=("itt",))
    cuped = [row for row in results if row.method == "cuped" and row.estimand == "itt"]
    unadjusted = [row for row in results if row.method == "unadjusted" and row.estimand == "itt"]

    assert [(row.metric, row.discovery) for row in cuped] == [
        ("m_a", True),
        ("m_b", False),
        ("m_c", False),
        ("m_d", False),
    ]
    assert next(row for row in cuped if row.metric == "m_a").require_lift().level == pytest.approx(
        0.975
    )
    assert next(row for row in cuped if row.metric == "m_a").family_threshold == pytest.approx(
        0.025
    )
    assert next(row for row in unadjusted if row.metric == "m_a").discovery is None


def test_asof_fixed_family_preserves_explicit_cuped_decision():
    cuped_method = Method(name="cuped", variance_reduction="cuped")
    results = readouts.asof_lift(_cuped_asof_source(), decision_method=cuped_method)
    cuped = [row for row in results if row.method == "cuped" and row.metric == "m_a"]
    assert len(cuped) == 2
    assert all(row.inference == "fixed" for row in cuped)
    assert all(row.lift is not None for row in cuped)


def test_secondary_discovery_matches_hand_bh():
    """4 secondaries with engineered effects: discovery flags equal
    bh_select on the rows' own p_values, and selected rows carry
    FCR-level (Benjamini-Yekutieli) intervals; non-selected rows stay at
    the plan's nominal level."""
    n = 1500
    control_mean = 10.0
    sd = 2.0
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            # m_a: unambiguous lift (selected); m_c: weak lift (not
            # selected at q=0.10); m_b/m_d: null.
            "m_a": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 1),
                    _normal_column(control_mean * 1.15, sd, n, 2),
                ]
            ),
            "m_b": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 3),
                    _normal_column(control_mean * 1.002, sd, n, 4),
                ]
            ),
            "m_c": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 5),
                    _normal_column(control_mean * 1.01, sd, n, 6),
                ]
            ),
            "m_d": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 7),
                    _normal_column(control_mean * 0.999, sd, n, 8),
                ]
            ),
        }
    )
    plan = AnalysisPlan()  # empty plan -> declared=True, every metric defaults to secondary
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name=name) for name in ("m_a", "m_b", "m_c", "m_d")],
        plan=plan,
    )
    results = readouts.run(src)
    secondary = [r for r in results if r.role == "secondary"]
    assert {r.metric for r in secondary} == {"m_a", "m_b", "m_c", "m_d"}

    p_values = [r.p_value() for r in secondary]
    selected_idx, _threshold = bh_select(p_values, plan.q)
    selected_set = set(selected_idx)
    assert [r.discovery for r in secondary] == [i in selected_set for i in range(len(secondary))]
    assert selected_set == {i for i, r in enumerate(secondary) if r.metric == "m_a"}

    expected_selected_alpha = _conservative_ratio(plan.q, len(selected_idx), len(p_values))
    for i, r in enumerate(secondary):
        if i in selected_set:
            assert r.require_lift().alpha == expected_selected_alpha
        else:
            assert r.require_lift().alpha == pytest.approx(plan.alpha)


def test_secondary_discovery_tracks_selection_with_one_sided_fcr_interval():
    """Under a one-sided plan (alternative="greater"), a secondary's BH
    `discovery` verdict records selection and its full-tail fixed FCR
    interval excludes the null in this uncapped fixture. m_a's one-sided
    p-value (~0.021) clears the q=0.10/m=4 BH rank-1 threshold (0.025), but an
    always-two-sided-against-zero p-value (~0.043, what the family
    selection used to naively compute) would have missed it."""
    from increment.estimation.family import bh_select

    n = 1500
    control_mean = 10.0
    sd = 2.0
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "m_a": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 1),
                    _normal_column(control_mean * 1.018, sd, n, 2),
                ]
            ),
            "m_b": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 3),
                    _normal_column(control_mean * 1.002, sd, n, 4),
                ]
            ),
            "m_c": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 5),
                    _normal_column(control_mean * 1.01, sd, n, 6),
                ]
            ),
            "m_d": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 7),
                    _normal_column(control_mean * 0.999, sd, n, 8),
                ]
            ),
        }
    )
    plan = AnalysisPlan(alternative="greater")
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name=name) for name in ("m_a", "m_b", "m_c", "m_d")],
        plan=plan,
    )
    results = readouts.run(src)
    secondary = [r for r in results if r.role == "secondary"]
    assert {r.metric for r in secondary} == {"m_a", "m_b", "m_c", "m_d"}
    for r in secondary:
        assert r.alternative == "greater"

    # discovery matches BH on the rows' own (one-sided) p_values -- m_a
    # is the sole discovery.
    p_values = [r.p_value() for r in secondary]
    selected_idx, _threshold = bh_select(p_values, plan.q)
    selected_set = set(selected_idx)
    assert [r.discovery for r in secondary] == [i in selected_set for i in range(len(secondary))]
    assert selected_set == {i for i, r in enumerate(secondary) if r.metric == "m_a"}

    m_a = next(r for r in secondary if r.metric == "m_a")
    assert m_a.discovery is True
    assert m_a.stat_sig() is True
    one_sided_lift = m_a.require_lift()
    assert one_sided_lift.open_side == "upper"
    assert one_sided_lift.ub is None

    # Reading the same panel without the declared direction gives the
    # always-two-sided p-value, which misses m_a's BH cutoff: selection
    # follows the plan's own alternative, not a direction-blind p-value.
    two_sided_src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name=name) for name in ("m_a", "m_b", "m_c", "m_d")],
        plan=AnalysisPlan(alternative="two-sided"),
    )
    two_sided = [r for r in readouts.run(two_sided_src) if r.role == "secondary"]
    two_sided_selected, _ = bh_select([r.p_value() for r in two_sided], plan.q)
    m_a_index = next(i for i, r in enumerate(two_sided) if r.metric == "m_a")
    assert m_a_index not in two_sided_selected


def test_secondary_discovery_can_disagree_with_stat_sig_when_nominal_cap_binds():
    """`discovery` is the family-selection verdict, not a restatement of
    the row's own `stat_sig()` -- BY's cutoff `R*q/m` can exceed the
    plan's nominal alpha, in which case the re-estimated interval is cut
    at the (tighter) nominal level rather than the (looser) uncapped BH
    level. A cell can then clear BH's own selection threshold on its
    nominal p-value (`discovery=True`) while its capped, wider interval
    still contains the null (`stat_sig=False`); this is not a bug to
    reconciled away, and the old `family_discovery(...) and stat_sig()`
    conjunct must not be restored to paper over it."""
    n = 1500
    control_mean = 10.0
    sd = 2.0
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "m_a": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 1),
                    _normal_column(control_mean * 1.01, sd, n, 2),
                ]
            ),
        }
    )
    # q=0.5 (generous FDR level) with a single-metric family means BH's own
    # realized cutoff is R*q/m = 0.5 whenever the cell is selected -- five
    # times alpha=0.01, so fcr_alpha = min(0.5, 0.01) = 0.01 (capped): the
    # re-estimated interval is the WIDE 99% one, not the narrow 50% BH one.
    plan = AnalysisPlan(alpha=0.01, q=0.5)
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="m_a")],
        plan=plan,
    )
    results = readouts.run(src)
    (row,) = [r for r in results if r.role == "secondary"]

    assert row.p_value() < plan.q  # clears BH's own (generous) selection bar
    assert row.discovery is True
    assert row.family_threshold == pytest.approx(plan.q)  # realized R*q/m, uncapped
    lift = row.require_lift()
    assert lift.level == pytest.approx(1.0 - plan.alpha)  # capped to nominal, not 0.5
    assert row.stat_sig() is False
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < 0.0 < lift.ub  # the capped interval still contains the null


def test_guardrail_one_sided_at_margin():
    """margin=0.01 decrease-guardrail: null shifted to +0.01, alternative
    one-sided "less", and alpha stays unsplit across both treatment arms
    (level = 1 - 2*alpha_share on every row, not divided by n_arms)."""
    n = 800
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(3 * n)],
            "variant": ["control"] * n + ["treatment_a"] * n + ["treatment_b"] * n,
            "loss_metric": np.concatenate(
                [
                    _normal_column(5.0, 1.0, n, 31),
                    _normal_column(5.02, 1.0, n, 32),
                    _normal_column(4.9, 1.0, n, 33),
                ]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="loss_metric", preferred_direction="decrease")],
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="loss_metric", margin=0.01)]),
    )
    test = src.context.plan.procedures["loss_metric"]
    assert test.role == "guardrail"
    assert test.null_lift == pytest.approx(0.01)  # ty: ignore[unresolved-attribute]
    assert test.alternative == "less"
    assert test.alpha == pytest.approx(0.05)  # unsplit at plan-resolution time

    results = readouts.run(src)
    assert {r.group_id for r in results} == {"treatment_a", "treatment_b"}
    for r in results:
        assert r.role == "guardrail"
        assert r.null_lift == pytest.approx(0.01)
        assert r.alternative == "less"
        # one-sided alpha-doubling identity, unsplit by n_arms=2
        assert r.require_lift().level == pytest.approx(1.0 - 2 * test.alpha)


def test_guardrail_one_sided_without_margin():
    """A decrease-guardrail declared with no margin/margin_abs still tests
    one-sided against zero (alternative="less"), never the plan's default
    two-sided at full alpha -- the interval-width machinery (alpha-doubling,
    unsplit across arms) is identical to the margin-declared case, just
    against a zero null instead of a shifted one."""
    n = 800
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(3 * n)],
            "variant": ["control"] * n + ["treatment_a"] * n + ["treatment_b"] * n,
            "loss_metric": np.concatenate(
                [
                    _normal_column(5.0, 1.0, n, 31),
                    _normal_column(5.02, 1.0, n, 32),
                    _normal_column(4.9, 1.0, n, 33),
                ]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="loss_metric", preferred_direction="decrease")],
        plan=AnalysisPlan(guardrails=["loss_metric"]),
    )
    test = src.context.plan.procedures["loss_metric"]
    assert test.role == "guardrail"
    assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute]
    assert test.alternative == "less"
    assert test.alpha == pytest.approx(0.05)  # unsplit at plan-resolution time

    results = readouts.run(src)
    assert {r.group_id for r in results} == {"treatment_a", "treatment_b"}
    by_group = {r.group_id: r for r in results}
    for r in results:
        assert r.role == "guardrail"
        assert r.null_lift == 0.0
        assert r.alternative == "less"
        # one-sided alpha-doubling identity, unsplit by n_arms=2
        assert r.require_lift().level == pytest.approx(1.0 - 2 * test.alpha)
    # treatment_a's latency rose (worse, for a decrease-preferred metric):
    # the one-sided "less" upper bound stays above the zero null.
    ub_a = by_group["treatment_a"].require_lift().ub
    assert ub_a is not None and ub_a > 0.0
    # treatment_b's latency fell enough that the one-sided upper bound
    # clears zero too -- the interval excludes the null on the tested side.
    ub_b = by_group["treatment_b"].require_lift().ub
    assert ub_b is not None and ub_b < 0.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_av_plan_uses_e_bh_and_keeps_nominal_cs():
    """Full registered e-BH selection and same-checkpoint selected inversion."""
    n = 1500
    control_mean = 0.4
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "exposure": list(range(2 * n)),
            "m_a": np.concatenate(
                [
                    np.random.default_rng(1).binomial(1, control_mean, n),
                    np.random.default_rng(2).binomial(1, 0.7, n),
                ]
            ),
            "m_b": np.concatenate(
                [
                    np.random.default_rng(3).binomial(1, control_mean, n),
                    np.random.default_rng(4).binomial(1, 0.42, n),
                ]
            ),
            "m_c": np.concatenate(
                [
                    np.random.default_rng(5).binomial(1, control_mean, n),
                    np.random.default_rng(6).binomial(1, 0.5, n),
                ]
            ),
        }
    )
    metric_specs = [MetricSpec(name=name, type="conversion") for name in ("m_a", "m_b", "m_c")]
    plan = gaussian_plan(metric_specs, law="bernoulli", exposure_date="exposure")
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        exposure_date="exposure",
        metrics=metric_specs,
        plan=plan,
    )
    assert isinstance(src.context.plan.inference, AlwaysValid)
    results = readouts.run(src)
    secondary = [r for r in results if r.role == "secondary"]
    assert len(secondary) == 3
    for r in secondary:
        assert r.inference == "always_valid"

    logs = [r.require_exact_sequential_result().log_e for r in secondary]
    selected_set = set(e_bh_select(logs, plan.q))
    assert [r.discovery for r in secondary] == [i in selected_set for i in range(len(secondary))]
    assert selected_set
    from fractions import Fraction

    from increment.estimation.decision_types import exact_fraction

    selected_alpha = min(
        exact_fraction(plan.q) * len(selected_set) / len(secondary), exact_fraction(plan.alpha)
    )
    for i, row in enumerate(secondary):
        expected = selected_alpha if i in selected_set else Fraction(plan.alpha)
        assert row.require_sequential_result().bounds.alpha == expected


def _moments_row(metric_name: str, group_id: str, n: int, mean: float, var: float) -> dict:
    return {
        "experiment_id": "e",
        "metric": metric_name,
        "group_id": group_id,
        "n": n,
        "ref_y": mean,
        "cy1": 0.0,
        "cy2": var * n,
        "winsor_lower_percentile": None,
        "winsor_upper_percentile": None,
        "winsor_lower_bound": None,
        "winsor_upper_bound": None,
        "winsor_n": None,
        "winsor_n_lower": None,
        "winsor_n_upper": None,
        "moments_format": 8,
    }


def _moments_with_plan(
    rows: list[dict], metrics: list[MeanMetric], plan: AnalysisPlan
) -> list[dict]:
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan

    wire = compiled_plan_to_json(
        compile_decision_plan(
            plan, metrics, path="warehouse", design=Randomized(control_group="control")
        )
    )
    return [{**row, "decision_plan": wire} for row in rows]


def test_prior_secondary_outside_family():
    """Effective priors stay outside the family until explicitly cleared."""
    n = 2000
    rows = [
        _moments_row("family_metric_1", "control", n, 10.0, 4.0),
        _moments_row("family_metric_1", "treatment", n, 11.5, 4.0),  # clear lift -- selected
        _moments_row("family_metric_2", "control", n, 10.0, 4.0),
        _moments_row("family_metric_2", "treatment", n, 10.02, 4.0),  # null -- not selected
        _moments_row("prior_bound_metric", "control", n, 10.0, 4.0),
        _moments_row("prior_bound_metric", "treatment", n, 12.0, 4.0),  # extreme, but out of family
    ]
    metrics = [
        MeanMetric(name=name, entity="user_id", fact=name, aggregation="avg_event")
        for name in ("family_metric_1", "family_metric_2", "prior_bound_metric")
    ]
    # q=0.5 (rather than the 0.10 default) makes the FCR-adjusted level
    # visibly differ from nominal for this test's n/effect sizes.
    plan = AnalysisPlan(
        q=0.5,
        secondaries=[
            "family_metric_1",
            "family_metric_2",
            ExperimentMetric(metric="prior_bound_metric", prior=NormalPriorSpec(mu=0.0, sigma=0.5)),
        ],
    )
    src = MomentsSource(
        _moments_with_plan(rows, metrics, plan),
        metrics=metrics,
        study_id="e",
        design=Randomized(control_group="control"),
        plan=plan,
        # MomentsSource defaults to path="frame", which refuses prior-bound plan
        # entries; path="warehouse" can express the prior-bound secondary.
        path="warehouse",
    )

    results = readouts.run(src)
    by_metric = {r.metric: r for r in results}
    assert by_metric["prior_bound_metric"].role == "secondary"
    assert by_metric["prior_bound_metric"].discovery is None
    assert by_metric["prior_bound_metric"].require_lift().level == pytest.approx(1.0 - plan.alpha)

    assert by_metric["family_metric_1"].discovery is True
    assert by_metric["family_metric_2"].discovery is False
    # m=2 (prior_bound_metric excluded), R=1 and q=0.5 give BH cutoff R*q/m = 0.25,
    # five times nominal 0.05. Uncapped, the selected interval would use level
    # 0.75, narrower than an uncorrected 0.95; the cap holds the nominal level.
    assert by_metric["family_metric_1"].require_lift().level == pytest.approx(1.0 - plan.alpha)
    assert by_metric["family_metric_2"].require_lift().level == pytest.approx(1.0 - plan.alpha)
    # The uncapped cutoff is still recorded, so the disclosure survives.
    assert by_metric["family_metric_1"].family_threshold == pytest.approx(0.25)
    assert by_metric["family_metric_1"].family_axes == ("metric", "arm")
    assert by_metric["family_metric_1"].family_q == pytest.approx(plan.q)

    prior_free_plan = AnalysisPlan(q=0.5, secondaries=[metric.name for metric in metrics])
    prior_free_source = MomentsSource(
        _moments_with_plan(rows, metrics, prior_free_plan),
        metrics=metrics,
        study_id="e",
        design=Randomized(control_group="control"),
        plan=prior_free_plan,
        path="warehouse",
    )
    expected = {row.metric: row for row in readouts.run(prior_free_source)}
    assert expected["prior_bound_metric"].discovery is True
    assert expected["prior_bound_metric"].require_lift().value == pytest.approx(0.2)
    for _ in range(2):
        cleared = {row.metric: row for row in readouts.run(src, prior=None)}
        assert cleared.keys() == expected.keys()
        for name, row in cleared.items():
            oracle = expected[name]
            assert row.discovery == oracle.discovery
            assert row.family_axes == oracle.family_axes
            actual_interval, expected_interval = row.require_lift(), oracle.require_lift()
            assert (actual_interval.value, actual_interval.lb, actual_interval.ub) == pytest.approx(
                (expected_interval.value, expected_interval.lb, expected_interval.ub)
            )
    inherited = {row.metric: row for row in readouts.run(src)}
    assert inherited["prior_bound_metric"].discovery is None
    assert inherited["prior_bound_metric"].family_axes is None
    assert inherited["family_metric_1"].family_threshold == pytest.approx(0.25)


def test_callwide_prior_is_outside_secondary_family():
    """A call-wide prior overlay excludes every secondary from BH."""
    n = 2000
    rows = [
        _moments_row("family_metric", "control", n, 10.0, 4.0),
        _moments_row("family_metric", "treatment", n, 12.0, 4.0),
        _moments_row("prior_metric", "control", n, 10.0, 4.0),
        _moments_row("prior_metric", "treatment", n, 12.0, 4.0),
    ]
    metrics = [
        MeanMetric(name=name, entity="user_id", fact=name, aggregation="avg_event")
        for name in ("family_metric", "prior_metric")
    ]
    plan = AnalysisPlan(q=0.5, secondaries=["family_metric", "prior_metric"])
    src = MomentsSource(
        _moments_with_plan(rows, metrics, plan),
        metrics=metrics,
        study_id="e",
        design=Randomized(control_group="control"),
        plan=plan,
        path="warehouse",
    )

    results = readouts.run(src, prior=Normal(mu=0.0, sigma=0.1))

    assert results
    assert all(r.role == "secondary" for r in results)
    assert all(r.discovery is None for r in results)
    assert all(r.family_axes is None for r in results)
    assert all(
        r.require_lift().level == pytest.approx(1.0 - src.context.plan.alpha) for r in results
    )


def test_no_plan_matches_legacy_run():
    """No declared plan: every metric resolves role="unassigned" at
    alpha_share=0.05/alternative="two-sided" (the pre-cutover run()
    default), but the row's stamped `role` is None, not "unassigned" -
    None is reserved for `src.plan.declared is False`."""
    n = 800
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "revenue": np.concatenate(
                [_normal_column(10.0, 2.0, n, 41), _normal_column(10.6, 2.0, n, 42)]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    assert src.context.plan.declared is False
    assert src.context.plan.procedures["revenue"].role == "unassigned"

    (row,) = readouts.run(src)
    assert row.role is None
    assert row.discovery is None

    metric_obj = next(m for m in src.context.metrics if m.name == "revenue")
    rows = src.moments(metric_obj, grain="total")
    (expected,) = estimate_lift(
        [metric_obj],
        rows,
        control_group="control",
        alpha=0.05,
        alternative="two-sided",
        null_lift=0.0,
        null_abs=None,
        preferred_direction=metric_obj.declared_preferred_direction,
        cluster=None,
    ).results
    assert row.require_lift().value == pytest.approx(expected.require_lift().value, rel=1e-12)
    assert row.require_lift().lb == pytest.approx(expected.require_lift().lb, rel=1e-12)
    assert row.require_lift().ub == pytest.approx(expected.require_lift().ub, rel=1e-12)
    assert row.require_lift().level == pytest.approx(expected.require_lift().level)
    assert row.null_lift == expected.null_lift
    assert row.alternative == expected.alternative


def test_declared_plan_leaves_an_unnamed_metric_unassigned():
    """Companion to test_no_plan_matches_legacy_run: a DECLARED plan that
    simply doesn't name a metric on the warehouse path (`path="warehouse"`,
    passed explicitly here - `MomentsSource`'s own default is now
    `path="frame"`, which would default `orders` to `role="secondary"`
    instead) resolves that metric to role="unassigned" (a real string),
    distinct from role=None (no plan declared at all)."""
    n = 500
    rows = [
        _moments_row("revenue", "control", n, 10.0, 4.0),
        _moments_row("revenue", "treatment", n, 10.5, 4.0),
        _moments_row("orders", "control", n, 3.0, 1.0),
        _moments_row("orders", "treatment", n, 3.1, 1.0),
    ]
    metrics = [
        MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event"),
        MeanMetric(name="orders", entity="user_id", fact="orders", aggregation="avg_event"),
    ]
    plan = AnalysisPlan(primary="revenue")
    src = MomentsSource(
        _moments_with_plan(rows, metrics, plan),
        metrics=metrics,
        study_id="e",
        design=Randomized(control_group="control"),
        plan=plan,
        path="warehouse",
    )
    assert src.context.plan.declared is True
    assert src.context.plan.procedures["orders"].role == "unassigned"

    results = readouts.run(src)
    by_metric = {r.metric: r.role for r in results}
    assert by_metric["revenue"] == "primary"
    assert by_metric["orders"] == "unassigned"


def test_secondary_family_dedups_by_method_not_by_method_x_arm():
    """2 methods x 2 secondaries x 1 arm: family size m must count
    (metric, arm) cells, not (metric, method, arm) rows - and every
    method's row for the same cell must carry the same discovery verdict
    (and, when selected, the same FCR-adjusted level)."""
    n = 1500
    control_mean = 10.0
    sd = 2.0
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            # m_a: unambiguous lift (selected); m_b: null (not selected).
            "m_a": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 51),
                    _normal_column(control_mean * 1.15, sd, n, 52),
                ]
            ),
            "m_b": np.concatenate(
                [
                    _normal_column(control_mean, sd, n, 53),
                    _normal_column(control_mean * 1.002, sd, n, 54),
                ]
            ),
        }
    )
    plan = AnalysisPlan()  # empty plan -> declared=True, every metric defaults to secondary
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name=name) for name in ("m_a", "m_b")],
        plan=plan,
    )
    results = readouts.run(
        src, decision_method=Method(name="unadjusted"), sensitivity_methods=(Method(name="m2"),)
    )
    secondary = [r for r in results if r.role == "secondary"]
    # 2 metrics x 2 methods x 1 arm = 4 rows, but the family denominator
    # must be 2 (metric, arm) cells, not 4 (metric, method, arm) rows.
    assert len(secondary) == 4
    assert {r.method for r in secondary} == {"unadjusted", "m2"}

    by_metric: dict[str, list] = {}
    for r in secondary:
        by_metric.setdefault(r.metric, []).append(r)

    # Family metadata belongs only to the decision row; sensitivities keep
    # nominal presentation metadata clear while sharing the FCR interval.
    for _metric_name, rows in by_metric.items():
        decision = next(r for r in rows if r.method == "unadjusted")
        sensitivity = next(r for r in rows if r.method == "m2")
        assert decision.discovery is not None
        assert sensitivity.discovery is None

    assert by_metric["m_a"][0].discovery is True
    assert by_metric["m_b"][0].discovery is False

    # m=2 (metric x arm), not 4 (metric x method x arm): the selected rows'
    # alpha must be q*1/2, not q*1/4.
    expected_selected_alpha = _conservative_ratio(plan.q, 1, 2)
    wrong_selected_alpha = _conservative_ratio(plan.q, 1, 4)
    assert expected_selected_alpha != pytest.approx(wrong_selected_alpha)
    for r in by_metric["m_a"]:
        assert r.require_lift().alpha == expected_selected_alpha
    for r in by_metric["m_b"]:
        assert r.require_lift().alpha == pytest.approx(plan.alpha)


def test_always_valid_margin_bearing_secondary_discovery_matches_stat_sig():
    """A margin-bearing secondary (declared ``margin=0.05``, so
    ``null_lift=-0.05``) under an AlwaysValid plan: `discovery` must be
    decided against the row's own declared null, not a silently-zeroed
    one, using the same registered likelihood and stopped state as its
    selected interval.

    q=alpha here makes e-BH's single-cell (m=1)
    threshold and the row's own confidence-sequence boundary the exact
    same cutoff, so `discovery` must equal `stat_sig` exactly, not just
    avoid contradicting it.
    """
    n = 3000
    true_lift = -0.045  # inside the -0.05 margin, but far from 0
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "exposure": list(range(2 * n)),
            "m_a": np.concatenate(
                [
                    np.random.default_rng(101).binomial(1, 0.5, n),
                    np.random.default_rng(102).binomial(1, 0.5 * (1 + true_lift), n),
                ]
            ),
        }
    )
    from fractions import Fraction

    from increment import SequentialCell

    metric_specs = [MetricSpec(name="m_a", type="conversion", preferred_direction="increase")]
    plan = gaussian_plan(
        metric_specs,
        law="bernoulli",
        q=0.05,
        exposure_date="exposure",
        secondaries=[ExperimentMetric(metric="m_a", margin=0.05)],
        cells=(
            SequentialCell(
                metric="m_a",
                group_id="treatment",
                family=True,
                alternative="greater",
                null_lift=Fraction(-0.05),
                alpha=Fraction(0.05),
            ),
        ),
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        exposure_date="exposure",
        metrics=metric_specs,
        plan=plan,
    )
    assert src.context.plan.procedures["m_a"].null_lift == pytest.approx(-0.05)  # ty: ignore[unresolved-attribute]
    assert src.context.plan.procedures["m_a"].alternative == "greater"
    assert src.context.plan.procedures["m_a"].family.member is True  # ty: ignore[unresolved-attribute]
    (r,) = readouts.run(src)
    assert r.role == "secondary"
    assert r.inference == "always_valid"
    assert r.null_lift == pytest.approx(-0.05)

    # The declared null is comfortably cleared (-4.5% > -5%), but not by
    # much -- own confidence sequence does not exclude it.
    assert r.stat_sig() is False
    assert r.discovery is False

    # Re-register the zero null on the same observations and preserve its
    zero_plan = gaussian_plan(metric_specs, law="bernoulli", q=0.05, exposure_date="exposure")
    zero_source = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        exposure_date="exposure",
        metrics=metric_specs,
        plan=zero_plan,
    )
    (zero_row,) = readouts.run(zero_source)
    assert zero_row.discovery == zero_row.stat_sig()
    margin_result = r.require_exact_sequential_result()
    zero_result = zero_row.require_exact_sequential_result()
    assert margin_result.checkpoint.control == zero_result.checkpoint.control
    assert margin_result.checkpoint.treatment == zero_result.checkpoint.treatment
    assert margin_result.checkpoint.cell.null_lift == Fraction(-0.05)
    assert zero_result.checkpoint.cell.null_lift == 0
    assert isinstance(margin_result.log_e, Fraction)
    assert isinstance(zero_result.log_e, Fraction)
    assert margin_result.log_e != zero_result.log_e
    for row in (r, zero_row):
        replay = type(row).model_validate_json(row.model_dump_json())
        assert replay.require_sequential_result() == row.require_sequential_result()
        assert replay.discovery == row.discovery
        assert replay.stat_sig() == row.stat_sig()


@pytest.mark.parametrize("q", [0.01, 0.50], ids=["uncapped", "capped"])
def test_cuped_row_never_reads_discovery_true_off_an_unadjusted_canonical(q: float):
    """methods=[unadjusted, cuped]: engineered so the unadjusted row is
    clearly significant (drives the family's selection, since dedup
    prefers "unadjusted" as canonical) while CUPED's OWN interval does
    NOT exclude the null -- an arm-imbalanced covariate mean shifts
    CUPED's adjusted point estimate toward zero without shrinking its
    variance as much on this particular draw. Before the fix, `discovery`
    was stamped `True` on every method's row for a selected cell; a
    CUPED row could then read ``stat_sig=False, discovery=True``, a
    verdict for a null the family selection never actually tested."""
    rng = np.random.default_rng(7)
    n = 400
    control_mean = 10.0
    sd = 2.0
    lift = 0.045
    y_c = rng.normal(control_mean, sd, n)
    y_t = rng.normal(control_mean * (1 + lift), sd, n)
    x_c = y_c * 0.3 + rng.normal(0.0, 3.0, n)
    x_t = y_t * 0.3 + rng.normal(2.0, 3.0, n)
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([y_c, y_t]),
            "x": np.concatenate([x_c, x_t]),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="y", covariate="x")],
        plan=AnalysisPlan(q=q),  # declared=True, "y" defaults to secondary
    )
    results = readouts.run(
        src,
        decision_method=Method(name="unadjusted"),
        sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
    )
    by_method = {r.method: r for r in results}
    assert set(by_method) == {"unadjusted", "cuped"}

    unadjusted, cuped = by_method["unadjusted"], by_method["cuped"]
    # Pins the engineered scenario: unadjusted drives the (sole) family
    # decision, CUPED's own evidence disagrees.
    assert unadjusted.stat_sig() is True
    assert cuped.stat_sig() is False

    assert unadjusted.discovery is True
    # The exact bug: a CUPED row must never read stat_sig=False,
    # discovery=True off the unadjusted canonical's decision.
    assert not (cuped.stat_sig() is False and cuped.discovery is True)
    assert cuped.discovery is None


def test_randomized_run_preserves_declaration_order_across_mixed_roles():
    """A plan mixing roles must not reorder metrics by role: the
    randomized branch estimates non-secondary rows first, then deferred
    secondary-family rows, but the returned list must come back in the
    original `metrics=` declaration order regardless."""
    n = 800
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "checkout_rate": np.concatenate(
                [_normal_column(0.30, 0.1, n, 61), _normal_column(0.31, 0.1, n, 62)]
            ),
            "revenue": np.concatenate(
                [_normal_column(10.0, 2.0, n, 63), _normal_column(10.5, 2.0, n, 64)]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="checkout_rate"), MetricSpec(name="revenue")],
        plan=AnalysisPlan(primary="revenue"),
    )
    assert src.context.plan.procedures["checkout_rate"].role == "secondary"
    assert src.context.plan.procedures["revenue"].role == "primary"

    results = readouts.run(src)
    assert [r.metric for r in results] == ["checkout_rate", "revenue"]


def test_control_only_primary_metric_refuses_with_no_treatment_code():
    """A primary metric whose moments carry only the control arm has no
    contrast to report: run() refuses with a stable code instead of
    silently returning no rows."""
    n = 400
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": ["control"] * n,
            "revenue": _normal_column(10.0, 2.0, n, 71),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        plan=AnalysisPlan(primary="revenue"),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.run(src)
    assert exc_info.value.code == "readout.arms.no_treatment"
    assert exc_info.value.context["observed_arms"] == ("control",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"design": Randomized(control_group="control")},
        {"alpha": 0.05},
        {"alternative": "greater"},
        {"inference": AlwaysValid(registration=registration("gaussian"))},
        {"margins": {"revenue": 0.01}},
        {"null_lifts": {"revenue": 0.01}},
        {"margins_abs": {"revenue": 0.01}},
    ],
)
def test_removed_kwargs_raise_typeerror(kwargs):
    n = 200
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "revenue": np.concatenate(
                [_normal_column(10.0, 2.0, n, 1), _normal_column(10.5, 2.0, n, 2)]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    with pytest.raises(TypeError):
        readouts.run(src, **kwargs)


# breakout(): plan-inherited alpha/inference, "bh"-default family selection
# across every (metric, arm, segment) cell, "bonferroni"/"none" byte-for-byte,
# and role="exploratory" stamping everywhere (breakout has no plan-declared
# primary/secondary/guardrail role concept, unlike run()).


def _breakout_moments_row(
    metric_name: str, group_id: str, segment: str, n: int, mean: float, var: float
) -> dict:
    """Centered `group_summary` row with an extra `segment` breakout column,
    for `FakeMomentSource` -- unlike `MomentsSource`, it does no format
    conversion, so a caller must hand it already-centered rows."""
    from increment.estimation.armstats import centered_row_from_raw_sums

    sum_y = float(n) * mean
    sum_y2 = var * (n - 1) + sum_y**2 / float(n)
    return centered_row_from_raw_sums(
        {
            "experiment_id": "e",
            "metric": metric_name,
            "group_id": group_id,
            "segment": segment,
            "n": float(n),
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


def test_breakout_bh_discovery_matches_hand_bh_via_plan():
    """breakout()'s default correction="bh" runs one flat BH family across
    every (metric, arm, segment) cell the call produces, sourcing alpha
    from src.plan (undeclared here -> alpha=0.05, matching the old
    hardcoded default)."""
    from increment.breakout.estimates import BreakoutEstimates
    from tests.test_readouts_encouragement import FakeMomentSource

    m_a = MeanMetric(name="m_a", entity="user", fact="m_a")
    m_b = MeanMetric(name="m_b", entity="user", fact="m_b")
    specs = [
        ("m_a", "US", 10.0, 12.0),
        ("m_a", "CA", 10.0, 12.0),
        ("m_b", "US", 10.0, 10.05),
        ("m_b", "CA", 10.0, 10.05),
    ]
    rows = []
    for metric_name, segment, control_mean, treatment_mean in specs:
        rows.append(_breakout_moments_row(metric_name, "control", segment, 500, control_mean, 4.0))
        rows.append(
            _breakout_moments_row(metric_name, "treatment", segment, 500, treatment_mean, 4.0)
        )
    src = FakeMomentSource(
        rows,
        metrics=[m_a, m_b],
        capabilities={"total"},
        design=Randomized(control_group="control"),
        breakouts=("segment",),
        plan=AnalysisPlan(q=0.1),
    )
    assert src.context.plan.declared is True
    assert src.context.plan.alpha == pytest.approx(0.05)
    assert isinstance(src.context.plan.inference, FixedInference)

    result = readouts.breakout(src, "segment")
    assert isinstance(result, BreakoutEstimates)
    real = [r for r in result if r.excluded is None]
    assert len(real) == 4
    assert all(r.role == "exploratory" for r in real)

    p_values = [_p_value_from_estimate(r.lift) for r in real]
    selected_idx, bh_threshold = bh_select(p_values, 0.1)
    selected_set = set(selected_idx)
    assert [r.discovery for r in real] == [i in selected_set for i in range(len(real))]
    assert {(real[i].metric, real[i].dimension_value) for i in selected_set} == {
        ("m_a", "US"),
        ("m_a", "CA"),
    }

    for i, r in enumerate(real):
        lift = r.lift
        assert lift is not None
        if i in selected_set:
            assert lift.alpha == bh_threshold
        else:
            assert lift.alpha == pytest.approx(0.05)


def test_breakout_explicit_correction_still_inherits_plan_q():
    """Passing correction= explicitly must not pin q back to the
    function's own bare default -- an omitted q always inherits the
    plan's q, whether or not correction was passed explicitly."""
    from tests.test_readouts_encouragement import FakeMomentSource

    m_a = MeanMetric(name="m_a", entity="user", fact="m_a")
    m_b = MeanMetric(name="m_b", entity="user", fact="m_b")
    specs = [
        ("m_a", "US", 10.0, 12.0),
        ("m_a", "CA", 10.0, 12.0),
        ("m_b", "US", 10.0, 10.05),
        ("m_b", "CA", 10.0, 10.05),
    ]
    rows = []
    for metric_name, segment, control_mean, treatment_mean in specs:
        rows.append(_breakout_moments_row(metric_name, "control", segment, 500, control_mean, 4.0))
        rows.append(
            _breakout_moments_row(metric_name, "treatment", segment, 500, treatment_mean, 4.0)
        )
    src = FakeMomentSource(
        rows,
        metrics=[m_a, m_b],
        capabilities={"total"},
        design=Randomized(control_group="control"),
        breakouts=("segment",),
        plan=AnalysisPlan(q=0.01),
    )

    implicit = readouts.breakout(src, "segment")
    explicit = readouts.breakout(src, "segment", correction="bh")
    for result in (implicit, explicit):
        real = [r for r in result if r.excluded is None]
        assert {r.family_q for r in real} == {0.01}


def _p_value_from_estimate(lift) -> float:
    """Mirrors `tests/breakout/test_estimates.py`'s helper of the same
    purpose: `BreakoutEstimate` carries no `.p_value()` method."""
    from scipy.stats import norm as _norm_dist

    z = lift.log_mean / lift.log_se
    return float(2.0 * _norm_dist.sf(abs(z)))


@pytest.mark.slow
def test_breakout_av_plan_inherited_no_kwarg():
    from increment import SequentialCell

    specs = [MetricSpec(name=name, type="conversion") for name in ("m_a", "m_b")]
    cells = tuple(
        SequentialCell(
            metric=spec.name,
            group_id="treatment",
            segment=(("segment", segment),),
            family=True,
        )
        for spec in specs
        for segment in ("US", "CA")
    )
    plan = gaussian_plan(
        specs, law="bernoulli", cells=cells, unit="unit", group="arm", exposure_date="exposure"
    )
    table = pd.DataFrame(
        [
            {
                "unit": f"{segment}-{i:04d}-{arm}",
                "arm": arm,
                "segment": segment,
                "m_a": int(arm == "treatment" and segment == "US"),
                "m_b": int(arm == "treatment" and segment == "CA"),
            }
            for segment in ("US", "CA")
            for i in range(500)
            for arm in ("control", "treatment")
        ]
    )
    table["exposure"] = range(len(table))
    src = FrameTotalsSource.from_frame(
        table,
        unit="unit",
        group="arm",
        control="control",
        exposure_date="exposure",
        metrics=specs,
        plan=plan,
    )
    result = readouts.breakout(src, "segment", source_name="src_a")
    assert len(result) == 4
    logs = []
    for row in result:
        assert row.sequential_result is not None
        from increment.estimation.sequential_result import SequentialInferenceResult

        assert isinstance(row.sequential_result, SequentialInferenceResult)
        logs.append(row.sequential_result.log_e)
    selected = set(e_bh_select(logs, plan.q))
    assert selected
    assert [row.discovery for row in result] == [i in selected for i in range(4)]
    assert {row.dimension_value for row in result} == {"US", "CA"}
    assert {row.source for row in result} == {"src_a"}


def test_breakout_bonferroni_inherits_alpha_from_plan():
    """correction="bonferroni" still divides alpha by the segment count,
    but alpha itself now comes from src.plan (a declared alpha=0.10 here),
    not a removed call kwarg."""
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(name="rev", entity="user", fact="rev")
    rows = []
    for i, segment in enumerate(["US", "CA", "GB", "DE"]):
        rows.append(_breakout_moments_row("rev", "control", segment, 500, 10.0 + i, 4.0))
        rows.append(_breakout_moments_row("rev", "treatment", segment, 500, 11.0 + i, 4.0))
    plan = AnalysisPlan(alpha=0.10)
    src = FakeMomentSource(
        rows,
        metrics=[metric],
        capabilities={"total"},
        design=Randomized(control_group="control"),
        plan=plan,
        breakouts=("segment",),
    )
    assert src.context.plan.alpha == pytest.approx(0.10)

    result = readouts.breakout(src, "segment", correction="bonferroni")
    real = [r for r in result if r.excluded is None]
    assert len(real) == 4
    for r in real:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(1 - 0.10 / 4)
        assert r.role == "exploratory"
        assert r.discovery is None


@pytest.mark.parametrize(
    ("mechanism", "correction"),
    [
        ("randomized", "none"),
        ("randomized", "bonferroni"),
        ("randomized", "bh"),
        ("encouragement", "none"),
    ],
)
def test_breakout_rows_keep_their_source_on_every_branch(mechanism, correction):
    """``source_name`` tells same-name dimensions from distinct sources
    apart, so every branch stamps it: two sources' segment rows stay in
    separate heterogeneity groups instead of merging into one."""
    from increment.breakout.estimates import BreakoutEstimates
    from increment.breakout.heterogeneity import segment_heterogeneity
    from tests.test_readouts_encouragement import METRIC, FakeMomentSource, _design, _rows

    rows = []
    for i, segment in enumerate(["US", "CA", "GB"]):
        if mechanism == "randomized":
            rows.append(_breakout_moments_row("rev", "control", segment, 500, 10.0 + i, 4.0))
            rows.append(
                _breakout_moments_row("rev", "treatment", segment, 500, 11.0 + 1.5 * i, 4.0)
            )
        else:
            rows.extend({**row, "segment": segment} for row in _rows(n=4000, seed=21 + i))
    src = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"total"},
        design=Randomized(control_group="control")
        if mechanism == "randomized"
        else _design(one_sided=True),
        breakouts=("segment",),
    )
    estimates = BreakoutEstimates(
        row
        for source in ("src_a", "src_b")
        for row in readouts.breakout(src, "segment", source_name=source, correction=correction)
    )
    assert {row.source for row in estimates} == {"src_a", "src_b"}
    assert {row.source for row in segment_heterogeneity(estimates).summary} == {"src_a", "src_b"}


def test_registered_sequential_breakout_tests_a_plan_bound_margin_against_its_registered_null():
    """Only fixed-horizon breakouts refuse a margin: they build no shifted
    null. A registered sequential breakout registers the plan-bound margin's
    shifted null on every segment cell, so its rows carry that null and the
    margin's implied tail instead of being refused."""
    from fractions import Fraction

    from increment import SequentialCell

    specs = [MetricSpec(name="m_a", type="conversion", preferred_direction="increase")]
    cells = tuple(
        SequentialCell(
            metric="m_a",
            group_id="treatment",
            segment=(("segment", segment),),
            family=True,
            alternative="greater",
            null_lift=Fraction(-0.05),
        )
        for segment in ("US", "CA")
    )
    plan = gaussian_plan(
        specs,
        law="bernoulli",
        cells=cells,
        secondaries=[ExperimentMetric(metric="m_a", margin=0.05)],
        unit="unit",
        group="arm",
        exposure_date="exposure",
    )
    table = pd.DataFrame(
        [
            {"unit": f"{segment}-{i:03d}-{arm}", "arm": arm, "segment": segment, "m_a": i % 2}
            for segment in ("US", "CA")
            for i in range(100)
            for arm in ("control", "treatment")
        ]
    )
    table["exposure"] = range(len(table))
    src = FrameTotalsSource.from_frame(
        table,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        plan=plan,
        exposure_date="exposure",
    )

    result = readouts.breakout(src, "segment")
    assert {row.dimension_value for row in result} == {"US", "CA"}
    for row in result:
        assert row.inference == "always_valid"
        assert row.alternative == "greater"
        assert row.null_lift == pytest.approx(-0.05)


def test_breakout_design_none_raises_valueerror():
    """A source constructed without a design (design=None) cannot resolve
    breakout()'s randomized dispatch -- raises rather than crashing deep
    inside on a missing `.mechanism` attribute, mirroring run()'s own
    `design is None` check."""
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(name="rev", entity="user", fact="rev")
    rows = [_breakout_moments_row("rev", "control", "US", 50, 10.0, 4.0)]
    src = FakeMomentSource(rows, metrics=[metric], capabilities={"total"}, design=None)
    with pytest.raises(InvalidRequestError) as raised:
        readouts.breakout(src, "segment")
    assert raised.value.code == "readout.design.required"


def test_breakout_encouragement_role_exploratory_alpha_from_plan():
    """Under Encouragement, breakout()'s correction=None sentinel default
    resolves to "none" (preserving pre-existing Encouragement behavior,
    unchanged by the new "bh" default for randomized/observational
    designs) - only an EXPLICITLY requested non-"none" correction is
    still refused. breakout() sources alpha directly from src.plan (no
    split, no BH/FCR family machinery), and stamps every row
    role="exploratory"."""
    from increment.semantics.design import Encouragement
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(name="rev", entity="user", fact="rev")
    rng = np.random.default_rng(91)
    n = 1500
    rows = []
    for gid, encouraged in (("control", 0), ("treat", 1)):
        d = rng.binomial(1, 0.6, size=n) if encouraged else np.zeros(n)
        y = 10.0 + 2.0 * d + rng.normal(0, 2.0, size=n)
        yd = y * d
        from increment.estimation.armstats import centered_row_from_raw_sums

        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "segment": "seg1",
                    "n": n,
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )
    design = Encouragement.model_validate(
        {
            "control_group": "control",
            "uptake": {"fact": "help_click"},
            "exclusion_restriction": {
                "acknowledged": True,
                "justification": "unclicked button assumed inert",
            },
        }
    )
    plan = AnalysisPlan(alpha=0.10)
    src = FakeMomentSource(
        rows,
        metrics=[metric],
        capabilities={"total"},
        design=design,
        plan=plan,
        breakouts=("segment",),
    )

    result = readouts.breakout(src, "segment")
    assert result
    assert all(r.role == "exploratory" for r in result)
    assert all(r.discovery is None for r in result)
    itt = [r for r in result if r.estimand == "itt"]
    assert itt
    for r in itt:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(1 - 0.10)

    with pytest.raises(UnsupportedRequestError) as raised:
        readouts.breakout(src, "segment", correction="bh")
    assert raised.value.code == "readout.encouragement.correction"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"design": Randomized(control_group="control")},
        {"alpha": 0.05},
        {"inference": AlwaysValid(registration=registration("gaussian"))},
    ],
)
def test_breakout_removed_kwargs_raise_typeerror(kwargs):
    n = 200
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "revenue": np.concatenate(
                [_normal_column(10.0, 2.0, n, 1), _normal_column(10.5, 2.0, n, 2)]
            ),
        }
    )
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    with pytest.raises(TypeError):
        readouts.breakout(src, "variant", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"design": Randomized(control_group="control")},
        {"alpha": 0.05},
        {"alternative": "greater"},
        {"inference": AlwaysValid(registration=registration("gaussian"))},
        {"margins": {"revenue": 0.01}},
        {"null_lifts": {"revenue": 0.01}},
        {"margins_abs": {"revenue": 0.01}},
    ],
)
def test_asof_lift_removed_kwargs_raise_typeerror(kwargs):
    from datetime import date

    from increment.frame import from_unit_panel

    n = 200
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "ds": [date(2025, 1, 1)] * (2 * n),
            "revenue": np.concatenate(
                [_normal_column(10.0, 2.0, n, 1), _normal_column(10.5, 2.0, n, 2)]
            ),
        }
    )
    src = from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    with pytest.raises(TypeError):
        readouts.asof_lift(src, **kwargs)


def test_daily_removed_kwargs_raise_typeerror():
    from datetime import date

    from increment.frame import from_unit_panel

    n = 200
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "ds": [date(2025, 1, 1)] * (2 * n),
            "revenue": np.concatenate(
                [_normal_column(10.0, 2.0, n, 1), _normal_column(10.5, 2.0, n, 2)]
            ),
        }
    )
    src = from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    with pytest.raises(TypeError):
        readouts.daily(
            src,
            design=Randomized(control_group="control"),  # ty: ignore[unknown-argument]  - the removed kwarg IS the case under test
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_asof_lift_fixed_horizon_secondary_never_discovers_and_notes_run():
    """A fixed-horizon (undeclared inference) plan's asof_lift never
    stamps a discovery verdict on a secondary row, on any date - BH/FCR
    family selection only ever runs in run(), never on the per-date
    view - and every such row's note says so."""
    from datetime import date

    from increment.frame import from_unit_panel

    n = 400
    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    rng = np.random.default_rng(11)
    rows = []
    for arm, bump in (("control", 0.0), ("treatment", 1.5)):
        for i in range(n):
            rows.append(
                {
                    "user_id": f"{arm}_{i}",
                    "variant": arm,
                    "ds": day1,
                    "revenue": float(rng.normal(10.0 + bump, 2.0)),
                }
            )
            rows.append(
                {
                    "user_id": f"{arm}_{i}",
                    "variant": arm,
                    "ds": day2,
                    "revenue": float(rng.normal(10.0 + bump, 2.0)),
                }
            )
    df = pd.DataFrame(rows)
    plan = AnalysisPlan(secondaries=["revenue"])  # declared, fixed-horizon (no inference=)
    src = from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        design=Randomized(control_group="control"),
        plan=plan,
    )
    assert isinstance(src.context.plan.inference, FixedInference)
    results = readouts.asof_lift(src)
    assert results
    assert {r.ds for r in results} == {day1, day2}
    for r in results:
        assert r.role == "secondary"
        assert r.discovery is None
        assert r.note is not None
        assert "family verdicts at run()" in r.note


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.slow
def test_asof_lift_always_valid_secondary_discovery_differs_across_finalized_prefixes():
    from increment.sequential_state import SequentialSnapshot

    def with_cursor(source, revealed):
        current = source.sequential_snapshot()
        return source.adopt_sequential_snapshot(
            SequentialSnapshot.model_validate({**current.model_dump(), "reveal_cursor": revealed})
        )

    specs = [MetricSpec(name=name, type="conversion") for name in ("m_a", "m_b")]
    plan = gaussian_plan(specs, law="bernoulli", unit="unit", group="arm", exposure_date="exposure")
    early = pd.DataFrame(
        [
            {
                "unit": f"{i:08d}-{arm}",
                "arm": arm,
                "exposure": 2 * i + (arm == "treatment"),
                "m_a": i % 2,
                "m_b": i % 2,
            }
            for i in range(8)
            for arm in ("control", "treatment")
        ]
    )
    first = FrameTotalsSource.from_frame(
        early,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        plan=plan,
        exposure_date="exposure",
    )
    with_cursor(first, "2025-01-01")
    before = readouts.asof_lift(first)
    later = pd.DataFrame(
        [
            {
                "unit": f"{i:08d}-{arm}",
                "arm": arm,
                "exposure": 2 * i + (arm == "treatment"),
                "m_a": int(arm == "treatment"),
                "m_b": i % 2,
            }
            for i in range(8, 1508)
            for arm in ("control", "treatment")
        ]
    )
    second = FrameTotalsSource.from_frame(
        pd.concat([early, later]),
        unit="unit",
        group="arm",
        control="control",
        exposure_date="exposure",
        metrics=specs,
        plan=plan,
    )
    second.sequential_snapshot(previous=first.sequential_snapshot())
    with_cursor(second, "2025-01-02")
    after = readouts.asof_lift(second)
    assert not any(row.discovery for row in before)
    assert next(row for row in after if row.metric == "m_a").discovery is True
    assert next(row for row in after if row.metric == "m_b").discovery is False
    assert all(row.require_sequential_result().checkpoint.control.n == 1508 for row in after)


def _asof_row(metric: str, group_id: str, ds, n: int, mean: float, sd: float, seed: int) -> dict:
    """Centered `group_summary` row for `FakeMomentSource`'s as-of grain --
    same raw-sums-then-center construction `_rows()` in
    `tests/test_readouts_encouragement.py` uses, plus a `ds` stamp."""
    rng = np.random.default_rng(seed)
    y = rng.normal(mean, sd, n)
    row = centered_row_from_raw_sums(
        {
            "experiment_id": "e",
            "metric": metric,
            "group_id": group_id,
            "n": n,
            "sum_y": float(y.sum()),
            "sum_y2": float((y**2).sum()),
        }
    )
    row["ds"] = ds
    return row


def test_asof_lift_primary_guardrail_unassigned_role_dispatch():
    """Direct coverage for `asof_lift`'s per-date role dispatch (only
    incidentally covered elsewhere): a primary ("A"), a guardrail ("B",
    decrease-preferred, no margin), and an unnamed metric ("C") that a
    warehouse-path plan resolution genuinely leaves `role="unassigned"`
    (the frame path always defaults an unnamed metric to "secondary" --
    see the module docstring -- so this is built over `FakeMomentSource`
    at `path="warehouse"`, the only concrete `MomentSource` offering
    `"asof"` grain; see that class's docstring). "A" carries one
    treatment arm on day 1 and gains a second on day 2, so its cell
    alpha must come from THAT DATE's own arm count (0.05/1 on day 1,
    0.05/2 on day 2), not a global count observed once. "B"/"C" each
    carry two arms on both dates and must NOT split by arm count at
    all -- alpha_share stays the plan's full 0.05, unsplit, on every
    row regardless of role or date. Every row (any role here) carries
    discovery=None: family selection is a secondary-only concept."""
    from datetime import date

    from tests.test_readouts_encouragement import FakeMomentSource

    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    n = 500
    rows = [
        # A: primary. day 1 = 1 treatment arm; day 2 gains a second.
        _asof_row("A", "control", day1, n, 10.0, 2.0, 1),
        _asof_row("A", "t1", day1, n, 10.5, 2.0, 2),
        _asof_row("A", "control", day2, n, 10.0, 2.0, 3),
        _asof_row("A", "t1", day2, n, 10.5, 2.0, 4),
        _asof_row("A", "t2", day2, n, 10.3, 2.0, 5),
        # B: guardrail, decrease-preferred -- 2 arms on both dates.
        _asof_row("B", "control", day1, n, 5.0, 1.0, 6),
        _asof_row("B", "t1", day1, n, 4.9, 1.0, 7),
        _asof_row("B", "t2", day1, n, 5.05, 1.0, 8),
        _asof_row("B", "control", day2, n, 5.0, 1.0, 9),
        _asof_row("B", "t1", day2, n, 4.85, 1.0, 10),
        _asof_row("B", "t2", day2, n, 5.1, 1.0, 11),
        # C: unassigned -- also 2 arms on both dates.
        _asof_row("C", "control", day1, n, 3.0, 1.0, 12),
        _asof_row("C", "t1", day1, n, 3.1, 1.0, 13),
        _asof_row("C", "t2", day1, n, 2.95, 1.0, 14),
        _asof_row("C", "control", day2, n, 3.0, 1.0, 15),
        _asof_row("C", "t1", day2, n, 3.2, 1.0, 16),
        _asof_row("C", "t2", day2, n, 2.9, 1.0, 17),
    ]
    metric_a = MeanMetric(name="A", entity="user_id", fact="A", aggregation="avg_event")
    metric_b = MeanMetric(
        name="B",
        entity="user_id",
        fact="B",
        aggregation="avg_event",
        preferred_direction="decrease",
    )
    metric_c = MeanMetric(name="C", entity="user_id", fact="C", aggregation="avg_event")
    plan = AnalysisPlan(primary="A", guardrails=["B"])
    src = FakeMomentSource(
        rows,
        metrics=[metric_a, metric_b, metric_c],
        capabilities={"asof"},
        design=Randomized(control_group="control"),
        plan=plan,
    )
    assert src.context.plan.declared is True
    assert src.context.plan.procedures["A"].role == "primary"
    assert src.context.plan.procedures["B"].role == "guardrail"
    assert src.context.plan.procedures["C"].role == "unassigned"

    results = readouts.asof_lift(src)
    assert results
    for r in results:
        assert r.discovery is None  # (d): no role here ever runs family selection

    # (a) primary: cell alpha divided by THAT DATE's own non-control arm
    # count, not a global count.
    a_day1 = [r for r in results if r.metric == "A" and r.ds == day1]
    assert {r.group_id for r in a_day1} == {"t1"}
    for r in a_day1:
        assert r.role == "primary"
        assert r.require_lift().level == pytest.approx(1.0 - 0.05 / 1)

    a_day2 = [r for r in results if r.metric == "A" and r.ds == day2]
    assert {r.group_id for r in a_day2} == {"t1", "t2"}
    for r in a_day2:
        assert r.role == "primary"
        assert r.require_lift().level == pytest.approx(1.0 - 0.05 / 2)

    # (b) guardrail: one-sided tail off its preferred_direction, full
    # alpha_share unsplit despite 2 arms on both dates.
    b_rows = [r for r in results if r.metric == "B"]
    assert {r.ds for r in b_rows} == {day1, day2}
    assert {r.group_id for r in b_rows} == {"t1", "t2"}
    for r in b_rows:
        assert r.role == "guardrail"
        assert r.alternative == "less"
        assert r.require_lift().level == pytest.approx(1.0 - 2 * 0.05)

    # (c) unassigned: two-sided, full alpha_share unsplit despite 2 arms
    # on both dates.
    c_rows = [r for r in results if r.metric == "C"]
    assert {r.ds for r in c_rows} == {day1, day2}
    assert {r.group_id for r in c_rows} == {"t1", "t2"}
    for r in c_rows:
        assert r.role == "unassigned"
        assert r.alternative == "two-sided"
        assert r.require_lift().level == pytest.approx(1.0 - 0.05)


def test_asof_lift_undeclared_plan_stamps_role_none_on_every_date():
    """Companion to `test_declared_plan_leaves_an_unnamed_metric_unassigned`:
    no `plan=` at all (`src.plan.declared is False`) stamps `role=None`
    on every row of every as-of date -- `None` is reserved for an
    undeclared plan, distinct from the real string role="unassigned" a
    DECLARED-but-unnamed metric gets."""
    from datetime import date

    from tests.test_readouts_encouragement import FakeMomentSource

    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    n = 500
    rows = [
        _asof_row("revenue", "control", day1, n, 10.0, 2.0, 21),
        _asof_row("revenue", "treatment", day1, n, 10.5, 2.0, 22),
        _asof_row("revenue", "control", day2, n, 10.0, 2.0, 23),
        _asof_row("revenue", "treatment", day2, n, 10.6, 2.0, 24),
    ]
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    src = FakeMomentSource(
        rows,
        metrics=[metric],
        capabilities={"asof"},
        design=Randomized(control_group="control"),
    )
    assert src.context.plan.declared is False

    results = readouts.asof_lift(src)
    assert results
    assert {r.ds for r in results} == {day1, day2}
    for r in results:
        assert r.role is None
        assert r.discovery is None


def test_breakout_correction_none_uses_compiled_policy_q():
    from tests.test_readouts_encouragement import FakeMomentSource

    metric = MeanMetric(name="m", entity="user", fact="m")
    rows = [
        _breakout_moments_row("m", "control", "US", 500, 10.0, 4.0),
        _breakout_moments_row("m", "treatment", "US", 500, 12.0, 4.0),
        _breakout_moments_row("m", "control", "CA", 500, 10.0, 4.0),
        _breakout_moments_row("m", "treatment", "CA", 500, 12.0, 4.0),
    ]
    source = FakeMomentSource(
        rows,
        metrics=[metric],
        capabilities={"total"},
        design=Randomized(control_group="control"),
        plan=AnalysisPlan(q=0.2),
        breakouts=("segment",),
    )

    result = readouts.breakout(source, "segment")
    real = [r for r in result if r.excluded is None]
    assert real
    assert {r.family_q for r in real} == {0.2}


@pytest.mark.filterwarnings("ignore:metric .* is open-ended:UserWarning")
@pytest.mark.parametrize("kind", ["fixed", "always_valid"])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_directional_secondary_fcr_dispatches_on_returned_inference(kind, alternative):
    import math

    from scipy.stats import norm, t

    from increment.estimation.inference import normal_posterior

    n = 800
    sign = 1 if alternative == "greater" else -1
    specs = [MetricSpec(name="revenue", type="conversion" if kind == "always_valid" else "mean")]
    if kind == "always_valid":
        values = np.concatenate(
            (
                np.tile([0, 1], n // 2),
                np.tile([1, 1, 1, 0] if sign > 0 else [1, 0, 0, 0], n // 4),
            )
        )
    else:
        values = np.concatenate(
            (_normal_column(10, 2, n, 41), _normal_column(10 + sign * 2, 2, n, 42))
        )
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "exposure": list(range(2 * n)),
            "revenue": values,
        }
    )
    from fractions import Fraction

    from increment import SequentialCell
    from tests.test_sequential_public_sources import gaussian_plan

    plan = AnalysisPlan(alternative=alternative)
    if kind == "always_valid":
        plan = gaussian_plan(
            specs,
            law="bernoulli",
            cells=(
                SequentialCell(
                    metric="revenue",
                    group_id="treatment",
                    family=True,
                    alternative=alternative,
                ),
            ),
            exposure_date="exposure",
        ).model_copy(update={"alternative": alternative})
    src = FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        exposure_date="exposure",
        metrics=specs,
        plan=plan,
    )
    (row,) = readouts.run(src)
    assert row.inference == kind
    assert row.discovery is True
    if kind == "always_valid":
        result = row.require_sequential_result()
        assert result.bounds.alternative == alternative
        # A single-cell secondary family is always fully selected (R == m), so
        # e-BH reinverts it at min(q * R / m, nominal_alpha) -- here q (0.1)
        # exceeds the plan's nominal alpha, so the cap binds at exactly 1/20:
        # the decimal the plan declared, not its raw float value.
        from increment.estimation.decision_types import exact_fraction

        assert row.family_threshold == row.family_q
        assert row.family_threshold is not None
        assert row.family_nominal_alpha is not None
        assert result.decision_alpha == min(
            Fraction(row.family_threshold), exact_fraction(row.family_nominal_alpha)
        )
        assert result.decision_alpha == Fraction(1, 20)
        assert result.checkpoint.cell.alternative == alternative
        return
    e = row.require_lift()
    assert e.open_side == ("upper" if alternative == "greater" else "lower")
    assert e.log_mean is not None
    assert e.log_se is not None
    posterior = normal_posterior(e.log_mean, e.log_se)
    critical = (
        norm.isf(e.alpha) if row.reference_kind == "normal" else t.isf(e.alpha, row.reference_df)
    )
    mu = e.log_mean if row.dof is not None else posterior.mu
    sigma = e.log_se if row.dof is not None else posterior.sigma
    half_width = critical * sigma
    expected = math.expm1(mu - sign * half_width)
    assert (e.lb if alternative == "greater" else e.ub) == pytest.approx(expected, rel=1e-12)


def test_explicit_none_prior_matches_prior_free_analysis_family_decision():
    """A per-call prior reset uses the effective prior, not stale declaration metadata."""
    from tests.analysis_factory import lift_rows

    frame = pa.table(
        {
            "u": range(40),
            "g": ["control"] * 20 + ["treatment"] * 20,
            "y": [10 + i % 3 for i in range(20)] + [20 + i % 3 for i in range(20)],
        }
    )
    prior_bound = Analysis.from_unit_summary(
        frame,
        unit="u",
        group="g",
        control="control",
        metrics=[MetricSpec(name="y", prior=Normal(mu=0.0, sigma=0.1))],
        plan=AnalysisPlan(secondaries=["y"]),
    )
    prior_free = Analysis.from_unit_summary(
        frame,
        unit="u",
        group="g",
        control="control",
        metrics=[MetricSpec(name="y")],
        plan=AnalysisPlan(secondaries=["y"]),
    )

    try:
        inherited_before = lift_rows(prior_bound.run())[0]
        reset_first = lift_rows(prior_bound.run(prior=None))[0]
        reset_second = lift_rows(prior_bound.run(prior=None))[0]
        expected = lift_rows(prior_free.run())[0]
        inherited_after = lift_rows(prior_bound.run())[0]

        assert reset_first.require_lift().value == pytest.approx(0.9132420091324197)
        for actual in (reset_first, reset_second):
            assert actual.discovery is True
            assert actual.family_axes == ("metric", "arm")
            for field in ("value", "lb", "ub"):
                wanted = getattr(expected.require_lift(), field)
                assert wanted is not None
                assert getattr(actual.require_lift(), field) == pytest.approx(wanted)
        for actual in (inherited_before, inherited_after):
            assert actual.discovery is None
            assert actual.family_axes is None
        for field in ("value", "lb", "ub"):
            assert getattr(inherited_after.require_lift(), field) == pytest.approx(
                getattr(inherited_before.require_lift(), field)
            )
    finally:
        prior_bound.close()
        prior_free.close()
