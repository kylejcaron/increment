"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from increment import Analysis
from increment.breakout.estimates import BreakoutEstimate
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.semantics.models import AnalysisPlan, Definitions, MultiplicitySpec
from tests.analysis_factory import make_analysis, make_analysis_like
from tests.sequential_cases import registration
from tests.warning_codes import warning_codes


def _analysis_with_country_breakout(con, *, unbounded_retention: bool = False):
    """Two countries with opposite revenue lifts and an optional open retention band."""
    if "breakout_run_events" not in con.list_tables():
        # Exposure: 2 control + 2 treatment units per country.
        exposure_groups = {
            "US": {"control": ["bu1", "bu2"], "treatment": ["bu3", "bu4"]},
            "CA": {"control": ["bu5", "bu6"], "treatment": ["bu7", "bu8"]},
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                # first_exposures' dedup semi-joins on (unit_id, experiment_id): a NULL experiment_id never matches itself (SQL NULL = NULL is not true), so every row needs the real experiment name.
                "experiment_id": "breakout_run_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        # Revenue (mean-metric fact), one purchase per unit: US control mean=10, US treatment mean=15 (+50% lift); CA control mean=20, CA treatment mean=15 (-25% lift) - opposite-signed so cross-segment contamination flips a sign.
        revenue_by_unit = {
            "bu1": 9.0,
            "bu2": 11.0,
            "bu3": 14.0,
            "bu4": 16.0,
            "bu5": 19.0,
            "bu6": 21.0,
            "bu7": 14.0,
            "bu8": 16.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in revenue_by_unit.items()
        ]
        con.create_table("breakout_run_events", obj=exposure_rows + purchase_rows)

    metric_definitions: list[dict[str, object]] = [
        {
            "type": "mean",
            "name": "revenue",
            "entity": "user_id",
            "fact": "purchase",
            "aggregation": "sum",
        }
    ]
    if unbounded_retention:
        metric_definitions.append(
            {
                "type": "retention",
                "name": "unbounded",
                "entity": "user_id",
                "fact": "purchase",
                "threshold_days": 7,
            }
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM breakout_run_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": metric_definitions,
            "experiments": [
                {
                    "name": "breakout_run_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "plan": {"secondaries": [metric["name"] for metric in metric_definitions]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


@pytest.fixture(scope="module")
def country_breakout_estimates(con):
    """Plain ``run_breakout()`` on the country fixture, computed once;
    the tests below assert different facets of one deterministic readout."""
    return _analysis_with_country_breakout(con).run_breakout()


@pytest.fixture(scope="module")
def country_daily_lift_by_segment(con):
    """Plain ``run_daily_lift(dimension="country")`` on the country fixture,
    computed once and shared the same way."""
    return _analysis_with_country_breakout(con).run_daily_lift(dimension="country")


def test_run_breakout_returns_per_segment_estimates(country_breakout_estimates):
    """Analysis.run_breakout() wires breakout_summaries into
    the run_breakout wrapper and returns BreakoutEstimate rows for every
    declared breakout x metric, correctly attributed per segment (not
    cross-contaminated - US has positive lift, CA has negative)."""
    results = country_breakout_estimates

    assert len(results) == 2  # 1 metric x 1 method x 1 treatment arm x 2 segments
    for r in results:
        assert isinstance(r, BreakoutEstimate)
        assert r.metric == "revenue"
        assert r.group_id == "treatment"
        assert r.dimension == "country"
        assert r.source == "events"  # the resolved FactSource name

    by_segment = {r.dimension_value: r for r in results}
    assert set(by_segment) == {"US", "CA"}
    assert by_segment["US"].require_lift().value > 0, (
        "US treatment (15) > control (10) -- expected positive lift"
    )
    assert by_segment["CA"].require_lift().value < 0, (
        "CA treatment (15) < control (20) -- expected negative lift"
    )


def _configured_country_breakout_analysis(con, correction: Literal["none", "bonferroni", "bh"]):
    from increment.semantics.models import ExperimentMetric, MethodSpec, NormalPriorSpec

    analysis = _analysis_with_country_breakout(con)
    return make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="revenue",
                            decision_method=MethodSpec(name="declared"),
                            prior=NormalPriorSpec(mu=0.0, sigma=0.01),
                        )
                    ],
                    view_multiplicity=MultiplicitySpec(correction=correction),
                )
            }
        ),
    )


def test_native_breakout_ignores_incompatible_declared_method(con):
    """Native breakout prevalidation must ignore observational-only metric
    declarations on a randomized design when no call-wide method is set."""
    from increment.semantics.models import ExperimentMetric, MethodSpec

    analysis = _analysis_with_country_breakout(con)
    configured = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="revenue",
                            decision_method=MethodSpec(name="iptw"),
                        )
                    ]
                )
            }
        ),
    )

    with pytest.raises(InvalidRequestError) as raised:
        configured.run_breakout()
    assert raised.value.code == "estimation.engine.method_name_observational"


@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
def test_native_breakout_uses_callwide_methods_not_declared_methods(con, correction):
    from increment.estimation.engine import Method

    analysis = _configured_country_breakout_analysis(con, correction)

    if correction == "bh":
        with pytest.raises(InvalidRequestError) as raised:
            analysis.run_breakout()
        assert raised.value.code == "readout.breakout_correction_bh"
    else:
        results = analysis.run_breakout()
        assert results
        assert {row.method for row in results} == {"declared"}

    if correction == "bh":
        with pytest.raises(InvalidRequestError) as raised:
            analysis.run_breakout(decision_method=Method(name="call-wide"))
        assert raised.value.code == "readout.breakout_correction_bh"
    else:
        explicit = analysis.run_breakout(decision_method=Method(name="call-wide"))
        assert explicit
        assert {row.method for row in explicit} == {"call-wide"}


@pytest.mark.parametrize("correction", ["none", "bonferroni"])
def test_native_breakout_ignores_declared_prior(con, correction):
    from increment.estimation import Normal

    configured = _configured_country_breakout_analysis(con, correction)
    baseline = make_analysis_like(
        configured,
        experiment=configured.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=["revenue"],
                    view_multiplicity=MultiplicitySpec(correction=correction),
                )
            }
        ),
    )

    declared_results = configured.run_breakout()
    baseline_results = baseline.run_breakout()

    def keyed_lifts(results):
        return {
            (row.metric, row.group_id, row.dimension_value, row.method): row.require_lift().value
            for row in results
            if row.lift is not None
        }

    declared_lifts = keyed_lifts(declared_results)
    baseline_lifts = keyed_lifts(baseline_results)
    assert {(key[0], key[1], key[2]) for key in declared_lifts} == {
        (key[0], key[1], key[2]) for key in baseline_lifts
    }
    assert list(declared_lifts.values()) != pytest.approx(list(baseline_lifts.values()))

    prior_results = baseline.run_breakout(prior=Normal(mu=0.0, sigma=0.01))
    prior_lifts = keyed_lifts(prior_results)
    assert prior_lifts.keys() == baseline_lifts.keys()
    assert any(prior_lifts[key] != pytest.approx(baseline_lifts[key]) for key in baseline_lifts)


def test_run_breakout_plan_declared_multiplicity_controls_level(con):
    """Breakout correction and alpha are read from the resolved plan."""
    analysis = _analysis_with_country_breakout(con)

    corrected = make_analysis_like(
        analysis,
        plan=AnalysisPlan(
            secondaries=["revenue"],
            view_multiplicity=MultiplicitySpec(correction="bonferroni"),
        ),
    )
    default_alpha = corrected.run_breakout()
    assert len(default_alpha) == 2
    for r in default_alpha:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(1 - 0.05 / 2)

    custom_alpha = make_analysis_like(
        analysis,
        plan=AnalysisPlan(
            secondaries=["revenue"],
            alpha=0.1,
            view_multiplicity=MultiplicitySpec(correction="bonferroni"),
        ),
    ).run_breakout()
    assert len(custom_alpha) == 2
    for r in custom_alpha:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(1 - 0.1 / 2)

    uncorrected = make_analysis_like(
        analysis,
        plan=AnalysisPlan(
            secondaries=["revenue"],
            view_multiplicity=MultiplicitySpec(correction="none"),
        ),
    ).run_breakout()
    assert len(uncorrected) == 2
    for r in uncorrected:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(0.95)


def test_run_breakout_default_correction_runs_bh_family(country_breakout_estimates):
    """`Analysis.run_breakout()`'s `correction` default is `None`, which
    delegates to `readouts.breakout()`'s own default (`"bh"` under a
    randomized design) instead of being pinned to the old `"none"` -- a
    plain `run_breakout()` call now stamps a real `discovery` bool on
    every row (the BH family actually ran), not the all-`None` a
    forced `correction="none"` would leave."""
    results = country_breakout_estimates
    assert len(results) == 2
    assert all(r.discovery is not None for r in results)
    assert all(isinstance(r.discovery, bool) for r in results)


def _panel_with_borderline_country_segment():
    """3-country panel (US/CA/MX): US is a strong lift, CA is null, and
    MX (seed/shift chosen empirically) sits right on the BH boundary --
    not selected at q=0.05, selected at q=0.10 -- so a test can prove
    `q=` actually reaches the underlying selection instead of being
    silently dropped between the facade and `readouts.breakout`."""
    import pandas as pd

    rng = np.random.default_rng(2)
    rows = []
    n = 15
    for country, shift in (("US", 4.0), ("CA", 0.0), ("MX", 0.5)):
        for i in range(n):
            rows.append(
                {
                    "user_id": f"c_{country}_{i}",
                    "variant": "control",
                    "day": "2026-01-01",
                    "country": country,
                    "revenue": 10.0 + rng.normal(0, 0.5),
                }
            )
            rows.append(
                {
                    "user_id": f"t_{country}_{i}",
                    "variant": "treatment",
                    "day": "2026-01-01",
                    "country": country,
                    "revenue": 10.0 + shift + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_run_breakout_declared_bh_q_controls_selection():
    """A declared breakout BH policy controls the family q level."""
    table = _panel_with_borderline_country_segment()
    common = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": {"revenue": "mean"},
        "breakouts": ["country"],
    }
    tight = Analysis.from_unit_panel(
        table,
        plan=AnalysisPlan(
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.05),
        ),
        **common,  # ty: ignore[invalid-argument-type]
    )
    wide = Analysis.from_unit_panel(
        table,
        plan=AnalysisPlan(
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.10),
        ),
        **common,  # ty: ignore[invalid-argument-type]
    )

    tight_discovery = {r.dimension_value: r.discovery for r in tight.run_breakout()}
    assert tight_discovery == {"US": True, "CA": False, "MX": False}
    wide_discovery = {r.dimension_value: r.discovery for r in wide.run_breakout()}
    assert wide_discovery == {"US": True, "CA": False, "MX": True}


def test_run_breakout_bh_family_survives_a_degenerate_segment_cell():
    """run_breakout's own BH correction calls the SAME select_family
    path as run(); a segment cell breakout pre-drops as
    excluded='nonpositive_mean' (raw non-positive mean, caught before
    estimate_lift runs) must not abort the whole breakout for every
    other segment/metric -- the same non-rejection rule that covers
    estimation.engine.lift_guard now also covers breakout's own
    breakout.nonpositive_mean/breakout.zero_variance prescreen codes."""
    rng = np.random.default_rng(11)
    n = 100
    rows = []
    for country in ("US", "CA", "GB"):
        for group, shift in (("control", 0.0), ("treatment", 3.0)):
            for i in range(n):
                refunds = (
                    0.0
                    if (country == "GB" and group == "treatment")
                    else float(rng.normal(1.0, 0.2))
                )
                rows.append(
                    {
                        "user_id": f"{country}_{group}_{i}",
                        "variant": group,
                        "day": "2026-01-01",
                        "country": country,
                        "revenue": 10.0 + shift + rng.normal(0, 0.5),
                        "refunds": max(refunds, 0.0),
                    }
                )
    table = pd.DataFrame(rows)
    analysis = Analysis.from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean", "refunds": "mean"},
        breakouts=["country"],
        plan=AnalysisPlan(
            secondaries=["revenue", "refunds"],
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.10),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = analysis.run_breakout()
    degenerate_rows = [r for r in results if r.metric == "refunds" and r.dimension_value == "GB"]
    assert degenerate_rows and all(
        r.excluded == "nonpositive_mean" and r.lift is None for r in degenerate_rows
    )
    healthy_rows = [
        r
        for r in results
        if r.metric == "revenue" and r.dimension_value == "US" and r.excluded is None
    ]
    assert healthy_rows  # the family completed instead of aborting for GB/refunds


def test_run_breakout_bh_family_survives_a_tiny_segment_extreme_ratio_cell():
    """A tiny segment trips estimate_lift's delta-method guard (combined
    log-scale SE >= 0.5), which breakout keys as
    excluded='extreme_ratio'. run() treats the same LiftGuardError as a
    non-rejection; run_breakout must too, not abort the BH family."""
    rng = np.random.default_rng(12)
    rows = []
    for country in ("US", "CA"):
        for group, shift in (("control", 0.0), ("treatment", 1.0)):
            for i in range(500):
                rows.append(
                    {
                        "user_id": f"{country}_{group}_{i}",
                        "variant": group,
                        "day": "2026-01-01",
                        "country": country,
                        "revenue": 10.0 + shift + rng.normal(0, 5.0),
                    }
                )
    # Three units per arm, positive means, non-zero variance: combined
    # log-scale SE is 0.601 >= 0.5, so only the delta-method guard fires.
    for group, values in (("control", (1.0, 5.0, 9.0)), ("treatment", (2.0, 6.0, 10.0))):
        for i, value in enumerate(values):
            rows.append(
                {
                    "user_id": f"GB_{group}_{i}",
                    "variant": group,
                    "day": "2026-01-01",
                    "country": "GB",
                    "revenue": value,
                }
            )
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
        plan=AnalysisPlan(
            secondaries=["revenue"],
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.10),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = analysis.run_breakout()
    gb = [r for r in results if r.dimension_value == "GB"]
    assert gb and all(r.excluded == "extreme_ratio" for r in gb)
    assert any(
        r.dimension_value == "US" and r.excluded is None and r.family_q is not None for r in results
    )


def test_breakout_propagates_a_cuped_nonpositive_mean_row_unmodified(con):
    """The previously-caught gap: a CUPED-adjusted mean that turns
    non-positive in exactly one segment must reach breakout as a real
    row (relative_unavailable_reason='nonpositive_arm_mean', a real
    abs_diff) -- NOT a warning-classified exclusion, and unlike the raw
    case above, this one is NOT pre-screened (breakout's own partition
    step only inspects the RAW arm mean, not a CUPED adjustment applied
    later inside estimate_lift), so it reaches estimate_lift and gets
    the row-based fix directly.

    Only run_breakout's own from_definitions/warehouse construction
    reaches this: from_unit_summary/from_moments refuse run_breakout
    by name, and from_unit_panel refuses a declared CUPED covariate --
    so this builds a definitions-backed DuckDB fixture directly, the
    same bypass-of-load() pattern _analysis_with_country_breakout uses
    above, with an added pre-period purchase fact for the covariate.

    GB's treatment arm carries a raw post-period mean of 1.025 (positive,
    passes breakout's raw pre-screen) but a pre-period covariate mean
    (1001.5) two orders of magnitude past its own control arm's (50),
    so the fitted CUPED slope drags its adjusted mean negative while
    every other cell -- GB control and both US arms -- stays positive.
    Verified directly against the unmodified `increment.estimation.cuped
    .fit_cuped` on these exact sums (independent of the guard change
    this test targets, which only touches what happens after the
    adjusted mean is known):
    `fit_cuped([gb_control, gb_treatment]).theta == 0.03`,
    pooled covariate mean `525.75`, giving adjusted means
    `control: 9.75 - 0.03*(50 - 525.75) == 24.0225` (positive) and
    `treatment: 1.025 - 0.03*(1001.5 - 525.75) == -13.2475` (<= 0),
    each with a strictly positive residual variance (0.4167 / 0.0343 --
    not the separate zero-variance guard). The US segment's arms (raw
    and adjusted means 10.0 and 13.0, residual variance 0.0267 each)
    stay ordinary and positive throughout, confirming only
    GB/treatment/adjusted_metric is degenerate.
    """
    from increment import Method

    exposure_ts = datetime(2025, 6, 10, 9, 0, 0)
    pre_ts = datetime(2025, 5, 28, 9, 0, 0)  # inside [2025-05-27, exposure_ts)
    post_ts = datetime(2025, 6, 11, 9, 0, 0)  # strictly after exposure_ts

    # (country, group_id) -> (post-period y values, pre-period x values).
    # GB/treatment's x values run two orders of magnitude past its own
    # control arm's while its y values stay an ordinary positive mean;
    # every other cell keeps x and y on the same scale.
    arms = {
        ("GB", "control"): ([9.0, 9.5, 10.0, 10.5], [50.0, 50.0, 50.0, 50.0]),
        ("GB", "treatment"): ([0.9, 1.0, 1.3, 0.9], [1000.0, 1001.0, 1002.0, 1003.0]),
        ("US", "control"): ([9.8, 10.5, 9.5, 10.2], [50.0, 51.0, 49.0, 50.0]),
        ("US", "treatment"): ([12.8, 13.5, 12.5, 13.2], [50.0, 51.0, 49.0, 50.0]),
    }

    table_name = "cuped_nonpositive_breakout_events"
    if table_name not in con.list_tables():
        exposure_rows: list[dict] = []
        purchase_rows: list[dict] = []
        for (country, group_id), (ys, xs) in arms.items():
            for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
                unit_id = f"{country}_{group_id}_{i}"
                exposure_rows.append(
                    {
                        "user_id": unit_id,
                        "ts": exposure_ts,
                        "event": "page_view",
                        "group_id": group_id,
                        "country_code": country,
                        "revenue": None,
                        "experiment_id": "cuped_nonpositive_breakout_exp",
                    }
                )
                purchase_rows.append(
                    {
                        "user_id": unit_id,
                        "ts": pre_ts,
                        "event": "purchase",
                        "group_id": None,
                        "country_code": country,
                        "revenue": x,
                        "experiment_id": None,
                    }
                )
                purchase_rows.append(
                    {
                        "user_id": unit_id,
                        "ts": post_ts,
                        "event": "purchase",
                        "group_id": None,
                        "country_code": country,
                        "revenue": y,
                        "experiment_id": None,
                    }
                )
        con.create_table(table_name, obj=pd.DataFrame(exposure_rows + purchase_rows))

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table_name}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "adjusted_metric",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "cuped_nonpositive_breakout_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "n_pre_periods": 14,
                    "plan": {"secondaries": ["adjusted_metric"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    results = analysis.run_breakout(
        decision_method=Method(name="cuped", variance_reduction="cuped")
    )
    cuped_row = next(
        r
        for r in results
        if r.metric == "adjusted_metric" and r.relative_unavailable_reason is not None
    )
    assert cuped_row.dimension_value == "GB"
    assert cuped_row.group_id == "treatment"
    assert cuped_row.relative_unavailable_reason == "nonpositive_arm_mean"
    assert cuped_row.abs_diff is not None
    assert cuped_row.excluded is None  # a real row, not an exclusion
    other_rows = [r for r in results if r.metric == "adjusted_metric" and r is not cuped_row]
    assert other_rows and all(
        r.relative_unavailable_reason is None and r.excluded is None for r in other_rows
    ), "every other GB/US cell must stay a healthy, non-degenerate row"


@pytest.mark.slow
def test_run_breakout_registered_raw_frame_results_roundtrip():
    import pyarrow as pa

    from increment import SequentialCell
    from increment.breakout.estimates import BreakoutEstimate
    from increment.frame import MetricSpec
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [MetricSpec(name="revenue", type="conversion", window_days=7)]
    cells = tuple(
        SequentialCell(
            metric="revenue", group_id="treatment", segment=(("country", segment),), family=True
        )
        for segment in ("US", "CA")
    )
    plan = gaussian_plan(
        specs,
        law="bernoulli",
        cells=cells,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
    )
    rows = [
        {
            "unit": f"{segment}-{arm}-{i}",
            "arm": arm,
            "country": segment,
            "day": date(2025, 1, 1),
            "exposed": date(2025, 1, 1),
            "revenue": int(i % 32 < center),
        }
        for segment, control, treatment in (("US", 10, 15), ("CA", 20, 15))
        for arm, center in (("control", control), ("treatment", treatment))
        for i in range(512)
    ]
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        control="control",
        date="day",
        exposure_date="exposed",
        observation_end=date(2025, 1, 20),
        metrics=specs,
        breakouts=["country"],
        plan=plan,
    )
    analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 15))
    result = analysis.run_breakout()
    assert {row.dimension_value: row.require_lift().value for row in result} == {
        "US": 0.5,
        "CA": -0.25,
    }
    assert all(row.discovery for row in result)
    for row in result:
        assert row.sequential_result is not None
        assert BreakoutEstimate.model_validate_json(row.model_dump_json()) == row


def test_run_breakout_removed_kwargs_raise_typeerror(con):
    """alpha=/inference= are no longer Analysis.run_breakout() call-time
    overrides - readouts.breakout() reads both off the source's own
    declared plan now, matching run()'s cutover."""
    from increment import AlwaysValid

    analysis = _analysis_with_country_breakout(con)
    with pytest.raises(TypeError):
        analysis.run_breakout(alpha=0.1)
    with pytest.raises(TypeError):
        analysis.run_breakout(inference=AlwaysValid(registration=registration()))
    with pytest.raises(TypeError):
        analysis.run_breakout(correction="bonferroni")
    with pytest.raises(TypeError):
        analysis.run_breakout(q=0.05)


def test_run_breakout_refuses_a_declared_margin_metric(con):
    """readouts.breakout() refuses any metric declaring
    Metric.margin/margin_abs with NotImplementedError - breakout rows are
    tested two-sided vs 0 and a per-segment shifted null is not built.
    This pins that refusal on Analysis.run_breakout()'s
    definitions-backed path, routed through readouts.breakout().
    model_copy pattern mirrors
    test_analysis_run_native_declared_margin_abs_reaches_the_estimate,
    applied to the breakout path instead of run()."""
    analysis = _analysis_with_country_breakout(con)
    analysis = make_analysis_like(
        analysis,
        [
            # margin_abs requires an explicitly-declared direction (the semantics
            # layer refuses a defaulted one) -- declare it so this test exercises
            # breakout()'s own margin refusal, not that upstream validation.
            m.model_copy(update={"margin_abs": 1.0, "preferred_direction": "decrease"})
            if m.name == "revenue"
            else m
            for m in analysis.metrics
        ],
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run_breakout()
    assert raised.value.code == "readout.margin.breakout"


def test_run_breakout_empty_when_no_breakouts_declared(con):
    """analysis.run_breakout() returns [] for an experiment with no
    declared breakouts - mirrors breakout_summaries' own empty-dict
    contract, and confirms run_breakout doesn't require breakouts to be
    present to avoid crashing. `pricing_tier_test` is the experiment with
    no breakouts declared here - `new_onboarding_v2` gained a `country`
    breakout for the example notebook (see examples/breakout.py)."""
    analysis = Analysis(
        experiment_name="pricing_tier_test",
        definitions_path="examples/definitions/",
        con=con,
    )
    assert analysis.experiment.breakouts == ()

    assert analysis.run_breakout() == []

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_breakout(metrics=["not_declared"])
    assert raised.value.code == "facade.analysis_config.unknown_metric_declared"


def test_panel_sql_and_summary_sql_with_breakouts_include_segment_sql(con):
    """panel_sql(breakouts=...)/summary_sql(breakouts=...), called
    directly on ``Analysis`` (not the lower-level query
    builders, and not ``run_breakout``/``breakout_summaries``, which are
    tested separately), include one valid, executable SQL entry per
    (metric, breakout) pair under the expected
    ``f"{metric.name}:{breakout.property}:{source_name}"`` key."""
    analysis = _analysis_with_country_breakout(con)
    breakouts = analysis.experiment.breakouts
    key = "revenue:country:events"

    panel_sql = analysis.panel_sql(breakouts=breakouts)
    assert key in panel_sql
    panel_df = con.sql(panel_sql[key]).execute()
    assert "country" in panel_df.columns

    summary_sql = analysis.summary_sql(breakouts=breakouts)
    assert key in summary_sql
    summary_df = con.sql(summary_sql[key]).execute()
    assert "country" in summary_df.columns
    assert set(summary_df["country"]) == {"US", "CA"}


def _analysis_with_missing_control_segment(con):
    """Build an ``Analysis`` with a 3-segment country breakout
    where one segment (MX) has ONLY treatment units - no control arm at
    all - to exercise ``run_breakout``'s per-segment "no control arm,
    skip the segment" handling without losing the other 2 segments'
    already-computed estimates."""
    if "breakout_missing_control_events" not in con.list_tables():
        exposure_groups = {
            "US": {"control": ["mu1", "mu2"], "treatment": ["mu3", "mu4"]},
            "CA": {"control": ["mu5", "mu6"], "treatment": ["mu7", "mu8"]},
            "MX": {"treatment": ["mu9", "mu10"]},  # deliberately no control group
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                "experiment_id": "breakout_missing_control_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        revenue_by_unit = {
            "mu1": 9.0,
            "mu2": 11.0,
            "mu3": 14.0,
            "mu4": 16.0,
            "mu5": 19.0,
            "mu6": 21.0,
            "mu7": 14.0,
            "mu8": 16.0,
            "mu9": 25.0,
            "mu10": 27.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in revenue_by_unit.items()
        ]
        con.create_table("breakout_missing_control_events", obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM breakout_missing_control_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "breakout_missing_control_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_breakout_sparse_segment_failure_aborts_complete_family(con):
    """A treatment-only segment fails its compiled family hypothesis."""
    analysis = _analysis_with_missing_control_segment(con)

    with pytest.warns(IncrementWarning) as rec:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            with pytest.raises(CapabilityError) as raised:
                analysis.run_breakout()

    assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)

    error = raised.value
    assert error.code == "family.evidence.incomplete"
    assert any("MX" in repr(key) for key in error.context["failed"])  # ty: ignore[not-iterable]


def _analysis_with_encouragement_country_breakout(con):
    """Native encouragement breakout with a declared uptake fact."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import Definitions

    table_name = "encouragement_breakout_events"
    if table_name not in con.list_tables():
        exposure_groups = {
            "US": {
                "control": [f"ec{i}" for i in range(1, 16)],
                "treatment": [f"et{i}" for i in range(1, 16)],
            },
            "CA": {
                "control": [f"cc{i}" for i in range(1, 16)],
                "treatment": [f"ct{i}" for i in range(1, 16)],
            },
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                "experiment_id": "encouragement_breakout_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        clicked_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 12, 0, 0),
                "event": "clicked",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": None,
                "experiment_id": None,
            }
            for country, groups in exposure_groups.items()
            for uid in groups["treatment"][:10]
        ]
        revenue_by_unit = {
            uid: 10.0
            + 0.2 * (int(uid[2:]) % 5)
            + (
                2.0
                if uid
                in {
                    click_uid
                    for country, groups in exposure_groups.items()
                    for click_uid in groups["treatment"][:10]
                }
                else 0.0
            )
            for uid in country_by_unit
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": revenue_by_unit[uid],
                "experiment_id": None,
            }
            for uid in country_by_unit
        ]
        con.create_table(table_name, obj=exposure_rows + clicked_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table_name}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "encouragement_breakout_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "end": "2025-06-04",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )
    return make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True,
                justification="clicking is the only path from assignment to revenue",
            ),
            one_sided=True,
        ),
    )


def test_native_encouragement_breakout_carries_uptake_moments(con):
    """Native breakout reductions must materialize declared uptake moments."""
    analysis = _analysis_with_encouragement_country_breakout(con)

    results = analysis.run_breakout()
    assert {row.estimand for row in results} == {"itt", "compliance", "late"}


@pytest.mark.slow
def test_retention_breakout_summary_uses_mature_exposure_cohorts():
    """The small parity fixture also exercises the one-segment control."""
    from tests.parity_harness.cases import _retention_breakout_cohorts_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _retention_breakout_cohorts_case(single_segment=True)
    assert_parity(case, run_case(case))


@pytest.mark.parametrize(
    "ingress",
    ["from_definitions", pytest.param("from_unit_day_artifact", marks=pytest.mark.slow)],
)
def test_unbounded_retention_breakout_refuses_before_warehouse_access(ingress):
    import ibis

    con = ibis.duckdb.connect()
    analysis = _analysis_with_country_breakout(con, unbounded_retention=True)
    if ingress == "from_unit_day_artifact":
        from tests.parity_harness.cases import _publish_and_adopt

        analysis = _publish_and_adopt(con, analysis, kinds=("breakout_dimension",))
    con.disconnect()
    try:
        with pytest.raises(InvalidRequestError) as caught:
            analysis.breakout_summaries(metrics=["revenue", "unbounded"])
        assert caught.value.code == "breakout.retention.unbounded"
        assert caught.value.context["names"] == ("unbounded",)
    finally:
        analysis.close()


@pytest.mark.slow
def test_retention_breakout_reports_censoring_once():
    from tests.parity_harness.cases import _retention_breakout_cohorts_case

    case = _retention_breakout_cohorts_case(single_segment=True, late_units=20)
    analysis = case.build["from_definitions"]()
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", IncrementWarning)
            tables = analysis.breakout_summaries(metrics=["d7_retention"])
        assert [w.message.code for w in recorded if isinstance(w.message, IncrementWarning)] == [
            "frame.censoring.dropped_units"
        ]
        totals = tables["d7_retention:store:events"]["group_summary"].to_pylist()
        assert {row["group_id"]: row["n"] for row in totals} == {"control": 20, "treatment": 20}
    finally:
        analysis.close()
        for connection in getattr(analysis, "_parity_connections", ()):
            connection.disconnect()
