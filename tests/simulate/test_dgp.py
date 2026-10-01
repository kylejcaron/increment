"""Tests for the data-generating process (Step 1 of the simulation brief)."""

from __future__ import annotations

import copy
import inspect
import math
import pickle

import numpy as np
import pyarrow.compute as pc
import pytest

import increment.simulate.dgp as dgp
from increment.errors import InvalidRequestError
from increment.simulate.dgp import (
    Scenario,
    _build_count_events,
    _check_count_string_capacity,
    _conversion_logit_shift,
    _count_means,
    simulate_raw_logs,
)

# Step 1a: deterministic under seed


class TestDeterministicUnderSeed:
    def test_same_seed_identical(self):
        sc = Scenario(
            n_units=100,
            n_days=7,
            true_lift={"conversion": 0.0, "count": 0.0, "revenue": 0.0},
            seed=42,
        )
        t1 = simulate_raw_logs(sc)
        t2 = simulate_raw_logs(sc)
        import math

        cols = t1.column_names
        for col in cols:
            a = t1.column(col).to_pylist()
            b = t2.column(col).to_pylist()
            for i, (va, vb) in enumerate(zip(a, b, strict=True)):
                if (
                    isinstance(va, float)
                    and isinstance(vb, float)
                    and math.isnan(va)
                    and math.isnan(vb)
                ):
                    continue
                assert va == vb, f"column '{col}' differs at index {i}: {va} != {vb}"


# Step 1b: exposure events emitted for every assigned unit


class TestExposureEvents:
    def test_every_unit_has_exposure(self):
        n = 1000
        sc = Scenario(n_units=n, n_days=7, true_lift={"conversion": 0.0}, seed=0)
        tbl = simulate_raw_logs(sc)
        exposures = tbl.filter(pc.equal(tbl.column("event"), "exposure"))  # ty: ignore[unresolved-attribute] - pyarrow.compute funcs are dynamically generated, no static stub coverage
        uids = exposures.column("unit_id").to_pylist()
        assert len(uids) == n, f"expected {n} exposures, got {len(uids)}"
        assert len(set(uids)) == n, "duplicate unit IDs in exposures"

    def test_exposure_columns_present(self):
        sc = Scenario(n_units=10, n_days=3, true_lift={"conversion": 0.0}, seed=0)
        tbl = simulate_raw_logs(sc)
        exposures = tbl.filter(pc.equal(tbl.column("event"), "exposure"))  # ty: ignore[unresolved-attribute] - pyarrow.compute funcs are dynamically generated, no static stub coverage
        row = exposures.to_pydict()
        assert len(row["unit_id"]) == 10
        assert all(e == "exposure" for e in row["event"])
        assert all(e == "sim" for e in row["experiment_id"])
        assert set(row["group_id"]) <= {"control", "treatment"}

    def test_no_mixed_group_units(self):
        """Each unit appears in exactly one group."""
        sc = Scenario(n_units=100, n_days=3, true_lift={"conversion": 0.0}, seed=0)
        tbl = simulate_raw_logs(sc)
        exposures = tbl.filter(pc.equal(tbl.column("event"), "exposure"))  # ty: ignore[unresolved-attribute] - pyarrow.compute funcs are dynamically generated, no static stub coverage
        pdf = exposures.to_pandas()
        grouped = pdf.groupby("unit_id")["group_id"].nunique()
        assert (grouped == 1).all(), "some units have multiple groups"

    def test_optional_realized_assignment_ratio_changes_allocation(self):
        """The DGP can realize a different allocation than the planned ratio."""

        def exposure_groups(scenario):
            table = simulate_raw_logs(scenario)
            exposures = table.filter(pc.equal(table.column("event"), "exposure"))  # ty: ignore[unresolved-attribute] - pyarrow.compute funcs are dynamically generated, no static stub coverage
            return exposures.column("group_id").to_pylist()

        planned = Scenario(
            n_units=2000,
            n_days=3,
            true_lift={"conversion": 0.0},
            assignment_ratio=0.5,
            seed=17,
        )
        explicit_default = planned.model_copy(update={"realized_assignment_ratio": 0.5})
        realized = planned.model_copy(update={"realized_assignment_ratio": 0.8})

        assert exposure_groups(planned) == exposure_groups(explicit_default)
        realized_groups = exposure_groups(realized)
        assert 1500 < realized_groups.count("treatment") < 1700


# Step 1c: realized lift on 50k units stays within tolerance of true_lift


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestRealizedLift:
    """Verify the DGP realizes ``true_lift`` on the metric mean - the
    same scale lifts are estimated on downstream. Root-finding calibrates
    conversion/revenue shifts so this holds even under heterogeneity.
    Tolerances are 4x the delta-method Monte Carlo SE.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def large_results(cls):
        """Generate data once per class; ``unit_heterogeneity=0.5`` on
        purpose so calibration is checked under a nonzero random effect,
        not just the homogeneous closed-form case."""
        sc = Scenario(
            n_units=50_000,
            n_days=14,
            true_lift={"conversion": 0.20, "count": 0.15, "revenue": 0.10},
            unit_heterogeneity=0.5,
            seed=42,
        )
        tbl = simulate_raw_logs(sc)
        pdf = tbl.to_pandas()
        exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
        return sc, pdf, exposures

    def _lift_and_4se(self, per_unit) -> tuple[float, float]:
        """Realized relative mean lift and 4x its delta-method SE; ``per_unit``
        has one row/unit with ``group_id`` and zero-filled ``v``."""
        import math

        g = per_unit.groupby("group_id")["v"].agg(["mean", "var", "count"])
        mean_t, var_t, n_t = g.loc["treatment"]
        mean_c, var_c, n_c = g.loc["control"]
        lift = mean_t / mean_c - 1.0
        se = (mean_t / mean_c) * math.sqrt(var_t / (n_t * mean_t**2) + var_c / (n_c * mean_c**2))
        return float(lift), 4.0 * float(se)

    def test_conversion_mean_lift(self, large_results):
        """Conversion lift is relative lift of the marginal rate: logit shift
        calibrated so ``E[p_treatment] = (1+true_lift)*E[p_control]`` even
        under heterogeneity - a fixed odds shift would undershoot it."""
        sc, pdf, exposures = large_results
        conv_uids = set(pdf[pdf["event"] == "conversion"]["unit_id"])
        per_unit = exposures.copy()
        per_unit["v"] = per_unit["unit_id"].isin(conv_uids).astype(float)
        realized, tol = self._lift_and_4se(per_unit)
        expected = sc.true_lift["conversion"]
        assert abs(realized - expected) < tol, (
            f"conversion rate lift {realized:.4f} differs from {expected} "
            f"by more than {tol:.4f} (4x Monte Carlo SE)"
        )

    def test_count_lift_within_tolerance(self, large_results):
        """Count lift is multiplicative on the mean count."""
        sc, pdf, exposures = large_results
        unit_counts = pdf[pdf["event"] == "visit"].groupby("unit_id").size()
        per_unit = exposures.copy()
        per_unit["v"] = per_unit["unit_id"].map(unit_counts).fillna(0.0)
        realized, tol = self._lift_and_4se(per_unit)
        expected = sc.true_lift["count"]
        assert abs(realized - expected) < tol, (
            f"count lift {realized:.4f} differs from {expected} "
            f"by more than {tol:.4f} (4x Monte Carlo SE)"
        )

    def test_revenue_mean_lift(self, large_results):
        """Revenue lift is relative lift of mean revenue/unit: the shift is
        split across purchase odds and log-amount, calibrated so realized
        mean lift equals ``true_lift`` (an uncalibrated split undershoots)."""
        sc, pdf, exposures = large_results
        unit_rev = pdf[pdf["event"] == "revenue"].groupby("unit_id")["value"].sum()
        per_unit = exposures.copy()
        per_unit["v"] = per_unit["unit_id"].map(unit_rev).fillna(0.0)
        realized, tol = self._lift_and_4se(per_unit)
        expected = sc.true_lift["revenue"]
        assert abs(realized - expected) < tol, (
            f"revenue lift {realized:.4f} differs from {expected} "
            f"by more than {tol:.4f} (4x Monte Carlo SE)"
        )

    def test_zero_lift_produces_near_zero_lift(self):
        """Zero true lift must realize near 0 within 4x the delta-method
        Monte Carlo SE (~0.085 at this N); a 0.05 band is only ~2.3 SE."""
        sc = Scenario(
            n_units=50_000,
            n_days=7,
            true_lift={"conversion": 0.0},
            unit_heterogeneity=0.0,
            seed=123,
        )
        tbl = simulate_raw_logs(sc)
        pdf = tbl.to_pandas()
        conv_uids = set(pdf[pdf["event"] == "conversion"]["unit_id"])
        per_unit = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
        per_unit["v"] = per_unit["unit_id"].isin(conv_uids).astype(float)
        realized, tol = self._lift_and_4se(per_unit)
        assert abs(realized) < tol, (
            f"zero-lift conversion realized lift = {realized:.4f} "
            f"exceeds {tol:.4f} (4x Monte Carlo SE)"
        )


def test_realized_lift_smoke_small_n():
    """Fast smoke: count lift has the right sign at loose tolerance (4 SE
    ~0.47 at this N); full-precision check lives in ``TestRealizedLift``."""
    sc = Scenario(
        n_units=500,
        n_days=3,
        true_lift={"count": 0.5},
        unit_heterogeneity=0.0,
        seed=7,
    )
    tbl = simulate_raw_logs(sc)
    pdf = tbl.to_pandas()
    exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
    fact = pdf[pdf["event"] == "visit"]
    unit_counts = fact.groupby(["unit_id", "group_id"]).size().reset_index(name="count")
    merged = exposures.merge(unit_counts, on=["unit_id", "group_id"], how="left")
    merged["count"] = merged["count"].fillna(0.0)
    means = merged.groupby("group_id")["count"].mean()
    realized = means["treatment"] / means["control"] - 1.0
    assert realized > 0.0, f"expected positive count lift, got {realized:.4f}"
    assert abs(realized - sc.true_lift["count"]) < 0.5, (
        f"count lift {realized:.4f} too far from {sc.true_lift['count']} even for small-N smoke"
    )


@pytest.mark.parametrize(
    ("lift", "eta"),
    [(-0.99, 0.0), (10_000.0, 0.0), (0.0, -30.0), (0.0, -72.0), (0.0, -100.0)],
)
def test_count_mean_matches_the_declared_law_across_old_clip_boundaries(lift, eta):
    """The count kernel keeps the requested mean on both sides of [-10, 10]."""
    groups = np.array(["control", "treatment"])
    means = _count_means(
        groups,
        np.full(2, eta),
        lift,
        3.0,
    )

    expected = 3.0 * math.exp(0.5 * eta) * np.array([1.0, 1.0 + lift])
    assert means == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("eta", [-1500.0, 1500.0])
def test_count_mean_refuses_unrepresentable_rates_with_recovery_controls(eta):
    groups = np.array(["control"])
    with pytest.raises(InvalidRequestError) as raised:
        _count_means(groups, np.array([eta]), 0.0, 3.0)
    assert raised.value.code == "simulate.dgp.count_rate_unrepresentable"
    controls = raised.value.context["controls"]
    assert isinstance(controls, tuple)
    assert "unit_heterogeneity" in controls


class _ControlledCountRng:
    def __init__(self, gamma: float, count: int):
        self.gamma_value = gamma
        self.count = count

    def gamma(self, shape, scale, size):
        return np.full(size, self.gamma_value)

    def poisson(self, rates):
        return np.full(rates.shape, self.count)

    def integers(self, low, high, size):
        return np.zeros(size, dtype=np.int64)


def _count_builder(rng, *, base_rate=3.0, unit_id="u"):
    return _build_count_events(
        np.array([unit_id]),
        np.array(["control"]),
        np.array([0.0]),
        0.0,
        base_rate,
        2.0,
        rng,
        1,
        exposure_ts=np.array([dgp._START_DATE], dtype="datetime64[us]"),
    )


def test_count_sampler_refuses_positive_rate_gamma_underflow_before_poisson():
    rng = _ControlledCountRng(1e-10, 1)
    with pytest.raises(InvalidRequestError) as raised:
        _count_builder(rng, base_rate=1e-320)
    assert raised.value.code == "simulate.dgp.count_rate_unrepresentable"
    assert raised.value.context["reason"] == "negative_binomial_rate_underflow"
    for transported in (copy.deepcopy(raised.value), pickle.loads(pickle.dumps(raised.value))):
        assert transported.code == raised.value.code
        assert transported.context == raised.value.context
        assert transported.context["route"] == raised.value.context["route"]


def test_count_string_capacity_uses_utf8_bytes_not_character_count():
    counts = np.array([200_000_000], dtype=np.int64)
    with pytest.raises(InvalidRequestError) as raised:
        _check_count_string_capacity(
            np.array(["é" * 8]), np.array(["g"]), counts, ne=int(counts[0])
        )
    assert raised.value.context["reason"] == "event_string_data_capacity"
    assert raised.value.context["column"] == "unit_id"
    assert raised.value.context["total_string_bytes"] == 3_200_000_000


def test_count_string_capacity_allows_small_representable_layout():
    table = _count_builder(_ControlledCountRng(1.0, 2), unit_id="é")
    assert table["unit_id"].to_pylist() == ["é", "é"]


def test_count_lift_at_domain_floor_is_continuous():
    """``lift=-0.99`` (domain floor) must realize ~99% reduction via
    ``log(0.01)`` semantics, not a total-shutdown sentinel two orders of
    magnitude lower - the domain edge behaves like its neighborhood."""
    sc = Scenario(
        n_units=10_000,
        n_days=7,
        true_lift={"count": -0.99},
        unit_heterogeneity=0.0,
        seed=13,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
    unit_counts = pdf[pdf["event"] == "visit"].groupby("unit_id").size()
    exposures["v"] = exposures["unit_id"].map(unit_counts).fillna(0.0)
    means = exposures.groupby("group_id")["v"].mean()
    realized = means["treatment"] / means["control"] - 1.0
    # Band is ~6x the Monte Carlo SE of the realized lift at this N; the
    # sentinel regression would land at ~-0.9999, far outside it.
    assert -0.995 < realized < -0.985, (
        f"count lift at the -0.99 domain floor realized {realized:.5f}; "
        f"expected ~-0.99 (log(0.01) semantics, continuous at the boundary)"
    )


def test_conversion_lift_smoke_small_n():
    """Fast smoke: conversion lift has the right sign at loose tolerance
    (4 SE ~0.45 at this N); full-precision check lives in ``TestRealizedLift``."""
    sc = Scenario(
        n_units=5000,
        n_days=7,
        true_lift={"conversion": 0.8},
        unit_heterogeneity=0.5,
        seed=7,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
    conv_uids = set(pdf[pdf["event"] == "conversion"]["unit_id"])
    exposures["converted"] = exposures["unit_id"].isin(conv_uids).astype(float)
    means = exposures.groupby("group_id")["converted"].mean()
    realized = means["treatment"] / means["control"] - 1.0
    assert realized > 0.0, f"expected positive conversion lift, got {realized:.4f}"
    assert abs(realized - sc.true_lift["conversion"]) < 0.45, (
        f"conversion lift {realized:.4f} too far from "
        f"{sc.true_lift['conversion']} even for small-N smoke"
    )


def test_revenue_lift_smoke_small_n():
    """Fast smoke: mean-revenue lift has the right sign; lognormal revenue
    is heavy-tailed so 4 SE is roughly the lift's size here - loose by
    design. Full-precision check lives in ``TestRealizedLift``."""
    sc = Scenario(
        n_units=8000,
        n_days=7,
        true_lift={"revenue": 1.0},
        unit_heterogeneity=0.5,
        seed=11,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
    unit_rev = pdf[pdf["event"] == "revenue"].groupby("unit_id")["value"].sum()
    exposures["rev"] = exposures["unit_id"].map(unit_rev).fillna(0.0)
    means = exposures.groupby("group_id")["rev"].mean()
    realized = means["treatment"] / means["control"] - 1.0
    assert realized > 0.0, f"expected positive revenue lift, got {realized:.4f}"
    assert abs(realized - sc.true_lift["revenue"]) < 1.0, (
        f"revenue lift {realized:.4f} too far from {sc.true_lift['revenue']} even for small-N smoke"
    )


def test_multi_month_window_generates_valid_timestamps():
    """A window past one calendar month must not break at the boundary:
    day offsets are calendar arithmetic (timedeltas), not a datetime day
    field. A 45-day window from January 1 must reach past January 31."""
    from datetime import datetime

    sc = Scenario(n_units=200, n_days=45, true_lift={"count": 0.0}, seed=3)
    tbl = simulate_raw_logs(sc)
    ts = tbl.column("ts").to_pylist()
    assert min(ts) >= datetime(2025, 1, 1), f"event before window start: {min(ts)}"
    assert max(ts) >= datetime(2025, 2, 1), (
        f"no event past January 31 in a 45-day window (max ts {max(ts)})"
    )


# Every post-exposure fact, including a sampled day-zero offset, must land after
# its own unit's randomized exposure timestamp, not the start date's midnight;
# otherwise the pipeline's `ts > first_exposure_ts` filter
# (`increment.query.builders.post_exposure_stats`) discards it.


@pytest.mark.parametrize(
    "true_lift,event_name",
    [
        ({"conversion": 0.3}, "conversion"),
        ({"count": 0.3}, "visit"),
        ({"revenue": 0.3}, "revenue"),
    ],
)
def test_post_exposure_events_never_predate_their_own_exposure(true_lift, event_name):
    """Every conversion/count(visit)/revenue event -- including one
    sampled at day offset 0 -- lands strictly after its own unit's
    exposure timestamp, not the shared midnight of the experiment start."""
    sc = Scenario(n_units=3000, n_days=5, true_lift=true_lift, unit_heterogeneity=0.5, seed=11)
    pdf = simulate_raw_logs(sc).to_pandas()
    exposure_ts = pdf[pdf["event"] == "exposure"].set_index("unit_id")["ts"]
    post = pdf[pdf["event"] == event_name]
    assert not post.empty, f"expected some {event_name} events to exercise the chronology check"
    own_exposure = post["unit_id"].map(exposure_ts)
    assert (post["ts"] > own_exposure).all(), (
        f"{event_name} events must all land strictly after their own unit's exposure"
    )
    # At least one event lands on the exposure day itself (day offset 0):
    # the original defect specifically discarded these.
    same_day = post["ts"].dt.date == own_exposure.dt.date
    assert same_day.any(), (
        f"expected at least one day-zero {event_name} event at n=3000/{sc.n_days} days"
    )


def test_encouragement_events_never_predate_their_own_exposure():
    """The encouragement design's uptake/outcome events use the same
    exposure-relative chronology as the three standard metrics."""
    sc = Scenario(
        n_units=3000,
        n_days=5,
        true_lift={},
        uptake_compliance_t=0.6,
        uptake_compliance_c=0.2,
        tau_complier=1.0,
        seed=13,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    exposure_ts = pdf[pdf["event"] == "exposure"].set_index("unit_id")["ts"]
    for event_name in ("uptake", "outcome"):
        post = pdf[pdf["event"] == event_name]
        assert not post.empty, f"expected some {event_name} events"
        own_exposure = post["unit_id"].map(exposure_ts)
        assert (post["ts"] > own_exposure).all(), (
            f"{event_name} events must all land strictly after their own unit's exposure"
        )


def test_post_exposure_day_offset_still_spans_the_full_window():
    """The sampled day-offset contract ([0, n_days)) survives the
    exposure-relative chronology fix: events still spread across the
    whole declared window, not just the first day."""
    sc = Scenario(n_units=4000, n_days=10, true_lift={"count": 0.3}, seed=17)
    pdf = simulate_raw_logs(sc).to_pandas()
    exposure_ts = pdf[pdf["event"] == "exposure"].set_index("unit_id")["ts"]
    visits = pdf[pdf["event"] == "visit"]
    day_offsets = (visits["ts"] - visits["unit_id"].map(exposure_ts)).dt.days
    assert day_offsets.min() == 0
    assert day_offsets.max() == sc.n_days - 1


# Encouragement design: uptake fact honors per-arm compliance


class TestEncouragementUptake:
    """``uptake_compliance_t``/``_c``/``tau_complier`` drive a Bernoulli
    uptake fact plus compliance-driven outcome; verified via a real DuckDB
    round-trip since the raw log is the contract the query layer consumes."""

    def test_uptake_only_for_encouraged_units_one_sided(self):
        ibis = pytest.importorskip("ibis")
        pytest.importorskip("duckdb")

        sc = Scenario(
            n_units=2000,
            n_days=7,
            true_lift={},
            uptake_compliance_t=0.5,
            uptake_compliance_c=0.0,
            tau_complier=2.0,
            seed=17,
        )
        tbl = simulate_raw_logs(sc)

        con = ibis.duckdb.connect()
        events = con.create_table("events", obj=tbl)
        uptake = events.filter(events.event == "uptake")

        n_uptake = con.to_pyarrow(uptake.count()).as_py()
        assert n_uptake > 0, "expected some uptake events with compliance_t=0.5"

        uptake_groups = set(
            con.to_pyarrow(uptake.select("group_id").distinct()).column("group_id").to_pylist()
        )
        assert uptake_groups == {"treatment"}, (
            f"uptake_compliance_c=0 should confine uptake events to encouraged "
            f"(treatment) units, got groups {uptake_groups}"
        )

        # Outcome events are emitted for every unit regardless of uptake.
        n_outcome = con.to_pyarrow(events.filter(events.event == "outcome").count()).as_py()
        assert n_outcome == sc.n_units


# Encouragement design: uptake-outcome confounding


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestEncouragementConfounding:
    """Nonzero ``uptake_confounding``/``outcome_confounding`` create
    selection into compliance, biasing naive as-treated contrasts away
    from ``tau_complier`` - randomized assignment keeps arm contrasts clean."""

    @pytest.fixture(scope="class")
    @classmethod
    def confounded(cls):
        sc = Scenario(
            n_units=20_000,
            n_days=7,
            true_lift={},
            unit_heterogeneity=1.0,
            uptake_compliance_t=0.7,
            uptake_compliance_c=0.3,
            tau_complier=2.0,
            uptake_confounding=1.0,
            outcome_confounding=1.0,
            seed=5,
        )
        return sc, simulate_raw_logs(sc).to_pandas()

    def test_naive_as_treated_is_biased(self, confounded):
        """In the control arm alone, the as-treated contrast
        ``mean(y|d=1) - mean(y|d=0)`` must exceed ``tau_complier`` by >4 SE:
        uptake selects on the same latent factor that raises the outcome."""
        import math

        sc, pdf = confounded
        exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]]
        uptook = set(pdf[pdf["event"] == "uptake"]["unit_id"])
        outcome = pdf[pdf["event"] == "outcome"].set_index("unit_id")["value"]
        ctrl = exposures[exposures["group_id"] == "control"].copy()
        ctrl["d"] = ctrl["unit_id"].isin(uptook).astype(int)
        ctrl["y"] = ctrl["unit_id"].map(outcome)
        by_d = ctrl.groupby("d")["y"].agg(["mean", "var", "count"])
        naive = by_d.loc[1, "mean"] - by_d.loc[0, "mean"]
        se = math.sqrt(
            by_d.loc[1, "var"] / by_d.loc[1, "count"] + by_d.loc[0, "var"] / by_d.loc[0, "count"]
        )
        assert naive - sc.tau_complier > 4.0 * se, (
            f"naive as-treated contrast {naive:.4f} not biased above "
            f"tau={sc.tau_complier} by more than 4*SE={4 * se:.4f}; "
            f"uptake appears independent of the outcome level"
        )

    def test_randomization_keeps_latent_balanced(self, confounded):
        """The latent effect is drawn independently of assignment, so
        arm-level means must agree within Monte Carlo error - selection
        biases as-treated contrasts, never randomized arm contrasts."""
        import numpy as np

        sc, _pdf = confounded
        rng = np.random.default_rng(sc.seed)
        eta = rng.normal(0, sc.unit_heterogeneity, size=sc.n_units)
        group = rng.binomial(1, sc.assignment_ratio, size=sc.n_units)
        eta_t, eta_c = eta[group == 1], eta[group == 0]
        diff = eta_t.mean() - eta_c.mean()
        se = float(np.sqrt(eta_t.var() / len(eta_t) + eta_c.var() / len(eta_c)))
        assert abs(diff) < 4.0 * se, (
            f"arm-level latent-effect difference {diff:.5f} exceeds "
            f"4*SE={4 * se:.5f}; assignment is not independent of the latent"
        )


def test_encouragement_confounding_smoke_small_n():
    """Fast smoke: nonzero confounding yields uptake in both arms, and
    the control-arm as-treated contrast lands above ``tau_complier`` at
    this seed; precise 4*SE bound lives in ``TestEncouragementConfounding``."""
    sc = Scenario(
        n_units=1500,
        n_days=7,
        true_lift={},
        unit_heterogeneity=1.0,
        uptake_compliance_t=0.7,
        uptake_compliance_c=0.3,
        tau_complier=2.0,
        uptake_confounding=1.0,
        outcome_confounding=1.0,
        seed=5,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    uptake_groups = set(pdf[pdf["event"] == "uptake"]["group_id"])
    assert uptake_groups == {"control", "treatment"}, (
        f"expected uptake in both arms with two-sided compliance, got {uptake_groups}"
    )
    exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]]
    uptook = set(pdf[pdf["event"] == "uptake"]["unit_id"])
    outcome = pdf[pdf["event"] == "outcome"].set_index("unit_id")["value"]
    ctrl = exposures[exposures["group_id"] == "control"].copy()
    ctrl["d"] = ctrl["unit_id"].isin(uptook).astype(int)
    ctrl["y"] = ctrl["unit_id"].map(outcome)
    naive = ctrl[ctrl["d"] == 1]["y"].mean() - ctrl[ctrl["d"] == 0]["y"].mean()
    assert naive > sc.tau_complier, (
        f"control-arm as-treated contrast {naive:.4f} should sit above "
        f"tau={sc.tau_complier} under positive confounding"
    )


def test_one_sided_compliance_with_confounding_stays_degenerate():
    """Zero control compliance must stay EXACTLY zero under nonzero
    confounding: logit(0) = -inf, so no finite shift resurrects it or yields NaN."""
    sc = Scenario(
        n_units=1500,
        n_days=7,
        true_lift={},
        unit_heterogeneity=1.0,
        uptake_compliance_t=0.7,
        uptake_compliance_c=0.0,
        tau_complier=2.0,
        uptake_confounding=1.0,
        outcome_confounding=1.0,
        seed=5,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    uptake = pdf[pdf["event"] == "uptake"]
    assert set(uptake["group_id"]) == {"treatment"}, (
        f"one-sided compliance must confine uptake to the treatment arm, "
        f"got {set(uptake['group_id'])}"
    )
    n_treat = (pdf[pdf["event"] == "exposure"]["group_id"] == "treatment").sum()
    assert 0 < len(uptake) < n_treat, (
        "treatment-arm uptake should be nondegenerate (some but not all units)"
    )
    assert pdf[pdf["event"] == "outcome"]["value"].notna().all(), (
        "outcomes must stay finite when a compliance logit is infinite"
    )


# Scenario domain validation


class TestScenarioValidation:
    """Out-of-domain knobs fail at construction with a clear pydantic
    error instead of a cryptic numpy/Arrow crash mid-generation."""

    def test_n_units_zero_rejected(self):
        with pytest.raises(InvalidRequestError) as raised:
            Scenario(n_units=0, n_days=7, true_lift={"conversion": 0.0})
        assert raised.value.code == "model.field.range"

    def test_n_days_zero_rejected(self):
        with pytest.raises(InvalidRequestError) as raised:
            Scenario(n_units=10, n_days=0, true_lift={"conversion": 0.0})
        assert raised.value.code == "model.field.range"

    @pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5, -0.1])
    def test_assignment_ratio_outside_open_interval_rejected(self, ratio):
        with pytest.raises(InvalidRequestError) as raised:
            Scenario(n_units=10, n_days=7, true_lift={"conversion": 0.0}, assignment_ratio=ratio)
        assert raised.value.code == "model.field.range"

    @pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5, -0.1])
    def test_realized_assignment_ratio_outside_open_interval_rejected(self, ratio):
        with pytest.raises(InvalidRequestError) as raised:
            Scenario(
                n_units=10,
                n_days=7,
                true_lift={"conversion": 0.0},
                realized_assignment_ratio=ratio,
            )
        assert raised.value.code == "model.field.range"

    def test_negative_unit_heterogeneity_rejected(self):
        with pytest.raises(InvalidRequestError) as raised:
            Scenario(n_units=10, n_days=7, true_lift={"conversion": 0.0}, unit_heterogeneity=-0.5)
        assert raised.value.code == "model.field.range"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("uptake_compliance_t", 1.5),
            ("uptake_compliance_t", -0.1),
            ("uptake_compliance_c", 1.5),
            ("uptake_compliance_c", -0.1),
        ],
    )
    def test_compliance_outside_unit_interval_rejected(self, field, value):
        kwargs = {"tau_complier": 1.0, "uptake_compliance_t": 0.5, field: value}
        with pytest.raises(InvalidRequestError) as raised:
            Scenario(n_units=10, n_days=7, true_lift={}, **kwargs)
        assert raised.value.code == "model.field.range"

    def test_valid_scenario_still_constructs(self):
        sc = Scenario(
            n_units=1,
            n_days=1,
            true_lift={"conversion": 0.0},
            assignment_ratio=0.05,
            unit_heterogeneity=0.0,
        )
        assert sc.n_units == 1

    @pytest.mark.parametrize("value", [-0.995, math.nan, math.inf, -math.inf])
    def test_true_lift_outside_supported_domain_rejected(self, value):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            Scenario(n_units=10, n_days=7, true_lift={"conversion": value})
        assert exc_info.value.code == "simulate.dgp.scenario.true_lift_finite"
        assert exc_info.value.context["metric"] == "conversion"

    def test_true_lift_domain_floor_accepted(self):
        scenario = Scenario(n_units=10, n_days=7, true_lift={"conversion": -0.99})
        assert scenario.true_lift == {"conversion": -0.99}

    def test_true_lift_forwarded_without_clamping(self, monkeypatch):
        declared = 0.123456789
        forwarded: list[float] = []
        original = dgp._build_conversion_events

        def spy(*args, **kwargs):
            bound = inspect.signature(original).bind(*args, **kwargs)
            forwarded.append(bound.arguments["lift"])
            return original(*args, **kwargs)

        monkeypatch.setattr(dgp, "_build_conversion_events", spy)
        simulate_raw_logs(Scenario(n_units=10, n_days=7, true_lift={"conversion": declared}))

        assert forwarded == [declared]


@pytest.mark.parametrize("field,value", [("n_segments", 0), ("covariate_sd", -0.1)])
def test_covariate_segment_knobs_reject_out_of_domain(field, value):
    with pytest.raises(InvalidRequestError) as raised:
        Scenario(n_units=10, n_days=7, true_lift={"conversion": 0.0}, **{field: value})
    assert raised.value.code == "model.field.range"


# Observable pre-treatment segment/covariate


class TestObservableSegmentAndCovariate:
    """``n_segments``/``covariate_sd`` gate emission of observable
    "segment"/"covariate" event rows -- absent by default so existing
    scenarios are byte-for-byte unaffected."""

    def test_no_segment_or_covariate_events_by_default(self):
        sc = Scenario(n_units=200, n_days=7, true_lift={"conversion": 0.0}, seed=0)
        tbl = simulate_raw_logs(sc)
        events = set(tbl.column("event").to_pylist())
        assert "segment" not in events
        assert "covariate" not in events

    def test_segment_event_one_row_per_unit_in_range(self):
        n = 500
        sc = Scenario(n_units=n, n_days=7, true_lift={"conversion": 0.0}, n_segments=4, seed=0)
        tbl = simulate_raw_logs(sc)
        seg = tbl.filter(pc.equal(tbl.column("event"), "segment"))  # ty: ignore[unresolved-attribute] - pyarrow.compute funcs are dynamically generated, no static stub coverage
        uids = seg.column("unit_id").to_pylist()
        assert len(uids) == n
        assert len(set(uids)) == n
        values = seg.column("value").to_pylist()
        assert set(values) <= {0.0, 1.0, 2.0, 3.0}

    def test_covariate_event_one_row_per_unit_finite(self):
        n = 500
        sc = Scenario(n_units=n, n_days=7, true_lift={"conversion": 0.0}, covariate_sd=2.0, seed=0)
        tbl = simulate_raw_logs(sc)
        cov = tbl.filter(pc.equal(tbl.column("event"), "covariate"))  # ty: ignore[unresolved-attribute] - pyarrow.compute funcs are dynamically generated, no static stub coverage
        uids = cov.column("unit_id").to_pylist()
        assert len(uids) == n
        assert len(set(uids)) == n
        values = cov.column("value").to_pylist()
        assert all(v == v for v in values), "covariate value must never be NaN"

    def test_segments_present_in_both_arms(self):
        """Segment is drawn independent of assignment -- both arms see
        every segment at moderate N."""
        sc = Scenario(n_units=4000, n_days=7, true_lift={"conversion": 0.0}, n_segments=3, seed=1)
        pdf = simulate_raw_logs(sc).to_pandas()
        exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]]
        seg = pdf[pdf["event"] == "segment"][["unit_id", "value"]]
        merged = exposures.merge(seg, on="unit_id")
        pairs = set(zip(merged["group_id"], merged["value"], strict=False))
        assert pairs == {(g, s) for g in ("control", "treatment") for s in (0.0, 1.0, 2.0)}


# Covariate interaction: heterogeneous per-unit treatment effect


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestCovariateInteraction:
    """A nonzero ``covariate_interaction`` must keep the realized AVERAGE
    lift exactly calibrated (the exact-mean corrections in
    ``_build_*_events``), while making the per-unit effect genuinely vary
    with the observable covariate."""

    @pytest.fixture(scope="class")
    @classmethod
    def interacted(cls):
        sc = Scenario(
            n_units=100_000,
            n_days=7,
            true_lift={"conversion": 0.2, "count": 0.15, "revenue": 0.1},
            unit_heterogeneity=0.5,
            covariate_sd=1.0,
            covariate_interaction=0.8,
            seed=42,
        )
        tbl = simulate_raw_logs(sc)
        pdf = tbl.to_pandas()
        exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
        return sc, pdf, exposures

    def _lift_and_4se(self, per_unit) -> tuple[float, float]:
        import math

        g = per_unit.groupby("group_id")["v"].agg(["mean", "var", "count"])
        mean_t, var_t, n_t = g.loc["treatment"]
        mean_c, var_c, n_c = g.loc["control"]
        lift = mean_t / mean_c - 1.0
        se = (mean_t / mean_c) * math.sqrt(var_t / (n_t * mean_t**2) + var_c / (n_c * mean_c**2))
        return float(lift), 4.0 * float(se)

    @pytest.mark.parametrize(
        "metric,event",
        [("conversion", "conversion"), ("count", "visit"), ("revenue", "revenue")],
    )
    def test_average_lift_stays_calibrated(self, interacted, metric, event):
        sc, pdf, exposures = interacted
        if event == "revenue":
            per = pdf[pdf["event"] == event].groupby("unit_id")["value"].sum()
        elif event == "visit":
            per = pdf[pdf["event"] == event].groupby("unit_id").size()
        else:
            uids = set(pdf[pdf["event"] == event]["unit_id"])
            per = exposures["unit_id"].isin(uids).astype(float)
            per.index = exposures["unit_id"]
        per_unit = exposures.copy()
        per_unit["v"] = per_unit["unit_id"].map(per).fillna(0.0)
        realized, tol = self._lift_and_4se(per_unit)
        expected = sc.true_lift[metric]
        assert abs(realized - expected) < tol, (
            f"{metric} lift {realized:.4f} differs from {expected} by more than "
            f"{tol:.4f} (4x Monte Carlo SE) under covariate_interaction={sc.covariate_interaction}"
        )

    def test_treatment_effect_increases_with_covariate(self, interacted):
        """The realized per-unit revenue lift must be monotonically larger
        for units with a higher covariate draw -- the whole point of the
        interaction term."""
        sc, pdf, exposures = interacted
        cov = pdf[pdf["event"] == "covariate"].set_index("unit_id")["value"]
        rev = pdf[pdf["event"] == "revenue"].groupby("unit_id")["value"].sum()
        merged = exposures.set_index("unit_id").copy()
        merged["w"] = cov
        merged["rev"] = rev.reindex(merged.index).fillna(0.0)
        merged["wbin"] = pd_qcut(merged["w"])
        lifts = []
        for b in sorted(merged["wbin"].unique()):
            sub = merged[merged["wbin"] == b]
            means = sub.groupby("group_id")["rev"].mean()
            lifts.append(means["treatment"] / means["control"] - 1.0)
        assert lifts == sorted(lifts), (
            f"per-covariate-bin revenue lift {lifts} is not monotonically increasing "
            f"with the covariate under covariate_interaction={sc.covariate_interaction}"
        )


def pd_qcut(series):
    import pandas as pd

    return pd.qcut(series, 4, labels=False)


def test_covariate_interaction_smoke_small_n():
    """Fast smoke: nonzero covariate_interaction must not corrupt the
    calibration's sign or produce NaNs at small N."""
    sc = Scenario(
        n_units=2000,
        n_days=7,
        true_lift={"revenue": 0.3},
        unit_heterogeneity=0.5,
        covariate_sd=1.0,
        covariate_interaction=1.0,
        seed=9,
    )
    pdf = simulate_raw_logs(sc).to_pandas()
    assert pdf[pdf["event"] == "revenue"]["value"].notna().all()
    exposures = pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]].drop_duplicates()
    rev = pdf[pdf["event"] == "revenue"].groupby("unit_id")["value"].sum()
    exposures["rev"] = exposures["unit_id"].map(rev).fillna(0.0)
    means = exposures.groupby("group_id")["rev"].mean()
    realized = means["treatment"] / means["control"] - 1.0
    assert realized > 0.0, f"expected positive revenue lift, got {realized:.4f}"


def test_zero_lift_conversion_interaction_still_calibrates_treatment_shift():
    shift = _conversion_logit_shift(0.0, 0.15, 0.5, treat_extra_var=0.25)
    assert shift != 0.0


def test_conversion_logit_shift_refuses_an_unattainable_lift():
    from increment.errors import InvalidRequestError

    scenario = Scenario(
        n_units=10,
        n_days=7,
        true_lift={"conversion": 6.0},
        unit_heterogeneity=0.0,
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        simulate_raw_logs(scenario)
    assert exc_info.value.code == "simulate.dgp.conversion_lift_unattainable"
    assert exc_info.value.context["lift"] == 6.0
    assert exc_info.value.context["p_control"] == pytest.approx(0.15)


def test_simulate_raw_logs_refuses_an_unsupported_metric():
    from increment.errors import InvalidRequestError

    scenario = Scenario(n_units=10, n_days=7, true_lift={"nonsense": 0.1})
    with pytest.raises(InvalidRequestError) as exc_info:
        simulate_raw_logs(scenario)
    assert exc_info.value.code == "simulate.dgp.unsupported_metric_supported"
    assert exc_info.value.context["unknown"] == ("nonsense",)


def test_simulate_raw_logs_refuses_a_metric_declared_supported_but_unhandled(monkeypatch):
    """``_SUPPORTED_METRICS`` and the per-metric dispatch chain must stay in
    sync; a metric declared supported but not dispatched is a bug, refused
    rather than silently skipped."""
    from increment.errors import InvalidRequestError

    monkeypatch.setattr(
        dgp, "_SUPPORTED_METRICS", frozenset({"conversion", "count", "revenue", "ghost"})
    )
    scenario = Scenario(n_units=10, n_days=7, true_lift={"ghost": 0.1})
    with pytest.raises(InvalidRequestError) as exc_info:
        simulate_raw_logs(scenario)
    assert exc_info.value.code == "simulate.dgp.unknown_metric_true"
    assert exc_info.value.context["metric_name"] == "ghost"


def test_simulate_raw_logs_refuses_uptake_compliance_without_tau_complier():
    from increment.errors import InvalidRequestError

    scenario = Scenario(
        n_units=10, n_days=7, true_lift={"conversion": 0.0}, uptake_compliance_t=0.5
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        simulate_raw_logs(scenario)
    assert exc_info.value.code == "simulate.dgp.uptake_compliance_t"


def test_observable_properties_are_strictly_pre_exposure():
    scenario = Scenario(
        n_units=100,
        n_days=3,
        true_lift={"conversion": 0.0},
        n_segments=2,
        covariate_sd=1.0,
        seed=4,
    )
    events = simulate_raw_logs(scenario).to_pandas()
    exposure_min = events.loc[events.event == "exposure", "ts"].min()
    properties = events[events.event.isin(["segment", "covariate"])]
    assert not properties.empty
    assert properties["ts"].max() < exposure_min
