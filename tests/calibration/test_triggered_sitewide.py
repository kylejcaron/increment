"""Frozen sitewide designs, production equivalence, and recovery sentinels."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest


def test_manifest_freezes_realized_targets_and_family_allocation():
    from calibration import triggered_sitewide as campaign
    from tests.mc import family_eta

    expected_slots = [
        ("T1-1", "T1-partial", "assigned", "absolute", "I*"),
        ("T1-2", "T1-partial", "triggered", "absolute", "I*"),
        ("T1-3", "T1-partial", "assigned", "relative", "rel*"),
        ("T1-4", "T1-partial", "triggered", "relative", "rel*"),
        ("T1-5", "T1-whole", "assigned", "absolute", "I*"),
        ("T1-6", "T1-whole", "triggered", "absolute", "I*"),
        ("T1-7", "T1-whole", "assigned", "relative", "rel*"),
        ("T1-8", "T1-whole", "triggered", "relative", "rel*"),
        ("T4-1", "T4-benign", "assigned", "absolute", "ratio_impact*"),
        ("T4-2", "T4-benign", "triggered", "absolute", "ratio_impact*"),
        ("T4-3", "T4-benign", "assigned", "relative", "relative_ratio_impact*"),
        ("T4-4", "T4-benign", "triggered", "relative", "relative_ratio_impact*"),
    ]
    manifest = campaign.manifest()
    assert [
        (s["id"], s["design"], s["population"], s["scale"], s["truth"]) for s in manifest["slots"]
    ] == expected_slots

    assert manifest["gated_reps"] == 107_100
    assert manifest["eta"] == family_eta(0.01, 67) == 0.00007462686567164179
    assert manifest["coverage_floor"] == 0.945
    assert manifest["seed_start"] == 1000
    expected_designs = [
        ("T1-partial", 50_000, 10_000, 0.5, 0.2, 0.3, 1.0, 0.0, 0.05, 0.5, 0.0, 107_100),
        ("T1-whole", 10_000, 10_000, 0.5, 0.2, 1.0, 1.0, 0.0, 0.05, 0.5, 0.0, 107_100),
        ("T4-benign", 20_000, 10_000, 0.5, 0.2, 0.3, 1.0, 0.0, 0.05, 0.5, 0.0, 107_100),
    ]
    assert [
        (
            d["id"],
            d["n_site"],
            d["n_enrolled"],
            d["p_t"],
            d["p_trig"],
            d["delta"],
            d["s_tau"],
            d["delta_nt"],
            d["lift"],
            d["sigma_den"],
            d["den_effect"],
            d["reps"],
        )
        for d in manifest["designs"]
        if d["id"] in {"T1-partial", "T1-whole", "T4-benign"}
    ] == expected_designs


def test_smoke_campaign_runs_every_cell(tmp_path):
    from calibration import triggered_sitewide as campaign

    assert callable(getattr(campaign, "run_campaign", None))
    report = campaign.run_campaign("smoke", tmp_path, workers=1)
    assert {cell["id"] for cell in report["cells"]} == {d.id for d in campaign.DESIGNS}
    assert len(report["gates"]) == 12
    assert all(g["decision"] == "SMOKE_ONLY" for g in report["gates"])
    assert all(cell["attempted"] == campaign.SMOKE_REPS for cell in report["cells"])
    assert json.loads((tmp_path / "report.json").read_text()) == report
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seed_start"] == 1000
    assert manifest["gated_reps"] == 107_100
    assert {s["id"] for s in manifest["slots"]} == {s.id for s in campaign.SLOTS}
    assert (tmp_path / "report.md").is_file()


@pytest.mark.parametrize(
    ("y_c", "y_t", "d_c", "d_t", "site_num", "site_den"),
    [
        # Two observations per arm; positive numerator-denominator covariance.
        ([1.0, 3.0], [2.0, 6.0], [1.0, 3.0], [2.0, 6.0], 300.0, 800.0),
        # Three observations per arm; negative covariance.
        ([1.0, 3.0, 5.0], [2.0, 4.0, 8.0], [3.0, 2.0, 1.0], [4.0, 2.0, 1.0], 500.0, 900.0),
        # Strongly imbalanced arms and zero covariance.
        (
            [2.0, 4.0],
            [1.0, 3.0, 5.0, 7.0, 9.0],
            [1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            700.0,
            1500.0,
        ),
        # Constant outcomes and denominators have zero variance.
        ([2.0, 2.0, 2.0], [4.0, 4.0], [3.0, 3.0, 3.0], [5.0, 5.0], 800.0, 1200.0),
        # Negative numerator means remain valid in production.
        ([-1.0, -2.0], [-2.0, -4.0, -6.0], [1.0, 1.0], [1.0, 1.0, 1.0], 3000.0, 8000.0),
    ],
)
def test_batched_intervals_match_production_on_edge_case_draws(
    y_c, y_t, d_c, d_t, site_num, site_den
):
    from calibration import triggered_sitewide as campaign
    from increment.estimation.sitewide import SitewideRatioContrast, sitewide_impact_ratio

    y_c, y_t = np.asarray(y_c), np.asarray(y_t)
    d_c, d_t = np.asarray(d_c), np.asarray(d_t)
    scalar = sitewide_impact_ratio(
        SitewideRatioContrast.from_iid(
            campaign.arm_stats("control", y_c, d_c),
            campaign.arm_stats("treatment", y_t, d_t),
        ),
        site_total_numerator=site_num,
        site_total_denominator=site_den,
    )
    batch = campaign.batched_estimates(
        np.array([campaign.arm_moments(y_c, d_c)]),
        np.array([campaign.arm_moments(y_t, d_t)]),
        np.array([site_num]),
        np.array([site_den]),
    )
    assert batch["eligible"].all()
    for field in (
        "absolute_impact",
        "absolute_impact_se",
        "absolute_impact_lb",
        "absolute_impact_ub",
        "relative_impact",
        "relative_impact_se",
        "relative_impact_lb",
        "relative_impact_ub",
    ):
        np.testing.assert_allclose(batch[field], [getattr(scalar, field)], rtol=0, atol=1e-12)


def test_ineligible_report_retains_null_and_registered_consequence(tmp_path):
    from calibration import triggered_sitewide as campaign

    design = replace(campaign.DESIGNS[0], n_site=10, n_enrolled=2)
    cell = campaign.run_design(design, reps=3, workers=1)
    gates = campaign._gates([cell], "full")
    assert all(g["decision"] == "INELIGIBLE" for g in gates[:4])
    report = {
        "status": "ASSIGNED_REGRESSION",
        "wall_seconds": 0.0,
        "workers": 1,
        "gates": gates,
        "cells": [cell],
    }
    campaign._write_report(tmp_path, report)
    restored = json.loads((tmp_path / "report.json").read_text())
    assert restored["gates"][1]["coverage"] is None
    assert "Stop and escalate assigned regression" in (tmp_path / "report.md").read_text()


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("slot_id", ["T1-2", "T4-2"])
def test_realized_impact_recovery_sentinel(slot_id):
    from calibration import triggered_sitewide as campaign
    from tests.mc import coverage_lower_bound

    assert callable(getattr(campaign, "run_design", None))
    slot = next(s for s in campaign.SLOTS if s.id == slot_id)
    design = next(d for d in campaign.DESIGNS if d.id == slot.design)
    cell = campaign.run_design(design, reps=campaign.SENTINEL_REPS, workers=1)
    coverage = cell["coverage"][f"{slot.population}.{slot.scale}"]
    assert coverage["eligible"] == campaign.SENTINEL_REPS
    # Reduced-R sentinel: 99% one-sided lower bound >= .925, not a full gate.
    assert (
        coverage_lower_bound(coverage["hits"], coverage["eligible"], campaign.SENTINEL_ETA)
        >= campaign.SENTINEL_FLOOR
    )
