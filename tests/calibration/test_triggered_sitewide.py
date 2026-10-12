"""Frozen sitewide designs, production equivalence, and recovery sentinels."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest


def test_manifest_freezes_realized_targets_and_family_allocation():
    assert importlib.util.find_spec("calibration.triggered_sitewide") is not None
    from calibration.triggered_sitewide import DESIGNS, ETA, GATED_REPS, SLOTS
    from tests.mc import family_eta

    assert GATED_REPS == 107_100
    assert ETA == family_eta(0.01, 67) == 0.00007462686567164179
    assert [s.id for s in SLOTS] == [f"T1-{i}" for i in range(1, 9)] + [
        f"T4-{i}" for i in range(1, 5)
    ]
    assert [(s.population, s.scale) for s in SLOTS[:4]] == [
        ("assigned", "absolute"),
        ("triggered", "absolute"),
        ("assigned", "relative"),
        ("triggered", "relative"),
    ]
    assert [(s.population, s.scale) for s in SLOTS[8:]] == [
        ("assigned", "absolute"),
        ("triggered", "absolute"),
        ("assigned", "relative"),
        ("triggered", "relative"),
    ]
    designs = {d.id: d for d in DESIGNS}
    assert (designs["T1-partial"].n_site, designs["T1-partial"].n_enrolled) == (50_000, 10_000)
    assert (designs["T1-whole"].n_site, designs["T1-whole"].n_enrolled) == (10_000, 10_000)
    assert designs["T4-benign"].sigma_den == 0.5
    assert all(s.truth in {"I*", "rel*", "ratio_impact*", "relative_ratio_impact*"} for s in SLOTS)
    assert all(s.consequence for s in SLOTS)


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


def test_batched_intervals_match_production_on_random_draws():
    from calibration import triggered_sitewide as campaign
    from increment.estimation.sitewide import (
        SitewideContrast,
        SitewideRatioContrast,
        sitewide_impact,
        sitewide_impact_ratio,
    )

    assert callable(getattr(campaign, "batched_estimates", None))
    rng = np.random.default_rng(7182026)
    controls, targets, volumes, denominators, expected = [], [], [], [], []
    sum_controls, sum_targets, sum_volumes, sum_expected = [], [], [], []
    for _ in range(120):
        n_c, n_t = rng.integers(25, 501, 2)
        d_c, d_t = rng.lognormal(0, rng.uniform(0.2, 1.6), n_c), rng.lognormal(0, 1, n_t)
        y_c = d_c * rng.normal(0.4, 0.1, n_c)
        y_t = d_t * rng.normal(0.45, 0.1, n_t)
        v_n, v_d = float(y_c.sum() + y_t.sum() + 3000), float(d_c.sum() + d_t.sum() + 8000)
        controls.append(campaign.arm_moments(y_c, d_c))
        targets.append(campaign.arm_moments(y_t, d_t))
        volumes.append(v_n)
        denominators.append(v_d)
        control = campaign.arm_stats("control", y_c, d_c)
        target = campaign.arm_stats("treatment", y_t, d_t)
        expected.append(
            sitewide_impact_ratio(
                SitewideRatioContrast.from_iid(control, target),
                site_total_numerator=v_n,
                site_total_denominator=v_d,
            )
        )
        a, b = rng.lognormal(0, 1, n_c), rng.lognormal(0, 1, n_t) + rng.uniform(-0.2, 1)
        v = float(a.sum() + b.sum() + 5000)
        sum_controls.append(campaign.arm_moments(a))
        sum_targets.append(campaign.arm_moments(b))
        sum_volumes.append(v)
        sum_expected.append(
            sitewide_impact(
                SitewideContrast.from_iid(
                    campaign.arm_stats("control", a), campaign.arm_stats("treatment", b)
                ),
                site_total_volume=v,
            )
        )
    for result, scalar in (
        (
            campaign.batched_estimates(
                np.array(controls), np.array(targets), np.array(volumes), np.array(denominators)
            ),
            expected,
        ),
        (
            campaign.batched_estimates(
                np.array(sum_controls), np.array(sum_targets), np.array(sum_volumes)
            ),
            sum_expected,
        ),
    ):
        assert result["eligible"].all()
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
            np.testing.assert_allclose(
                result[field], [getattr(x, field) for x in scalar], rtol=0, atol=1e-12
            )


def test_ineligible_replications_are_counted_not_replaced():
    from calibration import triggered_sitewide as campaign

    assert callable(getattr(campaign, "run_design", None))
    design = replace(campaign.DESIGNS[0], n_site=10, n_enrolled=2)
    cell = campaign.run_design(design, reps=3, workers=1)
    assert cell["attempted"] == 3
    assert cell["coverage"]["triggered.absolute"]["eligible"] == 0
    assert cell["coverage"]["triggered.absolute"]["refused"] == 3


def test_batched_ratio_admits_negative_arm_numerator_means():
    from calibration import triggered_sitewide as campaign
    from increment.estimation.sitewide import SitewideRatioContrast, sitewide_impact_ratio

    y_c, y_t = -np.arange(1, 26, dtype=float), -np.arange(1, 36, dtype=float)
    d_c, d_t = np.ones(25), np.ones(35)
    scalar = sitewide_impact_ratio(
        SitewideRatioContrast.from_iid(
            campaign.arm_stats("control", y_c, d_c),
            campaign.arm_stats("treatment", y_t, d_t),
        ),
        site_total_numerator=3000,
        site_total_denominator=8000,
    )
    batch = campaign.batched_estimates(
        np.array([campaign.arm_moments(y_c, d_c)]),
        np.array([campaign.arm_moments(y_t, d_t)]),
        np.array([3000.0]),
        np.array([8000.0]),
    )
    assert batch["eligible"].all()
    assert batch["absolute_impact"][0] == pytest.approx(scalar.absolute_impact, abs=1e-12)


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
