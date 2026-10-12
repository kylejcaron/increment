"""Frozen sitewide designs, production equivalence, and recovery sentinels."""

from __future__ import annotations

import importlib.util


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
