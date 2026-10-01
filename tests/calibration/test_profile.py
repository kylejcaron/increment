"""Profiles select declared cells and an enumerated tolerance; never a free one."""

from __future__ import annotations

import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

from calibration.profile import CellSet, ProfileError, available, campaigns, load
from calibration.stopping import SEQUENTIAL_RULES
from tests.mc import family_eta, scientific_delta

CASES = [
    {"case_id": "a", "dgp": {"n_c": 50, "n_t": 50, "quantile": 0.95}},
    {"case_id": "b", "dgp": {"n_c": 50, "n_t": 50, "quantile": 0.95}},
    {"case_id": "c", "dgp": {"n_c": 50, "n_t": 50, "quantile": 0.99}},
    {"case_id": "d", "dgp": {"n_c": 200, "n_t": 50, "quantile": 0.95}},
]


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "profiles.yaml"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_release_certifies_every_case_and_local_reports_its_omissions():
    release = load("release")
    assert (release.delta, release.cells.rule, release.stopping) == (0.005, "every-case", "fixed")
    assert release.sequential_rule is None
    assert all(release.certification(CASES).values())

    local = load("local")
    assert local.delta == 0.01
    certification = local.certification(CASES)
    # One case per (n_c, n_t, quantile): "b" duplicates "a" and is not certified.
    assert certification == {"a": True, "b": False, "c": True, "d": True}


def test_declared_profiles_are_discoverable():
    assert set(available()) == {"release", "release-sequential", "local"}
    sequential = load("release-sequential")
    # `stopping` names the class of rule; `sequential_rule` pins the exact
    # procedure, and only a procedure the package implements is accepted.
    assert sequential.stopping == "sequential"
    assert sequential.sequential_rule in SEQUENTIAL_RULES
    assert load("local").sequential_rule == sequential.sequential_rule


def test_the_strict_tolerance_reproduces_the_repository_scientific_delta():
    release = load("release")
    for nominal_error in (0.01, 0.025, 0.05, 0.1, 0.2):
        assert release.tolerance_at(nominal_error) == pytest.approx(scientific_delta(nominal_error))
    # The reduced tolerance is genuinely weaker where the release one saturates.
    assert load("local").tolerance_at(0.05) == 0.01


@pytest.mark.parametrize(
    "name,expected_rule",
    [("release", "fixed"), ("release-sequential", "sprt-v1"), ("local", "sprt-v1")],
)
def test_every_profile_builds_a_rule_that_runs_to_a_decision(name, expected_rule):
    profile = load(name)
    rule = profile.stopping_rule(
        nominal_error=0.05, eta=family_eta(0.01, 1692), repetitions=189_997
    )
    assert rule.rule == expected_rule
    assert rule.design.tolerance == profile.tolerance_at(0.05)
    # A stream that never misses is decisively inside every declared
    # tolerance, so each profile's rule must accept it.
    assert rule.run(iter(lambda: False, None)) == "accept"
    assert rule.parameters()["rule"] == expected_rule


@pytest.mark.parametrize(
    "document,reason",
    [
        ({"profiles": {}}, "no declared profiles"),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {"p": {"tolerance": "loose", "cells": "all", "stopping": "fixed"}},
            },
            "undeclared tolerance",
        ),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {"p": {"tolerance": "strict", "cells": "all", "stopping": "peek"}},
            },
            "undeclared stopping rule",
        ),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed"],
                "cell_sets": {"all": {"rule": "invent-cells"}},
                "profiles": {"p": {"tolerance": "strict", "cells": "all", "stopping": "fixed"}},
            },
            "unsupported cell rule",
        ),
        (
            {
                "tolerances": {"strict": 5.0},
                "stopping": ["fixed"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {"p": {"tolerance": "strict", "cells": "all", "stopping": "fixed"}},
            },
            "tolerance outside (0, 1)",
        ),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {"p": {"tolerance": "strict", "cells": "all", "delta": 0.2}},
            },
            "free delta override",
        ),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed", "sequential"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {
                    "p": {"tolerance": "strict", "cells": "all", "stopping": "sequential"}
                },
            },
            "sequential stopping with no named procedure",
        ),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed", "sequential"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {
                    "p": {
                        "tolerance": "strict",
                        "cells": "all",
                        "stopping": "sequential",
                        "sequential_rule": "sprt-v2",
                    }
                },
            },
            "unimplemented sequential procedure",
        ),
        (
            {
                "tolerances": {"strict": 0.005},
                "stopping": ["fixed", "sequential"],
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {
                    "p": {
                        "tolerance": "strict",
                        "cells": "all",
                        "stopping": "fixed",
                        "sequential_rule": "sprt-v1",
                    }
                },
            },
            "fixed stopping with a stray sequential rule",
        ),
    ],
)
def test_malformed_profiles_are_refused(tmp_path, document, reason):
    with pytest.raises(ProfileError):
        load("p", _write(tmp_path, document))
    assert reason


# Predicted reach: the longest cost-ordered prefix of the 817-cell manifest whose
# modelled core-hours fit the declared budget. The next cell would cross it:
# smoke 638.19 + 26.95 > 650; nightly 6,042.84 + 108.89 > 6,100;
# weekly 49,331.54 + 425.35 > 49,600; release 980,171.92 + 11,870.83 > 990,000.
_LADDER = {"smoke": 73, "nightly": 176, "weekly": 373, "release": 675}


def test_each_campaign_resolves_its_own_profiles_without_seeing_the_others():
    assert set(campaigns()) == {"i15", "sequential"}
    assert set(available(campaign="sequential")) == set(_LADDER)
    # Both campaigns declare a `release`, and they are different objects
    # resolved from different sections against the same enumerations.
    assert load("release", campaign="sequential").cells.id_field == "name"
    assert load("release").cells.id_field == "case_id"
    with pytest.raises(ProfileError):
        load("release-sequential", campaign="sequential")
    with pytest.raises(ProfileError):
        load("smoke")
    with pytest.raises(ProfileError):
        load("release", campaign="i11")


def test_no_sequential_tier_certifies_the_whole_manifest():
    """Every tier, including the largest, must name what it did not reach.

    The manifest is not certifiable on any schedule this project has, so a
    profile that reported full coverage would be claiming something no run can
    support. Each tier admits the cheapest cells its declared budget affords
    and reports the rest as uncertified rather than dropping them.
    """
    from calibration.sequential import manifest_records

    cases = manifest_records()
    reached = {}
    for name, expected in _LADDER.items():
        profile = load(name, campaign="sequential")
        certification = profile.certification(cases)
        # Every declared cell is accounted for, certified or not.
        assert set(certification) == {case["name"] for case in cases}
        certified = {case_id for case_id, value in certification.items() if value}
        assert len(certified) == expected
        assert len(certification) - len(certified) == len(cases) - expected
        spent = sum(case["cost_core_hours"] for case in cases if certification[case["name"]])
        assert spent <= profile.cells.budget
        reached[name] = certified
    assert len(reached["release"]) < len(cases)
    # The ladder is nested: a bigger budget never drops a cell a smaller one
    # certified, so a tier's claim always contains the tier below it.
    for smaller, larger in (("smoke", "nightly"), ("nightly", "weekly"), ("weekly", "release")):
        assert reached[smaller] < reached[larger]


def test_a_cost_budget_admits_the_cheapest_cells_not_the_first_ones():
    """Ordering by cost is the point; manifest order would spend it all up front."""
    cells = CellSet(
        name="budgeted",
        rule="cheapest-within-budget",
        axes=(),
        id_field="name",
        axis_field=None,
        budget=6.0,
    )
    cases = [
        {"name": "expensive-and-first", "cost_core_hours": 100.0},
        {"name": "cheap", "cost_core_hours": 1.0},
        {"name": "unaffordable", "cost_core_hours": 5.0},
        {"name": "also-cheap", "cost_core_hours": 2.0},
    ]
    # Cheapest first: 1 then 2 spends 3 of 6, and the 5.0 cell would take it
    # to 8, so it is refused even though 3 of the budget is unspent. The
    # result comes back in manifest order, not cost order.
    assert cells.select(cases) == ("cheap", "also-cheap")
    # Manifest order would have spent the whole budget on nothing.
    assert cells.select(cases[:1]) == ()
    with pytest.raises(ProfileError):
        cells.select([{"name": "uncosted"}])


def test_the_declared_cost_model_reproduces_its_measured_anchors():
    """The ordering rests on a fitted model, so its anchors are pinned here.

    The anchors are timed replications of the raw record-bearing path, so they
    are reproduced at that path's shape: the confidence sequence inverted at
    every look, and capture paying the cubic ancestor replay the snapshot
    identity performed when they were measured. A coefficient change that
    moves an anchor outside the declared band has changed what the ladder
    means and must fail here.
    """
    from calibration.sequential import case_cost_seconds, cost_model
    from tests.estimation._sequential_acceptance import principal_manifest

    model = cost_model()
    cases = {case.name: case for case in principal_manifest()}
    # Keyed by cell name, the manifest's identity, so adding or removing other
    # cells cannot silently point an anchor at a different cell.
    measured = {
        "bernoulli-0.001-(1, 1)-2--1/5": 0.00441,
        "bernoulli-0.001-(1, 1)-25--1/5": 10.150,
        "bernoulli-0.001-(1, 1)-100--1/5": 64.221,
        "gaussian-1-(1, 1)-25": 8.942,
        "gaussian_ratio-1-(1, 1)-14": 44.600,
        "family-10-1.0-independent-first_rejection-14": 58.800,
    }
    anchored = {**model, "bounds_at": "every-look", "capture": "pre-fix"}
    for name, seconds in measured.items():
        ratio = case_cost_seconds(cases[name], anchored) / seconds
        assert 1 - model["band"] <= ratio <= 1 + model["band"], (name, ratio)
    # The executed route builds no records and inverts once, so the same cell
    # is an order of magnitude cheaper than the path the anchors were timed on.
    assert model["capture"] == "none" and model["bounds_at"] == "stopped-look"
    hundred_looks = cases["bernoulli-0.001-(1, 1)-100--1/5"]
    assert case_cost_seconds(hundred_looks, model) < case_cost_seconds(hundred_looks, anchored) / 4


def test_a_reference_profile_over_every_cell_reproduces_the_declared_ledger(tmp_path):
    """The profile machinery must not be able to move the declared design.

    No shipped tier certifies every cell, so the invariant is pinned against a
    reference declaration: the strict tolerance over the whole manifest has a
    tolerance ratio of 1, restates each gate as itself, and must resolve the
    frozen ledger field for field -- the per-decision error, the binding gate
    and the whole replication total included.
    """
    from calibration.sequential import campaign_plan
    from tests.estimation._sequential_acceptance import CERTIFICATION_LEDGER

    path = _write(
        tmp_path,
        {
            "tolerances": {"strict": 0.005},
            "stopping": ["fixed", "sequential"],
            "cell_sets": {"all": {"rule": "every-case"}},
            "profiles": {"p": {"tolerance": "strict", "cells": "all", "stopping": "fixed"}},
            "sequential": {
                "cell_sets": {"all": {"rule": "every-case"}},
                "profiles": {
                    "reference": {"tolerance": "strict", "cells": "all", "stopping": "fixed"}
                },
            },
        },
    )
    plan = campaign_plan(load("reference", path, campaign="sequential"))
    assert plan["tolerance_ratio"] == 1
    assert plan["reproduces_declared_ledger"]
    assert plan["ledger"] == asdict(CERTIFICATION_LEDGER)
    # 3,833 gates at 8 looks give 30,664 decisions sharing the 1/100 family error.
    # Binding gate: rare-event availability floor 0.495765 (threshold 0.488265 vs
    # worst case 0.485765, divergence 1.2509e-5): ceil(ln(3,066,400) / 1.2509e-5)
    # = 1,194,022 -> 1,195,000 replications. The total sums each case's hardest gate.
    assert plan["ledger"]["case_count"] == 817
    assert plan["ledger"]["budgeted_gate_count"] == 3_833
    assert plan["ledger"]["per_decision_error"] == Fraction(1, 3_066_400)
    assert plan["ledger"]["binding_gate"] == "point_availability_rare_event"
    assert plan["ledger"]["max_case_replications"] == 1_195_000
    assert plan["ledger"]["total_replications"] == 365_743_000


def test_a_weaker_sequential_tolerance_rescales_the_margin_that_drives_cost():
    """Halving the resolution is what makes a profile cheaper, not loosening alone.

    A gate's replication count inverts a divergence whose two arguments are
    ``margin`` apart, so a profile that moved only the tolerance would make the
    campaign MORE expensive while claiming less. The reduced profile scales the
    margin with it, which is what buys the roughly fourfold cut per gate.
    """
    from calibration.sequential import campaign_plan, rescaled_gate
    from tests.estimation._sequential_acceptance import ACCEPTANCE_GATES, CERTIFICATION_DESIGN

    plan = campaign_plan(load("smoke", campaign="sequential"))
    assert plan["tolerance_ratio"] == 2
    assert plan["profile_margins"] == {
        "scientific_excess": Fraction(1, 100),
        "scientific_excess_margin": Fraction(1, 200),
        "availability_slack": Fraction(1, 50),
        "availability_margin": Fraction(1, 200),
    }
    gate = next(
        gate
        for _, gates in ACCEPTANCE_GATES
        for gate in gates
        if gate.name == "ever_null_rejection"
    )
    assert (gate.tolerance, gate.margin) == (
        CERTIFICATION_DESIGN.scientific_excess,
        CERTIFICATION_DESIGN.max_mc_margin,
    )
    rescaled = rescaled_gate(gate, Fraction(2))
    assert (rescaled.tolerance, rescaled.margin) == (gate.tolerance * 2, gate.margin * 2)
    assert rescaled_gate(gate, Fraction(1)) is gate
