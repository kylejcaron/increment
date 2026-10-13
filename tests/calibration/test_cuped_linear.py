"""Reduced checks of the preregistered mixed-covariate certification cells."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from calibration.cuped_linear import (
    MANIFEST,
    Cell,
    Draw,
    fit_draw,
    generate,
    population_ratio,
    run_chunk,
    summarize,
    trial,
)
from increment.errors import InvalidRequestError
from increment.estimation.armstats import ArmStats
from increment.estimation.cuped import fit_cuped
from tests.mc import nominal_band


def test_revision_eight_manifest_keeps_the_shared_family_allocation():
    from calibration.cuped_linear import (
        EDGE_CELLS,
        ETA,
        FAMILY_SIZE,
        GATED,
        REPLICATIONS,
        gates,
    )
    from tests.mc import family_eta

    assert FAMILY_SIZE == 67
    assert ETA == family_eta(0.01, 67)
    assert REPLICATIONS == {"smoke": 3, "full": 107_100, "reduced": 28_000}
    assert len(MANIFEST) == 55
    assert Counter(cell.statistic for cell in MANIFEST) == {"coverage": 46, "fcp": 9}
    assert len({cell.id for cell in MANIFEST}) == 55
    assert [cell.id for cell in MANIFEST if cell.id.startswith("D4-")] == [
        f"D4-{i}" for i in range(1, 12)
    ]
    assert all(cell.failure_consequence for cell in MANIFEST)
    assert Counter(cell.statistic for cell in GATED) == {"coverage": 40, "fcp": 7}
    assert {cell.id: cell.gating for cell in MANIFEST if cell.gating != "gated"} == dict(EDGE_CELLS)
    assert gates("full") == gates("full").__class__(
        "full", "strict", 107_100, 0.945, 0.055, 0.935, 0.065
    )
    assert gates("reduced") == gates("reduced").__class__(
        "reduced", "reduced", 28_000, 0.94, 0.06, 0.92, 0.08
    )


def test_edge_cells_are_declared_from_the_frozen_admission_evidence():
    from calibration.cuped_linear import ADMISSION_PROVENANCE, EDGE_CELLS, admission_summary

    assert ADMISSION_PROVENANCE["source_sha"] == "6d67ad86903548ca221d05f52cc79f2565bb78e1"
    for identifier, gating in EDGE_CELLS.items():
        upper = admission_summary(identifier)["confidence_interval_95"][1]
        if gating == "unsupported_by_support_region":
            assert upper < 0.01
        else:
            assert identifier == "D4-7" and 0.01 < upper < 0.05
    assert all(
        admission_summary(cell.id)["confidence_interval_95"][1] >= 0.01
        for cell in MANIFEST
        if cell.gating == "gated"
    )


@pytest.mark.parametrize("name", ["full", "reduced"])
def test_each_profile_states_an_acceptance_rule_it_meets(name):
    from calibration.cuped_linear import ETA, FAIL_TARGET, PASS_TARGET, gates, profile_sizing

    record = profile_sizing(name)
    frozen = record["frozen"]
    assert record["frozen_matches_derivation"]
    assert frozen["satisfied"]
    assert frozen["coverage_pass_at_nominal"] >= PASS_TARGET
    assert frozen["fcp_pass_at_nominal"] >= PASS_TARGET
    assert frozen["coverage_fail_at_detectable"] >= FAIL_TARGET
    assert frozen["fcp_fail_at_detectable"] >= FAIL_TARGET
    assert frozen["coverage_false_acceptance_at_tolerance_edge"] <= ETA
    assert frozen["fcp_false_acceptance_at_tolerance_edge"] <= ETA
    assert frozen["family_all_pass_lower_bound"] >= 0.90
    assert gates(name).stopping == "fixed"


def test_cost_projection_covers_only_gated_cells(capsys):
    from calibration.cuped_linear import GATED, cost_projection, main

    projected = cost_projection("reduced")
    assert [row["cell"] for row in projected["cells"]] == [*(c.id for c in GATED), "D8"]
    assert projected["projected_cpu_hours"] == pytest.approx(
        sum(row["projected_cpu_hours"] for row in projected["cells"])
    )
    assert projected["projected_cpu_hours"] < cost_projection("full")["projected_cpu_hours"]
    assert main(["--campaign", "reduced", "--project"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["projected_cpu_hours"] == projected["projected_cpu_hours"]


def test_fast_smoke_preserves_design_and_interval_truth():
    cell = next(cell for cell in MANIFEST if cell.id == "D3-4")
    draw = generate(cell, 0)
    assert len(draw.arms) == 2000
    assert draw.truth == 2.51
    assert np.isnan(draw.covariates["x1"]).sum() > 0
    result = trial(cell, 0)
    assert result["status"] == "eligible"
    assert result["diagnostics"]["dof"] > 0
    assert result["diagnostics"]["width"] > 0


def test_chunk_checkpoint_resume_is_deterministic(tmp_path):
    cell = next(cell for cell in MANIFEST if cell.id == "D3-4")
    job = (cell, str(tmp_path), 0, 2, 2, {"source": "fixed"})
    first = run_chunk(job)
    checkpoint = tmp_path / cell.id / "chunks" / "000000" / "checkpoint.json"
    second = run_chunk(job)
    assert second == first
    checkpoint.unlink()
    replayed = run_chunk(job)
    assert replayed["journal_digest"] == first["journal_digest"]
    assert replayed["hits"] == first["hits"]
    assert replayed["diagnostics"] == first["diagnostics"]
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        run_chunk((*job[:-1], {"source": "different"}))


def test_smoke_never_claims_certification(tmp_path):
    cell = next(cell for cell in MANIFEST if cell.id == "D3-4")
    record = run_chunk((cell, str(tmp_path), 0, 2, 2, {}))
    report = summarize(cell, [record], 2, "smoke")
    assert report["certified"] is False
    assert report["gate_passed"] is None
    assert report["bound"] is not None
    assert all(not d["gated"] for d in report["diagnostics"].values())


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("identifier", ["D3-2", "D3-4", "D4-1"])
def test_representative_coverage_sentinel(identifier, tmp_path):
    cell = next(cell for cell in MANIFEST if cell.id == identifier)
    record = run_chunk((cell, str(tmp_path), 0, 64, 64, {}))
    assert record["status"] == "complete", record
    observed = record["hits"] / record["eligible"]
    lower, upper = nominal_band(0.95, record["eligible"], k=4)
    assert lower <= observed <= upper


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_d1_scalar_point_and_anchor_equal_scalar_moments():
    rng = np.random.default_rng(20261012)
    arms = np.array(["c"] * 40 + ["t"] * 60)
    x = rng.normal(size=100)
    y = 10 + 2 * x + (arms == "t") + rng.normal(size=100)
    fit = fit_draw(Draw(arms, {"x": x}, {"x": "continuous"}, (y,), 1.0))
    stats = [
        ArmStats.from_raw_sums(
            study_id="s",
            metric="m",
            group_id=a,
            n=int(sum(arms == a)),
            sum_y=float(y[arms == a].sum()),
            sum_y2=float(y[arms == a] @ y[arms == a]),
            sum_x=float(x[arms == a].sum()),
            sum_x2=float(x[arms == a] @ x[arms == a]),
            sum_xy=float(x[arms == a] @ y[arms == a]),
        )
        for a in ("c", "t")
    ]
    scalar = fit_cuped(stats)
    assert fit.theta[0] == pytest.approx(scalar.theta, abs=1e-12)
    assert fit.anchor[0] == pytest.approx(scalar.mean_x_pooled, abs=1e-12)
    assert fit.absolute("t").estimate == pytest.approx(
        scalar.adjusted_mean(stats[1]) - scalar.adjusted_mean(stats[0]), abs=1e-12
    )


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize(
    "cells,expected",
    [
        ([(60, 40, 0, 0), (60, 40, 0, 10)], 5.0),
        ([(90, 60, 0, 0), (30, 20, 0, 10)], 2.5),
        (
            [(90, 10, 0, 0), (50, 50, 0, 10)],
            float(
                Fraction(500, 140)
                - (
                    10
                    * (Fraction(50, 140) * Fraction(90, 140) / 139)
                    / (
                        Fraction(50, 140) * Fraction(90, 140) / 139
                        + Fraction(50, 60) * Fraction(10, 60) / 59
                    )
                )
                * (Fraction(50, 140) - Fraction(50, 60))
            ),
        ),
    ],
)
def test_d2_categorical_weighting_witness(cells, expected):
    arms, categories, outcomes = [], [], []
    for k, (nt, nc, mu, effect) in enumerate(cells):
        arms.extend(["t"] * nt + ["c"] * nc)
        categories.extend([str(k)] * (nt + nc))
        outcomes.extend([mu + effect] * nt + [mu] * nc)
    draw = Draw(
        np.array(arms),
        {"country": np.array(categories)},
        {"country": "categorical"},
        (np.array(outcomes, dtype=float),),
        expected,
    )
    fit = fit_draw(draw)
    assert fit.absolute("t").estimate == pytest.approx(expected, abs=1e-10)
    poststrat = sum((nt + nc) * effect for nt, nc, _, effect in cells) / len(arms)
    if expected in (5.0, 2.5):
        assert fit.absolute("t").estimate == pytest.approx(poststrat, abs=1e-10)
    else:
        assert poststrat == 5.0
        weights = [nt * nc / (nt + nc) for nt, nc, _, _ in cells]
        angrist = sum(w * row[3] for w, row in zip(weights, cells, strict=True)) / sum(weights)
        assert angrist == pytest.approx(7.352941176470588)
        interacted = 0.0
        pooled = np.mean(np.array(categories) == "1")
        for sign, arm in ((1, "t"), (-1, "c")):
            mask = np.array(arms) == arm
            indicator = (np.array(categories)[mask] == "1").astype(float)
            observed = np.array(outcomes)[mask]
            slope = np.cov(indicator, observed, ddof=1)[0, 1] / indicator.var(ddof=1)
            interacted += sign * (observed.mean() - slope * (indicator.mean() - pooled))
        assert interacted == pytest.approx(poststrat, abs=1e-10)
        assert fit.absolute("t").estimate != pytest.approx(poststrat, abs=1e-10)


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize(
    "case,code",
    [
        ("duplicate", "rank_deficiency"),
        ("one_arm_level", "level_support"),
        ("single_level", "level_support"),
        ("all_null", "missing_refused"),
        ("negative", "nonpositive_component"),
        ("df", "residual_df"),
    ],
)
def test_d6_named_refusals(case, code):
    rng = np.random.default_rng(20)
    arms = np.array(["c"] * 20 + ["t"] * 20)
    x = rng.normal(size=40)
    covariates, kinds = {"x": x}, {"x": "continuous"}
    y = 10 + x
    if case == "duplicate":
        covariates["copy"] = x.copy()
        kinds["copy"] = "continuous"
    elif case in ("one_arm_level", "single_level"):
        category = np.array(["a"] * 40)
        category[0 if case == "single_level" else slice(0, 20)] = "b"
        covariates, kinds = {"category": category}, {"category": "categorical"}
    elif case == "all_null":
        covariates["x"] = np.full(40, np.nan)
    elif case == "negative":
        y = -10 + x
    elif case == "df":
        arms = np.array(["c"] * 3 + ["t"] * 3)
        matrix = rng.normal(size=(6, 4))
        covariates = {f"x{i}": matrix[:, i] for i in range(4)}
        kinds = dict.fromkeys(covariates, "continuous")
        y = np.arange(6.0)
    from increment.estimation.cuped_linear import fit_cuped_linear

    with pytest.raises(InvalidRequestError) as raised:
        fit = fit_cuped_linear(
            y, arms, control="c", arm_order=("c", "t"), covariates=covariates, kinds=kinds
        )
        fit.relative("t")
    assert raised.value.code == f"estimation.cuped_linear.{code}"
    assert raised.value.context
    expected_keys = {
        "rank_deficiency": {"columns", "partners"},
        "level_support": {"column", "level", "counts"},
        "missing_refused": {"name"},
        "nonpositive_component": {"family", "arm", "value"},
        "residual_df": {"df"},
    }
    assert expected_keys[code] <= set(raised.value.context)
    with pytest.raises(TypeError):
        raised.value.context["changed"] = True  # ty: ignore[invalid-assignment]


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_d7_outcome_independence_and_d9_weighted_hat_identity():
    draw = generate(next(c for c in MANIFEST if c.id == "D3-4"), 0)
    original = fit_draw(draw)
    unrelated = np.random.default_rng(912).normal(size=len(draw.arms))
    frame = {**draw.covariates, "outcome": draw.outcomes[0], "unrelated": unrelated}
    restored = replace(draw, outcomes=(frame["outcome"],))
    second = fit_draw(restored)
    assert second.theta == original.theta
    assert second.anchor == original.anchor
    assert second.absolute("t") == original.absolute("t")
    assert second.covariance == original.covariance
    assert second.dof == original.dof
    assert original.leverage_trace == pytest.approx(2 + original.p, abs=1e-10)


def test_population_ratio_clipping_matches_independent_quadrature():
    from scipy.integrate import quad
    from scipy.stats import gamma

    cmu = np.array([0.0, 0.3, -0.3, 0.1, 0.2])
    for lift in (0.0, 0.05):
        rates = []
        for country in cmu:
            a = 0.2 + 0.02 * country + lift
            cutoff = (0.99 - a) / 0.03
            integrated = quad(
                lambda x, a=a: (a + 0.03 * x) * gamma.pdf(x, 2, scale=1 / 0.6),
                0,
                cutoff,
                epsabs=1e-12,
            )[0]
            integrated += 0.99 * gamma.sf(cutoff, 2, scale=1 / 0.6)
            rates.append(0.3 * a + 0.7 / 0.6**2 * integrated)
        truth = np.exp(cmu) @ rates / (np.exp(cmu).sum() * (0.3 + 0.7 / 0.6**2))
        assert population_ratio(lift) == pytest.approx(truth, abs=1e-12)
    assert math.log(population_ratio(0.05) / population_ratio(0)) > 0


def test_refusal_context_is_serializable_without_losing_nested_counts():
    result = trial(Cell("sparse", "coverage", "mean", "absolute", 10), 0)
    assert result["status"] == "refused"
    restored = json.loads(json.dumps(result))
    assert restored["context"]["counts"]
    assert restored["code"] == result["code"]


def test_d8_precision_is_reported_without_an_acceptance_gate(tmp_path):
    cell = Cell("D8", "descriptive", "mean", "absolute", 2000)
    record = run_chunk((cell, str(tmp_path), 0, 2, 2, {}))
    report = summarize(cell, [record], 2, "full")
    assert report["bound"] is None and report["gate_passed"] is None
    assert report["certified"] is False
    for name in ("variance_ratio_unadjusted", "variance_ratio_scalar_x1"):
        assert report["diagnostics"][name]["mean"] > 0
        assert report["diagnostics"][name]["mcse"] is not None
        assert report["diagnostics"][name]["gated"] is False


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_d9_independently_constructed_weighted_hat_trace_and_welch():
    rng = np.random.default_rng(123)
    arms = np.array(["c"] * 40 + ["t"] * 60)
    x = rng.normal(size=(100, 3))
    y = rng.normal(size=100)
    weights = np.where(arms == "c", 1 / (40 * 39), 1 / (60 * 59))
    centered = x.copy()
    for arm in ("c", "t"):
        centered[arms == arm] -= x[arms == arm].mean(axis=0)
    design = np.column_stack([arms == "c", arms == "t", centered])
    weighted = np.sqrt(weights)[:, None] * design
    hat = weighted @ np.linalg.inv(weighted.T @ weighted) @ weighted.T
    fit = fit_draw(
        Draw(
            arms,
            {f"x{i}": x[:, i] for i in range(3)},
            {f"x{i}": "continuous" for i in range(3)},
            (y,),
            0.0,
        )
    )
    assert np.trace(hat) == pytest.approx(2 + fit.p, abs=1e-11)
    assert fit.leverage_trace == pytest.approx(np.trace(hat), abs=1e-11)
    assert fit.max_leverage == pytest.approx(np.diag(hat).max(), abs=1e-11)
    unadjusted = fit_draw(Draw(arms, {}, {}, (y,), 0.0)).absolute("t")
    welch = y[arms == "c"].var(ddof=1) / 40 + y[arms == "t"].var(ddof=1) / 60
    assert unadjusted.se**2 == pytest.approx(welch, rel=1e-12)


def test_full_gate_cutoffs_match_revision_seven_sizing():
    from calibration.cuped_linear import ETA, FULL_REPLICATIONS, REDUCED_REPLICATIONS
    from tests.mc import coverage_lower_bound, kl_chernoff_upper_bound

    r = FULL_REPLICATIONS
    assert coverage_lower_bound(r - 5609, r, ETA) >= 0.945
    assert coverage_lower_bound(r - 5610, r, ETA) < 0.945
    assert kl_chernoff_upper_bound(5568 / r, r, ETA) <= 0.055
    assert kl_chernoff_upper_bound(5569 / r, r, ETA) > 0.055
    r = REDUCED_REPLICATIONS
    assert coverage_lower_bound(r - 1530, r, ETA) >= 0.94
    assert coverage_lower_bound(r - 1531, r, ETA) < 0.94
    assert kl_chernoff_upper_bound(1509 / r, r, ETA) <= 0.06
    assert kl_chernoff_upper_bound(1510 / r, r, ETA) > 0.06


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("name", ["full", "reduced"])
def test_galloping_sizing_matches_the_linear_lattice_scan(name):
    from calibration.cuped_linear import gates, sizing

    gate = gates(name)
    detectable = (gate.detectable_coverage, gate.detectable_fcp)
    first = next(
        r
        for r in range(100, gate.replications + 1, 100)
        if sizing(r, gate.coverage_floor, gate.fcp_ceiling, detectable)["satisfied"]
    )
    assert first == gate.replications
    assert all(
        sizing(r, gate.coverage_floor, gate.fcp_ceiling, detectable)["satisfied"]
        for r in range(gate.replications, gate.replications + 5001, 100)
    )


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize(
    "pvalues,selected,alpha",
    [
        ([0.01, 0.20], [0], 0.025),
        ([0.005, 0.02, 0.80], [0, 1], 1 / 30),
    ],
)
def test_d9_partial_bh_selection_reinverts_cached_projections(pvalues, selected, alpha):
    from pathlib import Path

    from increment.estimation.family import bh_select
    from tests.oracles.generate.gen_cuped_linear_oracle import reference

    indices, threshold = bh_select(pvalues, 0.05)
    assert indices == selected
    assert threshold == pytest.approx(alpha, abs=1e-16)
    data = json.loads(
        (Path(__file__).parents[1] / "oracles" / "data" / "cuped_linear.json").read_text()
    )
    for index in indices:
        inp = data["unequal_allocation" if index == 0 else "ratio"]
        outcomes = np.asarray(inp["outcomes"])
        draw = Draw(np.asarray(inp["arms"]), inp["covariates"], inp["kinds"], tuple(outcomes), 0.0)
        fit = fit_draw(draw)
        scale = "absolute" if index == 0 else "ratio_absolute"
        result = (
            fit.absolute("t", alpha=alpha)
            if index == 0
            else fit.ratio("t", scale="absolute", alpha=alpha)
        )
        nominal = fit.absolute("t") if index == 0 else fit.ratio("t", scale="absolute")
        expected = reference(np.asarray(inp["encoded_x"]), draw.arms, outcomes, scale, alpha=alpha)
        assert result.lower == pytest.approx(expected["lower"], abs=1e-12)
        assert result.upper == pytest.approx(expected["upper"], abs=1e-12)
        assert result.lower < nominal.lower
        assert result.upper > nominal.upper


def test_support_admission_probe_is_descriptive_and_resumable(tmp_path):
    from calibration.cuped_linear import run_admission_cell

    cell = next(c for c in MANIFEST if c.id == "D3-4")
    job = (cell, str(tmp_path), 3, {"source": "fixed"})
    first = run_admission_cell(job)
    assert first["attempts"] == 3
    assert first["admitted"] == 3
    assert first["admitted_fraction"] == 1
    assert 0 < first["confidence_interval_95"][0] < 1
    assert first["confidence_interval_95"][1] == 1
    assert first["projected_attempts"] == {"full": 107_100, "reduced": 28_000}
    assert first["gated"] is False
    assert run_admission_cell(job) == first


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_d7_declared_unrelated_metric_does_not_change_public_inference():
    import pandas as pd

    from increment import Analysis, AnalysisPlan, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.frame import CupedCovariate, MetricSpec

    draw = generate(next(c for c in MANIFEST if c.id == "D3-4"), 0)
    frame = pd.DataFrame(
        {
            **draw.covariates,
            "unit": np.arange(len(draw.arms)),
            "arm": draw.arms,
            "outcome": draw.outcomes[0],
            "unrelated": 10 + np.random.default_rng(912).normal(size=len(draw.arms)),
        }
    )
    target = MetricSpec(
        name="outcome",
        covariates=tuple(
            CupedCovariate(name=name, kind=kind, missing="impute-indicator")
            for name, kind in draw.kinds.items()
        ),
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
    )
    observations = []
    for include_unrelated in (False, True):
        metrics = [target, MetricSpec(name="unrelated")] if include_unrelated else [target]
        plan = AnalysisPlan(
            primary="outcome", secondaries=("unrelated",) if include_unrelated else ()
        )
        analysis = Analysis.from_unit_summary(
            frame, unit="unit", group="arm", control="c", metrics=metrics, plan=plan
        )
        rows = analysis.run()
        assert isinstance(rows, LiftEstimates)
        (row,) = [row for row in rows if row.metric == "outcome"]
        observations.append(
            (
                row.abs_diff,
                row.abs_se,
                row.abs_reference_df,
                row.require_lift().value,
                row.require_lift().log_se,
            )
        )
    assert observations[0] == observations[1]


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_d6_all_null_imputed_column_is_disclosed_without_population_loss():
    arms = np.array(["c"] * 40 + ["t"] * 60)
    y = np.random.default_rng(97).normal(size=100)
    fit = fit_draw(Draw(arms, {"x": np.full(100, np.nan)}, {"x": "continuous"}, (y,), 0.0))
    assert fit.p == 0
    assert fit.layout.structural_omissions == ("x",)
    assert fit.n_by_arm == (40, 60)


def test_family_refuses_an_unavailable_projection_before_public_selection(monkeypatch):
    import calibration.cuped_linear as campaign

    cell = next(item for item in MANIFEST if item.id == "D9-4")
    members = campaign.family_members(cell, 0)
    draw, scale = members[1]
    negative_mean = replace(draw, outcomes=(-draw.outcomes[0],))
    altered = (members[0], (negative_mean, scale), members[2])
    monkeypatch.setattr(campaign, "family_members", lambda *_: altered)
    result = trial(cell, 0)
    assert result["status"] == "refused"
    assert result["code"] == "estimation.cuped_linear.nonpositive_component"


def test_metric_streams_never_collide_within_any_profile_attempt_limit():
    from calibration.cuped_linear import REPLICATIONS, design_stream, family_members

    limit = 100 * max(REPLICATIONS.values())
    assert limit == 10_710_000
    assert design_stream(7).entropy == 1007 and design_stream(7).spawn_key == ()
    assert np.random.default_rng(design_stream(7)).random() == np.random.default_rng(1007).random()
    assert design_stream(7, 2).spawn_key == (2,)
    # Distinct (entropy, spawn_key) pairs for every attempt and metric: entropy
    # is injective in the replication and the spawn key separates metrics.
    keys = {
        (design_stream(r, m).entropy, design_stream(r, m).spawn_key)
        for r in (0, limit - 1)
        for m in range(3)
    }
    assert len(keys) == 6
    for metric in (1, 2):
        crossover = 10_000_000 * metric
        assert (design_stream(crossover, 0).entropy, design_stream(crossover, 0).spawn_key) != (
            design_stream(0, metric).entropy,
            design_stream(0, metric).spawn_key,
        )
    cell = next(item for item in MANIFEST if item.id == "D9-4")
    late = family_members(cell, 10_000_000)
    early = family_members(cell, 0)
    assert not np.array_equal(late[0][0].outcomes[0], early[1][0].outcomes[0])
    assert len({draw.outcomes[0][:8].tobytes() for draw, _ in early}) == 3


def test_interrupted_chunk_resumes_without_overlapping_its_journal(tmp_path, monkeypatch):
    import calibration.cuped_linear as campaign
    from calibration.journal import verify

    cell = next(item for item in MANIFEST if item.id == "D3-4")
    job = (cell, str(tmp_path / "run"), 0, 3, 3, {"source": "fixed"})
    original = campaign.trial
    seen = []

    def interrupting(cell, replication):
        seen.append(replication)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return original(cell, replication)

    monkeypatch.setattr(campaign, "trial", interrupting)
    with pytest.raises(KeyboardInterrupt):
        run_chunk(job)
    directory = tmp_path / "run" / cell.id / "chunks" / "000000"
    interrupted = verify(directory, case_id=f"{cell.id}:0")
    assert (interrupted.started, interrupted.completed) == (1, 1)
    monkeypatch.setattr(campaign, "trial", original)
    resumed = run_chunk(job)
    clean = run_chunk((cell, str(tmp_path / "clean"), 0, 3, 3, {"source": "fixed"}))
    assert resumed["status"] == "complete"
    assert (resumed["hits"], resumed["attempted"], resumed["diagnostics"]) == (
        clean["hits"],
        clean["attempted"],
        clean["diagnostics"],
    )
    totals = verify(directory, case_id=f"{cell.id}:0")
    assert totals.started == totals.completed == resumed["attempted"]
    assert totals.digest == resumed["journal_digest"]


def test_interrupted_admission_probe_resumes_without_overlapping_its_journal(tmp_path, monkeypatch):
    import calibration.cuped_linear as campaign
    from calibration.journal import verify

    cell = next(item for item in MANIFEST if item.id == "D3-4")
    job = (cell, str(tmp_path), 3, {"source": "fixed"})
    original = campaign.admission_trial
    seen = []

    def interrupting(cell, replication):
        seen.append(replication)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return original(cell, replication)

    monkeypatch.setattr(campaign, "admission_trial", interrupting)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_admission_cell(job)
    monkeypatch.setattr(campaign, "admission_trial", original)
    resumed = campaign.run_admission_cell(job)
    assert resumed["admitted"] == 3
    totals = verify(tmp_path / cell.id, case_id=cell.id)
    assert totals.started == totals.completed == 3
    assert totals.digest == resumed["journal_digest"]
