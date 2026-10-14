"""Research data gates, an independent Normal oracle, and chronological replay."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import softmax
from scipy.stats import multivariate_normal

from calibration.historical_prior_stress import selection_sensitivity, simulated_corpus
from calibration.historical_priors import (
    Corpus,
    Evidence,
    ModelSpec,
    evaluate,
    freeze_prior,
    main,
    score_prior,
)
from increment.estimation.priors import MixturePrior


def small_spec() -> ModelSpec:
    return ModelSpec(
        mean_scale=0.12, tau_grid=(0.01, 0.04, 0.1), tau_weights=(0.2, 0.5, 0.3), minimum_history=2
    )


def test_hyperposterior_matches_independent_multivariate_normal_oracle():
    corpus = simulated_corpus("stationary", count=5)
    history, target = corpus.evidence[:-1], corpus.evidence[-1]
    spec = small_spec()
    prior, mass = freeze_prior(history, target, spec)
    y = np.array([row.estimate for row in history])
    se = np.array([row.standard_error for row in history])
    expected_mass, expected_means, expected_sigmas = [], [], []
    for tau, weight in zip(spec.tau_grid, spec.tau_weights, strict=True):
        covariance = np.diag(se**2 + tau**2) + spec.mean_scale**2 * np.ones((4, 4))
        expected_mass.append(np.log(weight) + multivariate_normal.logpdf(y, cov=covariance))
        cross = np.full(4, spec.mean_scale**2)
        mean = cross @ np.linalg.solve(covariance, y)
        var = spec.mean_scale**2 - cross @ np.linalg.solve(covariance, cross)
        expected_means.append(mean)
        expected_sigmas.append(np.sqrt(tau**2 + var))
    np.testing.assert_allclose(mass, softmax(expected_mass), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(prior.means, expected_means, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(prior.sigmas, expected_sigmas, rtol=1e-12, atol=1e-14)
    assert all(sigma > tau for sigma, tau in zip(prior.sigmas, spec.tau_grid, strict=True))


@pytest.mark.parametrize("hazard", ["target", "future", "duplicate", "class"])
def test_freezing_refuses_leakage_and_incomparable_history(hazard):
    corpus = simulated_corpus("stationary", count=4)
    history, target = list(corpus.evidence[:-1]), corpus.evidence[-1]
    if hazard == "target":
        history.append(target)
    elif hazard == "future":
        history[0] = history[0].model_copy(update={"available_at": target.started_at})
    elif hazard == "duplicate":
        history.append(history[0])
    else:
        history[0] = history[0].model_copy(update={"population": "other"})
    with pytest.raises(ValueError):
        freeze_prior(tuple(history), target, small_spec())


@pytest.mark.parametrize("hazard", ["posterior", "missing", "selected", "duplicate", "truth"])
def test_corpus_gate_rejects_invalid_evidence(hazard):
    payload = simulated_corpus("stationary", count=4).model_dump(mode="json")
    if hazard == "posterior":
        payload["evidence"][0]["evidence_kind"] = "shrunk_posterior"
    elif hazard == "missing":
        payload["evidence"].pop()
    elif hazard == "selected":
        payload["reporting_process"] = "significant_only"
    elif hazard == "duplicate":
        payload["evidence"].append(payload["evidence"][0])
    else:
        payload["synthetic"] = False
    with pytest.raises(ValidationError):
        Corpus.model_validate(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"estimate": float("nan")},
        {"estimate": 1e200},
        {"standard_error": 1e-200},
        {"standard_error": float("inf")},
        {"standard_error": 0},
        {"standard_error": 0.5},
        {"started_at": datetime(2020, 1, 1)},
        {"available_at": datetime(2019, 1, 1, tzinfo=UTC)},
    ],
)
def test_evidence_gate(update):
    payload = simulated_corpus("stationary", count=1).evidence[0].model_dump()
    payload.update(update)
    with pytest.raises(ValidationError):
        Evidence.model_validate(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"tau_grid": ()},
        {"tau_grid": (0.1, 0.01, 0.04)},
        {"tau_weights": (0.1, 0.1, 0.1)},
        {"tau_weights": (0.1, -0.1, 1)},
        {"mean_scale": float("nan")},
        {"mean_scale": 1e-200},
        {"mean_scale": 1e200},
        {"minimum_history": 1},
    ],
)
def test_model_gate(update):
    payload = small_spec().model_dump()
    payload.update(update)
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(payload)


def test_target_and_future_observations_cannot_change_frozen_prior():
    corpus = simulated_corpus("stationary", count=6)
    baseline = evaluate(corpus, small_spec())
    payload = corpus.model_dump(mode="json")
    for row in payload["evidence"][3:]:
        row["estimate"] += 1
    replay = evaluate(Corpus.model_validate(payload), small_spec())
    assert (
        baseline["rows"][3]["models"]["learned"]["prior"]
        == replay["rows"][3]["models"]["learned"]["prior"]
    )
    assert (
        baseline["rows"][3]["models"]["learned"]["scores"]
        != replay["rows"][3]["models"]["learned"]["scores"]
    )
    reverse = corpus.model_dump(mode="json")
    reverse["evidence"].reverse()
    assert evaluate(Corpus.model_validate(reverse), small_spec())["rows"] == baseline["rows"]


def test_chronology_uses_availability_and_keeps_every_target():
    corpus = simulated_corpus("stationary", count=5)
    payload = corpus.model_dump(mode="json")
    payload["evidence"][0]["available_at"] = corpus.evidence[3].started_at.isoformat()
    report = evaluate(Corpus.model_validate(payload), small_spec())
    assert len(report["rows"]) == 5
    assert report["rows"][2]["status"] == "unavailable"
    assert report["rows"][2]["reason"] == "insufficient_comparable_history"
    assert report["rows"][3]["training_ids"] == ["experiment-0001", "experiment-0002"]
    assert report["rows"][4]["history_count"] == 4
    assert report["summary"]["paired_targets"] == 2
    assert all(model["targets"] == 2 for model in report["summary"]["models"].values())


def test_class_history_does_not_pool_different_populations():
    payload = simulated_corpus("stationary", count=6).model_dump(mode="json")
    payload["evidence"][4]["population"] = "other"
    report = evaluate(Corpus.model_validate(payload), small_spec())
    assert report["rows"][4]["history_count"] == 0
    assert "experiment-0004" not in report["rows"][5]["training_ids"]


def test_real_corpus_keeps_truth_metrics_null_and_does_not_promote():
    payload = simulated_corpus("stationary", count=4).model_dump(mode="json")
    payload["synthetic"] = False
    for row in payload["evidence"]:
        row["true_effect"] = None
    report = evaluate(Corpus.model_validate(payload), small_spec())
    assert report["recommendation"] == "defer"
    for model in report["rows"][-1]["models"].values():
        assert model["scores"]["posterior_squared_error"] is None
        assert model["scores"]["posterior_covered"] is None
        assert model["scores"]["truth_unavailable_reason"] == "latent_effect_unknown"
    assert (
        report["summary"]["learned_vs_fixed"]["fixed_normal"]["negative_transfer_fraction"] is None
    )
    json.dumps(report, allow_nan=False)


def test_prior_conflict_retains_small_tail_and_measures_harmful_shrinkage():
    target = (
        simulated_corpus("stationary", count=1)
        .evidence[0]
        .model_copy(
            update={
                "estimate": 0.5,
                "standard_error": 0.03,
                "true_effect": 0.5,
            }
        )
    )
    prior = MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(0.02,))
    score = score_prior(prior, target)
    assert 0 < score["prior_conflict_tail"] < 1e-30
    assert score["excess_error_vs_sampling"] > 0
    assert not score["posterior_covered"]
    assert score["posterior_displacement"] < 0


def test_selection_sensitivity_discloses_unavailable_and_complete_scenarios():
    report = selection_sensitivity(simulated_corpus("stationary", count=30))
    assert report["scenarios"]["complete"]["status"] == "evaluated"
    selected = report["scenarios"]["positive_significant_only"]
    assert selected["status"] == "unavailable"
    assert selected["reason"] == "insufficient_comparable_history"


@pytest.mark.integration
def test_cli_evaluates_a_saved_corpus(tmp_path, monkeypatch):
    corpus = simulated_corpus("stationary", count=23)
    source, output = tmp_path / "corpus.json", tmp_path / "report.json"
    source.write_text(corpus.model_dump_json())
    model = tmp_path / "model.json"
    model.write_text(ModelSpec(minimum_history=22).model_dump_json())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibration.historical_priors",
            "--corpus",
            str(source),
            "--model",
            str(model),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["recommendation"] == "defer"
    assert report["summary"]["paired_targets"] == 1
    assert report["model"]["minimum_history"] == 22
    assert report["source_sha256"]
    assert len(report["rows"]) == 23


def test_extreme_common_likelihood_preserves_unequal_tau_weights():
    corpus = simulated_corpus("stationary", count=3)
    history = tuple(
        Evidence.model_validate(
            row.model_dump()
            | {
                "estimate": 1e8,
                "standard_error": 0.03,
            }
        )
        for row in corpus.evidence[:-1]
    )
    spec = ModelSpec(tau_grid=(1e-100, 2e-100), tau_weights=(0.2, 0.8), minimum_history=2)
    prior, mass = freeze_prior(history, corpus.evidence[-1], spec)
    np.testing.assert_allclose(mass, spec.tau_weights, rtol=1e-12, atol=1e-14)
    assert sum(mass) == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(prior.weights, mass, rtol=1e-12, atol=1e-14)


def test_custom_hyperprior_sensitivity_is_broader_and_keeps_relative_tau_weights():
    spec = ModelSpec(
        mean_scale=1.0, tau_grid=(0.6, 0.8, 1.0), tau_weights=(0.1, 0.3, 0.6), minimum_history=2
    )
    report = evaluate(simulated_corpus("stationary", count=3), spec)
    wide = report["sensitivity_model"]
    assert wide["mean_scale"] > spec.mean_scale
    assert tuple(wide["tau_grid"][:-1]) == spec.tau_grid
    assert wide["tau_grid"][-1] > spec.tau_grid[-1]
    old_weights = np.array(wide["tau_weights"][:-1])
    np.testing.assert_allclose(old_weights / old_weights.sum(), spec.tau_weights)
    target = report["rows"][-1]
    actual, _ = freeze_prior(
        simulated_corpus("stationary", count=3).evidence[:-1],
        simulated_corpus("stationary", count=3).evidence[-1],
        ModelSpec.model_validate(wide),
    )
    assert target["models"]["learned_wide"]["prior"] == actual.model_dump()


def test_withheld_targets_emit_every_method_cell_with_reason_and_null_scores():
    report = evaluate(simulated_corpus("stationary", count=3), small_spec())
    assert report["schema_version"] == 2
    for row in report["rows"]:
        assert set(row["models"]) == {"fixed_normal", "fixed_student_t", "learned", "learned_wide"}
        for name, cell in row["models"].items():
            if name.startswith("fixed") or row["status"] == "evaluated":
                assert cell["status"] == "evaluated"
                assert cell["reason"] is None
                assert cell["scores"]["predictive_log_score"] is not None
            else:
                assert cell["status"] == "unavailable"
                assert cell["reason"] == "insufficient_comparable_history"
                assert cell["prior"] is None
                assert cell["tau_posterior"] is None
                for field, value in cell["scores"].items():
                    if field == "truth_unavailable_reason":
                        assert value == cell["reason"]
                    else:
                        assert value is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["estimate", "standard_error", "true_effect"])
def test_boolean_sampling_numbers_are_not_coerced(field):
    payload = simulated_corpus("stationary", count=1).evidence[0].model_dump(mode="json")
    payload[field] = True
    with pytest.raises(ValidationError):
        Evidence.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "intervention_class",
        "metric",
        "population",
        "provenance",
    ],
)
def test_blank_evidence_declarations_are_refused(field):
    payload = simulated_corpus("stationary", count=1).evidence[0].model_dump(mode="json")
    payload[field] = " \t\n "
    with pytest.raises(ValidationError):
        Evidence.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field",
    [
        "corpus_id",
        "scale_justification",
        "class_justification",
        "independence_justification",
        "census_provenance",
    ],
)
def test_blank_corpus_declarations_are_refused(field):
    payload = simulated_corpus("stationary", count=1).model_dump(mode="json")
    payload[field] = " \t\n "
    with pytest.raises(ValidationError):
        Corpus.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "update",
    [
        {"mean_scale": True},
        {"tau_grid": [True, 2]},
        {"tau_weights": [True]},
        {"minimum_history": True},
    ],
)
def test_boolean_model_numbers_are_refused(update):
    payload = small_spec().model_dump(mode="json") | update
    with pytest.raises(ValidationError):
        ModelSpec.model_validate_json(json.dumps(payload))


def test_strict_numbers_keep_normal_json_numbers_arrays_and_timestamps():
    payload = simulated_corpus("stationary", count=3).model_dump(mode="json")
    payload["evidence"][0]["estimate"] = 0
    payload["evidence"][0]["provenance"] = "  audited raw sampling evidence  "
    corpus = Corpus.model_validate_json(json.dumps(payload))
    assert corpus.evidence[0].estimate == 0.0
    assert corpus.evidence[0].provenance == "audited raw sampling evidence"
    spec = ModelSpec.model_validate_json(json.dumps(small_spec().model_dump(mode="json")))
    report = evaluate(corpus, spec)
    json.dumps(report, allow_nan=False)


def test_large_offset_mass_matches_independent_100_digit_absolute_likelihood():
    from decimal import Decimal as D
    from decimal import localcontext

    corpus = simulated_corpus("stationary", count=5)
    spec = ModelSpec(
        mean_scale=1e8, tau_grid=(0.002, 0.01, 0.1), tau_weights=(0.2, 0.3, 0.5), minimum_history=2
    )
    observations = [1e8 + 0.001, 1e8 + 0.04, 1e8 - 0.02, 1e8 + 0.01]
    standard_errors = [0.02, 0.03, 0.04, 0.025]
    history = tuple(
        Evidence.model_validate(
            row.model_dump()
            | {
                "estimate": estimate,
                "standard_error": se,
            }
        )
        for row, estimate, se in zip(
            corpus.evidence[:-1], observations, standard_errors, strict=True
        )
    )
    _, actual = freeze_prior(history, corpus.evidence[-1], spec)
    with localcontext() as context:
        context.prec = 100
        mean_prior_variance = D(spec.mean_scale) ** 2
        log_masses = []
        for tau, weight in zip(spec.tau_grid, spec.tau_weights, strict=True):
            variances = [D(se) ** 2 + D(tau) ** 2 for se in standard_errors]
            precisions = [1 / variance for variance in variances]
            conditional_variance = 1 / (1 / mean_prior_variance + sum(precisions))
            conditional_mean = conditional_variance * sum(
                precision * D(observation)
                for precision, observation in zip(precisions, observations, strict=True)
            )
            quadratic = (
                sum(
                    precision * (D(observation) - conditional_mean) ** 2
                    for precision, observation in zip(precisions, observations, strict=True)
                )
                + conditional_mean**2 / mean_prior_variance
            )
            log_masses.append(
                D(weight).ln()
                - (
                    sum(variance.ln() for variance in variances)
                    + (mean_prior_variance / conditional_variance).ln()
                    + quadratic
                )
                / 2
            )
        unnormalized = [(value - max(log_masses)).exp() for value in log_masses]
        expected = [float(value / sum(unnormalized)) for value in unnormalized]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
