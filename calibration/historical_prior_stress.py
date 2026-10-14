"""Known-truth model, shift and reporting-selection sensitivity for issue 70."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from scipy.stats import norm

from calibration.historical_priors import (
    DEFAULT_MODEL,
    Corpus,
    Evidence,
    evaluate,
    freeze_prior,
    score_prior,
)


def simulated_corpus(regime: str, *, seed: int = 70, count: int = 80) -> Corpus:
    """One predeclared cohort, preserving unsuccessful and negative experiments."""
    if regime not in {"stationary", "heavy_tailed", "shift"}:
        raise ValueError("unknown synthetic regime")
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 0.02, count)
    if regime == "heavy_tailed":
        truth = rng.standard_t(3, count) * 0.05
    if regime == "shift":
        truth[count // 2 :] += 0.15
    se = rng.uniform(0.02, 0.06, count)
    observations = rng.normal(truth, se)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    evidence = tuple(
        Evidence(
            experiment_id=f"experiment-{i:04}",
            intervention_class="synthetic_intervention",
            metric="synthetic_positive_mean",
            population="synthetic_users",
            started_at=start + timedelta(days=2 * i),
            available_at=start + timedelta(days=2 * i + 1),
            estimate=float(observations[i]),
            standard_error=float(se[i]),
            evidence_kind="normal_sampling",
            provenance=f"Normal sampling generator seed={seed}; index={i}",
            true_effect=float(truth[i]),
        )
        for i in range(count)
    )
    return Corpus(
        corpus_id=f"synthetic-{regime}-{seed}",
        synthetic=True,
        effect_scale="log_relative_risk",
        scale_justification="Known latent log relative risk in a Normal observation model",
        class_justification="One simulated intervention, metric and population class",
        independence_justification="Independent latent effects and observation noise",
        reporting_process="complete_census",
        census_provenance=f"generator seed={seed}; n={count}",
        experiment_roster=tuple(row.experiment_id for row in evidence),
        evidence=evidence,
    )


def selection_sensitivity(corpus: Corpus) -> dict:
    """Deliberately biased training scenarios, evaluated on the same unselected target."""
    target = corpus.evidence[-1]
    full = corpus.evidence[:-1]
    histories = {
        "complete": full,
        "positive_significant_only": tuple(
            row for row in full if (row.estimate / row.standard_error > norm.isf(0.025))
        ),
        "nonpositive_missing": tuple(row for row in full if row.estimate > 0),
    }
    rows = {}
    for scenario, history in histories.items():
        if len(history) < DEFAULT_MODEL.minimum_history:
            rows[scenario] = {
                "status": "unavailable",
                "reason": "insufficient_comparable_history",
                "history_count": len(history),
            }
            continue
        prior, _ = freeze_prior(history, target, DEFAULT_MODEL)
        rows[scenario] = {
            "status": "evaluated",
            "history_count": len(history),
            "training_ids": [row.experiment_id for row in history],
            "prior": prior.model_dump(),
            "scores": score_prior(prior, target),
        }
    return {
        "scope": "synthetic reporting-mechanism sensitivity, not a correction for a selected real corpus",
        "target_id": target.experiment_id,
        "scenarios": rows,
    }


def run_stress() -> dict:
    """Seeded diagnostic runs; no Monte Carlo certification or promotion decision."""
    reports = {}
    for regime in ("stationary", "heavy_tailed", "shift"):
        corpus = simulated_corpus(regime)
        report = evaluate(corpus)
        midpoint = len(corpus.evidence) // 2
        from calibration.historical_priors import summarize

        report["early_summary"] = summarize(report["rows"][:midpoint])
        report["late_summary"] = summarize(report["rows"][midpoint:])
        reports[regime] = report
    return {
        "recommendation": "defer",
        "recommendation_reason": "no suitable historical corpus supplied",
        "seed": 70,
        "experiments_per_regime": 80,
        "evidence_status": "diagnostic synthetic runs; no empirical or coverage certification",
        "regimes": reports,
        "selection_sensitivity": selection_sensitivity(simulated_corpus("shift")),
    }
