"""Corpus-gated, chronological effect-prior research; never a production trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import logsumexp
from scipy.stats import norm

from increment.estimation.priors import (
    MixturePosterior,
    MixturePrior,
    StudentTPrior,
    mixture_posterior,
)


def _validate_scale(value: float) -> None:
    variance = value * value
    if variance == 0 or not math.isfinite(variance) or not math.isfinite(1 / variance):
        raise ValueError("scale variance and precision must be numerically representable")


class ResearchInput(BaseModel):
    """Strict, immutable research declarations with finite numeric inputs."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class Evidence(ResearchInput):
    """One independent experiment's unshrunk log-relative-risk sampling evidence."""

    experiment_id: str = Field(min_length=1)
    intervention_class: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    population: str = Field(min_length=1)
    started_at: datetime
    available_at: datetime
    estimate: float = Field(strict=True)
    standard_error: float = Field(gt=0, lt=0.5, strict=True)
    evidence_kind: Literal["normal_sampling"]
    provenance: str = Field(min_length=1)
    true_effect: float | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def chronology(self):
        _validate_scale(self.standard_error)
        standardized = self.estimate / self.standard_error
        if not math.isfinite(standardized * standardized):
            raise ValueError("sampling evidence quadratic must be numerically representable")
        if self.started_at.utcoffset() is None or self.available_at.utcoffset() is None:
            raise ValueError("timestamps must include timezones")
        if self.available_at <= self.started_at:
            raise ValueError("evidence must become available after the experiment starts")
        return self

    @property
    def comparable_class(self) -> tuple[str, str, str]:
        return self.intervention_class, self.metric, self.population


class Corpus(ResearchInput):
    """Audited complete census, including null, negative and unsuccessful tests."""

    corpus_id: str = Field(min_length=1)
    synthetic: bool = Field(default=False, strict=True)
    effect_scale: Literal["log_relative_risk"]
    scale_justification: str = Field(min_length=1)
    class_justification: str = Field(min_length=1)
    independence_justification: str = Field(min_length=1)
    reporting_process: Literal["complete_census"]
    census_provenance: str = Field(min_length=1)
    experiment_roster: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def census(self):
        ids = [row.experiment_id for row in self.evidence]
        if len(set(ids)) != len(ids) or len(set(self.experiment_roster)) != len(
            self.experiment_roster
        ):
            raise ValueError(
                "duplicate experiment: correlated contrasts are not independent history"
            )
        if set(ids) != set(self.experiment_roster):
            raise ValueError("missing or selected evidence: supply the complete experiment census")
        if not self.synthetic and any(row.true_effect is not None for row in self.evidence):
            raise ValueError("known true effects belong only to synthetic corpora")
        return self


class ModelSpec(ResearchInput):
    """Predeclared discrete heterogeneity prior and Normal mean hyperprior."""

    mean_scale: float = Field(default=0.1, gt=0, strict=True)
    tau_grid: tuple[Annotated[float, Field(strict=True)], ...] = (
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
    )
    tau_weights: tuple[Annotated[float, Field(strict=True)], ...] = (1 / 7,) * 7
    minimum_history: int = Field(default=20, ge=2, strict=True)

    @model_validator(mode="after")
    def grid(self):
        _validate_scale(self.mean_scale)
        if len(self.tau_grid) != len(self.tau_weights) or not self.tau_grid:
            raise ValueError("tau grid and weights must have matching nonempty lengths")
        if any(t <= 0 for t in self.tau_grid) or tuple(sorted(set(self.tau_grid))) != self.tau_grid:
            raise ValueError("tau grid must be positive and strictly increasing")
        for tau in self.tau_grid:
            _validate_scale(tau)
        if any(w <= 0 for w in self.tau_weights) or not math.isclose(sum(self.tau_weights), 1):
            raise ValueError("tau weights must be positive and sum to one")
        return self


DEFAULT_MODEL = ModelSpec()


def wider_model(spec: ModelSpec) -> ModelSpec:
    """Broaden the supplied hyperprior while retaining its existing relative tau weights."""
    count = len(spec.tau_grid)
    return ModelSpec(
        mean_scale=2 * spec.mean_scale,
        tau_grid=(*spec.tau_grid, 2.5 * spec.tau_grid[-1]),
        tau_weights=(
            *(weight * count / (count + 1) for weight in spec.tau_weights),
            1 / (count + 1),
        ),
        minimum_history=spec.minimum_history,
    )


def freeze_prior(
    history: tuple[Evidence, ...], target: Evidence, spec: ModelSpec
) -> tuple[MixturePrior, tuple[float, ...]]:
    """Integrate both hyperparameters using only evidence available before target start."""
    ids = [row.experiment_id for row in history]
    if target.experiment_id in ids or len(set(ids)) != len(ids):
        raise ValueError("target leakage or duplicate experiment in training history")
    if any(row.available_at >= target.started_at for row in history):
        raise ValueError("training evidence was not available before target start")
    if any(row.comparable_class != target.comparable_class for row in history):
        raise ValueError("training history must share the justified target class")
    if len(history) < spec.minimum_history:
        raise ValueError("insufficient comparable history")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return _fit_prior(history, spec)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("research fit is not numerically representable") from error


def _fit_prior(
    history: tuple[Evidence, ...], spec: ModelSpec
) -> tuple[MixturePrior, tuple[float, ...]]:
    ordered = sorted(history, key=lambda row: row.experiment_id)
    y = np.array([row.estimate for row in ordered])
    se = np.array([row.standard_error for row in ordered])
    tau = np.array(spec.tau_grid)
    variance = se[:, None] ** 2 + tau[None, :] ** 2
    precision = 1 / variance
    mean_variance = 1 / (1 / spec.mean_scale**2 + precision.sum(axis=0))
    mean = mean_variance * (precision * y[:, None]).sum(axis=0)
    # V_tau = diag(se² + tau²) + mean_scale² 11ᵀ. Relative to the first
    # node, V_tau = V_0 + delta I, so q_tau-q_0 = -delta yᵀV_tau⁻¹V_0⁻¹y.
    # This removes the large common quadratic before adding prior log weights.
    delta = (tau - tau[0]) * (tau + tau[0])
    centered = y - y[0]
    shift = mean_variance * (precision * centered[:, None]).sum(axis=0)
    shift -= (mean_variance / spec.mean_scale**2) * y[0]
    solved = precision * (centered[:, None] - shift)
    quadratic_delta = -(delta[None, :] * solved * solved[:, :1]).sum(axis=0)
    determinant_delta = np.log1p(delta[None, :] / variance[:, :1]).sum(axis=0)
    determinant_delta += np.log(mean_variance[0]) - np.log(mean_variance)
    likelihood = -0.5 * (determinant_delta + quadratic_delta)
    likelihood -= likelihood.max()
    log_mass = np.log(spec.tau_weights) + likelihood
    if not np.isfinite(log_mass).all():
        raise ValueError("hyperposterior is not numerically representable on this scale")
    log_mass -= log_mass.max()
    mass = np.exp(log_mass - logsumexp(log_mass))
    # Underflowed components carry no representable mass; MixturePrior requires
    # strictly positive weights. Keep the full tau posterior for diagnostics.
    active = mass > 0
    prior = MixturePrior(
        weights=tuple(mass[active] / mass[active].sum()),
        means=tuple(mean[active]),
        sigmas=tuple(np.sqrt(tau[active] ** 2 + mean_variance[active])),
    )
    return prior, tuple(float(value) for value in mass)


def fixed_priors() -> dict[str, MixturePrior]:
    """Fixed shapes illustrated in the existing priors-and-decisions guide."""
    return {
        "fixed_normal": MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(0.05,)),
        "fixed_student_t": StudentTPrior(nu=4, scale=0.1).components(),
    }


def _score_prior(prior: MixturePrior, target: Evidence) -> dict:
    """Predict the noisy estimate, then measure the separately updated posterior."""
    weights = np.asarray(prior.weights)
    means = np.asarray(prior.means)
    sigmas = np.asarray(prior.sigmas)
    predictive_sigmas = np.hypot(sigmas, target.standard_error)
    predictive = MixturePosterior(weights, means, predictive_sigmas)
    posterior = mixture_posterior(target.estimate, target.standard_error, prior)
    if not all(
        np.isfinite(array).all()
        for array in (
            posterior.weights,
            posterior.means,
            posterior.sigmas,
        )
    ):
        raise ValueError("posterior must be numerically representable")
    posterior_mean = float(posterior.weights @ posterior.means)
    lower, upper = posterior.quantile(0.025), posterior.isf(0.025)
    pred_lower, pred_upper = predictive.quantile(0.025), predictive.isf(0.025)
    pit = predictive.cdf(target.estimate)
    conflict = min(1.0, 2 * min(pit, predictive.survival(target.estimate)))
    truth = target.true_effect
    posterior_error = None if truth is None else (posterior_mean - truth) ** 2
    sampling_error = None if truth is None else (target.estimate - truth) ** 2
    log_score = float(
        logsumexp(np.log(weights) + norm.logpdf(target.estimate, means, predictive_sigmas))
    )
    if not math.isfinite(log_score):
        raise ValueError("predictive score must be numerically representable")
    return {
        "predictive_log_score": log_score,
        "predictive_pit": pit,
        "predictive_interval": [pred_lower, pred_upper],
        "predictive_covered": pred_lower <= target.estimate <= pred_upper,
        "prior_conflict_tail": conflict,
        "posterior_mean": posterior_mean,
        "posterior_interval": [lower, upper],
        "posterior_displacement": posterior_mean - target.estimate,
        "posterior_squared_error": posterior_error,
        "sampling_squared_error": sampling_error,
        "sampling_covered": None
        if truth is None
        else bool(abs(target.estimate - truth) <= norm.isf(0.025) * target.standard_error),
        "excess_error_vs_sampling": None
        if truth is None
        else ((posterior_mean - truth) ** 2 - (target.estimate - truth) ** 2),
        "posterior_covered": None if truth is None else lower <= truth <= upper,
        "truth_unavailable_reason": "latent_effect_unknown" if truth is None else None,
    }


def score_prior(prior: MixturePrior, target: Evidence) -> dict:
    """Refuse unrepresentable research inputs without leaking numerical crashes."""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return _score_prior(prior, target)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("research score is not numerically representable") from error


def summarize(rows: list[dict]) -> dict:
    """Compare methods on exactly the same eligible chronological targets."""
    paired = [row for row in rows if row["status"] == "evaluated"]
    summaries = {}
    for name in (*fixed_priors(), "learned", "learned_wide"):
        scores = [row["models"][name]["scores"] for row in paired]
        summary: dict = {"targets": len(scores)}
        for key in (
            "predictive_log_score",
            "predictive_covered",
            "posterior_displacement",
            "posterior_squared_error",
            "sampling_squared_error",
            "sampling_covered",
            "excess_error_vs_sampling",
            "posterior_covered",
        ):
            available = [score[key] for score in scores if score[key] is not None]
            summary[key] = float(np.mean(available)) if available else None
        summary["absolute_posterior_displacement"] = (
            float(np.mean([abs(score["posterior_displacement"]) for score in scores]))
            if scores
            else None
        )
        summary["conflict_rate_at_0.01"] = (
            float(np.mean([score["prior_conflict_tail"] < 0.01 for score in scores]))
            if scores
            else None
        )
        summary["pit_histogram"] = np.histogram(
            [score["predictive_pit"] for score in scores], bins=np.linspace(0, 1, 11)
        )[0].tolist()
        summary["truth_targets"] = sum(
            score["posterior_squared_error"] is not None for score in scores
        )
        summaries[name] = summary
    comparisons = {}
    for baseline in fixed_priors():
        score_deltas = [
            row["models"]["learned"]["scores"]["predictive_log_score"]
            - row["models"][baseline]["scores"]["predictive_log_score"]
            for row in paired
        ]
        error_deltas = [
            row["models"]["learned"]["scores"]["posterior_squared_error"]
            - row["models"][baseline]["scores"]["posterior_squared_error"]
            for row in paired
            if row["models"]["learned"]["scores"]["posterior_squared_error"] is not None
        ]
        comparisons[baseline] = {
            "mean_log_score_gain": float(np.mean(score_deltas)) if score_deltas else None,
            "mean_excess_squared_error": float(np.mean(error_deltas)) if error_deltas else None,
            "negative_transfer_fraction": float(np.mean(np.array(error_deltas) > 0))
            if error_deltas
            else None,
        }
    return {"paired_targets": len(paired), "models": summaries, "learned_vs_fixed": comparisons}


def evaluate(corpus: Corpus, spec: ModelSpec = DEFAULT_MODEL) -> dict:
    """Expanding-window evaluation with identity-preserving insufficient-history rows."""
    rows = []
    wide = wider_model(spec)
    for target in sorted(corpus.evidence, key=lambda row: (row.started_at, row.experiment_id)):
        history = tuple(
            row
            for row in corpus.evidence
            if (
                row.available_at < target.started_at
                and row.comparable_class == target.comparable_class
            )
        )
        models: dict[str, dict] = {
            name: {
                "status": "evaluated",
                "reason": None,
                "prior": prior.model_dump(),
                "scores": score_prior(prior, target),
            }
            for name, prior in fixed_priors().items()
        }
        row = {
            "experiment_id": target.experiment_id,
            "class": target.comparable_class,
            "started_at": target.started_at.isoformat(),
            "training_ids": sorted(item.experiment_id for item in history),
            "history_count": len(history),
            "status": "evaluated" if len(history) >= spec.minimum_history else "unavailable",
            "reason": None
            if len(history) >= spec.minimum_history
            else "insufficient_comparable_history",
            "models": models,
        }
        if row["status"] == "evaluated":
            for name, model in (("learned", spec), ("learned_wide", wide)):
                prior, tau_mass = freeze_prior(history, target, model)
                models[name] = {
                    "status": "evaluated",
                    "reason": None,
                    "prior": prior.model_dump(),
                    "tau_posterior": tau_mass,
                    "upper_grid_mass": tau_mass[-1],
                    "scores": score_prior(prior, target),
                }
        else:
            fields = models["fixed_normal"]["scores"]
            for name in ("learned", "learned_wide"):
                scores = dict.fromkeys(fields)
                scores["truth_unavailable_reason"] = row["reason"]
                models[name] = {
                    "status": "unavailable",
                    "reason": row["reason"],
                    "prior": None,
                    "tau_posterior": None,
                    "upper_grid_mass": None,
                    "scores": scores,
                }
        rows.append(row)
    canonical = corpus.model_dump_json()
    return {
        "schema_version": 2,
        "recommendation": "defer",
        "recommendation_reason": (
            "synthetic evidence does not replace the required historical corpus"
            if corpus.synthetic
            else "requires domain review of chronological calibration, shift and selection sensitivity"
        ),
        "corpus_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "corpus": corpus.model_dump(mode="json"),
        "model": spec.model_dump(),
        "sensitivity_model": wide.model_dump(),
        "truth_metrics_scope": "synthetic latent effects only",
        "summary": summarize(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--model", type=Path, help="predeclared ModelSpec JSON for a real corpus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.demo and args.model:
        parser.error("--demo uses its fixed manifest; --model belongs with --corpus")
    if args.demo:
        from calibration.historical_prior_stress import run_stress

        report = run_stress()
    else:
        spec = (
            DEFAULT_MODEL
            if args.model is None
            else ModelSpec.model_validate_json(args.model.read_text())
        )
        report = evaluate(Corpus.model_validate_json(args.corpus.read_text()), spec)
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path(__file__).with_name("historical_prior_stress.py"),
            Path(__file__).parents[1] / "increment/estimation/priors.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Recommendation: defer. Research report: {args.output}")


if __name__ == "__main__":
    main()
