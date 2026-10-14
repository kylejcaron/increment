# Historical effect-prior evaluation

[Issue #70](https://github.com/kylejcaron/increment/issues/70) asks whether earlier,
comparable experiments improve priors for a new experiment. **Recommendation:
defer production implementation until a suitable historical corpus is available.**
No real corpus was supplied for this research. The offline evaluator and synthetic
stress cases establish a reproducible workflow, not an empirical answer or a
certified inference method. Issue #70 remains open for the corpus-dependent study.

## Run the research workflow

From a development checkout:

```bash
uv run python -m calibration.historical_priors --demo --output notes/historical-prior-demo.json
uv run python -m calibration.historical_priors --corpus corpus.json --output notes/historical-prior-report.json
```

The research modules live under `calibration/`, outside the installed package and
ordinary pytest collection. They do not register an `Analysis` method or change
sampling inference. Production prior-training, target-dependent hyperparameter
fitting, and automatic promotion of a learned prior are outside this issue.

Reports use schema version 2 and retain the corpus, its content hash, model specifications, source hashes,
training experiment identities, frozen priors and per-target scores. Every target
has a row with all four method cells, including those with insufficient comparable
history. Each cell has its own status and reason; unavailable learned cells keep
numeric nulls and a null frozen prior. Summary
comparisons use the same eligible holdouts for all four candidates.

## Historical evidence gate

The input is a JSON representation of `calibration.historical_priors.Corpus`.
`Corpus.model_json_schema()` describes the exact schema. Unknown fields, nonfinite
numbers, boolean numeric inputs, blank declarations and already-shrunk evidence
declarations are rejected. Identifiers and audit strings have surrounding
whitespace removed; substantive text is required. Ordinary JSON numbers, arrays
and timezone-aware timestamp strings remain supported. Record:

| Field | Requirement |
| --- | --- |
| `corpus_id`, `synthetic` | Identify the corpus; real evidence uses `synthetic: false`. |
| `effect_scale`, `scale_justification` | `log_relative_risk`, with a defensible common estimand and scale. |
| `class_justification` | Explain why intervention classes, metrics and populations are exchangeable; labels alone are insufficient. |
| `independence_justification` | Explain independence across experiments; one contrast per experiment, with no shared-control or repeated-result double counting. |
| `reporting_process`, `census_provenance` | `complete_census`, reconciled against an independently audited roster of all experiments, including unsuccessful, negative and null tests. |
| `experiment_roster` | Unique experiment IDs; exactly matches evidence rows. |
| `evidence` | Unshrunk sampling estimates and SEs, identities, chronology and provenance. |

Each evidence row contains `experiment_id`, `intervention_class`, `metric`,
`population`, timezone-aware `started_at` and `available_at`, `estimate`,
`standard_error`, `evidence_kind: normal_sampling`, and `provenance`. SEs must be
positive and below 0.5 on the log scale, matching the existing log-Normal
approximation's operational boundary. Their derivation and observation-law
validity still require review. A posterior mean and posterior SD are never raw
sampling evidence. `true_effect` is optional for simulations and prohibited in
real corpora; a high-precision follow-up estimate is not known truth.

For illustration, one evidence row looks like:

```json
{
  "experiment_id": "experiment-001",
  "intervention_class": "checkout-layout",
  "metric": "orders-per-assigned-user",
  "population": "US-assigned-users",
  "started_at": "2026-01-01T00:00:00Z",
  "available_at": "2026-01-20T00:00:00Z",
  "estimate": -0.012,
  "standard_error": 0.018,
  "evidence_kind": "normal_sampling",
  "provenance": "immutable sampling result; fixed-horizon analysis; audited source ID"
}
```

Schema checks cannot establish that an estimate is truly unshrunk, a supplied
roster is complete, experiments are independent, or class assignments are
scientifically defensible. These are corpus audit obligations. If missing results
or selection cannot be resolved, withhold learning and perform an explicit
reporting-mechanism sensitivity study before considering admission. The main
evaluator refuses selected/incomplete corpora; it does not repair publication bias.

## Model and chronological evaluation

For unshrunk log-ratio estimates `y_i` and sampling SEs `s_i`, use
`y_i | theta_i ~ Normal(theta_i, s_i)` and
`theta_i | mu, tau ~ Normal(mu, tau)`.
The mean hyperprior is `mu ~ Normal(0, 0.1)`. The heterogeneity hyperprior assigns
equal mass to `tau = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)`.
This is an explicitly **discrete model**, not a fitted point estimate or a claimed
continuous-hyperprior approximation.

Conditional on each `tau`, integrate `mu` analytically. Reweight `tau` by its
integrated likelihood. The prior for a new latent effect is a Normal mixture with
conditional posterior mean `m_tau` and variance `tau² + Var(mu | history, tau)`.
Both hyperparameter uncertainties therefore contribute to the frozen prior.
Compute relative integrated likelihoods before adding the tau log weights and
normalize in shifted coordinates, so a large common likelihood cannot erase
unequal prior weights or produce masses that sum to more than one.
The independent test oracle uses a multivariate-Normal density with covariance
`diag(s_i² + tau²) + mean_scale² 11ᵀ` and a Gaussian conditional distribution.

For each target, include only experiments in its exact declared
intervention/metric/population class whose `available_at` strictly precedes the
target's `started_at`. Overlapping experiments and results available exactly at
start are excluded. Freeze the mixture before scoring or updating on the target;
its observation changes the target posterior only. Earlier holdouts can become
history for later tests once available: this is an expanding-window study.

Compare the learned model with the existing guide's skeptical `Normal(0, 0.05)`
and `StudentTPrior(nu=4, scale=0.1)` shapes. These are declared baselines, not
claimed package defaults. The Student-t uses the existing 80-component expansion.
The learned sensitivity is derived from the supplied model: double the mean
hyperprior SD, append a heterogeneity node at 2.5 times the existing maximum,
and assign the new node probability `1 / (K + 1)`, retaining the original nodes'
relative weights in the remaining probability. For the default model this gives
`mu ~ Normal(0, 0.2)` and an equally weighted grid including `tau = 0.5`.
This broadens the hyperprior; fitted posterior intervals need not be wider. Record upper-grid posterior mass and inspect sensitivity
to the discrete support before interpreting a fit. This sensitivity changes
hyperpriors; the heavy-tailed synthetic regime separately stresses the Normal
effect-distribution assumption.

The default history gate is **20 independent comparable experiments**. This is a
conservative research screening choice, not an empirically established minimum.
At least two are structurally required for this evaluator; enough data to identify
heterogeneity depends on SEs, effect dispersion, representativeness and shift.
Insufficient-history rows retain fixed-baseline results and an explicit learned
unavailable reason, and are excluded from paired summaries. To predeclare another
research model or history threshold, supply `--model model.json` using the
`ModelSpec` schema. Do not choose a grid, class definition or threshold by looking
at its scored targets; reserve a subsequent untouched cohort after model selection.

The construction follows integration over hyperparameters for prediction of new
groups, as described in [Stan's predictive checks](https://mc-stan.org/docs/stan-users-guide/ppcs.html).
Chronological holdouts address the future-prediction objective described by
[Bürkner, Gabry and Vehtari](https://arxiv.org/abs/1902.06281); this evaluator refits
exactly and does not use their importance-sampling approximation.

## What is measured

| Measurement | Interpretation |
| --- | --- |
| Predictive log score | Density of the held-out **noisy sampling estimate**, including its SE; larger is better. |
| Predictive PIT and 95% interval coverage | Calibration of the noisy-estimate prediction, not coverage of the true latent effect. The report includes a ten-bin PIT histogram. |
| Posterior displacement | Signed and absolute movement from the sampling estimate, in log-relative-risk units; a shrinkage diagnostic. |
| Prior-conflict tail | Twice the smaller predictive tail at the observation. Uses survival functions for small upper tails; not a multiple-testing alarm or a frequentist significance verdict. |
| Posterior error and 95% latent-effect interval coverage | Computable only when a simulation supplies known truth; real reports keep numeric nulls with `latent_effect_unknown`. |
| Excess squared error and negative transfer | Learned posterior error minus fixed-baseline error, and the fraction of targets where it is worse; also compare to the raw sampling estimate. Truth required. |

Inspect per-class and chronological rows as well as aggregate scores. A positive
average predictive score does not rule out harmful shrinkage after a shift.
These descriptive summaries have no hypothesis test, confidence guarantee,
multiple-comparison adjustment or automatic pass threshold. Expanding-window fits
share training evidence; treating all score differences as independent would
understate uncertainty.

## Synthetic diagnostic results

The fixed manifest uses seed 70, 80 experiments per regime, SEs drawn uniformly
from 0.02 to 0.06, and independent Normal observation noise. Stationary latent
effects are `Normal(0, 0.02)`; the heavy-tailed regime uses `0.05 * t_3`; the shifted
regime adds 0.15 to latent effects starting at experiment 40. All tests, including
unsuccessful ones, are retained. The generator and exact manifest are in
`calibration/historical_prior_stress.py`.

The last 40 chronological targets give these illustrative results:

| Regime | Learned predictive 95% coverage | Learned posterior MSE | Fixed Student-t posterior MSE | Learned posterior worse than fixed Student-t |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 0.950 | 0.00036 | 0.00134 | 0.300 |
| Heavy-tailed | 0.950 | 0.00081 | 0.00101 | 0.375 |
| Shifted | 0.775 | 0.00223 | 0.00175 | 0.525 |

These are one seeded diagnostic realization, **not** a Monte Carlo coverage study.
In the shifted realization, the learned model has worse posterior MSE than the
fixed Student-t and worse MSE than the raw sampling estimate, despite a slightly
better average predictive log score than Student-t. The wider hyperprior does
not remove that failure. This demonstrates why prediction gains alone cannot
justify adoption.

A separate synthetic reporting sensitivity trains on the complete 79-experiment
history, only positive significant results, or only positive results, and scores
the same unselected final target. It retains every scenario, including an explicit
unavailable result when selection leaves fewer than 20 observations. This is a
controlled sensitivity to specific reporting mechanisms, not a justification
for applying a learned prior to an arbitrary selected real corpus.

## Decision and remaining acceptance

Withhold learning for incomplete/selected history without a suitable sensitivity
study, poorly justified classes, correlated experiments, estimated posteriors
masquerading as evidence, inaccurate sampling SEs, insufficient representative
history, material grid sensitivity, or evidence of shift/conflict/negative
transfer. A large count alone does not establish exchangeability.

Before recommending implement/reject, acquire and audit a real corpus, predeclare
its classes and evaluation protocol, run chronological and reporting-selection
sensitivity checks, quantify uncertainty, investigate adverse subgroups and shift,
and determine a data-supported minimum history. A negative finding is acceptable.
Known-truth inference claims additionally need scientific campaigns; real-corpus
predictive calibration alone cannot establish latent-effect coverage.

Production adoption would be a separate approved method change subject to
[method acceptance](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md#adding-a-method-or-design):
independent scientific validation, coded refusals, ingress parity and explicit
CUPED, ratio, clustering, sequential, winsorization, breakout and multiplicity
compatibility. This research evaluator establishes none of those production
capabilities. It supports only declared independent fixed-horizon Normal sampling
evidence on a justified log-relative-risk scale. Existing `Analysis` entry-point
capabilities and prior-free sampling inference are unchanged.
