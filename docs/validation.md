# External validation coverage

Increment has frozen comparisons with external tools, in-repository derivations,
and bounded calibration measurements. These check named methods at named inputs;
they do not establish that every supported analysis is calibrated.

A mathematical guarantee follows from a construction **and its assumptions**.
A measured check can find a failure or support the tested cases, but cannot make
an approximate method exact. Full sequential and mixed-covariate CUPED campaigns
remain unrun; independent real-world cases have not been recorded.

## How to read this page

| Your question | Start here |
|---|---|
| Does this method fit my data and decision? | [Assumptions by method](limitations.md#at-a-glance) |
| What was compared, at what tolerance? | [Evidence records](reference/validation-evidence.md#find-an-evidence-record) |
| Do equivalent input paths agree? | [Entry-point capabilities](reference/capabilities-by-entry-point.md#what-runs-where) |
| What is not established? | [Unvalidated regimes](#unvalidated-regimes) |

Selectors are checked for repository paths and directly declared functions or
methods. From a source checkout, `python scripts/check_doc_references.py --check`
reports identities still pending collection; adding `--collect` checks dynamic
identities by collection only, without running tests or campaigns. A resolved
selector is not evidence of a passing test. Recorded generator runs are dated in
[Reproduce](#reproduce); no later run is claimed.

## Evidence status

Software compatibility, statistical guarantees, and empirical evidence answer
different questions:

- **Compatibility:** the current alpha release is not 1.0.0; the
  [compatibility policy](api.md#compatibility-policy) applies starting with
  1.0.0. Individual methods have no maturity tier.
- **Guarantee:** `exact` means finite-sample under a stated model or dependence
  condition; `asymptotic` means valid in the limit under stated assumptions;
  `planning approximation` is for planning, not inference; `experimental`
  withholds decision evidence. No stated guarantee means none is documented.
- **Evidence:** what was checked in the [reference inventory](#reference-inventory),
  or `unknown` when the inventory records nothing. This differs from the
  `unknown` status in the [compatibility reference](guides/compatibility.md).

Records marked `exact` also list open deployed-numerics items by tracker id; an
open item is not a demonstrated coverage failure or a repair.

Parity is separate. Supported matched-arm entry points
(`from_definitions`, `from_unit_day_artifact`, `from_frame(shape="summary")`,
`from_frame(shape="panel")`, `from_moments`) agree at the documented tolerance.
`from_switchback_frame` has a different estimand: `supported` means it runs
and is exercised, not that it matches those constructors. `source_limited`
means the source cannot supply required input; `not_expressible` means
declaration fails by coded refusal or signature/schema validation. See the
[capability matrix](reference/capabilities-by-entry-point.md#what-runs-where)
and `helper:tests/parity_harness/matrix.py`; counts and unfinished trackers
are in `helper:tests/parity_harness/COVERAGE.md`.

| Status | Meaning in the parity matrix |
|---|---|
| `supported` | Runs and agrees with other running matched-arm entry points within `1e-9` relative tolerance; switchback is not compared with them. |
| `source_limited` | The entry point cannot supply the input and refuses with a code. |
| `construction_limited` | The estimator or readout is undefined for the combination; not `unsound` or `unfinished`. |
| `not_expressible` | The value cannot be declared; coded refusal or uncoded signature/schema failure. |
| `unfinished` | Supportable or unifiable, but not implemented; names its tracker. |
| `unsound` | Mathematically invalid; includes a derivation. |

A cell id such as `mean-run-cuped-utc-error` names metric, view, option, day
boundary, and missing-data policy. Guarantees follow derivations and
assumptions, not evidence; route labels are mechanisms, not guarantee classes.
Finite evidence grids are not universal, and completing a campaign changes
evidence, not the guarantee.

??? info "How a record states its evidence"
    Each record states the guarantee from its linked limitation and the evidence's
    cells, seeds and tolerance, or the evidence ID that supplies them. A record
    cites an existing source for each claim or says `unknown`; completing a
    campaign changes its evidence, not its guarantee.

    A `not_expressible` request fails during declaration, either with a coded
    refusal or through signature/schema validation (`TypeError` or Pydantic
    `ValidationError` without a refusal code).

## Ordinary-user journey contracts and current evidence

The [journey records](reference/validation-evidence.md#ordinary-user-journey-contracts-and-current-evidence)
separate each user-facing contract from the assertions that actually exercise it.
A basic mean test is not proof of plan disclosure; component checks are not proof
of automatic `run(experiment, reader)` behavior. Trigger-declared producer behavior
and a complete dashboard journey are not established by matched-arm parity.
Missing feed certificates remain unavailable, and failed cells keep their reasons.

### Numerical qualification reconciliation

Exact-binomial statistical validity is distinct from its numerical enclosure,
which is conditional on the deployed SciPy/Boost error model. The cited decimal
oracle measurements used arm64 with fused multiply-subtract; they do not identify
the currently locked SciPy 1.18.0 build, and x86 has not been executed. This is a
verification gap, not a reproduced bound failure. See the
[numerical record](reference/validation-evidence.md#numerical-qualification-reconciliation)
for completed changes and remaining investigations (`1tce`, `xeth`, `0p4v`, `4xwx`).

## Reference inventory

The [evidence index](reference/validation-evidence.md#find-an-evidence-record) leads
to all 19 evidence IDs. Each record keeps the method, guarantee, assumptions,
reference version, tolerance, source files, direct and transitive input-path
coverage, and regimes not covered.

### Reference classes

A frozen third-party tool output is independent evidence of a matched comparison.
An independent primitive checks only that primitive; a derivation or calibration
written in this repository is not independent review. See the exact
[reference classes](reference/validation-evidence.md#reference-classes).

### Frozen comparisons against a third-party tool or independent primitive

| Method | Evidence record |
|---|---|
| Welch mean difference | [EV-WELCH-T](reference/validation-evidence.md#ev-welch-t) |
| Clopper–Pearson single-arm interval | [EV-CP-BINOM-TEST](reference/validation-evidence.md#ev-cp-binom-test) |
| Benjamini–Hochberg selection | [EV-BH-P-ADJUST](reference/validation-evidence.md#ev-bh-p-adjust) |
| Meta-analysis: DL, HKSJ and HT | [EV-META-DL-HKSJ-HT](reference/validation-evidence.md#ev-meta-dl-hksj-ht) |
| Quantile order-statistic bracket | [EV-QUANTILE-QBINOM](reference/validation-evidence.md#ev-quantile-qbinom) |
| DML, AIPW and IPTW aggregation | [EV-ADJ-DML-AIPW-IPTW](reference/validation-evidence.md#ev-adj-dml-aipw-iptw) |

### In-repo derivations and calibration evidence

[Derivation and calibration records](reference/validation-evidence.md#in-repo-derivations-and-calibration-evidence)
cover arithmetic, confounding bounds, binomial sets, cluster covariance, CUPED,
meta-analysis and targeting. Their finite grids and regression bands are not
universal coverage guarantees.

## Methods without an independent reference

Use the [method evidence index](reference/validation-evidence.md#methods-without-an-independent-reference)
for the exact status, guarantee and input-path coverage of all 25 methods and
planning routes. No matching external comparison is recorded for the complete
binomial risk-ratio set, sequential selection, CUPED or the planning solvers.
Only the single-arm Clopper–Pearson primitive has a frozen binomial comparison.
`none_found` does not mean a method is wrong; `campaign_pending` does not mean its
registered thresholds were met.

## Mixed-covariate linear CUPED

`Method.cuped_multicovariate` uses fixed-horizon HC2 uncertainty with a
Bell–McCaffrey Student-t approximation. It is distinct from scalar CUPED; neither
inherits the other's evidence. **Neither the full nor the reduced preregistered
profile has run. Coverage is not established.** Runtime support, an example's
interval and passing the M1 support checks are not calibration evidence.

The [campaign record](reference/validation-evidence.md#mixed-covariate-linear-cuped-campaign)
contains the predeclared thresholds, unimplemented extensions and complete
prerequisite, cost-projection and execution commands. Thresholds are prospective
gates, not observed results.

## Unvalidated regimes

- **Sequential and cluster inference:** the full registered sequential manifest
  has not run; clustered and unit-cycle research matrices remain uncertified.
  Bounded checks cannot establish universal calibration
  ([scope](reference/validation-evidence.md#sequential-campaign-scope)).
- **Mixed-covariate CUPED:** neither registered campaign has run
  ([details](#mixed-covariate-linear-cuped)).
- **Winsorization:** inference remains experimental; 12 alternative stress-grid
  rows remain unresolved ([metric types](guides/metric-types.md#winsorization)).
- **Small-K meta-analysis:** the plug-in random-effects interval is
  anticonservative; HKSJ evidence is bounded by one design
  ([limitation](limitations.md#meta-analysis-can-be-anticonservative-at-small-k)).
- **Switchbacks:** carryover is assumed away, not tested, and the default t
  reference is approximate ([carryover](limitations.md#switchback-carryover-is-assumed-away-not-tested),
  [reference](limitations.md#the-switchback-t-reference-is-itself-an-approximation)).
  [Measured coverage with few units or blocks](reference/switchback-inference.md#coverage-with-few-units-or-blocks)
  separates returned intervals from all attempted experiments.
  [Ratio and forecast-adjustment checks](reference/validation-evidence.md#switchback-unit-cycle-t)
  verify their calculations separately; they do not inherit the unadjusted
  mean's measured coverage. [Pilot-profile planning checks](reference/validation-evidence.md#switchback-planning)
  verify simulated power under the chosen model, not the model's fit to a future experiment.
- **Two-stage offers, rejected method:** With 40 groups per offer level,
  one influential untreated outcome per group, and 99% versus 1% treated,
  a proposed 5% Normal test called a change on **22.15%** of assignments
  when the average effect was zero. That test is not used for decisions
  or sizing. Reproduce it with `python -m calibration.randomized_saturation`.
- **Adaptive logged policies:** refused. Historical coverage figures predate the
  gate and cannot be reproduced from the retained estimator
  ([measurements](reference/validation-evidence.md#adaptive-logger-measurements),
  [provenance](reference/validation-evidence.md#adaptive-logger-measurement-provenance)).
- **Very large binomial arms:** the certificate margin's earlier SciPy primitive-error
  assumption was falsified above about $2^{26}$ trials per arm; the investigation
  remains open (`1tce`). Public undercoverage has not been demonstrated
  ([numerical qualification](#numerical-qualification-reconciliation)).
- **Independent causal-feature calibration:** open (`ryyg`); no result is claimed.

## Reproduce

Ordinary comparison tests read frozen JSON: no R, network or external tool is
needed at test time. [Reproduction instructions](reference/validation-evidence.md#reproduce)
list the test commands; each frozen record lists its generator, tool versions,
fixture digest and one-time regeneration result (2026-10-03, macOS arm64).
Regeneration is deliberate: record a numeric discrepancy; never widen a tolerance
to make a regenerated fixture agree.

## Report a discrepancy

1. Open a [GitHub issue](https://github.com/kylejcaron/increment/issues/new) naming the increment version or commit, the entry point, and the reference tool and version.
2. Give the two numbers (increment's and the reference's) and the tolerance at which you consider them to disagree.
3. Attach per-arm aggregate moments or the reference tool's output; never identifiers, credentials, or raw private data.

A reported discrepancy is listed in the [Discrepancy log](#discrepancy-log) until it is fixed or
explained with evidence. It is not closed as dismissed.

## Case ledger

No independently sourced case has been recorded. Maintainer-, automated-agent-
and synthetic-only comparisons do not count. The
[case criteria](reference/validation-evidence.md#case-ledger) require permission,
a stated version and enough information to rerun the independent result.

## Discrepancy log

Two discrepancies were recorded on 2026-10-03: a BH selection boundary explained
by conservative rounding, and a regenerated DML fixture differing only in its
trailing newline. Neither is dismissed; the
[full records](reference/validation-evidence.md#discrepancy-log) retain the inputs,
observations and dispositions. No other discrepancy was found against the listed
references.
