# Extending an estimator or its evidence

This guide traces two completed changes in the repository: an exact-count endpoint refinement to an existing fixed-horizon estimator, and an added diagnostic in public analysis results. It is a map through existing contracts, not a universal extension protocol. The [Adding a method or design checklist](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md#adding-a-method-or-design) applies to new methods/designs; CONTRIBUTING explicitly says it does not hold up fixes to already-shipped behavior. This page links to that checklist rather than duplicating it.

## First locate the owner and the boundary

For fixed-horizon lift, `increment/estimation/engine.py` owns `Method`, the `estimate_lift` dispatch and the per-contrast routes. `increment/estimation/adjust.py` owns the observational `estimate_ate` orchestration and the `ADJUSTMENTS` registry; built-ins register from `increment/estimation/_adjust/{iptw,dml,aipw}.py`. Those are different boundaries: do not add an observational adjustment to randomized `estimate_lift`, or teach the core estimator to build a warehouse query. The source/reduction side owns how evidence is obtained; estimators consume `ArmStats` or the documented per-unit input appropriate to their construction.

`Method.name` is the display/identity label stamped on result rows and used to refer to a method in the request. It is not an algorithm selector by itself. For randomized lift, the configured `variance_reduction` and inference settings choose behavior; observational dispatch looks up the configured name in `ADJUSTMENTS`. Validation enforces those different meanings. Do not invent a new alias or infer a method from presentation text.

The moment vocabulary in `increment/_moment_plan.py` distinguishes counts, references, centered first moments and centered second/cross moments. These are not interchangeable with unit records: aggregates cannot reconstruct unit membership or arbitrary distributions. Nor is either interchangeable with *evidence*: a persisted reason, guarantee, diagnostic, or source-availability record answers a different question from a numeric moment. Preserve the distinction as data crosses source, estimator, result, serialization and presentation layers.

## Set up before running the focused commands

From a clean checkout, install the test dependencies used by the commands below:

```bash
uv sync --group dev --extra demo --extra tables --extra dashboard
```

If setting up the repository hooks as well, `make install` also synchronizes dependencies and installs hooks; inspect existing shared-hook ownership first, as described in CONTRIBUTING. Hook installation is not required just to run the focused tests.

## Walkthrough 1: refine exact-count binomial endpoints

**Historical change:** `0f272c4` (`Resolve exact risk-ratio endpoints to a relative log width (#40)`), compared with its parent `0f272c4^`. This is a specialized exact-count estimator refinement, not a template for adding a moment. The route already had the exact `successes` passthrough and trial counts it needs; this change refines the endpoint search without adding source evidence or changing the statistic.

### Contract and assumptions

The route computes an exact independent-binomial risk-ratio confidence set for eligible conversion contrasts. Its input is exact binary success counts and trial counts for both arms; it deliberately canonicalizes Bernoulli arms before applying that construction. These exact counts are not the ordinary centered moments for a mean/ratio estimator. A new sufficient statistic would instead require coordinated source/reduction and wire changes, plus its own independent construction and ingress evidence. A reported finite-sample construction is not a license to apply this model to clustered units, sequential monitoring, or an unsupported adjustment. The construction and its eligibility/refusals are owned by `increment/estimation/binomial_rr.py` and route selection in `engine._lift_for_method`; check the current compatibility table rather than assuming neighboring methods share its support.

`0f272c4` changes endpoint-search resolution: the stopping criterion is relative in log-risk-ratio space rather than a fixed absolute risk-ratio increment. This matters near zero, where a fixed-width bracket can be coarse relative to the endpoint. Keep outward containment and finite/unbounded endpoint representation intact; do not trade a disclosed resolution limit for an inward-rounded bound.

### Trace the implementation and independent check

1. Follow `increment/estimation/engine.py::_contrast_counts` from `estimate_lift` through `_lift_for_method` to `_infer_binomial_lift_result`; it constructs the count tuple for one requested contrast and returns the result or its typed failure. `binomial_rr.confidence_interval` owns the construction; endpoint search and its numerical qualifications live beside it.
2. Read `tests/estimation/test_binomial_rr.py::TestEndpointContainment::test_endpoints_enclose_the_finer_reference` and `tests/estimation/_binomial_endpoint_reference.py`. The reference uses a different doubling-and-bisection crossing search at a finer relative bracket width. Decimal polynomial/enumeration checks in the test module independently exercise the binomial tails; don't treat a comparison to the production endpoint bits as an oracle.
3. The ordinary parity matrix includes fixed-horizon conversion cases in `tests/parity_harness/cases.py` (for example `_fixed_horizon_case` and `mean_ratio_conversion_cuped_fixed_horizon`). Use it to check matched source paths; an ingress that cannot supply the exact counts must retain its explicit source/construction refusal, not manufacture counts from rounded moments. The six entry-point dispositions are in `docs/reference/capabilities-by-entry-point.md`.
4. For structured result preservation, inspect `tests/estimation/test_engine.py::TestBinomialDirectionalAlphaConvention::test_a_produced_row_round_trips_from_a_dump_without_defaults`; it checks that the produced exact confidence set and its persisted metadata survive serialization, including after reinversion. Pair it with the parity/result assertions rather than checking only a point estimate.

Focused commands for this slice:

```bash
make test TESTS=tests/estimation/test_binomial_rr.py::TestEndpointContainment::test_endpoints_enclose_the_finer_reference
make test TESTS=tests/estimation/test_engine.py
make test TESTS=tests/test_parity_harness.py
```

`test_binomial_rr.py` owns the numeric route and independent reference checks; the named `test_engine.py` case covers result round-trip preservation; the parity harness checks the public source boundary. `make test-affected BASE=0f272c4^` is the follow-up for a change spanning additional owners. Do not start a calibration campaign for a documentation edit; calibration obligations for a numerical change remain those recorded by the owners.

## Walkthrough 2: add experiment-health evidence

**Historical changes:** `baf3d7a` (`Add callable experiment health diagnostics`) introduced `Analysis.health_check()` and its composed report; `b32338a` (`Cover experiment health sources and snapshots`) added the ingress/snapshot evidence, compared with `b32338a^`. This added-evidence extension composes existing assignment-integrity, trigger-rate and retained source-maturity evidence. It does not add an estimator or claim a causal-validity verdict.

### Contract, assumptions and source requirements

`Analysis.health_check()` returns the frozen `ExperimentHealth` report (`increment/estimation/health.py`) with ordered `HealthCheck` entries. Availability is distinct from conclusion: absent trigger membership or certified feed-completeness evidence must remain unavailable with its exact code/context, not become a negative result or a pass. Assigned allocation integrity and triggered-population balance remain separate populations. Balanced trigger rates do not establish treatment-independent membership, stationarity, randomization integrity or causal validity; report the assumption risk rather than converting the diagnostic into a guarantee.

`health_check()` returns a report across the public analysis factories, but the *trigger diagnostics* have source-specific support:

- `Analysis.from_definitions` supports them only with retained native trigger evidence and the required source-snapshot evidence.
- `Analysis.from_unit_day_artifact` supports them when the artifact retains `trigger_population`, `assignment_counts`, and `trigger_measure_stats`; without those extensions, trigger checks are unavailable.
- `Analysis.from_unit_summary`, `from_unit_panel`, `from_switchback_panel`, and `from_moments` do not retain trigger-population evidence. Their reports must preserve the exact unavailable/not-applicable disposition; they do not support inferred trigger diagnostics.

The callable-experiment-health row in `tests/capability_entry_point_catalog.py` and its consumers in `tests/capability_consumers_auxiliary.py` record the six-entry-point dispositions. The facade composes evidence in `increment/analysis.py`; report models belong to `increment/estimation/health.py`; dashboard consumption is owned by `increment/dashboard/_data.py` and `increment/dashboard/_html.py`.

This evidence is not an additive moment: a moment slot can carry an aggregate quantity, but it cannot reconstruct missing unit membership or a feed watermark. It is not a unit-record substitute either; source identity and temporal evidence must remain attached to the check that consumes it.

### Trace dispatch, independent checks and consumers

1. Follow `Analysis.health_check()` to `_collect_health()` in `increment/analysis.py`. It pins one source snapshot, composes existing diagnostics, retains their source identity and constructs the typed report. Dashboard snapshots consume the same public method rather than reimplementing health policy.
2. `tests/test_experiment_health.py::test_balanced_trigger_rates_do_not_clear_assumption_risk` calculates each arm's expected trigger share directly from the fixture's raw `enrolled` and `saw_surface` user IDs, then compares that independent count to the report. The same test confirms that balanced rates leave assumption-risk fields marked as assumed. `test_health_missing_certificates_are_unknown` checks the source-refusal code and missing watermark rather than guessing a completeness value.
3. The source-specific sentinels are explicit: `test_untriggered_summary_has_explicit_unavailable_triggered_rows`, `test_health_unit_panel_reports_only_supported_population`, `test_health_switchback_does_not_substitute_unit_arm_srm`, `test_health_moments_replay_does_not_reconstruct_trigger_population`, and `test_health_artifact_without_trigger_extensions_is_unavailable` guard unavailable/not-applicable paths; `test_health_artifact_trigger_extensions_are_optional_but_explicit` compares retained native/artifact evidence.
4. Preserve result identity and provenance through frozen models, JSON/pickle/deepcopy round trips and dashboard projection. `test_health_json_replay_preserves_checks_and_risks` and `test_health_models_round_trip_pickle_deepcopy_and_nested_immutability` guard report restoration. `tests/test_dashboard.py` exercises health rendering; `tests/test_dashboard_overview.py` contains the additional dashboard overview consumer checks. A serialized report must restore what was rendered; it cannot upgrade an unavailable check.
5. Read the capability row and [entry-point reference](../reference/capabilities-by-entry-point.md) before changing a source disposition. Keep any newly supported ingress paired with evidence tests; constructor presence alone is not parity evidence.

Focused commands for this slice:

```bash
make test TESTS=tests/test_experiment_health.py
make test TESTS=tests/test_dashboard.py
make test TESTS=tests/test_dashboard_overview.py
```

## Extraction decision

No extraction is proposed. No single duplicated scope/estimator policy was demonstrated that could be centralized without changing behavior: moment planning/reduction, estimator dispatch, multiplicity routing and result presentation own distinct contracts. A module move or abstraction based only on navigation or file size would be cosmetic.

## Small navigation/verification path

Start at the relevant section above, inspect the named implementation and sentinel, then run its focused commands. For a documentation-only change, run the strict rendered docs session and the repository's doc-reference checker after adding this page to navigation. The contributor checklist remains the completion authority for new methods; the two examples here are historical maps, not exemptions from it.
