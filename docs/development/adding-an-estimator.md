# Extending an estimator or its evidence

This guide follows two completed changes: refining exact conversion-interval endpoints and adding experiment-health diagnostics. These are repository examples, not a universal extension protocol. For new methods or designs, follow the [Adding a method or design checklist](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md#adding-a-method-or-design); it does not block fixes to shipped behavior.

## First locate the owner and the boundary

For randomized lift, `increment/estimation/engine.py` owns `Method`, `estimate_lift` dispatch, and per-contrast routes. `increment/estimation/adjust.py` owns observational `estimate_ate` and the `ADJUSTMENTS` registry; built-ins register from `increment/estimation/_adjust/{iptw,dml,aipw}.py`. Keep these boundaries separate: don't add observational adjustments to randomized lift or warehouse queries to the estimator. Sources and reductions obtain evidence; estimators consume `ArmStats` or the per-unit data their construction requires.

`Method.name` labels results and identifies the requested method; it does not select an algorithm alone. Randomized lift uses `variance_reduction` and inference settings, while observational dispatch looks up its name in `ADJUSTMENTS`. Validation checks these separately. Do not infer a method from display text or add aliases.

`increment/_moment_plan.py` distinguishes counts, references, centered first moments, and centered second/cross moments. These cannot replace unit records or reconstruct membership and arbitrary distributions. Evidence is distinct too: reasons, guarantees, diagnostics, and source availability answer different questions from numeric moments. Preserve these distinctions through source, estimator, result, serialization, and presentation.

## Set up before running the focused commands

Install the test dependencies used below:

```bash
uv sync --group dev --extra demo --extra tables --extra dashboard
```

`make install` also synchronizes dependencies and installs repository hooks.
Check shared-hook ownership first, as described in CONTRIBUTING. Hooks are not
needed to run the focused tests.

## Walkthrough 1: refine exact-count binomial endpoints

**Historical change:** `0f272c4` (`Resolve exact risk-ratio endpoints to a relative log width (#40)`), compared with parent `0f272c4^`. This refined an existing exact-count estimator. The route already had exact successes and trial counts; the change improved endpoint search without adding evidence or changing the statistic.

### Contract and assumptions

The route computes an exact independent-binomial risk-ratio confidence set for eligible conversion contrasts. It uses exact successes and trial counts from both arms and canonicalizes Bernoulli arms. These counts differ from ordinary centered mean/ratio moments. A new sufficient statistic needs coordinated source, reduction, and wire changes, plus an independent construction and ingress evidence. This construction does not extend to clustered units, sequential monitoring, or unsupported adjustments. `increment/estimation/binomial_rr.py` owns the construction and its eligibility/refusals.

The change made endpoint search relative in log-risk-ratio space instead of using a fixed absolute step. That gives finer resolution near zero. Preserve outward containment and finite or unbounded endpoint representation; never hide the resolution limit by rounding inward.

### Trace the implementation and independent check

1. Trace `increment/estimation/engine.py::_contrast_counts` from `estimate_lift` through `_lift_for_method` to `_infer_binomial_lift_result`. It builds the contrast counts and returns the result or typed failure. `binomial_rr.confidence_interval` owns the construction and its endpoint qualifications.
2. Read `tests/estimation/test_binomial_rr.py::TestEndpointContainment::test_endpoints_enclose_the_finer_reference` and `tests/estimation/_binomial_endpoint_reference.py`. The reference uses a separate, finer doubling-and-bisection search. Decimal polynomial and enumeration checks independently test binomial tails; production endpoint bits are not an oracle.
3. Check fixed-horizon conversion cases in `tests/parity_harness/cases.py`, such as `_fixed_horizon_case` and `mean_ratio_conversion_cuped_fixed_horizon`. Unsupported ingress must retain its explicit refusal, not invent counts from rounded moments. See the six entry-point dispositions in the [capability table](../reference/capabilities-by-entry-point.md).
4. `tests/estimation/test_engine.py::TestBinomialDirectionalAlphaConvention::test_a_produced_row_round_trips_from_a_dump_without_defaults` checks that results and metadata survive serialization and reinversion. Check it alongside parity and result assertions, not only the point estimate.

Focused commands for this slice:

```bash
make test TESTS=tests/estimation/test_binomial_rr.py::TestEndpointContainment::test_endpoints_enclose_the_finer_reference
make test TESTS=tests/estimation/test_engine.py
make test TESTS=tests/test_parity_harness.py
```

`test_binomial_rr.py` checks the numerical route against an independent reference; the named `test_engine.py` case checks result round-tripping; the parity harness checks source support. Use `make test-affected BASE=0f272c4^` when changes span more owners. Do not start a calibration campaign for a documentation edit; calibration obligations for a numerical change remain those recorded by the owners.

## Walkthrough 2: add experiment-health evidence

**Historical changes:** `baf3d7a` introduced `Analysis.health_check()` and its composed report; `b32338a` added source and snapshot evidence, compared with `b32338a^`. The feature combines assignment-integrity, trigger-rate, and retained source-maturity evidence. It does not add an estimator or determine causal validity.

### Contract, assumptions and source requirements

`Analysis.health_check()` returns a frozen `ExperimentHealth` report (`increment/estimation/health.py`) with ordered `HealthCheck` entries. Availability is not a conclusion: missing trigger membership or feed-completeness evidence remains unavailable with its exact code and context, not a failure or pass. Assigned allocation integrity and triggered-population balance are separate.

!!! warning "Balanced triggers do not establish causal validity"
    Balanced trigger rates do not establish treatment-independent membership,
    stationarity, randomization integrity or causal validity; report the
    assumption risk rather than converting the diagnostic into a guarantee.

The report is available from all public analysis factories, but trigger diagnostics need specific evidence:

- `Analysis.from_definitions` needs retained native trigger and source-snapshot evidence.
- `Analysis.from_unit_day_artifact` needs `trigger_population`, `assignment_counts`, and `trigger_measure_stats`; without them, trigger checks are unavailable.
- `Analysis.from_unit_summary`, `from_unit_panel`, `from_switchback_panel`, and `from_moments` do not retain trigger membership. They report the exact unavailable or not-applicable status; they do not infer trigger diagnostics.

`tests/capability_entry_point_catalog.py` and `tests/capability_consumers_auxiliary.py` record dispositions across six entry points. `increment/analysis.py` composes evidence; `increment/estimation/health.py` defines reports; `increment/dashboard/_data.py` and `increment/dashboard/_html.py` render them.

This evidence is not a moment: aggregates cannot restore unit membership or feed watermarks. Nor is it a substitute for unit records; source identity and timing evidence must remain attached to the check that uses them.

### Trace dispatch, independent checks and consumers

1. Trace `Analysis.health_check()` to `_collect_health()` in `increment/analysis.py`. It pins one source snapshot, combines existing diagnostics, retains source identity, and builds the report. Dashboards use this method rather than duplicate its policy.
2. `tests/test_experiment_health.py::test_balanced_trigger_rates_do_not_clear_assumption_risk` calculates each arm's expected trigger share directly from the fixture's raw `enrolled` and `saw_surface` user IDs, then compares that independent count to the report. The same test confirms that balanced rates leave assumption-risk fields marked as assumed. `test_health_missing_certificates_are_unknown` checks the refusal and missing watermark.
3. Sentinel tests cover unavailable and not-applicable paths: `test_untriggered_summary_has_explicit_unavailable_triggered_rows`, `test_health_unit_panel_reports_only_supported_population`, `test_health_switchback_does_not_substitute_unit_arm_srm`, `test_health_moments_replay_does_not_reconstruct_trigger_population`, and `test_health_artifact_without_trigger_extensions_is_unavailable`. `test_health_artifact_trigger_extensions_are_optional_but_explicit` compares retained native/artifact evidence.
4. Preserve identity and provenance through frozen models, JSON/pickle/deepcopy round trips, and dashboard projection. `test_health_json_replay_preserves_checks_and_risks` and `test_health_models_round_trip_pickle_deepcopy_and_nested_immutability` check restoration; dashboard tests check rendering. Serialization restores a report; it cannot make an unavailable check available.
5. Check the capability row and [entry-point reference](../reference/capabilities-by-entry-point.md) before changing support. Pair each newly supported ingress with evidence tests; a constructor alone is not parity evidence.

Focused commands for this slice:

```bash
make test TESTS=tests/test_experiment_health.py
make test TESTS=tests/test_dashboard.py
make test TESTS=tests/test_dashboard_overview.py
```

## Extraction decision

No extraction is proposed. The apparent overlap does not demonstrate a shared policy that can move without changing behavior: moment planning/reduction, estimator dispatch, multiplicity, and result presentation own separate contracts. Moving modules or adding abstraction for navigation or file size alone would be cosmetic.

For this historical walkthrough, inspect the relevant implementation and sentinel tests, then run the focused commands above. For documentation changes, run the strict docs session and reference checker after adding this page to navigation. New methods must follow the contributor checklist; these examples do not replace it.
