"""Pure metadata for scalar capability outcomes consumed by their real tests.

Each expectation is read at the assertion that measures it: the expected kind
decides whether the request must complete or refuse, and a refusal's exact
code comes from here rather than from a literal beside the request.  Claims
whose evidence is a registered row case live in ``PARITY_DECLARATIONS`` and
are consumed by that case through the merged declaration registry.
"""

from __future__ import annotations

from collections.abc import Callable

from tests.capability_contracts import (
    ConsumerContract,
    ConsumerExpectation,
    ConsumerOutcome,
    OutcomeKind,
    ParityDeclaration,
)

FRAME_INGRESSES = ("from_unit_summary", "from_unit_panel", "from_switchback_panel")
MATCHED_ARM_INGRESSES = (
    "from_unit_summary",
    "from_unit_panel",
    "from_definitions",
    "from_unit_day_artifact",
    "from_moments",
)


def _e(
    id: str,
    operation: str,
    ingress: str,
    condition: str,
    kind: OutcomeKind,
    *,
    codes: tuple[str, ...] = (),
    reason: str = "",
    stage: str = "read",
    evidence: tuple[str, ...],
) -> ConsumerExpectation:
    return ConsumerExpectation(
        id=id,
        operation=operation,
        ingress=ingress,
        condition=condition,
        expected=ConsumerOutcome(kind, codes, reason, stage),
        evidence=evidence,
    )


_RATIO_ADVISORY_TEST = "tests/estimation/test_ratio_denominator_precision.py::test_advisory_reaches_every_path_that_estimates_a_ratio_metric"
RATIO_DENOMINATOR_ADVISORY = ConsumerContract(
    "ratio-denominator-advisory-fixed-horizon",
    tuple(
        _e(
            f"{ingress}-advisory",
            "ratio lift with an imprecise denominator",
            ingress,
            "fixed horizon; same matched units; denominator relative SE exceeds advisory threshold",
            "supported",
            evidence=(_RATIO_ADVISORY_TEST,),
        )
        for ingress in MATCHED_ARM_INGRESSES
    ),
)

_PREDICTABLE = "tests/test_sequential_predictable_transforms.py"
FIXED_WINSOR_SEQUENTIAL = ConsumerContract(
    "fixed-threshold-winsor-sequential",
    (
        _e(
            "unit-summary-clipped-scalars",
            "mean with fixed-threshold winsorization under asymptotic_mean",
            "from_unit_summary",
            "upper_value=20; the retained scalar-mean state holds the clipped per-unit means",
            "supported",
            evidence=(
                f"{_PREDICTABLE}::test_fixed_winsorization_runs_and_retains_clipped_scalars",
            ),
        ),
        _e(
            "unit-summary-percentile-refused",
            "mean with percentile winsorization under asymptotic_mean",
            "from_unit_summary",
            "upper_percentile=0.9; the threshold would move with accumulated data",
            "refused",
            codes=("sequential.transform.unpredictable",),
            reason="a percentile threshold re-clips earlier observations as it moves",
            stage="construction",
            evidence=(f"{_PREDICTABLE}::test_percentile_winsorization_refuses_as_unpredictable",),
        ),
        _e(
            "shared-fixed-threshold-gate",
            "validate sequential transform",
            "shared sequential transform gate",
            "fixed lower/upper value thresholds are predictable",
            "supported",
            evidence=(
                f"{_PREDICTABLE}::test_fixed_threshold_passes_the_shared_gate_every_route_consults",
            ),
        ),
        _e(
            "shared-percentile-threshold-gate",
            "validate sequential transform",
            "shared sequential transform gate",
            "a percentile bound beside a fixed one is still data-dependent",
            "refused",
            codes=("sequential.transform.unpredictable",),
            reason="a percentile threshold re-clips earlier observations as it moves",
            stage="construction",
            evidence=(f"{_PREDICTABLE}::test_percentile_threshold_refused_at_the_shared_gate",),
        ),
    ),
)

_BINARY_PARITY = {
    "asymptotic_mean": "tests/test_binary_asymptotic_sequential.py::test_conversion_state_and_interval_agree_on_every_path",
    "always_valid": "tests/test_binary_always_valid_auto.py::test_binary_always_valid_state_and_interval_agree_on_every_path",
}
BOUNDED_CONVERSION_SEQUENTIAL = ConsumerContract(
    "bounded-conversion-sequential-state-and-interval",
    tuple(
        _e(
            f"{kind}-{ingress}-retained-state",
            "bounded conversion under sequential inference",
            ingress,
            f"{kind}; retain Bernoulli registration, model state and interval equal to unit-summary oracle",
            "supported",
            evidence=(evidence,),
        )
        for kind, evidence in _BINARY_PARITY.items()
        for ingress in MATCHED_ARM_INGRESSES
    ),
)

MEAN_CUPED_ASYMPTOTIC = ConsumerContract(
    "mean-cuped-asymptotic-sequential",
    (
        _e(
            "predeclared-cuped-summary-state",
            "mean CUPED under asymptotic_mean",
            "from_unit_summary",
            "predeclared coefficient and center; retained adjusted scalar narrows interval",
            "supported",
            evidence=(
                f"{_PREDICTABLE}::test_predeclared_cuped_narrows_the_asymptotic_interval_and_labels_the_row",
            ),
        ),
        _e(
            "in-experiment-cuped-summary-joint-moments",
            "mean CUPED under asymptotic_mean",
            "from_unit_summary",
            "in-experiment coefficient; the adjusted_mean law retains joint (Y, X) moments",
            "supported",
            evidence=(
                f"{_PREDICTABLE}::test_in_experiment_coefficient_is_retained_as_joint_moments_on_the_asymptotic_route",
            ),
        ),
        _e(
            "exact-route-cuped-refused",
            "validate sequential methods",
            "shared sequential method gate",
            "an in-experiment CUPED coefficient on the exact Bernoulli route",
            "refused",
            codes=("sequential.transform.unpredictable",),
            reason="the exact Bernoulli e-process admits no CUPED coefficient",
            stage="construction",
            evidence=(
                f"{_PREDICTABLE}::test_in_experiment_coefficient_still_refuses_on_the_exact_routes",
            ),
        ),
    ),
)

_IMPUTE = "tests/test_impute.py"
_FRAME_SPECS = "tests/test_frame_specs.py"
_MISSING_IMPUTE_REASON = "frame metric declarations do not support missing='impute'"
IMPUTE_POOLED_MEAN = ConsumerContract(
    "impute-pooled-mean-role-and-metric-policies",
    (
        *(
            _e(
                f"{ingress}-outcome-role-refused",
                "increment.impute.pooled_mean",
                ingress,
                f"frame prepared for {ingress} with declared roles; the outcome column is rejected before any requested column is filled",
                "refused",
                codes=("impute.pooled_mean_outcome",),
                reason="outcome columns must not be imputed as covariates",
                stage="construction",
                evidence=(
                    f"{_IMPUTE}::test_pooled_mean_refuses_a_declared_outcome_before_any_fill",
                ),
            )
            for ingress in FRAME_INGRESSES
        ),
        *(
            _e(
                f"{ingress}-undeclared-role-warning",
                "increment.impute.pooled_mean",
                ingress,
                f"frame prepared for {ingress} without declared roles; filled columns emit one coded warning with per-column counts",
                "supported",
                codes=("impute.pooled_mean_role_undeclared",),
                stage="construction",
                evidence=(
                    f"{_IMPUTE}::test_pooled_mean_warns_per_column_counts_for_undeclared_roles",
                ),
            )
            for ingress in FRAME_INGRESSES
        ),
        _e(
            "metric-missing-impute-refused",
            "frame MetricSpec(missing='impute')",
            "frame metric declaration",
            "MetricSpec refuses an unsupported impute missing policy",
            "refused",
            codes=("frame.metric.missing_impute",),
            reason=_MISSING_IMPUTE_REASON,
            stage="construction",
            evidence=(
                f"{_FRAME_SPECS}::test_metric_spec_impute_missing_is_a_named_refusal_with_a_route",
            ),
        ),
        *(
            _e(
                f"{ingress}-missing-impute-refused",
                "frame metric with missing='impute'",
                ingress,
                "frame source refuses imputation policy",
                "refused",
                codes=("frame.metric.missing_impute",),
                reason=_MISSING_IMPUTE_REASON,
                stage="construction",
                evidence=(evidence,),
            )
            for ingress, evidence in (
                (
                    "from_unit_summary",
                    "tests/test_frame_missing.py::test_missing_impute_is_refused_by_every_frame_constructor_with_one_code",
                ),
                (
                    "from_unit_panel",
                    "tests/test_frame_missing.py::test_missing_impute_is_refused_by_every_frame_constructor_with_one_code",
                ),
                (
                    "from_switchback_panel",
                    "tests/test_frame_switchback.py::test_switchback_missing_impute_refuses_with_the_shared_metric_code",
                ),
            )
        ),
        *(
            _e(
                f"{ingress}-missing-policy-not-declarable",
                "increment.impute.pooled_mean",
                ingress,
                "a warehouse experiment declares metrics over event facts, not a prepared frame",
                "not_comparable",
                reason="a warehouse Metric cannot express missing='impute'; the helper prepares dataframes only",
                stage="declaration",
                evidence=(
                    f"{_FRAME_SPECS}::test_warehouse_metrics_cannot_declare_a_missing_policy",
                ),
            )
            for ingress in ("from_definitions", "from_unit_day_artifact")
        ),
        _e(
            "from_moments-no-per-unit-rows",
            "increment.impute.pooled_mean",
            "from_moments",
            "a portable cube holds arm moments only",
            "not_comparable",
            reason="a moments cube carries no per-unit rows for the helper to repair",
            stage="read",
            evidence=(f"{_IMPUTE}::test_pooled_mean_has_no_per_unit_frame_on_a_moments_cube",),
        ),
    ),
)

_BASELINE = "tests/test_analysis_planning_baseline.py"
_BASELINE_PARITY = f"{_BASELINE}::TestPlanningBaselineParity::test_every_arm_constructor_matches_the_unit_summary_oracle"
_BASELINE_CLUSTER_PARITY = f"{_BASELINE}::TestPlanningBaselineParity::test_clustered_constructors_match_the_unit_summary_oracle"
_BASELINE_SWITCHBACK = f"{_BASELINE}::TestPlanningBaselineOnSwitchback::test_delegates_to_the_switchback_source_and_feeds_its_planner"
_BASELINE_TRIGGERED = f"{_BASELINE}::TestPlanningBaselineOnWarehouseRoutes::test_declared_trigger_reads_the_triggered_population"
_BASELINE_ASSIGNED = f"{_BASELINE}::TestPlanningBaselineOnDataframeRoutes::test_frame_routes_plan_the_assigned_population_without_a_trigger"
_NO_TRIGGER_DECLARATION = "the constructor accepts no Experiment declaration, so no trigger resolves and the assigned population is planned with trigger_rate=1.0"
PLANNING_BASELINE = ConsumerContract(
    "planning-baseline-mean-quantile-cluster-trigger",
    (
        _e(
            "mean-summary",
            "planning_baseline(mean)",
            "from_unit_summary",
            "unclustered mean baseline is the oracle",
            "supported",
            evidence=(_BASELINE_PARITY,),
        ),
        *(
            _e(
                f"mean-{short}",
                "planning_baseline(mean)",
                ingress,
                "unclustered mean baseline matches the unit-summary oracle field for field",
                "supported",
                evidence=(_BASELINE_PARITY,),
            )
            for short, ingress in (
                ("panel", "from_unit_panel"),
                ("definitions", "from_definitions"),
                ("artifact", "from_unit_day_artifact"),
                ("moments", "from_moments"),
            )
        ),
        _e(
            "mean-switchback-planner",
            "planning_baseline(mean)",
            "from_switchback_panel",
            "the pilot-fitted SwitchbackBaseline feeds the switchback planner",
            "supported",
            evidence=(_BASELINE_SWITCHBACK,),
        ),
        _e(
            "quantile-summary",
            "planning_baseline(quantile)",
            "from_unit_summary",
            "quantile baseline from raw unit values",
            "supported",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineOnDataframeRoutes::test_quantile_metric_from_control_arm_values",
            ),
        ),
        _e(
            "quantile-panel",
            "planning_baseline(quantile)",
            "from_unit_panel",
            "unwindowed quantile; raw per-unit panel values remain available",
            "supported",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineOnDataframeRoutes::test_quantile_metric_from_unit_panel_values",
            ),
        ),
        _e(
            "quantile-triggered-definitions",
            "planning_baseline(quantile)",
            "from_definitions",
            "declared trigger; the quantile reads triggered control values",
            "supported",
            evidence=(_BASELINE_TRIGGERED,),
        ),
        _e(
            "quantile-triggered-artifact",
            "planning_baseline(quantile)",
            "from_unit_day_artifact",
            "declared trigger with metric-scoped trigger evidence; the artifact quantile matches definitions",
            "supported",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineOnWarehouseRoutes::test_triggered_cuped_planning_uses_assignment_anchored_preperiod",
            ),
        ),
        _e(
            "quantile-switchback-refused",
            "planning_baseline(quantile)",
            "from_switchback_panel",
            "a quantile metric is refused at the switchback metric-type gate before any frame is read",
            "refused",
            codes=("source.frame.switchback.metric",),
            reason="switchback metrics are mean or conversion only",
            stage="construction",
            evidence=(_BASELINE_SWITCHBACK,),
        ),
        _e(
            "quantile-moments-refusal",
            "planning_baseline(quantile)",
            "from_moments",
            "a quantile declared over an exported scalar cube",
            "refused",
            codes=("analysis.planning_baseline.quantile_source_unavailable",),
            reason="moments do not retain per-unit values required by quantile planning",
            stage="read",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineOnMoments::test_from_moments_refuses_a_quantile_metric_by_source",
            ),
        ),
        _e(
            "cluster-summary",
            "planning_baseline(clustered mean)",
            "from_unit_summary",
            "declared cluster column; per-unit mean/var with ICC on the metric's per-unit score",
            "supported",
            evidence=(_BASELINE_CLUSTER_PARITY,),
        ),
        _e(
            "cluster-definitions",
            "planning_baseline(clustered mean)",
            "from_definitions",
            "declared experiment cluster; the plan agrees with the clustered unit-summary oracle",
            "supported",
            evidence=(_BASELINE_CLUSTER_PARITY,),
        ),
        _e(
            "cluster-artifact",
            "planning_baseline(clustered mean)",
            "from_unit_day_artifact",
            "published cluster identity; the plan agrees with the clustered unit-summary oracle",
            "supported",
            evidence=(_BASELINE_CLUSTER_PARITY,),
        ),
        _e(
            "cluster-panel-refusal",
            "planning_baseline(clustered mean)",
            "from_unit_panel",
            "cluster= on a per-day panel whose per-unit collapse cannot carry randomization grain",
            "refused",
            codes=("source.frame_panel.cluster_grain",),
            reason="panel grain cannot express clustered randomization",
            stage="construction",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineParity::test_clustered_planning_panel_and_moments_sources_refuse",
            ),
        ),
        _e(
            "cluster-moments-refusal",
            "planning_baseline(clustered mean)",
            "from_moments",
            "exporting a clustered unit summary to a portable cube",
            "refused",
            codes=("source.moments.cluster_grain",),
            reason="portable moments do not retain randomized cluster identity",
            stage="export",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineParity::test_clustered_planning_panel_and_moments_sources_refuse",
            ),
        ),
        _e(
            "cluster-switchback-absence",
            "planning_baseline(clustered mean)",
            "from_switchback_panel",
            "cluster= on the switchback constructor",
            "not_expressible",
            reason="from_switchback_panel has no cluster= parameter: a switchback contrast has no arm-cluster shape",
            stage="construction",
            evidence=(_BASELINE_SWITCHBACK,),
        ),
        _e(
            "trigger-definitions-complete",
            "planning_baseline(triggered mean)",
            "from_definitions",
            "declared trigger reads actual triggered population",
            "supported",
            evidence=(_BASELINE_TRIGGERED,),
        ),
        _e(
            "trigger-artifact-complete",
            "planning_baseline(triggered mean)",
            "from_unit_day_artifact",
            "artifact trigger evidence agrees with definitions",
            "supported",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineOnWarehouseRoutes::test_artifact_with_trigger_evidence_matches_definitions",
            ),
        ),
        _e(
            "trigger-artifact-missing-evidence",
            "planning_baseline(triggered mean)",
            "from_unit_day_artifact",
            "declared trigger but artifact omits required evidence",
            "refused",
            codes=("analysis.planning_baseline.trigger_evidence_unavailable",),
            reason="artifact lacks declared trigger evidence",
            stage="read",
            evidence=(
                f"{_BASELINE}::TestPlanningBaselineOnWarehouseRoutes::test_artifact_without_trigger_evidence_refuses_by_name",
            ),
        ),
        *(
            _e(
                f"trigger-{short}-assigned-population",
                "planning_baseline(triggered mean)",
                ingress,
                "no trigger can be declared; the assigned population is planned",
                "source_limited",
                reason=_NO_TRIGGER_DECLARATION,
                evidence=(evidence,),
            )
            for short, ingress, evidence in (
                ("summary", "from_unit_summary", _BASELINE_ASSIGNED),
                ("panel", "from_unit_panel", _BASELINE_ASSIGNED),
                (
                    "moments",
                    "from_moments",
                    f"{_BASELINE}::TestPlanningBaselineOnMoments::test_from_moments_keeps_the_default_trigger_rate",
                ),
            )
        ),
        _e(
            "trigger-switchback-no-trigger",
            "planning_baseline(triggered mean)",
            "from_switchback_panel",
            "no trigger can be declared; the pilot contrast carries no triggered population",
            "source_limited",
            reason="a switchback contrast has no assignment population to trigger into; its SwitchbackBaseline carries no trigger rate",
            evidence=(_BASELINE_SWITCHBACK,),
        ),
    ),
)

_ASOF = "tests/test_analysis_asof.py"
UNBOUNDED_RETENTION_SEQUENTIAL = ConsumerContract(
    "unbounded-retention-sequential",
    (
        _e(
            "summary-no-dates-refused",
            "sequential retention with threshold_days=int",
            "from_unit_summary",
            "summary has no exposure dates to resolve the unbounded retention band",
            "refused",
            codes=("source.frame.constructor",),
            reason="retention needs an exposure date; a one-row-per-unit summary has none",
            stage="construction",
            evidence=(
                f"{_FRAME_SPECS}::test_summary_constructor_refuses_windowed_and_retention_specs",
            ),
        ),
        _e(
            "definitions-unbounded-observation-refused",
            "capture_sequential(retention, threshold_days=int)",
            "from_definitions",
            "the unbounded observation band has no finalized window",
            "refused",
            codes=("sequential.route.unsupported",),
            reason="sequential capture requires every observation window to have a bounded end",
            stage="read",
            evidence=(
                f"{_ASOF}::test_run_asof_lift_unbounded_retention_refuses_sequential_capture",
            ),
        ),
        _e(
            "moments-no-checkpoint-to-replay",
            "capture_sequential(retention, threshold_days=int)",
            "from_moments",
            "cube exported from the fixed-horizon definitions readout; the registered route captured no checkpoint for the unbounded band",
            "refused",
            codes=("sequential.continuation.legacy",),
            reason="portable moments only replay an upstream checkpoint, and none can be captured for an unbounded retention band",
            stage="read",
            evidence=(f"{_ASOF}::test_unbounded_retention_cube_has_no_checkpoint_to_replay",),
        ),
        _e(
            "switchback-retention-refused",
            "sequential retention with threshold_days=int",
            "from_switchback_panel",
            "switchback source refuses retention metric before sequential monitoring",
            "refused",
            codes=("source.frame.switchback.metric",),
            reason="switchback source only constructs its supported mean/conversion metrics",
            stage="construction",
            evidence=(
                "tests/test_frame_switchback.py::test_switchback_unbounded_retention_refuses_before_sequential_monitoring",
            ),
        ),
    ),
)

MIXED_CUPED_FIXED_HORIZON = ConsumerContract(
    "mixed-cuped-fixed-horizon",
    (
        _e(
            "from_unit_summary_supported",
            "Analysis.run() mixed-covariate mean/ratio",
            "from_unit_summary",
            "same per-unit mean, ratio, categorical and missing-history data",
            "supported",
            evidence=(
                "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_rows_match_all_supported_ingresses",
            ),
        ),
        _e(
            "from_unit_panel_supported",
            "Analysis.run() mixed-covariate mean/ratio",
            "from_unit_panel",
            "same per-unit mean, ratio, categorical and missing-history data",
            "supported",
            evidence=(
                "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_rows_match_all_supported_ingresses",
            ),
        ),
        _e(
            "from_definitions_supported",
            "Analysis.run() mixed-covariate mean/ratio",
            "from_definitions",
            "same per-unit mean, ratio, categorical and missing-history data",
            "supported",
            evidence=(
                "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_rows_match_all_supported_ingresses",
            ),
        ),
        _e(
            "from_unit_day_artifact_supported",
            "Analysis.run() mixed-covariate mean/ratio",
            "from_unit_day_artifact",
            "published mixed-covariate extensions are adopted with the same per-unit outcomes",
            "supported",
            evidence=(
                "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_rows_match_all_supported_ingresses",
            ),
        ),
        _e(
            "from_moments_source_limited",
            "Analysis.run() mixed-covariate mean",
            "from_moments",
            "aggregate moments contain no per-unit covariates",
            "source_limited",
            reason="the portable cube cannot supply per-unit covariates; use a unit source",
            stage="construction",
            evidence=(
                "tests/test_cuped_covariate_sources.py::test_from_moments_refuses_cuped_linear_with_unit_source_route",
            ),
        ),
        _e(
            "switchback_source_limited",
            "Analysis.run() mixed-covariate mean",
            "from_switchback_panel",
            "switchback contrasts use a different design and construction",
            "source_limited",
            reason="use an independent-unit randomized ingress",
            stage="construction",
            evidence=(
                "tests/test_cuped_covariate_sources.py::test_switchback_refuses_cuped_linear_with_independent_unit_route",
            ),
        ),
    ),
)


CONSUMER_CONTRACTS = (
    RATIO_DENOMINATOR_ADVISORY,
    FIXED_WINSOR_SEQUENTIAL,
    BOUNDED_CONVERSION_SEQUENTIAL,
    MEAN_CUPED_ASYMPTOTIC,
    IMPUTE_POOLED_MEAN,
    PLANNING_BASELINE,
    UNBOUNDED_RETENTION_SEQUENTIAL,
    MIXED_CUPED_FIXED_HORIZON,
)
_SWITCHBACK_NO_SCHEDULE = (
    "SOURCE: a switchback contrast has no sequential construction, and from_switchback_panel "
    "needs a switchback-shaped frame (unit/cycle/period/step columns) this dataset does not carry."
)
PARITY_DECLARATIONS = {
    "scalar_sequential_fixed_threshold_winsor": ParityDeclaration(
        attempted=(
            "from_definitions",
            "from_unit_day_artifact",
            "from_unit_summary",
            "from_unit_panel",
            "from_moments",
        ),
        waive=(("from_switchback_panel", _SWITCHBACK_NO_SCHEDULE),),
    ),
    "scalar_sequential_mean_cuped_asymptotic": ParityDeclaration(
        attempted=(
            "from_definitions",
            "from_unit_day_artifact",
            "from_unit_summary",
            "from_unit_panel",
            "from_switchback_panel",
            "from_moments",
        ),
        waive=(
            (
                "from_unit_panel",
                "CONSTRUCTION: panel sequential capture needs a bounded window, and a windowed metric has no per-unit collapse to attach the CUPED covariate to.",
            ),
            (
                "from_switchback_panel",
                "SOURCE: the switchback plan gate refuses the CUPED decision override and the sequential inference before any frame is read; its contrasts are fixed-horizon.",
            ),
        ),
        refusals=(
            ("from_unit_panel", "frame.validation.from_unit_panel"),
            ("from_switchback_panel", "source.frame.switchback.plan"),
        ),
    ),
}


def expected(contract: ConsumerContract, outcome_id: str) -> ConsumerOutcome:
    """Return an expectation's executable outcome for use at its real assertion."""
    return contract.outcome(outcome_id).expected


def assert_code(actual: str, contract: ConsumerContract, outcome_id: str) -> None:
    """Compare a real consumer's observed refusal/warning code to its contract."""
    outcome = expected(contract, outcome_id)
    assert outcome.codes and actual == outcome.codes[0]


def observe[T](
    contract: ConsumerContract, outcome_id: str, request: Callable[[], T]
) -> T | Exception:
    """Run a real consumer request and hold it to its declared outcome.

    A refused outcome must raise a coded error carrying the declared code; a
    not-expressible one must fail without any code; every other kind must let
    the request complete so the caller can check what the source produced.
    Callers that declare a completing kind narrow with :func:`completed`;
    callers that declare a failing kind narrow with :func:`refused`.
    """
    outcome = expected(contract, outcome_id)
    if outcome.kind == "refused":
        try:
            request()
        except Exception as raised:
            code = getattr(raised, "code", None)
            assert code == outcome.codes[0], (contract.id, outcome_id, code)
            return raised
        raise AssertionError(f"{contract.id}:{outcome_id} completed instead of refusing")
    if outcome.kind == "not_expressible":
        try:
            request()
        except Exception as raised:
            assert not hasattr(raised, "code"), (contract.id, outcome_id, raised)
            return raised
        raise AssertionError(f"{contract.id}:{outcome_id} was expressed by the constructor")
    return request()


def completed[T](contract: ConsumerContract, outcome_id: str, request: Callable[[], T]) -> T:
    """``observe`` for an outcome declared to complete; returns the request's own value."""
    observed = observe(contract, outcome_id, request)
    if isinstance(observed, Exception):
        raise AssertionError(
            f"{contract.id}:{outcome_id} is declared {expected(contract, outcome_id).kind!r}, "
            "which does not fail"
        ) from observed
    return observed


def refused(
    contract: ConsumerContract, outcome_id: str, request: Callable[[], object]
) -> Exception:
    """``observe`` for an outcome declared refused or not expressible; returns the failure."""
    observed = observe(contract, outcome_id, request)
    if not isinstance(observed, Exception):
        raise AssertionError(
            f"{contract.id}:{outcome_id} is declared {expected(contract, outcome_id).kind!r}, "
            "which does not complete"
        )
    return observed
