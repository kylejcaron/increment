"""Pure metadata for auxiliary trigger, family-owned, and stored-state consumers.

Each expectation names the concrete consumer which must read it.  The small
assertion helper is deliberately used by those consumer tests, not a runner.
"""

from __future__ import annotations

from tests.capability_contracts import (
    ConsumerContract,
    ConsumerExpectation,
    ConsumerOutcome,
    OutcomeKind,
    ParityDeclaration,
)


def _expectation(
    id: str,
    operation: str,
    ingress: str,
    condition: str,
    kind: OutcomeKind,
    evidence: str,
    *,
    codes: tuple[str, ...] = (),
    reason: str = "",
    stage: str = "read",
) -> ConsumerExpectation:
    return ConsumerExpectation(
        id=id,
        operation=operation,
        ingress=ingress,
        condition=condition,
        expected=ConsumerOutcome(kind, codes, reason, stage),
        evidence=(
            f"tests/{evidence.split('::', 1)[0][6:].replace('.', '/')}.py::{evidence.split('::', 1)[1]}",
        ),
    )


_TRIGGER_OTHER = (
    "The frame/moments/switchback entry points carry no declared trigger membership for this read."
)
_TRIGGER_READS = (
    (
        "breakout",
        "run_breakout()/breakout_summaries()",
        "tests.test_analysis_trigger::test_run_breakout_returns_distinct_triggered_population_and_families",
    ),
    (
        "factor",
        "factor_summaries()",
        "tests.test_analysis_trigger::test_triggered_factor_group_and_allocation_preserve_population",
    ),
    (
        "dashboard",
        "dashboard_group_data(population=...)",
        "tests.test_analysis_trigger::test_triggered_factor_group_and_allocation_preserve_population",
    ),
    (
        "allocation",
        "allocation_history(population=...)",
        "tests.test_analysis_trigger::test_triggered_factor_group_and_allocation_preserve_population",
    ),
    (
        "sitewide",
        "sitewide(population=...)",
        "tests.test_analysis_trigger::test_sitewide_reports_assigned_impact_for_triggered_experiment",
    ),
    (
        "cate",
        "estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()",
        "tests.test_analysis_trigger::test_triggered_cate_uses_native_unit_population",
    ),
)

_TRIGGER_AUXILIARY_OUTCOMES: list[ConsumerExpectation] = []
for _key, _operation, _evidence in _TRIGGER_READS:
    if _key in {"breakout", "factor", "cate"}:
        _artifact_code = (
            "facade.analysis.operation" if _key == "factor" else "artifact.extension.missing"
        )
        _artifact_condition = (
            "randomized estimate_cate(interact=[]) with explicit trigger extensions"
            if _key == "cate"
            else "unit-day artifact has all read-specific extensions and explicit trigger evidence"
        )
        _TRIGGER_AUXILIARY_OUTCOMES.extend(
            (
                _expectation(
                    f"{_key}_definitions_triggered_population",
                    _operation,
                    "from_definitions",
                    "randomized definitions source includes per-unit trigger and covariate evidence"
                    if _key == "cate"
                    else "definitions source includes SourceSnapshotEvidence and per-unit trigger membership",
                    "supported",
                    _evidence,
                ),
                _expectation(
                    f"{_key}_artifact_complete_trigger_evidence",
                    "estimate_cate(interact=[])" if _key == "cate" else _operation,
                    "from_unit_day_artifact",
                    _artifact_condition,
                    "supported",
                    "tests.test_analysis_trigger::test_triggered_cate_randomized_artifact_without_covariates_preserves_population"
                    if _key == "cate"
                    else "tests.test_analysis_trigger::test_run_breakout_returns_distinct_triggered_population_and_families"
                    if _key == "factor"
                    else _evidence,
                ),
                _expectation(
                    f"{_key}_artifact_missing_extension",
                    _operation,
                    "from_unit_day_artifact",
                    "required extension kind is not published",
                    "refused",
                    "tests.test_analysis_facade::test_public_artifact_adoption_refuses_missing_extension"
                    if _key == "breakout"
                    else _evidence,
                    codes=(_artifact_code,),
                    reason="Artifact lacks the operation-specific extension.",
                    stage="read",
                ),
                _expectation(
                    f"{_key}_artifact_missing_evidence",
                    _operation,
                    "from_unit_day_artifact",
                    "extension producer cannot obtain operation-specific source evidence",
                    "refused",
                    "tests.test_unit_day_artifact_extensions::test_every_extension_rejects_missing_operation_and_extension",
                    codes=("artifact.evidence.unavailable",),
                    reason="The source cannot provide required evidence for the extension.",
                    stage="publication",
                ),
            )
        )
    elif _key == "dashboard":
        _TRIGGER_AUXILIARY_OUTCOMES.extend(
            (
                _expectation(
                    "dashboard_definitions_population",
                    _operation,
                    "from_definitions",
                    "assigned and triggered populations are requested",
                    "supported",
                    _evidence,
                ),
                _expectation(
                    "dashboard_artifact_refused",
                    _operation,
                    "from_unit_day_artifact",
                    "dashboard group operation is not implemented by the artifact facade",
                    "refused",
                    _evidence,
                    codes=("facade.analysis.operation",),
                    reason="Artifact-backed facade does not expose dashboard grouping.",
                ),
            )
        )
    elif _key == "allocation":
        _TRIGGER_AUXILIARY_OUTCOMES.extend(
            (
                _expectation(
                    "allocation_definitions_population",
                    _operation,
                    "from_definitions",
                    "assigned and triggered allocation histories are requested",
                    "supported",
                    _evidence,
                ),
                _expectation(
                    "allocation_artifact_refused",
                    _operation,
                    "from_unit_day_artifact",
                    "artifact-backed facade has no raw allocation event stream",
                    "refused",
                    "tests.test_analysis_trigger::test_triggered_factor_group_and_allocation_preserve_population",
                    codes=("facade.analysis.operation",),
                    reason="Artifact-backed facade does not retain raw allocation events.",
                ),
                _expectation(
                    "allocation_moments_refused",
                    _operation,
                    "from_moments",
                    "moments source contains no raw allocation event stream",
                    "refused",
                    "tests.test_allocation_history::test_allocation_history_refuses_a_seam_family_instance",
                    codes=("facade.analysis.operation",),
                    reason="Moments-backed facade does not retain raw allocation events.",
                ),
                *(
                    _expectation(
                        f"allocation_{ingress}_refused",
                        _operation,
                        ingress,
                        "source has no retained raw exposure-event stream for daily allocation history",
                        "refused",
                        "tests.test_allocation_history::test_allocation_history_refuses_non_native_ingresses",
                        codes=("facade.analysis.operation",),
                        reason="This source retains no raw allocation events.",
                    )
                    for ingress in ("from_unit_summary", "from_unit_panel", "from_switchback_panel")
                ),
            )
        )
    elif _key == "sitewide":
        _TRIGGER_AUXILIARY_OUTCOMES.extend(
            (
                _expectation(
                    "sitewide_definitions_assigned_supported",
                    _operation,
                    "from_definitions",
                    "sitewide requested with the default assigned population on an experiment that declares a trigger",
                    "supported",
                    _evidence,
                ),
                _expectation(
                    "sitewide_artifact_assigned_supported",
                    _operation,
                    "from_unit_day_artifact",
                    "sitewide requested with the default assigned population and a site-volume extension",
                    "supported",
                    _evidence,
                ),
                _expectation(
                    "sitewide_triggered_population_refused",
                    _operation,
                    "from_definitions",
                    "sitewide requested with population='triggered' on an experiment that declares a trigger",
                    "refused",
                    _evidence,
                    codes=("analysis.sitewide.triggered_population_unsupported",),
                    reason="The triggered-population sitewide estimand is not offered.",
                ),
                _expectation(
                    "sitewide_artifact_triggered_population_refused",
                    _operation,
                    "from_unit_day_artifact",
                    "sitewide requested with population='triggered' on an experiment that declares a trigger",
                    "refused",
                    _evidence,
                    codes=("analysis.sitewide.triggered_population_unsupported",),
                    reason="The triggered-population sitewide estimand is not offered.",
                ),
                *(
                    _expectation(
                        f"sitewide_{ingress}_triggered_population_refused",
                        _operation,
                        ingress,
                        "sitewide requested with population='triggered'",
                        "refused",
                        "tests.test_analysis_trigger::test_sitewide_triggered_population_refuses_before_loading_on_every_ingress",
                        codes=("analysis.sitewide.triggered_population_unsupported",),
                        reason="The triggered-population sitewide estimand is not offered.",
                    )
                    for ingress in ("from_unit_summary", "from_unit_panel", "from_moments")
                ),
            )
        )
    else:
        _TRIGGER_AUXILIARY_OUTCOMES.extend(
            (
                _expectation(
                    "sitewide_triggered_population_refused",
                    _operation,
                    "from_definitions",
                    "sitewide requested with population='triggered' on an experiment that declares a trigger",
                    "refused",
                    _evidence,
                    codes=("analysis.sitewide.triggered_population_unsupported",),
                    reason="The triggered-population sitewide estimand is not offered.",
                ),
                _expectation(
                    "sitewide_artifact_triggered_population_refused",
                    _operation,
                    "from_unit_day_artifact",
                    "sitewide requested with population='triggered' on an experiment that declares a trigger",
                    "refused",
                    _evidence,
                    codes=("analysis.sitewide.triggered_population_unsupported",),
                    reason="The triggered-population sitewide estimand is not offered.",
                ),
            )
        )

_TRIGGER_AUXILIARY_OUTCOMES.append(
    _expectation(
        "cate_artifact_observational_covariates_refused",
        "estimate_cate()",
        "from_unit_day_artifact",
        "estimate_cate() on an observational artifact with complete covariate and trigger evidence",
        "refused",
        "tests.test_analysis_trigger::test_triggered_cate_observational_artifact_refuses_even_with_covariate_evidence",
        codes=("cate.identification.randomized_only",),
        reason="Published covariates do not establish randomized assignment or an observational CATE estimator.",
    )
)

_TRIGGER_AUXILIARY_OUTCOMES.append(
    _expectation(
        "factor_artifact_extension_decoder_missing_extension",
        "read_extension(request=factor_dimension)",
        "from_unit_day_artifact",
        "typed factor extension decoding is requested without a published extension",
        "refused",
        "tests.test_unit_day_artifact_extensions::test_every_extension_rejects_missing_operation_and_extension",
        codes=("artifact.extension.missing",),
        reason="The typed artifact extension reader has no matching published factor extension.",
    )
)

for _method, _leaf in (
    ("validate_cate", "validation"),
    ("targeting_rule", "targeting"),
    ("select_targeting_rule", "selection"),
):
    for _ingress, _source_kind in (
        ("from_definitions", "definitions"),
        ("from_unit_day_artifact", "artifact"),
    ):
        _TRIGGER_AUXILIARY_OUTCOMES.append(
            _expectation(
                f"cate_{_source_kind}_observational_{_leaf}_supported",
                f"{_method}()",
                _ingress,
                f"{_method}() on an observational {_source_kind} source with complete covariate and trigger evidence",
                "supported",
                "tests.test_analysis_trigger::test_triggered_cate_observational_artifact_refuses_even_with_covariate_evidence",
            )
        )

_TRIGGER_AUXILIARY_OUTCOMES.append(
    _expectation(
        "cate_definitions_observational_covariates_refused",
        "estimate_cate()",
        "from_definitions",
        "estimate_cate() on an observational definitions source with complete covariate and trigger evidence",
        "refused",
        "tests.test_analysis_trigger::test_triggered_cate_observational_artifact_refuses_even_with_covariate_evidence",
        codes=("cate.identification.randomized_only",),
        reason="Published covariates do not establish randomized assignment or an observational CATE estimator.",
    )
)

for _key, _operation, _evidence in _TRIGGER_READS:
    if _key == "allocation":
        continue
    for _suffix, _ingress in (
        ("summary", "from_unit_summary"),
        ("panel", "from_unit_panel"),
        ("moments", "from_moments"),
        ("switchback", "from_switchback_panel"),
    ):
        _cate_boundary = _key == "cate"
        _code = (
            "facade.analysis.contrast_unavailable"
            if _suffix == "switchback"
            else "facade.analysis.operation"
        )
        _TRIGGER_AUXILIARY_OUTCOMES.append(
            _expectation(
                f"{_key}_{_suffix}",
                _operation,
                _ingress,
                "trigger declaration requested at construction"
                if _cate_boundary
                else "operation requested from a source without native auxiliary evidence",
                "not_comparable" if _cate_boundary else "refused",
                "tests.test_triggered_auxiliary_ingress_contracts::"
                + (
                    "test_cate_trigger_membership_is_not_expressible_at_public_constructor"
                    if _cate_boundary
                    else "test_triggered_auxiliary_operations_reach_real_ingress_boundaries"
                ),
                codes=() if _cate_boundary else (_code,),
                reason=_TRIGGER_OTHER
                if _cate_boundary
                else "This source cannot supply the requested auxiliary operation.",
                stage="construction" if _cate_boundary else "read",
            )
        )

CONSUMER_CONTRACTS: tuple[ConsumerContract, ...] = (
    ConsumerContract("triggered_auxiliary_reads", tuple(_TRIGGER_AUXILIARY_OUTCOMES)),
    ConsumerContract(
        "triggered-sequential-boundary",
        tuple(
            _expectation(
                ingress,
                "run(population='triggered')",
                ingress,
                "frame source or moments without a retained triggered checkpoint",
                "refused",
                "tests.test_analysis_moments::test_triggered_run_refuses_unit_summary_and_moment_replay",
                codes=("facade.analysis.trigger_unsupported",),
                reason="This source carries no trigger declaration or trigger membership.",
                stage="run",
            )
            for ingress in (
                "from_unit_summary",
                "from_unit_panel",
                "from_switchback_panel",
                "from_moments",
            )
        ),
    ),
    ConsumerContract(
        "triggered-sequential-certification",
        tuple(
            _expectation(
                key,
                "capture_sequential(finalized=True, as_of=...)",
                ingress,
                condition,
                "refused",
                f"tests.test_triggered_sequential::{test}",
                codes=(code,),
                reason=reason,
                stage="capture",
            )
            for key, ingress, condition, test, code, reason in (
                (
                    "native-feed",
                    "from_definitions",
                    "uncertified contributing feed",
                    "test_capture_refuses_feeds_not_certified_through_as_of",
                    "sequential.source.invalid",
                    "Every contributing feed must be certified complete through the capture day.",
                ),
                (
                    "native-sql",
                    "from_definitions",
                    "SQL-declared trigger",
                    "test_capture_refuses_a_sql_declared_trigger_and_missing_evidence",
                    "sequential.source.invalid",
                    "A SQL-declared trigger has no feed that can certify its completeness.",
                ),
                (
                    "native-unpinned",
                    "from_definitions",
                    "missing pinned source evidence",
                    "test_capture_refuses_a_sql_declared_trigger_and_missing_evidence",
                    "source.native.trigger_evidence_required",
                    "Trigger membership requires pinned source snapshot evidence.",
                ),
                (
                    "artifact-feed",
                    "from_unit_day_artifact",
                    "uncertified trigger feed",
                    "test_artifact_capture_refuses_an_uncertified_trigger_feed",
                    "sequential.source.invalid",
                    "Stored trigger evidence must certify completeness through the capture day.",
                ),
            )
        ),
    ),
    ConsumerContract(
        "triggered-sequential-uptake",
        tuple(
            _expectation(
                f"{ingress}-unavailable",
                "run(estimands=('itt', 'compliance'))",
                ingress,
                "registered encouragement plan with a triggered checkpoint",
                "unavailable",
                "tests.test_triggered_sequential::test_triggered_uptake_cells_stay_unavailable",
                codes=("readout.cell.unsupported_request",),
                reason="triggered_uptake_unsupported",
                stage="run",
            )
            for ingress in ("from_definitions", "from_unit_day_artifact", "from_moments")
        ),
    ),
    ConsumerContract(
        "triggered-sequential-continuation",
        tuple(
            _expectation(
                f"{ingress}-{suffix}",
                operation,
                ingress,
                "assigned-only commitment: "
                + (
                    "read under a declared trigger"
                    if suffix == "unavailable"
                    else "retroactive backfill"
                ),
                kind,
                "tests.test_triggered_sequential::"
                "test_assigned_only_state_retains_unavailable_triggered_rows_across_sources",
                codes=(code,),
                reason=reason,
                stage=stage,
            )
            for ingress in ("from_definitions", "from_unit_day_artifact", "from_moments")
            for suffix, operation, kind, code, reason, stage in (
                (
                    "unavailable",
                    "run()",
                    "unavailable",
                    "readout.cell.unsupported_request",
                    "triggered_chain_uncommitted",
                    "run",
                ),
                (
                    "backfill",
                    "SequentialSnapshot.verify_parent()",
                    "refused",
                    "sequential.continuation.legacy",
                    "A committed assigned-only process cannot acquire a triggered chain retroactively.",
                    "verify_parent",
                ),
            )
        ),
    ),
    ConsumerContract(
        "assignment_integrity_source_grain",
        (
            _expectation(
                "panel_unit_grain_supported",
                "srm()",
                "from_unit_panel",
                "multiple daily rows per unit must still count each assigned unit once",
                "supported",
                "tests.test_analysis_facade::test_assignment_integrity_uses_each_source_declared_grain",
            ),
            _expectation(
                "moments_unit_grain_supported",
                "srm()",
                "from_moments",
                "replayed unit-grain moments preserve assignment counts, not day-row counts",
                "supported",
                "tests.test_analysis_facade::test_assignment_integrity_uses_each_source_declared_grain",
            ),
            _expectation(
                "summary_cluster_grain_supported",
                "srm()",
                "from_unit_summary",
                "summary with cluster identities counts assigned clusters rather than member units",
                "supported",
                "tests.test_analysis_facade::test_assignment_integrity_uses_each_source_declared_grain",
            ),
        ),
    ),
    ConsumerContract(
        "moments_checkpoint_reexport_boundary",
        (
            _expectation(
                "moments_checkpoint_reexport_refused",
                "Analysis.export(sequential checkpoint)",
                "from_moments",
                "a moments-replayed source receives an export request for a sequential checkpoint",
                "refused",
                "tests.test_analysis_facade::test_moments_checkpoint_reexport_refused",
                codes=("facade.analysis.operation",),
                reason="A moments replay cannot be re-exported as a checkpoint.",
                stage="read",
            ),
        ),
    ),
    ConsumerContract(
        "observational_relative_margin_boundary",
        (
            _expectation(
                "relative_margin_summary_constructor_refused",
                "from_unit_summary(..., plan=relative margin)",
                "from_unit_summary",
                "Observational design declares relative margin",
                "refused",
                "tests.test_readouts_observational::test_run_observational_declared_relative_margin_refused_at_construction",
                codes=("plan.observational.relative_margin",),
                reason="Observational readouts do not implement a shifted relative null.",
                stage="construction",
            ),
            _expectation(
                "relative_margin_moments_constructor_refused",
                "Analysis.from_moments(..., plan=relative margin)",
                "from_moments",
                "Observational design declares relative margin",
                "refused",
                "tests.test_readouts_observational::test_moments_source_observational_declared_relative_margin_refused_at_construction",
                codes=("plan.observational.relative_margin",),
                reason="Observational readouts do not implement a shifted relative null.",
                stage="construction",
            ),
            _expectation(
                "relative_margin_sql_panel_constructor_refused",
                "SqlPanelSource.from_frame_via_memtable(..., plan=relative margin)",
                "SqlPanelSource",
                "Observational design declares relative margin",
                "refused",
                "tests.test_readouts_observational::test_sql_panel_source_observational_declared_relative_margin_refused_at_construction",
                codes=("plan.observational.relative_margin",),
                reason="Observational readouts do not implement a shifted relative null.",
                stage="construction",
            ),
            _expectation(
                "relative_margin_definitions_facade_refused",
                "make_analysis_like(..., design=Observational)",
                "from_definitions",
                "Definitions-backed metric declares a relative margin",
                "refused",
                "tests.test_analysis_facade::test_observational_relative_margin_refused_at_constructor",
                codes=("plan.observational.relative_margin",),
                reason="Observational readouts do not implement a shifted relative null.",
                stage="construction",
            ),
            _expectation(
                "relative_margin_quantile_moments_refused",
                "Analysis.from_moments(..., quantile + relative margin)",
                "from_moments",
                "Quantile metric has a declared relative margin under Observational design",
                "refused",
                "tests.test_readouts_observational::test_quantile_over_scalar_moments_refuses_in_the_documented_precedence",
                codes=("plan.observational.relative_margin",),
                reason="Declared relative-margin validation precedes quantile readout dispatch.",
                stage="construction",
            ),
            _expectation(
                "observational_quantile_two_sided_refused",
                "run() quantile",
                "from_moments",
                "Observational quantile request is two-sided and has no shifted margin",
                "refused",
                "tests.test_readouts_observational::test_quantile_over_scalar_moments_refuses_in_the_documented_precedence",
                codes=("readout.observational.quantile",),
                reason="No observational quantile estimator is available.",
                stage="read",
            ),
            _expectation(
                "observational_quantile_alternative_refused",
                "run() quantile with one-sided alternative",
                "from_moments",
                "Observational quantile request declares a one-sided alternative",
                "refused",
                "tests.test_readouts_observational::test_quantile_over_scalar_moments_refuses_in_the_documented_precedence",
                codes=("readout.metric.quantile_alternative",),
                reason="Quantiles do not support one-sided alternative inference.",
                stage="read",
            ),
            _expectation(
                "observational_quantile_guardrail_refused",
                "run() quantile guardrail",
                "from_moments",
                "Observational quantile request declares a marginless guardrail",
                "refused",
                "tests.test_readouts_observational::test_quantile_over_scalar_moments_refuses_in_the_documented_precedence",
                codes=("readout.metric.quantile_alternative",),
                reason="Quantile guardrails require unsupported one-sided inference.",
                stage="read",
            ),
            _expectation(
                "observational_quantile_absolute_margin_refused",
                "run() quantile with absolute margin",
                "from_moments",
                "Observational quantile request declares absolute margin",
                "refused",
                "tests.test_readouts_observational::test_quantile_over_scalar_moments_refuses_in_the_documented_precedence",
                codes=("readout.metric.quantile_alternative",),
                reason="Quantile shifted-null inference is unsupported.",
                stage="read",
            ),
            _expectation(
                "absolute_margin_passes_relative_gate",
                "run() absolute margin",
                "Observational",
                "Absolute margin reaches later treatment-arm validation rather than relative-margin refusal",
                "refused",
                "tests.test_readouts_observational::test_run_observational_declared_absolute_margin_is_no_longer_refused",
                codes=("readout.arms.no_treatment",),
                reason="Absolute margin passes relative-margin validation and reaches the treatment-arm gate.",
                stage="read",
            ),
            _expectation(
                "one_sided_observational_alternative_supported",
                "run(alternative='greater')",
                "from_unit_summary",
                "Observational design one-sided alternative is compatible and interval alpha doubling is applied",
                "supported",
                "tests.test_readouts_observational::test_run_one_sided_alternative_observational_doubles_alpha_and_labels",
            ),
            *(
                _expectation(
                    outcome,
                    f"{ingress}()/run()",
                    ingress,
                    condition,
                    "refused",
                    "tests.test_readouts_observational::test_public_observational_quantile_margin_precedence",
                    codes=(code,),
                    reason="Observational quantile requests retain construction/read refusal precedence.",
                    stage=stage,
                )
                for outcome, ingress, condition, code, stage in (
                    (
                        "panel_relative_margin_refused",
                        "from_unit_panel",
                        "observational relative-margin plan",
                        "plan.observational.relative_margin",
                        "construction",
                    ),
                    (
                        "panel_absolute_margin_precedence_refused",
                        "from_unit_panel",
                        "observational quantile with absolute margin",
                        "readout.metric.quantile_alternative",
                        "read",
                    ),
                    (
                        "panel_one_sided_precedence_refused",
                        "from_unit_panel",
                        "observational quantile with one-sided alternative",
                        "readout.metric.quantile_alternative",
                        "read",
                    ),
                    (
                        "summary_relative_margin_quantile_precedence_refused",
                        "from_unit_summary",
                        "observational quantile with relative margin",
                        "plan.observational.relative_margin",
                        "construction",
                    ),
                    (
                        "summary_absolute_margin_quantile_precedence_refused",
                        "from_unit_summary",
                        "observational quantile with absolute margin",
                        "readout.metric.quantile_alternative",
                        "read",
                    ),
                    (
                        "summary_one_sided_quantile_precedence_refused",
                        "from_unit_summary",
                        "observational quantile with one-sided alternative",
                        "readout.metric.quantile_alternative",
                        "read",
                    ),
                )
            ),
            _expectation(
                "relative_margin_public_definitions_constructor_refused",
                "from_definitions()",
                "from_definitions",
                "observational mean guardrail with relative margin",
                "refused",
                "tests.test_readouts_observational::test_public_definitions_relative_margin_refused_during_construction",
                codes=("plan.observational.relative_margin",),
                reason="Observational relative-margin plans refuse before loading evidence.",
                stage="construction",
            ),
            *(
                _expectation(
                    f"{ingress}_{request}_quantile_precedence_refused",
                    "run()",
                    ingress,
                    f"observational quantile with {request.replace('_', ' ')}",
                    "refused",
                    "tests.test_readouts_observational::test_native_and_artifact_observational_quantile_requests_keep_precedence",
                    codes=("readout.metric.quantile_alternative",),
                    reason="Quantile one-sided or shifted-null requests refuse before observational dispatch.",
                )
                for ingress in ("from_definitions", "from_unit_day_artifact")
                for request in ("absolute_margin", "one_sided")
            ),
            _expectation(
                "relative_margin_artifact_context_unexpressible",
                "produce artifact context",
                "from_unit_day_artifact",
                "observational relative-margin plan",
                "not_expressible",
                "tests.test_readouts_observational::test_observational_relative_margin_context_cannot_produce_an_artifact",
                reason="The producer refuses plan.observational.relative_margin before publication; "
                "artifact adoption cannot override its immutable plan.",
                stage="construction",
            ),
        ),
    ),
    ConsumerContract(
        "triggered_clustered_day_axis",
        (
            _expectation(
                "clustered_outcome_history_refused",
                "run_daily()/run_asof() outcome history",
                "from_definitions",
                "clustered experiment requests triggered outcome history",
                "refused",
                "tests.test_analysis_trigger::test_triggered_clustered_day_axis_keeps_existing_refusal",
                codes=("facade.analysis.clustered_day_axis",),
                reason="Clustered triggered outcome day-axis is refused.",
                stage="read",
            ),
            _expectation(
                "clustered_compliance_history_supported",
                "run_daily_lift()/run_asof_lift() compliance-only history",
                "from_definitions",
                "clustered triggered experiment requests uptake-only history",
                "supported",
                "tests.test_compliance_artifacts::test_triggered_compliance_matches_population_across_sources",
            ),
            _expectation(
                "clustered_artifact_outcome_history_refused",
                "run_daily()/run_asof() outcome history",
                "from_unit_day_artifact",
                "clustered experiment requests triggered outcome history from an adopted artifact",
                "refused",
                "tests.test_compliance_artifacts::test_triggered_compliance_matches_population_across_sources",
                codes=("facade.analysis.clustered_day_axis",),
                reason="Clustered triggered outcome day-axis is refused for artifact-backed history.",
                stage="read",
            ),
            _expectation(
                "clustered_artifact_compliance_history_supported",
                "run_daily_lift()/run_asof_lift() compliance-only history",
                "from_unit_day_artifact",
                "clustered triggered experiment requests uptake-only history from an adopted artifact",
                "supported",
                "tests.test_compliance_artifacts::test_triggered_compliance_matches_population_across_sources",
            ),
            _expectation(
                "clustered_triggered_population_supported",
                "run()",
                "from_definitions",
                "clustered experiment requests triggered outcome population",
                "supported",
                "tests.test_analysis_trigger::test_clustered_experiment_with_trigger_narrows_to_the_triggered_population",
            ),
        ),
    ),
    ConsumerContract(
        "triggered_day_axis_boundary",
        tuple(
            _expectation(
                f"day_axis_{ingress}_{operation}_refused",
                f"{operation}(population='triggered')",
                ingress,
                "triggered daily/as-of values or lift requested from an ingress without source trigger evidence",
                "refused",
                "tests.test_analysis_trigger::test_triggered_day_axis_refuses_non_native_ingresses",
                codes=(
                    "facade.analysis.contrast_unavailable"
                    if ingress == "from_switchback_panel" and operation in ("run_daily", "run_asof")
                    else "facade.analysis.trigger_unsupported",
                ),
                reason=(
                    "Switchback sources do not provide the parallel-arm contrast used by these day-axis APIs."
                    if ingress == "from_switchback_panel" and operation in ("run_daily", "run_asof")
                    else "Triggered day-axis reads require a supported definitions or unit-day artifact source."
                ),
                stage="read",
            )
            for ingress in (
                "from_unit_summary",
                "from_unit_panel",
                "from_switchback_panel",
                "from_moments",
            )
            for operation in ("run_daily", "run_asof", "run_daily_lift", "run_asof_lift")
        ),
    ),
    ConsumerContract(
        "trigger_source_snapshot_evidence",
        (
            _expectation(
                "triggered_read_requires_source_snapshot",
                "trigger_rates()/triggered population read",
                "from_definitions",
                "trigger membership read without explicit SourceSnapshotEvidence",
                "refused",
                "tests.test_analysis_trigger::test_triggered_request_requires_explicit_snapshot_evidence_but_assigned_does_not",
                codes=("source.native.trigger_evidence_required",),
                reason="Triggered membership requires a pinned snapshot and feed watermark.",
            ),
        ),
    ),
    ConsumerContract(
        "encouragement_retention_readouts",
        (
            _expectation(
                "itt_late_definitions_refused",
                "run() ITT/LATE with retention outcome",
                "from_definitions",
                "Encouragement design retains a retention outcome in the requested estimands",
                "refused",
                "tests.test_readouts_encouragement::test_encouragement_retention_refuses_on_every_reachable_constructor",
                codes=("readout.encouragement.retention",),
                reason="Retention is not estimable under an Encouragement design.",
                stage="read",
            ),
            _expectation(
                "itt_late_artifact_refused",
                "run() ITT/LATE with retention outcome",
                "from_unit_day_artifact",
                "Encouragement design retains a retention outcome in the requested estimands",
                "refused",
                "tests.test_readouts_encouragement::test_encouragement_retention_refuses_on_every_reachable_constructor",
                codes=("readout.encouragement.retention",),
                reason="Retention is not estimable under an Encouragement design.",
                stage="read",
            ),
            _expectation(
                "itt_late_moments_refused",
                "run() ITT/LATE with retention outcome",
                "from_moments",
                "Encouragement design retains a retention outcome in the requested estimands",
                "refused",
                "tests.test_readouts_encouragement::test_encouragement_retention_refuses_on_every_reachable_constructor",
                codes=("readout.encouragement.retention",),
                reason="Retention is not estimable under an Encouragement design.",
                stage="read",
            ),
            _expectation(
                "itt_late_panel_refused",
                "Analysis.from_unit_panel() with retention outcome",
                "from_unit_panel",
                "Encouragement design declares a retention outcome",
                "refused",
                "tests.test_readouts_encouragement::test_encouragement_retention_refuses_on_every_reachable_constructor",
                codes=("readout.encouragement.retention",),
                reason="The panel refuses the retention outcome before any readout.",
                stage="construction",
            ),
            _expectation(
                "itt_late_summary_refused",
                "Analysis.from_unit_summary() with retention outcome",
                "from_unit_summary",
                "retention threshold_days needs per-unit dates the summary does not carry",
                "refused",
                "tests.test_readouts_encouragement::test_encouragement_retention_refuses_on_every_reachable_constructor",
                codes=("source.frame.constructor",),
                reason="One-row-per-unit summaries carry no dates for retention thresholds.",
                stage="construction",
            ),
            _expectation(
                "compliance_only_definitions_supported",
                "run(estimands=('compliance',))",
                "from_definitions",
                "compliance-only request ignores the retention outcome configuration",
                "supported",
                "tests.test_compliance_summary::test_compliance_only_ignores_outcome_configuration",
            ),
            _expectation(
                "compliance_only_itt_still_refused",
                "run(estimands=('itt',))",
                "from_definitions",
                "same experiment requests the retention ITT outcome",
                "refused",
                "tests.test_compliance_summary::test_compliance_only_ignores_outcome_configuration",
                codes=("readout.encouragement.retention",),
                reason="Retention remains refused for the ITT outcome.",
                stage="read",
            ),
        ),
    ),
    ConsumerContract(
        "switchback_multi_arm_identification",
        tuple(
            _expectation(
                f"{metric_type}_multi_arm_refused",
                "Analysis.from_switchback_panel()",
                "from_switchback_panel",
                f"Randomized allocation has control plus multiple treatment arms; {metric_type} metric",
                "refused",
                "tests.simulate.test_switchback::test_switchback_invalid_allocation_refuses_before_frame_access",
                codes=("source.frame.switchback.identification",),
                reason="Switchback identification requires exactly one control and one treatment arm, before metric or sequential-plan validation.",
                stage="construction",
            )
            for metric_type in ("conversion", "retention")
        ),
    ),
    ConsumerContract(
        "retained_retention_checkpoint_reads",
        tuple(
            outcome
            for ingress in ("from_definitions", "from_unit_day_artifact", "from_moments")
            for outcome in (
                _expectation(
                    f"{ingress}_compliance_asof_supported",
                    "run_asof_lift(estimands=('compliance',), completed_windows_only=True)",
                    ingress,
                    "uptake-only registration retains an unbounded retention outcome; compliance-only as-of request",
                    "supported",
                    "tests.test_parity_harness::test_parity_case",
                ),
                _expectation(
                    f"{ingress}_itt_asof_refused",
                    "run_asof_lift(estimands=('itt',), completed_windows_only=False/True)",
                    ingress,
                    "same retained retention catalog; ITT as-of request with either completion policy",
                    "refused",
                    "tests.test_parity_harness::test_parity_case",
                    codes=("breakout.retention.encouragement",),
                    reason="Retention outcomes remain refused under encouragement even when compliance-only monitoring succeeds.",
                ),
                _expectation(
                    f"{ingress}_itt_run_refused",
                    "run(estimands=('itt',))",
                    ingress,
                    "same uptake-only registration and retained retention catalog; ITT outcome request",
                    "refused",
                    "tests.test_parity_harness::test_parity_case",
                    codes=("sequential.source.invalid",),
                    reason="The uptake-only checkpoint supplies no retained sequential state for the retention ITT outcome.",
                ),
            )
        ),
    ),
    ConsumerContract(
        "encouragement_switchback_boundary",
        (
            _expectation(
                "switchback_encouragement_identification_refused",
                "Analysis.from_switchback_panel(identification=Encouragement)",
                "from_switchback_panel",
                "Encouragement design supplies identification to the switchback facade",
                "refused",
                "tests.test_readouts_encouragement::test_encouragement_switchback_identification_is_refused",
                codes=("source.frame.switchback.identification",),
                reason="Switchback identification requires a supported randomized design.",
                stage="construction",
            ),
        ),
    ),
    ConsumerContract(
        "logged_policy_family_boundary",
        tuple(
            _expectation(
                f"analysis_ingress_{ingress}_not_expressible",
                "estimate_policy_contrast()",
                ingress,
                "Analysis ingresses cannot carry a LoggedTrace, which is the required contrast input",
                "not_expressible",
                "tests.logged_policy.test_public_api::test_root_and_results_exports_resolve_lazily_to_the_family_modules",
                reason="Policy contrast is a separate family-owned estimand, not an Analysis readout.",
            )
            for ingress in (
                "from_unit_summary",
                "from_unit_panel",
                "from_switchback_panel",
                "from_definitions",
                "from_unit_day_artifact",
                "from_moments",
            )
        )
        + tuple(
            _expectation(
                f"policy_persistence_analysis_ingress_{ingress}_not_comparable",
                "TabularPolicy JSON/Python persistence",
                ingress,
                "Analysis source persists effect-analysis results rather than policy tables",
                "not_comparable",
                "tests.logged_policy.test_public_api::test_builtin_integer_keyed_policies_persist_in_python_mode_only",
                reason="TabularPolicy persistence is family-owned and not an Analysis persistence operation.",
            )
            for ingress in (
                "from_unit_summary",
                "from_unit_panel",
                "from_switchback_panel",
                "from_definitions",
                "from_unit_day_artifact",
                "from_moments",
            )
        )
        + (
            _expectation(
                "logged_trace_contrast_supported",
                "estimate_policy_contrast()",
                "LoggedTrace.from_records/from_frame",
                "trace has valid logging-policy evidence and support",
                "supported",
                "tests.logged_policy.test_public_api::test_forty_unit_fixture_yields_the_exact_contrast_through_the_root_export",
            ),
            _expectation(
                "tabular_policy_typed_json_refused",
                "TabularPolicy.model_dump_json()",
                "TabularPolicy",
                "integer-keyed policy table uses JSON mode",
                "refused",
                "tests.logged_policy.test_public_api::test_builtin_integer_keyed_policies_persist_in_python_mode_only",
                codes=("logged_policy.policy.json_context_key",),
                reason="Typed context keys cannot be represented by JSON object keys.",
            ),
            _expectation(
                "tabular_policy_string_json_supported",
                "TabularPolicy.model_dump_json()/model_validate_json()",
                "TabularPolicy",
                "string-keyed policy table uses JSON mode",
                "supported",
                "tests.logged_policy.test_public_api::test_builtin_integer_keyed_policies_persist_in_python_mode_only",
            ),
            _expectation(
                "tabular_policy_typed_python_supported",
                "TabularPolicy.model_dump(mode='python')/trusted pickle",
                "TabularPolicy",
                "typed-key policy table uses Python mode or trusted pickle",
                "supported",
                "tests.logged_policy.test_public_api::test_builtin_integer_keyed_policies_persist_in_python_mode_only",
            ),
        ),
    ),
    ConsumerContract(
        "stored_bool_null_continuation",
        (
            _expectation(
                "raw_bool_null_replay",
                "sequential_snapshot()",
                "from_moments",
                "frozen checkpoint contains historical raw True/False/None spellings",
                "replay_only",
                "tests.test_sequential_registration_contract::test_pre_normalization_bool_null_prefix_replays_but_cannot_be_relabelled_forward",
                reason="Historical checkpoint replay is not proof of a live-source continuation.",
            ),
            _expectation(
                "same_spelling_continuation",
                "capture_sequential(previous=...)",
                "from_unit_summary",
                "incoming rows preserve historical raw spellings",
                "supported",
                "tests.test_sequential_registration_contract::test_pre_normalization_bool_null_prefix_replays_but_cannot_be_relabelled_forward",
            ),
            _expectation(
                "canonical_relabel_refused",
                "capture_sequential(previous=...)",
                "from_unit_summary",
                "incoming typed boolean/null levels are canonically relabelled",
                "refused",
                "tests.test_sequential_registration_contract::test_pre_normalization_bool_null_prefix_replays_but_cannot_be_relabelled_forward",
                codes=("sequential.continuation.rewrite",),
                reason="Canonical relabeling changes the frozen continuation identity.",
            ),
            _expectation(
                "panel_legacy_unbounded_prefix_refused",
                "capture_sequential(previous=...)",
                "from_unit_panel",
                "canonical labels attempt to continue a legacy unbounded-window prefix at an explicit panel horizon",
                "refused",
                "tests.test_sequential_registration_contract::test_pre_normalization_bool_null_prefix_replays_but_cannot_be_relabelled_forward",
                codes=("sequential.route.unsupported",),
                reason="Panel capture requires bounded metric windows covered by a common joint-reveal window; continue this legacy finalized prefix through unit summary.",
            ),
        ),
    ),
    ConsumerContract(
        "experiment_window_compatibility",
        (
            _expectation(
                "window_native_supported",
                "capture_sequential()",
                "from_definitions",
                "day-boundary-compatible start/end/horizon window identity",
                "supported",
                "tests.test_sequential_registration_contract::test_registered_native_and_artifact_capture_share_source_identity",
            ),
            _expectation(
                "window_artifact_supported",
                "capture_sequential()",
                "from_unit_day_artifact",
                "day-boundary-compatible start/end/horizon window identity",
                "supported",
                "tests.test_sequential_registration_contract::test_registered_native_and_artifact_capture_share_source_identity",
            ),
            _expectation(
                "window_old_native_refused",
                "capture_sequential()",
                "from_definitions",
                "legacy native registration predates bound window_days",
                "refused",
                "tests.test_sequential_registration_contract::test_a_registration_from_before_window_days_were_bound_refuses_new_native_capture",
                codes=("sequential.source.invalid",),
                reason="Legacy registration does not identify the derived observation window.",
                stage="construction",
            ),
            _expectation(
                "window_artifact_context_refused",
                "open/adopt artifact",
                "from_unit_day_artifact",
                "artifact context omits window_days",
                "refused",
                "tests.test_unit_day_artifact_adoption::test_frozen_collision_artifacts_are_refused_at_every_entry_point",
                codes=("artifact.context.mismatch",),
                reason="Artifact context identity omits the experiment window.",
                stage="admission",
            ),
        ),
    ),
    ConsumerContract(
        "artifact_publication_abort_recovery",
        (
            _expectation(
                "publication_from_definitions_supported",
                "publish_unit_day_artifact()",
                "from_definitions",
                "definitions-backed Analysis publishes its warehouse artifact",
                "supported",
                "tests.test_analysis_facade::test_public_artifact_adoption_matches_native_results",
            ),
            _expectation(
                "dropped_artifact_read_refused",
                "read adopted Analysis after generation drop",
                "from_unit_day_artifact",
                "an already-adopted reader attempts to read a generation tombstoned by the store",
                "refused",
                "tests.test_analysis_facade::test_public_artifact_adoption_refuses_a_dropped_generation",
                codes=("artifact.generation.dropped",),
                reason="The artifact generation was tombstoned and erased.",
                stage="read",
            ),
            _expectation(
                "abort_tombstones_generation",
                "abort publication after manifest insert",
                "WarehouseArtifactStore",
                "manifest insert has begun and publication body aborts",
                "refused",
                "tests.test_unit_day_artifact_production::test_abort_after_manifest_publication_invalidates_the_generation",
                codes=("artifact.generation.dropped",),
                reason="Aborted generation is tombstoned before its relations are erased.",
            ),
            _expectation(
                "abort_preserves_original_error",
                "abort publication with tombstone failure",
                "WarehouseArtifactStore",
                "cleanup tombstone write fails after body error",
                "supported",
                "tests.test_unit_day_artifact_production::test_abort_cleanup_failure_keeps_original_error_and_is_recoverable",
            ),
            _expectation(
                "tombstone_failure_keeps_relations",
                "publication abort with tombstone failure",
                "WarehouseArtifactStore",
                "tombstone cannot be persisted",
                "supported",
                "tests.test_unit_day_artifact_production::test_tombstone_failure_without_row_preserves_and_names_relations",
            ),
            _expectation(
                "fresh_store_recovers_abandoned_generation",
                "abandon_generation()/drop_generation()",
                "fresh WarehouseArtifactStore process on same persistent store",
                "a new process reopens the store and abandons/erases a generation left visible by the prior process",
                "supported",
                "tests.test_unit_day_artifact_production::test_fresh_process_abandons_and_erases_a_persisted_generation",
            ),
        )
        + tuple(
            _expectation(
                f"publication_store_not_comparable_{ingress}",
                "WarehouseArtifactStore publication/abort",
                ingress,
                "this ingress has no WarehouseArtifactStore publication-generation lifecycle",
                "not_comparable",
                "tests.test_unit_day_artifact_production::test_abort_after_manifest_publication_invalidates_the_generation",
                reason="Artifact-store generation lifecycle is not an operation on this Analysis ingress.",
            )
            for ingress in (
                "from_unit_summary",
                "from_unit_panel",
                "from_switchback_panel",
                "from_moments",
            )
        ),
    ),
    ConsumerContract(
        "artifact_immutable_properties",
        (
            _expectation(
                "manifest_in_place_edit_refused",
                "open_snapshot()",
                "WarehouseArtifactStore",
                "manifest JSON is changed without matching stored manifest digest",
                "refused",
                "tests.test_unit_day_artifact_adoption::test_open_refuses_a_manifest_edited_in_place",
                codes=("artifact.manifest.invalid",),
                reason="Manifest digest does not match the persisted manifest body.",
                stage="read",
            ),
            _expectation(
                "external_property_mutation_refused",
                "open_snapshot()/adopted read",
                "WarehouseArtifactStore",
                "published measure relation is mutated after snapshot creation",
                "refused",
                "tests.test_unit_day_artifact_adoption::test_streaming_snapshot_isolates_external_mutation",
                codes=("artifact.snapshot.mixed",),
                reason="Snapshot relation digest no longer matches the immutable published relation.",
                stage="read",
            ),
        ),
    ),
    ConsumerContract(
        "observational_cluster_weight_diagnostics",
        (
            _expectation(
                "summary_cluster_weight_grain",
                "run()",
                "from_unit_summary",
                "IPTW mean, declared cluster grain",
                "supported",
                "tests.test_weight_diagnostic_parity::test_cluster_grain_weight_diagnostics_are_retained_on_unit_summary",
            ),
            _expectation(
                "definitions_cluster_weight_grain",
                "run()",
                "from_definitions",
                "IPTW mean, declared cluster grain",
                "supported",
                "tests.test_weight_diagnostic_parity::test_definitions_cluster_weight_diagnostics_are_retained_at_cluster_grain",
            ),
        ),
    ),
    ConsumerContract(
        "callable_experiment_health",
        (
            _expectation(
                "health_definitions_triggered",
                "health_check()",
                "from_definitions",
                "declared trigger with retained native source evidence",
                "supported",
                "tests.test_experiment_health::test_balanced_trigger_rates_do_not_clear_assumption_risk",
            ),
            _expectation(
                "health_artifact_triggered",
                "health_check()",
                "from_unit_day_artifact",
                "published trigger population, assignment counts and metric-stat extensions",
                "supported",
                "tests.test_experiment_health::test_health_artifact_trigger_extensions_are_optional_but_explicit",
            ),
            _expectation(
                "health_artifact_without_extensions",
                "health_check()",
                "from_unit_day_artifact",
                "artifact omits trigger membership and rate extensions",
                "supported",
                "tests.test_experiment_health::test_health_artifact_without_trigger_extensions_is_unavailable",
            ),
            _expectation(
                "health_unit_panel",
                "health_check()",
                "from_unit_panel",
                "unit panel has no trigger declaration or raw trigger membership",
                "supported",
                "tests.test_experiment_health::test_health_unit_panel_reports_only_supported_population",
            ),
            _expectation(
                "health_switchback",
                "health_check()",
                "from_switchback_panel",
                "switchback assignment law cannot be tested as a unit-arm allocation",
                "supported",
                "tests.test_experiment_health::test_health_switchback_does_not_substitute_unit_arm_srm",
            ),
            _expectation(
                "health_moments",
                "health_check()",
                "from_moments",
                "moments replay retains no declared trigger membership",
                "supported",
                "tests.test_experiment_health::test_health_moments_replay_does_not_reconstruct_trigger_population",
            ),
            _expectation(
                "health_unit_summary",
                "health_check()",
                "from_unit_summary",
                "summary-backed analysis with declared randomized allocation",
                "supported",
                "tests.test_experiment_health::test_untriggered_summary_has_explicit_unavailable_triggered_rows",
            ),
        ),
    ),
)


def expectation(contract_id: str, outcome_id: str) -> ConsumerExpectation:
    """Return one immutable expectation for consumption by its owning test."""
    for contract in CONSUMER_CONTRACTS:
        if contract.id == contract_id:
            return contract.outcome(outcome_id)
    raise KeyError(f"unknown consumer contract {contract_id}")


def assert_expected(
    contract_id: str,
    outcome_id: str,
    *,
    kind: str,
    code: str | None = None,
    stage: str | None = None,
) -> ConsumerExpectation:
    """Bind an actual consumer assertion to its declared expected outcome."""
    declared = expectation(contract_id, outcome_id)
    assert declared.expected.kind == kind, (declared.id, declared.expected)
    if code is not None:
        assert declared.expected.codes == (code,), (declared.id, declared.expected.codes)
    if stage is not None:
        assert declared.expected.stage == stage, (declared.id, declared.expected.stage)

    return declared


def assert_refusal(
    contract_id: str,
    outcome_id: str,
    error: object,
    *,
    stage: str,
) -> ConsumerExpectation:
    """Bind a declared refusal to the code of a measured consumer exception."""
    declared = expectation(contract_id, outcome_id)
    actual_code = getattr(error, "code", None)
    assert declared.expected.kind == "refused", (declared.id, declared.expected)
    assert declared.expected.codes == (actual_code,), (declared.id, actual_code)
    assert declared.expected.stage == stage, (declared.id, stage)
    return declared


def assert_support(
    contract_id: str,
    outcome_id: str,
    result: object,
    *,
    stage: str = "read",
) -> ConsumerExpectation:
    """Bind a supported expectation to a value produced by its real consumer."""
    declared = expectation(contract_id, outcome_id)
    assert not isinstance(result, BaseException), (declared.id, result)
    assert declared.expected.kind == "supported", (declared.id, declared.expected)
    assert declared.expected.stage == stage, (declared.id, stage)
    return declared


def assert_replay_only(contract_id: str, outcome_id: str, replayed: object) -> ConsumerExpectation:
    """Bind a replay-only leaf to the actual immutable replayed value."""
    declared = expectation(contract_id, outcome_id)
    assert replayed is not None, declared.id
    assert declared.expected.kind == "replay_only", (declared.id, declared.expected)
    return declared


PARITY_DECLARATIONS: dict[str, ParityDeclaration] = {}
