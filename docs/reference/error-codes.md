# Error codes

Refusal codes are stable identifiers; their human-readable wording may change.
Context keys are part of the refusal contract. They include template fields, explicitly declared keys and named renderer parameters (including optional parameters).
Registering modules include the declaration and any modules that reuse it.
Retired codes and their replacements are listed below.

## Registered refusal codes

| Code | Error class | Context keys | Registering modules |
| --- | --- | --- | --- |
| `absorption.absorb_factor_needs` | `InvalidRequestError` | `arms` | `increment.absorption` |
| `absorption.control_group_present` | `InvalidRequestError` | `arms`, `control_group` | `increment.absorption` |
| `absorption.summary_duplicate_rows` | `InvalidRequestError` | `factor`, `named` | `increment.absorption` |
| `absorption.summary_eager_dataframe` | `InvalidRequestError` | — | `increment.absorption` |
| `absorption.summary_missing_columns` | `InvalidRequestError` | `missing` | `increment.absorption` |
| `adjust.variance_reduction.unsupported` | `UnsupportedRequestError` | `method`, `variance_reduction` | `increment.estimation.adjust` |
| `adjust.winsorization.percentile_unsupported` | `CapabilityError` | `metric` | `increment.estimation.adjust` |
| `allocation.alpha.underflow` | `InvalidRequestError` | `denominator`, `multiplier`, `numerator` | `increment.compatibility` |
| `analysis.planning_baseline.cluster_icc_not_estimable` | `CapabilityError` | `control_clusters`, `control_units` | `increment._planning_baseline` |
| `analysis.planning_baseline.covariate_source_unavailable` | `CapabilityError` | `metric` | `increment._planning_baseline` |
| `analysis.planning_baseline.cuped_no_residual_variance` | `CapabilityError` | `metric` | `increment._planning_baseline` |
| `analysis.planning_baseline.design_unavailable` | `CapabilityError` | `metric` | `increment._planning_baseline` |
| `analysis.planning_baseline.quantile_insufficient_data` | `CapabilityError` | `metric` | `increment._planning_baseline` |
| `analysis.planning_baseline.quantile_source_unavailable` | `CapabilityError` | `metric` | `increment._planning_baseline` |
| `analysis.planning_baseline.trigger_evidence_unavailable` | `CapabilityError` | `metric`, `missing`, `trigger` | `increment._planning_baseline` |
| `analysis.planning_baseline.trigger_membership_unavailable` | `CapabilityError` | `analyzed_clusters`, `assigned_clusters`, `control_group`, `metric`, `route` | `increment._planning_baseline` |
| `analysis.planning_baseline.trigger_metric_population_unavailable` | `CapabilityError` | `control_group`, `eligible_units`, `metric`, `observed_units`, `route`, `trigger` | `increment._planning_baseline` |
| `analysis.planning_baseline.trigger_population_inconsistent` | `CapabilityError` | `analyzed_clusters`, `assigned_clusters`, `assigned_units`, `derived_mean`, `metric`, `observed_mean`, `route`, `trigger_rate` | `increment._planning_baseline` |
| `arm.adjustment.cluster_cuped` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.adjustment.cluster_prior` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.adjustment.sequential_cuped` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.adjustment.sequential_prior` | `InvalidRequestError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.baseline.absorption_undeclared` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.baseline.cluster_size_required` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.baseline.compliance_undeclared` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.baseline.cuped_undeclared` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.baseline.iid_cluster_knobs` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.baseline.triggering_undeclared` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.inference.cluster` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.metric.quantile_cluster` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.metric.quantile_cuped` | `UnsupportedRequestError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.metric.quantile_sequential` | `UnsupportedRequestError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm.metric.unsupported` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `arm_planning.guardrail_neutral_requires_direction` | `InvalidRequestError` | `alternative`, `preferred_direction` | `increment.estimation.arm_contract` |
| `arm_planning.guardrail_preferred_direction_conflict` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.guardrail_requires_one_sided` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.inference_spec_adjustments_unplannable` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.inference_spec_expected_n_unplannable` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.inference_spec_registration_unplannable` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.inference_spec_segments_unplannable` | `InvalidRequestError` | `rejected_segments`, `route`, `segmented`, `view` | `increment.estimation.arm_contract` |
| `arm_planning.preferred_direction_without_guardrail_role` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.secondaries_without_secondary_role` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.secondary_requires_count` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `arm_planning.sequential_secondary_one_arm` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `artifact.artifact_contract.refusal_code` | `InvalidRequestError` | — | `increment.query.artifact_contract` |
| `artifact.context.legacy_exclusion_acknowledged` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.context.mismatch` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.coverage_dates_date` | `InvalidRequestError` | — | `increment.query.artifact_contract` |
| `artifact.cuped.scalar_breakout` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.digest.json` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.nonfinite` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.nullable` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.primary_key` | `ArtifactDigestError` | `constraint`, `field`, `primary_key`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.range` | `ArtifactDigestError` | `field`, `maximum`, `minimum`, `quantity`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.recanonicalize` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.role` | `ArtifactDigestError` | `constraint`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.row` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.rows` | `ArtifactDigestError` | `constraint`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.schema` | `ArtifactDigestError` | `constraint`, `field`, `position`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.timestamp` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.digest.type` | `ArtifactDigestError` | `constraint`, `field`, `route`, `value` | `increment.query.artifact_digest` |
| `artifact.evidence.unavailable` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.extension.invalid` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.extension.missing` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.first_ds_last` | `InvalidRequestError` | — | `increment.query.artifact_contract` |
| `artifact.format.unsupported` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.generation.dropped` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.identifier.unsafe` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.manifest.invalid` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.manifest.unreferenced_relation` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.metric.binding_mismatch` | `ArtifactContractError` | `message` | `increment.query.artifact_contract`, `increment.query.artifact_reader` |
| `artifact.on_mixed_assignment` | `InvalidRequestError` | — | `increment.query.artifact_contract` |
| `artifact.operation.unsupported` | `CapabilityError` | `offered`, `operation`, `request`, `route` | `increment.query.artifact_reader` |
| `artifact.refresh.context_mismatch` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.refresh.invalid_ref` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.relation.digest_mismatch` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.relation.invalid` | `CapabilityError` | `check`, `detail`, `relation` | `increment.query.artifact_reader` |
| `artifact.relation.schema_mismatch` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.request.trigger_metric_scope` | `InvalidRequestError` | — | `increment.semantics.artifact` |
| `artifact.snapshot.mixed` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `artifact.store.namespace` | `CapabilityError` | `message` | `increment.query.session` |
| `artifact_contract.encouragement_uptake_conflict` | `ArtifactContractError` | `message` | `increment.query.artifact_contract` |
| `breakout.breakout.exactly_one_lift` | `InvalidRequestError` | `reason` | `increment.breakout.estimates` |
| `breakout.breakout.point_unavailable` | `InvalidRequestError` | — | `increment.breakout.estimates` |
| `breakout.coerce_type_datetime` | `InvalidRequestError` | `type_name`, `v` | `increment.breakout.estimates` |
| `breakout.compliance_late_view` | `InvalidRequestError` | — | `increment.breakout.estimates` |
| `breakout.daily_group_summary` | `InvalidRequestError` | `missing` | `increment.breakout.estimates` |
| `breakout.daily_lift.exactly_one_unavailable` | `InvalidRequestError` | — | `increment.breakout.estimates` |
| `breakout.daily_lift.point_unavailable` | `InvalidRequestError` | — | `increment.breakout.estimates` |
| `breakout.daily_metric.exactly_one_value` | `InvalidRequestError` | — | `increment.breakout.estimates` |
| `breakout.metric_found_group` | `InvalidRequestError` | `declared_names`, `unknown_metrics` | `increment.breakout.estimates` |
| `breakout.price_rollout_metrics` | `InvalidRequestError` | `misdirected` | `increment.breakout.rollout` |
| `breakout.quantile` | `CapabilityError` | `method`, `names`, `reason`, `remedy` | `increment.breakout.estimates` |
| `breakout.retention.completion` | `InvalidRequestError` | `fn_name`, `names` | `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `breakout.retention.daily` | `InvalidRequestError` | `fn_name`, `names` | `increment.breakout.estimates` |
| `breakout.retention.encouragement` | `CapabilityError` | `fn_name`, `names` | `increment.breakout.estimates` |
| `breakout.retention.unbounded` | `InvalidRequestError` | `fn_name`, `names`, `remedy`, `supported_view` | `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `breakout.rollout_cost_names` | `InvalidRequestError` | `known`, `unknown` | `increment.breakout.rollout` |
| `breakout.rollout_cost_rollout` | `InvalidRequestError` | `name`, `value` | `increment.breakout.rollout` |
| `breakout.run_breakout_correction` | `InvalidRequestError` | `correction` | `increment.breakout.estimates` |
| `breakout.run_breakout_dimension` | `InvalidRequestError` | `columns`, `dimension` | `increment.breakout.estimates` |
| `breakout.run_breakout_duplicate` | `InvalidRequestError` | `dimension`, `group_id`, `metric`, `value` | `increment.breakout.estimates` |
| `breakout.run_breakout_group` | `InvalidRequestError` | `missing` | `increment.breakout.estimates` |
| `breakout.run_breakout_methods` | `InvalidRequestError` | — | `increment.breakout.estimates` |
| `breakout.run_daily_lift` | `InvalidRequestError` | `correction` | `increment.breakout.estimates` |
| `breakout.run_daily_lift_dimension_found` | `InvalidRequestError` | `columns`, `dimension` | `increment.breakout.estimates` |
| `breakout.run_daily_lift_ds_column_found` | `InvalidRequestError` | `columns` | `increment.breakout.estimates` |
| `breakout.run_daily_lift_metric_declared` | `InvalidRequestError` | `declared`, `unknown` | `increment.breakout.estimates` |
| `breakout.run_daily_lift_plan_missing_procedures` | `InvalidRequestError` | `missing_procedures` | `increment.breakout.estimates` |
| `breakout.segment.lift_none_excluded` | `InvalidRequestError` | — | `increment.breakout.heterogeneity` |
| `breakout.segment.point_unavailable` | `InvalidRequestError` | — | `increment.breakout.heterogeneity` |
| `breakout.segment_contrast_absolute_interval` | `InvalidRequestError` | — | `increment.breakout.heterogeneity` |
| `breakout.segment_contrast_dimension` | `InvalidRequestError` | `dimension_value_a`, `dimension_value_b`, `key_a`, `key_b` | `increment.breakout.heterogeneity` |
| `breakout.segment_contrast_estimate` | `InvalidRequestError` | — | `increment.breakout.heterogeneity` |
| `breakout.segment_contrast_expected` | `InvalidRequestError` | `label`, `row_count` | `increment.breakout.heterogeneity` |
| `breakout.segment_contrast_open_interval_variance` | `InvalidRequestError` | — | `increment.breakout.heterogeneity` |
| `breakout.segment_contrast_reference_fixed_horizon` | `InvalidRequestError` | `dimension_value`, `dof`, `inference`, `reference_df`, `reference_kind` | `increment.breakout.heterogeneity` |
| `breakout.segment_heterogeneity_rebuild` | `InvalidRequestError` | `alpha`, `alternative`, `dimension_value`, `dof`, `inference`, `metric`, `reference_df`, `reference_kind` | `increment.breakout.heterogeneity` |
| `breakout.tau_prior_scale` | `InvalidRequestError` | `tau_prior_scale` | `increment.breakout.heterogeneity` |
| `breakout.to_frame_infer` | `InvalidRequestError` | — | `increment.breakout.projection` |
| `breakout.to_frame_model` | `InvalidRequestError` | `inferred`, `model` | `increment.breakout.projection` |
| `breakout.unknown_estimand_supported` | `InvalidRequestError` | `supported`, `unknown_estimands` | `increment.breakout.estimates` |
| `cate.cate_supported_quantile` | `UnsupportedRequestError` | `metric` | `increment.cate` |
| `cate.cluster.covariate_conflict` | `InvalidRequestError` | `caller`, `column` | `increment.cate` |
| `cate.cluster.identity_unavailable` | `InvalidRequestError` | `caller`, `column` | `increment.cate` |
| `cate.cluster.intervention_grain_unavailable` | `InvalidRequestError` | `caller` | `increment.cate` |
| `cate.cluster.randomization_arm_purity` | `InvalidRequestError` | `caller`, `cluster`, `values` | `increment.cate` |
| `cate.contrasts_exactly_two` | `InvalidRequestError` | `arms`, `caller`, `control`, `metric`, `n_arms`, `others` | `increment.cate` |
| `cate.control_arm_present` | `InvalidRequestError` | `arms`, `control`, `metric` | `increment.cate` |
| `cate.does_support_ratio` | `UnsupportedRequestError` | `caller`, `metric` | `increment.cate` |
| `cate.identification.randomized_only` | `InvalidRequestError` | `caller`, `mechanism` | `increment.cate` |
| `cate.identification.unsupported_max_smd` | `InvalidRequestError` | `caller`, `max_smd` | `increment.cate` |
| `cate.identification.unsupported_mechanism` | `InvalidRequestError` | `caller`, `mechanism`, `supported` | `increment.cate` |
| `cate.identification.unsupported_missing_policy` | `InvalidRequestError` | `caller`, `missing` | `increment.cate` |
| `cate.metric_declared_source` | `InvalidRequestError` | `available`, `metric` | `increment.cate` |
| `compatibility.denominator` | `InvalidRequestError` | — | `increment.compatibility` |
| `compatibility.multiplier` | `InvalidRequestError` | — | `increment.compatibility` |
| `compatibility.support_unsupported` | `InvalidRequestError` | `support_type` | `increment.compatibility` |
| `contrast.decision` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `contrast.inference` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `contrast.metric` | `CapabilityError` | `_code`, `message`, `reason` | `increment.compatibility` |
| `conversion_inference.finite_sample.cuped` | `InvalidRequestError` | `adjusted_by`, `method` | `increment._finite_sample_refusals` |
| `conversion_inference.finite_sample.metric_type` | `InvalidRequestError` | `metric_type` | `increment._finite_sample_refusals` |
| `dashboard.invalid_config` | `InvalidRequestError` | `reason` | `increment.dashboard._data` |
| `dashboard.invalid_snapshot` | `InvalidRequestError` | `reason` | `increment.dashboard._data` |
| `dashboard.invalid_theme` | `InvalidRequestError` | `field`, `reason` | `increment.dashboard._theme` |
| `dashboard.invalid_view` | `InvalidRequestError` | `reason` | `increment.dashboard._data` |
| `dashboard.population_unavailable` | `InvalidRequestError` | `population` | `increment.dashboard._data` |
| `dashboard.unsupported_experiment` | `CapabilityError` | `reason` | `increment.dashboard._data` |
| `decision.absolute_arm.procedures_fixed_inference` | `InvalidRequestError` | — | `increment.decision` |
| `decision.arm_decision.one_sided_alpha` | `InvalidRequestError` | — | `increment.decision`, `increment.estimation.decision_types` |
| `decision.arm_planning.scalar_power_unsupported_family` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `decision.arm_planning.uncorrected_family_size` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `decision.compiled_decision.procedure_mapping_key` | `InvalidRequestError` | `key`, `metric` | `increment.decision` |
| `decision.contrast_context.duplicate_metrics` | `InvalidRequestError` | `metrics` | `increment.decision` |
| `decision.contrast_context.procedure_metrics` | `InvalidRequestError` | `missing`, `undeclared` | `increment.decision` |
| `decision.contrast_context.reference_mismatch` | `InvalidRequestError` | `metrics` | `increment.decision` |
| `decision.contrast_decision.reference_metric` | `InvalidRequestError` | — | `increment.estimation.decision_types` |
| `decision.family.axes_contain_duplicates` | `InvalidRequestError` | `axes` | `increment.estimation.arm_contract` |
| `decision.family.axes_contain_unknown` | `InvalidRequestError` | `unknown` | `increment.estimation.arm_contract` |
| `decision.family.axes_use_canonical` | `InvalidRequestError` | `allowed`, `axes` | `increment.estimation.arm_contract` |
| `decision.family.nofamily_membership_mark` | `InvalidRequestError` | — | `increment.decision` |
| `decision.family.policy_least_one` | `InvalidRequestError` | `kind` | `increment.estimation.arm_contract` |
| `decision.family.uncorrected_policy_empty` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `decision.multiplicity.bh` | `InvalidRequestError` | `correction` | `increment._multiplicity` |
| `decision.multiplicity.guarantee_mismatch` | `InvalidRequestError` | `correction`, `guarantee` | `increment._multiplicity` |
| `decision.multiplicity.validate_policy` | `InvalidRequestError` | `correction` | `increment._multiplicity` |
| `decision.p_value_evidence.finite_unit_interval` | `InvalidRequestError` | `p_value` | `increment.estimation.decision_types` |
| `decision.relative_arm.null_lift_greater` | `InvalidRequestError` | — | `increment.decision` |
| `decision.relative_decision.one_sided_nominal` | `InvalidRequestError` | — | `increment.estimation.arm_contract` |
| `definition.active.metric_window_days` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.adjustment_set.adjustmentset_covariates_contains` | `InvalidRequestError` | `dups` | `increment.semantics.design` |
| `definition.already_canonical_json` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.analysis.appears_both_metric` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.analysis.appears_more_once` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact.context_format_disagrees` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact.context_sha256_does` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact.trigger_metric_scope` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact.trigger_timestamp_timezone` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_datetimes_timezone` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_extension.definition_sha256_does` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_extension.source_provenance_sha256` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_extension.unsupported_version` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_json_values` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_relation.primary_key_contain` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_relation.primary_key_members` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.artifact_relation.row_count_integer` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.assignment.carryover_order_range` | `InvalidRequestError` | `carryover_order`, `observation_steps` | `increment.semantics.assignment` |
| `definition.assignment.carryover_order_type` | `InvalidRequestError` | — | `increment.semantics.assignment` |
| `definition.assignment.reject_bool` | `InvalidRequestError` | — | `increment.semantics.assignment`, `increment.semantics.design` |
| `definition.assignment_populations_contain` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.assignment_populations_include` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.base_relations.exposures_primary_key` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.base_relations.exposures_relation_use` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.base_relations.measure_stats_primary` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.base_relations.measure_stats_relation` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.base_relations.share_artifact_id` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.base_relations.share_generation_id` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.breakout.ambiguous_source` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.check_observational.covariate_could` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.check_observational.covariate_property_found` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.check_observational.covariate_source_found` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.check_observational.covariate_unit_but` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.contain_control_characters` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.cuped.ambiguous_covariate_source` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.cuped_preperiod.window_bounds_integers` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.cuped_preperiod.window_satisfy_window` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.day_boundary_accepts` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.duplicate_property_name` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.source_declares_no` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.source_no_validity` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.source_versioned_validity` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.validity_mutually_exclusive_encoding` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.validity_ranges_need` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.validity_requires_one_encoding` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.dim.validity_valid_from` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.duplicate_scalar` | `DefinitionError` | `field`, `files`, `path`, `route` | `increment.semantics.loader` |
| `definition.duplicates` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.encode_json_object` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.encouragement_uptake.window_days_integer` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.expected_datetime_bool` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.expected_iso_8601_epoch` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.expected_iso_8601_numeric_string` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.binding_margin_margin` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.cluster_combined_n` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.cluster_names_same` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.covariate_binding_exactly_one` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.covariate_measure_needs_preperiod` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.covariate_measure_numeric` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.covariate_name_reserved` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.covariate_names_unique` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.covariate_source_requires_property` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.design_tuning_key_not_yaml` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.end_before_start` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.intervention_grain_without_cluster` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.metric_cuped_linear_needs_covariates` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.metric_declares_cuped` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.observation_end_before_end` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.observation_end_requires_end` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.experiment.timezone_aware_but` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.exposure.filters_apply_fact` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.exposure.needs_exactly_one` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.exposure.sql_set_but` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.fact.source_declares_propert` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.fact.source_points_propert` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.fact.source_renames_its` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.fact.source_renames_propert` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.fact_source.missing_columns` | `DefinitionError` | `fact_source`, `missing` | `increment.query.fact_resolution` |
| `definition.filter.op_between_comparable` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.filter.op_between_values` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.filter.op_exactly_value` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.filter.op_exactly_values` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.filter.op_least_value` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.filter.values_finite` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.inference.always_valid_metric_type` | `DefinitionError` | `metric`, `metric_type` | `increment.sequential_source` |
| `definition.inference.baseline_rate_domain` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.inference.baseline_rate_route` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.invalid` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.json` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.lowercase_64_hex` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.method.covariates` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.method.methodspec_name_cuped` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_base.margin_margin_abs` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_base.rollout_cost_preferred` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_both_relative` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_margin_abs_requires_direction` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_margin_abs_requires_non_neutral_direction` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_margin_implies_unrepresentable_null_lift` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_margin_non` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_margin_requires_direction` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.metric_unknown_alternative` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.migration.guardrails` | `DefinitionError` | `experiment`, `path` | `increment.semantics.loader` |
| `definition.migration.metrics` | `DefinitionError` | `experiment`, `path` | `increment.semantics.loader` |
| `definition.models.reject_bool` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.multiplicity.bh_q_only_valid_for_bh` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.multiplicity.bh_requires_q` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.non_empty_string` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.observational.ambiguous_covariate_source` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.path.missing` | `DefinitionError` | `path` | `increment.semantics.loader` |
| `definition.quantile.metric_use_avg` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.ratio.metric_numerator_denominator` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.ratio.metric_use_avg` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.ratio_metric.numerator_denominator_differ` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.relation_bind_manifest` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.relation_primary_key` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.relation_role` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.resolve_breakout.experiment_property_could` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.resolve_breakout.experiment_property_found` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.resolve_breakout.experiment_property_references` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.resolve_breakout.experiment_unit_but` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.retention.metric_window_days` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.retention.threshold_days_lower_bound_non_negative` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.retention.threshold_days_non_negative` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.retention.threshold_days_upper_exceeds_lower` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.root` | `DefinitionError` | `expected`, `path`, `route`, `value_type` | `increment.semantics.loader` |
| `definition.section` | `DefinitionError` | `expected`, `field`, `path`, `route`, `value_type` | `increment.semantics.loader` |
| `definition.single_sql_identifier` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.site_volume.first_ds_last` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.site_volume.measure_keys_empty` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.site_volume.measure_keys_unique` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.site_volume.metric_names_empty` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.site_volume.metric_names_unique` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.source.sql_empty` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.sql` | `DefinitionError` | `error`, `error_type`, `label`, `path`, `route` | `increment.semantics.loader` |
| `definition.sql.dialect` | `InvalidRequestError` | `dialect`, `label`, `route` | `increment.semantics.loader` |
| `definition.sql.not_read_only` | `InvalidRequestError` | `construct`, `dialect`, `label`, `route`, `violation` | `increment.semantics.loader` |
| `definition.sql.parse` | `InvalidRequestError` | `column`, `dialect`, `error`, `label`, `line`, `route` | `increment.semantics.loader` |
| `definition.sql.statement_count` | `InvalidRequestError` | `dialect`, `empty`, `label`, `route`, `statements` | `increment.semantics.loader` |
| `definition.surrounding_whitespace` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.total.metric_use_avg` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.total.metric_window_days` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_cycle.aggregation` | `InvalidRequestError` | `reason` | `increment.semantics.unit_cycle` |
| `definition.unit_cycle.assignment` | `InvalidRequestError` | `reason` | `increment.semantics.unit_cycle` |
| `definition.unit_cycle.cycles` | `InvalidRequestError` | `reason` | `increment.semantics.unit_cycle` |
| `definition.unit_cycle.groups` | `InvalidRequestError` | `reason` | `increment.semantics.unit_cycle` |
| `definition.unit_cycle.nonempty` | `InvalidRequestError` | `reason` | `increment.semantics.unit_cycle` |
| `definition.unit_cycle.references` | `InvalidRequestError` | `reason` | `increment.semantics.unit_cycle` |
| `definition.unit_day.created_at_timezone` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_extension_absent` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_extensions_contain` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_extensions_use` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_first_ds` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_measure_keys` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_measures_use` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_metric_bindings` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_metric_bindings_canonical_order` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_metric_names` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.manifest_sha256_does` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.metric_references_undeclared` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.ratio_metric_references` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.site_volume_keys` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.site_volume_keys_absent_from_manifest` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.site_volume_request` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.trigger_measure_anchor` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.trigger_measure_keys` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.unit_day.trigger_measure_metric` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.utf` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.validation` | `DefinitionError` | `errors`, `path`, `route` | `increment.semantics.loader` |
| `definition.winsorization.finite` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.winsorization.least_one_bound` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.winsorization.lower_percentile_below` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.winsorization.lower_value_below` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.winsorization.set_percentile_fixed` | `DefinitionError` | `message` | `increment.semantics.loader`, `increment.semantics.models` |
| `definition.yaml` | `DefinitionError` | `column`, `error`, `line`, `path`, `route` | `increment.semantics.loader` |
| `design.allocation.invalid` | `InvalidRequestError` | `allocation`, `control_group` | `increment.semantics.design` |
| `design.allocation_scheme.incompatible` | `InvalidRequestError` | `allocation_scheme`, `design` | `increment.semantics.design`, `increment.semantics.models` |
| `estimation.absorption.absorption_interval_half` | `InvalidRequestError` | `crit`, `se` | `increment.estimation.absorption` |
| `estimation.absorption.all_moment_arrays` | `InvalidRequestError` | `lengths` | `increment.estimation.absorption` |
| `estimation.absorption.contain_exact_integer` | `InvalidRequestError` | `arr`, `name` | `increment.estimation.absorption` |
| `estimation.absorption.contains_non_finite` | `InvalidRequestError` | `name` | `increment.estimation.absorption` |
| `estimation.absorption.need_least_levels` | `InvalidRequestError` | `min_cell_n`, `min_levels`, `n_usable` | `increment.estimation.absorption` |
| `estimation.absorption.non_negative` | `InvalidRequestError` | `arr`, `name` | `increment.estimation.absorption` |
| `estimation.absorption.one_dimensional_shape` | `InvalidRequestError` | `arr`, `name` | `increment.estimation.absorption` |
| `estimation.absorption.pooling_one` | `InvalidRequestError` | `pooling`, `pooling_options` | `increment.estimation.absorption` |
| `estimation.adjust.estimate_ate_every` | `UnsupportedRequestError` | — | `increment.estimation.adjust`, `increment.readouts._common` |
| `estimation.adjust.method_name_unadjusted` | `InvalidRequestError` | — | `increment.estimation.adjust` |
| `estimation.adjust.names_metrics_source` | `InvalidRequestError` | `by_name`, `label`, `unknown` | `increment.estimation.adjust` |
| `estimation.adjust.prior.method_scale` | `InvalidRequestError` | — | `increment.estimation.adjust` |
| `estimation.adjust.prior.type` | `InvalidRequestError` | — | `increment.estimation.adjust`, `increment.estimation.encouragement` |
| `estimation.adjust.prior_interpreted_call` | `InvalidRequestError` | — | `increment.estimation.adjust` |
| `estimation.adjust.resolve_value_scales_null_abs_on_absolute_metric` | `InvalidRequestError` | `metric` | `increment.estimation.adjust` |
| `estimation.adjust.resolve_value_scales_null_lift_on_absolute_metric` | `InvalidRequestError` | `metric` | `increment.estimation.adjust` |
| `estimation.adjust.value_scale_names` | `InvalidRequestError` | `metric_type`, `name` | `increment.estimation.adjust` |
| `estimation.adjust.value_scale_relative` | `InvalidRequestError` | `name`, `requested` | `increment.estimation.adjust` |
| `estimation.adjust_common.control_group_found` | `InvalidRequestError` | `available`, `control_group`, `metric` | `increment.estimation._adjust.common` |
| `estimation.adjust_common.covariate_dtype` | `InvalidRequestError` | `contrast`, `covariate`, `dtype`, `method`, `metric` | `increment.estimation._adjust.common` |
| `estimation.adjust_common.fold_training_split` | `InvalidRequestError` | `contrast`, `j`, `method`, `metric` | `increment.estimation._adjust.common` |
| `estimation.adjust_common.fold_would_train` | `InvalidRequestError` | `contrast`, `j`, `method`, `metric` | `increment.estimation._adjust.common` |
| `estimation.adjust_common.needs_least_folds` | `InvalidRequestError` | `contrast`, `method`, `metric`, `n`, `needed` | `increment.estimation._adjust.common` |
| `estimation.adjust_common.statistically_indistinguishable_from` | `InvalidRequestError` | `control`, `denominator`, `estimator`, `label`, `metric`, `se`, `treatment` | `increment.estimation._adjust.common` |
| `estimation.adjust_common.supported_ratio_metric` | `UnsupportedRequestError` | `correction`, `family`, `method`, `metric`, `role` | `increment.estimation._adjust.common`, `increment.estimation.adjust` |
| `estimation.adjust_dml.dml_residualized_treatment` | `InvalidRequestError` | `contrast`, `metric_name` | `increment.estimation._adjust.dml` |
| `estimation.adjust_encoding.predict_called` | `InvalidRequestError` | — | `increment.estimation._adjust.encoding` |
| `estimation.adjust_learners.logistic_propensity.logisticpropensity_failed_converge` | `InvalidRequestError` | `res` | `increment.estimation._adjust.learners` |
| `estimation.adjust_learners.logistic_propensity.logisticpropensity_predict_called` | `InvalidRequestError` | — | `increment.estimation._adjust.learners` |
| `estimation.adjust_learners.ridge_outcome.ridgeoutcome_predict_called` | `InvalidRequestError` | — | `increment.estimation._adjust.learners` |
| `estimation.adjust_overlap.arm_cluster_but` | `InvalidRequestError` | `arm`, `clusters`, `k` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.arm_stratified_cross` | `InvalidRequestError` | `unique_arms` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.arm_unit_but` | `InvalidRequestError` | `arm`, `indices`, `k` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.arms_shape` | `InvalidRequestError` | `arms` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.arms_unit_ids` | `InvalidRequestError` | `arms`, `unit_ids` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.cluster_arm_needs_two` | `InvalidRequestError` | `k_c`, `k_t` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.cluster_ids_one` | `InvalidRequestError` | — | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.cluster_needs_two` | `InvalidRequestError` | `k` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.cross_fitting_needs` | `InvalidRequestError` | `k` | `increment.estimation._adjust.overlap` |
| `estimation.adjust_overlap.fold_ids_needs` | `InvalidRequestError` | `unit_ids` | `increment.estimation._adjust.overlap` |
| `estimation.armstats.arm_stats.cden3_complete_denominator` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.centered_sum_squares` | `InvalidRequestError` | `c2`, `what` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_family_some` | `InvalidRequestError` | `family`, `probe` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_inconsistent_winsorization` | `InvalidRequestError` | `field` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_needs_least` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_needs_one` | `InvalidRequestError` | `a_metric`, `a_study_id`, `head_metric`, `head_study_id` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_partitions_declaring` | `InvalidRequestError` | `roles` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_partitions_from` | `InvalidRequestError` | `labels` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.combine_uptake_family` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.cov_yd_needs_uptake_sum_cyd` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.cov_yden_needs_cyden` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.cross_field_finite` | `InvalidRequestError` | `name`, `value` | `increment.estimation.armstats`, `increment.estimation.sitewide` |
| `estimation.armstats.arm_stats.cxd_complete_uptake` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.cxden_complete_denominator` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.group_summary_moment` | `InvalidRequestError` | `group_id`, `metric`, `name`, `value` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.least_compute_metric` | `InvalidRequestError` | `group_id`, `metric`, `n`, `quantity` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.mean_d_needs_uptake_sum` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.mean_den_needs_ref_den` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.needs_covariate_cx2` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.needs_covariate_cxy` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.needs_covariate_ref` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.needs_covariate_uptake` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.needs_denominator_cross` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.needs_uptake_cyd` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.obsolete_armstats_fields` | `InvalidRequestError` | `obsolete` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.partial_family_missing` | `InvalidRequestError` | `label`, `missing`, `present` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.skew_den_inconsistent` | `InvalidRequestError` | `cden3`, `group_id`, `metric`, `n` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.skew_den_needs_cden3` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.skew_den_underflow` | `InvalidRequestError` | `cden2`, `group_id`, `metric` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.sum_d_outside` | `InvalidRequestError` | `group_id`, `metric`, `n`, `sum_d` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.sum_d_zero_but_masked_nonzero` | `InvalidRequestError` | `group_id`, `metric`, `nonzero` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.var_den_needs_cden2` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.winsorization_cap_counts` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.winsorization_lower_percentile` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.winsorization_metadata_finite` | `InvalidRequestError` | `name` | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.winsorization_metadata_winsor` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.winsorization_resolved_lower` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.winsorization_upper_percentile` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.x_role_declared` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.x_role_family` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.arm_stats.x_role_one` | `InvalidRequestError` | `x_role` | `increment.estimation.armstats` |
| `estimation.armstats.centered_raw_row` | `InvalidRequestError` | `unsupported` | `increment.estimation.armstats` |
| `estimation.armstats.centered_sum_squares` | `InvalidRequestError` | `value`, `variance_slack`, `what` | `increment.estimation.armstats` |
| `estimation.armstats.cross_moment_materially` | `InvalidRequestError` | `cross`, `floor`, `var_a`, `var_b`, `what` | `increment.estimation.armstats` |
| `estimation.armstats.cross_moment_violates` | `InvalidRequestError` | `bound`, `cross`, `excess`, `var_a`, `var_b`, `variance_slack`, `what` | `increment.estimation.armstats` |
| `estimation.armstats.independent_mean_contract` | `InvalidRequestError` | `reason` | `increment.estimation.armstats` |
| `estimation.armstats.score_stats.cluster_score_scale` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.score_stats.cluster_variance` | `InvalidRequestError` | — | `increment.estimation.armstats` |
| `estimation.armstats.score_stats.n_clusters_clustered` | `InvalidRequestError` | `n_clusters` | `increment.estimation.armstats` |
| `estimation.armstats.score_stats.score_scale` | `InvalidRequestError` | `field`, `value` | `increment.estimation.armstats` |
| `estimation.armstats.score_stats.se_needs_positive` | `InvalidRequestError` | `normalizer` | `increment.estimation.armstats` |
| `estimation.armstats.summary_stats.mean_finite` | `InvalidRequestError` | `mean` | `increment.estimation.armstats` |
| `estimation.armstats.summary_stats.var_finite` | `InvalidRequestError` | `var` | `increment.estimation.armstats` |
| `estimation.binomial.binary_provenance_required` | `InvalidRequestError` | `metric_type` | `increment.estimation.armstats` |
| `estimation.binomial.exact_counts_required` | `CapabilityError` | `group_id`, `metric` | `increment.estimation.armstats` |
| `estimation.binomial.finite_sample_arm_ceiling_exceeded` | `BinomialDataError` | `max_arm_size`, `n_c`, `n_t` | `increment.estimation.binomial_rr` |
| `estimation.binomial.finite_sample_unavailable` | `InvalidRequestError` | `metric`, `reason` | `increment.estimation.conversion_route` |
| `estimation.binomial.inconsistent_bernoulli_variance` | `BinomialDataError` | `cy2`, `expected_cy2`, `group_id`, `metric`, `n`, `tolerance` | `increment.estimation.armstats` |
| `estimation.binomial.independent_units_required` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.armstats` |
| `estimation.binomial.invalid_counts` | `BinomialDataError` | `n`, `x` | `increment.estimation.binomial_rr` |
| `estimation.binomial.reconstructed_counts_not_binary` | `BinomialDataError` | `group_id`, `metric`, `n`, `sum_y` | `increment.estimation.armstats` |
| `estimation.binomial.tail_unrepresentable` | `BinomialDataError` | — | `increment.estimation.binomial_rr` |
| `estimation.binomial.unknown_alternative` | `InvalidRequestError` | `alternative` | `increment.estimation.binomial_rr`, `increment.estimation.encouragement`, `increment.estimation.inference` |
| `estimation.cate.cate.covariate_missing_from` | `InvalidRequestError` | `names`, `transform` | `increment.estimation.cate` |
| `estimation.cate.cate.level_contradicts_alpha` | `InvalidRequestError` | `alpha`, `expected`, `level` | `increment.estimation.cate` |
| `estimation.cate.cluster_count` | `InvalidRequestError` | `n`, `n_clusters` | `increment.estimation.cate` |
| `estimation.cate.cluster_ids_shape` | `InvalidRequestError` | `n`, `shape` | `increment.estimation.cate` |
| `estimation.cate.cluster_weight` | `InvalidRequestError` | `weighting` | `increment.estimation.cate` |
| `estimation.cate.covariate.categorical_so_knots` | `InvalidRequestError` | `self` | `increment.estimation.cate` |
| `estimation.cate.covariate.needs_least_one` | `InvalidRequestError` | `knots`, `name` | `increment.estimation.cate` |
| `estimation.cate.covariate.needs_strictly_increasing` | `InvalidRequestError` | `knots`, `name` | `increment.estimation.cate` |
| `estimation.cate.covariate.non_finite_knots` | `InvalidRequestError` | `knots`, `name` | `increment.estimation.cate` |
| `estimation.cate.covariate.was_empty_knot` | `InvalidRequestError` | `self` | `increment.estimation.cate` |
| `estimation.cate.covariate_columns_rows` | `InvalidRequestError` | `n`, `rows` | `increment.estimation.cate` |
| `estimation.cate.covariate_declared_both` | `InvalidRequestError` | `kind`, `name`, `prior_kind` | `increment.estimation.cate` |
| `estimation.cate.covariate_declared_knots` | `InvalidRequestError` | `knots`, `name`, `prior_knots` | `increment.estimation.cate` |
| `estimation.cate.covariate_missing_from` | `InvalidRequestError` | `columns`, `name` | `increment.estimation.cate` |
| `estimation.cate.covariate_nulls` | `InvalidRequestError` | `name` | `increment.estimation.cate` |
| `estimation.cate.covariate_nulls_non` | `InvalidRequestError` | `name` | `increment.estimation.cate` |
| `estimation.cate.covariate_shape` | `InvalidRequestError` | `name`, `shape` | `increment.estimation.cate` |
| `estimation.cate.design.covariate_rows_expected` | `InvalidRequestError` | `n`, `name`, `size` | `increment.estimation.cate` |
| `estimation.cate.design.covariate_single_level` | `InvalidRequestError` | `level`, `name` | `increment.estimation.cate` |
| `estimation.cate.design.covariate_zero_variance` | `InvalidRequestError` | `name` | `increment.estimation.cate` |
| `estimation.cate.design_row_leverage` | `InvalidRequestError` | `culprits`, `leverage`, `row` | `increment.estimation.cate` |
| `estimation.cate.design_too_wide` | `InvalidRequestError` | `n`, `width` | `increment.estimation.cate` |
| `estimation.cate.each_arm_needs` | `InvalidRequestError` | `n_control`, `n_treated` | `increment.estimation.cate` |
| `estimation.cate.equal_weighting_without_cluster` | `InvalidRequestError` | — | `increment.estimation.cate` |
| `estimation.cate.inference_metadata` | `InvalidRequestError` | — | `increment.estimation.cate` |
| `estimation.cate.insufficient_clusters` | `InvalidRequestError` | `count` | `increment.estimation.cate` |
| `estimation.cate.intervention_grain` | `InvalidRequestError` | `grain` | `increment.estimation.cate` |
| `estimation.cate.intervention_grain_without_cluster` | `InvalidRequestError` | — | `increment.estimation.cate` |
| `estimation.cate.invalid_covariance` | `InvalidRequestError` | `reason` | `increment.estimation.cate` |
| `estimation.cate.names_entries_gram` | `InvalidRequestError` | `count`, `p` | `increment.estimation.cate` |
| `estimation.cate.outcome_constant_within` | `InvalidRequestError` | — | `increment.estimation.cate` |
| `estimation.cate.outcome_nulls_non` | `InvalidRequestError` | — | `increment.estimation.cate`, `increment.estimation.targeting` |
| `estimation.cate.outcome_shape` | `InvalidRequestError` | `shape` | `increment.estimation.cate` |
| `estimation.cate.protected_column_collinear` | `InvalidRequestError` | `conditional`, `diag`, `name` | `increment.estimation.cate` |
| `estimation.cate.rank_deficient` | `InvalidRequestError` | `dimension` | `increment.estimation.cate` |
| `estimation.cate.score_nonfinite` | `InvalidRequestError` | — | `increment.estimation.cate` |
| `estimation.cate.score_state` | `InvalidRequestError` | — | `increment.estimation.cate` |
| `estimation.cate.single_cluster_direction` | `InvalidRequestError` | `query` | `increment.estimation.cate` |
| `estimation.cate.singular_wald` | `InvalidRequestError` | `dimension` | `increment.estimation.cate` |
| `estimation.cate.thin_cells.one_hot_level_interacted_min` | `InvalidRequestError` | `n_c`, `n_t`, `name` | `increment.estimation.cate` |
| `estimation.cate.thin_cells.one_hot_level_singleton` | `InvalidRequestError` | `name`, `total` | `increment.estimation.cate` |
| `estimation.cate.treatment_binary` | `InvalidRequestError` | `values` | `increment.estimation.cate`, `increment.estimation.targeting` |
| `estimation.cate.treatment_shape_expected` | `InvalidRequestError` | `d_shape`, `y_shape` | `increment.estimation.cate` |
| `estimation.cate.uncertainty_unavailable` | `InvalidRequestError` | `query` | `increment.estimation.cate` |
| `estimation.cate.zz_square_gram` | `InvalidRequestError` | `shape` | `increment.estimation.cate` |
| `estimation.certified.decimal_arithmetic_capacity` | `InvalidRequestError` | `precision`, `signal`, `value`, `working_precision` | `increment.estimation._certified` |
| `estimation.certified.decimal_input_exponent_capacity` | `InvalidRequestError` | `precision`, `value`, `working_precision` | `increment.estimation._certified` |
| `estimation.certified.decimal_log_exponent_capacity` | `InvalidRequestError` | `precision`, `value`, `working_precision` | `increment.estimation._certified` |
| `estimation.certified.decimal_precision_capacity` | `InvalidRequestError` | `max_precision`, `precision`, `working_precision` | `increment.estimation._certified` |
| `estimation.certified.inexact_value` | `InvalidRequestError` | `value_type` | `increment.estimation._certified`, `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood` |
| `estimation.certified.invalid_count` | `InvalidRequestError` | `count` | `increment.estimation._certified`, `increment.estimation._sequential_likelihood` |
| `estimation.certified.invalid_precision` | `InvalidRequestError` | `precision` | `increment.estimation._certified` |
| `estimation.certified.nonpositive_gamma_value` | `InvalidRequestError` | `value` | `increment.estimation._certified` |
| `estimation.certified.nonpositive_log_domain` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation._certified` |
| `estimation.certified.nonpositive_rising_value` | `InvalidRequestError` | `value` | `increment.estimation._certified` |
| `estimation.certified.nonpositive_root_width` | `InvalidRequestError` | `max_width` | `increment.estimation._certified`, `increment.estimation._sequential_inversion` |
| `estimation.certified.reversed_interval` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation._certified` |
| `estimation.certified.reversed_root_domain` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation._certified` |
| `estimation.certified.zero_divisor_interval` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation._certified` |
| `estimation.certified.zero_polynomial` | `InvalidRequestError` | `coefficients` | `increment.estimation._certified` |
| `estimation.contrast.assignment_metadata` | `InvalidRequestError` | `expected_grain`, `independence_grain`, `randomization_law` | `increment.estimation.contrast_results` |
| `estimation.contrast.contrast_interval_half` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition.contain_least_one` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition.ct_counts` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition.cycles_mapping` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition.cycles_positive_integers` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition.unit_deltas_cycles` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition.unit_deltas_finite` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition_centered` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partition_does` | `InvalidRequestError` | `mismatch` | `increment.estimation.contrast` |
| `estimation.contrast.contrast_partitions_disjoint` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_reduction_delta_mean_finite` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_reduction_least` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats.control_group_treatment` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats_m2` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats_metric` | `InvalidRequestError` | `procedure`, `stats` | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats_n` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats_reconstructed` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats_reference` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.contrast_stats_standard` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.one_sided_alpha` | `InvalidRequestError` | `alpha_eff`, `procedure` | `increment.estimation.contrast` |
| `estimation.contrast.order_counts` | `InvalidRequestError` | — | `increment.estimation.contrast_results` |
| `estimation.contrast.procedure_alpha` | `InvalidRequestError` | `procedure` | `increment.estimation.contrast` |
| `estimation.contrast.procedure_contrastdecisionprocedure` | `InvalidRequestError` | `procedure` | `increment.estimation.contrast` |
| `estimation.contrast.reference_metadata` | `InvalidRequestError` | — | `increment.estimation.contrast_results` |
| `estimation.contrast.replication_counts` | `InvalidRequestError` | — | `increment.estimation.contrast_results` |
| `estimation.contrast.retained_window` | `InvalidRequestError` | `carryover_order`, `observation_steps`, `retained_steps` | `increment.estimation.contrast_results` |
| `estimation.contrast.shared_roster_size_mismatch` | `InvalidRequestError` | `roster_sizes` | `increment.estimation.contrast` |
| `estimation.contrast.stats_contraststats` | `InvalidRequestError` | `stats` | `increment.estimation.contrast` |
| `estimation.contrast.switchback_block_inference` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.contrast.switchback_unit_inference` | `InvalidRequestError` | — | `increment.estimation.contrast` |
| `estimation.crossfit.candidate_contribution_nonfinite` | `InvalidRequestError` | `fold`, `index`, `n_bad`, `n_total`, `route` | `increment.estimation.crossfit` |
| `estimation.crossfit.candidate_returned_contribution` | `InvalidRequestError` | `index`, `n_val`, `shape` | `increment.estimation.crossfit` |
| `estimation.crossfit.cluster_atomic_shape` | `InvalidRequestError` | `assignment`, `cluster_ids` | `increment.estimation.crossfit` |
| `estimation.crossfit.cluster_ids_shape_expected` | `InvalidRequestError` | `cluster_ids`, `unit_ids` | `increment.estimation.crossfit` |
| `estimation.crossfit.cluster_split` | `InvalidRequestError` | `cluster`, `values` | `increment.estimation.crossfit` |
| `estimation.crossfit.fold_assignments_unit_ids_shape` | `InvalidRequestError` | `unit_ids` | `increment.estimation._adjust.overlap`, `increment.estimation.crossfit` |
| `estimation.crossfit.identity_collision` | `InvalidRequestError` | `canonical`, `first`, `second`, `what` | `increment._identity` |
| `estimation.crossfit.identity_missing` | `InvalidRequestError` | `index`, `what` | `increment._identity` |
| `estimation.crossfit.insufficient_clusters` | `InvalidRequestError` | `n_clusters`, `n_folds` | `increment.estimation.crossfit` |
| `estimation.crossfit.n_candidates_least` | `InvalidRequestError` | `n_candidates` | `increment.estimation.crossfit` |
| `estimation.crossfit.n_folds_least` | `InvalidRequestError` | `n_folds` | `increment.estimation.crossfit`, `increment.estimation.targeting` |
| `estimation.crossfit.selection_needs_least` | `InvalidRequestError` | `n_labels` | `increment.estimation.crossfit` |
| `estimation.crossfit.stratify_shape_expected` | `InvalidRequestError` | `n`, `s` | `increment.estimation.crossfit` |
| `estimation.crossfit.test_size` | `InvalidRequestError` | `test_size` | `increment.estimation.crossfit` |
| `estimation.cuped.arm_no_covariate` | `InvalidRequestError` | `arm_group`, `arm_metric` | `increment.estimation.cuped.scalar` |
| `estimation.cuped.covariate_zero_variance` | `InvalidRequestError` | `weighted_var_x` | `increment.estimation.cuped.scalar` |
| `estimation.cuped.cuped_adjusted_variance` | `InvalidRequestError` | `arm_group`, `arm_metric`, `cross_label`, `naive_var`, `pair` | `increment.estimation.cuped.scalar` |
| `estimation.cuped.pooled_theta_least` | `InvalidRequestError` | — | `increment.estimation.cuped.scalar` |
| `estimation.cuped.ratio_arm_covariate_role` | `InvalidRequestError` | `arm_group`, `arm_metric`, `x_role` | `increment.estimation.cuped.scalar` |
| `estimation.cuped.ratio_arm_no_denominator_cross` | `InvalidRequestError` | `arm_group`, `arm_metric` | `increment.estimation.cuped.scalar` |
| `estimation.cuped_linear.arm_counts` | `InvalidRequestError` | `counts` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.covariate_input` | `InvalidRequestError` | `kinds`, `values` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.covariate_kind` | `InvalidRequestError` | `kind`, `name` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.incompatible_evidence` | `InvalidRequestError` | `reason` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.input_shape` | `InvalidRequestError` | `n`, `name`, `shape` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.invalid_alpha` | `InvalidRequestError` | `alpha` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.invalid_arms` | `InvalidRequestError` | `arm_order`, `control`, `observed` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.invalid_evidence` | `InvalidRequestError` | `reason` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.invalid_scale` | `InvalidRequestError` | `scale` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.level_support` | `InvalidRequestError` | `column`, `counts`, `level` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.leverage` | `InvalidRequestError` | `leverage`, `row` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.merge_precision` | `InvalidRequestError` | `arm`, `bound`, `component`, `mean`, `route` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.missing_refused` | `InvalidRequestError` | `name` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.nonfinite` | `InvalidRequestError` | `name`, `row` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.nonfinite_projection` | `InvalidRequestError` | `dof`, `se`, `value` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.nonpositive_component` | `InvalidRequestError` | `arm`, `family`, `value` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.outcome_components` | `InvalidRequestError` | — | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.projection_arm` | `InvalidRequestError` | `arm`, `control` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.projection_kind` | `InvalidRequestError` | `available`, `requested` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.projection_unavailable` | `InvalidRequestError` | `arm`, `control`, `kind` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.rank_deficiency` | `InvalidRequestError` | `columns`, `partners` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.residual_df` | `InvalidRequestError` | `df` | `increment.estimation.cuped._numeric` |
| `estimation.cuped_linear.support_region` | `InvalidRequestError` | `region`, `violations` | `increment.estimation.cuped._numeric` |
| `estimation.diagnostics.allocation_band_ds` | `InvalidRequestError` | `a`, `b`, `credible_level`, `ds`, `group_id`, `lower`, `upper` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.allocation_posterior_bands` | `InvalidRequestError` | `experiment_ids` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.alpha` | `InvalidRequestError` | `alpha` | `increment.breakout.estimates`, `increment.estimation.absorption`, `increment.estimation.cate`, `increment.estimation.diagnostics`, `increment.estimation.encouragement`, `increment.estimation.inference`, `increment.estimation.quantile`, `increment.estimation.targeting`, `increment.power.core` |
| `estimation.diagnostics.always_srm_predeclared` | `InvalidRequestError` | `route` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.counts_missing_columns` | `InvalidRequestError` | `missing` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.counts_non_negative` | `InvalidRequestError` | `negative` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.counts_whole_unit` | `InvalidRequestError` | `fractional` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.credible_level` | `InvalidRequestError` | `credible_level` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.credible_level_too` | `InvalidRequestError` | `credible_level` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.duplicate_ds_group` | `InvalidRequestError` | `ds_group` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.every_expected_share_finite` | `InvalidRequestError` | `nonfinite` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.every_expected_share_positive` | `InvalidRequestError` | `expected` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.expected_allocation_contain` | `InvalidRequestError` | `accounting` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.expected_keys_do` | `InvalidRequestError` | `counts_keys`, `expected_keys` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.expected_share_normalization` | `InvalidRequestError` | `invalid_normalized` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.inference_always_valid` | `InvalidRequestError` | `inference` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.n_cumulative_non` | `InvalidRequestError` | `ds`, `group_id`, `n_raw` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.n_cumulative_whole` | `InvalidRequestError` | `ds`, `group_id`, `n_raw` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.need_least_groups` | `InvalidRequestError` | — | `increment.estimation.diagnostics` |
| `estimation.diagnostics.predeclared_allocation_support` | `InvalidRequestError` | — | `increment.estimation.diagnostics` |
| `estimation.diagnostics.prior_alpha_finite` | `InvalidRequestError` | `prior_a` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.prior_beta_finite` | `InvalidRequestError` | `prior_b` | `increment.estimation.diagnostics` |
| `estimation.diagnostics.total_observed_count` | `InvalidRequestError` | — | `increment.estimation.diagnostics` |
| `estimation.encouragement.alpha_eff_too` | `InvalidRequestError` | — | `increment.estimation.encouragement` |
| `estimation.encouragement.cluster.arm_needs_two` | `InvalidRequestError` | `cluster`, `k_c`, `k_t`, `metric` | `increment.estimation.encouragement` |
| `estimation.encouragement.cluster.ratio` | `CapabilityError` | `cluster`, `names` | `increment.estimation.encouragement` |
| `estimation.encouragement.control.missing` | `InvalidRequestError` | `control_group` | `increment.estimation.encouragement` |
| `estimation.encouragement.cuped_adjusted_late` | `InvalidRequestError` | `arm_group`, `arm_metric` | `increment.estimation.encouragement` |
| `estimation.encouragement.dof_positive_reference` | `InvalidRequestError` | `dof` | `increment.estimation.encouragement` |
| `estimation.encouragement.late.ratio` | `UnsupportedRequestError` | `metric` | `increment.estimation.encouragement` |
| `estimation.encouragement.needs_uptake_cyd` | `InvalidRequestError` | — | `increment.estimation.encouragement` |
| `estimation.encouragement.negative_variance` | `InvalidRequestError` | `slack`, `value`, `what` | `increment.estimation.encouragement` |
| `estimation.encouragement.one_sided_encouragement` | `InvalidRequestError` | `control_uptake_total`, `metric` | `increment.estimation.encouragement` |
| `estimation.encouragement.prior_cluster_robust` | `InvalidRequestError` | — | `increment.estimation.encouragement` |
| `estimation.encouragement.prior_inference_are` | `InvalidRequestError` | — | `increment.estimation.encouragement` |
| `estimation.encouragement.sequential_inference_cluster` | `InvalidRequestError` | — | `increment.estimation.encouragement` |
| `estimation.encouragement.unknown_estimand_supported` | `InvalidRequestError` | `estimands`, `unknown` | `increment.estimation.encouragement`, `increment.readouts._common` |
| `estimation.engine.arm.duplicate_rows` | `InvalidRequestError` | `keys` | `increment.estimation.engine` |
| `estimation.engine.arm.invalid_count` | `InvalidRequestError` | `value`, `what` | `increment.estimation.engine` |
| `estimation.engine.carries_format_moments` | `InvalidRequestError` | `what` | `increment.estimation.engine` |
| `estimation.engine.carries_unsupported_weighted` | `InvalidRequestError` | `unsupported`, `what` | `increment.estimation.engine` |
| `estimation.engine.cluster.denominator` | `CapabilityError` | `cluster`, `control_group`, `metric`, `metric_type`, `treatment_group` | `increment.estimation.engine` |
| `estimation.engine.control_group_found` | `InvalidRequestError` | `control_group`, `known_groups` | `increment.estimation.engine` |
| `estimation.engine.cuped_linear_cluster` | `UnsupportedRequestError` | `cluster` | `increment.estimation.engine` |
| `estimation.engine.cuped_linear_encouragement` | `UnsupportedRequestError` | `metric` | `increment.estimation.engine` |
| `estimation.engine.cuped_linear_observational` | `UnsupportedRequestError` | `metric` | `increment.estimation.engine` |
| `estimation.engine.cuped_linear_sequential` | `UnsupportedRequestError` | — | `increment.estimation.engine` |
| `estimation.engine.cuped_linear_unit_evidence` | `UnsupportedRequestError` | `metric` | `increment.estimation.engine` |
| `estimation.engine.cuped_linear_winsorization` | `UnsupportedRequestError` | `metric` | `increment.estimation.engine` |
| `estimation.engine.group_summary_missing` | `InvalidRequestError` | `missing` | `increment.estimation.engine` |
| `estimation.engine.group_summary_row` | `InvalidRequestError` | `missing` | `increment.estimation.engine` |
| `estimation.engine.method.name_iptw_does` | `InvalidRequestError` | — | `increment.declarations` |
| `estimation.engine.method.name_without_variance` | `InvalidRequestError` | `self` | `increment.declarations` |
| `estimation.engine.method_covariates` | `InvalidRequestError` | `method` | `increment.declarations` |
| `estimation.engine.method_name_observational` | `InvalidRequestError` | `m` | `increment.estimation.engine` |
| `estimation.engine.method_names_unique` | `InvalidRequestError` | `caller`, `duplicates` | `increment.estimation.engine` |
| `estimation.engine.metric_found_group` | `InvalidRequestError` | `metric_types`, `unknown` | `increment.estimation.engine` |
| `estimation.engine.ratio.cuped` | `UnsupportedRequestError` | `experiment`, `metric` | `increment.estimation.engine` |
| `estimation.engine.ratio.negative_variance` | `InvalidRequestError` | `reason` | `increment.estimation.engine` |
| `estimation.engine.winsorization_metadata_differs` | `InvalidRequestError` | `field` | `increment.estimation.engine` |
| `estimation.engine.winsorization_metadata_present` | `InvalidRequestError` | — | `increment.estimation.engine` |
| `estimation.engine.x_role_declared` | `InvalidRequestError` | — | `increment.estimation.engine` |
| `estimation.estimator.adjustment.callback_contract` | `InvalidRequestError` | `estimator_id` | `increment.declarations` |
| `estimation.estimator.implementation.missing_ref` | `InvalidRequestError` | `estimator_id` | `increment.declarations` |
| `estimation.estimator.need.unsupported` | `InvalidRequestError` | `estimator_id`, `need_type` | `increment.estimators.registry` |
| `estimation.estimator.ref.already_registered` | `InvalidRequestError` | `estimator_id` | `increment.declarations` |
| `estimation.estimator.ref.invalid_id` | `InvalidRequestError` | `estimator_id` | `increment.declarations` |
| `estimation.estimator.ref.observational_needs` | `InvalidRequestError` | `estimator_id` | `increment.declarations` |
| `estimation.estimator.ref.raw_rows_unsupported` | `InvalidRequestError` | `estimator_id` | `increment.declarations` |
| `estimation.family.bh_select_q_finite` | `InvalidRequestError` | `q` | `increment.estimation.family` |
| `estimation.family.e_values_finite` | `InvalidRequestError` | `e`, `i` | `increment.estimation.family` |
| `estimation.family.exploratory_construction` | `UnsupportedRequestError` | `constructions`, `rows` | `increment.estimation.family` |
| `estimation.family.exploratory_non_decision` | `InvalidRequestError` | `rows` | `increment.estimation.family` |
| `estimation.family.exploratory_pre_corrected` | `InvalidRequestError` | `rows` | `increment.estimation.family` |
| `estimation.family.exploratory_row` | `InvalidRequestError` | `rows` | `increment.estimation.family` |
| `estimation.family.exploratory_sequential` | `UnsupportedRequestError` | `rows` | `increment.estimation.family` |
| `estimation.family.nominal_alpha_finite` | `InvalidRequestError` | `nominal_alpha` | `increment.estimation.family` |
| `estimation.family.p_values_finite` | `InvalidRequestError` | `i`, `p` | `increment.estimation.family` |
| `estimation.health.check.available_conclusion` | `InvalidRequestError` | — | `increment.estimation.health` |
| `estimation.health.check.unavailable_conclusion` | `InvalidRequestError` | — | `increment.estimation.health` |
| `estimation.health.check.unavailable_reason` | `InvalidRequestError` | — | `increment.estimation.health` |
| `estimation.health.experiment.checks_type` | `InvalidRequestError` | `received` | `increment.estimation.health` |
| `estimation.health.experiment.stable_order` | `InvalidRequestError` | — | `increment.estimation.health` |
| `estimation.inference.abs_dof_requires_dof` | `InvalidRequestError` | — | `increment.estimation.inference` |
| `estimation.inference.arm_ns_each` | `InvalidRequestError` | `arm_ns` | `increment.estimation.inference` |
| `estimation.inference.degenerate_data_zero` | `InvalidRequestError` | — | `increment.estimation.inference` |
| `estimation.inference.infer_ate_null_abs_finite` | `InvalidRequestError` | `null_abs` | `increment.estimation.inference` |
| `estimation.inference.infer_ate_null_abs_on_absolute_metric` | `InvalidRequestError` | `name` | `increment.estimation.inference` |
| `estimation.inference.infer_ate_null_lift_finite` | `InvalidRequestError` | `null_lift` | `increment.estimation.inference` |
| `estimation.inference.infer_ate_null_lift_on_absolute_metric` | `InvalidRequestError` | `name` | `increment.estimation.inference` |
| `estimation.inference.infer_lift_abs_diff_finite` | `InvalidRequestError` | `abs_diff` | `increment.estimation.inference` |
| `estimation.inference.infer_lift_abs_se_finite` | `InvalidRequestError` | `abs_se` | `increment.estimation.inference` |
| `estimation.inference.infer_lift_abs_se_positive` | `InvalidRequestError` | `abs_se` | `increment.estimation.inference` |
| `estimation.inference.joint_reference_invalid` | `InvalidRequestError` | `reason` | `increment.estimation.inference` |
| `estimation.inference.null_lift_log` | `InvalidRequestError` | `null_lift` | `increment.estimation.inference` |
| `estimation.inference.one_sided_alpha_doubles` | `InvalidRequestError` | `alpha`, `alpha_eff` | `increment.estimation.encouragement`, `increment.estimation.inference` |
| `estimation.inference.prior_excludes_cluster_robust_t` | `InvalidRequestError` | — | `increment.estimation.inference` |
| `estimation.inference.prior_excludes_joint_reference` | `InvalidRequestError` | — | `increment.estimation.inference` |
| `estimation.inference.prior_excludes_welch_reference` | `InvalidRequestError` | `arm_ns` | `increment.estimation.inference` |
| `estimation.inference.unknown_value_scale` | `InvalidRequestError` | `value_scale` | `increment.estimation.inference` |
| `estimation.meta.alpha_strictly_between` | `InvalidRequestError` | `alpha` | `increment.estimation.meta` |
| `estimation.meta.alpha_too_small` | `InvalidRequestError` | — | `increment.breakout.estimates`, `increment.estimation.meta`, `increment.estimation.quantile` |
| `estimation.meta.est_a_est` | `InvalidRequestError` | — | `increment.estimation.meta` |
| `estimation.meta.est_contains_non` | `InvalidRequestError` | — | `increment.estimation.meta` |
| `estimation.meta.est_var_same` | `InvalidRequestError` | `est_arr`, `var_arr` | `increment.estimation.meta` |
| `estimation.meta.est_var_shape` | `InvalidRequestError` | `est_arr` | `increment.estimation.meta` |
| `estimation.meta.hksj_pooled_variance` | `InvalidRequestError` | — | `increment.estimation.meta` |
| `estimation.meta.marginalized_segment.finite` | `InvalidRequestError` | `arr`, `name` | `increment.estimation.meta` |
| `estimation.meta.marginalized_segment.lb_exceed_ub` | `InvalidRequestError` | `lb`, `ub` | `increment.estimation.meta` |
| `estimation.meta.marginalized_segment.length_shape` | `InvalidRequestError` | `arr_shape`, `k`, `name` | `increment.estimation.meta` |
| `estimation.meta.marginalized_segment.level_contradicts_alpha` | `InvalidRequestError` | `alpha`, `expected_level`, `level` | `increment.estimation.meta` |
| `estimation.meta.need_least_estimable` | `InvalidRequestError` | `k` | `increment.estimation.meta` |
| `estimation.meta.posterior_integration_unresolved` | `InvalidRequestError` | `integration_error`, `k`, `node_budget`, `support`, `tau_prior_scale` | `increment.breakout.heterogeneity`, `increment.estimation.meta` |
| `estimation.meta.tau2_finite` | `InvalidRequestError` | `tau2` | `increment.estimation.meta` |
| `estimation.meta.tau_prior_scale` | `InvalidRequestError` | `tau_prior_scale` | `increment.estimation.meta` |
| `estimation.meta.var_a_finite` | `InvalidRequestError` | `var_a` | `increment.estimation.meta` |
| `estimation.meta.var_b_finite` | `InvalidRequestError` | `var_b` | `increment.estimation.meta` |
| `estimation.meta.var_finite_strictly` | `InvalidRequestError` | — | `increment.estimation.meta`, `increment.power.core` |
| `estimation.priors.mixture.every_sigma` | `InvalidRequestError` | — | `increment.estimation.priors` |
| `estimation.priors.mixture.every_weight` | `InvalidRequestError` | — | `increment.estimation.priors` |
| `estimation.priors.mixture.every_weight_mean` | `InvalidRequestError` | `means`, `sigmas`, `weights` | `increment.estimation.priors` |
| `estimation.priors.mixture.prior_needs_least` | `InvalidRequestError` | — | `increment.estimation.priors` |
| `estimation.priors.mixture.weights_means_sigmas` | `InvalidRequestError` | `k`, `means`, `sigmas` | `increment.estimation.priors` |
| `estimation.priors.mixture.weights_sum` | `InvalidRequestError` | `total` | `increment.estimation.priors` |
| `estimation.priors.mixture_posterior.expected_negative_part_scale` | `InvalidRequestError` | `scale` | `increment.estimation.inference`, `increment.estimation.priors` |
| `estimation.priors.mixture_posterior.expected_positive_part_scale` | `InvalidRequestError` | `scale` | `increment.estimation.inference`, `increment.estimation.priors` |
| `estimation.priors.mixture_posterior.probability` | `InvalidRequestError` | `probability` | `increment.estimation.priors` |
| `estimation.priors.mixture_posterior.tail_probability` | `InvalidRequestError` | `tail_probability` | `increment.estimation.priors` |
| `estimation.priors.standard_error` | `InvalidRequestError` | `standard_error` | `increment.estimation.priors` |
| `estimation.priors.student_t.nu_exceeds_200` | `InvalidRequestError` | `nu` | `increment.estimation.priors` |
| `estimation.quantile.control_group_found` | `InvalidRequestError` | `arms`, `control_group` | `increment.estimation.quantile` |
| `estimation.quantile.degenerate_spread_order` | `InvalidRequestError` | `q` | `increment.estimation.quantile` |
| `estimation.quantile.estimate_quantile.metric_arm` | `InvalidRequestError` | `arm`, `error`, `metric` | `increment.estimation.quantile` |
| `estimation.quantile.log_quantile_se` | `InvalidRequestError` | `q` | `increment.estimation.quantile` |
| `estimation.quantile.quantile_positive_log` | `InvalidRequestError` | `lo`, `point`, `q` | `increment.estimation.quantile` |
| `estimation.quantile.too_small_bound` | `InvalidRequestError` | `level`, `n`, `n_min`, `q` | `increment.estimation.quantile` |
| `estimation.quantile.values_contain_non` | `InvalidRequestError` | `n_bad` | `increment.estimation.quantile` |
| `estimation.quantile.values_one_dimensional` | `InvalidRequestError` | `shape` | `increment.estimation.quantile` |
| `estimation.results.binomial.counts_out_of_range` | `InvalidRequestError` | `n_c`, `n_t`, `x_c`, `x_t` | `increment.estimation.results` |
| `estimation.results.binomial.decision_metadata` | `InvalidRequestError` | `reason` | `increment.estimation.results` |
| `estimation.results.binomial.interval_inverted` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation.results` |
| `estimation.results.binomial.obsolete_construction` | `InvalidRequestError` | `method`, `supported` | `increment.estimation.results` |
| `estimation.results.binomial.posterior_unavailable` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.delta.counts_out_of_range` | `InvalidRequestError` | `n_c`, `n_t`, `x_c`, `x_t` | `increment.estimation.results` |
| `estimation.results.delta.dense_level` | `InvalidRequestError` | `reason` | `increment.estimation.results` |
| `estimation.results.delta.reference_mismatch` | `InvalidRequestError` | `group_id`, `metric`, `reference_kind` | `increment.estimation.results` |
| `estimation.results.estimate.alpha_none_lb` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.estimate.interval_inverted_lb` | `InvalidRequestError` | `lb`, `ub` | `increment.estimation.results` |
| `estimation.results.estimate.lb_ub_both` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.estimate.level_alpha_because` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.estimate.level_contradicts_effective` | `InvalidRequestError` | `alpha`, `expected_level`, `level` | `increment.estimation.results` |
| `estimation.results.estimate.level_interval` | `InvalidRequestError` | `level` | `increment.estimation.results` |
| `estimation.results.estimate.level_lb_ub` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.estimate.level_none_lb` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.estimate.open_side_bound_must_be_none` | `InvalidRequestError` | `field` | `increment.estimation.results` |
| `estimation.results.estimate.open_side_needs_calibrated_bound` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.joint.invalid_reference` | `InvalidRequestError` | `reason` | `increment.estimation.results` |
| `estimation.results.joint.invalid_set` | `InvalidRequestError` | `reason` | `increment.estimation.results` |
| `estimation.results.joint.unavailable` | `InvalidRequestError` | `reason` | `increment.estimation.results` |
| `estimation.results.lift.absolute_alpha_mismatch` | `InvalidRequestError` | `abs_alpha`, `alpha` | `increment.estimation.results` |
| `estimation.results.lift.absolute_alpha_without_interval` | `InvalidRequestError` | `abs_alpha` | `increment.estimation.results` |
| `estimation.results.lift.absolute_reference_mismatch` | `InvalidRequestError` | `df`, `kind` | `increment.estimation.results` |
| `estimation.results.lift.binomial_lift_availability` | `InvalidRequestError` | `group_id`, `metric`, `reason` | `increment.estimation.results` |
| `estimation.results.lift.dof_reference_df_mismatch` | `InvalidRequestError` | `dof`, `group_id`, `metric`, `reference_df` | `increment.estimation.results` |
| `estimation.results.lift.fcr_prior_unsupported` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.lift.inference_reference_kind_mismatch` | `InvalidRequestError` | `group_id`, `inference`, `metric`, `reference_kind` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_carries_no` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_chance_to` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_dof_set` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_interval_symmetric` | `InvalidRequestError` | `group_id`, `metric`, `scale` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_level_contradicts` | `InvalidRequestError` | `e_alpha`, `e_level`, `expected_level`, `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_log_se` | `InvalidRequestError` | `e_log_se`, `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_prior_spec` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_prob_favorable` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_risk_if` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.liftestimate_value_lb` | `InvalidRequestError` | `e_lb`, `e_value`, `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.open_interval_unrecoverable` | `InvalidRequestError` | — | `increment.estimation.results` |
| `estimation.results.lift.open_side_alternative_mismatch` | `InvalidRequestError` | `alternative`, `open_side` | `increment.estimation.results` |
| `estimation.results.lift.p_value_cluster_robust_null_abs` | `InvalidRequestError` | `group_id`, `metric`, `reference_df` | `increment.estimation.results` |
| `estimation.results.lift.p_value_missing_log_mean_or_se` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.p_value_null_abs_missing_abs_se` | `InvalidRequestError` | `group_id`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.p_value_null_lift_not_representable_dof` | `InvalidRequestError` | `group_id`, `metric`, `null_lift` | `increment.estimation.results` |
| `estimation.results.lift.posterior_components_invalid` | `InvalidRequestError` | `group_id`, `metric`, `reason` | `increment.estimation.results` |
| `estimation.results.lift.posterior_decision_stats_cluster_robust` | `InvalidRequestError` | `group_id`, `metric`, `reference_df` | `increment.estimation.results` |
| `estimation.results.lift.posterior_decision_stats_sequential` | `InvalidRequestError` | `group_id`, `inference`, `metric` | `increment.estimation.results` |
| `estimation.results.lift.prob_within_threshold_positive` | `InvalidRequestError` | `threshold` | `increment.estimation.results` |
| `estimation.results.lift.prob_within_threshold_unit_interval` | `InvalidRequestError` | `threshold` | `increment.estimation.results` |
| `estimation.results.lift.reference_df_required_for_t` | `InvalidRequestError` | `group_id`, `metric`, `reference_df` | `increment.estimation.results` |
| `estimation.results.lift.reference_df_set_for_normal` | `InvalidRequestError` | `group_id`, `metric`, `reference_df`, `reference_kind` | `increment.estimation.results` |
| `estimation.results.lift.t_posterior_sufficient_statistics` | `InvalidRequestError` | `group_id`, `metric`, `missing`, `reference_df` | `increment.estimation.results` |
| `estimation.results.lift.threshold_representable_log` | `InvalidRequestError` | `threshold` | `increment.estimation.results` |
| `estimation.rollout.cost_threshold_finite` | `InvalidRequestError` | `cost_threshold` | `increment.estimation.rollout` |
| `estimation.rollout.segment_rollout.policy_value_policy` | `InvalidRequestError` | — | `increment.estimation.rollout` |
| `estimation.rollout.segment_rollout.recommendation_match_value` | `InvalidRequestError` | — | `increment.estimation.rollout` |
| `estimation.rollout.segment_rollout.recommendation_refuse_carry` | `InvalidRequestError` | — | `increment.estimation.rollout` |
| `estimation.rollout.segment_rollout.selected_bool_length` | `InvalidRequestError` | `dtype`, `k`, `shape` | `increment.estimation.rollout` |
| `estimation.rollout.segment_rollout.selection_bias_share` | `InvalidRequestError` | — | `increment.estimation.rollout` |
| `estimation.sequential_inversion.accepted_coordinate_mismatch` | `InvalidRequestError` | `expected`, `ratio`, `side` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.bound_coordinate_mismatch` | `InvalidRequestError` | `bound`, `expected`, `side` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.endpoint_width_exceeded` | `InvalidRequestError` | `bracket`, `max_width`, `side` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.incomplete_finite_endpoint` | `InvalidRequestError` | `accepted`, `bracket`, `rejected` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_alpha` | `InvalidRequestError` | `alpha` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_bracket_length` | `InvalidRequestError` | `bracket` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_domain_evidence_type` | `InvalidRequestError` | `value_type` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_endpoint_type` | `InvalidRequestError` | `side`, `value_type` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_evidence_type` | `InvalidRequestError` | `value_type` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_threshold_type` | `InvalidRequestError` | `value_type` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.invalid_witness_type` | `InvalidRequestError` | `value_type`, `witness` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.missing_endpoint_reason` | `InvalidRequestError` | `status` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.negative_bernoulli_bound` | `InvalidRequestError` | `lower` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.rejected_coordinate_mismatch` | `InvalidRequestError` | `expected`, `ratio`, `side` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.reversed_bracket` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.uncertified_empty_endpoint` | `InvalidRequestError` | `domain_log_e`, `side`, `threshold` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.uncertified_nonrejection` | `InvalidRequestError` | `evidence`, `side`, `threshold` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.uncertified_rejection` | `InvalidRequestError` | `evidence`, `side`, `threshold` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.uncertified_reversed_bounds` | `InvalidRequestError` | `lower`, `upper` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.uncertified_uniform_bound` | `InvalidRequestError` | `log_e_upper`, `side`, `threshold` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.unknown_domain` | `InvalidRequestError` | `domain` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.unknown_endpoint_status` | `InvalidRequestError` | `status` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_inversion.witness_outside_bracket` | `InvalidRequestError` | `bracket`, `ratio`, `witness` | `increment.estimation._sequential_inversion` |
| `estimation.sequential_likelihood.asymmetric_matrix` | `InvalidRequestError` | `matrix` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.evidence_enclosure_mismatch` | `InvalidRequestError` | `log_e`, `required` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.inconsistent_nonfinite_logs` | `InvalidRequestError` | `log_e`, `log_null_sup`, `reason`, `status` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.insufficient_prior_nu` | `InvalidRequestError` | `dimension`, `nu` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.invalid_certificate_reason` | `InvalidRequestError` | `reason` | `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.invalid_dimension` | `InvalidRequestError` | `dimension` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.invalid_log_interval` | `InvalidRequestError` | `field`, `value_type` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.matrix_dimension_mismatch` | `InvalidRequestError` | `dimension`, `row_lengths` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.merge_dimension_mismatch` | `InvalidRequestError` | `left_dimension`, `right_dimension` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.missing_finite_logs` | `InvalidRequestError` | `log_e`, `log_null_sup`, `log_predictive` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.negative_bernoulli_ratio` | `InvalidRequestError` | `ratio` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.non_psd_matrix` | `InvalidRequestError` | `matrix`, `principal_minors` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonfinite_observation` | `InvalidRequestError` | `value` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonpositive_beta_parameter` | `InvalidRequestError` | `parameter`, `value` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonpositive_definite_matrix` | `InvalidRequestError` | `matrix`, `principal_minors` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonpositive_max_error` | `InvalidRequestError` | `max_error` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonpositive_prior_kappa` | `InvalidRequestError` | `kappa` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonzero_empty_mean` | `InvalidRequestError` | `mean` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.nonzero_small_sample_scatter` | `InvalidRequestError` | `n`, `scatter` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.row_dimension_mismatch` | `InvalidRequestError` | `dimension`, `row_dimension` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.scatter_rank_exceeds_count` | `InvalidRequestError` | `n`, `scatter` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.singular_candidate_residual` | `InvalidRequestError` | `candidate`, `mean`, `n`, `residual`, `scatter` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.state_prior_dimension_mismatch` | `InvalidRequestError` | `control_dimension`, `prior_control_dimension`, `prior_treatment_dimension`, `treatment_dimension` | `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.successes_exceed_count` | `InvalidRequestError` | `n`, `successes` | `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.unknown_alternative` | `InvalidRequestError` | `alternative` | `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood` |
| `estimation.sequential_likelihood.unknown_certificate_status` | `InvalidRequestError` | `status` | `increment.estimation._sequential_likelihood` |
| `estimation.sitewide.alpha` | `InvalidRequestError` | `alpha` | `increment.estimation.sitewide` |
| `estimation.sitewide.cluster_arm_needs_two` | `InvalidRequestError` | `arm_counts`, `cluster`, `metric` | `increment.estimation.sitewide` |
| `estimation.sitewide.control_carries_metric` | `InvalidRequestError` | `control`, `target` | `increment.estimation.sitewide` |
| `estimation.sitewide.control_carries_study` | `InvalidRequestError` | `control`, `target` | `increment.estimation.sitewide` |
| `estimation.sitewide.control_target_are` | `InvalidRequestError` | `control` | `increment.estimation.sitewide` |
| `estimation.sitewide.counterfactual_baseline_denominator` | `InvalidRequestError` | `baseline_denominator`, `delta_den`, `den_contribution`, `n_units`, `other_arms`, `site_total_denominator` | `increment.estimation.sitewide` |
| `estimation.sitewide.counterfactual_baseline_ratio` | `InvalidRequestError` | `baseline_denominator`, `baseline_numerator`, `delta_num`, `n_units`, `num_contribution`, `other_arms`, `site_total_numerator` | `increment.estimation.sitewide` |
| `estimation.sitewide.counterfactual_baseline_volume` | `InvalidRequestError` | `baseline_volume`, `delta`, `n_units`, `other_arms`, `other_contribution`, `site_total_volume` | `increment.estimation.sitewide` |
| `estimation.sitewide.from_iid.metric_arm_carries_ref_den` | `InvalidRequestError` | `arm`, `label`, `target` | `increment.estimation.sitewide` |
| `estimation.sitewide.other_arms_entry_metric_mismatch` | `InvalidRequestError` | `arm`, `metric`, `target` | `increment.estimation.sitewide` |
| `estimation.sitewide.other_arms_entry_study_mismatch` | `InvalidRequestError` | `arm`, `study_id`, `target_study_id` | `increment.estimation.sitewide` |
| `estimation.sitewide.other_arms_repeats` | `InvalidRequestError` | `arm` | `increment.estimation.sitewide` |
| `estimation.sitewide.ship_all_denominator` | `InvalidRequestError` | `baseline_denominator`, `delta_den`, `n_exp`, `shipped_denominator` | `increment.estimation.sitewide` |
| `estimation.sitewide.sitewide_ratio.metric_arm_declares` | `InvalidRequestError` | `arm`, `label`, `remedy`, `target`, `x_role` | `increment.estimation.sitewide` |
| `estimation.sitewide.sitewide_ratio.metric_arm_missing` | `InvalidRequestError` | `arm`, `metric` | `increment.estimation.sitewide` |
| `estimation.tails.unresolvable` | `InvalidRequestError` | `tail_or_alpha`, `value`, `what` | `increment.estimation._tails` |
| `estimation.targeting.adjustment_dtype` | `InvalidRequestError` | `caller`, `column`, `dtype` | `increment.estimation.targeting` |
| `estimation.targeting.bootstrap_empty_arm` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.bootstrap_insufficient_clusters` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.bootstrap_options` | `InvalidRequestError` | — | `increment.estimation.targeting` |
| `estimation.targeting.bootstrap_tail_resolution` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.bootstrap_unavailable_replicate` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.bootstrap_zero_variance` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.cluster_count_bounds` | `InvalidRequestError` | `n_clusters`, `n_holdout` | `increment.estimation.targeting` |
| `estimation.targeting.cluster_count_mismatch` | `InvalidRequestError` | `n_clusters`, `validation_n_clusters` | `increment.estimation.targeting` |
| `estimation.targeting.cluster_grain_count_required` | `InvalidRequestError` | — | `increment.estimation.targeting` |
| `estimation.targeting.cluster_ids_shape` | `InvalidRequestError` | `cluster_ids`, `y` | `increment.estimation.targeting` |
| `estimation.targeting.cluster_ids_string` | `InvalidRequestError` | `cluster_ids` | `increment.estimation.targeting` |
| `estimation.targeting.covariate_rows_expected` | `InvalidRequestError` | `cov`, `values`, `y` | `increment.estimation.targeting` |
| `estimation.targeting.degenerate_cluster_variance` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.degenerate_rank_distribution` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.deployment_cluster_ids_required` | `InvalidRequestError` | — | `increment.estimation._deployment` |
| `estimation.targeting.deployment_cluster_ids_shape` | `InvalidRequestError` | `n`, `shape` | `increment.estimation._deployment` |
| `estimation.targeting.empty_arm` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.empty_group` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.evaluation_population_invalid` | `InvalidRequestError` | `reason` | `increment.estimation.targeting` |
| `estimation.targeting.every_fraction` | `InvalidRequestError` | `f` | `increment.estimation.targeting` |
| `estimation.targeting.fraction_share_units` | `InvalidRequestError` | `fraction` | `increment.estimation.targeting` |
| `estimation.targeting.fractions_contains_duplicates` | `InvalidRequestError` | `fracs` | `increment.estimation.targeting` |
| `estimation.targeting.fractions_non_empty` | `InvalidRequestError` | — | `increment.estimation.targeting` |
| `estimation.targeting.half_split_holds` | `InvalidRequestError` | `n_control`, `n_treated`, `name` | `increment.estimation.targeting` |
| `estimation.targeting.holdout_mask_shape` | `InvalidRequestError` | `hold`, `y` | `increment.estimation.targeting` |
| `estimation.targeting.insufficient_arm_clusters` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.insufficient_clusters` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.invalid_deployment_grain` | `InvalidRequestError` | `deploy_grain`, `intervention_grain` | `increment.estimation._deployment` |
| `estimation.targeting.jackknife_unavailable_replicate` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.n_groups_least` | `InvalidRequestError` | `n_groups` | `increment.estimation.targeting` |
| `estimation.targeting.nonfinite_statistic` | `InvalidRequestError` | `message` | `increment.estimation.targeting` |
| `estimation.targeting.outcome_shape` | `InvalidRequestError` | `y` | `increment.estimation.targeting` |
| `estimation.targeting.overlap.empty_arm` | `IdentificationError` | `min_propensity`, `n_kept`, `n_total` | `increment.estimation.targeting` |
| `estimation.targeting.overlap.partial_cluster` | `IdentificationError` | — | `increment.estimation.targeting` |
| `estimation.targeting.overlap.refuse` | `IdentificationError` | `min_propensity`, `n_outside`, `n_total` | `increment.estimation.targeting` |
| `estimation.targeting.policy_grain_mismatch` | `InvalidRequestError` | — | `increment.estimation.targeting` |
| `estimation.targeting.policy_metadata` | `InvalidRequestError` | — | `increment.estimation.targeting` |
| `estimation.targeting.score_state_required` | `InvalidRequestError` | — | `increment.estimation.targeting` |
| `estimation.targeting.targeting_fraction_leaves` | `InvalidRequestError` | `fraction`, `n_control`, `n_treatment` | `increment.estimation.targeting` |
| `estimation.targeting.training_mask_shape` | `InvalidRequestError` | `train`, `y` | `increment.estimation.targeting` |
| `estimation.targeting.treatment_shape_expected` | `InvalidRequestError` | `d`, `y` | `increment.estimation.targeting` |
| `estimation.targeting.unit_ids_contains` | `InvalidRequestError` | `count`, `dupes` | `increment.estimation.targeting` |
| `estimation.targeting.unit_ids_shape` | `InvalidRequestError` | `unit_ids`, `y` | `increment.estimation.targeting` |
| `estimation.targeting.unit_ids_string` | `InvalidRequestError` | `unit_ids` | `increment.estimation.targeting` |
| `estimation.targeting.unsupported_unit_deployment` | `InvalidRequestError` | — | `increment.estimation._deployment` |
| `estimation.targeting.validation_fold_holds` | `InvalidRequestError` | `label`, `n_control`, `n_treated` | `increment.estimation.targeting` |
| `estimation.variance.cluster_outcome_moments_least_two_units` | `InvalidRequestError` | `group_id`, `metric`, `n` | `increment.estimation.variance` |
| `estimation.variance.cluster_outcome_moments_nonpositive_size` | `InvalidRequestError` | `group_id`, `metric`, `size_bar` | `increment.estimation.variance` |
| `estimation.variance.cluster_robust_late` | `InvalidRequestError` | `carries`, `x_role` | `increment.estimation.variance` |
| `estimation.variance.cluster_robust_late_needs_cluster_uptake_family` | `InvalidRequestError` | — | `increment.estimation.variance` |
| `estimation.variance.cluster_robust_late_needs_cxden` | `InvalidRequestError` | — | `increment.estimation.variance` |
| `estimation.variance.cluster_uptake_moments_least_two_units` | `InvalidRequestError` | `group_id`, `metric`, `n` | `increment.estimation.variance` |
| `estimation.variance.cluster_uptake_moments_needs_ref_den` | `InvalidRequestError` | — | `increment.estimation.variance` |
| `estimation.variance.mean_positive_log` | `InvalidRequestError` | `mean` | `increment.estimation.variance` |
| `estimation.variance.metric_group_log` | `InvalidRequestError` | `group_id`, `mean`, `metric`, `what` | `increment.estimation.variance` |
| `estimation.variance.positive_se_log` | `InvalidRequestError` | `n` | `increment.estimation.variance` |
| `estimation.variance.ratio_moments_least_two_units` | `InvalidRequestError` | `group_id`, `metric`, `n` | `increment.estimation.variance` |
| `estimation.variance.ratio_moments_needs_ref_den` | `InvalidRequestError` | — | `increment.estimation.variance` |
| `estimation.variance.ratio_moments_nonpositive_denominator_mean` | `InvalidRequestError` | `d_bar`, `group_id`, `metric` | `increment.estimation.variance` |
| `estimation.variance.ratio_residual_variance` | `InvalidRequestError` | `den_bar`, `num_bar`, `residual`, `what` | `increment.estimation.variance` |
| `estimation.variance.ratio_residual_variance_negative` | `InvalidRequestError` | `residual`, `what` | `increment.estimation.variance` |
| `estimation.variance.ratio_se_final_not_representable` | `InvalidRequestError` | `den_bar`, `num_bar`, `r`, `se`, `what` | `increment.estimation.variance` |
| `estimation.variance.ratio_se_representable` | `InvalidRequestError` | `clamped`, `den_bar`, `num_bar`, `what` | `increment.estimation.variance` |
| `estimation.variance.ratio_se_sqrt_not_representable` | `InvalidRequestError` | `den_bar`, `num_bar`, `root`, `what` | `increment.estimation.variance` |
| `estimation.variance.ratio_variance.correlation_not_representable` | `InvalidRequestError` | `group_id`, `log_rho`, `metric` | `increment.estimation.variance` |
| `estimation.variance.ratio_variance.se_not_representable` | `InvalidRequestError` | `group_id`, `metric`, `se` | `increment.estimation.variance` |
| `estimation.variance.ratio_variance.true_se_out_of_range` | `InvalidRequestError` | `group_id`, `log_se`, `metric` | `increment.estimation.variance` |
| `estimation.variance.ratio_variance.variance_scale_not_representable` | `InvalidRequestError` | `group_id`, `log_scale`, `metric`, `value`, `variance_slack` | `increment.estimation.variance` |
| `estimation.variance.ratio_variance_term` | `InvalidRequestError` | `den_bar`, `num_bar`, `rooted`, `var_den`, `what` | `increment.estimation.variance` |
| `estimation.variance.registry.no_registered_available` | `UnsupportedRequestError` | `available`, `key`, `registry_name` | `increment.declarations` |
| `estimation.variance.se_log_mean_not_representable` | `InvalidRequestError` | `mean`, `n`, `se`, `var` | `increment.estimation.variance` |
| `estimation.variance.se_log_mean_out_of_range` | `InvalidRequestError` | `half`, `mean`, `n`, `var` | `increment.estimation.variance` |
| `estimation.variance.var_non_negative` | `InvalidRequestError` | `var` | `increment.estimation.variance` |
| `estimation.winsor.cutoff_in_zero_atom` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.density_required` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.density_unresolved` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.design_unsupported` | `CapabilityError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.endpoint_unrepresentable` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.invalid_state` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.pilot_degenerate` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.pilot_negative_outcome` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.pool_mismatch` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.posterior_unavailable` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.rank_size_unsupported` | `CapabilityError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.raw_state_required` | `CapabilityError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.reinversion_required` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.studentization_degenerate` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.support_required` | `CapabilityError` | `reason` | `increment._winsor_errors` |
| `estimation.winsor.support_violation` | `InvalidRequestError` | `reason` | `increment._winsor_errors` |
| `evidence.assignment.identity_unavailable` | `CapabilityError` | `constructor`, `operation`, `route` | `increment._evidence_refusals` |
| `evidence.assignment.invalid` | `CapabilityError` | `details`, `reason` | `increment._assignment_refusals`, `increment._evidence_refusals` |
| `evidence.assignment.replay_mismatch` | `CapabilityError` | `details` | `increment._assignment_refusals`, `increment._evidence_refusals` |
| `evidence.assignment.selection_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested` | `increment._assignment_refusals`, `increment._evidence_refusals` |
| `evidence.assignment.unavailable` | `CapabilityError` | `constructor`, `offered`, `route` | `increment._assignment_refusals`, `increment._evidence_refusals` |
| `evidence.breakout_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `evidence.cluster_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `evidence.covariate_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `evidence.grain_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `evidence.joint_grouping_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested` | `increment._evidence_refusals` |
| `evidence.operation_unavailable` | `CapabilityError` | `constructor`, `offered`, `request`, `requested`, `route` | `increment._evidence_refusals`, `increment.query._native_refusals` |
| `evidence.provider_unavailable` | `CapabilityError` | `constructor` | `increment._evidence_refusals`, `increment._source_types` |
| `evidence.raw_outcomes_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `evidence.request_unsupported` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `evidence.unit_rows_unavailable` | `CapabilityError` | `constructor`, `offered`, `requested`, `route` | `increment._evidence_refusals` |
| `facade.analysis.bh_segmented_asof` | `UnsupportedRequestError` | `method` | `increment._day_axis` |
| `facade.analysis.clustered_day_axis` | `CapabilityError` | `cluster`, `experiment`, `method` | `increment._day_axis` |
| `facade.analysis.contrast_unavailable` | `CapabilityError` | `method` | `increment._evidence_refusals`, `increment.cate`, `increment.dashboard._data` |
| `facade.analysis.daily_compliance_dimension_unsupported` | `UnsupportedRequestError` | `method` | `increment._day_axis` |
| `facade.analysis.definitions_state_requires_identification` | `InvalidRequestError` | — | `increment.builders` |
| `facade.analysis.encouragement_daily_late` | `InvalidRequestError` | `experiment` | `increment._day_axis` |
| `facade.analysis.exploratory_metrics_sequential` | `UnsupportedRequestError` | `inference`, `method`, `names` | `increment._exploratory` |
| `facade.analysis.exploratory_metrics_source_limited` | `CapabilityError` | `method` | `increment._exploratory` |
| `facade.analysis.invalid_mixed_assignment_policy` | `InvalidRequestError` | `on_mixed_assignment` | `increment.builders` |
| `facade.analysis.invalid_population` | `InvalidRequestError` | `population` | `increment._evidence_refusals`, `increment._sitewide`, `increment._source_diagnostics` |
| `facade.analysis.invalid_store_policy` | `InvalidRequestError` | `store` | `increment.builders` |
| `facade.analysis.observational_day_axis` | `UnsupportedRequestError` | `method` | `increment._day_axis` |
| `facade.analysis.sitewide_arm_not_enrolled` | `InvalidRequestError` | `arm`, `arms`, `control_group`, `metric` | `increment._sitewide` |
| `facade.analysis.sitewide_arm_required` | `InvalidRequestError` | `arms`, `metric` | `increment._sitewide` |
| `facade.analysis.sitewide_cluster_counts_missing` | `CapabilityError` | `cluster` | `increment._sitewide` |
| `facade.analysis.sitewide_cluster_metric_type` | `CapabilityError` | `cluster`, `metric`, `metric_type` | `increment._sitewide` |
| `facade.analysis.sitewide_control_missing` | `InvalidRequestError` | `available`, `control_group`, `metric` | `increment._sitewide` |
| `facade.analysis.sitewide_metric_unaligned` | `CapabilityError` | `aggregation`, `metric`, `metric_type`, `route`, `window_days` | `increment._sitewide` |
| `facade.analysis.sitewide_no_treatment_arm` | `InvalidRequestError` | `control_group`, `metric` | `increment._sitewide` |
| `facade.analysis.sitewide_outcomes_incomplete` | `CapabilityError` | `route` | `increment.query.native_contract`, `increment.query.native_source`, `increment.query.source` |
| `facade.analysis.sitewide_ratio_denominator_missing` | `CapabilityError` | `metric` | `increment._sitewide` |
| `facade.analysis.sitewide_requires_randomized` | `CapabilityError` | — | `increment._sitewide` |
| `facade.analysis.sitewide_triggered_cluster_unsupported` | `CapabilityError` | `cluster` | `increment._sitewide` |
| `facade.analysis.sitewide_triggered_end_missing` | `CapabilityError` | — | `increment._sitewide` |
| `facade.analysis.sitewide_triggered_sequential_unsupported` | `CapabilityError` | — | `increment._sitewide` |
| `facade.analysis.sitewide_winsorized_metric` | `CapabilityError` | `metric` | `increment._sitewide` |
| `facade.analysis.source_context_design_disagrees_arm` | `InvalidRequestError` | — | `increment.builders` |
| `facade.analysis.switchback_only` | `CapabilityError` | `method` | `increment._source_diagnostics` |
| `facade.analysis.trigger_unsupported` | `CapabilityError` | `experiment`, `method`, `trigger` | `increment._breakout_readouts`, `increment._day_axis`, `increment._execute`, `increment.readouts._whole_window` |
| `facade.analysis.uncorrected_segments_sequential` | `UnsupportedRequestError` | `inference` | `increment._breakout_readouts` |
| `facade.analysis.undeclared_breakout` | `InvalidRequestError` | `declared`, `requested` | `increment._binding`, `increment._dashboard_capture` |
| `facade.analysis.undeclared_metric_for_dimension` | `InvalidRequestError` | `dimension`, `method`, `undeclared` | `increment._day_axis` |
| `facade.analysis.unknown_dimension` | `InvalidRequestError` | `declared`, `dimension`, `method` | `increment._day_axis` |
| `facade.analysis.unknown_experiment` | `InvalidRequestError` | `available`, `requested` | `increment.builders` |
| `facade.analysis.unknown_metric` | `InvalidRequestError` | `declared`, `metric` | `increment._sitewide`, `increment.readouts._whole_window` |
| `facade.analysis_config.covariate_binding_conflict` | `InvalidRequestError` | `covariate` | `increment._analysis_config` |
| `facade.analysis_config.cuped_linear_covariates` | `InvalidRequestError` | `metric` | `increment._analysis_config` |
| `facade.analysis_config.cuped_linear_metric_type` | `InvalidRequestError` | `metric`, `metric_type` | `increment._analysis_config` |
| `facade.analysis_config.cuped_linear_moments` | `UnsupportedRequestError` | `metric` | `increment._analysis_config` |
| `facade.analysis_config.cuped_scalar_moments` | `UnsupportedRequestError` | `metric` | `increment._analysis_config` |
| `facade.analysis_config.duplicate_metric_name` | `InvalidRequestError` | `caller`, `duplicate_names` | `increment._analysis_config` |
| `facade.analysis_config.exploratory_metric_unavailable` | `InvalidRequestError` | `available_names`, `caller`, `declared_names`, `names`, `unknown_names` | `increment._analysis_config` |
| `facade.analysis_config.metric_definition_mismatch` | `InvalidRequestError` | `caller`, `mismatched_names` | `increment._analysis_config` |
| `facade.analysis_config.metric_no_resolved` | `InvalidRequestError` | `metric_name` | `increment._analysis_config` |
| `facade.analysis_config.multifamily_scalar_breakout` | `UnsupportedRequestError` | `metric` | `increment._analysis_config` |
| `facade.analysis_config.unknown_metric_declared` | `InvalidRequestError` | `caller`, `declared_names`, `unknown_names` | `increment._analysis_config` |
| `facade.compatibility_inspector.compatibility_report.overall_fields_equal` | `InvalidRequestError` | `actual`, `expected` | `increment._compatibility_inspector` |
| `facade.evidence_dispatch.contrast_handler.readout_request_no` | `InvalidRequestError` | `metric_name` | `increment._evidence_dispatch` |
| `facade.plan_compatibility.plan_family.runtime_effect_limited` | `InvalidRequestError` | — | `increment._plan_compatibility` |
| `facade.study.switchback_study.allocation_include_its` | `InvalidRequestError` | — | `increment._study` |
| `facade.study.switchback_study.identification_allocation_exactly` | `InvalidRequestError` | — | `increment._study` |
| `facade.study.switchback_study.identification_exactly_one` | `InvalidRequestError` | — | `increment._study` |
| `facade.window.resolve_window_days` | `UnsupportedRequestError` | `metric_type` | `increment._window` |
| `frame.asof.day_axis_unorderable` | `CapabilityError` | `labels` | `increment._day_axis_labels` |
| `frame.asof.quantile_unsupported` | `CapabilityError` | `metric` | `increment._frame_moments`, `increment.frame` |
| `frame.duplicate_metric_name` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.frame_panel.asof_moments_completed` | `InvalidRequestError` | `band`, `name` | `increment.frame` |
| `frame.frame_panel.unit_covariate_column` | `InvalidRequestError` | `columns`, `missing` | `increment.frame` |
| `frame.frame_panel.unit_covariate_varies` | `InvalidRequestError` | `covariate`, `units` | `increment.frame` |
| `frame.frame_totals.control_group_absent` | `InvalidRequestError` | `control`, `groups` | `increment.frame` |
| `frame.frame_totals.metric_was_declared` | `InvalidRequestError` | `name` | `increment.frame` |
| `frame.frame_totals.moments_metric_undeclared` | `InvalidRequestError` | `declared`, `names` | `increment.frame` |
| `frame.frame_totals.unit_covariate_column` | `InvalidRequestError` | `columns`, `missing` | `increment.frame` |
| `frame.frame_totals.unit_covariate_reserved` | `InvalidRequestError` | `cluster`, `column` | `increment.frame` |
| `frame.metric.covariate_names_unique` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.cuped_does_apply` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.cuped_linear_supported_metrics` | `InvalidRequestError` | `name`, `type` | `increment._metric_specs` |
| `frame.metric.declares_cuped_linear` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.declares_cuped_method` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.method_name_cuped` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.missing_impute` | `InvalidRequestError` | `metric`, `route` | `increment._metric_specs` |
| `frame.metric.scalar_and_set_covariates` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.sets_quantile_but` | `InvalidRequestError` | `name`, `type` | `increment._metric_specs` |
| `frame.metric.sets_threshold_days` | `InvalidRequestError` | `name`, `type` | `increment._metric_specs` |
| `frame.metric.type_but_sets` | `InvalidRequestError` | `name`, `type` | `increment._metric_specs` |
| `frame.metric.type_quantile_needs` | `InvalidRequestError` | `name`, `quantile` | `increment._metric_specs` |
| `frame.metric.type_ratio_but` | `InvalidRequestError` | `missing`, `name` | `increment._metric_specs` |
| `frame.metric.type_retention_needs` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.window_days_supported` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric.winsorization_applies_type` | `InvalidRequestError` | `name`, `type` | `increment._metric_specs` |
| `frame.metric_type_quantile` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric_type_retention` | `InvalidRequestError` | `name` | `increment._metric_specs` |
| `frame.metric_unknown_type` | `InvalidRequestError` | `kind`, `name` | `increment._metric_specs` |
| `frame.metrics.type_required` | `InvalidRequestError` | `dtype`, `metric`, `options` | `increment._metric_specs` |
| `frame.metrics_empty_supply` | `InvalidRequestError` | — | `increment._metric_specs` |
| `frame.metrics_entries_metricspec` | `InvalidRequestError` | `type_name` | `increment._metric_specs` |
| `frame.missing_policy.panel_drop` | `UnsupportedRequestError` | `missing` | `increment._frame_validation` |
| `frame.moments.asof_completed_requires_uptake_window` | `InvalidRequestError` | — | `increment._frame_moments` |
| `frame.moments.asof_moments_completed` | `InvalidRequestError` | `band`, `metric` | `increment._frame_moments` |
| `frame.moments.asof_source_uptake_needs_anchor` | `InvalidRequestError` | — | `increment._frame_moments` |
| `frame.moments.asof_unit_rows` | `InvalidRequestError` | — | `increment._frame_moments` |
| `frame.moments.collapse_to_unit` | `InvalidRequestError` | — | `increment._frame_moments` |
| `frame.moments.metric_type_conversion` | `InvalidRequestError` | `metric`, `value` | `increment._frame_moments` |
| `frame.panel.declaration_canonical_collision` | `InvalidRequestError` | `canonical_name`, `role`, `route`, `source_name` | `increment._frame_panel` |
| `frame.panel.observable_end_dtype` | `InvalidRequestError` | `anchor_dtype`, `axis_kind`, `end_dtype`, `end_kind`, `observable_end` | `increment._frame_panel` |
| `frame.validation.breakout_overlaps_column` | `InvalidRequestError` | `name`, `role` | `increment._frame_validation` |
| `frame.validation.breakout_overlaps_metric` | `InvalidRequestError` | `name` | `increment._frame_validation` |
| `frame.validation.breakout_reserved_internal` | `InvalidRequestError` | `name` | `increment._frame_validation` |
| `frame.validation.breakout_unit_stable` | `InvalidRequestError` | `count`, `more`, `name`, `shown` | `increment._frame_validation` |
| `frame.validation.covariate_not_numeric` | `InvalidRequestError` | `covariate`, `dtype`, `metric`, `variance_reduction` | `increment._frame_validation` |
| `frame.validation.design_control_group` | `InvalidRequestError` | `control`, `control_group` | `increment._frame_validation` |
| `frame.validation.dtype_disagree_day` | `InvalidRequestError` | `anchor_dtype`, `anchor_name`, `ds_dtype`, `ds_name` | `increment._frame_validation` |
| `frame.validation.duplicate_breakout_column` | `InvalidRequestError` | `duplicates` | `increment._frame_validation` |
| `frame.validation.from_unit_panel` | `InvalidRequestError` | `covered`, `retention` | `increment._frame_validation` |
| `frame.validation.group_label_collision` | `InvalidRequestError` | `collision`, `group`, `route` | `increment._frame_validation` |
| `frame.validation.group_label_reserved` | `InvalidRequestError` | `group`, `labels`, `route` | `increment._frame_validation` |
| `frame.validation.metric_covariate_column` | `InvalidRequestError` | `covariate`, `metric`, `n_missing` | `increment._frame_moments`, `increment._frame_validation`, `increment.query.native_source` |
| `frame.validation.metric_covariate_impute_all_null` | `InvalidRequestError` | `covariate`, `metric` | `increment._frame_moments`, `increment._frame_validation`, `increment.query.native_source` |
| `frame.validation.metric_value_missing` | `InvalidRequestError` | `column`, `metric`, `n_missing` | `increment._frame_validation` |
| `frame.winsorization.collapsed_bound` | `CapabilityError` | `clipped`, `date`, `lower`, `metric`, `upper` | `increment._frame_moments` |
| `identification.encouragement.exclusion_required` | `IdentificationError` | `available_estimands`, `estimands`, `required_assumption` | `increment.estimation.encouragement` |
| `impute.impute_column_found` | `InvalidRequestError` | `helper`, `missing`, `present` | `increment.impute` |
| `impute.impute_name_least` | `InvalidRequestError` | `helper` | `increment.impute` |
| `impute.impute_pooled_mean` | `InvalidRequestError` | `col` | `increment.impute` |
| `impute.impute_pooled_mean_not_finite` | `InvalidRequestError` | `col`, `mean`, `observed` | `increment.impute` |
| `impute.impute_roles_columns` | `InvalidRequestError` | `columns`, `requested` | `increment.impute` |
| `impute.impute_roles_value` | `InvalidRequestError` | `columns` | `increment.impute` |
| `impute.pooled_mean_outcome` | `InvalidRequestError` | `columns`, `route` | `increment.impute` |
| `integrity.allocation_missing_or_nonconstant` | `CapabilityError` | `population` | `increment.estimation.assignment_integrity`, `increment.readouts._common` |
| `integrity.allocation_scheme_missing` | `CapabilityError` | `population` | `increment.estimation.assignment_integrity`, `increment.readouts._common` |
| `kernel.context.closed` | `InvalidRequestError` | — | `increment.kernel.contracts` |
| `kernel.context.undeclared_stream` | `InvalidRequestError` | `stream` | `increment.kernel.contracts` |
| `kernel.definitions.catalog_missing` | `InvalidRequestError` | `path` | `increment.kernel.contracts` |
| `kernel.definitions.catalog_stale` | `InvalidRequestError` | `path` | `increment.kernel.contracts` |
| `kernel.method.duplicate_instance` | `InvalidRequestError` | `method_instance_id` | `increment.kernel.contracts` |
| `kernel.method.implementation_ref_missing` | `InvalidRequestError` | `field` | `increment.kernel.contracts` |
| `kernel.request.capture_not_supported` | `InvalidRequestError` | — | `increment.kernel.contracts` |
| `kernel.request.datetime_timezone` | `InvalidRequestError` | `field` | `increment.kernel.contracts` |
| `kernel.request.invalid` | `InvalidRequestError` | `field` | `increment.kernel.contracts` |
| `kernel.request.population_scope_only` | `InvalidRequestError` | — | `increment.kernel.contracts` |
| `kernel.request.scope_mismatch` | `InvalidRequestError` | `field`, `kind` | `increment.kernel.contracts` |
| `kernel.request.unportable` | `InvalidRequestError` | `field`, `kind` | `increment.kernel.contracts` |
| `kernel.run.provider_execution_unavailable` | `InvalidRequestError` | — | `increment.kernel.contracts` |
| `kernel.scope.family_mismatch` | `CapabilityError` | `experiment_id`, `view` | `increment._runnability` |
| `kernel.scope.invalid` | `InvalidRequestError` | `field` | `increment.kernel.contracts` |
| `kernel.snapshot.integrity` | `InvalidRequestError` | `snapshot_id`, `stream` | `increment.kernel.contracts` |
| `logged_policy.inference.adaptive_logging_unsupported` | `CapabilityError` | `logging_policy_keys`, `logging_policy_versions`, `update_batches` | `increment.logged_policy._refusals` |
| `logged_policy.inference.alpha` | `InvalidRequestError` | `alpha` | `increment.logged_policy._refusals` |
| `logged_policy.inference.cluster_floor` | `CapabilityError` | `ess_by_time`, `max_weight_by_time`, `min_propensity_by_time`, `minimum_total_units`, `n_units` | `increment.logged_policy._refusals` |
| `logged_policy.inference.unrepresentable` | `CapabilityError` | `estimate`, `nonfinite`, `reference_value`, `se`, `target_value` | `increment.logged_policy._refusals` |
| `logged_policy.policy.json_context_key` | `CapabilityError` | `key_types`, `policy_id`, `route`, `version` | `increment.logged_policy._refusals` |
| `logged_policy.registry.duplicate` | `InvalidRequestError` | `policy_id`, `version` | `increment.logged_policy._refusals` |
| `logged_policy.support.action_unsupported` | `CapabilityError` | `action`, `decision_index`, `policy_id`, `policy_probability`, `unit_id`, `version` | `increment.logged_policy._refusals` |
| `logged_policy.support.context_unsupported` | `CapabilityError` | `context`, `context_key`, `policy_id`, `version` | `increment.logged_policy._refusals` |
| `logged_policy.support.ess_floor` | `CapabilityError` | `decision_index`, `ess`, `floor`, `policy_id`, `reference_ess_by_time`, `target_ess_by_time`, `version` | `increment.logged_policy._refusals` |
| `logged_policy.support.policy_not_normalized` | `CapabilityError` | `context`, `policy_id`, `probabilities`, `version` | `increment.logged_policy._refusals` |
| `logged_policy.trace.admission_route` | `InvalidRequestError` | `alternatives`, `supplied` | `increment.logged_policy._refusals` |
| `logged_policy.trace.boundary_awareness_mixed` | `InvalidRequestError` | `decision_index`, `decision_time`, `reward_observation_boundary`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.boundary_before_decision` | `InvalidRequestError` | `decision_index`, `decision_time`, `reward_observation_boundary`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.candidate_set_mismatch` | `InvalidRequestError` | `candidate_actions`, `chosen_action`, `decision_index`, `expected_candidate_actions`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.context_value_unsupported` | `InvalidRequestError` | `value_type` | `increment.logged_policy._refusals` |
| `logged_policy.trace.decision_time_awareness_mixed` | `InvalidRequestError` | `decision_index`, `decision_time`, `reference_decision_index`, `reference_decision_time`, `reference_unit_id`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.empty` | `InvalidRequestError` | `constructors`, `n_records` | `increment.logged_policy._refusals` |
| `logged_policy.trace.incomplete_horizon` | `InvalidRequestError` | `horizon`, `observed_horizon`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.logging_distribution_keys` | `InvalidRequestError` | `actions`, `decision_index`, `expected_actions`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.logging_distribution_misaligned` | `InvalidRequestError` | `n_distributions`, `n_records` | `increment.logged_policy._refusals` |
| `logged_policy.trace.logging_policy_unregistered` | `InvalidRequestError` | `decision_index`, `logging_policy_id`, `logging_policy_version`, `registered`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.missing_columns` | `InvalidRequestError` | `missing` | `increment.logged_policy._refusals` |
| `logged_policy.trace.open_reward` | `InvalidRequestError` | `decision_index`, `reward_observation_boundary`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.propensity_below_floor` | `InvalidRequestError` | `action`, `context`, `decision_index`, `floor`, `logging_policy_id`, `logging_policy_version`, `propensity`, `route`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.propensity_missing` | `InvalidRequestError` | `decision_index`, `logging_policy_id`, `logging_policy_version`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.propensity_nonfinite` | `InvalidRequestError` | `decision_index`, `propensity`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.propensity_out_of_range` | `InvalidRequestError` | `decision_index`, `propensity`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.propensity_stale` | `InvalidRequestError` | `chosen_action`, `decision_index`, `logged`, `logging_policy_id`, `logging_policy_version`, `registered`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.registry_audit_required` | `InvalidRequestError` | `constructors` | `increment.logged_policy._refusals` |
| `logged_policy.trace.reward_nonfinite` | `InvalidRequestError` | `decision_index`, `reward`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.unit_id_missing` | `InvalidRequestError` | `row`, `unit_id` | `increment.logged_policy._refusals` |
| `logged_policy.trace.unordered` | `InvalidRequestError` | `decision_indices`, `decision_times`, `unit_id` | `increment.logged_policy._refusals` |
| `model.field.length` | `InvalidRequestError` | `constraint`, `field`, `model`, `value` | `increment.errors`, `increment.winsor` |
| `model.field.literal` | `InvalidRequestError` | `constraint`, `field`, `model`, `value` | `increment.errors`, `increment.winsor` |
| `model.field.missing` | `InvalidRequestError` | `field`, `model` | `increment.errors`, `increment.winsor` |
| `model.field.nonfinite` | `InvalidRequestError` | `field`, `model`, `value` | `increment.errors`, `increment.winsor` |
| `model.field.pattern` | `InvalidRequestError` | `constraint`, `field`, `model`, `value` | `increment.errors`, `increment.winsor` |
| `model.field.range` | `InvalidRequestError` | `constraint`, `field`, `model`, `value` | `increment.errors`, `increment.winsor` |
| `model.field.type` | `InvalidRequestError` | `constraint`, `field`, `model`, `value` | `increment.errors`, `increment.winsor` |
| `model.field.unknown` | `InvalidRequestError` | `field`, `model` | `increment.errors`, `increment.winsor` |
| `moments.count_field_missing` | `WireFormatError` | `field` | `increment.sources` |
| `moments.count_not_integer` | `WireFormatError` | `field`, `value` | `increment.sources` |
| `moments.count_out_of_range` | `WireFormatError` | `field`, `n`, `value` | `increment.sources` |
| `moments.cuped_evidence.duplicate` | `WireFormatError` | `method`, `metric`, `population` | `increment._cuped_evidence` |
| `moments.cuped_evidence.invalid` | `WireFormatError` | `reason` | `increment._cuped_evidence` |
| `moments.cuped_evidence.unsupported` | `WireFormatError` | `field`, `received`, `supported` | `increment._cuped_evidence` |
| `moments.denominator_third_moment_missing` | `WireFormatError` | `group_id`, `metric`, `version` | `increment.sources` |
| `moments.duplicate_key` | `WireFormatError` | `key` | `increment.sources` |
| `moments.format.invalid` | `WireFormatError` | `received`, `required` | `increment.sources` |
| `moments.format.mixed` | `WireFormatError` | `required`, `seen` | `increment.sources` |
| `moments.format.unsupported_future` | `WireFormatError` | `received`, `required` | `increment.sources` |
| `moments.format.unsupported_legacy` | `WireFormatError` | `received`, `required` | `increment.sources` |
| `moments.plan.conflict` | `WireFormatError` | `constraint`, `field`, `value` | `increment.sources` |
| `moments.plan.invalid` | `WireFormatError` | `constraint`, `field`, `value` | `increment.sources` |
| `moments.rows.duplicate` | `WireFormatError` | `key` | `increment.sources` |
| `moments.rows.experiment_conflict` | `WireFormatError` | `experiments` | `increment.sources` |
| `moments.summaries.invalid` | `InvalidRequestError` | `field`, `group_id`, `metric`, `value` | `increment.summaries` |
| `moments.summaries.unsupported` | `CapabilityError` | `feature`, `route` | `increment.summaries` |
| `moments.v7_rows_missing_winsorization_fields` | `WireFormatError` | `missing` | `increment.sources` |
| `moments.winsorization.metadata_conflict` | `CapabilityError` | `actual_bound`, `actual_n`, `actual_percentile`, `expected_bound`, `expected_percentile`, `metric`, `side` | `increment.sources` |
| `moments.winsorization.metadata_missing` | `CapabilityError` | `field`, `metric`, `value` | `increment.sources` |
| `moments.winsorization.metadata_shape` | `CapabilityError` | `field`, `metric`, `value` | `increment.sources` |
| `package.builtin_provider_unavailable` | `CodedError` | `count`, `feature`, `group`, `provider`, `reason` | `increment.errors`, `increment.kernel.ports`, `increment.query.artifact_ports` |
| `plan.compile_configs_template_mismatch` | `InvalidRequestError` | `metric` | `increment.plan` |
| `plan.compile_configs_wrong_type` | `InvalidRequestError` | — | `increment.plan` |
| `plan.configs_contain_one` | `InvalidRequestError` | — | `increment.plan` |
| `plan.configs_keys_match` | `InvalidRequestError` | — | `increment.plan` |
| `plan.corrected_encouragement_breakout` | `UnsupportedRequestError` | — | `increment.plan` |
| `plan.duplicate_metric_name` | `InvalidRequestError` | `duplicate_names` | `increment.plan` |
| `plan.encouragement.margin` | `UnsupportedRequestError` | `metric` | `increment.plan` |
| `plan.encouragement.sequential_cuped` | `UnsupportedRequestError` | `inference`, `metric` | `increment.plan` |
| `plan.frame.override` | `InvalidRequestError` | `field`, `metric` | `increment.decision` |
| `plan.frame.secondaries` | `InvalidRequestError` | `derived`, `explicit` | `increment.decision` |
| `plan.frame_contrast_does` | `UnsupportedRequestError` | — | `increment.plan` |
| `plan.guardrail.direction` | `InvalidRequestError` | `metric` | `increment.decision` |
| `plan.metric.report_only` | `InvalidRequestError` | `metric`, `role` | `increment.decision` |
| `plan.metric_cuped_methods` | `UnsupportedRequestError` | `inference`, `metric` | `increment.plan` |
| `plan.metric_frame_contrast` | `InvalidRequestError` | `metric` | `increment.plan` |
| `plan.metric_informative_priors` | `UnsupportedRequestError` | `inference`, `metric` | `increment.plan` |
| `plan.metric_priors_relative` | `UnsupportedRequestError` | `metric`, `prior` | `increment.plan` |
| `plan.metrics.unknown` | `InvalidRequestError` | `declared`, `path`, `unknown` | `increment.decision` |
| `plan.observational.relative_margin` | `UnsupportedRequestError` | `metrics` | `increment.plan` |
| `plan.unsupported_decision_plan` | `InvalidRequestError` | `path` | `increment.plan` |
| `power.alpha_finite` | `InvalidRequestError` | `alpha` | `increment.power.sequential` |
| `power.baseline.avg_cluster_size` | `InvalidRequestError` | `avg_cluster_size` | `increment.power.core` |
| `power.baseline.cluster_analyzed_mean` | `InvalidRequestError` | `analyzed_mean`, `avg_cluster_size`, `cluster_participation`, `route`, `trigger_rate` | `increment.power.core` |
| `power.baseline.cluster_icc` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.cluster_icc_avg` | `InvalidRequestError` | `analyzed_mean`, `cluster_icc`, `route` | `increment.power.core` |
| `power.baseline.cluster_participation` | `InvalidRequestError` | `cluster_participation`, `route` | `increment.power.core` |
| `power.baseline.cluster_participation_triggered` | `InvalidRequestError` | `avg_cluster_size`, `cluster_participation`, `route`, `trigger_rate` | `increment.power.core` |
| `power.baseline.cluster_participation_untriggered` | `InvalidRequestError` | `cluster_participation`, `route`, `trigger_rate` | `increment.power.core` |
| `power.baseline.cluster_size_cv` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.compliance` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.confidence` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.confidence_produces_non` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.cuped_rho` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.icc` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.mean` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.mean_den` | `InvalidRequestError` | `mean_den` | `increment.power.core` |
| `power.baseline.summarystats_bound_pilot` | `InvalidRequestError` | `n` | `increment.power.core` |
| `power.baseline.trigger_rate` | `InvalidRequestError` | — | `increment.power.core` |
| `power.baseline.var_zero_variance` | `InvalidRequestError` | — | `increment.power.core` |
| `power.binomial_arm_ceiling_below_smallest_design` | `InvalidRequestError` | `allocation`, `conversion_inference`, `max_arm_size`, `n_c`, `n_t` | `increment.power.core` |
| `power.binomial_arm_ceiling_exceeded` | `InvalidRequestError` | `max_arm_size`, `n_c`, `n_t` | `increment.power.core` |
| `power.binomial_enclosure_requires_counts` | `InvalidRequestError` | `metric_type` | `increment.power.core` |
| `power.binomial_probability_unresolved` | `InvalidRequestError` | `error`, `lower`, `point`, `tolerance`, `upper` | `increment.power._binomial`, `increment.power.core` |
| `power.binomial_replay_bound_exceeded` | `InvalidRequestError` | `cells`, `conversion_inference`, `dense_min_count`, `max_arm_size`, `max_cells`, `n_c`, `n_t`, `p_c`, `p_t`, `planned_dense_min_count`, `power`, `power_reached` | `increment.power.core` |
| `power.binomial_size_search_unreachable` | `InvalidRequestError` | `conversion_inference`, `max_arm_size`, `maximum_power`, `n_per_arm`, `power` | `increment.power.core` |
| `power.binomial_tail_level_unrepresentable` | `InvalidRequestError` | `alpha`, `beta`, `cause`, `conversion_inference`, `decided_from`, `margin`, `n_c`, `n_t`, `p_c`, `scope`, `solver_floor`, `tail_alpha` | `increment.power.core` |
| `power.boundary_crossing_quadrature` | `InvalidRequestError` | `reason` | `increment.power.sequential` |
| `power.bounded_metric.baseline` | `InvalidRequestError` | `mean`, `metric_type` | `increment.power.core` |
| `power.bounded_metric.rate_above_one` | `InvalidRequestError` | `implied_rate`, `mean`, `metric_type`, `role` | `increment.power.core` |
| `power.cluster.too_few_clusters` | `InvalidRequestError` | `k_c`, `k_t`, `k_total`, `recruited_k_total`, `route` | `increment.power.core` |
| `power.core.n_per_arm_int` | `InvalidRequestError` | `n_per_arm` | `increment.power.core` |
| `power.core.n_per_arm_min` | `InvalidRequestError` | `metric_type`, `minimum`, `n_per_arm` | `increment.power.core` |
| `power.cuped_linear_unavailable` | `InvalidRequestError` | — | `increment.power.core` |
| `power.curve.n_per_arm_int` | `InvalidRequestError` | `value` | `increment.power.curve` |
| `power.curve.n_per_arm_min` | `InvalidRequestError` | `value` | `increment.power.curve` |
| `power.empty` | `InvalidRequestError` | `name` | `increment.power.curve` |
| `power.exactly_one_relative` | `InvalidRequestError` | — | `increment.power.curve` |
| `power.finite` | `InvalidRequestError` | `name`, `value` | `increment.power._validation` |
| `power.fractions_nonempty` | `InvalidRequestError` | — | `increment.estimation.sequential` |
| `power.information_fractions_lie` | `InvalidRequestError` | `fr` | `increment.estimation.sequential` |
| `power.information_fractions_strictly` | `InvalidRequestError` | `fr` | `increment.estimation.sequential` |
| `power.log_exp_needs` | `InvalidRequestError` | `x` | `increment.power.core` |
| `power.max_workers` | `InvalidRequestError` | — | `increment.power.curve` |
| `power.max_workers_positive` | `InvalidRequestError` | — | `increment.power.curve` |
| `power.mde_relative_unavailable` | `InvalidRequestError` | — | `increment.power.core` |
| `power.mde_unavailable_reason` | `InvalidRequestError` | `mde_relative`, `mde_unavailable_reason` | `increment.power.core` |
| `power.minimum_detectable_effect.design_search_minimum` | `InvalidRequestError` | `estimate`, `target` | `increment.power.core` |
| `power.minimum_detectable_effect.numerical_resolution` | `InvalidRequestError` | `direction`, `power_enclosure`, `stopping_reason`, `target_power`, `unresolved_interval` | `increment.power._search`, `increment.power.core`, `increment.power.sequential` |
| `power.minimum_detectable_effect.unattainable` | `InvalidRequestError` | `direction`, `limiting_condition`, `maximum_power`, `power_basis`, `target_power` | `increment.power.core` |
| `power.minimum_detectable_effect.unrepresentable` | `InvalidRequestError` | `direction`, `limiting_condition`, `target_power` | `increment.power.core` |
| `power.need_least_segments` | `InvalidRequestError` | `k` | `increment.power.core` |
| `power.nodes_multiple` | `InvalidRequestError` | `nodes`, `panel_nodes` | `increment.power.sequential` |
| `power.noncentral_t_unresolved` | `InvalidRequestError` | `dof`, `nc`, `tail_alpha` | `increment.power._shared` |
| `power.planned_looks` | `InvalidRequestError` | `planned_looks` | `increment.power._validation` |
| `power.planned_looks_positive` | `InvalidRequestError` | `planned_looks` | `increment.power._validation` |
| `power.power.check` | `InvalidRequestError` | `power` | `increment.power.core` |
| `power.power.effective_var` | `InvalidRequestError` | `effective_var` | `increment.power.core` |
| `power.power_bounds_enclose` | `InvalidRequestError` | `power`, `power_lower`, `power_upper` | `increment.power.core` |
| `power.power_bounds_paired` | `InvalidRequestError` | `power_lower`, `power_upper` | `increment.power.core` |
| `power.power_bounds_required` | `InvalidRequestError` | — | `increment.power.core` |
| `power.power_is_lower_bound` | `InvalidRequestError` | `power`, `power_lower` | `increment.power.core` |
| `power.power_solvers_relative` | `InvalidRequestError` | — | `increment.power._shared`, `increment.power.core`, `increment.power.pairwise` |
| `power.procedure_armplanningprocedure` | `InvalidRequestError` | `procedure_type` | `increment.power.core` |
| `power.q_a` | `InvalidRequestError` | — | `increment.power.pairwise` |
| `power.q_a_q` | `InvalidRequestError` | `q_a_plus_q_b` | `increment.power.pairwise` |
| `power.q_b` | `InvalidRequestError` | — | `increment.power.pairwise` |
| `power.quantile_baseline.cuped_unsupported` | `InvalidRequestError` | `cuped_rho` | `increment.power.core` |
| `power.quantile_baseline.metric_not_quantile` | `InvalidRequestError` | `metric`, `metric_type` | `increment.power.core` |
| `power.quantile_size_search_unreachable` | `InvalidRequestError` | `limiting_condition`, `maximum_power`, `power` | `increment.power.core` |
| `power.r_a_above` | `InvalidRequestError` | `r_a`, `r_b`, `theta` | `increment.power.pairwise` |
| `power.r_a_below` | `InvalidRequestError` | `r_a`, `r_b`, `theta` | `increment.power.pairwise` |
| `power.r_a_r` | `InvalidRequestError` | `r_a`, `r_b` | `increment.power.pairwise` |
| `power.relative_lift_lies` | `InvalidRequestError` | `null_lift`, `relative_lift` | `increment.power.core` |
| `power.relative_lift_lies_above_null` | `InvalidRequestError` | `null_lift`, `relative_lift` | `increment.power.core` |
| `power.require_relative_domain` | `InvalidRequestError` | `name`, `value` | `increment.power._validation` |
| `power.sample_size_detecting` | `InvalidRequestError` | `null_lift`, `relative_lift` | `increment.power.core` |
| `power.scalar_numeric_sequence` | `InvalidRequestError` | `name` | `increment.power.curve` |
| `power.se_full_finite` | `InvalidRequestError` | `se_full` | `increment.power.sequential` |
| `power.segment_clustered_baseline` | `InvalidRequestError` | `k_total`, `n_c`, `n_t`, `segment` | `increment.power.pairwise` |
| `power.segment_pairwise_achieved` | `InvalidRequestError` | `null_lift` | `increment.power.pairwise` |
| `power.segment_pairwise_achieved_n_per_arm_too_small` | `InvalidRequestError` | `exc`, `n_per_arm`, `q_a`, `q_b` | `increment.power.pairwise` |
| `power.segment_pairwise_minimum` | `InvalidRequestError` | `null_lift` | `increment.power.pairwise` |
| `power.segment_pairwise_required` | `InvalidRequestError` | `null_lift` | `increment.power.pairwise` |
| `power.segment_pairwise_solvers` | `InvalidRequestError` | `unsupported` | `increment.power.pairwise` |
| `power.segment_share_n` | `InvalidRequestError` | `allocation`, `min_n_total_seg`, `n_total`, `n_total_seg`, `q` | `increment.power.pairwise` |
| `power.sequential_design_more` | `InvalidRequestError` | `power` | `increment.power.core` |
| `power.sequential_mde.compliance_unrepresentable` | `InvalidRequestError` | `compliance`, `mde_relative`, `target_power` | `increment.power.sequential` |
| `power.sequential_mde.decrease_unrepresentable` | `InvalidRequestError` | `compliance`, `mde_relative`, `target_power` | `increment.power.sequential` |
| `power.sequential_mde.design_search_minimum` | `InvalidRequestError` | `estimate`, `target` | `increment.power.sequential` |
| `power.sequential_mde.unrepresentable` | `InvalidRequestError` | `mde_theta`, `se`, `target_power` | `increment.power.sequential` |
| `power.sequential_sample_size` | `InvalidRequestError` | `n_per_arm`, `reason` | `increment.power.core` |
| `power.sequential_sample_size_target_inside_enclosure` | `InvalidRequestError` | `lower`, `n_per_arm`, `upper` | `increment.power.core` |
| `power.size_design_relative` | `InvalidRequestError` | `null_lift`, `relative_lift` | `increment.power.core` |
| `power.solved_too_small` | `InvalidRequestError` | `exc`, `n_total`, `q_a`, `q_b` | `increment.power.pairwise` |
| `power.supports_fixed_horizon` | `InvalidRequestError` | `caller` | `increment.power.pairwise` |
| `power.switchback.baseline` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.bounded_model_required` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.delta` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.n` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.procedure` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.result` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.roster` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.switchback.target_power` | `InvalidRequestError` | `constraint`, `field`, `message`, `route`, `value` | `increment.power.switchback` |
| `power.tau_b` | `InvalidRequestError` | `tau_b` | `increment.power.core` |
| `power.theta_contains_non` | `InvalidRequestError` | — | `increment.power.core` |
| `power.theta_var_one` | `InvalidRequestError` | `theta_shape`, `var_shape` | `increment.power.core` |
| `power.theta_var_same` | `InvalidRequestError` | `theta_shape`, `var_shape` | `increment.power.core` |
| `power.unit_cycle.input` | `InvalidRequestError` | `constraint`, `detail`, `field`, `route`, `value` | `increment.power.unit_cycle` |
| `power.unit_cycle.law` | `InvalidRequestError` | `constraint`, `detail`, `field`, `route`, `value` | `increment.power.unit_cycle` |
| `power.unit_cycle.numerical` | `InvalidRequestError` | `constraint`, `detail`, `field`, `route`, `value` | `increment.power.unit_cycle` |
| `power.unit_cycle.procedure` | `InvalidRequestError` | `constraint`, `detail`, `field`, `route`, `value` | `increment.power.unit_cycle` |
| `power.unit_cycle.result` | `InvalidRequestError` | `constraint`, `detail`, `field`, `route`, `value` | `increment.power.unit_cycle` |
| `power.units_per_week_finite_positive` | `InvalidRequestError` | — | `increment.power.curve` |
| `query.artifact_reader.artifact_moment.manifest_no_metric` | `InvalidRequestError` | — | `increment.query.artifact_reader` |
| `query.artifact_reader.artifact_moment.verification_lazy_digest` | `InvalidRequestError` | — | `increment.query.artifact_reader` |
| `query.artifact_reader.snapshot_execution_returned` | `InvalidRequestError` | `type_name` | `increment.query.artifact_reader` |
| `query.builders.asof_group_summary` | `InvalidRequestError` | `name`, `threshold_days` | `increment.query.builders` |
| `query.builders.asof_group_summary_completed_requires_uptake_window` | `InvalidRequestError` | — | `increment.query.builders` |
| `query.builders.asof_group_summary_den_panel_retention` | `InvalidRequestError` | — | `increment.query.builders` |
| `query.builders.asof_group_summary_metric_type_not_implemented` | `UnsupportedRequestError` | `metric_type` | `increment.query.builders` |
| `query.builders.check_cluster.experiment_arm_no` | `InvalidRequestError` | `experiment_name`, `group` | `increment.query.builders` |
| `query.builders.cluster_column_missing` | `CapabilityError` | `cluster`, `columns`, `experiment`, `remedy`, `table` | `increment.query.builders` |
| `query.builders.cluster_size_imbalance` | `CapabilityError` | `cluster`, `control_group`, `control_mean_size`, `experiment`, `gap`, `target_group`, `target_mean_size`, `threshold` | `increment.query.builders` |
| `query.builders.cluster_total_grain` | `CapabilityError` | `caller`, `cluster`, `grain` | `increment.query.builders` |
| `query.builders.cluster_undeclared` | `CapabilityError` | `experiment`, `operation` | `increment.query.builders` |
| `query.builders.cohort_group_summary` | `UnsupportedRequestError` | `metric_type` | `increment.query.builders` |
| `query.builders.cohort_group_summary_unbounded_band` | `InvalidRequestError` | `name`, `threshold_days` | `increment.query.builders` |
| `query.builders.daily_state_not_implemented` | `UnsupportedRequestError` | `metric_type` | `increment.query.builders` |
| `query.builders.group_summary_declared` | `InvalidRequestError` | `cluster`, `columns` | `increment.query.builders` |
| `query.builders.group_summary_ratio` | `InvalidRequestError` | `columns`, `ratio_names` | `increment.query.builders` |
| `query.builders.group_summary_uptake` | `InvalidRequestError` | `columns` | `increment.query.builders` |
| `query.builders.invalid_day_boundary` | `InvalidRequestError` | `day_boundary` | `increment.query.builders` |
| `query.builders.part_ratiometric` | `InvalidRequestError` | `part` | `increment.query.builders` |
| `query.builders.ratiometric_den_stats` | `InvalidRequestError` | — | `increment.query.builders` |
| `query.builders.site_volume_metric_type` | `CapabilityError` | `metric`, `metric_type`, `reason` | `increment.query.builders` |
| `query.builders.trigger_population.cutoff_not_aware` | `InvalidRequestError` | `field`, `value` | `increment.query.builders` |
| `query.builders.union_event_horizon` | `InvalidRequestError` | — | `increment.query.builders` |
| `query.builders.unit_totals_by` | `InvalidRequestError` | — | `increment.query.builders` |
| `query.builders.unit_totals_properties` | `InvalidRequestError` | — | `increment.query.builders` |
| `query.builders.unknown_aggregation` | `InvalidRequestError` | `aggregation` | `increment.query.builders`, `increment.query.calendar` |
| `query.builders.unknown_part_ratiometric` | `InvalidRequestError` | `part` | `increment.query.builders` |
| `query.builders.window_bound_stats` | `UnsupportedRequestError` | `metric_type` | `increment.query.builders` |
| `query.calendar.period_moments_by` | `InvalidRequestError` | `dim_cols` | `increment.query.calendar` |
| `query.calendar.period_start_unknown_grain` | `InvalidRequestError` | `grain` | `increment.query.calendar` |
| `query.calendar.population_unit_id` | `InvalidRequestError` | `columns` | `increment.query.calendar` |
| `query.calendar.range` | `CapabilityError` | `end_edge`, `span`, `start` | `increment.query.calendar` |
| `query.cuped.general_qr_required` | `CapabilityError` | `metric`, `reason`, `route` | `increment.query._cuped_cells` |
| `query.cuped.magnitude_range` | `CapabilityError` | `arm`, `bound`, `component`, `metric` | `increment.query._cuped_cells` |
| `query.dims.dim_source_fact` | `InvalidRequestError` | `name` | `increment.query.dims` |
| `query.dims.dim_source_multiple` | `InvalidRequestError` | `entity`, `name` | `increment.query.dims` |
| `query.dims.dim_source_overlapping` | `InvalidRequestError` | `entity`, `name` | `increment.query.dims` |
| `query.dims.dim_source_property` | `InvalidRequestError` | `column`, `name` | `increment.query.dims` |
| `query.dims.dim_source_two` | `InvalidRequestError` | `entity`, `name` | `increment.query.dims` |
| `query.fact_resolution.breakout_property_could` | `InvalidRequestError` | `property`, `unit` | `increment.query.fact_resolution` |
| `query.fact_resolution.breakout_property_references` | `InvalidRequestError` | `property`, `source` | `increment.query.fact_resolution` |
| `query.fact_resolution.fact_declared_any` | `InvalidRequestError` | `available`, `fact_name` | `increment.query.fact_resolution` |
| `query.fact_resolution.fact_source_property` | `InvalidRequestError` | `name`, `prop_name` | `increment.query.fact_resolution` |
| `query.fact_resolution.fact_source_property_deletes_value_column` | `InvalidRequestError` | `name`, `prop_column`, `prop_name` | `increment.query.fact_resolution` |
| `query.fact_resolution.fact_source_property_reads_canonical_column` | `InvalidRequestError` | `name`, `prop_column`, `prop_name` | `increment.query.fact_resolution` |
| `query.fact_resolution.pass_exactly_one` | `InvalidRequestError` | `constructor`, `control`, `design` | `increment.query.fact_resolution` |
| `query.integrity.cluster_conflict` | `InvalidRequestError` | `cluster`, `experiment`, `units` | `increment.query.integrity` |
| `query.integrity.cluster_labels` | `InvalidRequestError` | `cluster`, `constraint`, `experiment`, `kind`, `value` | `increment.query.integrity` |
| `query.integrity.mixed_assignment_units` | `InvalidRequestError` | `finding`, `mixed_count`, `route`, `unassigned_count` | `increment.query.integrity` |
| `query.integrity.trigger_arm_missing` | `InvalidRequestError` | `rates`, `route`, `silent`, `trigger` | `increment._trigger_integrity` |
| `query.integrity.unassigned_assignment_units` | `InvalidRequestError` | `finding`, `mixed_count`, `route`, `unassigned_count` | `increment.query.integrity` |
| `query.session.artifact_publication.handle_closed` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.materialization_namespace` | `CapabilityError` | `backend` | `increment.query.session` |
| `query.session.qr_scratch_cleanup_failed` | `CapabilityError` | `relations`, `route` | `increment.query.session` |
| `query.session.snapshot.cutoff_not_aware` | `InvalidRequestError` | `field`, `value` | `increment.query.session` |
| `query.session.snapshot.feed_name_invalid` | `InvalidRequestError` | `value` | `increment.query.session` |
| `query.session.snapshot.materialization_failed` | `CapabilityError` | — | `increment.query.session` |
| `query.session.snapshot.source_missing` | `CapabilityError` | — | `increment.query.session` |
| `query.session.snapshot.watermark_not_aware` | `InvalidRequestError` | `feed`, `field`, `value` | `increment.query.session` |
| `query.session.warehouse_artifact.manifest_bound_publication` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.warehouse_artifact.manifest_context_differs` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.warehouse_artifact.manifest_digest_does` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.warehouse_artifact.manifest_references_unwritten` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.warehouse_artifact.manifest_relation_locators` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.warehouse_artifact.publication_cleanup_incomplete` | `InvalidRequestError` | `artifact_id`, `generation_id`, `relations`, `route` | `increment.query.session` |
| `query.session.warehouse_artifact.publication_handle_sealed` | `InvalidRequestError` | — | `increment.query.session` |
| `query.session.warehouse_artifact.publication_state_unknown` | `InvalidRequestError` | `artifact_id`, `generation_id`, `relations`, `route`, `tombstoned` | `increment.query.session` |
| `query.session.warehouse_artifact.unknown_digest_format` | `InvalidRequestError` | `digest_format` | `increment.query.session` |
| `query.session.warehouse_artifact.unknown_relation_role` | `InvalidRequestError` | `role` | `increment.query.session` |
| `query.session.warehouse_artifact.verification_unavailable` | `InvalidRequestError` | `digest_format`, `relation`, `route` | `increment.query.session` |
| `readout.adjustment.absolute_unadjusted` | `InvalidRequestError` | — | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.adjust` |
| `readout.alwaysvalid_under_encouragement` | `InvalidRequestError` | `method` | `increment.readouts._common` |
| `readout.arms.no_treatment` | `InvalidRequestError` | `control_group`, `observed_arms` | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.asof.cluster` | `CapabilityError` | `cluster` | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.asof.segment_unsupported` | `UnsupportedRequestError` | `by` | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.assignment.estimands` | `InvalidRequestError` | `estimands`, `mechanism` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.cell.key_invalid` | `InvalidRequestError` | `fields`, `kind` | `increment.declarations`, `increment.estimation.readout_types` |
| `readout.cell.missing_arm` | `InvalidRequestError` | `group_id`, `metric`, `observed_arms` | `increment.estimation.readout_types` |
| `readout.cell.missing_metric_observations` | `InvalidRequestError` | `group_id`, `metric`, `observed_arms` | `increment.estimation.readout_types` |
| `readout.cell.unsupported_request` | `InvalidRequestError` | `reason` | `increment.estimation.readout_types` |
| `readout.collection.concat_cell_collision` | `InvalidRequestError` | `cell_key`, `snapshot_ids` | `increment.estimation.readout_types` |
| `readout.collection.concat_metadata_mismatch` | `InvalidRequestError` | `fields`, `snapshot_ids` | `increment.estimation.readout_types` |
| `readout.collection.concat_type_mismatch` | `InvalidRequestError` | `left`, `right` | `increment.estimation.readout_types` |
| `readout.collection.duplicate_cell_key` | `InvalidRequestError` | `cell_key`, `source_snapshot_id` | `increment.estimation.readout_types` |
| `readout.collection.mutation_unsupported` | `InvalidRequestError` | `model`, `operation` | `increment.estimation.readout_types` |
| `readout.completed_encouragement_inference` | `InvalidRequestError` | `method`, `unbound`, `uptake_window_days` | `increment.readouts._common` |
| `readout.contrast.request` | `InvalidRequestError` | `message` | `increment._evidence_dispatch` |
| `readout.correction.invalid` | `InvalidRequestError` | `correction` | `increment.plan` |
| `readout.design.required` | `InvalidRequestError` | `view` | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.encouragement.asof_completion` | `InvalidRequestError` | `view` | `increment.estimation.encouragement` |
| `readout.encouragement.asof_unbounded` | `InvalidRequestError` | `metrics`, `uptake_window_days` | `increment.estimation.encouragement` |
| `readout.encouragement.cluster_ratio` | `CapabilityError` | `cluster`, `names` | `increment.estimation.encouragement` |
| `readout.encouragement.correction` | `UnsupportedRequestError` | — | `increment.plan` |
| `readout.encouragement.margin` | `UnsupportedRequestError` | `names` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.encouragement` |
| `readout.encouragement.retention` | `CapabilityError` | `names` | `increment.estimation.encouragement`, `increment.semantics.design` |
| `readout.encouragement.value_scale` | `UnsupportedRequestError` | — | `increment.estimation.encouragement`, `increment.readouts._common` |
| `readout.estimands.unknown` | `InvalidRequestError` | `supported`, `unknown` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.encouragement` |
| `readout.estimate_lift_every` | `UnsupportedRequestError` | `failures` | `increment.readouts._common` |
| `readout.failure.hypothesis_key_collision` | `InvalidRequestError` | `cell_keys`, `hypothesis` | `increment.estimation.readout_types` |
| `readout.inference.disjoint_slices` | `UnsupportedRequestError` | `inference`, `view` | `increment.breakout.estimates` |
| `readout.inference.observational_run` | `UnsupportedRequestError` | `route` | `increment.plan` |
| `readout.inference.sequential_view` | `UnsupportedRequestError` | `view` | `increment.estimation.encouragement` |
| `readout.legacy.sampling_unreconstructible` | `InvalidRequestError` | `cell_key`, `missing_fields`, `remedy` | `increment.estimation.readout_types` |
| `readout.legacy.scope_unreconstructible` | `UnsupportedRequestError` | `missing_fields`, `remedy`, `source` | `increment.estimation.readout_types` |
| `readout.margin.breakout` | `UnsupportedRequestError` | `names` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.metric.percentile_winsorization` | `CapabilityError` | `names`, `view` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.metric.quantile_alternative` | `UnsupportedRequestError` | `metric`, `route` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.metric.quantile_breakout` | `CapabilityError` | `metric`, `route` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.metric.quantile_grain` | `CapabilityError` | `grain`, `metric` | `increment.sources` |
| `readout.metric_declare_non` | `UnsupportedRequestError` | `combined` | `increment.readouts._common` |
| `readout.observational.prior` | `InvalidRequestError` | — | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.adjust` |
| `readout.observational.quantile` | `UnsupportedRequestError` | `metric` | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.randomized.value_scale` | `UnsupportedRequestError` | — | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.request.readoutrequest_requires_one_resolved` | `InvalidRequestError` | — | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.roster.undeclared_observed_arms` | `InvalidRequestError` | `analysis_population`, `declared_arms`, `undeclared_arms` | `increment.readouts._common` |
| `readout.run.segment_unsupported` | `UnsupportedRequestError` | `metric` | `increment._readout_request`, `increment.estimation._readout_refusals` |
| `readout.scalar_method_evidence` | `UnsupportedRequestError` | `method`, `route` | `increment._analysis_config` |
| `readout.scope.cell_incomplete` | `InvalidRequestError` | `cell` | `increment.readouts._common` |
| `readout.scope.cell_unresolved` | `InvalidRequestError` | `cell` | `increment.readouts._common` |
| `readout.scope.invalid_axis` | `InvalidRequestError` | `axis`, `value` | `increment.readouts._common` |
| `readout.scope.metric_undeclared` | `InvalidRequestError` | `declared`, `metric` | `increment.readouts._common` |
| `readout.scope.request_not_canonical` | `UnsupportedRequestError` | `fields` | `increment.estimation.readout_types` |
| `readout.scope.source_digest_unavailable` | `UnsupportedRequestError` | `route`, `source` | `increment.estimation.readout_types` |
| `readout.sequential_inference_supported` | `UnsupportedRequestError` | — | `increment.readouts._common` |
| `readout.serialization.duplicate_cell_key` | `WireFormatError` | `cell_key`, `source_snapshot_id` | `increment.estimation.readout_types` |
| `readout.serialization.duplicate_object_key` | `WireFormatError` | `key` | `increment.estimation.readout_types` |
| `readout.serialization.identity_mismatch` | `WireFormatError` | `cell_key`, `reason`, `source_snapshot_id` | `increment.estimation.readout_types` |
| `readout.serialization.model_mismatch` | `WireFormatError` | `collection`, `row_model` | `increment.estimation.readout_types` |
| `readout.serialization.noncanonical_order` | `WireFormatError` | `section` | `increment.estimation.readout_types` |
| `readout.serialization.sequential_snapshot_required` | `WireFormatError` | `checkpoint_ids` | `increment.estimation.readout_types` |
| `readout.serialization.unsupported_version` | `WireFormatError` | `kind`, `received`, `remedy`, `supported` | `increment.estimation.readout_types` |
| `readout.srm_source_declared` | `InvalidRequestError` | — | `increment.readouts._common` |
| `readout.value_scale.invalid` | `InvalidRequestError` | `metric`, `value_scale` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.adjust` |
| `readout.value_scale.null` | `InvalidRequestError` | `metric` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.adjust` |
| `readout.value_scale.unknown_metric` | `InvalidRequestError` | `selected`, `unknown` | `increment._readout_request`, `increment.estimation._readout_refusals`, `increment.estimation.engine` |
| `readout.value_scale_names` | `InvalidRequestError` | `metric_names`, `unknown_value_scale` | `increment.readouts._common` |
| `readout.value_scale_randomized_absolute` | `UnsupportedRequestError` | — | `increment.readouts._common` |
| `readout.view.observational` | `UnsupportedRequestError` | `view` | `increment.estimation.adjust`, `increment.plan` |
| `readout.weight_diagnostics.not_applicable` | `UnsupportedRequestError` | `method` | `increment.estimation.adjust` |
| `report.metric.unsupported` | `CapabilityError` | `errors` | `increment.reporting` |
| `reporting.alpha_finite` | `InvalidRequestError` | `alpha` | `increment.reporting` |
| `reporting.metric_trend.unknown_backend` | `InvalidRequestError` | `backend` | `increment.reporting` |
| `reporting.week_start_monday` | `InvalidRequestError` | `week_start` | `increment.reporting` |
| `results.frame.empty_columns` | `InvalidRequestError` | — | `increment.breakout.projection` |
| `results.frame.summary_unavailable` | `InvalidRequestError` | `model` | `increment.breakout.estimates` |
| `results.frame.unknown_columns` | `InvalidRequestError` | `available`, `unknown` | `increment.breakout.projection` |
| `results.readout.not_retained` | `InvalidRequestError` | `run_id` | `increment._run` |
| `results.readout.scope_required` | `InvalidRequestError` | `run_id`, `scope_hashes` | `increment._run` |
| `results.readout.scope_unknown` | `InvalidRequestError` | `run_id`, `scope_hash` | `increment._run` |
| `sensitivity.confounding.estimand_unsupported` | `UnsupportedRequestError` | `estimand`, `method` | `increment.estimation.confounding` |
| `sensitivity.confounding.metric_type_unavailable` | `UnsupportedRequestError` | `metric` | `increment.estimation.confounding` |
| `sensitivity.confounding.outcome_unsupported` | `UnsupportedRequestError` | `metric_type` | `increment.estimation.confounding` |
| `sensitivity.confounding.ratio_not_representable` | `InvalidRequestError` | `observed_risk_ratio`, `threshold_risk_ratio` | `increment.estimation.confounding` |
| `sensitivity.confounding.risk_ratio_unavailable` | `UnsupportedRequestError` | `relative_unavailable_reason` | `increment.estimation.confounding` |
| `sensitivity.confounding.scale_unsupported` | `UnsupportedRequestError` | `value_scale` | `increment.estimation.confounding` |
| `sensitivity.confounding.sequential_unsupported` | `UnsupportedRequestError` | `inference` | `increment.estimation.confounding` |
| `sensitivity.confounding.threshold_invalid` | `InvalidRequestError` | `threshold_risk_ratio` | `increment.estimation.confounding` |
| `sequential.asymptotic_joint.invalid` | `CapabilityError` | `reason` | `increment.estimation.asymptotic_joint` |
| `sequential.asymptotic_mean.invalid` | `CapabilityError` | `reason` | `increment.estimation.asymptotic_mean` |
| `sequential.compliance.segmented_family_unsupported` | `CapabilityError` | `alternatives`, `family`, `route`, `segments` | `increment.semantics.sequential` |
| `sequential.continuation.legacy` | `CapabilityError` | `reason` | `increment.semantics.sequential`, `increment.sequential_state` |
| `sequential.continuation.rewrite` | `CapabilityError` | `reason` | `increment.sequential_state` |
| `sequential.freeze.dropped` | `CapabilityError` | `reason` | `increment.sequential_state` |
| `sequential.freeze.invalid` | `CapabilityError` | `reason` | `increment.sequential_state` |
| `sequential.memo.invalid` | `InvalidRequestError` | `reason` | `increment.estimation.sequential_result` |
| `sequential.registration.invalid` | `InvalidRequestError` | `reason` | `increment.semantics.sequential` |
| `sequential.registration.mixed_requires_asymptotic_mean` | `CapabilityError` | `metrics` | `increment.estimation.sequential`, `increment.semantics.sequential` |
| `sequential.route.unsupported` | `CapabilityError` | `reason` | `increment.sequential_state` |
| `sequential.source.invalid` | `CapabilityError` | `reason` | `increment.sequential_state` |
| `sequential.transform.unpredictable` | `CapabilityError` | `reason` | `increment.sequential_state` |
| `sequential.wire.rational_invalid` | `InvalidRequestError` | `field`, `limit_digits`, `reason`, `route_forward` | `increment.decision_wire`, `increment.semantics.rational` |
| `simulate.bandit_loggers.count_positive` | `InvalidRequestError` | `name`, `value` | `increment.simulate.bandit_loggers` |
| `simulate.bandit_loggers.epsilon` | `InvalidRequestError` | `epsilon` | `increment.simulate.bandit_loggers` |
| `simulate.bandit_loggers.reward_mean_out_of_range` | `InvalidRequestError` | `action`, `mean`, `x` | `increment.simulate.bandit_loggers` |
| `simulate.cluster_dgp.invalid_scenario` | `InvalidRequestError` | `reason` | `increment.simulate._cluster_reconstruction`, `increment.simulate.cluster_dgp` |
| `simulate.cluster_dgp.overlap_roster_unavailable` | `InvalidRequestError` | — | `increment.simulate.cluster_dgp` |
| `simulate.dgp.binomial_fixture_counts` | `InvalidRequestError` | `n_c`, `n_t`, `x_c`, `x_t` | `increment.simulate.dgp` |
| `simulate.dgp.conversion_lift_unattainable` | `InvalidRequestError` | `lift`, `p_control`, `target` | `increment.simulate.dgp` |
| `simulate.dgp.count_rate_unrepresentable` | `InvalidRequestError` | `reason`, `route` | `increment.simulate.dgp` |
| `simulate.dgp.scenario.true_lift_finite` | `InvalidRequestError` | `metric`, `value` | `increment.simulate.dgp` |
| `simulate.dgp.switchback.extra_fields` | `InvalidRequestError` | `fields` | `increment.simulate.dgp` |
| `simulate.dgp.switchback.retained_window` | `InvalidRequestError` | `declared_carryover_order`, `observation_steps` | `increment.simulate.dgp` |
| `simulate.dgp.unknown_metric_true` | `InvalidRequestError` | `metric_name` | `increment.simulate.dgp` |
| `simulate.dgp.unsupported_metric_supported` | `InvalidRequestError` | `supported`, `unknown` | `increment.simulate.dgp` |
| `simulate.dgp.uptake_compliance_t` | `InvalidRequestError` | — | `increment.simulate.dgp` |
| `simulate.runner.cate_routing_does` | `InvalidRequestError` | `metric_name` | `increment.simulate.runner` |
| `simulate.runner.replications_invalid` | `InvalidRequestError` | `replications` | `increment.simulate.runner` |
| `source.breakout.assignment_counts` | `CapabilityError` | — | `increment.sources` |
| `source.breakout.compliance_summary` | `CapabilityError` | — | `increment.sources` |
| `source.compliance_arm.cluster_family_declaration_mismatch` | `InvalidRequestError` | `group_id`, `has_family`, `n_clusters` | `increment._evidence_types` |
| `source.compliance_arm.cluster_family_finite` | `InvalidRequestError` | `group_id`, `name`, `value` | `increment._evidence_types` |
| `source.compliance_arm.n_clusters_positive` | `InvalidRequestError` | `group_id`, `n_clusters` | `increment._evidence_types` |
| `source.compliance_arm.n_units_positive` | `InvalidRequestError` | `group_id`, `n_units` | `increment._evidence_types` |
| `source.compliance_arm.partial_cluster_family` | `InvalidRequestError` | `group_id` | `increment._evidence_types` |
| `source.compliance_arm.uptake_total_bounds` | `InvalidRequestError` | `group_id`, `n_units`, `uptake_total` | `increment._evidence_types` |
| `source.compliance_summary.cluster_family_mismatch` | `InvalidRequestError` | `declared_cluster`, `group_id`, `has_family`, `study_id` | `increment._evidence_types` |
| `source.compliance_summary.completed_window_required` | `InvalidRequestError` | — | `increment._evidence_types` |
| `source.compliance_summary.design_mismatch` | `InvalidRequestError` | `cohort`, `control_group`, `design_control_group`, `design_fact`, `design_one_sided`, `design_window_days`, `one_sided`, `study_id`, `window_days` | `increment._evidence_types` |
| `source.compliance_summary.duplicate_arms` | `InvalidRequestError` | `duplicates`, `study_id` | `increment._evidence_types` |
| `source.compliance_summary.invalid_state` | `InvalidRequestError` | `reason` | `increment._evidence_types` |
| `source.compliance_summary.legacy_uptake_state` | `CapabilityError` | `reason`, `study_id` | `increment._evidence_types` |
| `source.frame.cluster_capability` | `CapabilityError` | `cluster`, `feature`, `mechanism`, `metrics`, `route`, `uptake` | `increment._frame_validation`, `increment.frame` |
| `source.frame.cluster_labels` | `InvalidRequestError` | `cluster`, `examples`, `missing`, `reason`, `route`, `spanning` | `increment._frame_validation` |
| `source.frame.column_dtype` | `InvalidRequestError` | `column`, `dtype`, `offending_values` | `increment._frame_validation` |
| `source.frame.compliance_asof` | `CapabilityError` | `as_of`, `operation`, `route` | `increment.frame` |
| `source.frame.compliance_design_mismatch` | `InvalidRequestError` | `requested_design`, `route`, `source_design` | `increment.frame` |
| `source.frame.compliance_no_uptake` | `CapabilityError` | `route`, `source` | `increment.frame` |
| `source.frame.constructor` | `CapabilityError` | `capability`, `metric`, `route`, `value` | `increment._frame_validation` |
| `source.frame.control_invalid` | `InvalidRequestError` | `value` | `increment._frame_validation` |
| `source.frame.control_missing` | `InvalidRequestError` | `control`, `group`, `observed`, `route` | `increment._frame_validation` |
| `source.frame.conversion_not_binary` | `InvalidRequestError` | `column`, `metric`, `route`, `values` | `increment._frame_validation` |
| `source.frame.duplicate_units` | `InvalidRequestError` | `count`, `route`, `units` | `increment._frame_validation` |
| `source.frame.input_type` | `InvalidRequestError` | `constructor`, `input_type`, `route` | `increment._frame_validation` |
| `source.frame.intervention_grain_without_cluster` | `InvalidRequestError` | `intervention_grain` | `increment.frame` |
| `source.frame.metric_missing` | `InvalidRequestError` | `available`, `message`, `missing` | `increment._frame_validation` |
| `source.frame.non_finite` | `InvalidRequestError` | `column`, `count`, `role`, `route` | `increment._frame_validation` |
| `source.frame.null_identity` | `InvalidRequestError` | `column`, `count`, `role`, `route` | `increment._frame_validation` |
| `source.frame.on_unassigned_invalid` | `InvalidRequestError` | `allowed`, `value` | `increment._frame_validation` |
| `source.frame.quantile_daily` | `CapabilityError` | `metric`, `route` | `increment.frame` |
| `source.frame.quantile_no_moments` | `CapabilityError` | `metric`, `route` | `increment.frame`, `increment.sources` |
| `source.frame.ratio_same_column` | `InvalidRequestError` | `column`, `metric` | `increment._frame_validation` |
| `source.frame.retention_daily` | `CapabilityError` | `metric`, `route` | `increment.frame` |
| `source.frame.sequential_exposure_date` | `InvalidRequestError` | `constructor`, `route` | `increment.frame` |
| `source.frame.switchback.assignment` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.switchback.columns` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.switchback.domain` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.switchback.identification` | `InvalidRequestError` | `message` | `increment._study`, `increment.switchback` |
| `source.frame.switchback.metric` | `CapabilityError` | `message` | `increment.switchback` |
| `source.frame.switchback.missingness` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.switchback.numeric` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.switchback.plan` | `CapabilityError` | `message`, `reason` | `increment.decision` |
| `source.frame.switchback.schedule` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.switchback.units` | `InvalidRequestError` | `message` | `increment.switchback` |
| `source.frame.unassigned` | `InvalidRequestError` | `constructor`, `count`, `group`, `route`, `units` | `increment._frame_validation` |
| `source.frame.unit_frame_panel` | `CapabilityError` | `metric`, `route`, `shape` | `increment._frame_validation`, `increment.frame`, `increment.query.artifact_reader` |
| `source.frame.uptake_not_binary` | `InvalidRequestError` | `route`, `uptake`, `values` | `increment._frame_validation` |
| `source.frame.windowed_encouragement_total` | `CapabilityError` | `metric`, `route` | `increment.frame` |
| `source.moments.assignment_counts` | `CapabilityError` | — | `increment.sources` |
| `source.native.compliance_design_mismatch` | `InvalidRequestError` | `expected`, `received` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.native.covariate_ambiguous` | `CapabilityError` | `candidates`, `covariate` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.native.covariate_as_of` | `CapabilityError` | `as_of`, `covariate`, `source` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.native.covariate_dtype` | `CapabilityError` | `covariate`, `dtype`, `source` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.native.covariate_kind_mismatch` | `CapabilityError` | `covariate`, `dtype`, `kind`, `source` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.native.covariate_reserved` | `CapabilityError` | `cluster`, `covariate` | `increment.query.native_contract`, `increment.query.native_source`, `increment.query.source` |
| `source.native.covariate_unresolved` | `CapabilityError` | `covariate`, `source`, `unit` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.native.trigger_evidence_required` | `CapabilityError` | `operation`, `route` | `increment.query._native_refusals`, `increment.query.native_source` |
| `source.panel.day_grain_columns` | `InvalidRequestError` | `columns`, `route` | `increment._frame_validation` |
| `source.panel.duplicate_unit_days` | `InvalidRequestError` | `count`, `date`, `examples`, `route`, `unit` | `increment._frame_validation` |
| `source.panel.exposure_date` | `InvalidRequestError` | `column`, `count`, `examples`, `issue`, `route` | `increment._frame_validation` |
| `source.panel.multi_group_unit` | `InvalidRequestError` | `count`, `examples`, `group`, `route`, `unit` | `increment._frame_validation` |
| `source.panel.null_exposure` | `InvalidRequestError` | `column`, `constructor`, `count`, `route` | `increment._frame_validation` |
| `store.cell.not_found` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.decision.event_invalid` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.decision.log_corrupt` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.decision.outcome_mismatch` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.decision.reference_invalid` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.decision.schema_unsupported` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.decision.timestamp_invalid` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.history.ambiguous` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.corrupt` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.duplicate` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.empty` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.identity_incomplete` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.identity_mismatch` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.legacy_analysis_required` | `InvalidRequestError` | `route` | `increment._source_workflows` |
| `store.run.metadata_mismatch` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.not_found` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.provenance_mismatch` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.roster_mismatch` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.scope_manifest_invalid` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.scope_mapping_invalid` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `store.run.timestamp_invalid` | `InvalidRequestError` | `message` | `increment.kernel.program_refusals` |
| `tables.readout_table_ambiguous` | `InvalidRequestError` | `key` | `increment.tables` |
| `tables.readout_table_duplicate` | `InvalidRequestError` | — | `increment.tables` |
| `tables.readout_table_nest` | `InvalidRequestError` | — | `increment.tables` |
| `tables.readout_table_trend` | `InvalidRequestError` | `columns` | `increment.tables` |
| `tables.readout_table_trend_ds_all_null` | `InvalidRequestError` | — | `increment.tables` |
| `tables.readout_table_trend_duplicate_rows` | `InvalidRequestError` | `key`, `key_columns`, `rows`, `varying` | `increment.tables` |
| `tables.readout_table_trend_missing_key_columns` | `InvalidRequestError` | `key_columns`, `missing`, `nest_by` | `increment.tables` |
| `tables.readout_table_trend_mixed_ds_basis` | `InvalidRequestError` | `affected`, `bases` | `increment.tables` |
| `tables.trend_table_missing` | `InvalidRequestError` | `missing` | `increment.tables` |
| `tables.trend_table_supports` | `InvalidRequestError` | `dims` | `increment.tables` |
| `unit_cycle.envelope_required` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `unit_cycle.error_budget_exhausted` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `unit_cycle.implausible_realized_split` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `unit_cycle.invalid_design` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `unit_cycle.numerical` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `unit_cycle.reference_mismatch` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `unit_cycle.sufficient_state` | `InvalidRequestError` | `reason` | `increment._unit_cycle` |
| `wire.procedure.contrast_context` | `WireFormatError` | `action` | `increment._run` |
| `wire.result.raw_outcomes_unavailable` | `WireFormatError` | `action` | `increment._run` |

## Retired codes

| Retired code | Replacement |
| --- | --- |
| `artifact.observational.governed_evidence_missing` | Removed without replacement |
| `breakout.alpha` | `estimation.diagnostics.alpha` |
| `breakout.alpha_too_small` | `estimation.meta.alpha_too_small` |
| `breakout.metric.daily_winsorization` | `evidence.operation_unavailable` |
| `breakout.segment_contrast_interval` | `breakout.segment_contrast_absolute_interval` |
| `contrast.assignment` | Removed without replacement |
| `decision.absolute_decision.one_sided_nominal` | `decision.relative_decision.one_sided_nominal` |
| `decision.contrast_context.procedure_mapping_key` | `decision.compiled_decision.procedure_mapping_key` |
| `decision.contrast_decision.one_sided_alpha` | `decision.arm_decision.one_sided_alpha` |
| `definition.design.reject_bool` | `definition.assignment.reject_bool` |
| `definition.experiment.observation_schedule` | `definition.experiment.observation_end_before_end`, `definition.experiment.observation_end_requires_end` |
| `definition.method.finite_sample_cuped` | `conversion_inference.finite_sample.cuped` |
| `definition.validate_experiment.finite_sample_metric_type` | `conversion_inference.finite_sample.metric_type` |
| `design.exclusion_restriction.justification` | Removed without replacement |
| `estimation.absorption.alpha` | `estimation.diagnostics.alpha` |
| `estimation.adjust_aipw.aipw_control_mean` | Removed without replacement |
| `estimation.adjust_dml.dml_control_mean` | Removed without replacement |
| `estimation.adjust_iptw.iptw_hajek_control` | Removed without replacement |
| `estimation.adjust_overlap.unit_ids_shape` | `estimation.crossfit.fold_assignments_unit_ids_shape` |
| `estimation.armstats.arm_stats.family_field_finite` | `estimation.armstats.arm_stats.cross_field_finite` |
| `estimation.armstats.arm_stats.var_d_needs_uptake_sum` | `estimation.armstats.arm_stats.mean_d_needs_uptake_sum` |
| `estimation.binomial.arm_too_large_for_exact_enumeration` | `estimation.binomial.finite_sample_arm_ceiling_exceeded` |
| `estimation.cate.alpha` | `estimation.diagnostics.alpha` |
| `estimation.contrast.contrast_partition.control_group_treatment` | `estimation.contrast.contrast_stats.control_group_treatment` |
| `estimation.contrast.contrast_partition.positive_cycles_contain_least_one` | `estimation.contrast.contrast_partition.contain_least_one` |
| `estimation.contrast.contrast_reduction_m2_finite` | `estimation.contrast.contrast_reduction_delta_mean_finite` |
| `estimation.crossfit.outer_split_unit_ids_shape` | `estimation.crossfit.fold_assignments_unit_ids_shape` |
| `estimation.encouragement.alpha` | `estimation.diagnostics.alpha` |
| `estimation.encouragement.mixture_priors_are` | `estimation.adjust.prior.type` |
| `estimation.encouragement.n_comparison_inference` | Removed without replacement |
| `estimation.encouragement.nn_estimate_unknown_alternative` | `estimation.binomial.unknown_alternative` |
| `estimation.encouragement.one_sided_alpha` | `estimation.inference.one_sided_alpha_doubles` |
| `estimation.encouragement.prepare_encouragement_estimation_unknown_alternative` | `estimation.binomial.unknown_alternative` |
| `estimation.encouragement.retention` | `readout.encouragement.retention` |
| `estimation.engine.cluster.arm_needs_two` | `estimation.encouragement.cluster.arm_needs_two` |
| `estimation.engine.method.cuped_finite_sample` | `conversion_inference.finite_sample.cuped` |
| `estimation.family.e_bh_select_q_finite` | `estimation.family.bh_select_q_finite` |
| `estimation.inference.alpha_eff_underflows` | `estimation.encouragement.alpha_eff_too` |
| `estimation.inference.alpha_in_unit_interval` | `estimation.diagnostics.alpha` |
| `estimation.inference.combine_sequential_inference` | Removed without replacement |
| `estimation.inference.dof_positive_for_t` | `estimation.encouragement.dof_positive_reference` |
| `estimation.inference.infer_ate_abs_diff_finite` | `estimation.inference.infer_lift_abs_diff_finite` |
| `estimation.inference.infer_ate_abs_se_finite` | `estimation.inference.infer_lift_abs_se_finite` |
| `estimation.inference.infer_ate_abs_se_positive` | `estimation.inference.infer_lift_abs_se_positive` |
| `estimation.inference.mixture_priors_are` | `estimation.adjust.prior.type` |
| `estimation.inference.n_comparison_inference` | Removed without replacement |
| `estimation.inference.normal.expected_negative_part_scale` | `estimation.priors.mixture_posterior.expected_negative_part_scale` |
| `estimation.inference.normal.expected_positive_part_scale` | `estimation.priors.mixture_posterior.expected_positive_part_scale` |
| `estimation.inference.null_abs_finite` | `estimation.inference.infer_ate_null_abs_finite` |
| `estimation.inference.null_lift_finite` | `estimation.inference.infer_ate_null_lift_finite` |
| `estimation.inference.prior_sequential_inference` | Removed without replacement |
| `estimation.inference.sequential_inference_cluster` | Removed without replacement |
| `estimation.inference.unknown_alternative` | `estimation.binomial.unknown_alternative` |
| `estimation.meta.marginalized_segment.alpha_strictly_between` | `estimation.meta.alpha_strictly_between` |
| `estimation.meta.marginalized_segment.alpha_too_small` | `estimation.meta.alpha_too_small` |
| `estimation.meta.tau_prior_scale_overflow` | Removed without replacement |
| `estimation.quantile.alpha` | `estimation.diagnostics.alpha` |
| `estimation.quantile.alpha_too_small` | `estimation.meta.alpha_too_small` |
| `estimation.results.estimate.lb_ub_both_when_lb_set` | `estimation.results.estimate.lb_ub_both` |
| `estimation.results.lift.p_value_null_lift_not_representable` | `estimation.results.lift.p_value_null_lift_not_representable_dof` |
| `estimation.results.lift.prob_favorable_cluster_robust_null_abs` | `estimation.results.lift.p_value_cluster_robust_null_abs` |
| `estimation.results.lift.prob_favorable_null_abs_missing_abs_se` | `estimation.results.lift.p_value_null_abs_missing_abs_se` |
| `estimation.sequential_inversion.inexact_value` | `estimation.certified.inexact_value` |
| `estimation.sequential_inversion.invalid_endpoint_reason` | `estimation.sequential_likelihood.invalid_certificate_reason` |
| `estimation.sequential_inversion.nonpositive_max_width` | `estimation.certified.nonpositive_root_width` |
| `estimation.sequential_inversion.state_prior_dimension_mismatch` | `estimation.sequential_likelihood.state_prior_dimension_mismatch` |
| `estimation.sequential_inversion.unknown_alternative` | `estimation.sequential_likelihood.unknown_alternative` |
| `estimation.sequential_likelihood.inexact_value` | `estimation.certified.inexact_value` |
| `estimation.sequential_likelihood.invalid_count` | `estimation.certified.invalid_count` |
| `estimation.sitewide.finite` | `estimation.armstats.arm_stats.cross_field_finite` |
| `estimation.targeting.adjustment_nonnumeric` | `estimation.targeting.adjustment_dtype` |
| `estimation.targeting.alpha` | `estimation.diagnostics.alpha` |
| `estimation.targeting.n_folds_least` | `estimation.crossfit.n_folds_least` |
| `estimation.targeting.outcome_nulls_non` | `estimation.cate.outcome_nulls_non` |
| `estimation.targeting.treatment_binary` | `estimation.cate.treatment_binary` |
| `estimation.variance.cluster_outcome_moments_needs_ref_den` | `estimation.variance.ratio_moments_needs_ref_den` |
| `estimation.variance.cluster_uptake_moments_nonpositive_size` | `estimation.variance.cluster_outcome_moments_nonpositive_size` |
| `estimation.winsor.bootstrap_replicate_failure` | Removed without replacement |
| `estimation.winsor.bootstrap_tail_unresolved` | Removed without replacement |
| `estimation.winsor.empty_region` | Removed without replacement |
| `estimation.winsor.pilot_nonpositive_outcome` | `estimation.winsor.pilot_negative_outcome` |
| `facade.analysis.no_definitions` | `evidence.grain_unavailable` |
| `facade.analysis.operation` | `evidence.breakout_unavailable`, `evidence.operation_unavailable` |
| `facade.analysis.source_context_design_disagrees_contrast` | `facade.analysis.source_context_design_disagrees_arm` |
| `frame.frame_panel.metric_was_declared` | `frame.frame_totals.metric_was_declared` |
| `frame.metric.finite_sample_metric_type` | `conversion_inference.finite_sample.metric_type` |
| `frame.moments.metric_covariate_column` | `frame.validation.metric_covariate_impute_all_null` |
| `plan.contrast_configs_template_mismatch` | `plan.compile_configs_template_mismatch` |
| `plan.contrast_configs_wrong_type` | `plan.compile_configs_wrong_type` |
| `power.alpha` | `estimation.diagnostics.alpha` |
| `power.segment_pairwise_minimum_n_per_arm_too_small` | `power.segment_pairwise_achieved_n_per_arm_too_small` |
| `power.sequential_power_enclosure_se_full_finite` | `power.se_full_finite` |
| `power.units_per_week` | `power.units_per_week_finite_positive` |
| `power.var_finite_strictly` | `estimation.meta.var_finite_strictly` |
| `query.builder.operation` | `query.builders.cluster_column_missing`, `query.builders.cluster_size_imbalance`, `query.builders.cluster_total_grain`, `query.builders.cluster_undeclared`, `query.builders.site_volume_metric_type` |
| `query.builders.unit_totals_implemented` | Removed without replacement |
| `query.builders.unknown_filter_op` | Removed without replacement |
| `query.calendar.rolling_total_values_unknown_aggregation` | `query.builders.unknown_aggregation` |
| `query.calendar.total_period_values_unknown_aggregation` | `query.builders.unknown_aggregation` |
| `query.calendar.unknown_aggregation` | `query.builders.unknown_aggregation` |
| `query.calendar.unknown_grain` | `query.calendar.period_start_unknown_grain` |
| `query.fact_resolution.fact_source_property_overwrites_value_column` | `query.fact_resolution.fact_source_property` |
| `query.integrity.mixed_assignments` | `query.integrity.mixed_assignment_units`, `query.integrity.unassigned_assignment_units` |
| `query.session.warehouse_artifact.write_relation_sealed` | `query.session.warehouse_artifact.publication_handle_sealed` |
| `query.source.artifact_facade.verification_lazy_digest` | `query.artifact_reader.artifact_moment.verification_lazy_digest` |
| `readout.estimate_ate_every` | `estimation.adjust.estimate_ate_every` |
| `readout.inference.encouragement_run` | Removed without replacement |
| `readout.metric.daily_winsorization` | `evidence.operation_unavailable` |
| `readout.source.dimension` | `evidence.breakout_unavailable` |
| `readout.source.grain` | `evidence.grain_unavailable` |
| `readout.unknown_estimand_supported` | `estimation.encouragement.unknown_estimand_supported` |
| `readout.value_scale_encouragement_absolute` | `readout.encouragement.value_scale` |
| `sequential.ratio.denominator_near_zero` | Removed without replacement |
| `source.breakout.cluster_grain` | `evidence.cluster_unavailable` |
| `source.breakout.dimension` | `evidence.breakout_unavailable` |
| `source.breakout.grain` | `evidence.grain_unavailable` |
| `source.breakout.sql` | `evidence.operation_unavailable` |
| `source.breakout.unit_grain` | `evidence.unit_rows_unavailable` |
| `source.frame.breakouts_unsupported` | `evidence.breakout_unavailable` |
| `source.frame.cluster_grain` | `evidence.cluster_unavailable` |
| `source.frame.grain` | `evidence.grain_unavailable`, `evidence.breakout_unavailable` |
| `source.frame.multi_dimension` | `evidence.breakout_unavailable` |
| `source.frame.sql_unsupported` | `evidence.operation_unavailable` |
| `source.frame.undeclared_breakout` | `evidence.breakout_unavailable` |
| `source.frame_panel.cluster_grain` | `evidence.cluster_unavailable` |
| `source.frame_panel.grain` | `evidence.grain_unavailable` |
| `source.moments.breakout` | `evidence.breakout_unavailable` |
| `source.moments.cluster_grain` | `evidence.cluster_unavailable` |
| `source.moments.covariate_unavailable` | `evidence.covariate_unavailable` |
| `source.moments.grain` | `evidence.grain_unavailable` |
| `source.moments.sql` | `evidence.operation_unavailable` |
| `source.moments.unit_grain` | `evidence.unit_rows_unavailable` |
| `source.native.breakout_grain` | `evidence.grain_unavailable` |
| `source.native.cluster_grain` | `evidence.cluster_unavailable` |
| `source.native.grain` | `evidence.grain_unavailable` |
| `source.native.operation` | `evidence.grain_unavailable`, `evidence.breakout_unavailable`, `evidence.operation_unavailable` |
| `source.native.unit_grain` | `evidence.unit_rows_unavailable` |
| `source.native.warehouse_cluster_grain` | `evidence.cluster_unavailable` |
| `source.native_sql_grain` | `evidence.grain_unavailable` |
| `source.operation.unsupported` | `evidence.operation_unavailable` |
| `source.sql.cluster_grain` | `evidence.cluster_unavailable` |
| `source.sql.grain` | `evidence.grain_unavailable` |
| `source.sql.operation` | `evidence.breakout_unavailable`, `evidence.operation_unavailable` |
| `source.sql.sql_grain` | `evidence.grain_unavailable` |
| `source.sql.unit_grain` | `evidence.unit_rows_unavailable` |
